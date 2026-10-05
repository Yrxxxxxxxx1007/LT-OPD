# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import copy
import datetime
import hashlib
import json
import logging
import os
import random
import warnings
from dataclasses import asdict
from typing import Any, Optional
import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from peft.utils.save_and_load import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.chat_template import resolve_custom_chat_template
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.utils.transformers_compat import get_auto_model_for_vision2seq
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
device_name = get_device_name()


def _projector_aware_vllm_enabled(config) -> bool:
    projector_config = config.model.get("vision_token_projector", {})
    return (
        bool(projector_config.get("enabled", False))
        and config.rollout.name == "vllm"
        and bool(config.rollout.get("projector_aware", False))
    )


def _normalize_projector_param_name(name: str) -> str:
    marker = "vision_token_projector."
    if marker in name:
        return marker + name.split(marker, 1)[1]
    return name


def _normalize_native_visual_merger_name(name: str) -> str:
    marker = "visual.merger."
    if marker not in name:
        raise ValueError(f"Not a native visual merger parameter: {name}")
    suffix = name.split(marker, 1)[1]
    suffix = ".".join((part for part in suffix.split(".") if part != "_fsdp_wrapped_module"))
    return marker + suffix


_FULL_PARAMETER_OPTIMIZER_GROUP_ORDER = (
    "language_decay",
    "language_no_decay",
    "vision_decay",
    "vision_no_decay",
    "native_visual_merger_decay",
    "native_visual_merger_no_decay",
)
_SUMMARY_OPTIMIZER_GROUP_ORDER = ("summary_decay", "summary_no_decay")
_LEARNABLE_SUMMARY_PARAMETER_SUFFIXES = {"input.weight", "input.bias", "output.weight"}


def _normalize_full_parameter_name(name: str) -> str:
    """Return a stable parameter name before/after nested FSDP1 wrapping."""
    return ".".join((part for part in name.split(".") if part != "_fsdp_wrapped_module"))


def _full_parameter_domain(name: str) -> str:
    normalized = _normalize_full_parameter_name(name)
    if ".vision_token_compressor.learnable_summary." in f".{normalized}":
        return "summary"
    if "visual.merger." in f"{normalized}.":
        return "native_visual_merger"
    if ".visual." in f".{normalized}." or normalized.startswith("visual."):
        return "vision"
    return "language"


def _full_parameter_group_order(assignments: dict[str, str]) -> tuple[str, ...]:
    """Preserve the six backbone groups and append the optional learned head."""
    has_summary = any(group.startswith("summary_") for group in assignments.values())
    return _FULL_PARAMETER_OPTIMIZER_GROUP_ORDER + (_SUMMARY_OPTIMIZER_GROUP_ORDER if has_summary else ())


def _validate_learnable_summary_inventory(module) -> None:
    """Require one canonical registration and all three learned head tensors."""
    registered = [
        name for name, _ in module.named_modules(remove_duplicate=False)
        if name.endswith("vision_token_compressor.learnable_summary")
    ]
    if not registered:
        return
    if len(registered) != 1:
        raise RuntimeError(f"Learnable summary must have one module registration, got {registered}")
    prefix = registered[0] + "."
    actual = {
        name[len(prefix):] for name, _ in module.named_parameters(remove_duplicate=False)
        if name.startswith(prefix)
    }
    if actual != _LEARNABLE_SUMMARY_PARAMETER_SUFFIXES:
        raise RuntimeError(
            "Learnable summary parameter inventory mismatch: "
            f"expected={sorted(_LEARNABLE_SUMMARY_PARAMETER_SUFFIXES)}, actual={sorted(actual)}"
        )


def _initialize_learnable_aggregation_from_pretrained(module, compressor) -> Optional[dict[str, Any]]:
    """Bootstrap a fresh head from the loaded first language MLP projection."""
    head = getattr(compressor, "learnable_summary", None)
    if head is None:
        return None
    input_hidden_size = head.input_hidden_size
    donors = [
        (name, parameter)
        for name, parameter in module.named_parameters(remove_duplicate=True)
        if (
            name == "language_model.layers.0.mlp.gate_proj.weight"
            or name.endswith(".language_model.layers.0.mlp.gate_proj.weight")
        )
        and "visual" not in name.lower()
        and parameter.ndim == 2
        and parameter.shape[1] == input_hidden_size
    ]
    if len(donors) != 1:
        raise RuntimeError(
            "Learnable aggregation requires exactly one matching first language MLP gate projection: "
            f"hidden_size={input_hidden_size}, candidates={[name for name, _ in donors]}"
        )
    source_name, source_weight = donors[0]
    if source_weight.is_meta:
        raise RuntimeError(
            f"Learnable aggregation donor must be materialized after base checkpoint loading: {source_name}"
        )
    return head.initialize_from_pretrained_projection(source_weight, source_name=source_name)


def _full_parameter_uses_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    normalized = _normalize_full_parameter_name(name).lower()
    leaf = normalized.rsplit(".", 1)[-1]
    is_norm = any((token in normalized for token in ("layernorm", "layer_norm", "rmsnorm", "rms_norm", ".norm.")))
    return not (leaf == "bias" or parameter.ndim < 2 or is_norm)


def _build_full_parameter_assignment(module) -> dict[str, str]:
    """Inventory each actor parameter into a backbone or optional summary group."""
    _validate_learnable_summary_inventory(module)
    assignments: dict[str, str] = {}
    seen_parameter_ids: set[int] = set()
    for name, parameter in module.named_parameters(remove_duplicate=True):
        normalized = _normalize_full_parameter_name(name)
        if normalized in assignments:
            raise RuntimeError(f"Duplicate full-parameter name after FSDP normalization: {normalized}")
        if id(parameter) in seen_parameter_ids:
            raise RuntimeError(f"Duplicate parameter object in full-parameter inventory: {normalized}")
        seen_parameter_ids.add(id(parameter))
        if not parameter.requires_grad:
            raise RuntimeError(f"Full-parameter actor unexpectedly contains a frozen parameter: {normalized}")
        if not parameter.is_floating_point() or parameter.dtype != torch.float32:
            raise TypeError(
                f"Full-parameter actor master weight must be FP32 floating point: {normalized}={parameter.dtype}"
            )
        suffix = "decay" if _full_parameter_uses_weight_decay(normalized, parameter) else "no_decay"
        assignments[normalized] = f"{_full_parameter_domain(normalized)}_{suffix}"
    if not assignments:
        raise RuntimeError("Full-parameter actor inventory is empty")
    missing_groups = sorted(set(_full_parameter_group_order(assignments)) - set(assignments.values()))
    if missing_groups:
        raise RuntimeError(f"Full-parameter actor has empty optimizer groups: {missing_groups}")
    return assignments


def _build_full_parameter_optimizer_groups(module, assignments: dict[str, str], optim_config) -> list[dict[str, Any]]:
    """Resolve the pre-wrap inventory on FSDP original parameters exactly once."""
    group_order = _full_parameter_group_order(assignments)
    grouped: dict[str, list[torch.nn.Parameter]] = {name: [] for name in group_order}
    resolved_names: set[str] = set()
    resolved_ids: set[int] = set()
    for name, parameter in module.named_parameters(remove_duplicate=True):
        if not parameter.requires_grad:
            continue
        normalized = _normalize_full_parameter_name(name)
        group_name = assignments.get(normalized)
        if group_name is None:
            raise RuntimeError(f"FSDP trainable parameter is absent from the pre-wrap inventory: {normalized}")
        if normalized in resolved_names or id(parameter) in resolved_ids:
            raise RuntimeError(f"FSDP trainable parameter was resolved more than once: {normalized}")
        resolved_names.add(normalized)
        resolved_ids.add(id(parameter))
        grouped[group_name].append(parameter)
    if resolved_names != assignments.keys():
        missing = sorted(assignments.keys() - resolved_names)
        extra = sorted(resolved_names - assignments.keys())
        raise RuntimeError(f"FSDP full-parameter inventory mismatch: missing={missing[:20]}, extra={extra[:20]}")
    empty = [name for name, parameters in grouped.items() if not parameters]
    if empty:
        raise RuntimeError(f"FSDP full-parameter optimizer groups became empty: {empty}")
    domain_lrs = {
        "language": float(optim_config.lr),
        "vision": float(optim_config.vision_lr),
        "native_visual_merger": float(optim_config.merger_lr),
    }
    if "summary_decay" in grouped:
        summary_lr = optim_config.get("summary_lr", None)
        if summary_lr is None or not np.isfinite(float(summary_lr)) or float(summary_lr) <= 0:
            raise ValueError("Learnable summary requires a finite positive actor.optim.summary_lr")
        domain_lrs["summary"] = float(summary_lr)
    groups = []
    for group_name in group_order:
        domain = group_name.removesuffix("_no_decay").removesuffix("_decay")
        groups.append(
            {
                "params": grouped[group_name],
                "lr": domain_lrs[domain],
                "weight_decay": 0.0 if group_name.endswith("_no_decay") else float(optim_config.weight_decay),
                "group_name": group_name,
            }
        )
    return groups


def _validate_fixed_dense_teacher_module(module) -> None:
    """Verify the immutable dense teacher invariant on plain or FSDP modules."""
    if module.training:
        raise RuntimeError("Fixed dense teacher must remain in eval mode")
    trainable = [name for name, parameter in module.named_parameters(remove_duplicate=True) if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"Fixed dense teacher contains trainable parameters: {trainable[:20]}")
    compressor_modules = [name for name, _ in module.named_modules() if "vision_token_compressor" in name]
    compressor_enabled = any(
        (bool(getattr(submodule, "vision_token_compressor_enabled", False)) for _, submodule in module.named_modules())
    )
    if compressor_modules or compressor_enabled:
        raise RuntimeError(
            f"Fixed teacher must be dense and contain no visual compressor: modules={compressor_modules[:20]}, enabled={compressor_enabled}"
        )


def _native_visual_merger_parameters(module) -> dict[str, torch.nn.Parameter]:
    parameters: dict[str, torch.nn.Parameter] = {}
    for name, parameter in module.named_parameters(remove_duplicate=True):
        if "visual.merger." not in name:
            continue
        normalized = _normalize_native_visual_merger_name(name)
        if normalized in parameters:
            raise RuntimeError(f"Duplicate native visual merger parameter after FSDP normalization: {normalized}")
        parameters[normalized] = parameter
    return parameters


def _load_hf_rollout_replica_merger(replica, state: dict[str, torch.Tensor]) -> None:
    if not state:
        raise RuntimeError("HF rollout replica synchronization received an empty native visual merger")
    parameters = _native_visual_merger_parameters(replica)
    if parameters.keys() != state.keys():
        missing = sorted(state.keys() - parameters.keys())
        extra = sorted(parameters.keys() - state.keys())
        raise RuntimeError(f"Native visual merger key mismatch: missing={missing[:20]}, unexpected={extra[:20]}")
    with torch.no_grad():
        for name, source in state.items():
            target = parameters[name]
            if target.shape != source.shape:
                raise RuntimeError(f"Native visual merger shape mismatch for {name}: {target.shape} vs {source.shape}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    unequal = [
        name
        for name, source in state.items()
        if not torch.equal(parameters[name].detach().cpu(), source.to(dtype=parameters[name].dtype).cpu())
    ]
    if unequal:
        raise RuntimeError(f"Native visual merger differs after replica synchronization: {unequal[:20]}")


def _collect_projector_params(module) -> dict[str, torch.Tensor]:
    projector_params = {}
    for name, param in module.named_parameters(remove_duplicate=True):
        if "vision_token_projector." not in name:
            continue
        param_tensor = param.full_tensor() if hasattr(param, "full_tensor") else param
        projector_params[_normalize_projector_param_name(name)] = param_tensor.detach().cpu()
    return projector_params


def _exclude_projector_from_lora(exclude_modules):
    projector_regex = ".*vision_token_projector.*"
    if exclude_modules is None or exclude_modules == "null":
        return projector_regex
    if isinstance(exclude_modules, str):
        has_projector = "vision_token_projector" in exclude_modules
        if has_projector:
            return exclude_modules
        return f"(?:{exclude_modules})|(?:{projector_regex})"
    if isinstance(exclude_modules, (list, tuple, set)):
        values = list(exclude_modules)
        if not any(("vision_token_projector" in str(value) for value in values)):
            values.append(projector_regex)
        return values
    return exclude_modules


def _prepare_lora_target_inventory(module, peft_config):
    """Resolve PEFT shorthands and snapshot every module the adapter must target."""
    from peft.tuners.tuners_utils import _maybe_include_all_linear_layers, check_target_module_exists

    peft_config = _maybe_include_all_linear_layers(peft_config, module)
    expected = {
        name for name, _ in module.named_modules() if name and bool(check_target_module_exists(peft_config, name))
    }
    if not expected:
        raise RuntimeError("LoRA target selection resolved to an empty module inventory")
    return (peft_config, expected)


def _validate_lora_target_inventory(peft_model, expected, expected_linear_attn_targets=None):
    """Fail closed if PEFT silently skipped or unexpectedly added any target."""
    tuner = getattr(peft_model, "base_model", None)
    actual = set(getattr(tuner, "targeted_module_names", ()))
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA target inventory mismatch: expected={len(expected)}, actual={len(actual)}, missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    linear_attn_targets = {name for name in actual if ".linear_attn.in_proj_" in f".{name}"}
    if expected_linear_attn_targets is not None and len(linear_attn_targets) != expected_linear_attn_targets:
        raise RuntimeError(
            f"LoRA linear-attention inventory mismatch: expected={expected_linear_attn_targets}, actual={len(linear_attn_targets)}"
        )
    return actual


def _freeze_inference_module(module):
    """Make the fixed reference/teacher state explicit before and after wrapping."""
    module.requires_grad_(False)
    module.eval()
    return module


def _clone_hf_rollout_replica(actor_module, *, dtype: torch.dtype | None = None):
    """Clone the fully configured CPU actor without advancing Torch RNG.

    The clone happens before FSDP mutates the module hierarchy, so the rollout
    copy has no FSDP hooks.  It is intentionally limited to CPU construction:
    cloning a live CUDA/FSDP module would duplicate transient full-parameter
    buffers and would not have a defensible memory bound.
    """
    named_tensors = list(actor_module.named_parameters()) + list(actor_module.named_buffers())
    non_cpu = sorted({f"{name}={tensor.device}" for name, tensor in named_tensors if tensor.device.type != "cpu"})
    if non_cpu:
        raise RuntimeError(
            f"HF rollout replica must be cloned before FSDP from fully materialized CPU tensors; found non-CPU tensors={non_cpu[:20]}"
        )
    _validate_learnable_summary_inventory(actor_module)
    torch_rng = torch.get_rng_state().clone()
    replica = copy.deepcopy(actor_module)
    if not torch.equal(torch.get_rng_state(), torch_rng):
        torch.set_rng_state(torch_rng)
        raise RuntimeError("Cloning the HF rollout replica advanced the Torch RNG state")
    if dtype is not None:
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError(f"Unsupported HF rollout replica dtype: {dtype}")
        replica.to(dtype=dtype)
    disable_input_require_grads = getattr(replica, "disable_input_require_grads", None)
    if callable(disable_input_require_grads):
        disable_input_require_grads()
    disable_gradient_checkpointing = getattr(replica, "gradient_checkpointing_disable", None)
    if callable(disable_gradient_checkpointing):
        disable_gradient_checkpointing()
    _freeze_inference_module(replica)
    if any((parameter.requires_grad for parameter in replica.parameters())):
        raise RuntimeError("HF rollout replica contains trainable parameters")
    replica_tensors = list(replica.named_parameters()) + list(replica.named_buffers())
    if any((tensor.device.type == "meta" for _, tensor in replica_tensors)):
        raise RuntimeError("HF rollout replica contains unmaterialized meta tensors")
    return replica


def _load_hf_rollout_replica_adapter(replica, adapter_state) -> None:
    """Load and then bitwise-verify the complete portable PEFT adapter."""
    if not adapter_state:
        raise RuntimeError("HF rollout replica synchronization received an empty LoRA adapter")
    expected = {name: tensor.detach().cpu() for name, tensor in adapter_state.items()}
    load_result = set_peft_model_state_dict(replica, expected, adapter_name="default")
    unexpected = sorted(getattr(load_result, "unexpected_keys", ()) or ())
    if unexpected:
        raise RuntimeError(f"HF rollout replica adapter has unexpected keys: {unexpected[:20]}")
    actual_raw = get_peft_model_state_dict(replica, adapter_name="default")
    actual = {name: tensor.detach().cpu() for name, tensor in actual_raw.items()}
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise RuntimeError(f"HF rollout replica adapter key mismatch: missing={missing[:20]}, unexpected={extra[:20]}")
    unequal = [name for name in expected if not torch.equal(actual[name], expected[name])]
    if unequal:
        raise RuntimeError(f"HF rollout replica adapter tensors differ after synchronization: {unequal[:20]}")
    _freeze_inference_module(replica)


def _portable_full_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize distributed state before handing it to a plain HF replica."""
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu()


def _reference_param_offload_enabled(role, fsdp_config) -> bool:
    return role in {"ref", "teacher"} and bool(fsdp_config.get("param_offload", False))


def _seed_actor_model_initialization(base_seed: int, rank: int) -> int:
    """Seed every RNG before PEFT creates its random LoRA-A parameters."""
    worker_seed = int(base_seed) + int(rank)
    random.seed(worker_seed)
    np.random.seed(worker_seed % 2**32)
    torch.manual_seed(worker_seed)
    return worker_seed


def _rollout_rng_state_path(local_path: str, world_size: int, rank: int) -> str:
    return os.path.join(local_path, f"rollout_rng_world_size_{world_size}_rank_{rank}.pt")


def _save_rollout_rng_state(local_path: str, world_size: int, rank: int, torch_state, generation_state) -> None:
    path = _rollout_rng_state_path(local_path, world_size, rank)
    torch.save({"torch_random_states": torch_state.cpu(), "gen_random_states": generation_state.cpu()}, path)


def _load_rollout_rng_state(local_path: str, world_size: int, rank: int):
    path = _rollout_rng_state_path(local_path, world_size, rank)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Exact rollout resume requires the per-rank generation RNG sidecar, but it is missing: {path}"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"torch_random_states", "gen_random_states"}
    if set(state) != required or not all((isinstance(state[key], torch.Tensor) for key in required)):
        raise RuntimeError(f"Invalid rollout RNG sidecar: {path}")
    return state


def print_fsdp_children(module, tag="model"):
    found = False
    for name, submodule in module.named_modules():
        if isinstance(submodule, FSDP):
            print(f"[Already FSDP wrapped][{tag}] {name}: {type(submodule)}")
            found = True
    if not found:
        print(f"[No FSDP child found][{tag}]")


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh, zero3_enable=True):
    from torch.distributed.fsdp import ShardingStrategy

    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    if device_mesh.ndim == 1:
        sharding_strategy = fsdp_strategy
    elif device_mesh.ndim == 2:
        sharding_strategy = hsdp_strategy
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        return vl_model_instance.visual
    return None


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self.config.model.get("lora_adapter_path") is not None or self._lora_rank > 0
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.hf_rollout_replica = None
        self._hf_rollout_replica_dirty = False
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            )
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        use_prefix_grouper=False,
        use_tiled_mlp=False,
        tiled_mlp_shards=4,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText

        AutoModelForVision2Seq = get_auto_model_for_vision2seq()
        from verl.utils.model import (
            align_qwen35_chat_generation_config,
            get_generation_config,
            print_model_size,
            update_model_config,
        )
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref", "teacher"]
        training_mode = self.config.actor.get("training_mode", "legacy")
        full_parameter_actor = role == "actor" and training_mode == "full_parameter"
        apply_lora = self._is_lora and role == "actor"
        train_native_visual_merger = bool(
            role == "actor" and self.config.model.get("train_native_visual_merger", False)
        )
        use_replicated_hf_rollout = bool(
            role == "actor" and self.config.rollout.get("hf_use_replicated_module", False)
        )
        hf_rollout_replica = None
        full_parameter_assignments = None
        if full_parameter_actor:
            invalid_reasons = []
            if apply_lora or int(self.config.model.get("lora_rank", 0)) != 0:
                invalid_reasons.append("LoRA must be disabled (rank=0)")
            if self.config.model.get("lora_adapter_path") is not None:
                invalid_reasons.append("lora_adapter_path must be null")
            if self.config.actor.get("freeze_vision_tower", False):
                invalid_reasons.append("the vision tower must be trainable")
            if not train_native_visual_merger:
                invalid_reasons.append(
                    "train_native_visual_merger must explicitly declare the native merger trainable"
                )
            if self.config.actor.strategy != "fsdp":
                invalid_reasons.append("FSDP1 is required")
            if not bool(fsdp_config.get("use_orig_params", False)):
                invalid_reasons.append("FSDP1 use_orig_params=True is required")
            if bool(fsdp_config.get("reshard_after_forward", True)):
                invalid_reasons.append("actor reshard_after_forward must be false")
            if bool(fsdp_config.get("param_offload", False)) or bool(fsdp_config.get("optimizer_offload", False)):
                invalid_reasons.append("actor parameter/optimizer offload must be disabled")
            if enable_activation_offload:
                invalid_reasons.append("actor activation offload must be disabled")
            if invalid_reasons:
                raise ValueError("Invalid full-parameter actor contract: " + "; ".join(invalid_reasons))
        if use_replicated_hf_rollout:
            compressor_config = self.config.model.get("vision_token_compressor", {})
            projector_config = self.config.model.get("vision_token_projector", {})
            invalid_reasons = []
            if not self._is_actor or not self._is_rollout:
                invalid_reasons.append("the worker must own both actor and rollout roles")
            if self.config.rollout.name != "hf":
                invalid_reasons.append("rollout.name must be hf")
            if not train_native_visual_merger:
                invalid_reasons.append("the rollout replica requires train_native_visual_merger=True")
            if self.config.actor.strategy != "fsdp":
                invalid_reasons.append("actor.strategy must be fsdp (FSDP1)")
            if self.ulysses_sequence_parallel_size != 1:
                invalid_reasons.append("ulysses_sequence_parallel_size must be 1")
            if int(self.config.rollout.get("tensor_model_parallel_size", 1)) != 1:
                invalid_reasons.append("rollout.tensor_model_parallel_size must be 1")
            if int(self.config.rollout.get("pipeline_model_parallel_size", 1)) != 1:
                invalid_reasons.append("rollout.pipeline_model_parallel_size must be 1")
            if int(self.config.rollout.get("data_parallel_size", 1)) != 1:
                invalid_reasons.append("rollout.data_parallel_size must be 1")
            if bool(projector_config.get("enabled", False)):
                invalid_reasons.append("vision_token_projector must be disabled")
            if not bool(compressor_config.get("enabled", False)):
                invalid_reasons.append("vision_token_compressor must be enabled")
            if full_parameter_actor:
                if apply_lora or int(self.config.model.get("lora_rank", 0)) != 0:
                    invalid_reasons.append("the formal full-parameter replica forbids LoRA")
                if compressor_config.get("algorithm") not in {
                    "qwen35_holitom_dpc_spatial_merge_v1",
                    "qwen35_holitom_dpc_spatial_merge_curriculum_v2",
                    "qwen35_cdpruner_v1",
                }:
                    invalid_reasons.append(
                        "the formal full-parameter replica requires audited HoliTom-DPC or CDPruner"
                    )
                if not bool(self.config.rollout.get("hf_replica_cpu_offload_between_phases", False)):
                    invalid_reasons.append("the full-parameter replica must be parked on CPU during actor update")
                if not bool(self.config.rollout.get("hf_active_sequence_compaction", False)):
                    invalid_reasons.append("the optimized replica requires active sequence compaction")
                if not bool(self.config.rollout.get("hf_share_rollout_prefix", False)):
                    invalid_reasons.append("the optimized replica requires exact same-UID prefix sharing")
                if int(self.config.rollout.get("hf_dart_decode_max_prefill_cost", 0)) <= 0:
                    invalid_reasons.append("the optimized replica requires a positive prefill-cost ceiling")
            else:
                if not apply_lora:
                    invalid_reasons.append("the legacy replica actor must use LoRA")
                if compressor_config.get("algorithm") not in {
                    "qwen35_cdpruner_v1",
                    "qwen35_conditional_diversity_prune_v1",
                }:
                    invalid_reasons.append("legacy replica compression must be CDPruner or explicit legacy")
            if bool(fsdp_config.get("param_offload", False)):
                invalid_reasons.append("actor parameter offload must be disabled")
            if bool(fsdp_config.get("optimizer_offload", False)):
                invalid_reasons.append("actor optimizer offload must be disabled")
            if enable_activation_offload:
                invalid_reasons.append("activation offload must be disabled")
            if invalid_reasons:
                raise ValueError(
                    "hf_use_replicated_module violates its audited HF rollout contract: " + "; ".join(invalid_reasons)
                )
        if use_tiled_mlp and self.config.actor.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `actor_rollout_ref.actor.strategy=fsdp2`.")
        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor_resize_kwargs = {}
        processor_min_pixels = self.config.model.get("processor_min_pixels")
        processor_max_pixels = self.config.model.get("processor_max_pixels")
        if processor_min_pixels is not None or processor_max_pixels is not None:
            processor_resize_kwargs = {"min_pixels": processor_min_pixels, "max_pixels": processor_max_pixels}
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, **processor_resize_kwargs)
        custom_chat_template = resolve_custom_chat_template(self.config.model)
        if custom_chat_template is not None:
            self.tokenizer.chat_template = custom_chat_template
            if self.processor is not None:
                self.processor.chat_template = custom_chat_template
                processor_tokenizer = getattr(self.processor, "tokenizer", None)
                if processor_tokenizer is None:
                    raise RuntimeError("Multimodal processor does not expose its bound tokenizer")
                processor_tokenizer.chat_template = custom_chat_template
                template_hashes = {
                    hashlib.sha256(value.encode("utf-8")).hexdigest()
                    for value in (
                        self.tokenizer.chat_template,
                        self.processor.chat_template,
                        processor_tokenizer.chat_template,
                    )
                }
                if len(template_hashes) != 1:
                    raise RuntimeError("Worker tokenizer/processor chat-template identity drift")
        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        if role == "teacher" and training_mode == "full_parameter":
            teacher_errors = []
            if torch_dtype != torch.bfloat16:
                teacher_errors.append(f"model_dtype must be bfloat16, got {torch_dtype}")
            if PrecisionType.to_dtype(fsdp_config.dtype) != torch.bfloat16:
                teacher_errors.append(f"compute dtype must be bfloat16, got {fsdp_config.dtype}")
            if not bool(fsdp_config.get("reshard_after_forward", False)):
                teacher_errors.append("reshard_after_forward must be true")
            if bool(fsdp_config.get("param_offload", False)) or bool(fsdp_config.get("optimizer_offload", False)):
                teacher_errors.append("parameter and optimizer offload must be disabled")
            if enable_gradient_checkpointing or enable_activation_offload:
                teacher_errors.append("gradient checkpointing and activation offload must be disabled")
            if teacher_errors:
                raise ValueError("Invalid formal fixed-teacher contract: " + "; ".join(teacher_errors))
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"
        if (
            getattr(actor_model_config, "model_type", None) == "qwen2_5_vl"
            and attn_implementation == "flash_attention_3"
            and hasattr(actor_model_config, "vision_config")
        ):
            actor_model_config.vision_config._attn_implementation = "flash_attention_2"
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
        self.generation_config = align_qwen35_chat_generation_config(
            self.generation_config,
            tokenizer=self.tokenizer,
            model_config=actor_model_config,
            required=training_mode == "full_parameter",
        )
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings and (not use_replicated_hf_rollout),
            mesh=self.device_mesh,
        )
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                (actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values())
            )
            if has_remote_code:
                auto_class = next(
                    (k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v)
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            elif type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                actor_module_class = AutoModelForCausalLM
            elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                actor_module_class = AutoModelForImageTextToText
            else:
                actor_module_class = AutoModel
            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
            projector_config = self.config.model.get("vision_token_projector", {})
            projector_enabled = bool(projector_config.get("enabled", False)) and role == "actor"
            compressor_config = self.config.model.get("vision_token_compressor", {})
            compressor_enabled = bool(compressor_config.get("enabled", False)) and role == "actor"
            if projector_enabled and compressor_enabled:
                raise ValueError("vision_token_projector and vision_token_compressor cannot both be enabled")
            if projector_enabled:
                from verl.models.transformers.vision_token_projector import VisualTokenMLPProjector

                hidden_size = getattr(actor_model_config, "hidden_size", None)
                if hidden_size is None and hasattr(actor_model_config, "text_config"):
                    hidden_size = getattr(actor_model_config.text_config, "hidden_size", None)
                if hidden_size is None:
                    raise ValueError("vision_token_projector requires a model hidden_size or text_config.hidden_size.")
                if not hasattr(actor_model_config, "vision_config"):
                    raise ValueError("vision_token_projector can only be enabled for vision-language models.")
                num_tokens = int(projector_config.get("num_tokens", 64))
                mlp_ratio = float(projector_config.get("mlp_ratio", 2.0))
                projector = VisualTokenMLPProjector(
                    hidden_size=hidden_size, num_tokens=num_tokens, mlp_ratio=mlp_ratio
                )
                actor_module.vision_token_projector = projector
                actor_module.vision_token_projector_enabled = True
                actor_module.vision_token_projector_num_tokens = num_tokens
                if hasattr(actor_module, "model"):
                    actor_module.model.vision_token_projector = projector
                    actor_module.model.vision_token_projector_enabled = True
                    actor_module.model.vision_token_projector_num_tokens = num_tokens
                if self.rank == 0:
                    print(f"[actor model] Enabled visual token MLP projector with num_tokens={num_tokens}.")
            else:
                actor_module.vision_token_projector_enabled = False
                if hasattr(actor_module, "model"):
                    actor_module.model.vision_token_projector_enabled = False
            if compressor_enabled:
                from verl.models.transformers.vision_token_compressor import build_vision_token_compressor

                if not hasattr(actor_model_config, "vision_config"):
                    raise ValueError("Conditional visual pruning requires a vision-language model.")
                algorithm = compressor_config.get("algorithm")
                input_hidden_size = getattr(getattr(actor_model_config, "text_config", None), "hidden_size", None)
                if input_hidden_size is None:
                    input_hidden_size = getattr(actor_model_config, "hidden_size", None)
                compressor = build_vision_token_compressor(
                    compressor_config, allow_legacy=True, input_hidden_size=input_hidden_size
                )
                if getattr(compressor, "learnable_summary", None) is not None and not full_parameter_actor:
                    raise ValueError("Learnable CDPruner aggregation requires actor.training_mode='full_parameter'")
                initialization = _initialize_learnable_aggregation_from_pretrained(actor_module, compressor)
                if initialization is not None and self.rank == 0:
                    print(
                        "[actor model] Deterministic aggregation initialization: "
                        f"source={initialization['source_name']}, rows={initialization['row_indices']}."
                    )
                # The language-model forward consumes this owner. Register the
                # learned module once so FSDP and state_dict use one key path.
                compressor_owner = getattr(actor_module, "model", actor_module)
                compressor_owner.vision_token_compressor = compressor
                actor_module.vision_token_compressor_enabled = True
                actor_module.vision_token_compressor_minimum_tokens = compressor.minimum_tokens
                actor_module.vision_token_compressor_retention_ratio = compressor.retention_ratio
                actor_module.vision_token_compressor_retention_bps = compressor.active_retention_bps
                actor_module.vision_token_compressor_summary_enabled = bool(
                    getattr(compressor, "summary_enabled", False)
                )
                actor_module.vision_token_compressor_algorithm = algorithm
                compressor_owner.vision_token_compressor_last_routes = None
                if hasattr(actor_module, "model"):
                    actor_module.model.vision_token_compressor_enabled = True
                    actor_module.model.vision_token_compressor_minimum_tokens = compressor.minimum_tokens
                    actor_module.model.vision_token_compressor_retention_ratio = compressor.retention_ratio
                    actor_module.model.vision_token_compressor_retention_bps = compressor.active_retention_bps
                    actor_module.model.vision_token_compressor_summary_enabled = bool(
                        getattr(compressor, "summary_enabled", False)
                    )
                    actor_module.model.vision_token_compressor_algorithm = algorithm
                    actor_module.model.vision_token_compressor_last_routes = None
                if self.rank == 0:
                    aggregation_note = (
                        "; nearest-kept signed residual merge starts with zero discarded contribution, preserving K"
                        if getattr(compressor, "learnable_merge_enabled", False) else
                        "; add one discarded-token summary when K<N"
                        if getattr(compressor, "summary_enabled", False) else ""
                    )
                    print(
                        f"[actor model] Enabled {algorithm} ({compressor.__class__.__name__}) with K=min(N, max({compressor.minimum_tokens}, ceil({compressor.retention_ratio}*N))){aggregation_note}."
                    )
            else:
                actor_module.vision_token_compressor_enabled = False
                if hasattr(actor_module, "model"):
                    actor_module.model.vision_token_compressor_enabled = False
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)
            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )
            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )
            actor_module.to(torch_dtype)
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if apply_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftConfig, PeftModel

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))
                peft_config = PeftConfig.from_pretrained(local_adapter_path)
                peft_config, expected_lora_targets = _prepare_lora_target_inventory(actor_module, peft_config)
                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM
            else:
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": _exclude_projector_from_lora(
                        convert_to_regular_types(self.config.model.exclude_modules)
                    ),
                    "bias": "none",
                }
                peft_config, expected_lora_targets = _prepare_lora_target_inventory(
                    actor_module, LoraConfig(**lora_config)
                )
                actor_module = get_peft_model(actor_module, peft_config)
            expected_linear_attn_targets = os.getenv("VERL_EXPECTED_LINEAR_ATTN_LORA_TARGETS")
            if expected_linear_attn_targets is not None:
                try:
                    expected_linear_attn_targets = int(expected_linear_attn_targets)
                except ValueError as exc:
                    raise ValueError("VERL_EXPECTED_LINEAR_ATTN_LORA_TARGETS must be an integer") from exc
                if expected_linear_attn_targets < 0:
                    raise ValueError("VERL_EXPECTED_LINEAR_ATTN_LORA_TARGETS must be non-negative")
            actual_lora_targets = _validate_lora_target_inventory(
                actor_module, expected_lora_targets, expected_linear_attn_targets=expected_linear_attn_targets
            )
            if self.rank == 0:
                linear_attn_targets = sorted(
                    (name for name in actual_lora_targets if ".linear_attn.in_proj_" in f".{name}")
                )
                print(
                    f"[actor model] Verified exact LoRA target inventory: total={len(actual_lora_targets)}, linear_attn_in_proj={len(linear_attn_targets)}."
                )
            trainable_dtypes = {}
            for name, param in actor_module.named_parameters():
                if not param.requires_grad:
                    continue
                if not param.is_floating_point():
                    raise TypeError(f"Trainable parameter {name} is not floating point: {param.dtype}")
                if param.dtype != torch.float32:
                    param.data = param.data.to(dtype=torch.float32)
                trainable_dtypes[name] = param.dtype
            invalid_trainables = {name: dtype for name, dtype in trainable_dtypes.items() if dtype != torch.float32}
            if invalid_trainables:
                raise TypeError(f"All trainable LoRA parameters must be FP32, got {invalid_trainables}")
            if not trainable_dtypes:
                raise RuntimeError("LoRA was enabled but no trainable parameters were found")
            if self.rank == 0:
                print(f"[actor model] Verified {len(trainable_dtypes)} FP32 trainable LoRA parameters.")
            if projector_enabled:
                lora_param_dtype = next(
                    (
                        param.dtype
                        for name, param in actor_module.named_parameters()
                        if "lora_" in name and param.requires_grad
                    ),
                    None,
                )
                projector_modules = [
                    module
                    for module_name, module in actor_module.named_modules()
                    if module_name.endswith("vision_token_projector")
                ]
                for projector_module in projector_modules:
                    if lora_param_dtype is not None:
                        projector_module.to(dtype=lora_param_dtype)
                    projector_module.requires_grad_(True)
                if self.rank == 0:
                    if projector_modules:
                        dtype_msg = f" dtype={lora_param_dtype}" if lora_param_dtype is not None else ""
                        print(f"[actor model] Visual token projector is trainable with LoRA enabled.{dtype_msg}")
                    else:
                        print("[actor model] WARNING: visual token projector not found after applying LoRA.")
        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if apply_lora and (not self.use_orig_params):
            self.use_orig_params = True
            if self.rank == 0:
                print("[actor model] Forcing FSDP use_orig_params=True for mixed frozen BF16 / trainable FP32 LoRA.")
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            elif self.rank == 0:
                print("[actor model] No vision tower found.")
        if train_native_visual_merger and (not full_parameter_actor):
            if not self.config.actor.get("freeze_vision_tower", False):
                raise ValueError("Training only visual.merger requires freeze_vision_tower=True")
            merger_parameters = _native_visual_merger_parameters(actor_module)
            if not merger_parameters:
                raise RuntimeError("Qwen3.5 native visual.merger parameter inventory is empty")
            for parameter in merger_parameters.values():
                parameter.requires_grad_(True)
                if not parameter.is_floating_point():
                    raise TypeError("Native visual.merger parameters must be floating point")
                parameter.data = parameter.data.to(dtype=torch.float32)
            self.use_orig_params = True
        if full_parameter_actor:
            actor_module.requires_grad_(True)
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is None:
                raise RuntimeError("Full-parameter V6 requires a Qwen vision tower")
            if not _native_visual_merger_parameters(actor_module):
                raise RuntimeError("Full-parameter V6 requires the native Qwen visual.merger")
            for name, parameter in actor_module.named_parameters(remove_duplicate=True):
                if not parameter.is_floating_point():
                    raise TypeError(f"Full-parameter model parameter is not floating point: {name}={parameter.dtype}")
                if parameter.dtype != torch.float32:
                    parameter.data = parameter.data.to(dtype=torch.float32)
            full_parameter_assignments = _build_full_parameter_assignment(actor_module)
            self.use_orig_params = True
            if self.rank == 0:
                counts = {
                    group: sum((value == group for value in full_parameter_assignments.values()))
                    for group in _full_parameter_group_order(full_parameter_assignments)
                }
                print(
                    f"[actor model] Full-parameter FP32 inventory verified before FSDP: tensors={len(full_parameter_assignments)}, groups={counts}."
                )
        elif role == "actor":
            trainable_inventory = {
                name: parameter
                for name, parameter in actor_module.named_parameters(remove_duplicate=True)
                if parameter.requires_grad
            }
            lora_trainables = {name for name in trainable_inventory if "lora_" in name}
            merger_trainables = {name for name in trainable_inventory if "visual.merger." in name}
            unexpected_trainables = sorted(set(trainable_inventory) - lora_trainables - merger_trainables)
            if not lora_trainables or (train_native_visual_merger and (not merger_trainables)):
                raise RuntimeError(
                    f"Incomplete trainable inventory: lora={len(lora_trainables)}, merger={len(merger_trainables)}"
                )
            if unexpected_trainables:
                raise RuntimeError(
                    f"Unexpected trainable parameters outside LoRA/native merger: {unexpected_trainables[:20]}"
                )
            non_fp32 = sorted(
                (name for name, parameter in trainable_inventory.items() if parameter.dtype != torch.float32)
            )
            if non_fp32:
                raise TypeError(f"Trainable master parameters are not FP32: {non_fp32[:20]}")
            if self.rank == 0:
                print(
                    f"[actor model] Exact trainable inventory verified: lora_tensors={len(lora_trainables)}, native_visual_merger_tensors={len(merger_trainables)}."
                )
        if use_replicated_hf_rollout:
            if full_parameter_actor:
                if hasattr(actor_module, "peft_config"):
                    raise RuntimeError("Formal full-parameter rollout replication forbids a PEFT wrapper")
                hf_rollout_replica = _clone_hf_rollout_replica(actor_module, dtype=torch.bfloat16)
            else:
                if not hasattr(actor_module, "peft_config") or "default" not in actor_module.peft_config:
                    raise RuntimeError("HF rollout replication requires the actor's complete default PEFT adapter")
                trainable_names = sorted(
                    (name for name, parameter in actor_module.named_parameters() if parameter.requires_grad)
                )
                invalid_trainables = [
                    name for name in trainable_names if "lora_" not in name and "visual.merger." not in name
                ]
                if not trainable_names or invalid_trainables:
                    raise RuntimeError(
                        f"Legacy HF rollout replication permits only LoRA plus native visual.merger trainables; trainable_count={len(trainable_names)}, invalid={invalid_trainables[:20]}"
                    )
                hf_rollout_replica = _clone_hf_rollout_replica(actor_module)
            if self.rank == 0:
                print(
                    "[actor model] Built a frozen CPU native HF rollout replica before FSDP; its trainable state will be synchronized immediately before every dirty rollout."
                )
        if role in {"ref", "teacher"}:
            _freeze_inference_module(actor_module)
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(actor_module)
        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=apply_lora
        )
        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")
        fsdp_mesh = self.device_mesh
        fsdp_enable_zero3 = fsdp_config.reshard_after_forward
        sharding_strategy = get_sharding_strategy(fsdp_mesh, fsdp_enable_zero3)
        reference_param_offload = _reference_param_offload_enabled(role, fsdp_config)
        cpu_offload = CPUOffload(offload_params=True) if reference_param_offload else None
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            print_fsdp_children(actor_module, "actor_module before FSDP")
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = CPUOffloadPolicy(pin_memory=True) if reference_param_offload else None
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")
        if role in {"ref", "teacher"}:
            _freeze_inference_module(actor_module_fsdp)
            if role == "teacher" and training_mode == "full_parameter":
                _validate_fixed_dense_teacher_module(actor_module_fsdp)
        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)
        if hf_rollout_replica is not None:
            replica_device = torch.device(device_name, get_device_id())
            park_replica_on_cpu = bool(self.config.rollout.get("hf_replica_cpu_offload_between_phases", False))
            if not park_replica_on_cpu:
                hf_rollout_replica.to(device=replica_device)
            _freeze_inference_module(hf_rollout_replica)
            replica_tensors = list(hf_rollout_replica.named_parameters()) + list(hf_rollout_replica.named_buffers())
            expected_replica_device = torch.device("cpu") if park_replica_on_cpu else replica_device
            wrong_device = sorted(
                (
                    f"{name}={tensor.device}"
                    for name, tensor in replica_tensors
                    if tensor.device != expected_replica_device
                )
            )
            if wrong_device:
                raise RuntimeError(
                    f"HF rollout replica was not fully materialized on {expected_replica_device}: {wrong_device[:20]}"
                )
            if full_parameter_actor:
                non_bf16 = sorted(
                    (
                        name
                        for name, parameter in hf_rollout_replica.named_parameters()
                        if parameter.dtype != torch.bfloat16
                    )
                )
                if non_bf16:
                    raise TypeError(f"Formal rollout replica has non-BF16 parameters: {non_bf16[:20]}")
            if any((parameter.requires_grad for parameter in hf_rollout_replica.parameters())):
                raise RuntimeError("GPU HF rollout replica unexpectedly contains trainable parameters")
            self.hf_rollout_replica = hf_rollout_replica
            self._hf_rollout_replica_dirty = True
            if self.rank == 0:
                replica_bytes = sum((tensor.numel() * tensor.element_size() for _, tensor in replica_tensors))
                print(
                    f"[actor model] Native HF rollout replica is resident and excluded from training/checkpoint state: device={expected_replica_device}, tensor_storage={replica_bytes / 1024**3:.2f} GiB."
                )
        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            if full_parameter_actor:
                if full_parameter_assignments is None:
                    raise RuntimeError("Full-parameter optimizer has no pre-FSDP inventory")
                if optim_config.get("vision_lr") is None or optim_config.get("merger_lr") is None:
                    raise ValueError("Full-parameter optimizer requires vision_lr and merger_lr")
                parameter_groups = _build_full_parameter_optimizer_groups(
                    actor_module_fsdp, full_parameter_assignments, optim_config
                )
            else:
                named_trainables = [
                    (name, parameter)
                    for name, parameter in actor_module_fsdp.named_parameters(remove_duplicate=True)
                    if parameter.requires_grad
                ]
                lora_parameters = [parameter for name, parameter in named_trainables if "lora_" in name]
                merger_parameters = [parameter for name, parameter in named_trainables if "visual.merger." in name]
                covered_ids = [id(parameter) for parameter in lora_parameters + merger_parameters]
                if len(covered_ids) != len(set(covered_ids)) or len(covered_ids) != len(named_trainables):
                    missing = [name for name, parameter in named_trainables if id(parameter) not in set(covered_ids)]
                    raise RuntimeError(
                        f"Optimizer groups do not cover trainables exactly once: missing={missing[:20]}"
                    )
                projector_lr = optim_config.get("projector_lr", None)
                if train_native_visual_merger and projector_lr is None:
                    raise ValueError("train_native_visual_merger=True requires actor.optim.projector_lr")
                parameter_groups = [
                    {
                        "params": lora_parameters,
                        "lr": float(optim_config.lr),
                        "weight_decay": float(optim_config.weight_decay),
                        "group_name": "language_lora",
                    }
                ]
                if merger_parameters:
                    parameter_groups.append(
                        {
                            "params": merger_parameters,
                            "lr": float(projector_lr),
                            "weight_decay": float(optim_config.weight_decay),
                            "group_name": "native_visual_merger",
                        }
                    )
            actor_optimizer = build_optimizer(parameter_groups, optim_config)
            if self.rank == 0:
                print(
                    "[actor optimizer] Exact groups: "
                    + ", ".join(
                        (
                            f"{group['group_name']}:tensors={len(group['params'])},lr={group['lr']},wd={group['weight_decay']}"
                            for group in parameter_groups
                        )
                    )
                )
            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            warmup_update_indexing = optim_config.get("lr_warmup_update_indexing", "zero_based_legacy_v1")
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")
            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    warmup_update_indexing=warmup_update_indexing,
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                    warmup_update_indexing=warmup_update_indexing,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None
        return (actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config)

    def _sync_hf_rollout_replica(self) -> None:
        """Refresh the frozen rollout replica without perturbing either RNG stream."""
        if not bool(self.config.rollout.get("hf_use_replicated_module", False)):
            return
        replica = self.hf_rollout_replica
        if replica is None:
            raise RuntimeError("hf_use_replicated_module=True but the native HF rollout replica was not built")
        if not self._hf_rollout_replica_dirty:
            if replica.training or any((parameter.requires_grad for parameter in replica.parameters())):
                raise RuntimeError("Clean HF rollout replica is not frozen in eval mode")
            return
        cpu_rng_before = torch.get_rng_state().clone()
        device_rng_before = get_torch_device().get_rng_state().clone()
        sync_error = None
        try:
            full_parameter_replica = self.config.actor.get("training_mode", "legacy") == "full_parameter"
            if full_parameter_replica:
                replica_device = torch.device(device_name, get_device_id())
                if bool(self.config.rollout.get("hf_replica_cpu_offload_between_phases", False)):
                    non_cpu = sorted(
                        (
                            f"{name}={tensor.device}"
                            for name, tensor in list(replica.named_parameters()) + list(replica.named_buffers())
                            if tensor.device.type != "cpu"
                        )
                    )
                    if non_cpu:
                        raise RuntimeError(
                            f"Dirty rollout replica must be CPU-parked before synchronization: {non_cpu[:20]}"
                        )

                def canonical_name(name: str) -> str:
                    return name.replace("_fsdp_wrapped_module.", "").replace("._fsdp_wrapped_module", "")

                def canonical_inventory(items, *, label: str):
                    result = {}
                    for name, tensor in items:
                        canonical = canonical_name(name)
                        if canonical in result:
                            raise RuntimeError(f"Duplicate canonical {label} name during replica sync: {canonical}")
                        result[canonical] = tensor
                    return result

                wrapped_actor = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
                with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False, recurse=True):
                    source_parameters = canonical_inventory(wrapped_actor.named_parameters(), label="actor parameter")
                    target_parameters = canonical_inventory(replica.named_parameters(), label="replica parameter")
                    if source_parameters.keys() != target_parameters.keys():
                        missing = sorted(target_parameters.keys() - source_parameters.keys())
                        extra = sorted(source_parameters.keys() - target_parameters.keys())
                        raise RuntimeError(
                            f"Full-parameter rollout replica inventory mismatch: missing={missing[:20]}, extra={extra[:20]}"
                        )
                    with torch.no_grad():
                        for name, target in target_parameters.items():
                            source = source_parameters[name]
                            if source.shape != target.shape:
                                raise RuntimeError(
                                    f"Replica parameter shape mismatch for {name}: actor={tuple(source.shape)}, replica={tuple(target.shape)}"
                                )
                            target.copy_(source.to(device=target.device, dtype=target.dtype))
                        source_buffers = canonical_inventory(wrapped_actor.named_buffers(), label="actor buffer")
                        target_buffers = canonical_inventory(replica.named_buffers(), label="replica buffer")
                        missing_buffers = sorted(target_buffers.keys() - source_buffers.keys())
                        if missing_buffers:
                            raise RuntimeError(
                                f"Full-parameter rollout replica buffers are missing from actor: {missing_buffers[:20]}"
                            )
                        for name, target in target_buffers.items():
                            source = source_buffers[name]
                            if source.shape != target.shape:
                                raise RuntimeError(f"Replica buffer shape mismatch for {name}")
                            target.copy_(source.to(device=target.device, dtype=target.dtype))
                replica.to(device=replica_device, dtype=torch.bfloat16)
                _freeze_inference_module(replica)
            else:
                peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
                with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False):
                    adapter_state = {
                        name: _portable_full_cpu_tensor(tensor)
                        for name, tensor in get_peft_model_state_dict(peft_model, adapter_name="default").items()
                    }
                    merger_state = {
                        name: _portable_full_cpu_tensor(parameter)
                        for name, parameter in _native_visual_merger_parameters(peft_model).items()
                    }
                _load_hf_rollout_replica_adapter(replica, adapter_state)
                _load_hf_rollout_replica_merger(replica, merger_state)
            if not bool(self.config.rollout.get("hf_preserve_cuda_cache", False)):
                get_torch_device().empty_cache()
        except Exception as exc:
            sync_error = exc
        cpu_rng_changed = not torch.equal(torch.get_rng_state(), cpu_rng_before)
        device_rng_changed = not torch.equal(get_torch_device().get_rng_state(), device_rng_before)
        if cpu_rng_changed or device_rng_changed:
            torch.set_rng_state(cpu_rng_before)
            get_torch_device().set_rng_state(device_rng_before)
            raise RuntimeError(
                "HF rollout replica synchronization changed Torch RNG state; refusing to alter sampling trajectory"
            ) from sync_error
        if sync_error is not None:
            raise sync_error
        if replica.training or any((parameter.requires_grad for parameter in replica.parameters())):
            raise RuntimeError("HF rollout replica synchronization did not preserve frozen eval mode")
        self._hf_rollout_replica_dirty = False
        if self.rank == 0:
            if self.config.actor.get("training_mode", "legacy") == "full_parameter":
                print("[actor model] Synchronized the complete BF16 full-parameter rollout replica.")
            else:
                print("[actor model] Synchronized and bitwise-verified rollout LoRA plus native visual.merger.")

    def _park_hf_rollout_replica(self) -> None:
        """Remove the inference replica from CUDA before the FP32 actor update."""
        if not bool(self.config.rollout.get("hf_use_replicated_module", False)):
            return
        if not bool(self.config.rollout.get("hf_replica_cpu_offload_between_phases", False)):
            return
        replica = self.hf_rollout_replica
        if replica is None:
            raise RuntimeError("Cannot park a missing HF rollout replica")
        replica.to(device=torch.device("cpu"))
        _freeze_inference_module(replica)
        non_cpu = sorted(
            (
                f"{name}={tensor.device}"
                for name, tensor in list(replica.named_parameters()) + list(replica.named_buffers())
                if tensor.device.type != "cpu"
            )
        )
        if non_cpu:
            raise RuntimeError(f"HF rollout replica did not fully return to CPU: {non_cpu[:20]}")

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name
        self.rollout_device_mesh = rollout_device_mesh
        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        if rollout_name == "hf":
            from verl.workers.rollout.hf_rollout import HFRollout

            use_replicated_module = bool(self.config.rollout.get("hf_use_replicated_module", False))
            rollout_module = self.hf_rollout_replica if use_replicated_module else self.actor_module_fsdp
            if rollout_module is None:
                raise RuntimeError("Native HF rollout replication was requested, but no replica is available")
            self.rollout = HFRollout(
                config=rollout_config,
                model_config=model_config,
                device_mesh=rollout_device_mesh,
                module=rollout_module,
                tokenizer=self.tokenizer,
                processor=self.processor,
                keep_module_in_eval=use_replicated_module,
            )
        else:
            self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)
        if self.config.rollout.name == "hf":
            self.actor_module_fsdp.eval()
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)
            return
        peft_config = None
        projector_params = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            elif _projector_aware_vllm_enabled(self.config) and getattr(self.rollout, "sleep_level", None) != 2:
                with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False):
                    projector_params = _collect_projector_params(peft_model)
        else:
            params = self.actor_module_fsdp.state_dict()
        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp, layered_summon=self.layered_summon, base_sync_done=False
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )
        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)
        set_expandable_segments(False)
        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params
        else:
            device = get_device_id()
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params
        if projector_params:
            projector_device = get_device_id()
            per_tensor_projector_params = (
                (
                    name,
                    param.to(projector_device, non_blocking=True).full_tensor()
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in projector_params.items()
            )
            await self.rollout.update_weights(per_tensor_projector_params, base_sync_done=False)
            del projector_params, per_tensor_projector_params
        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done
        )
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)
        self.base_sync_done = True
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)
        self.actor_module_fsdp.train()
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(True)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._is_actor:
            model_init_seed = int(self.config.actor.get("data_loader_seed", 42))
            worker_seed = _seed_actor_model_initialization(model_init_seed, self.rank)
            if self.rank == 0:
                print(
                    f"[actor model] Seeded Python/NumPy/Torch before model and LoRA initialization: base_seed={model_init_seed}, rank0_seed={worker_seed}."
                )
        from verl.workers.actor import DataParallelPPOActor
        from verl.workers.actor.dp_actor import TrustRegionTeacher

        import_external_libs(self.config.model.get("external_lib", None))
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            use_tiled_mlp = tiled_mlp_config.get("enabled", False)
            tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
                self._build_model_optimizer(
                    model_path=local_path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    use_liger=self.config.model.get("use_liger", False),
                    role="actor",
                    enable_activation_offload=self.config.model.get("enable_activation_offload", False),
                    use_prefix_grouper=self.config.actor.get("use_prefix_grouper", False),
                    use_tiled_mlp=use_tiled_mlp,
                    tiled_mlp_shards=tiled_mlp_shards,
                )
            )
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                actor_lr_scheduler=self.actor_lr_scheduler,
            )
            if getattr(self, "tokenizer", None) is not None:
                self.actor.tokenizer = self.tokenizer
        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        if self._is_ref:
            self_distillation_cfg = self.config.actor.get("self_distillation", None) if self._is_actor else None
            fixed_teacher_uses_ref_slot = bool(
                self_distillation_cfg is not None and self_distillation_cfg.get("fixed_teacher_uses_ref_slot", False)
            )
            if fixed_teacher_uses_ref_slot:
                ref_model_path = self_distillation_cfg.get("teacher_model_path")
                if not ref_model_path:
                    raise ValueError("Fixed teacher ref-slot reuse requires teacher_model_path")
            else:
                ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None and (not fixed_teacher_uses_ref_slot):
                ref_model_path = ref_model.get("path", self.config.model.path)
            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            use_prefix_grouper = hasattr(self.config, "actor") and self.config.actor.get("use_prefix_grouper", False)
            ref_tiled_mlp_config = self.config.ref.get("tiled_mlp", None)
            if ref_tiled_mlp_config is None:
                ref_tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            ref_use_tiled_mlp = ref_tiled_mlp_config.get("enabled", False)
            ref_tiled_mlp_shards = ref_tiled_mlp_config.get("num_shards", 4)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="teacher" if fixed_teacher_uses_ref_slot else "ref",
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=ref_use_tiled_mlp,
                tiled_mlp_shards=ref_tiled_mlp_shards,
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
                if use_prefix_grouper:
                    self.config.ref.use_prefix_grouper = use_prefix_grouper
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
            if getattr(self, "tokenizer", None) is not None:
                self.ref_policy.tokenizer = self.tokenizer
            if self._is_actor:
                self_distillation_cfg = self.config.actor.get("self_distillation", None)
                loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
                if self_distillation_cfg is not None and loss_mode == "vopd":
                    teacher_model_source = self_distillation_cfg.get("teacher_model_source", "legacy")
                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    if teacher_model_source == "current":
                        self.actor.teacher_module = None
                    elif teacher_model_source == "fixed" and fixed_teacher_uses_ref_slot:
                        self.actor.teacher_module = _freeze_inference_module(self.ref_module_fsdp)
                        if self.config.actor.get("training_mode", "legacy") == "full_parameter":
                            _validate_fixed_dense_teacher_module(self.actor.teacher_module)
                            self.actor.teacher_update_count = 0
                    elif teacher_model_source == "fixed":
                        if self.config.actor.get("training_mode", "legacy") == "full_parameter":
                            raise RuntimeError(
                                "Formal V6 requires fixed_teacher_uses_ref_slot=True; refusing a duplicate ref+teacher model"
                            )
                        teacher_model_path = self_distillation_cfg.get("teacher_model_path")
                        if self.rank == 0:
                            print("self-distillation fixed teacher model:", teacher_model_path)
                        teacher_local_path = copy_to_local(teacher_model_path, use_shm=use_shm)
                        self.teacher_module_fsdp = self._build_model_optimizer(
                            model_path=teacher_local_path,
                            fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                            optim_config=None,
                            override_model_config=override_model_config,
                            use_remove_padding=use_remove_padding,
                            use_fused_kernels=use_fused_kernels,
                            trust_remote_code=self.config.model.get("trust_remote_code", False),
                            use_liger=self.config.model.get("use_liger", False),
                            role="teacher",
                            use_prefix_grouper=use_prefix_grouper,
                            use_tiled_mlp=ref_use_tiled_mlp,
                            tiled_mlp_shards=ref_tiled_mlp_shards,
                        )[0]
                        self.actor.teacher_module = _freeze_inference_module(self.teacher_module_fsdp)
                    elif teacher_regularization == "trust-region":
                        self.actor.teacher_module = TrustRegionTeacher(
                            ref_module=self.ref_module_fsdp,
                            student_module=self.actor_module_fsdp,
                            mix_coef=self_distillation_cfg.get("teacher_update_rate", 0.0),
                        )
                    else:
                        self.actor.teacher_module = self.ref_module_fsdp
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )
        if not self._is_actor and self._is_rollout:
            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_visual_token_curriculum_step(self, completed_optimizer_steps: int) -> dict[str, Any]:
        """Bind every actor/rollout copy to the same committed-step curriculum state."""
        from verl.models.transformers.vision_token_compressor import (
            VisionCDPrunerCompressor, VisionHoliTomDPCSpatialMergeCompressor,
        )

        if not self._is_actor or not self._is_rollout:
            raise RuntimeError("visual-token curriculum requires the colocated actor/rollout worker")
        if isinstance(completed_optimizer_steps, bool) or not isinstance(completed_optimizer_steps, int):
            raise TypeError("completed_optimizer_steps must be an integer")
        if completed_optimizer_steps < 0:
            raise ValueError("completed_optimizer_steps must be non-negative")
        roots = [self.actor_module_fsdp]
        replica = getattr(self, "hf_rollout_replica", None)
        if replica is not None:
            roots.append(replica)
        compressors: dict[int, torch.nn.Module] = {}
        owners = []
        for root in roots:
            for module in root.modules():
                compressor = module.__dict__.get("_modules", {}).get("vision_token_compressor")
                if isinstance(compressor, (VisionCDPrunerCompressor, VisionHoliTomDPCSpatialMergeCompressor)) and compressor.curriculum is not None:
                    compressors[id(compressor)] = compressor
                    owners.append(module)
        if not compressors:
            raise RuntimeError("curriculum binding found no actor compressor with a configured curriculum")
        states = [compressor.set_curriculum_step(completed_optimizer_steps) for compressor in compressors.values()]
        state = states[0]
        if any((candidate != state for candidate in states[1:])):
            raise RuntimeError(f"actor/rollout curriculum copies disagree: {states!r}")
        for owner in owners:
            owner.vision_token_compressor_retention_ratio = state["retention_bps"] / 10000.0
            owner.vision_token_compressor_retention_bps = state["retention_bps"]
            owner.vision_token_compressor_curriculum_completed_steps = state["completed_optimizer_steps"]
            owner.vision_token_compressor_curriculum_schedule_sha256 = state["schedule_sha256"]
        rank_states = [None] * dist.get_world_size()
        dist.all_gather_object(rank_states, state)
        if any((candidate != state for candidate in rank_states)):
            raise RuntimeError(f"distributed curriculum state drift: {rank_states!r}")
        self._visual_token_curriculum_state = dict(state)
        return dict(state)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
        gate_global_step = int(data.meta_info.get("global_steps", -1))
        curriculum_config = self.config.model.get("vision_token_compressor", {}).get("curriculum")
        if curriculum_config is not None:
            state = getattr(self, "_visual_token_curriculum_state", None)
            if not isinstance(state, dict):
                raise RuntimeError("actor update has no bound visual-token curriculum state")
            expected_completed = gate_global_step - 1
            if state.get("completed_optimizer_steps") != expected_completed:
                raise RuntimeError(
                    f"actor update curriculum/global-step drift: completed={state.get('completed_optimizer_steps')!r}, global_step={gate_global_step}"
                )
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            if bool(self.config.rollout.get("hf_use_replicated_module", False)):
                self._hf_rollout_replica_dirty = True
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            images_seqlens = data.meta_info.get("images_seqlens", None)
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time, images_seqlens=images_seqlens
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / 1024**3
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / 1024**3
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / 1024**3
            scheduler_per_optimizer_step = bool(
                self.config.actor.optim.get("lr_scheduler_step_per_optimizer_step", False)
            )
            if not scheduler_per_optimizer_step:
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
                self.actor_lr_scheduler.step()
            elif "actor/lr" not in metrics:
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        curriculum_config = self.config.model.get("vision_token_compressor", {}).get("curriculum")
        if curriculum_config is not None:
            state = getattr(self, "_visual_token_curriculum_state", None)
            expected = {
                "visual_token_curriculum_schema_version": state.get("schema_version")
                if isinstance(state, dict)
                else None,
                "visual_token_curriculum_schedule_sha256": state.get("schedule_sha256")
                if isinstance(state, dict)
                else None,
                "visual_token_curriculum_completed_steps": state.get("completed_optimizer_steps")
                if isinstance(state, dict)
                else None,
                "visual_token_retention_bps": state.get("retention_bps") if isinstance(state, dict) else None,
            }
            actual = {key: prompts.meta_info.get(key) for key in expected}
            if state is None or actual != expected:
                raise RuntimeError(
                    f"rollout request is not bound to the active visual-token curriculum: expected={expected!r}, actual={actual!r}"
                )
        prompts = prompts.to(get_device_id())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        preserve_hf_cuda_cache = bool(
            self.config.rollout.name == "hf" and self.config.rollout.get("hf_preserve_cuda_cache", False)
        )
        if self._is_actor:
            if self.config.rollout.name == "hf":
                if not preserve_hf_cuda_cache:
                    aggressive_empty_cache(force_sync=True)
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self._sync_hf_rollout_replica()
                self.actor_module_fsdp.eval()
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)
            else:
                loop = get_event_loop()
                loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)
        if self._is_actor:
            if self.config.rollout.name == "hf":
                self._park_hf_rollout_replica()
                self.actor_module_fsdp.train()
                set_expandable_segments(True)
                self.gen_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.torch_random_states)
            else:
                loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        if not preserve_hf_cuda_cache:
            aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        calculate_entropy = not is_lora
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                outputs = self.actor.compute_log_prob(data=data, calculate_entropy=calculate_entropy)
            if not is_lora:
                tensors = {"old_log_probs": outputs["log_probs"]}
            else:
                tensors = {"ref_log_prob": outputs["log_probs"]}
            if calculate_entropy:
                tensors["entropys"] = outputs["entropys"]
            if "sum_pi_squared" in outputs:
                tensors["sum_pi_squared"] = outputs["sum_pi_squared"]
            output = DataProto.from_dict(tensors=tensors, meta_info={"temperature": self.config.rollout.temperature})
        output = output.to("cpu")
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            data.meta_info["is_lora"] = True
            return self.compute_log_prob(data)
        assert self._is_ref
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            outputs = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": outputs["log_probs"]})
        output = output.to("cpu")
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_rollout:
            if not hasattr(self, "torch_random_states") or not hasattr(self, "gen_random_states"):
                raise RuntimeError("Rollout checkpoint save requires initialized trainer and generation RNG states")
            os.makedirs(local_path, exist_ok=True)
            _save_rollout_rng_state(
                local_path,
                world_size=dist.get_world_size(),
                rank=dist.get_rank(),
                torch_state=self.torch_random_states,
                generation_state=self.gen_random_states,
            )
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()
        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            save_error = None
            try:
                peft_config = {}
                if dist.get_rank() == 0:
                    os.makedirs(lora_save_path, exist_ok=True)
                    peft_config = asdict(peft_model.peft_config.get("default", {}))
                    peft_config["task_type"] = peft_config["task_type"].value
                    peft_config["peft_type"] = peft_config["peft_type"].value
                    peft_config["target_modules"] = sorted(peft_config["target_modules"])
                    if float(peft_config.get("lora_alpha", 0)) <= 0:
                        raise ValueError("Checkpoint LoRA adapter must have a positive lora_alpha")
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                save_error = f"rank={dist.get_rank()} {type(e).__name__}: {e}"
            save_errors = [None] * dist.get_world_size()
            dist.all_gather_object(save_errors, save_error)
            save_errors = [error for error in save_errors if error is not None]
            if save_errors:
                raise RuntimeError(f"Failed to save the LoRA adapter: {save_errors}")
            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )
        if self.config.actor.get("training_mode", "legacy") != "full_parameter" and bool(
            self.config.model.get("train_native_visual_merger", False)
        ):
            merger_save_path = os.path.join(local_path, "native_visual_merger")
            merger_save_error = None
            try:
                with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False):
                    peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
                    merger_state = {
                        name: parameter.detach().cpu().contiguous()
                        for name, parameter in _native_visual_merger_parameters(peft_model).items()
                    }
                    if not merger_state:
                        raise RuntimeError("Native visual merger checkpoint inventory is empty")
                    if dist.get_rank() == 0:
                        os.makedirs(merger_save_path, exist_ok=True)
                        save_file(merger_state, os.path.join(merger_save_path, "model.safetensors"))
                        manifest = {
                            "schema_version": "qwen35_native_visual_merger_v1",
                            "tensor_count": len(merger_state),
                            "all_fp32": all((value.dtype == torch.float32 for value in merger_state.values())),
                            "tensor_shapes": {name: list(value.shape) for name, value in merger_state.items()},
                        }
                        with open(os.path.join(merger_save_path, "manifest.json"), "w", encoding="utf-8") as handle:
                            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
                            handle.write("\n")
            except Exception as exc:
                merger_save_error = f"rank={dist.get_rank()} {type(exc).__name__}: {exc}"
            merger_save_errors = [None] * dist.get_world_size()
            dist.all_gather_object(merger_save_errors, merger_save_error)
            merger_save_errors = [error for error in merger_save_errors if error is not None]
            if merger_save_errors:
                raise RuntimeError(f"Failed to save the native visual merger: {merger_save_errors}")
            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved native visual merger to: {merger_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finalize_checkpoint(self, local_path, max_ckpt_to_keep=None):
        """Apply retention only after the controller atomically commits the checkpoint.

        ``save_checkpoint`` finishes the distributed model/optimizer/adapter
        files first.  The controller then writes its dataloader state,
        completion marker and latest tracker.  Deferring destructive retention
        until this method ensures a partial incoming save can never delete the
        last resumable checkpoint.
        """
        assert self._is_actor
        dist.barrier()
        if dist.get_rank() == 0:
            self.checkpoint_manager.register_checkpoint(local_path, max_ckpt_to_keep)
        dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got {self._is_actor} and {self._is_rollout}"
        )
        if local_path is None:
            if bool(self.config.rollout.get("hf_use_replicated_module", False)):
                self._hf_rollout_replica_dirty = True
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self.config.actor.get("training_mode", "legacy") != "full_parameter" and bool(
            self.config.model.get("train_native_visual_merger", False)
        ):
            merger_path = os.path.join(local_path, "native_visual_merger", "model.safetensors")
            manifest_path = os.path.join(local_path, "native_visual_merger", "manifest.json")
            if not os.path.isfile(merger_path) or not os.path.isfile(manifest_path):
                raise FileNotFoundError("Exact resume requires native_visual_merger model and manifest artifacts")
            saved_merger = load_file(merger_path, device="cpu")
            with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False):
                peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
                loaded_merger = _native_visual_merger_parameters(peft_model)
                if loaded_merger.keys() != saved_merger.keys():
                    raise RuntimeError("Native visual merger resume inventory differs from its checkpoint artifact")
                unequal = [
                    name
                    for name, expected in saved_merger.items()
                    if not torch.equal(loaded_merger[name].detach().cpu(), expected)
                ]
                if unequal:
                    raise RuntimeError(f"Native visual merger differs after checkpoint resume: {unequal[:20]}")
        if bool(self.config.rollout.get("hf_use_replicated_module", False)):
            self._hf_rollout_replica_dirty = True
        if self._is_rollout:
            if not hasattr(self, "torch_random_states") or not hasattr(self, "gen_random_states"):
                raise RuntimeError("Rollout checkpoint load requires initialized trainer and generation RNG states")
            rollout_rng_state = _load_rollout_rng_state(
                local_path, world_size=dist.get_world_size(), rank=dist.get_rank()
            )
            self.torch_random_states = rollout_rng_state["torch_random_states"]
            self.gen_random_states = rollout_rng_state["gen_random_states"]
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception:
                pass


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        self.config: FSDPCriticConfig = config
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)

    def _build_critic_model_optimizer(self, config):
        from torch.distributed.fsdp import MixedPrecision
        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        custom_chat_template = resolve_custom_chat_template(self.config.model)
        if custom_chat_template is not None:
            self.tokenizer.chat_template = custom_chat_template
            if self.processor is not None:
                self.processor.chat_template = custom_chat_template
                processor_tokenizer = getattr(self.processor, "tokenizer", None)
                if processor_tokenizer is None:
                    raise RuntimeError("Multimodal processor does not expose its bound tokenizer")
                processor_tokenizer.chat_template = custom_chat_template
                template_hashes = {
                    hashlib.sha256(value.encode("utf-8")).hexdigest()
                    for value in (
                        self.tokenizer.chat_template,
                        self.processor.chat_template,
                        processor_tokenizer.chat_template,
                    )
                }
                if len(template_hashes) != 1:
                    raise RuntimeError("Worker tokenizer/processor chat-template identity drift")
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        from transformers import AutoConfig

        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation=attn_implementation,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            critic_model_config.vision_config._attn_implementation = "eager"
        critic_model_config.num_labels = 1
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )
        tiled_mlp_config = config.model.get("tiled_mlp", {})
        use_tiled_mlp = tiled_mlp_config.get("enabled", False)
        tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)
        if use_tiled_mlp and config.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `critic.strategy=fsdp2`.")
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0
            critic_module = load_valuehead_model(
                local_path, torch_dtype, critic_model_config, config.model.get("trust_remote_code", False)
            )
            use_remove_padding = config.model.get("use_remove_padding", False)
            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )
            critic_module.to(torch_dtype)
            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to critic from: {lora_adapter_path}")
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))
                critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, is_trainable=True)
                peft_config = critic_module.peft_config["default"]
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.TOKEN_CLS
            else:
                lora_config = {
                    "task_type": TaskType.TOKEN_CLS,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))
        if self.rank == 0:
            print_model_size(critic_module)
        self.critic_model_config = critic_model_config
        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module, config=self.config.model.fsdp_config.wrap_policy, is_lora=self._is_lora
        )
        log_gpu_memory_usage("Before critic FSDP", logger=None)
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            elif self.rank == 0:
                print("[critic model] No vision tower found.")
        if config.strategy == "fsdp":
            print_fsdp_children(critic_module, "critic_module before FSDP")
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")
        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)
        log_gpu_memory_usage("After critic FSDP", logger=None)
        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)
        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                warmup_update_indexing=config.optim.get("lr_warmup_update_indexing", "zero_based_legacy_v1"),
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
                warmup_update_indexing=config.optim.get("lr_warmup_update_indexing", "zero_based_legacy_v1"),
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
        return (critic_module, critic_optimizer, critic_lr_scheduler)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get("external_lib", None))
        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)
        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()
            output = DataProto(batch=None, meta_info={"metrics": metrics})
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "reward", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))
        trust_remote_code = config.model.get("trust_remote_code", False)
        override_config = OmegaConf.to_container(OmegaConf.create(config.model.get("override_config", {})))
        model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=override_config.get("attn_implementation", "flash_attention_2"),
        )
        model_config.num_labels = 1
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
            )
            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        if config.strategy == "fsdp":
            print_fsdp_children(reward_module, "reward_module before FSDP")
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits
                rm_score = rm_score.squeeze(-1)
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores
        token_level_scores = token_level_scores[:, -response_length:]
        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]
        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer
        rm_input_ids = []
        rm_attention_mask = []
        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response = src_tokenizer.decode(valid_response_ids)
            response = response.replace(src_tokenizer.eos_token, "")
            chat.append({"role": "assistant", "content": response})
            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                print(f"Switch template. chat: {prompt_with_chat_template}")
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length
            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,
                truncation=self.config.get("truncation", "right"),
            )
            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)
        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)
        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}
        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown", role="compute_rm_score")
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)
        rm_data = rm_data.to(get_device_id())
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]
            token_level_scores = self._expand_to_token_level(data, scores)
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)
        output = output.to("cpu")
        return output


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
