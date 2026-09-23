# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import inspect
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Mapping
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pprint import pprint
from string import Template
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.batch_schedule import TrainingBatchPlan
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.chat_template import resolve_custom_chat_template
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def _nested_exact_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        if left.dtype.hasobject:
            return all(
                _nested_exact_equal(a, b)
                for a, b in zip(left.reshape(-1).tolist(), right.reshape(-1).tolist(), strict=True)
            )
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def _atomic_json(path: str, payload: dict) -> None:
    target = os.path.realpath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = f"{target}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _audit_exp3_migrated_step301(metrics: dict, rollout_data_dir: str | None) -> None:
    """Fail before step302 unless observable step301 behavior matches attempt1.

    Attempt1 did not persist per-image route tensors or a post-step RNG
    snapshot, so those fields are explicitly reported as unavailable rather
    than being falsely claimed exact.  Discrete behavior, route aggregates,
    optimizer state and learning rate remain exact.  Floating training
    objectives are allowed at most 1% fresh-process numerical drift, while
    rollout-correction diagnostics must stay inside an explicit safety
    envelope.  Every observed difference is retained in the report.
    """
    report_path = os.environ.get("VERL_EXP3_RESUME_REPLAY_REPORT")
    reference_root = os.environ.get("VERL_EXP3_RESUME_REPLAY_REFERENCE_OUTPUT")
    if not report_path and not reference_root:
        return
    if not report_path or not reference_root:
        raise RuntimeError("Exp3 resume replay requires report and reference output together")
    if int(metrics.get("training/global_step", -1)) != 301:
        return
    if not rollout_data_dir:
        raise RuntimeError("Exp3 resume replay requires rollout_data_dir")

    reference_root = os.path.realpath(reference_root)
    old_rollout_path = os.path.join(reference_root, "rollouts", "301.jsonl")
    new_rollout_path = os.path.join(os.path.realpath(rollout_data_dir), "301.jsonl")
    old_metrics_path = os.path.join(reference_root, "logs", "metrics.jsonl")
    if not all(os.path.isfile(path) for path in (old_rollout_path, new_rollout_path, old_metrics_path)):
        raise RuntimeError("Exp3 resume replay reference/current artifacts are missing")

    def jsonl(path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    old_rollouts, new_rollouts = jsonl(old_rollout_path), jsonl(new_rollout_path)
    old_metric_rows = [row for row in jsonl(old_metrics_path) if int(row.get("step", -1)) == 301]
    if len(old_metric_rows) != 1:
        raise RuntimeError(f"Expected exactly one attempt1 step301 metric row, got {len(old_metric_rows)}")
    old_metrics = old_metric_rows[0]["data"]

    included_prefixes = (
        "global_seqlen/", "compression/", "self_distillation/", "rollout_corr/",
        "actor/", "response_length", "response/", "prompt_length/", "training/",
    )
    old_scientific = {key: value for key, value in old_metrics.items() if key.startswith(included_prefixes)}
    new_scientific = {key: value for key, value in metrics.items() if key.startswith(included_prefixes)}
    metric_differences = {}
    for key in sorted(set(old_scientific) | set(new_scientific)):
        old, new = old_scientific.get(key, "<MISSING>"), new_scientific.get(key, "<MISSING>")
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            equal = bool(np.isfinite(old) and np.isfinite(new) and np.isclose(old, new, rtol=1e-7, atol=1e-8))
        else:
            equal = old == new
        if not equal:
            metric_differences[key] = {"attempt1": old, "resumed": new}

    def digest(path):
        result = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()

    bounded_objective_keys = {
        "actor/grad_norm", "actor/pg_loss", "actor/vopd_loss", "actor/vopd_loss_weighted",
        "self_distillation/raw_jsd_token_mean", "self_distillation/weighted_jsd_token_mean",
        "self_distillation/support_unique_mean", "rollout_corr/training_log_ppl",
        "rollout_corr/training_ppl", "rollout_corr/rollout_log_ppl", "rollout_corr/rollout_ppl",
    }
    diagnostic_keys = {
        "rollout_corr/chi2_seq", "rollout_corr/chi2_token", "rollout_corr/k3_kl", "rollout_corr/kl",
        "rollout_corr/log_ppl_abs_diff", "rollout_corr/log_ppl_diff", "rollout_corr/log_ppl_diff_max",
        "rollout_corr/log_ppl_diff_min", "rollout_corr/ppl_ratio",
    }
    bounded_objective_differences = {
        key: values for key, values in metric_differences.items()
        if key in bounded_objective_keys and not np.isclose(values["attempt1"], values["resumed"], rtol=0.01, atol=1e-6)
    }
    unexpected_exact_differences = {
        key: values for key, values in metric_differences.items()
        if key not in bounded_objective_keys | diagnostic_keys
    }
    def diagnostic_safe(metric_key, values):
        old, new = float(values["attempt1"]), float(values["resumed"])
        return {
            "rollout_corr/chi2_seq": max(abs(old), abs(new)) <= 0.5,
            "rollout_corr/chi2_token": max(abs(old), abs(new)) <= 0.02,
            "rollout_corr/k3_kl": max(abs(old), abs(new)) <= 0.01,
            "rollout_corr/kl": max(abs(old), abs(new)) <= 0.01,
            "rollout_corr/log_ppl_abs_diff": max(abs(old), abs(new)) <= 0.05,
            "rollout_corr/log_ppl_diff": max(abs(old), abs(new)) <= 0.05,
            "rollout_corr/log_ppl_diff_max": max(abs(old), abs(new)) <= 0.10,
            "rollout_corr/log_ppl_diff_min": max(abs(old), abs(new)) <= 0.10,
            "rollout_corr/ppl_ratio": 0.95 <= old <= 1.05 and 0.95 <= new <= 1.05,
        }[metric_key]
    unsafe_diagnostics = {
        key: values for key, values in metric_differences.items()
        if key in diagnostic_keys and not diagnostic_safe(key, values)
    }

    gates = {
        "rollout_record_count_exact": len(old_rollouts) == len(new_rollouts) == 16,
        "rollout_prompt_response_gold_stream_exact": old_rollouts == new_rollouts,
        "scientific_metric_keyset_exact": old_scientific.keys() == new_scientific.keys(),
        "bounded_training_objectives_within_one_percent": not bounded_objective_differences,
        "categorical_route_and_optimizer_metrics_exact": not unexpected_exact_differences,
        "rollout_correction_diagnostics_within_safety_envelope": not unsafe_diagnostics,
        "dynamic_route_aggregate_exact": all(
            old_scientific.get(key) == new_scientific.get(key)
            for key in old_scientific if key.startswith("compression/")
        ),
        "optimizer_step_and_lr_exact": all(
            old_scientific.get(key) == new_scientific.get(key)
            for key in ("actor/optimizer_steps", "actor/lr")
        ),
    }
    report = {
        "schema_version": "vision_opd_exp3_resume_step301_replay_v1",
        "passed": all(gates.values()),
        "failed_gates": sorted(key for key, value in gates.items() if not value),
        "gates": gates,
        "reference_output": reference_root,
        "reference_rollout_sha256": digest(old_rollout_path),
        "resumed_rollout_sha256": digest(new_rollout_path),
        "reference_metrics_sha256": digest(old_metrics_path),
        "scientific_metric_count": len(old_scientific),
        "metric_differences": metric_differences,
        "failed_bounded_objective_differences": bounded_objective_differences,
        "unexpected_exact_differences": unexpected_exact_differences,
        "unsafe_rollout_correction_diagnostics": unsafe_diagnostics,
        "comparison_tolerance": {
            "discrete_and_contract_metrics": "exact",
            "bounded_training_objectives": {"rtol": 0.01, "atol": 1e-6},
            "rollout_correction_diagnostics": "explicit safety envelope; never used as equality evidence",
        },
        "limitations": {
            "per_image_route_tensor": "unavailable in read-only attempt1 step301 artifacts",
            "uid_field": "unavailable; exact rendered prompt stream is compared instead",
            "post_step_rng_snapshot": "unavailable in read-only attempt1 step301 artifacts",
            "dataloader_and_rng_at_step300": "loaded by existing exact checkpoint restore checks",
        },
    }
    _atomic_json(report_path, report)
    if not report["passed"]:
        raise RuntimeError(f"Exp3 migrated step301 replay mismatch: {report['failed_gates']}")


def _atomic_json_file(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_dataproto_file(path: str, payload: DataProto) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    payload.save_to_disk(temporary)
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _exp3_rollout_exact_gates(expected: DataProto, actual: DataProto) -> dict[str, bool]:
    expected_tensor_keys = set(expected.batch.keys())
    actual_tensor_keys = set(actual.batch.keys())
    tensors_exact = expected_tensor_keys == actual_tensor_keys and all(
        _nested_exact_equal(expected.batch[key], actual.batch[key]) for key in expected_tensor_keys
    )
    required = {"prompts", "responses", "input_ids", "attention_mask", "position_ids", "rollout_log_probs"}
    routes_expected = (expected.non_tensor_batch or {}).get("dart_merge_routes")
    routes_actual = (actual.non_tensor_batch or {}).get("dart_merge_routes")
    routes_exact = (
        routes_expected is not None
        and routes_actual is not None
        and _nested_exact_equal(routes_expected, routes_actual)
    )
    return {
        "rollout_tensor_inventory_exact": expected_tensor_keys == actual_tensor_keys,
        "all_rollout_tensors_bitwise_exact": tensors_exact,
        "required_rollout_tensors_present": required <= expected_tensor_keys,
        "sampled_response_tokens_bitwise_exact": (
            "responses" in expected_tensor_keys
            and "responses" in actual_tensor_keys
            and _nested_exact_equal(expected.batch["responses"], actual.batch["responses"])
        ),
        "behavior_logprobs_bitwise_exact": (
            "rollout_log_probs" in expected_tensor_keys
            and "rollout_log_probs" in actual_tensor_keys
            and _nested_exact_equal(expected.batch["rollout_log_probs"], actual.batch["rollout_log_probs"])
        ),
        "cdpruner_routes_bitwise_exact": routes_exact,
        "dart_routes_bitwise_exact": routes_exact,
    }


def _v6_rollout_exact_gates(expected: DataProto, actual: DataProto) -> dict[str, bool]:
    """Exact next-rollout gates for the independent HoliTom-DPC schema."""

    expected_tensor_keys = set(expected.batch.keys())
    actual_tensor_keys = set(actual.batch.keys())
    tensors_exact = expected_tensor_keys == actual_tensor_keys and all(
        _nested_exact_equal(expected.batch[key], actual.batch[key]) for key in expected_tensor_keys
    )
    required = {"prompts", "responses", "input_ids", "attention_mask", "position_ids", "rollout_log_probs"}
    expected_non_tensors = expected.non_tensor_batch or {}
    actual_non_tensors = actual.non_tensor_batch or {}
    routes_expected = expected_non_tensors.get("dpc_merge_routes")
    routes_actual = actual_non_tensors.get("dpc_merge_routes")
    routes_exact = (
        routes_expected is not None
        and routes_actual is not None
        and _nested_exact_equal(routes_expected, routes_actual)
    )
    return {
        "rollout_tensor_inventory_exact": expected_tensor_keys == actual_tensor_keys,
        "all_rollout_tensors_bitwise_exact": tensors_exact,
        "required_rollout_tensors_present": required <= expected_tensor_keys,
        "sampled_response_tokens_bitwise_exact": (
            "responses" in expected_tensor_keys
            and "responses" in actual_tensor_keys
            and _nested_exact_equal(expected.batch["responses"], actual.batch["responses"])
        ),
        "behavior_logprobs_bitwise_exact": (
            "rollout_log_probs" in expected_tensor_keys
            and "rollout_log_probs" in actual_tensor_keys
            and _nested_exact_equal(expected.batch["rollout_log_probs"], actual.batch["rollout_log_probs"])
        ),
        "dpc_merge_routes_bitwise_exact": routes_exact,
        "legacy_dart_routes_absent": (
            "dart_merge_routes" not in expected_non_tensors
            and "dart_merge_routes" not in actual_non_tensors
        ),
    }


def _v8_rollout_exact_gates(expected: DataProto, actual: DataProto) -> dict[str, bool]:
    """Exact next-rollout gates for V8 CDPruner plus its virtual-open protocol.

    V8 deliberately retains ``dart_merge_routes`` as the public transport key,
    but every route carries the CDPruner identity triple and a UID-bound query
    audit.  Comparing the complete nested route payload therefore covers the
    selected indices, assignment, original M-RoPE anchors, semantic query, and
    sample binding without weakening the historical V6 checker.
    """

    expected_tensor_keys = set(expected.batch.keys())
    actual_tensor_keys = set(actual.batch.keys())
    tensors_exact = expected_tensor_keys == actual_tensor_keys and all(
        _nested_exact_equal(expected.batch[key], actual.batch[key]) for key in expected_tensor_keys
    )
    required = {"prompts", "responses", "input_ids", "attention_mask", "position_ids", "rollout_log_probs"}
    expected_non_tensors = expected.non_tensor_batch or {}
    actual_non_tensors = actual.non_tensor_batch or {}
    routes_expected = expected_non_tensors.get("dart_merge_routes")
    routes_actual = actual_non_tensors.get("dart_merge_routes")
    routes_exact = (
        routes_expected is not None
        and routes_actual is not None
        and _nested_exact_equal(routes_expected, routes_actual)
    )

    def exact_non_tensor(key: str) -> bool:
        return (
            key in expected_non_tensors
            and key in actual_non_tensors
            and _nested_exact_equal(expected_non_tensors[key], actual_non_tensors[key])
        )

    return {
        "rollout_tensor_inventory_exact": expected_tensor_keys == actual_tensor_keys,
        "all_rollout_tensors_bitwise_exact": tensors_exact,
        "required_rollout_tensors_present": required <= expected_tensor_keys,
        "sampled_response_tokens_bitwise_exact": (
            "responses" in expected_tensor_keys
            and "responses" in actual_tensor_keys
            and _nested_exact_equal(expected.batch["responses"], actual.batch["responses"])
        ),
        "behavior_logprobs_bitwise_exact": (
            "rollout_log_probs" in expected_tensor_keys
            and "rollout_log_probs" in actual_tensor_keys
            and _nested_exact_equal(expected.batch["rollout_log_probs"], actual.batch["rollout_log_probs"])
        ),
        "cdpruner_routes_and_query_audits_bitwise_exact": routes_exact,
        "archived_holitom_routes_absent": (
            "dpc_merge_routes" not in expected_non_tensors
            and "dpc_merge_routes" not in actual_non_tensors
        ),
        "semantic_stop_status_bitwise_exact": exact_non_tensor("rollout_protocol_status"),
        "semantic_stop_reason_bitwise_exact": exact_non_tensor("rollout_stop_reason"),
        "virtual_open_transport_prefix_bitwise_exact": exact_non_tensor(
            "rollout_response_transport_prefix"
        ),
    }


def _post_checkpoint_next_curriculum_state(curriculum: Any, checkpoint_step: int) -> dict[str, Any]:
    """Return the curriculum state for the first rollout after checkpoint S.

    Checkpoint ``S`` is written only after optimizer update ``S`` commits, so
    the next rollout is driven by ``completed_optimizer_steps=S``.  Keeping
    this boundary in one pure helper prevents resume evidence from silently
    replaying update S's pre-update state ``S-1``.
    """

    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise TypeError("checkpoint_step must be an integer")
    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be positive")
    return curriculum.runtime_state(checkpoint_step)


def _require_v6_single_finite_actor_update(
    actor_metrics: Mapping[str, Any], *, require_nonzero_lr: bool = False
) -> None:
    """Fail closed unless a formal V6 outer step performed one finite update.

    Formal V6 fixes ``ppo_epochs=1`` and the optimizer mini-batch to the full
    global batch, so every consumed outer batch must produce exactly one
    optimizer step.  ``DataParallelPPOActor`` deliberately skips an optimizer
    step after a non-finite gradient norm; without this controller-side gate,
    the dataloader and global step could still advance and silently undertrain.
    """

    required_finite_metrics = (
        "actor/grad_norm",
        "actor/vopd_loss",
        "actor/vopd_loss_weighted",
        "self_distillation/raw_jsd_token_mean",
        "self_distillation/weighted_jsd_token_mean",
    )
    for key in required_finite_metrics:
        if key not in actor_metrics:
            raise RuntimeError(f"Formal V6 actor update did not report required metric {key!r}")
        value = actor_metrics[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            raise TypeError(
                f"Formal V6 actor metric {key!r} must be a real scalar, "
                f"got {type(value).__name__}"
            )
        if not np.isfinite(float(value)):
            raise FloatingPointError(f"Formal V6 actor metric {key!r} is non-finite: {value!r}")

    optimizer_steps = actor_metrics.get("actor/optimizer_steps")
    if isinstance(optimizer_steps, (bool, np.bool_)) or not isinstance(
        optimizer_steps, (int, float, np.number)
    ):
        raise RuntimeError(
            "Formal V6 actor update must report numeric 'actor/optimizer_steps'; "
            f"got {optimizer_steps!r}"
        )
    if not np.isfinite(float(optimizer_steps)) or float(optimizer_steps) != 1.0:
        raise RuntimeError(
            "Formal V6 requires exactly one successful optimizer update for every consumed "
            f"outer batch; got actor/optimizer_steps={optimizer_steps!r}. Training is stopped "
            "before the global step, sampler, log, or checkpoint can advance."
        )
    if require_nonzero_lr:
        nonzero_steps = actor_metrics.get("actor/nonzero_lr_optimizer_steps")
        if isinstance(nonzero_steps, (bool, np.bool_)) or not isinstance(
            nonzero_steps, (int, float, np.number)
        ):
            raise RuntimeError(
                "V7 actor update must report numeric 'actor/nonzero_lr_optimizer_steps'; "
                f"got {nonzero_steps!r}"
            )
        if not np.isfinite(float(nonzero_steps)) or float(nonzero_steps) != 1.0:
            raise RuntimeError(
                "V7 requires exactly one optimizer update with finite, strictly positive "
                f"learning rates in every parameter group; got {nonzero_steps!r}"
            )

logger = logging.getLogger(__name__)


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def should_reuse_rollout_log_probs_as_old_log_probs(config, batch: DataProto) -> bool:
    """Reuse rollout log-probs only when rollout and update use the same HF backend.

    Batch counts and PPO epochs do not establish policy identity.  In particular,
    vLLM sampled-token log-probs are not HF-old log-probs even when the update has
    one epoch and exactly one mini-batch.
    """
    if "rollout_log_probs" not in batch.batch:
        return False

    if config.actor_rollout_ref.rollout.name != "hf":
        return False

    if config.actor_rollout_ref.actor.ppo_epochs != 1:
        return False
    rollout_config = config.actor_rollout_ref.rollout
    if (
        float(getattr(rollout_config, "temperature", 1.0)) != 1.0
        or float(getattr(rollout_config, "top_p", 1.0)) != 1.0
        or int(getattr(rollout_config, "top_k", 0)) > 0
    ):
        # HFRollout records the probability after sampling warpers.  That is
        # identical to the actor policy only for the production unwarped
        # distribution; otherwise retain the established recomputation path.
        return False
    rollout_log_probs = batch.batch["rollout_log_probs"]
    responses = batch.batch.get("responses")
    if responses is None or rollout_log_probs.shape != responses.shape:
        raise RuntimeError(
            "HF behavior log-probabilities must align with responses before they can be the PPO old snapshot: "
            f"log_probs={tuple(rollout_log_probs.shape)}, responses="
            f"{None if responses is None else tuple(responses.shape)}"
        )
    if not bool(torch.isfinite(rollout_log_probs).all().item()):
        raise FloatingPointError("HF behavior log-probabilities contain NaN or Inf")
    # All mini-batches were sampled before the first update.  The behavior
    # log-probability is therefore the correct frozen PPO old snapshot for
    # every mini-batch, including batch=16 split into two updates.  Recomputing
    # it with the FSDP training wrapper is both redundant and numerically less
    # faithful to the native HF replica that actually sampled the token.
    return True


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = config.actor_rollout_ref.actor.get("self_distillation", {}).get("reprompt_truncation", "error")
        self.processor = processor
        self.config = config
        custom_chat_template = resolve_custom_chat_template(self.config.actor_rollout_ref.model)
        if custom_chat_template is not None:
            if self.processor is not None:
                self.processor.chat_template = custom_chat_template
            self.tokenizer.chat_template = custom_chat_template
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        self.best_metric_key = self.config.trainer.get("best_metric_key", None)
        self.best_metric_mode = self.config.trainer.get("best_metric_mode", "max")
        if self.best_metric_mode not in {"max", "min"}:
            raise ValueError("trainer.best_metric_mode must be 'max' or 'min'")
        self.best_metric_value = None
        self.best_metric_step = None

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _is_ai4s_v6_full_parameter(self) -> bool:
        """Return whether this trainer is running the fail-closed V6 contract."""

        root = getattr(self.config, "actor_rollout_ref", None)
        if root is None and isinstance(self.config, Mapping):
            root = self.config.get("actor_rollout_ref")
        if root is None:
            return False
        actor = getattr(root, "actor", None)
        if actor is None and isinstance(root, Mapping):
            actor = root.get("actor")
        getter = getattr(actor, "get", None)
        return callable(getter) and getter("training_mode", "legacy") == "full_parameter"

    def _formal_compressor_algorithm(self) -> Optional[str]:
        """Return the selected full-parameter compressor identity.

        The historical helper above intentionally remains the admission gate for
        the V6/V7/V8 full-parameter checkpoint machinery.  It must not, however,
        be used as a method discriminator: V8 is also full-parameter but carries
        CDPruner routes rather than HoliTom-DPC routes.  Resolve the algorithm
        from both Hydra and the selected immutable contract and fail closed if
        those two independently projected identities differ.
        """

        if not self._is_ai4s_v6_full_parameter():
            return None
        compressor = self.config.actor_rollout_ref.model.get("vision_token_compressor", {})
        configured = compressor.get("algorithm")
        from training.contract import load_contract

        contract, _ = load_contract()
        contracted = contract["compressor"]["algorithm"]
        if configured != contracted:
            raise RuntimeError(
                "Full-parameter compressor algorithm differs from the selected static contract: "
                f"configured={configured!r}, contracted={contracted!r}"
            )
        return str(contracted)

    def _is_ai4s_v8_cdpruner(self) -> bool:
        return self._formal_compressor_algorithm() == "qwen35_cdpruner_v1"

    def _reject_v6_legacy_audit_controls(self) -> None:
        """Prevent a formal eight-rank run from entering archived Exp3 gates."""

        if not self._is_ai4s_v6_full_parameter():
            return
        legacy_environment = sorted(
            name
            for name in (
                "VERL_EXP3_FRESH_LOAD_AUDIT_DIR",
                "VERL_EXP3_NEXT_ROLLOUT_AUDIT_DIR",
                "VERL_EXP3_GATE_PARAMETER_AUDIT_DIR",
                "VERL_EXP3_RESUME_REPLAY_REPORT",
                "VERL_EXP3_RESUME_REPLAY_REFERENCE_OUTPUT",
            )
            if os.environ.get(name)
        )
        if legacy_environment:
            raise RuntimeError(
                "Formal V6 forbids archived four-rank Exp3 audit controls: "
                f"{legacy_environment}"
            )

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None and self.config.data.val_files:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        train_drop_last = bool(self.config.data.get("train_drop_last", True))
        train_batch_size = int(self.config.data.get("gen_batch_size", self.config.data.train_batch_size))
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            drop_last=train_drop_last,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        if self.val_dataset is not None:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_dataloader = None

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader) if self.val_dataloader else 0}"
        )

        exact_batch_plan = bool(self.config.data.get("enforce_exact_train_batch_plan", False))
        optimizer_training_steps = None
        if exact_batch_plan:
            rollout_n = int(self.config.actor_rollout_ref.rollout.n)
            if rollout_n <= 0:
                raise ValueError("enforce_exact_train_batch_plan requires a positive rollout.n")
            if self._is_ai4s_v6_full_parameter():
                from training.contract import load_contract

                formal_contract, _ = load_contract()
                expected_rollout_n = int(formal_contract["training"]["rollout_n"])
                if rollout_n != expected_rollout_n:
                    raise ValueError(
                        "Formal V6 exact prompt-batch plan requires "
                        f"rollout.n={expected_rollout_n}, got {rollout_n}"
                    )
            batch_plan = TrainingBatchPlan(
                dataset_size=len(self.train_dataset),
                train_batch_size=train_batch_size,
                ppo_mini_batch_size=int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size),
                world_size=int(self.resource_pool_manager.get_n_gpus()),
                epochs=int(self.config.trainer.total_epochs),
                drop_last=train_drop_last,
                rollout_n=rollout_n,
            )
            if len(self.train_dataloader) != batch_plan.outer_steps_per_epoch:
                raise RuntimeError(
                    "StatefulDataLoader length disagrees with the exact batch plan: "
                    f"loader={len(self.train_dataloader)}, plan={batch_plan.outer_steps_per_epoch}"
                )
            configured_steps = self.config.trainer.total_training_steps
            v6_smoke = (
                self.config.actor_rollout_ref.actor.get("training_mode", "legacy") == "full_parameter"
                and self.config.trainer.get("dart_smoke_audit_dir", None) is not None
            )
            if configured_steps is not None and int(configured_steps) != batch_plan.total_outer_steps:
                configured_steps = int(configured_steps)
                if not (
                    v6_smoke
                    and configured_steps in {1, 10}
                    and configured_steps < batch_plan.total_outer_steps
                ):
                    raise ValueError(
                        "trainer.total_training_steps disagrees with the exact full-data plan: "
                        f"configured={configured_steps}, required={batch_plan.total_outer_steps}"
                    )
                # A ten-step V6 smoke consumes the first ten exact balanced
                # batches from the same stateful sampler and scientific data
                # plan.  Only the isolated horizon is shortened.
                total_training_steps = configured_steps
                optimizer_training_steps = configured_steps
            else:
                total_training_steps = batch_plan.total_outer_steps
                optimizer_training_steps = batch_plan.total_optimizer_steps
            self.training_batch_plan = batch_plan
            print(
                "Exact training batch plan: "
                f"samples={batch_plan.samples_per_epoch * batch_plan.epochs}, "
                f"outer_steps={total_training_steps}, optimizer_steps={optimizer_training_steps}, "
                f"rollout_n={batch_plan.rollout_n}, "
                f"trajectories_per_full_update={batch_plan.trajectories_per_full_outer_step}, "
                f"total_trajectories={batch_plan.total_trajectories}, "
                f"final_batch={batch_plan.final_batch_size}, drop_last={batch_plan.drop_last}"
            )
        else:
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            if self.config.trainer.total_training_steps is not None:
                total_training_steps = self.config.trainer.total_training_steps
            optimizer_training_steps = total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = optimizer_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _decode_v8_rollout_rows(self, batch: DataProto) -> tuple[list[str], list[str], dict[str, list]]:
        """Decode only sampled actions and reconstruct prompt-owned transport.

        This path is selected by the immutable V8 static schema, never by a
        template-name heuristic.  The opening delimiter remains presentation
        context and is never included in sampled token IDs or log-probability
        accounting.
        """

        from training.contract import load_contract
        from verl.utils.response_protocol import (
            PROTOCOL_MISSING_CLOSE,
            PROTOCOL_NESTED_OPEN,
            classify_answer_protocol,
        )

        contract, _ = load_contract()
        if contract.get("schema_version") != "vision_opd_ai4s_v8_fullimage_curriculum_static_contract_v1":
            raise RuntimeError("V8 rollout decoder was invoked for a non-V8 static contract")
        semantic = contract["rollout"]["semantic_stop"]
        prefix = semantic["transport_prefix"]
        rollout = self.config.actor_rollout_ref.rollout
        if (
            not bool(rollout.get("semantic_stop_virtual_open", False))
            or str(rollout.get("semantic_stop_transport_prefix", "")) != prefix
        ):
            raise RuntimeError("Resolved rollout config lost the V8 virtual-open protocol")

        responses = batch.batch["responses"]
        response_mask = batch.batch.get("response_mask")
        if response_mask is None or response_mask.shape != responses.shape:
            raise RuntimeError("V8 rollout logging requires an explicit response-aligned mask")
        mask = response_mask.to(torch.bool)
        if not torch.equal(response_mask, mask.to(response_mask.dtype)):
            raise RuntimeError("V8 response_mask must be binary")
        if mask.shape[1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any().item()):
            raise RuntimeError("V8 response_mask must be a contiguous sampled prefix")
        attention = batch.batch.get("attention_mask")
        if attention is None or attention.shape[0] != responses.shape[0]:
            raise RuntimeError("V8 rollout logging requires a batch-aligned attention mask")
        if not torch.equal(attention[:, -responses.shape[1] :].to(torch.bool), mask):
            raise RuntimeError("V8 response_mask differs from the response attention suffix")

        non_tensors = batch.non_tensor_batch or {}
        prefixes = np.asarray(non_tensors.get("rollout_response_transport_prefix"), dtype=object)
        statuses = np.asarray(non_tensors.get("rollout_protocol_status"), dtype=object)
        stop_reasons = np.asarray(non_tensors.get("rollout_stop_reason"), dtype=object)
        expected_shape = (responses.shape[0],)
        for name, values in (
            ("rollout_response_transport_prefix", prefixes),
            ("rollout_protocol_status", statuses),
            ("rollout_stop_reason", stop_reasons),
        ):
            if values.shape != expected_shape:
                raise RuntimeError(f"V8 {name} is not trajectory aligned")
        if any(str(value) != prefix for value in prefixes):
            raise RuntimeError("V8 rollout transport prefix drifted across trajectories")

        continuations: list[str] = []
        protocols: list[str] = []
        token_ids: list[list[int]] = []
        offline_statuses: list[str] = []
        strict_valid: list[bool] = []
        for row in range(responses.shape[0]):
            ids = responses[row][mask[row]].detach().cpu().tolist()
            continuation = self.tokenizer.decode(
                ids,
                # This is the forensic sampled-action record.  Preserve EOS
                # and every other special token so malformed early stops are
                # not made to look like ordinary text by the logger.
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            classified = classify_answer_protocol(continuation, virtual_open=True)
            reported = str(statuses[row])
            if reported == "not_closed":
                # ``not_closed`` is the online termination observation: the
                # decoder never confirmed an exact closing delimiter before
                # EOS/length.  It is not an offline strict-format category.
                # In particular, a model may redundantly sample ``<answer>``
                # and then hit EOS without ever closing; the exact offline
                # diagnosis is ``nested_open``, while ``not_closed`` remains
                # the correct online state.  Check the property that semantic
                # stopping actually guarantees instead of conflating the two
                # classifications.
                if semantic["stop_string"] in continuation:
                    raise RuntimeError(
                        "A not_closed rollout contains the exact closing delimiter"
                    )
                if classified.status not in {
                    PROTOCOL_MISSING_CLOSE,
                    PROTOCOL_NESTED_OPEN,
                }:
                    raise RuntimeError(
                        "A not_closed rollout has an impossible offline protocol classification: "
                        f"{classified.status!r}"
                    )
            elif reported != classified.status:
                raise RuntimeError(
                    "Rollout semantic status differs from independent protocol classification: "
                    f"reported={reported!r}, classified={classified.status!r}"
                )
            token_ids.append([int(value) for value in ids])
            continuations.append(continuation)
            protocols.append(classified.reconstructed_text)
            offline_statuses.append(classified.status)
            strict_valid.append(classified.strict_valid)
        return continuations, protocols, {
            "sampled_continuation": continuations,
            "protocol_response": protocols,
            "response_token_ids": token_ids,
            "response_token_count": [len(values) for values in token_ids],
            "transport_prefix": [prefix] * len(continuations),
            "transport_prefix_sampled": [False] * len(continuations),
            # Public consumers receive the independently recomputed strict
            # whole-response classification.  Preserve the online stopping
            # state separately: ``not_closed`` is an execution observation,
            # not a public protocol category, and conflating the two made
            # capped/EOS rows impossible for the panel to validate.
            "protocol_status": offline_statuses,
            "semantic_stop_protocol_status": [str(value) for value in statuses],
            "offline_protocol_classification": offline_statuses,
            "strict_protocol_valid": strict_valid,
            "stop_reason": [str(value) for value in stop_reasons],
        }

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            from training.contract import load_contract

            static_contract, _ = load_contract()
            is_v8 = static_contract.get("schema_version") == "vision_opd_ai4s_v8_fullimage_curriculum_static_contract_v1"
            if is_v8:
                prompt_ids = batch.batch["prompts"]
                prompt_mask = batch.batch["attention_mask"][:, : prompt_ids.shape[1]].to(torch.bool)
                inputs = [
                    self.tokenizer.decode(
                        prompt_ids[row][prompt_mask[row]].detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for row in range(prompt_ids.shape[0])
                ]
                _, outputs, protocol_dump = self._decode_v8_rollout_rows(batch)
            else:
                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                protocol_dump = {}
            score_tensor = batch.batch.get("token_level_scores")
            scores = score_tensor.sum(-1).cpu().tolist() if score_tensor is not None else [None] * len(batch)
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            reward_extra_infos_to_dump.update(protocol_dump)
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            for key in ("uid", "data_source", "sampling_bucket", "task_type"):
                if key in batch.non_tensor_batch:
                    values = batch.non_tensor_batch[key]
                    if isinstance(values, np.ndarray):
                        values = values.tolist()
                    if len(values) == len(batch):
                        reward_extra_infos_to_dump.setdefault(key, list(values))
            if is_v8 and "uid" in reward_extra_infos_to_dump:
                occurrences: dict[str, int] = {}
                trajectory_indices = []
                for raw_uid in reward_extra_infos_to_dump["uid"]:
                    uid = str(raw_uid)
                    trajectory_indices.append(occurrences.get(uid, 0))
                    occurrences[uid] = occurrences.get(uid, 0) + 1
                reward_extra_infos_to_dump["trajectory_index_within_uid"] = trajectory_indices

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    @staticmethod
    def _audit_tensor_summary(value: torch.Tensor) -> dict[str, Any]:
        """Return a reproducible byte hash without changing the source tensor."""
        cpu_value = value.detach().contiguous().cpu()
        byte_view = cpu_value.view(torch.uint8)
        return {
            "shape": list(cpu_value.shape),
            "dtype": str(cpu_value.dtype),
            "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
        }

    @classmethod
    def _audit_input_summary(cls, value: Any) -> Any:
        """Hash large multimodal inputs while preserving their nested contract."""
        if isinstance(value, torch.Tensor):
            return {"kind": "tensor", **cls._audit_tensor_summary(value)}
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [cls._audit_input_summary(item) for item in value.tolist()]
            contiguous = np.ascontiguousarray(value)
            return {
                "kind": "ndarray",
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            }
        if isinstance(value, Image.Image):
            return {
                "kind": "PIL.Image",
                "mode": value.mode,
                "size": list(value.size),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {str(key): cls._audit_input_summary(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [cls._audit_input_summary(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"kind": type(value).__qualname__, "repr": repr(value)}

    @staticmethod
    def _validated_route_query_audit(value: Any) -> dict[str, Any]:
        """Validate and normalize the semantic provenance attached to one route."""

        from verl.utils.route_query import (
            ROUTE_QUERY_POLICY,
            ROUTE_QUERY_SCHEMA_VERSION_V1,
            ROUTE_QUERY_SCHEMA_VERSION_V2,
            SUPPORTED_ROUTE_QUERY_SCHEMAS,
        )

        if not isinstance(value, Mapping):
            raise ValueError("query_audit must be a mapping")
        audit = deepcopy(dict(value))
        route_schema = audit.get("schema_version")
        if route_schema not in SUPPORTED_ROUTE_QUERY_SCHEMAS:
            raise ValueError("query_audit has an unsupported schema_version")
        if audit.get("query_policy") != ROUTE_QUERY_POLICY:
            raise ValueError("query_audit has an unsupported query_policy")

        source = audit.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("query_audit source must be a non-empty string")
        canonical_text = audit.get("canonical_text")
        if not isinstance(canonical_text, str) or not canonical_text.strip():
            raise ValueError("query_audit canonical_text must be a non-empty string")
        canonical_sha256 = audit.get("canonical_sha256")
        expected_sha256 = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        if canonical_sha256 != expected_sha256:
            raise ValueError("query_audit canonical_sha256 does not match canonical_text")

        segments = audit.get("segments")
        if not isinstance(segments, (list, tuple)) or any(
            not isinstance(segment, str) or not segment.strip() for segment in segments
        ):
            raise ValueError("query_audit segments must contain non-empty semantic strings")
        if route_schema == ROUTE_QUERY_SCHEMA_VERSION_V1 and len(segments) != 5:
            raise ValueError("V1 query_audit segments must contain one question and options A-D")
        if route_schema == ROUTE_QUERY_SCHEMA_VERSION_V2 and len(segments) < 1:
            raise ValueError("V2 query_audit must contain at least one question segment")
        normalized_segments = [segment.strip() for segment in segments]
        if "\n".join(normalized_segments) != canonical_text:
            raise ValueError("query_audit segments do not reconstruct canonical_text")

        def integer_list(field: str) -> list[int]:
            raw = audit.get(field)
            if isinstance(raw, np.ndarray):
                raw = raw.tolist()
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValueError(f"query_audit {field} must be a non-empty integer sequence")
            if any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
                for item in raw
            ):
                raise ValueError(f"query_audit {field} must contain only integers")
            normalized = [int(item) for item in raw]
            if any(item < 0 for item in normalized):
                raise ValueError(f"query_audit {field} must contain non-negative integers")
            return normalized

        selected_token_indices = integer_list("selected_token_indices")
        selected_token_ids = integer_list("selected_token_ids")
        if len(selected_token_indices) != len(selected_token_ids):
            raise ValueError("query_audit selected_token_indices/token_ids must have equal length")
        if selected_token_indices != sorted(set(selected_token_indices)):
            raise ValueError("query_audit selected_token_indices must be strictly increasing and unique")
        selected_token_count = audit.get("selected_token_count")
        if (
            isinstance(selected_token_count, (bool, np.bool_))
            or not isinstance(selected_token_count, (int, np.integer))
            or int(selected_token_count) != len(selected_token_indices)
        ):
            raise ValueError("query_audit selected_token_count does not match selected token provenance")

        audit["segments"] = normalized_segments
        audit["selected_token_indices"] = selected_token_indices
        audit["selected_token_ids"] = selected_token_ids
        audit["selected_token_count"] = len(selected_token_indices)
        try:
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("query_audit must remain JSON-serializable for the smoke summary") from exc
        return audit

    @staticmethod
    def _validated_v8_smoke_protocol_evidence(
        batch: DataProto,
        expected_count: int,
    ) -> dict[str, list[str]]:
        """Return exact trajectory-aligned V8 answer-protocol evidence."""

        protocol_evidence: dict[str, list[str]] = {}
        for key in (
            "rollout_protocol_status",
            "rollout_stop_reason",
            "rollout_response_transport_prefix",
        ):
            raw_values = batch.non_tensor_batch.get(key)
            if isinstance(raw_values, np.ndarray):
                raw_values = raw_values.tolist()
            if (
                not isinstance(raw_values, (list, tuple))
                or len(raw_values) != expected_count
                or any(not isinstance(value, str) or not value for value in raw_values)
            ):
                raise RuntimeError(f"V8 smoke audit requires sample-aligned {key}")
            protocol_evidence[key] = list(raw_values)
        if set(protocol_evidence["rollout_response_transport_prefix"]) != {"<answer>"}:
            raise RuntimeError("V8 smoke audit found a non-canonical virtual-open prefix")
        return protocol_evidence

    def _dump_dart_smoke_audit(self, batch: DataProto, audit_dir: str) -> None:
        """Persist exact pre-update tensors/routes for a CDPruner/legacy smoke."""
        required_tensors = {
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "rollout_log_probs",
            "rollout_is_weights",
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
            "teacher_response_start_idx",
            "self_distillation_mask",
        }
        missing = sorted(required_tensors.difference(batch.batch.keys()))
        if missing:
            raise RuntimeError(f"CDPruner smoke audit is missing required tensors: {missing}")
        if "dart_merge_routes" not in batch.non_tensor_batch:
            raise RuntimeError("CDPruner smoke audit requires rollout-provided legacy-schema dart_merge_routes")

        tensor_payload = {
            key: value.detach().cpu()
            for key, value in batch.batch.items()
            if isinstance(value, torch.Tensor)
            and (key in required_tensors or key in {"prompts", "response_start_idx"})
        }
        if tensor_payload["old_log_probs"].shape != tensor_payload["rollout_log_probs"].shape:
            raise RuntimeError("CDPruner smoke audit found mismatched HF-old/rollout log-prob shapes")

        from verl.models.transformers.vision_token_compressor import DARTMergeRoute

        raw_route_samples = batch.non_tensor_batch["dart_merge_routes"]
        if isinstance(raw_route_samples, np.ndarray):
            raw_route_samples = raw_route_samples.tolist()
        compressor_algorithm = "qwen35_cdpruner_v1"
        try:
            compressor_algorithm = self.config.actor_rollout_ref.model.vision_token_compressor.algorithm
        except (AttributeError, KeyError, TypeError):
            pass
        route_samples = []
        route_summary = []
        route_core_keys = {
            "selected_indices",
            "assignment",
            "source_counts",
            "original_tokens",
            "output_tokens",
            "anchor_coordinates",
            "schema_version",
            "algorithm",
            "method",
        }
        for sample_index, raw_sample in enumerate(raw_route_samples):
            if isinstance(raw_sample, np.ndarray):
                raw_sample = raw_sample.tolist()
            if isinstance(raw_sample, (dict, DARTMergeRoute)):
                raw_sample = [raw_sample]
            serialized_sample = []
            summarized_sample = []
            for route_index, raw_route in enumerate(raw_sample):
                route = raw_route if isinstance(raw_route, DARTMergeRoute) else DARTMergeRoute.from_dict(raw_route)
                route.validate_for_algorithm(str(compressor_algorithm))
                if not isinstance(raw_route, Mapping):
                    raise ValueError(
                        "CDPruner smoke routes must retain serialized query_audit provenance; "
                        f"sample={sample_index}, route={route_index}"
                    )
                non_string_extension_keys = [
                    key for key in raw_route if key not in route_core_keys and not isinstance(key, str)
                ]
                if non_string_extension_keys:
                    raise ValueError(
                        "CDPruner smoke route extension keys must be strings: "
                        f"sample={sample_index}, route={route_index}, keys={non_string_extension_keys!r}"
                    )
                extensions = {
                    key: deepcopy(value) for key, value in raw_route.items() if key not in route_core_keys
                }
                try:
                    extensions["query_audit"] = self._validated_route_query_audit(extensions.get("query_audit"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "CDPruner smoke route has invalid query_audit provenance: "
                        f"sample={sample_index}, route={route_index}: {exc}"
                    ) from exc
                serialized = route.as_dict(cpu=True)
                serialized.update(extensions)
                serialized_sample.append(serialized)
                summarized = {
                    "original_tokens": route.original_tokens,
                    "output_tokens": route.output_tokens,
                    "selected_indices": self._audit_tensor_summary(serialized["selected_indices"]),
                    "assignment": self._audit_tensor_summary(serialized["assignment"]),
                    "source_counts": self._audit_tensor_summary(serialized["source_counts"]),
                    "anchor_coordinates": self._audit_tensor_summary(serialized["anchor_coordinates"]),
                }
                summarized.update(
                    {key: self._audit_input_summary(value) for key, value in sorted(extensions.items())}
                )
                summarized_sample.append(summarized)
            if not serialized_sample:
                raise RuntimeError("CDPruner smoke audit encountered an empty route sample")
            route_samples.append(serialized_sample)
            route_summary.append(summarized_sample)

        multimodal_summary = self._audit_input_summary(batch.non_tensor_batch.get("multi_modal_inputs"))
        os.makedirs(audit_dir, exist_ok=True)
        stem = f"pre_update_step_{int(self.global_steps)}"
        tensor_path = os.path.join(audit_dir, f"{stem}.pt")
        summary_path = os.path.join(audit_dir, f"{stem}.json")
        if os.path.exists(tensor_path) or os.path.exists(summary_path):
            raise FileExistsError(f"Refusing to overwrite an existing CDPruner smoke audit artifact: {stem}")

        artifact = {
            "schema_version": "vision_opd_cdpruner_smoke_v1",
            "vision_token_compressor_algorithm": str(compressor_algorithm),
            "global_step": int(self.global_steps),
            "tensors": tensor_payload,
            "dart_merge_routes": route_samples,
            "multi_modal_input_summary": multimodal_summary,
        }
        tensor_tmp = f"{tensor_path}.tmp-{os.getpid()}"
        summary_tmp = f"{summary_path}.tmp-{os.getpid()}"
        torch.save(artifact, tensor_tmp)
        summary = {
            "schema_version": "vision_opd_cdpruner_smoke_v1",
            "vision_token_compressor_algorithm": str(compressor_algorithm),
            "global_step": int(self.global_steps),
            "tensor_summaries": {
                key: self._audit_tensor_summary(value) for key, value in sorted(tensor_payload.items())
            },
            "dart_merge_routes": route_summary,
            "multi_modal_input_summary": multimodal_summary,
        }
        with open(summary_tmp, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tensor_tmp, tensor_path)
        os.replace(summary_tmp, summary_path)

    def _dump_v6_dpc_smoke_audit(self, batch: DataProto, audit_dir: str) -> None:
        """Persist method-specific V6/V7/V8 full-parameter smoke evidence."""

        from training.contract import load_contract

        release_contract, _ = load_contract()
        compressor_algorithm = self._formal_compressor_algorithm()
        is_v8 = compressor_algorithm == "qwen35_cdpruner_v1"
        if compressor_algorithm not in {
            "qwen35_holitom_dpc_spatial_merge_v1",
            "qwen35_cdpruner_v1",
        }:
            raise RuntimeError(
                "Full-parameter smoke producer does not support compressor algorithm "
                f"{compressor_algorithm!r}"
            )

        smoke_steps = int(self.config.trainer.total_training_steps)
        if smoke_steps not in {1, 10}:
            raise RuntimeError("Formal full-parameter smoke producer is restricted to a 1/10-step isolated run")
        if int(self.global_steps) < 1 or int(self.global_steps) > smoke_steps:
            raise RuntimeError(
                "Formal full-parameter smoke producer step is outside its isolated horizon: "
                f"global_step={self.global_steps}, horizon={smoke_steps}"
            )

        curriculum_state = None
        if is_v8:
            from verl.models.transformers.visual_token_curriculum import (
                VisualTokenCurriculum,
            )

            curriculum = VisualTokenCurriculum.from_mapping(
                release_contract["compressor"]["curriculum"]
            )
            curriculum_state = curriculum.runtime_state(int(self.global_steps) - 1)

        required_tensors = {
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
            "teacher_response_start_idx",
            "self_distillation_mask",
        }
        if is_v8:
            required_tensors.update({"old_log_probs", "rollout_log_probs", "rollout_is_weights"})
        missing = sorted(required_tensors.difference(batch.batch.keys()))
        if missing:
            raise RuntimeError(f"Formal smoke audit is missing required tensors: {missing}")
        route_key = "dart_merge_routes" if is_v8 else "dpc_merge_routes"
        forbidden_route_key = "dpc_merge_routes" if is_v8 else "dart_merge_routes"
        if route_key not in batch.non_tensor_batch:
            raise RuntimeError(
                f"Formal {compressor_algorithm} smoke audit requires rollout-provided {route_key}"
            )
        if forbidden_route_key in batch.non_tensor_batch:
            raise RuntimeError(
                f"Formal {compressor_algorithm} smoke audit forbids {forbidden_route_key}"
            )

        from verl.models.transformers.vision_token_compressor import (
            DARTMergeRoute,
            HoliTomDPCSpatialMergeRoute,
            validate_cdpruner_curriculum_route,
        )

        raw_route_samples = batch.non_tensor_batch[route_key]
        if isinstance(raw_route_samples, np.ndarray):
            raw_route_samples = raw_route_samples.tolist()
        protocol_evidence = (
            self._validated_v8_smoke_protocol_evidence(batch, len(raw_route_samples))
            if is_v8
            else {}
        )
        sample_uids = batch.non_tensor_batch.get("uid") if is_v8 else None
        if is_v8 and (sample_uids is None or len(sample_uids) != len(raw_route_samples)):
            raise RuntimeError("V8 CDPruner smoke routes require one batch-aligned immutable UID")
        route_samples = []
        route_summary = []
        for sample_index, raw_sample in enumerate(raw_route_samples):
            if isinstance(raw_sample, np.ndarray):
                raw_sample = raw_sample.tolist()
            if isinstance(raw_sample, (dict, DARTMergeRoute, HoliTomDPCSpatialMergeRoute)):
                raw_sample = [raw_sample]
            if not isinstance(raw_sample, (list, tuple)) or not raw_sample:
                raise RuntimeError(f"Formal smoke route sample {sample_index} is empty or invalid")
            serialized_sample = []
            summarized_sample = []
            for route_index, raw_route in enumerate(raw_sample):
                if not isinstance(raw_route, Mapping):
                    raise RuntimeError(
                        "Formal routes must remain serialized mappings so provenance cannot be dropped: "
                        f"sample={sample_index}, route={route_index}"
                    )
                if is_v8:
                    core_keys = {
                        "schema_version",
                        "algorithm",
                        "method",
                        "selected_indices",
                        "assignment",
                        "source_counts",
                        "original_tokens",
                        "output_tokens",
                        "anchor_coordinates",
                        "retention_bps",
                        "curriculum_completed_steps",
                        "curriculum_schedule_sha256",
                    }
                    if set(raw_route) != core_keys | {"query_audit"}:
                        raise RuntimeError(
                            "V8 CDPruner route has a non-canonical field inventory: "
                            f"sample={sample_index}, route={route_index}, keys={sorted(raw_route)}"
                        )
                    route = DARTMergeRoute.from_dict(raw_route)
                    validate_cdpruner_curriculum_route(route, curriculum_state)
                    query_audit = self._validated_route_query_audit(raw_route.get("query_audit"))
                    expected_uid = str(sample_uids[sample_index])
                    if not expected_uid or query_audit.get("sample_uid") != expected_uid:
                        raise RuntimeError(
                            "V8 CDPruner query audit is not bound to its exact rollout UID: "
                            f"sample={sample_index}, route={route_index}"
                        )
                else:
                    expected_route_keys = {
                        "schema_version",
                        "algorithm",
                        "method",
                        "center_indices",
                        "assignment",
                        "source_counts",
                        "original_tokens",
                        "output_tokens",
                        "anchor_coordinates",
                    }
                    if set(raw_route) != expected_route_keys:
                        raise RuntimeError(
                            "V6/V7 DPC smoke route has a non-canonical field inventory: "
                            f"sample={sample_index}, route={route_index}, keys={sorted(raw_route)}"
                        )
                    route = HoliTomDPCSpatialMergeRoute.from_dict(raw_route)
                    route.validate()
                    query_audit = None
                if route.anchor_coordinates is None:
                    raise RuntimeError("Formal smoke route must bind selected original M-RoPE coordinates")
                if not is_v8:
                    expected_output = min(
                        route.original_tokens,
                        max(32, (route.original_tokens + 19) // 20),
                    )
                    if route.output_tokens != expected_output:
                        raise RuntimeError(
                            "Formal smoke route violates K=min(N,max(32,ceil(0.05N))): "
                            f"sample={sample_index}, route={route_index}, "
                            f"N={route.original_tokens}, K={route.output_tokens}, "
                            f"expected={expected_output}"
                        )
                serialized = route.as_dict(cpu=True)
                if is_v8:
                    serialized["query_audit"] = query_audit
                serialized_sample.append(serialized)
                index_key = "selected_indices" if is_v8 else "center_indices"
                route_item = {
                    "schema_version": serialized["schema_version"],
                    "algorithm": serialized["algorithm"],
                    "method": serialized["method"],
                    "original_tokens": route.original_tokens,
                    "output_tokens": route.output_tokens,
                    index_key: self._audit_tensor_summary(serialized[index_key]),
                    "assignment": self._audit_tensor_summary(serialized["assignment"]),
                    "source_counts": self._audit_tensor_summary(serialized["source_counts"]),
                    "anchor_coordinates": self._audit_tensor_summary(serialized["anchor_coordinates"]),
                }
                if is_v8:
                    route_item["query_audit"] = self._audit_input_summary(query_audit)
                    route_item.update(
                        {
                            "retention_bps": route.retention_bps,
                            "curriculum_completed_steps": route.curriculum_completed_steps,
                            "curriculum_schedule_sha256": route.curriculum_schedule_sha256,
                        }
                    )
                summarized_sample.append(route_item)
            route_samples.append(serialized_sample)
            route_summary.append(summarized_sample)

        tensor_payload = {
            key: value.detach().cpu()
            for key, value in batch.batch.items()
            if isinstance(value, torch.Tensor)
            and (key in required_tensors or key in {"prompts", "response_start_idx", "old_log_probs"})
        }
        multimodal_summary = self._audit_input_summary(batch.non_tensor_batch.get("multi_modal_inputs"))
        selected_profile = os.environ.get("VERL_V6_SELECTED_PROFILE")
        launch_contract_sha256 = os.environ.get("VERL_V6_LAUNCH_CONTRACT_SHA256")
        preflight_sha256 = os.environ.get("VERL_V6_PREFLIGHT_SHA256")
        run_root = os.environ.get("VERL_V6_RUN_ROOT")
        canonical_profiles = {
            str(profile["name"])
            for profile in release_contract["hardware"]["micro_profiles_descending"]
        }
        if selected_profile not in canonical_profiles:
            raise RuntimeError("V6 DPC smoke producer has no canonical selected profile")
        for label, value in (
            ("launch contract", launch_contract_sha256),
            ("preflight", preflight_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeError(f"V6 DPC smoke producer {label} hash is invalid")
        if not isinstance(run_root, str) or not os.path.isabs(run_root):
            raise RuntimeError("V6 DPC smoke producer run root must be absolute")
        producer_context = {
            "release_variant": release_contract["release_variant"],
            "algorithm": release_contract["compressor"]["algorithm"],
            "profile": selected_profile,
            "max_pixels": int(self.config.actor_rollout_ref.model.processor_max_pixels),
            "max_composite_cost_per_gpu": int(
                self.config.actor_rollout_ref.actor.vision_packing.max_cost_per_gpu
            ),
            "smoke_steps": smoke_steps,
            "run_root": os.path.abspath(run_root),
            "launch_contract_sha256": launch_contract_sha256,
            "preflight_sha256": preflight_sha256,
            **({"visual_token_curriculum_state": curriculum_state} if is_v8 else {}),
        }
        route_count = sum(len(sample) for sample in route_samples)
        producer_gates = {
            "formal_full_parameter_mode": (
                self.config.actor_rollout_ref.actor.training_mode == "full_parameter"
            ),
            "world_size_eight": int(self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes) == 8,
            (
                "cdpruner_route_inventory_nonempty"
                if is_v8
                else "dpc_route_inventory_nonempty"
            ): route_count > 0,
            (
                "archived_holitom_routes_absent"
                if is_v8
                else "legacy_routes_absent"
            ): forbidden_route_key not in batch.non_tensor_batch,
            "audit_path_bound_to_run_root": (
                os.path.abspath(audit_dir) == os.path.join(os.path.abspath(run_root), "audit")
            ),
            "checkpoint_path_bound_to_run_root": (
                os.path.abspath(self.config.trainer.default_local_dir)
                == os.path.join(os.path.abspath(run_root), "checkpoints")
            ),
            "fixed_dense_teacher_configured": (
                self.config.actor_rollout_ref.actor.self_distillation.teacher_model_source == "fixed"
                and self.config.actor_rollout_ref.actor.self_distillation.teacher_regularization == "fixed"
                and self.config.actor_rollout_ref.actor.self_distillation.teacher_visual_compression_mode == "dense"
            ),
            "six_group_optimizer_declared": (
                float(self.config.actor_rollout_ref.actor.optim.lr)
                == float(release_contract["optimizer"]["language_lr"])
                and float(self.config.actor_rollout_ref.actor.optim.vision_lr)
                == float(release_contract["optimizer"]["vision_lr"])
                and float(self.config.actor_rollout_ref.actor.optim.merger_lr)
                == float(release_contract["optimizer"]["native_visual_merger_lr"])
            ),
        }
        if not all(producer_gates.values()):
            raise RuntimeError(f"Formal smoke producer context gates failed: {producer_gates}")
        os.makedirs(audit_dir, exist_ok=True)
        stem = f"{'v8' if is_v8 else 'v6'}_pre_update_step_{int(self.global_steps)}"
        tensor_path = os.path.join(audit_dir, f"{stem}.pt")
        summary_path = os.path.join(audit_dir, f"{stem}.json")
        if os.path.exists(tensor_path) or os.path.exists(summary_path):
            raise FileExistsError(f"Refusing to overwrite an existing formal smoke artifact: {stem}")
        artifact = {
            "schema_version": (
                "vision_opd_ai4s_v8_fullparam_cdpruner_smoke_step_v1"
                if is_v8
                else "vision_opd_ai4s_v6_fullparam_dpc_smoke_step_v1"
            ),
            "vision_token_compressor_algorithm": compressor_algorithm,
            "world_size": 8,
            "global_step": int(self.global_steps),
            **producer_context,
            "producer_gates": producer_gates,
            "tensors": tensor_payload,
            route_key: route_samples,
            "multi_modal_input_summary": multimodal_summary,
            **(
                {"sample_uids": [str(value) for value in sample_uids]}
                if is_v8
                else {}
            ),
            **protocol_evidence,
        }
        summary = {
            "schema_version": artifact["schema_version"],
            "vision_token_compressor_algorithm": artifact["vision_token_compressor_algorithm"],
            "world_size": 8,
            "global_step": int(self.global_steps),
            **producer_context,
            "producer_gates": producer_gates,
            "route_count": route_count,
            "tensor_summaries": {
                key: self._audit_tensor_summary(value) for key, value in sorted(tensor_payload.items())
            },
            route_key: route_summary,
            "multi_modal_input_summary": multimodal_summary,
            **(
                {"sample_uids": [str(value) for value in sample_uids]}
                if is_v8
                else {}
            ),
            **protocol_evidence,
        }
        tensor_tmp = f"{tensor_path}.tmp-{os.getpid()}"
        summary_tmp = f"{summary_path}.tmp-{os.getpid()}"
        torch.save(artifact, tensor_tmp)
        with open(summary_tmp, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tensor_tmp, tensor_path)
        os.replace(summary_tmp, summary_path)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        if reward_fn is None and self._use_reward_free_teacher_vopd():
            reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            return reward_tensor, {}

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _use_reward_free_teacher_vopd(self) -> bool:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode != "vopd":
            return False
        if not self_distillation_cfg.get("teacher_always_on", False):
            return False
        if self_distillation_cfg.get("fallback_to_policy_loss_on_missing_teacher", False):
            return False
        if self_distillation_cfg.get("teacher_image_key", None) is not None:
            return True
        if self_distillation_cfg.get("teacher_prompt_mode", None) == "answer_hint":
            return True
        return False

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int
    ) -> list[Any]:
        """
        Collect environment feedback from reward_extra_infos_dict.

        Args:
            include_environment_feedback: Whether to include environment feedback
            reward_extra_infos_dict: Dictionary containing reward extra information
            batch_size: Size of the batch

        Returns:
            List of feedback strings (or None for entries without feedback)
        """
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                # Only include non-empty feedback strings
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type == "image":
                    text_parts.append("<image>")
                elif item_type == "video":
                    text_parts.append("<video>")
            return "".join(text_parts)
        if isinstance(content, dict):
            return str(content.get("text", ""))
        return str(content)

    @staticmethod
    def _normalize_teacher_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            with Image.open(image) as pil_image:
                return pil_image.convert("RGB")
        if isinstance(image, dict):
            if "image" in image:
                return RayPPOTrainer._normalize_teacher_image(image["image"])
            image_bytes = image.get("bytes", None)
            if image_bytes is not None:
                return Image.open(BytesIO(image_bytes)).convert("RGB")
            if "path" in image:
                with Image.open(image["path"]) as pil_image:
                    return pil_image.convert("RGB")
        raise TypeError(f"Unsupported teacher image type: {type(image)}")

    def _swap_images_in_messages(self, messages: list[dict], teacher_images: list[Any]) -> list[dict]:
        teacher_messages = deepcopy(messages)
        image_offset = 0
        for message in teacher_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    if image_offset >= len(teacher_images):
                        raise ValueError(
                            f"Teacher image count is smaller than prompt image count: {len(teacher_images)=}, {image_offset=}"
                        )
                    new_content.append(
                        {
                            "type": "image",
                            "image": self._normalize_teacher_image(teacher_images[image_offset]),
                        }
                    )
                    image_offset += 1
                else:
                    new_content.append(item)
            message["content"] = new_content
        if image_offset != len(teacher_images):
            raise ValueError(
                f"Teacher image count does not match prompt image placeholders: {len(teacher_images)=}, {image_offset=}"
            )
        return teacher_messages

    def _build_teacher_messages_from_template(self, messages: list[dict], teacher_images: list[Any]) -> list[dict]:
        teacher_messages = deepcopy(messages)
        normalized_images = [self._normalize_teacher_image(image) for image in teacher_images]
        image_offset = 0

        for message in teacher_messages:
            content = message.get("content")
            if isinstance(content, list):
                continue
            if not isinstance(content, str):
                continue

            content_list = []
            segments = [segment for segment in re.split(r"(<image>)", content) if segment != ""]
            for segment in segments:
                if segment == "<image>":
                    if image_offset >= len(normalized_images):
                        raise ValueError(
                            "Teacher image count is smaller than teacher_prompt placeholders: "
                            f"{len(normalized_images)=}, {image_offset=}"
                        )
                    content_list.append({"type": "image", "image": normalized_images[image_offset]})
                    image_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        if image_offset != len(normalized_images):
            raise ValueError(
                "Teacher image count does not match teacher_prompt placeholders: "
                f"{len(normalized_images)=}, {image_offset=}"
            )
        return teacher_messages

    def _prepare_teacher_messages(
        self,
        prompt_messages: list[dict],
        teacher_images: list[Any],
        teacher_prompt_messages: Optional[list[dict]] = None,
    ) -> list[dict]:
        if teacher_prompt_messages is not None:
            return self._build_teacher_messages_from_template(teacher_prompt_messages, teacher_images)
        return self._swap_images_in_messages(prompt_messages, teacher_images)

    def _prepare_opsd_teacher_messages(
        self,
        raw_prompt_messages: list[dict],
        answer: str,
        hint_template: str,
    ) -> list[dict]:
        teacher_messages = deepcopy(raw_prompt_messages)
        last_msg = teacher_messages[-1]
        content = last_msg["content"]
        hint_suffix = hint_template.format(answer=answer)

        if isinstance(content, list):
            teacher_messages[-1]["content"] = list(content) + [{"type": "text", "text": hint_suffix}]
        elif isinstance(content, str):
            teacher_messages[-1]["content"] = content + hint_suffix
        else:
            raise TypeError(f"Unsupported message content type: {type(content)}")
        return teacher_messages

    @staticmethod
    def _extract_images_from_messages(messages: list[dict]) -> list[Image.Image]:
        images = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                if "image" in item:
                    images.append(RayPPOTrainer._normalize_teacher_image(item["image"]))
                elif "path" in item:
                    images.append(RayPPOTrainer._normalize_teacher_image(item["path"]))
                elif "bytes" in item:
                    images.append(RayPPOTrainer._normalize_teacher_image({"bytes": item["bytes"]}))
        return images

    @staticmethod
    def _teacher_images_available(teacher_images: Any) -> bool:
        if teacher_images is None:
            return False
        if isinstance(teacher_images, np.ndarray):
            teacher_images = teacher_images.tolist()
        elif not isinstance(teacher_images, (list, tuple)):
            teacher_images = [teacher_images]
        for image in teacher_images:
            if image is None:
                continue
            if isinstance(image, str):
                if image:
                    return True
                continue
            if isinstance(image, dict):
                if image.get("path") or image.get("bytes") is not None or image.get("image") is not None:
                    return True
                continue
            return True
        return False

    @staticmethod
    def _normalize_teacher_image_sequence(teacher_images: Any) -> list[Any]:
        """Normalize one sample's image container without splitting strings."""

        if teacher_images is None:
            return []
        if isinstance(teacher_images, np.ndarray):
            teacher_images = teacher_images.tolist()
        if isinstance(teacher_images, (list, tuple)):
            return list(teacher_images)
        # A scalar path/dict/PIL image denotes one image.  Calling list() on a
        # string used to turn it into characters and corrupt the teacher input.
        return [teacher_images]

    @staticmethod
    def _teacher_reuse_stable_value(value: Any) -> Any:
        """Convert processor metadata to a deterministic JSON-safe value."""

        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if isinstance(value, dict):
            return {
                str(key): RayPPOTrainer._teacher_reuse_stable_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [RayPPOTrainer._teacher_reuse_stable_value(item) for item in value]
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        return f"{type(value).__module__}.{type(value).__qualname__}:{value}"

    @staticmethod
    def _teacher_reuse_digest(payload: Any) -> str:
        serialized = json.dumps(
            RayPPOTrainer._teacher_reuse_stable_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _teacher_reuse_file_descriptor(self, raw_path: str) -> Optional[dict[str, Any]]:
        if raw_path.startswith(("http://", "https://", "data:")):
            return None
        path = os.path.realpath(os.path.abspath(os.path.expanduser(raw_path.removeprefix("file://"))))
        try:
            stat = os.stat(path)
        except OSError:
            return None
        if not os.path.isfile(path):
            return None

        cache_key = (os.path.normcase(path), int(stat.st_size), int(stat.st_mtime_ns))
        digest_cache = getattr(self, "_teacher_reuse_file_digest_cache", None)
        if digest_cache is None:
            digest_cache = {}
            self._teacher_reuse_file_digest_cache = digest_cache
        digest = digest_cache.get(cache_key)
        if digest is None:
            hasher = hashlib.sha256()
            try:
                with open(path, "rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(block)
            except OSError:
                return None
            digest = hasher.hexdigest()
            digest_cache[cache_key] = digest
        return {
            "kind": "file",
            "path": os.path.normcase(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest,
        }

    def _teacher_reuse_image_descriptor(self, image: Any) -> Optional[dict[str, Any]]:
        """Fingerprint an image source without decoding/resizing it again."""

        if isinstance(image, np.ndarray) and image.dtype == object and image.size == 1:
            image = image.reshape(-1)[0]
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
            hasher = hashlib.sha256()
            hasher.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
            hasher.update(rgb.tobytes())
            return {
                "kind": "pil-rgb",
                "size": [rgb.width, rgb.height],
                "sha256": hasher.hexdigest(),
            }
        if isinstance(image, os.PathLike):
            image = os.fspath(image)
        if isinstance(image, str):
            return self._teacher_reuse_file_descriptor(image)
        if not isinstance(image, dict):
            return None

        candidates: list[dict[str, Any]] = []
        if image.get("bytes") is not None:
            image_bytes = image["bytes"]
            if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
                return None
            candidates.append(
                {
                    "kind": "bytes",
                    "size": len(image_bytes),
                    "sha256": hashlib.sha256(bytes(image_bytes)).hexdigest(),
                }
            )
        for key in ("image", "path"):
            if image.get(key) is not None:
                descriptor = self._teacher_reuse_image_descriptor(image[key])
                if descriptor is None:
                    return None
                candidates.append(descriptor)
        if not candidates:
            return None
        first = candidates[0]
        if any(candidate != first for candidate in candidates[1:]):
            # Ambiguous dictionaries are interpreted differently by the dataset,
            # DART helper and legacy teacher normalizer.  Never guess.
            return None
        return first

    def _teacher_reuse_message_image_descriptors(
        self, messages: list[dict]
    ) -> Optional[list[dict[str, Any]]]:
        descriptors: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                if not set(item).issubset({"type", "image", "path", "bytes"}):
                    return None
                descriptor = self._teacher_reuse_image_descriptor(item)
                if descriptor is None:
                    return None
                descriptors.append(descriptor)
        return descriptors

    def _teacher_reuse_prompt_digest(self, messages: list[dict]) -> Optional[str]:
        def canonicalize(value: Any) -> Any:
            if isinstance(value, Image.Image):
                return self._teacher_reuse_image_descriptor(value)
            if isinstance(value, (bytes, bytearray, memoryview)):
                return {
                    "kind": "bytes",
                    "size": len(value),
                    "sha256": hashlib.sha256(bytes(value)).hexdigest(),
                }
            if isinstance(value, np.ndarray):
                return canonicalize(value.tolist())
            if isinstance(value, os.PathLike):
                return os.fspath(value)
            if isinstance(value, dict):
                if value.get("type") in {"image", "image_url"}:
                    descriptor = self._teacher_reuse_image_descriptor(value)
                    if descriptor is None:
                        raise ValueError("unsupported image source")
                    non_source = {
                        str(key): canonicalize(item)
                        for key, item in value.items()
                        if key not in {"image", "image_url", "path", "bytes"}
                    }
                    non_source["source"] = descriptor
                    return non_source
                return {
                    str(key): canonicalize(item)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                }
            if isinstance(value, (list, tuple)):
                return [canonicalize(item) for item in value]
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            raise ValueError(f"unsupported prompt value: {type(value)}")

        try:
            return self._teacher_reuse_digest(canonicalize(messages))
        except (OSError, TypeError, ValueError):
            return None

    def _teacher_reuse_processor_fingerprint(self) -> Optional[str]:
        if self.processor is None:
            return None

        def component_payload(component: Any) -> Optional[dict[str, Any]]:
            if component is None:
                return None
            payload: dict[str, Any] = {
                "class": f"{type(component).__module__}.{type(component).__qualname__}",
            }
            for attr in ("name_or_path", "chat_template", "init_kwargs"):
                if hasattr(component, attr):
                    payload[attr] = getattr(component, attr)
            if hasattr(component, "to_dict"):
                try:
                    payload["config"] = component.to_dict()
                except Exception:
                    return None
            return payload

        payload = {
            "processor": component_payload(self.processor),
            "tokenizer": component_payload(getattr(self.processor, "tokenizer", None)),
            "image_processor": component_payload(getattr(self.processor, "image_processor", None)),
            "video_processor": component_payload(getattr(self.processor, "video_processor", None)),
            "processor_config": component_payload(getattr(self.processor, "config", None)),
            "apply_chat_template_kwargs": dict(self.config.data.get("apply_chat_template_kwargs", {}) or {}),
        }
        try:
            return self._teacher_reuse_digest(payload)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _teacher_reuse_local_path(path: Any) -> Optional[str]:
        if not isinstance(path, (str, os.PathLike)):
            return None
        normalized = os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path))))
        if not os.path.isdir(normalized):
            return None
        return os.path.normcase(normalized)

    @staticmethod
    def _teacher_reuse_qwen3_model_config(model_path: str) -> Optional[dict[str, Any]]:
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, TypeError, ValueError):
            return None
        model_type = config.get("model_type")
        if model_type not in {"qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"}:
            return None
        hasher = hashlib.sha256()
        try:
            with open(config_path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
        except OSError:
            return None
        return {"model_type": model_type, "config_sha256": hasher.hexdigest()}

    def _teacher_reuse_contract(self) -> Optional[dict[str, Any]]:
        cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        compressor = self.config.actor_rollout_ref.model.get("vision_token_compressor", {})
        if cfg is None or not bool(cfg.get("reuse_rollout_teacher_inputs", False)):
            return None
        if not bool(cfg.get("teacher_always_on", False)) or cfg.get("teacher_prompt_mode", None) is not None:
            return None
        if cfg.get("teacher_model_source", None) != "fixed" or not cfg.get("teacher_image_key", None):
            return None
        if self.processor is None or getattr(self, "async_rollout_mode", False):
            return None
        if self.config.actor_rollout_ref.rollout.get("name", None) != "hf":
            return None
        if bool(self.config.actor_rollout_ref.rollout.get("skip_rollout", False)):
            return None
        compressor_algorithm = compressor.get("algorithm", None)
        if not bool(compressor.get("enabled", False)) or compressor_algorithm not in {
            "qwen35_cdpruner_v1",
            "qwen35_conditional_diversity_prune_v1",
        }:
            return None
        # HFRollout's audited compressed entry point currently applies no extra chat
        # template kwargs.  A non-empty driver-side value could change IDs.
        if dict(self.config.data.get("apply_chat_template_kwargs", {}) or {}):
            return None

        model_path = self._teacher_reuse_local_path(self.config.actor_rollout_ref.model.get("path", None))
        teacher_path = self._teacher_reuse_local_path(cfg.get("teacher_model_path", None))
        if model_path is None or teacher_path != model_path:
            return None
        model_config = self._teacher_reuse_qwen3_model_config(model_path)
        if model_config is None:
            return None

        # Transformers 5.5 Qwen3VLProcessor does not expose name_or_path even
        # when created with from_pretrained().  Its nested tokenizer does, and
        # the driver model path/config hash above is authoritative.  If a
        # processor origin is present validate it, but do not require an
        # attribute that the production processor class does not define.
        processor_origin = self._teacher_reuse_local_path(getattr(self.processor, "name_or_path", None))
        tokenizer_origin = self._teacher_reuse_local_path(
            getattr(getattr(self.processor, "tokenizer", None), "name_or_path", None)
        )
        if tokenizer_origin != model_path or (processor_origin is not None and processor_origin != model_path):
            return None

        processor_fingerprint = self._teacher_reuse_processor_fingerprint()
        if processor_fingerprint is None:
            return None
        return {
            "schema": "rollout_teacher_input_reuse_v1",
            "processor_fingerprint": processor_fingerprint,
            "model_path": model_path,
            "teacher_model_path": teacher_path,
            "model_config": model_config,
            "rollout": "hf",
            "compressor": compressor_algorithm,
        }

    def _stamp_rollout_teacher_input_provenance(self, batch: DataProto) -> None:
        """Record immutable pre-rollout inputs used to validate cached tensors later."""

        contract = self._teacher_reuse_contract()
        if contract is None:
            return
        cfg = self.config.actor_rollout_ref.actor.self_distillation
        teacher_image_key = cfg.teacher_image_key
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        teacher_image_batch = batch.non_tensor_batch.get(teacher_image_key)
        if raw_prompts is None or teacher_image_batch is None:
            return
        # The audited source is HFRollout's raw-message processor path.  If a
        # future/custom dataset supplies pre-tokenized inputs, HFRollout gives
        # those precedence and this provenance would not describe the source.
        if batch.batch is None or set(batch.batch.keys()) != {"dummy_tensor"}:
            return

        provenance = np.empty((len(batch),), dtype=object)
        provenance[:] = None
        teacher_prompt_batch = batch.non_tensor_batch.get("teacher_prompt")
        # Share file hashes only within this pre-rollout snapshot.  The reuse
        # check installs a fresh cache and therefore re-reads each file after
        # rollout instead of trusting unchanged size/mtime metadata.
        self._teacher_reuse_file_digest_cache = {}
        try:
            for sample_idx in range(len(batch)):
                if teacher_prompt_batch is not None:
                    continue
                try:
                    messages = list(raw_prompts[sample_idx])
                    teacher_images = self._normalize_teacher_image_sequence(
                        teacher_image_batch[sample_idx]
                    )
                    raw_image_descriptors = self._teacher_reuse_message_image_descriptors(messages)
                    teacher_image_descriptors = [
                        self._teacher_reuse_image_descriptor(image) for image in teacher_images
                    ]
                    prompt_digest = self._teacher_reuse_prompt_digest(messages)
                    if (
                        prompt_digest is None
                        or raw_image_descriptors is None
                        or not raw_image_descriptors
                        or any(descriptor is None for descriptor in teacher_image_descriptors)
                        or raw_image_descriptors != teacher_image_descriptors
                    ):
                        continue
                    provenance[sample_idx] = {
                        "contract": contract,
                        "prompt_digest": prompt_digest,
                        "image_descriptors": raw_image_descriptors,
                    }
                except (OSError, TypeError, ValueError):
                    continue
        finally:
            self._teacher_reuse_file_digest_cache = {}
        batch.non_tensor_batch["_rollout_teacher_input_provenance_v1"] = provenance

    @staticmethod
    def _clone_teacher_reuse_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: RayPPOTrainer._clone_teacher_reuse_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [RayPPOTrainer._clone_teacher_reuse_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(RayPPOTrainer._clone_teacher_reuse_value(item) for item in value)
        return deepcopy(value)

    @staticmethod
    def _teacher_reuse_route_original_tokens(route: Any) -> Optional[int]:
        if isinstance(route, dict):
            value = route.get("original_tokens")
        else:
            value = getattr(route, "original_tokens", None)
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _recompute_teacher_reuse_prompt_position_ids(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        multimodal_inputs: dict[str, Any],
    ) -> torch.Tensor:
        """Rebuild prompt positions from IDs/grid without touching image pixels."""

        model_path = self._teacher_reuse_local_path(self.config.actor_rollout_ref.model.get("path", None))
        if model_path is not None and self._teacher_reuse_qwen3_model_config(model_path) is not None:
            from verl.models.transformers.vision_token_compressor import build_qwen3_5_position_ids

            mm_token_type_ids = torch.zeros_like(prompt_input_ids).unsqueeze(0)
            mm_token_type_ids[0][prompt_input_ids == int(self.processor.image_token_id)] = 1
            video_token_id = getattr(self.processor, "video_token_id", None)
            if video_token_id is not None:
                mm_token_type_ids[0][prompt_input_ids == int(video_token_id)] = 2
            return build_qwen3_5_position_ids(
                self.processor,
                input_ids=prompt_input_ids.unsqueeze(0),
                attention_mask=prompt_attention_mask.unsqueeze(0),
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=multimodal_inputs.get("image_grid_thw"),
                video_grid_thw=multimodal_inputs.get("video_grid_thw"),
            ).squeeze(1)

        if not hasattr(self.processor, "get_rope_index"):
            return compute_position_id_with_mask(prompt_attention_mask.unsqueeze(0)).squeeze(0)
        rope_index_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "image_grid_thw": multimodal_inputs.get("image_grid_thw"),
            "video_grid_thw": multimodal_inputs.get("video_grid_thw"),
        }
        try:
            signature = inspect.signature(self.processor.get_rope_index)
        except (TypeError, ValueError):
            signature = None
        if signature is None or "second_per_grid_ts" in signature.parameters:
            rope_index_kwargs["second_per_grid_ts"] = multimodal_inputs.get("second_per_grid_ts")
        try:
            position_ids = self.processor.get_rope_index(**rope_index_kwargs)
        except IndexError as exc:
            if prompt_input_ids.dim() != 1 or "tuple index out of range" not in str(exc):
                raise
            rope_index_kwargs["input_ids"] = prompt_input_ids.unsqueeze(0)
            rope_index_kwargs["attention_mask"] = prompt_attention_mask.unsqueeze(0)
            position_ids = self.processor.get_rope_index(**rope_index_kwargs)
        if isinstance(position_ids, tuple):
            position_ids = position_ids[0]
        if position_ids.dim() == 3 and position_ids.shape[1] == 1:
            position_ids = position_ids.squeeze(1)
        return self._maybe_expand_qwen2_5_vl_prompt_position_ids(position_ids, prompt_attention_mask)

    def _try_reuse_rollout_teacher_prompt_inputs(
        self,
        batch: DataProto,
        sample_idx: int,
        raw_prompt_messages: list[dict],
        teacher_images: list[Any],
        teacher_prompt_messages: Optional[list[dict]],
        responses: torch.Tensor,
        response_mask: torch.Tensor,
        max_prompt_len: int,
    ) -> tuple[
        Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]],
        str,
    ]:
        """Reuse full-token rollout inputs only when every equivalence gate passes."""

        contract = self._teacher_reuse_contract()
        if contract is None:
            return None, "contract"
        if teacher_prompt_messages is not None:
            return None, "teacher_prompt"

        # Re-hash current sources after rollout.  This deliberately does not
        # reuse the pre-rollout file digest cache.
        self._teacher_reuse_file_digest_cache = {}

        provenance_batch = batch.non_tensor_batch.get("_rollout_teacher_input_provenance_v1")
        if provenance_batch is None or sample_idx >= len(provenance_batch):
            return None, "provenance"
        provenance = provenance_batch[sample_idx]
        if not isinstance(provenance, dict) or provenance.get("contract") != contract:
            return None, "processor_fingerprint"

        prompt_digest = self._teacher_reuse_prompt_digest(raw_prompt_messages)
        raw_image_descriptors = self._teacher_reuse_message_image_descriptors(raw_prompt_messages)
        teacher_image_descriptors = [self._teacher_reuse_image_descriptor(image) for image in teacher_images]
        if prompt_digest is None or prompt_digest != provenance.get("prompt_digest"):
            return None, "prompt"
        if (
            raw_image_descriptors is None
            or not raw_image_descriptors
            or any(descriptor is None for descriptor in teacher_image_descriptors)
            or raw_image_descriptors != teacher_image_descriptors
            or raw_image_descriptors != provenance.get("image_descriptors")
        ):
            return None, "images"

        required_tensor_keys = {"prompts", "input_ids", "attention_mask", "position_ids", "responses"}
        if batch.batch is None or not required_tensor_keys.issubset(batch.batch.keys()):
            return None, "source_tensors"
        prompt_width = int(batch.batch["prompts"].shape[-1])
        source_input_ids = batch.batch["input_ids"][sample_idx]
        source_attention_mask = batch.batch["attention_mask"][sample_idx]
        source_position_ids = batch.batch["position_ids"][sample_idx]
        source_prompts = batch.batch["prompts"][sample_idx]
        source_responses = batch.batch["responses"][sample_idx]
        if (
            source_input_ids.dim() != 1
            or source_attention_mask.dim() != 1
            or source_prompts.dim() != 1
            or prompt_width <= 0
            or source_input_ids.shape[-1] != prompt_width + responses.shape[-1]
            or source_attention_mask.shape != source_input_ids.shape
            or source_prompts.shape[-1] != prompt_width
            or source_position_ids.shape[-1] != source_input_ids.shape[-1]
        ):
            return None, "source_shapes"
        if not torch.equal(source_input_ids[:prompt_width], source_prompts):
            return None, "prompt_prefix"
        if not torch.equal(source_input_ids[prompt_width:].to(responses.device), responses):
            return None, "responses"
        if not torch.equal(source_responses.to(responses.device), responses):
            return None, "responses"
        if not torch.equal(source_attention_mask[prompt_width:].to(response_mask.device), response_mask):
            return None, "response_mask"

        prompt_mask = source_attention_mask[:prompt_width]
        if not torch.all((prompt_mask == 0) | (prompt_mask == 1)):
            return None, "prompt_mask"
        prompt_mask_bool = prompt_mask.bool()
        active_count = int(prompt_mask_bool.sum().item())
        if active_count <= 0 or active_count > int(max_prompt_len):
            return None, "prompt_length"
        first_active = int(torch.nonzero(prompt_mask_bool, as_tuple=False)[0].item())
        if first_active + active_count != prompt_width or not bool(prompt_mask_bool[first_active:].all().item()):
            return None, "left_padding"

        prompt_input_ids = source_prompts[prompt_mask_bool]
        if source_position_ids.dim() == 1:
            prompt_position_ids = source_position_ids[:prompt_width][prompt_mask_bool]
            response_position_ids = source_position_ids[prompt_width:]
            expected_response_positions = (
                torch.arange(responses.shape[-1], device=source_position_ids.device, dtype=source_position_ids.dtype)
                + prompt_position_ids[-1]
                + 1
            )
        elif source_position_ids.dim() == 2:
            prompt_position_ids = source_position_ids[:, :prompt_width][:, prompt_mask_bool]
            response_position_ids = source_position_ids[:, prompt_width:]
            expected_response_positions = (
                torch.arange(responses.shape[-1], device=source_position_ids.device, dtype=source_position_ids.dtype)
                .unsqueeze(0)
                + prompt_position_ids[:, -1:]
                + 1
            )
        else:
            return None, "position_rank"
        if not torch.equal(response_position_ids, expected_response_positions):
            return None, "response_positions"

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(getattr(self.processor, "tokenizer", None), "image_token_id", None)
        if image_token_id is None:
            return None, "image_token_id"
        visual_indices = torch.nonzero(prompt_input_ids == int(image_token_id), as_tuple=False).flatten()
        if visual_indices.numel() == 0:
            return None, "visual_tokens"
        split_points = torch.nonzero(torch.diff(visual_indices) != 1, as_tuple=False).flatten().tolist()
        spans = []
        span_start = 0
        for split_point in split_points:
            spans.append(visual_indices[span_start : split_point + 1])
            span_start = split_point + 1
        spans.append(visual_indices[span_start:])

        routes_batch = batch.non_tensor_batch.get("dart_merge_routes")
        multimodal_batch = batch.non_tensor_batch.get("multi_modal_inputs")
        if routes_batch is None or multimodal_batch is None:
            return None, "multimodal_source"
        routes = routes_batch[sample_idx]
        if isinstance(routes, np.ndarray):
            routes = routes.tolist()
        if not isinstance(routes, (list, tuple)):
            return None, "routes"
        multimodal_inputs = multimodal_batch[sample_idx]
        if hasattr(multimodal_inputs, "data") and not isinstance(multimodal_inputs, dict):
            multimodal_inputs = multimodal_inputs.data
        if not isinstance(multimodal_inputs, dict):
            return None, "multimodal_inputs"
        image_grid_thw = multimodal_inputs.get("image_grid_thw")
        pixel_values = multimodal_inputs.get("pixel_values")
        if (
            not isinstance(image_grid_thw, torch.Tensor)
            or image_grid_thw.dim() != 2
            or image_grid_thw.shape[1] != 3
            or image_grid_thw.dtype == torch.bool
            or image_grid_thw.is_floating_point()
            or not bool((image_grid_thw > 0).all().item())
            or not isinstance(pixel_values, torch.Tensor)
            or pixel_values.dim() != 2
            or not pixel_values.is_floating_point()
            or pixel_values.numel() == 0
            or not bool(torch.isfinite(pixel_values).all().item())
        ):
            return None, "pixel_grid"
        image_count = len(raw_image_descriptors)
        if len(spans) != image_count or len(routes) != image_count or image_grid_thw.shape[0] != image_count:
            return None, "multi_image_contract"

        from verl.models.transformers.vision_token_compressor import DARTMergeRoute

        validated_routes = []
        try:
            for route in routes:
                validated_route = route if isinstance(route, DARTMergeRoute) else DARTMergeRoute.from_dict(route)
                validated_route.validate()
                if validated_route.anchor_coordinates is None:
                    return None, "routes"
                validated_routes.append(validated_route)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None, "routes"
        route_lengths = [route.original_tokens for route in validated_routes]
        if [int(span.numel()) for span in spans] != route_lengths:
            # This is the critical full-token gate: compressed public prompts
            # have output-token spans and can never be reused by the teacher.
            return None, "full_visual_tokens"
        spatial_merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        if spatial_merge_size <= 0:
            return None, "pixel_grid"
        grid_patch_counts = image_grid_thw.to(torch.long).prod(dim=-1)
        merge_area = spatial_merge_size**2
        if bool((grid_patch_counts % merge_area != 0).any().item()):
            return None, "pixel_grid"
        grid_visual_lengths = (grid_patch_counts // merge_area).tolist()
        if route_lengths != grid_visual_lengths:
            return None, "pixel_grid"
        expected_patch_rows = int(grid_patch_counts.sum().item())
        if int(pixel_values.shape[0]) != expected_patch_rows:
            return None, "pixel_grid"
        for video_key in ("pixel_values_videos", "video_grid_thw"):
            video_value = multimodal_inputs.get(video_key)
            if isinstance(video_value, torch.Tensor) and video_value.numel() > 0:
                return None, "video"

        recomputed_prompt_position_ids = self._recompute_teacher_reuse_prompt_position_ids(
            prompt_input_ids,
            prompt_mask[prompt_mask_bool],
            multimodal_inputs,
        ).to(prompt_position_ids.device)
        if not torch.equal(prompt_position_ids, recomputed_prompt_position_ids):
            return None, "prompt_positions"
        for span, route in zip(spans, validated_routes, strict=True):
            selected = route.selected_indices.to(prompt_position_ids.device)
            expected_anchors = prompt_position_ids[-3:, span].index_select(1, selected).transpose(0, 1).to(torch.long)
            if not torch.equal(route.anchor_coordinates.to(expected_anchors.device), expected_anchors):
                return None, "route_positions"

        teacher_multi_modal_inputs = {
            key: self._clone_teacher_reuse_value(value)
            for key, value in multimodal_inputs.items()
            if key not in {"input_ids", "attention_mask", "mm_token_type_ids", "images_seqlens"}
        }
        full_input_ids = torch.cat((prompt_input_ids.detach().cpu(), responses.detach().cpu()), dim=0).clone()
        full_attention_mask = torch.cat(
            (prompt_mask[prompt_mask_bool].detach().cpu(), response_mask.detach().cpu()), dim=0
        ).clone()
        full_position_ids = torch.cat(
            (prompt_position_ids.detach().cpu(), response_position_ids.detach().cpu()), dim=-1
        ).clone()
        response_start_idx = torch.tensor(active_count, dtype=torch.long)
        return (
            full_input_ids,
            full_attention_mask,
            full_position_ids,
            response_start_idx,
            teacher_multi_modal_inputs,
        ), "reused"

    def _get_visual_special_token_mappings(self) -> dict[int, str]:
        processing_class = self.processor or self.tokenizer
        if processing_class is None:
            return {}

        token_mappings: dict[int, str] = {}
        attr_names = {
            "image_token_id": "<|image_pad|>",
            "video_token_id": "<|video_pad|>",
        }
        # Qwen3.5 boundary delimiters use ordinary text embeddings in the
        # response. Only image/video placeholders consume visual features.
        for attr_name, fallback_token in attr_names.items():
            token_id = getattr(processing_class, attr_name, None)
            if token_id is None:
                token_id = getattr(getattr(processing_class, "tokenizer", None), attr_name, None)
            if token_id is None:
                tokenizer = getattr(processing_class, "tokenizer", processing_class)
                if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
                    converted = tokenizer.convert_tokens_to_ids(fallback_token)
                    if converted is not None and converted != getattr(tokenizer, "unk_token_id", None):
                        token_id = converted
            if token_id is not None:
                token_mappings[int(token_id)] = fallback_token
        return token_mappings

    def _raise_if_response_contains_visual_special_tokens(
        self,
        response: torch.Tensor,
        response_mask: torch.Tensor,
        batch: DataProto,
        sample_idx: int,
    ) -> None:
        visual_special_tokens = self._get_visual_special_token_mappings()
        if not visual_special_tokens:
            return

        active_response = response[response_mask.bool()].detach().cpu()
        bad_positions = [
            (position, int(token_id))
            for position, token_id in enumerate(active_response.tolist())
            if int(token_id) in visual_special_tokens
        ]
        if not bad_positions:
            return

        bad_tokens = [
            {
                "position": position,
                "token_id": token_id,
                "token": visual_special_tokens[token_id],
            }
            for position, token_id in bad_positions
        ]
        uid = batch.non_tensor_batch["uid"][sample_idx] if "uid" in batch.non_tensor_batch else None
        index = batch.non_tensor_batch["index"][sample_idx] if "index" in batch.non_tensor_batch else None
        decoded_response = None
        tokenizer = getattr(self.processor, "tokenizer", None) if self.processor is not None else self.tokenizer
        if tokenizer is not None:
            try:
                decoded_response = tokenizer.decode(active_response.tolist(), skip_special_tokens=False)
            except Exception:
                decoded_response = None

        message = (
            "Response contains vision special tokens before teacher SDPO forward, which would corrupt "
            f"image token accounting. sample_idx={sample_idx}, uid={uid}, index={index}, bad_tokens={bad_tokens}"
        )
        if decoded_response is not None:
            message += f", response={decoded_response!r}"
        logger.error(message)
        raise ValueError(message)

    def _needs_qwen2_5_vl_teacher_position_ids_compat(self) -> bool:
        if self.processor is None:
            return False

        processor_name = type(self.processor).__name__.lower()
        if "qwen2_5" in processor_name and "vl" in processor_name:
            return True

        model_path = str(getattr(self.config.actor_rollout_ref.model, "path", "")).lower()
        return "qwen2.5-vl" in model_path or "qwen2_5_vl" in model_path or "qwen25_vl" in model_path

    def _maybe_expand_qwen2_5_vl_prompt_position_ids(
        self,
        prompt_position_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self._needs_qwen2_5_vl_teacher_position_ids_compat():
            return prompt_position_ids

        if prompt_position_ids.dim() != 2 or prompt_position_ids.shape[0] != 3:
            return prompt_position_ids

        valid_mask = prompt_attention_mask.bool()
        text_position_ids = torch.ones(
            (1, prompt_attention_mask.shape[-1]),
            dtype=prompt_position_ids.dtype,
            device=prompt_position_ids.device,
        )
        text_position_ids[0, valid_mask] = torch.arange(
            valid_mask.sum().item(),
            dtype=prompt_position_ids.dtype,
            device=prompt_position_ids.device,
        )
        return torch.cat((text_position_ids, prompt_position_ids), dim=0)

    def _build_teacher_prompt_inputs(
        self,
        messages: list[dict],
        responses: torch.Tensor,
        response_mask: torch.Tensor,
        max_prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        apply_kwargs = dict(self.config.data.apply_chat_template_kwargs or {})
        teacher_multi_modal_inputs = None
        if self.processor is not None:
            compressor = self.config.actor_rollout_ref.model.get("vision_token_compressor", {})
            if bool(compressor.get("enabled", False)):
                from verl.utils.multimodal_preprocessing import process_dart_messages_once

                raw_prompt, model_inputs = process_dart_messages_once(
                    self.processor,
                    messages,
                    apply_chat_template_kwargs=apply_kwargs,
                )
            else:
                raw_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **apply_kwargs,
                )
                prompt_images = self._extract_images_from_messages(messages)
                model_inputs = dict(
                    self.processor(
                        text=[raw_prompt],
                        images=prompt_images or None,
                        videos=None,
                        return_tensors="pt",
                        truncation=False,
                    )
                )
            teacher_multi_modal_inputs = model_inputs.copy()
            prompt_input_ids = teacher_multi_modal_inputs.pop("input_ids").squeeze(0)
            prompt_attention_mask = teacher_multi_modal_inputs.pop("attention_mask").squeeze(0)
            if prompt_input_ids.numel() > max_prompt_len:
                raise ValueError(
                    f"Teacher prompt has {prompt_input_ids.numel()} tokens, exceeding max_reprompt_len="
                    f"{max_prompt_len}; refusing silent multimodal truncation."
                )

            mm_token_type_ids = teacher_multi_modal_inputs.pop("mm_token_type_ids", None)
            if mm_token_type_ids is not None:
                from verl.models.transformers.vision_token_compressor import build_qwen3_5_position_ids

                prompt_position_ids = build_qwen3_5_position_ids(
                    self.processor,
                    input_ids=prompt_input_ids.unsqueeze(0),
                    attention_mask=prompt_attention_mask.unsqueeze(0),
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=teacher_multi_modal_inputs.get("image_grid_thw"),
                    video_grid_thw=teacher_multi_modal_inputs.get("video_grid_thw"),
                ).squeeze(1)
            elif hasattr(self.processor, "get_rope_index"):
                processor_model_type = getattr(getattr(self.processor, "config", None), "model_type", None)
                if processor_model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"}:
                    if mm_token_type_ids is None:
                        mm_token_type_ids = torch.zeros_like(prompt_input_ids).unsqueeze(0)
                        mm_token_type_ids[0][prompt_input_ids == self.processor.image_token_id] = 1
                        video_token_id = getattr(self.processor, "video_token_id", None)
                        if video_token_id is not None:
                            mm_token_type_ids[0][prompt_input_ids == video_token_id] = 2

                    prompt_position_ids = self.processor.get_rope_index(
                        input_ids=prompt_input_ids.unsqueeze(0),
                        mm_token_type_ids=mm_token_type_ids,
                        image_grid_thw=teacher_multi_modal_inputs.get("image_grid_thw"),
                        video_grid_thw=teacher_multi_modal_inputs.get("video_grid_thw"),
                        attention_mask=prompt_attention_mask.unsqueeze(0),
                    )
                else:
                    rope_index_kwargs = dict(
                        input_ids=prompt_input_ids,
                        attention_mask=prompt_attention_mask,
                        image_grid_thw=teacher_multi_modal_inputs.get("image_grid_thw"),
                        video_grid_thw=teacher_multi_modal_inputs.get("video_grid_thw"),
                    )
                    try:
                        rope_index_signature = inspect.signature(self.processor.get_rope_index)
                    except (TypeError, ValueError):
                        rope_index_signature = None
                    if rope_index_signature is None or "second_per_grid_ts" in rope_index_signature.parameters:
                        rope_index_kwargs["second_per_grid_ts"] = teacher_multi_modal_inputs.get("second_per_grid_ts")

                    try:
                        prompt_position_ids = self.processor.get_rope_index(**rope_index_kwargs)
                    except IndexError as exc:
                        # Some Qwen-VL implementations expect a batch dimension here.
                        if prompt_input_ids.dim() != 1 or "tuple index out of range" not in str(exc):
                            raise
                        rope_index_kwargs["input_ids"] = prompt_input_ids.unsqueeze(0)
                        rope_index_kwargs["attention_mask"] = prompt_attention_mask.unsqueeze(0)
                        prompt_position_ids = self.processor.get_rope_index(**rope_index_kwargs)
                if isinstance(prompt_position_ids, tuple):
                    prompt_position_ids = prompt_position_ids[0]
                if prompt_position_ids.dim() == 3 and prompt_position_ids.shape[1] == 1:
                    prompt_position_ids = prompt_position_ids.squeeze(1)
                if (
                    processor_model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"}
                    and prompt_position_ids.dim() == 2
                    and prompt_position_ids.shape[0] == 3
                ):
                    text_position_ids = torch.arange(
                        prompt_input_ids.shape[-1],
                        dtype=prompt_position_ids.dtype,
                        device=prompt_position_ids.device,
                    ).unsqueeze(0)
                    prompt_position_ids = torch.cat((text_position_ids, prompt_position_ids), dim=0)
                prompt_position_ids = self._maybe_expand_qwen2_5_vl_prompt_position_ids(
                    prompt_position_ids,
                    prompt_attention_mask,
                )
            else:
                prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask.unsqueeze(0)).squeeze(0)
        else:
            teacher_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                continue_final_message=False,
                add_generation_prompt=True,
                max_length=max_prompt_len,
                padding=False,
                truncation=False,
            )
            prompt_input_ids = teacher_prompt["input_ids"].squeeze(0)
            prompt_attention_mask = teacher_prompt["attention_mask"].squeeze(0)
            if prompt_input_ids.numel() > max_prompt_len:
                raise ValueError(
                    f"Teacher prompt has {prompt_input_ids.numel()} tokens, exceeding max_reprompt_len="
                    f"{max_prompt_len}; refusing silent truncation."
                )
            prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask.unsqueeze(0)).squeeze(0)

        return self._append_teacher_response_to_prompt_prefix(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            responses=responses,
            response_mask=response_mask,
            teacher_multi_modal_inputs=teacher_multi_modal_inputs,
        )

    @staticmethod
    def _append_teacher_response_to_prompt_prefix(
        *,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_position_ids: torch.Tensor,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
        teacher_multi_modal_inputs: Optional[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Append a trajectory response without reprocessing its shared image prompt."""

        full_input_ids = torch.cat([prompt_input_ids, responses.cpu()], dim=0)
        full_attention_mask = torch.cat([prompt_attention_mask, response_mask.cpu()], dim=0)
        response_start_idx = torch.tensor(prompt_input_ids.shape[0], dtype=torch.long)
        if prompt_position_ids.dim() == 1:
            response_positions = torch.arange(
                responses.shape[0], dtype=prompt_position_ids.dtype
            ) + prompt_position_ids[-1] + 1
            full_position_ids = torch.cat([prompt_position_ids, response_positions], dim=0)
        else:
            response_positions = (
                torch.arange(responses.shape[0], dtype=prompt_position_ids.dtype).unsqueeze(0)
                + prompt_position_ids[:, -1:].cpu()
                + 1
            )
            full_position_ids = torch.cat([prompt_position_ids.cpu(), response_positions], dim=-1)

        return full_input_ids, full_attention_mask, full_position_ids, response_start_idx, teacher_multi_modal_inputs

    def _collect_solutions_by_uid(self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        """Remove <think>...</think> tags and their content from text."""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
    ) -> Optional[str]:
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]
        if len(solution_idxs) == 0:
            return None
        solution_idx = solution_idxs[0]  # taking the first successful demonstration effectively selects a random one
        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)
        return solution_str


    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode != "vopd":
            return None

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        batch_size = batch.batch.batch_size[0]

        # Determine teacher input construction mode
        teacher_prompt_mode = self_distillation_cfg.get("teacher_prompt_mode", None)
        use_opsd_answer_hint = (
            self_distillation_cfg.get("teacher_always_on", False)
            and teacher_prompt_mode == "answer_hint"
        )
        # Build teacher inputs directly from teacher_image_key whenever teacher_always_on is enabled
        # for both SDPO and OPD modes. Note: OPD is NOT reward-free globally; this branch only
        # controls teacher-side input construction.
        use_teacher_always_on_inputs = (
            self_distillation_cfg.get("teacher_always_on", False)
            and self_distillation_cfg.get("teacher_image_key", None) is not None
            and not use_opsd_answer_hint
        )

        if use_opsd_answer_hint:
            answer_hint_template = self_distillation_cfg.get(
                "answer_hint_template",
                "\n\nHere is a reference solution to this problem:\n{answer}\n\n"
                "After understanding the reference solution, please try to solve this problem using your own approach below:\n",
            )

            teacher_input_ids_list = []
            teacher_attention_mask_list = []
            teacher_position_ids_list = []
            teacher_response_start_idx_list = []
            teacher_multi_modal_inputs_list = []
            teacher_present_mask_list = []

            for i in range(batch_size):
                self._raise_if_response_contains_visual_special_tokens(
                    responses[i],
                    response_mask[i],
                    batch,
                    i,
                )

                reward_model_info = batch.non_tensor_batch.get("reward_model", [None] * batch_size)
                answer = None
                if reward_model_info[i] is not None and isinstance(reward_model_info[i], dict):
                    answer = reward_model_info[i].get("ground_truth", None)
                if answer is None:
                    extra_info = batch.non_tensor_batch.get("extra_info", [None] * batch_size)
                    if extra_info[i] is not None and isinstance(extra_info[i], dict):
                        answer = extra_info[i].get("answer", None)

                has_answer = answer is not None and str(answer).strip() != ""
                teacher_present_mask_list.append(1.0 if has_answer else 0.0)

                if not has_answer:
                    answer = ""

                raw_prompt_messages = list(batch.non_tensor_batch["raw_prompt"][i])
                teacher_messages = self._prepare_opsd_teacher_messages(
                    raw_prompt_messages,
                    str(answer),
                    answer_hint_template,
                )

                (
                    teacher_input_ids,
                    teacher_attention_mask,
                    teacher_position_ids,
                    teacher_response_start_idx,
                    teacher_multi_modal_inputs,
                ) = self._build_teacher_prompt_inputs(
                    teacher_messages,
                    responses[i],
                    response_mask[i],
                    max_prompt_len=self_distillation_cfg.max_reprompt_len,
                )
                teacher_input_ids_list.append(teacher_input_ids)
                teacher_attention_mask_list.append(teacher_attention_mask)
                teacher_position_ids_list.append(teacher_position_ids)
                teacher_response_start_idx_list.append(teacher_response_start_idx)
                teacher_multi_modal_inputs_list.append(teacher_multi_modal_inputs)

            teacher_input_ids = torch.nn.utils.rnn.pad_sequence(
                teacher_input_ids_list,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id or 0,
            ).to(device)
            teacher_attention_mask = torch.nn.utils.rnn.pad_sequence(
                teacher_attention_mask_list,
                batch_first=True,
                padding_value=0,
            ).to(device)

            max_teacher_len = teacher_input_ids.shape[1]
            if teacher_position_ids_list[0].dim() == 1:
                teacher_position_ids = torch.zeros(
                    (batch_size, max_teacher_len),
                    dtype=teacher_position_ids_list[0].dtype,
                    device=device,
                )
                for i, position_ids in enumerate(teacher_position_ids_list):
                    teacher_position_ids[i, : position_ids.shape[-1]] = position_ids.to(device)
            else:
                rope_dims = teacher_position_ids_list[0].shape[0]
                teacher_position_ids = torch.zeros(
                    (batch_size, rope_dims, max_teacher_len),
                    dtype=teacher_position_ids_list[0].dtype,
                    device=device,
                )
                for i, position_ids in enumerate(teacher_position_ids_list):
                    teacher_position_ids[i, :, : position_ids.shape[-1]] = position_ids.to(device)

            teacher_present_mask = torch.tensor(teacher_present_mask_list, dtype=torch.float32, device=device)
            grpo_fallback_count = float(batch_size - teacher_present_mask.sum().item())
            metrics = {
                "self_distillation/teacher_always_on_fraction": teacher_present_mask.mean().item(),
                "self_distillation/opsd_answer_hint_fraction": teacher_present_mask.mean().item(),
                "self_distillation/policy_fallback_fraction": (1.0 - teacher_present_mask.mean()).item(),
                "self_distillation/grpo_fallback_count": grpo_fallback_count,
            }
            return DataProto.from_dict(
                tensors={
                    "teacher_input_ids": teacher_input_ids,
                    "teacher_attention_mask": teacher_attention_mask,
                    "teacher_position_ids": teacher_position_ids,
                    "teacher_response_start_idx": torch.stack(teacher_response_start_idx_list).to(device),
                    "self_distillation_mask": teacher_present_mask,
                },
                non_tensors={"teacher_multi_modal_inputs": teacher_multi_modal_inputs_list},
            ), metrics

        if use_teacher_always_on_inputs:
            teacher_image_key = self_distillation_cfg.teacher_image_key
            if teacher_image_key not in batch.non_tensor_batch:
                raise KeyError(f"Teacher image key `{teacher_image_key}` not found in batch.non_tensor_batch")
            fallback_to_policy_loss = self_distillation_cfg.get("fallback_to_policy_loss_on_missing_teacher", False)

            teacher_input_ids_list = []
            teacher_attention_mask_list = []
            teacher_position_ids_list = []
            teacher_response_start_idx_list = []
            teacher_multi_modal_inputs_list = []
            teacher_present_mask_list = []
            teacher_input_reuse_count = 0
            teacher_input_reuse_reasons: dict[str, int] = defaultdict(int)
            # rollout.n trajectories for one uid have different responses but
            # the exact same deterministic teacher prompt/image prefix.  Keep
            # ``reuse_rollout_teacher_inputs=False`` semantics: the first item
            # still executes the independent canonical teacher processor path.
            # Subsequent items may share that canonical prefix only when Ray
            # preserved the exact same source-object identities.
            teacher_prompt_prefix_cache: dict[
                tuple[str, int, int, int],
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]],
            ] = {}

            for i in range(batch_size):
                self._raise_if_response_contains_visual_special_tokens(
                    responses[i],
                    response_mask[i],
                    batch,
                    i,
                )
                teacher_prompt_messages = None
                teacher_prompt_source = None
                if "teacher_prompt" in batch.non_tensor_batch:
                    teacher_prompt_source = batch.non_tensor_batch["teacher_prompt"][i]
                    teacher_prompt_messages = list(teacher_prompt_source)

                teacher_image_source = batch.non_tensor_batch[teacher_image_key][i]
                teacher_images = self._normalize_teacher_image_sequence(
                    teacher_image_source
                )
                has_teacher_images = self._teacher_images_available(teacher_images)
                teacher_present_mask_list.append(1.0 if has_teacher_images else 0.0)
                if not has_teacher_images:
                    if not fallback_to_policy_loss:
                        raise ValueError(
                            f"Teacher image key `{teacher_image_key}` is empty for sample {i}, "
                            "but fallback_to_policy_loss_on_missing_teacher=False."
                        )
                    teacher_images = self._extract_images_from_messages(list(batch.non_tensor_batch["raw_prompt"][i]))

                raw_prompt_source = batch.non_tensor_batch["raw_prompt"][i]
                raw_prompt_messages = list(raw_prompt_source)
                try:
                    reused_inputs, _reuse_reason = self._try_reuse_rollout_teacher_prompt_inputs(
                        batch=batch,
                        sample_idx=i,
                        raw_prompt_messages=raw_prompt_messages,
                        teacher_images=teacher_images,
                        teacher_prompt_messages=teacher_prompt_messages,
                        responses=responses[i],
                        response_mask=response_mask[i],
                        max_prompt_len=self_distillation_cfg.max_reprompt_len,
                    )
                except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                    # Cache/provenance data is an optimization hint, never a
                    # correctness dependency. Malformed hints fail closed to
                    # the canonical processor path.
                    reused_inputs, _reuse_reason = None, "cache_validation_error"
                if reused_inputs is not None:
                    (
                        teacher_input_ids,
                        teacher_attention_mask,
                        teacher_position_ids,
                        teacher_response_start_idx,
                        teacher_multi_modal_inputs,
                    ) = reused_inputs
                    teacher_input_reuse_count += 1
                    teacher_input_reuse_reasons["reused"] += 1
                else:
                    teacher_input_reuse_reasons[_reuse_reason] += 1
                    uid_value = (
                        batch.non_tensor_batch["uid"][i]
                        if "uid" in batch.non_tensor_batch
                        else None
                    )
                    prefix_cache_key = (
                        (str(uid_value), id(raw_prompt_source), id(teacher_image_source), id(teacher_prompt_source))
                        if uid_value is not None
                        else None
                    )
                    cached_prefix = (
                        teacher_prompt_prefix_cache.get(prefix_cache_key)
                        if prefix_cache_key is not None
                        else None
                    )
                    if cached_prefix is not None:
                        (
                            prompt_input_ids,
                            prompt_attention_mask,
                            prompt_position_ids,
                            teacher_multi_modal_inputs,
                        ) = cached_prefix
                        (
                            teacher_input_ids,
                            teacher_attention_mask,
                            teacher_position_ids,
                            teacher_response_start_idx,
                            teacher_multi_modal_inputs,
                        ) = self._append_teacher_response_to_prompt_prefix(
                            prompt_input_ids=prompt_input_ids,
                            prompt_attention_mask=prompt_attention_mask,
                            prompt_position_ids=prompt_position_ids,
                            responses=responses[i],
                            response_mask=response_mask[i],
                            teacher_multi_modal_inputs=teacher_multi_modal_inputs,
                        )
                    else:
                        teacher_messages = self._prepare_teacher_messages(
                            raw_prompt_messages,
                            teacher_images,
                            teacher_prompt_messages=teacher_prompt_messages,
                        )
                        (
                            teacher_input_ids,
                            teacher_attention_mask,
                            teacher_position_ids,
                            teacher_response_start_idx,
                            teacher_multi_modal_inputs,
                        ) = self._build_teacher_prompt_inputs(
                            teacher_messages,
                            responses[i],
                            response_mask[i],
                            max_prompt_len=self_distillation_cfg.max_reprompt_len,
                        )
                        if prefix_cache_key is not None:
                            response_start = int(teacher_response_start_idx.item())
                            teacher_prompt_prefix_cache[prefix_cache_key] = (
                                teacher_input_ids[:response_start].clone(),
                                teacher_attention_mask[:response_start].clone(),
                                teacher_position_ids[..., :response_start].clone(),
                                teacher_multi_modal_inputs,
                            )
                teacher_input_ids_list.append(teacher_input_ids)
                teacher_attention_mask_list.append(teacher_attention_mask)
                teacher_position_ids_list.append(teacher_position_ids)
                teacher_response_start_idx_list.append(teacher_response_start_idx)
                teacher_multi_modal_inputs_list.append(teacher_multi_modal_inputs)

            teacher_input_ids = torch.nn.utils.rnn.pad_sequence(
                teacher_input_ids_list,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id or 0,
            ).to(device)
            teacher_attention_mask = torch.nn.utils.rnn.pad_sequence(
                teacher_attention_mask_list,
                batch_first=True,
                padding_value=0,
            ).to(device)

            max_teacher_len = teacher_input_ids.shape[1]
            if teacher_position_ids_list[0].dim() == 1:
                teacher_position_ids = torch.zeros(
                    (batch_size, max_teacher_len),
                    dtype=teacher_position_ids_list[0].dtype,
                    device=device,
                )
                for i, position_ids in enumerate(teacher_position_ids_list):
                    teacher_position_ids[i, : position_ids.shape[-1]] = position_ids.to(device)
            else:
                rope_dims = teacher_position_ids_list[0].shape[0]
                teacher_position_ids = torch.zeros(
                    (batch_size, rope_dims, max_teacher_len),
                    dtype=teacher_position_ids_list[0].dtype,
                    device=device,
                )
                for i, position_ids in enumerate(teacher_position_ids_list):
                    teacher_position_ids[i, :, : position_ids.shape[-1]] = position_ids.to(device)

            teacher_present_mask = torch.tensor(teacher_present_mask_list, dtype=torch.float32, device=device)
            grpo_fallback_count = float(batch_size - teacher_present_mask.sum().item())
            metrics = {
                "self_distillation/teacher_always_on_fraction": teacher_present_mask.mean().item(),
                "self_distillation/teacher_image_swap_fraction": teacher_present_mask.mean().item(),
                "self_distillation/policy_fallback_fraction": (1.0 - teacher_present_mask.mean()).item(),
                "self_distillation/grpo_fallback_count": grpo_fallback_count,
                "self_distillation/teacher_input_reuse_fraction": teacher_input_reuse_count / batch_size,
                "self_distillation/teacher_input_reuse_fallback_fraction":
                    (batch_size - teacher_input_reuse_count) / batch_size,
            }
            for reason, count in sorted(teacher_input_reuse_reasons.items()):
                metrics[f"self_distillation/teacher_input_reuse_reason/{reason}"] = count / batch_size
            return DataProto.from_dict(
                tensors={
                    "teacher_input_ids": teacher_input_ids,
                    "teacher_attention_mask": teacher_attention_mask,
                    "teacher_position_ids": teacher_position_ids,
                    "teacher_response_start_idx": torch.stack(teacher_response_start_idx_list).to(device),
                    "self_distillation_mask": teacher_present_mask,
                },
                non_tensors={"teacher_multi_modal_inputs": teacher_multi_modal_inputs_list},
            ), metrics

        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [self._message_content_to_text(msgs[-1]["content"]) for msgs in batch.non_tensor_batch["raw_prompt"]]

        # Extract feedback if available and include_environment_feedback is enabled
        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, success_reward_threshold=self_distillation_cfg.success_reward_threshold)
        solution_strs = [
            self._get_solution(
                i,
                success_by_uid,
                batch.non_tensor_batch["uid"],
                response_texts,
                self_distillation_cfg.dont_reprompt_on_self_success,
                self_distillation_cfg.get("remove_thinking_from_demonstration", False),
            )
            for i in range(batch_size)
        ]

        def _build_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)

            # If feedback_only_without_solution is True, only use feedback when no solution exists
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            # build solution section
            solution_section = ""
            if has_solution:
                solution_section = self_distillation_cfg.solution_template.format(
                    successful_previous_attempt=solution_strs[i]
                )

            # build feedback section
            feedback_section = ""
            if use_feedback:
                feedback_section = self_distillation_cfg.feedback_template.format(
                    feedback_raw=feedback_list[i]
                )

            # combine solution and feedback sections
            if use_feedback or has_solution:
                reprompt_text = self_distillation_cfg.reprompt_template.format(
                    prompt=prompt_texts[i],
                    solution=solution_section,
                    feedback=feedback_section,
                )
            else:
                reprompt_text = prompt_texts[i]

            return system_messages + [
                {"role": "user", "content": reprompt_text},
            ]


        messages = [_build_teacher_message(i) for i in range(batch_size)]
        enable_thinking = self.config.data.apply_chat_template_kwargs.get("enable_thinking", True) if self.config.data.apply_chat_template_kwargs else True
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=self_distillation_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
        teacher_response_start_idx = torch.full(
            (batch_size,),
            teacher_prompt["input_ids"].shape[1],
            dtype=torch.long,
            device=device,
        )

        # Compute which samples actually use feedback (accounting for environment_feedback_only_without_solution)
        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]

        # self_distillation_mask is True if sample has a solution OR feedback is used (i.e., will get a reprompted message)
        self_distillation_mask = torch.tensor(
            [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)],
            dtype=torch.float32,
            device=device
        )

        uids = set(batch.non_tensor_batch["uid"])
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)
        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
        }
        return DataProto.from_dict(tensors={
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "teacher_response_start_idx": teacher_response_start_idx,
            "self_distillation_mask": self_distillation_mask,
        }), metrics

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # The stamp is intentionally created before rollout and retained on the
        # controller's copy of the batch.  Reuse later requires this exact
        # prompt/image/processor provenance; missing or stale stamps simply
        # select the legacy teacher preprocessing path.
        self._stamp_rollout_teacher_input_provenance(batch)
        reward_model_keys = (
            set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt", "teacher_prompt"})
            & batch.non_tensor_batch.keys()
        )
        teacher_image_key = self.config.actor_rollout_ref.actor.get("self_distillation", {}).get("teacher_image_key", None)
        if teacher_image_key and teacher_image_key in batch.non_tensor_batch:
            reward_model_keys.add(teacher_image_key)

        if self.async_rollout_mode:
            non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
            gen_batch = batch.pop(
                batch_keys=[],
                non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
            )
            # For agent loop, we need reward model keys to compute score.
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        else:
            # HF rollout calls transformers.generate directly and needs both
            # tokenized prompt tensors and multimodal non-tensor inputs. Keep
            # non-tensor data on the original batch for reward/teacher logic.
            gen_batch = batch.pop(
                batch_keys=list(batch.batch.keys()),
                non_tensor_batch_keys=[],
            )
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            from training.contract import load_contract

            validation_contract, _ = load_contract()
            if validation_contract.get("schema_version") == "vision_opd_ai4s_v8_fullimage_curriculum_static_contract_v1":
                _, output_texts, _ = self._decode_v8_rollout_rows(test_output_gen_batch)
            else:
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            if validation_contract.get("schema_version") == "vision_opd_ai4s_v8_fullimage_curriculum_static_contract_v1":
                prompt_mask = test_batch.batch["attention_mask"][:, : input_ids.shape[1]].to(torch.bool)
                input_texts = [
                    self.tokenizer.decode(
                        input_ids[row][prompt_mask[row]].detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for row in range(input_ids.shape[0])
                ]
            else:
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # HF rollout reuses the actor worker directly so the patched model forward
        # path is executed. AgentLoopManager only supports vLLM/SGLang replicas.
        self.async_rollout_mode = self.config.actor_rollout_ref.rollout.name != "hf"
        if not self.async_rollout_mode:
            self.async_rollout_manager = None
            return

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe
        from verl.utils.checkpoint.integrity import (
            GLOBAL_MANIFEST_NAME,
            GLOBAL_MARKER_V2,
            artifact_binding,
            atomic_json_dump,
            atomic_text_write,
            atomic_torch_save,
            canonical_sha256,
            quarantine_incomplete_global_checkpoint,
            validate_global_checkpoint,
        )

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        integrity_v2 = False
        if self._is_ai4s_v6_full_parameter():
            from training.contract import load_contract

            integrity_v2 = (
                load_contract()[0]["checkpoint"].get("integrity_schema")
                == "sha256_all_payloads_v2"
            )
        completion_marker = os.path.join(local_global_step_folder, ".checkpoint_complete")
        if integrity_v2:
            quarantine_incomplete_global_checkpoint(
                local_global_step_folder,
                expected_step=self.global_steps,
            )

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        if integrity_v2:
            atomic_torch_save(dataloader_state_dict, dataloader_local_path)
        else:
            torch.save(dataloader_state_dict, dataloader_local_path)

        # Persist the best-candidate metadata inside the incoming checkpoint
        # before committing it.  The root-level pointer must not move yet: a
        # crash before the completion marker/tracker would otherwise leave it
        # pointing at an incomplete checkpoint.
        best_metadata_written = self._write_best_checkpoint_metadata(directory=local_global_step_folder)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        # Commit the checkpoint only after all worker state and the dataloader
        # state exist.  Retention ignores directories without this marker, so
        # a crash during a future save cannot displace the last valid tracker
        # target.  Both files use atomic replace to avoid torn metadata.
        if integrity_v2:
            global_artifacts = [
                artifact_binding(local_global_step_folder, "actor/CHECKPOINT_COMPLETE.json"),
                artifact_binding(local_global_step_folder, "data.pt"),
            ]
            global_artifacts.sort(key=lambda item: item["relative_path"])
            global_manifest = {
                "schema_version": GLOBAL_MARKER_V2,
                "global_step": int(self.global_steps),
                "artifacts": global_artifacts,
                "artifact_inventory_sha256": canonical_sha256(global_artifacts),
            }
            atomic_json_dump(
                global_manifest,
                os.path.join(local_global_step_folder, GLOBAL_MANIFEST_NAME),
            )
            atomic_text_write(f"{self.global_steps}\n", completion_marker)
            validate_global_checkpoint(
                local_global_step_folder,
                expected_step=self.global_steps,
                allow_legacy_marker=False,
            )
        else:
            completion_tmp = completion_marker + ".tmp"
            with open(completion_tmp, "w", encoding="utf-8") as f:
                f.write(f"{self.global_steps}\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(completion_tmp, completion_marker)

        if integrity_v2:
            atomic_text_write(str(self.global_steps), local_latest_checkpointed_iteration)
        else:
            tracker_tmp = local_latest_checkpointed_iteration + ".tmp"
            with open(tracker_tmp, "w", encoding="utf-8") as f:
                f.write(str(self.global_steps))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tracker_tmp, local_latest_checkpointed_iteration)
        # The checkpoint is now durable and discoverable.  It is safe to move
        # the root best pointer; if this write is interrupted, resume reconciles
        # it from the committed step-local candidate above.
        if best_metadata_written:
            self._write_best_checkpoint_metadata()
        self.actor_rollout_wg.finalize_checkpoint(actor_local_path, max_actor_ckpt_to_keep)
        self._maybe_write_v6_monitoring_request(self.global_steps)

    def _resolve_v6_monitoring_contract(self):
        return None

    def _maybe_write_v6_monitoring_request(self, global_step: int) -> Optional[str]:
        return None

    def _is_v6_monitoring_segment_boundary(self, global_step: int) -> bool:
        """Return true only for a formal, science-gated V6 segment boundary.

        One- and ten-step V6 smoke runs use the same full-parameter actor but
        intentionally have no formal full-run monitoring contract.  They must
        save their terminal checkpoint without attempting to index a missing
        monitoring contract.
        """

        resolved = self._resolve_v6_monitoring_contract()
        if resolved is None:
            return False
        evaluation_steps = {
            int(step) for step in resolved["contract"]["monitoring"]["evaluation_steps"]
            if int(step) > 0
        }
        return int(global_step) in evaluation_steps

    def _require_v6_checkpoint_science_gate(self, global_step: int) -> Optional[dict[str, Any]]:
        """Require an independently recomputed passing audit before a new segment."""

        resolved = self._resolve_v6_monitoring_contract()
        if resolved is None:
            return None
        evaluation_steps = [int(step) for step in resolved["contract"]["monitoring"]["evaluation_steps"]]
        if global_step == 0:
            return None
        if global_step not in evaluation_steps:
            raise RuntimeError(
                f"Formal V6 resume step {global_step} is not a scientific segment boundary"
            )
        run_root = os.path.abspath(os.environ["VERL_V6_RUN_ROOT"])
        from verl.trainer.ppo.v6_monitoring import validate_checkpoint_science_gate

        return validate_checkpoint_science_gate(
            run_root=run_root,
            global_step=int(global_step),
            baseline=resolved["baseline"],
            require_pass=True,
        )

    def _set_v6_gpu_phase(self, phase: str) -> None:
        """Publish the driver phase used by the out-of-process NVML sampler.

        The file is evidence only: no training decision reads it.  Formal V6
        launchers bind it inside the run root, and an atomic replace prevents
        the sampler from observing a partial value.  A malformed path fails
        closed instead of silently dropping the utilization evidence.
        """

        path = os.environ.get("VERL_V6_GPU_PHASE_FILE")
        if path is None:
            return
        if phase not in {"idle", "rollout", "actor_update"}:
            raise ValueError(f"Unsupported formal V6 GPU phase: {phase!r}")
        if not os.path.isabs(path):
            raise RuntimeError("VERL_V6_GPU_PHASE_FILE must be absolute")
        run_root = os.environ.get("VERL_V6_RUN_ROOT")
        if not isinstance(run_root, str) or not os.path.isabs(run_root):
            raise RuntimeError("Formal V6 GPU phase evidence requires VERL_V6_RUN_ROOT")
        expected = os.path.join(os.path.realpath(run_root), "audit", "gpu_phase.txt")
        if os.path.realpath(path) != expected:
            raise RuntimeError("Formal V6 GPU phase path is outside its canonical run root")
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        if os.path.islink(path):
            raise RuntimeError("Formal V6 GPU phase path must not be a symlink")
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(phase + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _consider_best_checkpoint(self, val_metrics: dict[str, Any]) -> bool:
        if not self.best_metric_key:
            return False
        if self.best_metric_key not in val_metrics:
            available = sorted(key for key in val_metrics if key.startswith("val-"))
            raise KeyError(
                f"Configured best metric {self.best_metric_key!r} was not produced. Available metrics: {available}"
            )
        value = float(val_metrics[self.best_metric_key])
        if not np.isfinite(value):
            raise FloatingPointError(f"Best-checkpoint metric {self.best_metric_key} is non-finite: {value}")
        improved = self.best_metric_value is None
        if self.best_metric_value is not None:
            improved = value > self.best_metric_value if self.best_metric_mode == "max" else value < self.best_metric_value
        if improved:
            self.best_metric_value = value
            self.best_metric_step = int(self.global_steps)
        return improved

    def _write_best_checkpoint_metadata(self, directory: str | None = None, *, require_current_step: bool = True) -> bool:
        if self.best_metric_value is None or self.best_metric_step is None:
            return False
        if require_current_step and self.best_metric_step != self.global_steps:
            return False
        destination_dir = directory or self.config.trainer.default_local_dir
        os.makedirs(destination_dir, exist_ok=True)
        destination = os.path.join(destination_dir, "best_checkpoint.json")
        temporary = destination + ".tmp"
        payload = {
            "schema_version": "vision_opd_best_checkpoint_v1",
            "metric_key": self.best_metric_key,
            "metric_mode": self.best_metric_mode,
            "metric_value": self.best_metric_value,
            "global_step": self.best_metric_step,
            "checkpoint_path": os.path.join(
                self.config.trainer.default_local_dir, f"global_step_{self.best_metric_step}"
            ),
        }
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return True

    def _load_best_checkpoint_metadata(self) -> None:
        if not self.best_metric_key:
            return
        checkpoint_root = os.path.abspath(self.config.trainer.default_local_dir)
        root_metadata = os.path.join(checkpoint_root, "best_checkpoint.json")
        candidate_paths = []
        if os.path.isfile(root_metadata):
            candidate_paths.append(root_metadata)
        for step in range(self.global_steps + 1):
            step_metadata = os.path.join(checkpoint_root, f"global_step_{step}", "best_checkpoint.json")
            if os.path.isfile(step_metadata):
                candidate_paths.append(step_metadata)
        if not candidate_paths:
            return

        candidates = []
        errors = []
        for path in candidate_paths:
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload.get("metric_key") != self.best_metric_key or payload.get("metric_mode") != self.best_metric_mode:
                    raise ValueError("validation contract mismatch")
                value = float(payload["metric_value"])
                step = int(payload["global_step"])
                expected_step_path = os.path.join(checkpoint_root, f"global_step_{step}")
                checkpoint_path = os.path.abspath(os.fspath(payload["checkpoint_path"]))
                if not np.isfinite(value) or step < 0 or step > self.global_steps:
                    raise ValueError("invalid or uncommitted metric step")
                if checkpoint_path != expected_step_path:
                    raise ValueError("checkpoint_path does not match the checkpoint root and global step")
                if not os.path.isfile(os.path.join(checkpoint_path, "data.pt")):
                    raise ValueError("checkpoint has no dataloader state")
                marker = os.path.join(checkpoint_path, ".checkpoint_complete")
                if not os.path.isfile(marker):
                    raise ValueError("checkpoint has no matching atomic completion marker")
                with open(marker, encoding="utf-8") as marker_handle:
                    if marker_handle.read().strip() != str(step):
                        raise ValueError("checkpoint has no matching atomic completion marker")
                actor_path = os.path.join(checkpoint_path, "actor")
                if not os.path.isdir(actor_path):
                    raise ValueError("checkpoint has no actor state")
                formal_v6 = self._is_ai4s_v6_full_parameter()
                expected_world_size = 8 if formal_v6 else 4
                for kind in ("model", "optim", "extra_state", "rollout_rng"):
                    ranks = {
                        int(match.group(1))
                        for name in os.listdir(actor_path)
                        if (
                            match := re.fullmatch(
                                rf"{kind}_world_size_{expected_world_size}_rank_(\d+)\.pt", name
                            )
                        )
                    }
                    if ranks != set(range(expected_world_size)):
                        raise ValueError(
                            f"checkpoint has incomplete {expected_world_size}-rank {kind} state: {sorted(ranks)}"
                        )
                required_artifacts = ["checkpoint_provenance.json"]
                if formal_v6:
                    # Full-parameter state lives entirely in the eight FSDP
                    # shards.  Adapter/standalone-merger files are forbidden,
                    # while the inner distributed commit marker is mandatory.
                    required_artifacts.append("CHECKPOINT_COMPLETE.json")
                    forbidden_artifacts = (
                        "lora_adapter/adapter_model.safetensors",
                        "native_visual_merger/model.safetensors",
                        "native_visual_merger/manifest.json",
                    )
                    unexpected = [
                        relative
                        for relative in forbidden_artifacts
                        if os.path.exists(os.path.join(actor_path, relative))
                    ]
                    if unexpected:
                        raise ValueError(
                            f"formal V6 checkpoint contains legacy actor artifacts: {unexpected}"
                        )
                else:
                    required_artifacts.extend(
                        (
                            "lora_adapter/adapter_model.safetensors",
                            "native_visual_merger/model.safetensors",
                            "native_visual_merger/manifest.json",
                        )
                    )
                for relative in required_artifacts:
                    if not os.path.isfile(os.path.join(actor_path, relative)):
                        raise ValueError(f"checkpoint is missing committed actor artifact: {relative}")
                candidates.append((value, step))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: {exc}")
        if not candidates:
            raise ValueError("No valid committed best-checkpoint metadata remains: " + "; ".join(errors))

        if self.best_metric_mode == "max":
            value, step = max(candidates, key=lambda item: (item[0], -item[1]))
        else:
            value, step = min(candidates, key=lambda item: (item[0], item[1]))
        self.best_metric_value = value
        self.best_metric_step = step
        # Reconcile a stale/missing root pointer after a crash between tracker
        # commit and the root metadata replace.
        self._write_best_checkpoint_metadata(require_current_step=False)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        integrity_v2 = False
        if self._is_ai4s_v6_full_parameter():
            from training.contract import load_contract

            integrity_v2 = (
                load_contract()[0]["checkpoint"].get("integrity_schema")
                == "sha256_all_payloads_v2"
            )

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder,
                allow_legacy_marker=not integrity_v2,
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        from verl.utils.checkpoint.integrity import validate_global_checkpoint

        # Validate data.pt and the recursively hash-bound actor marker before
        # any worker or controller calls torch.load.
        validate_global_checkpoint(
            global_step_folder,
            expected_step=self.global_steps,
            allow_legacy_marker=not integrity_v2,
        )

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if not os.path.exists(dataloader_local_path):
            raise FileNotFoundError(
                f"Committed checkpoint is missing data.pt; exact resume is impossible: {dataloader_local_path}"
            )
        dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
        self.train_dataloader.load_state_dict(dataloader_state_dict)
        restore_exact = _nested_exact_equal(
            dataloader_state_dict, self.train_dataloader.state_dict()
        )
        if self._is_ai4s_v6_full_parameter():
            self._v6_dataloader_restore_exact = restore_exact
        else:
            self._exp3_dataloader_restore_exact = restore_exact
        if not restore_exact:
            raise RuntimeError("Dataloader state differs immediately after checkpoint restore")
        return self.global_steps

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = int(self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size"))
            if minibatch_size <= 0 or minibatch_size % dp_size or batch_size % minibatch_size:
                raise ValueError(
                    "Mini-batch-preserving balance requires a positive PPO mini-batch divisible by DP size "
                    "and an integral number of mini-batches: "
                    f"batch={batch_size}, mini_batch={minibatch_size}, dp={dp_size}"
                )
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                if keep_minibatch:
                    # Each rank receives one contiguous slice of the flattened
                    # index list.  Preserve the same mini-batch ordinal on all
                    # ranks; otherwise sorting the whole per-rank partition can
                    # swap mini-batch 0/1 on only some ranks and make a single
                    # FSDP optimizer step aggregate samples from two different
                    # global mini-batches.  Bubble ordering remains safe inside
                    # each mini-batch segment.
                    local_minibatch_size = minibatch_size // dp_size
                    if local_minibatch_size <= 0 or len(partition) % local_minibatch_size:
                        raise ValueError(
                            "Mini-batch-preserving balance produced an invalid per-rank partition: "
                            f"partition={len(partition)}, local_minibatch={local_minibatch_size}"
                        )
                    ordered_partition = []
                    for start in range(0, len(partition), local_minibatch_size):
                        segment = partition[start : start + local_minibatch_size]
                        segment.sort(key=lambda x: (workload_lst[x], x))
                        ordered_partition.extend(segment[::2] + segment[1::2][::-1])
                else:
                    partition.sort(key=lambda x: (workload_lst[x], x))
                    ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _balance_v6_rollout_uid_groups(
        self, repeated_prompts: DataProto
    ) -> tuple[DataProto, torch.Tensor | None, dict[str, float]]:
        """Balance same-UID rollout groups across ranks and return an exact inverse.

        The controller reorders only the generation request.  The returned
        inverse is applied to generated responses/routes before they are unioned
        with the original training batch, so sample alignment and the actor
        objective remain byte-for-byte in the original order.  A UID group is
        indivisible, which also preserves the n-way shared-prefix optimization.
        """

        rollout = self.config.actor_rollout_ref.rollout
        if not bool(rollout.get("hf_rollout_group_balance", False)):
            return repeated_prompts, None, {}
        if rollout.name != "hf" or not bool(rollout.get("hf_use_replicated_module", False)):
            raise RuntimeError("Formal rollout UID balancing requires the independent HF replica")
        rollout_n = int(rollout.n)
        if rollout_n <= 0 or len(repeated_prompts) % rollout_n:
            raise RuntimeError("Rollout UID balancing received an invalid n-way batch")
        if repeated_prompts.non_tensor_batch is None or "uid" not in repeated_prompts.non_tensor_batch:
            raise RuntimeError("Rollout UID balancing requires batch-aligned uid values")

        uids = list(repeated_prompts.non_tensor_batch["uid"])
        prompt_count = len(repeated_prompts) // rollout_n
        has_attention_mask = (
            repeated_prompts.batch is not None
            and "attention_mask" in repeated_prompts.batch.keys()
        )
        extra_infos = repeated_prompts.non_tensor_batch.get("extra_info")
        curriculum_config = self.config.actor_rollout_ref.model.get(
            "vision_token_compressor", {}
        ).get("curriculum")
        active_retention_bps = repeated_prompts.meta_info.get("visual_token_retention_bps")
        if curriculum_config is not None:
            if (
                isinstance(active_retention_bps, bool)
                or not isinstance(active_retention_bps, int)
                or not 1 <= active_retention_bps <= 10_000
            ):
                raise RuntimeError(
                    "curriculum rollout balancing requires an integer visual_token_retention_bps"
                )
        selected_max_pixels = int(
            getattr(self.config.actor_rollout_ref.model, "processor_max_pixels", 0)
        )
        if not has_attention_mask and (extra_infos is None or selected_max_pixels <= 0):
            raise RuntimeError(
                "Raw-prompt rollout balancing requires bound extra_info visual capacity metadata"
            )
        group_costs: list[int] = []
        for prompt_index in range(prompt_count):
            start = prompt_index * rollout_n
            end = start + rollout_n
            group_uids = uids[start:end]
            if len(set(map(str, group_uids))) != 1:
                raise RuntimeError("Interleaved rollout group contains multiple UIDs")
            if has_attention_mask:
                group_attention = repeated_prompts.batch["attention_mask"][start:end]
                if not torch.equal(group_attention, group_attention[:1].expand_as(group_attention)):
                    raise RuntimeError("Same-UID rollout group has different prompt attention masks")
                prompt_tokens = int(group_attention[0].sum().item())
                raw_patches = 0
                if "multi_modal_inputs" in repeated_prompts.non_tensor_batch:
                    entry = getattr(
                        repeated_prompts.non_tensor_batch["multi_modal_inputs"][start], "data",
                        repeated_prompts.non_tensor_batch["multi_modal_inputs"][start],
                    )
                    if isinstance(entry, dict) and entry.get("image_grid_thw") is not None:
                        grid = torch.as_tensor(entry["image_grid_thw"], dtype=torch.long)
                        if grid.ndim == 1:
                            grid = grid.unsqueeze(0)
                        if grid.ndim != 2 or grid.shape[-1] != 3 or bool((grid <= 0).any().item()):
                            raise RuntimeError("Rollout group has an invalid image_grid_thw")
                        raw_patches = int(grid.prod(dim=-1).sum().item())
                group_costs.append(prompt_tokens + raw_patches)
                continue

            group_metadata_costs: list[int] = []
            for extra_info in extra_infos[start:end]:
                if not isinstance(extra_info, dict):
                    raise RuntimeError("Rollout extra_info must be a dictionary")
                profiles = extra_info.get("visual_capacity_profiles")
                profile = profiles.get(str(selected_max_pixels)) if isinstance(profiles, dict) else None
                if not isinstance(profile, dict) or profile.get("max_pixels") != selected_max_pixels:
                    raise RuntimeError("Rollout extra_info lacks the selected visual capacity profile")
                merged_prompt_tokens = profile.get("merged_student_prompt_tokens")
                raw_patch_tokens = profile.get("raw_patch_tokens_per_view")
                if curriculum_config is not None:
                    from verl.models.transformers.visual_token_curriculum import (
                        curriculum_merged_prompt_tokens,
                    )

                    merged_prompt_tokens = curriculum_merged_prompt_tokens(
                        profile,
                        retention_bps=active_retention_bps,
                    )
                for label, value in (
                    ("merged_student_prompt_tokens", merged_prompt_tokens),
                    ("raw_patch_tokens_per_view", raw_patch_tokens),
                ):
                    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                        raise RuntimeError(f"Rollout capacity metadata has invalid {label}")
                group_metadata_costs.append(int(merged_prompt_tokens) + int(raw_patch_tokens))
            if len(set(group_metadata_costs)) != 1:
                raise RuntimeError("Same-UID rollout group has inconsistent capacity metadata")
            group_costs.append(group_metadata_costs[0])

        dp_size = self._get_dp_size(self.actor_rollout_wg, "rollout")
        if prompt_count % dp_size:
            raise RuntimeError(
                f"Rollout UID groups ({prompt_count}) must divide evenly across {dp_size} ranks"
            )
        from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

        prompt_partitions = get_seqlen_balanced_partitions(
            group_costs, k_partitions=dp_size, equal_size=True
        )
        expanded_order = [
            prompt_index * rollout_n + trajectory_index
            for partition in prompt_partitions
            for prompt_index in partition
            for trajectory_index in range(rollout_n)
        ]
        if sorted(expanded_order) != list(range(len(repeated_prompts))):
            raise RuntimeError("Rollout UID balancing did not produce a permutation")
        order = torch.tensor(expanded_order, dtype=torch.long)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), dtype=torch.long)
        balanced = repeated_prompts[order]
        partition_costs = [sum(group_costs[index] for index in part) for part in prompt_partitions]
        metrics = {
            "rollout_balance/cost_min": float(min(partition_costs)),
            "rollout_balance/cost_max": float(max(partition_costs)),
            "rollout_balance/cost_max_min_ratio": float(max(partition_costs) / max(min(partition_costs), 1)),
        }
        return balanced, inverse, metrics

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        batch.meta_info["global_steps"] = self.global_steps
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def _exp3_rollout_state_digests(self) -> list[dict[str, Any]]:
        states = self.actor_rollout_wg.exp3_rollout_state_digest()
        if not isinstance(states, list) or len(states) != 4:
            raise RuntimeError(f"Exp3 rollout-state audit expected four rank results, got {states!r}")
        states = sorted(states, key=lambda item: int(item["rank"]))
        if [int(item["rank"]) for item in states] != list(range(4)):
            raise RuntimeError(f"Exp3 rollout-state audit rank coverage is invalid: {states!r}")
        if any(int(item.get("world_size", -1)) != 4 for item in states):
            raise RuntimeError(f"Exp3 rollout-state audit world-size mismatch: {states!r}")
        return states

    def _write_exp3_post_checkpoint_rollout_golden(self, probe: DataProto, audit_dir: str) -> None:
        """Exercise post-update replica sync from the exact checkpoint RNG."""

        os.makedirs(audit_dir, exist_ok=False)
        input_path = os.path.join(audit_dir, "input.dataproto")
        output_path = os.path.join(audit_dir, "expected_output.dataproto")
        _atomic_dataproto_file(input_path, probe)
        output = self.actor_rollout_wg.generate_sequences(deepcopy(probe))
        output.meta_info.pop("timing", None)
        states = self._exp3_rollout_state_digests()
        _atomic_dataproto_file(output_path, output)
        _atomic_json_file(
            os.path.join(audit_dir, "golden.json"),
            {
                "schema_version": "exp3_post_checkpoint_next_rollout_v1",
                "passed": True,
                "global_step": self.global_steps,
                "world_size": 4,
                "input_path": os.path.abspath(input_path),
                "expected_output_path": os.path.abspath(output_path),
                "post_rollout_rank_states": states,
            },
        )

    def _verify_exp3_fresh_next_rollout(self, audit_dir: str) -> dict[str, Any]:
        golden_path = os.path.join(audit_dir, "golden.json")
        if not os.path.isfile(golden_path):
            raise FileNotFoundError(f"Exp3 next-rollout golden manifest is missing: {golden_path}")
        with open(golden_path, encoding="utf-8") as handle:
            golden = json.load(handle)
        if (
            golden.get("schema_version") != "exp3_post_checkpoint_next_rollout_v1"
            or golden.get("passed") is not True
            or int(golden.get("global_step", -1)) != self.global_steps
            or int(golden.get("world_size", -1)) != 4
        ):
            raise RuntimeError(f"Exp3 next-rollout golden contract is invalid: {golden}")
        input_path = os.path.abspath(golden["input_path"])
        expected_path = os.path.abspath(golden["expected_output_path"])
        expected_parent = os.path.abspath(audit_dir) + os.sep
        if not input_path.startswith(expected_parent) or not expected_path.startswith(expected_parent):
            raise RuntimeError("Exp3 next-rollout golden paths escape the audited directory")
        probe = DataProto.load_from_disk(input_path)
        expected = DataProto.load_from_disk(expected_path)
        actual = self.actor_rollout_wg.generate_sequences(probe)
        actual.meta_info.pop("timing", None)
        states = self._exp3_rollout_state_digests()
        gates = _exp3_rollout_exact_gates(expected, actual)
        gates.update(
            {
                "post_rollout_rng_states_bitwise_exact": states == golden.get("post_rollout_rank_states"),
                "native_rollout_replica_resynchronized_on_all_ranks": all(
                    item.get("native_rollout_replica_enabled") is True
                    and item.get("native_rollout_replica_dirty") is False
                    for item in states
                ),
            }
        )
        return {
            "schema_version": "exp3_fresh_process_next_rollout_v1",
            "passed": all(gates.values()),
            "failed_gates": sorted(key for key, value in gates.items() if not value),
            "gates": gates,
            "global_step": self.global_steps,
            "world_size": 4,
            "post_rollout_rank_states": states,
        }

    def _v6_rollout_state_digests(self) -> list[dict[str, Any]]:
        """Collect the independent eight-rank formal rollout-state schema."""

        states = self.actor_rollout_wg.v6_rollout_state_digest()
        if not isinstance(states, list) or len(states) != 8:
            raise RuntimeError(f"V6 rollout-state audit expected eight rank results, got {states!r}")
        states = sorted(states, key=lambda item: int(item["rank"]))
        if [int(item["rank"]) for item in states] != list(range(8)):
            raise RuntimeError(f"V6 rollout-state audit rank coverage is invalid: {states!r}")
        is_v8 = self._formal_compressor_algorithm() == "qwen35_cdpruner_v1"
        expected_schema = (
            "vision_opd_ai4s_v8_rollout_state_v1"
            if is_v8
            else "vision_opd_ai4s_v6_rollout_state_v2"
        )
        if any(
            item.get("schema_version") != expected_schema
            or int(item.get("world_size", -1)) != 8
            or item.get("native_rollout_replica_dirty") is not False
            or item.get("native_rollout_replica_parked") is not True
            for item in states
        ):
            raise RuntimeError(f"V6 rollout-state audit contract mismatch: {states!r}")
        if is_v8:
            from training.contract import load_contract
            from verl.models.transformers.visual_token_curriculum import (
                VisualTokenCurriculum,
            )

            release_contract, _ = load_contract()
            curriculum = VisualTokenCurriculum.from_mapping(
                release_contract["compressor"]["curriculum"]
            )
            expected_keys = {
                "schema_version",
                "rank",
                "world_size",
                "torch_random_state_sha256",
                "generation_random_state_sha256",
                "native_rollout_replica_enabled",
                "native_rollout_replica_dirty",
                "native_rollout_replica_parked",
                "visual_token_curriculum_state",
            }
            canonical_states = []
            for item in states:
                if set(item) != expected_keys:
                    raise RuntimeError(f"V8 rollout-state audit has a non-canonical inventory: {item!r}")
                state = item.get("visual_token_curriculum_state")
                if not isinstance(state, dict):
                    raise RuntimeError(f"V8 rollout-state audit has no curriculum state: {item!r}")
                completed = state.get("completed_optimizer_steps")
                if isinstance(completed, bool) or not isinstance(completed, int):
                    raise RuntimeError(f"V8 rollout-state audit has an invalid curriculum step: {item!r}")
                expected_state = curriculum.runtime_state(completed)
                if state != expected_state:
                    raise RuntimeError(
                        "V8 rollout-state audit curriculum schedule/control state drift: "
                        f"expected={expected_state!r}, actual={state!r}"
                    )
                canonical_states.append(state)
            if any(state != canonical_states[0] for state in canonical_states[1:]):
                raise RuntimeError(
                    "V8 rollout-state audit found a curriculum state disagreement across ranks: "
                    f"{canonical_states!r}"
                )
        return states

    def _write_v6_post_checkpoint_rollout_golden(self, probe: DataProto, audit_dir: str) -> None:
        """Bind a post-checkpoint rollout and RNG state to the eight-rank V6 run."""

        os.makedirs(audit_dir, exist_ok=False)
        input_path = os.path.join(audit_dir, "input.dataproto")
        output_path = os.path.join(audit_dir, "expected_output.dataproto")
        algorithm = self._formal_compressor_algorithm()
        is_v8 = algorithm == "qwen35_cdpruner_v1"
        curriculum_state = None
        if is_v8:
            from training.contract import load_contract
            from verl.models.transformers.visual_token_curriculum import (
                VisualTokenCurriculum,
            )

            release_contract, _ = load_contract()
            curriculum = VisualTokenCurriculum.from_mapping(
                release_contract["compressor"]["curriculum"]
            )
            # Checkpoint S commits update S.  The first rollout after restoring
            # that checkpoint must therefore use completed=S, not the state
            # completed=S-1 that produced update S's training rollout.
            curriculum_state = _post_checkpoint_next_curriculum_state(
                curriculum, int(self.global_steps)
            )
            probe.meta_info.update(
                {
                    "visual_token_curriculum_schema_version": curriculum_state[
                        "schema_version"
                    ],
                    "visual_token_curriculum_schedule_sha256": curriculum_state[
                        "schedule_sha256"
                    ],
                    "visual_token_curriculum_completed_steps": curriculum_state[
                        "completed_optimizer_steps"
                    ],
                    "visual_token_retention_bps": curriculum_state["retention_bps"],
                }
            )
            self.actor_rollout_wg.set_visual_token_curriculum_step(
                completed_optimizer_steps=curriculum_state[
                    "completed_optimizer_steps"
                ]
            )
        _atomic_dataproto_file(input_path, probe)
        output = self.actor_rollout_wg.generate_sequences(deepcopy(probe))
        output.meta_info.pop("timing", None)
        states = self._v6_rollout_state_digests()
        _atomic_dataproto_file(output_path, output)
        payload = {
            "schema_version": (
                "vision_opd_ai4s_v8_post_checkpoint_next_rollout_v1"
                if is_v8
                else "vision_opd_ai4s_v6_post_checkpoint_next_rollout_v1"
            ),
            "passed": True,
            "global_step": self.global_steps,
            "world_size": 8,
            "input_path": os.path.abspath(input_path),
            "expected_output_path": os.path.abspath(output_path),
            "post_rollout_rank_states": states,
        }
        if is_v8:
            payload["algorithm"] = algorithm
            payload["visual_token_curriculum_state"] = curriculum_state
        _atomic_json_file(
            os.path.join(audit_dir, "golden.json"),
            payload,
        )

    def _verify_v6_fresh_next_rollout(self, audit_dir: str) -> dict[str, Any]:
        golden_path = os.path.join(audit_dir, "golden.json")
        if not os.path.isfile(golden_path):
            raise FileNotFoundError(f"V6 next-rollout golden manifest is missing: {golden_path}")
        with open(golden_path, encoding="utf-8") as handle:
            golden = json.load(handle)
        algorithm = self._formal_compressor_algorithm()
        is_v8 = algorithm == "qwen35_cdpruner_v1"
        expected_schema = (
            "vision_opd_ai4s_v8_post_checkpoint_next_rollout_v1"
            if is_v8
            else "vision_opd_ai4s_v6_post_checkpoint_next_rollout_v1"
        )
        curriculum_state = None
        if is_v8:
            from training.contract import load_contract
            from verl.models.transformers.visual_token_curriculum import (
                VisualTokenCurriculum,
            )

            release_contract, _ = load_contract()
            curriculum = VisualTokenCurriculum.from_mapping(
                release_contract["compressor"]["curriculum"]
            )
            # A restored checkpoint at S must reproduce the next rollout with
            # exactly S optimizer updates already committed.
            curriculum_state = _post_checkpoint_next_curriculum_state(
                curriculum, int(self.global_steps)
            )
        if (
            golden.get("schema_version") != expected_schema
            or golden.get("passed") is not True
            or int(golden.get("global_step", -1)) != self.global_steps
            or int(golden.get("world_size", -1)) != 8
            or (is_v8 and golden.get("algorithm") != algorithm)
            or (
                is_v8
                and golden.get("visual_token_curriculum_state") != curriculum_state
            )
        ):
            raise RuntimeError(f"V6 next-rollout golden contract is invalid: {golden}")
        input_path = os.path.abspath(golden["input_path"])
        expected_path = os.path.abspath(golden["expected_output_path"])
        expected_parent = os.path.abspath(audit_dir) + os.sep
        if not input_path.startswith(expected_parent) or not expected_path.startswith(expected_parent):
            raise RuntimeError("V6 next-rollout golden paths escape the audited directory")
        probe = DataProto.load_from_disk(input_path)
        expected = DataProto.load_from_disk(expected_path)
        if is_v8:
            probe_curriculum = {
                "schema_version": probe.meta_info.get(
                    "visual_token_curriculum_schema_version"
                ),
                "schedule_sha256": probe.meta_info.get(
                    "visual_token_curriculum_schedule_sha256"
                ),
                "completed_optimizer_steps": probe.meta_info.get(
                    "visual_token_curriculum_completed_steps"
                ),
                "retention_bps": probe.meta_info.get("visual_token_retention_bps"),
            }
            if probe_curriculum != {
                key: curriculum_state[key] for key in probe_curriculum
            }:
                raise RuntimeError("fresh-process golden probe curriculum metadata drift")
            self.actor_rollout_wg.set_visual_token_curriculum_step(
                completed_optimizer_steps=curriculum_state[
                    "completed_optimizer_steps"
                ]
            )
        actual = self.actor_rollout_wg.generate_sequences(probe)
        actual.meta_info.pop("timing", None)
        states = self._v6_rollout_state_digests()
        gates = (
            _v8_rollout_exact_gates(expected, actual)
            if is_v8
            else _v6_rollout_exact_gates(expected, actual)
        )
        gates.update(
            {
                "post_rollout_rng_states_bitwise_exact": states == golden.get("post_rollout_rank_states"),
                "hf_replicated_rollout_clean_and_parked_on_all_ranks": all(
                    item.get("native_rollout_replica_dirty") is False
                    and item.get("native_rollout_replica_parked") is True
                    for item in states
                ),
                **(
                    {"curriculum_runtime_state_exact": True}
                    if is_v8
                    else {}
                ),
            }
        )
        return {
            "schema_version": (
                "vision_opd_ai4s_v8_fresh_process_next_rollout_v1"
                if is_v8
                else "vision_opd_ai4s_v6_fresh_process_next_rollout_v1"
            ),
            "passed": all(gates.values()),
            "failed_gates": sorted(key for key, value in gates.items() if not value),
            "gates": gates,
            "global_step": self.global_steps,
            "world_size": 8,
            "post_rollout_rank_states": states,
            **(
                {
                    "algorithm": algorithm,
                    "visual_token_curriculum_state": curriculum_state,
                }
                if is_v8
                else {}
            ),
        }

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            group_name=self.config.trainer.get("group_name", None),
        )

        self.global_steps = 0
        self._reject_v6_legacy_audit_controls()

        # load checkpoint before doing anything
        loaded_step = self._load_checkpoint()
        self._load_best_checkpoint_metadata()

        formal_v6 = self._is_ai4s_v6_full_parameter()
        formal_v8 = formal_v6 and self._is_ai4s_v8_cdpruner()
        self._resolve_v6_monitoring_contract()
        if formal_v6:
            self._set_v6_gpu_phase("idle")
        if loaded_step > 0:
            # Reconcile the narrow crash window after checkpoint commit but
            # before queue creation. Existing exact requests are only verified.
            self._maybe_write_v6_monitoring_request(loaded_step)
        fresh_audit_dir = os.environ.get(
            "VERL_V6_FRESH_LOAD_AUDIT_DIR" if formal_v6 else "VERL_EXP3_FRESH_LOAD_AUDIT_DIR"
        )
        if fresh_audit_dir:
            if loaded_step <= 0:
                raise RuntimeError(
                    f"{'V6' if formal_v6 else 'Exp3'} fresh-process audit requires a committed checkpoint"
                )
            audit_world_size = 8 if formal_v6 else 4
            worker_paths = [
                os.path.join(fresh_audit_dir, f"worker_rank{rank}.json")
                for rank in range(audit_world_size)
            ]
            missing = [path for path in worker_paths if not os.path.isfile(path)]
            if missing:
                raise RuntimeError(f"Fresh-process worker audit artifacts are missing: {missing}")
            workers = []
            for path in worker_paths:
                with open(path, encoding="utf-8") as handle:
                    workers.append(json.load(handle))
            expected_worker_schema = (
                "vision_opd_ai4s_v6_full_parameter_fresh_worker_load_v1"
                if formal_v6
                else "exp3_fresh_process_worker_load_v1"
            )
            if [worker.get("rank") for worker in workers] != list(range(audit_world_size)) or not all(
                worker.get("schema_version") == expected_worker_schema
                and int(worker.get("world_size", -1)) == audit_world_size
                and worker.get("passed") is True
                for worker in workers
            ):
                raise RuntimeError(f"Fresh-process worker audit failed: {workers}")
            next_rollout_dir = os.environ.get(
                "VERL_V6_NEXT_ROLLOUT_AUDIT_DIR" if formal_v6 else "VERL_EXP3_NEXT_ROLLOUT_AUDIT_DIR"
            )
            if not next_rollout_dir:
                raise RuntimeError(
                    f"{'V6' if formal_v6 else 'Exp3'} fresh-process audit requires its next-rollout audit dir"
                )
            next_rollout = (
                self._verify_v6_fresh_next_rollout(next_rollout_dir)
                if formal_v6
                else self._verify_exp3_fresh_next_rollout(next_rollout_dir)
            )
            gates = {
                "loaded_committed_checkpoint": loaded_step > 0,
                "dataloader_state_exact": bool(
                    getattr(
                        self,
                        "_v6_dataloader_restore_exact" if formal_v6 else "_exp3_dataloader_restore_exact",
                        False,
                    )
                ),
                "all_worker_load_audits_passed": len(workers) == audit_world_size and all(
                    worker["passed"] for worker in workers
                ),
                "loaded_step_matches_training_horizon": loaded_step == self.total_training_steps,
                "next_rollout_and_rng_bitwise_exact": next_rollout["passed"] is True,
            }
            payload = {
                "schema_version": (
                    (
                        "vision_opd_ai4s_v8_full_parameter_fresh_process_restore_v1"
                        if formal_v8
                        else "vision_opd_ai4s_v6_full_parameter_fresh_process_restore_v1"
                    )
                    if formal_v6
                    else "exp3_fresh_process_restore_v1"
                ),
                "world_size": audit_world_size,
                "passed": all(gates.values()),
                "failed_gates": sorted(key for key, value in gates.items() if not value),
                "gates": gates,
                "loaded_step": loaded_step,
                "worker_audits": workers,
                "next_rollout_audit": next_rollout,
                **(
                    {"algorithm": "qwen35_cdpruner_v1"}
                    if formal_v8
                    else {}
                ),
            }
            _atomic_json_file(os.path.join(fresh_audit_dir, "driver.json"), payload)
            if not payload["passed"]:
                raise RuntimeError(
                    f"{'V6' if formal_v6 else 'Exp3'} fresh-process restore audit failed: {payload}"
                )
            print(
                f"{'V6' if formal_v6 else 'Exp3'} fresh-process checkpoint and next-rollout audit passed; "
                "exiting without another update."
            )
            return

        if loaded_step >= self.total_training_steps:
            print(
                f"Checkpoint step {loaded_step} already reaches total_training_steps={self.total_training_steps}; "
                "exiting without an unintended extra update."
            )
            return

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                best_improved = False
                curriculum_state = None

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                compressor_config = self.config.actor_rollout_ref.model.get(
                    "vision_token_compressor", {}
                )
                curriculum_config = compressor_config.get("curriculum")
                if curriculum_config is not None:
                    if self.async_rollout_mode:
                        raise RuntimeError("formal visual-token curriculum requires synchronous rollout")
                    from omegaconf import OmegaConf

                    from verl.models.transformers.visual_token_curriculum import (
                        VisualTokenCurriculum,
                    )

                    curriculum_mapping = OmegaConf.to_container(
                        curriculum_config, resolve=True
                    )
                    curriculum = VisualTokenCurriculum.from_mapping(curriculum_mapping)
                    completed_optimizer_steps = int(self.global_steps) - 1
                    curriculum_state = curriculum.runtime_state(completed_optimizer_steps)
                    self.actor_rollout_wg.set_visual_token_curriculum_step(
                        completed_optimizer_steps=completed_optimizer_steps
                    )
                    gen_batch.meta_info.update(
                        {
                            "visual_token_curriculum_schema_version": curriculum_state[
                                "schema_version"
                            ],
                            "visual_token_curriculum_schedule_sha256": curriculum_state[
                                "schedule_sha256"
                            ],
                            "visual_token_curriculum_completed_steps": curriculum_state[
                                "completed_optimizer_steps"
                            ],
                            "visual_token_retention_bps": curriculum_state["retention_bps"],
                        }
                    )
                    metrics.update(
                        {
                            "curriculum/completed_optimizer_steps": float(
                                curriculum_state["completed_optimizer_steps"]
                            ),
                            "curriculum/next_optimizer_step": float(
                                curriculum_state["next_optimizer_step"]
                            ),
                            "curriculum/stage_index": float(curriculum_state["stage_index"]),
                            "curriculum/requested_retention_bps": float(
                                curriculum_state["retention_bps"]
                            ),
                            "curriculum/requested_retention_ratio": float(
                                curriculum_state["retention_bps"] / 10_000.0
                            ),
                        }
                    )
                next_rollout_audit_dir = os.environ.get(
                    "VERL_V6_NEXT_ROLLOUT_AUDIT_DIR"
                    if formal_v6
                    else "VERL_EXP3_NEXT_ROLLOUT_AUDIT_DIR"
                )
                next_rollout_probe = (
                    deepcopy(gen_batch)
                    if next_rollout_audit_dir and self.global_steps >= self.total_training_steps
                    else None
                )
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                gen_batch_output, rollout_inverse_order, rollout_balance_metrics = (
                    self._balance_v6_rollout_uid_groups(gen_batch_output)
                )
                metrics.update(rollout_balance_metrics)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    if formal_v6:
                        self._set_v6_gpu_phase("rollout")
                    try:
                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)
                            if rollout_inverse_order is not None:
                                # Restore the original interleaved trajectory
                                # order before unioning with batch.repeat().
                                gen_batch_output.reorder(rollout_inverse_order)
                    finally:
                        if formal_v6:
                            self._set_v6_gpu_phase("idle")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(
                            batch,
                            metrics=metrics,
                            # Balance ranks inside each global mini-batch
                            # without moving a sample across the two optimizer
                            # update boundaries of the exact batch=16 plan.
                            keep_minibatch=bool(
                                self.config.data.get("enforce_exact_train_batch_plan", False)
                            ),
                        )

                    from verl.models.transformers.vision_token_compressor import (
                        HOLITOM_DPC_MERGE_ROUTES_KEY,
                        HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM,
                        DARTMergeRoute,
                        HoliTomDPCSpatialMergeRoute,
                        validate_cdpruner_curriculum_route,
                    )

                    compressor_config = self.config.actor_rollout_ref.model.get("vision_token_compressor", {})
                    holitom_dpc_merge = (
                        compressor_config.get("algorithm") == HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM
                    )
                    route_key = HOLITOM_DPC_MERGE_ROUTES_KEY if holitom_dpc_merge else "dart_merge_routes"
                    route_type = HoliTomDPCSpatialMergeRoute if holitom_dpc_merge else DARTMergeRoute
                    raw_route_samples = batch.non_tensor_batch.get(route_key)
                    if raw_route_samples is None:
                        raise RuntimeError(
                            f"Visual-compression training requires replayable {route_key} metadata per image"
                        )

                    route_original_tokens: list[int] = []
                    route_output_tokens: list[int] = []
                    for raw_routes in raw_route_samples:
                        if isinstance(raw_routes, np.ndarray):
                            raw_routes = raw_routes.tolist()
                        if not isinstance(raw_routes, (list, tuple)) or not raw_routes:
                            raise RuntimeError(
                                "Every visual-compression sample must contain at least one image route"
                            )
                        for raw_route in raw_routes:
                            route = (
                                raw_route
                                if isinstance(raw_route, route_type)
                                else route_type.from_dict(raw_route, device="cpu")
                            )
                            route.validate()
                            if not holitom_dpc_merge:
                                route.validate_for_algorithm(str(compressor_config.get("algorithm")))
                            if curriculum_config is not None:
                                if holitom_dpc_merge or curriculum_state is None:
                                    raise RuntimeError(
                                        "visual-token curriculum has no active CDPruner runtime state"
                                    )
                                validate_cdpruner_curriculum_route(route, curriculum_state)
                            else:
                                expected = min(
                                    route.original_tokens,
                                    max(32, int(np.ceil(0.05 * route.original_tokens))),
                                )
                                if route.output_tokens != expected:
                                    raise RuntimeError(
                                        "Dynamic visual-token budget mismatch: "
                                        f"{route.output_tokens} != {expected}"
                                    )
                            route_original_tokens.append(route.original_tokens)
                            route_output_tokens.append(route.output_tokens)
                    retention = np.asarray(route_output_tokens, dtype=np.float64) / np.asarray(
                        route_original_tokens, dtype=np.float64
                    )
                    metrics.update(
                        {
                            "compression/original_tokens_mean": float(np.mean(route_original_tokens)),
                            "compression/output_tokens_mean": float(np.mean(route_output_tokens)),
                            "compression/retention_ratio_mean": float(retention.mean()),
                            "compression/retention_ratio_min": float(retention.min()),
                            "compression/retention_ratio_max": float(retention.max()),
                        }
                    )

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
                        if multi_modal_input is None:
                            continue
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    reward_free_teacher_vopd = self._use_reward_free_teacher_vopd()
                    reward_extra_infos_dict = {}
                    if reward_free_teacher_vopd:
                        reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
                    else:
                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # Compute or extract reward for training
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                    batch, reward_fn=self.reward_fn, return_dict=False
                                )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: 蟺_rollout, 蟺_胃)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: 蟺_rollout, 蟺_old, 蟺_胃)
                    #   Note: 蟺_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    reuse_rollout_log_probs = (
                        not bypass_recomputing_logprobs
                        and should_reuse_rollout_log_probs_as_old_log_probs(self.config, batch)
                    )
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    elif reuse_rollout_log_probs:  # Same actor, same HF backend and same context
                        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                        metrics["rollout_corr/old_logprob_source_rollout_alias"] = 1.0
                        metrics["rollout_corr/logprob_gap_independent"] = 0.0
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if (
                                self.config.actor_rollout_ref.rollout.name == "hf"
                                and "rollout_log_probs" not in batch.batch.keys()
                            ):
                                # HF rollout and HF-old are the same backend and
                                # unchanged parameter snapshot.  HFRollout does
                                # not emit sampled-token log-probs during
                                # generate(), so the single HF-old scoring pass
                                # is both quantities by construction.
                                batch.batch["rollout_log_probs"] = batch.batch["old_log_probs"].detach().clone()
                            if "rollout_log_probs" in batch.batch.keys():
                                metrics["rollout_corr/old_logprob_source_rollout_alias"] = 0.0
                                metrics["rollout_corr/logprob_gap_independent"] = 1.0
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        if not reward_free_teacher_vopd and self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                        self_distillation_data = self._maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra_infos_dict)
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Reward-free teacher SDPO never consumes real rewards or KL-in-reward.
                        if reward_free_teacher_vopd:
                            pass
                        # compute rewards. apply_kl_penalty if available
                        elif self.config.algorithm.use_kl_in_reward:
                            batch.batch["token_level_scores"] = reward_tensor
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_scores"] = reward_tensor
                            batch.batch["token_level_rewards"] = reward_tensor

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable 蟺_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving 蟺_胃 vs 蟺_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
                        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
                        skip_advantage_for_vopd = self_distillation_cfg is not None and loss_mode == "vopd"
                        if skip_advantage_for_vopd and "self_distillation_mask" in batch.batch.keys():
                            skip_advantage_for_vopd = bool(torch.all(batch.batch["self_distillation_mask"] > 0.5).item())

                        if not skip_advantage_for_vopd:
                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        smoke_audit_dir = self.config.trainer.get("dart_smoke_audit_dir", None)
                        if smoke_audit_dir:
                            smoke_steps = int(self.config.trainer.total_training_steps)
                            if formal_v6:
                                if smoke_steps not in {1, 10}:
                                    raise RuntimeError(
                                        "Formal V6 trainer.dart_smoke_audit_dir is restricted to 1/10-step probes"
                                    )
                                self._dump_v6_dpc_smoke_audit(batch, smoke_audit_dir)
                            else:
                                if smoke_steps != 1:
                                    raise RuntimeError(
                                        "trainer.dart_smoke_audit_dir is restricted to one-step legacy smoke runs"
                                    )
                                self._dump_dart_smoke_audit(batch, smoke_audit_dir)
                        # update actor
                        if formal_v6:
                            self._set_v6_gpu_phase("actor_update")
                        try:
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self._update_actor(batch)
                        finally:
                            if formal_v6:
                                self._set_v6_gpu_phase("idle")
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        if formal_v6:
                            from training.contract import load_contract

                            warmup_semantics = load_contract()[0]["optimizer"].get(
                                "warmup_update_indexing", "zero_based_legacy_v1"
                            )
                            _require_v6_single_finite_actor_update(
                                actor_output_metrics,
                                require_nonzero_lr=warmup_semantics == "one_based_nonzero_v2",
                            )
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        best_improved = self._consider_best_checkpoint(val_metrics)
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if is_last_step or best_improved or (self.config.trainer.save_freq > 0 and (
                    self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                )):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                    if is_last_step and next_rollout_probe is not None:
                        if formal_v6:
                            self._write_v6_post_checkpoint_rollout_golden(
                                next_rollout_probe,
                                next_rollout_audit_dir,
                            )
                        else:
                            self._write_exp3_post_checkpoint_rollout_golden(
                                next_rollout_probe,
                                next_rollout_audit_dir,
                            )

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(
                    compute_data_metrics(
                        batch=batch,
                        use_critic=self.use_critic,
                        configured_max_prompt_length=int(self.config.data.max_prompt_length),
                        configured_max_response_length=int(self.config.data.max_response_length),
                    )
                )
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if not formal_v6:
                    _audit_exp3_migrated_step301(
                        metrics,
                        self.config.trainer.get("rollout_data_dir", None),
                    )

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
