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
Single Process Actor
"""

import logging
import math
import os
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#from torch.distributed.tensor import DTensor
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_self_distillation_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (
    build_vopd_composite_costs,
    prepare_dynamic_batch,
    prepare_dynamic_batch_by_composite_cost,
    restore_dynamic_batch,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Qwen3.5-4B has a 248,320-token vocabulary.  A monolithic logsumexp can
# materialize a temporary as large as the logits tensor; bounding the reduced
# slice to 4,096 vocabulary entries cuts that temporary by roughly 60x while
# preserving the exact full-vocabulary normalization objective.
_DISTILLATION_VOCAB_LOGSUMEXP_CHUNK_SIZE = 4096


class _ChunkedVocabLogsumexpFunction(torch.autograd.Function):
    """Memory-bounded autograd for a last-dimension logsumexp.

    ``Function.forward`` runs without graph recording.  Consequently the
    temporary FP32 chunk conversions are released instead of all being retained
    for backward.  Backward recomputes one softmax chunk at a time and writes it
    into the preallocated input-dtype gradient.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
        reduction_dtype = (
            torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        )
        vocab_size = int(logits.shape[-1])
        running_logsumexp = None
        for start in range(0, vocab_size, chunk_size):
            chunk = logits[..., start : start + chunk_size].to(dtype=reduction_dtype)
            partial = torch.logsumexp(chunk, dim=-1, keepdim=True)
            running_logsumexp = (
                partial if running_logsumexp is None else torch.logaddexp(running_logsumexp, partial)
            )

        ctx.chunk_size = int(chunk_size)
        ctx.save_for_backward(logits, running_logsumexp)
        # The surrounding actor forward runs under autocast, where PyTorch's
        # native logsumexp is an FP32-policy op.  Returning the reduction dtype
        # preserves that behavior for BF16/FP16 logits.
        return running_logsumexp

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits, logsumexp = ctx.saved_tensors
        vocab_size = int(logits.shape[-1])
        grad_logits = torch.empty_like(logits)
        grad_output_work = grad_output.to(dtype=logsumexp.dtype)

        for start in range(0, vocab_size, ctx.chunk_size):
            end = min(vocab_size, start + ctx.chunk_size)
            chunk = logits[..., start:end].to(dtype=logsumexp.dtype)
            chunk_grad = torch.exp(chunk - logsumexp) * grad_output_work
            grad_logits[..., start:end].copy_(chunk_grad.to(dtype=logits.dtype))
        return grad_logits, None


def _chunked_vocab_logsumexp(
    logits: torch.Tensor,
    *,
    chunk_size: int = _DISTILLATION_VOCAB_LOGSUMEXP_CHUNK_SIZE,
) -> torch.Tensor:
    """Differentiable full-vocabulary logsumexp with bounded temporary memory.

    Each chunk is reduced over the vocabulary dimension, then the partial
    log-partition functions are combined with ``logaddexp``.  This is
    mathematically the same reduction as ``torch.logsumexp(logits, -1)`` and
    does not detach or mutate ``logits``.  Only reduction order differs.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim == 0:
        raise ValueError("logits must have a vocabulary dimension")
    if not logits.is_floating_point():
        raise TypeError(f"logits must have a floating-point dtype, got {logits.dtype}")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    vocab_size = int(logits.shape[-1])
    if vocab_size == 0:
        # The empty reduction is already allocation-free and returns the
        # canonical -Inf value with the correct leading shape.
        reduction_dtype = (
            torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        )
        return torch.logsumexp(logits.to(dtype=reduction_dtype), dim=-1, keepdim=True)

    return _ChunkedVocabLogsumexpFunction.apply(logits, chunk_size)


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
        actor_lr_scheduler: Scheduler advanced after every successful optimizer
            update.  Keeping it beside the optimizer is important when one
            rollout batch contains more than one PPO mini-batch.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        actor_lr_scheduler=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.actor_lr_scheduler = actor_lr_scheduler
        if actor_optimizer is None and actor_lr_scheduler is not None:
            raise ValueError("A learning-rate scheduler requires an actor optimizer")
        self._last_optimizer_step_succeeded = False
        self._last_optimizer_lr = None
        self._last_optimizer_lrs_by_group = None
        self.teacher_module: Optional[nn.Module] = None
        self.teacher_update_count = 0
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "vopd":
            return
        teacher_model_source = getattr(self_distillation_cfg, "teacher_model_source", "legacy")
        if self.config.get("training_mode", "legacy") == "full_parameter":
            teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", None)
            update_rate = float(getattr(self_distillation_cfg, "teacher_update_rate", -1.0))
            if teacher_model_source != "fixed" or teacher_regularization != "fixed" or update_rate != 0.0:
                raise RuntimeError("Formal V6 forbids EMA/current/progressive teacher updates")
            if self.teacher_module is None or self.teacher_module is self.actor_module:
                raise RuntimeError("Formal V6 requires a separate fixed dense teacher module")
            if self.teacher_module.training:
                raise RuntimeError("Formal fixed teacher left eval mode")
            if any(parameter.requires_grad for parameter in self.teacher_module.parameters()):
                raise RuntimeError("Formal fixed teacher contains a trainable parameter")
            if self.teacher_update_count != 0:
                raise RuntimeError("Formal fixed teacher update_count must remain zero")
            return
        if teacher_model_source != "legacy":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("Teacher updates require a separate teacher_module in the actor worker.")
        with torch.no_grad():
            if teacher_regularization == "ema":
                update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
                if update_rate == 0.0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    student_data = student_param.data.to(device=teacher_param.device)
                    teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
                return

            if teacher_regularization == "progressive":
                teacher_update_interval = getattr(self_distillation_cfg, "teacher_update_interval", None)
                if teacher_update_interval is None:
                    raise ValueError("Progressive teacher requires self_distillation.teacher_update_interval.")
                global_steps = getattr(self, "_current_global_steps", None)
                if global_steps is None or global_steps % teacher_update_interval != 0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    teacher_param.data.copy_(student_param.data.to(device=teacher_param.device))
                for teacher_buffer, student_buffer in zip(
                    self.teacher_module.buffers(),
                    self.actor_module.buffers(),
                ):
                    teacher_buffer.data.copy_(student_buffer.data.to(device=teacher_buffer.device))
                return

            return

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    @staticmethod
    def _global_distillation_batch_info(mini_batch: DataProto) -> dict[str, int]:
        """Return exact global denominators for one optimizer mini-batch.

        FSDP averages gradients across ranks.  A rank-local token mean therefore
        gives every rank equal weight even when EOS makes their valid-token
        counts different.  We instead divide every micro-batch numerator by the
        same global denominator and multiply by the data-parallel world size;
        FSDP's gradient average then recovers the true global mean.
        """

        loss_mask = mini_batch.batch["response_mask"]
        distillation_mask = mini_batch.batch.get("self_distillation_mask")
        if distillation_mask is not None:
            loss_mask = loss_mask * distillation_mask.unsqueeze(1)
        local_counts = torch.tensor(
            [
                int(loss_mask.sum().item()),
                int((loss_mask.sum(dim=-1) > 0).sum().item()),
            ],
            dtype=torch.long,
            device=get_device_id(),
        )
        dp_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_counts, op=torch.distributed.ReduceOp.SUM)
            dp_size = torch.distributed.get_world_size()
        return {
            # A fully masked VOPD mini-batch has a zero numerator.  Use a unit
            # denominator so the loss remains exactly zero instead of NaN.
            "batch_num_tokens": max(int(local_counts[0].item()), 1),
            "global_batch_size": max(int(local_counts[1].item()), 1),
            "dp_size": int(dp_size),
        }

    @staticmethod
    def _slice_batch_aligned_value(value, start: int, end: int, batch_size: int):
        if isinstance(value, torch.Tensor):
            return value[start:end] if value.ndim > 0 and int(value.shape[0]) == batch_size else value
        if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0 and int(value.shape[0]) == batch_size:
            return value[start:end]
        if isinstance(value, list) and len(value) == batch_size:
            return value[start:end]
        if isinstance(value, tuple) and len(value) == batch_size:
            return value[start:end]
        return value

    def _forward_fixed_teacher_in_chunks(
        self,
        teacher_inputs: dict,
        *,
        module: nn.Module,
        micro_batch_size: int,
        **forward_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run the immutable dense teacher with a rank-synchronous small batch."""

        if self.config.get("training_mode", "legacy") != "full_parameter":
            raise RuntimeError("Fixed-teacher chunking is restricted to the formal full-parameter path")
        if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
            raise ValueError("teacher micro_batch_size must be a positive integer")
        if module is None or module is self.actor_module:
            raise RuntimeError("Formal fixed teacher must be separate from the current actor")
        if module.training or any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("Formal fixed teacher must remain frozen in eval mode")
        responses = teacher_inputs.get("responses")
        if not isinstance(responses, torch.Tensor) or responses.ndim == 0:
            raise ValueError("Teacher chunking requires batch-aligned responses")
        batch_size = int(responses.shape[0])
        if micro_batch_size == 1:
            # Preserve the original one-sample path exactly.  Recovery tiers
            # may deliberately return to it, and it does not need a cost proxy
            # or multimodal geometry merely to construct singleton chunks.
            plan = [(index, index + 1) for index in range(batch_size)]
        else:
            plan = None
        # Build one identical chunk plan on every FSDP rank.  A locally chosen
        # number of teacher forwards would desynchronize FSDP collective order.
        # The proxy counts dense attended tokens, raw ViT patches, and all
        # response positions projected by the LM head (including padded ones).
        if plan is None:
            attention_mask = teacher_inputs.get("attention_mask")
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape[0] != batch_size:
                raise ValueError("Teacher dynamic chunking requires a batch-aligned attention_mask")
            sample_costs = attention_mask.to(torch.long).sum(dim=-1)
            sample_costs = sample_costs + int(responses.shape[-1])
            raw_inputs = teacher_inputs.get("multi_modal_inputs")
            if raw_inputs is not None:
                if len(raw_inputs) != batch_size:
                    raise ValueError("Teacher multimodal inputs are not batch aligned")
                patch_counts = []
                for sample in raw_inputs:
                    sample = getattr(sample, "data", sample)
                    if not isinstance(sample, dict) or sample.get("image_grid_thw") is None:
                        raise ValueError("Teacher dynamic chunking requires image_grid_thw for every sample")
                    grid = torch.as_tensor(sample["image_grid_thw"], dtype=torch.long)
                    if grid.ndim == 1:
                        grid = grid.unsqueeze(0)
                    if grid.ndim != 2 or grid.shape[-1] != 3 or bool((grid <= 0).any().item()):
                        raise ValueError("Teacher dynamic chunking received invalid image_grid_thw")
                    patch_counts.append(int(grid.prod(dim=-1).sum().item()))
                sample_costs = sample_costs + torch.tensor(
                    patch_counts, dtype=sample_costs.dtype, device=sample_costs.device
                )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(sample_costs, op=torch.distributed.ReduceOp.MAX)

            cost_ceiling = int(
                self.config.vision_packing.get("teacher_max_cost_per_microbatch", 0)
            )
            if cost_ceiling <= 0:
                raise ValueError("Dynamic teacher batching requires teacher_max_cost_per_microbatch")
            plan = []
            start = 0
            while start < batch_size:
                end = min(start + micro_batch_size, batch_size)
                while end > start + 1 and int(sample_costs[start:end].sum().item()) > cost_ceiling:
                    end -= 1
                if int(sample_costs[start:end].sum().item()) > cost_ceiling:
                    raise RuntimeError(
                        "One dense-teacher sample exceeds teacher_max_cost_per_microbatch: "
                        f"cost={int(sample_costs[start].item())}, ceiling={cost_ceiling}"
                    )
                plan.append((start, end))
                start = end

        chunks: dict[str, list[torch.Tensor]] = {}
        for start, end in plan:
            chunk_inputs = {
                key: self._slice_batch_aligned_value(value, start, end, batch_size)
                for key, value in teacher_inputs.items()
            }
            chunk_kwargs = {
                key: self._slice_batch_aligned_value(value, start, end, batch_size)
                for key, value in forward_kwargs.items()
            }
            output = self._forward_micro_batch(chunk_inputs, module=module, **chunk_kwargs)
            for key, value in output.items():
                if not isinstance(value, torch.Tensor) or value.ndim == 0 or int(value.shape[0]) != end - start:
                    raise RuntimeError(f"Teacher output {key!r} is not batch-aligned and cannot be concatenated")
                chunks.setdefault(key, []).append(value)
        self._teacher_forward_chunk_calls = int(
            getattr(self, "_teacher_forward_chunk_calls", 0)
        ) + len(plan)
        self._teacher_forward_chunk_samples = int(
            getattr(self, "_teacher_forward_chunk_samples", 0)
        ) + batch_size
        return {key: torch.cat(values, dim=0) for key, values in chunks.items()}

    @staticmethod
    def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _build_union_support_indices(
        student_indices: torch.Tensor,
        teacher_indices: torch.Tensor,
        rollout_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stable union: student top-k, teacher top-k, rollout token.

        Duplicate slots remain physically present for a fixed tensor shape but
        are marked invalid and assigned log-probability ``-1e9`` by the caller,
        so they contribute exactly zero probability mass and cannot corrupt the
        tail bucket.
        """

        if student_indices.ndim != 3 or teacher_indices.ndim != 3:
            raise ValueError("Student and teacher support indices must be [batch, response, k]")
        if student_indices.shape[:2] != teacher_indices.shape[:2]:
            raise ValueError("Student and teacher support indices are response-misaligned")
        if rollout_token_ids.shape != student_indices.shape[:2]:
            raise ValueError("Rollout token IDs are response-misaligned")
        combined = torch.cat([student_indices, teacher_indices, rollout_token_ids.unsqueeze(-1)], dim=-1)
        equality = combined.unsqueeze(-1) == combined.unsqueeze(-2)
        earlier = torch.tril(
            torch.ones(combined.shape[-1], combined.shape[-1], dtype=torch.bool, device=combined.device),
            diagonal=-1,
        )
        valid_mask = ~(equality & earlier).any(dim=-1)
        if not bool(valid_mask[..., 0].all().item()):
            raise RuntimeError("Union support unexpectedly invalidated its first element")
        return combined, valid_mask

    @staticmethod
    def _build_response_positions(
        response_start_idx: torch.Tensor,
        response_length: int,
        seqlen: int,
    ) -> torch.Tensor:
        if response_start_idx.dim() != 1:
            raise ValueError(f"response_start_idx must be rank-1, got shape {tuple(response_start_idx.shape)}")
        if (response_start_idx < 1).any():
            raise ValueError("response_start_idx must be >= 1 so response logits have a preceding context token.")

        offsets = torch.arange(response_length, device=response_start_idx.device, dtype=response_start_idx.dtype)
        response_positions = response_start_idx.unsqueeze(1) - 1 + offsets.unsqueeze(0)
        if response_positions.numel() > 0:
            if response_positions.min().item() < 0 or response_positions.max().item() >= seqlen:
                raise ValueError(
                    f"Response positions out of bounds for seqlen={seqlen}: "
                    f"min={response_positions.min().item()}, max={response_positions.max().item()}"
                )
        return response_positions.to(dtype=torch.long)

    @staticmethod
    def _select_response_positions(
        hidden_states: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if response_start_idx is None:
            return hidden_states[:, -response_length - 1 : -1, ...]

        response_positions = DataParallelPPOActor._build_response_positions(
            response_start_idx=response_start_idx,
            response_length=response_length,
            seqlen=hidden_states.size(1),
        )
        if hidden_states.dim() == 2:
            return torch.gather(hidden_states, dim=1, index=response_positions)

        gather_index = response_positions.view(
            response_positions.size(0),
            response_positions.size(1),
            *([1] * (hidden_states.dim() - 2)),
        ).expand(response_positions.size(0), response_positions.size(1), *hidden_states.shape[2:])
        return torch.gather(hidden_states, dim=1, index=gather_index)

    @staticmethod
    def _get_unwrapped_module(module: nn.Module) -> nn.Module:
        return getattr(module, "module", getattr(module, "_fsdp_wrapped_module", module))

    @staticmethod
    def _visual_token_compression_enabled(module: nn.Module) -> bool:
        unwrapped = DataParallelPPOActor._get_unwrapped_module(module)
        nested_model = getattr(unwrapped, "model", None)
        return bool(
            getattr(unwrapped, "vision_token_projector_enabled", False)
            or (nested_model is not None and getattr(nested_model, "vision_token_projector_enabled", False))
            or getattr(unwrapped, "vision_token_compressor_enabled", False)
            or (nested_model is not None and getattr(nested_model, "vision_token_compressor_enabled", False))
        )

    @staticmethod
    def _visual_token_compressor_algorithm(module: nn.Module) -> Optional[str]:
        """Return the installed compressor algorithm without mutating model state."""

        unwrapped = DataParallelPPOActor._get_unwrapped_module(module)
        nested_model = getattr(unwrapped, "model", None)
        algorithm = getattr(unwrapped, "vision_token_compressor_algorithm", None)
        if algorithm is None and nested_model is not None:
            algorithm = getattr(nested_model, "vision_token_compressor_algorithm", None)
        return algorithm

    @staticmethod
    def _visual_token_route_transport_key(module: nn.Module) -> str:
        """Resolve the fail-closed route key from the live installed model."""

        from verl.models.transformers.vision_token_compressor import (
            CDPRUNER_ALGORITHM,
            HOLITOM_DPC_MERGE_ROUTES_KEY,
            HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM,
            LEGACY_CONDITIONAL_DIVERSITY_ALGORITHM,
        )

        algorithm = DataParallelPPOActor._visual_token_compressor_algorithm(module)
        if algorithm == HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM:
            return HOLITOM_DPC_MERGE_ROUTES_KEY
        if algorithm in {CDPRUNER_ALGORITHM, LEGACY_CONDITIONAL_DIVERSITY_ALGORITHM}:
            return "dart_merge_routes"
        raise RuntimeError(
            "Formal composite packing requires a supported live visual compressor, "
            f"got compressor algorithm={algorithm!r}"
        )

    @staticmethod
    def _supports_functional_visual_compression_mode(module: nn.Module) -> bool:
        unwrapped = DataParallelPPOActor._get_unwrapped_module(module)
        config = getattr(unwrapped, "config", None)
        if config is None:
            nested_model = getattr(unwrapped, "model", None)
            config = getattr(nested_model, "config", None)
        return getattr(config, "model_type", None) in {"qwen3_5", "qwen3_5_moe"}

    @staticmethod
    def _visual_compression_mode_for_inputs(module: nn.Module, multi_modal_inputs) -> str:
        has_images = DataParallelPPOActor._has_non_empty_multi_modal_inputs(multi_modal_inputs)
        if not has_images:
            return "no_image"
        if DataParallelPPOActor._visual_token_compression_enabled(module):
            return "merge"
        return "dense"

    @staticmethod
    def _adjust_response_start_for_visual_compression(
        *,
        module: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        unwrapped = DataParallelPPOActor._get_unwrapped_module(module)
        nested_model = getattr(unwrapped, "model", None)
        config = getattr(unwrapped, "config", getattr(nested_model, "config", None))
        image_token_id = getattr(config, "image_token_id", None)
        minimum_tokens = getattr(unwrapped, "vision_token_compressor_minimum_tokens", None)
        retention_ratio = getattr(unwrapped, "vision_token_compressor_retention_ratio", None)
        retention_bps = getattr(unwrapped, "vision_token_compressor_retention_bps", None)
        if nested_model is not None:
            minimum_tokens = minimum_tokens or getattr(nested_model, "vision_token_compressor_minimum_tokens", None)
            retention_ratio = retention_ratio or getattr(nested_model, "vision_token_compressor_retention_ratio", None)
            retention_bps = retention_bps or getattr(
                nested_model, "vision_token_compressor_retention_bps", None
            )
        if image_token_id is None or minimum_tokens is None or retention_ratio is None:
            raise ValueError("Visual compression requires image token id and dynamic-budget metadata.")

        if response_start_idx is None:
            response_start_idx = torch.full(
                (input_ids.size(0),),
                input_ids.size(1) - response_length,
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            response_start_idx = response_start_idx.to(device=input_ids.device, dtype=torch.long)

        token_positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        prompt_mask = token_positions < response_start_idx.unsqueeze(1)
        image_mask = (input_ids == image_token_id) & prompt_mask & attention_mask.to(torch.bool)
        old_image_tokens = image_mask.sum(dim=1)

        from verl.models.transformers.vision_token_compressor import count_compressed_image_tokens

        # The previous num_images * B shortcut was wrong whenever an image had
        # N < B tokens.  Count every real contiguous image span independently.
        new_image_tokens = count_compressed_image_tokens(
            image_mask,
            int(minimum_tokens),
            float(retention_ratio),
            retention_bps=int(retention_bps) if retention_bps is not None else None,
        )
        return response_start_idx - (old_image_tokens - new_image_tokens)

    @staticmethod
    def _crop_common_dart_prompt_left_padding(
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Remove only prompt-leading pad columns shared by the local batch.

        Qwen3.5's hybrid linear-attention implementation assumes a batch of one
        has no padding and skips its padding-state mask in that case.  Global
        rollout concatenation can nevertheless add prompt-left padding before
        the actor receives a one-sample DP mini-batch.  Cropping those inert
        columns makes actor route replay use the same physical prefix as the
        compact cached rollout.  Response-tail padding is intentionally kept so
        response tensors, masks, and loss positions remain fixed-width.
        """

        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "DART left-padding crop requires matching rank-2 input_ids/attention_mask, "
                f"got ids={tuple(input_ids.shape)}, mask={tuple(attention_mask.shape)}"
            )
        batch_size, seqlen = input_ids.shape
        if response_length <= 0 or response_length >= seqlen:
            raise ValueError(
                f"Invalid DART response_length={response_length} for sequence length {seqlen}"
            )
        if response_start_idx is None:
            prompt_ends = torch.full(
                (batch_size,), seqlen - response_length, dtype=torch.long, device=input_ids.device
            )
        else:
            prompt_ends = response_start_idx.to(device=input_ids.device, dtype=torch.long)
            if prompt_ends.shape != (batch_size,):
                raise ValueError(
                    f"response_start_idx shape must be ({batch_size},), got {tuple(prompt_ends.shape)}"
                )
        if bool(((prompt_ends <= 0) | (prompt_ends > seqlen - response_length)).any().item()):
            raise ValueError(
                "DART response_start_idx must identify a non-empty prompt before the fixed response suffix"
            )

        leading_pad_counts = []
        mask_bool = attention_mask.to(torch.bool)
        for row_idx, prompt_end_tensor in enumerate(prompt_ends):
            prompt_end = int(prompt_end_tensor.item())
            row_prompt_mask = mask_bool[row_idx, :prompt_end]
            valid = torch.nonzero(row_prompt_mask, as_tuple=False).flatten()
            if valid.numel() == 0:
                raise ValueError(f"DART sample {row_idx} has an empty prompt after masking")
            first_valid = int(valid[0].item())
            if not bool(torch.all(row_prompt_mask[first_valid:]).item()):
                raise ValueError(
                    "DART actor replay supports only left padding in the prompt; "
                    f"sample {row_idx} has an internal mask gap"
                )
            leading_pad_counts.append(first_valid)

        crop = min(leading_pad_counts)
        if crop == 0:
            return input_ids, attention_mask, position_ids, response_start_idx
        input_ids = input_ids[:, crop:]
        attention_mask = attention_mask[:, crop:]
        position_ids = position_ids[..., crop:]
        if response_start_idx is not None:
            response_start_idx = prompt_ends - crop
        return input_ids, attention_mask, position_ids, response_start_idx

    @staticmethod
    def _prepare_replay_dart_routes(
        routes_by_sample,
        multi_modal_inputs: dict,
        *,
        expected_algorithm: str,
        sample_uids=None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_length: int,
    ) -> list:
        """Validate sample-aligned rollout routes before any actor re-score."""

        if routes_by_sample is None:
            raise RuntimeError("DART actor scoring requires rollout-provided dart_merge_routes")
        if hasattr(routes_by_sample, "tolist"):
            routes_by_sample = routes_by_sample.tolist()
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("DART replay identity validation requires aligned rank-2 prompt tensors")
        if len(routes_by_sample) != input_ids.shape[0]:
            raise RuntimeError("DART replay route metadata is not sample aligned")
        normalized_uids = None
        if expected_algorithm == "qwen35_cdpruner_v1":
            if sample_uids is None:
                raise RuntimeError("V8 CDPruner actor replay requires sample-aligned rollout UIDs")
            if hasattr(sample_uids, "tolist"):
                sample_uids = sample_uids.tolist()
            normalized_uids = [str(value) for value in sample_uids]
            if len(normalized_uids) != input_ids.shape[0] or any(not value for value in normalized_uids):
                raise RuntimeError("V8 CDPruner actor replay UIDs are missing or not sample aligned")
        prompt_width = int(input_ids.shape[-1]) - int(response_length)
        if prompt_width <= 0:
            raise ValueError("DART replay has no public prompt prefix")
        flattened = []
        for sample_idx, sample_routes in enumerate(routes_by_sample):
            if sample_routes is None:
                raise RuntimeError(f"DART route metadata is missing for sample {sample_idx}")
            if isinstance(sample_routes, dict):
                sample_routes = [sample_routes]
            sample_routes = list(sample_routes)
            if not sample_routes:
                raise RuntimeError(f"DART route metadata is empty for sample {sample_idx}")
            for raw_route in sample_routes:
                from verl.models.transformers.vision_token_compressor import (
                    CDPRUNER_ALGORITHM,
                    DARTMergeRoute,
                )

                route = (
                    raw_route
                    if isinstance(raw_route, DARTMergeRoute)
                    else DARTMergeRoute.from_dict(raw_route)
                )
                route.validate_for_algorithm(expected_algorithm)
                if route.anchor_coordinates is None:
                    raise RuntimeError(
                        f"DART replay route for sample {sample_idx} has no anchor M-RoPE coordinates"
                    )
                if expected_algorithm == CDPRUNER_ALGORITHM:
                    if not isinstance(raw_route, Mapping):
                        raise RuntimeError("CDPruner actor replay requires a serialized identity-bound route")
                    audit = raw_route.get("query_audit")
                    if not isinstance(audit, Mapping):
                        raise RuntimeError("CDPruner actor replay route is missing query_audit")
                    from verl.utils.route_query import (
                        ROUTE_QUERY_SCHEMA_VERSION_V2,
                        parse_route_query_value,
                    )

                    if audit.get("source") != "route_query":
                        raise RuntimeError("V8 CDPruner forbids inferred/fallback route-query provenance")
                    if audit.get("sample_uid") != normalized_uids[sample_idx]:
                        raise RuntimeError(
                            "CDPruner replay route UID does not match the current sample"
                        )
                    if audit.get("schema_version") != ROUTE_QUERY_SCHEMA_VERSION_V2:
                        raise RuntimeError("V8 CDPruner replay requires the explicit V2 route-query schema")
                    spec = parse_route_query_value(
                        {
                            "schema_version": audit.get("schema_version"),
                            "policy": audit.get("query_policy"),
                            "segments": audit.get("segments"),
                            "canonical_text": audit.get("canonical_text"),
                            "sha256": audit.get("canonical_sha256"),
                        }
                    )
                    # parse_route_query_value already enforces the one allowed
                    # policy and binds the canonical text/hash.  Keep the
                    # parsed object live here so that malformed semantic
                    # segments fail before route replay.
                    if not spec.canonical_text:
                        raise RuntimeError("CDPruner replay query is empty")
                    audited_width = audit.get("prefill_prompt_length")
                    audited_left_pad = audit.get("prefill_left_padding")
                    if (
                        isinstance(audited_width, bool)
                        or not isinstance(audited_width, int)
                        or audited_width != prompt_width
                        or isinstance(audited_left_pad, bool)
                        or not isinstance(audited_left_pad, int)
                        or audited_left_pad < 0
                    ):
                        raise RuntimeError("CDPruner replay query audit has stale prompt geometry")
                    prompt_mask = attention_mask[sample_idx, :prompt_width].to(torch.bool)
                    active_positions = torch.nonzero(prompt_mask, as_tuple=False).flatten()
                    if active_positions.numel() == 0 or int(active_positions[0].item()) != audited_left_pad:
                        raise RuntimeError("CDPruner replay query audit has the wrong left padding")
                    positions = audit.get("selected_token_indices")
                    token_ids = audit.get("selected_token_ids")
                    if (
                        not isinstance(positions, (list, tuple))
                        or not positions
                        or not isinstance(token_ids, (list, tuple))
                        or len(positions) != len(token_ids)
                        or audit.get("selected_token_count") != len(positions)
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in positions)
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
                    ):
                        raise RuntimeError("CDPruner replay query audit has invalid token provenance")
                    normalized_positions = [int(value) for value in positions]
                    if (
                        normalized_positions != sorted(set(normalized_positions))
                        or normalized_positions[0] < 0
                        or normalized_positions[-1] >= prompt_width
                        or not bool(prompt_mask[normalized_positions].all().item())
                    ):
                        raise RuntimeError("CDPruner replay query positions are outside the active prompt")
                    actual_ids = [
                        int(input_ids[sample_idx, position].item()) for position in normalized_positions
                    ]
                    if actual_ids != [int(value) for value in token_ids]:
                        raise RuntimeError(
                            "CDPruner replay route/query identity does not match the current sample prompt"
                        )
                flattened.append(raw_route)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            raise RuntimeError("DART route replay requires image_grid_thw")
        expected_images = int(image_grid_thw.shape[0])
        if len(flattened) != expected_images:
            raise RuntimeError(
                f"DART route/image count mismatch: routes={len(flattened)}, images={expected_images}"
            )

        return flattened

    @staticmethod
    def _prepare_replay_holitom_dpc_routes(routes_by_sample, multi_modal_inputs: dict) -> list:
        """Validate exact rollout-owned HoliTom-DPC routes before actor replay."""

        from verl.models.transformers.vision_token_compressor import (
            HOLITOM_DPC_MERGE_ROUTES_KEY,
            HoliTomDPCSpatialMergeRoute,
        )

        if routes_by_sample is None:
            raise RuntimeError(
                "HoliTom-DPC actor scoring requires rollout-provided " f"{HOLITOM_DPC_MERGE_ROUTES_KEY}"
            )
        if hasattr(routes_by_sample, "tolist"):
            routes_by_sample = routes_by_sample.tolist()
        flattened = []
        for sample_idx, sample_routes in enumerate(routes_by_sample):
            if sample_routes is None:
                raise RuntimeError(f"HoliTom-DPC route metadata is missing for sample {sample_idx}")
            if isinstance(sample_routes, (dict, HoliTomDPCSpatialMergeRoute)):
                sample_routes = [sample_routes]
            sample_routes = list(sample_routes)
            if not sample_routes:
                raise RuntimeError(f"HoliTom-DPC route metadata is empty for sample {sample_idx}")
            flattened.extend(sample_routes)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            raise RuntimeError("HoliTom-DPC route replay requires image_grid_thw")
        expected_images = int(image_grid_thw.shape[0])
        if len(flattened) != expected_images:
            raise RuntimeError(
                "HoliTom-DPC route/image count mismatch: "
                f"routes={len(flattened)}, images={expected_images}"
            )

        for route_index, payload in enumerate(flattened):
            route = (
                payload
                if isinstance(payload, HoliTomDPCSpatialMergeRoute)
                else HoliTomDPCSpatialMergeRoute.from_dict(payload)
            )
            route.validate()
            if route.anchor_coordinates is None:
                raise RuntimeError(f"HoliTom-DPC replay route {route_index} has no anchor_coordinates")
        return flattened

    def _dump_self_distillation_log_probs(
        self,
        *,
        meta_info: dict,
        self_distillation_cfg,
        dump_chunks: list[dict[str, torch.Tensor]],
    ) -> None:
        dump_root = self_distillation_cfg.get("log_prob_dump_dir", None)
        if not dump_root or not dump_chunks:
            return

        global_step = meta_info.get("global_steps")
        if global_step is None:
            return

        experiment_name = os.environ.get("EXPERIMENT", "unknown_experiment")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        normalized_root = os.path.normpath(dump_root)
        if os.path.basename(normalized_root) == experiment_name:
            save_dir = normalized_root
        else:
            save_dir = os.path.join(normalized_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{int(global_step)}.rank{rank}.pt")

        student_log_probs = torch.cat([chunk["student_log_probs"] for chunk in dump_chunks], dim=0)
        teacher_log_probs = torch.cat([chunk["teacher_log_probs"] for chunk in dump_chunks], dim=0)
        student_response_start_idx = torch.cat(
            [chunk["student_response_start_idx"] for chunk in dump_chunks], dim=0
        )
        teacher_response_start_idx = torch.cat(
            [chunk["teacher_response_start_idx"] for chunk in dump_chunks], dim=0
        )
        topk_valid_mask = None
        if all("topk_valid_mask" in chunk for chunk in dump_chunks):
            topk_valid_mask = torch.cat([chunk["topk_valid_mask"] for chunk in dump_chunks], dim=0)
        support_indices = teacher_seed_indices = rollout_token_ids = None
        if all("support_indices" in chunk for chunk in dump_chunks):
            support_indices = torch.cat([chunk["support_indices"] for chunk in dump_chunks], dim=0)
            teacher_seed_indices = torch.cat([chunk["teacher_seed_indices"] for chunk in dump_chunks], dim=0)
            rollout_token_ids = torch.cat([chunk["rollout_token_ids"] for chunk in dump_chunks], dim=0)

        payload = {
                "student_log_probs": student_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "student_response_start_idx": student_response_start_idx,
                "teacher_response_start_idx": teacher_response_start_idx,
                "global_step": int(global_step),
                "rank": rank,
                "experiment_name": experiment_name,
                "distribution_size": int(student_log_probs.shape[-1]),
                "num_valid_tokens": int(student_log_probs.shape[0]),
                "distillation_topk": self_distillation_cfg.get("distillation_topk", None),
                "distillation_add_tail": bool(self_distillation_cfg.get("distillation_add_tail", False)),
                "distillation_support_policy": self_distillation_cfg.get(
                    "distillation_support_policy", "student_topk"
                ),
            }
        if topk_valid_mask is not None:
            payload["topk_valid_mask"] = topk_valid_mask
            payload["support_indices"] = support_indices
            payload["teacher_seed_indices"] = teacher_seed_indices
            payload["rollout_token_ids"] = rollout_token_ids
        torch.save(payload, save_path)

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        topk_valid_mask: Optional[torch.Tensor] = None,
        additional_topk_indices: Optional[torch.Tensor] = None,
        rollout_token_ids: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
        visual_compression_mode: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        union_support = additional_topk_indices is not None or rollout_token_ids is not None
        if union_support and (additional_topk_indices is None or rollout_token_ids is None):
            raise ValueError("Union distillation support requires teacher indices and rollout token ids")
        if union_support and topk_indices is not None:
            raise ValueError("Cannot build a new union while replaying explicit support indices")
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")
        if union_support and self.use_remove_padding:
            raise ValueError("Union distillation support is audited only with use_remove_padding=False")

        model = module or self.actor_module
        compressor_enabled = self._visual_token_compression_enabled(model)
        compressor_algorithm = self._visual_token_compressor_algorithm(model)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])
            # This is trainer-side FLOP metadata, not a model forward input.
            # Leaving it in kwargs reaches Qwen's language_model and is either
            # rejected or silently changes backend behavior across versions.
            multi_modal_inputs.pop("images_seqlens", None)
        has_multi_modal_inputs = bool(multi_modal_inputs)

        if visual_compression_mode is None:
            if has_multi_modal_inputs:
                visual_compression_mode = "merge" if compressor_enabled else "dense"
            else:
                visual_compression_mode = "no_image"
        if visual_compression_mode not in {"merge", "dense", "no_image"}:
            raise ValueError(
                "visual_compression_mode must be one of {'merge', 'dense', 'no_image'}, "
                f"got {visual_compression_mode!r}"
            )
        if visual_compression_mode == "merge":
            if not compressor_enabled:
                raise RuntimeError("visual_compression_mode='merge' requires an installed visual compressor")
            if not has_multi_modal_inputs:
                raise RuntimeError("visual_compression_mode='merge' requires non-empty multimodal inputs")
        elif visual_compression_mode == "dense":
            if not has_multi_modal_inputs:
                raise RuntimeError("visual_compression_mode='dense' requires non-empty multimodal inputs")

        merge_visual_tokens = visual_compression_mode == "merge"
        if visual_compression_mode == "no_image":
            # Keep the exact public token/placeholder stream for the formal
            # no-image ablation, but never forward retained pixel metadata.
            # Policy: same_prompt_image_placeholder_without_visual_replacement_v1.
            multi_modal_inputs = {}
        if merge_visual_tokens:
            if self.use_remove_padding:
                raise ValueError("Visual token compression is not compatible with use_remove_padding=True.")
            if self.use_fused_kernels:
                raise ValueError("Visual token compression is not compatible with fused-kernel logprob backends.")

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
                and not merge_visual_tokens
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        dart_merge_routes = None
        dpc_merge_routes = None
        if merge_visual_tokens:
            from verl.models.transformers.vision_token_compressor import HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM

            if compressor_algorithm == HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM:
                if micro_batch.get("dart_merge_routes") is not None:
                    raise RuntimeError("HoliTom-DPC replay forbids legacy dart_merge_routes")
                dpc_merge_routes = self._prepare_replay_holitom_dpc_routes(
                    micro_batch.get("dpc_merge_routes"), multi_modal_inputs
                )
            else:
                if micro_batch.get("dpc_merge_routes") is not None:
                    raise RuntimeError("Legacy DART replay forbids dpc_merge_routes")
                dart_merge_routes = self._prepare_replay_dart_routes(
                    micro_batch.get("dart_merge_routes"),
                    multi_modal_inputs,
                    expected_algorithm=str(compressor_algorithm),
                    sample_uids=micro_batch.get("uid"),
                    input_ids=micro_batch["input_ids"],
                    attention_mask=micro_batch["attention_mask"],
                    response_length=response_length,
                )
        elif micro_batch.get("dart_merge_routes") is not None or micro_batch.get("dpc_merge_routes") is not None:
            raise RuntimeError(f"visual_compression_mode={visual_compression_mode!r} forbids merge-route replay")

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            response_start_idx = micro_batch.get("response_start_idx")
            if merge_visual_tokens:
                input_ids, attention_mask, position_ids, response_start_idx = (
                    self._crop_common_dart_prompt_left_padding(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                )
                batch_size, seqlen = input_ids.shape
                response_start_idx = self._adjust_response_start_for_visual_compression(
                    module=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    response_length=response_length,
                    response_start_idx=response_start_idx,
                )
            if response_start_idx is not None:
                response_start_idx = response_start_idx.to(device=input_ids.device, dtype=torch.long)
                if response_start_idx.shape != (batch_size,):
                    raise ValueError(
                        f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
                    )
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if dart_merge_routes is not None:
                    extra_args["dart_merge_routes"] = dart_merge_routes
                if dpc_merge_routes is not None:
                    extra_args["dpc_merge_routes"] = dpc_merge_routes
                if self._supports_functional_visual_compression_mode(model):
                    extra_args["visual_compression_mode"] = visual_compression_mode

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            if response_start_idx is None:
                                full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            else:
                                response_positions = self._build_response_positions(
                                    response_start_idx=response_start_idx.to(device=topk_indices.device),
                                    response_length=response_length,
                                    seqlen=seqlen,
                                )
                                batch_indices = torch.arange(batch_size, device=topk_indices.device).unsqueeze(1)
                                full_topk_indices[batch_indices, response_positions, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = _chunked_vocab_logsumexp(logits_rmpad)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = self._select_response_positions(
                        full_entropy.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = self._select_response_positions(
                        full_sum_pi_squared.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                log_probs = self._select_response_positions(
                    full_log_probs.squeeze(-1),
                    response_length=response_length,
                    response_start_idx=response_start_idx,
                )
                if compute_all_logps:
                    all_logps = self._select_response_positions(
                        full_all_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if use_topk:
                    topk_logps = self._select_response_positions(
                        full_topk_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    if return_topk_indices:
                        topk_indices = self._select_response_positions(
                            full_topk_indices,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                logits_are_response_only = False
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if dart_merge_routes is not None:
                    extra_args["dart_merge_routes"] = dart_merge_routes
                if dpc_merge_routes is not None:
                    extra_args["dpc_merge_routes"] = dpc_merge_routes
                if self._supports_functional_visual_compression_mode(model):
                    extra_args["visual_compression_mode"] = visual_compression_mode

                # Qwen3.5 accepts explicit per-sample hidden-state indices for
                # its vocabulary projection.  OPD only consumes logits that
                # predict response tokens, so projecting every visual/prompt
                # position wastes substantial compute and memory (especially
                # for the full-token teacher) without contributing to the
                # objective.  DART response_start_idx has already been shifted
                # to the compressed physical sequence above.
                model_config = getattr(getattr(model, "module", model), "config", None)
                model_type = getattr(model_config, "model_type", None)
                if not self.use_fused_kernels and model_type in {"qwen3_5", "qwen3_5_moe"}:
                    if response_start_idx is None:
                        response_logit_indices = torch.arange(
                            seqlen - response_length - 1,
                            seqlen - 1,
                            dtype=torch.long,
                            device=input_ids.device,
                        )
                    else:
                        response_logit_indices = self._build_response_positions(
                            response_start_idx=response_start_idx,
                            response_length=response_length,
                            seqlen=seqlen,
                        )
                    extra_args["logits_to_keep"] = response_logit_indices
                    logits_are_response_only = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    if logits_are_response_only:
                        if logits.shape[:2] != (batch_size, response_length):
                            raise RuntimeError(
                                "Qwen3.5 indexed lm_head returned an unexpected shape: "
                                f"got {tuple(logits.shape)}, expected "
                                f"({batch_size}, {response_length}, vocab_size)"
                            )
                    else:
                        logits = self._select_response_positions(
                            logits,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                            if union_support:
                                if additional_topk_indices.shape[:2] != logits.shape[:2]:
                                    raise ValueError("Teacher top-k indices do not align with student response logits")
                                if rollout_token_ids.shape != logits.shape[:2]:
                                    raise ValueError("Rollout token IDs do not align with student response logits")
                                topk_indices, topk_valid_mask = self._build_union_support_indices(
                                    topk_indices, additional_topk_indices, rollout_token_ids
                                )
                                topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                            if topk_valid_mask is not None and topk_valid_mask.shape != topk_indices.shape:
                                raise ValueError("topk_valid_mask must match explicit support indices")
                        logsumexp = _chunked_vocab_logsumexp(logits)
                        topk_logps = topk_logits - logsumexp
                        if topk_valid_mask is not None:
                            topk_logps = topk_logps.masked_fill(~topk_valid_mask, -1.0e9)
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            if response_start_idx is None:
                response_start_idx = torch.full(
                    (batch_size,),
                    seqlen - response_length,
                    device=input_ids.device,
                    dtype=torch.long,
                )
            outputs = {"log_probs": log_probs, "response_start_idx": response_start_idx}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
                    if topk_valid_mask is not None:
                        outputs["topk_valid_mask"] = topk_valid_mask
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        self._last_optimizer_step_succeeded = False
        self._last_optimizer_lrs_by_group = {
            str(group.get("group_name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(self.actor_optimizer.param_groups)
        }
        if any(not math.isfinite(value) or value < 0 for value in self._last_optimizer_lrs_by_group.values()):
            raise FloatingPointError(
                f"Optimizer parameter-group learning rates are invalid: {self._last_optimizer_lrs_by_group}"
            )
        self._last_optimizer_lr = float(self.actor_optimizer.param_groups[0]["lr"])
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        if self.actor_lr_scheduler is not None and bool(
            self.config.optim.get("lr_scheduler_step_per_optimizer_step", False)
        ):
            # A scheduler is defined over optimizer updates, not outer rollout
            # batches.  Advancing it here keeps batch=8 behavior unchanged and
            # preserves the intended 685-step schedule when batch=16 is split
            # into two optimizer mini-batches.
            self.actor_lr_scheduler.step()
        self._last_optimizer_step_succeeded = True
        if not getattr(self, "_fp32_optimizer_state_validated", False):
            moment_keys = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
            moment_count = 0
            for group in self.actor_optimizer.param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue
                    if param.dtype != torch.float32:
                        raise TypeError(f"Trainable optimizer parameter must be FP32, got {param.dtype}")
                    if not torch.isfinite(param).all().item():
                        raise FloatingPointError("Trainable parameter contains NaN or Inf after optimizer step")
                    state = self.actor_optimizer.state.get(param, {})
                    for key in moment_keys.intersection(state):
                        value = state[key]
                        if value.dtype != torch.float32:
                            raise TypeError(f"Adam {key} must be FP32, got {value.dtype}")
                        if not torch.isfinite(value).all().item():
                            raise FloatingPointError(f"Adam {key} contains NaN or Inf")
                        moment_count += 1
            if moment_count == 0:
                raise RuntimeError("No Adam FP32 moment tensors were created after the first optimizer step")
            self._fp32_optimizer_state_validated = True
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        visual_compression_mode = data.meta_info.get("visual_compression_mode")
        if visual_compression_mode is not None and visual_compression_mode not in {
            "merge",
            "dense",
            "no_image",
        }:
            raise ValueError(f"Unsupported visual_compression_mode={visual_compression_mode!r}")
        diagnostic_model_role = data.meta_info.get("diagnostic_model_role", "current_actor")
        if diagnostic_model_role == "current_actor":
            score_model = self.actor_module
        elif diagnostic_model_role == "fixed_teacher":
            if self.teacher_module is None or self.teacher_module is self.actor_module:
                raise RuntimeError("fixed_teacher diagnostics require a separate immutable teacher module")
            if any(parameter.requires_grad for parameter in self.teacher_module.parameters()):
                raise RuntimeError("fixed_teacher diagnostics require a frozen teacher module")
            if visual_compression_mode is None:
                visual_compression_mode = "dense"
            if visual_compression_mode != "dense":
                raise ValueError("fixed_teacher diagnostics require visual_compression_mode='dense'")
            score_model = self.teacher_module
        else:
            raise ValueError(f"Unsupported diagnostic_model_role={diagnostic_model_role!r}")
        score_model.eval()
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if visual_compression_mode in {None, "merge"}:
            if "dart_merge_routes" in data.non_tensor_batch:
                non_tensor_select_keys.append("dart_merge_routes")
            if "dpc_merge_routes" in data.non_tensor_batch:
                non_tensor_select_keys.append("dpc_merge_routes")
            if "dart_merge_routes" in data.non_tensor_batch:
                from verl.models.transformers.vision_token_compressor import CDPRUNER_ALGORITHM

                if self._visual_token_compressor_algorithm(score_model) == CDPRUNER_ALGORITHM:
                    if "uid" not in data.non_tensor_batch:
                        raise RuntimeError("V8 CDPruner scoring requires sample-aligned rollout UIDs")
                    non_tensor_select_keys.append("uid")
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch and "uid" not in non_tensor_select_keys:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    module=score_model,
                    visual_compression_mode=visual_compression_mode,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._current_global_steps = data.meta_info.get("global_steps")
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "vopd"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        self_distillation_gamma = 1.0
        if self_distillation_enabled:
            if self_distillation_cfg is None:
                raise ValueError(f"loss_mode={loss_mode} requires actor.self_distillation config.")
            self_distillation_gamma = float(self_distillation_cfg.get("gamma", 1.0))
            if not math.isfinite(self_distillation_gamma) or self_distillation_gamma < 0.0:
                raise ValueError(
                    "actor.self_distillation.gamma must be finite and non-negative, "
                    f"got {self_distillation_gamma}"
                )
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "teacher_response_start_idx",
                "self_distillation_mask",
            }
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
        ]
        if not self_distillation_enabled or "advantages" in data.batch.keys():
            select_keys.append("advantages")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        has_teacher_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_teacher_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_multi_modal_inputs")
        if "dart_merge_routes" in data.non_tensor_batch:
            non_tensor_select_keys.append("dart_merge_routes")
        if "dpc_merge_routes" in data.non_tensor_batch:
            non_tensor_select_keys.append("dpc_merge_routes")
        if "dart_merge_routes" in data.non_tensor_batch:
            from verl.models.transformers.vision_token_compressor import CDPRUNER_ALGORITHM

            if self._visual_token_compressor_algorithm(self.actor_module) == CDPRUNER_ALGORITHM:
                if "uid" not in data.non_tensor_batch:
                    raise RuntimeError("V8 CDPruner update requires sample-aligned rollout UIDs")
                non_tensor_select_keys.append("uid")
        elif self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        if self_distillation_enabled:
            metrics["actor/grpo_loss"] = 0.0
            metrics["actor/vopd_loss"] = 0.0
            metrics["actor/vopd_loss_weighted"] = 0.0
            self._teacher_forward_chunk_calls = 0
            self._teacher_forward_chunk_samples = 0
        distill_dump_chunks = []
        stage_wall_time_totals = None
        if self_distillation_enabled:
            stage_wall_time_totals = {
                "timing_s/update_actor/student_forward": 0.0,
                "timing_s/update_actor/teacher_forward": 0.0,
                "timing_s/update_actor/loss_compute": 0.0,
                "timing_s/update_actor/backward": 0.0,
                "timing_s/update_actor/optimizer_step": 0.0,
                "timing_s/update_actor/teacher_ema_update": 0.0,
            }
        did_update = False
        successful_optimizer_steps = 0
        nonzero_lr_optimizer_steps = 0
        last_optimizer_lr = None
        last_optimizer_lrs_by_group = None
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                distillation_batch_info = (
                    self._global_distillation_batch_info(mini_batch) if self_distillation_enabled else None
                )
                if self.config.use_dynamic_bsz:
                    if self.config.get("training_mode", "legacy") == "full_parameter":
                        packing = self.config.vision_packing
                        # ActorConfig.model_config is intentionally a generic
                        # BaseConfig in real Ray workers and does not carry the
                        # nested compressor schema.  Resolve the immutable route
                        # contract from the actually installed model instead of
                        # relying on a richer test-only config object.
                        route_key = self._visual_token_route_transport_key(self.actor_module)
                        composite_costs = build_vopd_composite_costs(
                            mini_batch,
                            route_key=route_key,
                            teacher_forward_multiplier=int(packing["teacher_forward_multiplier"]),
                            raw_patch_weight=float(packing["raw_patch_weight"]),
                            image_overhead=float(packing["image_overhead"]),
                        )
                        micro_batches, _, packing_report = prepare_dynamic_batch_by_composite_cost(
                            mini_batch,
                            composite_costs,
                            max_cost_per_gpu=int(packing["max_cost_per_gpu"]),
                            max_trajectories_per_microbatch=int(
                                packing["max_trajectories_per_microbatch"]
                            ),
                        )
                        metrics["packing/composite_micro_batches"] = packing_report["micro_batch_count"]
                        metrics["packing/composite_samples_per_micro_batch"] = packing_report[
                            "samples_per_micro_batch"
                        ]
                        metrics["packing/max_trajectories_per_microbatch"] = packing_report[
                            "max_trajectories_per_microbatch"
                        ]
                        metrics["packing/composite_partition_cost_max"] = packing_report[
                            "partition_cost_max"
                        ]
                        metrics["packing/composite_sample_cost_max"] = packing_report["sample_cost_max"]
                    else:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs.get("advantages")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    policy_fallback_mask = None
                    if self_distillation_enabled and self_distillation_mask is not None:
                        policy_fallback_mask = (self_distillation_mask <= 0.5).to(response_mask.dtype)
                        micro_batch_metrics["actor/policy_fallback_fraction"] = (
                            policy_fallback_mask.float().mean().detach().item()
                        )

                    if self.config.use_dynamic_bsz:
                        micro_batch_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        micro_batch_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    teacher_model_source = self_distillation_cfg.get("teacher_model_source", "legacy")
                    use_trust_region_teacher = teacher_model_source == "legacy" and teacher_regularization == "trust-region"
                    if use_trust_region_teacher and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                    distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                    support_policy = self_distillation_cfg.get("distillation_support_policy", "student_topk")
                    use_union_support = bool(
                        self_distillation_enabled
                        and distill_topk
                        and support_policy == "student_teacher_topk_plus_rollout"
                    )
                    if support_policy not in {"student_topk", "student_teacher_topk_plus_rollout"}:
                        raise ValueError(f"Unsupported distillation_support_policy={support_policy!r}")

                    teacher_inputs = None
                    teacher_model = None
                    teacher_seed_indices = None
                    if use_union_support:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                            "response_start_idx": model_inputs["teacher_response_start_idx"],
                        }
                        if "teacher_multi_modal_inputs" in model_inputs:
                            teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
                        teacher_model = self.teacher_module or self.actor_module
                        if use_trust_region_teacher and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module")
                        with torch.no_grad():
                            teacher_seed_start = time.perf_counter()
                            teacher_seed_kwargs = {
                                "temperature": temperature,
                                "calculate_entropy": False,
                                "return_all_logps": False,
                                "distill_topk": distill_topk,
                                "visual_compression_mode": (
                                    "dense"
                                    if self._has_non_empty_multi_modal_inputs(
                                        teacher_inputs.get("multi_modal_inputs")
                                    )
                                    else "no_image"
                                ),
                            }
                            if self.config.get("training_mode", "legacy") == "full_parameter":
                                teacher_seed_outputs = self._forward_fixed_teacher_in_chunks(
                                    teacher_inputs,
                                    module=teacher_model,
                                    micro_batch_size=int(
                                        self.config.vision_packing["teacher_micro_batch_size_per_gpu"]
                                    ),
                                    **teacher_seed_kwargs,
                                )
                            else:
                                teacher_seed_outputs = self._forward_micro_batch(
                                    teacher_inputs, module=teacher_model, **teacher_seed_kwargs
                                )
                            teacher_seed_indices = teacher_seed_outputs["topk_indices"]
                            teacher_seed_time = time.perf_counter() - teacher_seed_start
                        stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += teacher_seed_time
                    student_forward_start = time.perf_counter()
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                        additional_topk_indices=teacher_seed_indices,
                        rollout_token_ids=model_inputs["responses"] if use_union_support else None,
                        visual_compression_mode=self._visual_compression_mode_for_inputs(
                            self.actor_module, model_inputs.get("multi_modal_inputs")
                        ),
                    )
                    if self_distillation_enabled:
                        student_forward_time = time.perf_counter() - student_forward_start
                        stage_wall_time_totals["timing_s/update_actor/student_forward"] += student_forward_time
                    # V8_FULL_LOGIT_VOPD_AUXILIARY_BRANCH_RELEASE_V1
                    # Full-distribution JSD differentiates the top-k/full-logit
                    # branch. Token log-probabilities only form detached IS
                    # weights here. Drop their unused CE autograd branch before
                    # teacher work/backward; otherwise it can retain the full
                    # vocabulary logits throughout checkpoint recomputation.
                    # Keep all legacy, fallback, KL and entropy paths intact.
                    if (
                        self_distillation_enabled
                        and self.config.get("training_mode", "legacy") == "full_parameter"
                        and self_distillation_cfg.full_logit_distillation
                        and not self.config.use_kl_loss
                        and not calculate_entropy
                        and (policy_fallback_mask is None or not policy_fallback_mask.any().item())
                    ):
                        outputs["log_probs"] = outputs["log_probs"].detach()
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None
                    student_topk_valid_mask = outputs.get("topk_valid_mask") if distill_topk else None
                    if use_union_support and student_topk_valid_mask is None:
                        raise RuntimeError("Union support construction did not return its validity mask")

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        # For OPD, old_log_probs is the HF-old snapshot used by
                        # the existing self_distillation.is_clip update ratio.
                        # Never replace it with HF-current when central backend
                        # correction is present; doing so made the second ratio
                        # mechanically equal to one and erased the old snapshot.
                        if on_policy and not self_distillation_enabled and "rollout_is_weights" not in model_inputs:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        if teacher_inputs is None:
                            teacher_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["teacher_input_ids"],
                                "attention_mask": model_inputs["teacher_attention_mask"],
                                "position_ids": model_inputs["teacher_position_ids"],
                                "response_start_idx": model_inputs["teacher_response_start_idx"],
                            }
                            if "teacher_multi_modal_inputs" in model_inputs:
                                teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
                        teacher_model = teacher_model or self.teacher_module or self.actor_module
                        if use_trust_region_teacher and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                        with torch.no_grad():
                            teacher_forward_start = time.perf_counter()
                            teacher_forward_kwargs = {
                                "temperature": temperature,
                                "calculate_entropy": False,
                                "return_all_logps": return_all_logps,
                                "distill_topk": distill_topk,
                                "topk_indices": student_topk_indices,
                                "topk_valid_mask": student_topk_valid_mask,
                                "visual_compression_mode": (
                                    "dense"
                                    if self._has_non_empty_multi_modal_inputs(
                                        teacher_inputs.get("multi_modal_inputs")
                                    )
                                    else "no_image"
                                ),
                            }
                            if self.config.get("training_mode", "legacy") == "full_parameter":
                                teacher_outputs = self._forward_fixed_teacher_in_chunks(
                                    teacher_inputs,
                                    module=teacher_model,
                                    micro_batch_size=int(
                                        self.config.vision_packing["teacher_micro_batch_size_per_gpu"]
                                    ),
                                    **teacher_forward_kwargs,
                                )
                            else:
                                teacher_outputs = self._forward_micro_batch(
                                    teacher_inputs, module=teacher_model, **teacher_forward_kwargs
                                )
                            teacher_forward_time = time.perf_counter() - teacher_forward_start
                        stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += teacher_forward_time
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                        if self_distillation_cfg.get("log_prob_dump_dir", None):
                            if distill_topk:
                                student_distill_log_probs = student_topk_logps
                                teacher_distill_log_probs = teacher_topk_logps
                                if self_distillation_cfg.distillation_add_tail:
                                    student_distill_log_probs = self._add_tail_bucket(student_distill_log_probs)
                                    teacher_distill_log_probs = self._add_tail_bucket(teacher_distill_log_probs)
                            else:
                                student_distill_log_probs = student_all_logps
                                teacher_distill_log_probs = teacher_all_logps

                            if student_distill_log_probs is None or teacher_distill_log_probs is None:
                                raise ValueError("Missing distillation log_probs for dump.")

                            loss_mask = response_mask
                            if self_distillation_mask is not None:
                                loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
                            valid_rows = loss_mask > 0
                            if valid_rows.any():
                                distill_dump_chunks.append(
                                    {
                                        "student_log_probs": student_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                        "teacher_log_probs": teacher_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                        "student_response_start_idx": outputs["response_start_idx"].detach().cpu(),
                                        "teacher_response_start_idx": teacher_outputs["response_start_idx"].detach().cpu(),
                                        "topk_valid_mask": student_topk_valid_mask[valid_rows].detach().cpu(),
                                        "support_indices": student_topk_indices[valid_rows].detach().cpu(),
                                        "teacher_seed_indices": teacher_seed_indices[valid_rows].detach().cpu(),
                                        "rollout_token_ids": model_inputs["responses"][valid_rows].detach().cpu(),
                                    }
                                )
                        loss_compute_start = time.perf_counter()
                        vopd_loss, vopd_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            topk_valid_mask=student_topk_valid_mask,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            batch_num_tokens=distillation_batch_info["batch_num_tokens"],
                            global_batch_size=distillation_batch_info["global_batch_size"],
                            loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                            dp_size=distillation_batch_info["dp_size"],
                        )
                        loss_compute_time = time.perf_counter() - loss_compute_start
                        stage_wall_time_totals["timing_s/update_actor/loss_compute"] += loss_compute_time

                        vopd_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                        micro_batch_metrics.update(vopd_metrics)

                        if policy_fallback_mask is not None and policy_fallback_mask.any().item():
                            if advantages is None:
                                raise ValueError(
                                    "Mixed SDPO/GRPO fallback requires advantages for samples without teacher images."
                                )
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            grpo_loss, grpo_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask * policy_fallback_mask.unsqueeze(1),
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            grpo_contribution = grpo_loss * micro_batch_scale_factor
                            pg_loss = vopd_loss * self_distillation_gamma + grpo_contribution
                            micro_batch_metrics.update(
                                {f"actor/policy_fallback/{key.split('/', 1)[1]}": value for key, value in grpo_metrics.items()}
                            )
                        else:
                            grpo_loss = None
                            grpo_contribution = None
                            pg_loss = vopd_loss * self_distillation_gamma
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            entropy_scale = micro_batch_scale_factor if self_distillation_enabled else 1.0
                            policy_loss -= entropy_agg * entropy_coeff * entropy_scale

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        kl_scale = micro_batch_scale_factor if self_distillation_enabled else 1.0
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef * kl_scale
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * micro_batch_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # VOPD already uses a global token/sequence denominator and
                    # dp_size compensation, so applying the legacy sample-count
                    # scale here would make the gradient depend on dynamic
                    # micro-batch partitioning.  Other policy losses retain the
                    # existing accumulation scale.
                    loss = policy_loss if self_distillation_enabled else policy_loss * micro_batch_scale_factor
                    backward_start = time.perf_counter()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self_distillation_enabled:
                        backward_time = time.perf_counter() - backward_start
                        stage_wall_time_totals["timing_s/update_actor/backward"] += backward_time

                    metric_scale = 1.0 if self_distillation_enabled else micro_batch_scale_factor
                    metrics["actor/pg_loss"] += pg_loss.detach().item() * metric_scale
                    if self_distillation_enabled:
                        metrics["actor/vopd_loss"] += vopd_loss.detach().item()
                        metrics["actor/vopd_loss_weighted"] += (
                            vopd_loss.detach().item() * self_distillation_gamma
                        )
                        if grpo_loss is not None:
                            metrics["actor/grpo_loss"] += grpo_contribution.detach().item()
                    append_to_dict(metrics, micro_batch_metrics)

                    if self_distillation_enabled and self.config.get("training_mode", "legacy") == "full_parameter":
                        # Metrics are scalars and optional dumps are already on
                        # CPU. Do not carry completed microbatch graph/output
                        # references into the next teacher/student forward.
                        outputs = teacher_outputs = teacher_seed_outputs = None
                        log_prob = entropy = student_all_logps = student_topk_logps = None
                        student_topk_indices = student_topk_valid_mask = teacher_seed_indices = None
                        teacher_log_prob = teacher_all_logps = teacher_topk_logps = None
                        student_distill_log_probs = teacher_distill_log_probs = None
                        loss = policy_loss = pg_loss = vopd_loss = None
                        grpo_loss = grpo_contribution = None
                        teacher_forward_kwargs = teacher_seed_kwargs = None
                        model_inputs = teacher_inputs = rollout_is_weights = rollout_log_prob = None

                optimizer_step_start = time.perf_counter()
                grad_norm = self._optimizer_step()
                if self_distillation_enabled:
                    optimizer_step_time = time.perf_counter() - optimizer_step_start
                if torch.isfinite(grad_norm).item():
                    did_update = True
                if self._last_optimizer_step_succeeded:
                    successful_optimizer_steps += 1
                    last_optimizer_lr = self._last_optimizer_lr
                    last_optimizer_lrs_by_group = dict(self._last_optimizer_lrs_by_group)
                    if all(value > 0 for value in last_optimizer_lrs_by_group.values()):
                        nonzero_lr_optimizer_steps += 1
                grad_norm_value = grad_norm.detach().item()
                grad_clip_threshold = float(self.config.grad_clip)
                mini_batch_metrics = {
                    "actor/grad_norm": grad_norm_value,
                    "actor/grad_clip_threshold": grad_clip_threshold,
                    "actor/grad_clip_ratio": grad_norm_value / grad_clip_threshold,
                    "actor/gradient_clipped": float(grad_norm_value > grad_clip_threshold),
                }
                if self_distillation_enabled:
                    stage_wall_time_totals["timing_s/update_actor/optimizer_step"] += optimizer_step_time
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self_distillation_enabled and distill_dump_chunks:
            self._dump_self_distillation_log_probs(
                meta_info=data.meta_info,
                self_distillation_cfg=self_distillation_cfg,
                dump_chunks=distill_dump_chunks,
            )
        if did_update:
            teacher_update_start = time.perf_counter()
            self._update_teacher()
            if self_distillation_enabled:
                stage_wall_time_totals["timing_s/update_actor/teacher_ema_update"] += (
                    time.perf_counter() - teacher_update_start
                )
        metrics["actor/optimizer_steps"] = successful_optimizer_steps
        metrics["actor/nonzero_lr_optimizer_steps"] = nonzero_lr_optimizer_steps
        if last_optimizer_lr is not None:
            metrics["actor/lr"] = last_optimizer_lr
        if last_optimizer_lrs_by_group is not None:
            for group_name, value in sorted(last_optimizer_lrs_by_group.items()):
                metrics[f"actor/lr_group/{group_name}"] = value
        if self_distillation_enabled:
            metrics["teacher/chunk_forward_calls"] = float(self._teacher_forward_chunk_calls)
            metrics["teacher/samples_per_chunk"] = float(
                self._teacher_forward_chunk_samples / max(self._teacher_forward_chunk_calls, 1)
            )
            for key, total_time in stage_wall_time_totals.items():
                metrics[key] = Metric(aggregation=AggregationType.MAX, value=total_time)
        metric_keys_to_keep_unreduced = set(stage_wall_time_totals.keys()) if stage_wall_time_totals is not None else set()
        local_metrics_to_reduce = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, list) or (isinstance(value, Metric) and key not in metric_keys_to_keep_unreduced)
        }
        if local_metrics_to_reduce:
            reduced_local_metrics = reduce_metrics(local_metrics_to_reduce)
            metrics.update(reduced_local_metrics)
        return metrics
