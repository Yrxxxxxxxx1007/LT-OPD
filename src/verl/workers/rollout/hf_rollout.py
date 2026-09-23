# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0

"""
Rollout with huggingface models.

Patched version.

Supports both:

1. Old verl format:
   prompts.batch contains:
       input_ids / attention_mask / position_ids

2. New verl format:
   prompts.batch contains:
       dummy_tensor

   prompts.non_tensor_batch contains one of:
       raw_prompt / prompt / prompts

Your current case:
   prompts.batch.keys() = ['dummy_tensor']
   prompts.non_tensor_batch.keys() = [
       'index',
       'ability',
       'prompt',
       'route_query',
       'extra_info',
       'interaction_kwargs',
       'bbox_images',
       'tools_kwargs'
   ]
"""

import contextlib
import copy
from collections.abc import Mapping

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList

from verl import DataProto
from verl.models.transformers.vision_token_compressor import (
    CDPRUNER_ALGORITHM,
    HOLITOM_DPC_MERGE_ROUTES_KEY,
    HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM,
    DARTMergeRoute,
    HoliTomDPCSpatialMergeRoute,
    NO_IMAGE_ABLATION_POLICY,
    VISUAL_COMPRESSION_MODES,
)
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.route_query import (
    ROUTE_QUERY_POLICY,
    SUPPORTED_ROUTE_QUERY_SCHEMAS,
    RouteQuerySpec,
    parse_route_query_text,
    parse_route_query_value,
    user_text_for_prompt_fallback,
)
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.semantic_stop import (
    STOP_REASON_ANSWER_TAG,
    STOP_REASON_TOKEN_EOS,
    WellFormedAnswerTagCriteria,
    response_action_mask,
    stop_reason_names,
)

from .base import BaseRollout

__all__ = ["HFRollout"]


def _split_dataproto_by_max_batch_size(prompts: DataProto, maximum_batch_size: int) -> list[DataProto]:
    """Split a rollout batch into bounded chunks, allowing one final partial chunk.

    ``DataProto.chunk`` requires every chunk to have exactly the same size.  A
    validation shard can be divisible by the FSDP world size without also
    being divisible by the configured HF decode batch size (for example,
    612 / 4 ranks = 153 samples and a decode batch size of 4).  ``split`` has
    the required fixed-size semantics and preserves input order.
    """

    maximum_batch_size = int(maximum_batch_size)
    if maximum_batch_size <= 0:
        raise ValueError(f"maximum_batch_size must be positive, got {maximum_batch_size}")
    chunks = prompts.split(split_size=maximum_batch_size)
    if not chunks or sum(len(chunk) for chunk in chunks) != len(prompts):
        raise RuntimeError("HF rollout batch splitting did not conserve the input batch")
    if any(len(chunk) <= 0 or len(chunk) > maximum_batch_size for chunk in chunks):
        raise RuntimeError("HF rollout batch splitting produced an invalid chunk size")
    return chunks


class HFRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        module: nn.Module,
        tokenizer=None,
        processor=None,
        keep_module_in_eval: bool = False,
    ):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        self.module = module
        self.keep_module_in_eval = bool(keep_module_in_eval)

        # Some verl versions put tokenizer / processor inside model_config.
        # HFModelConfig is a serializable config object and normally does not
        # carry live ProcessorMixin instances.  The worker that loaded the
        # model owns the authoritative tokenizer/processor and must pass them
        # explicitly; falling back to model_config keeps older call sites
        # compatible.  A multimodal DART rollout fails closed below if the
        # processor is missing instead of silently tokenizing only the text.
        self.tokenizer = tokenizer or getattr(model_config, "tokenizer", None)
        self.processor = processor or getattr(model_config, "processor", None)

        # Fallback: load tokenizer from model path.
        if self.tokenizer is None:
            tokenizer_path = (
                getattr(model_config, "tokenizer_path", None)
                or getattr(model_config, "path", None)
                or getattr(model_config, "model_path", None)
            )

            if tokenizer_path is None:
                raise ValueError(
                    "HFRollout cannot find tokenizer. "
                    "Expected model_config.tokenizer or "
                    "model_config.tokenizer_path/path/model_path."
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )

        # Make sure tokenizer has pad token.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer has no pad_token_id and no eos_token. "
                    "Please set tokenizer.pad_token or tokenizer.pad_token_id."
                )

    async def resume(self, tags: list[str]):
        return None

    async def update_weights(self, weights, **kwargs):
        return None

    async def release(self):
        return None

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = len(prompts)
        compressor_enabled = bool(getattr(self.model_config, "vision_token_compressor", {}).get("enabled", False))
        if compressor_enabled and self.processor is None:
            raise RuntimeError(
                "Visual-compression HF rollout requires the live multimodal processor; "
                "tokenizer-only rollout would omit image pixels and is forbidden."
            )
        if compressor_enabled:
            # Every decode chunk below executes distributed collectives.
            # Unequal local batch sizes would make some ranks enter an extra
            # generate call while their peers have already returned, causing a
            # collective-order mismatch or hang.  Detect that before the first
            # model forward so a bad global batch fails deterministically.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                local_batch_size = torch.tensor(
                    batch_size,
                    dtype=torch.int64,
                    device=self._get_module_device(),
                )
                rank_batch_sizes = [
                    torch.empty_like(local_batch_size)
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(rank_batch_sizes, local_batch_size)
                rank_batch_sizes = [int(value.item()) for value in rank_batch_sizes]
                if len(set(rank_batch_sizes)) != 1:
                    raise RuntimeError(
                        "HF DART rollout requires the same local batch size on every FSDP rank; "
                        f"got {rank_batch_sizes}. Make the generated batch divisible by world size."
                    )
            decode_batch_size = int(self.config.get("hf_dart_decode_batch_size", 1))
            if decode_batch_size <= 0:
                raise ValueError(
                    "rollout.hf_dart_decode_batch_size must be positive, "
                    f"got {decode_batch_size}"
                )
            decode_max_prefill_cost = int(
                self.config.get("hf_dart_decode_max_prefill_cost", 0)
            )
            if decode_max_prefill_cost > 0:
                batch_prompts = self._split_holitom_uid_groups_by_prefill_cost(
                    prompts,
                    maximum_batch_size=decode_batch_size,
                    maximum_prefill_cost=decode_max_prefill_cost,
                )
            else:
                batch_prompts = _split_dataproto_by_max_batch_size(prompts, decode_batch_size)
        else:
            num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
            batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        if compressor_enabled:
            output = self._left_pad_dart_outputs(output)
        output = DataProto.concat(output)
        return output

    def _split_holitom_uid_groups_by_prefill_cost(
        self,
        prompts: DataProto,
        *,
        maximum_batch_size: int,
        maximum_prefill_cost: int,
    ) -> list[DataProto]:
        """Pack complete rollout groups without changing rows or their order.

        Prefix sharing makes prefill cost proportional to unique prompts, while
        decode state is proportional to trajectories.  This deterministic
        contiguous packer constrains both.  It deliberately does not perform
        rolling refill: every returned chunk owns an ordinary, self-contained
        Qwen hybrid cache and therefore has no cache-splicing ambiguity.
        """

        if not bool(self.config.get("hf_use_replicated_module", False)):
            raise RuntimeError("Cost-bounded rollout packing requires the independent HF replica")
        if not bool(self.config.get("hf_share_rollout_prefix", False)):
            raise RuntimeError("Cost-bounded rollout packing requires exact shared-prefix validation")
        rollout_n = int(self.config.get("n", 0))
        if rollout_n <= 0 or len(prompts) % rollout_n:
            raise RuntimeError("Cost-bounded rollout packing requires complete n-way trajectory groups")
        if maximum_batch_size < rollout_n or maximum_batch_size % rollout_n:
            raise ValueError("maximum_batch_size must be a positive multiple of rollout.n")
        if maximum_prefill_cost <= 0:
            raise ValueError("maximum_prefill_cost must be positive")
        non_tensor = prompts.non_tensor_batch
        if non_tensor is None or "uid" not in non_tensor:
            raise RuntimeError("Cost-bounded rollout packing requires batch-aligned UID bindings")
        uids = list(non_tensor["uid"])
        if len(uids) != len(prompts):
            raise RuntimeError("Cost-bounded rollout UIDs are not batch aligned")

        # Legacy/pretokenized batches already carry the exact attention mask
        # and processed image grid.  The current RLHFDataset deliberately
        # transports raw chat objects plus signed capacity metadata; image
        # preprocessing happens inside ``_generate_minibatch`` *after* this
        # packer.  Requiring ``multi_modal_inputs`` here therefore rejects every
        # valid raw-prompt formal batch before decode.  Use the materialized
        # per-profile cost in that path -- the same binding used by the
        # controller's cross-rank group balancer -- without preprocessing an
        # image twice or changing row order.
        attention_mask = prompts.batch.get("attention_mask") if prompts.batch is not None else None
        multimodal = non_tensor.get("multi_modal_inputs")
        processed_inputs = (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.shape[0] == len(prompts)
            and multimodal is not None
            and len(multimodal) == len(prompts)
        )
        extra_infos = None
        selected_max_pixels = None
        curriculum_config = getattr(self.model_config, "vision_token_compressor", {}).get(
            "curriculum"
        )
        active_retention_bps = prompts.meta_info.get("visual_token_retention_bps")
        if curriculum_config is not None:
            if (
                isinstance(active_retention_bps, bool)
                or not isinstance(active_retention_bps, int)
                or not 1 <= active_retention_bps <= 10_000
            ):
                raise RuntimeError(
                    "curriculum rollout packing requires an integer visual_token_retention_bps"
                )
        if not processed_inputs:
            extra_infos = non_tensor.get("extra_info")
            model_config = getattr(self, "model_config", None)
            selected_max_pixels = getattr(model_config, "processor_max_pixels", None)
            if selected_max_pixels is None and isinstance(model_config, Mapping):
                selected_max_pixels = model_config.get("processor_max_pixels")
            if extra_infos is None or len(extra_infos) != len(prompts):
                raise RuntimeError(
                    "Raw-prompt cost-bounded rollout packing requires batch-aligned extra_info"
                )
            if isinstance(selected_max_pixels, bool) or not isinstance(selected_max_pixels, int):
                raise RuntimeError("Raw-prompt rollout packing requires processor_max_pixels")
            if selected_max_pixels <= 0:
                raise RuntimeError("Raw-prompt rollout packing requires positive processor_max_pixels")

        group_costs: list[int] = []
        for start in range(0, len(prompts), rollout_n):
            end = start + rollout_n
            if len(set(map(str, uids[start:end]))) != 1:
                raise RuntimeError("Cost-bounded rollout packing found a mixed-UID trajectory group")
            if processed_inputs:
                group_masks = attention_mask[start:end]
                if not torch.equal(group_masks, group_masks[:1].expand_as(group_masks)):
                    raise RuntimeError("Same-UID rollout group has different attention masks")
                entry = getattr(multimodal[start], "data", multimodal[start])
                if not isinstance(entry, Mapping) or entry.get("image_grid_thw") is None:
                    raise RuntimeError("Cost-bounded rollout packing requires image_grid_thw")
                grid = torch.as_tensor(entry["image_grid_thw"], dtype=torch.long)
                if grid.ndim == 1:
                    grid = grid.unsqueeze(0)
                if grid.ndim != 2 or grid.shape[-1] != 3 or bool((grid <= 0).any().item()):
                    raise RuntimeError("Cost-bounded rollout packing received invalid image_grid_thw")
                raw_patches = int(grid.prod(dim=-1).sum().item())
                group_cost = int(group_masks[0].sum().item()) + raw_patches
            else:
                metadata_costs: list[int] = []
                for extra_info in extra_infos[start:end]:
                    if not isinstance(extra_info, Mapping):
                        raise RuntimeError("Raw-prompt rollout extra_info must be a mapping")
                    profiles = extra_info.get("visual_capacity_profiles")
                    profile = profiles.get(str(selected_max_pixels)) if isinstance(profiles, Mapping) else None
                    if not isinstance(profile, Mapping) or profile.get("max_pixels") != selected_max_pixels:
                        raise RuntimeError("Raw-prompt rollout lacks the selected visual capacity profile")
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
                            raise RuntimeError(f"Raw-prompt rollout has invalid {label}")
                    metadata_costs.append(int(merged_prompt_tokens) + int(raw_patch_tokens))
                if len(set(metadata_costs)) != 1:
                    raise RuntimeError("Same-UID rollout group has inconsistent capacity metadata")
                group_cost = metadata_costs[0]
            if group_cost > maximum_prefill_cost:
                raise RuntimeError(
                    "A single rollout UID group exceeds hf_dart_decode_max_prefill_cost: "
                    f"cost={group_cost}, maximum={maximum_prefill_cost}"
                )
            group_costs.append(group_cost)

        maximum_groups = maximum_batch_size // rollout_n
        spans: list[tuple[int, int]] = []
        group_start = 0
        while group_start < len(group_costs):
            group_end = group_start + 1
            packed_cost = group_costs[group_start]
            while group_end < len(group_costs) and group_end - group_start < maximum_groups:
                candidate = packed_cost + group_costs[group_end]
                if candidate > maximum_prefill_cost:
                    break
                packed_cost = candidate
                group_end += 1
            spans.append((group_start * rollout_n, group_end * rollout_n))
            group_start = group_end
        chunks = [prompts[start:end] for start, end in spans]
        if sum(len(chunk) for chunk in chunks) != len(prompts):
            raise RuntimeError("Cost-bounded rollout packing did not conserve every trajectory")
        if any(len(chunk) % rollout_n or len(chunk) > maximum_batch_size for chunk in chunks):
            raise RuntimeError("Cost-bounded rollout packing split a trajectory group")
        return chunks

    def _left_pad_dart_outputs(self, outputs: list[DataProto]) -> list[DataProto]:
        """Pad DART decode chunks to the global batch's real maximum prompt.

        Padding every prompt to the configured safety limit (8192 in the smoke
        contract) made the HF backend perform thousands of useless masked-token
        operations.  Completed chunks are left-padded only as much as
        DataProto concatenation requires.  Responses remain at the fixed right
        edge, preserving the actor's response-start convention.
        """

        if not outputs:
            raise ValueError("HF rollout produced no DART outputs")
        target_prompt_length = max(int(output.batch["prompts"].shape[-1]) for output in outputs)
        # Each FSDP rank normally receives only its data-parallel shard.  A
        # local maximum is therefore insufficient: the controller concatenates
        # the per-rank DataProto objects and requires every non-batch dimension
        # to agree.  Exchange only the scalar length (after decoding, so this
        # adds no masked-token model compute) and left-pad to the real global
        # batch maximum.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            target_length_tensor = torch.tensor(
                target_prompt_length,
                dtype=torch.int64,
                device=outputs[0].batch["prompts"].device,
            )
            torch.distributed.all_reduce(target_length_tensor, op=torch.distributed.ReduceOp.MAX)
            target_prompt_length = int(target_length_tensor.item())
        pad_token_id = int(self.tokenizer.pad_token_id)
        for output in outputs:
            output_batch_size = len(output)
            if output_batch_size <= 0:
                raise ValueError("DART HF rollout produced an empty decode chunk")
            prompt_length = int(output.batch["prompts"].shape[-1])
            pad_length = target_prompt_length - prompt_length
            if pad_length <= 0:
                continue
            prompt_pad = torch.full(
                (output_batch_size, pad_length),
                pad_token_id,
                dtype=output.batch["prompts"].dtype,
                device=output.batch["prompts"].device,
            )
            output.batch["prompts"] = torch.cat([prompt_pad, output.batch["prompts"]], dim=-1)
            input_pad = torch.full(
                (output_batch_size, pad_length),
                pad_token_id,
                dtype=output.batch["input_ids"].dtype,
                device=output.batch["input_ids"].device,
            )
            output.batch["input_ids"] = torch.cat([input_pad, output.batch["input_ids"]], dim=-1)
            attention_pad = torch.zeros(
                (output_batch_size, pad_length),
                dtype=output.batch["attention_mask"].dtype,
                device=output.batch["attention_mask"].device,
            )
            output.batch["attention_mask"] = torch.cat(
                [attention_pad, output.batch["attention_mask"]], dim=-1
            )
            positions = output.batch["position_ids"]
            position_pad_shape = list(positions.shape)
            position_pad_shape[-1] = pad_length
            position_pad = torch.ones(
                position_pad_shape,
                dtype=positions.dtype,
                device=positions.device,
            )
            output.batch["position_ids"] = torch.cat([position_pad, positions], dim=-1)
            # Conditional-route query provenance is expressed in public
            # prompt coordinates.  Global chunk/rank padding changes those
            # coordinates even though it does not change the selected query
            # tokens.  Rebase every attached audit atomically with the tensor
            # padding; otherwise a route from another same-shaped sample can
            # pass replay while its semantic-query evidence points elsewhere.
            route_samples = (output.non_tensor_batch or {}).get("dart_merge_routes")
            if route_samples is not None:
                if len(route_samples) != output_batch_size:
                    raise RuntimeError("DART route metadata lost batch alignment during global padding")
                for sample_idx, sample_routes in enumerate(route_samples):
                    if isinstance(sample_routes, np.ndarray):
                        sample_routes = sample_routes.tolist()
                    if not isinstance(sample_routes, (list, tuple)) or not sample_routes:
                        raise RuntimeError("DART route metadata is empty during global padding")
                    for route in sample_routes:
                        if not isinstance(route, Mapping):
                            raise RuntimeError("Conditional route payload must remain a mapping")
                        audit = route.get("query_audit")
                        if not isinstance(audit, Mapping):
                            raise RuntimeError("Conditional route lost query_audit during global padding")
                        previous_width = audit.get("prefill_prompt_length")
                        previous_left_pad = audit.get("prefill_left_padding")
                        if (
                            isinstance(previous_width, bool)
                            or not isinstance(previous_width, (int, np.integer))
                            or int(previous_width) != prompt_length
                            or isinstance(previous_left_pad, bool)
                            or not isinstance(previous_left_pad, (int, np.integer))
                            or int(previous_left_pad) < 0
                        ):
                            raise RuntimeError("Conditional query audit has stale pre-padding geometry")
                        old_positions = audit.get("selected_token_indices")
                        if not isinstance(old_positions, (list, tuple)) or not old_positions:
                            raise RuntimeError("Conditional query audit has no selected token positions")
                        new_positions = [pad_length + int(value) for value in old_positions]
                        expected_ids = [int(value) for value in audit.get("selected_token_ids", [])]
                        actual_ids = [
                            int(output.batch["prompts"][sample_idx, position].item())
                            for position in new_positions
                        ]
                        if actual_ids != expected_ids:
                            raise RuntimeError("Global left padding changed route-query token identity")
                        audit["selected_token_indices"] = new_positions
                        audit["prefill_left_padding"] = int(previous_left_pad) + pad_length
                        audit["prefill_prompt_length"] = target_prompt_length
        return outputs

    def _get_module_device(self):
        try:
            return next(self.module.parameters()).device
        except StopIteration:
            try:
                return next(self.module.buffers()).device
            except StopIteration:
                # Parameter-free fixtures and wrappers legitimately have no
                # tensor from which to infer placement.  Real rollout models
                # always return above; keep the empty-module fallback usable
                # on CPU-only hosts instead of assuming an accelerator API.
                current_device = getattr(get_torch_device(), "current_device", None)
                return current_device() if callable(current_device) else torch.device("cpu")

    def _vision_compressor_config(self):
        return getattr(self.model_config, "vision_token_compressor", {}) or {}

    def _uses_holitom_dpc_spatial_merge(self) -> bool:
        config = self._vision_compressor_config()
        return bool(config.get("enabled", False)) and config.get("algorithm") == HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM

    def _uses_cdpruner(self) -> bool:
        config = self._vision_compressor_config()
        return bool(config.get("enabled", False)) and config.get("algorithm") == CDPRUNER_ALGORITHM

    def _restore_module_mode_after_rollout(self) -> None:
        """Restore a shared actor to train mode while keeping replicas inference-only."""

        if getattr(self, "keep_module_in_eval", False):
            self.module.eval()
        else:
            self.module.train()

    def _clear_and_collect_dart_routes(self, *, clear: bool):
        owner = self._get_dart_route_owner()
        if clear:
            owner.vision_token_compressor_last_routes = None
            return None
        collected = []
        for submodule in (owner,):
            if clear:
                submodule.vision_token_compressor_last_routes = None
            elif submodule.vision_token_compressor_last_routes is not None:
                collected.append(submodule.vision_token_compressor_last_routes)
        if len(collected) != 1:
            raise RuntimeError(
                "HF DART rollout must expose exactly one route owner after generation; "
                f"found {len(collected)}"
            )
        routes = collected[0]
        validated = []
        expected_algorithm = self._vision_compressor_config().get("algorithm")
        for payload in routes:
            route = (
                HoliTomDPCSpatialMergeRoute.from_dict(payload, device="cpu")
                if self._uses_holitom_dpc_spatial_merge()
                else DARTMergeRoute.from_dict(payload, device="cpu")
            )
            if isinstance(route, DARTMergeRoute):
                route.validate_for_algorithm(str(expected_algorithm))
            if route.anchor_coordinates is None:
                raise RuntimeError("HF visual-compression route is missing anchor M-RoPE coordinates")
            validated.append(route.as_dict(cpu=True))
        return validated

    def _get_dart_route_owner(self) -> nn.Module:
        owners = [
            submodule
            for submodule in self.module.modules()
            if hasattr(submodule, "vision_token_compressor_last_routes")
        ]
        if not owners:
            # On the first rollout no route has been produced yet.  Identify
            # the patched Qwen multimodal base by direct module ownership (not
            # wrapper __getattr__ delegation), then clear/create its route slot.
            owners = [
                submodule
                for submodule in self.module.modules()
                if bool(submodule.__dict__.get("vision_token_compressor_enabled", False))
                and "vision_token_compressor" in submodule.__dict__.get("_modules", {})
                and "visual" in submodule.__dict__.get("_modules", {})
                and "language_model" in submodule.__dict__.get("_modules", {})
            ]
        if len(owners) != 1:
            raise RuntimeError(
                "HF DART rollout must expose exactly one route owner; "
                f"found {len(owners)}"
            )
        return owners[0]

    @contextlib.contextmanager
    def _dart_compact_padding(self, *, enabled: bool):
        """Temporarily enable rollout-only compact padding on the compressor."""

        if not enabled:
            yield
            return
        owner = self._get_dart_route_owner()
        existed = hasattr(owner, "vision_token_compressor_compact_padding")
        previous = getattr(owner, "vision_token_compressor_compact_padding", False)
        owner.vision_token_compressor_compact_padding = bool(enabled)
        try:
            yield
        finally:
            if existed:
                owner.vision_token_compressor_compact_padding = previous
            else:
                delattr(owner, "vision_token_compressor_compact_padding")

    def _split_dart_routes_by_sample(
        self,
        routes: list[dict],
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
        """Split the compressor's flat, batch-ordered routes per sample."""

        owner = self._get_dart_route_owner()
        image_token_id = getattr(getattr(owner, "config", None), "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("DART route owner has no config.image_token_id")
        counts = []
        for row_ids, row_mask in zip(input_ids, attention_mask, strict=True):
            image_mask = (row_ids == int(image_token_id)) & row_mask.to(torch.bool)
            starts = image_mask.clone()
            starts[1:] &= ~image_mask[:-1]
            counts.append(int(starts.sum().item()))
        if sum(counts) != len(routes):
            raise RuntimeError(
                "DART route/image-span count mismatch: "
                f"routes={len(routes)}, per_sample_spans={counts}"
            )
        route_array = np.empty((input_ids.shape[0],), dtype=object)
        offset = 0
        for sample_idx, count in enumerate(counts):
            route_array[sample_idx] = routes[offset : offset + count]
            offset += count
        return route_array

    @staticmethod
    def _attach_query_audits_to_routes(routes_by_sample: np.ndarray, query_audits: list[dict]) -> np.ndarray:
        """Bind each replayable image route to the query that created it.

        ``DARTMergeRoute.from_dict`` intentionally ignores additional audit
        keys, so actor replay consumes exactly the serialized indices while the
        semantic query provenance remains attached for inspection.
        """

        if len(routes_by_sample) != len(query_audits):
            raise RuntimeError(
                "DART route/query-audit batch mismatch: "
                f"routes={len(routes_by_sample)}, audits={len(query_audits)}"
            )
        for sample_idx, audit in enumerate(query_audits):
            if not isinstance(audit, Mapping):
                raise RuntimeError(f"Missing route-query audit for sample {sample_idx}")
            sample_routes = routes_by_sample[sample_idx]
            if not isinstance(sample_routes, list) or not sample_routes:
                raise RuntimeError(f"Missing serialized DART route for sample {sample_idx}")
            for route in sample_routes:
                if not isinstance(route, dict):
                    raise RuntimeError("Serialized DART routes must be dictionaries before actor replay")
                route["query_audit"] = copy.deepcopy(dict(audit))
        return routes_by_sample

    @staticmethod
    def _warp_dart_sampling_logits(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Apply the sampling transforms used by the controlled DART decoder."""

        if temperature <= 0:
            raise ValueError(f"DART sampling temperature must be positive, got {temperature}")
        if top_k < 0:
            raise ValueError(f"DART sampling top_k must be non-negative, got {top_k}")
        if not 0 < top_p <= 1:
            raise ValueError(f"DART sampling top_p must be in (0, 1], got {top_p}")

        warped = logits.float() / float(temperature)
        if 0 < top_k < warped.shape[-1]:
            threshold = torch.topk(warped, top_k, dim=-1).values[..., -1, None]
            warped = warped.masked_fill(warped < threshold, -torch.inf)

        if top_p < 1.0:
            # Match Hugging Face's TopPLogitsWarper: remove the low-probability
            # tail in ascending order while always retaining at least one token.
            sorted_logits, sorted_indices = torch.sort(warped, descending=False, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_remove = cumulative_probs <= (1.0 - float(top_p))
            sorted_remove[..., -1:] = False
            remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
            warped = warped.masked_fill(remove, -torch.inf)
        return warped

    @classmethod
    def _sample_dart_next_token(
        cls,
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the sampled token and its behavior-policy log-probability."""

        if logits.ndim != 2:
            raise ValueError(f"DART next-token logits must be [batch, vocab], got {tuple(logits.shape)}")
        if do_sample:
            behavior_logits = cls._warp_dart_sampling_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            behavior_log_probs = torch.log_softmax(behavior_logits, dim=-1)
            next_token = torch.multinomial(behavior_log_probs.exp(), num_samples=1)
        else:
            behavior_log_probs = torch.log_softmax(logits.float(), dim=-1)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        sampled_log_prob = torch.gather(behavior_log_probs, dim=-1, index=next_token).squeeze(-1)
        return next_token, sampled_log_prob

    @staticmethod
    def _reorder_cached_batch(cache, indices: torch.Tensor):
        """Select/repeat every Qwen hybrid-cache row without changing time state.

        Qwen3.5 mixes ordinary attention cache layers with recurrent linear-
        attention layers.  Selecting only ``key_cache``/``value_cache`` would
        silently leave the recurrent and convolution states misaligned.  The
        Transformers cache/layer ``reorder_cache`` API covers both state types;
        fail closed when an installed version cannot provide that contract.
        """

        if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError("Cache reorder indices must be a rank-1 torch.long tensor")
        if indices.numel() <= 0:
            raise ValueError("Cache reorder cannot produce an empty active batch")
        sequence_length = int(cache.get_seq_length())
        reorder = getattr(cache, "reorder_cache", None)
        if not callable(reorder):
            raise RuntimeError(
                "Active Qwen3.5 cache compaction requires cache.reorder_cache; "
                "the installed Transformers cache API is incompatible"
            )
        reordered = reorder(indices)
        if reordered is not None:
            cache = reordered
        if int(cache.get_seq_length()) != sequence_length:
            raise RuntimeError("Cache batch reorder unexpectedly changed its sequence length")
        return cache

    @staticmethod
    def _index_position_batch(value: torch.Tensor, indices: torch.Tensor, previous_batch_size: int) -> torch.Tensor:
        """Index the batch axis of ordinary or Qwen M-RoPE position tensors."""

        if value.ndim == 3 and value.shape[0] in {3, 4} and value.shape[1] == previous_batch_size:
            return value.index_select(1, indices)
        if value.ndim >= 1 and value.shape[0] == previous_batch_size:
            return value.index_select(0, indices)
        raise RuntimeError(
            "Cannot identify position-id batch axis during cached decode: "
            f"shape={tuple(value.shape)}, batch={previous_batch_size}"
        )

    @staticmethod
    def _multi_modal_entry_equal(left, right) -> bool:
        """Exact recursive equality used before sharing a multimodal prefix."""

        left = getattr(left, "data", left)
        right = getattr(right, "data", right)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return left.keys() == right.keys() and all(
                HFRollout._multi_modal_entry_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            return len(left) == len(right) and all(
                HFRollout._multi_modal_entry_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        return type(left) is type(right) and left == right

    def _shared_prefix_plan(
        self,
        prompts: DataProto,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        compression_query_mask: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor]:
        """Return physical prefill rows and the exact logical branch mapping."""

        if not bool(self.config.get("hf_share_rollout_prefix", False)) or len(prompts) <= 1:
            indices = list(range(len(prompts)))
            return indices, torch.arange(len(prompts), dtype=torch.long)
        if not bool(self.config.get("hf_use_replicated_module", False)) or isinstance(self.module, FSDP):
            raise RuntimeError("Shared rollout prefixes require the independent native HF replica")
        non_tensor = prompts.non_tensor_batch
        if non_tensor is None or "uid" not in non_tensor or "multi_modal_inputs" not in non_tensor:
            indices = list(range(len(prompts)))
            return indices, torch.arange(len(prompts), dtype=torch.long)
        uids = list(non_tensor["uid"])
        entries = list(non_tensor["multi_modal_inputs"])
        if len(uids) != len(prompts) or len(entries) != len(prompts):
            raise RuntimeError("Shared-prefix multimodal inputs are not batch aligned")
        for tensor, label in (
            (input_ids, "input_ids"),
            (attention_mask, "attention_mask"),
        ):
            if tensor.shape[0] != len(prompts):
                raise RuntimeError(f"Shared-prefix {label} is not batch aligned")
        if compression_query_mask is not None and compression_query_mask.shape != input_ids.shape:
            raise RuntimeError("Shared-prefix compression_query_mask is not prompt aligned")
        position_batch = (
            int(position_ids.shape[0])
            if position_ids.shape[0] == len(prompts)
            else int(position_ids.shape[1])
            if position_ids.ndim == 3 and position_ids.shape[0] in {3, 4}
            else -1
        )
        if position_batch != len(prompts):
            raise RuntimeError("Shared-prefix position_ids is not batch aligned")

        unique_indices: list[int] = []
        uid_to_prefill: dict[str, int] = {}
        branches: list[int] = []
        for row, uid_value in enumerate(uids):
            uid = str(uid_value)
            if uid not in uid_to_prefill:
                uid_to_prefill[uid] = len(unique_indices)
                unique_indices.append(row)
            physical = uid_to_prefill[uid]
            reference_row = unique_indices[physical]
            if not torch.equal(input_ids[row], input_ids[reference_row]):
                raise RuntimeError("Same-UID rollout group has non-identical input_ids")
            if not torch.equal(attention_mask[row], attention_mask[reference_row]):
                raise RuntimeError("Same-UID rollout group has non-identical attention_mask")
            if compression_query_mask is not None and not torch.equal(
                compression_query_mask[row], compression_query_mask[reference_row]
            ):
                raise RuntimeError("Same-UID rollout group has non-identical compression query masks")
            if position_ids.ndim == 3 and position_ids.shape[0] in {3, 4}:
                positions_equal = torch.equal(position_ids[:, row], position_ids[:, reference_row])
            else:
                positions_equal = torch.equal(position_ids[row], position_ids[reference_row])
            if not positions_equal:
                raise RuntimeError("Same-UID rollout group has non-identical position_ids")
            if not self._multi_modal_entry_equal(entries[row], entries[reference_row]):
                raise RuntimeError("Same-UID rollout group has non-identical multimodal inputs")
            branches.append(physical)
        return unique_indices, torch.tensor(branches, dtype=torch.long)

    def _generate_dart_cached(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        multi_modal_inputs: dict,
        response_length: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id,
        pad_token_id: int,
        compression_query_mask: torch.Tensor | None = None,
        prefix_branch_indices: torch.Tensor | None = None,
        semantic_stop_criteria: WellFormedAnswerTagCriteria | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate with one compressed prefill followed by cached token decode.

        FSDP-backed ranks execute the same number of model calls.  The audited
        native rollout replica contains no distributed forward collectives, so
        it stops at the local batch EOS and avoids a scalar all-reduce plus
        dummy cache forward on every generated token.
        """

        prefill_batch_size = int(input_ids.shape[0])
        if prefill_batch_size <= 0:
            raise ValueError("Controlled DART cached generation requires a non-empty batch")
        if response_length <= 0:
            raise ValueError(f"DART response_length must be positive, got {response_length}")
        if prefix_branch_indices is not None:
            if (
                not isinstance(prefix_branch_indices, torch.Tensor)
                or prefix_branch_indices.ndim != 1
                or prefix_branch_indices.dtype != torch.long
                or prefix_branch_indices.numel() <= 0
            ):
                raise TypeError("prefix_branch_indices must be a non-empty rank-1 torch.long tensor")
            if int(prefix_branch_indices.min().item()) < 0 or int(prefix_branch_indices.max().item()) >= prefill_batch_size:
                raise ValueError("prefix_branch_indices refers outside the physical prefill batch")

        outputs = self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compression_query_mask=compression_query_mask,
            visual_compression_mode="merge",
            **multi_modal_inputs,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        cache_attention_mask = outputs.dart_attention_mask
        cache_position_ids = outputs.dart_position_ids
        if cache is None:
            raise RuntimeError("DART prefill did not return past_key_values")
        if cache_attention_mask is None or cache_position_ids is None:
            raise RuntimeError("DART prefill did not return its compressed mask and position ids")
        if cache_attention_mask.ndim != 2 or cache_attention_mask.shape[0] != prefill_batch_size:
            raise RuntimeError(
                "DART compressed attention mask must be [batch, sequence], got "
                f"{tuple(cache_attention_mask.shape)}"
            )
        cache_length = int(cache.get_seq_length())
        if cache_length != cache_attention_mask.shape[-1]:
            raise RuntimeError(
                "DART prefill cache/mask length mismatch: "
                f"cache={cache_length}, mask={cache_attention_mask.shape[-1]}"
            )
        if outputs.logits.shape[:2] != (prefill_batch_size, 1):
            raise RuntimeError(f"DART prefill must return one next-token logit row, got {tuple(outputs.logits.shape)}")

        next_logits = outputs.logits[:, -1, :]
        last_position = cache_position_ids[..., -1:]
        if prefix_branch_indices is not None:
            repeat_indices = prefix_branch_indices.to(device=input_ids.device)
            cache = self._reorder_cached_batch(cache, repeat_indices)
            cache_attention_mask = cache_attention_mask.index_select(0, repeat_indices)
            last_position = self._index_position_batch(
                last_position, repeat_indices, previous_batch_size=prefill_batch_size
            )
            next_logits = next_logits.index_select(0, repeat_indices)
            input_ids = input_ids.index_select(0, repeat_indices)

        batch_size = int(input_ids.shape[0])
        response_tokens = torch.full(
            (batch_size, response_length),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        sampled_log_probs = torch.zeros(
            (batch_size, response_length), dtype=torch.float32, device=input_ids.device
        )
        sampled_action_mask = torch.zeros(
            (batch_size, response_length), dtype=torch.bool, device=input_ids.device
        )
        stop_reason_codes = torch.zeros((batch_size,), dtype=torch.long, device=input_ids.device)
        if semantic_stop_criteria is not None:
            semantic_stop_criteria.reset()
        active_original_indices = torch.arange(batch_size, dtype=torch.long, device=input_ids.device)
        local_finished = torch.zeros((batch_size,), dtype=torch.bool, device=input_ids.device)
        eos_ids = None
        if eos_token_id is not None:
            eos_ids = torch.as_tensor(eos_token_id, device=input_ids.device, dtype=input_ids.dtype).reshape(-1)
            if eos_ids.numel() == 0:
                eos_ids = None
        rollout_config = getattr(self, "config", {})
        replicated_module = bool(
            rollout_config.get("hf_use_replicated_module", False)
            if hasattr(rollout_config, "get")
            else getattr(rollout_config, "hf_use_replicated_module", False)
        )
        compact_active = bool(
            rollout_config.get("hf_active_sequence_compaction", False)
            if hasattr(rollout_config, "get")
            else getattr(rollout_config, "hf_active_sequence_compaction", False)
        )
        if compact_active and (not replicated_module or isinstance(self.module, FSDP)):
            raise RuntimeError("Active cached-decode compaction requires an independent HF replica")

        initial_cache_length = int(cache_attention_mask.shape[-1])
        mask_storage = torch.zeros(
            (batch_size, initial_cache_length + response_length),
            dtype=cache_attention_mask.dtype,
            device=cache_attention_mask.device,
        )
        mask_storage[:, :initial_cache_length] = cache_attention_mask

        for step in range(response_length):
            current_batch_size = int(next_logits.shape[0])
            if compact_active:
                active_indices = torch.arange(current_batch_size, dtype=torch.long, device=input_ids.device)
            else:
                active_indices = torch.nonzero(~local_finished, as_tuple=False).flatten()
            next_token = torch.full(
                (current_batch_size, 1), int(pad_token_id), dtype=input_ids.dtype, device=input_ids.device
            )
            current_log_prob = torch.zeros(
                (current_batch_size,), dtype=torch.float32, device=input_ids.device
            )
            if active_indices.numel():
                active_token, active_log_prob = self._sample_dart_next_token(
                    next_logits.index_select(0, active_indices),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                next_token.index_copy_(0, active_indices, active_token.to(next_token.dtype))
                current_log_prob.index_copy_(0, active_indices, active_log_prob)
                if eos_ids is not None:
                    active_finished = torch.isin(active_token.reshape(-1), eos_ids)
                    local_finished[active_indices] = active_finished
            response_tokens[active_original_indices, step] = next_token.reshape(-1)
            sampled_log_probs[active_original_indices, step] = current_log_prob
            if active_indices.numel():
                sampled_original_indices = active_original_indices.index_select(0, active_indices)
                sampled_action_mask[sampled_original_indices, step] = True
            if eos_ids is not None and active_indices.numel():
                sampled_original_indices = active_original_indices.index_select(0, active_indices)
                previous_reasons = stop_reason_codes.index_select(0, sampled_original_indices)
                eos_reasons = torch.full_like(previous_reasons, STOP_REASON_TOKEN_EOS)
                # A boolean advanced index lowers through aten::nonzero and
                # synchronizes CUDA.  Keep the update fixed-shape so the
                # semantic path has only its intentional all-finished copy.
                stop_reason_codes.index_copy_(
                    0,
                    sampled_original_indices,
                    torch.where(active_finished, eos_reasons, previous_reasons),
                )
            if semantic_stop_criteria is not None:
                semantic_finished, close_candidates = semantic_stop_criteria.update_candidates(
                    response_tokens[:, : step + 1]
                )
                semantic_current = semantic_finished.index_select(0, active_original_indices)
                local_finished |= semantic_current
                current_candidates = close_candidates.index_select(0, active_original_indices)
                # Fold semantic candidate bits and the active->original map
                # into the same host synchronization that this loop already
                # needs for its all-finished decision. Exact decoding then
                # transfers response rows only on the rare close-hit steps.
                current_count = int(local_finished.numel())
                synchronized = torch.cat(
                    (
                        local_finished.to(dtype=torch.int64),
                        current_candidates.to(dtype=torch.int64),
                        active_original_indices.to(dtype=torch.int64),
                    )
                ).detach().cpu().tolist()
                local_finished_cpu = [bool(value) for value in synchronized[:current_count]]
                candidate_cpu = [
                    bool(value)
                    for value in synchronized[current_count : 2 * current_count]
                ]
                original_cpu = [
                    int(value) for value in synchronized[2 * current_count :]
                ]
                candidate_rows = sorted(
                    original
                    for original, candidate in zip(original_cpu, candidate_cpu, strict=True)
                    if candidate
                )
                confirmed_rows = semantic_stop_criteria.confirm_candidates(
                    response_tokens[:, : step + 1],
                    candidate_rows,
                )
                if confirmed_rows:
                    confirmed_tensor = torch.tensor(
                        confirmed_rows,
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    stop_reason_codes[confirmed_tensor] = STOP_REASON_ANSWER_TAG
                    confirmed_set = set(confirmed_rows)
                    for index, original in enumerate(original_cpu):
                        if original in confirmed_set:
                            local_finished_cpu[index] = True
                semantic_finished = semantic_stop_criteria.finished_lengths > 0
                local_finished |= semantic_finished.index_select(0, active_original_indices)
                local_batch_finished = all(local_finished_cpu)
            else:
                local_batch_finished = bool(local_finished.all().item())
            all_ranks_finished = local_batch_finished
            if (
                not replicated_module
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                finished_flag = torch.tensor(
                    int(local_batch_finished),
                    dtype=torch.int32,
                    device=input_ids.device,
                )
                torch.distributed.all_reduce(finished_flag, op=torch.distributed.ReduceOp.MIN)
                all_ranks_finished = bool(finished_flag.item())

            if step + 1 == response_length or all_ranks_finished:
                break

            if compact_active:
                if semantic_stop_criteria is not None:
                    # The semantic path already copied local_finished above to
                    # make the exact all-finished decision.  Reuse that result
                    # instead of issuing a second CUDA nonzero synchronization.
                    survivor_indices = torch.tensor(
                        [index for index, finished in enumerate(local_finished_cpu) if not finished],
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                else:
                    survivor_indices = torch.nonzero(~local_finished, as_tuple=False).flatten()
                if survivor_indices.numel() == 0:
                    break
                previous_batch_size = current_batch_size
                cache = self._reorder_cached_batch(cache, survivor_indices)
                mask_storage = mask_storage.index_select(0, survivor_indices)
                last_position = self._index_position_batch(
                    last_position, survivor_indices, previous_batch_size=previous_batch_size
                )
                next_token = next_token.index_select(0, survivor_indices)
                active_original_indices = active_original_indices.index_select(0, survivor_indices)
                local_finished = torch.zeros(
                    (int(survivor_indices.numel()),), dtype=torch.bool, device=input_ids.device
                )

            mask_storage[:, initial_cache_length + step] = 1
            cache_attention_mask = mask_storage[:, : initial_cache_length + step + 1]
            decode_position_ids = last_position + (step + 1)
            outputs = self.module(
                input_ids=next_token,
                attention_mask=cache_attention_mask,
                position_ids=decode_position_ids,
                past_key_values=cache,
                # Cached decode remains the same merged view even though pixel
                # tensors and routes are consumed only by the prefill call.
                visual_compression_mode="merge",
                use_cache=True,
                logits_to_keep=1,
            )
            cache = outputs.past_key_values
            if cache is None:
                raise RuntimeError(f"DART decode step {step + 1} did not return past_key_values")
            expected_cache_length = cache_attention_mask.shape[-1]
            actual_cache_length = int(cache.get_seq_length())
            if actual_cache_length != expected_cache_length:
                raise RuntimeError(
                    f"DART decode cache length mismatch at step {step + 1}: "
                    f"cache={actual_cache_length}, mask={expected_cache_length}"
                )
            if outputs.logits.shape[:2] != (next_token.shape[0], 1):
                raise RuntimeError(
                    f"DART decode step {step + 1} must return one logit row, got {tuple(outputs.logits.shape)}"
                )
            next_logits = outputs.logits[:, -1, :]

        sequence = torch.cat([input_ids, response_tokens], dim=-1)
        sampled_log_probs = sampled_log_probs.masked_fill(~sampled_action_mask, 0.0)
        return sequence, sampled_log_probs, sampled_action_mask, stop_reason_codes

    def _get_max_prompt_length(self, prompts: DataProto):
        """
        Try to get fixed max prompt length.

        Fixed prompt length is important because DataProto.concat requires
        tensors from different minibatches to have same shape.
        """
        max_prompt_length = None

        try:
            max_prompt_length = self.config.get("prompt_length", None)
        except Exception:
            pass

        if max_prompt_length is None:
            try:
                max_prompt_length = self.config.get("max_prompt_length", None)
            except Exception:
                pass

        if max_prompt_length is None:
            max_prompt_length = prompts.meta_info.get("prompt_length", None)

        if max_prompt_length is None:
            max_prompt_length = prompts.meta_info.get("max_prompt_length", None)

        return max_prompt_length

    def _get_prompt_key_from_non_tensor_batch(self, prompts: DataProto):
        """
        Different verl versions / datasets may use different keys.

        Your current batch uses:
            non_tensor_batch["prompt"]
        """
        if prompts.non_tensor_batch is None:
            return None

        for key in ["raw_prompt", "prompt", "prompts"]:
            if key in prompts.non_tensor_batch:
                return key

        return None

    def _normalize_raw_prompt_to_text(self, raw_prompt):
        """
        Convert one prompt item into text.

        Supported formats:
        - str
        - dict, e.g. {"role": "user", "content": "..."}
        - list[dict], e.g. [{"role": "user", "content": "..."}]
        - numpy object array containing the above
        """

        if isinstance(raw_prompt, str):
            return raw_prompt

        if hasattr(raw_prompt, "tolist"):
            raw_prompt = raw_prompt.tolist()

        if isinstance(raw_prompt, tuple):
            raw_prompt = list(raw_prompt)

        # Single message dict.
        if isinstance(raw_prompt, dict):
            if "role" in raw_prompt and "content" in raw_prompt:
                raw_prompt = [raw_prompt]
            else:
                raise TypeError(
                    f"Unsupported prompt dict format: {raw_prompt}"
                )

        # Sometimes it may be wrapped as [[{"role": ...}]]
        if (
            isinstance(raw_prompt, list)
            and len(raw_prompt) == 1
            and isinstance(raw_prompt[0], list)
        ):
            raw_prompt = raw_prompt[0]

        # Chat format.
        if isinstance(raw_prompt, list):
            if self.tokenizer.chat_template is None:
                raise ValueError(
                    "Prompt is a chat-message list, but tokenizer.chat_template is not set. "
                    "Please set tokenizer.chat_template, or convert prompt to plain string."
                )

            return self.tokenizer.apply_chat_template(
                raw_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

        raise TypeError(
            f"Unsupported prompt type: {type(raw_prompt)}. "
            f"prompt={raw_prompt}"
        )

    def _encode_query_boundary(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("Query-boundary tokenizer unexpectedly returned a batch")
            ids = ids[0]
        return [int(token_id) for token_id in ids]

    @staticmethod
    def _find_subsequence(sequence: list[int], needle: list[int], start: int = 0) -> int:
        if not needle:
            raise ValueError("Cannot locate an empty query boundary")
        stop = len(sequence) - len(needle) + 1
        for offset in range(start, max(start, stop)):
            if sequence[offset : offset + len(needle)] == needle:
                return offset
        return -1

    def _resolve_image_token_id(self) -> int:
        """Resolve the image placeholder ID from the actual HF model config.

        ``BaseRollout`` stores a :class:`HFModelConfig` wrapper in
        ``self.model_config``.  Qwen special-token IDs live in the wrapper's
        ``hf_config`` (and on the instantiated route owner's ``config``), not
        on the wrapper itself.  Collect every available authoritative value
        and fail closed if the loaded model and wrapper disagree.
        """

        candidates: dict[str, int] = {}

        hf_config = getattr(self.model_config, "hf_config", None)
        hf_value = getattr(hf_config, "image_token_id", None)
        if hf_value is not None:
            candidates["model_config.hf_config"] = int(hf_value)

        # Keep the direct field as a backwards-compatible test/config fallback,
        # but never prefer it over the actual Transformers configuration.
        wrapper_value = getattr(self.model_config, "image_token_id", None)
        if wrapper_value is not None:
            candidates["model_config"] = int(wrapper_value)

        if hasattr(self, "module"):
            owner = self._get_dart_route_owner()
            owner_value = getattr(getattr(owner, "config", None), "image_token_id", None)
            if owner_value is not None:
                candidates["dart_route_owner.config"] = int(owner_value)

        if not candidates or any(value < 0 for value in candidates.values()):
            raise ValueError(
                "Conditional visual pruning requires a non-negative image_token_id "
                "on model_config.hf_config or the DART route owner's config"
            )
        unique_values = set(candidates.values())
        if len(unique_values) != 1:
            raise ValueError(f"Conflicting image_token_id values across runtime configs: {candidates}")
        return next(iter(unique_values))

    @staticmethod
    def _sample_non_tensor_value(prompts: DataProto, key: str, sample_idx: int, batch_size: int):
        """Return one sample-aligned non-tensor value without unsafe broadcasting."""

        non_tensor_batch = prompts.non_tensor_batch or {}
        if key not in non_tensor_batch:
            return None
        values = non_tensor_batch[key]
        if isinstance(values, np.ndarray):
            if values.ndim == 0:
                if batch_size != 1:
                    raise ValueError(f"Scalar non-tensor field {key!r} cannot describe batch_size={batch_size}")
                return values.item()
            if values.shape[0] != batch_size:
                raise ValueError(
                    f"Non-tensor field {key!r} has {values.shape[0]} rows for batch_size={batch_size}"
                )
            value = values[sample_idx]
            return value.item() if isinstance(value, np.generic) else value
        if isinstance(values, (list, tuple)):
            if len(values) != batch_size:
                if batch_size == 1:
                    return values
                raise ValueError(f"Non-tensor field {key!r} is not sample-aligned")
            return values[sample_idx]
        if batch_size != 1:
            raise ValueError(f"Non-tensor field {key!r} must be sample-aligned for batch_size={batch_size}")
        return values

    def _resolve_sample_route_query(
        self,
        *,
        prompts: DataProto,
        sample_idx: int,
        batch_size: int,
        normalized_prompt,
    ) -> tuple[RouteQuerySpec, str]:
        """Resolve an explicit semantic query, then use strict legacy fallbacks."""

        explicit = self._sample_non_tensor_value(prompts, "route_query", sample_idx, batch_size)
        if explicit is not None:
            return parse_route_query_value(explicit), "route_query"

        extra_info = self._sample_non_tensor_value(prompts, "extra_info", sample_idx, batch_size)
        if hasattr(extra_info, "tolist") and not isinstance(extra_info, Mapping):
            extra_info = extra_info.tolist()
        if isinstance(extra_info, Mapping):
            nested = extra_info.get("route_query")
            if nested is not None:
                return parse_route_query_value(nested), "extra_info.route_query"
            question = extra_info.get("question")
            options = extra_info.get("options")
            if isinstance(question, str) and question.strip():
                if options is not None:
                    return (
                        parse_route_query_value({"question": question, "options": options}),
                        "extra_info.question+options",
                    )
                try:
                    return parse_route_query_text(question), "extra_info.question"
                except ValueError:
                    # Older datasets sometimes store only the stem here.  The
                    # complete structured prompt remains a valid fallback, but
                    # it must independently parse as one exact A-D MCQ below.
                    pass

        fallback_text = user_text_for_prompt_fallback(normalized_prompt)
        return parse_route_query_text(fallback_text, prompt_fallback=True), "prompt_fallback"

    def _locate_user_content_token_spans(
        self, *, sample_input_ids: torch.Tensor, normalized_prompt
    ) -> list[tuple[int, int]]:
        if not isinstance(normalized_prompt, list):
            raise ValueError("Conditional visual pruning requires structured chat messages")
        user_count = sum(
            int(isinstance(message, dict) and message.get("role") == "user")
            for message in normalized_prompt
        )
        if user_count <= 0:
            raise ValueError("Conditional visual pruning requires at least one user message")
        if any(isinstance(message, dict) and message.get("role") == "assistant" for message in normalized_prompt):
            raise ValueError("Rollout prefill must not contain an assistant response when creating a route")

        ids = [int(token_id) for token_id in sample_input_ids.squeeze(0).detach().cpu().tolist()]
        user_start = self._encode_query_boundary("<|im_start|>user\n")
        message_end = self._encode_query_boundary("<|im_end|>")
        spans: list[tuple[int, int]] = []
        cursor = 0
        for located in range(user_count):
            start = self._find_subsequence(ids, user_start, cursor)
            if start < 0:
                raise ValueError(
                    f"Could not locate all user spans in the rendered prompt: expected={user_count}, found={located}"
                )
            content_start = start + len(user_start)
            end = self._find_subsequence(ids, message_end, content_start)
            if end < 0:
                raise ValueError("Rendered user message has no <|im_end|> boundary")
            spans.append((content_start, end))
            cursor = end + len(message_end)
        return spans

    @staticmethod
    def _encoding_field_as_list(encoded, key: str) -> list:
        value = encoded[key] if isinstance(encoded, Mapping) else getattr(encoded, key)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if value and isinstance(value[0], list) and key == "input_ids":
            if len(value) != 1:
                raise ValueError("Rendered prompt tokenizer unexpectedly returned a batch")
            value = value[0]
        if (
            value
            and isinstance(value[0], list)
            and key == "offset_mapping"
            and value[0]
            and isinstance(value[0][0], (list, tuple))
        ):
            if len(value) != 1:
                raise ValueError("Rendered prompt offsets unexpectedly returned a batch")
            value = value[0]
        return list(value)

    @staticmethod
    def _collapse_expanded_image_tokens(ids: list[int], image_token_id: int) -> tuple[list[int], list[tuple[int, ...]]]:
        collapsed_ids: list[int] = []
        source_positions: list[tuple[int, ...]] = []
        cursor = 0
        while cursor < len(ids):
            token_id = int(ids[cursor])
            end = cursor + 1
            if token_id == image_token_id:
                while end < len(ids) and int(ids[end]) == image_token_id:
                    end += 1
            collapsed_ids.append(token_id)
            source_positions.append(tuple(range(cursor, end)))
            cursor = end
        return collapsed_ids, source_positions

    @staticmethod
    def _all_subsequence_offsets(sequence: list[int], needle: list[int]) -> list[int]:
        if not needle or len(needle) > len(sequence):
            return []
        return [
            offset
            for offset in range(len(sequence) - len(needle) + 1)
            if sequence[offset : offset + len(needle)] == needle
        ]

    def _build_user_query_mask(
        self,
        *,
        sample_input_ids: torch.Tensor,
        normalized_prompt,
        rendered_text: str,
        route_query_spec: RouteQuerySpec,
        query_source: str,
    ) -> tuple[torch.Tensor, dict]:
        """Map only the exact question and source-option text onto processed IDs.

        Fast-tokenizer character offsets are aligned to the processor output.
        The only permitted alignment difference is Qwen's expansion of one
        rendered image placeholder into a run of image tokens.  Any ambiguous
        text occurrence or other tokenizer/processor drift fails closed.
        """

        if sample_input_ids.ndim != 2 or sample_input_ids.shape[0] != 1:
            raise ValueError("Query-mask construction expects input_ids with shape [1, sequence]")
        if not isinstance(rendered_text, str) or not rendered_text:
            raise ValueError("Conditional visual pruning requires the exact rendered chat text")
        user_spans = self._locate_user_content_token_spans(
            sample_input_ids=sample_input_ids,
            normalized_prompt=normalized_prompt,
        )
        try:
            rendered_encoding = self.tokenizer(
                rendered_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (TypeError, NotImplementedError) as exc:
            raise ValueError("Conditional routing requires a fast tokenizer with offset_mapping") from exc
        if "offset_mapping" not in rendered_encoding:
            raise ValueError("Conditional routing requires tokenizer offset_mapping")
        rendered_ids = [int(value) for value in self._encoding_field_as_list(rendered_encoding, "input_ids")]
        offsets = [
            tuple(int(part) for part in pair)
            for pair in self._encoding_field_as_list(rendered_encoding, "offset_mapping")
        ]
        if len(rendered_ids) != len(offsets):
            raise ValueError("Rendered prompt token IDs and offsets have different lengths")

        image_token_id = self._resolve_image_token_id()
        sample_ids = [int(value) for value in sample_input_ids.squeeze(0).detach().cpu().tolist()]
        collapsed_ids, collapsed_positions = self._collapse_expanded_image_tokens(sample_ids, image_token_id)
        alignments = self._all_subsequence_offsets(collapsed_ids, rendered_ids)
        if len(alignments) != 1:
            raise ValueError(
                "Rendered prompt does not align uniquely with processed input IDs after image-token expansion: "
                f"matches={len(alignments)}"
            )
        alignment = alignments[0]
        rendered_to_sample = collapsed_positions[alignment : alignment + len(rendered_ids)]
        special_ids = {int(value) for value in (getattr(self.tokenizer, "all_special_ids", []) or [])}

        def is_user_position(position: int) -> bool:
            return any(start <= position < end for start, end in user_spans)

        semantic_character_spans: list[list[int]] = []
        per_segment_positions: list[list[int]] = []
        previous_end = -1
        for segment in route_query_spec.segments:
            occurrences: list[int] = []
            cursor = 0
            while True:
                occurrence = rendered_text.find(segment, cursor)
                if occurrence < 0:
                    break
                occurrences.append(occurrence)
                cursor = occurrence + 1

            valid_candidates: list[tuple[int, int, list[int]]] = []
            for occurrence in occurrences:
                segment_end = occurrence + len(segment)
                # The materialized query segments are ordered exactly as they
                # appear in the rendered source question.  Ignore an earlier
                # incidental occurrence (for example an option phrase quoted
                # in the question stem), while still failing closed if two
                # later occurrences remain ambiguous.
                if occurrence < previous_end:
                    continue
                expanded_start = occurrence
                expanded_end = segment_end
                while expanded_start > 0 and rendered_text[expanded_start - 1].isspace():
                    expanded_start -= 1
                while expanded_end < len(rendered_text) and rendered_text[expanded_end].isspace():
                    expanded_end += 1
                rendered_indices = [
                    index
                    for index, (start, end) in enumerate(offsets)
                    if end > start and start >= expanded_start and end <= expanded_end
                ]
                if not rendered_indices:
                    continue
                positions = sorted(
                    {
                        position
                        for index in rendered_indices
                        for position in rendered_to_sample[index]
                        if sample_ids[position] != image_token_id
                        and sample_ids[position] not in special_ids
                        and is_user_position(position)
                    }
                )
                if not positions:
                    continue
                mapped_all = [position for index in rendered_indices for position in rendered_to_sample[index]]
                if not all(is_user_position(position) for position in mapped_all):
                    continue
                selected_offsets = [offsets[index] for index in rendered_indices if any(
                    position in positions for position in rendered_to_sample[index]
                )]
                if not selected_offsets:
                    continue
                if min(start for start, _ in selected_offsets) > occurrence:
                    continue
                if max(end for _, end in selected_offsets) < segment_end:
                    continue
                valid_candidates.append((occurrence, segment_end, positions))

            if len(valid_candidates) != 1:
                raise ValueError(
                    f"Route-query segment does not occur uniquely inside user text: {segment!r}, "
                    f"valid_matches={len(valid_candidates)}"
                )
            start, end, positions = valid_candidates[0]
            if start < previous_end:
                raise ValueError("Route-query segments are not ordered in the rendered user prompt")
            previous_end = end
            semantic_character_spans.append([start, end])
            per_segment_positions.append(positions)

        mask = torch.zeros_like(sample_input_ids, dtype=torch.bool)
        selected_positions = sorted({position for positions in per_segment_positions for position in positions})
        mask[0, selected_positions] = True
        if int(mask.sum().item()) <= 0:
            raise ValueError("Question/options query contains no selectable text tokens")
        if any(not is_user_position(position) for position in selected_positions):
            raise RuntimeError("Internal error: semantic query mask escaped the user message")
        if torch.any(mask & sample_input_ids.eq(image_token_id)).item():
            raise RuntimeError("Internal error: semantic query mask includes image placeholders")
        special_tensor = torch.as_tensor(
            sorted(special_ids),
            dtype=sample_input_ids.dtype,
            device=sample_input_ids.device,
        )
        if special_tensor.numel() and torch.any(mask & torch.isin(sample_input_ids, special_tensor)).item():
            raise RuntimeError("Internal error: semantic query mask includes tokenizer special IDs")

        audit = {
            "schema_version": route_query_spec.schema_version,
            "query_policy": ROUTE_QUERY_POLICY,
            "source": query_source,
            "canonical_text": route_query_spec.canonical_text,
            "canonical_sha256": route_query_spec.sha256,
            "segments": list(route_query_spec.segments),
            "semantic_character_spans": semantic_character_spans,
            "selected_token_count": len(selected_positions),
            "selected_token_indices": selected_positions,
            "selected_token_ids": [sample_ids[position] for position in selected_positions],
        }
        return mask, audit

    def _build_prompt_tensors_from_non_tensor_prompt(
        self,
        prompts: DataProto,
        pad_token_id: int,
    ):
        """
        Compatible with new verl dataset format:

        prompts.batch:
            {"dummy_tensor": ...}

        prompts.non_tensor_batch:
            {"prompt": ...}
            or {"raw_prompt": ...}
            or {"prompts": ...}

        Returns:
            input_ids
            attention_mask
            position_ids
            compression_query_mask
            compression_query_audits
        """

        prompt_key = self._get_prompt_key_from_non_tensor_batch(prompts)

        if prompt_key is None:
            non_tensor_keys = []
            if prompts.non_tensor_batch is not None:
                non_tensor_keys = list(prompts.non_tensor_batch.keys())

            raise KeyError(
                "HFRollout cannot find prompt in non_tensor_batch. "
                f"batch keys={list(prompts.batch.keys())}, "
                f"non_tensor_batch keys={non_tensor_keys}. "
                "Expected one of non_tensor_batch['raw_prompt'], "
                "non_tensor_batch['prompt'], or non_tensor_batch['prompts']."
            )

        raw_prompts = prompts.non_tensor_batch[prompt_key]
        max_prompt_length = self._get_max_prompt_length(prompts)

        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        compression_query_mask_list = []
        compression_query_audit_list = []
        multi_modal_inputs_list = []
        # ``DataProto.repeat(..., interleave=True)`` keeps the same Python
        # prompt object for the n trajectories belonging to one rollout uid.
        # Qwen image preprocessing is deterministic for the formal single-image
        # path, and repeating it materializes n identical (potentially very
        # large) CPU pixel tensors.  Cache only when both the immutable rollout
        # uid and the exact Python prompt object match.  If transport ever
        # reconstructs separate objects, this deliberately falls back to the
        # canonical per-sample path instead of relying on value heuristics.
        compressor_enabled = bool(
            getattr(self.model_config, "vision_token_compressor", {}).get("enabled", False)
        )
        rollout_uids = (
            prompts.non_tensor_batch.get("uid")
            if prompts.non_tensor_batch is not None
            else None
        )
        if self._uses_cdpruner() and (
            rollout_uids is None or len(rollout_uids) != len(raw_prompts)
        ):
            raise RuntimeError("V8 CDPruner requires one immutable rollout UID per prompt")
        deterministic_preprocess_cache: dict[tuple[str, int], tuple[str, object]] = {}

        raw_prompt_count = len(raw_prompts)
        for sample_idx, raw_prompt in enumerate(raw_prompts):
            normalized_prompt = raw_prompt.tolist() if hasattr(raw_prompt, "tolist") else raw_prompt
            if isinstance(normalized_prompt, tuple):
                normalized_prompt = list(normalized_prompt)
            if isinstance(normalized_prompt, dict):
                normalized_prompt = [normalized_prompt]
            if (
                isinstance(normalized_prompt, list)
                and len(normalized_prompt) == 1
                and isinstance(normalized_prompt[0], list)
            ):
                normalized_prompt = normalized_prompt[0]

            preprocess_cache_key = None
            if compressor_enabled and rollout_uids is not None and len(rollout_uids) == raw_prompt_count:
                preprocess_cache_key = (str(rollout_uids[sample_idx]), id(raw_prompt))
            cached_preprocess = (
                deterministic_preprocess_cache.get(preprocess_cache_key)
                if preprocess_cache_key is not None
                else None
            )

            if cached_preprocess is not None:
                text, processed = cached_preprocess
            elif self.processor is not None and isinstance(normalized_prompt, list):
                if compressor_enabled:
                    # DART student and full-token teacher must consume pixels
                    # produced by the same single-resize processor path.  In
                    # particular, qwen-vl-utils must not resize still images
                    # before the model processor sees them.
                    from verl.utils.multimodal_preprocessing import process_dart_messages_once

                    text, processed = process_dart_messages_once(self.processor, normalized_prompt)
                else:
                    from qwen_vl_utils import process_vision_info

                    text = self.processor.apply_chat_template(
                        normalized_prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    images, videos = process_vision_info(
                        normalized_prompt,
                        image_patch_size=14,
                        return_video_metadata=True,
                    )
                    if videos:
                        video_tensors, video_metadatas = zip(*videos, strict=True)
                        video_tensors = list(video_tensors)
                        video_metadatas = list(video_metadatas)
                    else:
                        video_tensors = None
                        video_metadatas = None
                    processed = self.processor(
                        text=[text],
                        images=images or None,
                        videos=video_tensors,
                        video_metadata=video_metadatas,
                        return_tensors="pt",
                        do_sample_frames=False,
                    )
            else:
                text = self._normalize_raw_prompt_to_text(normalized_prompt)
                processed = self.tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=False,
                )
            if preprocess_cache_key is not None and cached_preprocess is None:
                deterministic_preprocess_cache[preprocess_cache_key] = (text, processed)

            sample_input_ids = processed["input_ids"]
            sample_attention_mask = processed.get("attention_mask", torch.ones_like(sample_input_ids))
            if sample_input_ids.shape[0] != 1:
                raise ValueError("HF rollout expects each raw prompt to produce exactly one sequence")
            if max_prompt_length is not None and sample_input_ids.shape[-1] > int(max_prompt_length):
                raise ValueError(
                    f"HF rollout prompt has {sample_input_ids.shape[-1]} tokens, exceeding "
                    f"max_prompt_length={int(max_prompt_length)}; refusing silent truncation."
                )

            if compressor_enabled and not self._uses_holitom_dpc_spatial_merge():
                route_query_spec, query_source = self._resolve_sample_route_query(
                    prompts=prompts,
                    sample_idx=sample_idx,
                    batch_size=raw_prompt_count,
                    normalized_prompt=normalized_prompt,
                )
                sample_query_mask, sample_query_audit = self._build_user_query_mask(
                    sample_input_ids=sample_input_ids,
                    normalized_prompt=normalized_prompt,
                    rendered_text=text,
                    route_query_spec=route_query_spec,
                    query_source=query_source,
                )
                if self._uses_cdpruner():
                    sample_uid = str(rollout_uids[sample_idx])
                    if not sample_uid:
                        raise RuntimeError("V8 CDPruner rollout UID must be non-empty")
                    sample_query_audit["sample_uid"] = sample_uid
            else:
                sample_query_mask = torch.zeros_like(sample_attention_mask, dtype=torch.bool)
                sample_query_audit = None

            multi_modal_inputs = {
                key: value
                for key, value in dict(processed).items()
                if key not in {"input_ids", "attention_mask"} and value is not None
            }
            image_grid_thw = multi_modal_inputs.get("image_grid_thw")
            if image_grid_thw is not None:
                # Metadata used by the trainer's FLOP accounting.  Keep it in
                # the per-sample payload so repeat/sort/micro-batch operations
                # cannot detach it from its image, but never forward it to the
                # Hugging Face model (see _generate_minibatch below).
                multi_modal_inputs["images_seqlens"] = torch.repeat_interleave(
                    image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
                )
            position_kwargs = {
                "input_ids": sample_input_ids,
                "attention_mask": sample_attention_mask,
                "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
                "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
            }
            mm_token_type_ids = multi_modal_inputs.pop("mm_token_type_ids", None)
            if mm_token_type_ids is not None:
                from verl.models.transformers.vision_token_compressor import build_qwen3_5_position_ids

                sample_position_ids = build_qwen3_5_position_ids(
                    self.processor,
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=position_kwargs["image_grid_thw"],
                    video_grid_thw=position_kwargs["video_grid_thw"],
                ).transpose(0, 1)
            elif self.processor is not None and hasattr(self.processor, "get_rope_index"):
                sample_position_ids = self.processor.get_rope_index(**position_kwargs)
                if isinstance(sample_position_ids, tuple):
                    sample_position_ids = sample_position_ids[0]
                if sample_position_ids.dim() == 2:
                    sample_position_ids = sample_position_ids.unsqueeze(1)
                if sample_position_ids.dim() != 3 or sample_position_ids.shape[1] != 1:
                    raise ValueError(
                        "processor.get_rope_index must return [rope_dims, 1, seq], got "
                        f"{tuple(sample_position_ids.shape)}"
                    )
                sample_position_ids = sample_position_ids.transpose(0, 1)
            else:
                sample_position_ids = sample_attention_mask.long().cumsum(dim=-1) - 1
                sample_position_ids.masked_fill_(sample_attention_mask == 0, 0)

            input_ids_list.append(sample_input_ids.squeeze(0))
            attention_mask_list.append(sample_attention_mask.squeeze(0))
            position_ids_list.append(sample_position_ids.squeeze(0))
            compression_query_mask_list.append(sample_query_mask.squeeze(0))
            compression_query_audit_list.append(sample_query_audit)
            multi_modal_inputs_list.append(multi_modal_inputs or None)

        # max_prompt_length is a hard safety bound, not a padding target.  The
        # DART path decodes one sample at a time and pads completed outputs to
        # the real batch maximum in _left_pad_dart_outputs.
        target_length = max(item.numel() for item in input_ids_list)
        batch_size = len(input_ids_list)
        input_ids = torch.full((batch_size, target_length), pad_token_id, dtype=input_ids_list[0].dtype)
        attention_mask = torch.zeros((batch_size, target_length), dtype=attention_mask_list[0].dtype)
        compression_query_mask = torch.zeros((batch_size, target_length), dtype=torch.bool)
        rope_dims = position_ids_list[0].shape[0] if position_ids_list[0].dim() == 2 else None
        if any((item.dim() == 2) != (rope_dims is not None) for item in position_ids_list):
            raise ValueError("HF rollout batch mixes scalar and M-RoPE position tensors")
        if rope_dims is not None and any(item.shape[0] != rope_dims for item in position_ids_list):
            raise ValueError("HF rollout batch has inconsistent M-RoPE channel counts")
        position_ids = (
            torch.ones((batch_size, rope_dims, target_length), dtype=position_ids_list[0].dtype)
            if rope_dims is not None
            else torch.zeros((batch_size, target_length), dtype=position_ids_list[0].dtype)
        )
        for sample_idx, (ids, mask, positions, query_mask) in enumerate(
            zip(
                input_ids_list,
                attention_mask_list,
                position_ids_list,
                compression_query_mask_list,
                strict=True,
            )
        ):
            length = ids.numel()
            start = target_length - length
            input_ids[sample_idx, start:] = ids
            attention_mask[sample_idx, start:] = mask
            position_ids[sample_idx, ..., start:] = positions
            compression_query_mask[sample_idx, start:] = query_mask
            query_audit = compression_query_audit_list[sample_idx]
            if query_audit is not None:
                query_audit = copy.deepcopy(query_audit)
                query_audit["selected_token_indices"] = [
                    start + int(position) for position in query_audit["selected_token_indices"]
                ]
                query_audit["prefill_left_padding"] = start
                query_audit["prefill_prompt_length"] = target_length
                audited_ids = [
                    int(input_ids[sample_idx, position].item())
                    for position in query_audit["selected_token_indices"]
                ]
                if audited_ids != [int(value) for value in query_audit["selected_token_ids"]]:
                    raise RuntimeError("Left padding changed route-query token identity")
                compression_query_audit_list[sample_idx] = query_audit

        prompts.non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        device = self._get_module_device()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        compression_query_mask = compression_query_mask.to(device)

        return input_ids, attention_mask, position_ids, compression_query_mask, compression_query_audit_list

    def _validated_pretokenized_query_audits(
        self,
        *,
        prompts: DataProto,
        input_ids: torch.Tensor,
        compression_query_mask: torch.Tensor,
    ) -> list[dict]:
        """Validate producer-supplied audits for legacy pre-tokenized prompts.

        A bare mask is not enough to prove that boilerplate was excluded.  Old
        pre-tokenized callers remain supported only when they provide the exact
        semantic text, hash, token indices, and token IDs used to build it.
        """

        batch_size = int(input_ids.shape[0])
        audits: list[dict] = []
        image_token_id = self._resolve_image_token_id()
        special_ids = {int(value) for value in (getattr(self.tokenizer, "all_special_ids", []) or [])}
        for sample_idx in range(batch_size):
            raw_audit = self._sample_non_tensor_value(
                prompts,
                "compression_query_audit",
                sample_idx,
                batch_size,
            )
            if hasattr(raw_audit, "tolist") and not isinstance(raw_audit, Mapping):
                raw_audit = raw_audit.tolist()
            if not isinstance(raw_audit, Mapping):
                raise ValueError(
                    "Pre-tokenized conditional prompts require sample-aligned "
                    "non_tensor_batch['compression_query_audit']"
                )
            audit = dict(raw_audit)
            audit_schema = audit.get("schema_version")
            if audit_schema not in SUPPORTED_ROUTE_QUERY_SCHEMAS:
                raise ValueError("Pre-tokenized query audit has an unsupported schema_version")
            if audit.get("query_policy") != ROUTE_QUERY_POLICY:
                raise ValueError("Pre-tokenized query audit has an unsupported query_policy")
            spec = parse_route_query_value(
                {
                    "schema_version": audit_schema,
                    "policy": ROUTE_QUERY_POLICY,
                    "segments": audit.get("segments"),
                    "canonical_text": audit.get("canonical_text"),
                    "sha256": audit.get("canonical_sha256"),
                }
            )
            positions = [int(value) for value in audit.get("selected_token_indices", [])]
            expected_positions = torch.nonzero(
                compression_query_mask[sample_idx].to(torch.bool), as_tuple=False
            ).flatten().detach().cpu().tolist()
            if positions != expected_positions:
                raise ValueError("Pre-tokenized query audit indices do not match compression_query_mask")
            ids = [int(input_ids[sample_idx, position].item()) for position in positions]
            if ids != [int(value) for value in audit.get("selected_token_ids", [])]:
                raise ValueError("Pre-tokenized query audit token IDs do not match input_ids")
            if not positions or any(token_id == image_token_id or token_id in special_ids for token_id in ids):
                raise ValueError("Pre-tokenized semantic query is empty or contains image/special tokens")
            audit["canonical_text"] = spec.canonical_text
            audit["canonical_sha256"] = spec.sha256
            audit["selected_token_count"] = len(positions)
            audits.append(audit)
        return audits

    def _extract_prompt_tensors(self, prompts: DataProto):
        """
        Return:
            idx
            attention_mask
            position_ids
            compression_query_mask
            compression_query_audits
            eos_token_id
            pad_token_id
        """

        if prompts.batch is None:
            raise ValueError("HFRollout requires DataProto.batch not None.")

        eos_token_id = prompts.meta_info.get("eos_token_id", None)
        pad_token_id = prompts.meta_info.get("pad_token_id", None)

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            pad_token_id = eos_token_id

        compression_query_mask = None
        compression_query_audits = None

        # Case 1: old verl / already-tokenized dataset.
        if "input_ids" in prompts.batch:
            idx = prompts.batch["input_ids"]
            attention_mask = prompts.batch.get("attention_mask", None)
            position_ids = prompts.batch.get("position_ids", None)
            compression_query_mask = prompts.batch.get("compression_query_mask", None)

        elif "prompts" in prompts.batch:
            idx = prompts.batch["prompts"]
            attention_mask = prompts.batch.get("attention_mask", None)
            position_ids = prompts.batch.get("position_ids", None)
            compression_query_mask = prompts.batch.get("compression_query_mask", None)

        elif "prompt_ids" in prompts.batch:
            idx = prompts.batch["prompt_ids"]
            attention_mask = prompts.batch.get("attention_mask", None)
            position_ids = prompts.batch.get("position_ids", None)
            compression_query_mask = prompts.batch.get("compression_query_mask", None)

        # Case 2: new verl dataset format.
        # Your current case:
        #   batch keys = ['dummy_tensor']
        #   non_tensor_batch contains 'prompt'
        elif self._get_prompt_key_from_non_tensor_batch(prompts) is not None:
            (
                idx,
                attention_mask,
                position_ids,
                compression_query_mask,
                compression_query_audits,
            ) = self._build_prompt_tensors_from_non_tensor_prompt(
                prompts=prompts,
                pad_token_id=pad_token_id,
            )

        else:
            non_tensor_keys = []
            if prompts.non_tensor_batch is not None:
                non_tensor_keys = list(prompts.non_tensor_batch.keys())

            raise KeyError(
                "HFRollout cannot find prompt ids. "
                f"batch keys={list(prompts.batch.keys())}, "
                f"non_tensor_batch keys={non_tensor_keys}. "
                "Expected one of batch['input_ids'], batch['prompts'], batch['prompt_ids'], "
                "or non_tensor_batch['raw_prompt'/'prompt'/'prompts']."
            )

        device = self._get_module_device()

        idx = idx.to(device)

        if attention_mask is None:
            attention_mask = idx.ne(pad_token_id).long()
        else:
            attention_mask = attention_mask.to(device)

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        else:
            position_ids = position_ids.to(device)

        if compression_query_mask is not None:
            compression_query_mask = compression_query_mask.to(device=device, dtype=torch.bool)
            if compression_query_mask.shape != idx.shape:
                raise ValueError(
                    "compression_query_mask must match prompt input_ids, got "
                    f"{tuple(compression_query_mask.shape)} vs {tuple(idx.shape)}"
                )

        compressor_enabled = bool(getattr(self.model_config, "vision_token_compressor", {}).get("enabled", False))
        if (
            compressor_enabled
            and not self._uses_holitom_dpc_spatial_merge()
            and compression_query_mask is not None
            and compression_query_audits is None
        ):
            compression_query_audits = self._validated_pretokenized_query_audits(
                prompts=prompts,
                input_ids=idx,
                compression_query_mask=compression_query_mask,
            )

        return (
            idx,
            attention_mask,
            position_ids,
            compression_query_mask,
            compression_query_audits,
            eos_token_id,
            pad_token_id,
        )

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # Make sampling args can be overridden by inputs.
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))

        if not do_sample:
            # Greedy decoding.
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }

        elif is_validate:
            # Validation sampling.
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,
            }

        else:
            # Training rollout sampling.
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,
            }

        generation_config = GenerationConfig(**kwargs)

        (
            idx,
            attention_mask,
            position_ids,
            compression_query_mask,
            compression_query_audits,
            eos_token_id,
            pad_token_id,
        ) = self._extract_prompt_tensors(prompts)

        prompt_length = idx.size(1)
        semantic_stop_enabled = bool(self.config.get("semantic_stop_enabled", False))
        semantic_stop_criteria = None
        if semantic_stop_enabled:
            if self.config.get("semantic_stop_string", "</answer>") != "</answer>":
                raise ValueError("V7 semantic stop is pinned to the exact '</answer>' delimiter")
            semantic_stop_criteria = WellFormedAnswerTagCriteria(
                self.tokenizer,
                prompt_width=prompt_length,
                open_string="<answer>",
                close_string="</answer>",
                virtual_open=bool(self.config.get("semantic_stop_virtual_open", False)),
            )

        multi_modal_inputs = {}
        raw_multi_modal_inputs = None
        if prompts.non_tensor_batch is not None and "multi_modal_inputs" in prompts.non_tensor_batch:
            raw_multi_modal_inputs = prompts.non_tensor_batch["multi_modal_inputs"]
            multi_modal_inputs = extract_multi_modal_inputs(raw_multi_modal_inputs)
            multi_modal_inputs.pop("images_seqlens", None)

        generation_position_ids = position_ids.transpose(0, 1) if position_ids.dim() == 3 else position_ids

        compressor_enabled = bool(getattr(self.model_config, "vision_token_compressor", {}).get("enabled", False))
        holitom_dpc_merge = self._uses_holitom_dpc_spatial_merge()
        visual_compression_mode = prompts.meta_info.get("visual_compression_mode")
        if visual_compression_mode is None:
            # Preserve legacy DART rollout behavior.  Formal V6 callers may
            # select dense/no_image explicitly through per-request meta_info.
            visual_compression_mode = (
                "merge"
                if compressor_enabled
                else ("dense" if multi_modal_inputs else "no_image")
            )
        if visual_compression_mode not in VISUAL_COMPRESSION_MODES:
            raise ValueError(
                f"visual_compression_mode must be one of {sorted(VISUAL_COMPRESSION_MODES)}, "
                f"got {visual_compression_mode!r}"
            )
        if visual_compression_mode == "merge":
            if not compressor_enabled:
                raise RuntimeError("HF merge rollout requires an enabled visual compressor")
            if holitom_dpc_merge and not multi_modal_inputs:
                raise RuntimeError("Formal HoliTom-DPC merge rollout requires non-empty multimodal inputs")
        elif visual_compression_mode == "dense":
            if not multi_modal_inputs:
                raise RuntimeError("HF dense rollout requires non-empty multimodal inputs")
        merge_visual_tokens = visual_compression_mode == "merge"
        prefix_source_indices = list(range(len(prompts)))
        prefix_branch_indices: torch.Tensor | None = None
        if merge_visual_tokens:
            prefix_source_indices, prefix_branch_indices = self._shared_prefix_plan(
                prompts,
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                compression_query_mask=(
                    None if holitom_dpc_merge else compression_query_mask
                ),
            )
            if len(prefix_source_indices) < len(prompts):
                # ``extract_multi_modal_inputs`` concatenates flattened image
                # patches, so slice the original per-sample objects rather than
                # guessing a batch axis on the concatenated pixel tensor.
                multi_modal_inputs = extract_multi_modal_inputs(
                    raw_multi_modal_inputs, indices=prefix_source_indices
                )
                multi_modal_inputs.pop("images_seqlens", None)
        device = self._get_module_device()
        multi_modal_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in multi_modal_inputs.items()
        }
        # no_image keeps the exact prompt token stream (including image
        # placeholders) but withholds pixel tensors from the model under
        # same_prompt_image_placeholder_without_visual_replacement_v1.
        model_multi_modal_inputs = (
            {} if visual_compression_mode == "no_image" else multi_modal_inputs
        )
        if merge_visual_tokens:
            self._clear_and_collect_dart_routes(clear=True)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse=False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(
                self.module,
                writeback=False,
                recurse=False,
            )

        rollout_log_probs = None
        explicit_response_mask = None
        stop_reason_codes = None
        compact_padding = merge_visual_tokens and idx.shape[0] > 1
        with (
            param_ctx,
            torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16),
            self._dart_compact_padding(enabled=compact_padding)
            if merge_visual_tokens
            else contextlib.nullcontext(),
        ):
            if merge_visual_tokens:
                if semantic_stop_enabled:
                    semantic_stop_criteria = WellFormedAnswerTagCriteria(
                        self.tokenizer,
                        prompt_width=0,
                        open_string="<answer>",
                        close_string="</answer>",
                        virtual_open=bool(self.config.get("semantic_stop_virtual_open", False)),
                    )
                if not holitom_dpc_merge and compression_query_mask is None:
                    raise ValueError(
                        "Conditional visual pruning requires an explicit user question/options mask; "
                        "pre-tokenized prompts must provide batch['compression_query_mask']"
                    )
                if not holitom_dpc_merge:
                    if torch.any(compression_query_mask & ~attention_mask.to(torch.bool)).item():
                        raise ValueError("compression_query_mask includes padded prompt positions")
                    if torch.any(compression_query_mask.sum(dim=-1) <= 0).item():
                        raise ValueError("Every prompt must expose at least one user question/options token")
                # HF generate tracks masks/cache positions against the public,
                # uncompressed input_ids.  DART instead builds a shorter
                # physical cache, so use an explicit fixed-length loop whose
                # state is returned by the patched Qwen3.5 forward.
                prefix_index_tensor = torch.tensor(
                    prefix_source_indices, dtype=torch.long, device=idx.device
                )
                prefill_position_ids = self._index_position_batch(
                    generation_position_ids,
                    prefix_index_tensor,
                    previous_batch_size=len(prompts),
                )
                prefill_query_mask = None
                if not holitom_dpc_merge:
                    prefill_query_mask = compression_query_mask.index_select(
                        0, prefix_index_tensor.to(compression_query_mask.device)
                    )
                seq, rollout_log_probs, explicit_response_mask, stop_reason_codes = self._generate_dart_cached(
                    input_ids=idx.index_select(0, prefix_index_tensor),
                    attention_mask=attention_mask.index_select(0, prefix_index_tensor),
                    position_ids=prefill_position_ids,
                    multi_modal_inputs=multi_modal_inputs,
                    response_length=response_length,
                    do_sample=bool(kwargs.get("do_sample", False)),
                    temperature=float(kwargs.get("temperature", 1.0)),
                    top_k=int(kwargs.get("top_k", 0)),
                    top_p=float(kwargs.get("top_p", 1.0)),
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    compression_query_mask=prefill_query_mask,
                    prefix_branch_indices=(
                        prefix_branch_indices
                        if len(prefix_source_indices) < len(prompts)
                        else None
                    ),
                    semantic_stop_criteria=semantic_stop_criteria,
                )
            else:
                hf_config = getattr(self.model_config, "hf_config", None)
                functional_mode_kwargs = (
                    {"visual_compression_mode": visual_compression_mode}
                    if getattr(hf_config, "model_type", None) in {"qwen3_5", "qwen3_5_moe"}
                    else {}
                )
                stopping_criteria = (
                    StoppingCriteriaList([semantic_stop_criteria])
                    if semantic_stop_criteria is not None
                    else None
                )
                if semantic_stop_criteria is not None:
                    semantic_stop_criteria.reset()
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    position_ids=generation_position_ids,
                    **model_multi_modal_inputs,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    # Dense/no-image are ordinary uncompressed Qwen decode
                    # paths.  Recomputing the full visual/text prefix at every
                    # token makes the 1,500-row four-view gate quadratic and
                    # needlessly reruns the vision tower.  Keep the standard
                    # KV cache; fresh-process functional-mode parity audits
                    # verify the cached implementation before GPU release.
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                    **functional_mode_kwargs,
                )
                seq = output.sequences

        generated_batch_size = seq.size(0)

        # HuggingFace generate stops when all samples reach EOS.
        # Pad to fixed response length.
        sequence_length = prompt_length + response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(
                size=(generated_batch_size, delta_length),
                device=seq.device,
                dtype=seq.dtype,
            )
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        assert seq.shape[1] == sequence_length, (
            f"Generated sequence length mismatch: "
            f"seq.shape[1]={seq.shape[1]}, expected={sequence_length}"
        )

        # Repeat masks if num_return_sequences > 1.
        num_return_sequences = kwargs.get("num_return_sequences", 1)

        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]

        response_length = response.size(1)

        delta_position_id = torch.arange(
            1,
            response_length + 1,
            device=position_ids.device,
        )
        delta_position_id = delta_position_id.unsqueeze(0).repeat(
            generated_batch_size,
            1,
        )

        if position_ids.dim() == 3:
            response_position_ids = position_ids[..., -1:] + delta_position_id.unsqueeze(1)
        else:
            response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        if explicit_response_mask is not None:
            if explicit_response_mask.shape != response.shape:
                raise RuntimeError("Explicit rollout response mask is not response aligned")
            response_attention_mask = explicit_response_mask.to(dtype=attention_mask.dtype)
        else:
            semantic_lengths = (
                semantic_stop_criteria.finished_lengths
                if semantic_stop_criteria is not None
                else None
            )
            response_attention_mask, stop_reason_codes = response_action_mask(
                response,
                eos_token_ids=eos_token_id,
                semantic_finished_lengths=semantic_lengths,
                dtype=attention_mask.dtype,
            )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch_tensors = {
            "prompts": prompt,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_attention_mask,
        }
        if rollout_log_probs is not None:
            if rollout_log_probs.shape != response.shape:
                raise RuntimeError(
                    "DART sampled log-probability shape mismatch: "
                    f"log_probs={tuple(rollout_log_probs.shape)}, response={tuple(response.shape)}"
                )
            batch_tensors["rollout_log_probs"] = rollout_log_probs
        batch = TensorDict(batch_tensors, batch_size=generated_batch_size)

        # Do not empty the allocator inside a decode chunk.  ``generate_sequences``
        # may execute many chunks (80 trajectories/rank at rollout_n=8), and a
        # synchronized empty_cache here would serialize every chunk while doing
        # nothing to the live KV cache.  The owning FSDP worker applies the
        # hf_preserve_cuda_cache policy once at the rollout/training boundary,
        # after all chunk outputs have been moved to CPU.  This keeps fast cache
        # reuse between chunks while still allowing M7 to release rollout-only
        # allocator blocks before the full-parameter actor update.

        # The ordinary HF path shares the actor module and restores its train
        # mode here.  A dedicated rollout replica must remain inference-only;
        # switching it to train mode would enable dropout-like behavior in any
        # future architecture and invalidate rollout parity.
        self._restore_module_mode_after_rollout()

        output_non_tensors = {}
        if stop_reason_codes is None:
            raise RuntimeError("HF rollout did not produce per-row stop reasons")
        output_non_tensors["rollout_stop_reason"] = np.asarray(
            stop_reason_names(stop_reason_codes), dtype=object
        )
        if semantic_stop_criteria is not None:
            output_non_tensors["rollout_protocol_status"] = np.asarray(
                semantic_stop_criteria.protocol_status_names(), dtype=object
            )
            if bool(self.config.get("semantic_stop_virtual_open", False)):
                prefix = str(self.config.get("semantic_stop_transport_prefix", ""))
                if prefix != "<answer>":
                    raise RuntimeError("V8 virtual-open rollout lost its exact transport prefix")
                output_non_tensors["rollout_response_transport_prefix"] = np.asarray(
                    [prefix] * generated_batch_size, dtype=object
                )
        if prompts.non_tensor_batch is not None and "multi_modal_inputs" in prompts.non_tensor_batch:
            output_non_tensors["multi_modal_inputs"] = prompts.non_tensor_batch["multi_modal_inputs"]
        if merge_visual_tokens:
            routes = self._clear_and_collect_dart_routes(clear=False)
            prefix_index_tensor = torch.tensor(
                prefix_source_indices, dtype=torch.long, device=idx.device
            )
            prefix_position_mask = attention_mask.index_select(0, prefix_index_tensor)
            routes_by_prefix = self._split_dart_routes_by_sample(
                routes,
                input_ids=idx.index_select(0, prefix_index_tensor),
                attention_mask=prefix_position_mask[:, : idx.shape[-1]],
            )
            if prefix_branch_indices is not None and len(prefix_source_indices) < len(prompts):
                routes_by_sample = np.empty((len(prompts),), dtype=object)
                for row, physical in enumerate(prefix_branch_indices.tolist()):
                    routes_by_sample[row] = copy.deepcopy(routes_by_prefix[physical])
            else:
                routes_by_sample = routes_by_prefix
            if holitom_dpc_merge:
                output_non_tensors[HOLITOM_DPC_MERGE_ROUTES_KEY] = routes_by_sample
            else:
                if compression_query_audits is None:
                    raise RuntimeError("Conditional rollout did not retain route-query audit metadata")
                output_non_tensors["dart_merge_routes"] = self._attach_query_audits_to_routes(
                    routes_by_sample,
                    compression_query_audits,
                )
        output_meta_info = {"visual_compression_mode": visual_compression_mode}
        if visual_compression_mode == "no_image":
            output_meta_info["no_image_ablation_policy"] = NO_IMAGE_ABLATION_POLICY
        return DataProto(
            batch=batch,
            non_tensor_batch=output_non_tensors,
            meta_info=output_meta_info,
        )
