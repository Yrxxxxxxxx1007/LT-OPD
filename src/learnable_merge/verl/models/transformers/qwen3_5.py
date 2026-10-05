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

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ModelOutputWithPast,
)

from verl.models.transformers.vision_token_compressor import (
    DARTMergeRoute,
    HoliTomDPCSpatialMergeRoute,
    VISUAL_COMPRESSION_MODES,
    VisionHoliTomDPCSpatialMergeCompressor,
    compress_image_embeds,
    visual_token_compressor_enabled,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def fast_pos_embed_interpolate(self, grid_thw):
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = grid_thw.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


def _get_input_embeds(
    model: "Qwen3_5CausalLMOutputWithPast",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    dart_merge_routes=None,
    dpc_merge_routes=None,
    compression_query_mask: Optional[torch.Tensor] = None,
    visual_compression_mode: Optional[str] = None,
    run_dummy_vision: bool = True,
):
    compressor_is_enabled = visual_token_compressor_enabled(model)
    if visual_compression_mode is None:
        # Preserve old checkpoints while making all new actor/teacher callers
        # pass an explicit functional mode.  The mode is per-forward state and
        # never mutates the shared module, so dense teacher calls cannot race
        # with merged student calls.
        visual_compression_mode = (
            "merge"
            if compressor_is_enabled
            and (pixel_values is not None or dart_merge_routes is not None or dpc_merge_routes is not None)
            else ("dense" if pixel_values is not None or pixel_values_videos is not None else "no_image")
        )
    if visual_compression_mode not in VISUAL_COMPRESSION_MODES:
        raise ValueError(
            f"visual_compression_mode must be one of {sorted(VISUAL_COMPRESSION_MODES)}, "
            f"got {visual_compression_mode!r}"
        )
    if dart_merge_routes is not None and dpc_merge_routes is not None:
        raise ValueError("Legacy dart_merge_routes and formal dpc_merge_routes are mutually exclusive")
    if visual_compression_mode != "merge" and (
        dart_merge_routes is not None or dpc_merge_routes is not None or compression_query_mask is not None
    ):
        raise ValueError("Replay routes/query masks are valid only in visual_compression_mode='merge'")
    if visual_compression_mode == "merge" and pixel_values_videos is not None:
        algorithm = getattr(model, "vision_token_compressor_algorithm", "visual-token compressor")
        raise ValueError(f"{algorithm} supports image inputs only, not video.")
    if visual_compression_mode == "no_image" and (
        pixel_values is not None
        or pixel_values_videos is not None
        or image_grid_thw is not None
        or video_grid_thw is not None
    ):
        raise ValueError("visual_compression_mode='no_image' forbids pixel/video/grid values")

    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        if visual_compression_mode == "merge":
            if not compressor_is_enabled:
                raise ValueError("visual_compression_mode='merge' requires an enabled visual compressor")
            replay_routes = None
            if isinstance(model.vision_token_compressor, VisionHoliTomDPCSpatialMergeCompressor):
                if compression_query_mask is not None:
                    raise ValueError("Formal HoliTom DPC merge is query-independent; query masks are forbidden")
                if dart_merge_routes is not None:
                    raise ValueError("Formal HoliTom DPC merge cannot consume legacy dart_merge_routes")
                if dpc_merge_routes is not None:
                    replay_routes = [
                        route
                        if isinstance(route, HoliTomDPCSpatialMergeRoute)
                        else HoliTomDPCSpatialMergeRoute.from_dict(route, device=image_embeds.device)
                        for route in dpc_merge_routes
                    ]
                    for route in replay_routes:
                        route.validate_for_algorithm(model.vision_token_compressor.algorithm)
            else:
                if dpc_merge_routes is not None:
                    raise ValueError("Legacy visual compressors cannot consume formal dpc_merge_routes")
            if dart_merge_routes is not None:
                expected_algorithm = getattr(
                    model,
                    "vision_token_compressor_algorithm",
                    getattr(model.vision_token_compressor, "algorithm", None),
                )
                replay_routes = []
                for raw_route in dart_merge_routes:
                    route = (
                        raw_route.to(image_embeds.device)
                        if isinstance(raw_route, DARTMergeRoute)
                        else DARTMergeRoute.from_dict(raw_route, device=image_embeds.device)
                    )
                    route.validate_for_algorithm(str(expected_algorithm))
                    replay_routes.append(route)
            inputs_embeds, attention_mask, position_ids, used_routes = compress_image_embeds(
                model=model,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_embeds=image_embeds,
                image_token_id=model.config.image_token_id,
                routes=replay_routes,
                compression_query_mask=compression_query_mask,
            )
            model.vision_token_compressor_last_routes = [route.as_dict(cpu=True) for route in used_routes]
        else:
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if hasattr(model, "vision_token_compressor_last_routes"):
                model.vision_token_compressor_last_routes = None

    elif dart_merge_routes is not None or dpc_merge_routes is not None:
        raise ValueError("Visual replay routes were supplied without image pixel values")

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if (
        pixel_values is None
        and pixel_values_videos is None
        and run_dummy_vision
        and visual_compression_mode != "no_image"
    ):
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()

    if visual_compression_mode == "no_image":
        # Formal ablation policy:
        # same_prompt_image_placeholder_without_visual_replacement_v1.
        # Image/video placeholders deliberately remain in the exact public
        # token stream and use their ordinary token embeddings; no vision
        # features are computed or scattered and no M-RoPE coordinates are
        # selected/rebased.  This makes cross-view prompt hashes identical.
        if inputs_embeds.shape[:2] != input_ids.shape:
            raise RuntimeError("no_image mode changed the public placeholder-token count")
        if hasattr(model, "vision_token_compressor_last_routes"):
            model.vision_token_compressor_last_routes = None

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "position_ids": position_ids}


@dataclass
class Qwen3_5DARTModelOutputWithPast(Qwen3_5ModelOutputWithPast):
    """Qwen3.5 output plus physical tensors used by CDPruner/legacy prefill.

    The caller starts with an uncompressed token sequence, while the language
    model cache is built from the compressed embedding sequence.  Returning
    these tensors makes that physical cache contract explicit and prevents
    generation code from extending the stale, uncompressed attention mask.
    """

    dart_attention_mask: Optional[torch.Tensor] = None
    dart_position_ids: Optional[torch.LongTensor] = None


def qwen3_5_base_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    dart_merge_routes=None,
    dpc_merge_routes=None,
    compression_query_mask: Optional[torch.Tensor] = None,
    visual_compression_mode: Optional[str] = None,
    **kwargs,
):
    past_key_values = kwargs.get("past_key_values")
    if (
        visual_compression_mode == "merge"
        and pixel_values is None
        and past_key_values is None
    ):
        raise ValueError("A merge prefill requires image pixel values; pixel-free merge is cached decode only")
    # The dummy vision dependency is needed only for training batches without
    # visual inputs.  Cached decode must never re-enter the vision tower.
    run_dummy_vision = bool(self.training and past_key_values is None)
    input_kwargs = _get_input_embeds(
        self,
        input_ids,
        attention_mask,
        kwargs.get("position_ids"),
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        dart_merge_routes=dart_merge_routes,
        dpc_merge_routes=dpc_merge_routes,
        compression_query_mask=compression_query_mask,
        visual_compression_mode=visual_compression_mode,
        run_dummy_vision=run_dummy_vision,
    )
    kwargs.update(input_kwargs)
    outputs = self.language_model(
        input_ids=None,
        **kwargs,
    )
    return Qwen3_5DARTModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self, "rope_deltas", None),
        dart_attention_mask=input_kwargs["attention_mask"],
        dart_position_ids=input_kwargs["position_ids"],
    )


@dataclass
class Qwen3_5CausalLMOutputForPPO(Qwen3_5CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    dart_attention_mask: Optional[torch.Tensor] = None
    dart_position_ids: Optional[torch.LongTensor] = None


def _select_hidden_states_for_logits(
    hidden_states: torch.Tensor,
    logits_to_keep: int | torch.Tensor,
) -> torch.Tensor:
    """Select only hidden states that need vocabulary logits.

    Transformers generation passes an integer (normally ``1``).  A compressed actor
    scoring additionally needs per-sample response positions because visual
    compression can produce a different physical response start for every
    row, so a rank-2 index tensor is supported as an audited extension.
    """

    if isinstance(logits_to_keep, int):
        if logits_to_keep < 0:
            raise ValueError(f"logits_to_keep must be non-negative, got {logits_to_keep}")
        return hidden_states if logits_to_keep == 0 else hidden_states[:, -logits_to_keep:, :]

    indices = torch.as_tensor(logits_to_keep, device=hidden_states.device, dtype=torch.long)
    if indices.ndim == 1:
        return hidden_states.index_select(1, indices)
    if indices.ndim == 2 and indices.shape[0] == hidden_states.shape[0]:
        gather_index = indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        return torch.gather(hidden_states, dim=1, index=gather_index)
    raise ValueError(
        "logits_to_keep tensor must be [positions] or [batch, positions], got "
        f"{tuple(indices.shape)} for hidden states {tuple(hidden_states.shape)}"
    )


def forward_with_normal_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    logits_to_keep: int | torch.Tensor = 0,
    visual_compression_mode: Optional[str] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    if visual_compression_mode not in VISUAL_COMPRESSION_MODES:
        raise ValueError(
            f"visual_compression_mode must be one of {sorted(VISUAL_COMPRESSION_MODES)}, "
            f"got {visual_compression_mode!r}"
        )
    outputs = self.model(
        input_ids,
        visual_compression_mode=visual_compression_mode,
        **kwargs,
    )
    hidden_states = _select_hidden_states_for_logits(outputs[0], logits_to_keep)
    logits = self.lm_head(hidden_states)
    return Qwen3_5CausalLMOutputForPPO(
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        dart_attention_mask=outputs.dart_attention_mask,
        dart_position_ids=outputs.dart_position_ids,
    )


def forward_with_torch_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
