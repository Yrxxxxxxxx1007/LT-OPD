# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class VisualTokenMLPProjector(nn.Module):
    """Pool ViT tokens to a fixed budget and refine them with a small MLP."""

    def __init__(self, hidden_size: int, num_tokens: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        inner_size = max(1, int(hidden_size * mlp_ratio))
        self.num_tokens = int(num_tokens)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size),
        )

    def forward(self, image_embeds: torch.Tensor, num_tokens: Optional[int] = None) -> torch.Tensor:
        target_tokens = int(num_tokens or self.num_tokens)
        if image_embeds.dim() != 2:
            raise ValueError(f"image_embeds must be rank-2 [tokens, hidden], got {tuple(image_embeds.shape)}")
        if target_tokens <= 0:
            raise ValueError(f"target token count must be positive, got {target_tokens}")

        pooled = F.adaptive_avg_pool1d(image_embeds.transpose(0, 1).unsqueeze(0), target_tokens)
        pooled = pooled.squeeze(0).transpose(0, 1).contiguous()
        return pooled + self.mlp(self.norm(pooled))


def visual_token_projector_enabled(model: nn.Module) -> bool:
    return bool(getattr(model, "vision_token_projector_enabled", False)) and hasattr(
        model,
        "vision_token_projector",
    )


def compress_image_embeds(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    image_embeds: torch.Tensor,
    image_token_id: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Replace each contiguous image-token span with fixed-budget projected embeddings."""

    if not visual_token_projector_enabled(model):
        return inputs_embeds, attention_mask, position_ids

    projector = model.vision_token_projector
    batch_size, seq_len, hidden_size = inputs_embeds.shape
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.long)

    if position_ids is not None and position_ids.dim() == 2:
        position_ids_work = position_ids.unsqueeze(0)
        squeeze_position_ids = True
    else:
        position_ids_work = position_ids
        squeeze_position_ids = False

    embed_offset = 0
    new_embeds = []
    new_masks = []
    new_positions = [] if position_ids_work is not None else None

    for batch_idx in range(batch_size):
        ids = input_ids[batch_idx]
        spans = []
        cursor = 0
        while cursor < seq_len:
            if ids[cursor].item() != image_token_id:
                cursor += 1
                continue
            start = cursor
            while cursor < seq_len and ids[cursor].item() == image_token_id:
                cursor += 1
            spans.append((start, cursor))

        pieces = []
        mask_pieces = []
        pos_pieces = [] if position_ids_work is not None else None
        last = 0
        for start, end in spans:
            if start > last:
                pieces.append(inputs_embeds[batch_idx, last:start])
                mask_pieces.append(attention_mask[batch_idx, last:start])
                if pos_pieces is not None:
                    pos_pieces.append(position_ids_work[:, batch_idx, last:start])

            span_len = end - start
            span_embeds = image_embeds[embed_offset : embed_offset + span_len]
            embed_offset += span_len
            reduced = projector(span_embeds).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            pieces.append(reduced)
            mask_pieces.append(torch.ones((reduced.size(0),), device=attention_mask.device, dtype=attention_mask.dtype))
            if pos_pieces is not None:
                gather_idx = torch.linspace(
                    0,
                    span_len - 1,
                    steps=reduced.size(0),
                    device=position_ids_work.device,
                ).round().to(torch.long)
                pos_pieces.append(position_ids_work[:, batch_idx, start:end].index_select(dim=1, index=gather_idx))
            last = end

        if last < seq_len:
            pieces.append(inputs_embeds[batch_idx, last:])
            mask_pieces.append(attention_mask[batch_idx, last:])
            if pos_pieces is not None:
                pos_pieces.append(position_ids_work[:, batch_idx, last:])

        new_embeds.append(torch.cat(pieces, dim=0))
        new_masks.append(torch.cat(mask_pieces, dim=0))
        if new_positions is not None and pos_pieces is not None:
            new_positions.append(torch.cat(pos_pieces, dim=1))

    if embed_offset != image_embeds.size(0):
        raise ValueError(
            f"Unused image embeddings after compression: used {embed_offset}, total {image_embeds.size(0)}"
        )

    max_len = max(item.size(0) for item in new_embeds)
    padded_embeds = inputs_embeds.new_zeros((batch_size, max_len, hidden_size))
    padded_masks = attention_mask.new_zeros((batch_size, max_len))
    padded_positions = None
    if new_positions is not None:
        padded_positions = position_ids_work.new_zeros((position_ids_work.size(0), batch_size, max_len))

    for batch_idx, embeds in enumerate(new_embeds):
        length = embeds.size(0)
        padded_embeds[batch_idx, :length] = embeds
        padded_masks[batch_idx, :length] = new_masks[batch_idx]
        if padded_positions is not None:
            padded_positions[:, batch_idx, :length] = new_positions[batch_idx]

    if padded_positions is not None and squeeze_position_ids:
        padded_positions = padded_positions.squeeze(0)

    return padded_embeds, padded_masks, padded_positions
