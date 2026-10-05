# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.

"""Visual-token pruning and aggregation for Qwen3.5.

CDPruner selects tokens after the visual merger. Discarded tokens are assigned
to their nearest retained token by FP32 cosine similarity. An MLP learns
bounded signed residual contributions, initialized to zero. The output keeps
the retained tokens' sequence order and M-RoPE positions without adding tokens.

Routes are shared between generation and training. The module also supports
plain CDPruner selection, DPC spatial merging, and single-token summaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


CDPRUNER_ALGORITHM = "qwen35_cdpruner_v1"
CDPRUNER_METHOD = "cdpruner"
CDPRUNER_ROUTE_SCHEMA_VERSION = "vision_opd_cdpruner_route_v1"
CDPRUNER_CURRICULUM_ROUTE_SCHEMA_VERSION = "vision_opd_cdpruner_curriculum_route_v2"
CDPRUNER_SUMMARY_ROUTE_SCHEMA_VERSION = "vision_opd_cdpruner_summary_route_v2"
CDPRUNER_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION = "vision_opd_cdpruner_summary_curriculum_route_v3"
CDPRUNER_LEARNABLE_SUMMARY_ROUTE_SCHEMA_VERSION = "vision_opd_cdpruner_learnable_summary_route_v3"
CDPRUNER_LEARNABLE_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION = "vision_opd_cdpruner_learnable_summary_curriculum_route_v4"
CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE = "learnable_residual_mlp_v1"
CDPRUNER_LEARNABLE_MERGE_INITIALIZATION = "pretrained_mlp_gate_rows_zero_output_v1"
CDPRUNER_LEARNABLE_MERGE_ROUTE_SCHEMA_VERSION = "vision_opd_cdpruner_learnable_merge_zero_residual_route_v1"
CDPRUNER_LEARNABLE_MERGE_CURRICULUM_ROUTE_SCHEMA_VERSION = "vision_opd_cdpruner_learnable_merge_zero_residual_curriculum_route_v1"
CDPRUNER_LEARNABLE_MERGE_AGGREGATION = "zero_init_signed_residual_v1"
CDPRUNER_LEARNABLE_MERGE_SCHEMAS = frozenset({
    CDPRUNER_LEARNABLE_MERGE_ROUTE_SCHEMA_VERSION,
    CDPRUNER_LEARNABLE_MERGE_CURRICULUM_ROUTE_SCHEMA_VERSION,
})
CDPRUNER_LEARNABLE_SUMMARY_SCHEMAS = frozenset({
    CDPRUNER_LEARNABLE_SUMMARY_ROUTE_SCHEMA_VERSION,
    CDPRUNER_LEARNABLE_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION,
})
CDPRUNER_SUMMARY_SCHEMAS = frozenset({
    CDPRUNER_SUMMARY_ROUTE_SCHEMA_VERSION, CDPRUNER_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION,
}) | CDPRUNER_LEARNABLE_SUMMARY_SCHEMAS
CDPRUNER_SUMMARY_WEIGHT_FORMAT = "float32_ieee754_int32_bits_v1"
CDPRUNER_RETENTION_RATIO = 0.05
CDPRUNER_MINIMUM_TOKENS = 32
LEGACY_CONDITIONAL_DIVERSITY_ALGORITHM = "qwen35_conditional_diversity_prune_v1"
HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM = "qwen35_holitom_dpc_spatial_merge_v1"
HOLITOM_DPC_SPATIAL_MERGE_METHOD = "holitom_inspired_dpc_diversity_merge"
HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA = "vision_opd_holitom_dpc_merge_route_v1"
HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM = "qwen35_holitom_dpc_spatial_merge_curriculum_v2"
HOLITOM_DPC_CURRICULUM_MERGE_ROUTE_SCHEMA = "vision_opd_holitom_dpc_curriculum_merge_route_v2"
HOLITOM_DPC_MERGE_ALGORITHMS = frozenset({
    HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM, HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM,
})
HOLITOM_DPC_MERGE_ROUTES_KEY = "dpc_merge_routes"
VISUAL_COMPRESSION_MODES = frozenset({"merge", "dense", "no_image"})
NO_IMAGE_ABLATION_POLICY = "same_prompt_image_placeholder_without_visual_replacement_v1"


def build_qwen3_5_position_ids(
    processor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build canonical four-channel Qwen3.5 M-RoPE positions.

    Transformers 5.x exposes the model as ``Qwen3VLProcessor`` and does not
    reliably provide ``processor.config.model_type``.  Detecting Qwen3.5 via
    that attribute made the teacher omit the required ``mm_token_type_ids``.
    This strict helper is shared by HF rollout and teacher construction.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Qwen3.5 input_ids and attention_mask must be [batch, sequence]")
    if mm_token_type_ids.shape != input_ids.shape:
        raise ValueError(
            f"Qwen3.5 mm_token_type_ids shape {tuple(mm_token_type_ids.shape)} does not match "
            f"input_ids {tuple(input_ids.shape)}"
        )
    result = processor.get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    vision_positions = result[0] if isinstance(result, tuple) else result
    if vision_positions.ndim == 2 and input_ids.size(0) == 1:
        vision_positions = vision_positions.unsqueeze(1)
    if vision_positions.ndim != 3:
        raise ValueError(f"Qwen3.5 get_rope_index returned invalid shape {tuple(vision_positions.shape)}")
    if vision_positions.shape[0] == input_ids.size(0) and vision_positions.shape[1] == 3:
        vision_positions = vision_positions.transpose(0, 1)
    if vision_positions.shape != (3, input_ids.size(0), input_ids.size(1)):
        raise ValueError(
            "Qwen3.5 vision M-RoPE positions must be [3, batch, sequence], got "
            f"{tuple(vision_positions.shape)}"
        )
    text_positions = attention_mask.long().cumsum(dim=-1) - 1
    text_positions.masked_fill_(attention_mask == 0, 1)
    return torch.cat([text_positions.unsqueeze(0), vision_positions.to(text_positions.device)], dim=0)


@dataclass(frozen=True)
class DARTMergeRoute:
    """Selected visual token indices and positions for replay."""

    selected_indices: torch.Tensor
    original_tokens: int
    output_tokens: int
    anchor_coordinates: Optional[torch.Tensor] = None
    schema_version: Optional[str] = None
    algorithm: Optional[str] = None
    method: Optional[str] = None
    retention_bps: Optional[int] = None
    curriculum_completed_steps: Optional[int] = None
    curriculum_schedule_sha256: Optional[str] = None
    summary_anchor_index: Optional[int] = None
    summary_source_indices: Optional[torch.Tensor] = None
    # Fixed-summary weights, or detached cosine priors for learnable summaries.
    # Learned alpha is recomputed from current differentiable features/head.
    summary_weights: Optional[torch.Tensor] = None
    # Map each original node exactly once to a retained output slot.
    # merge_prior_weights holds detached discarded-only priors, never learned
    # signed residual weights; retained-anchor priors are exactly zero.
    merge_assignment: Optional[torch.Tensor] = None
    merge_prior_weights: Optional[torch.Tensor] = None

    @property
    def summary_enabled(self) -> bool:
        return self.schema_version in CDPRUNER_SUMMARY_SCHEMAS

    @property
    def learnable_merge_enabled(self) -> bool:
        return self.schema_version in CDPRUNER_LEARNABLE_MERGE_SCHEMAS

    def validate(self) -> None:
        if (
            isinstance(self.original_tokens, bool)
            or not isinstance(self.original_tokens, Integral)
            or isinstance(self.output_tokens, bool)
            or not isinstance(self.output_tokens, Integral)
        ):
            raise TypeError("DART route token counts must be integers")
        if self.original_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("DART route token counts must be positive")
        if self.output_tokens > self.original_tokens:
            raise ValueError("DART output token count cannot exceed its source count")
        if self.selected_indices.shape != (self.output_tokens,):
            raise ValueError("selected_indices has an invalid shape")
        if self.anchor_coordinates is not None and self.anchor_coordinates.shape != (self.output_tokens, 3):
            raise ValueError("anchor_coordinates must be [output_tokens, 3]")
        identity = (self.schema_version, self.algorithm, self.method)
        if any(value is not None for value in identity) and not all(
            isinstance(value, str) and value for value in identity
        ):
            raise ValueError("DART route identity must be either absent or a complete non-empty triple")
        curriculum_identity = (
            self.retention_bps,
            self.curriculum_completed_steps,
            self.curriculum_schedule_sha256,
        )
        if self.schema_version in {
            CDPRUNER_CURRICULUM_ROUTE_SCHEMA_VERSION,
            CDPRUNER_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION,
            CDPRUNER_LEARNABLE_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION,
            CDPRUNER_LEARNABLE_MERGE_CURRICULUM_ROUTE_SCHEMA_VERSION,
        }:
            if any(value is None for value in curriculum_identity):
                raise ValueError("curriculum CDPruner routes require complete schedule metadata")
            if (
                isinstance(self.retention_bps, bool)
                or not isinstance(self.retention_bps, Integral)
                or not 1 <= int(self.retention_bps) <= 10_000
            ):
                raise ValueError("curriculum route retention_bps must be an integer in [1, 10000]")
            if (
                isinstance(self.curriculum_completed_steps, bool)
                or not isinstance(self.curriculum_completed_steps, Integral)
                or int(self.curriculum_completed_steps) < 0
            ):
                raise ValueError("curriculum route completed steps must be a non-negative integer")
            if (
                not isinstance(self.curriculum_schedule_sha256, str)
                or len(self.curriculum_schedule_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.curriculum_schedule_sha256)
            ):
                raise ValueError("curriculum route schedule hash must be lowercase SHA-256")
        elif any(value is not None for value in curriculum_identity):
            raise ValueError("curriculum metadata is valid only for the curriculum route schema")
        if self.selected_indices.dtype != torch.long:
            raise TypeError("DART route indices must use torch.long")
        if self.anchor_coordinates is not None:
            if self.anchor_coordinates.is_floating_point() or self.anchor_coordinates.is_complex():
                raise TypeError("DART anchor_coordinates must contain integer M-RoPE coordinates")
            if not torch.isfinite(self.anchor_coordinates).all().item():
                raise ValueError("DART anchor_coordinates contains NaN or Inf")
        if torch.unique(self.selected_indices).numel() != self.output_tokens:
            raise ValueError("DART anchors must be unique")
        if self.output_tokens > 1 and torch.any(self.selected_indices[1:] <= self.selected_indices[:-1]).item():
            raise ValueError("DART selected_indices must be strictly increasing in original token order")
        if self.selected_indices.min().item() < 0 or self.selected_indices.max().item() >= self.original_tokens:
            raise ValueError("DART anchor index is out of range")
        summary_fields = (self.summary_anchor_index, self.summary_source_indices, self.summary_weights)
        if not self.summary_enabled:
            if any(value is not None for value in summary_fields):
                raise ValueError("Pure-prune routes forbid summary fields")
        elif all(value is None for value in summary_fields):
            if self.output_tokens != self.original_tokens:
                raise ValueError("A summary route without discarded sources must retain all tokens")
        elif any(value is None for value in summary_fields):
            raise ValueError("Summary route fields must be either complete or absent")
        else:
            anchor = self.summary_anchor_index
            if isinstance(anchor, bool) or not isinstance(anchor, Integral) or not 0 <= anchor < self.original_tokens:
                raise ValueError("Summary anchor must be an original token index")
            source = self.summary_source_indices
            weights = self.summary_weights
            if source.ndim != 1 or source.numel() == 0 or source.dtype != torch.long:
                raise ValueError("Summary source indices must be a non-empty torch.long vector")
            if weights.shape != source.shape or weights.dtype != torch.float32:
                raise ValueError("Summary weights must be an aligned FP32 vector")
            if not torch.isfinite(weights).all().item() or torch.any(weights <= 0).item():
                raise ValueError("Every discarded token must have a finite positive summary weight")
            if not torch.isclose(weights.sum(), weights.new_tensor(1.0), rtol=0.0, atol=1e-5).item():
                raise ValueError("Summary weights must sum to one")
            if source.min().item() < 0 or source.max().item() >= self.original_tokens:
                raise ValueError("Summary source index is out of range")
            if source.numel() > 1 and torch.any(source[1:] <= source[:-1]).item():
                raise ValueError("Summary source indices must follow original token order")
            if int((self.selected_indices == anchor).sum().item()) != 1:
                raise ValueError("Summary anchor must occupy exactly one output slot")
            kept = self.selected_indices[self.selected_indices != anchor]
            discarded_mask = torch.ones(self.original_tokens, device=self.selected_indices.device, dtype=torch.bool)
            discarded_mask[kept] = False
            expected_sources = torch.nonzero(discarded_mask, as_tuple=False).flatten()
            if not torch.equal(source.to(expected_sources.device), expected_sources):
                raise ValueError("Summary sources must contain all and only the discarded tokens, including its anchor")
        merge_fields = (self.merge_assignment, self.merge_prior_weights)
        if not self.learnable_merge_enabled:
            if any(value is not None for value in merge_fields):
                raise ValueError("Only learnable merge routes may carry merge assignments/priors")
        else:
            if any(value is None for value in merge_fields):
                raise ValueError("Learnable merge routes require complete assignments/priors")
            assignment, priors = merge_fields
            if assignment.shape != (self.original_tokens,) or assignment.dtype != torch.long:
                raise ValueError("Learnable merge assigns exactly one torch.long output slot per original node")
            if priors.shape != assignment.shape or priors.dtype != torch.float32:
                raise ValueError("Learnable merge priors must be one aligned FP32 value per original node")
            if assignment.min().item() < 0 or assignment.max().item() >= self.output_tokens:
                raise ValueError("Learnable merge assignment is outside the retained output slots")
            if not torch.isfinite(priors).all().item() or torch.any(priors < 0).item():
                raise ValueError("Learnable merge priors must be finite non-negative FP32 values")
            anchors = self.selected_indices.to(assignment.device)
            if not torch.equal(assignment[anchors], torch.arange(self.output_tokens, device=assignment.device)):
                raise ValueError("Each retained merge anchor must be assigned to its own output slot")
            counts = torch.bincount(assignment, minlength=self.output_tokens)
            group_sums = priors.new_zeros((self.output_tokens,)).scatter_add(0, assignment, priors)
            expected_sums = (counts > 1).to(priors.dtype)
            if not torch.allclose(group_sums, expected_sums, rtol=0.0, atol=1e-5):
                raise ValueError("Discarded priors must sum to one per non-singleton group and zero per singleton")
            if torch.any(priors[anchors] != 0).item():
                raise ValueError("Retained anchors must have exactly zero residual prior")
            discarded = torch.ones(self.original_tokens, device=assignment.device, dtype=torch.bool)
            discarded[anchors] = False
            if torch.any(priors[discarded] <= 0).item():
                raise ValueError("Every discarded node must have a positive finite residual prior")
            expected_tokens = _exact_cdpruner_budget(
                self.original_tokens, retention_bps=int(self.retention_bps) if self.retention_bps is not None else 500,
            )
            if self.output_tokens != expected_tokens:
                raise ValueError("Learnable merge output count must equal the original CDPruner K without an extra token")

    def to(self, device: torch.device | str) -> "DARTMergeRoute":
        return DARTMergeRoute(
            selected_indices=self.selected_indices.to(device=device),
            original_tokens=self.original_tokens,
            output_tokens=self.output_tokens,
            anchor_coordinates=(
                self.anchor_coordinates.to(device=device) if self.anchor_coordinates is not None else None
            ),
            schema_version=self.schema_version,
            algorithm=self.algorithm,
            method=self.method,
            retention_bps=self.retention_bps,
            curriculum_completed_steps=self.curriculum_completed_steps,
            curriculum_schedule_sha256=self.curriculum_schedule_sha256,
            summary_anchor_index=self.summary_anchor_index,
            summary_source_indices=(self.summary_source_indices.to(device=device) if self.summary_source_indices is not None else None),
            summary_weights=(self.summary_weights.to(device=device) if self.summary_weights is not None else None),
            merge_assignment=(self.merge_assignment.to(device=device) if self.merge_assignment is not None else None),
            merge_prior_weights=(self.merge_prior_weights.to(device=device) if self.merge_prior_weights is not None else None),
        )

    def as_dict(self, *, cpu: bool = True) -> dict:
        route = self.to("cpu") if cpu else self
        payload = {
            "selected_indices": route.selected_indices,
            "original_tokens": route.original_tokens,
            "output_tokens": route.output_tokens,
            "anchor_coordinates": route.anchor_coordinates,
        }
        if route.schema_version is not None:
            payload.update(
                {
                    "schema_version": route.schema_version,
                    "algorithm": route.algorithm,
                    "method": route.method,
                }
            )
        if route.retention_bps is not None:
            payload.update(
                {
                    "retention_bps": route.retention_bps,
                    "curriculum_completed_steps": route.curriculum_completed_steps,
                    "curriculum_schedule_sha256": route.curriculum_schedule_sha256,
                }
            )
        if route.summary_enabled:
            payload.update({
                "summary_anchor_index": route.summary_anchor_index,
                "summary_source_indices": route.summary_source_indices,
                # FSDP recursively casts floating forward inputs to BF16.
                # Integer bit transport preserves the exact FP32 route instead.
                # For learnable schemas these bits carry only baseline priors,
                # never the final differentiable aggregation weights.
                "summary_weight_format": CDPRUNER_SUMMARY_WEIGHT_FORMAT,
                "summary_weight_bits": (
                    route.summary_weights.detach().contiguous().view(torch.int32)
                    if route.summary_weights is not None else None
                ),
            })
        if route.learnable_merge_enabled:
            payload.update({
                "merge_assignment": route.merge_assignment,
                "merge_prior_weight_format": CDPRUNER_SUMMARY_WEIGHT_FORMAT,
                "merge_prior_weight_bits": route.merge_prior_weights.detach().contiguous().view(torch.int32),
            })
        return payload

    @classmethod
    def from_dict(cls, payload: dict, *, device: Optional[torch.device | str] = None) -> "DARTMergeRoute":
        required = {"selected_indices", "original_tokens", "output_tokens"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Serialized DART route is missing keys: {sorted(missing)}")
        original_tokens = payload["original_tokens"]
        output_tokens = payload["output_tokens"]
        if (
            isinstance(original_tokens, bool)
            or not isinstance(original_tokens, Integral)
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, Integral)
        ):
            raise TypeError("Serialized DART route token counts must be integers")

        def exact_long_tensor(field: str) -> torch.Tensor:
            value = torch.as_tensor(payload[field])
            if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise TypeError(f"Serialized DART route {field} must contain integers")
            return value.to(dtype=torch.long)

        def exact_weight_tensor(*, merge: bool = False) -> Optional[torch.Tensor]:
            schema = payload.get("schema_version") in (CDPRUNER_LEARNABLE_MERGE_SCHEMAS if merge else CDPRUNER_SUMMARY_SCHEMAS)
            prefix = "merge_prior" if merge else "summary"
            field = "merge_prior_weights" if merge else "summary_weights"
            if field in payload:
                raise ValueError("Floating weight transport is forbidden; use exact FP32 integer bits")
            weight_format = payload.get(prefix + "_weight_format")
            raw = payload.get(prefix + "_weight_bits")
            if schema:
                if weight_format != CDPRUNER_SUMMARY_WEIGHT_FORMAT:
                    raise ValueError(f"Serialized {prefix} weight format is missing or unsupported")
            elif weight_format is not None or raw is not None:
                raise ValueError(f"Route schema forbids {prefix} weight transport")
            if raw is None:
                return None
            value = torch.as_tensor(raw)
            if value.ndim != 1 or value.numel() == 0:
                raise ValueError("Serialized summary weight bits must be a non-empty vector")
            if isinstance(raw, torch.Tensor):
                if value.dtype != torch.int32:
                    raise TypeError("Serialized summary weight bit tensors must be int32")
            else:
                if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                    raise TypeError("Serialized summary weight bits must contain integers")
                limits = torch.iinfo(torch.int32)
                if value.min().item() < limits.min or value.max().item() > limits.max:
                    raise ValueError("Serialized summary weight bits exceed int32 range")
                value = value.to(dtype=torch.int32)
            return value.contiguous().view(torch.float32)

        raw_coordinates = payload.get("anchor_coordinates")
        coordinates = None
        if raw_coordinates is not None:
            coordinates = torch.as_tensor(raw_coordinates)
            if coordinates.dtype == torch.bool or coordinates.is_floating_point() or coordinates.is_complex():
                raise TypeError("Serialized DART anchor_coordinates must contain integers")
            coordinates = coordinates.to(dtype=torch.long)
        route = cls(
            selected_indices=exact_long_tensor("selected_indices"),
            original_tokens=int(original_tokens),
            output_tokens=int(output_tokens),
            anchor_coordinates=coordinates,
            schema_version=payload.get("schema_version"),
            algorithm=payload.get("algorithm"),
            method=payload.get("method"),
            retention_bps=payload.get("retention_bps"),
            curriculum_completed_steps=payload.get("curriculum_completed_steps"),
            curriculum_schedule_sha256=payload.get("curriculum_schedule_sha256"),
            summary_anchor_index=payload.get("summary_anchor_index"),
            summary_source_indices=(exact_long_tensor("summary_source_indices") if payload.get("summary_source_indices") is not None else None),
            summary_weights=exact_weight_tensor(),
            merge_assignment=(exact_long_tensor("merge_assignment") if payload.get("merge_assignment") is not None else None),
            merge_prior_weights=exact_weight_tensor(merge=True),
        )
        if device is not None:
            route = route.to(device)
        route.validate()
        return route

    def validate_for_algorithm(
        self,
        expected_algorithm: str,
        *,
        expected_schema_version: Optional[str] = None,
    ) -> None:
        """Fail closed on cross-method replay while retaining old DART compatibility."""

        self.validate()
        if expected_algorithm == CDPRUNER_ALGORITHM:
            actual = (self.schema_version, self.algorithm, self.method)
            admitted_schemas = {
                CDPRUNER_ROUTE_SCHEMA_VERSION,
                CDPRUNER_CURRICULUM_ROUTE_SCHEMA_VERSION,
                CDPRUNER_SUMMARY_ROUTE_SCHEMA_VERSION,
                CDPRUNER_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION,
                CDPRUNER_LEARNABLE_SUMMARY_ROUTE_SCHEMA_VERSION,
                CDPRUNER_LEARNABLE_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION,
                CDPRUNER_LEARNABLE_MERGE_ROUTE_SCHEMA_VERSION,
                CDPRUNER_LEARNABLE_MERGE_CURRICULUM_ROUTE_SCHEMA_VERSION,
            }
            required_schema = expected_schema_version
            if required_schema is not None and required_schema not in admitted_schemas:
                raise ValueError(f"unsupported expected CDPruner route schema: {required_schema!r}")
            schema_ok = (
                self.schema_version == required_schema
                if required_schema is not None
                else self.schema_version in admitted_schemas
            )
            if not schema_ok or self.algorithm != CDPRUNER_ALGORITHM or self.method != CDPRUNER_METHOD:
                expected = (
                    required_schema if required_schema is not None else sorted(admitted_schemas),
                    CDPRUNER_ALGORITHM,
                    CDPRUNER_METHOD,
                )
                raise ValueError(
                    "CDPruner replay route identity mismatch: "
                    f"expected={expected!r}, actual={actual!r}"
                )
        elif self.algorithm is not None and self.algorithm != expected_algorithm:
            raise ValueError(
                "Visual-compression replay route algorithm mismatch: "
                f"expected={expected_algorithm!r}, actual={self.algorithm!r}"
            )


@dataclass(frozen=True)
class HoliTomDPCSpatialMergeRoute:
    """Replayable DPC clustering route for spatial merging.

    ``assignment[i]`` is the output-cluster slot receiving source token ``i``;
    ``center_indices`` are strictly increasing, so both output embeddings and
    anchor M-RoPE coordinates remain in original visual sequence order.
    """

    center_indices: torch.Tensor
    assignment: torch.Tensor
    source_counts: torch.Tensor
    original_tokens: int
    output_tokens: int
    anchor_coordinates: Optional[torch.Tensor] = None
    schema_version: str = HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA
    algorithm: str = HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM
    method: str = HOLITOM_DPC_SPATIAL_MERGE_METHOD
    retention_bps: Optional[int] = None
    curriculum_completed_steps: Optional[int] = None
    curriculum_schedule_sha256: Optional[str] = None

    @property
    def selected_indices(self) -> torch.Tensor:
        """Compatibility accessor used by the shared sequence compactor."""

        return self.center_indices

    def validate(self) -> None:
        expected_schema = {
            HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM: HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA,
            HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM: HOLITOM_DPC_CURRICULUM_MERGE_ROUTE_SCHEMA,
        }.get(self.algorithm)
        if expected_schema is None or self.schema_version != expected_schema or self.method != HOLITOM_DPC_SPATIAL_MERGE_METHOD:
            raise ValueError("HoliTom DPC route algorithm/schema/method identity mismatch")
        metadata = (self.retention_bps, self.curriculum_completed_steps, self.curriculum_schedule_sha256)
        if self.algorithm == HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM:
            if isinstance(self.retention_bps, bool) or not isinstance(self.retention_bps, Integral) or not 1 <= self.retention_bps <= 10000:
                raise ValueError("HoliTom curriculum route requires integer retention_bps in [1, 10000]")
            if isinstance(self.curriculum_completed_steps, bool) or not isinstance(self.curriculum_completed_steps, Integral) or self.curriculum_completed_steps < 0:
                raise ValueError("HoliTom curriculum route requires non-negative completed steps")
            digest = self.curriculum_schedule_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("HoliTom curriculum route requires a lowercase SHA-256 schedule digest")
        elif any(value is not None for value in metadata):
            raise ValueError("Static HoliTom routes forbid curriculum metadata")
        if (
            isinstance(self.original_tokens, bool)
            or not isinstance(self.original_tokens, Integral)
            or isinstance(self.output_tokens, bool)
            or not isinstance(self.output_tokens, Integral)
        ):
            raise TypeError("HoliTom DPC route token counts must be integers")
        if self.original_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("HoliTom DPC route token counts must be positive")
        if self.output_tokens > self.original_tokens:
            raise ValueError("HoliTom DPC output token count cannot exceed its source count")
        if self.center_indices.shape != (self.output_tokens,):
            raise ValueError("center_indices has an invalid shape")
        if self.assignment.shape != (self.original_tokens,):
            raise ValueError("assignment has an invalid shape")
        if self.source_counts.shape != (self.output_tokens,):
            raise ValueError("source_counts has an invalid shape")
        if self.anchor_coordinates is not None and self.anchor_coordinates.shape != (self.output_tokens, 3):
            raise ValueError("anchor_coordinates must be [output_tokens, 3]")
        if self.center_indices.dtype != torch.long or self.assignment.dtype != torch.long:
            raise TypeError("HoliTom DPC route indices must use torch.long")
        if self.source_counts.dtype != torch.long:
            raise TypeError("HoliTom DPC source_counts must use torch.long")
        if torch.unique(self.center_indices).numel() != self.output_tokens:
            raise ValueError("HoliTom DPC centers must be unique")
        if self.output_tokens > 1 and torch.any(self.center_indices[1:] <= self.center_indices[:-1]).item():
            raise ValueError("HoliTom DPC centers must be strictly increasing in original token order")
        if self.center_indices.min().item() < 0 or self.center_indices.max().item() >= self.original_tokens:
            raise ValueError("HoliTom DPC center index is out of range")
        if self.assignment.min().item() < 0 or self.assignment.max().item() >= self.output_tokens:
            raise ValueError("HoliTom DPC assignment is out of range")
        if int(self.source_counts.sum().item()) != self.original_tokens:
            raise ValueError("HoliTom DPC source counts do not conserve all input tokens")
        if torch.any(self.source_counts <= 0).item():
            raise ValueError("Every HoliTom DPC center must own at least one source token")
        expected = torch.bincount(self.assignment, minlength=self.output_tokens).to(self.source_counts.device)
        if not torch.equal(expected, self.source_counts):
            raise ValueError("HoliTom DPC source_counts does not match assignment")
        center_assignment = self.assignment.index_select(0, self.center_indices.to(self.assignment.device))
        expected_center_assignment = torch.arange(
            self.output_tokens, device=self.assignment.device, dtype=torch.long
        )
        if not torch.equal(center_assignment, expected_center_assignment):
            raise ValueError("Every HoliTom DPC center must be assigned to its own output slot")
        if self.anchor_coordinates is not None:
            if self.anchor_coordinates.is_floating_point() or self.anchor_coordinates.is_complex():
                raise TypeError("HoliTom DPC anchor_coordinates must contain integer M-RoPE coordinates")
            if not torch.isfinite(self.anchor_coordinates).all().item():
                raise ValueError("HoliTom DPC anchor_coordinates contains NaN or Inf")

    def to(self, device: torch.device | str) -> "HoliTomDPCSpatialMergeRoute":
        return HoliTomDPCSpatialMergeRoute(
            center_indices=self.center_indices.to(device=device),
            assignment=self.assignment.to(device=device),
            source_counts=self.source_counts.to(device=device),
            original_tokens=self.original_tokens,
            output_tokens=self.output_tokens,
            anchor_coordinates=(
                self.anchor_coordinates.to(device=device) if self.anchor_coordinates is not None else None
            ),
            schema_version=self.schema_version,
            algorithm=self.algorithm,
            method=self.method,
            retention_bps=self.retention_bps,
            curriculum_completed_steps=self.curriculum_completed_steps,
            curriculum_schedule_sha256=self.curriculum_schedule_sha256,
        )

    def as_dict(self, *, cpu: bool = True) -> dict[str, Any]:
        route = self.to("cpu") if cpu else self
        payload = {
            "schema_version": route.schema_version,
            "algorithm": route.algorithm,
            "method": route.method,
            "center_indices": route.center_indices,
            "assignment": route.assignment,
            "source_counts": route.source_counts,
            "original_tokens": route.original_tokens,
            "output_tokens": route.output_tokens,
            "anchor_coordinates": route.anchor_coordinates,
        }
        if route.retention_bps is not None:
            payload.update(
                retention_bps=route.retention_bps,
                curriculum_completed_steps=route.curriculum_completed_steps,
                curriculum_schedule_sha256=route.curriculum_schedule_sha256,
            )
        return payload

    def validate_for_algorithm(self, expected_algorithm: str) -> None:
        self.validate()
        if self.algorithm != expected_algorithm:
            raise ValueError(f"HoliTom route algorithm mismatch: expected={expected_algorithm!r}, actual={self.algorithm!r}")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        device: Optional[torch.device | str] = None,
    ) -> "HoliTomDPCSpatialMergeRoute":
        required = {
            "center_indices",
            "assignment",
            "source_counts",
            "original_tokens",
            "output_tokens",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Serialized HoliTom DPC route is missing keys: {sorted(missing)}")
        original_tokens = payload["original_tokens"]
        output_tokens = payload["output_tokens"]
        if (
            isinstance(original_tokens, bool)
            or not isinstance(original_tokens, Integral)
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, Integral)
        ):
            raise TypeError("Serialized HoliTom DPC token counts must be integers")

        def _integer_tensor(value: Any, *, name: str) -> torch.Tensor:
            tensor = torch.as_tensor(value)
            if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
                raise TypeError(f"Serialized HoliTom DPC {name} must contain integers")
            return tensor.to(dtype=torch.long)

        route = cls(
            center_indices=_integer_tensor(payload["center_indices"], name="center_indices"),
            assignment=_integer_tensor(payload["assignment"], name="assignment"),
            source_counts=_integer_tensor(payload["source_counts"], name="source_counts"),
            original_tokens=int(original_tokens),
            output_tokens=int(output_tokens),
            schema_version=payload.get("schema_version"),
            algorithm=payload.get("algorithm"),
            method=payload.get("method"),
            retention_bps=payload.get("retention_bps"),
            curriculum_completed_steps=payload.get("curriculum_completed_steps"),
            curriculum_schedule_sha256=payload.get("curriculum_schedule_sha256"),
            anchor_coordinates=(
                _integer_tensor(payload["anchor_coordinates"], name="anchor_coordinates")
                if payload.get("anchor_coordinates") is not None
                else None
            ),
        )
        if device is not None:
            route = route.to(device)
        route.validate()
        return route


# CDPruner name.  The serialized schema remains
# identical so existing rollout/replay code can consume new CDPruner routes.
CDPrunerRoute = DARTMergeRoute


def _stable_argsort(values: torch.Tensor, *, descending: bool) -> torch.Tensor:
    try:
        return torch.argsort(values, descending=descending, stable=True)
    except TypeError:  # pragma: no cover - compatibility for old torch releases
        # The index-sized perturbation only resolves exact ties and is computed
        # in float64 to avoid disturbing representable non-tied FP32 values.
        values64 = values.to(torch.float64)
        indices = torch.arange(values.numel(), device=values.device, dtype=torch.float64)
        eps = torch.finfo(torch.float64).eps
        adjusted = values64 - indices * eps if descending else values64 + indices * eps
        return torch.argsort(adjusted, descending=descending)


def _reduce_features(features: torch.Tensor, output_dim: int = 64) -> torch.Tensor:
    """Deterministically reduce the hidden dimension without learned weights."""

    if features.ndim != 2:
        raise ValueError(f"features must be rank-2, got {tuple(features.shape)}")
    hidden = int(features.shape[-1])
    if hidden <= 0:
        raise ValueError("features must have a positive hidden dimension")
    output_dim = min(int(output_dim), hidden)
    pad = (-hidden) % output_dim
    work = features.detach().to(torch.float32)
    if pad:
        work = F.pad(work, (0, pad))
    return work.reshape(work.shape[0], output_dim, -1).mean(dim=-1)


def _query_prototypes(query_embeds: torch.Tensor, *, max_prototypes: int = 8) -> torch.Tensor:
    """Pool ordered query tokens into at most eight deterministic prototypes."""

    reduced = _reduce_features(query_embeds)
    count = min(max_prototypes, int(reduced.shape[0]))
    if count <= 0:
        raise ValueError("At least one inference-visible query token is required")
    boundaries = torch.linspace(0, reduced.shape[0], count + 1, device=reduced.device).round().to(torch.long)
    prototypes = []
    for index in range(count):
        start, end = int(boundaries[index].item()), int(boundaries[index + 1].item())
        end = max(end, start + 1)
        prototypes.append(reduced[start:end].mean(dim=0))
    return F.normalize(torch.stack(prototypes), p=2, dim=-1, eps=1e-12)


def _normalized_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(f"image coordinates must be [tokens, 3], got {tuple(coordinates.shape)}")
    work = coordinates.detach().to(torch.float32)
    low = work.amin(dim=0, keepdim=True)
    high = work.amax(dim=0, keepdim=True)
    return (work - low) / (high - low).clamp_min(1.0)


def _even_prototype_indices(indices: torch.Tensor, maximum: int) -> torch.Tensor:
    if indices.numel() <= maximum:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, maximum, device=indices.device).round().to(torch.long)
    return indices.index_select(0, positions)


def _exact_cdpruner_budget(num_tokens: int, retention_bps: int = 500) -> int:
    """Return the exact integer-basis-point CDPruner budget for one image."""

    from verl.models.transformers.visual_token_curriculum import visual_token_budget

    return visual_token_budget(
        num_tokens,
        retention_bps=retention_bps,
        minimum_tokens=CDPRUNER_MINIMUM_TOKENS,
    )


def validate_cdpruner_curriculum_route(
    route: DARTMergeRoute,
    runtime_state: Mapping[str, Any],
    *,
    curriculum=None,
    summary_enabled: bool = False,
    summary_weight_rule: str = "softmax_cosine_to_anchor",
    learnable_merge_enabled: bool = False,
) -> int:
    """Validate one V2 route against the sole active optimizer-step state.

    This is shared by rollout and actor replay. Keeping the integer budget and the three
    schedule fields here prevents a stale fixed-5% assertion from silently
    diverging from the optimizer-step curriculum.
    """

    if not isinstance(route, DARTMergeRoute):
        raise TypeError("curriculum route validation requires a DARTMergeRoute")
    if not isinstance(runtime_state, Mapping):
        raise TypeError("curriculum runtime state must be a mapping")
    if summary_weight_rule not in {"softmax_cosine_to_anchor", CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE}:
        raise ValueError("curriculum summary weight rule is unsupported")
    from verl.models.transformers.visual_token_curriculum import (
        VisualTokenCurriculum,
        visual_token_budget,
    )

    schedule = (
        VisualTokenCurriculum() if curriculum is None else
        VisualTokenCurriculum.from_mapping(curriculum) if isinstance(curriculum, Mapping) else curriculum
    )
    required = set(schedule.runtime_state(0))
    if set(runtime_state) != required:
        raise ValueError(
            "curriculum runtime state has a non-canonical field inventory: "
            f"expected={sorted(required)}, actual={sorted(runtime_state)}"
        )
    completed = runtime_state.get("completed_optimizer_steps")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, Integral)
        or int(completed) < 0
    ):
        raise ValueError("curriculum runtime-state completed step is invalid")
    expected_state = schedule.runtime_state(int(completed))
    if dict(runtime_state) != expected_state:
        raise ValueError(
            "curriculum runtime-state schedule/control boundary drift: "
            f"expected={expected_state!r}, actual={dict(runtime_state)!r}"
        )
    retention_bps = expected_state["retention_bps"]
    minimum_tokens = expected_state["minimum_tokens_per_image"]
    schedule_sha256 = expected_state["schedule_sha256"]
    route.validate_for_algorithm(
        CDPRUNER_ALGORITHM,
        expected_schema_version=_cdpruner_route_schema(
            summary_enabled=summary_enabled,
            learnable_summary_enabled=(summary_weight_rule == CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE),
            curriculum_enabled=True,
            learnable_merge_enabled=learnable_merge_enabled,
        ),
    )
    expected_metadata = (int(retention_bps), int(completed), schedule_sha256)
    actual_metadata = (
        route.retention_bps,
        route.curriculum_completed_steps,
        route.curriculum_schedule_sha256,
    )
    if actual_metadata != expected_metadata:
        raise ValueError(
            "curriculum CDPruner route state drift: "
            f"expected={expected_metadata!r}, actual={actual_metadata!r}"
        )
    expected_budget = visual_token_budget(
        route.original_tokens,
        retention_bps=int(retention_bps),
        minimum_tokens=int(minimum_tokens),
    )
    expected_budget += int(summary_enabled and expected_budget < route.original_tokens)
    if route.output_tokens != expected_budget:
        raise ValueError(
            "curriculum CDPruner route budget drift: "
            f"N={route.original_tokens}, K={route.output_tokens}, expected={expected_budget}, "
            f"retention_bps={retention_bps}"
        )
    return expected_budget


def validate_holitom_curriculum_route(
    route: HoliTomDPCSpatialMergeRoute,
    runtime_state: Mapping[str, Any],
    *,
    curriculum,
) -> int:
    """Validate replay against the configured schedule, never a default horizon."""
    from verl.models.transformers.visual_token_curriculum import VisualTokenCurriculum, visual_token_budget

    if not isinstance(route, HoliTomDPCSpatialMergeRoute):
        raise TypeError("HoliTom curriculum requires a HoliTom DPC route")
    route.validate_for_algorithm(HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM)
    schedule = VisualTokenCurriculum.from_mapping(curriculum) if isinstance(curriculum, Mapping) else curriculum
    if not isinstance(schedule, VisualTokenCurriculum):
        raise TypeError("HoliTom curriculum validation requires the configured schedule")
    if not isinstance(runtime_state, Mapping):
        raise TypeError("HoliTom curriculum runtime state must be a mapping")
    completed = runtime_state.get("completed_optimizer_steps")
    if isinstance(completed, bool) or not isinstance(completed, Integral):
        raise ValueError("HoliTom curriculum completed step must be an integer")
    expected = schedule.runtime_state(int(completed))
    if dict(runtime_state) != expected:
        raise ValueError("HoliTom curriculum runtime state does not match its configured schedule")
    actual_metadata = (route.retention_bps, route.curriculum_completed_steps, route.curriculum_schedule_sha256)
    expected_metadata = (expected["retention_bps"], completed, expected["schedule_sha256"])
    if actual_metadata != expected_metadata:
        raise ValueError(f"HoliTom curriculum replay state drift: expected={expected_metadata!r}, actual={actual_metadata!r}")
    budget = visual_token_budget(route.original_tokens, retention_bps=expected["retention_bps"], minimum_tokens=expected["minimum_tokens_per_image"])
    if route.output_tokens != budget:
        raise ValueError(f"HoliTom curriculum replay budget drift: {route.output_tokens} != {budget}")
    return budget


def _exact_holitom_dpc_budget(num_tokens: int, retention_bps: int = 500) -> int:
    """Return the exact integer-basis-point per-image merge budget."""
    return _exact_cdpruner_budget(num_tokens, retention_bps=retention_bps)


def _scaled_euclidean_block(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute the FP32 Euclidean/sqrt(D) distance for one block."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[-1] != right.shape[-1]:
        raise ValueError("distance operands must be rank-2 with a shared hidden dimension")
    hidden_size = int(left.shape[-1])
    if hidden_size <= 0:
        raise ValueError("distance operands must have a positive hidden dimension")
    # The non-matmul implementation evaluates every pair with the same direct
    # FP32 reduction independent of chunk shape.  This is slower than the GEMM
    # shortcut but gives reference/chunk route parity and stable exact ties.
    return torch.cdist(
        left,
        right,
        p=2.0,
        compute_mode="donot_use_mm_for_euclid_dist",
    ) / math.sqrt(hidden_size)


def _dpc_density_and_delta(
    features: torch.Tensor,
    *,
    knn_k: int,
    distance_chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact DPC-KNN density and nearest-higher distance by blocks.

    The largest persistent tensors are ``O(N*k)`` and ``O(N)``.  A pairwise
    distance block is released before the next block, so an ``N x N`` matrix is
    never retained.  KNN includes the source token itself, exactly matching the
    ``k=min(7, N)`` rule.
    """

    if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] <= 0:
        raise ValueError("DPC features must be a non-empty [tokens, hidden] tensor")
    num_tokens = int(features.shape[0])
    if knn_k != min(7, num_tokens):
        raise ValueError(f"HoliTom DPC knn_k must equal min(7, N), got {knn_k} for N={num_tokens}")
    if distance_chunk_tokens <= 0:
        raise ValueError("distance_chunk_tokens must be positive")

    nearest_distances = features.new_full((num_tokens, knn_k), torch.inf)
    # HoliTom Eq. (5) defines the no-higher-density fallback independently
    # for every token: delta_i=max_{j != i} d(v_i, v_j).  A single global
    # pairwise maximum is not equivalent when several tokens share the
    # maximum density, and can change the stable center ranking.
    row_max_distances = features.new_zeros((num_tokens,))
    for query_start in range(0, num_tokens, distance_chunk_tokens):
        query_end = min(num_tokens, query_start + distance_chunk_tokens)
        query = features[query_start:query_end]
        local_nearest = features.new_full((query_end - query_start, knn_k), torch.inf)
        local_row_max = features.new_zeros((query_end - query_start,))
        for candidate_start in range(0, num_tokens, distance_chunk_tokens):
            candidate_end = min(num_tokens, candidate_start + distance_chunk_tokens)
            distances = _scaled_euclidean_block(query, features[candidate_start:candidate_end])
            local_row_max = torch.maximum(local_row_max, distances.amax(dim=-1))
            candidates = torch.cat((local_nearest, distances), dim=-1)
            local_nearest = torch.topk(
                candidates,
                k=knn_k,
                dim=-1,
                largest=False,
                sorted=True,
            ).values
        nearest_distances[query_start:query_end] = local_nearest
        row_max_distances[query_start:query_end] = local_row_max

    density = torch.exp(-nearest_distances.square().mean(dim=-1))
    if not torch.isfinite(density).all().item():
        raise FloatingPointError("HoliTom DPC density contains NaN or Inf")

    delta = features.new_full((num_tokens,), torch.inf)
    has_higher = torch.zeros(num_tokens, device=features.device, dtype=torch.bool)
    for query_start in range(0, num_tokens, distance_chunk_tokens):
        query_end = min(num_tokens, query_start + distance_chunk_tokens)
        query = features[query_start:query_end]
        query_density = density[query_start:query_end]
        local_delta = features.new_full((query_end - query_start,), torch.inf)
        local_has_higher = torch.zeros(query_end - query_start, device=features.device, dtype=torch.bool)
        for candidate_start in range(0, num_tokens, distance_chunk_tokens):
            candidate_end = min(num_tokens, candidate_start + distance_chunk_tokens)
            distances = _scaled_euclidean_block(query, features[candidate_start:candidate_end])
            higher = density[candidate_start:candidate_end].unsqueeze(0) > query_density.unsqueeze(1)
            masked = distances.masked_fill(~higher, torch.inf)
            block_delta = masked.amin(dim=-1)
            improves = block_delta < local_delta
            local_delta = torch.where(improves, block_delta, local_delta)
            local_has_higher |= higher.any(dim=-1)
        delta[query_start:query_end] = torch.where(
            local_has_higher,
            local_delta,
            row_max_distances[query_start:query_end],
        )
        has_higher[query_start:query_end] = local_has_higher

    if not torch.isfinite(delta).all().item():
        raise FloatingPointError("HoliTom DPC nearest-higher delta contains NaN or Inf")
    return density, delta


def _nearest_center_assignment_euclidean(
    features: torch.Tensor,
    center_indices: torch.Tensor,
    *,
    distance_chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign every non-center source to its exact nearest selected center."""

    if distance_chunk_tokens <= 0:
        raise ValueError("distance_chunk_tokens must be positive")
    num_tokens = int(features.shape[0])
    output_tokens = int(center_indices.numel())
    if output_tokens <= 0:
        raise ValueError("At least one DPC center is required")
    centers = features.index_select(0, center_indices)
    assignment = torch.empty(num_tokens, device=features.device, dtype=torch.long)

    for query_start in range(0, num_tokens, distance_chunk_tokens):
        query_end = min(num_tokens, query_start + distance_chunk_tokens)
        query = features[query_start:query_end]
        best_distance = features.new_full((query_end - query_start,), torch.inf)
        best_slot = torch.zeros(query_end - query_start, device=features.device, dtype=torch.long)
        for center_start in range(0, output_tokens, distance_chunk_tokens):
            center_end = min(output_tokens, center_start + distance_chunk_tokens)
            distances = _scaled_euclidean_block(query, centers[center_start:center_end])
            block_distance, block_offset = distances.min(dim=-1)
            # Center indices are sorted.  Strict improvement therefore keeps
            # the lower original center index on an exact distance tie.
            improves = block_distance < best_distance
            best_distance = torch.where(improves, block_distance, best_distance)
            candidate_slot = block_offset.to(torch.long) + center_start
            best_slot = torch.where(improves, candidate_slot, best_slot)
        assignment[query_start:query_end] = best_slot

    assignment[center_indices] = torch.arange(output_tokens, device=features.device, dtype=torch.long)
    source_counts = torch.bincount(assignment, minlength=output_tokens).to(torch.long)
    return assignment, source_counts


@torch.no_grad()
def build_holitom_dpc_spatial_merge_route(
    image_embeds: torch.Tensor,
    *,
    image_coordinates: torch.Tensor,
    distance_chunk_tokens: int = 256,
    retention_bps: int = 500,
    curriculum_completed_steps: Optional[int] = None,
    curriculum_schedule_sha256: Optional[str] = None,
) -> HoliTomDPCSpatialMergeRoute:
    """Build the HoliTom-inspired DPC merge route."""

    if image_embeds.ndim != 2 or image_embeds.shape[0] <= 0 or image_embeds.shape[1] <= 0:
        raise ValueError("image_embeds must be a non-empty [tokens, hidden] tensor")
    if not torch.isfinite(image_embeds).all().item():
        raise ValueError("image_embeds contains NaN or Inf")
    num_tokens = int(image_embeds.shape[0])
    if image_coordinates.shape != (num_tokens, 3):
        raise ValueError("image_coordinates must be [image_tokens, 3]")
    if image_coordinates.is_floating_point() or image_coordinates.is_complex():
        raise TypeError("image_coordinates must contain integer M-RoPE coordinates")
    if not torch.isfinite(image_coordinates).all().item():
        raise ValueError("image_coordinates contains NaN or Inf")
    if distance_chunk_tokens <= 0:
        raise ValueError("distance_chunk_tokens must be positive")

    curriculum_enabled = curriculum_completed_steps is not None or curriculum_schedule_sha256 is not None
    if curriculum_enabled and (curriculum_completed_steps is None or curriculum_schedule_sha256 is None):
        raise ValueError("HoliTom curriculum route construction requires both completed steps and schedule hash")
    if not curriculum_enabled and retention_bps != 500:
        raise ValueError("Static HoliTom routes require the fixed 5% budget")
    identity = {}
    if curriculum_enabled:
        identity = dict(
            schema_version=HOLITOM_DPC_CURRICULUM_MERGE_ROUTE_SCHEMA,
            algorithm=HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM,
            retention_bps=retention_bps,
            curriculum_completed_steps=curriculum_completed_steps,
            curriculum_schedule_sha256=curriculum_schedule_sha256,
        )
    output_tokens = _exact_holitom_dpc_budget(num_tokens, retention_bps=retention_bps)
    coordinates = image_coordinates.detach().to(device=image_embeds.device, dtype=torch.long)
    if output_tokens == num_tokens:
        center_indices = torch.arange(num_tokens, device=image_embeds.device, dtype=torch.long)
        route = HoliTomDPCSpatialMergeRoute(
            center_indices=center_indices,
            assignment=center_indices.clone(),
            source_counts=torch.ones(num_tokens, device=image_embeds.device, dtype=torch.long),
            original_tokens=num_tokens,
            output_tokens=num_tokens,
            anchor_coordinates=coordinates,
            **identity,
        )
        route.validate()
        return route

    features = image_embeds.detach().to(torch.float32)
    density, delta = _dpc_density_and_delta(
        features,
        knn_k=min(7, num_tokens),
        distance_chunk_tokens=distance_chunk_tokens,
    )
    center_score = density * delta
    # Stable descending sort gives the lower original index precedence for an
    # exact score tie.  Output is then restored to visual sequence order.
    center_indices = torch.sort(
        _stable_argsort(center_score, descending=True)[:output_tokens]
    ).values.to(torch.long)
    if center_indices.numel() != output_tokens or torch.unique(center_indices).numel() != output_tokens:
        raise RuntimeError("HoliTom DPC did not produce the exact unique center budget")

    assignment, source_counts = _nearest_center_assignment_euclidean(
        features,
        center_indices,
        distance_chunk_tokens=distance_chunk_tokens,
    )
    route = HoliTomDPCSpatialMergeRoute(
        center_indices=center_indices,
        assignment=assignment,
        source_counts=source_counts,
        original_tokens=num_tokens,
        output_tokens=output_tokens,
        anchor_coordinates=coordinates.index_select(0, center_indices),
        **identity,
    )
    route.validate()
    return route


def _cdpruner_features_and_quality(
    image_embeds: torch.Tensor,
    query_embeds: torch.Tensor,
    *,
    relevance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CDPruner cosine features and min-max-normalized query quality.

    CDPruner Eq. (3) uses cosine similarity between post-projector visual
    tokens.  Equations (5)--(6) use cosine relevance to the mean instruction
    embedding followed by min-max normalization.  Qwen has no paired text
    tower, so both operands intentionally live in the LLM embedding space.
    """

    if image_embeds.ndim != 2:
        raise ValueError(f"image_embeds must be [tokens, hidden], got {tuple(image_embeds.shape)}")
    if query_embeds.ndim != 2 or query_embeds.shape[-1] != image_embeds.shape[-1]:
        raise ValueError("query_embeds must be [query_tokens, hidden] with the same hidden size as image_embeds")
    if image_embeds.shape[0] <= 0:
        raise ValueError("image_embeds must contain at least one token")
    if query_embeds.shape[0] <= 0:
        raise ValueError("query_embeds must be non-empty")
    if relevance_epsilon <= 0.0:
        raise ValueError("relevance_epsilon must be positive")
    if not torch.isfinite(image_embeds).all().item():
        raise ValueError("image_embeds contains NaN or Inf")
    if not torch.isfinite(query_embeds).all().item():
        raise ValueError("query_embeds contains NaN or Inf")

    visual = F.normalize(image_embeds.detach().to(torch.float32), p=2, dim=-1, eps=relevance_epsilon)
    mean_query = query_embeds.detach().to(device=image_embeds.device, dtype=torch.float32).mean(dim=0)
    mean_query = F.normalize(mean_query, p=2, dim=0, eps=relevance_epsilon)
    raw_relevance = torch.mv(visual, mean_query)
    relevance_min = raw_relevance.amin()
    relevance_range = raw_relevance.amax() - relevance_min
    normalized_relevance = (raw_relevance - relevance_min) / relevance_range.clamp_min(relevance_epsilon)

    # A zero relevance range contains no query-discriminative information.  In
    # that case CDPruner should reduce to an unconditional diversity DPP, not a
    # near-zero kernel in which diagonal jitter determines every choice.
    quality = torch.where(
        relevance_range <= relevance_epsilon,
        torch.ones_like(normalized_relevance),
        normalized_relevance,
    )
    return visual, quality.clamp(min=relevance_epsilon, max=1.0)


def _fast_greedy_conditional_dpp(
    normalized_visual: torch.Tensor,
    quality: torch.Tensor,
    *,
    output_tokens: int,
    kernel_jitter: float,
    residual_epsilon: float,
) -> torch.Tensor:
    """Greedy Cholesky MAP for ``diag(q) @ (V V.T) @ diag(q)``.

    This is CDPruner Appendix-A's update, but each selected kernel row is
    generated as a matrix-vector product.  The implementation therefore keeps
    ``O(ND + NK)`` workspace rather than materializing an ``N x N`` similarity
    or kernel matrix.  A small diagonal jitter makes the PSD kernel full-rank
    when visual tokens are duplicated or ``K`` exceeds the feature rank.
    """

    if normalized_visual.ndim != 2:
        raise ValueError("normalized_visual must be rank-2")
    num_tokens = int(normalized_visual.shape[0])
    if quality.shape != (num_tokens,):
        raise ValueError("quality must contain one value per visual token")
    if not 0 < output_tokens <= num_tokens:
        raise ValueError("output_tokens must be in [1, num_tokens]")
    if kernel_jitter <= 0.0 or residual_epsilon <= 0.0:
        raise ValueError("kernel_jitter and residual_epsilon must be positive")

    # C[k, i] stores the incremental Cholesky coordinate c_i[k].  It is the
    # only N-by-budget workspace and is bounded by the requested output size.
    cholesky_coordinates = normalized_visual.new_zeros((output_tokens, num_tokens))
    # Use the actual feature norm rather than assuming normalization produced
    # an exact unit vector.  This keeps the matrix-free diagonal bit-for-bit
    # consistent with the represented kernel even for zero or subnormal rows.
    feature_norm_squared = normalized_visual.square().sum(dim=-1)
    residual_diagonal = quality.square() * feature_norm_squared + float(kernel_jitter)
    selected_mask = torch.zeros(num_tokens, device=normalized_visual.device, dtype=torch.bool)
    selected = torch.empty(output_tokens, device=normalized_visual.device, dtype=torch.long)

    for step in range(output_tokens):
        # torch.argmax returns the first maximum, providing an index-stable tie
        # break.  Selected entries are always masked to -Inf below.
        pivot = torch.argmax(residual_diagonal)
        selected[step] = pivot
        selected_mask[pivot] = True
        if step + 1 == output_tokens:
            break

        similarity_row = torch.mv(normalized_visual, normalized_visual[pivot])
        kernel_row = quality * quality[pivot] * similarity_row
        kernel_row[pivot] += float(kernel_jitter)
        if step:
            projection = torch.mv(
                cholesky_coordinates[:step].transpose(0, 1),
                cholesky_coordinates[:step, pivot],
            )
            kernel_row = kernel_row - projection

        pivot_scale = residual_diagonal[pivot].clamp_min(residual_epsilon).sqrt()
        new_coordinate = torch.nan_to_num(kernel_row / pivot_scale, nan=0.0, posinf=0.0, neginf=0.0)
        cholesky_coordinates[step] = new_coordinate
        residual_diagonal = (residual_diagonal - new_coordinate.square()).clamp_min(0.0)
        residual_diagonal.masked_fill_(selected_mask, -torch.inf)

    return torch.sort(selected).values


def _validate_cdpruner_summary_config(summary: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if summary is None:
        return {"enabled": False}
    if not isinstance(summary, Mapping):
        raise TypeError("CDPruner summary configuration must be a mapping")
    allowed = {
        "enabled", "knn_k", "distance", "anchor_rule", "weight_rule", "temperature",
        "hidden_size", "input_hidden_size", "residual_logit_limit",
        "activation_checkpointing", "chunk_tokens",
    }
    if set(summary).difference(allowed):
        raise ValueError("CDPruner summary configuration contains unsupported fields")
    enabled = summary.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("CDPruner summary enabled must be boolean")
    if not enabled:
        if set(summary).difference({"enabled"}):
            raise ValueError("Disabled CDPruner summary must not contain active settings")
        return {"enabled": False}
    exact = {
        "enabled": True,
        "knn_k": 7,
        "distance": "cosine",
        "anchor_rule": "max_mean_knn_similarity_excluding_self",
        "temperature": 1.0,
    }
    weight_rule = summary.get("weight_rule", "softmax_cosine_to_anchor")
    if weight_rule not in {"softmax_cosine_to_anchor", CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE}:
        raise ValueError(f"Unsupported CDPruner summary weight_rule: {weight_rule!r}")
    exact["weight_rule"] = weight_rule
    if weight_rule == CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE:
        exact.update({"hidden_size": 64, "residual_logit_limit": 4.0})
        checkpointing = summary.get("activation_checkpointing", True)
        if not isinstance(checkpointing, bool):
            raise TypeError("Learnable summary activation_checkpointing must be boolean")
        chunk_tokens = summary.get("chunk_tokens", 1024)
        if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, Integral) or chunk_tokens <= 0:
            raise ValueError("Learnable summary chunk_tokens must be a positive integer")
        exact.update({"activation_checkpointing": checkpointing, "chunk_tokens": int(chunk_tokens)})
        hidden_size = summary.get("hidden_size", 64)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, Integral):
            raise TypeError("Learnable summary hidden_size must be integer 64")
        if isinstance(summary.get("residual_logit_limit", 4.0), bool):
            raise TypeError("Learnable summary residual_logit_limit must be numeric 4.0")
        if "input_hidden_size" in summary:
            input_hidden_size = summary["input_hidden_size"]
            if (
                isinstance(input_hidden_size, bool)
                or not isinstance(input_hidden_size, Integral)
                or input_hidden_size <= 0
            ):
                raise ValueError("Learnable summary input_hidden_size must be a positive integer from model config")
            exact["input_hidden_size"] = int(input_hidden_size)
    elif set(summary).intersection({
        "hidden_size", "input_hidden_size", "residual_logit_limit", "activation_checkpointing", "chunk_tokens",
    }):
        raise ValueError("Fixed cosine summary does not accept learnable MLP settings")
    if isinstance(summary.get("knn_k", 7), bool) or not isinstance(summary.get("knn_k", 7), Integral):
        raise TypeError("Summary knn_k must be integer 7")
    if isinstance(summary.get("temperature", 1.0), bool):
        raise TypeError("Summary temperature must be numeric 1.0")
    for key, expected in exact.items():
        if summary.get(key, expected) != expected:
            raise ValueError(f"CDPruner summary {key} must be {expected!r}")
    return exact


def _cdpruner_route_schema(
    *,
    summary_enabled: bool,
    learnable_summary_enabled: bool,
    curriculum_enabled: bool,
    learnable_merge_enabled: bool = False,
) -> str:
    if learnable_merge_enabled:
        if summary_enabled or learnable_summary_enabled:
            raise ValueError("Learnable merge cannot add a separate summary token")
        return (
            CDPRUNER_LEARNABLE_MERGE_CURRICULUM_ROUTE_SCHEMA_VERSION
            if curriculum_enabled else CDPRUNER_LEARNABLE_MERGE_ROUTE_SCHEMA_VERSION
        )
    if learnable_summary_enabled:
        if not summary_enabled:
            raise ValueError("Learnable summary requires the summary token to be enabled")
        return (
            CDPRUNER_LEARNABLE_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION
            if curriculum_enabled else CDPRUNER_LEARNABLE_SUMMARY_ROUTE_SCHEMA_VERSION
        )
    if summary_enabled:
        return (
            CDPRUNER_SUMMARY_CURRICULUM_ROUTE_SCHEMA_VERSION
            if curriculum_enabled else CDPRUNER_SUMMARY_ROUTE_SCHEMA_VERSION
        )
    return CDPRUNER_CURRICULUM_ROUTE_SCHEMA_VERSION if curriculum_enabled else CDPRUNER_ROUTE_SCHEMA_VERSION


class NoRandomLinear(nn.Linear):
    """Allocate linear parameters without consuming any random-number state."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class LearnableResidualSummaryMLP(nn.Module):
    """Learn a bounded log-weight correction around a detached cosine prior.

    The first linear layer acts on [LN(source), LN(anchor)], but its weight
    halves are applied separately to avoid copying a repeated anchor N times.
    There are precisely three parameter tensors, created in FP32. Explicit
    FP32 operations also cover an FSDP/replica forward with BF16 parameters.
    """

    def __init__(self, input_hidden_size: int, *, hidden_size: int = 64, residual_logit_limit: float = 4.0):
        super().__init__()
        if isinstance(input_hidden_size, bool) or not isinstance(input_hidden_size, Integral) or input_hidden_size <= 0:
            raise ValueError("Learnable summary needs a positive model input_hidden_size")
        if hidden_size != 64 or residual_logit_limit != 4.0:
            raise ValueError("Learnable summary uses hidden_size=64 and residual_logit_limit=4.0")
        self.input_hidden_size = int(input_hidden_size)
        self.hidden_size = int(hidden_size)
        self.residual_logit_limit = float(residual_logit_limit)
        self.norm = nn.LayerNorm(self.input_hidden_size, elementwise_affine=False)
        self.input = NoRandomLinear(2 * self.input_hidden_size, self.hidden_size, bias=True, dtype=torch.float32)
        self.output = NoRandomLinear(self.hidden_size, 1, bias=False, dtype=torch.float32)
        # A scalar only: no extra parameters/buffers enter the checkpoint.
        # Zero placeholders may never be used as an active scoring head.
        self.ready = False

    def _require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(
                "Learnable aggregation head is uninitialized; initialize from loaded pretrained "
                "MLP gate rows or strictly restore a learned checkpoint before forward"
            )

    @torch.no_grad()
    def initialize_from_pretrained_projection(
        self, source_weight: torch.Tensor, source_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Use 64 uniformly spaced pretrained gate rows without any randomness.

        Source rows are selected before copying to CPU FP32. Their unit-norm
        projection is copied independently into both input halves / sqrt(2).
        The zero output layer preserves initial baseline aggregation weights.
        """
        if not isinstance(source_weight, torch.Tensor) or source_weight.ndim != 2:
            raise ValueError("Pretrained MLP gate source must be a loaded [intermediate, hidden] tensor")
        if source_weight.device.type == "meta":
            raise ValueError("Pretrained MLP gate source cannot be an unmaterialized meta tensor")
        rows, width = source_weight.shape
        if rows < self.hidden_size or width != self.input_hidden_size:
            raise ValueError("Pretrained MLP gate source shape does not match the aggregation head")
        if not source_weight.is_floating_point() or not torch.isfinite(source_weight).all().item():
            raise ValueError("Pretrained MLP gate source must contain finite floating-point weights")
        if any(parameter.device.type == "meta" for parameter in self.parameters()):
            raise ValueError("Aggregation head must be materialized before pretrained-row initialization")
        row_indices = [index * (rows - 1) // (self.hidden_size - 1) for index in range(self.hidden_size)]
        indices = torch.tensor(row_indices, device=source_weight.device, dtype=torch.long)
        selected_rows = source_weight.detach().index_select(0, indices).to(device="cpu", dtype=torch.float32)
        norms = selected_rows.norm(p=2, dim=1, keepdim=True)
        if not torch.isfinite(selected_rows).all().item() or not torch.isfinite(norms).all().item() or torch.any(norms <= 0).item():
            raise ValueError("Selected pretrained MLP gate rows must have finite non-zero FP32 norms")
        projection = (selected_rows / norms) / math.sqrt(2.0)
        self.input.weight[:, : self.input_hidden_size].copy_(projection)
        self.input.weight[:, self.input_hidden_size :].copy_(projection)
        self.input.bias.zero_()
        self.output.weight.zero_()
        self.ready = True
        return {
            "source_name": source_name,
            "row_indices": row_indices,
            "rule": CDPRUNER_LEARNABLE_MERGE_INITIALIZATION,
        }

    @torch.no_grad()
    def mark_checkpoint_loaded(self) -> None:
        """Enable a strictly restored head after checking its three tensors.

        Trained weights need not retain the initialization's normalized form.
        The caller must perform strict state restoration before this method.
        """
        self.ready = False
        parameters = tuple(self.parameters())
        if len(parameters) != 3 or any(parameter.device.type == "meta" for parameter in parameters):
            raise ValueError("Restored aggregation head must contain three materialized parameter tensors")
        if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
            raise ValueError("Restored aggregation head contains non-finite weights")
        if not torch.any(self.input.weight != 0).item():
            raise ValueError("Restored aggregation input projection cannot be entirely zero")
        self.ready = True

    def forward(
        self, sources: torch.Tensor, anchor: torch.Tensor, baseline_prior: torch.Tensor,
    ) -> torch.Tensor:
        self._require_ready()
        if baseline_prior.shape != (sources.shape[0],):
            raise ValueError("Learnable summary baseline prior shape mismatch")
        with torch.autocast(device_type=sources.device.type, enabled=False):
            delta = self.residual_logits(sources, anchor)
            limit = self.residual_logit_limit
            prior_log = torch.log(baseline_prior.detach().float())
            alpha = torch.softmax(prior_log + limit * torch.tanh(delta / limit), dim=0, dtype=torch.float32)
        return alpha

    def residual_logits(self, sources: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Return unbounded correction logits for one source chunk in FP32."""
        self._require_ready()
        if sources.ndim != 2 or sources.shape[-1] != self.input_hidden_size:
            raise ValueError("Learnable summary source features do not match model hidden size")
        if anchor.shape != (self.input_hidden_size,):
            raise ValueError("Learnable summary anchor shape mismatch")
        with torch.autocast(device_type=sources.device.type, enabled=False):
            source_norm = self.norm(sources.float())
            anchor_norm = self.norm(anchor.float())
            input_weight = self.input.weight.float()
            source_hidden = F.linear(source_norm, input_weight[:, : self.input_hidden_size], self.input.bias.float())
            anchor_hidden = F.linear(anchor_norm, input_weight[:, self.input_hidden_size :])
            delta = F.linear(F.silu(source_hidden + anchor_hidden), self.output.weight.float()).squeeze(-1)
        return delta

    def summarize(
        self,
        image_embeds: torch.Tensor,
        source_indices: torch.Tensor,
        anchor_index: int,
        baseline_prior: torch.Tensor,
        *,
        chunk_tokens: int = 1024,
        activation_checkpointing: bool = True,
    ) -> torch.Tensor:
        """Recompute a differentiable summary with bounded FP32 workspaces.

        Global softmax retains all discarded sources. Chunking avoids a full
        FP32 N-by-D copy; non-reentrant checkpoint saves just the existing
        image tensor and recomputes head activations and weighted sum in bwd.
        """
        self._require_ready()
        if image_embeds.ndim != 2 or image_embeds.shape[-1] != self.input_hidden_size:
            raise ValueError("Learnable summary features do not match model hidden size")

        def reduce_current_features(features: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=features.device.type, enabled=False):
                anchor = features[anchor_index]
                logits = []
                for start in range(0, source_indices.numel(), chunk_tokens):
                    indices = source_indices[start : start + chunk_tokens]
                    source_chunk = features.index_select(0, indices)
                    logits.append(self.residual_logits(source_chunk, anchor))
                delta = torch.cat(logits)
                limit = self.residual_logit_limit
                alpha = torch.softmax(
                    torch.log(baseline_prior.detach().float()) + limit * torch.tanh(delta / limit),
                    dim=0, dtype=torch.float32,
                )
                summary = features.new_zeros((self.input_hidden_size,), dtype=torch.float32)
                for start in range(0, source_indices.numel(), chunk_tokens):
                    indices = source_indices[start : start + chunk_tokens]
                    source_chunk = features.index_select(0, indices).float()
                    summary = summary + torch.matmul(alpha[start : start + indices.numel()], source_chunk)
                return summary

        needs_grad = image_embeds.requires_grad or any(parameter.requires_grad for parameter in self.parameters())
        if activation_checkpointing and torch.is_grad_enabled() and needs_grad:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(reduce_current_features, image_embeds, use_reentrant=False, preserve_rng_state=False)
        return reduce_current_features(image_embeds)

    def merge_nodes(
        self,
        image_embeds: torch.Tensor,
        selected_indices: torch.Tensor,
        assignment: torch.Tensor,
        baseline_prior: torch.Tensor,
        *,
        chunk_tokens: int = 1024,
        activation_checkpointing: bool = True,
    ) -> torch.Tensor:
        """Add bounded signed discarded residuals to exactly K retained anchors.

        Anchors are normalized/projected once, then gathered in each source
        chunk. Discarded priors sum to one per group and anchor priors are zero.
        A zero output layer makes every effective residual weight exactly zero.
        K-by-64 anchor projections persist;
        source FP32 copies and weighted N-by-D intermediates remain chunked.
        The complete differentiable operation is checkpointed for training.
        """
        self._require_ready()
        if image_embeds.ndim != 2 or image_embeds.shape[-1] != self.input_hidden_size:
            raise ValueError("Learnable merge features do not match model hidden size")
        output_tokens = selected_indices.numel()
        if output_tokens == image_embeds.shape[0]:
            return image_embeds.index_select(0, selected_indices)

        def reduce_current_features(features: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=features.device.type, enabled=False):
                anchors = features.index_select(0, selected_indices).float()
                anchor_norm = self.norm(anchors)
                weight = self.input.weight.float()
                anchor_hidden = F.linear(anchor_norm, weight[:, self.input_hidden_size :])
                correction = features.new_zeros((output_tokens, self.input_hidden_size), dtype=torch.float32)
                residual_mass = features.new_zeros((output_tokens,), dtype=torch.float32)
                for start in range(0, features.shape[0], chunk_tokens):
                    end = min(start + chunk_tokens, features.shape[0])
                    source_norm = self.norm(features[start:end].float())
                    source_hidden = F.linear(source_norm, weight[:, : self.input_hidden_size], self.input.bias.float())
                    hidden = source_hidden + anchor_hidden.index_select(0, assignment[start:end])
                    delta = F.linear(F.silu(hidden), self.output.weight.float()).squeeze(-1)
                    signed_weights = baseline_prior[start:end].detach().float() * torch.tanh(delta)
                    weighted_chunk = features[start:end].float() * signed_weights.unsqueeze(-1)
                    correction = correction.index_add(0, assignment[start:end], weighted_chunk)
                    residual_mass = residual_mass.index_add(0, assignment[start:end], signed_weights)
                # Equivalent to sum w_i * (x_i - anchor), without materializing
                # a source-chunk-by-D repeated-anchor tensor. This is signed
                # residual aggregation, not a convex weighted mean.
                output = anchors + correction - anchors * residual_mass.unsqueeze(-1)
                return output.to(features.dtype)

        needs_grad = image_embeds.requires_grad or any(parameter.requires_grad for parameter in self.parameters())
        if activation_checkpointing and torch.is_grad_enabled() and needs_grad:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(reduce_current_features, image_embeds, use_reentrant=False, preserve_rng_state=False)
        return reduce_current_features(image_embeds)


def _validate_cdpruner_learnable_merge_config(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if config is None:
        return {"enabled": False}
    if not isinstance(config, Mapping):
        raise TypeError("CDPruner learnable_merge configuration must be a mapping")
    allowed = {
        "enabled", "weight_rule", "input_hidden_size", "hidden_size",
        "temperature", "assignment", "activation_checkpointing", "chunk_tokens",
        "initialization", "aggregation",
    }
    if set(config).difference(allowed):
        raise ValueError("CDPruner learnable_merge configuration contains unsupported fields")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("CDPruner learnable_merge enabled must be boolean")
    if not enabled:
        if set(config).difference({"enabled"}):
            raise ValueError("Disabled learnable merge must not contain active settings")
        return {"enabled": False}
    if config.get("initialization") != CDPRUNER_LEARNABLE_MERGE_INITIALIZATION:
        raise ValueError(
            "Learnable merge requires explicit initialization='pretrained_mlp_gate_rows_zero_output_v1'"
        )
    if config.get("aggregation") != CDPRUNER_LEARNABLE_MERGE_AGGREGATION:
        raise ValueError("Learnable merge requires explicit aggregation='zero_init_signed_residual_v1'")
    exact = {
        "enabled": True,
        "weight_rule": CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE,
        "hidden_size": 64,
        "temperature": 1.0,
        "assignment": "cosine_nearest_retained_original_index_tie",
        "initialization": CDPRUNER_LEARNABLE_MERGE_INITIALIZATION,
        "aggregation": CDPRUNER_LEARNABLE_MERGE_AGGREGATION,
    }
    if isinstance(config.get("hidden_size", 64), bool) or not isinstance(config.get("hidden_size", 64), Integral):
        raise TypeError("Learnable merge hidden_size must be integer 64")
    for key in ("temperature",):
        if isinstance(config.get(key, exact[key]), bool):
            raise TypeError(f"Learnable merge {key} must be numeric")
    for key, expected in exact.items():
        if config.get(key, expected) != expected:
            raise ValueError(f"Learnable merge {key} must be {expected!r}")
    if "input_hidden_size" in config:
        size = config["input_hidden_size"]
        if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
            raise ValueError("Learnable merge input_hidden_size must be a positive model-config integer")
        exact["input_hidden_size"] = int(size)
    checkpointing = config.get("activation_checkpointing", True)
    if not isinstance(checkpointing, bool):
        raise TypeError("Learnable merge activation_checkpointing must be boolean")
    chunk_tokens = config.get("chunk_tokens", 1024)
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, Integral) or chunk_tokens <= 0:
        raise ValueError("Learnable merge chunk_tokens must be a positive integer")
    exact.update({"activation_checkpointing": checkpointing, "chunk_tokens": int(chunk_tokens)})
    return exact


@torch.no_grad()
def _build_cdpruner_learnable_merge_fields(
    normalized_visual: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route full-D cosine nearest retained, then construct discarded-only priors.

    Every source has one assignment. Retained anchors are forced to their own
    groups; argmax over original-index-sorted anchors resolves discarded ties.
    Similarity workspace is at most chunk_tokens-by-K, never N-by-N.
    """
    num_tokens = normalized_visual.shape[0]
    output_tokens = selected_indices.numel()
    anchors = normalized_visual.index_select(0, selected_indices)
    assignment = selected_indices.new_empty((num_tokens,))
    anchor_similarity = normalized_visual.new_empty((num_tokens,))
    with torch.autocast(device_type=normalized_visual.device.type, enabled=False):
        for start in range(0, num_tokens, chunk_tokens):
            end = min(start + chunk_tokens, num_tokens)
            similarity = torch.mm(normalized_visual[start:end].float(), anchors.float().T).clamp(-1.0, 1.0)
            scores, slots = similarity.max(dim=1)
            assignment[start:end] = slots
            anchor_similarity[start:end] = scores
        assignment[selected_indices] = torch.arange(output_tokens, device=assignment.device)
        retained_mask = torch.zeros(num_tokens, device=assignment.device, dtype=torch.bool)
        retained_mask[selected_indices] = True
        logits = (anchor_similarity / float(temperature)).masked_fill(retained_mask, -torch.inf)
        maxima = logits.new_full((output_tokens,), -torch.inf).scatter_reduce(
            0, assignment, logits, reduce="amax", include_self=True,
        )
        maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
        numerators = torch.exp(logits - maxima.index_select(0, assignment))
        denominators = logits.new_zeros((output_tokens,)).scatter_add(0, assignment, numerators)
        priors = numerators / denominators.index_select(0, assignment).clamp_min(1e-20)
        priors[selected_indices] = 0.0
    return assignment, priors.float()


@torch.no_grad()
def _build_cdpruner_summary_fields(
    normalized_visual: torch.Tensor,
    kept_indices: torch.Tensor,
    *,
    knn_k: int = 7,
    temperature: float = 1.0,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Select a real discarded anchor by cosine-kNN density, then weight all sources.

    Exact row chunks bound the temporary similarity workspace to 256 by the
    discarded count. Self-neighbors are excluded; index-order argmax resolves
    equal density. The anchor remains one of the positively weighted sources.
    """
    discarded_mask = torch.ones(normalized_visual.shape[0], device=normalized_visual.device, dtype=torch.bool)
    discarded_mask[kept_indices] = False
    sources = torch.nonzero(discarded_mask, as_tuple=False).flatten()
    if sources.numel() == 0:
        raise ValueError("A summary requires at least one discarded token")
    features = normalized_visual.index_select(0, sources)
    if sources.numel() == 1:
        return int(sources[0].item()), sources, features.new_ones((1,))
    neighbors = min(int(knn_k), int(sources.numel()) - 1)
    density = features.new_empty((features.shape[0],))
    for start in range(0, features.shape[0], 256):
        end = min(start + 256, features.shape[0])
        with torch.autocast(device_type=features.device.type, enabled=False):
            similarities = torch.mm(features[start:end], features.transpose(0, 1)).clamp(-1.0, 1.0)
        local_rows = torch.arange(end - start, device=features.device)
        similarities[local_rows, torch.arange(start, end, device=features.device)] = -torch.inf
        density[start:end] = torch.topk(similarities, k=neighbors, dim=-1, sorted=True).values.mean(dim=-1)
    anchor_slot = torch.argmax(density)
    similarity_to_anchor = torch.mv(features, features[anchor_slot]).clamp(-1.0, 1.0)
    weights = torch.softmax(similarity_to_anchor / float(temperature), dim=0)
    return int(sources[anchor_slot].item()), sources, weights


@torch.no_grad()
def build_cdpruner_route(
    image_embeds: torch.Tensor,
    *,
    query_embeds: torch.Tensor,
    image_coordinates: torch.Tensor,
    kernel_jitter: float = 1e-6,
    relevance_epsilon: float = 1e-6,
    residual_epsilon: float = 1e-12,
    retention_bps: int = 500,
    curriculum_completed_steps: Optional[int] = None,
    curriculum_schedule_sha256: Optional[str] = None,
    summary: Optional[Mapping[str, Any]] = None,
    learnable_merge: Optional[Mapping[str, Any]] = None,
) -> DARTMergeRoute:
    """Build the exact-budget Qwen CDPruner pure-index route.

    The selected subset greedily maximizes the determinant of CDPruner's
    instruction-conditioned DPP kernel.  Selection happens on post-merger Qwen
    embeddings; the returned indices are sorted before use so original token
    order and the corresponding M-RoPE coordinates remain unchanged.
    """

    if image_embeds.ndim != 2:
        raise ValueError(f"image_embeds must be [tokens, hidden], got {tuple(image_embeds.shape)}")
    num_tokens = int(image_embeds.shape[0])
    if num_tokens <= 0:
        raise ValueError("image_embeds must contain at least one token")
    if not torch.isfinite(image_embeds).all().item():
        raise ValueError("image_embeds contains NaN or Inf")
    if query_embeds.ndim != 2 or query_embeds.shape[-1] != image_embeds.shape[-1]:
        raise ValueError("query_embeds must be [query_tokens, hidden] with the same hidden size as image_embeds")
    if query_embeds.shape[0] <= 0 or not torch.isfinite(query_embeds).all().item():
        raise ValueError("query_embeds must be non-empty and finite")
    if image_coordinates.shape != (num_tokens, 3):
        raise ValueError("image_coordinates must be [image_tokens, 3]")
    if image_coordinates.is_floating_point() or image_coordinates.is_complex():
        raise TypeError("image_coordinates must contain integer M-RoPE coordinates")
    if not torch.isfinite(image_coordinates).all().item():
        raise ValueError("image_coordinates contains NaN or Inf")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (kernel_jitter, relevance_epsilon, residual_epsilon)
    ):
        raise ValueError("CDPruner numerical epsilons must be positive")

    summary_config = _validate_cdpruner_summary_config(summary)
    summary_enabled = summary_config["enabled"]
    merge_config = _validate_cdpruner_learnable_merge_config(learnable_merge)
    merge_enabled = merge_config["enabled"]
    if summary_enabled and merge_enabled:
        raise ValueError("CDPruner summary and learnable_merge cannot both be enabled")
    curriculum_enabled = curriculum_completed_steps is not None or curriculum_schedule_sha256 is not None
    if curriculum_enabled and (curriculum_completed_steps is None or curriculum_schedule_sha256 is None):
        raise ValueError("curriculum route construction requires both completed steps and schedule hash")
    route_schema_version = _cdpruner_route_schema(
        summary_enabled=summary_enabled,
        learnable_summary_enabled=(summary_config.get("weight_rule") == CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE),
        curriculum_enabled=curriculum_enabled,
        learnable_merge_enabled=merge_enabled,
    )
    output_tokens = _exact_cdpruner_budget(num_tokens, retention_bps=retention_bps)
    coordinates = image_coordinates.detach().to(device=image_embeds.device, dtype=torch.long)
    if output_tokens == num_tokens:
        selected_indices = torch.arange(num_tokens, device=image_embeds.device, dtype=torch.long)
        route = DARTMergeRoute(
            selected_indices=selected_indices,
            original_tokens=num_tokens,
            output_tokens=num_tokens,
            anchor_coordinates=coordinates,
            schema_version=route_schema_version,
            algorithm=CDPRUNER_ALGORITHM,
            method=CDPRUNER_METHOD,
            retention_bps=int(retention_bps) if curriculum_enabled else None,
            curriculum_completed_steps=(
                int(curriculum_completed_steps) if curriculum_enabled else None
            ),
            curriculum_schedule_sha256=(
                str(curriculum_schedule_sha256) if curriculum_enabled else None
            ),
            merge_assignment=(selected_indices.clone() if merge_enabled else None),
            merge_prior_weights=(image_embeds.new_zeros((num_tokens,), dtype=torch.float32) if merge_enabled else None),
        )
        route.validate()
        return route

    normalized_visual, quality = _cdpruner_features_and_quality(
        image_embeds,
        query_embeds,
        relevance_epsilon=relevance_epsilon,
    )
    selected_indices = _fast_greedy_conditional_dpp(
        normalized_visual,
        quality,
        output_tokens=output_tokens,
        kernel_jitter=kernel_jitter,
        residual_epsilon=residual_epsilon,
    )
    if selected_indices.numel() != output_tokens or torch.unique(selected_indices).numel() != output_tokens:
        raise RuntimeError("CDPruner did not produce the exact unique token budget")

    summary_anchor_index = None
    summary_source_indices = None
    summary_weights = None
    if summary_enabled:
        summary_anchor_index, summary_source_indices, summary_weights = _build_cdpruner_summary_fields(
            normalized_visual, selected_indices,
            knn_k=summary_config["knn_k"], temperature=summary_config["temperature"],
        )
        selected_indices = torch.sort(torch.cat([
            selected_indices, selected_indices.new_tensor([summary_anchor_index]),
        ])).values
        output_tokens += 1

    merge_assignment = None
    merge_prior_weights = None
    if merge_enabled:
        merge_assignment, merge_prior_weights = _build_cdpruner_learnable_merge_fields(
            normalized_visual, selected_indices,
            temperature=merge_config["temperature"], chunk_tokens=merge_config["chunk_tokens"],
        )

    route = DARTMergeRoute(
        selected_indices=selected_indices,
        original_tokens=num_tokens,
        output_tokens=output_tokens,
        anchor_coordinates=coordinates.index_select(0, selected_indices),
        schema_version=route_schema_version,
        algorithm=CDPRUNER_ALGORITHM,
        method=CDPRUNER_METHOD,
        retention_bps=int(retention_bps) if curriculum_enabled else None,
        curriculum_completed_steps=(int(curriculum_completed_steps) if curriculum_enabled else None),
        curriculum_schedule_sha256=(str(curriculum_schedule_sha256) if curriculum_enabled else None),
        summary_anchor_index=summary_anchor_index,
        summary_source_indices=summary_source_indices,
        summary_weights=summary_weights,
        merge_assignment=merge_assignment,
        merge_prior_weights=merge_prior_weights,
    )
    route.validate()
    return route


def build_dart_merge_route(
    image_embeds: torch.Tensor,
    *,
    query_embeds: torch.Tensor,
    image_coordinates: torch.Tensor,
    retention_ratio: float = 0.05,
    minimum_tokens: int = 32,
    relevance_fraction: float = 0.60,
    candidate_multiplier: int = 4,
) -> DARTMergeRoute:
    """Build an exact dynamic-budget query-relevance/diversity prune route.

    Sixty percent of the budget protects the most query-relevant tokens.  The
    remainder is selected from a bounded relevance/saliency candidate pool by
    eight rounds of semantic and spatial novelty scoring.  The implementation
    never constructs an ``N x N`` matrix.
    """

    if image_embeds.ndim != 2:
        raise ValueError(f"image_embeds must be [tokens, hidden], got {tuple(image_embeds.shape)}")
    num_tokens = int(image_embeds.shape[0])
    if num_tokens <= 0:
        raise ValueError("image_embeds must contain at least one token")
    if query_embeds.ndim != 2 or query_embeds.shape[-1] != image_embeds.shape[-1]:
        raise ValueError("query_embeds must be [query_tokens, hidden] with the same hidden size as image_embeds")
    if query_embeds.shape[0] <= 0 or not torch.isfinite(query_embeds).all().item():
        raise ValueError("query_embeds must be non-empty and finite")
    if image_coordinates.shape != (num_tokens, 3):
        raise ValueError("image_coordinates must be [image_tokens, 3]")
    if not 0.0 < retention_ratio <= 1.0:
        raise ValueError("retention_ratio must be in (0, 1]")
    if minimum_tokens <= 0:
        raise ValueError("minimum_tokens must be positive")
    if not 0.0 < relevance_fraction <= 1.0:
        raise ValueError("relevance_fraction must be in (0, 1]")
    if candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be at least one")
    if not torch.isfinite(image_embeds).all().item():
        raise ValueError("image_embeds contains NaN or Inf")

    output_tokens = min(num_tokens, max(int(minimum_tokens), int(torch.ceil(torch.tensor(num_tokens * retention_ratio)).item())))
    if output_tokens == num_tokens:
        indices = torch.arange(num_tokens, device=image_embeds.device, dtype=torch.long)
        route = DARTMergeRoute(
            selected_indices=indices,
            original_tokens=num_tokens,
            output_tokens=num_tokens,
        )
        route.validate()
        return route

    visual = _reduce_features(image_embeds)
    normalized = F.normalize(visual, p=2, dim=-1, eps=1e-12)
    query = _query_prototypes(query_embeds)
    relevance = (normalized @ query.transpose(0, 1)).amax(dim=-1)
    saliency = visual.norm(dim=-1)
    saliency = (saliency - saliency.amin()) / (saliency.amax() - saliency.amin()).clamp_min(1e-12)
    coordinates = _normalized_coordinates(image_coordinates)

    protected_count = min(output_tokens, max(1, int(torch.ceil(torch.tensor(output_tokens * relevance_fraction)).item())))
    protected = _stable_argsort(relevance, descending=True)[:protected_count]
    selected_mask = torch.zeros(num_tokens, device=image_embeds.device, dtype=torch.bool)
    selected_mask[protected] = True

    candidate_count = min(num_tokens, max(output_tokens, int(candidate_multiplier) * output_tokens, 256))
    candidate_base_score = 0.8 * relevance + 0.2 * saliency
    candidates = _stable_argsort(candidate_base_score, descending=True)[:candidate_count]
    selected = protected
    remaining = output_tokens - int(selected.numel())
    rounds_left = 8
    while remaining > 0:
        available = candidates[~selected_mask.index_select(0, candidates)]
        if available.numel() == 0:
            available = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        prototypes = _even_prototype_indices(selected, 64)
        semantic_novelty = 1.0 - (
            normalized.index_select(0, available)
            @ normalized.index_select(0, prototypes).transpose(0, 1)
        ).amax(dim=-1)
        spatial_novelty = torch.cdist(
            coordinates.index_select(0, available), coordinates.index_select(0, prototypes)
        ).amin(dim=-1)
        spatial_novelty = spatial_novelty / spatial_novelty.amax().clamp_min(1e-12)
        score = (
            0.35 * relevance.index_select(0, available)
            + 0.40 * semantic_novelty
            + 0.25 * spatial_novelty
        )
        take = min(remaining, max(1, (remaining + rounds_left - 1) // rounds_left))
        chosen = available.index_select(0, _stable_argsort(score, descending=True)[:take])
        selected_mask[chosen] = True
        selected = torch.cat([selected, chosen])
        remaining -= int(chosen.numel())
        rounds_left = max(1, rounds_left - 1)

    selected_indices = torch.sort(selected).values.to(torch.long)
    if selected_indices.numel() != output_tokens or torch.unique(selected_indices).numel() != output_tokens:
        raise RuntimeError("Conditional diversity selector did not produce the exact unique token budget")

    route = DARTMergeRoute(
        selected_indices=selected_indices,
        original_tokens=num_tokens,
        output_tokens=output_tokens,
    )
    route.validate()
    return route


def prune_image_embeds(image_embeds: torch.Tensor, route: DARTMergeRoute) -> torch.Tensor:
    """Return selected original embeddings without averaging or synthesis."""

    route = route.to(image_embeds.device)
    route.validate()
    if route.summary_anchor_index is not None:
        raise ValueError("A discarded-summary route requires weighted summary reduction")
    if route.learnable_merge_enabled:
        raise ValueError("A learnable node merge route requires group aggregation")
    if image_embeds.ndim != 2 or image_embeds.shape[0] != route.original_tokens:
        raise ValueError(
            f"image_embeds shape {tuple(image_embeds.shape)} is incompatible with route "
            f"original_tokens={route.original_tokens}"
        )
    selected = image_embeds.index_select(0, route.selected_indices)
    if not torch.isfinite(selected).all().item():
        raise FloatingPointError("Selected visual embeddings contain NaN or Inf")
    return selected


def summarize_discarded_image_embeds(
    image_embeds: torch.Tensor,
    route: DARTMergeRoute,
    *,
    learnable_summary: Optional[LearnableResidualSummaryMLP] = None,
    activation_checkpointing: bool = True,
    chunk_tokens: int = 1024,
) -> torch.Tensor:
    """Keep CDPruner's original embeddings and replace only its extra anchor slot.

    Routing and the cosine prior are detached. The learnable head recomputes alpha
    from the current sources and current parameters on every actor forward;
    alpha never enters route serialization. The FP32 weighted sum propagates
    gradients to every source and the head before casting back at output.
    """
    route = route.to(image_embeds.device)
    route.validate()
    if not route.summary_enabled:
        raise ValueError("Weighted discarded summary requires a summary route schema")
    learned_route = route.schema_version in CDPRUNER_LEARNABLE_SUMMARY_SCHEMAS
    if learned_route != (learnable_summary is not None):
        raise ValueError("Learnable summary route schema and aggregation head must agree")
    if learnable_summary is not None:
        learnable_summary._require_ready()
    if image_embeds.ndim != 2 or image_embeds.shape[0] != route.original_tokens:
        raise ValueError("Summary source embeddings do not match the route")
    if not torch.isfinite(image_embeds).all().item():
        raise FloatingPointError("Summary source embeddings contain NaN or Inf")
    if route.summary_anchor_index is None:
        return prune_image_embeds(image_embeds, route)
    selected = image_embeds.index_select(0, route.selected_indices)
    with torch.autocast(device_type=image_embeds.device.type, enabled=False):
        if learnable_summary is None:
            sources = image_embeds.index_select(0, route.summary_source_indices).float()
            summary_fp32 = torch.matmul(route.summary_weights.detach(), sources)
        else:
            summary_fp32 = learnable_summary.summarize(
                image_embeds, route.summary_source_indices, route.summary_anchor_index, route.summary_weights,
                activation_checkpointing=activation_checkpointing, chunk_tokens=chunk_tokens,
            )
        summary = summary_fp32.to(image_embeds.dtype)
    anchor_slot = torch.nonzero(route.selected_indices == route.summary_anchor_index, as_tuple=False).flatten()
    output = selected.index_copy(0, anchor_slot, summary.unsqueeze(0))
    if not torch.isfinite(output).all().item():
        raise FloatingPointError("Weighted summary embeddings contain NaN or Inf")
    return output


def merge_image_embeds(image_embeds: torch.Tensor, route: DARTMergeRoute) -> torch.Tensor:
    """Legacy compatibility name for :func:`prune_image_embeds`."""

    return prune_image_embeds(image_embeds, route)


def learnable_merge_image_embeds(
    image_embeds: torch.Tensor,
    route: DARTMergeRoute,
    learnable_summary: LearnableResidualSummaryMLP,
    *,
    activation_checkpointing: bool = True,
    chunk_tokens: int = 1024,
) -> torch.Tensor:
    """Aggregate every original node into one of the K retained anchor slots."""
    route = route.to(image_embeds.device)
    route.validate()
    if not route.learnable_merge_enabled:
        raise ValueError("Learnable node aggregation requires a learnable merge route schema")
    if image_embeds.ndim != 2 or image_embeds.shape[0] != route.original_tokens:
        raise ValueError("Learnable merge features do not match the original route nodes")
    if not torch.isfinite(image_embeds).all().item():
        raise FloatingPointError("Learnable merge source embeddings contain NaN or Inf")
    output = learnable_summary.merge_nodes(
        image_embeds, route.selected_indices, route.merge_assignment, route.merge_prior_weights,
        activation_checkpointing=activation_checkpointing, chunk_tokens=chunk_tokens,
    )
    if not torch.isfinite(output).all().item():
        raise FloatingPointError("Learnable merged embeddings contain NaN or Inf")
    return output


def spatial_mean_merge_image_embeds(
    image_embeds: torch.Tensor,
    route: HoliTomDPCSpatialMergeRoute,
) -> torch.Tensor:
    """Average every DPC cluster while preserving gradients to every source."""

    route = route.to(image_embeds.device)
    route.validate()
    if image_embeds.ndim != 2 or image_embeds.shape[0] != route.original_tokens:
        raise ValueError(
            f"image_embeds shape {tuple(image_embeds.shape)} is incompatible with HoliTom route "
            f"original_tokens={route.original_tokens}"
        )
    if not torch.isfinite(image_embeds).all().item():
        raise FloatingPointError("HoliTom DPC source embeddings contain NaN or Inf")
    accumulation = (
        image_embeds.float()
        if route.algorithm == HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM
        else image_embeds
    )
    sums = accumulation.new_zeros((route.output_tokens, image_embeds.shape[-1]))
    sums = sums.index_add(0, route.assignment, accumulation)
    counts = route.source_counts.to(device=image_embeds.device, dtype=accumulation.dtype).unsqueeze(-1)
    merged = (sums / counts).to(image_embeds.dtype)
    if not torch.isfinite(merged).all().item():
        raise FloatingPointError("HoliTom DPC merged embeddings contain NaN or Inf")
    return merged


class VisionHoliTomDPCSpatialMergeCompressor(nn.Module):
    """Parameter-free HoliTom-inspired spatial DPC merge at the LLM boundary."""

    algorithm = HOLITOM_DPC_SPATIAL_MERGE_ALGORITHM
    method = HOLITOM_DPC_SPATIAL_MERGE_METHOD
    route_schema_version = HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA
    merge = "differentiable_arithmetic_mean"

    def __init__(
        self,
        *,
        retention_ratio: float = CDPRUNER_RETENTION_RATIO,
        minimum_tokens: int = CDPRUNER_MINIMUM_TOKENS,
        distance_chunk_tokens: int = 256,
        curriculum: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not math.isclose(float(retention_ratio), CDPRUNER_RETENTION_RATIO, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"HoliTom DPC retention_ratio is fixed at {CDPRUNER_RETENTION_RATIO}")
        if int(minimum_tokens) != CDPRUNER_MINIMUM_TOKENS:
            raise ValueError(f"HoliTom DPC minimum_tokens is fixed at {CDPRUNER_MINIMUM_TOKENS}")
        if int(distance_chunk_tokens) <= 0:
            raise ValueError("HoliTom DPC distance_chunk_tokens must be positive")
        self.retention_ratio = CDPRUNER_RETENTION_RATIO
        self.minimum_tokens = CDPRUNER_MINIMUM_TOKENS
        self.distance_chunk_tokens = int(distance_chunk_tokens)
        self.curriculum = None
        self.curriculum_completed_steps = None
        self.curriculum_schedule_sha256 = None
        self.active_retention_bps = 500
        if curriculum is not None:
            from verl.models.transformers.visual_token_curriculum import VisualTokenCurriculum

            self.curriculum = VisualTokenCurriculum.from_mapping(curriculum)
            if self.curriculum.minimum_tokens_per_image != 32 or self.curriculum.final_retention_bps != 500:
                raise ValueError("HoliTom curriculum requires final 5% and per-image minimum 32")
            self.algorithm = HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM
            self.route_schema_version = HOLITOM_DPC_CURRICULUM_MERGE_ROUTE_SCHEMA
            self.curriculum_schedule_sha256 = self.curriculum.schedule_sha256
            self.active_retention_bps = None

    def set_curriculum_step(self, completed_optimizer_steps: int) -> dict[str, Any]:
        if self.curriculum is None:
            raise RuntimeError("Cannot bind a curriculum step on static HoliTom")
        state = self.curriculum.runtime_state(completed_optimizer_steps)
        self.curriculum_completed_steps = state["completed_optimizer_steps"]
        self.active_retention_bps = state["retention_bps"]
        self.retention_ratio = self.active_retention_bps / 10000.0
        return state

    def set_final_retention_for_evaluation(self) -> dict[str, Any]:
        if self.curriculum is not None:
            return self.set_curriculum_step(self.curriculum.total_optimizer_steps - 1)
        return dict(schema_version=None, schedule_sha256=None, completed_optimizer_steps=None,
                    retention_bps=500, minimum_tokens_per_image=self.minimum_tokens)

    def forward(
        self,
        image_embeds: torch.Tensor,
        route: Optional[HoliTomDPCSpatialMergeRoute] = None,
        *,
        query_embeds: Optional[torch.Tensor] = None,
        image_coordinates: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, HoliTomDPCSpatialMergeRoute]:
        if self.active_retention_bps is None or (self.curriculum is not None and self.curriculum_completed_steps is None):
            raise RuntimeError("HoliTom curriculum must bind the completed optimizer step before rollout or replay")
        if query_embeds is not None:
            raise ValueError("HoliTom DPC is query-independent; query embeddings are forbidden")
        if route is None:
            if image_coordinates is None:
                raise ValueError("HoliTom DPC route construction requires original image M-RoPE coordinates")
            route = build_holitom_dpc_spatial_merge_route(
                image_embeds,
                image_coordinates=image_coordinates,
                distance_chunk_tokens=self.distance_chunk_tokens,
                retention_bps=self.active_retention_bps,
                curriculum_completed_steps=self.curriculum_completed_steps,
                curriculum_schedule_sha256=self.curriculum_schedule_sha256,
            )
        else:
            if not isinstance(route, HoliTomDPCSpatialMergeRoute):
                raise TypeError("HoliTom DPC replay requires HoliTomDPCSpatialMergeRoute")
            route = route.to(image_embeds.device)
            route.validate_for_algorithm(self.algorithm)
            if self.curriculum is not None:
                validate_holitom_curriculum_route(
                    route, self.curriculum.runtime_state(self.curriculum_completed_steps),
                    curriculum=self.curriculum,
                )
            expected = _exact_holitom_dpc_budget(route.original_tokens, self.active_retention_bps)
            if route.output_tokens != expected:
                raise ValueError(
                    f"Replayed HoliTom DPC route has {route.output_tokens} tokens; expected {expected}"
                )
        return spatial_mean_merge_image_embeds(image_embeds, route), route


class VisionCDPrunerCompressor(nn.Module):
    """Qwen CDPruner with optional fixed or learnable discarded aggregation."""

    algorithm = CDPRUNER_ALGORITHM
    method = CDPRUNER_METHOD
    route_schema_version = CDPRUNER_ROUTE_SCHEMA_VERSION
    merge = "none_pure_index_prune"

    def __init__(
        self,
        *,
        retention_ratio: float = CDPRUNER_RETENTION_RATIO,
        minimum_tokens: int = CDPRUNER_MINIMUM_TOKENS,
        kernel_jitter: float = 1e-6,
        relevance_epsilon: float = 1e-6,
        residual_epsilon: float = 1e-12,
        curriculum: Optional[Mapping[str, Any]] = None,
        summary: Optional[Mapping[str, Any]] = None,
        input_hidden_size: Optional[int] = None,
        learnable_merge: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not math.isclose(float(retention_ratio), CDPRUNER_RETENTION_RATIO, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"CDPruner retention_ratio is fixed at {CDPRUNER_RETENTION_RATIO}")
        if int(minimum_tokens) != CDPRUNER_MINIMUM_TOKENS:
            raise ValueError(f"CDPruner minimum_tokens is fixed at {CDPRUNER_MINIMUM_TOKENS}")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in (kernel_jitter, relevance_epsilon, residual_epsilon)
        ):
            raise ValueError("CDPruner numerical epsilons must be positive")
        self.curriculum = None
        self.summary_config = _validate_cdpruner_summary_config(summary)
        self.summary_enabled = self.summary_config["enabled"]
        self.learnable_merge_config = _validate_cdpruner_learnable_merge_config(learnable_merge)
        self.learnable_merge_enabled = self.learnable_merge_config["enabled"]
        if self.summary_enabled and self.learnable_merge_enabled:
            raise ValueError("CDPruner summary and learnable_merge cannot both be enabled")
        self.learnable_summary_enabled = (
            self.summary_config.get("weight_rule") == CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE
        )
        self.learnable_summary = None
        if self.learnable_summary_enabled or self.learnable_merge_enabled:
            head_config = self.learnable_merge_config if self.learnable_merge_enabled else self.summary_config
            configured_input_hidden_size = head_config.get("input_hidden_size")
            if input_hidden_size is not None and (
                isinstance(input_hidden_size, bool)
                or not isinstance(input_hidden_size, Integral)
                or input_hidden_size <= 0
            ):
                raise ValueError("Learnable summary input_hidden_size must be a positive model-config integer")
            if (
                input_hidden_size is not None and configured_input_hidden_size is not None
                and int(input_hidden_size) != configured_input_hidden_size
            ):
                raise ValueError("Learnable summary input_hidden_size differs from model config")
            effective_input_hidden_size = (
                input_hidden_size if input_hidden_size is not None else configured_input_hidden_size
            )
            if effective_input_hidden_size is None:
                raise ValueError("Learnable summary requires input_hidden_size explicitly from model config")
            head_config["input_hidden_size"] = int(effective_input_hidden_size)
            self.learnable_summary = LearnableResidualSummaryMLP(
                int(effective_input_hidden_size),
                hidden_size=head_config["hidden_size"],
                residual_logit_limit=head_config.get("residual_logit_limit", 4.0),
            )
        self.curriculum_completed_steps: Optional[int] = None
        self.curriculum_schedule_sha256: Optional[str] = None
        self.active_retention_bps: Optional[int] = 500
        if curriculum is not None:
            from verl.models.transformers.visual_token_curriculum import VisualTokenCurriculum

            self.curriculum = VisualTokenCurriculum.from_mapping(curriculum)
            if self.curriculum.minimum_tokens_per_image != CDPRUNER_MINIMUM_TOKENS:
                raise ValueError("CDPruner curriculum must retain the immutable per-image minimum of 32")
            if self.curriculum.final_retention_bps != 500:
                raise ValueError("CDPruner curriculum must retain the immutable final 5% budget")
            self.curriculum_schedule_sha256 = self.curriculum.schedule_sha256
            self.active_retention_bps = None
        self.route_schema_version = _cdpruner_route_schema(
            summary_enabled=self.summary_enabled,
            learnable_summary_enabled=self.learnable_summary_enabled,
            curriculum_enabled=self.curriculum is not None,
            learnable_merge_enabled=self.learnable_merge_enabled,
        )
        if self.summary_enabled:
            self.merge = "single_knn_anchor_weighted_discarded_summary"
        elif self.learnable_merge_enabled:
            self.merge = "learnable_nearest_retained_zero_init_residual"
        self.retention_ratio = CDPRUNER_RETENTION_RATIO
        self.minimum_tokens = CDPRUNER_MINIMUM_TOKENS
        self.kernel_jitter = float(kernel_jitter)
        self.relevance_epsilon = float(relevance_epsilon)
        self.residual_epsilon = float(residual_epsilon)

    def set_curriculum_step(self, completed_optimizer_steps: int) -> dict[str, Any]:
        """Bind the compressor to one pre-update optimizer-step state."""

        if self.curriculum is None:
            raise RuntimeError("cannot set a curriculum step on a fixed-ratio CDPruner")
        state = self.curriculum.runtime_state(completed_optimizer_steps)
        self.curriculum_completed_steps = int(state["completed_optimizer_steps"])
        self.active_retention_bps = int(state["retention_bps"])
        self.retention_ratio = self.active_retention_bps / 10_000.0
        return state

    def set_final_retention_for_evaluation(self) -> dict[str, Any]:
        """Use the final 5% budget for inference."""

        if self.curriculum is None:
            return {
                "schema_version": None,
                "schedule_sha256": None,
                "completed_optimizer_steps": None,
                "retention_bps": 500,
                "minimum_tokens_per_image": self.minimum_tokens,
            }
        return self.set_curriculum_step(self.curriculum.total_optimizer_steps - 1)

    def _active_budget_state(self) -> tuple[int, Optional[int], Optional[str]]:
        if self.curriculum is not None and (
            self.active_retention_bps is None or self.curriculum_completed_steps is None
        ):
            raise RuntimeError(
                "curriculum CDPruner has no optimizer-step binding; training must set the "
                "completed optimizer step before every rollout"
            )
        if self.active_retention_bps is None:
            raise RuntimeError("CDPruner active retention budget is unavailable")
        return (
            int(self.active_retention_bps),
            self.curriculum_completed_steps,
            self.curriculum_schedule_sha256,
        )

    def forward(
        self,
        image_embeds: torch.Tensor,
        route: Optional[DARTMergeRoute] = None,
        *,
        query_embeds: Optional[torch.Tensor] = None,
        image_coordinates: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, DARTMergeRoute]:
        retention_bps, curriculum_completed_steps, schedule_sha256 = self._active_budget_state()
        if route is None:
            if query_embeds is None or image_coordinates is None:
                raise ValueError("New CDPruner routes require query embeddings and image coordinates")
            route = build_cdpruner_route(
                image_embeds,
                query_embeds=query_embeds,
                image_coordinates=image_coordinates,
                kernel_jitter=self.kernel_jitter,
                relevance_epsilon=self.relevance_epsilon,
                residual_epsilon=self.residual_epsilon,
                retention_bps=retention_bps,
                curriculum_completed_steps=curriculum_completed_steps,
                curriculum_schedule_sha256=schedule_sha256,
                summary=self.summary_config,
                learnable_merge=self.learnable_merge_config,
            )
        else:
            route = route.to(image_embeds.device)
            route.validate_for_algorithm(
                self.algorithm,
                expected_schema_version=self.route_schema_version,
            )
            if self.curriculum is not None:
                state = self.curriculum.runtime_state(int(curriculum_completed_steps))
                validate_cdpruner_curriculum_route(
                    route, state, curriculum=self.curriculum, summary_enabled=self.summary_enabled,
                    summary_weight_rule=self.summary_config.get("weight_rule", "softmax_cosine_to_anchor"),
                    learnable_merge_enabled=self.learnable_merge_enabled,
                )
            else:
                expected = _exact_cdpruner_budget(
                    route.original_tokens, retention_bps=retention_bps
                )
                expected += int(self.summary_enabled and expected < route.original_tokens)
                if route.output_tokens != expected:
                    raise ValueError(
                        f"Replayed CDPruner route has {route.output_tokens} tokens; "
                        f"expected exact budget {expected}"
                    )
        if self.learnable_merge_enabled:
            reduced = learnable_merge_image_embeds(
                image_embeds, route, self.learnable_summary,
                activation_checkpointing=self.learnable_merge_config["activation_checkpointing"],
                chunk_tokens=self.learnable_merge_config["chunk_tokens"],
            )
        elif self.summary_enabled:
            reduced = summarize_discarded_image_embeds(
                image_embeds, route, learnable_summary=self.learnable_summary,
                activation_checkpointing=self.summary_config.get("activation_checkpointing", True),
                chunk_tokens=self.summary_config.get("chunk_tokens", 1024),
            )
        else:
            reduced = prune_image_embeds(image_embeds, route)
        return reduced, route


class VisionDARTMergeCompressor(nn.Module):
    """Legacy fixed-quota conditional-diversity path."""

    algorithm = LEGACY_CONDITIONAL_DIVERSITY_ALGORITHM

    def __init__(
        self,
        *,
        retention_ratio: float = 0.05,
        minimum_tokens: int = 32,
        relevance_fraction: float = 0.60,
        candidate_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.retention_ratio = float(retention_ratio)
        self.minimum_tokens = int(minimum_tokens)
        self.relevance_fraction = float(relevance_fraction)
        self.candidate_multiplier = int(candidate_multiplier)

    def forward(
        self,
        image_embeds: torch.Tensor,
        route: Optional[DARTMergeRoute] = None,
        *,
        query_embeds: Optional[torch.Tensor] = None,
        image_coordinates: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, DARTMergeRoute]:
        if route is None:
            if query_embeds is None or image_coordinates is None:
                raise ValueError("New conditional prune routes require query embeddings and image coordinates")
            route = build_dart_merge_route(
                image_embeds,
                query_embeds=query_embeds,
                image_coordinates=image_coordinates,
                retention_ratio=self.retention_ratio,
                minimum_tokens=self.minimum_tokens,
                relevance_fraction=self.relevance_fraction,
                candidate_multiplier=self.candidate_multiplier,
            )
        else:
            route = route.to(image_embeds.device)
            route.validate_for_algorithm(self.algorithm)
        return merge_image_embeds(image_embeds, route), route


def build_vision_token_compressor(
    config: Mapping[str, Any],
    *,
    allow_legacy: bool = True,
    input_hidden_size: Optional[int] = None,
) -> VisionHoliTomDPCSpatialMergeCompressor | VisionCDPrunerCompressor | VisionDARTMergeCompressor:
    """Instantiate the configured visual token compressor.

    In particular, a ``qwen35_cdpruner_v1`` checkpoint can never be silently
    reconstructed with the legacy selector.  Legacy construction is
    available only when its legacy algorithm identifier is explicit.
    """

    algorithm = config.get("algorithm")
    if algorithm in HOLITOM_DPC_MERGE_ALGORITHMS:
        curriculum_config = config.get("curriculum")
        curriculum_enabled = algorithm == HOLITOM_DPC_CURRICULUM_MERGE_ALGORITHM
        if curriculum_enabled != (curriculum_config is not None):
            raise ValueError("HoliTom static/curriculum algorithm must match the presence of curriculum configuration")
        exact_identifiers = {
            "method": HOLITOM_DPC_SPATIAL_MERGE_METHOD,
            "route_schema_version": (HOLITOM_DPC_CURRICULUM_MERGE_ROUTE_SCHEMA if curriculum_enabled else HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA),
            "transport_key": HOLITOM_DPC_MERGE_ROUTES_KEY,
            "placement": "post_native_merger_pre_llm",
            "budget_policy": ("per_image_min_N_max_32_ceil_active_optimizer_step_retention_bps_N_v1" if curriculum_enabled else "per_image_min_N_max_32_ceil_0.05N"),
            "distance": "euclidean_div_sqrt_hidden_dim_fp32",
            "routing_autograd": "detached_no_grad_fp32",
            "center_score": "density_times_holitom_per_token_max_or_nearest_higher_delta",
            "assignment": "nearest_center_euclidean_stable_original_index_tie",
            "reduction": "differentiable_arithmetic_mean_of_original_embeddings",
            "position_policy": "inherit_center_original_three_axis_mrope_no_rebase",
        }
        for key, expected in exact_identifiers.items():
            if key in config and config.get(key) != expected:
                raise ValueError(f"HoliTom DPC {key} must be {expected!r}, got {config.get(key)!r}")
        retention_ratio = float(config.get("retention_ratio", CDPRUNER_RETENTION_RATIO))
        minimum_tokens = int(config.get("minimum_tokens_per_image", CDPRUNER_MINIMUM_TOKENS))
        if not math.isclose(retention_ratio, CDPRUNER_RETENTION_RATIO, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"HoliTom DPC retention_ratio is fixed at {CDPRUNER_RETENTION_RATIO}")
        if minimum_tokens != CDPRUNER_MINIMUM_TOKENS:
            raise ValueError(f"HoliTom DPC minimum_tokens_per_image is fixed at {CDPRUNER_MINIMUM_TOKENS}")
        if int(config.get("knn_k", 7)) != 7:
            raise ValueError("HoliTom DPC knn_k is fixed at 7 (and clipped to N at runtime)")
        distance_chunk_tokens = config.get("distance_chunk_tokens", 256)
        if (
            isinstance(distance_chunk_tokens, bool)
            or not isinstance(distance_chunk_tokens, Integral)
            or int(distance_chunk_tokens) <= 0
        ):
            raise ValueError("HoliTom DPC distance_chunk_tokens must be a positive integer")
        boolean_contract = {
            "query_conditioned": False,
            "attention_conditioned": False,
            "bbox_conditioned": False,
            "persistent_n_by_n_matrix_forbidden": True,
            "deterministic": True,
        }
        for key, expected in boolean_contract.items():
            if key in config and config.get(key) is not expected:
                raise ValueError(f"HoliTom DPC {key} must be {expected}")
        forbidden = [
            key
            for key in (
                "relevance_fraction",
                "candidate_multiplier",
                "kernel_jitter",
                "relevance_epsilon",
                "residual_epsilon",
                "selector_dtype",
                "merge",
                "query_policy",
                "query_weight",
                "attention_weight",
                "bbox_weight",
            )
            if config.get(key) is not None
        ]
        if forbidden:
            raise ValueError(f"HoliTom DPC contract contains forbidden conditional settings: {forbidden}")
        return VisionHoliTomDPCSpatialMergeCompressor(
            retention_ratio=retention_ratio,
            minimum_tokens=minimum_tokens,
            distance_chunk_tokens=int(distance_chunk_tokens),
            curriculum=curriculum_config,
        )

    if algorithm == CDPRUNER_ALGORITHM:
        curriculum_config = config.get("curriculum")
        curriculum_enabled = curriculum_config is not None
        summary_config = _validate_cdpruner_summary_config(config.get("summary"))
        summary_enabled = summary_config["enabled"]
        learnable_summary_enabled = summary_config.get("weight_rule") == CDPRUNER_LEARNABLE_SUMMARY_WEIGHT_RULE
        merge_config = _validate_cdpruner_learnable_merge_config(config.get("learnable_merge"))
        learnable_merge_enabled = merge_config["enabled"]
        if summary_enabled and learnable_merge_enabled:
            raise ValueError("CDPruner summary and learnable_merge cannot both be enabled")
        exact_identifiers = {
            "method": CDPRUNER_METHOD,
            "transport_key": "dart_merge_routes",
            "placement": "post_native_merger_pre_llm",
            "conditional_objective": "instruction_relevance_weighted_visual_diversity_greedy_dpp_matrix_free_fp32",
            "selector_dtype": "float32",
            "merge": (
                "learnable_nearest_retained_zero_init_residual" if learnable_merge_enabled else
                "single_knn_anchor_weighted_discarded_summary" if summary_enabled else "none_pure_index_prune"
            ),
            "position_policy": "selected_and_summary_anchor_original_mrope_sorted_no_rebase" if summary_enabled else "selected_original_mrope",
            "query_policy": "user_question_options_only_excluding_special_and_image_tokens",
            "query_route_schema_version": "vision_opd_route_query_v2",
            "routing_autograd": "detached_no_grad_fp32",
            "assignment": "greedy_conditional_dpp_stable_original_index_tie",
            "reduction": (
                "differentiable_fp32_zero_init_signed_residual" if learnable_merge_enabled else
                "differentiable_fp32_learnable_residual_softmax_weighted_mean_of_all_discarded_embeddings"
                if learnable_summary_enabled else
                "differentiable_fp32_softmax_weighted_mean_of_all_discarded_embeddings"
                if summary_enabled else "none_pure_index_prune"
            ),
        }
        for key, expected in exact_identifiers.items():
            if key in config and config.get(key) != expected:
                raise ValueError(f"CDPruner {key} must be {expected!r}, got {config.get(key)!r}")
        expected_route_schema = _cdpruner_route_schema(
            summary_enabled=summary_enabled,
            learnable_summary_enabled=learnable_summary_enabled,
            curriculum_enabled=curriculum_enabled,
            learnable_merge_enabled=learnable_merge_enabled,
        )
        expected_budget_policy = (
            ("per_image_min_N_max_32_ceil_active_optimizer_step_retention_bps_N_plus_one_if_discarded_v1" if curriculum_enabled else "per_image_min_N_max_32_ceil_0.05N_plus_one_if_discarded_v1")
            if summary_enabled else
            ("per_image_min_N_max_32_ceil_active_optimizer_step_retention_bps_N_v1" if curriculum_enabled else "per_image_max_32_ceil_0.05N")
        )
        if config.get("route_schema_version") != expected_route_schema:
            raise ValueError(
                "CDPruner route_schema_version drift: "
                f"expected={expected_route_schema!r}, actual={config.get('route_schema_version')!r}"
            )
        if config.get("budget_policy") != expected_budget_policy:
            raise ValueError(
                "CDPruner budget_policy drift: "
                f"expected={expected_budget_policy!r}, actual={config.get('budget_policy')!r}"
            )
        boolean_contract = {
            "query_conditioned": True,
            "attention_conditioned": False,
            "bbox_conditioned": False,
            "persistent_n_by_n_matrix_forbidden": True,
            "deterministic": True,
        }
        for key, expected in boolean_contract.items():
            if key in config and config.get(key) is not expected:
                raise ValueError(f"CDPruner {key} must be {expected}")
        retention_ratio = float(config.get("retention_ratio", CDPRUNER_RETENTION_RATIO))
        minimum_tokens = int(config.get("minimum_tokens_per_image", CDPRUNER_MINIMUM_TOKENS))
        if not math.isclose(retention_ratio, CDPRUNER_RETENTION_RATIO, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"CDPruner retention_ratio is fixed at {CDPRUNER_RETENTION_RATIO}")
        if minimum_tokens != CDPRUNER_MINIMUM_TOKENS:
            raise ValueError(f"CDPruner minimum_tokens_per_image is fixed at {CDPRUNER_MINIMUM_TOKENS}")
        forbidden_legacy = [
            key
            for key in (
                "relevance_fraction",
                "candidate_multiplier",
                "query_weight",
                "attention_weight",
                "bbox_weight",
                "distance",
                "distance_chunk_tokens",
                "knn_k",
                "center_score",
            )
            if config.get(key) is not None
        ]
        if forbidden_legacy:
            raise ValueError(f"CDPruner contract contains legacy selector settings: {forbidden_legacy}")
        return VisionCDPrunerCompressor(
            retention_ratio=retention_ratio,
            minimum_tokens=minimum_tokens,
            kernel_jitter=float(config.get("kernel_jitter", 1e-6)),
            relevance_epsilon=float(config.get("relevance_epsilon", 1e-6)),
            residual_epsilon=float(config.get("residual_epsilon", 1e-12)),
            curriculum=curriculum_config,
            summary=summary_config,
            input_hidden_size=input_hidden_size,
            learnable_merge=merge_config,
        )

    if algorithm == LEGACY_CONDITIONAL_DIVERSITY_ALGORITHM and allow_legacy:
        return VisionDARTMergeCompressor(
            retention_ratio=float(config.get("retention_ratio", 0.05)),
            minimum_tokens=int(config.get("minimum_tokens_per_image", 32)),
            relevance_fraction=float(config.get("relevance_fraction", 0.60)),
            candidate_multiplier=int(config.get("candidate_multiplier", 4)),
        )
    if algorithm == LEGACY_CONDITIONAL_DIVERSITY_ALGORITHM:
        raise ValueError("The legacy conditional-diversity selector is not supported by this loader")
    raise ValueError(f"Unsupported vision token compressor algorithm: {algorithm!r}")


def visual_token_compressor_enabled(model: nn.Module) -> bool:
    return bool(getattr(model, "vision_token_compressor_enabled", False)) and isinstance(
        getattr(model, "vision_token_compressor", None),
        (VisionHoliTomDPCSpatialMergeCompressor, VisionCDPrunerCompressor, VisionDARTMergeCompressor),
    )


def _contiguous_true_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open spans with one device synchronization per row.

    Calling ``Tensor.item()`` once per prompt token serializes thousands of GPU
    kernels for real Qwen image prompts.  Moving only the sparse indices to CPU
    preserves the exact span semantics without that performance regression.
    """

    if mask.ndim != 1:
        raise ValueError("span mask must be rank-1")
    indices = torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist()
    if not indices:
        return []
    spans: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            spans.append((start, previous + 1))
            start = index
        previous = index
    spans.append((start, previous + 1))
    return spans


def count_compressed_image_tokens(
    image_mask: torch.Tensor,
    minimum_tokens: int,
    retention_ratio: float = 0.05,
    *,
    retention_bps: Optional[int] = None,
    summary_enabled: bool = False,
) -> torch.Tensor:
    """Return the actual compressed image-token count for every batch row."""

    if image_mask.ndim != 2:
        raise ValueError("image_mask must be [batch, sequence]")
    if not isinstance(summary_enabled, bool):
        raise TypeError("summary_enabled must be boolean")
    if retention_bps is None:
        if not math.isfinite(float(retention_ratio)) or not 0.0 < float(retention_ratio) <= 1.0:
            raise ValueError("retention_ratio must be finite and in (0, 1]")
        retention_bps = round(float(retention_ratio) * 10_000)
        if not math.isclose(
            float(retention_ratio),
            retention_bps / 10_000.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("retention_ratio must have an exact integer-basis-point representation")
    from verl.models.transformers.visual_token_curriculum import visual_token_budget

    counts = []
    for row in image_mask:
        spans = _contiguous_true_spans(row.to(torch.bool))
        counts.append(
            sum(
                visual_token_budget(
                    end - start,
                    retention_bps=int(retention_bps),
                    minimum_tokens=int(minimum_tokens),
                )
                + int(summary_enabled and visual_token_budget(
                    end - start,
                    retention_bps=int(retention_bps),
                    minimum_tokens=int(minimum_tokens),
                ) < end - start)
                for start, end in spans
            )
        )
    return torch.tensor(counts, device=image_mask.device, dtype=torch.long)


def compress_image_embeds(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    image_embeds: torch.Tensor,
    image_token_id: int,
    routes: Optional[Sequence[DARTMergeRoute | HoliTomDPCSpatialMergeRoute]] = None,
    compression_query_mask: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    list[DARTMergeRoute | HoliTomDPCSpatialMergeRoute],
]:
    """Replace each image span with pruned or spatially merged embeddings."""

    if not visual_token_compressor_enabled(model):
        return inputs_embeds, attention_mask, position_ids, []

    compressor: VisionHoliTomDPCSpatialMergeCompressor | VisionCDPrunerCompressor | VisionDARTMergeCompressor
    compressor = model.vision_token_compressor
    holitom_merge = isinstance(compressor, VisionHoliTomDPCSpatialMergeCompressor)
    if holitom_merge and compression_query_mask is not None:
        raise ValueError("HoliTom spatial merge is query-independent; query masks are forbidden")
    batch_size, seq_len, hidden_size = inputs_embeds.shape
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.long)
    cdpruner_spans_by_row: Optional[list[list[tuple[int, int]]]] = None
    if isinstance(compressor, VisionCDPrunerCompressor):
        cdpruner_spans_by_row = []
        for batch_idx in range(batch_size):
            row_spans = _contiguous_true_spans(
                (input_ids[batch_idx] == image_token_id)
                & attention_mask[batch_idx].to(device=input_ids.device, dtype=torch.bool)
            )
            cdpruner_spans_by_row.append(row_spans)
    if routes is None and not holitom_merge:
        if compression_query_mask is None or compression_query_mask.shape != input_ids.shape:
            raise ValueError("New conditional routes require compression_query_mask [batch, sequence]")
        query_mask = compression_query_mask.to(device=input_ids.device, dtype=torch.bool)
        if torch.any(query_mask & ~attention_mask.to(torch.bool)).item():
            raise ValueError("compression_query_mask includes masked prompt positions")
        if torch.any(query_mask & (input_ids == image_token_id)).item():
            raise ValueError("compression_query_mask must exclude image placeholders")

    if position_ids is not None and position_ids.ndim == 2:
        position_ids_work = position_ids.unsqueeze(0)
        squeeze_position_ids = True
    else:
        position_ids_work = position_ids
        squeeze_position_ids = False
    if position_ids_work is not None and (
        position_ids_work.ndim != 3
        or position_ids_work.shape[1] != batch_size
        or position_ids_work.shape[2] != seq_len
    ):
        raise ValueError(
            "position_ids must be [batch, seq] or [rope_dims, batch, seq], got "
            f"{tuple(position_ids_work.shape)}"
        )

    route_iter = iter(routes) if routes is not None else None
    used_routes: list[DARTMergeRoute | HoliTomDPCSpatialMergeRoute] = []
    embed_offset = 0
    new_embeds: list[torch.Tensor] = []
    new_masks: list[torch.Tensor] = []
    new_positions: Optional[list[torch.Tensor]] = [] if position_ids_work is not None else None
    previous_row_ids: Optional[torch.Tensor] = None
    previous_row_mask: Optional[torch.Tensor] = None
    previous_row_sources: list[torch.Tensor] = []
    previous_row_coordinates: list[Optional[torch.Tensor]] = []
    previous_row_routes: list[HoliTomDPCSpatialMergeRoute] = []

    for batch_idx in range(batch_size):
        ids = input_ids[batch_idx]
        spans = (
            cdpruner_spans_by_row[batch_idx]
            if cdpruner_spans_by_row is not None
            else _contiguous_true_spans(
                (ids == image_token_id) & attention_mask[batch_idx].to(device=ids.device, dtype=torch.bool)
            )
        )
        # GRPO/VOPD places the n rollouts for one source sample next to each
        # other.  Their deterministic eval-mode vision embeddings are exactly
        # equal, so rebuilding the same O(N^2) DPC route n times is pure waste.
        # Reuse is deliberately local to this forward and guarded by exact
        # input-ID, mask, source-embedding, and coordinate equality.  Any
        # mismatch falls back to an ordinary exact route build.
        repeated_holitom_row = bool(
            holitom_merge
            and routes is None
            and previous_row_ids is not None
            and len(previous_row_sources) == len(spans)
            and torch.equal(ids, previous_row_ids)
            and torch.equal(attention_mask[batch_idx], previous_row_mask)
        )
        current_row_sources: list[torch.Tensor] = []
        current_row_coordinates: list[Optional[torch.Tensor]] = []
        current_row_routes: list[HoliTomDPCSpatialMergeRoute] = []
        row_query_embeds = None
        if routes is None and not holitom_merge:
            row_query_embeds = inputs_embeds[batch_idx][query_mask[batch_idx]]
            if row_query_embeds.shape[0] == 0:
                raise ValueError(f"No inference-visible query tokens for batch row {batch_idx}")

        pieces: list[torch.Tensor] = []
        mask_pieces: list[torch.Tensor] = []
        pos_pieces: Optional[list[torch.Tensor]] = [] if position_ids_work is not None else None
        last = 0
        for span_index, (start, end) in enumerate(spans):
            if start > last:
                pieces.append(inputs_embeds[batch_idx, last:start])
                mask_pieces.append(attention_mask[batch_idx, last:start])
                if pos_pieces is not None:
                    pos_pieces.append(position_ids_work[:, batch_idx, last:start])

            span_len = end - start
            span_embeds = image_embeds[embed_offset : embed_offset + span_len]
            if span_embeds.shape[0] != span_len:
                raise ValueError("Image placeholder spans consume more embeddings than provided")
            embed_offset += span_len
            replay_route = next(route_iter, None) if route_iter is not None else None
            if route_iter is not None and replay_route is None:
                raise ValueError("Fewer replay routes were supplied than image spans")
            span_coordinates = None
            if position_ids_work is not None:
                if position_ids_work.shape[0] < 3:
                    raise ValueError("Conditional pruning requires three visual M-RoPE coordinate channels")
                span_coordinates = position_ids_work[-3:, batch_idx, start:end].transpose(0, 1).contiguous()
            elif replay_route is None:
                raise ValueError("New visual compression routes require image M-RoPE coordinates")
            if repeated_holitom_row:
                same_source = torch.equal(span_embeds.detach(), previous_row_sources[span_index])
                previous_coordinates = previous_row_coordinates[span_index]
                same_coordinates = (
                    span_coordinates is None
                    and previous_coordinates is None
                    or span_coordinates is not None
                    and previous_coordinates is not None
                    and torch.equal(span_coordinates, previous_coordinates)
                )
                if same_source and same_coordinates:
                    replay_route = previous_row_routes[span_index]
            reduced, route = compressor(
                span_embeds,
                replay_route,
                query_embeds=row_query_embeds,
                image_coordinates=span_coordinates,
            )
            pieces.append(reduced.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            mask_pieces.append(
                torch.ones((route.output_tokens,), device=attention_mask.device, dtype=attention_mask.dtype)
            )
            if pos_pieces is not None:
                selected = route.selected_indices.to(position_ids_work.device)
                anchor_positions = position_ids_work[:, batch_idx, start:end].index_select(1, selected)
                pos_pieces.append(anchor_positions)
                if anchor_positions.size(0) < 3:
                    raise ValueError("DART anchor coordinates require at least three M-RoPE channels")
                computed_coordinates = anchor_positions[-3:].transpose(0, 1).contiguous().to(torch.long)
                if route.anchor_coordinates is not None and not torch.equal(
                    route.anchor_coordinates.to(computed_coordinates.device), computed_coordinates
                ):
                    raise ValueError("Replayed DART route anchor coordinates do not match current M-RoPE positions")
                route = replace(route, anchor_coordinates=computed_coordinates)
                route.validate()
            used_routes.append(route)
            if holitom_merge and routes is None:
                current_row_sources.append(span_embeds.detach())
                current_row_coordinates.append(
                    span_coordinates.detach() if span_coordinates is not None else None
                )
                current_row_routes.append(route)
            last = end

        if last < seq_len:
            pieces.append(inputs_embeds[batch_idx, last:])
            mask_pieces.append(attention_mask[batch_idx, last:])
            if pos_pieces is not None:
                pos_pieces.append(position_ids_work[:, batch_idx, last:])

        row_embeds = torch.cat(pieces, dim=0)
        row_mask = torch.cat(mask_pieces, dim=0)
        row_positions = torch.cat(pos_pieces, dim=1) if pos_pieces is not None else None
        compact_padding = bool(getattr(model, "vision_token_compressor_compact_padding", False))
        if compact_padding:
            # Batched cached decoding needs every row's last physical slot to
            # be a real token.  Remove semantically inert input padding before
            # the compressed rows are re-padded below.  Training keeps the
            # legacy physical layout unless this explicit rollout-only flag is
            # enabled, so batch=1/replay numerics are unchanged.
            row_mask_bool = row_mask.to(torch.bool)
            valid_indices = torch.nonzero(row_mask_bool, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                raise ValueError("DART compact padding produced an empty prompt")
            first_valid = int(valid_indices[0].item())
            if not bool(torch.all(row_mask_bool[first_valid:]).item()):
                raise ValueError(
                    "DART compact padding only supports left-padded prompts; "
                    "an internal or right-side attention-mask gap was found"
                )
            row_embeds = row_embeds.index_select(0, valid_indices)
            row_mask = row_mask.index_select(0, valid_indices)
            if row_positions is not None:
                row_positions = row_positions.index_select(1, valid_indices)
        new_embeds.append(row_embeds)
        new_masks.append(row_mask)
        if new_positions is not None and row_positions is not None:
            new_positions.append(row_positions)
        if holitom_merge and routes is None:
            previous_row_ids = ids.detach()
            previous_row_mask = attention_mask[batch_idx].detach()
            previous_row_sources = current_row_sources
            previous_row_coordinates = current_row_coordinates
            previous_row_routes = current_row_routes

    if route_iter is not None:
        try:
            next(route_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("More replay routes were supplied than image spans")
    if embed_offset != image_embeds.size(0):
        raise ValueError(f"Unused image embeddings: used {embed_offset}, total {image_embeds.size(0)}")

    max_len = max(item.size(0) for item in new_embeds)
    padded_embeds = inputs_embeds.new_zeros((batch_size, max_len, hidden_size))
    padded_masks = attention_mask.new_zeros((batch_size, max_len))
    padded_positions = None
    if new_positions is not None:
        padded_positions = position_ids_work.new_zeros((position_ids_work.size(0), batch_size, max_len))

    compact_padding = bool(getattr(model, "vision_token_compressor_compact_padding", False))
    for batch_idx, embeds in enumerate(new_embeds):
        length = embeds.size(0)
        # Autoregressive generation consumes the final physical position.  In
        # compact rollout mode left-pad variable compressed lengths; otherwise
        # retain the original physical token layout.
        start = max_len - length if compact_padding else 0
        padded_embeds[batch_idx, start : start + length] = embeds
        padded_masks[batch_idx, start : start + length] = new_masks[batch_idx]
        if padded_positions is not None:
            if compact_padding and start:
                # Qwen3.5 uses one as the canonical masked M-RoPE pad value.
                padded_positions[:, batch_idx, :start] = 1
            padded_positions[:, batch_idx, start : start + length] = new_positions[batch_idx]

    if padded_positions is not None and squeeze_position_ids:
        padded_positions = padded_positions.squeeze(0)
    return padded_embeds, padded_masks, padded_positions, used_routes
