"""Fail-closed CPU contract for a future Qwen3.5 DPC-aware vLLM runtime.

This module intentionally imports no vLLM symbols.  It defines the numerical
and provenance boundary that a backend plugin must satisfy before the formal
rollout can change from HF: one route per unique UID, exact cluster mean,
center-token three-axis M-RoPE, the dense-grid text cursor and an exact actor
weight epoch.  A boolean capability flag is never sufficient for admission.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from collections.abc import Mapping
from typing import Any

import torch

from verl.models.transformers.vision_token_compressor import (
    HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA,
    HoliTomDPCSpatialMergeRoute,
    spatial_mean_merge_image_embeds,
)


IMPLEMENTATION_ID = "qwen35_dpc_route_replay_mean_mrope_v1"
REQUEST_SCHEMA = "vision_opd_v7_vllm_dpc_request_v1"
PARITY_SCHEMA = "vision_opd_v7_vllm_dpc_gpu_parity_v1"
ROUTE_AUTHORITY = "hf_actor_once_per_unique_uid_before_rollout_repeat_v1"
WEIGHT_EPOCH_POLICY = "exact_actor_checkpoint_epoch_match_v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _serializable_route(route: HoliTomDPCSpatialMergeRoute) -> dict[str, Any]:
    route = route.to("cpu")
    route.validate()
    if route.anchor_coordinates is None:
        raise ValueError("vLLM DPC replay requires exact three-axis center coordinates")
    expected_output = min(route.original_tokens, max(32, math.ceil(0.05 * route.original_tokens)))
    if route.output_tokens != expected_output:
        raise ValueError(
            "vLLM DPC replay route violates K=min(N,max(32,ceil(0.05N)))"
        )
    return {
        "schema_version": HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA,
        "center_indices": route.center_indices.tolist(),
        "assignment": route.assignment.tolist(),
        "source_counts": route.source_counts.tolist(),
        "original_tokens": int(route.original_tokens),
        "output_tokens": int(route.output_tokens),
        "anchor_coordinates": route.anchor_coordinates.tolist(),
    }


def build_request_envelope(
    *,
    sample_uid: str,
    actor_weight_epoch: int,
    route: HoliTomDPCSpatialMergeRoute,
    dense_text_position_cursor: int,
    rollout_repeat_count: int = 8,
) -> dict[str, Any]:
    """Serialize a group-sticky request without duplicating its DPC route."""

    if not isinstance(sample_uid, str) or _HEX64.fullmatch(sample_uid) is None:
        raise ValueError("sample_uid must be one lowercase SHA-256")
    epoch = _strict_nonnegative_int(actor_weight_epoch, name="actor_weight_epoch")
    cursor = _strict_nonnegative_int(dense_text_position_cursor, name="dense_text_position_cursor")
    if isinstance(rollout_repeat_count, bool) or rollout_repeat_count != 8:
        raise ValueError("formal DPC vLLM requests require exactly rollout_repeat_count=8")
    route_payload = _serializable_route(route)
    envelope = {
        "schema_version": REQUEST_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "route_authority": ROUTE_AUTHORITY,
        "weight_epoch_policy": WEIGHT_EPOCH_POLICY,
        "sample_uid": sample_uid,
        "actor_weight_epoch": epoch,
        "rollout_repeat_count": 8,
        "group_sticky_by_uid": True,
        "dense_text_position_cursor": cursor,
        "route_schema_version": HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA,
        "route": route_payload,
        "route_sha256": canonical_sha256(route_payload),
    }
    envelope["request_sha256"] = canonical_sha256(envelope)
    return envelope


def validate_request_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_sample_uid: str,
    expected_actor_weight_epoch: int,
) -> HoliTomDPCSpatialMergeRoute:
    required = {
        "schema_version",
        "implementation_id",
        "route_authority",
        "weight_epoch_policy",
        "sample_uid",
        "actor_weight_epoch",
        "rollout_repeat_count",
        "group_sticky_by_uid",
        "dense_text_position_cursor",
        "route_schema_version",
        "route",
        "route_sha256",
        "request_sha256",
    }
    if not isinstance(envelope, Mapping) or set(envelope) != required:
        raise ValueError("vLLM DPC request has a non-canonical field inventory")
    expected_literals = {
        "schema_version": REQUEST_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "route_authority": ROUTE_AUTHORITY,
        "weight_epoch_policy": WEIGHT_EPOCH_POLICY,
        "rollout_repeat_count": 8,
        "group_sticky_by_uid": True,
        "route_schema_version": HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA,
    }
    for key, value in expected_literals.items():
        if envelope.get(key) != value:
            raise ValueError(f"vLLM DPC request field {key} is incompatible")
    if envelope.get("sample_uid") != expected_sample_uid:
        raise ValueError("vLLM DPC request UID differs from the rollout group")
    if envelope.get("actor_weight_epoch") != _strict_nonnegative_int(
        expected_actor_weight_epoch, name="expected_actor_weight_epoch"
    ):
        raise ValueError("vLLM DPC request uses a stale or future actor weight epoch")
    _strict_nonnegative_int(envelope.get("dense_text_position_cursor"), name="dense_text_position_cursor")
    route_payload = envelope.get("route")
    if not isinstance(route_payload, Mapping):
        raise ValueError("vLLM DPC request has no route payload")
    if envelope.get("route_sha256") != canonical_sha256(route_payload):
        raise ValueError("vLLM DPC route payload differs from its SHA-256 binding")
    without_digest = dict(envelope)
    request_digest = without_digest.pop("request_sha256")
    if request_digest != canonical_sha256(without_digest):
        raise ValueError("vLLM DPC request differs from its envelope SHA-256 binding")
    expanded = {
        **dict(route_payload),
        "algorithm": "qwen35_holitom_dpc_spatial_merge_v1",
        "method": "holitom_inspired_dpc_diversity_merge",
    }
    route = HoliTomDPCSpatialMergeRoute.from_dict(expanded)
    expected_output = min(route.original_tokens, max(32, math.ceil(0.05 * route.original_tokens)))
    if route.output_tokens != expected_output:
        raise ValueError("vLLM DPC request route violates the immutable exact-K budget")
    return route


def replay_exact_dpc_inputs(
    image_embeds: torch.Tensor,
    dense_position_ids: torch.Tensor,
    envelope: Mapping[str, Any],
    *,
    expected_sample_uid: str,
    expected_actor_weight_epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the exact merged embeddings, center positions and text cursor."""

    route = validate_request_envelope(
        envelope,
        expected_sample_uid=expected_sample_uid,
        expected_actor_weight_epoch=expected_actor_weight_epoch,
    ).to(image_embeds.device)
    if dense_position_ids.shape != (3, route.original_tokens):
        raise ValueError("dense_position_ids must have shape [3, original_tokens]")
    if dense_position_ids.is_floating_point() or dense_position_ids.is_complex():
        raise TypeError("dense_position_ids must be integral")
    expected_anchor = dense_position_ids.index_select(
        1, route.center_indices.to(dense_position_ids.device)
    ).transpose(0, 1)
    if not torch.equal(expected_anchor.to(route.anchor_coordinates.device), route.anchor_coordinates):
        raise ValueError("route center coordinates differ from the dense Qwen M-RoPE grid")
    dense_cursor = int(dense_position_ids.max().item()) + 1
    if envelope.get("dense_text_position_cursor") != dense_cursor:
        raise ValueError("vLLM request rebases the post-image text cursor after pruning")
    merged = spatial_mean_merge_image_embeds(image_embeds, route)
    return merged, expected_anchor.transpose(0, 1), dense_cursor


def validate_gpu_parity_artifact(
    path: pathlib.Path,
    *,
    static_contract_summary: Mapping[str, Any],
    expected_vllm_version: str,
) -> dict[str, Any]:
    """Validate proof required before a DPC-aware vLLM backend is admitted."""

    path = pathlib.Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("vLLM parity artifact must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("vLLM parity artifact is not valid UTF-8 JSON") from exc
    required = {
        "schema_version",
        "passed",
        "implementation_id",
        "vllm_version",
        "route_schema_version",
        "static_contract",
        "model_type",
        "gpu_count",
        "cases",
        "results",
        "canonical_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("vLLM parity artifact has a non-canonical field inventory")
    without_digest = dict(payload)
    digest = without_digest.pop("canonical_sha256")
    if digest != canonical_sha256(without_digest):
        raise ValueError("vLLM parity artifact canonical SHA-256 is invalid")
    expected_static = {
        key: static_contract_summary[key]
        for key in ("schema_version", "release_variant", "canonical_sha256", "file_sha256")
    }
    if (
        payload.get("schema_version") != PARITY_SCHEMA
        or payload.get("passed") is not True
        or payload.get("implementation_id") != IMPLEMENTATION_ID
        or payload.get("vllm_version") != expected_vllm_version
        or payload.get("route_schema_version") != HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA
        or payload.get("static_contract") != expected_static
        or payload.get("model_type") != "qwen3_5"
        or payload.get("gpu_count") != 8
        or isinstance(payload.get("cases"), bool)
        or not isinstance(payload.get("cases"), int)
        or payload["cases"] < 80
    ):
        raise ValueError("vLLM parity artifact identity or coverage is incompatible")
    expected_results = {
        "route_bytes_exact": True,
        "merged_embeddings_max_abs_diff_max": 0.0,
        "mrope_position_ids_exact": True,
        "dense_text_position_cursor_exact": True,
        "prefill_logits_max_abs_diff_max": 0.0,
        "greedy_token_ids_exact": True,
        "sampled_logprob_max_abs_diff_max": 0.0,
        "weight_epoch_sync_exact": True,
        "uid_group_sticky_exact": True,
    }
    if payload.get("results") != expected_results:
        raise ValueError("vLLM parity results are not bit-exact for the admitted boundary")
    return payload


def validate_vllm_rollout_admission_fields(config: Mapping[str, Any]) -> None:
    """Reject the historical boolean-only gate and incomplete runtime claims."""

    required = {
        "dpc_aware_impl_id": IMPLEMENTATION_ID,
        "dpc_route_authority": ROUTE_AUTHORITY,
        "dpc_route_schema_version": HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA,
        "dpc_weight_epoch_policy": WEIGHT_EPOCH_POLICY,
        "dpc_group_sticky_by_uid": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"DPC-aware vLLM admission requires {key}={expected!r}")
    artifact = config.get("dpc_gpu_parity_artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        raise ValueError("DPC-aware vLLM requires a hash-bound GPU parity artifact")


def validate_formal_vllm_admission(
    config: Mapping[str, Any],
    *,
    static_contract: Mapping[str, Any] | None = None,
    static_contract_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the immutable release decision and the parity artifact itself.

    ``validate_vllm_rollout_admission_fields`` is intentionally usable by the
    rollout dataclass before the release contract is loaded.  It only proves
    that the request contains the complete field inventory.  Formal admission
    must additionally prove that the selected immutable release names vLLM as
    its backend *and* authenticate the referenced GPU parity artifact.  Keeping
    these two levels separate avoids treating a non-empty path as evidence.
    """

    validate_vllm_rollout_admission_fields(config)
    if (static_contract is None) != (static_contract_summary is None):
        raise ValueError(
            "static_contract and static_contract_summary must be supplied together"
        )
    if static_contract is None:
        from training.contract import load_contract

        static_contract, static_contract_summary = load_contract()
    assert static_contract_summary is not None

    rollout_contract = static_contract.get("rollout")
    if not isinstance(rollout_contract, Mapping):
        raise ValueError("selected immutable release has no rollout contract")
    vllm_contract = rollout_contract.get("vllm")
    if not isinstance(vllm_contract, Mapping):
        raise ValueError("selected immutable release has no DPC-aware vLLM contract")
    expected_contract = {
        "admission": "gpu_parity_admitted_formal_v1",
        "implementation_id": IMPLEMENTATION_ID,
        "route_authority": ROUTE_AUTHORITY,
        "route_schema_version": HOLITOM_DPC_SPATIAL_MERGE_ROUTE_SCHEMA,
        "weight_epoch_policy": WEIGHT_EPOCH_POLICY,
        "group_sticky_by_uid": True,
        "requires_gpu_parity_artifact": True,
        "stock_qwen35_multimodal_pruning_supported": False,
    }
    if rollout_contract.get("formal_backend") != "vllm":
        raise ValueError(
            "selected immutable release does not admit vLLM as its formal rollout backend"
        )
    if dict(vllm_contract) != expected_contract:
        raise ValueError("selected immutable release has not admitted the exact vLLM contract")
    compatibility = static_contract.get("compatibility")
    if (
        not isinstance(compatibility, Mapping)
        or compatibility.get("formal_vllm_allowed_without_gpu_parity_artifact") is not False
    ):
        raise ValueError("formal vLLM must remain fail-closed without a parity artifact")
    runtime = static_contract.get("runtime")
    optional_versions = runtime.get("optional_backend_versions") if isinstance(runtime, Mapping) else None
    expected_version = (
        optional_versions.get("vllm") if isinstance(optional_versions, Mapping) else None
    )
    if not isinstance(expected_version, str) or not expected_version:
        raise ValueError("selected immutable release does not pin a vLLM version")
    return validate_gpu_parity_artifact(
        pathlib.Path(str(config["dpc_gpu_parity_artifact"])),
        static_contract_summary=static_contract_summary,
        expected_vllm_version=expected_version,
    )
