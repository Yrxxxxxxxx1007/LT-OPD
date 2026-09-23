"""Fail-closed access to the release-08 operational capacity census.

Release 08 starts the student at 25% visual-token retention and later decays
to 5%.  The materialized row profile retained for backward compatibility is
the final 5% profile, so launch, recovery, and runtime admission must never
read their peak cost from that field.  This module is the single consumer-side
interpretation of the all-stage census emitted by the V8 data builder.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from verl.models.transformers.visual_token_curriculum import VisualTokenCurriculum


V8_STATIC_SCHEMA = "vision_opd_ai4s_v8_fullimage_curriculum_static_contract_v1"
V8_DATA_SCHEMA = "vision_opd_v8_vqa14k_fullimage_curriculum_raw_data_v1"
V8_CENSUS_SCHEMA = "vision_opd_v8_visual_token_curriculum_capacity_census_v1"
_HEX64 = re.compile(r"[0-9a-f]{64}")

_ROOT_KEYS = {
    "schema_version",
    "schedule_sha256",
    "driver",
    "minimum_tokens_per_image",
    "validation_control",
    "rows",
    "stages",
    "operational_worst_stage",
    "all_stages_all_rows_within_capacity",
}
_STAGE_KEYS = {"optimizer_updates", "profiles"}
_PROFILE_KEYS = {
    "visual_budget_tokens",
    "merged_prompt_tokens",
    "vopd_composite_cost_upper_bound",
    "all_rows_within_capacity",
    "row_binding_schema",
    "row_binding_sha256",
}
_SUMMARY_KEYS = {
    "count",
    "min",
    "max",
    "mean",
    "p50_nearest_rank",
    "p90_nearest_rank",
    "p95_nearest_rank",
    "p99_nearest_rank",
}
_WORST_KEYS = {
    "retention_bps",
    "stage_key",
    "profile_cost_field",
    "stage_profiles_sha256",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} keys drift: expected={sorted(expected)}, actual={actual}")
    return value


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _validate_summary(value: Any, *, rows: int, label: str) -> Mapping[str, Any]:
    summary = _require_exact_keys(value, _SUMMARY_KEYS, label=label)
    if summary["count"] != rows:
        raise ValueError(f"{label}.count must equal the complete train row count")
    ordered_names = (
        "min",
        "p50_nearest_rank",
        "p90_nearest_rank",
        "p95_nearest_rank",
        "p99_nearest_rank",
        "max",
    )
    ordered = [_positive_int(summary[name], label=f"{label}.{name}") for name in ordered_names]
    if ordered != sorted(ordered):
        raise ValueError(f"{label} nearest-rank summaries are not monotone")
    mean = summary["mean"]
    if (
        isinstance(mean, bool)
        or not isinstance(mean, (int, float))
        or not math.isfinite(float(mean))
    ):
        raise ValueError(f"{label}.mean must be finite numeric data")
    if not ordered[0] <= float(mean) <= ordered[-1]:
        raise ValueError(f"{label}.mean is outside its observed range")
    return summary


def operational_curriculum_profile(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    max_pixels: int,
) -> Mapping[str, Any] | None:
    """Return the authenticated peak-stage profile, or ``None`` for legacy data.

    A full-image V8 contract/data mismatch is always an error.  Every stage is
    checked, including its exact update multiplicity, and the declared first
    stage is proven no smaller than every later stage for both student prompt
    and full composite admission cost.
    """

    is_v8_contract = contract.get("schema_version") == V8_STATIC_SCHEMA
    is_v8_data = manifest.get("data_schema_version") == V8_DATA_SCHEMA
    if is_v8_contract != is_v8_data:
        raise ValueError(
            "release-08 capacity lookup requires matching full-image static and data schemas"
        )
    if not is_v8_contract:
        return None

    prompt_census = manifest.get("prompt_capacity_census")
    prompt_profiles = (
        prompt_census.get("profiles") if isinstance(prompt_census, Mapping) else None
    )
    if not isinstance(prompt_profiles, Mapping) or str(max_pixels) not in prompt_profiles:
        raise ValueError("release-08 prompt census lacks the selected max_pixels profile")

    curriculum_mapping = contract.get("compressor", {}).get("curriculum")
    if not isinstance(curriculum_mapping, Mapping) or curriculum_mapping.get("enabled") is not True:
        raise ValueError("release-08 static contract lacks its enabled curriculum")
    curriculum = VisualTokenCurriculum.from_mapping(curriculum_mapping)
    census = _require_exact_keys(
        manifest.get("curriculum_capacity_census"),
        _ROOT_KEYS,
        label="release-08 curriculum capacity census",
    )
    expected_identity = {
        "schema_version": V8_CENSUS_SCHEMA,
        "schedule_sha256": curriculum.schedule_sha256,
        "driver": curriculum.driver,
        "minimum_tokens_per_image": curriculum.minimum_tokens_per_image,
        "validation_control": curriculum.validation_control,
        "rows": manifest.get("train_rows"),
        "all_stages_all_rows_within_capacity": True,
    }
    for key, expected in expected_identity.items():
        if census.get(key) != expected:
            raise ValueError(
                f"release-08 curriculum capacity census {key} drift: "
                f"expected={expected!r}, actual={census.get(key)!r}"
            )
    rows = _positive_int(census["rows"], label="curriculum census rows")

    expected_counts = collections.Counter(
        curriculum.retention_bps(completed)
        for completed in range(curriculum.total_optimizer_steps)
    )
    stages = census.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != {
        str(retention) for retention in expected_counts
    }:
        raise ValueError("release-08 curriculum capacity stage inventory drift")
    expected_profile_keys = set(prompt_profiles)
    validated_profiles: dict[str, Mapping[str, Any]] = {}
    for retention_bps, expected_updates in expected_counts.items():
        stage_key = str(retention_bps)
        stage = _require_exact_keys(
            stages[stage_key], _STAGE_KEYS, label=f"curriculum stage {stage_key}"
        )
        if stage["optimizer_updates"] != expected_updates:
            raise ValueError(f"curriculum stage {stage_key} update multiplicity drift")
        profiles = stage["profiles"]
        if not isinstance(profiles, Mapping) or set(profiles) != expected_profile_keys:
            raise ValueError(f"curriculum stage {stage_key} max_pixels inventory drift")
        profile = _require_exact_keys(
            profiles[str(max_pixels)],
            _PROFILE_KEYS,
            label=f"curriculum stage {stage_key} selected profile",
        )
        if profile["all_rows_within_capacity"] is not True:
            raise ValueError(f"curriculum stage {stage_key} contains a capacity overflow")
        expected_binding_schema = (
            "canonical_sha256_list_of_sample_uid_visual_budget_merged_prompt_tokens_"
            "vopd_composite_cost_upper_bound_v1"
        )
        if profile["row_binding_schema"] != expected_binding_schema:
            raise ValueError(f"curriculum stage {stage_key} row-binding schema drift")
        binding = profile["row_binding_sha256"]
        if not isinstance(binding, str) or _HEX64.fullmatch(binding) is None:
            raise ValueError(f"curriculum stage {stage_key} row-binding SHA-256 is invalid")
        _validate_summary(
            profile["visual_budget_tokens"], rows=rows, label=f"stage {stage_key} visual budget"
        )
        _validate_summary(
            profile["merged_prompt_tokens"], rows=rows, label=f"stage {stage_key} merged prompt"
        )
        _validate_summary(
            profile["vopd_composite_cost_upper_bound"],
            rows=rows,
            label=f"stage {stage_key} composite cost",
        )
        validated_profiles[stage_key] = profile

    first_retention = curriculum.retention_bps(0)
    first_key = str(first_retention)
    worst = _require_exact_keys(
        census.get("operational_worst_stage"),
        _WORST_KEYS,
        label="curriculum operational worst stage",
    )
    expected_worst = {
        "retention_bps": first_retention,
        "stage_key": first_key,
        "profile_cost_field": "vopd_composite_cost_upper_bound",
        "stage_profiles_sha256": _canonical_sha256(stages[first_key]["profiles"]),
    }
    if dict(worst) != expected_worst:
        raise ValueError("release-08 operational worst-stage binding drift")
    first_profile = validated_profiles[first_key]
    first_merged = first_profile["merged_prompt_tokens"]["max"]
    first_composite = first_profile["vopd_composite_cost_upper_bound"]["max"]
    for stage_key, profile in validated_profiles.items():
        if profile["merged_prompt_tokens"]["max"] > first_merged:
            raise ValueError(f"curriculum stage {stage_key} exceeds declared peak merged prompt")
        if profile["vopd_composite_cost_upper_bound"]["max"] > first_composite:
            raise ValueError(f"curriculum stage {stage_key} exceeds declared peak composite cost")
    return first_profile


def operational_composite_cost_max(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    max_pixels: int,
) -> int:
    """Return one-row peak composite cost for V8, legacy fixed-profile cost otherwise."""

    operational = operational_curriculum_profile(
        manifest, contract, max_pixels=max_pixels
    )
    if operational is not None:
        value = operational["vopd_composite_cost_upper_bound"]["max"]
    else:
        census = manifest.get("prompt_capacity_census")
        profiles = census.get("profiles") if isinstance(census, Mapping) else None
        profile = profiles.get(str(max_pixels)) if isinstance(profiles, Mapping) else None
        metrics = profile.get("metrics") if isinstance(profile, Mapping) else None
        composite = (
            metrics.get("vopd_composite_cost_upper_bound")
            if isinstance(metrics, Mapping)
            else None
        )
        value = composite.get("max") if isinstance(composite, Mapping) else None
    return _positive_int(value, label="operational maximum composite cost")
