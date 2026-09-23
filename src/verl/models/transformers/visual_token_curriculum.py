"""Deterministic optimizer-step curriculum for visual-token retention.

The schedule is deliberately integer-only.  It is driven by the number of
successfully committed optimizer updates *before* the current update and is
therefore invariant to rollout count, micro-batching, teacher chunks, retries,
wall time, and diagnostic validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Mapping


CURRICULUM_SCHEMA_VERSION = "vision_opd_v8_visual_token_curriculum_v2"
CURRICULUM_DRIVER = "completed_optimizer_steps_before_current_update"
CURRICULUM_SCHEDULE_TYPE = "warmup_hold_raised_cosine_frozen_bps"
CURRICULUM_VALIDATION_CONTROL = "forbidden_diagnostic_only"
CURRICULUM_ROUNDING_MODE = "round_half_up"
CURRICULUM_RETENTION_BPS_VECTOR_SHA256 = (
    "deea218dec57429ecfc439fe7aca1bc68ad72ef17af40f65b69e4947985642fa"
)
CURRICULUM_SCHEDULE_SHA256 = (
    "a02cbf6e99fa6dc002373656ee6d2492b55c29f968df3c9558b67252633f9a9b"
)


# Offline-frozen, integer basis-point schedule.  It is the round-half-up
# realization of
#   500 + 1000 * (1 + cos(pi * (c - 13) / 87))  for 14 <= c <= 99,
# with a 25% hold for c=0..13 and the immutable 5% plateau for c=100..174.
# Runtime code intentionally performs no floating-point or trigonometric work.
RETENTION_BPS_BY_COMPLETED_STEP = (
    2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500,
    2500, 2500, 2500, 2500, 2499, 2497, 2494, 2490, 2484, 2477,
    2468, 2459, 2448, 2436, 2422, 2408, 2392, 2375, 2357, 2338,
    2317, 2296, 2274, 2250, 2226, 2201, 2174, 2147, 2119, 2091,
    2061, 2031, 2000, 1968, 1936, 1903, 1870, 1836, 1802, 1768,
    1733, 1697, 1662, 1626, 1590, 1554, 1518, 1482, 1446, 1410,
    1374, 1338, 1303, 1267, 1232, 1198, 1164, 1130, 1097, 1064,
    1032, 1000, 969, 939, 909, 881, 853, 826, 799, 774,
    750, 726, 704, 683, 662, 643, 625, 608, 592, 578,
    564, 552, 541, 532, 523, 516, 510, 506, 503, 501,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500,
)


def _require_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_frozen_retention_vector() -> None:
    if len(RETENTION_BPS_BY_COMPLETED_STEP) != 175:
        raise RuntimeError("frozen V8 curriculum retention vector must contain exactly 175 updates")
    actual_hash = _canonical_sha256(RETENTION_BPS_BY_COMPLETED_STEP)
    if actual_hash != CURRICULUM_RETENTION_BPS_VECTOR_SHA256:
        raise RuntimeError(
            "frozen V8 curriculum retention vector hash drift: "
            f"expected={CURRICULUM_RETENTION_BPS_VECTOR_SHA256}, actual={actual_hash}"
        )
    golden = {0: 2500, 13: 2500, 14: 2499, 99: 501, 100: 500, 174: 500}
    if any(RETENTION_BPS_BY_COMPLETED_STEP[index] != value for index, value in golden.items()):
        raise RuntimeError("frozen V8 curriculum retention vector golden boundary drift")
    if any(
        left < right
        for left, right in zip(
            RETENTION_BPS_BY_COMPLETED_STEP[:-1],
            RETENTION_BPS_BY_COMPLETED_STEP[1:],
            strict=True,
        )
    ):
        raise RuntimeError("frozen V8 curriculum retention vector must be monotone non-increasing")
    if len(set(RETENTION_BPS_BY_COMPLETED_STEP)) != 88:
        raise RuntimeError("frozen V8 curriculum retention vector must contain exactly 88 stages")
    if sum(RETENTION_BPS_BY_COMPLETED_STEP) != 201_500:
        raise RuntimeError("frozen V8 curriculum retention-vector area drift")
    if RETENTION_BPS_BY_COMPLETED_STEP.count(2500) != 14:
        raise RuntimeError("frozen V8 curriculum must hold 25% for exactly 14 updates")
    if RETENTION_BPS_BY_COMPLETED_STEP.count(500) != 75:
        raise RuntimeError("frozen V8 curriculum must hold 5% for exactly 75 updates")
    drops = tuple(
        left - right
        for left, right in zip(
            RETENTION_BPS_BY_COMPLETED_STEP[:-1],
            RETENTION_BPS_BY_COMPLETED_STEP[1:],
            strict=True,
        )
    )
    if max(drops) != 36 or sum(drop > 0 for drop in drops) != 87:
        raise RuntimeError("frozen V8 curriculum transition smoothness/stage-count drift")
    if any(
        RETENTION_BPS_BY_COMPLETED_STEP[c]
        + RETENTION_BPS_BY_COMPLETED_STEP[113 - c]
        != 3000
        for c in range(14, 100)
    ):
        raise RuntimeError("frozen V8 curriculum raised-cosine symmetry drift")


_validate_frozen_retention_vector()


@dataclass(frozen=True)
class VisualTokenCurriculum:
    """Immutable 25% -> 5% retention schedule for the 175-update release."""

    schema_version: str = CURRICULUM_SCHEMA_VERSION
    driver: str = CURRICULUM_DRIVER
    schedule_type: str = CURRICULUM_SCHEDULE_TYPE
    total_optimizer_steps: int = 175
    start_retention_bps: int = 2500
    final_retention_bps: int = 500
    warmup_hold_optimizer_steps: int = 14
    final_plateau_start_completed_steps: int = 100
    rounding_mode: str = CURRICULUM_ROUNDING_MODE
    retention_bps_vector_sha256: str = CURRICULUM_RETENTION_BPS_VECTOR_SHA256
    minimum_tokens_per_image: int = 32
    validation_control: str = CURRICULUM_VALIDATION_CONTROL

    def __post_init__(self) -> None:
        if self.schema_version != CURRICULUM_SCHEMA_VERSION:
            raise ValueError(f"unsupported curriculum schema: {self.schema_version!r}")
        if self.driver != CURRICULUM_DRIVER:
            raise ValueError(f"unsupported curriculum driver: {self.driver!r}")
        if self.schedule_type != CURRICULUM_SCHEDULE_TYPE:
            raise ValueError(f"unsupported curriculum schedule type: {self.schedule_type!r}")
        if self.validation_control != CURRICULUM_VALIDATION_CONTROL:
            raise ValueError("diagnostic validation is forbidden from controlling the curriculum")
        if self.rounding_mode != CURRICULUM_ROUNDING_MODE:
            raise ValueError(f"unsupported curriculum rounding mode: {self.rounding_mode!r}")
        if self.retention_bps_vector_sha256 != CURRICULUM_RETENTION_BPS_VECTOR_SHA256:
            raise ValueError(
                "curriculum retention vector hash drift: "
                f"expected={CURRICULUM_RETENTION_BPS_VECTOR_SHA256}, "
                f"actual={self.retention_bps_vector_sha256!r}"
            )
        total = _require_int(self.total_optimizer_steps, name="total_optimizer_steps", minimum=1)
        start = _require_int(self.start_retention_bps, name="start_retention_bps", minimum=1)
        final = _require_int(self.final_retention_bps, name="final_retention_bps", minimum=1)
        warmup = _require_int(
            self.warmup_hold_optimizer_steps,
            name="warmup_hold_optimizer_steps",
            minimum=1,
        )
        plateau = _require_int(
            self.final_plateau_start_completed_steps,
            name="final_plateau_start_completed_steps",
            minimum=0,
        )
        _require_int(self.minimum_tokens_per_image, name="minimum_tokens_per_image", minimum=1)
        expected_scalars = {
            "total_optimizer_steps": (total, 175),
            "start_retention_bps": (start, 2500),
            "final_retention_bps": (final, 500),
            "warmup_hold_optimizer_steps": (warmup, 14),
            "final_plateau_start_completed_steps": (plateau, 100),
            "minimum_tokens_per_image": (self.minimum_tokens_per_image, 32),
        }
        drift = {
            name: {"expected": expected, "actual": actual}
            for name, (actual, expected) in expected_scalars.items()
            if actual != expected
        }
        if drift:
            raise ValueError(f"immutable V8 curriculum scalar drift: {drift}")
        if self.schedule_sha256 != CURRICULUM_SCHEDULE_SHA256:
            raise RuntimeError(
                "immutable V8 curriculum identity hash drift: "
                f"expected={CURRICULUM_SCHEDULE_SHA256}, actual={self.schedule_sha256}"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VisualTokenCurriculum":
        if not isinstance(value, Mapping):
            raise TypeError("visual-token curriculum config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        expected_inventory = allowed | {"enabled", "schedule_sha256"}
        actual_inventory = set(value)
        if actual_inventory != expected_inventory:
            raise ValueError(
                "non-canonical visual-token curriculum field inventory: "
                f"missing={sorted(expected_inventory - actual_inventory)}, "
                f"unexpected={sorted(actual_inventory - expected_inventory)}"
            )
        if value.get("enabled") is not True:
            raise ValueError("visual-token curriculum must be explicitly enabled")
        schedule = cls(**{key: value[key] for key in allowed if key in value})
        declared_hash = value.get("schedule_sha256")
        if not isinstance(declared_hash, str) or declared_hash != schedule.schedule_sha256:
            raise ValueError(
                "visual-token curriculum schedule hash drift: "
                f"expected={schedule.schedule_sha256}, actual={declared_hash!r}"
            )
        return schedule

    @property
    def identity(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def schedule_sha256(self) -> str:
        return _canonical_sha256(self.identity)

    @property
    def final_plateau_updates(self) -> int:
        return self.total_optimizer_steps - self.final_plateau_start_completed_steps

    def retention_bps(self, completed_optimizer_steps: int) -> int:
        completed = _require_int(
            completed_optimizer_steps,
            name="completed_optimizer_steps",
            minimum=0,
        )
        if completed >= self.total_optimizer_steps:
            raise ValueError(
                "completed_optimizer_steps is outside the training horizon: "
                f"completed={completed}, total={self.total_optimizer_steps}"
            )
        return RETENTION_BPS_BY_COMPLETED_STEP[completed]

    def stage_index(self, completed_optimizer_steps: int) -> int:
        completed = _require_int(
            completed_optimizer_steps,
            name="completed_optimizer_steps",
            minimum=0,
        )
        if completed >= self.total_optimizer_steps:
            raise ValueError("completed_optimizer_steps is outside the training horizon")
        if completed < self.warmup_hold_optimizer_steps:
            return 0
        return min(completed - (self.warmup_hold_optimizer_steps - 1), 87)

    def budget(self, original_tokens: int, *, completed_optimizer_steps: int) -> int:
        return visual_token_budget(
            original_tokens,
            retention_bps=self.retention_bps(completed_optimizer_steps),
            minimum_tokens=self.minimum_tokens_per_image,
        )

    def runtime_state(self, completed_optimizer_steps: int) -> dict[str, Any]:
        completed = _require_int(
            completed_optimizer_steps,
            name="completed_optimizer_steps",
            minimum=0,
        )
        return {
            "schema_version": self.schema_version,
            "schedule_sha256": self.schedule_sha256,
            "completed_optimizer_steps": completed,
            "next_optimizer_step": completed + 1,
            "retention_bps": self.retention_bps(completed),
            "stage_index": self.stage_index(completed),
            "minimum_tokens_per_image": self.minimum_tokens_per_image,
            "validation_control": self.validation_control,
        }


def visual_token_budget(original_tokens: int, *, retention_bps: int, minimum_tokens: int = 32) -> int:
    """Return ``min(N, max(minimum, ceil(N * bps / 10000)))`` exactly."""

    tokens = _require_int(original_tokens, name="original_tokens", minimum=1)
    bps = _require_int(retention_bps, name="retention_bps", minimum=1)
    minimum = _require_int(minimum_tokens, name="minimum_tokens", minimum=1)
    if bps > 10_000:
        raise ValueError("retention_bps must be <= 10000")
    retained = (tokens * bps + 9_999) // 10_000
    return min(tokens, max(minimum, retained))


def curriculum_merged_prompt_tokens(profile: Mapping[str, Any], *, retention_bps: int) -> int:
    """Recompute a materialized prompt cost for the active curriculum stage."""

    if not isinstance(profile, Mapping):
        raise TypeError("visual capacity profile must be a mapping")
    dense_visual = _require_int(profile.get("dense_visual_tokens"), name="dense_visual_tokens", minimum=1)
    dense_prompt = _require_int(
        profile.get("dense_teacher_prompt_tokens"),
        name="dense_teacher_prompt_tokens",
        minimum=1,
    )
    non_visual = dense_prompt - dense_visual
    if non_visual <= 0:
        raise ValueError("dense prompt does not contain a positive non-visual token count")
    return non_visual + visual_token_budget(dense_visual, retention_bps=retention_bps, minimum_tokens=32)
