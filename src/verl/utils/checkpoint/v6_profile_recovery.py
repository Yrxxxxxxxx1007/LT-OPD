"""Fail-closed runtime-profile recovery for the formal AI4S V6 run.

The launch contract and checkpoint provenance bind the initially profiled
geometry.  A small, append-only ledger may subsequently authorize only an
adjacent move towards a more conservative runtime micro-profile.  The current
effective profile is deliberately checkpointed outside immutable provenance so
that an OOM restart can still load the last complete checkpoint without making
scientific settings mutable.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any


RUNTIME_PROFILE_STATE_SCHEMA = "vision_opd_ai4s_v6_runtime_profile_state_v1"
RECOVERY_LEDGER_ENTRY_SCHEMA = "vision_opd_ai4s_v6_profile_recovery_entry_v1"
PROFILE_ORDER = ("M8", "M7", "M6", "M5", "T4", "T3", "B2", "S1")
ZERO_SHA256 = "0" * 64

_LEDGER_ENTRY_KEYS = frozenset(
    {
        "schema_version",
        "sequence",
        "from_profile",
        "to_profile",
        "from_max_composite_cost_per_gpu",
        "to_max_composite_cost_per_gpu",
        "max_pixels",
        "train_batch_size",
        "ppo_mini_batch_size",
        "rollout_n",
        "launch_contract_sha256",
        "preflight_sha256",
        "previous_entry_sha256",
    }
)
_RUNTIME_STATE_KEYS = frozenset(
    {
        "schema_version",
        "initial_profile",
        "effective_profile",
        "initial_max_composite_cost_per_gpu",
        "effective_max_composite_cost_per_gpu",
        "runtime_profile_ladder",
        "max_pixels",
        "train_batch_size",
        "ppo_mini_batch_size",
        "rollout_n",
        "launch_contract_sha256",
        "effective_preflight_sha256",
        "run_root",
        "recovery_ledger",
    }
)
_RECOVERY_LEDGER_STATE_KEYS = frozenset(
    {
        "path",
        "file_sha256",
        "entry_count",
        "entry_sha256s",
        "tail_entry_sha256",
        "entries",
    }
)


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} has a non-canonical field inventory: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, label: str, *, allow_zero: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or (not allow_zero and value == ZERO_SHA256)
    ):
        raise ValueError(f"{label} must be a lowercase nonzero SHA-256")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def canonical_recovery_entry_bytes(entry: Mapping[str, Any]) -> bytes:
    """Return the semantic bytes used by the ledger hash chain."""

    return (
        json.dumps(dict(entry), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def canonical_recovery_entry_sha256(entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_recovery_entry_bytes(entry)).hexdigest()


def _normalize_ladder(
    micro_profiles: Sequence[Mapping[str, Any]],
    initial_profile: str,
) -> list[dict[str, Any]]:
    if not isinstance(initial_profile, str) or initial_profile not in PROFILE_ORDER:
        raise ValueError("initial profile must be one of M8/M7/M6/M5/T4/T3/B2/S1")
    profiles = [dict(profile) for profile in micro_profiles]
    names = [profile.get("name") for profile in profiles]
    admitted_ladders = (list(PROFILE_ORDER), list(PROFILE_ORDER[1:]))
    if names not in admitted_ladders:
        raise ValueError(
            "runtime micro-profile ladder must be exactly the V6/V8 M8..S1 or "
            f"V7 M7..S1 ladder, got {names}"
        )
    if initial_profile not in names:
        raise ValueError("initial profile is not admitted by the release micro-profile ladder")
    required_profile_keys = {"name", "max_trajectories", "decode_batch_size", "preserve_cuda_cache"}
    for profile in profiles:
        if set(profile) != required_profile_keys:
            raise ValueError(f"runtime profile {profile.get('name')} has a non-canonical field inventory")
        _require_positive_int(profile["max_trajectories"], f"{profile['name']}.max_trajectories")
        _require_positive_int(profile["decode_batch_size"], f"{profile['name']}.decode_batch_size")
        if not isinstance(profile["preserve_cuda_cache"], bool):
            raise ValueError(f"{profile['name']}.preserve_cuda_cache must be boolean")
    return profiles[names.index(initial_profile) :]


def _read_and_validate_ledger(
    *,
    ledger_path: str | None,
    ledger_sha256: str | None,
    run_root: str,
    initial_profile: str,
    effective_profile: str,
    initial_max_cost: int,
    effective_max_cost: int,
    max_pixels: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    rollout_n: int,
    launch_contract_sha256: str,
    effective_preflight_sha256: str,
) -> dict[str, Any]:
    expected_path = os.path.join(run_root, "recovery", "profile_recovery.jsonl")
    if bool(ledger_path) != bool(ledger_sha256):
        raise ValueError("V6 recovery requires both ledger path and ledger SHA-256, or neither")
    if not ledger_path:
        if os.path.exists(expected_path):
            raise ValueError("V6 recovery ledger exists but is not hash-bound by the launcher environment")
        if effective_profile != initial_profile or effective_max_cost != initial_max_cost:
            raise ValueError("Without a recovery ledger, effective profile/cost must equal the initial profile/cost")
        return {
            "path": None,
            "file_sha256": None,
            "entry_count": 0,
            "entry_sha256s": [],
            "tail_entry_sha256": ZERO_SHA256,
            "entries": [],
        }

    ledger_path = os.path.abspath(os.path.expanduser(str(ledger_path)))
    if ledger_path != expected_path:
        raise ValueError(f"V6 recovery ledger must be exactly {expected_path}, got {ledger_path}")
    if os.path.islink(ledger_path) or os.path.realpath(ledger_path) != ledger_path:
        raise ValueError("V6 recovery ledger must be a regular non-symlink path")
    if not os.path.isfile(ledger_path):
        raise FileNotFoundError(f"V6 recovery ledger does not exist: {ledger_path}")
    ledger_sha256 = _require_sha256(ledger_sha256, "recovery ledger SHA-256")
    with open(ledger_path, "rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != ledger_sha256:
        raise ValueError("V6 recovery ledger bytes differ from VERL_V6_RECOVERY_LEDGER_SHA256")
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("V6 recovery ledger must be nonempty canonical LF-terminated JSONL")
    raw_lines = raw.splitlines(keepends=True)
    if not raw_lines or any(not line.strip() for line in raw_lines):
        raise ValueError("V6 recovery ledger contains an empty record")

    entries: list[dict[str, Any]] = []
    entry_hashes: list[str] = []
    previous_profile = initial_profile
    previous_cost = initial_max_cost
    previous_hash = ZERO_SHA256
    ladder_names = list(PROFILE_ORDER[PROFILE_ORDER.index(initial_profile) :])
    for index, raw_line in enumerate(raw_lines, start=1):
        try:
            entry = json.loads(raw_line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"V6 recovery ledger line {index} is not valid canonical UTF-8 JSON") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"V6 recovery ledger line {index} must be an object")
        _require_exact_keys(entry, _LEDGER_ENTRY_KEYS, f"V6 recovery ledger line {index}")
        if raw_line != canonical_recovery_entry_bytes(entry):
            raise ValueError(f"V6 recovery ledger line {index} is not canonical JSON")
        if entry["schema_version"] != RECOVERY_LEDGER_ENTRY_SCHEMA:
            raise ValueError(f"V6 recovery ledger line {index} has an invalid schema")
        if entry["sequence"] != index:
            raise ValueError(f"V6 recovery ledger sequence must be contiguous from one; line={index}")
        if entry["from_profile"] != previous_profile:
            raise ValueError(f"V6 recovery ledger line {index} does not continue the prior profile")
        if previous_profile not in ladder_names or entry["to_profile"] not in ladder_names:
            raise ValueError(f"V6 recovery ledger line {index} leaves the initial profile ladder")
        from_index = PROFILE_ORDER.index(previous_profile)
        to_index = PROFILE_ORDER.index(entry["to_profile"])
        if to_index != from_index + 1:
            raise ValueError(
                f"V6 recovery ledger line {index} must move exactly one adjacent profile towards S1"
            )
        from_cost = _require_positive_int(
            entry["from_max_composite_cost_per_gpu"],
            f"V6 recovery ledger line {index} from cost",
        )
        to_cost = _require_positive_int(
            entry["to_max_composite_cost_per_gpu"],
            f"V6 recovery ledger line {index} to cost",
        )
        if from_cost != previous_cost or to_cost > from_cost:
            raise ValueError(f"V6 recovery ledger line {index} must continue a nonincreasing cost sequence")
        exact_invariants = {
            "max_pixels": max_pixels,
            "train_batch_size": train_batch_size,
            "ppo_mini_batch_size": ppo_mini_batch_size,
            "rollout_n": rollout_n,
            "launch_contract_sha256": launch_contract_sha256,
        }
        for key, expected in exact_invariants.items():
            if entry[key] != expected:
                raise ValueError(f"V6 recovery ledger line {index} changed immutable {key}")
        _require_sha256(entry["preflight_sha256"], f"V6 recovery ledger line {index} preflight")
        _require_sha256(
            entry["previous_entry_sha256"],
            f"V6 recovery ledger line {index} previous-entry hash",
            allow_zero=index == 1,
        )
        if entry["previous_entry_sha256"] != previous_hash:
            raise ValueError(f"V6 recovery ledger line {index} breaks the append-only hash chain")
        entry_hash = canonical_recovery_entry_sha256(entry)
        entries.append(entry)
        entry_hashes.append(entry_hash)
        previous_profile = entry["to_profile"]
        previous_cost = to_cost
        previous_hash = entry_hash

    if previous_profile != effective_profile or previous_cost != effective_max_cost:
        raise ValueError("V6 recovery ledger tail disagrees with the launcher effective profile/cost")
    if entries[-1]["preflight_sha256"] != effective_preflight_sha256:
        raise ValueError("V6 recovery ledger tail does not bind the effective-profile preflight")
    return {
        "path": ledger_path,
        "file_sha256": ledger_sha256,
        "entry_count": len(entries),
        "entry_sha256s": entry_hashes,
        "tail_entry_sha256": entry_hashes[-1],
        "entries": entries,
    }


def build_v6_runtime_profile_state(
    *,
    micro_profiles: Sequence[Mapping[str, Any]],
    initial_profile: str,
    effective_profile: str,
    initial_max_composite_cost_per_gpu: int,
    effective_max_composite_cost_per_gpu: int,
    max_pixels: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    rollout_n: int,
    launch_contract_sha256: str,
    effective_preflight_sha256: str,
    run_root: str,
    recovery_ledger_path: str | None = None,
    recovery_ledger_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve and fully validate the launcher's current V6 runtime profile."""

    initial_cost = _require_positive_int(
        initial_max_composite_cost_per_gpu,
        "initial max composite cost per GPU",
    )
    effective_cost = _require_positive_int(
        effective_max_composite_cost_per_gpu,
        "effective max composite cost per GPU",
    )
    max_pixels = _require_positive_int(max_pixels, "max_pixels")
    train_batch_size = _require_positive_int(train_batch_size, "train_batch_size")
    ppo_mini_batch_size = _require_positive_int(ppo_mini_batch_size, "ppo_mini_batch_size")
    rollout_n = _require_positive_int(rollout_n, "rollout_n")
    launch_contract_sha256 = _require_sha256(launch_contract_sha256, "launch contract SHA-256")
    effective_preflight_sha256 = _require_sha256(effective_preflight_sha256, "effective preflight SHA-256")
    if not isinstance(effective_profile, str) or effective_profile not in PROFILE_ORDER:
        raise ValueError("effective profile must be one of M8/M7/M6/M5/T4/T3/B2/S1")
    ladder = _normalize_ladder(micro_profiles, initial_profile)
    if effective_profile not in {profile["name"] for profile in ladder}:
        raise ValueError("effective profile cannot be more aggressive than the initial profile")
    if not isinstance(run_root, str) or not os.path.isabs(run_root):
        raise ValueError("formal V6 run root must be absolute")
    run_root = os.path.abspath(run_root)
    ledger = _read_and_validate_ledger(
        ledger_path=recovery_ledger_path,
        ledger_sha256=recovery_ledger_sha256,
        run_root=run_root,
        initial_profile=initial_profile,
        effective_profile=effective_profile,
        initial_max_cost=initial_cost,
        effective_max_cost=effective_cost,
        max_pixels=max_pixels,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        rollout_n=rollout_n,
        launch_contract_sha256=launch_contract_sha256,
        effective_preflight_sha256=effective_preflight_sha256,
    )
    return {
        "schema_version": RUNTIME_PROFILE_STATE_SCHEMA,
        "initial_profile": initial_profile,
        "effective_profile": effective_profile,
        "initial_max_composite_cost_per_gpu": initial_cost,
        "effective_max_composite_cost_per_gpu": effective_cost,
        "runtime_profile_ladder": ladder,
        "max_pixels": max_pixels,
        "train_batch_size": train_batch_size,
        "ppo_mini_batch_size": ppo_mini_batch_size,
        "rollout_n": rollout_n,
        "launch_contract_sha256": launch_contract_sha256,
        "effective_preflight_sha256": effective_preflight_sha256,
        "run_root": run_root,
        "recovery_ledger": ledger,
    }


def _validate_runtime_state_shape(state: Mapping[str, Any], label: str) -> None:
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} must be a mapping")
    _require_exact_keys(state, _RUNTIME_STATE_KEYS, label)
    if state["schema_version"] != RUNTIME_PROFILE_STATE_SCHEMA:
        raise ValueError(f"{label} has an invalid schema")
    initial_profile = state["initial_profile"]
    effective_profile = state["effective_profile"]
    if initial_profile not in PROFILE_ORDER or effective_profile not in PROFILE_ORDER:
        raise ValueError(
            f"{label} profile names must be canonical M8/M7/M6/M5/T4/T3/B2/S1 values"
        )
    for key in (
        "initial_max_composite_cost_per_gpu",
        "effective_max_composite_cost_per_gpu",
        "max_pixels",
        "train_batch_size",
        "ppo_mini_batch_size",
        "rollout_n",
    ):
        _require_positive_int(state[key], f"{label}.{key}")
    _require_sha256(state["launch_contract_sha256"], f"{label}.launch_contract_sha256")
    _require_sha256(state["effective_preflight_sha256"], f"{label}.effective_preflight_sha256")
    if (
        not isinstance(state["run_root"], str)
        or not os.path.isabs(state["run_root"])
        or os.path.abspath(state["run_root"]) != state["run_root"]
    ):
        raise ValueError(f"{label}.run_root must be a normalized absolute path")
    ladder = state["runtime_profile_ladder"]
    if not isinstance(ladder, list) or not ladder:
        raise ValueError(f"{label}.runtime_profile_ladder must be a nonempty list")
    expected_ladder_names = list(PROFILE_ORDER[PROFILE_ORDER.index(initial_profile) :])
    ladder_names = [profile.get("name") if isinstance(profile, Mapping) else None for profile in ladder]
    if ladder_names != expected_ladder_names or effective_profile not in ladder_names:
        raise ValueError(f"{label}.runtime_profile_ladder is not the canonical suffix from initial_profile")
    required_profile_keys = {"name", "max_trajectories", "decode_batch_size", "preserve_cuda_cache"}
    for profile in ladder:
        if set(profile) != required_profile_keys:
            raise ValueError(f"{label} runtime profile has a non-canonical field inventory")
        _require_positive_int(
            profile["max_trajectories"],
            f"{label}.{profile['name']}.max_trajectories",
        )
        _require_positive_int(
            profile["decode_batch_size"],
            f"{label}.{profile['name']}.decode_batch_size",
        )
        if not isinstance(profile["preserve_cuda_cache"], bool):
            raise ValueError(f"{label}.{profile['name']}.preserve_cuda_cache must be boolean")
    ledger = state["recovery_ledger"]
    if not isinstance(ledger, Mapping):
        raise ValueError(f"{label}.recovery_ledger must be a mapping")
    _require_exact_keys(ledger, _RECOVERY_LEDGER_STATE_KEYS, f"{label}.recovery_ledger")
    entries = ledger["entries"]
    hashes = ledger["entry_sha256s"]
    if not isinstance(entries, list) or not isinstance(hashes, list):
        raise ValueError(f"{label} recovery entries and hashes must be lists")
    if (
        isinstance(ledger["entry_count"], bool)
        or not isinstance(ledger["entry_count"], int)
        or ledger["entry_count"] < 0
        or ledger["entry_count"] != len(entries)
        or len(entries) != len(hashes)
    ):
        raise ValueError(f"{label} recovery ledger counts are inconsistent")
    previous_hash = ZERO_SHA256
    previous_profile = state["initial_profile"]
    previous_cost = state["initial_max_composite_cost_per_gpu"]
    for index, (entry, expected_hash) in enumerate(zip(entries, hashes, strict=True), start=1):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} recovery entry {index} must be a mapping")
        _require_exact_keys(entry, _LEDGER_ENTRY_KEYS, f"{label} recovery entry {index}")
        if entry["schema_version"] != RECOVERY_LEDGER_ENTRY_SCHEMA:
            raise ValueError(f"{label} recovery entry {index} has an invalid schema")
        if isinstance(entry["sequence"], bool) or entry["sequence"] != index:
            raise ValueError(f"{label} recovery entry {index} has an invalid sequence")
        _require_sha256(
            entry["preflight_sha256"],
            f"{label} recovery entry {index} preflight hash",
        )
        _require_sha256(
            entry["previous_entry_sha256"],
            f"{label} recovery entry {index} previous hash",
            allow_zero=index == 1,
        )
        _require_sha256(expected_hash, f"{label} recovery entry {index} canonical hash")
        actual_hash = canonical_recovery_entry_sha256(entry)
        if expected_hash != actual_hash or entry["previous_entry_sha256"] != previous_hash:
            raise ValueError(f"{label} recovery entry {index} has an invalid hash chain")
        if entry["from_profile"] != previous_profile:
            raise ValueError(f"{label} recovery entry {index} is not contiguous")
        if entry["to_profile"] not in PROFILE_ORDER or PROFILE_ORDER.index(
            entry["to_profile"]
        ) != PROFILE_ORDER.index(previous_profile) + 1:
            raise ValueError(f"{label} recovery entry {index} is not an adjacent downgrade")
        from_cost = _require_positive_int(
            entry["from_max_composite_cost_per_gpu"],
            f"{label} recovery entry {index} from cost",
        )
        to_cost = _require_positive_int(
            entry["to_max_composite_cost_per_gpu"],
            f"{label} recovery entry {index} to cost",
        )
        if from_cost != previous_cost:
            raise ValueError(f"{label} recovery entry {index} does not continue the saved cost")
        if to_cost > previous_cost:
            raise ValueError(f"{label} recovery entry {index} increases the saved cost")
        for key in (
            "max_pixels",
            "train_batch_size",
            "ppo_mini_batch_size",
            "rollout_n",
            "launch_contract_sha256",
        ):
            if entry[key] != state[key]:
                raise ValueError(f"{label} recovery entry {index} changed immutable {key}")
        previous_hash = actual_hash
        previous_profile = entry["to_profile"]
        previous_cost = to_cost
    if previous_profile != state["effective_profile"] or previous_cost != state[
        "effective_max_composite_cost_per_gpu"
    ]:
        raise ValueError(f"{label} recovery tail disagrees with the effective profile/cost")
    expected_tail = hashes[-1] if hashes else ZERO_SHA256
    _require_sha256(
        ledger["tail_entry_sha256"],
        f"{label} recovery ledger tail hash",
        allow_zero=not hashes,
    )
    if ledger["tail_entry_sha256"] != expected_tail:
        raise ValueError(f"{label} recovery tail hash is inconsistent")
    if entries and entries[-1]["preflight_sha256"] != state["effective_preflight_sha256"]:
        raise ValueError(f"{label} recovery tail does not bind the effective preflight")
    if not entries:
        if ledger["path"] is not None or ledger["file_sha256"] is not None:
            raise ValueError(f"{label} empty recovery state cannot bind a ledger file")
        if state["effective_profile"] != state["initial_profile"] or state[
            "effective_max_composite_cost_per_gpu"
        ] != state["initial_max_composite_cost_per_gpu"]:
            raise ValueError(f"{label} has an unledgered profile/cost change")
    else:
        expected_path = os.path.join(state["run_root"], "recovery", "profile_recovery.jsonl")
        if ledger["path"] != expected_path:
            raise ValueError(f"{label} recovery ledger path is not bound to run_root")
        _require_sha256(ledger["file_sha256"], f"{label} recovery ledger file hash")


def validate_v6_runtime_profile_resume(
    saved_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only an exact state or an append-only adjacent downgrade extension."""

    _validate_runtime_state_shape(saved_state, "saved V6 runtime profile state")
    _validate_runtime_state_shape(current_state, "current V6 runtime profile state")
    immutable_keys = (
        "schema_version",
        "initial_profile",
        "initial_max_composite_cost_per_gpu",
        "runtime_profile_ladder",
        "max_pixels",
        "train_batch_size",
        "ppo_mini_batch_size",
        "rollout_n",
        "launch_contract_sha256",
        "run_root",
    )
    changed = [key for key in immutable_keys if saved_state[key] != current_state[key]]
    if changed:
        raise ValueError(f"V6 runtime-profile resume changed immutable fields: {changed}")

    saved_entries = saved_state["recovery_ledger"]["entries"]
    current_entries = current_state["recovery_ledger"]["entries"]
    if len(current_entries) < len(saved_entries):
        raise ValueError("V6 runtime-profile resume truncated the append-only recovery ledger")
    if current_entries[: len(saved_entries)] != saved_entries:
        raise ValueError("V6 runtime-profile resume rewrote the recovery ledger prefix")
    extension = current_entries[len(saved_entries) :]
    if not extension:
        exact_runtime_keys = (
            "effective_profile",
            "effective_max_composite_cost_per_gpu",
            "effective_preflight_sha256",
            "recovery_ledger",
        )
        if any(saved_state[key] != current_state[key] for key in exact_runtime_keys):
            raise ValueError("V6 runtime-profile resume changed effective state without a ledger extension")
    else:
        first = extension[0]
        if (
            first["from_profile"] != saved_state["effective_profile"]
            or first["from_max_composite_cost_per_gpu"]
            != saved_state["effective_max_composite_cost_per_gpu"]
        ):
            raise ValueError("V6 runtime-profile recovery does not continue the checkpoint effective state")
    return {
        "passed": True,
        "saved_effective_profile": saved_state["effective_profile"],
        "effective_runtime_profile": current_state["effective_profile"],
        "authorized_recovery_transitions": len(extension),
        "ledger_prefix_exact": True,
        "immutable_scientific_fields_exact": True,
    }
