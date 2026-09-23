"""Deterministic, stateful samplers for the audited VQA mixtures.

The materialized parquet is intentionally not trusted to already be in a
balanced order.  This module preserves the legacy exact-global-mixture order
byte-for-byte, while explicitly selected cumulative-Hamilton schedules emit
either the 13K release as 325 exact 40-row half batches or the full-image 14K
release as 350 such half batches.  Adjacent half batches are packed rank-wise
into 80-row outer batches.  The 13K release ends in one 40-row half batch; the
14K release has 175 complete outer batches and no tail.

The iterator implements the ``state_dict`` protocol consumed by
``torchdata.stateful_dataloader.StatefulDataLoader``.  Its state binds both the
release contract and the complete index order, so a checkpoint cannot be
loaded against a reordered or substituted parquet.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence, Sized
from typing import Any

from verl.experimental.dataset.sampler import AbstractSampler


SAMPLER_STATE_SCHEMA = "vision_opd_v6_vqa20k_balanced_sampler_state_v1"
SAMPLER_ALGORITHM = "sha256_bucket_permutation_rank_sharded_exact_global_mix_v1"

HAMILTON_SCHEDULE_SCHEMA = "cumulative_hamilton_exact_half_batch_v1"
HAMILTON_SAMPLER_STATE_SCHEMA = "vision_opd_v8_vqa13k_balanced_sampler_state_v1"
HAMILTON_SAMPLER_ALGORITHM = (
    "sha256_bucket_permutation_cumulative_hamilton_half_batch_rank_sharded_v2"
)
FULLIMAGE_HAMILTON_SCHEDULE_SCHEMA = HAMILTON_SCHEDULE_SCHEMA
FULLIMAGE_HAMILTON_SAMPLER_STATE_SCHEMA = (
    "vision_opd_v8_vqa14k_fullimage_balanced_sampler_state_v1"
)
FULLIMAGE_HAMILTON_SAMPLER_ALGORITHM = (
    "sha256_bucket_permutation_cumulative_hamilton_fullimage_half_batch_rank_sharded_v1"
)

_HAMILTON_BUCKET_ORDER = (
    "onethinker_mcq",
    "onethinker_math",
    "onethinker_numerical",
    "onethinker_ocr",
    "onethinker_regression",
    "pixmo_ask_model_anything",
    "llava_v1_5_mix665k",
    "textvqa",
)
_HAMILTON_QUOTAS = (3000, 1500, 900, 900, 900, 3000, 1800, 1000)
_HAMILTON_HALF_BATCH_SIZE = 40
_HAMILTON_HALF_BATCHES = 325
_FULLIMAGE_HAMILTON_BUCKET_ORDER = _HAMILTON_BUCKET_ORDER + (
    "vision_opd_fullimage_mcq",
)
_FULLIMAGE_HAMILTON_QUOTAS = _HAMILTON_QUOTAS + (1000,)
_FULLIMAGE_HAMILTON_HALF_BATCHES = 350
_DP_WORLD_SIZE = 8

_SOURCE_BUCKET = {
    "pixmo_ask_model_anything": "pixmo_ask_model_anything",
    "llava_v1_5_mix665k": "llava_v1_5_mix665k",
    "textvqa": "textvqa",
}


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_release_data_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    # Import lazily: importing a sampler must not parse release files unless it
    # is actually selected by the data config.
    from training.contract import load_contract

    contract, summary = load_contract()
    return contract["data"], summary


def _column(data_source: Sized, name: str) -> list[Any]:
    dataframe = getattr(data_source, "dataframe", None)
    column_names = getattr(dataframe, "column_names", ())
    if dataframe is not None and name in column_names:
        return list(dataframe[name])

    values: list[Any] = []
    for index in range(len(data_source)):
        row = data_source[index]
        if not isinstance(row, Mapping) or name not in row:
            raise ValueError(f"VQA20K row {index} is missing required sampler column {name!r}")
        values.append(row[name])
    return values


def _expected_bucket_counts(data_contract: Mapping[str, Any]) -> dict[str, int]:
    sources = data_contract["sources"]
    result = {
        f"onethinker_{name}": int(count)
        for name, count in sources["onethinker"]["strata"].items()
    }
    result[_SOURCE_BUCKET["pixmo_ask_model_anything"]] = int(
        sources["pixmo_ask_model_anything"]["rows"]
    )
    result[_SOURCE_BUCKET["llava_v1_5_mix665k"]] = int(sources["llava_v1_5_mix665k"]["rows"])
    if "textvqa" in sources:
        result[_SOURCE_BUCKET["textvqa"]] = int(sources["textvqa"]["rows"])
    if "vision_opd_fullimage_mcq" in sources:
        result["vision_opd_fullimage_mcq"] = int(sources["vision_opd_fullimage_mcq"]["rows"])
    return result


def _per_batch_bucket_counts(data_contract: Mapping[str, Any]) -> dict[str, int]:
    batch = data_contract["global_batch"]
    one = batch["onethinker"]
    result = {
        f"onethinker_{name}": int(one[name])
        for name in data_contract["sources"]["onethinker"]["strata"]
    }
    result[_SOURCE_BUCKET["pixmo_ask_model_anything"]] = int(batch["pixmo_ask_model_anything"])
    result[_SOURCE_BUCKET["llava_v1_5_mix665k"]] = int(batch["llava_v1_5_mix665k"])
    return result


def _rank_key(*, seed: int, bucket: str, sample_uid: str, index: int) -> tuple[str, str, int]:
    encoded = f"{seed}\0{bucket}\0{sample_uid}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), sample_uid, index


def _build_legacy_balanced_index_order(
    *,
    sampling_buckets: Sequence[Any],
    sample_uids: Sequence[Any],
    data_contract: Mapping[str, Any],
) -> list[int]:
    """Return the legacy exact-mixture order without changing its semantics."""

    if len(sampling_buckets) != len(sample_uids):
        raise ValueError("sampling_bucket and sample_uid columns have different lengths")
    expected = _expected_bucket_counts(data_contract)
    if len(sampling_buckets) != int(data_contract["accepted_train_rows"]):
        raise ValueError(
            "VQA20K sampler requires the exact release row count: "
            f"expected={data_contract['accepted_train_rows']}, actual={len(sampling_buckets)}"
        )

    inventory: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_uids: set[str] = set()
    for index, (raw_bucket, raw_uid) in enumerate(zip(sampling_buckets, sample_uids, strict=True)):
        if not isinstance(raw_bucket, str) or raw_bucket not in expected:
            raise ValueError(f"VQA20K row {index} has an unknown sampling bucket: {raw_bucket!r}")
        if not isinstance(raw_uid, str) or not raw_uid.strip():
            raise ValueError(f"VQA20K row {index} has an invalid sample_uid")
        uid = raw_uid.strip()
        if uid in seen_uids:
            raise ValueError(f"VQA20K sample_uid is not unique: {uid!r}")
        seen_uids.add(uid)
        inventory[raw_bucket].append((index, uid))

    actual = {bucket: len(inventory.get(bucket, ())) for bucket in expected}
    if actual != expected:
        raise ValueError(f"VQA20K bucket inventory drift: expected={expected!r}, actual={actual!r}")
    unexpected = set(inventory) - set(expected)
    if unexpected:
        raise ValueError(f"VQA20K has unexpected sampling buckets: {sorted(unexpected)!r}")

    seed = int(data_contract["sampling_seed"])
    queues: dict[str, list[int]] = {}
    for bucket in expected:
        ranked = sorted(
            inventory[bucket],
            key=lambda item: _rank_key(seed=seed, bucket=bucket, sample_uid=item[1], index=item[0]),
        )
        queues[bucket] = [index for index, _ in ranked]

    per_batch = _per_batch_bucket_counts(data_contract)
    global_batch_size = int(data_contract["global_batch"]["size"])
    if sum(per_batch.values()) != global_batch_size:
        raise ValueError("VQA20K per-batch bucket counts do not sum to the global batch size")
    steps, remainder = divmod(len(sampling_buckets), global_batch_size)

    cursors = {bucket: 0 for bucket in expected}
    result: list[int] = []
    world_size = 8
    local_batch = global_batch_size // world_size
    if global_batch_size % world_size or local_batch != 10:
        raise ValueError("release-06 requires global batch 80 and exactly 10 prompts per DP rank")

    one_buckets = [bucket for bucket in expected if bucket.startswith("onethinker_")]
    pixmo_bucket = _SOURCE_BUCKET["pixmo_ask_model_anything"]
    llava_bucket = _SOURCE_BUCKET["llava_v1_5_mix665k"]

    for step in range(steps):
        batch_items: dict[str, list[int]] = {}
        for bucket, count in per_batch.items():
            start = cursors[bucket]
            stop = start + count
            batch_items[bucket] = queues[bucket][start:stop]
            if len(batch_items[bucket]) != count:
                raise RuntimeError(f"VQA20K bucket {bucket!r} exhausted at outer step {step}")
            cursors[bucket] = stop

        # Spread the 48 OneThinker prompts across ranks first.  Rotating the
        # stratum order prevents a fixed rank from always receiving the same
        # task family, while preserving the exact global composition.
        one_stream: list[int] = []
        rotated = one_buckets[step % len(one_buckets) :] + one_buckets[: step % len(one_buckets)]
        remaining = {bucket: list(batch_items[bucket]) for bucket in one_buckets}
        while any(remaining.values()):
            for bucket in rotated:
                if remaining[bucket]:
                    one_stream.append(remaining[bucket].pop(0))
        if len(one_stream) != 48:
            raise RuntimeError("release-06 OneThinker batch must contain exactly 48 prompts")

        # Four ranks receive 3 PixMo + 1 LLaVA and four receive 2 PixMo + 2
        # LLaVA.  The high-PixMo ranks rotate by step.  Every rank receives 6
        # OneThinker prompts and exactly 10 prompts in total.
        pixmo = list(batch_items[pixmo_bucket])
        llava = list(batch_items[llava_bucket])
        high_pixmo = {(step + offset) % world_size for offset in range(4)}
        for rank in range(world_size):
            chunk = one_stream[rank * 6 : (rank + 1) * 6]
            pixmo_count = 3 if rank in high_pixmo else 2
            llava_count = 1 if rank in high_pixmo else 2
            chunk.extend(pixmo[:pixmo_count])
            del pixmo[:pixmo_count]
            chunk.extend(llava[:llava_count])
            del llava[:llava_count]
            if len(chunk) != local_batch:
                raise RuntimeError(f"VQA20K rank {rank} did not receive exactly 10 prompts")
            result.extend(chunk)
        if pixmo or llava:
            raise RuntimeError("VQA20K source prompts were not consumed exactly within a global batch")

    if remainder:
        # The 15K revision is exactly 187.5 copies of the immutable 80-row
        # mixture.  Retain all rows without repetition by emitting one final
        # 40-row half batch: 24 OneThinker (10/5/3/3/3), 10 PixMo and 6
        # LLaVA.  Forty rows divide evenly across DP8, so every rank executes
        # five prompts / forty rollout trajectories and reaches collectives in
        # the same order.
        tail_counts = {bucket: expected[bucket] - cursors[bucket] for bucket in expected}
        expected_tail = {bucket: count // 2 for bucket, count in per_batch.items()}
        if remainder != global_batch_size // 2 or tail_counts != expected_tail:
            raise ValueError(
                "VQA20K partial batch must be the exact 40-row half-mixture: "
                f"expected={expected_tail!r}, actual={tail_counts!r}"
            )
        batch_items = {}
        for bucket, count in tail_counts.items():
            start = cursors[bucket]
            stop = start + count
            batch_items[bucket] = queues[bucket][start:stop]
            if len(batch_items[bucket]) != count:
                raise RuntimeError(f"VQA20K bucket {bucket!r} exhausted in the final half batch")
            cursors[bucket] = stop

        one_stream = []
        step = steps
        rotated = one_buckets[step % len(one_buckets) :] + one_buckets[: step % len(one_buckets)]
        remaining = {bucket: list(batch_items[bucket]) for bucket in one_buckets}
        while any(remaining.values()):
            for bucket in rotated:
                if remaining[bucket]:
                    one_stream.append(remaining[bucket].pop(0))
        if len(one_stream) != 24:
            raise RuntimeError("release-06 final half batch must contain exactly 24 OneThinker prompts")

        pixmo = list(batch_items[pixmo_bucket])
        llava = list(batch_items[llava_bucket])
        high_pixmo = {(step + offset) % world_size for offset in range(2)}
        for rank in range(world_size):
            chunk = one_stream[rank * 3 : (rank + 1) * 3]
            pixmo_count = 2 if rank in high_pixmo else 1
            llava_count = 0 if rank in high_pixmo else 1
            chunk.extend(pixmo[:pixmo_count])
            del pixmo[:pixmo_count]
            chunk.extend(llava[:llava_count])
            del llava[:llava_count]
            if len(chunk) != 5:
                raise RuntimeError(f"VQA20K final half batch rank {rank} did not receive exactly 5 prompts")
            result.extend(chunk)
        if pixmo or llava:
            raise RuntimeError("VQA20K source prompts were not consumed exactly in the final half batch")

    if cursors != expected:
        raise RuntimeError(f"VQA20K sampler did not consume every bucket exactly: {cursors!r}")
    if len(result) != len(sampling_buckets) or len(set(result)) != len(result):
        raise RuntimeError("VQA20K sampler order is not a one-to-one permutation")
    return result


def _sampler_identity(data_contract: Mapping[str, Any]) -> tuple[str, str]:
    global_batch = data_contract.get("global_batch")
    if not isinstance(global_batch, Mapping):
        raise ValueError("VQA20K data contract is missing the global_batch mapping")
    schedule_schema = global_batch.get("schedule_schema")
    if schedule_schema is None:
        if "textvqa" in data_contract.get("sources", {}):
            raise ValueError(
                "a VQA contract containing textvqa must explicitly select "
                f"global_batch.schedule_schema={HAMILTON_SCHEDULE_SCHEMA!r}"
            )
        legacy_state_schema = data_contract.get("vqa20k_sampler_state_schema", SAMPLER_STATE_SCHEMA)
        legacy_algorithm = data_contract.get("vqa20k_sampler_algorithm", SAMPLER_ALGORITHM)
        if legacy_state_schema != SAMPLER_STATE_SCHEMA or legacy_algorithm != SAMPLER_ALGORITHM:
            raise ValueError(
                "legacy VQA sampler resume identity drift without an explicit schedule_schema: "
                f"state_schema={legacy_state_schema!r}, algorithm={legacy_algorithm!r}"
            )
        return SAMPLER_STATE_SCHEMA, SAMPLER_ALGORITHM
    if schedule_schema != HAMILTON_SCHEDULE_SCHEMA:
        raise ValueError(f"unsupported VQA20K global_batch.schedule_schema: {schedule_schema!r}")

    state_schema = data_contract.get("vqa20k_sampler_state_schema")
    algorithm = data_contract.get("vqa20k_sampler_algorithm")
    allowed_algorithms = {
        HAMILTON_SAMPLER_STATE_SCHEMA: HAMILTON_SAMPLER_ALGORITHM,
        FULLIMAGE_HAMILTON_SAMPLER_STATE_SCHEMA: FULLIMAGE_HAMILTON_SAMPLER_ALGORITHM,
    }
    expected_algorithm = allowed_algorithms.get(state_schema)
    if expected_algorithm is None:
        raise ValueError(
            "Hamilton sampler state schema drift: "
            f"expected_one_of={tuple(allowed_algorithms)!r}, actual={state_schema!r}"
        )
    if algorithm != expected_algorithm:
        raise ValueError(
            "Hamilton sampler algorithm drift: "
            f"expected={expected_algorithm!r}, actual={algorithm!r}"
        )
    return state_schema, algorithm


def _hamilton_layout(
    *,
    state_schema: str,
    algorithm: str,
) -> tuple[tuple[str, ...], tuple[int, ...], int, str]:
    """Resolve an explicitly versioned Hamilton layout without row-count inference."""

    if (
        state_schema == HAMILTON_SAMPLER_STATE_SCHEMA
        and algorithm == HAMILTON_SAMPLER_ALGORITHM
    ):
        return (
            _HAMILTON_BUCKET_ORDER,
            _HAMILTON_QUOTAS,
            _HAMILTON_HALF_BATCHES,
            "VQA13K",
        )
    if (
        state_schema == FULLIMAGE_HAMILTON_SAMPLER_STATE_SCHEMA
        and algorithm == FULLIMAGE_HAMILTON_SAMPLER_ALGORITHM
    ):
        return (
            _FULLIMAGE_HAMILTON_BUCKET_ORDER,
            _FULLIMAGE_HAMILTON_QUOTAS,
            _FULLIMAGE_HAMILTON_HALF_BATCHES,
            "VQA14K full-image",
        )
    raise ValueError(
        "unsupported Hamilton sampler identity: "
        f"state_schema={state_schema!r}, algorithm={algorithm!r}"
    )


def _hamilton_cumulative_counts(
    t: int,
    *,
    quotas: Sequence[int] = _HAMILTON_QUOTAS,
    half_batches: int = _HAMILTON_HALF_BATCHES,
    half_batch_size: int = _HAMILTON_HALF_BATCH_SIZE,
) -> tuple[int, ...]:
    """Allocate the first ``t`` half batches using exact integer Hamilton."""

    if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t <= half_batches:
        raise ValueError(f"invalid cumulative Hamilton half-batch index: {t!r}")
    if sum(quotas) != half_batches * half_batch_size:
        raise ValueError("Hamilton quotas do not equal half_batches * half_batch_size")

    floors = [t * int(quota) // half_batches for quota in quotas]
    target = t * half_batch_size
    seats = target - sum(floors)
    if not 0 <= seats <= len(quotas):
        raise RuntimeError(f"invalid Hamilton remainder seat count at half batch {t}: {seats}")
    # Integer remainders avoid float/platform drift.  Python's stable sort plus
    # the explicit bucket index gives the immutable contract tie-break order.
    remainder_order = sorted(
        range(len(quotas)),
        key=lambda index: (-(t * int(quotas[index]) % half_batches), index),
    )
    for index in remainder_order[:seats]:
        floors[index] += 1
    if sum(floors) != target:
        raise RuntimeError(f"Hamilton cumulative allocation does not sum to {target} at t={t}")
    return tuple(floors)


def _hamilton_half_batch_plan(
    *,
    quotas: Sequence[int] = _HAMILTON_QUOTAS,
    half_batches: int = _HAMILTON_HALF_BATCHES,
    half_batch_size: int = _HAMILTON_HALF_BATCH_SIZE,
) -> tuple[tuple[int, ...], ...]:
    previous = _hamilton_cumulative_counts(
        0,
        quotas=quotas,
        half_batches=half_batches,
        half_batch_size=half_batch_size,
    )
    plan: list[tuple[int, ...]] = []
    for t in range(1, half_batches + 1):
        cumulative = _hamilton_cumulative_counts(
            t,
            quotas=quotas,
            half_batches=half_batches,
            half_batch_size=half_batch_size,
        )
        delta = tuple(current - before for current, before in zip(cumulative, previous, strict=True))
        if any(count < 0 for count in delta) or sum(delta) != half_batch_size:
            raise RuntimeError(f"invalid cumulative-Hamilton delta at half batch {t}: {delta!r}")
        plan.append(delta)
        previous = cumulative
    final = tuple(sum(step[index] for step in plan) for index in range(len(quotas)))
    expected_final = tuple(int(quota) for quota in quotas)
    if final != expected_final:
        raise RuntimeError(f"Hamilton plan does not end at the exact quotas: {final!r}")
    return tuple(plan)


def _fullimage_hamilton_half_batch_plan() -> tuple[tuple[int, ...], ...]:
    """Return the immutable 350-half full-image VQA14K plan."""

    return _hamilton_half_batch_plan(
        quotas=_FULLIMAGE_HAMILTON_QUOTAS,
        half_batches=_FULLIMAGE_HAMILTON_HALF_BATCHES,
        half_batch_size=_HAMILTON_HALF_BATCH_SIZE,
    )


def _build_hamilton_balanced_index_order(
    *,
    sampling_buckets: Sequence[Any],
    sample_uids: Sequence[Any],
    data_contract: Mapping[str, Any],
) -> list[int]:
    """Build an exact DP8-packed cumulative-Hamilton one-epoch order."""

    state_schema, algorithm = _sampler_identity(data_contract)
    bucket_order, quotas, half_batches, release_name = _hamilton_layout(
        state_schema=state_schema,
        algorithm=algorithm,
    )
    if len(sampling_buckets) != len(sample_uids):
        raise ValueError("sampling_bucket and sample_uid columns have different lengths")

    expected = _expected_bucket_counts(data_contract)
    required_expected = dict(zip(bucket_order, quotas, strict=True))
    if expected != required_expected:
        raise ValueError(
            f"Hamilton {release_name} bucket quotas drift: "
            f"expected={required_expected!r}, actual={expected!r}"
        )
    accepted_rows = data_contract.get("accepted_train_rows")
    expected_rows = sum(quotas)
    if accepted_rows != expected_rows:
        raise ValueError(
            f"Hamilton {release_name} accepted_train_rows drift: "
            f"expected={expected_rows}, actual={accepted_rows!r}"
        )
    if len(sampling_buckets) != accepted_rows:
        raise ValueError(
            "VQA20K sampler requires the exact release row count: "
            f"expected={accepted_rows}, actual={len(sampling_buckets)}"
        )

    global_batch = data_contract["global_batch"]
    if global_batch.get("size") != 80:
        raise ValueError(
            f"Hamilton {release_name} requires global_batch.size=80, "
            f"got {global_batch.get('size')!r}"
        )
    if "half_batch_size" in global_batch and global_batch["half_batch_size"] != _HAMILTON_HALF_BATCH_SIZE:
        raise ValueError(
            f"Hamilton {release_name} half_batch_size drift: "
            f"expected={_HAMILTON_HALF_BATCH_SIZE}, actual={global_batch['half_batch_size']!r}"
        )
    if "half_batches" in global_batch and global_batch["half_batches"] != half_batches:
        raise ValueError(
            f"Hamilton {release_name} half_batches drift: "
            f"expected={half_batches}, actual={global_batch['half_batches']!r}"
        )
    expected_outer_steps = (half_batches + 1) // 2
    if "outer_steps" in global_batch and global_batch["outer_steps"] != expected_outer_steps:
        raise ValueError(
            f"Hamilton {release_name} outer_steps drift: "
            f"expected={expected_outer_steps}, actual={global_batch['outer_steps']!r}"
        )
    if state_schema == FULLIMAGE_HAMILTON_SAMPLER_STATE_SCHEMA:
        if half_batches % 2 or expected_rows % 80:
            raise RuntimeError("Hamilton VQA14K full-image layout must not have a partial outer batch")
        if "has_partial_outer_batch" in global_batch and global_batch["has_partial_outer_batch"] is not False:
            raise ValueError(
                "Hamilton VQA14K full-image has_partial_outer_batch must be false"
            )
    if _HAMILTON_HALF_BATCH_SIZE % _DP_WORLD_SIZE:
        raise RuntimeError("Hamilton half batch must divide evenly across DP8")
    local_half_batch = _HAMILTON_HALF_BATCH_SIZE // _DP_WORLD_SIZE
    if local_half_batch != 5:
        raise RuntimeError("Hamilton release requires exactly five prompts per DP rank and half batch")

    inventory: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_uids: set[str] = set()
    for index, (raw_bucket, raw_uid) in enumerate(zip(sampling_buckets, sample_uids, strict=True)):
        if not isinstance(raw_bucket, str) or raw_bucket not in required_expected:
            raise ValueError(f"VQA20K row {index} has an unknown sampling bucket: {raw_bucket!r}")
        if not isinstance(raw_uid, str) or not raw_uid.strip():
            raise ValueError(f"VQA20K row {index} has an invalid sample_uid")
        uid = raw_uid.strip()
        if uid in seen_uids:
            raise ValueError(f"VQA20K sample_uid is not unique: {uid!r}")
        seen_uids.add(uid)
        inventory[raw_bucket].append((index, uid))

    actual = {bucket: len(inventory.get(bucket, ())) for bucket in bucket_order}
    if actual != required_expected:
        raise ValueError(f"VQA20K bucket inventory drift: expected={required_expected!r}, actual={actual!r}")
    unexpected = set(inventory) - set(bucket_order)
    if unexpected:
        raise ValueError(f"VQA20K has unexpected sampling buckets: {sorted(unexpected)!r}")

    raw_seed = data_contract.get("sampling_seed")
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ValueError(f"Hamilton {release_name} sampling_seed must be an integer, got {raw_seed!r}")
    queues: dict[str, list[int]] = {}
    for bucket in bucket_order:
        ranked = sorted(
            inventory[bucket],
            key=lambda item: _rank_key(seed=raw_seed, bucket=bucket, sample_uid=item[1], index=item[0]),
        )
        queues[bucket] = [index for index, _ in ranked]

    cursors = {bucket: 0 for bucket in bucket_order}
    half_rank_chunks: list[tuple[tuple[int, ...], ...]] = []
    plan = _hamilton_half_batch_plan(
        quotas=quotas,
        half_batches=half_batches,
        half_batch_size=_HAMILTON_HALF_BATCH_SIZE,
    )
    for half_index, counts in enumerate(plan):
        batch_items: dict[str, list[int]] = {}
        for bucket, count in zip(bucket_order, counts, strict=True):
            start = cursors[bucket]
            stop = start + count
            items = queues[bucket][start:stop]
            if len(items) != count:
                raise RuntimeError(f"VQA20K bucket {bucket!r} exhausted at half batch {half_index}")
            batch_items[bucket] = items
            cursors[bucket] = stop

        # Rotate the fixed tie-break bucket order only for within-half
        # interleaving.  This does not alter Hamilton composition or the
        # deterministic per-bucket SHA-256 permutation.
        rotation = half_index % len(bucket_order)
        rotated = bucket_order[rotation:] + bucket_order[:rotation]
        positions = {bucket: 0 for bucket in bucket_order}
        stream: list[int] = []
        while len(stream) < _HAMILTON_HALF_BATCH_SIZE:
            before = len(stream)
            for bucket in rotated:
                position = positions[bucket]
                if position < len(batch_items[bucket]):
                    stream.append(batch_items[bucket][position])
                    positions[bucket] = position + 1
            if len(stream) == before:
                raise RuntimeError(f"Hamilton half batch {half_index} interleaver made no progress")
        if len(stream) != _HAMILTON_HALF_BATCH_SIZE:
            raise RuntimeError(f"Hamilton half batch {half_index} does not contain exactly 40 prompts")
        if positions != {bucket: len(batch_items[bucket]) for bucket in bucket_order}:
            raise RuntimeError(f"Hamilton half batch {half_index} did not consume its composition exactly")
        observed = {bucket: 0 for bucket in bucket_order}
        for index in stream:
            observed[str(sampling_buckets[index])] += 1
        required = dict(zip(bucket_order, counts, strict=True))
        if observed != required:
            raise RuntimeError(
                f"Hamilton half batch {half_index} composition drift: expected={required!r}, actual={observed!r}"
            )

        rank_chunks = tuple(
            tuple(stream[rank * local_half_batch : (rank + 1) * local_half_batch])
            for rank in range(_DP_WORLD_SIZE)
        )
        if len(rank_chunks) != _DP_WORLD_SIZE or any(len(chunk) != 5 for chunk in rank_chunks):
            raise RuntimeError(f"Hamilton half batch {half_index} is not an exact DP8 x 5 partition")
        if len({index for chunk in rank_chunks for index in chunk}) != _HAMILTON_HALF_BATCH_SIZE:
            raise RuntimeError(f"Hamilton half batch {half_index} DP chunks are not disjoint")
        half_rank_chunks.append(rank_chunks)

    # Pack adjacent halves by rank.  A naive ``half_a + half_b`` concatenation
    # would give each 10-row DP slice two different ranks from half A; this
    # rank-major packing is therefore part of the algorithm identity.
    result: list[int] = []
    paired_half_batches = half_batches - (half_batches % 2)
    for first_half in range(0, paired_half_batches, 2):
        second_half = first_half + 1
        outer_start = len(result)
        for rank in range(_DP_WORLD_SIZE):
            result.extend(half_rank_chunks[first_half][rank])
            result.extend(half_rank_chunks[second_half][rank])
        outer = result[outer_start:]
        if len(outer) != 80 or any(len(outer[rank * 10 : (rank + 1) * 10]) != 10 for rank in range(8)):
            raise RuntimeError(f"Hamilton outer batch {first_half // 2} is not an exact DP8 x 10 partition")

    if half_batches % 2:
        final_start = len(result)
        for rank in range(_DP_WORLD_SIZE):
            result.extend(half_rank_chunks[-1][rank])
        final_outer = result[final_start:]
        if len(final_outer) != 40 or any(
            len(final_outer[rank * 5 : (rank + 1) * 5]) != 5
            for rank in range(_DP_WORLD_SIZE)
        ):
            raise RuntimeError("Hamilton final half batch is not an exact DP8 x 5 partition")

    if cursors != required_expected:
        raise RuntimeError(f"VQA20K sampler did not consume every bucket exactly: {cursors!r}")
    if len(result) != len(sampling_buckets) or len(set(result)) != len(result):
        raise RuntimeError("VQA20K Hamilton sampler order is not a one-to-one permutation")
    return result


def build_balanced_index_order(
    *,
    sampling_buckets: Sequence[Any],
    sample_uids: Sequence[Any],
    data_contract: Mapping[str, Any],
) -> list[int]:
    """Return the selected exact one-epoch order, rejecting contract drift."""

    _state_schema, algorithm = _sampler_identity(data_contract)
    if algorithm == SAMPLER_ALGORITHM:
        return _build_legacy_balanced_index_order(
            sampling_buckets=sampling_buckets,
            sample_uids=sample_uids,
            data_contract=data_contract,
        )
    return _build_hamilton_balanced_index_order(
        sampling_buckets=sampling_buckets,
        sample_uids=sample_uids,
        data_contract=data_contract,
    )


class _BalancedIterator(Iterator[int]):
    def __init__(
        self,
        order: Sequence[int],
        *,
        contract_sha256: str,
        state_schema: str = SAMPLER_STATE_SCHEMA,
        algorithm: str = SAMPLER_ALGORITHM,
    ):
        self._order = tuple(int(index) for index in order)
        self._contract_sha256 = contract_sha256
        self._order_sha256 = _canonical_sha256(self._order)
        self._state_schema = state_schema
        self._algorithm = algorithm
        self._yielded = 0

    def __iter__(self) -> "_BalancedIterator":
        return self

    def __next__(self) -> int:
        if self._yielded >= len(self._order):
            raise StopIteration
        value = self._order[self._yielded]
        self._yielded += 1
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self._state_schema,
            "algorithm": self._algorithm,
            "contract_sha256": self._contract_sha256,
            "order_sha256": self._order_sha256,
            "length": len(self._order),
            "yielded": self._yielded,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": self._state_schema,
            "algorithm": self._algorithm,
            "contract_sha256": self._contract_sha256,
            "order_sha256": self._order_sha256,
            "length": len(self._order),
        }
        for key, value in expected.items():
            if state_dict.get(key) != value:
                raise ValueError(
                    f"VQA20K sampler resume identity drift for {key}: "
                    f"expected={value!r}, actual={state_dict.get(key)!r}"
                )
        yielded = state_dict.get("yielded")
        if isinstance(yielded, bool) or not isinstance(yielded, int) or not 0 <= yielded <= len(self._order):
            raise ValueError(f"invalid VQA20K sampler yielded count: {yielded!r}")
        self._yielded = yielded


class VQA20KBalancedSampler(AbstractSampler):
    """Exact, resumable sampler selected through ``data.sampler`` config."""

    def __init__(self, data_source: Sized, data_config: Any):
        data_contract, contract_summary = _load_release_data_contract()
        self._state_schema, self._algorithm = _sampler_identity(data_contract)
        expected_class = _config_get(data_config, "vqa20k_sampler_algorithm", self._algorithm)
        if expected_class != self._algorithm:
            raise ValueError(
                f"data.vqa20k_sampler_algorithm must be {self._algorithm!r}, got {expected_class!r}"
            )
        buckets = _column(data_source, "sampling_bucket")
        uids = _column(data_source, "sample_uid")
        self._order = build_balanced_index_order(
            sampling_buckets=buckets,
            sample_uids=uids,
            data_contract=data_contract,
        )
        self._contract_sha256 = str(contract_summary["canonical_sha256"])

    def __iter__(self) -> Iterator[int]:
        return _BalancedIterator(
            self._order,
            contract_sha256=self._contract_sha256,
            state_schema=self._state_schema,
            algorithm=self._algorithm,
        )

    def __len__(self) -> int:
        return len(self._order)
