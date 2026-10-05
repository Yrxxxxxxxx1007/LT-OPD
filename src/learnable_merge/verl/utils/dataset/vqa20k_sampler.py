"""Deterministic LT-14K and LT-15K sampling with resumable iteration.

Adjacent 40-example Hamilton allocations are interleaved into 80-example
batches using the original eight-group ordering. This defines dataset order;
it does not restrict the number of GPUs that consume each batch.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence, Sized
from copy import deepcopy
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
FULLIMAGE15K_HAMILTON_SAMPLER_STATE_SCHEMA = (
    "vision_opd_v9_vqa15k_fullimage_balanced_sampler_state_v1"
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
_FULLIMAGE15K_HAMILTON_QUOTAS = _HAMILTON_QUOTAS + (2000,)
_FULLIMAGE15K_HAMILTON_HALF_BATCHES = 375
_ORDER_GROUPS = 8
DATA_MIXTURE = {'accepted_train_rows': 14000,
 'sampling_seed': 20260824,
 'global_batch': {'size': 80,
                  'schedule_schema': 'cumulative_hamilton_exact_half_batch_v1',
                  'half_batch_size': 40,
                  'half_batches': 350,
                  'full_outer_batches': 175,
                  'final_outer_batch_size': 0,
                  'hamilton_tie_break_bucket_order': ['onethinker_mcq',
                                                      'onethinker_math',
                                                      'onethinker_numerical',
                                                      'onethinker_ocr',
                                                      'onethinker_regression',
                                                      'pixmo_ask_model_anything',
                                                      'llava_v1_5_mix665k',
                                                      'textvqa',
                                                      'vision_opd_fullimage_mcq'],
                  'exact_epoch_bucket_quotas': {'onethinker_mcq': 3000,
                                                'onethinker_math': 1500,
                                                'onethinker_numerical': 900,
                                                'onethinker_ocr': 900,
                                                'onethinker_regression': 900,
                                                'pixmo_ask_model_anything': 3000,
                                                'llava_v1_5_mix665k': 1800,
                                                'textvqa': 1000,
                                                'vision_opd_fullimage_mcq': 1000},
                  'rank_partition': 'adjacent_half_batches_rank_major_5_plus_5_dp8_no_tail_v2'},
 'vqa20k_sampler_state_schema': 'vision_opd_v8_vqa14k_fullimage_balanced_sampler_state_v1',
 'vqa20k_sampler_algorithm': 'sha256_bucket_permutation_cumulative_hamilton_fullimage_half_batch_rank_sharded_v1',
 'sources': {'onethinker': {'rows': 7200,
                            'strata': {'mcq': 3000,
                                       'math': 1500,
                                       'numerical': 900,
                                       'ocr': 900,
                                       'regression': 900}},
             'pixmo_ask_model_anything': {'rows': 3000},
             'llava_v1_5_mix665k': {'rows': 1800},
             'textvqa': {'rows': 1000},
             'vision_opd_fullimage_mcq': {'rows': 1000}}}

DATA_MIXTURE_15K = deepcopy(DATA_MIXTURE)
DATA_MIXTURE_15K["accepted_train_rows"] = 15000
DATA_MIXTURE_15K["vqa20k_sampler_state_schema"] = FULLIMAGE15K_HAMILTON_SAMPLER_STATE_SCHEMA
DATA_MIXTURE_15K["sources"]["vision_opd_fullimage_mcq"]["rows"] = 2000
DATA_MIXTURE_15K["global_batch"].update(
    half_batches=375,
    full_outer_batches=187,
    outer_steps=188,
    final_outer_batch_size=40,
    has_partial_outer_batch=True,
    rank_partition="adjacent_half_batches_rank_major_5_plus_5_with_final_half_v1",
)
DATA_MIXTURE_15K["global_batch"]["exact_epoch_bucket_quotas"]["vision_opd_fullimage_mcq"] = 2000

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




def _rank_key(*, seed: int, bucket: str, sample_uid: str, index: int) -> tuple[str, str, int]:
    encoded = f"{seed}\0{bucket}\0{sample_uid}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), sample_uid, index




def _sampler_identity(data_contract):
    schema = data_contract.get("vqa20k_sampler_state_schema", FULLIMAGE_HAMILTON_SAMPLER_STATE_SCHEMA)
    algorithm = data_contract.get("vqa20k_sampler_algorithm", FULLIMAGE_HAMILTON_SAMPLER_ALGORITHM)
    _hamilton_layout(state_schema=schema, algorithm=algorithm)
    return schema, algorithm


def _hamilton_layout(*, state_schema, algorithm):
    if algorithm != FULLIMAGE_HAMILTON_SAMPLER_ALGORITHM:
        raise ValueError(f"Unsupported full-image sampler algorithm: {algorithm!r}")
    if state_schema == FULLIMAGE15K_HAMILTON_SAMPLER_STATE_SCHEMA:
        return (_FULLIMAGE_HAMILTON_BUCKET_ORDER, _FULLIMAGE15K_HAMILTON_QUOTAS,
                _FULLIMAGE15K_HAMILTON_HALF_BATCHES, "LT-15K")
    if state_schema != FULLIMAGE_HAMILTON_SAMPLER_STATE_SCHEMA:
        raise ValueError(f"Unsupported full-image sampler state: {state_schema!r}")
    return (_FULLIMAGE_HAMILTON_BUCKET_ORDER, _FULLIMAGE_HAMILTON_QUOTAS,
            _FULLIMAGE_HAMILTON_HALF_BATCHES, "LT-14K")


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
    elif state_schema == FULLIMAGE15K_HAMILTON_SAMPLER_STATE_SCHEMA:
        for key, expected_value in {
            "full_outer_batches": 187,
            "final_outer_batch_size": 40,
            "has_partial_outer_batch": True,
        }.items():
            if global_batch.get(key) != expected_value:
                raise ValueError(f"Hamilton LT-15K {key} must be {expected_value!r}")
    if _HAMILTON_HALF_BATCH_SIZE % _ORDER_GROUPS:
        raise RuntimeError("Hamilton half batch must divide evenly across DP8")
    local_half_batch = _HAMILTON_HALF_BATCH_SIZE // _ORDER_GROUPS
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
            for rank in range(_ORDER_GROUPS)
        )
        if len(rank_chunks) != _ORDER_GROUPS or any(len(chunk) != 5 for chunk in rank_chunks):
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
        for rank in range(_ORDER_GROUPS):
            result.extend(half_rank_chunks[first_half][rank])
            result.extend(half_rank_chunks[second_half][rank])
        outer = result[outer_start:]
        if len(outer) != 80 or any(len(outer[rank * 10 : (rank + 1) * 10]) != 10 for rank in range(8)):
            raise RuntimeError(f"Hamilton outer batch {first_half // 2} is not an exact DP8 x 10 partition")

    if half_batches % 2:
        final_start = len(result)
        for rank in range(_ORDER_GROUPS):
            result.extend(half_rank_chunks[-1][rank])
        final_outer = result[final_start:]
        if len(final_outer) != 40 or any(
            len(final_outer[rank * 5 : (rank + 1) * 5]) != 5
            for rank in range(_ORDER_GROUPS)
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
        state_schema: str = SAMPLER_STATE_SCHEMA,
        algorithm: str = SAMPLER_ALGORITHM,
    ):
        self._order = tuple(int(index) for index in order)
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
            "order_sha256": self._order_sha256,
            "length": len(self._order),
            "yielded": self._yielded,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": self._state_schema,
            "algorithm": self._algorithm,
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
        mixture = _config_get(data_config, "vqa20k_mixture", "lt14k")
        mixtures = {"lt14k": DATA_MIXTURE, "lt15k": DATA_MIXTURE_15K}
        if mixture not in mixtures:
            raise ValueError(f"Unsupported LT-OPD mixture: {mixture!r}")
        data_contract = mixtures[mixture]
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

    def __iter__(self) -> Iterator[int]:
        return _BalancedIterator(
            self._order,
            state_schema=self._state_schema,
            algorithm=self._algorithm,
        )

    def __len__(self) -> int:
        return len(self._order)
