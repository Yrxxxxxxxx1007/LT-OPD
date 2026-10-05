"""Resource selection for the local training launcher."""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

GIB = 1024**3
MIN_OBJECT_STORE_BYTES = 75 * 1024**2


def available_cpus():
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    root = Path('/sys/fs/cgroup')
    if (root/'cpu.max').is_file():
        quota, period = (root/'cpu.max').read_text().split()
        if quota != 'max':
            count = min(count, max(1, int(quota) // int(period)))
    elif (root/'cpu/cpu.cfs_quota_us').is_file():
        quota = int((root/'cpu/cpu.cfs_quota_us').read_text())
        period = int((root/'cpu/cpu.cfs_period_us').read_text())
        if quota > 0:
            count = min(count, max(1, quota // period))
    return count


def available_memory():
    import psutil

    available = int(psutil.virtual_memory().available)
    root = Path('/sys/fs/cgroup')
    if (root/'memory.max').is_file():
        raw = (root/'memory.max').read_text().strip()
        if raw == 'max':
            return available
        limit = int(raw)
        usage = int((root/'memory.current').read_text())
        stats = dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
        reclaimable = int(stats.get('inactive_file', 0))
    elif (root/'memory/memory.limit_in_bytes').is_file():
        root = root/'memory'
        limit = int((root/'memory.limit_in_bytes').read_text())
        usage = int((root/'memory.usage_in_bytes').read_text())
        stats = dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
        reclaimable = int(stats.get('total_inactive_file', stats.get('inactive_file', 0)))
    else:
        return available
    return min(available, max(0, limit - usage) + min(usage, reclaimable))


def bytes_from_gib(value, name, *, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name} must be a finite {"nonnegative" if allow_zero else "positive"} number')
    return int(value * GIB)


def store_capacity(directory):
    directory = Path(directory)
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    return shutil.disk_usage(directory).free


def select_resources(config, plasma):
    settings = config.runtime_resources
    ray_init = config.ray_kwargs.ray_init
    gpus = int(config.trainer.n_gpus_per_node)
    cpu_limit = available_cpus()
    # The worker pool reserves three CPUs per GPU, plus one for its driver.
    minimum_cpus = 3 * gpus + 1
    cpus = ray_init.get('num_cpus')
    cpus = min(cpu_limit, max(32, minimum_cpus)) if cpus is None else int(cpus)
    if not minimum_cpus <= cpus <= cpu_limit:
        raise ValueError(
            f'Ray needs at least {minimum_cpus} CPUs for {gpus} GPUs; '
            f'--cpus={cpus}, available={cpu_limit}. Increase the CPU allocation or select fewer GPUs.'
        )
    headroom = bytes_from_gib(settings.memory_headroom_gib, 'memory_headroom_gib', allow_zero=True)
    available = available_memory()
    capacity = store_capacity(plasma)
    object_bytes = ray_init.get('object_store_memory')
    if object_bytes is None:
        object_bytes = min(64 * GIB, capacity * 2 // 3, max(0, available - headroom))
    object_bytes = int(object_bytes)
    validate_store_capacity(plasma, object_bytes, headroom)
    omp_threads = settings.get('omp_threads')
    if omp_threads is None:
        omp_threads = os.environ.get('OMP_NUM_THREADS', min(4, max(1, (cpus - 1) // gpus)))
    omp_threads = int(omp_threads)
    if omp_threads < 1 or omp_threads > cpu_limit:
        raise ValueError(f'OMP thread count must be between 1 and {cpu_limit}')
    return cpus, object_bytes, omp_threads


def validate_store_capacity(directory, object_bytes, headroom):
    if object_bytes < MIN_OBJECT_STORE_BYTES:
        raise RuntimeError(
            'Ray requires at least 75 MiB for its object store. '
            'Increase container --shm-size and its memory limit, or explicitly select '
            'an object-store directory under --user-root with --ram-object-store-dir.'
        )
    required_space = math.ceil(object_bytes * 1.5)
    if store_capacity(directory) < required_space:
        raise RuntimeError(
            f'Object store needs {required_space / GIB:.2f} GiB free at {directory}. '
            'Increase container --shm-size or reduce --object-store-gib.'
        )
    if available_memory() < object_bytes + headroom:
        raise RuntimeError(
            f'Object store and memory headroom need {(object_bytes + headroom) / GIB:.2f} GiB. '
            'Increase the memory allocation or reduce --object-store-gib/--memory-headroom-gib.'
        )


def validate_batch_layout(config):
    from verl.trainer.ppo.batch_schedule import TrainingBatchPlan
    from verl.utils.dataset.vqa20k_sampler import DATA_MIXTURE, DATA_MIXTURE_15K

    if int(config.trainer.nnodes) != 1:
        raise ValueError('This launcher starts a local Ray cluster; --nodes must be 1.')
    world_size = int(config.trainer.n_gpus_per_node)
    if world_size < 1:
        raise ValueError('Select at least one visible GPU with --gpus or CUDA_VISIBLE_DEVICES.')
    mixture = config.data.get('vqa20k_mixture')
    contracts = {'lt14k': DATA_MIXTURE, 'lt15k': DATA_MIXTURE_15K}
    if mixture not in contracts:
        raise ValueError(f'Unsupported data mixture: {mixture!r}')
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    plan = TrainingBatchPlan(
        dataset_size=contracts[mixture]['accepted_train_rows'],
        train_batch_size=int(config.data.train_batch_size),
        ppo_mini_batch_size=int(actor.ppo_mini_batch_size),
        world_size=world_size,
        epochs=int(config.trainer.total_epochs),
        drop_last=bool(config.data.train_drop_last),
        rollout_n=int(rollout.n),
    )
    if plan.total_outer_steps != int(config.trainer.total_training_steps):
        raise ValueError('Training steps do not match the data batch plan.')
    if int(rollout.tensor_model_parallel_size) != 1 or int(actor.get('ulysses_sequence_parallel_size', 1)) != 1:
        raise ValueError('The HF/FSDP launcher requires tensor and sequence parallel sizes of 1.')
    if rollout.name != 'hf' or not rollout.hf_use_replicated_module or not rollout.hf_rollout_group_balance:
        raise ValueError('The data batch plan requires HF replica rollout with UID-group balancing.')
    decode_batch = int(rollout.hf_dart_decode_batch_size)
    if decode_batch < plan.rollout_n or decode_batch % plan.rollout_n:
        raise ValueError('The rollout decode batch must contain complete rollout.n groups.')
    if not actor.use_dynamic_bsz or not actor.vision_packing.enabled:
        raise ValueError('The launcher requires dynamic vision-cost packing for equal FSDP micro-batches.')
    return plan
