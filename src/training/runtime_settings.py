"""CPU allocation for the training launcher."""
from __future__ import annotations

import os
import json
from pathlib import Path


def _quota_cpus(directory):
    try:
        if (directory / 'cpu.max').is_file():
            quota, period = (directory / 'cpu.max').read_text().split()
            return None if quota == 'max' else max(1, int(quota) // int(period))
        quota = int((directory / 'cpu.cfs_quota_us').read_text())
        period = int((directory / 'cpu.cfs_period_us').read_text())
        return max(1, quota // period) if quota > 0 else None
    except (OSError, ValueError, ZeroDivisionError):
        return None


def available_cpus():
    """Respect the process affinity and applicable Linux CPU quotas."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    root = Path('/sys/fs/cgroup')
    roots = [root, root / 'cpu', root / 'cpu,cpuacct', root / 'cpuacct,cpu']
    candidates = set(roots)
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            if controllers and 'cpu' not in controllers.split(','):
                continue
            for base in roots[:1] if not controllers else roots[1:]:
                current = base / relative.lstrip('/')
                if '..' in current.parts or not current.is_dir():
                    continue
                while current != base:
                    candidates.add(current)
                    current = current.parent
    except (OSError, ValueError):
        pass
    for directory in candidates:
        quota = _quota_cpus(directory)
        if quota is not None:
            count = min(count, quota)
    return max(1, count)


def uses_existing_ray(ray_init, *, ray_initialized=False):
    address = ray_init.get('address') or os.environ.get('RAY_ADDRESS')
    return bool(ray_initialized or (address and address != 'local'))


def configure_cpus(ray_init, gpus, *, cpus=None, ray_initialized=False):
    """Set local Ray resources without overriding an existing cluster."""
    if uses_existing_ray(ray_init, ray_initialized=ray_initialized):
        if cpus is not None:
            raise ValueError('--cpus applies to a new local Ray instance; configure CPUs on the existing cluster.')
        ray_init.pop('num_cpus', None)
        return
    available = available_cpus()
    minimum = 3 * int(gpus) + 1
    requested = cpus if cpus is not None else ray_init.get('num_cpus')
    selected = min(available, max(32, minimum)) if requested is None else int(requested)
    if not minimum <= selected <= available:
        raise ValueError(
            f'Local Ray needs at least {minimum} CPUs for {gpus} GPUs; '
            f'requested={selected}, available={available}. '
            'Adjust --cpus/--gpus or the CPU allocation.'
        )
    ray_init.num_cpus = selected


def validate_max_steps(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError('--max-steps must be a positive integer')


def validate_local_gpus(config, *, ray_initialized=False):
    if int(config.trainer.n_gpus_per_node) < 1 or int(config.trainer.nnodes) < 1:
        raise ValueError('--gpus and --nodes must be positive')
    if not uses_existing_ray(config.ray_kwargs.ray_init, ray_initialized=ray_initialized):
        import torch
        visible = torch.cuda.device_count()
        if int(config.trainer.n_gpus_per_node) > visible:
            raise ValueError(f'Requested {config.trainer.n_gpus_per_node} GPUs but only {visible} are visible.')


def validate_resume_settings(saved, config):
    """Keep the saved training recipe when resuming optimizer state."""
    from omegaconf import OmegaConf
    for key in ('data', 'actor_rollout_ref', 'algorithm', 'reward_model', 'custom_reward_function', 'max_model_len',
                'trainer.total_training_steps', 'trainer.total_epochs', 'trainer.balance_batch'):
        before = OmegaConf.select(saved, key)
        after = OmegaConf.select(config, key)
        if before != after:
            raise ValueError(f'Resume changes the saved training setting {key}; start a new output directory instead.')


def prepare_resume(config):
    """Select a committed checkpoint with every optimizer and RNG shard."""
    root = Path(config.trainer.default_local_dir).expanduser().resolve()
    tracker = root / 'latest_checkpointed_iteration.txt'
    if not tracker.is_file():
        raise FileNotFoundError(f'--resume requires a saved checkpoint: {tracker}')
    value = tracker.read_text().strip()
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f'Invalid checkpoint tracker: {tracker}')
    step = int(value)
    checkpoint = root / f'global_step_{step}'
    marker = checkpoint / '.checkpoint_complete'
    if not marker.is_file() or marker.read_text().strip() != str(step):
        raise ValueError(f'Checkpoint has not completed saving: {checkpoint}')
    actor = checkpoint / 'actor'
    metadata = json.loads((actor / 'fsdp_config.json').read_text())
    world_size = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    saved_world_size = metadata.get('world_size')
    if isinstance(saved_world_size, bool) or saved_world_size != world_size:
        raise ValueError(f'Checkpoint world size {saved_world_size} does not match requested {world_size}')
    required = [checkpoint / 'data.pt']
    required.extend(actor / f'{kind}_world_size_{world_size}_rank_{rank}.pt'
                    for rank in range(world_size) for kind in ('model', 'optim', 'extra_state'))
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError('Checkpoint is missing saved state: ' + ', '.join(missing))
    config.trainer.resume_mode = 'resume_path'
    config.trainer.resume_from_path = str(checkpoint)
    return step
