"""CPU allocation for the training launcher."""
from __future__ import annotations

import os
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
    roots = [root, root / 'cpu', root / 'cpu,cpuacct']
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


def configure_cpus(ray_init, gpus, *, cpus=None, ray_initialized=False):
    """Set local Ray resources without overriding an existing cluster."""
    address = ray_init.get('address') or os.environ.get('RAY_ADDRESS')
    if ray_initialized or (address and address != 'local'):
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
