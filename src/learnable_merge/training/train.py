"""Train LT-OPD with a frozen full-token teacher."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path


def private_ram_log_root(value):
    """Validate the explicitly enabled ephemeral logging namespace."""
    directory = Path(value)
    shm = Path('/dev/shm')
    if not directory.is_absolute() or not directory.is_relative_to(shm) or len(directory.relative_to(shm).parts) < 2:
        raise ValueError('RAM logs require a private /dev/shm/<user>/<run> directory')
    if '..' in directory.parts or not any(
        len(fields) >= 3 and fields[1:3] == ['/dev/shm', 'tmpfs']
        for fields in (line.split() for line in Path('/proc/mounts').read_text().splitlines())
    ):
        raise ValueError('RAM logs require a real /dev/shm tmpfs mount')
    current = shm
    for part in directory.relative_to(shm).parts:
        current = current / part
        current.mkdir(mode=0o700, exist_ok=True)
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or current.is_symlink()
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError(f'RAM log directory must be owned, regular and mode 0700: {current}')
    return directory


def build_config(model, data_dir, output, resume=False, gpus=None, nodes=1, config_path=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    root = Path(__file__).resolve().parent
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'), version_base=None):
        config = compose(config_name='vopd')
    OmegaConf.set_struct(config, False)
    recipe = Path(config_path).resolve() if config_path else root/'v12.yaml'
    config = OmegaConf.merge(config, OmegaConf.load(recipe))
    config.paths = {'model': str(Path(model).resolve()), 'data': str(Path(data_dir).resolve()),
                    'output': str(Path(output).resolve()), 'training': str(root)}
    config.trainer.resume_mode = 'auto' if resume else 'disable'
    if gpus is not None:
        config.trainer.n_gpus_per_node = gpus
    config.trainer.nnodes = nodes
    OmegaConf.resolve(config)
    return config


def configure_runtime(config, *, output, user_root, ram_object_store_dir=None):
    """Keep persistent state private and optionally use a private RAM object store."""
    output, user_root = Path(output).resolve(), Path(user_root).resolve()
    if not output.is_relative_to(user_root):
        raise ValueError('Training output must be inside --user-root')
    output.mkdir(parents=True, exist_ok=True)
    runtime = output/'runtime'
    cache = runtime/'cache'
    # Ray's Unix socket paths must remain short even with a long output path.
    run_id = hashlib.sha256(str(output).encode()).hexdigest()[:8]
    ray_root = user_root/'r'/run_id
    ram_logs = (private_ram_log_root(os.environ['LT_OPD_RAM_LOG_ROOT'])
                if os.environ.get('LT_OPD_RAM_LOG_ROOT') else None)
    if ram_logs is not None:
        ray_root = ram_logs/'ray'
    directories = {
        'TMPDIR': runtime/'tmp', 'TMP': runtime/'tmp', 'TEMP': runtime/'tmp',
        'XDG_CACHE_HOME': cache, 'HF_HOME': cache/'hf', 'TORCH_HOME': cache/'torch',
        'TORCHINDUCTOR_CACHE_DIR': cache/'torchinductor', 'TRITON_CACHE_DIR': cache/'triton',
        'CUDA_CACHE_PATH': cache/'cuda', 'RAY_TMPDIR': ray_root,
    }
    for directory in set(directories.values()):
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.resolve().is_relative_to(user_root) and not (
            ram_logs is not None and directory == ray_root
            and directory.resolve().is_relative_to(ram_logs)
        ):
            raise ValueError(f'Runtime directory escapes --user-root: {directory}')
    spill = runtime/'spill'
    spill.mkdir(parents=True, exist_ok=True)
    logs = output/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    if ram_logs is not None:
        logs = ram_logs/'logs'
        logs.mkdir(mode=0o700, exist_ok=True)
    # This is an ephemeral, user-owned RAM cache. Spill and all persistent
    # artifacts remain under user_root; never silently mmap the shared filesystem.
    plasma = (Path(ram_object_store_dir).resolve() if ram_object_store_dir
              else Path('/dev/shm')/user_root.name/f'ltopd-{run_id}')
    env = {key: str(value) for key, value in directories.items()}
    env.update({
        'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
        'NCCL_DEBUG': 'WARN', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'NCCL_RAS_ENABLE': '0',
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
        'VERL_FILE_LOGGER_PATH': str(logs/'metrics.jsonl'),
        'VERL_FILE_LOGGER_APPEND': '1', 'VERL_RUN_ATTEMPT': '1',
    })
    os.environ.update(env)
    config.ray_kwargs.ray_init.update({
        'num_cpus': 32, 'num_gpus': int(config.trainer.n_gpus_per_node),
        'object_store_memory': 64 * 1024**3, 'include_dashboard': False,
        '_temp_dir': str(ray_root), '_plasma_directory': str(plasma),
        'object_spilling_directory': str(spill), 'runtime_env': {'env_vars': env},
    })
    return plasma


def prepare_ram_store(directory, user_root):
    """Create only an explicitly requested private ephemeral tmpfs namespace."""
    import stat
    directory = Path(directory).resolve()
    if directory.is_relative_to(Path(user_root).resolve()):
        directory.mkdir(parents=True, exist_ok=True)
        return
    shm = Path('/dev/shm')
    if not directory.is_relative_to(shm) or len(directory.relative_to(shm).parts) < 2:
        raise ValueError('External object-store cache must use a private /dev/shm/<user>/<run> directory')
    if not shm.is_dir() or not any(
        len(fields) >= 3 and fields[1:3] == ['/dev/shm', 'tmpfs']
        for fields in (line.split() for line in Path('/proc/mounts').read_text().splitlines())
    ):
        raise RuntimeError('The default object store requires a RAM-backed /dev/shm tmpfs mount')
    if shutil.disk_usage(shm).free < 96 * 1024**3:
        raise RuntimeError('The 64 GiB RAM object store requires at least 96 GiB available in /dev/shm')
    cgroup = Path('/sys/fs/cgroup')
    if (cgroup/'memory.max').is_file():
        raw_limit = (cgroup/'memory.max').read_text().strip()
        limit = None if raw_limit == 'max' else int(raw_limit)
        usage = int((cgroup/'memory.current').read_text())
        stats = dict(line.split() for line in (cgroup/'memory.stat').read_text().splitlines())
        reclaimable = int(stats.get('inactive_file', 0))
    elif (cgroup/'memory/memory.limit_in_bytes').is_file():
        cgroup = cgroup/'memory'
        limit = int((cgroup/'memory.limit_in_bytes').read_text())
        usage = int((cgroup/'memory.usage_in_bytes').read_text())
        stats = dict(line.split() for line in (cgroup/'memory.stat').read_text().splitlines())
        reclaimable = int(stats.get('total_inactive_file', stats.get('inactive_file', 0)))
    else:
        limit, usage, reclaimable = None, 0, 0
    if limit is not None and max(0, limit - usage) + min(usage, reclaimable) < 80 * 1024**3:
        raise RuntimeError('The RAM object store requires 64 GiB plus 16 GiB of instance memory headroom')
    current = shm
    for part in directory.relative_to(shm).parts:
        current = current/part
        current.mkdir(mode=0o700, exist_ok=True)
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or current.is_symlink() or info.st_uid != os.getuid():
            raise ValueError(f'RAM cache path must be an owned regular directory: {current}')
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError(f'RAM cache directory must have mode 0700: {current}')


@contextmanager
def short_ray_session(ray_root):
    """Keep Ray sockets below Linux's path limit while retaining logs in user storage."""
    import inspect
    import ray
    from ray._private.parameter import RayParams

    if ray.__version__ != '2.53.0' or 'session_name' not in inspect.signature(RayParams.__init__).parameters:
        raise RuntimeError('The short-session adapter requires the pinned Ray 2.53.0 runtime')
    ray_root = str(Path(ray_root).resolve())
    session_name = 's' + os.urandom(3).hex()
    while (Path(ray_root)/session_name).exists():
        session_name = 's' + os.urandom(3).hex()
    # Leave four bytes for any incremental suffix Ray appends to a socket name.
    socket_path = Path(ray_root)/session_name/'sockets/plasma_store'
    if len(os.fsencode(socket_path)) > 103:
        raise RuntimeError(f'User Ray socket path exceeds the supported length: {socket_path}')
    original = RayParams.__init__

    def scoped_params(instance, *args, **kwargs):
        if kwargs.get('temp_dir') == ray_root:
            kwargs['session_name'] = session_name
        original(instance, *args, **kwargs)

    RayParams.__init__ = scoped_params
    try:
        yield session_name
    finally:
        RayParams.__init__ = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local Qwen3.5-4B directory')
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('v12.yaml'))
    parser.add_argument('--gpus', type=int, help='GPUs per node; defaults to the selected recipe')
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--user-root', type=Path, required=True)
    parser.add_argument('--ram-object-store-dir', type=Path,
                        help='Override private /dev/shm/<user>/ltopd-<runhash> cache; persistent files remain under --user-root')
    parser.add_argument('--prepare-only', action='store_true', help='Write configuration without starting Ray or training')
    args = parser.parse_args()
    config = build_config(args.model, args.data_dir, args.output, args.resume, args.gpus, args.nodes, args.config)
    if config.data.get('vqa20k_mixture') == 'lt15k' and (config.trainer.n_gpus_per_node, args.nodes) != (4, 1):
        raise ValueError('The LT-15K batch plan requires the GPU layout specified in the default recipe')
    from verl.utils.config import validate_config
    validate_config(config, use_reference_policy=True, use_critic=False)
    if not (args.data_dir/'train.parquet').is_file():
        raise FileNotFoundError(args.data_dir/'train.parquet')
    if not args.resume and (args.output/'checkpoints/latest_checkpointed_iteration.txt').exists():
        raise FileExistsError('Output contains a checkpoint. Use --resume or a new output directory.')
    plasma = configure_runtime(config, output=args.output, user_root=args.user_root,
                               ram_object_store_dir=args.ram_object_store_dir)
    from omegaconf import OmegaConf
    OmegaConf.save(config, args.output/'config.yaml')
    if args.prepare_only:
        print(json.dumps({'status': 'prepared_not_started', 'config': str((args.output/'config.yaml').resolve()),
                          'gpus': config.trainer.n_gpus_per_node, 'training_steps': config.trainer.total_training_steps}))
        return
    prepare_ram_store(plasma, args.user_root)
    from verl.trainer.main_ppo import run_ppo
    with short_ray_session(config.ray_kwargs.ray_init._temp_dir):
        run_ppo(config)


if __name__ == '__main__':
    main()
