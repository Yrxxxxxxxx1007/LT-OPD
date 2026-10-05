"""Train LT-OPD with a frozen full-token teacher."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path

from training.runtime_settings import (
    bytes_from_gib,
    prepare_resume,
    select_resources,
    uses_existing_ray,
    validate_batch_layout,
    validate_max_steps,
    validate_resume_settings,
    validate_store_capacity,
)


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


def build_config(model, data_dir, output, resume=False, gpus=None, nodes=None, config_path=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    root = Path(__file__).resolve().parent
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'), version_base=None):
        config = compose(config_name='vopd')
    OmegaConf.set_struct(config, False)
    recipe = Path(config_path).expanduser().resolve() if config_path else root/'default.yaml'
    if recipe == root/'v12.yaml' and not recipe.exists():
        recipe = root/'default.yaml'
    config = OmegaConf.merge(
        config,
        {'trainer': {'max_steps': None}},
        {'runtime_resources': {'memory_headroom_gib': 16, 'omp_threads': None}},
        OmegaConf.load(root/'default.yaml'),
    )
    saved = None
    if resume:
        saved_path = Path(output).expanduser().resolve()/'config.yaml'
        if not saved_path.is_file():
            raise FileNotFoundError(f'--resume requires the saved recipe: {saved_path}')
        saved = OmegaConf.load(saved_path)
        for name, value in (('model', model), ('data', data_dir)):
            if Path(saved.paths[name]).expanduser().resolve() != Path(value).expanduser().resolve():
                raise ValueError(f'--resume requires the saved {name} path: {saved.paths[name]}')
        config = OmegaConf.merge(config, saved)
        config.trainer.max_steps = None
    if config_path is not None:
        config = OmegaConf.merge(config, OmegaConf.load(recipe))
    config.paths = {'model': str(Path(model).expanduser().resolve()), 'data': str(Path(data_dir).expanduser().resolve()),
                    'output': str(Path(output).expanduser().resolve()), 'training': str(root)}
    config.trainer.resume_mode = 'auto' if resume else 'disable'
    if gpus is not None:
        config.trainer.n_gpus_per_node = gpus
    elif config.trainer.get('n_gpus_per_node') is None:
        import torch
        config.trainer.n_gpus_per_node = torch.cuda.device_count()
    if nodes is not None:
        config.trainer.nnodes = nodes
    OmegaConf.resolve(config)
    if saved is not None:
        validate_resume_settings(saved, config)
    return config


def configure_runtime(config, *, output, user_root, ram_object_store_dir=None, ray_initialized=False):
    """Keep persistent state private and optionally use a private RAM object store."""
    from omegaconf import OmegaConf
    output, user_root = Path(output).expanduser().resolve(), Path(user_root).expanduser().resolve()
    ray_init = config.ray_kwargs.ray_init
    existing_cluster = uses_existing_ray(ray_init, ray_initialized=ray_initialized)
    runtime_env = ray_init.get('runtime_env') or {}
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
    plasma = (Path(ram_object_store_dir).expanduser().resolve() if ram_object_store_dir
              else Path('/dev/shm')/user_root.name/f'ltopd-{run_id}')
    if existing_cluster:
        plasma = None
        omp_threads = config.runtime_resources.get('omp_threads')
        configured_env = runtime_env.get('env_vars') or {}
        if omp_threads is None:
            omp_threads = configured_env.get('OMP_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', 4))
        omp_threads = int(omp_threads)
        if omp_threads < 1:
            raise ValueError('OMP thread count must be positive')
    else:
        cpus, object_bytes, omp_threads = select_resources(config, plasma)
    env = {key: str(value) for key, value in directories.items()}
    env.update({
        'OMP_NUM_THREADS': str(omp_threads), 'TOKENIZERS_PARALLELISM': 'false',
        'NCCL_DEBUG': 'WARN', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'NCCL_RAS_ENABLE': '0',
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'VERL_FILE_LOGGER_PATH': str(logs/'metrics.jsonl'),
        'VERL_FILE_LOGGER_APPEND': '1', 'VERL_RUN_ATTEMPT': '1',
    })
    env.update(dict(runtime_env.get('env_vars') or {}))
    paths = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]
    paths.extend(str(env.get('PYTHONPATH', '')).split(os.pathsep))
    paths.extend(os.environ.get('PYTHONPATH', '').split(os.pathsep))
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    env['LT_OPD_IMPLEMENTATION'] = 'current'
    env['OMP_NUM_THREADS'] = str(omp_threads)
    env = {key: str(value) for key, value in env.items()}
    os.environ.update(env)
    if existing_cluster:
        for key in ('num_cpus', 'num_gpus', 'object_store_memory', 'include_dashboard',
                    '_temp_dir', '_plasma_directory', 'object_spilling_directory'):
            ray_init.pop(key, None)
    else:
        ray_init.update({
            'num_cpus': cpus, 'num_gpus': int(config.trainer.n_gpus_per_node),
            'object_store_memory': object_bytes, 'include_dashboard': False,
            '_temp_dir': str(ray_root), '_plasma_directory': str(plasma),
            'object_spilling_directory': str(spill),
        })
    ray_init.runtime_env = OmegaConf.merge(runtime_env, {'env_vars': env})
    return plasma


def prepare_ram_store(directory, user_root, *, object_bytes, headroom):
    """Create only an explicitly requested private ephemeral tmpfs namespace."""
    import stat
    directory = Path(directory).resolve()
    validate_store_capacity(directory, object_bytes, headroom)
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
    parser.add_argument('--config', type=Path, help='YAML overrides for the default or resumed training recipe')
    parser.add_argument('--gpus', type=int, help='Number of visible GPUs to use; defaults to the recipe or all visible GPUs')
    parser.add_argument('--nodes', type=int, help='Number of nodes (the local launcher supports one)')
    parser.add_argument('--cpus', type=int, help='Ray CPU allocation; defaults to available resources')
    parser.add_argument('--object-store-gib', type=float, help='Ray object-store size in GiB')
    parser.add_argument('--memory-headroom-gib', type=float, help='Memory reserved outside the Ray object store')
    parser.add_argument('--omp-threads', type=int, help='CPU threads per worker; respects OMP_NUM_THREADS by default')
    parser.add_argument('--max-steps', type=int, help='Stop at this training step while retaining the full recipe schedule')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--user-root', type=Path, required=True)
    parser.add_argument('--ram-object-store-dir', type=Path,
                        help='Override private /dev/shm/<user>/ltopd-<runhash> cache; persistent files remain under --user-root')
    parser.add_argument('--prepare-only', action='store_true', help='Write configuration without starting Ray or training')
    args = parser.parse_args()
    config = build_config(args.model, args.data_dir, args.output, args.resume, args.gpus, args.nodes, args.config)
    if args.max_steps is not None:
        config.trainer.max_steps = args.max_steps
    validate_max_steps(config.trainer.max_steps)
    args.data_dir, args.output = (Path(config.paths.data), Path(config.paths.output))
    import ray
    existing_cluster = uses_existing_ray(config.ray_kwargs.ray_init, ray_initialized=ray.is_initialized())
    if existing_cluster and any(value is not None for value in
                                (args.cpus, args.object_store_gib, args.ram_object_store_dir)):
        raise ValueError('--cpus and object-store allocation apply to a new local Ray instance; '
                         'configure these resources on the existing cluster.')
    if args.cpus is not None:
        config.ray_kwargs.ray_init.num_cpus = args.cpus
    if args.object_store_gib is not None:
        config.ray_kwargs.ray_init.object_store_memory = bytes_from_gib(args.object_store_gib, '--object-store-gib')
    if args.memory_headroom_gib is not None:
        config.runtime_resources.memory_headroom_gib = args.memory_headroom_gib
    if args.omp_threads is not None:
        config.runtime_resources.omp_threads = args.omp_threads
    validate_batch_layout(config)
    from verl.utils.config import validate_config
    validate_config(config, use_reference_policy=True, use_critic=False)
    if not (args.data_dir/'train.parquet').is_file():
        raise FileNotFoundError(args.data_dir/'train.parquet')
    if args.resume:
        prepare_resume(config)
    if not args.resume and (args.output/'checkpoints/latest_checkpointed_iteration.txt').exists():
        raise FileExistsError('Output contains a checkpoint. Use --resume or a new output directory.')
    if not args.prepare_only and not existing_cluster:
        import torch
        visible_gpus = torch.cuda.device_count()
        if int(config.trainer.n_gpus_per_node) > visible_gpus:
            raise ValueError(f'Requested {config.trainer.n_gpus_per_node} GPUs but only {visible_gpus} are visible.')
    plasma = configure_runtime(config, output=args.output, user_root=args.user_root,
                               ram_object_store_dir=args.ram_object_store_dir,
                               ray_initialized=ray.is_initialized())
    from omegaconf import OmegaConf
    OmegaConf.save(config, args.output/'config.yaml')
    if args.prepare_only:
        print(json.dumps({'status': 'prepared_not_started', 'config': str((args.output/'config.yaml').resolve()),
                          'gpus': config.trainer.n_gpus_per_node, 'training_steps': config.trainer.total_training_steps}))
        return
    from verl.trainer.main_ppo import run_ppo
    if existing_cluster:
        run_ppo(config)
    else:
        prepare_ram_store(
            plasma, args.user_root,
            object_bytes=int(config.ray_kwargs.ray_init.object_store_memory),
            headroom=bytes_from_gib(config.runtime_resources.memory_headroom_gib, 'memory_headroom_gib', allow_zero=True),
        )
        with short_ray_session(config.ray_kwargs.ray_init._temp_dir):
            run_ppo(config)


if __name__ == '__main__':
    main()
