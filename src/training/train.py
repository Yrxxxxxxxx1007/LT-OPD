"""Train LT-OPD with a frozen full-token teacher."""
from __future__ import annotations
import argparse
import os
from pathlib import Path


def build_config(model, data_dir, output, resume=False, gpus=None, nodes=None, config_path=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    root = Path(__file__).resolve().parent
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'), version_base=None):
        config = compose(config_name='vopd')
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, {'trainer': {'max_steps': None}}, OmegaConf.load(root/'v8.yaml'))
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
        config = OmegaConf.merge(config, OmegaConf.load(Path(config_path).expanduser().resolve()))
    config.paths = {'model': str(Path(model).expanduser().resolve()), 'data': str(Path(data_dir).expanduser().resolve()),
                    'output': str(Path(output).expanduser().resolve()), 'training': str(root)}
    config.trainer.resume_mode = 'auto' if resume else 'disable'
    if gpus is not None:
        config.trainer.n_gpus_per_node = gpus
    if nodes is not None:
        config.trainer.nnodes = nodes
    OmegaConf.resolve(config)
    if saved is not None:
        from training.runtime_settings import validate_resume_settings
        validate_resume_settings(saved, config)
    return config


def configure_runtime(config, *, cpus=None, ray_initialized=False):
    from omegaconf import OmegaConf
    from training.runtime_settings import configure_cpus

    ray_init = config.ray_kwargs.ray_init
    configure_cpus(ray_init, config.trainer.n_gpus_per_node,
                   cpus=cpus, ray_initialized=ray_initialized)
    runtime_env = ray_init.get('runtime_env') or {}
    env = {'NCCL_DEBUG': 'WARN', 'TOKENIZERS_PARALLELISM': 'false',
           'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'NCCL_RAS_ENABLE': '0'}
    env.update(dict(runtime_env.get('env_vars') or {}))
    paths = [str(Path(__file__).resolve().parents[1])]
    paths.extend(str(env.get('PYTHONPATH', '')).split(os.pathsep))
    paths.extend(os.environ.get('PYTHONPATH', '').split(os.pathsep))
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    env['LT_OPD_IMPLEMENTATION'] = 'legacy'
    env = {key: str(value) for key, value in env.items()}
    os.environ.update(env)
    ray_init.runtime_env = OmegaConf.merge(runtime_env, {'env_vars': env})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local Qwen3.5-4B directory')
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='YAML overrides for the default training recipe')
    parser.add_argument('--gpus', type=int, help='GPUs per node; defaults to the recipe')
    parser.add_argument('--nodes', type=int, help='Number of nodes; defaults to the recipe')
    parser.add_argument('--cpus', type=int, help='Local Ray CPU allocation; defaults to available resources')
    parser.add_argument('--max-steps', type=int, help='Stop at this training step while retaining the full recipe schedule')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = build_config(args.model, args.data_dir, args.output, args.resume, args.gpus, args.nodes, args.config)
    if args.max_steps is not None:
        config.trainer.max_steps = args.max_steps
    from training.runtime_settings import prepare_resume, validate_local_gpus, validate_max_steps
    validate_max_steps(config.trainer.max_steps)
    args.data_dir, args.output = (Path(config.paths.data), Path(config.paths.output))
    from verl.utils.config import validate_config
    validate_config(config, use_reference_policy=True, use_critic=False)
    if not (args.data_dir/'train.parquet').is_file():
        raise FileNotFoundError(args.data_dir/'train.parquet')
    if args.resume:
        prepare_resume(config)
    if not args.resume and (args.output/'checkpoints/latest_checkpointed_iteration.txt').exists():
        raise FileExistsError('Output contains a checkpoint. Use --resume or a new output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    import ray
    validate_local_gpus(config, ray_initialized=ray.is_initialized())
    configure_runtime(config, cpus=args.cpus, ray_initialized=ray.is_initialized())
    from omegaconf import OmegaConf
    OmegaConf.save(config, args.output/'config.yaml')
    from verl.trainer.main_ppo import run_ppo
    run_ppo(config)


if __name__ == '__main__':
    main()
