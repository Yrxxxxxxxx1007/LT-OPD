"""Train LT-OPD with a frozen full-token teacher."""
from __future__ import annotations
import argparse
import os
from pathlib import Path


def build_config(model, data_dir, output, resume=False, gpus=8, nodes=1):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    root = Path(__file__).resolve().parent
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'), version_base=None):
        config = compose(config_name='vopd')
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(root/'v8.yaml'))
    config.paths = {'model': str(Path(model).resolve()), 'data': str(Path(data_dir).resolve()),
                    'output': str(Path(output).resolve()), 'training': str(root)}
    config.trainer.resume_mode = 'auto' if resume else 'disable'
    config.trainer.n_gpus_per_node = gpus
    config.trainer.nnodes = nodes
    OmegaConf.resolve(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local Qwen3.5-4B directory')
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', type=int, default=8, help='GPUs per node')
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = build_config(args.model, args.data_dir, args.output, args.resume, args.gpus, args.nodes)
    from verl.utils.config import validate_config
    validate_config(config, use_reference_policy=True, use_critic=False)
    if not (args.data_dir/'train.parquet').is_file():
        raise FileNotFoundError(args.data_dir/'train.parquet')
    if not args.resume and (args.output/'checkpoints/latest_checkpointed_iteration.txt').exists():
        raise FileExistsError('Output contains a checkpoint. Use --resume or a new output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    env = {'NCCL_DEBUG':'WARN', 'TOKENIZERS_PARALLELISM':'false',
           'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1', 'NCCL_RAS_ENABLE':'0'}
    os.environ.update(env)
    config.ray_kwargs.ray_init.runtime_env = {'env_vars':env}
    from omegaconf import OmegaConf
    OmegaConf.save(config, args.output/'config.yaml')
    from verl.trainer.main_ppo import run_ppo
    run_ppo(config)


if __name__ == '__main__':
    main()
