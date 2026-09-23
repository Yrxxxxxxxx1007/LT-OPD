"""Train the LT-OPD V8 recipe with a frozen full-token teacher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from training.contract import canonical_sha256, load_contract


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_config(model, data_dir, output, resume=False):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    root = Path(__file__).resolve().parent
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'),
                               version_base=None):
        config = compose(config_name='vopd')
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(root/'v8.yaml'))
    config.paths = {'model': str(Path(model).resolve()),
                    'data': str(Path(data_dir).resolve()),
                    'output': str(Path(output).resolve()), 'training': str(root)}
    config.trainer.resume_mode = 'auto' if resume else 'disable'
    OmegaConf.resolve(config)
    return config


def validate_dataset(data_dir):
    import pyarrow.parquet as pq
    from collections import Counter
    data_dir = Path(data_dir)
    path = data_dir/'train.parquet'
    table = pq.read_table(path, columns=['sample_uid','sampling_bucket','images','teacher_images','extra_info'])
    rows = table.to_pylist()
    spec = load_contract()[0]['data']
    quotas = spec['global_batch']['exact_epoch_bucket_quotas']
    if len(rows) != 14000 or Counter(row['sampling_bucket'] for row in rows) != quotas:
        raise ValueError('Expected the complete LT-14K mixture with the published source quotas.')
    if len({row['sample_uid'] for row in rows}) != 14000:
        raise ValueError('Training sample identifiers must be unique.')
    for row in rows:
        images = row['images']
        if images != row['teacher_images'] or len(images) != 1:
            raise ValueError(f"Student/teacher must use the same single full image: {row['sample_uid']}")
        image_path = images[0]
        if isinstance(image_path, dict):
            image_path = image_path.get('path')
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError(f"Missing image for {row['sample_uid']}: {image_path}")
        extra = row['extra_info']
        if isinstance(extra, str):
            extra = json.loads(extra)
        if not isinstance(extra, dict) or not extra.get('visual_capacity_profiles'):
            raise ValueError(f"Missing image capacity metadata for {row['sample_uid']}")
    manifest = json.loads((data_dir/'manifest.json').read_text())
    digest = sha256_file(path)
    if manifest.get('train_sha256') != digest:
        raise ValueError('train.parquet differs from manifest.json; run data.prepare again.')
    return digest


def configure_runtime(config, dataset_digest):
    from omegaconf import OmegaConf
    root = Path(config.paths.output)
    root.mkdir(parents=True, exist_ok=True)
    model = Path(config.paths.model)
    # Hash metadata locally; the documented HF revision identifies the base weights.
    identity = {p.name: sha256_file(p) for p in sorted(model.iterdir())
                if p.is_file() and p.suffix in {'.json','.jinja'}}
    if 'config.json' not in identity:
        raise FileNotFoundError(f'Missing model config: {model}/config.json')
    model_digest = canonical_sha256(identity)
    source = Path(__file__).resolve().parents[1]
    source_digest = canonical_sha256({str(p.relative_to(source)):sha256_file(p)
                                     for package in ('verl','training')
                                     for p in sorted((source/package).rglob('*.py'))})
    run_identity = OmegaConf.to_container(config, resolve=True)
    run_identity['trainer']['resume_mode'] = 'auto'
    run_digest = canonical_sha256(run_identity)
    env = {
        'VERL_RUNTIME_SOURCE_FINGERPRINT': source_digest,
        'VERL_MODEL_ASSET_FINGERPRINT': model_digest,
        'VERL_TEACHER_ASSET_FINGERPRINT': model_digest,
        'VERL_IMAGE_ASSET_FINGERPRINT': dataset_digest,
        'VERL_V6_LAUNCH_CONTRACT_SHA256': run_digest,
        'VERL_V6_PREFLIGHT_SHA256': run_digest,
        'VERL_V6_RUN_ROOT': str(root),
        'VERL_V6_INITIAL_PROFILE':'M7', 'VERL_V6_SELECTED_PROFILE':'M7',
        'VERL_V6_INITIAL_MAX_COMPOSITE_COST_PER_GPU':'1664327',
        'VERL_FILE_LOGGER_PATH': str(root/'metrics.jsonl'),
        'VERL_FILE_LOGGER_APPEND': '1' if config.trainer.resume_mode == 'auto' else '0',
        'NCCL_DEBUG':'WARN', 'TOKENIZERS_PARALLELISM':'false',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1', 'NCCL_RAS_ENABLE':'0',
    }
    os.environ.update(env)
    config.ray_kwargs.ray_init.runtime_env = {'env_vars':env}
    OmegaConf.save(config, root/'config.yaml')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local Qwen3.5-4B directory')
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate data/config on CPU without launching Ray')
    args = parser.parse_args()
    config = build_config(args.model, args.data_dir, args.output, args.resume)
    from verl.utils.config import validate_config
    validate_config(config, use_reference_policy=True, use_critic=False)
    dataset_digest = validate_dataset(args.data_dir)
    if args.dry_run:
        print(json.dumps({'status':'ok','training_rows':14000,'steps':175,'gpus':8,
                          'curriculum':'25% hold 14 steps, cosine to 5%, final 75-step hold',
                          'train_sha256':dataset_digest},indent=2))
        return
    if not args.resume and (args.output/'checkpoints/latest_checkpointed_iteration.txt').exists():
        raise FileExistsError('Output already contains a checkpoint. Use --resume or a new output directory.')
    configure_runtime(config, dataset_digest)
    from verl.trainer.main_ppo import run_ppo
    run_ppo(config)


if __name__ == '__main__':
    main()
