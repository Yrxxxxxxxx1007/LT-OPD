"""Merge a final FSDP checkpoint into a CDPruner inference bundle on CPU."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.contract import ROOT, load_contract
from training.train import sha256_file


def export_checkpoint(checkpoint, output, base_model):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from training.fsdp_merge import FSDPShardMerger

    checkpoint, output, base_model = map(lambda p: Path(p).resolve(),
                                        (checkpoint, output, base_model))
    actor = checkpoint/'actor'
    if checkpoint.name != 'global_step_175' or (checkpoint/'.checkpoint_complete').read_text().strip() != '175':
        raise ValueError('Use the complete final global_step_175 checkpoint.')
    provenance = json.loads((actor/'checkpoint_provenance.json').read_text())
    if provenance.get('vision_token_compressor',{}).get('algorithm') != 'qwen35_cdpruner_v1':
        raise ValueError('Checkpoint does not contain the V8 CDPruner actor.')
    if provenance.get('training',{}).get('outer_steps') != 175:
        raise ValueError('Unexpected training horizon.')
    marker = json.loads((actor/'CHECKPOINT_COMPLETE.json').read_text())
    for binding in marker['artifacts']:
        name = binding['relative_path']
        if name.startswith('model_world_size_'):
            path = actor/name
            expected = binding.get('size_bytes',binding.get('size'))
            if not path.is_file() or (expected is not None and path.stat().st_size != expected):
                raise ValueError(f'Incomplete model shard: {name}')
            if sha256_file(path) != binding['sha256']:
                raise ValueError(f'Model shard checksum mismatch: {name}')
    output.mkdir(parents=True, exist_ok=False)
    model_dir=output/'compressed_actor_model'
    model_dir.mkdir()
    state=FSDPShardMerger(actor).merge()
    config=AutoConfig.from_pretrained(actor/'huggingface',trust_remote_code=True)
    with init_empty_weights():
        model=AutoModelForImageTextToText.from_config(config,trust_remote_code=True)
    expected_shapes={name:tuple(value.shape) for name,value in model.state_dict().items()}
    observed_shapes={name:tuple(value.shape) for name,value in state.items()}
    if expected_shapes != observed_shapes:
        missing=set(expected_shapes)-set(observed_shapes)
        extra=set(observed_shapes)-set(expected_shapes)
        shapes={name:(expected_shapes[name],observed_shapes[name])
                for name in set(expected_shapes)&set(observed_shapes)
                if expected_shapes[name]!=observed_shapes[name]}
        raise ValueError(f'Merged state mismatch: missing={missing}, extra={extra}, shapes={shapes}')
    del model
    save_file(state,str(model_dir/'model.safetensors'),metadata={'format':'pt'})
    del state
    contract=json.loads((ROOT/'export_contract.json').read_text())
    static, summary=load_contract()
    contract['static_contract']={key:summary[key] for key in ('schema_version','canonical_sha256','file_sha256')}
    compressor=contract['vision_token_compressor']
    if provenance['vision_token_compressor'] != static['compressor']:
        raise ValueError('Checkpoint curriculum/compressor differs from this release.')
    config.vision_token_compressor=compressor
    config.vision_opd_requires_v6_dpc_runtime=True
    config.v6_dpc_export_contract='../cdpruner_export_contract.json'
    config.vision_opd_static_contract_schema=static['schema_version']
    config.save_pretrained(model_dir)
    shutil.copy2(model_dir/'config.json',model_dir/'v6_dpc_config.json')
    processor=AutoProcessor.from_pretrained(base_model,trust_remote_code=True)
    processor.image_processor.size={'shortest_edge':65536,'longest_edge':16777216}
    if hasattr(processor.image_processor,'min_pixels'):
        processor.image_processor.min_pixels=65536
        processor.image_processor.max_pixels=16777216
    template=(ROOT/'perception_chat_template_qwen35_answer_prefill.jinja').read_text()
    processor.chat_template=template
    processor.tokenizer.chat_template=template
    processor.save_pretrained(model_dir)
    processor.tokenizer.save_pretrained(model_dir)
    # Compression needs the custom cache-aware loader; keep its explicit config.
    (model_dir/'config.json').unlink(missing_ok=True)
    (output/'chat_template.jinja').write_text(template)
    for name in ('generation_config.json',):
        source=actor/'huggingface'/name
        if source.is_file():shutil.copy2(source,model_dir/name)
        elif (base_model/name).is_file():shutil.copy2(base_model/name,model_dir/name)
    shutil.copy2(actor/'checkpoint_provenance.json',output/'checkpoint_provenance.json')
    loader=output/'load_cdpruner_runtime.py'
    loader.write_text('"""LT-OPD CDPruner loader. Install this repository before use."""\n'
                      'from training.runtime import load_cdpruner_runtime, CDPrunerInferenceRuntime\n')
    contract['artifact_bindings']={
        'export_loader_sha256':sha256_file(loader),
        'compressed_actor_weight_sha256':sha256_file(model_dir/'model.safetensors'),
        'checkpoint_provenance_sha256':sha256_file(output/'checkpoint_provenance.json'),
        'training_chat_template_sha256':sha256_file(output/'chat_template.jinja'),
    }
    (output/'cdpruner_export_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (output/'EXPORT_COMPLETE.json').write_text(json.dumps({'step':175,'method':'cdpruner'})+'\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--base-model',required=True,type=Path)
    args=parser.parse_args()
    print(export_checkpoint(args.checkpoint,args.output,args.base_model))


if __name__=='__main__':
    main()
