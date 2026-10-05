"""Merge an FSDP actor checkpoint into a Hugging Face model directory on CPU."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def export_checkpoint(checkpoint, output, base_model, *, recipe_path=None):
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoProcessor
    from training.fsdp_merge import FSDPShardMerger
    from training.runtime import _model_input_hidden_size, _validate_learnable_summary_state

    checkpoint, output, base_model = (Path(p).expanduser().resolve()
                                      for p in (checkpoint, output, base_model))
    actor = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
    config_dir = actor / "huggingface" if (actor / "huggingface").is_dir() else base_model
    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    compressor = getattr(config, "vision_token_compressor", None)
    if not compressor:
        if recipe_path is None:
            raise ValueError("Checkpoint has no compressor metadata; provide its exact --recipe")
        recipe = OmegaConf.load(recipe_path)
        compressor = OmegaConf.to_container(
            recipe.actor_rollout_ref.model.vision_token_compressor, resolve=True,
        )
    elif recipe_path is not None:
        recipe = OmegaConf.load(recipe_path)
        expected = OmegaConf.to_container(recipe.actor_rollout_ref.model.vision_token_compressor, resolve=True)
        if compressor != expected:
            raise ValueError("Checkpoint compressor differs from the supplied recipe")
    from verl.models.transformers.vision_token_compressor import build_vision_token_compressor
    input_hidden_size = _model_input_hidden_size(config)
    build_vision_token_compressor(compressor, input_hidden_size=input_hidden_size, allow_legacy=False)
    state = FSDPShardMerger(actor).merge()
    # Validation precedes output creation: old exporters/checkpoints that omit
    # the learned head cannot produce an apparently valid learned export.
    _validate_learnable_summary_state(state, compressor, input_hidden_size=input_hidden_size)
    output.mkdir(parents=True, exist_ok=False)
    save_file(state, str(output / "model.safetensors"), metadata={"format": "pt"})
    config.vision_token_compressor = compressor
    config.save_pretrained(output)
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 16777216}
    if hasattr(processor.image_processor, "min_pixels"):
        processor.image_processor.min_pixels = 65536
        processor.image_processor.max_pixels = 16777216
    template = (ROOT / "perception_chat_template_qwen35_answer_prefill.jinja").read_text(encoding="utf-8")
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    processor.save_pretrained(output)
    processor.tokenizer.save_pretrained(output)
    (output / "chat_template.jinja").write_text(template, encoding="utf-8")
    for directory in (config_dir, base_model):
        source = directory / "generation_config.json"
        if source.is_file():
            shutil.copy2(source, output / source.name)
            break
    settings = {"attn_implementation": "flash_attention_2", "vision_token_compressor": compressor}
    (output / "visual_compression_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--recipe", type=Path, help="Exact training recipe, required for checkpoints missing compression metadata")
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.output, args.base_model, recipe_path=args.recipe))


if __name__ == "__main__":
    main()
