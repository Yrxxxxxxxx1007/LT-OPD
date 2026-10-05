"""Merge an FSDP actor checkpoint into a Hugging Face model directory on CPU."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@contextmanager
def _export_directory(output):
    """Publish a complete export without leaving partial files at its destination."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        yield staging
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _saved_recipe(actor):
    """Find config.yaml only in the run that declares this checkpoint directory."""
    from omegaconf import OmegaConf

    step_dir = actor.parent
    if actor.name != "actor" or re.fullmatch(r"global_step_\d+", step_dir.name) is None:
        return None
    run_dir = step_dir.parent.parent
    path = run_dir / "config.yaml"
    if not path.is_file():
        return None
    recipe = OmegaConf.load(path)
    checkpoint_root = OmegaConf.select(recipe, "trainer.default_local_dir")
    output_root = OmegaConf.select(recipe, "paths.output")
    if (not isinstance(checkpoint_root, str) or not Path(checkpoint_root).is_absolute()
            or Path(checkpoint_root).resolve() != step_dir.parent
            or not isinstance(output_root, str) or not Path(output_root).is_absolute()
            or Path(output_root).resolve() != run_dir):
        raise ValueError("Saved config.yaml does not identify this training run; provide its exact --recipe")
    return recipe


def export_checkpoint(checkpoint, output, base_model, *, recipe_path=None):
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoProcessor
    from training.fsdp_merge import FSDPShardMerger

    checkpoint, output, base_model = (Path(p).expanduser().resolve()
                                      for p in (checkpoint, output, base_model))
    if output.exists():
        raise FileExistsError(output)
    actor = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
    config_dir = actor / "huggingface" if (actor / "huggingface").is_dir() else base_model
    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    compressor = getattr(config, "vision_token_compressor", None)
    recipe = OmegaConf.load(recipe_path) if recipe_path is not None else None
    if not compressor:
        if recipe is None:
            recipe = _saved_recipe(actor)
        if recipe is None:
            raise ValueError("Checkpoint has no compressor metadata or saved run config; provide its exact --recipe")
        compressor = OmegaConf.to_container(
            recipe.actor_rollout_ref.model.vision_token_compressor, resolve=True,
        )
    elif recipe is not None:
        expected = OmegaConf.to_container(recipe.actor_rollout_ref.model.vision_token_compressor, resolve=True)
        if compressor != expected:
            raise ValueError("Checkpoint compressor differs from the supplied recipe")
    state = FSDPShardMerger(actor).merge()
    with _export_directory(output) as destination:
        save_file(state, str(destination / "model.safetensors"), metadata={"format": "pt"})
        config.vision_token_compressor = compressor
        config.save_pretrained(destination)
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 16777216}
        if hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = 65536
            processor.image_processor.max_pixels = 16777216
        template = (ROOT / "perception_chat_template_qwen35_answer_prefill.jinja").read_text(encoding="utf-8")
        processor.chat_template = template
        processor.tokenizer.chat_template = template
        processor.save_pretrained(destination)
        processor.tokenizer.save_pretrained(destination)
        (destination / "chat_template.jinja").write_text(template, encoding="utf-8")
        for directory in (config_dir, base_model):
            source = directory / "generation_config.json"
            if source.is_file():
                shutil.copy2(source, destination / source.name)
                break
        settings = {"attn_implementation": "flash_attention_2", "vision_token_compressor": compressor}
        (destination / "cdpruner_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--recipe", type=Path, help="Exact training recipe; otherwise use checkpoint metadata or its saved run config")
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.output, args.base_model, recipe_path=args.recipe))


if __name__ == "__main__":
    main()
