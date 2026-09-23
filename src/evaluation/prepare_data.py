#!/usr/bin/env python3
"""Fetch the pinned evaluation splits and write a ready-to-use datasets.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "benchmark_sources.json").read_text())


def file_identity(path):
    value = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            value.update(block)
    return size, value.hexdigest()


def configuration(output, selected, textvqa_json=None, textvqa_images=None):
    """Resolve relative paths without accessing the network."""
    config = {}
    for name in selected:
        if name == "textvqa":
            if textvqa_json is None or textvqa_images is None:
                raise ValueError("TextVQA requires --textvqa-json and --textvqa-images")
            config[name] = {"source": str(textvqa_json.expanduser().resolve()),
                            "image_root": str(textvqa_images.expanduser().resolve())}
            continue
        item = SOURCES[name]
        config[name] = ({"source": f"{name}/dataset"} if item["format"] == "hf_disk" else
                        {key: f"{name}/{value}" for key, value in item["paths"].items()})
    return config


def prepare(output, selected, textvqa_json=None, textvqa_images=None):
    output = output.expanduser().resolve()
    selected = list(dict.fromkeys(selected))
    if (textvqa_json is None) != (textvqa_images is None):
        raise ValueError("Supply both TextVQA paths together")
    if textvqa_json is not None and "textvqa" not in selected:
        selected.append("textvqa")
    config = configuration(output, selected, textvqa_json, textvqa_images)
    if "textvqa" in config:
        if not Path(config["textvqa"]["source"]).is_file() or not Path(config["textvqa"]["image_root"]).is_dir():
            raise FileNotFoundError("TextVQA JSON or image directory does not exist")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "datasets.json"
    existing = json.loads(destination.read_text()) if destination.exists() else {}
    for name, value in config.items():
        if name in existing and existing[name] != value:
            raise ValueError(f"datasets.json already maps {name} to a different source")

    cache = output / ".cache"
    for variable, folder in (("HF_HUB_CACHE", "huggingface"), ("HF_DATASETS_CACHE", "datasets"),
                             ("HF_XET_CACHE", "xet"), ("HF_ASSETS_CACHE", "assets")):
        os.environ[variable] = str(cache / folder)
    from huggingface_hub import hf_hub_download
    for name in selected:
        if name == "textvqa":
            continue
        item = SOURCES[name]
        directory = output / name
        files = []
        for entry in item["files"]:
            path = directory / entry["path"]
            if not path.exists():
                hf_hub_download(repo_id=item["repo_id"], repo_type="dataset",
                                revision=item["revision"], filename=entry["path"],
                                local_dir=directory, cache_dir=cache / "huggingface")
            size, digest = file_identity(path)
            if size != entry["size"] or digest != entry["sha256"]:
                raise ValueError(f"Benchmark file differs from its pinned source: {path}")
            files.append(str(path))
        if item["format"] == "hf_disk":
            from datasets import load_dataset, load_from_disk
            dataset_path = directory / "dataset"
            marker = dataset_path / ".lt_opd_source.json"
            if dataset_path.exists():
                if not marker.is_file() or json.loads(marker.read_text()) != item:
                    raise ValueError(f"Existing dataset has a different preparation identity: {dataset_path}")
                dataset = load_from_disk(str(dataset_path))
            else:
                dataset = load_dataset("parquet", data_files={"test": files},
                                       cache_dir=str(cache / "datasets"))
                if len(dataset["test"]) != item["rows"]:
                    raise ValueError(f"Unexpected {name} test split size")
                with tempfile.TemporaryDirectory(prefix="prepare_", dir=directory) as temporary:
                    staged = Path(temporary) / "dataset"
                    dataset.save_to_disk(str(staged))
                    (staged / marker.name).write_text(json.dumps(item, indent=2) + "\n")
                    staged.rename(dataset_path)
            if set(dataset.keys()) != {"test"} or len(dataset["test"]) != item["rows"]:
                raise ValueError(f"Unexpected {name} saved split")
        print(f"{name}: ready", flush=True)
    existing.update(config)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=[*SOURCES, "textvqa"], default=list(SOURCES),
                        help="Default: all eight downloadable benchmarks")
    parser.add_argument("--textvqa-json", type=Path, help="Existing official TextVQA_0.5.1_val.json")
    parser.add_argument("--textvqa-images", type=Path, help="Existing TextVQA validation JPEG directory")
    args = parser.parse_args()
    print(prepare(args.output, args.datasets, args.textvqa_json, args.textvqa_images))


if __name__ == "__main__":
    main()
