#!/usr/bin/env python3
"""Run the nine benchmarks across visible GPUs, resuming existing predictions."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROTOCOLS = json.loads((ROOT / "protocols.json").read_text())


def select_gpus(requested, inherited):
    """Resolve logical GPU indices without escaping the inherited CUDA mask."""
    def parse(value):
        devices = [item.strip() for item in value.split(",")]
        if not all(re.fullmatch(r"(?:\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)", item)
                   for item in devices):
            raise ValueError("Use CUDA device indices or GPU/MIG UUIDs")
        devices = [str(int(item)) if item.isdigit() else item for item in devices]
        if len(devices) != len(set(devices)):
            raise ValueError("GPU devices must be distinct")
        return devices

    if inherited is not None and inherited.strip() in ("", "-1"):
        raise ValueError("CUDA_VISIBLE_DEVICES exposes no GPUs")
    visible = parse(inherited) if inherited is not None else None
    if requested is None:
        return visible if visible is not None else ["0"]
    selected = parse(requested)
    if visible is None:
        return selected
    resolved = []
    for device in selected:
        if device.isdigit():
            index = int(device)
            if index >= len(visible):
                raise ValueError("--gpus indices refer to devices within CUDA_VISIBLE_DEVICES")
            device = visible[index]
        elif device not in visible:
            raise ValueError("Requested GPU UUID is outside CUDA_VISIBLE_DEVICES")
        resolved.append(device)
    if len(resolved) != len(set(resolved)):
        raise ValueError("GPU devices must be distinct")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", help="Comma-separated visible GPU indices or UUIDs; defaults to CUDA_VISIBLE_DEVICES, or 0 when unset")
    parser.add_argument("--implementation", choices=("auto", "legacy", "current"), default="auto",
                        help="Runtime implementation; auto reads the exported compressor configuration")
    parser.add_argument("--datasets", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    args = parser.parse_args()
    try:
        gpus = select_gpus(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    except ValueError as error:
        parser.error(str(error))
    if args.workers_per_gpu < 1:
        parser.error("workers-per-gpu must be >= 1")
    # Also support launching this script directly from a source checkout.
    sys.path.insert(0, str(ROOT.parent))
    from lt_opd.cli import _implementation_for_export, _select
    implementation = (_implementation_for_export(args.export_dir)
                      if args.implementation == "auto" else args.implementation)
    _select(implementation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        # Preserve the logical shards and microbatch sizes of the V8 evaluator.
        world_size = 24 if dataset in ("gqa", "pope", "textvqa") else 8
        pending = list(range(world_size))
        active = []
        try:
            while pending or active:
                for gpu in gpus:
                    while pending and sum(item["gpu"] == gpu for item in active) < args.workers_per_gpu:
                        rank = pending.pop(0)
                        command = [sys.executable, str(ROOT / "infer.py"), "--dataset", dataset,
                                   "--export-dir", str(args.export_dir.resolve()),
                                   "--data-config", str(args.data_config.resolve()),
                                   "--output-dir", str(args.output_dir.resolve()),
                                   "--rank", str(rank), "--world-size", str(world_size)]
                        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                        child = subprocess.Popen(command, env=environment)
                        active.append({"process": child, "gpu": gpu, "rank": rank})
                for item in active[:]:
                    code = item["process"].poll()
                    if code is not None:
                        active.remove(item)
                        if code:
                            raise RuntimeError(f"{dataset} rank {item['rank']} exited {code}")
                if active:
                    time.sleep(1)
        finally:
            for item in active:
                if item["process"].poll() is None:
                    item["process"].terminate()
                item["process"].wait()
        print(f"{dataset}: inference complete", flush=True)


if __name__ == "__main__":
    main()
