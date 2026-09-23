#!/usr/bin/env python3
"""Run the nine benchmarks across visible GPUs, resuming existing predictions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROTOCOLS = json.loads((ROOT / "protocols.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated CUDA device IDs")
    parser.add_argument("--datasets", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != len(set(gpus)) or not all(x.isdigit() for x in gpus) or args.workers_per_gpu < 1:
        parser.error("Use distinct numeric GPU IDs and workers-per-gpu >= 1")
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
                        log_path = args.output_dir / dataset / f"rank{rank}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open("a", encoding="utf-8")
                        command = [sys.executable, str(ROOT / "infer.py"), "--dataset", dataset,
                                   "--export-dir", str(args.export_dir.resolve()),
                                   "--data-config", str(args.data_config.resolve()),
                                   "--output-dir", str(args.output_dir.resolve()),
                                   "--rank", str(rank), "--world-size", str(world_size)]
                        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                        active.append({"process": child, "log": log, "gpu": gpu, "rank": rank})
                for item in active[:]:
                    code = item["process"].poll()
                    if code is not None:
                        item["log"].close()
                        active.remove(item)
                        if code:
                            raise RuntimeError(f"{dataset} rank {item['rank']} exited {code}; see its log")
                if active:
                    time.sleep(1)
        finally:
            for item in active:
                if item["process"].poll() is None:
                    item["process"].terminate()
                item["process"].wait()
                item["log"].close()
        print(f"{dataset}: inference complete", flush=True)


if __name__ == "__main__":
    main()
