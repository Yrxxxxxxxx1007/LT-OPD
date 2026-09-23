#!/usr/bin/env python3
"""Generate benchmark answers with the exported LT-OPD model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOLS = json.loads((ROOT / "protocols.json").read_text())


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def mme_rows(config, rank, world_size):
    from datasets import load_from_disk
    if __package__:
        from .mme import build_shard_plan, official_prompt, row_metadata
    else:
        from mme import build_shard_plan, official_prompt, row_metadata
    dataset = load_from_disk(config["source"])
    if hasattr(dataset, "keys"):
        dataset = dataset["test"]
    if len(dataset) != 2374:
        raise ValueError("MME requires the complete 2374-row test split")
    metadata = [row_metadata(row, i) for i, row in enumerate(dataset)]
    plan = build_shard_plan(metadata, world_size)
    for i, row in enumerate(dataset):
        if plan[i] == rank:
            yield {"uid": str(i), "source_index": i, "image": row["image"].convert("RGB"),
                   "prompt": official_prompt(row["question"]), "question": row["question"],
                   "question_id": str(row["question_id"]), "category": row["category"],
                   "gold": metadata[i]["answer"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--data-config", type=Path, required=True,
                        help="JSON mapping datasets to source/image paths; relative to this JSON")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("require 0 <= rank < world-size")

    if __package__:
        from .data import public_row, row_messages, rows_for
        from .mme import runtime_prediction_content, runtime_route_query
        from .runtime import build_route_query, load_export_runtime, prediction_record
    else:
        from data import public_row, row_messages, rows_for
        from mme import runtime_prediction_content, runtime_route_query
        from runtime import build_route_query, load_export_runtime, prediction_record

    settings = PROTOCOLS[args.dataset]
    config = json.loads(args.data_config.read_text(encoding="utf-8"))[args.dataset]
    config = {k: str((args.data_config.resolve().parent / v).resolve()) for k, v in config.items()}
    output_dir = args.output_dir / args.dataset / f"rank{args.rank}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"model": str(args.export_dir.resolve()), "dataset": args.dataset,
                  "data": config, "settings": settings, "world_size": args.world_size}
    config_path = output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError("Use a different output directory for a different evaluation configuration")
    atomic_json(config_path, run_config)
    iterator = (mme_rows(config, args.rank, args.world_size) if args.dataset == "mme" else
                rows_for(args.dataset, config, rank=args.rank, world_size=args.world_size))
    runtime = None
    pending = []
    seen = set()
    count = 0

    def flush():
        nonlocal runtime, pending
        if not pending:
            return
        if runtime is None:
            runtime = load_export_runtime(args.export_dir)
        generated = runtime.generate(
            [p["messages"] for p in pending],
            response_length=settings["official_max_new_tokens"],
            visual_compression_mode=pending[0]["mode"],
            route_queries_batch=[p["query"] for p in pending],
        )
        for i, item in enumerate(pending):
            response = prediction_record(runtime, generated, i,
                                         settings["official_max_new_tokens"], settings["until"])
            if args.dataset == "mme":
                response["pred"] = runtime_prediction_content(
                    decoded_prediction=generated["decoded_predictions"][i],
                    sampled_continuation=generated["decoded_raw_continuations"][i])
            atomic_json(item["path"], {
                "dataset": args.dataset, "source_index": item["row"]["source_index"],
                "uid": item["row"]["uid"], "row": item["row"], "response": response,
            })
        pending = []

    for row in iterator:
        count += 1
        if row["uid"] in seen:
            raise ValueError(f"Duplicate question ID: {row['uid']}")
        seen.add(row["uid"])
        public = public_row(row)
        target = output_dir / f"{row['source_index']:06d}.json"
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            if previous["row"] != public:
                raise ValueError(f"Saved question differs from the selected dataset: {target}")
            continue
        messages, image_count = row_messages(row)
        query = runtime_route_query(row["question"]) if args.dataset == "mme" else build_route_query(row["prompt"])
        mode = "merge" if image_count else "no_image"
        multiple = image_count > 1
        if pending and (multiple or pending[0]["image_count"] > 1 or pending[0]["mode"] != mode):
            flush()
        pending.append({"row": public, "messages": messages, "image_count": image_count,
                        "query": query, "mode": mode, "path": target})
        if multiple or len(pending) == settings["batch_size"]:
            flush()
    flush()
    if args.dataset != "mme":
        expected = len(range(args.rank, settings["expected_count"], args.world_size))
        if count != expected:
            raise ValueError(f"Expected {expected} rows in shard, received {count}")
    print(json.dumps({"dataset": args.dataset, "rank": args.rank, "completed": count}))


if __name__ == "__main__":
    main()
