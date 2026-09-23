#!/usr/bin/env python3
"""Generate benchmark answers with the exported LT-OPD runtime."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOLS = json.loads((ROOT / "protocols.json").read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


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
        from .multispan_compat import _multispan_callsite
        from .runtime import build_route_query, canonical_export_binding, load_export_runtime, prediction_record
    else:
        from data import public_row, row_messages, rows_for
        from mme import runtime_prediction_content, runtime_route_query
        from multispan_compat import _multispan_callsite
        from runtime import build_route_query, canonical_export_binding, load_export_runtime, prediction_record

    settings = PROTOCOLS[args.dataset]
    config = json.loads(args.data_config.read_text(encoding="utf-8"))[args.dataset]
    config = {k: str((args.data_config.resolve().parent / v).resolve())
              for k, v in config.items()}
    binding = canonical_export_binding(args.export_dir)
    contract = {"dataset": args.dataset, "settings": settings, "data": config,
                "export": binding, "world_size": args.world_size,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in ROOT.glob("*.py")}}
    identity = digest(contract)
    output_dir = args.output_dir / args.dataset / f"rank{args.rank}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["identity"] != identity:
        raise ValueError("Output directory belongs to a different model, dataset, or shard plan")
    atomic_json(run_path, {"identity": identity, "contract": contract})
    iterator = (mme_rows(config, args.rank, args.world_size) if args.dataset == "mme" else
                rows_for(args.dataset, config, rank=args.rank, world_size=args.world_size))
    runtime = None
    pending = []
    seen = set()
    image_cache = {}
    count = 0

    def flush():
        nonlocal runtime, pending
        if not pending:
            return
        if runtime is None:
            runtime = load_export_runtime(args.export_dir, binding)
        multiple = len(pending[0]["images"]) > 1
        scope = _multispan_callsite(runtime) if multiple else contextlib.nullcontext()
        with scope:
            generated = runtime.generate(
                [p["messages"] for p in pending],
                response_length=settings["official_max_new_tokens"],
                visual_compression_mode=pending[0]["mode"],
                route_queries_batch=[p["query"] for p in pending],
                audit_level="route" if multiple else "none",
            )
        for i, item in enumerate(pending):
            response = prediction_record(runtime, generated, i,
                                         settings["official_max_new_tokens"], settings["until"])
            if args.dataset == "mme":
                response["pred"] = runtime_prediction_content(
                    decoded_prediction=generated["decoded_predictions"][i],
                    sampled_continuation=generated["decoded_raw_continuations"][i])
            record = {"dataset": args.dataset, "source_index": item["row"]["source_index"],
                      "uid": item["row"]["uid"], "row": item["row"], "response": response,
                      "input_sha256": item["input_sha256"], "run_identity": identity}
            record["record_sha256"] = digest(record)
            atomic_json(item["path"], record)
        pending = []

    for row in iterator:
        count += 1
        if row["uid"] in seen:
            raise ValueError(f"Duplicate question ID: {row['uid']}")
        seen.add(row["uid"])
        public = public_row(row)
        messages, images = row_messages(row, image_cache)
        query = runtime_route_query(row["question"]) if args.dataset == "mme" else build_route_query(row["prompt"])
        input_sha = digest({"row": public, "images": images, "query": query})
        target = output_dir / f"{row['source_index']:06d}.json"
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            seal = previous.pop("record_sha256")
            if (digest(previous) != seal or previous["input_sha256"] != input_sha or
                    previous["run_identity"] != identity):
                raise ValueError(f"Incompatible prediction record: {target}")
            continue
        mode = "merge" if images else "no_image"
        multiple = len(images) > 1
        if pending and (multiple or len(pending[0]["images"]) > 1 or pending[0]["mode"] != mode):
            flush()
        pending.append({"row": public, "messages": messages, "images": images, "query": query,
                        "mode": mode, "path": target, "input_sha256": input_sha})
        if multiple or len(pending) == settings["batch_size"]:
            flush()
    flush()
    if args.dataset != "mme":
        expected = len(range(args.rank, settings["expected_count"], args.world_size))
        if count != expected:
            raise ValueError(f"Expected {expected} rows in shard, received {count}")
    atomic_json(output_dir / "complete.json", {"identity": identity, "count": count})
    print(json.dumps({"dataset": args.dataset, "rank": args.rank, "completed": count}))


if __name__ == "__main__":
    main()
