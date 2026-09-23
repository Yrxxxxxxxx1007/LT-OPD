#!/usr/bin/env python3
"""Score saved predictions using the pinned benchmark implementations."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import string
import tempfile

ROOT = Path(__file__).resolve().parent
PROTOCOLS = json.loads((ROOT / "protocols.json").read_text())
SOURCES = json.loads((ROOT / "sources.json").read_text())


def source(root, name):
    path = root / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCES[name]["sha256"]:
        raise ValueError(f"Scoring source differs from the pinned version: {path}")
    return path


def definitions(path, names, namespace=None, assignments=()):
    """Load unchanged upstream AST nodes without their model/API imports."""
    body = []
    found = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in assignments for t in node.targets):
            body.append(node)
    if found != set(names):
        raise ValueError(f"Missing upstream definitions: {set(names) - found}")
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_records(path, dataset, mme_manifest=None):
    if path.is_file():
        raw = path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(raw)
    else:
        records = []
        for item in sorted(path.rglob("*.json")):
            value = json.loads(item.read_text(encoding="utf-8"))
            if isinstance(value, dict) and (("row" in value and "response" in value) or
                                           (dataset == "mme" and "scored_content" in value)):
                records.append(value)
    metadata = None
    if mme_manifest:
        manifest = json.loads(mme_manifest.read_text(encoding="utf-8"))
        metadata = manifest.get("rows", manifest.get("metadata_rows"))
    for row in records:
        if "record_sha256" in row:
            body = {k: v for k, v in row.items() if k != "record_sha256"}
            actual = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if row["record_sha256"] != actual:
                raise ValueError("Prediction record checksum mismatch")
        if "canonical_sha256" in row:
            body = {k: v for k, v in row.items() if k != "canonical_sha256"}
            actual = hashlib.sha256((json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
            if row["canonical_sha256"] != actual:
                raise ValueError("Legacy prediction record checksum mismatch")
        if "source_index" not in row and dataset == "mme":
            # Historical MME records use a separate transport schema.
            if metadata is None:
                raise ValueError("Legacy MME records require --mme-manifest")
            row.update(source_index=row["row_index"], uid=str(row["row_index"]),
                       row={"gold": row["ground_truth"], "category": row["category"],
                            "question_id": row["question_id"],
                            "question": metadata[row["row_index"]]["question"]},
                       response={"pred": row["scored_content"]})
    records.sort(key=lambda r: int(r["source_index"]))
    count = PROTOCOLS[dataset]["expected_count"]
    if len(records) != count or [int(r["source_index"]) for r in records] != list(range(count)):
        raise ValueError(f"Require exactly {count} unique ordered rows for {dataset}")
    if len({str(r["uid"]) for r in records}) != count:
        raise ValueError("Duplicate question IDs")
    identities = {r["run_identity"] for r in records if "run_identity" in r}
    if len(identities) > 1:
        raise ValueError("Predictions from different runs were mixed")
    return records


def choice_frame(rows):
    import pandas as pd
    items = []
    for record in rows:
        item = dict(record["row"])
        item.update(item.pop("options"))
        item["prediction"] = record["response"]["pred"]
        item["answer"] = item["GT"] = item["gold"]
        item["index"] = int(item.get("index", record["uid"]))
        if "L2-category" in item:
            item["l2-category"] = item.pop("L2-category")
        items.append(item)
    return pd.DataFrame(items).sort_values("index").reset_index(drop=True)


def score_mcq(rows, dataset, root):
    import numpy as np
    import pandas as pd
    logger = logging.getLogger("benchmark")
    logger.addHandler(logging.NullHandler())
    namespace = dict(np=np, pd=pd, re=re, os=os, cp=copy, string=string,
                     defaultdict=defaultdict, logger=logger, eval_logger=logger,
                     get_logger=lambda *args: logger)
    if dataset == "vstar":
        definitions(source(root, "vstar.py"),
                    ["extract_answer_letter", "vstar_process_results", "vstar_aggregate_results"], namespace)
        items = [namespace["vstar_process_results"](
            {"label": r["row"]["gold"], "category": r["row"]["category"],
             "text": r["row"]["prompt"], "question_id": r["uid"]},
            [r["response"]["pred"]])["vstar_overall_acc"] for r in rows]
        return {"score": namespace["vstar_aggregate_results"](items), "unit": "percent",
                "protocol": "lmms-eval generated-answer accuracy",
                "parse_failures": sum(x["prediction"] is None for x in items),
                "author_likelihood_protocol": "not measured by generated-answer evaluation"}
    definitions(source(root, "matching.py"), ["can_infer_option", "can_infer_text", "can_infer"],
                namespace, ["_VERBOSE_ANSWER_RE"])
    definitions(source(root, "multiple_choice.py"),
                ["build_choices", "prefetch_answer", "prefetch_circular_group", "report_acc"],
                namespace, ["MMB_abbrs"])
    frame = choice_frame(rows)
    if dataset == "hrbench4k":
        definitions(source(root, "hrbench.py"), ["hrbench_score", "report_acc_hrbench"], namespace)
        predictions = [namespace["can_infer"](str(r["prediction"]),
                       {c: r[c] for c in string.ascii_uppercase if c in r and not pd.isna(r[c])})
                       for _, r in frame.iterrows()]
        pending = [int(frame.iloc[i]["source_index"]) for i, p in enumerate(predictions) if not p]
        if pending:
            return {"state": "pending_judge", "score": None, "unresolved_source_indices": pending}
        frame["hit"] = [int(p == frame.iloc[i]["GT"]) for i, p in enumerate(predictions)]
        table = namespace["report_acc_hrbench"](frame)
        result = float(table[(table["cycle"] == "Average") & (table["type"] == "all")]["accuracy"].iloc[0])
        return {"score": result * 100, "unit": "percent", "protocol": "VLMEvalKit cycle macro",
                "table": table.to_dict(orient="records"), "judge_calls": 0}
    frame["g_index"] = [int(i % 1000000) for i in frame["index"]]
    main = frame[frame["index"] == frame["g_index"]].copy()
    hits, pending = {}, []
    for index in main["index"]:
        result = namespace["prefetch_circular_group"](frame[frame["g_index"] == index], verbose=False)
        if result is None:
            pending.append(int(index))
        else:
            hits[int(index)] = int(result["hit"])
    if pending:
        return {"state": "pending_judge", "score": None, "unresolved_groups": pending}
    main["hit"] = [hits[int(i)] for i in main["index"]]
    table = namespace["report_acc"](main)
    if len(table) != 1:
        raise ValueError("MMBench input contains multiple splits")
    return {"score": float(table["Overall"].iloc[0]) * 100, "unit": "percent",
            "protocol": "VLMEvalKit CircularEval", "correct_groups": sum(hits.values()),
            "groups": len(main), "judge_calls": 0, "table": table.to_dict(orient="records")}


def score_mmmu(rows, root):
    path = source(root, "mmmu.py")
    spec = importlib.util.spec_from_file_location("mmmu_author", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    random.seed(42)
    original_choice = random.choice
    fallbacks, samples = [], []
    active = None

    def counted_choice(values):
        fallbacks.append(active)
        return original_choice(values)

    random.choice = counted_choice
    try:
        for row in rows:
            active = row["uid"]
            doc, pred = row["row"], row["response"]["pred"]
            if doc["question_type"] == "multiple-choice":
                options = ast.literal_eval(doc["options"])
                choices = [chr(65 + i) for i in range(len(options))]
                parsed = module.parse_multi_choice_response(pred, choices, dict(zip(choices, options)))
            else:
                parsed = module.parse_open_response(pred)
            samples.append({"id": active, "question_type": doc["question_type"],
                            "answer": doc["gold"], "parsed_pred": parsed})
    finally:
        random.choice = original_choice
    judgments, metric = module.evaluate(samples)
    return {"score": metric["acc"] * 100, "unit": "percent", "seed": 42,
            "correct": sum(x == "Correct" for x in judgments.values()),
            "random_fallbacks": len(fallbacks), "random_fallback_ids": fallbacks}


def score_gqa(rows, root):
    path = source(root, "gqa.py")
    namespace = definitions(path, ["toScore", "avg"])
    loop = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.For) and
                isinstance(n.target, ast.Tuple) and [x.id for x in n.target.elts] == ["qid", "question"])
    code = compile(ast.Module(body=loop.body[:4], type_ignores=[]), str(path), "exec")
    values = []
    for row in rows:
        scope = dict(namespace, question={"answer": row["row"]["gold"]}, qid=row["uid"],
                     predictions={row["uid"]: row["response"]["pred"]})
        exec(code, scope)
        values.append(scope["score"])
    return {"score": namespace["avg"](values) * 100, "unit": "percent",
            "correct": sum(values), "normalization": "none (author raw exact match)"}


def score_textvqa(rows, root):
    module = definitions(source(root, "textvqa.py"), ["EvalAIAnswerProcessor", "TextVQAAccuracyEvaluator"], {"re": re})
    for row in rows:
        if len(row["row"]["gold"]) != 10:
            raise ValueError("TextVQA requires all ten reference answers")
    value = module["TextVQAAccuracyEvaluator"]().eval_pred_list([
        {"pred_answer": r["response"]["pred"], "gt_answers": r["row"]["gold"]} for r in rows])
    return {"score": value * 100, "unit": "percent", "protocol": "author ten-answer VQA soft accuracy"}


def score_pope(rows, root):
    path = source(root, "pope.py")
    tree = ast.parse(path.read_text())
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.For) and
                 isinstance(n.target, ast.Name) and n.target.id == "answer")
    code = compile(ast.Module(body=tree.body[start:], type_ignores=[]), str(path), "exec")

    def evaluate(items):
        scope = {"answers": [{"answer": r["response"]["pred"]} for r in items],
                 "label_list": [r["row"]["gold"] for r in items]}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, scope)
        return {key: float(scope[name]) for key, name in
                [("accuracy", "acc"), ("precision", "precision"), ("recall", "recall"),
                 ("f1", "f1"), ("yes_ratio", "yes_ratio")]}

    categories = {}
    for category in ("adversarial", "popular", "random"):
        subset = [r for r in rows if r["row"]["category"] == category]
        if len(subset) != 3000:
            raise ValueError("POPE requires 3000 examples per category")
        categories[category] = evaluate(subset)
    macro = {k: sum(v[k] for v in categories.values()) / 3 for k in categories["random"]}
    return {"score": macro["accuracy"] * 100, "unit": "percent", "macro": macro,
            "per_category": categories, "pooled": evaluate(rows)}


def score_ocrbench(rows, root):
    path = source(root, "ocrbench.py")
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare))
    loops = [n for n in main.body if isinstance(n, ast.For) and "data_type = data[i]['type']" in ast.unparse(n)]
    if len(loops) != 1:
        raise ValueError("Unexpected author OCRBench scoring loop")
    data = [{"type": r["row"]["question_type"], "dataset_name": r["row"]["dataset_name"],
             "answers": r["row"]["gold"], "predict": r["response"]["pred"]} for r in rows]
    exec(compile(ast.Module(body=loops, type_ignores=[]), str(path), "exec"), {"data": data})
    return {"score": sum(x["result"] for x in data), "unit": "points/1000"}


def score_mme(rows, root):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    scope = dict(os=os, accuracy_score=accuracy_score, precision_score=precision_score,
                 recall_score=recall_score, confusion_matrix=confusion_matrix)
    definitions(source(root, "mme.py"), ["calculate_metrics"], scope, ["eval_type_dict"])
    evaluator = scope["calculate_metrics"]()
    groups = defaultdict(list)
    for row in rows:
        groups[(row["row"]["category"], str(row["row"]["question_id"]))].append(row)
    if len(groups) != 1187 or any(len(v) != 2 for v in groups.values()):
        raise ValueError("MME requires 1187 complete question pairs")
    with tempfile.TemporaryDirectory(prefix="lt_opd_mme_") as directory:
        for category in sum(scope["eval_type_dict"].values(), []):
            lines = []
            for (cat, qid), items in sorted(groups.items()):
                if cat != category:
                    continue
                for row in sorted(items, key=lambda r: r["source_index"]):
                    pred = row["response"]["pred"]
                    transported = pred.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                    if evaluator.parse_pred_ans((pred + "\n").lower()) != evaluator.parse_pred_ans((transported + "\n").lower()):
                        raise ValueError("MME TSV transport changes answer parsing")
                    question = row["row"]["question"]
                    if any(c in question for c in "\t\r\n"):
                        raise ValueError("MME question cannot contain TSV delimiters")
                    lines.append(f"{qid}\t{question}\t{row['row']['gold']}\t{transported}\n")
            (Path(directory) / f"{category}.txt").write_text("".join(lines), encoding="utf-8")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            evaluator.process_result(directory)
    totals = [float(x) for x in re.findall(r"total score:\s*([0-9.]+)", stdout.getvalue())]
    if len(totals) != 2:
        raise ValueError("Unexpected author MME output")
    return {"score": sum(totals), "perception": totals[0], "cognition": totals[1], "unit": "points"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="One dataset's prediction directory or JSONL")
    parser.add_argument("--sources", type=Path, required=True, help="Directory populated by fetch_scorers.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mme-manifest", type=Path, help="Dataset manifest for legacy MME records only")
    args = parser.parse_args()
    rows = load_records(args.predictions, args.dataset, args.mme_manifest)
    if args.dataset in ("vstar", "hrbench4k", "mmbench_circular"):
        result = score_mcq(rows, args.dataset, args.sources)
    else:
        function = {"mmmu": score_mmmu, "gqa": score_gqa, "textvqa": score_textvqa,
                    "pope": score_pope, "ocrbench": score_ocrbench, "mme": score_mme}[args.dataset]
        result = function(rows, args.sources)
    result = {"dataset": args.dataset, "count": len(rows), "state": "complete", **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
