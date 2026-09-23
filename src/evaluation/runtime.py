"""Load the exported CDPruner model and preserve its answer transport."""
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, re
from typing import Any

GENERATION_SUFFIXES = (
    "\nAnswer with the option's letter from the given choices directly.",
    "\nAnswer the option letter directly.",
    "\nAnswer with the option letter only.",
    "\nPlease answer the question directly.",
    "\nAnswer the question using a single word or phrase.",
)


ROUTE_QUERY_SCHEMA = "vision_opd_route_query_v2"


ROUTE_QUERY_POLICY = "user_question_options_only_excluding_special_and_image_tokens"


_OPTION_LINE = re.compile(r"(?m)^[ \t]*([A-Z])\.[ \t]*(.*?)[ \t]*(?=\r?$)")


_NUMBERED_IMAGE = re.compile(r"(?i)<image[ \t]+\d+>")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_export_binding(export_dir: pathlib.Path) -> dict[str, Any]:
    """Bind the export without recursively hashing model/runtime/data trees."""

    export_dir = export_dir.resolve(strict=True)
    choices = (
        ("load_v6_dpc_runtime.py", "load_v6_dpc_runtime"),
        ("load_cdpruner_runtime.py", "load_cdpruner_runtime"),
    )
    available = [(name, entry) for name, entry in choices if (export_dir / name).is_file()]
    if not available:
        raise FileNotFoundError(
            "export has neither load_v6_dpc_runtime.py nor load_cdpruner_runtime.py"
        )
    contract_names = ("v6_dpc_export_contract.json", "cdpruner_export_contract.json")
    contract_paths = [export_dir / name for name in contract_names if (export_dir / name).is_file()]
    if len(contract_paths) != 1:
        raise RuntimeError(f"expected one export contract, found {contract_paths}")
    contract_path = contract_paths[0]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    deployment = contract.get("deployment")
    if not isinstance(deployment, dict):
        raise RuntimeError("export contract has no deployment binding")
    authorized = deployment.get("authorized_loader")
    if (
        not isinstance(authorized, str)
        or pathlib.PurePath(authorized).name != authorized
    ):
        raise RuntimeError("export contract has no safe authorized loader name")
    selected = next((item for item in available if item[0] == authorized), None)
    if selected is None:
        raise RuntimeError(
            f"authorized loader is unavailable or unsupported: {authorized}"
        )
    loader_path = export_dir / selected[0]
    loader_sha = sha256_file(loader_path)
    artifact_bindings = contract.get("artifact_bindings")
    expected_loader_sha = (
        artifact_bindings.get("export_loader_sha256")
        if isinstance(artifact_bindings, dict)
        else None
    )
    if (
        not isinstance(expected_loader_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_loader_sha) is None
    ):
        raise RuntimeError("export contract has no valid loader SHA-256 binding")
    if loader_sha != expected_loader_sha:
        raise RuntimeError(
            "authorized loader differs from artifact_bindings.export_loader_sha256"
        )
    compressor = contract.get("vision_token_compressor", {})
    if compressor.get("algorithm") != "qwen35_cdpruner_v1":
        raise RuntimeError("export is not a CDPruner actor")
    minimum = compressor.get("minimum_tokens_per_image")
    if int(minimum if minimum is not None else -1) != 32:
        raise RuntimeError("export does not preserve the minimum-32 deployment budget")
    schema = str(contract.get("schema_version", ""))
    if "v8" in schema:
        if contract.get("checkpoint_step") != 175:
            raise RuntimeError("V8 exact-nine evaluation requires the final step-175 export")
        if contract.get("artifact_role") != "final_deployment":
            raise RuntimeError("V8 exact-nine evaluation refuses a diagnostic-only export")
        state = contract.get("evaluation_curriculum_state", {})
        if int(state.get("retention_bps", -1)) != 500:
            raise RuntimeError("V8 export is not bound to the final 5% evaluation state")
    provenance = export_dir / "checkpoint_provenance.json"
    if not provenance.is_file():
        raise FileNotFoundError(provenance)
    return {
        "export_dir": str(export_dir),
        "loader": selected[0],
        "loader_sha256": loader_sha,
        "entrypoint": selected[1],
        "contract": contract_path.name,
        "contract_sha256": sha256_file(contract_path),
        "checkpoint_provenance_sha256": sha256_file(provenance),
        "schema_version": schema,
        "checkpoint_step": contract.get("checkpoint_step"),
        "artifact_role": contract.get("artifact_role"),
        "algorithm": compressor.get("algorithm"),
    }


def load_export_runtime(export_dir: pathlib.Path, binding: dict[str, Any]):
    """Load either supported export spelling through its authorized entry point."""

    current = canonical_export_binding(export_dir)
    if current != binding:
        raise RuntimeError("export binding changed before runtime load")
    loader = export_dir.resolve() / binding["loader"]
    spec = importlib.util.spec_from_file_location("v8_exact9_authorized_loader", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import export loader: {loader}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entrypoint = getattr(module, binding["entrypoint"], None)
    if not callable(entrypoint):
        raise RuntimeError(f"loader has no callable {binding['entrypoint']}")
    import torch

    return entrypoint(
        export_dir.resolve(),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )


def build_route_query(prompt: str) -> dict[str, Any]:
    """Build the exact V2 semantic query and exclude only pinned decode suffixes."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("official prompt must be non-empty")
    matches = [suffix for suffix in GENERATION_SUFFIXES if prompt.endswith(suffix)]
    if len(matches) > 1:
        raise RuntimeError("prompt matched more than one generation suffix")
    core = prompt[: -len(matches[0])] if matches else prompt
    core = core.strip()
    if not core:
        raise ValueError("empty semantic query after suffix removal")

    # Match the training query granularity for byte-exact, ordinary A-D MCQs.
    # Fall back to one exact semantic-core segment for E+ options, MMMU image
    # references, open QA, or any ambiguous/non-unique segmentation.
    segments: tuple[str, ...] = (core,)
    option_matches = list(_OPTION_LINE.finditer(core))
    if (
        [match.group(1) for match in option_matches] == list("ABCD")
        and not _NUMBERED_IMAGE.search(core)
    ):
        candidate = (core[: option_matches[0].start()].strip(),) + tuple(
            match.group(0).strip() for match in option_matches
        )
        cursor = -1
        exact = bool(candidate[0])
        for segment in candidate:
            positions = [match.start() for match in re.finditer(re.escape(segment), core)]
            if len(positions) != 1 or positions[0] <= cursor:
                exact = False
                break
            cursor = positions[0]
        if exact:
            segments = candidate

    canonical = "\n".join(segment.strip() for segment in segments)
    return {
        "schema_version": ROUTE_QUERY_SCHEMA,
        "policy": ROUTE_QUERY_POLICY,
        "segments": list(segments),
        "canonical_text": canonical,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def prediction_record(runtime, output: dict[str, Any], index: int, cap: int, until: list[str]):
    """Extract the scored answer from V8's prompt-owned opening-tag transport."""

    mask = output["response_mask"][index].to(dtype=__import__("torch").bool)
    ids = output["responses"][index][mask].detach().cpu().tolist()
    raw_rows = output.get("decoded_raw_continuations")
    reconstructed_rows = output.get("decoded_predictions")
    statuses = output.get("protocol_status")
    status = str(statuses[index]) if statuses is not None else None
    raw = str(raw_rows[index]) if raw_rows is not None else ""
    reconstructed = str(reconstructed_rows[index]) if reconstructed_rows is not None else raw
    if status == "valid" and reconstructed.startswith("<answer>") and reconstructed.endswith("</answer>"):
        prediction = reconstructed[len("<answer>") : -len("</answer>")].strip()
    else:
        prediction = raw.split("</answer>", 1)[0].strip()
        if not prediction:
            prediction = reconstructed.strip()
    matched_until = None
    for terminator in until:
        if terminator in prediction:
            prediction = prediction.split(terminator, 1)[0]
            matched_until = terminator
            break
    reasons = output.get("rollout_stop_reason")
    reason = str(reasons[index]) if reasons is not None else None
    return {
        "pred": prediction.strip(),
        "raw_continuation": raw,
        "protocol_status": status,
        "rollout_stop_reason": reason,
        "output_token_ids": [int(value) for value in ids],
        "generated_token_count": len(ids),
        "max_new_tokens": int(cap),
        "cap_hit": reason == "length" or (len(ids) == int(cap) and status != "valid"),
        "matched_until": matched_until,
        "until_strings": list(until),
    }
