"""Load the exported CDPruner model and preserve its answer transport."""
from __future__ import annotations
import hashlib, pathlib, re
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


def load_export_runtime(export_dir: pathlib.Path):
    from training.runtime import load_cdpruner_runtime
    import torch

    return load_cdpruner_runtime(export_dir, torch_dtype=torch.bfloat16,
                                 device_map={"": "cuda:0"})


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
    for terminator in until:
        if terminator in prediction:
            prediction = prediction.split(terminator, 1)[0]
            break
    return {"pred": prediction.strip()}
