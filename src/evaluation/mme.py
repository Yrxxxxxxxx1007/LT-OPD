"""MME prompt, routing, answer transport, and paired sharding from the V8 evaluator."""
from __future__ import annotations
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

MME_REPLACE_PROMPT = " Please answer yes or no."


MME_POST_PROMPT = "\nAnswer the question using a single word or phrase."


MME_SYSTEM_PROMPT = "You are a helpful assistant."


PERCEPTION_CATEGORIES = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
)


COGNITION_CATEGORIES = (
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)


ALL_CATEGORIES = PERCEPTION_CATEGORIES + COGNITION_CATEGORIES


def semantic_question(question: str) -> str:
    """Return the benchmark question before lmms-eval's default post-prompt."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("MME question must be a non-empty string")
    semantic = question.strip().replace(MME_REPLACE_PROMPT, "")
    if not semantic.strip():
        raise ValueError("MME question became empty after canonical prompt normalization")
    return semantic


def official_prompt(question: str) -> str:
    """Reproduce the current lmms-eval default MME ``doc_to_text`` output."""

    return semantic_question(question) + MME_POST_PROMPT


def runtime_prediction_content(
    *, decoded_prediction: str, sampled_continuation: str | None = None
) -> str:
    """Remove the runtime-owned answer transport before official MME parsing."""

    if not isinstance(decoded_prediction, str):
        raise TypeError("decoded_prediction must be a string")
    if isinstance(sampled_continuation, str):
        content = sampled_continuation.strip()
        if content.endswith("</answer>"):
            content = content[: -len("</answer>")]
        return content.strip()
    content = decoded_prediction.strip()
    if content.startswith("<answer>"):
        content = content[len("<answer>") :]
    if "</answer>" in content:
        content = content.split("</answer>", 1)[0]
    return content.strip()


def canonical_route_query(question: str) -> dict[str, Any]:
    """Build the exact V2 query mapping expected by the exported CDPruner runtime."""

    text = semantic_question(question)
    return {
        "schema_version": "vision_opd_route_query_v2",
        "policy": "user_question_options_only_excluding_special_and_image_tokens",
        "segments": [text],
        "canonical_text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def runtime_route_query(question: str) -> dict[str, Any]:
    """Serialize routing text exactly as the runtime's RouteQuerySpec does.

    The pinned official MME prompt preserves seven source rows with two spaces
    before the removable yes/no suffix.  Removing that suffix leaves one
    insignificant trailing space in the benchmark semantic identity, while
    the exported runtime strips every explicit route segment.  Normalize only
    this routing transport; model prompt bytes, row metadata, record identity,
    answer parsing, and scoring remain unchanged.
    """

    identity = canonical_route_query(question)
    segments = [str(segment).strip() for segment in identity["segments"]]
    if any(not segment for segment in segments):
        raise ValueError("MME runtime route query contains an empty semantic segment")
    canonical_text = "\n".join(segments)
    return {
        "schema_version": identity["schema_version"],
        "policy": identity["policy"],
        "segments": segments,
        "canonical_text": canonical_text,
        "sha256": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
    }


def normalized_answer(answer: str) -> str:
    if not isinstance(answer, str):
        raise TypeError("MME ground-truth answer must be a string")
    value = answer.lower().strip().replace(".", "")
    if value not in {"yes", "no"}:
        raise ValueError(f"MME ground-truth answer must be yes or no, got {answer!r}")
    return value


def row_metadata(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    """Select the non-image bytes that define one pinned MME row."""

    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError("row_index must be a non-negative integer")
    required = ("question_id", "question", "answer", "category")
    missing = [key for key in required if key not in row]
    if missing:
        raise KeyError(f"MME row {row_index} is missing fields: {missing}")
    question_id = str(row["question_id"])
    question = str(row["question"])
    category = str(row["category"])
    if not question_id or not question.strip() or category not in ALL_CATEGORIES:
        raise ValueError(f"MME row {row_index} has malformed identity fields")
    payload = {
        "row_index": row_index,
        "question_id": question_id,
        "question": question,
        "answer": normalized_answer(str(row["answer"])),
        "category": category,
    }
    return payload


def build_shard_plan(rows: Sequence[Mapping[str, Any]], world_size: int) -> list[int]:
    """Assign complete question pairs to balanced deterministic worker ranks."""

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    groups = sorted({(str(row["category"]), str(row["question_id"])) for row in rows})
    group_rank = {group: index % world_size for index, group in enumerate(groups)}
    return [group_rank[(str(row["category"]), str(row["question_id"]))] for row in rows]


