"""Question extraction for conditional visual pruning.

Routing uses the question and any answer options, excluding prompt formatting,
image placeholders, and assistant text.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


# V1 supports A-D multiple-choice queries. V2 supports a question with any
# number of options. Both schemas use only the user's question text.
ROUTE_QUERY_SCHEMA_VERSION_V1 = "vision_opd_route_query_v1"
ROUTE_QUERY_SCHEMA_VERSION_V2 = "vision_opd_route_query_v2"
ROUTE_QUERY_SCHEMA_VERSION = ROUTE_QUERY_SCHEMA_VERSION_V1
SUPPORTED_ROUTE_QUERY_SCHEMAS = frozenset(
    {ROUTE_QUERY_SCHEMA_VERSION_V1, ROUTE_QUERY_SCHEMA_VERSION_V2}
)
ROUTE_QUERY_POLICY = "user_question_options_only_excluding_special_and_image_tokens"
VISION_OPD_RED_BOX_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)

_OPTION_LINE_PATTERN = re.compile(r"(?m)^[ \t]*([A-D])\.[ \t]*(.*?)[ \t]*(?=\r?$)")
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"(?i)<image>")


@dataclass(frozen=True)
class RouteQuerySpec:
    """Exact semantic text segments that are allowed to influence routing."""

    segments: tuple[str, ...]
    schema_version: str = ROUTE_QUERY_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_ROUTE_QUERY_SCHEMAS:
            raise ValueError(f"Unsupported route-query schema_version={self.schema_version!r}")
        if self.schema_version == ROUTE_QUERY_SCHEMA_VERSION_V1 and len(self.segments) != 5:
            raise ValueError("A V1 route query must contain one question segment and options A-D")
        if self.schema_version == ROUTE_QUERY_SCHEMA_VERSION_V2 and len(self.segments) < 1:
            raise ValueError("A V2 route query must contain at least the question segment")
        if any(not isinstance(segment, str) or not segment.strip() for segment in self.segments):
            raise ValueError("Route-query segments must be non-empty strings")

    @property
    def canonical_text(self) -> str:
        return "\n".join(segment.strip() for segment in self.segments)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_text.encode("utf-8")).hexdigest()

    def as_data_field(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": ROUTE_QUERY_POLICY,
            "segments": list(self.segments),
            "canonical_text": self.canonical_text,
            "sha256": self.sha256,
        }


def build_general_route_query(question: str, options: Sequence[str] = ()) -> RouteQuerySpec:
    """Build the V2 query used by mixed MCQ/open/OCR/numerical data.

    The caller supplies already-normalized source text.  Keeping each option as
    one explicit segment avoids parsing labels or guessing task semantics.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("General route query requires a non-empty question")
    if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
        raise TypeError("General route-query options must be a sequence of strings")
    normalized_options = tuple(str(option).strip() for option in options)
    if any(not option for option in normalized_options):
        raise ValueError("General route-query options must be non-empty strings")
    return RouteQuerySpec(
        (question.strip(), *normalized_options),
        schema_version=ROUTE_QUERY_SCHEMA_VERSION_V2,
    )


def _question_prefix(text: str, first_option_start: int, *, prompt_fallback: bool) -> str:
    prefix = _IMAGE_PLACEHOLDER_PATTERN.sub("", text[:first_option_start])
    hint_start = prefix.find(VISION_OPD_RED_BOX_HINT)
    if hint_start >= 0:
        after_hint = prefix[hint_start + len(VISION_OPD_RED_BOX_HINT) :]
        if after_hint.strip():
            raise ValueError("Unexpected text appears between the red-box template and option A")
        prefix = prefix[:hint_start]
    elif prompt_fallback and "red bounding box" in prefix.casefold():
        # Do not guess around a drifted instruction: a partial/translated
        # template could otherwise leak boilerplate into the route query.
        raise ValueError("Unrecognized red-box instruction in prompt fallback")
    question = prefix.strip()
    if not question:
        raise ValueError("Route query has an empty question stem")
    return question


def parse_route_query_text(text: str, *, prompt_fallback: bool = False) -> RouteQuerySpec:
    """Parse one question plus exact option lines A-D, discarding later text.

    Stopping at the end of option D intentionally excludes answer-format
    instructions.  When parsing a complete prompt fallback, the recognized
    Vision-OPD red-box sentence is also removed.  Ambiguous or incomplete MCQ
    layouts fail closed instead of selecting the whole user message.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Route query text must be a non-empty string")
    matches = list(_OPTION_LINE_PATTERN.finditer(text))
    labels = [match.group(1) for match in matches]
    if labels != ["A", "B", "C", "D"]:
        raise ValueError(f"Route query requires exactly one ordered A-D option set, got {labels}")

    question = _question_prefix(text, matches[0].start(), prompt_fallback=prompt_fallback)
    option_segments = []
    for expected_label, match in zip(("A", "B", "C", "D"), matches, strict=True):
        value = match.group(2).strip()
        if not value:
            raise ValueError(f"Route-query option {expected_label} is empty")
        option_segments.append(f"{expected_label}. {value}")
    return RouteQuerySpec((question, *option_segments))


def parse_route_query_value(value: Any) -> RouteQuerySpec:
    """Parse a serialized explicit ``route_query`` value.

    The canonical data form is a mapping produced by ``RouteQuerySpec``.  A
    plain string and a ``question``/``options`` mapping remain accepted for
    compatibility with hand-authored datasets.
    """

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, Mapping)):
        value = value.tolist()
    if isinstance(value, str):
        return parse_route_query_text(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"route_query must be text or a mapping, got {type(value)}")

    schema_version = value.get("schema_version")
    if schema_version not in (None, *SUPPORTED_ROUTE_QUERY_SCHEMAS):
        raise ValueError(f"Unsupported route-query schema_version={schema_version!r}")
    policy = value.get("policy")
    if policy not in (None, ROUTE_QUERY_POLICY):
        raise ValueError(f"Unsupported route-query policy={policy!r}")

    segments = value.get("segments")
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
        raw_segments = tuple(str(segment).strip() for segment in segments)
        raw_canonical = "\n".join(raw_segments)
        if schema_version == ROUTE_QUERY_SCHEMA_VERSION_V2:
            spec = RouteQuerySpec(
                raw_segments,
                schema_version=ROUTE_QUERY_SCHEMA_VERSION_V2,
            )
        else:
            spec = parse_route_query_text(raw_canonical)
        if spec.canonical_text != raw_canonical:
            raise ValueError("route_query segments do not reconstruct canonical semantic text")
    elif isinstance(value.get("canonical_text"), str):
        if schema_version == ROUTE_QUERY_SCHEMA_VERSION_V2:
            raise ValueError("V2 route_query requires explicit segments; canonical text alone is ambiguous")
        spec = parse_route_query_text(value["canonical_text"])
    elif isinstance(value.get("text"), str):
        if schema_version == ROUTE_QUERY_SCHEMA_VERSION_V2:
            raise ValueError("V2 route_query requires explicit segments; free text is ambiguous")
        spec = parse_route_query_text(value["text"])
    else:
        question = value.get("question")
        options = value.get("options")
        if not isinstance(question, str):
            raise ValueError("route_query mapping requires segments, text, or question+options")
        if schema_version == ROUTE_QUERY_SCHEMA_VERSION_V2:
            if options is None:
                options = ()
            if isinstance(options, Mapping):
                option_sequence = tuple(str(item) for item in options.values())
            elif isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
                option_sequence = tuple(str(item) for item in options)
            else:
                raise ValueError("V2 route_query.options must be a mapping or sequence")
            spec = build_general_route_query(question, option_sequence)
        elif isinstance(options, Mapping):
            if set(options) != {"A", "B", "C", "D"}:
                raise ValueError("route_query.options must contain exactly A-D")
            option_values = {label: options[label] for label in ("A", "B", "C", "D")}
            spec = parse_route_query_text(
                "\n".join([question, *(f"{label}. {option_values[label]}" for label in ("A", "B", "C", "D"))])
            )
        elif isinstance(options, Sequence) and not isinstance(options, (str, bytes)) and len(options) == 4:
            option_values = dict(zip(("A", "B", "C", "D"), options, strict=True))
            spec = parse_route_query_text(
                "\n".join([question, *(f"{label}. {option_values[label]}" for label in ("A", "B", "C", "D"))])
            )
        else:
            raise ValueError("route_query.options must be an A-D mapping or four-item sequence")

    expected_text = value.get("canonical_text")
    if expected_text is not None and expected_text != spec.canonical_text:
        raise ValueError("route_query canonical_text does not match its semantic segments")
    expected_hash = value.get("sha256")
    if expected_hash is not None and expected_hash != spec.sha256:
        raise ValueError("route_query sha256 does not match its canonical text")
    return spec


def user_text_for_prompt_fallback(messages: Any) -> str:
    """Collect user-authored text while ignoring image/video content items."""

    if not isinstance(messages, list):
        raise ValueError("Prompt fallback requires structured chat messages")
    user_chunks: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            user_chunks.append(content)
            continue
        if not isinstance(content, list):
            raise ValueError("User message content must be text or a multimodal content list")
        for item in content:
            if isinstance(item, str):
                user_chunks.append(item)
            elif isinstance(item, Mapping) and item.get("type") == "text":
                text = item.get("text")
                if not isinstance(text, str):
                    raise ValueError("Text content items require a string 'text' field")
                user_chunks.append(text)
    if not user_chunks:
        raise ValueError("Prompt fallback contains no user text")
    return "\n".join(user_chunks)
