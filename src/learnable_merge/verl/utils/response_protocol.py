"""Single-answer transport helpers shared by rollout, logging, and evaluation.

The response protocol places the opening ``<answer>`` delimiter in the assistant prefill.
It is therefore context, not a sampled action.  Any consumer that presents or
scores the assistant response must reconstruct that transport prefix exactly
once before applying the strict whole-response protocol.
"""

from __future__ import annotations

from dataclasses import dataclass


ANSWER_TRANSPORT_PREFIX = "<answer>"
ANSWER_TRANSPORT_SUFFIX = "</answer>"

PROTOCOL_VALID = "valid"
PROTOCOL_EMPTY = "empty"
PROTOCOL_NESTED_OPEN = "nested_open"
PROTOCOL_MULTIPLE_CLOSE = "multiple_close"
PROTOCOL_TRAILING_TEXT = "trailing_text"
PROTOCOL_MISSING_CLOSE = "missing_close"
PROTOCOL_MISSING_OPEN = "missing_open"

PROTOCOL_STATUSES = (
    PROTOCOL_VALID,
    PROTOCOL_EMPTY,
    PROTOCOL_NESTED_OPEN,
    PROTOCOL_MULTIPLE_CLOSE,
    PROTOCOL_TRAILING_TEXT,
    PROTOCOL_MISSING_CLOSE,
    PROTOCOL_MISSING_OPEN,
)


@dataclass(frozen=True)
class AnswerProtocolResult:
    reconstructed_text: str
    status: str
    answer: str | None

    @property
    def strict_valid(self) -> bool:
        return self.status == PROTOCOL_VALID


def reconstruct_sampled_response(
    sampled_text: str,
    *,
    virtual_open: bool,
    transport_prefix: str = ANSWER_TRANSPORT_PREFIX,
) -> str:
    """Restore prompt-owned transport bytes without normalizing model output."""

    if not isinstance(sampled_text, str):
        raise TypeError("sampled_text must be a string")
    if not isinstance(transport_prefix, str) or not transport_prefix:
        raise ValueError("transport_prefix must be a non-empty string")
    return transport_prefix + sampled_text if virtual_open else sampled_text


def classify_answer_protocol(
    sampled_text: str,
    *,
    virtual_open: bool,
    transport_prefix: str = ANSWER_TRANSPORT_PREFIX,
    transport_suffix: str = ANSWER_TRANSPORT_SUFFIX,
) -> AnswerProtocolResult:
    """Classify the exact, case-sensitive, whole-response transport contract."""

    text = reconstruct_sampled_response(
        sampled_text,
        virtual_open=virtual_open,
        transport_prefix=transport_prefix,
    )
    if not text.startswith(transport_prefix):
        return AnswerProtocolResult(text, PROTOCOL_MISSING_OPEN, None)
    if text.count(transport_prefix) != 1:
        return AnswerProtocolResult(text, PROTOCOL_NESTED_OPEN, None)
    close_count = text.count(transport_suffix)
    if close_count == 0:
        return AnswerProtocolResult(text, PROTOCOL_MISSING_CLOSE, None)
    if close_count != 1:
        return AnswerProtocolResult(text, PROTOCOL_MULTIPLE_CLOSE, None)
    if not text.endswith(transport_suffix):
        return AnswerProtocolResult(text, PROTOCOL_TRAILING_TEXT, None)
    close_start = len(text) - len(transport_suffix)
    content = text[len(transport_prefix) : close_start]
    if not content.strip():
        return AnswerProtocolResult(text, PROTOCOL_EMPTY, None)
    return AnswerProtocolResult(text, PROTOCOL_VALID, content.strip())


def strict_answer_or_none(
    sampled_text: str,
    *,
    virtual_open: bool,
    transport_prefix: str = ANSWER_TRANSPORT_PREFIX,
    transport_suffix: str = ANSWER_TRANSPORT_SUFFIX,
) -> str | None:
    """Return content only for an exact strict transport response."""

    return classify_answer_protocol(
        sampled_text,
        virtual_open=virtual_open,
        transport_prefix=transport_prefix,
        transport_suffix=transport_suffix,
    ).answer
