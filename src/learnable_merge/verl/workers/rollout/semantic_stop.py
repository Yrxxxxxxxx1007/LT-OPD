"""Per-row answer-protocol stopping with explicit sampled-action masks."""

from __future__ import annotations

from typing import Any

import torch
from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteria

from verl.utils.response_protocol import (
    ANSWER_TRANSPORT_PREFIX,
    ANSWER_TRANSPORT_SUFFIX,
    PROTOCOL_STATUSES,
    classify_answer_protocol,
)


STOP_REASON_LENGTH = 0
STOP_REASON_TOKEN_EOS = 1
STOP_REASON_ANSWER_TAG = 2

_ANSWER_BEFORE_OPEN = 0
_ANSWER_OPEN = 1
_ANSWER_INVALID = 2
_ANSWER_FINISHED = 3


def _encoded_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Semantic-stop tokenizer unexpectedly returned a batch")
        ids = ids[0]
    result = [int(token_id) for token_id in ids]
    if not result:
        raise ValueError(f"Semantic-stop delimiter tokenized to an empty sequence: {text!r}")
    return result


class _StableVocabStopStringCriteria(StopStringCriteria):
    """Reuse Transformers' lookup tables when get_vocab() changes dict order."""

    def clean_and_embed_tokens_with_cache(self, token_list, token_indices, tokenizer):
        # Fast tokenizers may enumerate the same vocabulary in a different order
        # on every call. Canonicalize only the cache key; the lookup construction
        # and token-level matching remain the upstream implementation.
        pairs = sorted(zip(token_indices, token_list, strict=True))
        return super().clean_and_embed_tokens_with_cache(
            tuple(token for _, token in pairs), tuple(index for index, _ in pairs), tokenizer
        )


class WellFormedAnswerTagCriteria(StoppingCriteria):
    """Stop after a non-empty ``<answer>...</answer>`` in generated tokens.

    The prompt itself contains the protocol delimiters, so matching is always
    restricted to the response suffix. The state is per row and the closing
    delimiter remains part of the sampled sequence.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_width: int = 0,
        open_string: str = "<answer>",
        close_string: str = "</answer>",
        virtual_open: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(prompt_width, bool) or int(prompt_width) < 0:
            raise ValueError("semantic-stop prompt_width must be non-negative")
        self.prompt_width = int(prompt_width)
        self.open_string = open_string
        self.close_string = close_string
        self.virtual_open = bool(virtual_open)
        if self.virtual_open and (
            self.open_string != ANSWER_TRANSPORT_PREFIX
            or self.close_string != ANSWER_TRANSPORT_SUFFIX
        ):
            raise ValueError("virtual-open semantic stop requires the exact answer transport delimiters")
        self._tokenizer = tokenizer
        self.open_token_ids = _encoded_ids(tokenizer, open_string)
        self.close_token_ids = _encoded_ids(tokenizer, close_string)
        self._open = _StableVocabStopStringCriteria(tokenizer, [open_string])
        self._close = _StableVocabStopStringCriteria(tokenizer, [close_string])
        self._seen_open: torch.Tensor | None = None
        self._open_end_length: torch.Tensor | None = None
        self._answer_state: torch.Tensor | None = None
        self.finished_lengths: torch.Tensor | None = None
        self.protocol_status_codes: torch.Tensor | None = None

    def reset(self) -> None:
        self._seen_open = None
        self._open_end_length = None
        self._answer_state = None
        self.finished_lengths = None
        self.protocol_status_codes = None

    def _ensure_state(self, batch_size: int, device: torch.device) -> None:
        if self._seen_open is None:
            self._seen_open = torch.full(
                (batch_size,), self.virtual_open, dtype=torch.bool, device=device
            )
            self._open_end_length = torch.full(
                (batch_size,), 0 if self.virtual_open else -1, dtype=torch.long, device=device
            )
            self._answer_state = torch.full(
                (batch_size,),
                _ANSWER_OPEN if self.virtual_open else _ANSWER_BEFORE_OPEN,
                dtype=torch.int8,
                device=device,
            )
            self.finished_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            self.protocol_status_codes = torch.full(
                (batch_size,), -1, dtype=torch.int8, device=device
            )
        elif (
            self._seen_open.numel() != batch_size
            or self._seen_open.device != device
            or self._answer_state is None
            or self._answer_state.numel() != batch_size
            or self._answer_state.device != device
        ):
            raise RuntimeError("Semantic-stop batch identity changed during one decode")

    def _decode_response(self, token_ids: torch.Tensor) -> str:
        values = token_ids.detach().cpu().tolist()
        try:
            return self._tokenizer.decode(
                values,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self._tokenizer.decode(values)

    def _strict_complete_answer(self, token_ids: torch.Tensor) -> bool:
        """Accept one exact suffix block under the tokenizer's actual decode."""

        text = self._decode_response(token_ids)
        if not text.endswith(self.close_string):
            return False
        if text.count(self.open_string) != 1 or text.count(self.close_string) != 1:
            return False
        open_start = text.find(self.open_string)
        close_start = text.find(self.close_string)
        if open_start != 0 or close_start < open_start + len(self.open_string):
            return False
        content = text[open_start + len(self.open_string) : close_start]
        return bool(content.strip())

    def update_candidates(
        self,
        response_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance tensor state and return prior finishes plus close candidates.

        This method performs no host transfer.  The cached HF loop can
        copy candidate bits together with its already-required all-finished
        synchronization, then call :meth:`confirm_candidates` only for the
        rare matching rows.
        """

        if response_ids.ndim != 2:
            raise ValueError("Semantic-stop response_ids must be rank-2")
        batch_size, response_length = response_ids.shape
        self._ensure_state(batch_size, response_ids.device)
        if response_length == 0:
            empty = torch.zeros(batch_size, dtype=torch.bool, device=response_ids.device)
            return empty, empty
        scores = torch.empty((batch_size, 0), dtype=torch.float32, device=response_ids.device)
        open_hit = self._open(response_ids, scores).to(torch.bool)
        close_hit = self._close(response_ids, scores).to(torch.bool)
        active = (self._answer_state != _ANSWER_INVALID) & (
            self._answer_state != _ANSWER_FINISHED
        )
        first_open = open_hit & active & (self._answer_state == _ANSWER_BEFORE_OPEN)
        repeated_open = open_hit & active & ~first_open
        # Boolean advanced assignment lowers through aten::nonzero on CUDA.
        # Keep the per-token state transition fixed-shape so the cached loop's
        # explicit all-finished copy remains its only routine host sync.
        answer_state = torch.where(
            first_open,
            torch.full_like(self._answer_state, _ANSWER_OPEN),
            self._answer_state,
        )
        if not self.virtual_open:
            answer_state = torch.where(
                repeated_open,
                torch.full_like(answer_state, _ANSWER_INVALID),
                answer_state,
            )
        self._seen_open = self._seen_open | first_open
        self._open_end_length = torch.where(
            first_open,
            torch.full_like(self._open_end_length, int(response_length)),
            self._open_end_length,
        )

        close_candidate = close_hit & (answer_state == _ANSWER_OPEN)
        if self.virtual_open:
            # Termination and protocol validity are deliberately separate.  A
            # malformed/empty answer is still stopped at its first close and is
            # recorded as malformed, instead of wasting tokens until the hard
            # length cap.  This changes neither sampled tokens nor the loss.
            self._answer_state = answer_state
        else:
            close_active = close_hit & (answer_state != _ANSWER_FINISHED)
            self._answer_state = torch.where(
                close_active,
                torch.full_like(answer_state, _ANSWER_INVALID),
                answer_state,
            )
        return self.finished_lengths > 0, close_candidate

    def confirm_candidates(
        self,
        response_ids: torch.Tensor,
        candidate_rows: list[int] | tuple[int, ...],
    ) -> list[int]:
        """Decode and confirm caller-synchronized close-candidate rows."""

        if (
            response_ids.ndim != 2
            or self._answer_state is None
            or self.finished_lengths is None
            or self.protocol_status_codes is None
        ):
            raise RuntimeError("Semantic-stop candidates were confirmed before state initialization")
        rows = [int(row) for row in candidate_rows]
        if rows != sorted(set(rows)) or any(row < 0 or row >= response_ids.shape[0] for row in rows):
            raise ValueError("Semantic-stop candidate rows must be unique, sorted and batch aligned")
        confirmed: list[int] = []
        for row in rows:
            if self.virtual_open:
                decoded = self._decode_response(response_ids[row])
                # ``StopStringCriteria`` may match a close whose final bytes
                # share one atomic tokenizer token with trailing text (for
                # example, Qwen tokenizes ``</answer>x`` with a final ``>x``
                # token).  The caller has already proved that this row is a
                # suffix candidate at the current token boundary, so require
                # the exact delimiter to be present without pretending that a
                # sampled token can be split after ``>``.  Strict protocol
                # classification below deliberately retains the trailing text
                # and records it as malformed.
                if self.close_string not in decoded:
                    continue
                protocol = classify_answer_protocol(decoded, virtual_open=True)
                self.protocol_status_codes[row] = PROTOCOL_STATUSES.index(protocol.status)
                self._answer_state[row] = _ANSWER_FINISHED
                self.finished_lengths[row] = int(response_ids.shape[1])
                confirmed.append(row)
            elif self._strict_complete_answer(response_ids[row]):
                protocol = classify_answer_protocol(
                    self._decode_response(response_ids[row]), virtual_open=False
                )
                self.protocol_status_codes[row] = PROTOCOL_STATUSES.index(protocol.status)
                self._answer_state[row] = _ANSWER_FINISHED
                self.finished_lengths[row] = int(response_ids.shape[1])
                confirmed.append(row)
        return confirmed

    def protocol_status_names(self) -> list[str]:
        if self.protocol_status_codes is None:
            raise RuntimeError("Semantic-stop protocol status requested before initialization")
        result = []
        for raw in self.protocol_status_codes.detach().cpu().tolist():
            code = int(raw)
            result.append(PROTOCOL_STATUSES[code] if 0 <= code < len(PROTOCOL_STATUSES) else "not_closed")
        return result

    def update(self, response_ids: torch.Tensor) -> torch.Tensor:
        finished, close_candidate = self.update_candidates(response_ids)
        # Generic Transformers generation has no exposed synchronization hook,
        # so keep a correctness-first fallback here.  The DPC cached
        # loop calls the split API and folds these bits into its existing sync.
        candidate_rows = (
            torch.nonzero(close_candidate, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        self.confirm_candidates(response_ids, candidate_rows)
        return finished | (self.finished_lengths > 0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        if input_ids.ndim != 2 or input_ids.shape[-1] < self.prompt_width:
            raise ValueError("Semantic-stop generation input is shorter than its bound prompt")
        return self.update(input_ids[:, self.prompt_width :])


def response_action_mask(
    response_ids: torch.Tensor,
    *,
    eos_token_ids: int | list[int] | tuple[int, ...] | None,
    semantic_finished_lengths: torch.Tensor | None = None,
    dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return explicit valid-action mask and one stop-reason code per row."""

    if response_ids.ndim != 2:
        raise ValueError("response_ids must be rank-2")
    batch_size, width = response_ids.shape
    device = response_ids.device
    lengths = torch.full((batch_size,), width, dtype=torch.long, device=device)
    reasons = torch.full((batch_size,), STOP_REASON_LENGTH, dtype=torch.long, device=device)
    if eos_token_ids is not None:
        eos = torch.as_tensor(eos_token_ids, dtype=response_ids.dtype, device=device).reshape(-1)
        if eos.numel():
            hits = torch.isin(response_ids, eos)
            positions = torch.arange(width, device=device).unsqueeze(0).expand(batch_size, -1)
            first = torch.where(hits, positions, width).min(dim=-1).values
            has_eos = first < width
            lengths = torch.where(has_eos, first + 1, lengths)
            reasons = torch.where(
                has_eos,
                torch.full_like(reasons, STOP_REASON_TOKEN_EOS),
                reasons,
            )
    if semantic_finished_lengths is not None:
        semantic = semantic_finished_lengths.to(device=device, dtype=torch.long)
        if semantic.ndim != 1 or semantic.numel() != batch_size:
            raise ValueError("semantic_finished_lengths must be batch aligned")
        if bool(((semantic < 0) | (semantic > width)).any().item()):
            raise ValueError("semantic_finished_lengths is outside the response width")
        use_semantic = (semantic > 0) & (semantic <= lengths)
        lengths = torch.where(use_semantic, semantic, lengths)
        reasons = torch.where(
            use_semantic,
            torch.full_like(reasons, STOP_REASON_ANSWER_TAG),
            reasons,
        )
    offsets = torch.arange(width, device=device).unsqueeze(0)
    mask = offsets < lengths.unsqueeze(1)
    return mask.to(dtype=dtype), reasons


def stop_reason_names(reason_codes: torch.Tensor) -> list[str]:
    names = {
        STOP_REASON_LENGTH: "length",
        STOP_REASON_TOKEN_EOS: "token_eos",
        STOP_REASON_ANSWER_TAG: "answer_tag",
    }
    return [names.get(int(value), "invalid") for value in reason_codes.detach().cpu().tolist()]
