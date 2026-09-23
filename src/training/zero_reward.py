"""Formal reward sentinel for reward-free VOPD.

The release-06 optimization objective is entirely fixed-teacher VOPD.  This
function exists only so a generic VERL reward-manager construction cannot
silently select a task-specific scorer.  Any accidental invocation returns an
exact zero and records that the value is diagnostic-only; it never attempts to
judge PixMo/LLaVA open answers with lexical overlap.
"""

from __future__ import annotations


_SOURCES = {
    "onethinker",
    "pixmo_ask_model_anything",
    "llava_v1_5_mix665k",
}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del solution_str, ground_truth, extra_info, kwargs
    if data_source not in _SOURCES:
        raise ValueError(f"release-06 reward sentinel received an unknown source: {data_source!r}")
    return {
        "score": 0.0,
        "formal_grpo_contribution": 0.0,
        "reward_policy": "reward_free_fixed_teacher_vopd_v1",
    }
