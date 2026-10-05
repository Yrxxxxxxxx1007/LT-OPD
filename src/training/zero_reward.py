"""Zero reward for fixed-teacher VOPD.

The generic VERL reward manager uses this scorer to return zero without
adding a task-specific reward to the distillation objective.
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
        raise ValueError(f"Zero-reward scorer received an unknown source: {data_source!r}")
    return {
        "score": 0.0,
        "formal_grpo_contribution": 0.0,
        "reward_policy": "reward_free_fixed_teacher_vopd_v1",
    }
