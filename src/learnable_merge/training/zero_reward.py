"""Zero reward for teacher distillation through VERL's reward interface."""

from __future__ import annotations


_SOURCES = {
    "onethinker",
    "pixmo_ask_model_anything",
    "llava_v1_5_mix665k",
}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del solution_str, ground_truth, extra_info, kwargs
    if data_source not in _SOURCES:
        raise ValueError(f"Unknown training data source: {data_source!r}")
    return {
        "score": 0.0,
        "formal_grpo_contribution": 0.0,
        "reward_policy": "reward_free_fixed_teacher_vopd_v1",
    }
