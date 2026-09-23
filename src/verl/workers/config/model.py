# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from transformers import AutoConfig

from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import resolve_custom_chat_template
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config

__all__ = ["HFModelConfig"]


_V8_FULLIMAGE_CURRICULUM_SCHEDULE_SHA256 = (
    "a02cbf6e99fa6dc002373656ee6d2492b55c29f968df3c9558b67252633f9a9b"
)
_V8_FULLIMAGE_CURRICULUM_ROUTE_SCHEMA = "vision_opd_cdpruner_curriculum_route_v2"
_V8_FULLIMAGE_CURRICULUM_BUDGET_POLICY = (
    "per_image_min_N_max_32_ceil_active_optimizer_step_retention_bps_N_v1"
)


@dataclass
class HFModelConfig(BaseConfig):
    # note that we separate model_path, model_config_path and tokenizer_path in case they are different
    _mutable_fields = {
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
    }

    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None
    # Explicit runtime image resize; both values must be set together.
    processor_min_pixels: Optional[int] = None
    processor_max_pixels: Optional[int] = None

    # whether to load tokenizer. This is useful when we only want to load model config
    load_tokenizer: bool = True

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    # path to a custom chat template file for the model
    custom_chat_template_file: Optional[str] = None

    # path to a custom chat template file used only during validation
    val_custom_chat_template_file: Optional[str] = None

    external_lib: Optional[str] = None

    override_config: dict = field(default_factory=dict)

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    use_remove_padding: bool = True

    # TODO: unify fsdp and megatron lora config
    # fsdp lora related. We may setup a separate config later
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[str] = "all-linear"

    exclude_modules: Optional[str] = None

    # megatron lora config
    lora: dict[str, Any] = field(default_factory=dict)

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None
    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    # TiledMLP configuration for memory-efficient MLP computation
    tiled_mlp: dict = field(default_factory=lambda: {"enabled": False, "num_shards": 4})

    # Visual token projector for low-token student VLM training.
    vision_token_projector: dict = field(
        default_factory=lambda: {"enabled": False, "num_tokens": 64, "mlp_ratio": 2.0}
    )

    # Sixth-release parameter-free spatial DPC merge defaults.  The compressor
    # remains disabled until a launcher opts in.  Archived CDPruner launchers
    # explicitly override their complete legacy contract.
    vision_token_compressor: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "algorithm": "qwen35_holitom_dpc_spatial_merge_v1",
            "method": "holitom_inspired_dpc_diversity_merge",
            "paper_url": "https://proceedings.neurips.cc/paper_files/paper/2025/file/c573258c38d0a3919d8c1364053c45df-Paper-Conference.pdf",
            "official_repo": "cokeshao/HoliTom",
            "official_repo_revision": "e9b2972f6895c9e7d7fe74eb8c3a2ecaab8056e0",
            "adaptation_scope": "single_image_dpc_only_qwen35_post_native_merger_pre_llm_exact_budget",
            "original_method_reproduction_claim_allowed": False,
            "route_schema_version": "vision_opd_holitom_dpc_merge_route_v1",
            "transport_key": "dpc_merge_routes",
            "placement": "post_native_merger_pre_llm",
            "budget_policy": "per_image_min_N_max_32_ceil_0.05N",
            "retention_ratio": 0.05,
            "minimum_tokens_per_image": 32,
            "distance": "euclidean_div_sqrt_hidden_dim_fp32",
            "distance_chunk_tokens": 256,
            "routing_autograd": "detached_no_grad_fp32",
            "knn_k": 7,
            "center_score": "density_times_holitom_per_token_max_or_nearest_higher_delta",
            "assignment": "nearest_center_euclidean_stable_original_index_tie",
            "reduction": "differentiable_arithmetic_mean_of_original_embeddings",
            "position_policy": "inherit_center_original_three_axis_mrope_no_rebase",
            "query_conditioned": False,
            "attention_conditioned": False,
            "bbox_conditioned": False,
            "persistent_n_by_n_matrix_forbidden": True,
            "kernel_jitter": None,
            "relevance_epsilon": None,
            "residual_epsilon": None,
            "assignment_chunk_tokens": None,
            "selector_dtype": None,
            "merge": None,
            "query_policy": None,
            "relevance_fraction": None,
            "candidate_multiplier": None,
            "query_weight": None,
            "attention_weight": None,
            "bbox_weight": None,
            "deterministic": True,
            "data_manifest_path": None,
            "data_manifest_schema": None,
        }
    )

    # Legacy mode uses this as an isolated native-merger trainability switch.
    # Full-parameter mode uses it as an explicit declaration that the merger is
    # included alongside every language and vision parameter.
    train_native_visual_merger: bool = False

    architectures: Optional[list[str]] = None

    def __post_init__(self):
        import_external_libs(self.external_lib)

        if (self.processor_min_pixels is None) != (self.processor_max_pixels is None):
            raise ValueError("model processor_min_pixels and processor_max_pixels must be set together")
        if self.processor_min_pixels is not None:
            if (
                isinstance(self.processor_min_pixels, bool)
                or isinstance(self.processor_max_pixels, bool)
                or not isinstance(self.processor_min_pixels, int)
                or not isinstance(self.processor_max_pixels, int)
                or self.processor_min_pixels <= 0
                or self.processor_max_pixels < self.processor_min_pixels
            ):
                raise ValueError("model processor pixel bounds must be positive integers with min <= max")

        projector_enabled = bool(self.vision_token_projector.get("enabled", False))
        compressor_enabled = bool(self.vision_token_compressor.get("enabled", False))
        if projector_enabled and compressor_enabled:
            raise ValueError("vision_token_projector and vision_token_compressor cannot both be enabled")
        if projector_enabled:
            raise ValueError(
                "The legacy vision_token_projector is fail-closed in Vision-OPD-CDPruner; "
                "use vision_token_compressor or an archived legacy repository."
            )
        if compressor_enabled:
            compressor = self.vision_token_compressor
            algorithm = compressor.get("algorithm")
            common_expected = {
                "selector_dtype": "float32",
                "merge": "none_pure_index_prune",
                "position_policy": "selected_original_mrope",
                "deterministic": True,
            }
            if algorithm == "qwen35_cdpruner_v1":
                curriculum_config = compressor.get("curriculum")
                curriculum_enabled = curriculum_config is not None
                expected = {
                    **common_expected,
                    "algorithm": "qwen35_cdpruner_v1",
                    "method": "cdpruner",
                    "placement": "post_native_merger_pre_llm",
                    "budget_policy": (
                        _V8_FULLIMAGE_CURRICULUM_BUDGET_POLICY
                        if curriculum_enabled
                        else "per_image_max_32_ceil_0.05N"
                    ),
                    "route_schema_version": (
                        _V8_FULLIMAGE_CURRICULUM_ROUTE_SCHEMA
                        if curriculum_enabled
                        else "vision_opd_cdpruner_route_v1"
                    ),
                    "query_policy": "user_question_options_only_excluding_special_and_image_tokens",
                    "kernel_jitter": 1e-6,
                    "relevance_epsilon": 1e-6,
                    "residual_epsilon": 1e-12,
                    "assignment_chunk_tokens": 2048,
                }
                forbidden_legacy = [
                    key for key in ("relevance_fraction", "candidate_multiplier") if compressor.get(key) is not None
                ]
                if forbidden_legacy:
                    raise ValueError(f"CDPruner config contains legacy selector settings: {forbidden_legacy}")
                if curriculum_enabled:
                    from verl.models.transformers.visual_token_curriculum import (
                        CURRICULUM_DRIVER,
                        CURRICULUM_VALIDATION_CONTROL,
                        VisualTokenCurriculum,
                    )

                    curriculum = VisualTokenCurriculum.from_mapping(curriculum_config)
                    if curriculum.schedule_sha256 != _V8_FULLIMAGE_CURRICULUM_SCHEDULE_SHA256:
                        raise ValueError(
                            "CDPruner curriculum is not the immutable V8 full-image schedule: "
                            f"expected={_V8_FULLIMAGE_CURRICULUM_SCHEDULE_SHA256}, "
                            f"actual={curriculum.schedule_sha256}"
                        )
                    if curriculum.driver != CURRICULUM_DRIVER:
                        raise ValueError(
                            "CDPruner curriculum must be driven only by committed optimizer steps"
                        )
                    if curriculum.validation_control != CURRICULUM_VALIDATION_CONTROL:
                        raise ValueError(
                            "Diagnostic validation is forbidden from controlling CDPruner curriculum"
                        )
                    if (
                        curriculum.total_optimizer_steps != 175
                        or curriculum.final_retention_bps != 500
                        or curriculum.minimum_tokens_per_image != 32
                    ):
                        raise ValueError(
                            "V8 full-image CDPruner curriculum requires total_optimizer_steps=175, "
                            "final_retention_bps=500, and minimum_tokens_per_image=32"
                        )
                elif compressor.get("route_schema_version") == _V8_FULLIMAGE_CURRICULUM_ROUTE_SCHEMA:
                    raise ValueError(
                        "CDPruner curriculum route schema requires an explicit, hash-bound curriculum"
                    )
            elif algorithm == "qwen35_conditional_diversity_prune_v1":
                # Explicit third-release replay/resume compatibility only.
                expected = {
                    **common_expected,
                    "algorithm": "qwen35_conditional_diversity_prune_v1",
                    "query_policy": "user_question_options_only_excluding_special_and_image_tokens",
                }
            elif algorithm == "qwen35_holitom_dpc_spatial_merge_v1":
                from training.contract import load_contract

                release_contract, _ = load_contract()
                expected = release_contract["compressor"]
                conditional_only = (
                    "kernel_jitter",
                    "relevance_epsilon",
                    "residual_epsilon",
                    "assignment_chunk_tokens",
                    "selector_dtype",
                    "merge",
                    "query_policy",
                    "relevance_fraction",
                    "candidate_multiplier",
                    "query_weight",
                    "attention_weight",
                    "bbox_weight",
                )
                non_null_conditional = [
                    key for key in conditional_only if compressor.get(key) is not None
                ]
                if non_null_conditional:
                    raise ValueError(
                        "Formal HoliTom DPC config contains non-null conditional-pruning fields: "
                        f"{non_null_conditional}"
                    )
            else:
                raise ValueError(f"Unsupported vision_token_compressor.algorithm={algorithm!r}")
            for key, value in expected.items():
                if compressor.get(key) != value:
                    raise ValueError(
                        f"vision_token_compressor.{key} must be {value!r}, got {compressor.get(key)!r}"
                    )
            minimum_tokens = int(compressor.get("minimum_tokens_per_image", 0))
            retention_ratio = float(compressor.get("retention_ratio", 0.0))
            if algorithm in {"qwen35_cdpruner_v1", "qwen35_holitom_dpc_spatial_merge_v1"}:
                if minimum_tokens != 32 or retention_ratio != 0.05:
                    raise ValueError(
                        "CDPruner/formal DPC requires immutable minimum_tokens_per_image=32 "
                        "and retention_ratio=0.05"
                    )
            else:
                relevance_fraction = float(compressor.get("relevance_fraction", 0.0))
                candidate_multiplier = int(compressor.get("candidate_multiplier", 0))
                if minimum_tokens <= 0 or not 0.0 < retention_ratio <= 1.0:
                    raise ValueError("Legacy conditional pruning requires positive minimum_tokens and ratio in (0,1]")
                if not 0.0 < relevance_fraction <= 1.0 or candidate_multiplier < 1:
                    raise ValueError("Invalid legacy relevance_fraction or candidate_multiplier")
            if algorithm != "qwen35_holitom_dpc_spatial_merge_v1":
                data_manifest_path = compressor.get("data_manifest_path")
                if not data_manifest_path:
                    raise ValueError("Visual-token pruning requires vision_token_compressor.data_manifest_path")
                from pathlib import Path

                if not Path(str(data_manifest_path)).is_file():
                    raise FileNotFoundError(f"Visual-token pruning data manifest does not exist: {data_manifest_path}")

        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        # construct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
            processor_resize_kwargs = {}
            if self.processor_min_pixels is not None:
                processor_resize_kwargs = {
                    "min_pixels": self.processor_min_pixels,
                    "max_pixels": self.processor_max_pixels,
                }
            self.processor = hf_processor(
                self.local_tokenizer_path,
                trust_remote_code=self.trust_remote_code,
                **processor_resize_kwargs,
            )

        custom_chat_template = resolve_custom_chat_template(self)

        if custom_chat_template is not None:
            if self.processor is not None:
                self.processor.chat_template = custom_chat_template
            else:
                self.tokenizer.chat_template = custom_chat_template

        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )

        # construct hf_config
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code, attn_implementation=attn_implementation
        )

        override_config_kwargs = {}

        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )

        # TODO: (vermouth1992). self.config.model in megatron differs from that of fsdp in the override_config.
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)

        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        # get model architectures
        self.architectures = getattr(self.hf_config, "architectures", None)
        assert self.architectures is not None and len(self.architectures) == 1, (
            "Expect only one architecture, got {}".format(self.architectures)
        )
        if compressor_enabled and "Qwen3_5" not in self.architectures[0]:
            raise ValueError(
                f"{self.vision_token_compressor.get('algorithm')} is correctness-gated to Qwen3.5; "
                f"got architecture {self.architectures[0]!r}."
            )

        # per model patch
        if getattr(self.hf_config, "model_type", None) == "kimi_vl":
            self.hf_config.text_config.topk_method = "greedy"

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
