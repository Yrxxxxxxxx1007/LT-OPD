"""Visual token compression inference with the training-time cached decoder."""
from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from types import MethodType, SimpleNamespace


LEARNABLE_SUMMARY_WEIGHT_RULE = "learnable_residual_mlp_v1"
LEARNABLE_SUMMARY_PREFIX = "model.vision_token_compressor.learnable_summary."
LEARNABLE_MERGE_INITIALIZATION = "pretrained_mlp_gate_rows_zero_output_v1"
LEARNABLE_MERGE_AGGREGATION = "zero_init_signed_residual_v1"
LEARNABLE_MERGE_REDUCTION = "differentiable_fp32_zero_init_signed_residual"
LEARNABLE_MERGE_ROUTE_SCHEMA = "vision_opd_cdpruner_learnable_merge_zero_residual_route_v1"
LEARNABLE_MERGE_CURRICULUM_ROUTE_SCHEMA = "vision_opd_cdpruner_learnable_merge_zero_residual_curriculum_route_v1"


def _model_input_hidden_size(config):
    text_config = getattr(config, "text_config", config)
    size = (text_config.get("hidden_size") if isinstance(text_config, Mapping)
            else getattr(text_config, "hidden_size", None))
    if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
        raise ValueError("Qwen post-merger hidden_size must be a positive integer")
    return int(size)


def _expected_learnable_summary_shapes(settings, *, input_hidden_size):
    """Keep the canonical head format shared by summary and retained-node merge."""
    summary = settings.get("summary", {})
    if summary is None:
        summary = {}
    if not isinstance(summary, Mapping):
        raise TypeError("Compressor summary settings must be a mapping")
    merge = settings.get("learnable_merge", {})
    if merge is None:
        merge = {}
    if not isinstance(merge, Mapping):
        raise TypeError("Compressor learnable_merge settings must be a mapping")
    summary_enabled = summary.get("enabled", False)
    merge_enabled = merge.get("enabled", False)
    if not isinstance(summary_enabled, bool) or not isinstance(merge_enabled, bool):
        raise TypeError("Summary and learnable_merge enabled flags must be boolean")
    if summary_enabled and merge_enabled:
        raise ValueError("Learnable retained-node merge and extra summary cannot both be enabled")
    if merge_enabled and merge.get("initialization") != LEARNABLE_MERGE_INITIALIZATION:
        raise ValueError(
            "Learnable merge initialization must be explicitly "
            "pretrained_mlp_gate_rows_zero_output_v1"
        )
    if merge_enabled:
        if merge.get("aggregation") != LEARNABLE_MERGE_AGGREGATION:
            raise ValueError("Learnable merge aggregation must explicitly be zero_init_signed_residual_v1")
        if {"anchor_prior_mass", "residual_logit_limit"}.intersection(merge):
            raise ValueError("Zero-initialized signed residual merge forbids anchor_prior_mass and residual_logit_limit")
        curriculum = settings.get("curriculum") or {}
        if not isinstance(curriculum, Mapping):
            raise TypeError("Compressor curriculum settings must be a mapping")
        curriculum_enabled = curriculum.get("enabled", False)
        if not isinstance(curriculum_enabled, bool):
            raise TypeError("Compressor curriculum enabled flag must be boolean")
        expected_schema = (LEARNABLE_MERGE_CURRICULUM_ROUTE_SCHEMA if curriculum_enabled
                           else LEARNABLE_MERGE_ROUTE_SCHEMA)
        if settings.get("route_schema_version") != expected_schema:
            raise ValueError("Learnable merge route schema does not describe zero-initialized signed residual aggregation")
        if settings.get("reduction") != LEARNABLE_MERGE_REDUCTION:
            raise ValueError("Learnable merge reduction must be differentiable_fp32_zero_init_signed_residual")
    active = merge if merge_enabled else summary
    default_rule = LEARNABLE_SUMMARY_WEIGHT_RULE if merge_enabled else None
    weight_rule = active.get("weight_rule", default_rule)
    if merge_enabled and weight_rule != LEARNABLE_SUMMARY_WEIGHT_RULE:
        raise ValueError("Learnable merge weight_rule must be learnable_residual_mlp_v1")
    if weight_rule != LEARNABLE_SUMMARY_WEIGHT_RULE:
        return {}
    if active.get("enabled") is not True:
        raise ValueError("Learnable compressor weights require their aggregation method to be enabled")
    hidden_size = active.get("hidden_size")
    stored_input_size = active.get("input_hidden_size")
    if (isinstance(hidden_size, bool) or not isinstance(hidden_size, Integral)
            or hidden_size != 64):
        raise ValueError("Learnable compressor hidden_size must be 64")
    if (isinstance(stored_input_size, bool) or not isinstance(stored_input_size, Integral)
            or stored_input_size != input_hidden_size):
        raise ValueError("Learnable compressor input_hidden_size differs from Qwen hidden_size")
    if not merge_enabled:
        limit = active.get("residual_logit_limit")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit != 4.0:
            raise ValueError("Learnable summary residual_logit_limit must be 4.0")
    return {
        LEARNABLE_SUMMARY_PREFIX + "input.weight": (64, 2 * input_hidden_size),
        LEARNABLE_SUMMARY_PREFIX + "input.bias": (64,),
        LEARNABLE_SUMMARY_PREFIX + "output.weight": (1, 64),
    }


def _validate_learnable_summary_state(state, settings, *, input_hidden_size, require_fp32=True):
    """Reject omitted, duplicate, malformed, or nonfinite learned checkpoint heads."""
    import torch

    expected = _expected_learnable_summary_shapes(settings, input_hidden_size=input_hidden_size)
    actual = {name for name in state if "vision_token_compressor." in name}
    if actual != set(expected):
        raise ValueError(
            "Compressor checkpoint parameter schema differs: "
            f"missing={sorted(set(expected) - actual)}, unexpected={sorted(actual - set(expected))}"
        )
    for name, shape in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"Learnable compressor tensor {name} must have shape {shape}")
        if require_fp32 and tensor.dtype != torch.float32:
            raise TypeError(f"Learnable compressor tensor {name} must be exported in FP32")
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"Learnable compressor tensor {name} contains NaN or Inf")
    if expected and not torch.count_nonzero(state[LEARNABLE_SUMMARY_PREFIX + "input.weight"]).item():
        raise ValueError("Learnable compressor input.weight is an uninitialized all-zero placeholder")
    return expected


def _safetensor_inventory(model_dir):
    """Read tensor names from one safe checkpoint or its declared shards."""
    from safetensors import safe_open

    model_dir = Path(model_dir).resolve()
    single = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    if single.is_file() and index.is_file():
        raise ValueError("Export contains both monolithic and sharded safetensors checkpoints")
    declared = None
    if index.is_file():
        index_contents = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(index_contents.get("metadata"), dict):
            raise ValueError("Safetensors checkpoint index is missing its HF metadata mapping")
        declared = index_contents.get("weight_map")
        if (not isinstance(declared, dict) or not declared
                or any(not isinstance(key, str) or not isinstance(value, str)
                       for key, value in declared.items())):
            raise ValueError("Invalid safetensors checkpoint weight_map")
        files = []
        for filename in sorted(set(declared.values())):
            path = (model_dir / filename).resolve()
            if not path.is_relative_to(model_dir) or path.suffix != ".safetensors":
                raise ValueError("Safetensors checkpoint index points outside its export directory")
            files.append(path)
    elif single.is_file():
        files = [single]
    else:
        raise FileNotFoundError("Export must contain model.safetensors or its safetensors index")
    inventory = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name in inventory:
                    raise ValueError(f"Duplicate tensor {name} in safetensors checkpoint shards")
                inventory[name] = path
    if declared is not None:
        actual = {name: str(path.relative_to(model_dir)).replace("\\", "/")
                  for name, path in inventory.items()}
        normalized = {name: value.replace("\\", "/") for name, value in declared.items()}
        if actual != normalized:
            raise ValueError("Safetensors shard contents differ from their checkpoint index")
    return inventory


def _read_learnable_summary_state(inventory, settings, *, input_hidden_size):
    from safetensors import safe_open

    state = {}
    for name, path in inventory.items():
        if "vision_token_compressor." in name:
            with safe_open(str(path), framework="pt", device="cpu") as weights:
                state[name] = weights.get_tensor(name)
    _validate_learnable_summary_state(state, settings, input_hidden_size=input_hidden_size)
    return state


def _build_compressor(model_config, settings):
    from verl.models.transformers.vision_token_compressor import build_vision_token_compressor

    return build_vision_token_compressor(
        settings, input_hidden_size=_model_input_hidden_size(model_config), allow_legacy=False,
    )


def _compression_checkpoint_model_class(auto_class, config, settings):
    """Make learned keys expected by HF before its checkpoint-loading pass."""
    model_class = auto_class._model_mapping[type(config)]
    if isinstance(model_class, (tuple, list)):
        candidates = [cls for cls in model_class
                      if cls.__name__ in (getattr(config, "architectures", None) or [])]
        if len(candidates) != 1:
            raise ValueError("Cannot resolve the exported model architecture unambiguously")
        model_class = candidates[0]

    class CompressionCheckpointModel(model_class):
        def __init__(self, model_config):
            super().__init__(model_config)
            # The outer model contains the inner model already: registering here
            # once produces only model.vision_token_compressor.* state keys.
            self.model.vision_token_compressor = _build_compressor(model_config, settings)

    return CompressionCheckpointModel


def _decode_masked_response_rows(tokenizer, responses, response_mask):
    return [
        tokenizer.decode(
            row[mask.bool()].detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for row, mask in zip(responses, response_mask, strict=True)
    ]


def _attach_compressor(model, settings, *, compressor=None):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=False,
                      use_fused_kernels=False, use_prefix_grouper=False)
    if compressor is None:
        compressor = _build_compressor(model.config, settings)
    compressor.set_final_retention_for_evaluation()
    embedding_weight = model.get_input_embeddings().weight
    compressor.to(device=embedding_weight.device, dtype=embedding_weight.dtype)
    model.model.vision_token_compressor = compressor
    for module in (model, model.model):
        module.vision_token_compressor_enabled = True
        module.vision_token_compressor_algorithm = compressor.algorithm
        module.vision_token_compressor_minimum_tokens = compressor.minimum_tokens
        module.vision_token_compressor_retention_ratio = compressor.retention_ratio
        module.vision_token_compressor_retention_bps = compressor.active_retention_bps
        module.vision_token_compressor_summary_enabled = bool(
            getattr(compressor, "summary_enabled", False)
        )
    model.model.vision_token_compressor_last_routes = None
    model.config.vision_token_compressor = settings
    model.requires_grad_(False).eval()


class VisualCompressionInferenceRuntime:
    """Greedy answer generation with the compressor saved in the checkpoint."""

    def __init__(self, model, processor, compressor_config):
        from verl.workers.config import RolloutConfig
        from verl.workers.rollout.hf_rollout import HFRollout
        from verl.utils.model import align_qwen35_chat_generation_config

        self.model = model
        self.processor = processor
        self.generation_config = align_qwen35_chat_generation_config(
            getattr(model, "generation_config", None), tokenizer=processor.tokenizer,
            model_config=model.config, required=True,
        )
        rollout = HFRollout.__new__(HFRollout)
        rollout.config = RolloutConfig(
            name="hf", mode="async", do_sample=False, temperature=1.0,
            top_k=-1, top_p=1.0, n=1, prompt_length=24576,
            response_length=1024, hf_dart_decode_batch_size=1, dtype="bfloat16",
            hf_use_replicated_module=True, hf_preserve_cuda_cache=True,
            semantic_stop_enabled=True, semantic_stop_string="</answer>",
            semantic_stop_include_tokens=True, semantic_stop_scan_response_only=True,
            semantic_stop_virtual_open=True, semantic_stop_transport_prefix="<answer>",
        )
        rollout.model_config = SimpleNamespace(
            vision_token_compressor=compressor_config,
            image_token_id=int(model.config.image_token_id), hf_config=model.config,
        )
        rollout.device_mesh = None
        rollout.module = model
        rollout.keep_module_in_eval = True
        rollout.tokenizer = processor.tokenizer
        rollout.processor = processor
        self.rollout = rollout

    def _make_prompts(self, messages_batch, *, response_length,
                      visual_compression_mode, route_queries_batch):
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl import DataProto
        from verl.utils.route_query import parse_route_query_value

        if visual_compression_mode not in {"merge", "dense", "no_image"}:
            raise ValueError("visual_compression_mode must be merge, dense, or no_image")
        if not messages_batch or len(route_queries_batch) != len(messages_batch):
            raise ValueError("Messages and route queries must have matching nonempty batches")
        if not 1 <= int(response_length) <= 1024:
            raise ValueError("response_length must be in [1, 1024]")
        raw = np.empty(len(messages_batch), dtype=object)
        queries = np.empty(len(messages_batch), dtype=object)
        uids = np.empty(len(messages_batch), dtype=object)
        for index, (messages, query) in enumerate(zip(messages_batch, route_queries_batch, strict=True)):
            raw[index] = messages
            queries[index] = parse_route_query_value(query).as_data_field()
            uids[index] = str(index)
        return DataProto(
            batch=TensorDict(
                {"dummy_tensor": torch.zeros((len(messages_batch), 1), dtype=torch.float32)},
                batch_size=[len(messages_batch)],
            ),
            non_tensor_batch={"raw_prompt": raw, "route_query": queries, "uid": uids},
            meta_info={
                "do_sample": False, "temperature": 1.0, "top_k": -1, "top_p": 1.0,
                "response_length": int(response_length),
                "eos_token_id": list(self.generation_config.eos_token_id),
                "pad_token_id": int(self.generation_config.pad_token_id),
                "visual_compression_mode": visual_compression_mode,
            },
        )

    def generate(self, messages_batch, *, response_length=1024,
                 visual_compression_mode="merge", route_queries_batch):
        import torch
        from verl.utils.response_protocol import classify_answer_protocol

        prompts = self._make_prompts(
            messages_batch, response_length=response_length,
            visual_compression_mode=visual_compression_mode,
            route_queries_batch=route_queries_batch,
        )
        with torch.inference_mode():
            output = self.rollout.generate_sequences(prompts)
        responses = output.batch["responses"].detach()
        response_mask = output.batch["response_mask"]
        raw = _decode_masked_response_rows(self.processor.tokenizer, responses, response_mask)
        classified = [classify_answer_protocol(value, virtual_open=True, transport_prefix="<answer>")
                      for value in raw]
        return {
            "responses": responses.cpu(),
            "response_mask": response_mask.cpu(),
            "decoded_raw_continuations": raw,
            "decoded_predictions": [item.reconstructed_text for item in classified],
            "protocol_status": [item.status for item in classified],
            "rollout_stop_reason": output.non_tensor_batch["rollout_stop_reason"],
        }


def load_compression_runtime(model_dir, *, torch_dtype="auto", device_map="auto"):
    """Load an exported Hugging Face model and its explicit compression method."""
    import torch
    # Import the compression stack on CPU before HF enters meta initialization.
    with torch.device("cpu"):
        import verl.models.transformers.vision_token_compressor
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5Model

    model_dir = Path(model_dir).expanduser().resolve()
    settings_path = model_dir / "visual_compression_config.json"
    if not settings_path.is_file():
        settings_path = model_dir / "cdpruner_config.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    compressor_settings = settings["vision_token_compressor"]
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    embedded_settings = getattr(config, "vision_token_compressor", None)
    if embedded_settings is not None and embedded_settings != compressor_settings:
        raise ValueError("Model and runtime visual compression settings differ")
    input_hidden_size = _model_input_hidden_size(config)
    inventory = _safetensor_inventory(model_dir)
    learned_state = _read_learnable_summary_state(
        inventory, compressor_settings, input_hidden_size=input_hidden_size,
    )
    loader = (_compression_checkpoint_model_class(AutoModelForImageTextToText, config, compressor_settings)
              if learned_state else AutoModelForImageTextToText)
    model, loading_info = loader.from_pretrained(
        model_dir, config=config, torch_dtype=torch_dtype, device_map=device_map,
        attn_implementation=settings["attn_implementation"], trust_remote_code=True,
        output_loading_info=True,
    )
    learned_load_errors = {
        kind: [name for name in loading_info.get(kind, [])
               if "vision_token_compressor." in (name[0] if isinstance(name, (tuple, list)) else name)]
        for kind in ("missing_keys", "unexpected_keys", "mismatched_keys")
    }
    if any(learned_load_errors.values()):
        raise ValueError(f"HF loading did not restore the declared compressor head: {learned_load_errors}")
    # Keep the trained native visual merger in FP32 when loading the towers in BF16.
    parameters = dict(model.named_parameters())
    for path in set(inventory.values()):
        with safe_open(str(path), framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if "visual.merger." not in name or weights.get_slice(name).get_dtype() != "F32":
                    continue
                if name not in parameters:
                    raise ValueError(f"Exported visual merger tensor {name} is absent from the loaded model")
                set_module_tensor_to_device(model, name, parameters[name].device,
                                            value=weights.get_tensor(name), dtype=torch.float32)
    _attach_compressor(model, compressor_settings,
                       compressor=getattr(model.model, "vision_token_compressor", None))
    if learned_state:
        head = model.model.vision_token_compressor.learnable_summary
        head.load_state_dict({name.removeprefix(LEARNABLE_SUMMARY_PREFIX): value
                              for name, value in learned_state.items()}, strict=True)
        head.mark_checkpoint_loaded()
        _validate_learnable_summary_state(
            {LEARNABLE_SUMMARY_PREFIX + name: value for name, value in head.state_dict().items()},
            compressor_settings, input_hidden_size=input_hidden_size, require_fp32=False,
        )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    processor.config = model.config
    processor.get_rope_index = MethodType(Qwen3_5Model.get_rope_index, processor)
    processor.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, processor)
    template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    return VisualCompressionInferenceRuntime(model, processor, settings["vision_token_compressor"])


# Existing CDPruner callers continue to load their saved method.
CDPrunerInferenceRuntime = VisualCompressionInferenceRuntime
load_cdpruner_runtime = load_compression_runtime
