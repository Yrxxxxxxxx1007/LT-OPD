"""CDPruner inference with the same cached decoding used during training."""
from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace


def _decode_masked_response_rows(tokenizer, responses, response_mask):
    return [
        tokenizer.decode(
            row[mask.bool()].detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for row, mask in zip(responses, response_mask, strict=True)
    ]


def _attach_cdpruner(model, settings):
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.models.transformers.vision_token_compressor import build_vision_token_compressor

    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=False,
                      use_fused_kernels=False, use_prefix_grouper=False)
    compressor = build_vision_token_compressor(settings, allow_legacy=False)
    compressor.set_final_retention_for_evaluation()
    for module in (model, model.model):
        module.vision_token_compressor = compressor
        module.vision_token_compressor_enabled = True
        module.vision_token_compressor_algorithm = compressor.algorithm
        module.vision_token_compressor_minimum_tokens = compressor.minimum_tokens
        module.vision_token_compressor_retention_ratio = compressor.retention_ratio
        module.vision_token_compressor_retention_bps = compressor.active_retention_bps
    model.model.vision_token_compressor_last_routes = None
    model.config.vision_token_compressor = settings
    model.requires_grad_(False).eval()


class CDPrunerInferenceRuntime:
    """Greedy answer generation with question-conditioned visual token pruning."""

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


def load_cdpruner_runtime(model_dir, *, torch_dtype="auto", device_map="auto"):
    """Load an exported Hugging Face model directory for CDPruner generation."""
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5Model

    model_dir = Path(model_dir).expanduser().resolve()
    settings = json.loads((model_dir / "cdpruner_config.json").read_text(encoding="utf-8"))
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map=device_map,
        attn_implementation=settings["attn_implementation"], trust_remote_code=True,
    )
    # Keep the trained native visual merger in FP32 when loading the towers in BF16.
    parameters = dict(model.named_parameters())
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as weights:
        for name in weights.keys():
            if "visual.merger." in name and weights.get_slice(name).get_dtype() == "F32":
                set_module_tensor_to_device(model, name, parameters[name].device,
                                            value=weights.get_tensor(name), dtype=torch.float32)
    _attach_cdpruner(model, settings["vision_token_compressor"])
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    processor.config = model.config
    processor.get_rope_index = MethodType(Qwen3_5Model.get_rope_index, processor)
    processor.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, processor)
    template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    return CDPrunerInferenceRuntime(model, processor, settings["vision_token_compressor"])
