# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Utils for tokenization."""

import types
import warnings

__all__ = ["hf_tokenizer", "hf_processor", "validate_processor_resize_contract"]


def validate_processor_resize_contract(processor, *, min_pixels: int, max_pixels: int) -> None:
    """Fail closed when the loaded image processor ignored resize overrides."""

    if processor is None or not hasattr(processor, "image_processor"):
        raise ValueError("A visual resize contract requires processor.image_processor")
    image_processor = processor.image_processor
    size = getattr(image_processor, "size", None)
    try:
        # Transformers 5.x represents image geometry with ``SizeDict``.  It
        # exposes canonical mapping semantics but intentionally does not
        # subclass ``dict``.  Materialize it before the exact value check so
        # supported Transformers versions share one fail-closed contract.
        size_values = dict(size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"processor.image_processor.size must be mapping-like, got {size!r}"
        ) from exc
    expected_size = {"shortest_edge": int(min_pixels), "longest_edge": int(max_pixels)}
    actual_size = {key: size_values.get(key) for key in expected_size}
    if actual_size != expected_size:
        raise ValueError(
            "processor.image_processor.size ignored the immutable resize contract: "
            f"expected={expected_size}, actual={actual_size}"
        )
    for attribute, expected in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        actual = getattr(image_processor, attribute, expected)
        if int(actual) != int(expected):
            raise ValueError(f"processor.image_processor.{attribute} must be {expected}, got {actual}")


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoConfig, AutoProcessor

    try:
        processor_min_pixels = kwargs.get("min_pixels")
        processor_max_pixels = kwargs.get("max_pixels")
        if (processor_min_pixels is None) != (processor_max_pixels is None):
            raise ValueError("hf_processor requires both min_pixels and max_pixels, or neither")
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        config_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"min_pixels", "max_pixels", "use_fast"}
        }
        config = AutoConfig.from_pretrained(name_or_path, **config_kwargs)

        if processor_min_pixels is not None:
            validate_processor_resize_contract(
                processor,
                min_pixels=int(processor_min_pixels),
                max_pixels=int(processor_max_pixels),
            )

        # Bind vlm model's get_rope_index method to processor
        processor.config = config
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                processor.get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, processor)
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                processor.get_rope_index = types.MethodType(Qwen2_5_VLModel.get_rope_index, processor)
            case "Qwen3VLProcessor":
                if getattr(config, "model_type", None) == "qwen3_5":
                    from transformers.models.qwen3_5 import Qwen3_5Model

                    processor.get_rope_index = types.MethodType(Qwen3_5Model.get_rope_index, processor)
                    if hasattr(Qwen3_5Model, "get_vision_position_ids"):
                        processor.get_vision_position_ids = types.MethodType(
                            Qwen3_5Model.get_vision_position_ids, processor
                        )
                elif getattr(config, "model_type", None) == "qwen3_5_moe":
                    from transformers.models.qwen3_5_moe import Qwen3_5MoeModel

                    processor.get_rope_index = types.MethodType(Qwen3_5MoeModel.get_rope_index, processor)
                    if hasattr(Qwen3_5MoeModel, "get_vision_position_ids"):
                        processor.get_vision_position_ids = types.MethodType(
                            Qwen3_5MoeModel.get_vision_position_ids, processor
                        )
                else:
                    from transformers.models.qwen3_vl import Qwen3VLModel

                    processor.get_rope_index = types.MethodType(Qwen3VLModel.get_rope_index, processor)
                    if hasattr(Qwen3VLModel, "get_vision_position_ids"):
                        processor.get_vision_position_ids = types.MethodType(
                            Qwen3VLModel.get_vision_position_ids, processor
                        )
            case "Glm4vImageProcessor":
                from transformers.models.glm4v import Glm4vModel

                processor.get_rope_index = types.MethodType(Glm4vModel.get_rope_index, processor)
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
