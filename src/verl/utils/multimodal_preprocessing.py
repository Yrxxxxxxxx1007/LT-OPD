"""Multimodal preprocessing shared by DART student and teacher paths.

Qwen's convenience ``process_vision_info`` helper resizes still images before
the Hugging Face image processor sees them.  Passing those already-resized
images to the processor performs a second interpolation and can make the
student pixels/grid differ from the fixed teacher even when both rows point to
the same source file.  DART still images must therefore be loaded as raw RGB
PIL images and resized exactly once by the model's processor.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


_PER_ITEM_RESIZE_KEYS = {"min_pixels", "max_pixels", "resized_height", "resized_width"}


def _is_image_item(item: Any) -> bool:
    return isinstance(item, dict) and (
        item.get("type") in {"image", "image_url"}
        or "image" in item
        or "image_url" in item
    )


def _is_video_item(item: Any) -> bool:
    return isinstance(item, dict) and (item.get("type") == "video" or "video" in item)


def _decode_image_source(item: dict[str, Any]) -> Image.Image:
    overrides = sorted(_PER_ITEM_RESIZE_KEYS.intersection(item))
    if overrides:
        raise ValueError(
            "DART image items cannot carry pre-resize overrides; the model processor must be the only resize owner. "
            f"Found {overrides}."
        )

    source = item.get("image")
    if source is None:
        source = item.get("image_url")
    if source is None:
        source = item.get("path")
    if isinstance(source, dict):
        source = source.get("url") or source.get("path") or source.get("image")

    image_bytes = item.get("bytes")
    if image_bytes is not None:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, Path):
        source = str(source)
    if not isinstance(source, str) or not source:
        raise TypeError(f"Unsupported DART image source: {type(source)}")

    if source.startswith("data:image") and "base64," in source:
        source = source.split("base64,", 1)[1]
        with Image.open(BytesIO(base64.b64decode(source))) as image:
            return image.convert("RGB")
    if source.startswith(("http://", "https://")):
        # Keep network loading in the maintained Transformers helper, which
        # performs no spatial resize. Local training data never takes this path.
        from transformers.image_utils import load_image

        return load_image(source).convert("RGB")

    with Image.open(source.removeprefix("file://")) as image:
        return image.convert("RGB")


def extract_raw_images_from_messages(messages: list[dict[str, Any]]) -> list[Image.Image]:
    """Load every still image without spatial preprocessing.

    The returned RGB images retain the source width/height.  Only the model's
    Hugging Face processor may resize them afterwards.
    """

    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if _is_image_item(item):
                images.append(_decode_image_source(item))
    return images


def _extract_videos_without_processing_images(
    messages: list[dict[str, Any]], *, image_patch_size: int
) -> list[tuple[Any, dict[str, Any]]]:
    """Delegate video decoding to qwen-vl-utils while excluding still images."""

    video_messages = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        video_content = [deepcopy(item) for item in content if _is_video_item(item)]
        if video_content:
            video_messages.append({"role": message.get("role", "user"), "content": video_content})
    if not video_messages:
        return []

    from qwen_vl_utils import process_vision_info

    unexpected_images, videos = process_vision_info(
        video_messages,
        image_patch_size=image_patch_size,
        return_video_metadata=True,
    )
    if unexpected_images:
        raise RuntimeError("Video-only DART preprocessing unexpectedly produced still images")
    return videos or []


def process_dart_messages_once(
    processor,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Apply the chat template and the model processor exactly once.

    This is the canonical preprocessing entry point for both the compressed
    student and the full-token teacher.  It intentionally does not call
    qwen-vl-utils for still images.
    """

    template_kwargs = dict(apply_chat_template_kwargs or {})
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
    images = extract_raw_images_from_messages(messages)
    image_patch_size = int(processor.image_processor.patch_size)
    videos = _extract_videos_without_processing_images(messages, image_patch_size=image_patch_size)
    if videos:
        video_tensors, video_metadatas = zip(*videos, strict=True)
        video_tensors = list(video_tensors)
        video_metadatas = list(video_metadatas)
    else:
        video_tensors = None
        video_metadatas = None

    processed = dict(
        processor(
            text=[text],
            images=images or None,
            videos=video_tensors,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
            truncation=False,
        )
    )
    return text, processed
