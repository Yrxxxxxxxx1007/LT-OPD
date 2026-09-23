"""Dataset prompts and images used by LT-OPD evaluation."""
from __future__ import annotations
import ast, base64, hashlib, io, json, math, re
from pathlib import Path
from typing import Any, Iterable
import pyarrow.parquet as pq
from PIL import Image

OFFICIAL_SYSTEM_PROMPT = "You are a helpful assistant."
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def resolve_runtime_path(value):
    return Path(value).expanduser().resolve(strict=True)

def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_option(value: Any) -> bool:
    return value is not None and not (
        isinstance(value, float) and math.isnan(value)
    ) and str(value) != "nan"


def image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        # HR-Bench stores JPEG payloads as very long base64 strings.  Never
        # pass those to stat(2); only a bounded string can be a filesystem path.
        if len(value) <= 4096:
            candidate = Path(value)
            try:
                if candidate.is_file():
                    return Image.open(candidate).convert("RGB")
            except OSError:
                pass
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    raise TypeError(f"unsupported image value: {type(value)}")


def image_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(rgb.width.to_bytes(8, "big"))
    digest.update(rgb.height.to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def iter_parquet_rows(source: Path, batch_size: int = 32) -> Iterable[dict[str, Any]]:
    files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {source}")
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def _vstar_prompt(text: str) -> str:
    options = re.findall(r"\([A-D]\)\s*([^()]+?)(?=\s*\([A-D]\)|$)", text)
    if options:
        question = text.split("(A)", 1)[0].strip()
        text = question + "\n" + "\n".join(
            f"{letter}. {option.strip()}" for letter, option in zip("ABCD", options)
        )
    return text + "\nAnswer with the option's letter from the given choices directly."


def _mmmu_images(row: dict[str, Any]) -> list[Image.Image]:
    # The pinned lmms-eval mmmu_doc_to_visual scans construct_prompt(doc),
    # which includes both the question and formatted answer options.  Thirteen
    # validation rows place every image marker in the options; scanning only
    # row["question"] silently turned them into text-only samples.
    image_tokens = sorted(set(re.findall(r"<image \d+>", _mmmu_prompt(row))))
    keys = [token.strip("<>").replace(" ", "_") for token in image_tokens]
    missing = [key for key in keys if key not in row or row[key] is None]
    if missing:
        raise RuntimeError(f"MMMU prompt references missing images: {missing}")
    return [image_from_value(row[key]) for key in keys]


def _mmmu_prompt(row: dict[str, Any]) -> str:
    options = ast.literal_eval(row["options"])
    formatted = "\n".join(f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options))
    if row["question_type"] == "multiple-choice":
        return f"Question: {row['question']}\nOptions:\n{formatted}\nAnswer with the option letter only."
    return f"Question: {row['question']}\nOptions:\n{formatted}\nPlease answer the question directly."


def rows_for(
    dataset: str,
    config: dict[str, Any],
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Iterable[dict[str, Any]]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    source = resolve_runtime_path(config["source"])
    if dataset == "pope":
        from datasets import load_from_disk

        split = load_from_disk(str(source))["test"]
        for index, row in enumerate(split):
            if index % world_size != rank:
                continue
            image = row["image"].convert("RGB")
            yield {
                "uid": str(row["id"]), "source_index": index, "image": image,
                "prompt": row["question"].strip() + "\nAnswer the question using a single word or phrase.",
                "gold": row["answer"], "category": row["category"],
                "question_id": str(row["question_id"]),
            }
        return

    if dataset == "textvqa":
        rows = json.loads(source.read_text(encoding="utf-8"))["data"]
        image_root = resolve_runtime_path(config["image_root"])
        for index, row in enumerate(rows):
            if index % world_size != rank:
                continue
            image_path = image_root / f"{row['image_id']}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            yield {
                "uid": str(row["question_id"]), "source_index": index,
                "image": image_path,
                "prompt": row["question"].capitalize() + "\nAnswer the question using a single word or phrase.",
                "gold": list(row["answers"]), "image_id": str(row["image_id"]),
            }
        return

    if dataset == "gqa":
        image_rows = pq.read_table(resolve_runtime_path(config["image_source"])).to_pylist()
        # Only 398 images back 12,578 questions. Decode each shared image once
        # rather than repeatedly allocating/decoding the same byte payload.
        images = {str(row["id"]): image_from_value(row["image"]) for row in image_rows}
        for index, row in enumerate(iter_parquet_rows(source)):
            if index % world_size != rank:
                continue
            image_id = str(row["imageId"])
            if image_id not in images:
                raise KeyError(f"GQA image missing: {image_id}")
            yield {
                "uid": str(row["id"]), "source_index": index,
                "image": images[image_id],
                "prompt": row["question"].strip() + "\nAnswer the question using a single word or phrase.",
                "gold": row["answer"], "image_id": image_id,
                "structural_type": row["types"]["structural"],
                "semantic_type": row["types"]["semantic"],
                "detailed_type": row["types"]["detailed"],
            }
        return

    for index, row in enumerate(iter_parquet_rows(source)):
        if index % world_size != rank:
            continue
        if dataset == "vstar":
            yield {
                "uid": str(row["question_id"]), "source_index": index,
                "image": image_from_value(row["image"]), "prompt": _vstar_prompt(row["text"]),
                "gold": str(row["label"]), "category": str(row["category"]),
            }
        elif dataset in {"hrbench4k", "hrbench8k"}:
            options = {letter: str(row[letter]) for letter in LETTERS if valid_option(row.get(letter))}
            prompt = row["question"].strip() + "\n" + "".join(
                f"{letter}. {value}\n" for letter, value in options.items()
            ) + "Answer the option letter directly."
            yield {
                "uid": str(row["index"]), "source_index": index,
                "image": image_from_value(row["image"]), "prompt": prompt,
                "gold": str(row["answer"]), "options": options,
                "question": str(row["question"]), "category": str(row["category"]),
                "cycle_category": int(row["cycle_category"]),
            }
        elif dataset == "mmmu":
            yield {
                "uid": str(row["id"]), "source_index": index,
                "images": _mmmu_images(row), "prompt": _mmmu_prompt(row),
                "gold": str(row["answer"]), "options": str(row["options"]),
                "question_type": str(row["question_type"]),
                "subfield": str(row.get("subfield", "")),
            }
        elif dataset == "ocrbench":
            yield {
                "uid": str(index), "source_index": index,
                "image": image_from_value(row["image"]), "prompt": row["question"].strip(),
                "gold": row["answer"], "dataset_name": str(row["dataset"]),
                "question_type": str(row["question_type"]),
            }
        elif dataset == "mmbench_circular":
            options = {letter: str(row[letter]) for letter in "ABCDE" if valid_option(row.get(letter))}
            option_prompt = "There are several options:\n" + "\n".join(
                f"{letter}. {value}" for letter, value in options.items()
            )
            core = (
                f"{row['hint']} {row['question']} {option_prompt}"
                if valid_option(row.get("hint"))
                else f"{row['question']} {option_prompt}"
            )
            yield {
                "uid": str(row["index"]), "source_index": index,
                "image": image_from_value(row["image"]),
                # Pinned lmms-eval calls ``f"{query_prompt}\n{post_prompt}"``
                # while the configured post_prompt itself starts with ``\n``.
                # Preserve both newlines byte-for-byte for comparable results.
                "prompt": core + "\n\nAnswer with the option's letter from the given choices directly.",
                "gold": str(row["answer"]), "options": options,
                "question": str(row["question"]), "hint": row.get("hint"),
                "index": int(row["index"]), "category": str(row["category"]),
                "L2-category": str(row["L2-category"]), "source_name": str(row["source"]),
                "split": str(row["split"]),
            }
        else:
            raise ValueError(dataset)


def row_messages(
    row: dict[str, Any],
    image_audit_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # MMMU includes a small number of official text-only questions.  Preserve
    # their empty visual list exactly; every other loader supplies `image`.
    images = row["images"] if "images" in row else [row["image"]]
    content: list[dict[str, Any]] = []
    image_audit: list[dict[str, Any]] = []
    for image_value in images:
        cache_key = None
        if isinstance(image_value, Path):
            resolved = image_value.resolve()
            cache_key = ("path", str(resolved))
            content.append({"type": "image", "image": str(resolved)})
        else:
            # GQA reuses 398 images for 12,578 questions.  The stable image ID
            # is a safe cache key and avoids hashing identical pixels 30x.
            if len(images) == 1 and row.get("image_id") is not None:
                cache_key = ("image_id", str(row["image_id"]))
            rgb = image_from_value(image_value)
            content.append({"type": "image", "image": rgb})
        audit = image_audit_cache.get(cache_key) if image_audit_cache is not None and cache_key is not None else None
        if audit is None:
            if isinstance(image_value, Path):
                with Image.open(image_value) as loaded:
                    rgb = loaded.convert("RGB")
                    audit = {
                        "path": str(image_value.resolve()), "file_sha256": sha256_file(image_value),
                        "pixel_sha256": image_pixel_sha256(rgb), "width": rgb.width, "height": rgb.height,
                    }
            else:
                audit = {
                    "path": None, "file_sha256": None, "pixel_sha256": image_pixel_sha256(rgb),
                    "width": rgb.width, "height": rgb.height,
                }
            if image_audit_cache is not None and cache_key is not None:
                image_audit_cache[cache_key] = dict(audit)
        image_audit.append(dict(audit))
    content.append({"type": "text", "text": row["prompt"]})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": OFFICIAL_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ], image_audit


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"image", "images"}}
