"""Durable, hash-bound checkpoint primitives used by the V7 release.

The checkpoint directory itself is intentionally not renamed: every FSDP
rank writes into it concurrently.  Atomic discoverability is instead provided
by a final manifest which is written only after every payload has been flushed,
fsynced and SHA-256 bound.  Consumers validate the manifest before deserializing
any tensor payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

import torch


ACTOR_MARKER_V2 = "verl_fsdp_atomic_complete_v2"
GLOBAL_MARKER_V2 = "verl_global_checkpoint_complete_v2"
GLOBAL_MANIFEST_NAME = "CHECKPOINT_MANIFEST.json"
INCOMPLETE_QUARANTINE_NAME = "INCOMPLETE_CHECKPOINT.json"


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fsync_directory(path: os.PathLike[str] | str) -> None:
    """Persist directory-entry replacements where the platform supports it."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError:
        # Windows does not expose POSIX directory fsync. File flush/fsync and
        # same-directory os.replace still provide the strongest available path.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: os.PathLike[str] | str) -> None:
    """Public durability primitive for same-filesystem metadata transactions."""

    _fsync_directory(path)


def cleanup_atomic_temporary_siblings(
    directory: os.PathLike[str] | str,
    *,
    allowed_targets: Iterable[str],
) -> list[str]:
    """Remove only provably uncommitted siblings made by our atomic writers.

    A SIGKILL can leave ``.<target>.<pid>.<uuid>.tmp`` after the final target
    was either published or never published.  Such files are never commits and
    are safe to discard, but arbitrary files, directories and symlinks remain
    fail-closed so recovery cannot hide foreign evidence.
    """

    root = pathlib.Path(directory)
    if not root.exists() and not root.is_symlink():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Atomic temporary-file root must be a regular directory: {root}")
    targets = list(allowed_targets)
    if (
        not targets
        or len(targets) != len(set(targets))
        or any(
            not isinstance(target, str)
            or not target
            or pathlib.PurePath(target).name != target
            for target in targets
        )
    ):
        raise ValueError("Atomic temporary cleanup requires unique target basenames")
    patterns = {
        target: re.compile(
            rf"^\.{re.escape(target)}\.([1-9][0-9]*)\.([0-9a-f]{{32}})\.tmp$"
        )
        for target in targets
    }
    removed: list[str] = []
    for path in root.iterdir():
        if not any(pattern.fullmatch(path.name) for pattern in patterns.values()):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Atomic temporary sibling is not a regular file: {path}")
        path.unlink()
        removed.append(path.name)
    if removed:
        _fsync_directory(root)
    return sorted(removed)


def _temporary_sibling(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _ensure_parent_directories_durable(directory: pathlib.Path) -> None:
    """Create missing parents and persist each new directory entry."""

    missing: list[pathlib.Path] = []
    cursor = directory
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise FileNotFoundError(f"Atomic target has no existing filesystem ancestor: {directory}")
        cursor = parent
    for path in reversed(missing):
        path.mkdir(exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Atomic target parent must be a regular directory: {path}")
        _fsync_directory(path)
        _fsync_directory(path.parent)


def _prepare_atomic_target(path: os.PathLike[str] | str) -> pathlib.Path:
    target = pathlib.Path(path)
    _ensure_parent_directories_durable(target.parent)
    if target.is_symlink():
        raise ValueError(f"Atomic checkpoint target must not be a symlink: {target}")
    return target


def atomic_torch_save(value: Any, path: os.PathLike[str] | str) -> None:
    target = _prepare_atomic_target(path)
    temporary = _temporary_sibling(target)
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json_dump(value: Any, path: os.PathLike[str] | str) -> None:
    target = _prepare_atomic_target(path)
    temporary = _temporary_sibling(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_text_write(text: str, path: os.PathLike[str] | str) -> None:
    target = _prepare_atomic_target(path)
    temporary = _temporary_sibling(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def quarantine_incomplete_global_checkpoint(
    step_directory: os.PathLike[str] | str,
    *,
    expected_step: int,
) -> pathlib.Path | None:
    """Atomically preserve an uncommitted retry target outside its live name.

    A distributed Actor marker can become durable just before the controller
    writes ``data.pt`` and the global commit. A later exact replay of that step
    must not overwrite the evidence, but it must also not be permanently
    blocked by the inner marker. Moving the whole uncommitted directory to a
    unique sibling is atomic on the same filesystem and keeps it recoverable.
    """

    if isinstance(expected_step, bool) or not isinstance(expected_step, int) or expected_step <= 0:
        raise ValueError("expected_step must be a positive integer")
    step_path = pathlib.Path(step_directory)
    if step_path.name != f"global_step_{expected_step}":
        raise ValueError("checkpoint quarantine target does not match expected_step")
    if step_path.is_symlink():
        raise ValueError("checkpoint quarantine target must not be a symlink")
    if not os.path.lexists(step_path):
        return None
    if not step_path.is_dir():
        raise ValueError("checkpoint quarantine target must be a directory")

    completion = step_path / ".checkpoint_complete"
    if completion.exists() or completion.is_symlink():
        validate_global_checkpoint(
            step_path,
            expected_step=expected_step,
            allow_legacy_marker=False,
        )
        raise FileExistsError(f"Refusing to overwrite committed checkpoint: {step_path}")

    tracker = step_path.parent / "latest_checkpointed_iteration.txt"
    if tracker.exists() or tracker.is_symlink():
        if not tracker.is_file() or tracker.is_symlink():
            raise ValueError("checkpoint tracker is not a regular file")
        raw_tracker = tracker.read_text(encoding="utf-8").strip()
        if not raw_tracker.isdigit():
            raise ValueError("checkpoint tracker is not a step integer")
        if int(raw_tracker) == expected_step:
            raise RuntimeError("checkpoint tracker points at an uncommitted retry target")

    quarantine_id = uuid.uuid4().hex
    quarantine_path = step_path.with_name(
        f".incomplete_global_step_{expected_step}.{quarantine_id}"
    )
    atomic_json_dump(
        {
            "schema_version": "verl_incomplete_global_checkpoint_quarantine_v1",
            "global_step": expected_step,
            "original_directory": step_path.name,
            "quarantine_directory": quarantine_path.name,
            "reason": "global_commit_missing_before_retry",
        },
        step_path / INCOMPLETE_QUARANTINE_NAME,
    )
    os.replace(step_path, quarantine_path)
    _fsync_directory(step_path.parent)
    return quarantine_path


def fsync_regular_tree(root: os.PathLike[str] | str) -> None:
    """Flush every regular, non-symlink file before publishing a manifest."""

    root_path = pathlib.Path(root)
    if not root_path.is_dir() or root_path.is_symlink():
        raise FileNotFoundError(f"Checkpoint tree is not a regular directory: {root_path}")
    directories = [root_path]
    for current, directory_names, file_names in os.walk(root_path, followlinks=False):
        current_path = pathlib.Path(current)
        for name in (*directory_names, *file_names):
            if (current_path / name).is_symlink():
                raise ValueError(f"Checkpoint tree contains a symlink: {current_path / name}")
        for name in file_names:
            file_path = current_path / name
            with file_path.open("rb") as handle:
                os.fsync(handle.fileno())
        directories.extend(current_path / name for name in directory_names)
    for directory in reversed(directories):
        _fsync_directory(directory)


def artifact_binding(
    root: os.PathLike[str] | str,
    relative_path: str,
) -> dict[str, Any]:
    pure = pathlib.PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"Unsafe checkpoint artifact path: {relative_path!r}")
    normalized = pure.as_posix()
    path = pathlib.Path(root).joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Checkpoint artifact is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Checkpoint artifact is empty: {path}")
    return {"relative_path": normalized, "size_bytes": size, "sha256": sha256_file(path)}


def build_artifact_inventory(
    root: os.PathLike[str] | str,
    relative_paths: Iterable[str],
) -> list[dict[str, Any]]:
    paths = list(relative_paths)
    if paths != sorted(set(paths)):
        raise ValueError("Checkpoint artifact paths must be unique and sorted")
    return [artifact_binding(root, relative_path) for relative_path in paths]


def validate_artifact_inventory(
    root: os.PathLike[str] | str,
    artifacts: Any,
    *,
    expected_paths: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Checkpoint artifact inventory must be a non-empty list")
    paths: list[str] = []
    normalized: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {"relative_path", "size_bytes", "sha256"}:
            raise ValueError("Malformed checkpoint artifact binding")
        relative = item["relative_path"]
        size = item["size_bytes"]
        digest = item["sha256"]
        if (
            not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Malformed checkpoint artifact binding")
        actual = artifact_binding(root, relative)
        if actual != dict(item):
            raise ValueError(f"Checkpoint artifact differs from its manifest: {relative}")
        paths.append(relative)
        normalized.append(dict(item))
    if paths != sorted(set(paths)):
        raise ValueError("Checkpoint artifact inventory is non-canonical")
    if expected_paths is not None and paths != sorted(set(expected_paths)):
        raise ValueError("Checkpoint artifact inventory does not match the required payload set")
    return normalized


def validate_global_checkpoint(
    step_directory: os.PathLike[str] | str,
    *,
    expected_step: int | None = None,
    allow_legacy_marker: bool = True,
    validate_actor_payloads: bool = False,
) -> dict[str, Any]:
    """Validate the global commit marker before actor/data deserialization."""

    step_path = pathlib.Path(step_directory)
    if step_path.is_symlink() or not step_path.is_dir():
        raise ValueError(f"Checkpoint step must be a regular directory: {step_path}")
    marker_path = step_path / ".checkpoint_complete"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise FileNotFoundError(f"Checkpoint has no regular global completion marker: {marker_path}")
    raw = marker_path.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        raise ValueError("Checkpoint global completion sentinel is not a step integer")
    sentinel_step = int(raw)
    if expected_step is not None and sentinel_step != int(expected_step):
        raise ValueError("Checkpoint global completion sentinel step mismatch")
    manifest_path = step_path / GLOBAL_MANIFEST_NAME
    if not manifest_path.exists() and not manifest_path.is_symlink():
        if not allow_legacy_marker:
            raise FileNotFoundError(f"Checkpoint has no V2 global manifest: {manifest_path}")
        if not (step_path / "data.pt").is_file():
            raise FileNotFoundError("Legacy checkpoint is missing data.pt")
        return {
            "schema_version": "legacy_step_only_v1",
            "global_step": sentinel_step,
            "integrity": "partial",
        }
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Checkpoint global manifest is not a regular file")
    try:
        marker = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Checkpoint global manifest is invalid JSON") from exc
    if marker.get("schema_version") != GLOBAL_MARKER_V2:
        raise ValueError("Checkpoint global completion marker has an unsupported schema")
    actor_path = step_path / "actor"
    if actor_path.is_symlink() or not actor_path.is_dir():
        raise ValueError("Checkpoint V2 actor state must be a regular directory")
    step = marker.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("Checkpoint global completion marker has an invalid step")
    if expected_step is not None and step != int(expected_step):
        raise ValueError("Checkpoint global completion marker step mismatch")
    if step != sentinel_step:
        raise ValueError("Checkpoint global manifest and completion sentinel disagree")
    artifacts = validate_artifact_inventory(
        step_path,
        marker.get("artifacts"),
        expected_paths=["actor/CHECKPOINT_COMPLETE.json", "data.pt"],
    )
    if marker.get("artifact_inventory_sha256") != canonical_sha256(artifacts):
        raise ValueError("Checkpoint global artifact inventory digest is invalid")
    if validate_actor_payloads:
        actor_marker_path = actor_path / "CHECKPOINT_COMPLETE.json"
        if not actor_marker_path.is_file() or actor_marker_path.is_symlink():
            raise FileNotFoundError("Checkpoint has no regular V2 actor completion marker")
        try:
            actor_marker = json.loads(actor_marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Actor checkpoint completion marker is invalid JSON") from exc
        world_size = actor_marker.get("world_size")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise ValueError("Actor checkpoint completion marker has an invalid world size")
        validate_actor_checkpoint_v2(
            actor_path,
            expected_world_size=world_size,
            expected_step=step,
        )
    return dict(marker)


def validate_actor_checkpoint_v2(
    actor_directory: os.PathLike[str] | str,
    *,
    expected_world_size: int,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Validate every V2 actor payload without importing torch checkpoint code."""

    actor_path = pathlib.Path(actor_directory)
    if actor_path.is_symlink() or not actor_path.is_dir():
        raise ValueError(f"Actor checkpoint must be a regular directory: {actor_path}")
    marker_path = actor_path / "CHECKPOINT_COMPLETE.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise FileNotFoundError(f"Actor checkpoint has no regular completion marker: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Actor checkpoint completion marker is invalid JSON") from exc
    if marker.get("schema_version") != ACTOR_MARKER_V2:
        raise ValueError("Actor checkpoint does not use the full-integrity V2 schema")
    if marker.get("world_size") != int(expected_world_size):
        raise ValueError("Actor checkpoint world size differs from the expected runtime")
    step = marker.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("Actor checkpoint V2 marker has an invalid step")
    if expected_step is not None and step != int(expected_step):
        raise ValueError("Actor checkpoint V2 marker step mismatch")
    expected_core = sorted(
        [
            f"{kind}_world_size_{expected_world_size}_rank_{rank}.pt"
            for rank in range(expected_world_size)
            for kind in ("model", "optim", "extra_state", "rollout_rng")
        ]
        + ["V6_RUNTIME_PROFILE.json", "checkpoint_provenance.json", "fsdp_config.json"]
    )
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Actor checkpoint V2 marker has no artifact inventory")
    paths = [item.get("relative_path") if isinstance(item, Mapping) else None for item in artifacts]
    hf_paths = sorted(path for path in paths if isinstance(path, str) and path.startswith("huggingface/"))
    if not hf_paths or "huggingface/config.json" not in hf_paths:
        raise ValueError("Actor checkpoint V2 marker has no Hugging Face config binding")
    validated = validate_artifact_inventory(
        actor_path,
        artifacts,
        expected_paths=expected_core + hf_paths,
    )
    if marker.get("artifact_inventory_sha256") != canonical_sha256(validated):
        raise ValueError("Actor checkpoint V2 artifact inventory digest is invalid")
    return dict(marker)
