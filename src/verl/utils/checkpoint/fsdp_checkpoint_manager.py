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

import hashlib
import json
import logging
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank

from .checkpoint_manager import BaseCheckpointManager
from .integrity import (
    ACTOR_MARKER_V2,
    artifact_binding,
    atomic_json_dump,
    atomic_torch_save,
    canonical_sha256,
    fsync_regular_tree,
    validate_artifact_inventory,
)

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_ATOMIC_COMPLETE_MARKER = "CHECKPOINT_COMPLETE.json"
_V6_RUNTIME_PROFILE_SIDECAR = "V6_RUNTIME_PROFILE.json"
_FORMAL_V8_FULLIMAGE_CURRICULUM_CHECKPOINT_SCHEMA = (
    "vision_opd_ai4s_v8_fullimage_curriculum_checkpoint_v1"
)
_FORMAL_V8_FULLIMAGE_CURRICULUM_STATIC_SCHEMA = (
    "vision_opd_ai4s_v8_fullimage_curriculum_static_contract_v1"
)
_FORMAL_V8_FULLIMAGE_CURRICULUM_RELEASE = (
    "qwen35_vqa14k_cdpruner_fullimage_curriculum_fixed_teacher_hf_v1"
)
_FORMAL_V8_FULLIMAGE_CURRICULUM_DATA_SCHEMA = (
    "vision_opd_v8_vqa14k_fullimage_curriculum_raw_data_v1"
)
_FORMAL_V8_FULLIMAGE_CURRICULUM_MANIFEST_SCHEMA = (
    "vision_opd_v8_vqa14k_fullimage_curriculum_materialization_manifest_v1"
)
_FORMAL_V8_FULLIMAGE_CURRICULUM_SAMPLER_SCHEMA = (
    "vision_opd_v8_vqa14k_fullimage_balanced_sampler_state_v1"
)
_FORMAL_V8_FULLIMAGE_CURRICULUM_ROUTE_SCHEMA = "vision_opd_cdpruner_curriculum_route_v2"
_FORMAL_V8_FULLIMAGE_CURRICULUM_SCHEDULE_SHA256 = (
    "a02cbf6e99fa6dc002373656ee6d2492b55c29f968df3c9558b67252633f9a9b"
)
_FORBIDDEN_PRIOR_FORMAL_CHECKPOINT_SCHEMAS = frozenset(
    {"vision_opd_ai4s_v6_full_parameter_checkpoint_v1"}
)


_EXP3_MIGRATION_OLD_FINGERPRINT = "126bf544e6419826d652d754cdc59eba85eb0f4c77f74cee8d4f567ff27481a7"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_formal_v8_fullimage_curriculum_provenance(provenance: dict) -> None:
    """Reject a forged or partially migrated V8 checkpoint identity at construction."""

    if provenance.get("schema_version") != _FORMAL_V8_FULLIMAGE_CURRICULUM_CHECKPOINT_SCHEMA:
        raise ValueError("Formal V8 checkpoint provenance has the wrong checkpoint schema")
    if provenance.get("release_variant") != _FORMAL_V8_FULLIMAGE_CURRICULUM_RELEASE:
        raise ValueError("Formal V8 checkpoint provenance has the wrong release variant")
    static = provenance.get("static_contract")
    if not isinstance(static, dict) or static.get("schema_version") != (
        _FORMAL_V8_FULLIMAGE_CURRICULUM_STATIC_SCHEMA
    ):
        raise ValueError("Formal V8 checkpoint provenance has the wrong static-contract schema")
    data = provenance.get("data")
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != _FORMAL_V8_FULLIMAGE_CURRICULUM_DATA_SCHEMA
        or data.get("materialized_manifest_schema_version")
        != _FORMAL_V8_FULLIMAGE_CURRICULUM_MANIFEST_SCHEMA
        or data.get("student_teacher_image_policy")
        != "same_clean_original_full_image_no_bbox_overlay_v2"
    ):
        raise ValueError("Formal V8 checkpoint provenance has the wrong VQA14K data schema")
    compressor = provenance.get("vision_token_compressor")
    compressor_curriculum = (
        compressor.get("curriculum") if isinstance(compressor, dict) else None
    )
    if (
        not isinstance(compressor, dict)
        or not isinstance(compressor_curriculum, dict)
        or compressor.get("route_schema_version") != _FORMAL_V8_FULLIMAGE_CURRICULUM_ROUTE_SCHEMA
        or compressor.get("minimum_tokens_per_image") != 32
        or compressor.get("bbox_conditioned") is not False
        or compressor_curriculum.get("schedule_sha256")
        != _FORMAL_V8_FULLIMAGE_CURRICULUM_SCHEDULE_SHA256
    ):
        raise ValueError("Formal V8 checkpoint compressor is not the full-image curriculum compressor")
    curriculum = provenance.get("visual_token_curriculum")
    if not isinstance(curriculum, dict):
        raise ValueError("Formal V8 checkpoint provenance is missing visual_token_curriculum")
    if curriculum.get("schedule_sha256") != _FORMAL_V8_FULLIMAGE_CURRICULUM_SCHEDULE_SHA256:
        raise ValueError("Formal V8 checkpoint provenance has a different curriculum schedule hash")
    if curriculum.get("route_schema_version") != _FORMAL_V8_FULLIMAGE_CURRICULUM_ROUTE_SCHEMA:
        raise ValueError("Formal V8 checkpoint provenance has a different curriculum route schema")
    specification = curriculum.get("specification")
    if (
        not isinstance(specification, dict)
        or specification.get("total_optimizer_steps") != 175
        or specification.get("final_retention_bps") != 500
        or specification.get("minimum_tokens_per_image") != 32
        or specification.get("validation_control") != "forbidden_diagnostic_only"
    ):
        raise ValueError("Formal V8 checkpoint provenance has a non-canonical curriculum specification")
    sampler = provenance.get("sampler")
    if (
        not isinstance(sampler, dict)
        or sampler.get("curriculum_schedule_sha256")
        != _FORMAL_V8_FULLIMAGE_CURRICULUM_SCHEDULE_SHA256
        or sampler.get("prior_release_state_allowed") is not False
        or sampler.get("state_schema_version") != _FORMAL_V8_FULLIMAGE_CURRICULUM_SAMPLER_SCHEMA
    ):
        raise ValueError("Formal V8 checkpoint sampler is not bound to the curriculum or rejects no legacy state")


def _checkpoint_artifact_binding(path: str, relative_path: str) -> dict[str, object]:
    """Return the immutable identity used by the V6 checkpoint marker."""

    if relative_path != os.path.basename(relative_path):
        raise ValueError(f"Checkpoint artifact name must be a basename: {relative_path!r}")
    if not os.path.isfile(path) or os.path.islink(path):
        raise FileNotFoundError(f"Checkpoint artifact is not a regular file: {path}")
    size_bytes = os.path.getsize(path)
    if size_bytes <= 0:
        raise ValueError(f"Checkpoint artifact is empty: {path}")
    return {
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "sha256": _sha256_file(path),
    }


def _checkpoint_tree_artifacts(root: str, relative_root: str) -> list[dict[str, object]]:
    """Bind every regular file in a small checkpoint metadata tree."""

    root = os.path.abspath(root)
    if not os.path.isdir(root) or os.path.islink(root):
        raise FileNotFoundError(f"Checkpoint metadata tree is not a regular directory: {root}")
    artifacts = []
    for current, directories, files in os.walk(root, followlinks=False):
        symlinked = [name for name in (*directories, *files) if os.path.islink(os.path.join(current, name))]
        if symlinked:
            raise ValueError(f"Checkpoint metadata tree contains symlinks: {symlinked[:20]}")
        for filename in files:
            path = os.path.join(current, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative.startswith("../") or relative == "..":
                raise ValueError("Checkpoint metadata artifact escapes its tree")
            size_bytes = os.path.getsize(path)
            if size_bytes <= 0:
                raise ValueError(f"Checkpoint metadata artifact is empty: {path}")
            artifacts.append(
                {
                    "relative_path": f"{relative_root}/{relative}",
                    "size_bytes": size_bytes,
                    "sha256": _sha256_file(path),
                }
            )
    artifacts.sort(key=lambda item: item["relative_path"])
    if not artifacts or not any(item["relative_path"] == f"{relative_root}/config.json" for item in artifacts):
        raise RuntimeError("Checkpoint Hugging Face metadata tree is empty or has no config.json")
    return artifacts


def _artifact_inventory_sha256(artifacts: list[dict[str, object]]) -> str:
    payload = (json.dumps(artifacts, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_v6_checkpoint_free_space(checkpoint_parent: str, minimum_free_bytes: int) -> int:
    """Fail before a formal save rather than evicting a pending monitoring pin."""

    if (
        isinstance(minimum_free_bytes, bool)
        or not isinstance(minimum_free_bytes, int)
        or minimum_free_bytes <= 0
    ):
        raise ValueError("Formal V8 minimum_free_bytes must be a positive integer")
    free_bytes = shutil.disk_usage(checkpoint_parent).free
    if free_bytes < minimum_free_bytes:
        raise OSError(
            "Formal V8 checkpoint save stopped before writing because pending-monitoring "
            f"retention leaves {free_bytes} free bytes, below the immutable "
            f"minimum_free_bytes={minimum_free_bytes}"
        )
    return free_bytes


def _json_leaf_differences(left, right, prefix="") -> dict[str, tuple[object, object]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result[path] = (left.get(key, "<MISSING>"), right.get(key, "<MISSING>"))
            else:
                result.update(_json_leaf_differences(left[key], right[key], path))
        return result
    return {} if left == right else {prefix: (left, right)}


def _allow_exp3_resume_migration(saved: dict, expected: dict, actor_path: str) -> bool:
    """Authorize exactly the audited Exp3 validation-only source migration.

    The old checkpoint stays byte-for-byte unchanged.  The external manifest is
    hash-pinned by the launcher and may only bridge the two provenance leaves
    that necessarily change when the validation-tail fix is loaded.
    """
    manifest_path = os.environ.get("VERL_EXP3_RESUME_MIGRATION_MANIFEST")
    manifest_sha = os.environ.get("VERL_EXP3_RESUME_MIGRATION_SHA256")
    if not manifest_path and not manifest_sha:
        return False
    if not manifest_path or not manifest_sha or len(manifest_sha) != 64:
        raise RuntimeError("Exp3 resume migration requires both manifest path and SHA256")
    manifest_path = os.path.realpath(manifest_path)
    if not os.path.isfile(manifest_path) or _sha256_file(manifest_path) != manifest_sha:
        raise RuntimeError("Exp3 resume migration manifest is absent or hash-mismatched")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "vision_opd_exp3_resume_migration_v1" or manifest.get("passed") is not True:
        raise RuntimeError("Exp3 resume migration manifest did not pass")
    actor_real = os.path.realpath(actor_path)
    if actor_real != os.path.realpath(manifest.get("destination_actor_path", "")):
        raise RuntimeError("Exp3 resume migration actor path mismatch")
    provenance_path = os.path.join(actor_real, "checkpoint_provenance.json")
    if _sha256_file(provenance_path) != manifest.get("saved_checkpoint_provenance_sha256"):
        raise RuntimeError("Exp3 migrated checkpoint provenance bytes changed")

    old_fp = saved.get("runtime", {}).get("runtime_source_fingerprint")
    new_fp = expected.get("runtime", {}).get("runtime_source_fingerprint")
    if old_fp != _EXP3_MIGRATION_OLD_FINGERPRINT:
        raise RuntimeError(f"unsupported Exp3 migration source fingerprint: {old_fp!r}")
    if new_fp != manifest.get("new_runtime_source_fingerprint"):
        raise RuntimeError("Exp3 migration target fingerprint mismatch")
    if manifest.get("old_runtime_source_fingerprint") != _EXP3_MIGRATION_OLD_FINGERPRINT:
        raise RuntimeError("Exp3 migration manifest has an unsupported source fingerprint")

    old_audit = saved.get("runtime", {}).get("prelaunch_audit_sha256")
    new_audit = expected.get("runtime", {}).get("prelaunch_audit_sha256")
    allowed = {
        "runtime.runtime_source_fingerprint": (old_fp, new_fp),
        "runtime.prelaunch_audit_sha256": (old_audit, new_audit),
    }
    differences = _json_leaf_differences(saved, expected)
    if differences != allowed:
        raise RuntimeError(
            "Exp3 migration attempted to change non-allowlisted checkpoint provenance: "
            f"{json.dumps(differences, ensure_ascii=False, sort_keys=True)}"
        )
    if old_audit != manifest.get("old_prelaunch_audit_sha256"):
        raise RuntimeError("Exp3 migration old prelaunch audit mismatch")
    if new_audit != manifest.get("new_prelaunch_audit_sha256"):
        raise RuntimeError("Exp3 migration new prelaunch audit mismatch")
    return True


def _nested_exact_equal(left, right) -> bool:
    # FSDP1 SHARDED_STATE_DICT values are ShardedTensor objects.  They do not
    # implement elementwise equality, so compare both the global shard layout
    # and every rank-local payload explicitly.  Duck typing keeps this helper
    # compatible with the PyTorch public/private import-path changes.
    if all(hasattr(value, "local_shards") and hasattr(value, "metadata") for value in (left, right)):
        def metadata_signature(value):
            metadata = value.metadata()
            shards = tuple(
                (
                    tuple(shard.shard_offsets),
                    tuple(shard.shard_sizes),
                    str(shard.placement),
                )
                for shard in metadata.shards_metadata
            )
            properties = metadata.tensor_properties
            return (
                tuple(metadata.size),
                shards,
                str(properties.dtype),
                str(properties.layout),
                bool(properties.requires_grad),
                str(properties.memory_format),
                bool(properties.pin_memory),
            )

        if metadata_signature(left) != metadata_signature(right):
            return False
        left_local, right_local = left.local_shards(), right.local_shards()
        if len(left_local) != len(right_local):
            return False
        return all(
            (
                tuple(a.metadata.shard_offsets) == tuple(b.metadata.shard_offsets)
                and tuple(a.metadata.shard_sizes) == tuple(b.metadata.shard_sizes)
                and str(a.metadata.placement) == str(b.metadata.placement)
                and _nested_exact_equal(a.tensor, b.tensor)
            )
            for a, b in zip(left_local, right_local, strict=True)
        )
    if hasattr(left, "_local_tensor") and hasattr(right, "_local_tensor"):
        if tuple(getattr(left, "placements", ())) != tuple(getattr(right, "placements", ())):
            return False
        return _nested_exact_equal(left._local_tensor, right._local_tensor)
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        self.checkpoint_provenance = kwargs.pop("checkpoint_provenance", None)
        self.runtime_profile_state = kwargs.pop("runtime_profile_state", None)
        provenance_schema = (self.checkpoint_provenance or {}).get("schema_version")
        if provenance_schema in _FORBIDDEN_PRIOR_FORMAL_CHECKPOINT_SCHEMAS:
            raise ValueError(
                "This isolated V8 repository refuses legacy V6/V7 checkpoint provenance"
            )
        formal_v8 = provenance_schema == _FORMAL_V8_FULLIMAGE_CURRICULUM_CHECKPOINT_SCHEMA
        if formal_v8 != (self.runtime_profile_state is not None):
            raise ValueError(
                "Formal V8 full-image curriculum checkpoint provenance and "
                "runtime_profile_state must be provided together"
            )
        if formal_v8:
            _validate_formal_v8_fullimage_curriculum_provenance(self.checkpoint_provenance)

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )

    def _validate_checkpoint_provenance(self, local_path: str) -> None:
        if self.checkpoint_provenance is None:
            return
        remote_path = os.path.join(local_path, "checkpoint_provenance.json")
        local_manifest_path = copy_to_local(remote_path)
        if not os.path.isfile(local_manifest_path):
            raise FileNotFoundError(
                "Visual-compression checkpoint is missing checkpoint_provenance.json; "
                "refusing an ambiguous/legacy resume"
            )
        with open(local_manifest_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        formal_v8 = self._is_formal_v8_fullimage_curriculum()
        saved_schema = saved.get("schema_version") if isinstance(saved, dict) else None
        if formal_v8 and saved_schema != _FORMAL_V8_FULLIMAGE_CURRICULUM_CHECKPOINT_SCHEMA:
            if saved_schema in _FORBIDDEN_PRIOR_FORMAL_CHECKPOINT_SCHEMAS:
                raise ValueError(
                    "V8 full-image curriculum resume refuses a prior V6/V7 checkpoint schema"
                )
            raise ValueError(
                "V8 full-image curriculum resume requires its unique checkpoint schema, "
                f"got {saved_schema!r}"
            )
        migration_allowed = (
            not formal_v8
            and saved != self.checkpoint_provenance
            and _allow_exp3_resume_migration(saved, self.checkpoint_provenance, local_path)
        )
        if saved != self.checkpoint_provenance and not migration_allowed:
            saved_text = json.dumps(saved, ensure_ascii=False, sort_keys=True)
            expected_text = json.dumps(self.checkpoint_provenance, ensure_ascii=False, sort_keys=True)
            raise ValueError(
                "Visual-compression checkpoint provenance differs from the resolved run contract; refusing resume. "
                f"saved={saved_text}, expected={expected_text}"
            )

    def _atomic_complete_marker_enabled(self) -> bool:
        checkpoint = (self.checkpoint_provenance or {}).get("distributed_checkpoint", {})
        return bool(checkpoint.get("atomic_complete_marker", False))

    def _checkpoint_integrity_v2_enabled(self) -> bool:
        checkpoint = (self.checkpoint_provenance or {}).get("distributed_checkpoint", {})
        return checkpoint.get("integrity_schema") == "sha256_all_payloads_v2"

    def _allow_legacy_global_checkpoint(self) -> bool:
        return not self._checkpoint_integrity_v2_enabled()

    def _is_formal_v8_fullimage_curriculum(self) -> bool:
        return (
            (self.checkpoint_provenance or {}).get("schema_version")
            == _FORMAL_V8_FULLIMAGE_CURRICULUM_CHECKPOINT_SCHEMA
        )

    def _validate_checkpoint_runtime_profile(self, local_path: str) -> dict[str, object]:
        if not self._is_formal_v8_fullimage_curriculum():
            return {"passed": True, "formal_v8_fullimage_curriculum": False}
        if self.runtime_profile_state is None:
            raise RuntimeError("Formal V8 checkpoint load has no resolved runtime profile state")
        remote_path = os.path.join(local_path, _V6_RUNTIME_PROFILE_SIDECAR)
        local_state_path = copy_to_local(remote_path)
        if not os.path.isfile(local_state_path):
            raise FileNotFoundError("Formal V8 checkpoint is missing its runtime profile sidecar")
        with open(local_state_path, encoding="utf-8") as handle:
            saved_state = json.load(handle)
        from verl.utils.checkpoint.v6_profile_recovery import validate_v6_runtime_profile_resume

        audit = validate_v6_runtime_profile_resume(saved_state, self.runtime_profile_state)
        self.saved_runtime_profile_state = saved_state
        return {"formal_v8_fullimage_curriculum": True, **audit}

    def _validate_atomic_complete_marker(self, local_path: str) -> None:
        if not self._atomic_complete_marker_enabled():
            return
        marker_path = copy_to_local(os.path.join(local_path, _ATOMIC_COMPLETE_MARKER))
        if not os.path.isfile(marker_path):
            raise FileNotFoundError("Formal V8 checkpoint has no atomic completion marker")
        with open(marker_path, encoding="utf-8") as handle:
            marker = json.load(handle)
        if marker.get("schema_version") == ACTOR_MARKER_V2:
            if marker.get("world_size") != self.world_size:
                raise ValueError(
                    f"Checkpoint marker world_size={marker.get('world_size')} != runtime {self.world_size}"
                )
            if isinstance(marker.get("global_step"), bool) or not isinstance(marker.get("global_step"), int):
                raise ValueError("Checkpoint V2 marker has an invalid global_step")
            expected_core = sorted(
                [
                    f"{kind}_world_size_{self.world_size}_rank_{rank}.pt"
                    for rank in range(self.world_size)
                    for kind in ("model", "optim", "extra_state", "rollout_rng")
                ]
                + ["checkpoint_provenance.json", "fsdp_config.json", _V6_RUNTIME_PROFILE_SIDECAR]
            )
            artifacts = marker.get("artifacts")
            if not isinstance(artifacts, list):
                raise ValueError("Checkpoint V2 marker has no artifact inventory")
            artifact_paths = [
                item.get("relative_path") if isinstance(item, dict) else None for item in artifacts
            ]
            hf_paths = sorted(
                path for path in artifact_paths if isinstance(path, str) and path.startswith("huggingface/")
            )
            if not hf_paths or "huggingface/config.json" not in hf_paths:
                raise ValueError("Checkpoint V2 marker has no complete Hugging Face metadata inventory")
            validated = validate_artifact_inventory(
                local_path,
                artifacts,
                expected_paths=expected_core + hf_paths,
            )
            if marker.get("artifact_inventory_sha256") != canonical_sha256(validated):
                raise ValueError("Checkpoint V2 marker artifact inventory digest is invalid")
            if not self._checkpoint_integrity_v2_enabled():
                raise ValueError(
                    "Checkpoint uses full-integrity V2 but the resolved provenance does not authorize it"
                )
            return
        if marker.get("schema_version") != "verl_fsdp_atomic_complete_v1":
            raise ValueError("Formal V8 checkpoint completion marker has an invalid schema")
        if self._checkpoint_integrity_v2_enabled():
            raise ValueError("V8 full-integrity resume refuses a legacy partial-hash actor checkpoint")
        if marker.get("world_size") != self.world_size:
            raise ValueError(
                f"Formal V8 checkpoint marker world_size={marker.get('world_size')} != runtime {self.world_size}"
            )
        required_files = marker.get("required_files")
        if not isinstance(required_files, list) or len(required_files) != len(set(required_files)):
            raise ValueError("Formal V8 checkpoint completion marker has an invalid required-file inventory")
        if self._is_formal_v8_fullimage_curriculum():
            expected_files = [
                f"{kind}_world_size_{self.world_size}_rank_{rank}.pt"
                for rank in range(self.world_size)
                for kind in ("model", "optim", "extra_state", "rollout_rng")
            ] + ["checkpoint_provenance.json", "fsdp_config.json", _V6_RUNTIME_PROFILE_SIDECAR]
            if required_files != expected_files:
                raise ValueError("Formal V8 checkpoint marker has a non-canonical eight-rank file inventory")
            expected_model_files = [
                f"model_world_size_{self.world_size}_rank_{rank}.pt" for rank in range(self.world_size)
            ]
            model_artifacts = marker.get("model_artifacts")
            if not isinstance(model_artifacts, list) or len(model_artifacts) != self.world_size:
                raise ValueError("Formal V8 checkpoint marker has no complete model-shard hash inventory")
            if [item.get("relative_path") if isinstance(item, dict) else None for item in model_artifacts] != (
                expected_model_files
            ):
                raise ValueError("Formal V8 checkpoint marker model-shard order is non-canonical")
            for item in model_artifacts:
                if set(item) != {"relative_path", "size_bytes", "sha256"}:
                    raise ValueError("Formal V8 checkpoint marker model-shard binding is malformed")
                if (
                    isinstance(item["size_bytes"], bool)
                    or not isinstance(item["size_bytes"], int)
                    or item["size_bytes"] <= 0
                    or not isinstance(item["sha256"], str)
                    or len(item["sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in item["sha256"])
                ):
                    raise ValueError("Formal V8 checkpoint marker model-shard binding is malformed")
            local_binding = model_artifacts[self.rank]
            local_model_path = copy_to_local(os.path.join(local_path, local_binding["relative_path"]))
            if (
                os.path.getsize(local_model_path) != local_binding["size_bytes"]
                or _sha256_file(local_model_path) != local_binding["sha256"]
            ):
                raise ValueError("Formal V8 checkpoint model shard differs from its completion marker")
            hf_artifacts = marker.get("huggingface_artifacts")
            if not isinstance(hf_artifacts, list) or not hf_artifacts:
                raise ValueError("Formal V8 checkpoint marker has no Hugging Face metadata inventory")
            relative_paths = []
            for item in hf_artifacts:
                if (
                    not isinstance(item, dict)
                    or set(item) != {"relative_path", "size_bytes", "sha256"}
                    or not isinstance(item["relative_path"], str)
                    or not item["relative_path"].startswith("huggingface/")
                    or ".." in item["relative_path"].split("/")
                    or isinstance(item["size_bytes"], bool)
                    or not isinstance(item["size_bytes"], int)
                    or item["size_bytes"] <= 0
                    or not isinstance(item["sha256"], str)
                    or len(item["sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in item["sha256"])
                ):
                    raise ValueError("Formal V8 checkpoint Hugging Face metadata binding is malformed")
                relative_paths.append(item["relative_path"])
            if relative_paths != sorted(set(relative_paths)) or "huggingface/config.json" not in relative_paths:
                raise ValueError("Formal V8 checkpoint Hugging Face metadata inventory is non-canonical")
            if marker.get("huggingface_bundle_sha256") != _artifact_inventory_sha256(hf_artifacts):
                raise ValueError("Formal V8 checkpoint marker has an invalid Hugging Face metadata bundle hash")
            for item in hf_artifacts:
                local_hf_path = copy_to_local(os.path.join(local_path, *item["relative_path"].split("/")))
                if (
                    not os.path.isfile(local_hf_path)
                    or os.path.getsize(local_hf_path) != item["size_bytes"]
                    or _sha256_file(local_hf_path) != item["sha256"]
                ):
                    raise ValueError(
                        "Formal V8 checkpoint Hugging Face metadata differs from its completion marker: "
                        f"{item['relative_path']}"
                    )
        for filename in required_files:
            if not isinstance(filename, str) or filename != os.path.basename(filename):
                raise ValueError(f"Invalid checkpoint marker filename: {filename!r}")
            if not os.path.isfile(copy_to_local(os.path.join(local_path, filename))):
                raise FileNotFoundError(f"Formal V8 checkpoint is incomplete: missing {filename}")
        provenance_path = copy_to_local(os.path.join(local_path, "checkpoint_provenance.json"))
        if marker.get("checkpoint_provenance_sha256") != _sha256_file(provenance_path):
            raise ValueError("Formal V8 checkpoint completion marker does not bind the saved provenance")
        if self._is_formal_v8_fullimage_curriculum():
            if _V6_RUNTIME_PROFILE_SIDECAR not in required_files:
                raise ValueError("Formal V8 checkpoint marker does not require the runtime profile sidecar")
            runtime_path = copy_to_local(os.path.join(local_path, _V6_RUNTIME_PROFILE_SIDECAR))
            if marker.get("runtime_profile_state_sha256") != _sha256_file(runtime_path):
                raise ValueError("Formal V8 checkpoint completion marker does not bind runtime profile state")

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        self._validate_atomic_complete_marker(local_path)
        self._validate_checkpoint_provenance(local_path)
        runtime_profile_audit = self._validate_checkpoint_runtime_profile(local_path)

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        # every rank download its own checkpoint
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        model_loaded = optimizer_loaded = False
        model_state_exact = optimizer_state_exact = True
        formal_v8 = self._is_formal_v8_fullimage_curriculum()
        if formal_v8 and os.environ.get("VERL_EXP3_FRESH_LOAD_AUDIT_DIR"):
            raise RuntimeError("Formal V8 checkpoint load forbids the archived Exp3 fresh-load audit")
        require_fresh_exact = bool(
            os.environ.get(
                "VERL_V6_FRESH_LOAD_AUDIT_DIR"
                if formal_v8
                else "VERL_EXP3_FRESH_LOAD_AUDIT_DIR"
            )
        )
        extra_state_dict = None
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if self.should_load_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                incompatible = self.model.load_state_dict(model_state_dict)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "FSDP model checkpoint load was not exact: "
                        f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                    )
                model_loaded = True
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                optimizer_loaded = True
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

            if require_fresh_exact:
                if self.should_load_model:
                    model_state_exact = _nested_exact_equal(model_state_dict, self.model.state_dict())
                if self.should_load_optimizer:
                    optimizer_state_exact = _nested_exact_equal(optimizer_state_dict, self.optimizer.state_dict())

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            if formal_v8:
                if "v6_runtime_profile" not in extra_state_dict:
                    raise RuntimeError("Formal V8 extra_state shard is missing its runtime profile")
                if extra_state_dict["v6_runtime_profile"] != getattr(
                    self, "saved_runtime_profile_state", None
                ):
                    raise RuntimeError(
                        "Formal V8 extra_state runtime profile differs from its hash-bound sidecar"
                    )
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

        # wait for everyone to load checkpoints
        torch.distributed.barrier()
        scheduler_exact = True
        rng_exact = True
        if extra_state_dict is not None:
            expected_scheduler = extra_state_dict.get("lr_scheduler")
            actual_scheduler = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            scheduler_exact = _nested_exact_equal(expected_scheduler, actual_scheduler)
            expected_rng = extra_state_dict.get("rng")
            actual_rng = self.get_rng_state() if expected_rng is not None else None
            rng_exact = _nested_exact_equal(expected_rng, actual_rng)
        self.last_load_audit = {
            "model_loaded_strict": (not self.should_load_model) or model_loaded,
            "model_shard_values_exact": (not self.should_load_model) or model_state_exact,
            "optimizer_loaded": (not self.should_load_optimizer) or optimizer_loaded,
            "optimizer_state_values_exact": (not self.should_load_optimizer) or optimizer_state_exact,
            "scheduler_exact": scheduler_exact,
            "trainer_rng_exact": rng_exact,
            "runtime_profile_resume_authorized": bool(runtime_profile_audit.get("passed", False)),
        }
        if not all(self.last_load_audit.values()):
            raise RuntimeError(f"Fresh checkpoint load audit failed: {self.last_load_audit}")
        if self.rank == 0:
            self.rebuild_previous_saved_paths(local_path)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        formal_v8 = self._is_formal_v8_fullimage_curriculum()
        integrity_v2 = self._checkpoint_integrity_v2_enabled()
        if formal_v8:
            if not (self.should_save_model and self.should_save_optimizer and self.should_save_extra):
                raise RuntimeError(
                    "Formal V8 checkpoint must save model, optimizer, and runtime extra_state on all eight ranks"
                )
            from verl.utils.checkpoint.v6_profile_recovery import validate_v6_runtime_profile_resume

            validate_v6_runtime_profile_resume(self.runtime_profile_state, self.runtime_profile_state)

        # record the previous global step
        self.previous_global_step = global_step

        retention_error = None
        if self.rank == 0:
            try:
                self.ensure_checkpoint_capacity(max_ckpt_to_keep, incoming_path=local_path)
                if formal_v8:
                    minimum_free_bytes = int(
                        self.checkpoint_provenance["distributed_checkpoint"]["minimum_free_bytes"]
                    )
                    checkpoint_parent = os.path.dirname(os.path.abspath(local_path))
                    _require_v6_checkpoint_free_space(checkpoint_parent, minimum_free_bytes)
            except Exception as exc:
                retention_error = f"{type(exc).__name__}: {exc}"
        retention_errors = [retention_error]
        torch.distributed.broadcast_object_list(retention_errors, src=0)
        if retention_errors[0] is not None:
            raise RuntimeError(f"Checkpoint retention/capacity preflight failed: {retention_errors[0]}")

        local_path = local_mkdir_safe(local_path)
        if self.rank == 0 and self._atomic_complete_marker_enabled():
            stale_marker = os.path.join(local_path, _ATOMIC_COMPLETE_MARKER)
            if os.path.isfile(stale_marker):
                if integrity_v2:
                    raise FileExistsError(
                        "Refusing to overwrite a checkpoint directory that already has a completion marker"
                    )
                os.remove(stale_marker)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    model_state_dict = self.model.state_dict()
                    if integrity_v2:
                        atomic_torch_save(model_state_dict, model_path)
                    else:
                        torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    if integrity_v2:
                        atomic_torch_save(optimizer_state_dict, optim_path)
                    else:
                        torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    if formal_v8:
                        extra_state_dict["v6_runtime_profile"] = self.runtime_profile_state
                    if integrity_v2:
                        atomic_torch_save(extra_state_dict, extra_path)
                    else:
                        torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            if integrity_v2:
                atomic_json_dump(asdict(fsdp_config), fsdp_config_path)
            else:
                with open(fsdp_config_path, "w") as f:
                    json.dump(asdict(fsdp_config), f, indent=4)
            if self.checkpoint_provenance is not None:
                provenance_path = os.path.join(local_path, "checkpoint_provenance.json")
                if integrity_v2:
                    atomic_json_dump(self.checkpoint_provenance, provenance_path)
                else:
                    temporary_path = provenance_path + ".tmp"
                    with open(temporary_path, "w", encoding="utf-8") as handle:
                        json.dump(self.checkpoint_provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
                        handle.write("\n")
                    os.replace(temporary_path, provenance_path)
            if formal_v8:
                runtime_profile_path = os.path.join(local_path, _V6_RUNTIME_PROFILE_SIDECAR)
                if integrity_v2:
                    atomic_json_dump(self.runtime_profile_state, runtime_profile_path)
                else:
                    temporary_path = runtime_profile_path + ".tmp"
                    with open(temporary_path, "w", encoding="utf-8") as handle:
                        json.dump(self.runtime_profile_state, handle, ensure_ascii=False, indent=2, sort_keys=True)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary_path, runtime_profile_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    # Handle different transformers versions for Vision2Seq models
                    import transformers
                    from packaging import version

                    if version.parse(transformers.__version__) >= version.parse("4.54.0"):
                        # transformers >= 4.54.0 uses AutoModelForImageTextToText
                        from transformers import AutoModelForImageTextToText

                        auto_model_cls = AutoModelForImageTextToText
                    else:
                        # transformers < 4.54.0 uses AutoModelForVision2Seq
                        from transformers import AutoModelForVision2Seq

                        auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if self._atomic_complete_marker_enabled():
            torch.distributed.barrier()
            if integrity_v2:
                local_artifacts = []
                local_error = None
                try:
                    for kind in ("model", "optim", "extra_state", "rollout_rng"):
                        relative = f"{kind}_world_size_{self.world_size}_rank_{self.rank}.pt"
                        local_artifacts.append(artifact_binding(local_path, relative))
                except Exception as exc:
                    local_error = f"rank={self.rank} {type(exc).__name__}: {exc}"
                gathered = [None] * self.world_size
                torch.distributed.all_gather_object(
                    gathered,
                    {"rank": self.rank, "artifacts": local_artifacts, "error": local_error},
                )
                gathered_errors = [item["error"] for item in gathered if item["error"]]
                if gathered_errors:
                    raise RuntimeError(f"Checkpoint V2 payload hash binding failed: {gathered_errors}")
                if [item["rank"] for item in gathered] != list(range(self.world_size)):
                    raise RuntimeError("Checkpoint V2 payload gather returned non-canonical ranks")
                marker_error = None
                if self.rank == 0:
                    try:
                        artifacts = [binding for item in gathered for binding in item["artifacts"]]
                        for relative in (
                            "checkpoint_provenance.json",
                            "fsdp_config.json",
                            _V6_RUNTIME_PROFILE_SIDECAR,
                        ):
                            artifacts.append(artifact_binding(local_path, relative))
                        hf_artifacts = _checkpoint_tree_artifacts(
                            os.path.join(local_path, "huggingface"), "huggingface"
                        )
                        artifacts.extend(hf_artifacts)
                        artifacts.sort(key=lambda item: item["relative_path"])
                        if len(artifacts) != len({item["relative_path"] for item in artifacts}):
                            raise RuntimeError("Checkpoint V2 artifact inventory contains duplicates")
                        fsync_regular_tree(local_path)
                        marker = {
                            "schema_version": ACTOR_MARKER_V2,
                            "world_size": self.world_size,
                            "global_step": int(global_step),
                            "artifacts": artifacts,
                            "artifact_inventory_sha256": canonical_sha256(artifacts),
                        }
                        atomic_json_dump(marker, os.path.join(local_path, _ATOMIC_COMPLETE_MARKER))
                    except Exception as exc:
                        marker_error = f"{type(exc).__name__}: {exc}"
                marker_errors = [marker_error]
                torch.distributed.broadcast_object_list(marker_errors, src=0)
                if marker_errors[0] is not None:
                    raise RuntimeError(f"FSDP checkpoint V2 marker commit failed: {marker_errors[0]}")
                torch.distributed.barrier()
                if self.rank == 0:
                    self.register_checkpoint(local_path, max_ckpt_to_keep)
                return
            model_artifacts = None
            if formal_v8:
                local_model_name = f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
                local_model_binding = None
                local_model_binding_error = None
                try:
                    local_model_binding = _checkpoint_artifact_binding(
                        os.path.join(local_path, local_model_name),
                        local_model_name,
                    )
                except Exception as exc:
                    local_model_binding_error = f"rank={self.rank} {type(exc).__name__}: {exc}"
                gathered_model_bindings = [None] * self.world_size
                torch.distributed.all_gather_object(
                    gathered_model_bindings,
                    {
                        "rank": self.rank,
                        "binding": local_model_binding,
                        "error": local_model_binding_error,
                    },
                )
                model_binding_errors = [item["error"] for item in gathered_model_bindings if item["error"]]
                if model_binding_errors:
                    raise RuntimeError(f"Formal V8 model-shard hash binding failed: {model_binding_errors}")
                if [item["rank"] for item in gathered_model_bindings] != list(range(self.world_size)):
                    raise RuntimeError("Formal V8 model-shard hash gather returned non-canonical ranks")
                model_artifacts = [item["binding"] for item in gathered_model_bindings]
            marker_error = None
            if self.rank == 0:
                try:
                    required_files = []
                    for rank in range(self.world_size):
                        if self.should_save_model:
                            required_files.append(f"model_world_size_{self.world_size}_rank_{rank}.pt")
                        if self.should_save_optimizer:
                            required_files.append(f"optim_world_size_{self.world_size}_rank_{rank}.pt")
                        if self.should_save_extra:
                            required_files.append(f"extra_state_world_size_{self.world_size}_rank_{rank}.pt")
                        if (self.checkpoint_provenance or {}).get("schema_version") == (
                            _FORMAL_V8_FULLIMAGE_CURRICULUM_CHECKPOINT_SCHEMA
                        ):
                            required_files.append(f"rollout_rng_world_size_{self.world_size}_rank_{rank}.pt")
                    required_files.extend(["checkpoint_provenance.json", "fsdp_config.json"])
                    if formal_v8:
                        required_files.append(_V6_RUNTIME_PROFILE_SIDECAR)
                    missing = [
                        name for name in required_files if not os.path.isfile(os.path.join(local_path, name))
                    ]
                    if missing:
                        raise RuntimeError(f"Refusing to mark an incomplete FSDP checkpoint: {missing[:20]}")
                    marker_path = os.path.join(local_path, _ATOMIC_COMPLETE_MARKER)
                    temporary_path = marker_path + ".tmp"
                    marker = {
                        "schema_version": "verl_fsdp_atomic_complete_v1",
                        "world_size": self.world_size,
                        "global_step": int(global_step),
                        "required_files": required_files,
                        "checkpoint_provenance_sha256": _sha256_file(
                            os.path.join(local_path, "checkpoint_provenance.json")
                        ),
                    }
                    if formal_v8:
                        marker["runtime_profile_state_sha256"] = _sha256_file(
                            os.path.join(local_path, _V6_RUNTIME_PROFILE_SIDECAR)
                        )
                        expected_model_names = [
                            f"model_world_size_{self.world_size}_rank_{rank}.pt"
                            for rank in range(self.world_size)
                        ]
                        if [item.get("relative_path") for item in model_artifacts] != expected_model_names:
                            raise RuntimeError("Formal V8 model-shard hash gather returned a non-canonical inventory")
                        marker["model_artifacts"] = model_artifacts
                        marker["huggingface_artifacts"] = _checkpoint_tree_artifacts(
                            os.path.join(local_path, "huggingface"),
                            "huggingface",
                        )
                        marker["huggingface_bundle_sha256"] = _artifact_inventory_sha256(
                            marker["huggingface_artifacts"]
                        )
                    with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
                        json.dump(marker, handle, ensure_ascii=False, indent=2, sort_keys=True)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary_path, marker_path)
                except Exception as exc:
                    marker_error = f"{type(exc).__name__}: {exc}"
            marker_errors = [marker_error]
            torch.distributed.broadcast_object_list(marker_errors, src=0)
            if marker_errors[0] is not None:
                raise RuntimeError(f"FSDP checkpoint marker commit failed: {marker_errors[0]}")
            torch.distributed.barrier()

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)
