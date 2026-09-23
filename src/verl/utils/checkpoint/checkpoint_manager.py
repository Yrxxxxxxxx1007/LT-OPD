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

import json
import os
import random
import re
import shutil

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.trainer.config import CheckpointConfig
from verl.utils.device import get_device_name, get_torch_device


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads the following states in a SPMD way:
    - model
    - optimizer
    - lr_scheduler
    - extra_states

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig | CheckpointConfig = None,
    ):
        self.checkpoint_config = checkpoint_config
        checkpoint_load_contents = checkpoint_config.get("load_contents", None) if checkpoint_config else None
        checkpoint_save_contents = checkpoint_config.get("save_contents", None) if checkpoint_config else None
        if checkpoint_load_contents is None:
            checkpoint_load_contents = ["model", "optimizer", "extra"]
        if checkpoint_save_contents is None:
            checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_load_contents = checkpoint_load_contents
        self.checkpoint_save_contents = checkpoint_save_contents

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    @property
    def should_save_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_save_contents, indicating the model state should be saved.
        """
        return "model" in self.checkpoint_save_contents

    @property
    def should_save_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_save_contents, indicating the optimizer state should be saved.
        """
        return "optimizer" in self.checkpoint_save_contents

    @property
    def should_save_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_save_contents, indicating the extra state should be saved.
        """
        return "extra" in self.checkpoint_save_contents

    @property
    def should_save_hf_model(self) -> bool:
        """
        Returns True if 'hf_model' is in checkpoint_save_contents, indicating the model should be converted to hf
        model and saved.
        """
        return "hf_model" in self.checkpoint_save_contents

    @property
    def should_load_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_load_contents, indicating the model state should be loaded.
        """
        return "model" in self.checkpoint_load_contents

    @property
    def should_load_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_load_contents, indicating the optimizer state should be loaded.
        """
        return "optimizer" in self.checkpoint_load_contents

    @property
    def should_load_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_load_contents, indicating the extra state should be loaded.
        """
        return "extra" in self.checkpoint_load_contents

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        raise NotImplementedError

    def save_checkpoint(
        self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None
    ):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    _GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")
    _COMPLETION_MARKER = ".checkpoint_complete"

    def _allow_legacy_global_checkpoint(self) -> bool:
        """Whether retention may treat a partial-integrity V1 marker as committed."""

        return True

    @classmethod
    def _checkpoint_layout(cls, checkpoint_path: str):
        """Return ``(root, role)`` for ``root/global_step_N/role`` paths."""

        checkpoint_path = os.path.abspath(checkpoint_path)
        step_path = os.path.dirname(checkpoint_path)
        if cls._GLOBAL_STEP_PATTERN.fullmatch(os.path.basename(step_path)) is None:
            return None
        return os.path.dirname(step_path), os.path.basename(checkpoint_path)

    def rebuild_previous_saved_paths(self, checkpoint_path: str):
        """Rebuild retention state from existing numeric ``global_step_*`` paths.

        Worker recreation during resume resets the in-memory queue.  Scanning the
        checkpoint root prevents retention from forgetting pre-resume saves.
        Only the same role directory (for example ``actor``) is included.  A
        directory created by a failed save is *not* a checkpoint: it becomes
        eligible only after the controller has saved ``data.pt`` and either
        committed the completion marker or advanced the atomic latest tracker.
        """

        layout = self._checkpoint_layout(checkpoint_path)
        if layout is None:
            return list(self.previous_saved_paths)
        checkpoint_root, role = layout
        from verl.utils.checkpoint.integrity import validate_global_checkpoint

        tracker_step = None
        tracker_path = get_checkpoint_tracker_filename(checkpoint_root)
        if os.path.isfile(tracker_path):
            try:
                with open(tracker_path, "rb") as handle:
                    tracker_step = int(handle.read().decode().strip())
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid checkpoint tracker at {tracker_path}: {exc}") from exc

        discovered = []
        if os.path.isdir(checkpoint_root):
            for entry in os.scandir(checkpoint_root):
                match = self._GLOBAL_STEP_PATTERN.fullmatch(entry.name)
                role_path = os.path.join(entry.path, role)
                if match is None or not entry.is_dir() or not os.path.isdir(role_path):
                    continue
                step = int(match.group(1))
                has_data_state = os.path.isfile(os.path.join(entry.path, "data.pt"))
                has_completion_marker = os.path.isfile(os.path.join(entry.path, self._COMPLETION_MARKER))
                if has_data_state and has_completion_marker:
                    validate_global_checkpoint(
                        entry.path,
                        expected_step=step,
                        allow_legacy_marker=self._allow_legacy_global_checkpoint(),
                    )
                    discovered.append((step, os.path.abspath(role_path)))
        discovered.sort(key=lambda item: (item[0], item[1]))
        self.previous_saved_paths = [path for _, path in discovered]
        return list(self.previous_saved_paths)

    def _best_protected_paths(self, checkpoint_path: str) -> set[str]:
        layout = self._checkpoint_layout(checkpoint_path)
        if layout is None:
            return set()
        checkpoint_root, role = layout
        metadata_path = os.path.join(checkpoint_root, "best_checkpoint.json")
        if not os.path.isfile(metadata_path):
            return set()
        try:
            with open(metadata_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            best_step_path = os.path.abspath(os.fspath(payload["checkpoint_path"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid best-checkpoint metadata at {metadata_path}: {exc}") from exc
        if (
            os.path.dirname(best_step_path) != os.path.abspath(checkpoint_root)
            or self._GLOBAL_STEP_PATTERN.fullmatch(os.path.basename(best_step_path)) is None
        ):
            raise ValueError(
                f"best_checkpoint.json points outside the checkpoint root or to a non-step path: {best_step_path}"
            )
        protected = os.path.join(best_step_path, role)
        if not os.path.isdir(protected):
            raise FileNotFoundError(
                f"best_checkpoint.json points to a checkpoint missing its {role!r} state: {protected}"
            )
        return {protected}

    def _monitoring_protected_paths(self, checkpoint_path: str) -> set[str]:
        """Protect formal V6 actor checkpoints until a receipt is hash-valid."""

        layout = self._checkpoint_layout(checkpoint_path)
        if layout is None:
            return set()
        checkpoint_root, role = layout
        if role != "actor" or not os.path.isdir(os.path.join(checkpoint_root, "..", "monitoring")):
            return set()
        from verl.trainer.ppo.v6_monitoring import pending_monitoring_actor_paths

        return pending_monitoring_actor_paths(checkpoint_root)

    def _trim_previous_saved_paths(self, target_count: int, protected_paths: set[str]) -> None:
        self.previous_saved_paths = [
            os.path.abspath(path) for path in self.previous_saved_paths if os.path.isdir(os.path.abspath(path))
        ]
        remove_count = max(0, len(self.previous_saved_paths) - target_count)
        victims = [path for path in self.previous_saved_paths if path not in protected_paths][:remove_count]
        self.remove_previous_save_local_path(victims)
        victim_set = set(victims)
        self.previous_saved_paths = [path for path in self.previous_saved_paths if path not in victim_set]
        if len(self.previous_saved_paths) > target_count:
            print(
                "Checkpoint retention is over capacity because protected checkpoints cannot be removed: "
                f"target={target_count}, kept={self.previous_saved_paths}"
            )

    def ensure_checkpoint_capacity(self, max_ckpt_to_keep: int, incoming_path: str = None):
        """
        Remove old checkpoints to make room for a new one, keeping a safety buffer.

        With max_ckpt_to_keep=1, this does nothing - we keep the existing checkpoint
        until the new save completes successfully (handled by register_checkpoint).
        For max_ckpt_to_keep >= 2, we keep (max_ckpt_to_keep - 1) checkpoints before save.
        A best checkpoint consumes one retention slot and is never selected as a
        victim.  If max_ckpt_to_keep=1 and best differs from the new checkpoint,
        correctness wins over the cap and both are retained.  The latest
        completed checkpoint is also protected until the incoming save succeeds,
        so the tracker never points to a pre-emptively deleted directory.
        """
        if not (max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 1):
            return
        if incoming_path is not None:
            self.rebuild_previous_saved_paths(incoming_path)
            protected_paths = self._best_protected_paths(incoming_path)
            protected_paths.update(self._monitoring_protected_paths(incoming_path))
            # The latest completed checkpoint remains the tracker target until
            # the incoming save and its tracker update both finish.  Protect it
            # during the pre-save capacity pass even when an older best
            # checkpoint already occupies the other retention slot.
            layout = self._checkpoint_layout(incoming_path)
            if layout is not None:
                checkpoint_root, role = layout
                tracker_path = get_checkpoint_tracker_filename(checkpoint_root)
                if os.path.isfile(tracker_path):
                    try:
                        with open(tracker_path, "rb") as handle:
                            tracker_step = int(handle.read().decode().strip())
                    except (OSError, UnicodeDecodeError, ValueError) as exc:
                        raise ValueError(f"Invalid checkpoint tracker at {tracker_path}: {exc}") from exc
                    tracker_role_path = os.path.abspath(
                        os.path.join(checkpoint_root, f"global_step_{tracker_step}", role)
                    )
                    if not os.path.isdir(tracker_role_path):
                        raise FileNotFoundError(
                            f"Checkpoint tracker points to a missing {role!r} state: {tracker_role_path}"
                        )
                    from verl.utils.checkpoint.integrity import validate_global_checkpoint

                    validate_global_checkpoint(
                        os.path.dirname(tracker_role_path),
                        expected_step=tracker_step,
                        allow_legacy_marker=self._allow_legacy_global_checkpoint(),
                    )
                    protected_paths.add(tracker_role_path)
            # A marker may have been atomically committed immediately before a
            # crash that prevented the latest-tracker replace.  It is a fully
            # written checkpoint too, so keep the newest discovered completed
            # path until a later successful save reconciles retention.
            if self.previous_saved_paths:
                protected_paths.add(self.previous_saved_paths[-1])
        else:
            protected_paths = set()
        self._trim_previous_saved_paths(max_ckpt_to_keep - 1, protected_paths)

    def register_checkpoint(self, new_path: str, max_ckpt_to_keep: int):
        """
        Register a successfully saved checkpoint and enforce retention limit.

        Adds the new checkpoint path to tracking and removes excess old
        checkpoints beyond max_ckpt_to_keep.
        """
        new_path = os.path.abspath(new_path)
        self.rebuild_previous_saved_paths(new_path)
        if new_path not in self.previous_saved_paths:
            self.previous_saved_paths.append(new_path)
        layout = self._checkpoint_layout(new_path)
        if layout is not None:
            _, role = layout
            step_path = os.path.dirname(new_path)
            # The FSDP worker returns before the controller writes data.pt and
            # atomically commits the global checkpoint.  Never evict the last
            # valid tracker target while this incoming checkpoint is partial.
            if not (
                os.path.isfile(os.path.join(step_path, "data.pt"))
                and os.path.isfile(os.path.join(step_path, self._COMPLETION_MARKER))
                and os.path.basename(new_path) == role
            ):
                return
            from verl.utils.checkpoint.integrity import validate_global_checkpoint

            step = int(self._GLOBAL_STEP_PATTERN.fullmatch(os.path.basename(step_path)).group(1))
            validate_global_checkpoint(
                step_path,
                expected_step=step,
                allow_legacy_marker=self._allow_legacy_global_checkpoint(),
            )
        if not (max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0):
            return
        protected_paths = self._best_protected_paths(new_path)
        protected_paths.update(self._monitoring_protected_paths(new_path))
        protected_paths.add(new_path)
        self._trim_previous_saved_paths(max_ckpt_to_keep, protected_paths)

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if get_device_name() != "cpu":
            rng_state[get_device_name()] = get_torch_device().get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_state[get_device_name()])


def find_latest_ckpt_path(
    path,
    directory_format="global_step_{}",
    *,
    allow_legacy_marker: bool = True,
):
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)

    # Preserve upstream compatibility for legacy checkpoint layouts. V7 opts
    # into the strict branch explicitly and gains crash reconciliation only
    # because every candidate is fully SHA-256 bound.
    if allow_legacy_marker:
        if not os.path.exists(tracker_file):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"Checkpoint tracker file does not exist: {tracker_file}")
            return None
        with open(tracker_file, "rb") as f:
            iteration = int(f.read().decode())
        ckpt_path = os.path.join(path, directory_format.format(iteration))
        if not os.path.exists(ckpt_path):
            print("Checkpoint does not exist: %s", ckpt_path)
            return None
        print("Found checkpoint: %s", ckpt_path)
        return ckpt_path

    if directory_format != "global_step_{}":
        raise ValueError("Strict V2 checkpoint reconciliation requires the canonical global_step layout")
    from verl.utils.checkpoint.integrity import atomic_text_write, validate_global_checkpoint

    checkpoint_root = os.path.abspath(os.fspath(path))
    if os.path.lexists(checkpoint_root) and (
        not os.path.isdir(checkpoint_root) or os.path.islink(checkpoint_root)
    ):
        raise ValueError(f"Strict checkpoint root must be a regular directory: {checkpoint_root}")

    tracker_step = None
    if os.path.lexists(tracker_file):
        if not os.path.isfile(tracker_file) or os.path.islink(tracker_file):
            raise ValueError(f"Strict checkpoint tracker must be a regular file: {tracker_file}")
        try:
            with open(tracker_file, "rb") as handle:
                raw_tracker = handle.read().decode().strip()
            if not raw_tracker.isdigit() or int(raw_tracker) <= 0:
                raise ValueError("tracker is not a positive step integer")
            tracker_step = int(raw_tracker)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid strict checkpoint tracker at {tracker_file}: {exc}") from exc

    candidates: list[tuple[int, str]] = []
    if os.path.isdir(checkpoint_root):
        for entry in os.scandir(checkpoint_root):
            match = BaseCheckpointManager._GLOBAL_STEP_PATTERN.fullmatch(entry.name)
            if match is None or not entry.is_dir(follow_symlinks=False):
                continue
            step = int(match.group(1))
            if tracker_step is not None and step < tracker_step:
                continue
            actor_path = os.path.join(entry.path, "actor")
            marker_path = os.path.join(entry.path, BaseCheckpointManager._COMPLETION_MARKER)
            # Retention removes the role directory after a newer checkpoint is
            # committed. Such historical shells are not resume candidates.
            if (
                not os.path.isdir(actor_path)
                or os.path.islink(actor_path)
                or not os.path.isfile(marker_path)
            ):
                continue
            try:
                validate_global_checkpoint(
                    entry.path,
                    expected_step=step,
                    allow_legacy_marker=False,
                    validate_actor_payloads=True,
                )
            except (OSError, ValueError) as exc:
                raise ValueError(f"Invalid committed V2 checkpoint at {entry.path}: {exc}") from exc
            candidates.append((step, os.path.abspath(entry.path)))

    if tracker_step is not None and not any(step == tracker_step for step, _ in candidates):
        raise ValueError(
            f"Strict checkpoint tracker points to a missing, pruned, or invalid step: {tracker_step}"
        )
    if not candidates:
        if tracker_step is None:
            print(f"Checkpoint tracker file does not exist: {tracker_file}")
            return None
        raise ValueError("Strict checkpoint tracker has no valid committed target")

    iteration, ckpt_path = max(candidates, key=lambda item: item[0])
    if tracker_step != iteration:
        is_writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_writer:
            atomic_text_write(f"{iteration}\n", tracker_file)
            print(
                "Reconciled checkpoint tracker from committed V2 manifest: "
                f"{tracker_step!r} -> {iteration}"
            )
    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")


def should_save_ckpt_esi(max_steps_duration: float, save_ckpt_duration: float = 60, redundant_time: float = 0) -> bool:
    """
    Determine if checkpoint should be saved based on capacity esi expiration.

    Args:
        max_steps_duration: Max estimated time (seconds) required to complete one training step
        save_ckpt_duration: Estimated time (seconds) required to save checkpoint (default: 60)
        redundant_time: Additional buffer time (seconds) for unexpected delays (default: 0)
    """
    exp_ts_mlp = os.getenv("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # vemlp
    exp_ts_aws = os.getenv("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # aws
    if exp_ts_mlp:
        try:
            import time

            remaining = float(exp_ts_mlp) - time.time()
        except ValueError:
            return False
        return (
            remaining > 0
            and max_steps_duration > 0
            and remaining <= save_ckpt_duration + max_steps_duration + redundant_time
        )
    elif exp_ts_aws:
        from datetime import datetime, timedelta

        expiration_time = datetime.fromtimestamp(int(exp_ts_aws))
        time_difference = expiration_time - datetime.now()
        threshold_minutes = (save_ckpt_duration + max_steps_duration + redundant_time) / 60
        return time_difference < timedelta(minutes=threshold_minutes)
    else:
        return False
