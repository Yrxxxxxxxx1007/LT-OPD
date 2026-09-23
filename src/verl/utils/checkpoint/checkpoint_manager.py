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

    def remove_previous_save_local_path(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            if os.path.isdir(path):
                shutil.rmtree(path)

    _GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")
    _COMPLETION_MARKER = ".checkpoint_complete"

    @classmethod
    def _checkpoint_layout(cls, checkpoint_path: str):
        step_path = os.path.dirname(os.path.abspath(checkpoint_path))
        if cls._GLOBAL_STEP_PATTERN.fullmatch(os.path.basename(step_path)) is None:
            return None
        return os.path.dirname(step_path), os.path.basename(checkpoint_path)

    @classmethod
    def _is_complete(cls, role_path: str) -> bool:
        step_path = os.path.dirname(role_path)
        return (
            os.path.isdir(role_path)
            and os.path.isfile(os.path.join(step_path, "data.pt"))
            and os.path.isfile(os.path.join(step_path, cls._COMPLETION_MARKER))
        )

    def rebuild_previous_saved_paths(self, checkpoint_path: str):
        """Recover the retention queue when resuming training."""
        layout = self._checkpoint_layout(checkpoint_path)
        if layout is None:
            return list(self.previous_saved_paths)
        root, role = layout
        discovered = []
        if os.path.isdir(root):
            for entry in os.scandir(root):
                match = self._GLOBAL_STEP_PATTERN.fullmatch(entry.name)
                role_path = os.path.join(entry.path, role)
                if match and entry.is_dir(follow_symlinks=False) and self._is_complete(role_path):
                    discovered.append((int(match.group(1)), os.path.abspath(role_path)))
        self.previous_saved_paths = [path for _, path in sorted(discovered)]
        return list(self.previous_saved_paths)

    def _trim_previous_saved_paths(self, target_count: int, protected_paths: set[str]):
        self.previous_saved_paths = [path for path in self.previous_saved_paths if os.path.isdir(path)]
        excess = max(0, len(self.previous_saved_paths) - target_count)
        victims = [path for path in self.previous_saved_paths if path not in protected_paths][:excess]
        self.remove_previous_save_local_path(victims)
        self.previous_saved_paths = [path for path in self.previous_saved_paths if path not in victims]

    def ensure_checkpoint_capacity(self, max_ckpt_to_keep: int, incoming_path: str = None):
        """Keep the latest completed save while making room for the next one."""
        if not isinstance(max_ckpt_to_keep, int) or max_ckpt_to_keep <= 1:
            return
        if incoming_path is not None:
            self.rebuild_previous_saved_paths(incoming_path)
        protected = set(self.previous_saved_paths[-1:])
        self._trim_previous_saved_paths(max_ckpt_to_keep - 1, protected)

    def register_checkpoint(self, new_path: str, max_ckpt_to_keep: int):
        """Apply retention after model, optimizer and dataloader state are saved."""
        new_path = os.path.abspath(new_path)
        self.rebuild_previous_saved_paths(new_path)
        if self._checkpoint_layout(new_path) is not None and not self._is_complete(new_path):
            return
        if new_path not in self.previous_saved_paths:
            self.previous_saved_paths.append(new_path)
        if isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0:
            self._trim_previous_saved_paths(max_ckpt_to_keep, {new_path})

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


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """Return the checkpoint named by the latest completed-save tracker."""
    if path is None:
        return None
    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.isfile(tracker_file):
        return None
    with open(tracker_file, encoding="utf-8") as stream:
        iteration = int(stream.read().strip())
    checkpoint_path = os.path.join(path, directory_format.format(iteration))
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint tracker points to a missing directory: {checkpoint_path}")
    return checkpoint_path


def get_checkpoint_tracker_filename(root_path: str):
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")
