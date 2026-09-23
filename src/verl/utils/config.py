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
import os
from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from verl.utils.curriculum_capacity import operational_curriculum_profile

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object

def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(config: DictConfig, use_reference_policy: bool, use_critic: bool) -> None:
    """Check the published LT-OPD recipe without machine-specific release gates."""
    from pathlib import Path
    from training.contract import ROOT, load_contract
    expected = OmegaConf.to_container(OmegaConf.load(ROOT / "v8.yaml"), resolve=False)
    runtime_keys = {"paths", "trainer", "ray_kwargs", "custom_reward_function"}
    def compare(actual, wanted, path):
        for key, value in wanted.items():
            name = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                compare(actual.get(key, {}), value, name)
            elif isinstance(value, str) and "${paths." in value:
                continue
            elif isinstance(value, list) and any("${paths." in str(v) for v in value):
                continue
            elif actual.get(key) != value:
                raise ValueError(f"{name}: expected {value!r}, got {actual.get(key)!r}")
    for section in expected:
        if section in runtime_keys:
            continue
        value = expected[section]
        if isinstance(value, dict):
            compare(config[section], value, section)
        elif config[section] != value:
            raise ValueError(f"Unexpected {section}: {config[section]}")
    if config.trainer.n_gpus_per_node != 8 or config.trainer.nnodes != 1:
        raise ValueError("The published V8 recipe uses one node with 8 GPUs.")
    if config.trainer.total_training_steps != 175 or config.trainer.total_epochs != 1:
        raise ValueError("LT-14K requires exactly 175 updates in one epoch.")
    if use_critic or config.reward_model.enable or config.data.val_files:
        raise ValueError("LT-OPD uses distribution distillation without a critic or validation feedback.")
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor.validate(8, config.data.train_batch_size, config.actor_rollout_ref.model)
    print("LT-OPD configuration verified.")
