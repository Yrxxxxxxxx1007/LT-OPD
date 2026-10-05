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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.distributed._tensor import Placement, Shard

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from tqdm import tqdm

from types import SimpleNamespace


class FSDPShardMerger:
    def __init__(self, actor_dir):
        self.config = SimpleNamespace(local_dir=str(actor_dir))

    def _get_world_size(self) -> int:
        """_summary_
        From FSDP json config file, extract the world size.

        Returns:
            int: world size
        """
        config_path = Path(self.config.local_dir) / "fsdp_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} does not exist.")

        with open(config_path) as f:
            config = json.load(f)

        # Extract world size from the config
        world_size = config.get("world_size", None)
        if world_size is None:
            raise ValueError("World size not found in the config file.")

        return world_size


    def _load_rank_zero_state_dict(self, world_size: int) -> dict:
        return torch.load(
            Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_0.pt",
            map_location="cpu",
            weights_only=False,
        )


    def _extract_device_mesh_info(self, state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
        """
        Retrieves sharding information (device_mesh, mesh_dim_names) from a DTensor in the state_dict.
        If no DTensor is found, infers a simple FSDP mesh based on world_size.
        """
        pivot_key = sorted(list(state_dict.keys()))[0]
        weight = state_dict[pivot_key]

        if isinstance(weight, DTensor):
            # get sharding info
            device_mesh = weight.device_mesh
            mesh = device_mesh.mesh
            mesh_dim_names = device_mesh.mesh_dim_names
        else:
            # for non-DTensor
            mesh = np.array([world_size], dtype=np.int64)
            mesh_dim_names = ("fsdp",)

        return mesh, mesh_dim_names


    def _calculate_shard_configuration(
        self, mesh: np.ndarray, mesh_dim_names: tuple[str, ...]
    ) -> tuple[int, tuple[int, ...]]:
        """Calculates the total number of shards and the shape of the device mesh."""
        assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

        if "tp" in mesh_dim_names:
            # TODO: "tp" is not supported yet due to the above assert
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        else:
            total_shards = mesh.shape[-1]
            mesh_shape = (mesh.shape[-1],)

        return total_shards, mesh_shape


    def _merge_by_placement(self, tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        """Merges a list of tensors based on their DTensor placement"""
        if placement.is_replicate():
            return tensors[0]
        elif placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        elif placement.is_shard():
            return torch.cat(tensors, dim=placement.dim).contiguous()

        raise NotImplementedError(f"Unsupported placement: {placement}")


    def _load_and_merge_state_dicts(
        self, world_size: int, total_shards: int, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, model_state_dict_lst: list):
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return state_dict

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                future.result()

        # Merge state dicts from all shards
        state_dict = {}
        param_placements: dict[str, list] = {}

        for key in set(model_state_dict_lst[0].keys()):
            state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                # add tensor shard in order of rank to state_dict[key]
                tensor = model_state_shard.pop(key)
                # Preserve both trainable visual components in FP32 exports.
                export_dtype = (
                    torch.float32
                    if ("lora_" in key or "visual.merger." in key
                        or "vision_token_compressor.learnable_summary." in key)
                    else torch.bfloat16
                )
                if isinstance(tensor, DTensor):
                    state_dict[key].append(tensor._local_tensor.to(dtype=export_dtype))

                    placements = tuple(tensor.placements)
                    # replicated placement at dp dimension can be discarded
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(tensor.to(dtype=export_dtype))

        del model_state_dict_lst

        # Merge tensors
        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                print(f"No need to merge key {key}")
                continue
            if key in param_placements:
                # merge shards
                placements: tuple[Shard] = param_placements[key]
                if len(mesh_shape) == 1:
                    # 1-D list, FSDP without TP
                    assert len(placements) == 1
                    shards = state_dict[key]
                    state_dict[key] = self._merge_by_placement(shards, placements[0])
                else:
                    # 2-D list, FSDP + TP
                    raise NotImplementedError("FSDP + TP is not supported yet")
            else:
                # Plain tensors from an unsharded checkpoint are complete
                # parameters. Concatenating replicas would change head shapes
                # (including when world_size=1 for scalar parameters).
                replicas = state_dict[key]
                if any(not torch.equal(replicas[0], replica) for replica in replicas[1:]):
                    raise ValueError(f"Non-DTensor checkpoint replicas differ for {key}")
                state_dict[key] = replicas[0].contiguous()

        return state_dict


    def merge(self):
        world_size = self._get_world_size()
        first = self._load_rank_zero_state_dict(world_size)
        mesh, names = self._extract_device_mesh_info(first, world_size)
        del first
        count, shape = self._calculate_shard_configuration(mesh, names)
        return self._load_and_merge_state_dicts(world_size, count, shape, names)
