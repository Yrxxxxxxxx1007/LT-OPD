# LT-OPD

[Paper](https://arxiv.org/abs/2609.32353) · [Model](https://huggingface.co/yyy051007/LT-OPD)

CDPruner selects the visual tokens. Each discarded node is assigned to its nearest retained node in feature space, and an MLP learns its residual contribution. The retained nodes keep their original sequence order and M-RoPE positions; aggregation adds no tokens.

The MLP uses selected rows from a pretrained projection for its input layer and a zero output layer. Initial residual contributions are zero, so the model starts from ordinary CDPruner selection.

## Setup

Follow the repository's [setup instructions](../../README.md#setup) and run the commands below from the repository root. Both implementations share one installation; `--implementation current` selects this version.

## Training

Training uses a frozen full-token teacher, on-policy JSD, and a visual-token curriculum. Prepare [LT-OPD-14K](../../README.md#data) and download the base model as shown in the [training instructions](../../README.md#training). The configuration below uses the public 14K dataset; the default 15K recipe remains available for a matching dataset.

```bash
USER_STORAGE=/path/to/user-storage
RUN_DIR="$USER_STORAGE/outputs/lt-opd-current"

lt-opd --implementation current train \
  --config src/learnable_merge/training/lt14k.yaml \
  --model models/Qwen3.5-4B \
  --data-dir data/LT-OPD-14K \
  --user-root "$USER_STORAGE" \
  --output "$RUN_DIR"
```

Set `CUDA_VISIBLE_DEVICES` to select devices and `--gpus` to set how many to use. Use `--cpus` to set the local Ray CPU allocation and `--resume` to continue a saved run.

Persistent outputs and caches stay under `--user-root`. Ray's temporary object store uses `/dev/shm` by default; `--ram-object-store-dir` can select a directory under `--user-root` when shared memory is limited.

## Inference

Download the model, then load it through the compression runtime:

```bash
hf download yyy051007/LT-OPD \
  --local-dir models/LT-OPD
```

```python
import torch
from evaluation.runtime import build_route_query
from lt_opd import load_compression_runtime

runtime = load_compression_runtime(
    "models/LT-OPD",
    implementation="current",
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
)
question = "What is written on the sign?"
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": question},
    ],
}]
output = runtime.generate(
    [messages],
    route_queries_batch=[build_route_query(question)],
)
print(output["decoded_predictions"][0])
```

## Export and Evaluation

```bash
lt-opd --implementation current export \
  --checkpoint "$RUN_DIR/checkpoints/global_step_175" \
  --recipe "$RUN_DIR/config.yaml" \
  --base-model models/Qwen3.5-4B \
  --output "$RUN_DIR/export"
```

Follow the shared [evaluation commands](../evaluation/README.md) with `--export-dir "$RUN_DIR/export"`; they select the runtime from the model's saved compression configuration.
