# LT-OPD

[Paper](https://arxiv.org/abs/2609.32353) · [Model](https://huggingface.co/yyy051007/LT-OPD)

CDPruner selects the visual tokens. Each discarded node is assigned to its nearest retained node in feature space, and an MLP learns its residual contribution. The retained nodes keep their original sequence order and M-RoPE positions; aggregation adds no tokens.

The MLP uses selected rows from a pretrained projection for its input layer and a zero output layer. Initial residual contributions are zero, so the model starts from ordinary CDPruner selection.

## Setup

Follow the repository's [setup instructions](../../README.md#setup) and run the commands below from the repository root. Both implementations share one installation; `--implementation current` selects this version.

## Training

Training uses a frozen full-token teacher, on-policy JSD, and a visual-token curriculum. The data directory should contain `train.parquet` and its images:

```bash
lt-opd --implementation current train \
  --model /path/to/Qwen3.5-4B \
  --data-dir /path/to/prepared-data \
  --user-root /path/to/user-storage \
  --output /path/to/user-storage/outputs/lt-opd
```

Use `--resume` to continue a saved run. Persistent outputs and runtime caches stay under `--user-root`.

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
  --checkpoint /path/to/checkpoint \
  --base-model /path/to/Qwen3.5-4B \
  --output /path/to/export
```

The shared [evaluation commands](../evaluation/README.md) select the runtime from the compression configuration saved with the model.
