# LT-OPD

[Paper](https://arxiv.org/abs/2609.32353) · [Model](https://huggingface.co/yyy051007/LT-OPD)

CDPruner selects the visual tokens. Each discarded node is assigned to its nearest retained node in feature space, and an MLP learns its residual contribution. The retained nodes keep their original sequence order and M-RoPE positions; aggregation adds no tokens.

The MLP uses selected rows from a pretrained projection for its input layer and a zero output layer. Initial residual contributions are zero, so the model starts from ordinary CDPruner selection.

## Setup

Use the environment in the repository's [setup instructions](../../README.md#setup). From the repository root, select this implementation explicitly:

```bash
export PYTHONPATH="$PWD/src/learnable_merge:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

For an independent installation, this directory also provides a `pyproject.toml`:

```bash
pip install -e 'src/learnable_merge[train,eval]'
```

## Training

Training uses a frozen full-token teacher, on-policy JSD, and a visual-token curriculum. The data directory should contain `train.parquet` and its images:

```bash
python -m training.train \
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
from training.runtime import load_compression_runtime

runtime = load_compression_runtime(
    "models/LT-OPD",
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
python -m training.export \
  --checkpoint /path/to/checkpoint \
  --base-model /path/to/Qwen3.5-4B \
  --output /path/to/export
```

With the `PYTHONPATH` above, the existing [evaluation commands](../evaluation/README.md) use this runtime and the compressor configuration saved with the model.
