# Evaluation

Inference uses the exported Qwen3.5 CDPruner runtime at 5% retention, with a minimum of 32 tokens per image. MMMU multi-image questions use independent pruning for each image.

| Dataset | Split / examples | New tokens | Scoring |
| --- | --- | ---: | --- |
| [V*Bench](https://github.com/penghao-wu/vstar) | test / 191 | 16 | lmms-eval generated-answer accuracy |
| [HRBench-4K](https://github.com/DreamMr/HR-Bench) | 4K / 800 | 1024 | VLMEvalKit cycle macro accuracy |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/) | balanced test-dev / 12,578 | 16 | Author raw exact match |
| [MMMU](https://github.com/MMMU-Benchmark/MMMU) | validation / 900 | 128 | Author parser and evaluator, seed 42 |
| [MMBench](https://github.com/open-compass/MMBench) | English dev / 4,329 | 1024 | VLMEvalKit CircularEval |
| [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | test / 2,374 | 16 | Author tool: perception + cognition |
| [POPE](https://github.com/RUCAIBox/POPE) | all three test categories / 9,000 | 128 | Author accuracy, macro F1 |
| [TextVQA](https://textvqa.org/) | v0.5.1 validation / 5,000 | 1024 | Author ten-answer VQA soft accuracy |
| [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR) | test / 1,000 | 128 | Author score / 1,000 |

## Data and inference

Run these commands from the repository root. Prepare the benchmark data and create `data/benchmarks/datasets.json`:

```bash
python src/evaluation/prepare_data.py --output data/benchmarks \
  --textvqa-json /path/to/TextVQA_0.5.1_val.json --textvqa-images /path/to/textvqa/images
```

```bash
pip install -e '.[eval]'
lt-opd eval \
  --export-dir outputs/lt-opd/export --data-config data/benchmarks/datasets.json \
  --output-dir results/lt-opd
```

Set `--export-dir` to either version's export directory; the model selects its runtime automatically. Without `--gpus`, evaluation uses `CUDA_VISIBLE_DEVICES`, or GPU 0 if unset. Explicit indices refer to the visible list when it is set.

Select a subset with `--datasets mmmu textvqa`. `--workers-per-gpu 2` increases concurrency when memory permits. Batch sizes remain 1 for the five visual/MCQ tasks and 4 for GQA, POPE, TextVQA, and MME. Existing predictions are resumed; use a separate output directory for each model.

## Scoring

```bash
python src/evaluation/fetch_scorers.py --output-dir data/scorers
lt-opd score --dataset mmmu \
  --predictions results/lt-opd/mmmu --sources data/scorers \
  --output results/lt-opd/mmmu-score.json
```

Scoring runs on CPU after inference and requires a complete split. `sources.json` records upstream commits and license locations. The downloader obtains original author / harness files; the wrapper executes their unchanged scoring functions. Those files and benchmark data retain their upstream terms.
