# Fewer Tokens, More Self-Teaching: On-Policy Self-Distillation for Extreme Visual Token Reduction    
[Junxian Li](https://lijunxian111.github.io), [Ruixuan Yang](https://openreview.net/profile?id=~Ruixuan_Yang2), [Tianao Zhang](https://scholar.google.com/citations?user=Cb34iaEAAAAJ&hl=en&oi=ao), [Tiange Xu](http://openreview.net/profile?id=~Tiange_Xu1), [Weisheng Dong](https://scholar.google.com/citations?user=-g58LsoAAAAJ&hl=en&oi=ao), and [Yulun Zhang](https://yulunzhang.com)  

"Fewer Tokens, More Self-Teaching: On-Policy Self-Distillation for Extreme Visual Token Reduction", arXiv 2026  

<div>
<a href="https://github.com/Yrxxxxxxxx1007/LT-OPD" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/downloads/Yrxxxxxxxx1007/LT-OPD/total?color=green"></a>
<a href="https://github.com/Yrxxxxxxxx1007/LT-OPD" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=Yrxxxxxxxx1007/LT-OPD"></a>
<a href="">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoSvg">
  </a>
<a href="https://github.com/Yrxxxxxxxx1007/LT-OPD/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/Yrxxxxxxxx1007/LT-OPD"></a>
</div>  

[project](https://github.com/Yrxxxxxxxx1007/LT-OPD) · [Training data](https://huggingface.co/datasets/yyy051007/LT-OPD-14K) · [Evaluation](src/evaluation/README.md)


#### 🔥🔥🔥 News

- **2026-09-25:** This repo is released.

---

> **Brief Description** On-policy distillation for vision-language models with extremely reduced visual tokens.

> A student using CDPruner learns from a frozen full-token teacher on its own generated prefixes. This repository contains the Qwen3.5-4B training implementation, the LT-OPD-14K data pipeline, and evaluation code for nine benchmarks.

---


## Setup

Use Python 3.12 and a CUDA environment. Install the package and its training dependencies:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e '.[train,eval]'
pip install flash-attn==2.8.3 --no-build-isolation
```

## Data

LT-OPD-14K contains 14,000 examples from OneThinker, PixMo, LLaVA, TextVQA, and Vision-OPD. The Hugging Face release keeps the training order and records per-sample provenance and image hashes. Image archives and the instructions for obtaining the remaining upstream images are in the dataset card.

```bash
python -m data.prepare --dataset yyy051007/LT-OPD-14K \
  --output data/LT-OPD-14K --download-pixmo \
  --source-media /path/to/sa1b/images \
  --source-media /path/to/onethinker
```

Obtain the SA-1B and IAM images from their original sources as described in the dataset card. Preparation verifies all 14,000 images before writing the training files.

## Training

Download the base model, then launch training:

```bash
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir models/Qwen3.5-4B

python -m training.train \
  --model models/Qwen3.5-4B \
  --data-dir data/LT-OPD-14K \
  --output outputs/lt-opd
```

The default configuration uses eight GPUs and full-parameter training, including the visual encoder and merger.

| Setting | Value |
| --- | --- |
| Training updates | 175 |
| Batch | 80 questions × 8 student rollouts |
| Teacher | Frozen Qwen3.5-4B with all visual tokens |
| Objective | Token-level JSD |
| Visual-token curriculum | 25% for 14 updates, cosine decay, 5% for the final 75 updates |

Set `--gpus` and `--nodes` for your hardware. Use `--resume` to continue a saved run.

## Evaluation

Export the trained checkpoint:

```bash
python -m training.export \
  --checkpoint outputs/lt-opd/checkpoints/global_step_175 \
  --base-model models/Qwen3.5-4B \
  --output outputs/lt-opd/export
```

The evaluation suite covers **V*Bench, HRBench-4K, GQA, MMMU, MMBench, MME, POPE, TextVQA, and OCRBench**. Inference and scoring are separate commands; dataset preparation, protocol details, and scorer versions are listed in the [evaluation guide](src/evaluation/README.md).

## Code layout

```text
src/
├── training/      # Training configuration, launcher, and checkpoint export
├── verl/          # Training core, JSD, Qwen3.5 integration, and CDPruner
├── data/          # LT-OPD-14K preparation
└── evaluation/    # Benchmark inference and scoring
```

## Acknowledgements

Built on [verl](https://github.com/volcengine/verl), [SDPO](https://github.com/lasgroup/SDPO), and [CDPruner](https://github.com/Theia-4869/CDPruner). Benchmark scorers retain their upstream attribution and versions.
