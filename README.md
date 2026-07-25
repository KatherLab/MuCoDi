# MuCoDi: Multi-Teacher Contrastive Distillation for Edge-Efficient Pathology Foundation Models

<a href="https://arxiv.org/abs/2607.05533"><img src="https://img.shields.io/badge/Preprint-arXiv-red"></a>
<a href="https://sites.google.com/view/ema2026"><img src="https://img.shields.io/badge/Workshop-EMA4MICCAI%202026-blue"></a>
<a href="#citation"><img src="https://img.shields.io/badge/Paper-Cite-blue"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey"></a>

Official pretraining implementation of:

> **Multi-Teacher Contrastive Distillation for Edge-Efficient Pathology Foundation Models**  
> Tim Lenz<sup>†</sup>, Maurice Heide<sup>†</sup>, Marco Gustav, Nic G. Reitsam, and Jakob Nikolas Kather  
> <sup>†</sup> Equal contribution  
> Accepted at the **2nd MICCAI Workshop on Efficient Medical AI (EMA4MICCAI 2026)**.

## Table of Contents

- [Abstract](#abstract)
- [News](#news)
- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Scope](#repository-scope)
- [Installation](#installation)
- [Data Format](#data-format)
- [Pretraining](#pretraining)
- [Architectures](#architectures)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Downstream Evaluation](#downstream-evaluation)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Abstract

Computational pathology foundation models (PFMs) have advanced whole-slide image analysis. However, their size and inference cost hinder local deployment in pathology departments. We propose **MuCoDi**, a pretraining framework that distills frozen tile embeddings from multiple PFMs into compact edge-oriented encoders. Instead of regressing individual teacher features, MuCoDi trains lightweight MobileOne and RepViT students with a contrastive distillation objective adapted from MoCo v3, where cached Virchow2, UNI2, and H-Optimus-1 embeddings replace momentum-encoder keys. We pretrain students on 14.3 million TCGA tiles from only 11.8 thousand WSIs and evaluate frozen encoders on 23 clinically curated downstream classification tasks. RepViT-based MuCoEdge students retain near-teacher performance while reducing model size by orders of magnitude: MuCoEdge-R2.3 and MuCoEdge-R1.5 reach 71.0% external AUROC, within 0.8 percentage points of the best teacher (Virchow2, 71.8%), while MuCoEdge-R2.3 obtains the best external F1 and the second-best AUPRC (51.8% and 53.3%). MuCoEdge-R1.0 reaches 70.9% AUROC with only 6.4 million parameters and 1.12 GFLOPs. On a Raspberry Pi 5, sub-million-parameter MobileOne students achieve up to a 605-fold single-tile speedup over Virchow2 while retaining 66.5% to 66.9% external AUROC, demonstrating that PFM-quality pathology representations can be moved toward practical edge deployment.

## News

- **[July 2026]** MuCoDi was accepted at the 2nd MICCAI Workshop on Efficient Medical AI (EMA4MICCAI 2026).
- **[July 2026]** The MuCoDi preprint was released on arXiv.

## Overview

MuCoDi adapts the two-view contrastive learning formulation of MoCo v3 for multi-teacher knowledge distillation:

1. Frozen pathology foundation models provide one cached embedding per training tile.
2. A lightweight MobileOne or RepViT student processes two independently augmented views of the same tile.
3. One linear projection head per teacher maps the shared student representation into the corresponding teacher embedding space.
4. The student query is aligned with the cached teacher embedding of the same tile and contrasted against teacher embeddings from all other tiles in the global batch.

Only the student backbone and teacher-specific projection heads are optimized. Teacher models are not executed during student pretraining once their tile embeddings have been cached.

## Key Results

Selected external CPTAC results reported in the paper:

| Model | Parameters | GFLOPs / tile | AUROC | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|
| H-Optimus-1 | 1,135.0M | 295.97 | 68.9 | 48.1 | 51.1 |
| UNI2 | 681.0M | 180.39 | 71.7 | 50.6 | 53.4 |
| Virchow2 | 632.0M | 164.59 | 71.8 | 50.7 | 52.9 |
| MuCoEdge-R2.3 | 22.4M | 4.57 | 71.0 | 51.8 | 53.3 |
| MuCoEdge-R1.5 | 13.6M | 2.31 | 71.0 | 51.1 | 52.8 |
| MuCoEdge-R1.0 | 6.4M | 1.12 | 70.9 | 50.2 | 52.0 |
| MuCoEdge-R0.6 | 2.2M | 0.394 | 69.4 | 48.8 | 51.1 |
| MuCoEdge-Mµ0 | 0.2M | 0.069 | 66.5 | 44.5 | 48.4 |

Metrics are means across 23 clinically curated binary classification tasks. MobileOne parameter counts are reported after structural reparameterization into the fused inference graph; the multi-branch training graph contains additional parameters. See the paper for the complete benchmark, confidence intervals, organ-specific results, and Raspberry Pi 5 latency measurements.

## Repository Scope

This repository contains the code used for **MuCoDi student pretraining**, including:

- Multi-teacher contrastive distillation
- MoCo v3-style two-view augmentations
- WebDataset-based loading of image tiles and cached teacher embeddings
- MobileOne and RepViT student backbones
- Distributed data parallel training
- Step-based warmup, cosine decay, gradient clipping, AMP, and checkpointing

The repository does not contain TCGA data, cached teacher embeddings, teacher-model code, or the downstream STAMP evaluation pipeline.

## Installation

MuCoDi requires Python 3.12 and an NVIDIA GPU with CUDA support. The environment is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/KatherLab/MuCoDi.git
cd MuCoDi

uv sync
source .venv/bin/activate
```

The lock file pins the environment used for this release. On Linux, PyTorch and torchvision are installed from the CUDA 12.8 wheel index configured in `pyproject.toml`.

Verify the installation with:

```bash
python main.py --help
```

## Data Format

Training reads WebDataset shards named:

```text
train-000000.tar
train-000001.tar
...
```

By default, shards are loaded from:

```text
$DATA_ROOT/shards
```

If `DATA_ROOT` is not set, the code uses `./data/shards`.

Each sample in a shard must contain the following files with a shared sample key:

```text
<key>.png
<key>.virchow2.npy
<key>.uni2.npy
<key>.h-optimus-1.npy
<key>.json
```

The files contain:

| File | Content |
|---|---|
| `<key>.png` | RGB pathology tile |
| `<key>.<teacher>.npy` | Cached one-dimensional teacher embedding |
| `<key>.json` | Metadata containing at least `{"wsi": "<slide-id>"}` |

The default dataset profile in [`dataprep/datasets.yaml`](dataprep/datasets.yaml) uses:

| Dataset key | Teacher model | Embedding dimension |
|---|---|---:|
| `virchow2` | Virchow2 | 1280 |
| `uni2` | UNI2-h | 1536 |
| `h-optimus-1` | H-Optimus-1 | 1536 |

Tiles used in the paper were `224 × 224` pixels at `2.0 µm/px`.

### Custom Dataset Profiles

Additional profiles can be added to [`dataprep/datasets.yaml`](dataprep/datasets.yaml):

```yaml
custom:
  description: "Custom teacher combination"
  teacher_dims:
    teacher_a: 1024
    teacher_b: 768
  shard_dir: "custom_shards"
```

Select the profile with `--dataset custom`. Relative `shard_dir` values are resolved against `DATA_ROOT`; absolute paths are used unchanged.

## Pretraining

The following command reproduces the paper-scale setup on one node with four GPUs:

```bash
DATA_ROOT=/path/to/data python main.py \
  --arch repvit_m1_5 \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --dist-url tcp://127.0.0.1:10001 \
  --batch-size 2048 \
  --workers 48 \
  --train-steps 70000 \
  --warmup-steps 3500 \
  --save-dir ./checkpoints
```

The global batch size is divided evenly across all participating GPUs. The paper configuration uses:

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Base learning rate | `2e-2` |
| Weight decay | `1e-6` |
| Global batch size | 2048 |
| Temperature | `0.2` |
| Projection-head layers | 1 |
| Gradient clipping | `1.0` |
| Warmup | 5% of optimizer steps |
| Schedule | Cosine decay to zero |
| Precision | bfloat16 AMP |
| Data-loading workers | 12 per GPU |
| Training length | 10 epochs, approximately 70,000 steps |

### Single-GPU Run

```bash
DATA_ROOT=/path/to/data python main.py \
  --arch mobileone_mu0 \
  --gpu 0 \
  --batch-size 512 \
  --workers 12 \
  --train-steps 1000 \
  --warmup-steps 50 \
  --save-dir ./checkpoints/smoke_test
```

### Resume Training

```bash
DATA_ROOT=/path/to/data python main.py \
  --arch repvit_m1_5 \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --batch-size 2048 \
  --train-steps 70000 \
  --warmup-steps 3500 \
  --resume ./checkpoints/step_0035000.pth.tar \
  --save-dir ./checkpoints
```

### Weights & Biases Logging

Logging is disabled by default. Enable it explicitly with `--wandb`:

```bash
WANDB_PROJECT=mucodi \
WANDB_ENTITY=<entity> \
python main.py --wandb <other-arguments>
```

`WANDB_NAME` can be set to override the automatically generated run name.

### Gradient Checkpointing

RepViT gradient checkpointing can be enabled through:

```bash
MUCODI_GRAD_CKPT=1 python main.py <arguments>
```

## Architectures

The student architectures used in the paper are:

**RepViT**

```text
repvit_m0_6
repvit_m0_9
repvit_m1_0
repvit_m1_1
repvit_m1_5
repvit_m2_3
```

**MobileOne**

```text
mobileone_mu0
mobileone_mu1
mobileone_mu2
mobileone_s0
mobileone_s1
mobileone_s2
mobileone_s3
mobileone_s4
```

Other architectures supported by `timm.create_model` can also be supplied through `--arch`, provided that the model returns a two-dimensional pooled feature tensor when constructed with `num_classes=0`.

## Configuration

Important command-line options:

| Option | Description |
|---|---|
| `--arch` | Student backbone |
| `--dataset` | Dataset profile from `dataprep/datasets.yaml` |
| `--teacher-dims` | Override teacher names and dimensions, for example `a:1024,b:768` |
| `--num-layers` | Number of layers in each teacher-specific projection head |
| `--moco-t` | Contrastive temperature |
| `--batch-size` | Global batch size across all GPUs |
| `--workers` | Worker budget per node for a spawned multi-GPU run; divided across local GPUs |
| `--train-steps` | Total optimizer-step budget |
| `--warmup-steps` | Number of linear warmup steps |
| `--save-every` | Checkpoint interval in optimizer steps |
| `--shard-count` | Deterministically use only the first sampled subset of shards |
| `--seed` | Random seed |
| `--wandb` | Enable Weights & Biases logging |

Run `python main.py --help` for the complete interface.

## Repository Structure

```text
MuCoDi/
├── main.py                  # Training entry point
├── dataprep/
│   ├── datasets.yaml        # Dataset and teacher profiles
│   ├── config.py            # Profile loading and path resolution
│   └── loader.py            # WebDataset pipeline and augmentations
├── moco/
│   ├── loader.py            # Two-view augmentations
│   └── loss.py              # Multi-teacher KD-InfoNCE loss
├── models/
│   ├── factory.py           # Backbone construction
│   ├── mobileone.py         # MobileOne implementation
│   ├── repvit.py            # RepViT implementation
│   └── student.py           # Student and projection heads
├── utils/
│   ├── engine.py            # Training loop
│   ├── options.py           # Command-line arguments
│   └── utils.py             # Scheduling, metrics, and checkpointing
├── LICENSES/                # Third-party license texts
├── pyproject.toml
└── uv.lock
```

## Downstream Evaluation

The paper evaluates frozen MuCoDi encoders by extracting tile features and training slide-level transformer MIL heads with [STAMP](https://github.com/KatherLab/STAMP). Training is performed on TCGA using patient-level stratified five-fold cross-validation, followed by unchanged deployment to matched CPTAC cohorts.

The downstream extraction and evaluation workflow is not duplicated in this repository. Refer to STAMP for WSI tiling, feature extraction, and weakly supervised slide-level evaluation.

## References

- [MuCoDi preprint](https://arxiv.org/abs/2607.05533)
- [Virchow2](https://huggingface.co/paige-ai/Virchow2)
- [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)
- [H-Optimus-1](https://huggingface.co/bioptimus/H-optimus-1)
- [MoCo v3](https://github.com/facebookresearch/moco-v3)
- [MobileOne](https://github.com/apple/ml-mobileone)
- [RepViT](https://github.com/THU-MIG/RepViT)
- [STAMP](https://github.com/KatherLab/STAMP)

## Acknowledgements

The authors gratefully acknowledge computing-time support from the German Joint Science Conference through the Center for Information Services and High Performance Computing at TU Dresden, and from the Gauss Centre for Supercomputing through the John von Neumann Institute for Computing on the JUWELS and JUPITER Booster systems at the Jülich Supercomputing Centre. We also acknowledge the TCGA Research Network and the Clinical Proteomic Tumor Analysis Consortium, which generated the data used in this study.

## License

The original MuCoDi code is released under the **Creative Commons Attribution-NonCommercial 4.0 International License**. See [`LICENSE`](LICENSE).

This repository includes adapted third-party components with their own copyright and license terms. See [`NOTICE`](NOTICE) and [`LICENSES/`](LICENSES/) before reuse or redistribution.

## Citation

```bibtex
@misc{lenz2026mucodi,
  title         = {Multi-Teacher Contrastive Distillation for Edge-Efficient Pathology Foundation Models},
  author        = {Lenz, Tim and Heide, Maurice and Gustav, Marco and Reitsam, Nic G. and Kather, Jakob Nikolas},
  year          = {2026},
  eprint        = {2607.05533},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.05533},
  note          = {Accepted at the 2nd MICCAI Workshop on Efficient Medical AI (EMA4MICCAI 2026)}
}
```

## Contact

For technical questions, please open an issue in this repository.

For questions about the paper, contact **Jakob Nikolas Kather** at `kather.jn@tu-dresden.de`.
