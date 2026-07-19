# ICD Loss for Person Re-identification — Training Code

This repository contains the initial public release of the training code for **Intra-class Diversity (ICD) Loss** in person re-identification.

ICD is designed to preserve useful intra-class variation while maintaining identity discrimination. The current release provides the original research training pipeline in a transparent form. Documentation, verified experiment settings, result tables, and the arXiv link will be added progressively.

## Code base and contribution

The training framework is adapted from the official [TransReID](https://github.com/damo-cv/TransReID) repository. The original project structure and much of the upstream naming are intentionally retained so that the relationship to TransReID remains clear.

My main contribution is the design and implementation of the ICD loss and its integration into the ReID training objective:

- `loss/metric_learning.py` — `class ICD_loss`
- `model/make_model.py` — ICD classifier integration
- `loss/make_loss.py` — joint identity, triplet, and ICD objective
- `config/defaults.py` and `configs/` — ICD-related configuration

A lightweight standalone implementation is also available in [ICD-Loss-ReID-Code-Sample](https://github.com/321zhou-zhou/ICD-Loss-ReID-Code-Sample).

## Core idea

The ICD module follows an ArcFace-style identity classification formulation. For each normalized feature, it selects the normalized class weight corresponding to that identity, removes the component along the class-weight direction, and measures the similarity of same-identity samples in the remaining orthogonal space.

The auxiliary ICD term penalizes excessive positive similarity in this class-orthogonal space. The goal is to avoid collapsing all samples of the same identity into an overly compact representation while preserving a stable and discriminative identity direction.

The current training objective is:

```python
loss = ID_LOSS_WEIGHT * ID_LOSS \
     + TRIPLET_LOSS_WEIGHT * TRI_LOSS \
     + INTRA_CLASS_LOSS_WEIGHT * intra_class_loss_mean
```

## Current release status

This is an **initial public release** of the original training project.

- The code structure and coding style are kept close to the original implementation.
- The ICD module has passed a CPU forward/backward smoke test.
- The arXiv manuscript and citation information will be added later.

The current ICD integration is intended for the global ViT branch with:

```yaml
MODEL:
  JPM: False
  ID_LOSS_TYPE: 'ICD_loss'
```

## Repository structure

```text
config/                 default configuration
configs/                example dataset configurations
datasets/               ReID datasets and data loaders
loss/                   ICD, triplet, identity, and auxiliary losses
model/                  TransReID-based model construction
processor/              training and evaluation loops
solver/                 optimizers and learning-rate schedulers
utils/                  metrics, logging, and re-ranking
pretrained/             instructions for pretrained ViT weights
train.py                training entry point
test.py                 evaluation entry point
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The original project was developed with a CUDA-enabled PyTorch environment. A fully pinned environment file will be added after the original setup is verified.

## Dataset preparation

Prepare a supported ReID dataset under `data/`. For example:

```text
data/
└── market1501/
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```

Dataset files are not distributed in this repository. Please follow the license and access conditions of each dataset.

## Pretrained ViT model

Download the ImageNet-pretrained ViT-Base weight used by TransReID:

[jx_vit_base_p16_224-80ecf9dd.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)

Place it at:

```text
pretrained/jx_vit_base_p16_224-80ecf9dd.pth
```

The weight file is intentionally not stored in this repository.

## Training

Example training command for Market-1501:

```bash
python train.py --config_file configs/Market/vit_base.yml
```

Configuration values can also be overridden from the command line:

```bash
python train.py --config_file configs/Market/vit_base.yml \
  MODEL.DEVICE_ID "('0')" \
  MODEL.INTRA_CLASS_LOSS_WEIGHT 1.0
```

The public configuration currently uses a short run for code checking. Before full training, review the dataset path, batch size, epoch count, output directory, and ICD loss weight in the selected YAML file.

## Evaluation

Set `TEST.WEIGHT` in the YAML file or override it from the command line:

```bash
python test.py --config_file configs/Market/vit_base.yml \
  TEST.WEIGHT "./logs/market_vit_base/transformer_120.pth"
```

## Main ICD settings

```yaml
MODEL:
  ID_LOSS_TYPE: 'ICD_loss'
  INTRA_CLASS_LOSS_WEIGHT: 1.0
  JPM: False
```

Setting `INTRA_CLASS_LOSS_WEIGHT` to `0.0` disables the auxiliary ICD term while retaining the current classifier path. The exact baseline protocol and historical experiment settings will be documented in a later reproducibility update.

## Files not included

The repository intentionally excludes:

- ReID datasets
- ImageNet pretrained weights
- trained checkpoints
- experiment logs and local output files
- local machine paths and cached files

## Acknowledgement and license

This project is built on [TransReID](https://github.com/damo-cv/TransReID). Please cite the original TransReID paper and repository when using the upstream training framework.

The upstream MIT license and attribution are retained. See `LICENSE` for details.
