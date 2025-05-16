# HyperCap

**HyperCap: A Hyperspectral Imaging Dataset with Pixel-Level Captions and Benchmarking**

---

## Overview

This repository contains the codebase and model configurations for **HyperCap**

Note: A subset of the dataset has been provided for peer review. Upon acceptance of the paperS, the full dataset, complete training code, and evaluation pipeline will be released under an MIT license.

---

## Dataset

Complete Captioning Dataset is present along with the respective HSI data. Please refer to the `Datasets/` folder.

---

## Vision Encoders

We explore and support multiple 3D vision backbones that serve as the feature extractors for our VL tasks:

- 3DRCNet: https://github.com/wanggynpuer/3D-RCNet
- DBCTNet: https://github.com/xurui-joei/DBCTNet
- 3DConvSST: https://github.com/ShyamVarahagiri/3D-ConvSST
- FAHM: https://github.com/zhangxc0105/FAHM

---

## Text Encoders

HyperCap supports the integration of state-of-the-art pretrained language models:

- BERT (bert-large-uncased): https://huggingface.co/google-bert/bert-large-uncased
- T5 (t5-large): https://huggingface.co/google-t5/t5-large

---

## VL Classification - Training Code

To train the model for Vision-Language Classification:

```bash
python train.py
```

Note: Ensure that the subset data is correctly placed and the config file is set accordingly. Full training support will be enabled after acceptance.

---

## Captioning - Training Code

For the Captioning task, we build upon and adapt the codebase of prior works. As of now, we have tutorial of the training code and visual encoder integration.

The current setup includes:

- The adapted vision backbone for captioning is located in the `captioning/` directory.
- The code supports finetuning BLIP.
- To train and understand how to integrate the Vision Encoder on BLIP Check - `Tutorial_Captioning_BLIP.py`

Supported Vision-Language Captioning Models:

- BLIP: https://huggingface.co/docs/transformers/en/model_doc/blip
- mPLUG: https://github.com/X-PLUG/mPLUG
- GIT: https://huggingface.co/docs/transformers/en/model_doc/git
- VinVL: https://github.com/microsoft/Oscar
- VisualBERT: https://huggingface.co/docs/transformers/en/model_doc/visual_bert

---

## Code and Dataset

- Complete Dataset in `Datasets/` Folder
- Tutorial captioning code and adapted vision model released (`captioning/vision_model.py` and `Tutorial_Captioning_BLIP.py`)
- Full codebase, pretrained checkpoints, and complete dataset to be released upon acceptance of the paper

---
