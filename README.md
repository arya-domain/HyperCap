# HyperCap

<p align="center">
  <img src="https://github.com/user-attachments/assets/aaa3a758-7f94-4eb3-b532-bfcd2c359064" width="400"/>
</p>

<div align="center">
  <a href="http://hypercap.netlify.app" style="margin: 0 20px;">
    <img src="https://img.shields.io/badge/Project-Website-87CEEB" alt="Project Website">
  </a>
  <a href="https://arxiv.org/pdf/2505.12217" style="margin: 0 20px;">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv Paper">
  </a>
  <a href="https://github.com/arya-domain/HyperCap" style="margin: 0 20px;">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub Repo">
  </a>
</div>







**HyperCap: A Hyperspectral Imaging Dataset with Pixel-Level Captions and Benchmarking**
## HyperCap: A Benchmark Dataset for Hyperspectral Image Captioning

---

## Overview

This repository contains the codebase and model configurations for **HyperCap**

Note: Complete Dataset has been provided for review. Complete training code for captioning and evaluation pipeline will be released in future under an MIT license.

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

Note: Complete Dataset Released. Full training support will be enabled after infuture.

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

## Captioning - Evaluation
We utilized the Microsoft COCO Caption Evaluation for evaluating.

- PyCocoEval - https://github.com/salaniz/pycocoevalcap

---

## Code and Dataset

- Complete Dataset in `Datasets/` Folder
- Tutorial captioning code and adapted vision model released (`captioning/vision_model.py` and `Tutorial_Captioning_BLIP.py`)
- Full codebase, pretrained checkpoints, and complete dataset to be released upon acceptance of the paper

---
