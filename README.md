# HyperCap: Hyperspectral Land Cover Captioning Dataset for Vision Language Models

<p align="center">
  <img src="https://github.com/user-attachments/assets/aaa3a758-7f94-4eb3-b532-bfcd2c359064" width="400"/>
</p>

<div align="center">
  <a href="https://ieeexplore.ieee.org/document/11550167/">
    <img src="https://img.shields.io/badge/IEEE_Xplore-Paper-blue.svg?style=for-the-badge" alt="IEEE Xplore Paper">
  </a>
  <a href="https://arxiv.org/abs/2505.12217">
    <img src="https://img.shields.io/badge/arXiv-2505.12217-b31b1b.svg?style=for-the-badge" alt="arXiv Paper">
  </a>
  <a href="https://github.com/arya-domain/HyperCap">
    <img src="https://img.shields.io/badge/GitHub-Repository-black.svg?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/Task-Hyperspectral_Captioning-blue.svg?style=flat-square" alt="Task">
</div>

---

Official repository for **"HyperCap: A Hyperspectral Land-Cover Captioning Dataset for Vision–Language Models"**, published in **IEEE Geoscience and Remote Sensing Magazine (GRSM), 2026**.

---

## 👥 Authors

**Aryan Das\*, Tanishq Rachamalla\*, Pravendra Singh, Koushik Biswas, Vinay Kumar Verma, Salvador Garcia, Antonio Plaza†, Swalpa Kumar Roy†**

*\* Equal contribution | † Corresponding authors*

---

## 📌 Publication Details

- **Journal**: IEEE Geoscience and Remote Sensing Magazine (IEEE GRSM), 2026
- **DOI**: [10.1109/MGRS.2026.3693613](https://doi.org/10.1109/MGRS.2026.3693613)
- **arXiv Preprint**: [2505.12217](https://arxiv.org/abs/2505.12217)

---

## 📖 Abstract

We introduce **HyperCap**, the first large-scale hyperspectral captioning dataset designed to enhance model performance and effectiveness in remote sensing applications. Unlike traditional hyperspectral imaging (HSI) benchmarks, HyperCap integrates spectral data with pixel-wise textual annotations, enabling deeper semantic understanding. This dataset enhances model performance in tasks like classification and feature extraction, providing a valuable resource for advanced remote sensing applications. HyperCap is constructed from four benchmark datasets — Botswana, Houston 2013, Indian Pines, and Kennedy Space Center — and annotated through a hybrid approach combining automated and manual methods to ensure accuracy and consistency. Empirical evaluations using state-of-the-art encoders and diverse fusion techniques demonstrate significant improvements in classification performance. These results underscore the potential of vision-language learning in HSI and position HyperCap as a foundational dataset for future research in the field.

---

## 📊 Dataset Statistics

HyperCap provides **21,237 pixel-wise, expert-refined captions** spanning **50 land cover classes** across four standard hyperspectral imaging benchmarks:

| Dataset | Total Pixel Captions | Labeled Classes | Bands | Location / Sensor |
| :--- | :---: | :---: | :---: | :--- |
| **Indian Pines** | 10,248 | 16 | 220 | Indiana, USA (AVIRIS) |
| **Kennedy Space Center (KSC)** | 5,211 | 13 | 224 | Florida, USA (AVIRIS) |
| **Botswana** | 3,248 | 14 | 242 | Okavango Delta (Hyperion) |
| **Houston 2013** | 2,530 | 15 | 144 | Houston, USA (ITRES CASI-1500) |
| **Total** | **21,237** | **50** | — | — |

---

## 💡 Key Highlights

1. **New Pixel-Level Benchmark**: The first HSI captioning dataset to provide one expert-refined caption per labeled pixel (21,237 annotations total). Prior datasets like LDGNet only offered template-based captions at the patch level.
2. **Dual-LLM + Expert Refinement**: Candidates were generated in parallel by ChatGPT-4o and Mistral Large, then refined by three remote sensing experts with strict rules (no class names, no functional language, no hallucinated world knowledge).
3. **Massive Accuracy Gains**: Multimodal fusion using text encoders boosts overall classification accuracy by up to **+27%** on KSC (70.5% → 97.6%) and **+23%** on Indian Pines (76.0% → 99.4%) with DBCTNet. FAHM and 3D-ConvSST reach 100% OA on Botswana.
4. **Resilience to Data Scarcity**: Multimodal models using captions maintain above 98% OA even when trained on just **3% of training labels**, while vision-only models drop significantly.
5. **No Label Leakage**: Captions are semantically additive rather than simply leaking class labels. Text-only models perform 15–29% worse than full multimodal models.
6. **New Task Benchmarking**: Establishes baseline results for image captioning (GIT leads with BLEU-1: 0.43) and cross-modal retrieval (IR ~62%, TR ~70%).

---

## 📁 Repository Structure

```
HyperCap/
├── Datasets/
│   ├── Botswana/
│   │   ├── Bostwana.csv           # Pixel-wise captions
│   │   ├── Botswana_data.mat      # Raw HSI data
│   │   ├── Botswana_gt.mat        # Ground truth labels
│   │   └── class_map.json         # Class index mappings
│   ├── Houston13/                 # Houston 2013 dataset files
│   ├── Indian Pines/              # Indian Pines dataset files
│   └── KSC/                       # Kennedy Space Center dataset files
├── model/
│   ├── vision/
│   │   ├── DBCTNet.py             # DBCTNet vision backbone
│   │   ├── FAHM.py                # FAHM vision backbone
│   │   ├── _3DRCNet.py            # 3D-RCNet vision backbone
│   │   └── _3D_ConvSST.py         # 3D-ConvSST vision backbone
│   ├── text/
│   │   ├── bert.py                # BERT text encoder adapter
│   │   └── t5.py                  # T5 text encoder adapter
│   └── base.py                    # Base multimodal fusion models
├── captioning/
│   └── vision_model.py            # Adapted FAHM vision model for captioning
├── train.py                       # Training script for Vision-Language Classification
├── Tutorial_Captioning_BLIP.py    # Training tutorial for BLIP Captioning model
└── README.md
```

---

## 🛠️ Supported Encoders & VL Models

### 3D Vision Encoders (Backbones)
*   **3DRCNet**: [3D-RCNet Repository](https://github.com/wanggynpuer/3D-RCNet)
*   **DBCTNet**: [DBCTNet Repository](https://github.com/xurui-joei/DBCTNet)
*   **3DConvSST**: [3D-ConvSST Repository](https://github.com/ShyamVarahagiri/3D-ConvSST)
*   **FAHM**: [FAHM Repository](https://github.com/zhangxc0105/FAHM)

### Text Encoders & Models
*   **BERT** (`bert-large-uncased`): [HuggingFace Hub](https://huggingface.co/google-bert/bert-large-uncased)
*   **T5** (`t5-large`): [HuggingFace Hub](https://huggingface.co/google-t5/t5-large)

### Supported Vision-Language Captioning Models
*   **BLIP**: [Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/blip)
*   **mPLUG**: [mPLUG Repository](https://github.com/X-PLUG/mPLUG)
*   **GIT**: [Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/git)
*   **VinVL**: [Oscar Repository](https://github.com/microsoft/Oscar)
*   **VisualBERT**: [Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/visual_bert)

---

## 🚀 Getting Started

### 📋 Prerequisites

Install the required PyTorch and scientific computing dependencies:
```bash
pip install torch torchvision numpy scipy pandas h5py scikit-learn transformers datasets tqdm matplotlib
```

### 🏋️ Vision-Language Classification Training
To run the multimodal classification training script:
```bash
python train.py
```

### 📝 Captioning Training (BLIP Tutorial)
To run the tutorial script training BLIP with the FAHM 3D vision encoder backbone:
```bash
python Tutorial_Captioning_BLIP.py
```

---

## 🎗️ Funding & Acknowledgements

This research is supported in part by the **Consejería de Economía, Ciencia y Agenda Digital · Junta de Extremadura** and the **European Regional Development Fund (ERDF)** under Grant **GR24035**.

---

## ✏️ Citation

If you find this work, dataset, or codebase useful for your research, please cite our paper:

```bibtex
@article{das2026hypercap,
  title   = {HyperCap: A Hyperspectral Land-Cover Captioning Dataset for Vision--Language Models},
  author  = {Das, Aryan and Rachamalla, Tanishq and Singh, Pravendra and Biswas, Koushik and Verma, Vinay Kumar and Garcia, Salvador and Plaza, Antonio and Roy, Swalpa Kumar},
  journal = {IEEE Geoscience and Remote Sensing Magazine},
  year    = {2026},
  doi     = {10.1109/MGRS.2026.3693613}
}
```
