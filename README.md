# XGait: A Multi-Modality Wireless Sensing Dataset for Indoor Human Tracking and Identification

<div align="center">

[![UbiComp 2026](https://img.shields.io/badge/Accepted%20by-UbiComp%202026-c0392b?style=flat-square&logo=acm)](https://doi.org/10.1145/3832029)
[![Python 3.12](https://img.shields.io/badge/Python-3.12.1-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![MATLAB R2025a](https://img.shields.io/badge/MATLAB-R2025a-0076A8?style=flat-square)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

> **XGait** has been accepted by **The 2026 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp 2026)**, October 13–15, 2025, Shanghai, China.

---

## Overview

XGait is a comprehensive dataset and system for device-free indoor human tracking and gait-based identification using multi-modality wireless sensing.

### Key Features

- **Multi-Modality Sensing** — Combines WiFi CSI and Acoustic signals for wireless sensing
- **Large-Scale Dataset** — 27 subjects, 4 scenarios, comprehensive gait patterns across diverse environments

---

## Dataset

### Modalities

| Modality | Source | Capability |
|:---------|:-------|:------------|
| **WiFi CSI** | Intel 5300 NIC (1 Tx, 3 Rx) | Captures stable torso movement |
| **Acoustic** | Standard microphone/speaker (1 Tx, 3 Rx) | Captures subtle gait dynamics |
| **Video** | DSLR Camera | Ground truth for gait validation and analysis |

### Scenarios

For detailed scenario descriptions and data organization, see [data/readme.md](data/readme.md).

### Data Availability

Due to the large size of the full dataset (100GB+), it is hosted on Kaggle rather than directly in this repository.

| Platform | Link |
|:---------|:-----|
| **Kaggle** | [XGait-Paper-Dataset](https://www.kaggle.com/datasets/weilaiyilai55/xgait-paper-dataset) |

After downloading the four scenario folders (**Home**, **Laboratory**, **MeetingRoom**, **MeetingRoom-DifferentDays**) from Kaggle, place them under the `data/FusionGait/` directory:

```
data/FusionGait/
├── Home/
├── Laboratory/
├── MeetingRoom/
└── MeetingRoom-DifferentDays/
```

If you encounter any issues accessing the data, please feel free to contact us.

---

## System Requirements

| Component | Specification |
|:----------|:--------------|
| **OS** | Windows 10+ / Ubuntu 20.04 |
| **Python** | 3.12.1 |
| **MATLAB** | R2025a (Runtime or full installation) |
| **RAM** | 8 GB minimum (16 GB recommended) |
| **Storage** | 500 GB+ for full dataset |
| **GPU** | NVIDIA GPU with CUDA support (optional, for faster training) |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://xxxxx
cd XGait
```

### 2. Install Dependencies

**Windows:**
```bash
pip install -r env/requirements.win.txt
```

**Linux:**
```bash
pip install -r env/requirements.linux.txt
```

> **Note:** MATLAB Runtime R2025a must be installed separately. See the [User Guide](src/USER_GUIDE.md) for detailed setup instructions, including environment variable configuration and MATLAB package installation.

---

## Project Structure

```
XGait/
├── README.md                    # This file
├── data/                        # Dataset directory
│   ├── FusionGait/              # Multi-modality sensor data
│   │   ├── Home/
│   │   ├── Laboratory/
│   │   ├── MeetingRoom/
│   │   └── MeetingRoom-DifferentDays
│   └── GaitVideo/               # Video recordings
├── src/                         # Source code
│   ├── data_path_cfg.py         # Data path configuration
│   └── Apps/FusionGait/
│       ├── AlgorithmUtils/      # Signal processing (MATLAB-compiled)
│       ├── ConfigUtils/         # Configuration management
│       ├── GaitRecognition/     # Recognition pipeline
│       │   ├── GaitExtraction/  # Feature extraction
│       │   ├── IdentityModel/   # Classification models & training
│       │   └── Preprocessing/   # Data preprocessing & alignment
│       └── MatlabPlugins/       # MATLAB Runtime integration
├── tools/                       # Data collection tutorials & utilities
│   └── Tutorials/
│       ├── Acoustic/            # Acoustic recording guide
│       ├── Wi-Fi/               # WiFi CSI collection guide
│       └── Video/               # Video ground-truth extraction tools
└── env/                         # Environment dependencies
    ├── requirements.win.txt
    └── requirements.linux.txt
```

---

## Quick Start

### Data Collection

- **WiFi CSI** — See [WiFi Tutorial](tools/Tutorials/Wi-Fi/README.md) for MiniPC setup and CSI logging
- **Acoustic** — See [Acoustic Tutorial](tools/Tutorials/Acoustic/README.md) for distributed recording setup
- **Video Ground-truth** — See [Video Tutorial](tools/Tutorials/Video/VideoTrackExtraction/README.md) for trajectory extraction

### Feature Extraction & Model Training

For comprehensive setup instructions, environment configuration, and detailed guides on running feature extraction and model training, see the [User Guide](src/USER_GUIDE.md).

| Step | Description | Documentation |
|:-----|:------------|:---------------|
| 1. Environment Setup | Python venv, MATLAB Runtime, dependencies | [User Guide §1](src/USER_GUIDE.md#1-environment-setup) |
| 2. Feature Extraction | Process raw sensor data into gait features | [User Guide §2.2.2](src/USER_GUIDE.md#222-gaitextraction-sub-module) |
| 3. Model Training | Train & evaluate identity recognition models | [User Guide §2.5](src/USER_GUIDE.md#25-main-entry-points-for-users) |

---

## Citation

If you find XGait useful for your research, please cite our paper:

```bibtex
@article{xu2026xgait,
  author    = {Xu, Wei and Wang, Zhu and Guo, Yifan and Cheng, Changlong and
               Zhang, Yin and Ren, Zhihui and Guo, Bin and Yu, Zhiwen},
  title     = {XGait: A Multi-Modality Wireless Sensing Dataset for Indoor
               Human Tracking and Identification},
  journal   = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  volume    = {10},
  number    = {3},
  articleno = {179},
  numpages  = {27},
  year      = {2026},
  month     = {September},
  doi       = {10.1145/3832029},
  url       = {https://doi.org/10.1145/3832029}
}
```

**ACM Reference Format:**
> Wei Xu, Zhu Wang, Yifan Guo, Changlong Cheng, Yin Zhang, Zhihui Ren, Bin Guo, and Zhiwen Yu. 2026. XGait: A Multi-Modality Wireless Sensing Dataset for Indoor Human Tracking and Identification. *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.* 10, 3, Article 179 (September 2026), 27 pages. https://doi.org/10.1145/3832029

For any questions or issues, please feel free to contact us.

---

## Acknowledgments

We sincerely thank all **27 volunteers** who participated in the XGait data collection. Their time and effort made this dataset possible.

This project builds upon open-source works including Widar, Linux 802.11n CSI Tool, csiread, TFTB, Ultralytics YOLOv8, and AprilTag. We thank the authors of these projects for making their work publicly available.

---

<p align="right"><sub>Last Updated: July 27, 2026</sub></p>
