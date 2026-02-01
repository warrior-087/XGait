# XGait: A Multi-Modality Wireless Sensing Dataset for Indoor Human Tracking and Identification

> **Status:** This repository contains the official implementation and dataset for **XGait: A Multi-Modality Wireless Sensing Dataset for Indoor Human Tracking and Identification**. The paper is currently **under review**.

## Overview

XGait is a comprehensive dataset and system for device-free indoor human tracking and gait-based identification using multi-modality wireless sensing. 

### Key Features

- **Multi-Modality Sensing**: Combines WiFi CSI and Acoustic signals for wireless sensing
- **Large-Scale Dataset**: Comprehensive gait patterns from multiple users and scenarios

## Dataset Description

### Modalities

| Modality | Source | Capability |
|----------|--------|-----------|
| **WiFi CSI** | Intel 5300 NIC (1Tx,3Rx) | Captures stable torso movement |
| **Acoustic** | Standard microphone/speaker (1Tx,3Rx) | Captures subtle gait dynamics|
| **Video** |  DSLR Camera | Ground truth for gait validation and analysis |

### Scenarios and Data Organization

For detailed scenario descriptions and the full data organization, see [data/readme.md](data/readme.md).


## System Requirements

| Component | Specification |
|-----------|----------------|
| **OS** | Windows 10+ / Ubuntu 20.04
| **Python** | 3.12.1 |
| **Matlab** | R2025a |
| **RAM** | 8GB minimum (16GB recommended) |
| **Storage** | 500GB+ for full dataset |
| **GPU** | NVIDIA GPU with CUDA support (optional, for faster processing) |

## Installation

### 1. Clone Repository

```bash
git clone https://xxxxx
cd XGait
```

### 2. Install Dependencies

**On Windows:**
```bash
pip install -r env/requirements.win.txt
```

**On Linux:**
```bash
pip install -r env/requirements.linux.txt
```


## Project Structure

```
XGait/
├── README.md                 # This file
├── data/                     # Datasets
│   ├── FusionGait/          # Multi-modality sensor data
│   │   ├── Home/
│   │   └── Laboratory/
│   └── GaitVideo/           # Video recordings
├── src/                      # Source code
│   ├── data_path_cfg.py     # Data path configuration
│   ├── Apps/
│   │   └── FusionGait/
│   │       ├── AlgorithmUtils/      # Signal processing 
│   │       ├── ConfigUtils/          # Configuration management
│   │       ├── GaitRecognition/      # Recognition pipeline
│   │       │   ├── GaitExtraction/   # Feature extraction
│   │       │   ├── IdentityModel/    # Classification models
│   │       │   └── Preprocessing/    # Data preprocessing
│   │       ├── MatlabPlugins/        # MATLAB integration
│   │       └── exeDir/               # Program execution directory (auto-generated, stores intermediate features)
│   └── ...
├── tools/                    # Utilities and tutorials
│   ├── Tutorials/
│   │   ├── Acoustic/        # Acoustic data collection guide
│   │   ├── Wi-Fi/           # WiFi CSI collection guide
│   │   └── Video/           # Video processing tools
│   └── Image_res/           # Resource images
└── env/                      # Environment configurations
    ├── requirements.win.txt  # Windows dependencies
    └── requirements.linux.txt # Linux dependencies
```

## Quick Start

### Data Collection

#### WiFi CSI Collection
Refer to [WiFi Tutorial](tools/Tutorials/Wi-Fi/README.md) for detailed setup instructions.

#### Acoustic Data Collection
Refer to [Acoustic Tutorial](tools/Tutorials/Acoustic/README.md) for detailed setup instructions.

#### Ground-truth Extraction
Refer to [Video Tutorial](tools/Tutorials/Video/README.md) for detailed setup instructions.

### Project Execution

For comprehensive setup instructions, environment configuration, and detailed guides on running feature extraction and model training, refer to [User Guide](src/USER_GUIDE.md).

The guide covers:
- **Environment Setup**: Python virtual environment, MATLAB Runtime configuration
- **Feature Extraction**: Single-modality and multi-modal fusion feature extraction
- **Model Training**: Gait identity recognition model training and evaluation
- **Project Architecture**: Complete code structure and module descriptions



**Last Updated:** January 30, 2026
