# XGait Project User Guide

## 1. Environment Setup

This project supports both **Windows** and **Linux** environments.

### 1.1 System Requirements
- **Operating System**: Windows 10/11 or Linux (Ubuntu 20.04+ recommended)
  - **Linux**: Recommended for batch processing large datasets (requires high-performance server with large storage)
  - **Windows**: Primarily for development, visualization, and trajectory tracking verification
- **Python Version**: Python 3.12.1
- **MATLAB Environment**: 
  - MATLAB R2025a
  - MATLAB Runtime R2025a
- **IDE**: JetBrains PyCharm (Community or Professional Edition)

### 1.2 Installation Steps

#### Step 1: Open the Project
1. Launch PyCharm.
2. Select `Open` and open the project folder `XGait`.

#### Step 2: Create Virtual Environment
It is recommended to use PyCharm's built-in Terminal to create a clean virtual environment.

1. Click the **Terminal** tab at the bottom of PyCharm.
2. Ensure the current path is the project root directory.
3. Run the following command to create a virtual environment (named `venv`):
   
   **Windows:**
   ```powershell
   py -3.12.1 -m venv venv
   ```
   
   **Linux:**
   ```bash
   python3.12.1 -m venv venv
   ```

#### Step 3: Activate Virtual Environment
**Windows:**
```powershell
.\venv\Scripts\activate
```

**Linux:**
```bash
source venv/bin/activate
```
*After successful activation, the command line prefix will display `(venv)`.*

#### Step 4: Install Python Dependencies
Choose the appropriate dependency file based on your operating system. Execute in the activated virtual environment terminal:

```powershell
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# For Windows users
pip install -r env/requirements.win.txt

# For Linux users
pip install -r env/requirements.linux.txt
```

#### Step 5: Configure MATLAB Runtime System Variable
Before installing MATLAB packages, you must set the `MATLAB_RUNTIME_ROOT` environment variable.

**Windows:**
1. Press `Win + X` and select "System (Settings)".
2. Click "Advanced system settings" → "Environment Variables".
3. Click "New" under "System variables".
4. Set the variable name as `MATLAB_RUNTIME_ROOT`.
5. Set the variable value to your MATLAB Runtime R2025a installation path (e.g., `C:\Program Files\MATLAB\MATLAB Runtime\R2025a`).
6. Click "OK" and restart your terminal for changes to take effect.

**Linux:**

> **Important Note**: On Linux, environment variables are typically loaded through shell startup scripts (like `.bashrc`). PyCharm launched from the desktop may not inherit these shell environment variables. Choose the configuration method that matches your usage pattern.

##### Option 1: Running from Terminal (Command Line)

If you prefer running scripts directly from the command line:

1. Open a terminal.
2. Modify the MATLAB Runtime path in the commands below to match your installation (e.g., `/usr/local/MATLAB/MATLAB_Runtime/R2025a`), then execute:

```bash
# Set MATLAB Runtime root directory
echo 'export MATLAB_RUNTIME_ROOT="/usr/local/MATLAB/MATLAB_Runtime/R2025a"' >> ~/.bashrc

# Update LD_LIBRARY_PATH to include necessary dynamic libraries
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+:}$MATLAB_RUNTIME_ROOT/runtime/glnxa64:$MATLAB_RUNTIME_ROOT/bin/glnxa64:$MATLAB_RUNTIME_ROOT/sys/os/glnxa64:$MATLAB_RUNTIME_ROOT/sys/opengl/lib/glnxa64"' >> ~/.bashrc

# Activate the configuration
source ~/.bashrc
```

##### Option 2: Running from PyCharm IDE

If you run programs using PyCharm's "Run" button, you must explicitly configure environment variables in the **Run Configurations**:

1. Click the run/debug configuration dropdown in the top-right corner of PyCharm.
2. Select **"Edit Configurations..."**.
3. Click the **+** button in the top-left and select **Python** from the dropdown list.
4. Fill in the configuration:
   - **Name**: Enter a descriptive name (e.g., `single_extractor_main`)
   - **Script path**: Select the Python script you want to run (e.g., the full path to `single_extractor_main.py`)
   - **Environment variables**: Click the file icon or text field and add a new variable:
     - **Name**: `MATLAB_RUNTIME_ROOT`
     - **Value**: Your MATLAB Runtime installation path (e.g., `/usr/local/MATLAB/MATLAB_Runtime/R2025a`)
5. Click **Apply** and **OK** to save.

> **Note**: Without this configuration, the program may fail with errors about missing MATLAB dynamic libraries (`.so` files).

#### Step 6: Install MATLAB Algorithm Packages
This project includes compiled MATLAB algorithm components that must be installed using a dedicated script.

1. Ensure **MATLAB Runtime R2025a** is correctly installed.
2. From the project root directory, run:
   
   ```powershell
   python src/Apps/FusionGait/MatlabPlugins/install_matlab_packages.py
   ```
   
   *This script will automatically detect your operating system (Windows/Linux) and install the corresponding algorithm packages located in `src/Apps/FusionGait/MatlabPlugins/build_py`.*

#### Step 7: Verify Installation
After installation is complete, verify your environment with:

1. **Check dependency status**:
   ```powershell
   pip check
   ```
   *Should display "No broken requirements found".*

2. **Check package list**:
   ```powershell
   pip list
   ```
   *Verify that the list includes the base dependencies and MATLAB-related packages installed in Step 6.*

### 1.3 Configure PyCharm Interpreter
To enable proper code navigation and intelligent hints in PyCharm:
1. Click the Python interpreter status bar at the bottom right of PyCharm.
2. Select `Add New Interpreter` → `Add Local Interpreter...`.
3. Choose `Virtualenv Environment`.
4. In the `Existing` option, browse and select the interpreter from your created `venv` environment:
   - Windows path: `venv\Scripts\python.exe`
   - Linux path: `venv/bin/python`
5. Click `OK` to complete the configuration.

## 2. Project Code Structure and Architecture

### 2.1 Overview
The XGait project is organized in `src/Apps/FusionGait/` with four main modules:
- **GaitRecognition**: Core gait recognition pipeline
- **AlgorithmUtils**: Signal processing algorithms
- **ConfigUtils**: Configuration and experiment management
- **MatlabPlugins**: MATLAB Runtime integration

### 2.2 GaitRecognition Module: Core Recognition Pipeline

**Location**: `src/Apps/FusionGait/GaitRecognition/`

This module handles the complete gait recognition workflow from feature extraction to model training.

#### 2.2.1 Sub-modules Overview

**a) GaitExtraction** - Raw Feature Extraction from Sensor Data
**b) Preprocessing** - Data Preprocessing and Normalization
**c) IdentityModel** - Data Loading, Model Training and Evaluation

#### 2.2.2 GaitExtraction Sub-module
**Entry Points for Feature Extraction**:
- **`fusion_extract_main.py`**: Multi-modal (WiFi + Acoustic) feature extraction
  - Processes multiple modalities simultaneously
  - Multi-processing for parallel execution
  - Generates intermediate data in `exeDir`
  
- **`single_extractor_main.py`**: Single-modality feature extraction
  - Processes individual modalities independently
  - Lightweight alternative when fusion not needed

- **`utils.py`**: Supporting utilities
  - Signal loading and preprocessing
  - Spectrum generation (Doppler, BVP, etc.)
  - User and tracker management

#### 2.2.3 Preprocessing Sub-module
**Data Preprocessing Utilities**:
- **`fusion_preprocesor.py`**: Multi-modal preprocessing
- **`calc_velocity.py`**: Velocity computation

#### 2.2.4 IdentityModel Sub-module - Model Training Pipeline

**Location**: `src/Apps/FusionGait/GaitRecognition/IdentityModel/`

This is the core training pipeline that loads features from `exeDir` and performs model training.

**Sub-Components**:

**a) DataSetConfigs** - Dataset Loading & Management
- **`dataloader.py`**: Dataset classes
  - `DopplerDataset`: Single-modality dataset loading
  - `DopplerFusionDataset`: Multi-modal fusion dataset loading
  - Loads PPVP features from `exeDir/{user}/wifi/PPVP/` and `exeDir/{user}/acoustic/PPVP/`
  - Supports metadata-driven batch loading with caching
  
- **`data_factory.py`**: DataFactory class
  - `create_loaders()`: Creates PyTorch DataLoader instances
  - Handles single-modality vs. fusion modality routing
  - Train/validation/test split management
  - Device-agnostic (CPU/GPU) batch preparation

- **`dataset_config.py`**: Dataset configuration
  - Path specifications for PPVP features
  - Batch collation functions for variable-length sequences
  - Data preprocessing parameters

- **`dataset_registry.py`**: Dataset registration system
  - Maps dataset names to dataset classes
  - Enables extensibility for new dataset types

**b) ModelConfigs** - Model Architecture & Training

- **`models.py`**: Model architectures
  - Base models: CNN, ResNet
  - Single-modality: CNN, Conv3D-LSTM, Transformer variants
  - Fusion models: Late Fusion CNN, Fusion Transformer, Cross-Modal Attention
  
- **`model_config.py`**: Model configuration management
  - Model hyperparameters (embedding_dim, layer configs, etc.)
  - Architecture-specific settings
  
- **`model_factory.py`**: ModelFactory class
  - Model registration and instantiation
  - Checkpoint management and save paths
  
- **`trainer_factory.py`**: TrainerFactory class
  - Creates trainer instances (single-modality or fusion)
  - Handles training loop management

**c) DopplerBaseLearning** - Training Entry Points

- **`main.py`**: Single-scenario training
  - Control panel for experiment configuration
  - Sets model, modality, hyperparameters, phase (train/eval/test)
  - Orchestrates DataFactory → ModelFactory → Trainer pipeline
  - Example:
    ```python
    TARGET_MODEL = "fusion_cnn"
    TARGET_MODALITY = ["wifi", "acoustic"]  # or single modality
    CURRENT_PHASE = "train"  # or "eval", "test"
    ```

- **`cross_scene_main.py`**: Cross-scenario training
  - Trains on one scenario, evaluates on another
  - Tests model generalization across environments (Laboratory ↔ Home)

- **`run_config.py`**: Runtime configuration dataclass
  - Task specification (gait_identity, etc.)
  - Trainer type: "single" or "fusion"
  - Phase: "train", "eval", or "test"
  - Model name and dataset name
  - Modality selection

- **`train_config.py`**: Training hyperparameters
  - Epochs, learning rate, batch size
  - Patience (early stopping)
  - Optimizer and scheduler settings

**d) Utils** - Utilities

- **`logger.py`**: Experiment logging
- **`metrics.py`**: Evaluation metrics (accuracy, precision, recall, etc.)
- **`preprocess.py`**: Data normalization and preprocessing
- **`constants.py`**: Constants and enums for model names, modalities, phases

### 2.3 Supporting Modules

#### 2.3.1 AlgorithmUtils Module
**Location**: `src/Apps/FusionGait/AlgorithmUtils/`

Core signal processing algorithms (compiled MATLAB components):
- **`position_tracker.py`**: Multi-modal position tracking
- **`bvp_generator.py`**: Basic velocity profile generation
- **`dfss_generator.py`**: Doppler frequency spectrum surface generation
- **`gbvp_generator.py`**: Grid-based velocity profile generation
- **`ppvp_generator.py`**: Perspective projection velocity profile generation

#### 2.3.2 ConfigUtils Module
**Location**: `src/Apps/FusionGait/ConfigUtils/`

Configuration and experiment management:
- **`expriment_config.py`**: Experiment-level configuration (data paths, parameters)
- **`track_config.py`**: Ground truth trajectory configuration
- **`dfs_config.py`**: Doppler frequency spectrum configuration
- **`User.py`**: User data structure and utilities

#### 2.3.3 MatlabPlugins Module
**Location**: `src/Apps/FusionGait/MatlabPlugins/`

MATLAB Runtime integration:
- **`install_matlab_packages.py`**: Install compiled MATLAB algorithm packages
- **`init_matlab_path.py`**: Initialize MATLAB Runtime environment
- **`matlab_runtime_singleton.py`**: Singleton pattern for MATLAB engine management
- **`compile_matlab_packages.py`**: Compile MATLAB code (developer use)
- **`build_py/`**: Pre-compiled MATLAB algorithm packages

### 2.4 Processing Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW                             │
└─────────────────────────────────────────────────────────────────┘

Raw Sensor Data (data/FusionGait/)
    │
    ├─► [GaitExtraction] (fusion_extract_main.py / single_extractor_main.py)
    │   └─ Processes raw WiFi CSI and Acoustic signals
    │   └─ Applies MATLAB algorithms (Doppler, spectrum generation)
    │   └─ Generates intermediate features
    │
    └─► exeDir: {scenario}/user{id}/
        ├── wifi/BaseData/ & wifi/PPVP/
        ├── acoustic/BaseData/ & acoustic/PPVP/
        └── fusion/FusionPPVP/
            │
            ├─► [IdentityModel] (main.py / cross_scene_main.py)
            │   ├─ DataFactory loads PPVP features
            │   ├─ Splits into train/val/test sets
            │   ├─ Creates PyTorch DataLoaders
            │   │
            │   ├─ ModelFactory creates model architecture
            │   ├─ TrainerFactory creates trainer
            │   │
            │   └─ Model Training Loop
            │       └─ Trains on exeDir PPVP features
            │       └─ Evaluates on test set
            │
            └─► Model Checkpoints (./checkpoints/)
                └─ Final Gait Identity Recognition Models
```

### 2.5 Main Entry Points for Users

#### Entry Point 1: Feature Extraction (Required First Step)
**Purpose**: Extract gait features from raw sensor data and populate `exeDir`

**Single-Modality Extraction**:
```bash
python src/Apps/FusionGait/GaitRecognition/GaitExtraction/single_extractor_main.py
```

**Multi-Modal Fusion Extraction** (Recommended):
```bash
python src/Apps/FusionGait/GaitRecognition/GaitExtraction/fusion_extract_main.py
```

**Configuration** (in `ExperimentConfig`):
- Input data source (Wi-Fi, acoustic, or both)
- Input/output directories
- User and scenario information  
- Tracking parameters

**Output**: Populates `src/Apps/FusionGait/exeDir/` with locations and features

#### Entry Point 2: Model Training
**Purpose**: Train and evaluate gait identity recognition models using features from `exeDir`

**Location**: `src/Apps/FusionGait/GaitRecognition/IdentityModel/DopplerBaseLearning/`

**Training Workflow**:

All configuration is done in `main.py`. Modify parameters in the "Experiment control panel" section (lines 26-74):

##### Case 1: Single-Modality Training (WiFi Only)

**Edit `main.py` control panel**:
```python
# ━━━ Step 1: Select single-modality model (without "Fusion") ━━━
TARGET_MODEL = ModelNames.CNN  # or ResNet, Transformer, etc. (no "Fusion" in name)

# ━━━ Step 2: Select one modality ━━━
TARGET_MODALITY = Modalities.WIFI  # Choose: Modalities.WIFI or Modalities.ACOUSTIC

# ━━━ Step 3: Configure dataset parameters ━━━
BASE_DATASET_PARAMS = {
    "use_cache": True,
    "dataset_name": "doppler",        # Single-modality dataset
    # "dataset_name": "ppvp",         # Alternative single-modality dataset
    "traj_sample_percentage": 100,
    "split_strategy": "traj"
}

# ━━━ Step 4: Set phase ━━━
CURRENT_PHASE = Phases.TRAIN  # or Phases.TEST
EXP_VERSION = "v1_single_wifi"  # Experiment tag

# ━━━ Step 5: Other settings (keep as is) ━━━
TRAIN_PARAMS = {
    "num_workers": 10,
    "epochs": 50,
    "patience": 20
}
MODEL_PARAMS = {
    "name": TARGET_MODEL,
    "version": EXP_VERSION,
    "embedding_dim": 256
}
```

**Inside `main.py`, the training pipeline automatically handles**:
- `RunConfig` trainer type → Set to "single" automatically
- Dataset loading → Loads only WIFI/ACOUSTIC features from exeDir
- Model creation → Uses single-modality architecture

**Execute**:
```bash
python src/Apps/FusionGait/GaitRecognition/IdentityModel/DopplerBaseLearning/main.py
```

---

##### Case 2: Single-Modality Training (Acoustic Only)

**Edit `main.py` control panel** (same as WiFi, just change modality):
```python
TARGET_MODEL = ModelNames.CNN  # Single-modality model

TARGET_MODALITY = Modalities.ACOUSTIC  # Switch to acoustic

BASE_DATASET_PARAMS = {
    "use_cache": True,
    "dataset_name": "doppler",  # Single-modality dataset
    "traj_sample_percentage": 100,
    "split_strategy": "traj"
}

CURRENT_PHASE = Phases.TRAIN
EXP_VERSION = "v1_single_acoustic"  # Different tag
```

---

##### Case 3: Multi-Modal Fusion Training (WiFi + Acoustic)

**Edit `main.py` control panel**:
```python
# ━━━ Step 1: Select fusion model (with "Fusion" in name) ━━━
TARGET_MODEL = ModelNames.FUSION_CNN  # or FusionTransformer, etc.

# ━━━ Step 2: Select multiple modalities as a list ━━━
TARGET_MODALITY = [Modalities.WIFI, Modalities.ACOUSTIC]

# ━━━ Step 3: Configure dataset for fusion ━━━
BASE_DATASET_PARAMS = {
    "use_cache": True,
    "dataset_name": "doppler",           # Will be used for individual modalities
    "fusion_dataset_name": "doppler_fusion",  # Fusion dataset
    # "fusion_dataset_name": "ppvp_fusion",  # Alternative fusion dataset
    "traj_sample_percentage": 100,
    "split_strategy": "traj"
}

# ━━━ Step 4: Set phase ━━━
CURRENT_PHASE = Phases.TRAIN
EXP_VERSION = "v1_fusion"  # Experiment tag

# ━━━ Other settings ━━━
TRAIN_PARAMS = {
    "num_workers": 10,
    "epochs": 50,
    "patience": 20
}
MODEL_PARAMS = {
    "name": TARGET_MODEL,
    "version": EXP_VERSION,
    "embedding_dim": 256
}
```

**Inside `main.py`, the training pipeline automatically handles**:
- `RunConfig` trainer type → Set to "fusion" automatically
- Dataset loading → Loads both WIFI and ACOUSTIC features from exeDir
- DataFactory → Creates fusion dataset that combines modalities
- Model creation → Uses fusion architecture combining multiple inputs

**Execute**:
```bash
python src/Apps/FusionGait/GaitRecognition/IdentityModel/DopplerBaseLearning/main.py
```

---

**Quick Configuration Reference**:

| Configuration | Single-Modality (WiFi) | Single-Modality (Acoustic) | Multi-Modal Fusion |
|---|---|---|---|
| **TARGET_MODEL** | CNN, ResNet, Transformer | CNN, ResNet, Transformer | FUSION_CNN, FusionTransformer |
| **TARGET_MODALITY** | `Modalities.WIFI` | `Modalities.ACOUSTIC` | `[Modalities.WIFI, Modalities.ACOUSTIC]` |
| **dataset_name** | "doppler" or "ppvp" | "doppler" or "ppvp" | "doppler" or "ppvp" |
| **fusion_dataset_name** | ✗ Not used | ✗ Not used | "doppler_fusion" or "ppvp_fusion" |
| **RunConfig trainer** | "single" | "single" | "fusion" |
| **PPVP source** | exeDir/.../wifi/PPVP/ | exeDir/.../acoustic/PPVP/ | exeDir/.../wifi/PPVP/ + exeDir/.../acoustic/PPVP/ |

**Training Steps** (same for all configurations):

1. Modify parameters in `main.py` (control panel section)
2. Execute training script:
   ```bash
   python src/Apps/FusionGait/GaitRecognition/IdentityModel/DopplerBaseLearning/main.py
   ```
3. For cross-scenario evaluation (train on one scenario, test on another):
   ```bash
   python src/Apps/FusionGait/GaitRecognition/IdentityModel/DopplerBaseLearning/cross_scene_main.py
   ```

**Automatic Pipeline** (handled internally):
- DataFactory → Loads appropriate PPVP features from exeDir
- Splits data into train/val/test according to scenario parameters
- Creates PyTorch DataLoaders
- ModelFactory → Creates model architecture
- TrainerFactory → Creates trainer instance
- Training Loop → Trains on features, evaluates on test set
- Checkpoints saved to `./checkpoints/`

### 2.6 Configuration Files

**Key Configuration Files**:
- `src/data_path_cfg.py`: Define data directory paths
- `data/FusionGait/Home/senario_cfg.json`: Scenario-specific parameters
- `data/FusionGait/Laboratory/senario_cfg.json`: Laboratory scenario parameters

Modify these files according to your dataset location and experimental requirements.

### 2.7 Execution Directory (exeDir) Structure

The `exeDir` directory is automatically generated during feature extraction and stores all intermediate data products from the gait recognition pipeline.

**Location**: `src/Apps/FusionGait/exeDir/`

**Top-Level Organization**:
```
exeDir/
├── Home/                    # Scenario: Home environment
└── Laboratory/              # Scenario: Laboratory environment
    └── user{id}/            # User identifier (e.g., user1, user2)
        ├── wifi/            # WiFi CSI modality features
        ├── acoustic/        # Acoustic signal modality features
        └── fusion/          # Multi-modal fusion features
```

#### 2.7.1 Acoustic/WiFi Feature Directories

Both `acoustic/` and `wifi/` directories have identical organizational structures:

```
acoustic/ (or wifi/)
├── basedata_metadata.npy    # Metadata index file (paths to intermediate data)
├── BaseData/                # Raw intermediate processed data
│   ├── Doppler/             # Raw Doppler frequency spectrum matrices
│   ├── Freq/                # Frequency domain information
│   ├── Duration/            # Signal duration metadata
│   ├── Speeds/              # Extracted velocity information
│   └── Locations/           # Tracked trajectory coordinates
└── PPVP/                    # Perspective Projection Velocity Profile (final features)
    ├── basedata_metadata.npy
    └── track{i}_rep{j}_pro.npy  # Per-track, per-repetition feature files
```

**Data Content**:

- **basedata_metadata.npy**: NumPy array storing paths to BaseData files
  - Enables efficient metadata-driven batch loading during training
  - Each entry maps to a corresponding file in BaseData/

- **BaseData/**: Intermediate signal processing results
  - **Doppler/**: Raw 2D Doppler frequency spectrum (time × frequency)
  - **Freq/**: Frequency-domain features extracted from radar signals
  - **Duration/**: Signal acquisition duration and timing information
  - **Speeds/**: Doppler-derived velocity profile data
  - **Locations/**: Trajectory positions from ground truth (fusion tracking) or algorithm output

- **PPVP/** (Perspective Projection Velocity Profile):
  - High-level gait feature representation optimized for classification
  - Converts raw Doppler data into perspective-projected coordinates
  - File naming: `track{track_id}_rep{repetition_id}_pro.npy`
  - Used as primary input for model training

#### 2.7.2 Fusion Directory

```
fusion/
└── FusionPPVP/              # Multi-modal combined features
    ├── pro_metadata.npy
    └── track{i}_rep{j}_pro.npy
```

**Data Content**:

- **FusionPPVP/**: Unified multi-modal gait features
  - Combines WiFi and acoustic modality features for joint representation
  - Uses consistent PPVP format across modalities
  - Serves as primary input for multi-modal identity recognition models

### 2.8 Feature Extraction to Training Data Flow

```
Raw Sensor Data (data/FusionGait/)
    ↓
[GaitExtraction] → Processes raw signals with MATLAB algorithms
    ↓
exeDir: {scenario}/user{id}/{modality}/BaseData/
    ├─ Doppler spectra, Duration, Speeds, Locations
    └─ Indexed with metadata for efficient access
    ↓
[Feature Processing] → Converts to Perspective Projection format
    ↓
exeDir: {scenario}/user{id}/{modality}/PPVP/
    └─ Standardized velocity profile features
    ↓
[Fusion (if enabled)] → Combines multi-modality features
    ↓
exeDir: {scenario}/user{id}/fusion/FusionPPVP/
    └─ Unified multi-modal feature representation
    ↓
[DataFactory] → Loads PPVP from exeDir
    ├─ train/val/test split
    └─ Creates PyTorch DataLoaders
    ↓
[Model Training] → Trains on DataLoader batches
    ↓
Model Checkpoints (./checkpoints/)
```

**Key Design Points**:
- PPVP features in exeDir are reusable for different model architectures
- Metadata files enable efficient batch loading without re-scanning
- File naming maintains traceability to original dataset sequences
- Intermediate BaseData files persist for debugging and feature analysis

## 3. Quick Reference: Document Structure

This guide contains the following sections:

| Section | Content |
|---------|---------|
| **Section 1** | Environment Setup (7 steps including MATLAB Runtime configuration) |
| **Section 2.1-2.3** | Module Architecture (GaitRecognition, AlgorithmUtils, ConfigUtils, MatlabPlugins) |
| **Section 2.4** | Processing Pipeline Overview (visual workflow diagram) |
| **Section 2.5** | Main Entry Points (Feature Extraction and Model Training with examples) |
| **Section 2.6** | Configuration Files (key files to modify) |
| **Section 2.7-2.8** | exeDir Structure and Data Flow (intermediate feature storage and loading) |

## 4. Troubleshooting Quick Links

- **Environment Issues**: See Section 1.3 (PyCharm Interpreter Configuration)
- **MATLAB Runtime Setup**: See Section 1.2, Step 5
- **Feature Extraction**: See Section 2.5, Entry Point 1
- **Model Training**: See Section 2.5, Entry Point 2
- **Data Organization**: See Section 2.7 (exeDir Structure)
- **Understanding Pipeline**: See Section 2.4 (Processing Pipeline Overview) and 2.8 (Feature to Training Data Flow)

## 5. Acknowledgments

The matlab code in this project references open-source implementations from the **Widar** project for WiFi-based human tracking algorithms.

- **Widar**: Decimeter-Level Passive Tracking via Velocity Monitoring with Commodity Wi-Fi
  - link:http://tns.thss.tsinghua.edu.cn/wifiradar/WidarProject.zip
