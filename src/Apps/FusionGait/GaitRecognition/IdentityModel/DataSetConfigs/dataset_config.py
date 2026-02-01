# dataset_config.py
import os.path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src.Apps.FusionGait.ConfigUtils.track_config import TrackConfig


@dataclass
class DatasetConfig:
    root: str = ""  # Dataset root directory
    doppler_dir: str = "BaseData/Doppler"  # Relative path for Doppler data
    GBVP_dir: str = "GBVP"        # Relative path for GBVP
    PPVP_dir: str = "PPVP"        # Relative path for PPVP

    FusionGBVP_dir: str = "FusionGBVP"  # Relative path for early-fusion GBVP
    FusionPPVP_dir: str = "FusionPPVP"  # Relative path for early-fusion PPVP

    dataset_name: str = "doppler"        # Single-modality dataset to use
    fusion_dataset_name: str = "doppler_fusion"  # Multi-modality (fusion) dataset to use
    users_prefix: str = "user"           # Optional: filter user folders by prefix

    # Doppler downsampling factors (integers)
    freq_down: int = 1
    time_down: int = 0
    speed_down: int = 5

    # PPVP downsampling factor
    direc_down: int = 10

    # Whether to load large files with mmap (to reduce peak memory usage)
    use_mmap: bool = True
    use_cache: bool = True  # Whether to use cache to accelerate loading (preprocess + preload into memory)

    # Whether to merge the ncnt channels into the input (True: merge into a single 2D; False: keep channel dimension)
    merge_ncnt_as_channel: bool = False
    active_channels: List[int] = field(default_factory=lambda: [0, 1, 2])  # Default active channel indices
    wifi_channels: List[int] = field(default_factory=lambda: [0, 1])       # Active channel indices for WiFi
    acoustic_channels: List[int] = field(default_factory=lambda: [0, 1, 2])  # Active channel indices for Acoustic

    # User selection
    user_num_classes: int = 5
    selected_users: np.ndarray = field(default_factory=lambda: np.array([1, 4, 5, 6, 7]))  # Select specific user IDs

    # Number of trajectories per user, or a specific trajectory list
    # Note: max_files_per_user and selected_trajs are mutually exclusive
    max_files_per_user: int = None
    selected_trajs: np.ndarray = field(default_factory=lambda: np.array([1, 2, 3, 4]))  # e.g., select trajectories
    traj_sample_percentage: int = 50

    # Dataset split strategy
    # strategy: "random" (split by ratio) or "traj" (split by specified trajectory IDs)
    split_strategy: str = "random"

    # If split_strategy == "random"
    split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)  # train, val, test

    # If split_strategy == "traj"
    train_trajs: List[int] = field(default_factory=lambda: [1, 2, 3])
    val_trajs: List[int] = field(default_factory=lambda: [4])
    test_trajs: List[int] = field(default_factory=lambda: [5])

    def __post_init__(self):
        """
        Core behavior:
        Automatically called after dataclass initialization.
        If time_down keeps the default value 0, compute it automatically.
        """
        if self.time_down == 0:
            self.time_down = self.calc_time_resample()

    def calc_time_resample(self) -> int:
        """Encapsulated logic for computing time downsampling factor."""
        # Ensure TrackConfig is defined and accessible
        try:
            track_config = TrackConfig()
            # Logic: (sample_rate / plcr_rate) * (plcr_rate / track_rate)
            # This simplifies to: sample_rate / track_rate (under consistent integer ratios)
            res = int(track_config.sample_rate / track_config.plcr_rate) * \
                  int(track_config.plcr_rate / track_config.track_rate)
            return res
        except NameError:
            print("Warning: TrackConfig is not defined; using default time_down = 1")
            return 1

    def update(self, **kwargs):
        """Allow injecting parameters from main.py with automatic type conversion."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Optional: handle a few common type conversions
                target_type = type(getattr(self, key))
                if target_type == np.ndarray and not isinstance(value, np.ndarray):
                    value = np.array(value)

                setattr(self, key, value)
        return self


def get_dataset_cfg(root_dir):
    return DatasetConfig(root=root_dir)


def get_finetune_dataset_cfg(root_dir):
    split_ratio: Tuple[float, float, float] = (0.30, 0.1, 0.6)  # train, val, test
    return DatasetConfig(root=root_dir, split_ratio=split_ratio)


def collate_fn_variable_length(batch: List[Tuple]):
    """
    Collate function for data with layout (Channel, Time, Frequency).
    T is variable-length (dim=1), and F is fixed (dim=2).
    """
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)

    xs, ys = zip(*batch)

    # 1) Normalize inputs to numpy arrays (norm_xs)
    norm_xs = []
    for i, xi in enumerate(xs):
        if isinstance(xi, torch.Tensor):
            arr = xi.detach().cpu().numpy()
        else:
            arr = np.array(xi)

        if arr.ndim == 2:
            # Assume 2D input is (T, F); expand to (1, T, F)
            arr = arr[np.newaxis, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"Sample {i} has invalid dimensions: got {arr.ndim}D")
        norm_xs.append(arr)

    # 2) Determine padding boundaries for (C, T, F)
    B = len(norm_xs)
    Cs = [arr.shape[0] for arr in norm_xs]
    Ts = [arr.shape[1] for arr in norm_xs]  # variable-length dim is index 1
    Fs = [arr.shape[2] for arr in norm_xs]  # fixed dim is index 2

    if len(set(Fs)) != 1:
        raise ValueError(f"Inconsistent frequency dimension F in the batch: {set(Fs)}.")

    C_max = max(Cs)
    T_max = max(Ts)
    F = Fs[0]

    # 3) Create zero-filled tensors
    # Note: output shape is (B, C, T, F) to match model input
    padded_x = torch.zeros((B, C_max, T_max, F), dtype=torch.float32)

    # Mask records padding on the time dimension
    mask = torch.zeros((B, T_max), dtype=torch.bool)
    labels = torch.tensor(ys, dtype=torch.long)

    # 4) Fill data
    for i, arr in enumerate(norm_xs):
        C, T, f_dim = arr.shape  # C, T, F
        arr_t = torch.from_numpy(arr).float()

        # Key: pad along T dimension
        padded_x[i, :C, :T, :] = arr_t

        # Mask: mark padded positions as True
        if T < T_max:
            mask[i, T:] = True

    return padded_x, labels, mask


def collate_fn_fusionDoppler_variable_length(batch: List[Tuple]):
    """
    Collate function for fused (multi-modality) Doppler data.

    batch format: [([data_m1, data_m2, ...], label), (...)]

    Returns:
        all_padded: List[Tensor], each tensor has shape (B, C, T_max_m, F)
        labels:     Tensor of shape (B,)
        all_masks:  List[Tensor], each mask has shape (B, T_max_m)
    """
    if len(batch) == 0:
        return [], torch.empty(0), []

    # 1) Unpack
    # samples_list: [ [data_m1, data_m2], [data_m1, data_m2], ... ]
    # labels_list:  [ label, label, ... ]
    samples_list, labels_list = zip(*batch)
    num_modalities = len(samples_list[0])
    batch_size = len(batch)

    all_padded = []
    all_masks = []

    # 2) Pad each modality independently
    for m_idx in range(num_modalities):
        # Collect all samples of the current modality: List[(C, T, F)]
        curr_m_data = [s[m_idx] for s in samples_list]

        # Normalize to numpy and collect dimension info
        norm_xs = []
        for xi in curr_m_data:
            arr = xi.detach().cpu().numpy() if isinstance(xi, torch.Tensor) else np.array(xi)
            if arr.ndim == 2:  # (T, F) -> (1, T, F)
                arr = arr[np.newaxis, :, :]
            norm_xs.append(arr)

        # Compute max dims within this modality in the current batch
        Cs = [arr.shape[0] for arr in norm_xs]
        Ts = [arr.shape[1] for arr in norm_xs]
        Fs = [arr.shape[2] for arr in norm_xs]

        C_max = max(Cs)
        T_max = max(Ts)
        F_fixed = Fs[0]  # assume F is fixed within the modality

        # Allocate padding containers
        padded_x = torch.zeros((batch_size, C_max, T_max, F_fixed), dtype=torch.float32)
        mask = torch.zeros((batch_size, T_max), dtype=torch.bool)

        # Fill
        for i, arr in enumerate(norm_xs):
            c, t, f = arr.shape
            padded_x[i, :c, :t, :f] = torch.from_numpy(arr).float()
            if t < T_max:
                mask[i, t:] = True

        all_padded.append(padded_x)
        all_masks.append(mask)

    labels = torch.tensor(labels_list, dtype=torch.long)
    return all_padded, labels, all_masks


def collate_fn_fusionPPVP_variable_length(batch: List[Tuple]):
    """
    Collate function optimized for PPVP fusion datasets.

    Supports multi-modality inputs (WiFi/Acoustic, etc.) where each modality
    may have its own time length T that needs independent padding.

    batch format: [([data_m1, data_m2], label), ...]
    Returns:
        all_padded: List[Tensor], each tensor has shape (B, C, T_max_m, F)
        labels:     Tensor of shape (B,)
        all_masks:  List[Tensor], each mask has shape (B, T_max_m)
    """
    if not batch:
        return [], torch.empty(0), []

    # 1) Unpack
    samples_list, labels_list = zip(*batch)
    num_modalities = len(samples_list[0])
    batch_size = len(batch)

    all_padded = []
    all_masks = []

    # 2) For each modality (m1, m2...), compute max T within the batch independently
    for m_idx in range(num_modalities):
        # Collect all data for the current modality
        curr_m_data = [s[m_idx] for s in samples_list]

        # Normalize dimension info (handle Tensor/Numpy mixed inputs)
        norm_xs = []
        for xi in curr_m_data:
            if isinstance(xi, torch.Tensor):
                arr = xi
            else:
                arr = torch.from_numpy(np.array(xi)).float()

            # Ensure (C, T, F)
            if arr.ndim == 2:  # (T, F) -> (1, T, F)
                arr = arr.unsqueeze(0)
            norm_xs.append(arr)

        # Get max C/T/F for this modality in the current batch
        # Note: for PPVP, F is typically fixed (e.g., 150), but we keep it flexible here.
        c_max = max(arr.shape[0] for arr in norm_xs)
        t_max = max(arr.shape[1] for arr in norm_xs)
        f_max = max(arr.shape[2] for arr in norm_xs)

        # Allocate containers
        padded_x = torch.zeros((batch_size, c_max, t_max, f_max), dtype=torch.float32)
        mask = torch.zeros((batch_size, t_max), dtype=torch.bool)

        # Fill tensors
        for i, arr in enumerate(norm_xs):
            c, t, f = arr.shape
            # This fill logic handles possible inconsistencies across C/T/F
            padded_x[i, :c, :t, :f] = arr

            # Mask: True indicates padding positions
            if t < t_max:
                mask[i, t:] = True

        all_padded.append(padded_x)
        all_masks.append(mask)

    labels = torch.tensor(labels_list, dtype=torch.long)

    # This return format matches your FusionInputTrainer.prepare_batch perfectly
    return all_padded, labels, all_masks
