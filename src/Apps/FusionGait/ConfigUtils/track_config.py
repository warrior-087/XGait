from dataclasses import dataclass
import numpy as np
import os

# Tracking parameter configuration (values assigned later)
@dataclass
class TrackConfig:
    plcr_rate: int = 64  # PLCR calculation rate
    track_rate: int = 16  # Tracking rate
    sample_rate: int = 1000  # Sample_rate: this is NOT the sampling rate of the raw signal, but the sampling rate of the spectrum.
                             # For WiFi, the spectrum is obtained via wavelet, so it matches the signal sampling rate.
                             # For acoustic, STFT is already sampled/aggregated, so no further downsampling is needed.
    wave_length: float = None  # wavelength

    tx_loc: np.ndarray = None  # to be assigned
    rx_locs: np.ndarray = None  # to be assigned
    trajectories: dict = None  # to be assigned

class GroundTruthTrajectory:

    def __init__(self, data_root: str):
        self.data_root = data_root

    # ================================================================
    #                      1. File-loading mode
    # ================================================================
    def load_from_file(self, filepath: str):
        """Load a trajectory from a file"""
        if not os.path.exists(filepath):
            return None  # the file might not exist

        traj = np.load(filepath)
        return traj  # shape: (N, D)

    # ================================================================
    #                      2. Trajectory-generation mode
    # ================================================================
    def generate(self, config, nums=200):
        """Automatically generate a trajectory based on the configuration"""
        traj_type = config.type

        if traj_type == "line":
            return self._generate_linear(config, nums)
        elif traj_type == "multilines":
            return self._generate_multi_linear(config, nums)
        else:
            raise NotImplementedError(f"Unknown trajectory type: {traj_type}")

    def _generate_multi_linear(self, configs, nums):
        """
        Generate a concatenated trajectory from multiple linear segment configurations.

        Args:
            configs: list, each element has .start and .end, both are point coordinates convertible to ndarray
            nums_list: list, same length as configs, interpolation point count for each segment

        Returns:
            traj: ndarray, shape (dim, total_nums)
        """
        all_segments = []

        for cfg in zip(configs.segments):
            start = cfg[0]["start"]
            end = cfg[0]["end"]

            t = np.linspace(0, 1, nums)
            seg = start + (end - start) * t[:, None]  # shape (nums, dim)
            all_segments.append(seg)

        # Concatenate into a continuous trajectory
        traj = np.concatenate(all_segments, axis=0)  # (sum(nums), dim)

        return traj.T  # (dim, total_nums)

    def _generate_linear(self, config, nums):
        start = np.array(config.start)  # (dim,)
        end = np.array(config.end)

        # Linear interpolation
        t = np.linspace(0, 1, nums)  # (nums,)
        traj = start + (end - start) * t[:, None]  # (nums, dim)

        return traj.T  # (dim, nums)

    # ================================================================
    #                 Unified interface: users call only one function
    # ================================================================
    def get_trajectory(self, config, filename: str = None):
        """
        If filename is provided, load from file;
        otherwise, generate automatically from config.
        """
        if filename is not None:
            if self.load_from_file(filename) is not None:
                return self.load_from_file(filename)
            else:
                return self.generate(config)
        else:
            return self.generate(config)
