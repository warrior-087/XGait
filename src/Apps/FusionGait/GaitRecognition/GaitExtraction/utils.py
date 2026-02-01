import os
import re
from pathlib import Path
from typing import List

import numpy as np

from src.Apps.FusionGait.AlgorithmUtils.bvp_generator import PhysicalSetup, VelocityMappingParams
from src.Apps.FusionGait.AlgorithmUtils.dfss_generator import DFSSGeneratorFactory
from src.Apps.FusionGait.AlgorithmUtils.gbvp_generator import GBVPGenerator, FusionGBVPGenerator
from src.Apps.FusionGait.AlgorithmUtils.position_tracker import PositionTracker
from src.Apps.FusionGait.AlgorithmUtils.ppvp_generator import PPVPGenerator, PolarVelocityMappingParams, \
    FusionPPVPGenerator
from src.Apps.FusionGait.ConfigUtils.User import UserData, User
from src.Apps.FusionGait.ConfigUtils.dfs_config import WiFiDFSConfig, AcousticDFSConfig
from src.Apps.FusionGait.ConfigUtils.expriment_config import ExperimentConfig
from src.Apps.FusionGait.ConfigUtils.track_config import TrackConfig, GroundTruthTrajectory


# Suppress MATLAB Runtime output logs
class SuppressMCR:
    def __enter__(self):
        # Save original stdout/stderr file descriptors
        self.stdout_fno = os.dup(1)
        self.stderr_fno = os.dup(2)
        # Redirect to devnull
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, 1)
        os.dup2(self.devnull, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        os.dup2(self.stdout_fno, 1)
        os.dup2(self.stderr_fno, 2)
        os.close(self.devnull)
        os.close(self.stdout_fno)
        os.close(self.stderr_fno)


def load_all_users(data_root: str, user_range: range = range(1, 12)) -> List[User]:
    """
    Instantiate User objects in a loop
    Args:
        data_root (str): dataset root directory (contains userX folders)
        user_range (range): range of user IDs (default 1~11)
    Returns:
        List[User]: all user objects
    """
    users = []
    for uid in user_range:
        user_dir = Path(data_root) / f"user{uid}"
        if not user_dir.exists():
            print(f"[Warning] User directory {user_dir} not found, skipped.")
            continue
        user = User(uid, data_root, UserData(uid))
        users.append(user)
    return users

def base_env(modality: str, expCfg: ExperimentConfig):
    cfg = expCfg.get_current_scenario_config(modality)

    # Body information is filled by the trajectory algorithm, so leave it empty first
    torso_pos = np.array([])
    torso_ori = np.array([])

    envPhy = PhysicalSetup(
        torso_pos=torso_pos,
        torso_ori=torso_ori,
        Tx_pos=cfg.Tx_pos,
        Rx_pos=cfg.Rx_pos,
        rx_cnt=cfg.rx_cnt
    )

    return envPhy, cfg

def gbvp_env(modality: str, expCfg: ExperimentConfig):
    envPhy, cfg = base_env(modality, expCfg)

    envVecMap = VelocityMappingParams(
        wave_length=cfg.wave_length
    )

    return envPhy, envVecMap

def ppvp_env(modality: str, expCfg: ExperimentConfig):
    envPhy, cfg = base_env(modality, expCfg)
    envVecMap = PolarVelocityMappingParams(wave_length=cfg.wave_length)

    return envPhy, envVecMap


def parse_track_rep(path: str):
    """
    Extract trajectory ID and repetition ID from the path
    Example: '.../user1/user1-1-1' -> (1, 1)
    """
    p = Path(path)
    # Take the last part: user1-1-1
    stem = p.name
    match = re.match(r"user\d+-(\d+)-(\d+)", stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid path format: {path}")
    track_id, rep_id = map(int, match.groups())
    return track_id, rep_id

def get_dfss_config(modality: str):
    if modality == "wifi":
        return WiFiDFSConfig()
    elif modality == "acoustic":
        return AcousticDFSConfig()
    else:
        return

def get_track_config(modality: str, expCfg:ExperimentConfig):
    if modality == "wifi":
        return TrackConfig( wave_length=expCfg.wifi.wave_length,tx_loc=expCfg.wifi.Tx_pos,rx_locs=expCfg.wifi.Rx_pos,
                            trajectories=expCfg.current_scene_config.trajectories)
    if modality == "acoustic":
        return TrackConfig( wave_length=expCfg.acoustic.wave_length,tx_loc=expCfg.acoustic.Tx_pos,rx_locs=expCfg.acoustic.Rx_pos,
                            trajectories=expCfg.current_scene_config.trajectories)
    return NotImplementedError


_generator_cache = {}
def get_spectrum_generator(method: str, modality, expCfg: ExperimentConfig):
    """
    Return a singleton generator; create it only once across users to avoid repeatedly loading MATLAB classes.

    method: str, "GBVP", "PPVP", "FusionGBVP", etc.
    modality: str or List[str] (for FusionGBVP, multiple modalities)
    """
    global _generator_cache

    # --- Build cache key ---
    if isinstance(modality, (list, tuple)):
        key = (method, tuple(modality))
    else:
        key = (method, modality)

    # --- Return cached if exists ---
    if key in _generator_cache:
        return _generator_cache[key]

    # --- Create instance ---
    if method == "GBVP":
        envPhy, envVecMap = gbvp_env(modality, expCfg)
        generator = GBVPGenerator(envPhy, envVecMap)

    elif method == "PPVP":
        envPhy, envVecMap = ppvp_env(modality, expCfg)
        generator = PPVPGenerator(envPhy, envVecMap)

    elif method == "FusionGBVP":
        if not isinstance(modality, (list, tuple)):
            raise ValueError("FusionGBVP requires a list of modalities")

        # Build dictionaries: modality -> (PhysicalSetup / VelocityMappingParams)
        phys_cfg_dict = {}
        vel_cfg_dict = {}
        for mod in modality:
            envPhy, envVecMap = gbvp_env(mod, expCfg)
            phys_cfg_dict[mod] = envPhy
            vel_cfg_dict[mod] = envVecMap

        generator = FusionGBVPGenerator(phys_cfg_dict, vel_cfg_dict)

    elif method == "FusionPPVP":
        if not isinstance(modality, (list, tuple)):
            raise ValueError("FusionGBVP requires a list of modalities")

        # Build dictionaries: modality -> (PhysicalSetup / VelocityMappingParams)
        phys_cfg_dict = {}
        vel_cfg_dict = {}
        for mod in modality:
            envPhy, envVecMap = ppvp_env(mod, expCfg)
            phys_cfg_dict[mod] = envPhy
            vel_cfg_dict[mod] = envVecMap

        generator = FusionPPVPGenerator(phys_cfg_dict, vel_cfg_dict)

    else:
        raise NotImplementedError(f"Method {method} not implemented")

    # --- Cache it ---
    _generator_cache[key] = generator
    return generator

## Per-process singleton cache for DFS generators
_process_dfss_cache = {}

def get_dfss_in_process(modality: str):
    global _process_dfss_cache
    if modality in _process_dfss_cache:
        return _process_dfss_cache[modality]

    dfs_cfg = get_dfss_config(modality)
    # Ensure the factory-created instance is cached
    instance = DFSSGeneratorFactory.create(modality, dfs_cfg)
    _process_dfss_cache[modality] = instance
    return instance

# =============================
# Per-process singleton cache for trackers
# =============================
_process_tracker_cache = {}


def get_tracker_in_process(modality: str, expCfg: ExperimentConfig):
    global _process_tracker_cache
    if modality in _process_tracker_cache:
        return _process_tracker_cache[modality]

    track_config = get_track_config(modality, expCfg)
    traj_truth = GroundTruthTrajectory(expCfg.data_root_path)
    tracker = PositionTracker(track_config, traj_truth)

    _process_tracker_cache[modality] = tracker
    return tracker

# =============================
# Per-process singleton cache for Users
# =============================
_process_user_cache = {}
def get_user_in_process(user_id: int, data_root: str):
    """
    Per-process singleton User (not shared across processes)
    """
    global _process_user_cache

    if user_id in _process_user_cache:
        return _process_user_cache[user_id]

    user = User(user_id, data_root, UserData(user_id))
    _process_user_cache[user_id] = user
    return user

# ========================================
# Core function executed in child process (one task = one user + modality)
# ========================================
def process_user_modality(task, expCfg):
    """
    task = (scenario_name, user_id, modality)
    executed inside a child process
    """
    scenario_name, user_id, modality = task

    print(f"[Start] Processing User {user_id}, Modality: {modality}, Scenario: {scenario_name}")

    # -------- Per-process singleton user instance built from user_id --------
    data_root = os.path.join(expCfg.data_root_path, scenario_name)
    user = get_user_in_process(user_id, data_root)

    # -------- Per-process singleton tracker --------
    tracker = get_tracker_in_process(modality, expCfg)

    exe_base_dir = os.path.normpath(os.path.join(
        expCfg.exe_path, scenario_name, f"user{user_id}", modality
    ))

    # Load cache file first
    is_load_success = user.data.load_base_data(exe_base_dir, modality)
    print(f"[Info] Cache load {'successful' if is_load_success else 'failed'} for User {user_id}, Modality: {modality}")

    #if not is_load_success:
    paths_prefix = user.get_path_groups(modality)
    paths_prefix.sort()
    for path in paths_prefix:
        track_id, rep_id = parse_track_rep(path)
        print(f"[Processing] Path: {path}, Track ID: {track_id}, Rep ID: {rep_id}")

        if is_load_success:
            doppler_spectrum, freq_bin, locations, speeds, duration = \
                user.data.get_base_data(modality).get_data_by_id(track_id, rep_id)
            # Bug fix: some RX data can be NaN, causing loaded doppler_spectrum to have wrong dimensions
            if doppler_spectrum is None or doppler_spectrum.shape[0] != tracker.trackConfig.rx_locs.shape[0]:
                doppler_spectrum, freq_bin, locations, speeds, duration = \
                    tracker._successive_track(get_dfss_in_process(modality), path, track_id, rep_id, user, modality, exe_base_dir)
        else:
            with SuppressMCR():
                doppler_spectrum, freq_bin, locations, speeds, duration = \
                    tracker._successive_track(get_dfss_in_process(modality),path, track_id, rep_id, user, modality, exe_base_dir)

        base_data = user.data.get_base_data(modality)
        base_data.set_all_data(
            track_id, rep_id,
            doppler_spectrum, freq_bin,
            locations, speeds, duration,
            exe_base_dir
        )

        methods = user.data.get_base_methods()
        print(methods)
        for method in methods:
            if method == 'GBVP' and modality == 'acoustic':
                 continue
            spectrum_dir = os.path.normpath(os.path.join(expCfg.exe_path, scenario_name, f"user{user.user_id}", modality, method))
            generator = get_spectrum_generator(method, modality, expCfg)
            # The tracker updated the RX configuration; apply it to the GBVP generator as well
            generator.phys_cfg.Rx_pos=tracker.trackConfig.rx_locs
            generator.phys_cfg.rx_cnt=generator.phys_cfg.Rx_pos.shape[0]

            # Load cached spectrum files first
            is_load_velocitySpec_success = user.data.load_velocity_spectrum(spectrum_dir, modality, method)
            if is_load_velocitySpec_success and user.data.check_if_velocity_spectrum_exists(modality, method, track_id, rep_id):
                print(f"[Skip] Velocity spectrum already exists for User {user.user_id}, Modality: {modality}, Method: {method}, Track ID: {track_id}, Rep ID: {rep_id}")
                continue
            else:
                with SuppressMCR():
                    velocity_spectrum_ro = generator.run(locations, doppler_spectrum, freq_bin, duration)
                print(f"[User {user.user_id} | {modality}-{method} | Track {track_id} | Rep {rep_id}] {method} extract done!")
                user.data.get_velocity_spectrum(modality, method).set_processed_spectrum(track_id, rep_id, velocity_spectrum_ro, base_dir=spectrum_dir)
                del velocity_spectrum_ro
                # Save the spectrum path record: saved once per instance here; won't break logic but keeps overwriting, leaving only the last one
                user.save_velocity_spectrum(spectrum_dir, modality, method)
                print(f"[Info] Data processed and set for Track ID: {track_id}, Rep ID: {rep_id}")

        user.save_base_data(exe_base_dir, modality)

    return f"[Done] User {user_id} / {modality}"
