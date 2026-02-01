import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.data_path_cfg import base_paths   # <-- your project base_paths


# ============================================================
# Trajectory definitions
# ============================================================
@dataclass
class LineTrajectory:
    id: int
    type: str
    start: np.ndarray
    end: np.ndarray


@dataclass
class MultiLinesTrajectory:
    id: int
    type: str
    start: np.ndarray
    end: np.ndarray
    segments: List[Dict[str, np.ndarray]]


@dataclass
class ArcTrajectory:
    id: int
    type: str
    center: np.ndarray
    radius: float
    start_angle: float
    end_angle: float


def parse_trajectory(traj_json: Dict[str, Any]):
    """Parse trajectory JSON → Python object"""
    t = traj_json["type"]

    if t == "line":
        return LineTrajectory(
            id=traj_json["id"],
            type=t,
            start=np.array(traj_json["start"]),
            end=np.array(traj_json["end"])
        )

    elif t == "multilines":
        segments = [
            {"start": np.array(s["start"]), "end": np.array(s["end"])}
            for s in traj_json["segments"]
        ]
        return MultiLinesTrajectory(
            id=traj_json["id"],
            type= t,
            start=np.array(traj_json["start"]),
            end=np.array(traj_json["end"]),
            segments=segments
        )

    elif t == "arc":
        return ArcTrajectory(
            id=traj_json["id"],
            type=t,
            center=np.array(traj_json["center"]),
            radius=traj_json["radius"],
            start_angle=traj_json["start_angle"],
            end_angle=traj_json["end_angle"]
        )

    else:
        raise ValueError(f"Unknown trajectory type: {t}")


# ============================================================
# ScenarioConfig: loaded from JSON
# ============================================================
@dataclass
class ScenarioConfig:
    name: str
    users_list: list[int]
    json_path: str
    config_json: dict

    trajectories: dict = field(default_factory=dict)

    def wifi_data_dir(self):
        return os.path.join(os.path.dirname(self.json_path), "wifi")

    def acoustic_data_dir(self):
        return os.path.join(os.path.dirname(self.json_path), "acoustic")

    @property
    def trajectories_num(self):
        return len(self.trajectories)


# ============================================================
# WiFi & Acoustic global configs
# ============================================================
@dataclass
class WiFiConfig:
    tx_cnt: int = 1
    rx_cnt: int = 3
    rx_acnt: int = 3

    sample_rate: int = 1000
    wave_length: float = 299792458 / 5.32e9

    Tx_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    Rx_pos: np.ndarray = field(default_factory=lambda: np.array([
        [0.455, -0.455],
        [1.365, -0.455]
    ]))

    data_path: str = None


@dataclass
class AcousticConfig:
    tx_cnt: int = 1
    rx_cnt: int = 3

    sample_rate: int = 48000
    central_fre: int = 19000
    wave_length: float = 343 / 19000

    Tx_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    Rx_pos: np.ndarray = field(default_factory=lambda: np.array([
        [0.455, -0.455],
        [1.365, -0.455]
    ]))

    data_path: str = None


# ============================================================
# Main ExperimentConfig
# ============================================================
@dataclass
class ExperimentConfig:
    current_scene: str = "Laboratory"

    # WiFi and acoustic config for the current scene; position parameters are loaded from ScenarioConfig
    wifi: WiFiConfig = field(default_factory=WiFiConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)

    scenarios: Dict[str, ScenarioConfig] = field(default_factory=dict)

    data_root_path: str = os.path.normpath(os.path.join(base_paths['data_base'], "FusionGait"))  # dataset root directory
    exe_path: str = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../exeDir/")
    )

    fusion_config = {
        "time_align_mode": "correlation",  # or "reference"
        "reference_modality": "wifi",      # optional
         "rx_policy": {                    # selected Rx indices
             "wifi": {"indices": [0,1,2]},
             "acoustic": {"indices": [0,1,2]}
         },
        "enable_sanity_check": True
    }

    def __post_init__(self):
        # Load all scenes automatically
        self._load_all_scene_configs()

        # Bind current scene paths
        self._apply_scene(self.current_scene)

    # --------------------------------------------------------
    # Load all scene JSON files automatically
    # --------------------------------------------------------
    def _load_all_scene_configs(self):
        if not os.path.exists(self.data_root_path):
            raise FileNotFoundError(f"No scene folder found: {self.data_root_path}")

        for scene_name in os.listdir(self.data_root_path):
            folder = os.path.join(self.data_root_path, scene_name)
            if not os.path.isdir(folder):
                continue

            json_file = os.path.join(folder, "senario_cfg.json")
            if not os.path.exists(json_file):
                continue

            with open(json_file, "r") as f:
                cfg = json.load(f)

            # Create ScenarioConfig
            sc = ScenarioConfig(
                name=scene_name,
                users_list= cfg["users_list"],
                json_path=json_file,
                config_json=cfg
            )

            # Parse trajectories
            for t in cfg.get("trajectories", []):
                traj = parse_trajectory(t)
                sc.trajectories[traj.id] = traj

            self.scenarios[scene_name] = sc

    # --------------------------------------------------------
    # Apply scene → override WiFi/Acoustic
    # --------------------------------------------------------
    def _apply_scene(self, scene_name):
        if scene_name not in self.scenarios:
            raise ValueError(f"Scene '{scene_name}' does not exist")

        sc = self.scenarios[scene_name]

        # Override WiFi
        if "wifi" in sc.config_json:
            w = sc.config_json["wifi"]
            self.wifi.Tx_pos = np.array(w["Tx_pos"])
            self.wifi.Rx_pos = np.array(w["Rx_pos"])
        self.wifi.data_path =  sc.wifi_data_dir()

        # Override acoustic
        if "acoustic" in sc.config_json:
            a = sc.config_json["acoustic"]
            self.acoustic.Tx_pos = np.array(a["Tx_pos"])
            self.acoustic.Rx_pos = np.array(a["Rx_pos"])
        self.acoustic.data_path = sc.acoustic_data_dir()

    # --------------------------------------------------------
    # Switch scene at runtime
    # --------------------------------------------------------
    def set_scene(self, scene_name):
        self.current_scene = scene_name
        self._apply_scene(scene_name)

    @property
    def current_scene_config(self):
        return self.scenarios[self.current_scene]

    def get_all_scene_configs(self):
        return self.scenarios

    def get_current_scenario_config(self, modality: str):
        if modality == "wifi":
            return self.wifi
        elif modality == "acoustic":
            return self.acoustic

        return NotImplementedError


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    cfg = ExperimentConfig(current_scene="Loboratory")

    print("Current scene:", cfg.current_scene)
    print("Trajectories:", cfg.current_scene_config.trajectories_num)

    print("\nWiFi Tx:", cfg.wifi.Tx_pos)
    print("WiFi Rx:", cfg.wifi.Rx_pos)
    print("WiFi Data Dir:", cfg.wifi.data_path)

    print("\nAcoustic Tx:", cfg.acoustic.Tx_pos)
    print("Acoustic Rx:", cfg.acoustic.Rx_pos)
    print("Acoustic Data Dir:", cfg.acoustic.data_path)
