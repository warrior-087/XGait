# run_config.py
from dataclasses import dataclass
from typing import List, Union


@dataclass
class RunConfig:
    # ---------- task level ----------

    task: str = "gait_identity"          # task name
    trainer: str = "single"              # "single" | "fusion"
    phase: str = "train"                 # "train" | "eval" | "test"

    # ---------- model ----------
    model_name: str = "cnn"              # CNN / ResNet / ...
    model_cfg: object = None             # ModelConfig

    # ---------- data ----------
    dataset_name: str = "FusionGait"

    # ---------- modality ----------
    # Supports single-modality ("wifi") or multi-modality (["wifi", "acoustic"])
    modality: Union[str, List[str]] = "wifi"

    # Convenience helper: whether this run is in fusion (multi-modality) mode
    @property
    def is_fusion(self) -> bool:
        return isinstance(self.modality, list) and len(self.modality) > 1
