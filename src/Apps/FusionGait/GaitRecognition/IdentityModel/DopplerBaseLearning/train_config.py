# train_config.py
from dataclasses import dataclass
import torch

from src.Apps.FusionGait.GaitRecognition.IdentityModel.Utils.constants import ModelNames


@dataclass
class TrainConfig:
    # -------------------------
    # Device & basic training settings
    # -------------------------
    # device: str = "cpu"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    batch_size: int = 16
    drop_last: bool = True

    # -------------------------
    # Optimizer settings
    # -------------------------
    lr: float = 1e-3                     # initial learning rate
    weight_decay: float = 1e-4           # L2 regularization (weight decay)
    betas: tuple = (0.9, 0.999)          # Adam betas

    # -------------------------
    # Learning rate scheduler
    # -------------------------
    lr_scheduler: str = "cosine"         # options: none | step | cosine
    step_size: int = 10                  # step decay interval
    gamma: float = 0.01                  # learning rate decay factor

    # -------------------------
    # data loading
    # -------------------------
    num_workers: int = 10
    pin_memory: bool = True
    seed: int = 42                       # random seed

    # -------------------------
    # Saving & logging
    # -------------------------
    save_dir: str = "./checkpoints"
    log_interval: int = 10

    # -------------------------
    # Early stopping
    # -------------------------
    enable_early_stopping: bool = True
    patience: int = 30                   # stop if no improvement for 30 epochs on validation set

    # external override helper
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


# @dataclass
# class get_train_cfg():
#     return TrainConfig()


@dataclass
class CNNTrainConfig(TrainConfig):
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 5e-4


@dataclass
class FusionCNNTrainConfig(CNNTrainConfig):
    pass


@dataclass
class LSTMTrainConfig(TrainConfig):
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 5e-4


@dataclass
class FusionLSTMTrainConfig(LSTMTrainConfig):
    pass


# --- Aggressively tuned config for Transformer ---
@dataclass
class TransformerTrainConfig(TrainConfig):
    """Transformer-specific tuning parameters"""
    lr: float = 1e-3             # lower LR to reduce validation oscillation
    batch_size: int = 32         # smaller batch adds stochasticity; may help escape local minima
    weight_decay: float = 4e-5   # AdamW-style higher weight decay
    lr_scheduler: str = "warmup_cosine"
    warmup_epochs: int = 5       # warmup for the first 5 epochs
    patience: int = 20           # Transformer convergence differs; adjust patience accordingly


@dataclass
class FusionTransformerTrainConfig(TransformerTrainConfig):
    pass


@dataclass
class ResNetTransformerTrainConfig(TransformerTrainConfig):
    pass


@dataclass
class FusionResNetTransformerTrainConfig(ResNetTransformerTrainConfig):
    pass


@dataclass
class CNNTransformerWithSelfAttentionTrainConfig(TransformerTrainConfig):
    pass


@dataclass
class FusionCNNTransformerWithCrossModelAttentionTrainConfig(CNNTransformerWithSelfAttentionTrainConfig):
    pass


# --- Registry ---
_TRAIN_REGISTRY = {
    ModelNames.CNN: CNNTrainConfig,
    ModelNames.FUSION_CNN: FusionCNNTrainConfig,

    ModelNames.LSTM: LSTMTrainConfig,
    ModelNames.FUSION_LSTM: FusionLSTMTrainConfig,

    ModelNames.TRANSFORMER: TransformerTrainConfig,
    ModelNames.FUSION_TRANSFORMER: FusionTransformerTrainConfig,

    ModelNames.RESNETTRANSFORMER: ResNetTransformerTrainConfig,
    ModelNames.FUSION_RESNETTRANSFORMER: FusionResNetTransformerTrainConfig,

    ModelNames.TRANSFORMERWITHSELFATTENTION: CNNTransformerWithSelfAttentionTrainConfig,
    ModelNames.FUSION_TRANSFORMERWITHCROSSMODELATTENTION: FusionCNNTransformerWithCrossModelAttentionTrainConfig
}


def get_train_cfg(name: str) -> TrainConfig:
    """Get training config by model name: keep old models unchanged while allowing flexible configs for new models."""
    config_cls = _TRAIN_REGISTRY.get(name.lower(), TrainConfig)
    return config_cls()
