# model_config.py

from dataclasses import dataclass
from typing import Optional, List

from src.Apps.FusionGait.GaitRecognition.IdentityModel.Utils.constants import ModelNames


@dataclass
class ModelConfig:
    """Basic parameters shared by all models."""
    name: str = "base"
    version: str = "v1"
    input_channels: int = 1
    num_classes: int = 19
    num_modalities: int = 1
    modalities_channels: Optional[List[int]] = None
    freq_dim: int = 128
    embedding_dim: int = 256
    dropout: float = 0.3

    def __post_init__(self):
        """Runs after initialization to handle dependent logic."""
        # If no explicit channel list is provided, auto-fill it based on input_channels
        if self.modalities_channels is None:
            self.modalities_channels = [self.input_channels] * self.num_modalities

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    @property
    def model_id(self):
        return f"{self.name}_{self.version}"

# --- Define model-specific configuration classes ---

@dataclass
class CNNConfig(ModelConfig):
    name: str = ModelNames.CNN
    dropout: float = 0.3#0.5
    kernel_size: int = 3

@dataclass
class FusionCNNConfig(CNNConfig):
    name: str = ModelNames.FUSION_CNN


@dataclass
class TransformerConfig(ModelConfig):
    name: str = ModelNames.TRANSFORMER
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.3

@dataclass
class FusionTransformerConfig(TransformerConfig):
    name: str = ModelNames.FUSION_TRANSFORMER

@dataclass
class LSTMConfig(ModelConfig):
    name: str = ModelNames.LSTM
    hidden_dim: int = 256
    num_layers: int = 2

@dataclass
class FusionLSTMConfig(LSTMConfig):
    name: str = ModelNames.LSTM
    hidden_dim: int = 256
    num_layers: int = 2

@dataclass
class ResNetTransformerConfig(ModelConfig):
    name: str = ModelNames.RESNETTRANSFORMER

@dataclass
class FusionResNetTransformerConfig(ResNetTransformerConfig):
    name: str = ModelNames.FUSION_RESNETTRANSFORMER


@dataclass
class CNNTransformerWithSelfAttentionConfig(ModelConfig):
    name: str = ModelNames.TRANSFORMERWITHSELFATTENTION

@dataclass
class FusionCNNTransformerWithCrossAttentionConfig(CNNTransformerWithSelfAttentionConfig):
    name: str = ModelNames.FUSION_TRANSFORMERWITHCROSSMODELATTENTION

# --- Config registry ---

_CONFIG_REGISTRY = {
    ModelNames.CNN: CNNConfig,
    ModelNames.FUSION_CNN: FusionCNNConfig,

    ModelNames.LSTM: LSTMConfig,
    ModelNames.FUSION_LSTM:FusionLSTMConfig,

    ModelNames.TRANSFORMER: TransformerConfig,
    ModelNames.FUSION_TRANSFORMER:FusionTransformerConfig,

    ModelNames.RESNETTRANSFORMER: ResNetTransformerConfig,
    ModelNames.FUSION_RESNETTRANSFORMER:FusionResNetTransformerConfig,

    ModelNames.TRANSFORMERWITHSELFATTENTION: CNNTransformerWithSelfAttentionConfig,
    ModelNames.FUSION_TRANSFORMERWITHCROSSMODELATTENTION:FusionCNNTransformerWithCrossAttentionConfig
}

def get_model_cfg(name: str) -> ModelConfig:
    """Return the corresponding config instance by name."""
    config_cls = _CONFIG_REGISTRY.get(name.lower(), ModelConfig)
    return config_cls()
