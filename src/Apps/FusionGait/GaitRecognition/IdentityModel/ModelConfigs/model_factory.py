import os
from typing import Dict, Type

from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.model_config import ModelConfig
from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.models import BaseModel, CNN, LateFusionCNN, \
    Conv3dLSTM, FusionConv3dLSTM, CNNTransformer, FusionCNNTransformer, ResNetTransformer, FusionResnetTransformer, \
    CNNTransformerWithSelfAttention, FusionCNNTransformerWithCrossModalAttention


class ModelFactory:
    """
    Used to centrally manage all models: registration, instance creation, model save paths, etc.
    """
    _registry: Dict[str, Type[BaseModel]] = {}
    _model_save_root: str = "./checkpoints"  # default path, configurable

    @classmethod
    def register(cls, name: str, model_cls: Type[BaseModel]):
        """
        Register a model. The model class must inherit from BaseModel.
        """
        if not issubclass(model_cls, BaseModel):
            raise TypeError(f"{model_cls} must inherit BaseModel")

        cls._registry[name] = model_cls

    @classmethod
    def create(cls, name: str, cfg: ModelConfig):
        """
        Create a model instance by name.
        """
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' is not registered. Available: {list(cls._registry.keys())}")

        model_cls = cls._registry[name]
        return model_cls(cfg)

    @classmethod
    def get_model_save_dir(cls, cfg: ModelConfig, scenario: str = ""):
        """
        Generate a save path based on the model config and scenario name, e.g.:
        ./checkpoints/cnn_v1/Laboratory/
        """
        path = os.path.join(cls._model_save_root, cfg.model_id, scenario)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def available_models(cls):
        return list(cls._registry.keys())

# Default registration
from .model_factory import ModelFactory
ModelFactory.register("cnn", CNN)
ModelFactory.register("fusion_cnn", LateFusionCNN)

ModelFactory.register("lstm", Conv3dLSTM)
ModelFactory.register("fusion_lstm", FusionConv3dLSTM)

ModelFactory.register("transformer", CNNTransformer)
ModelFactory.register("fusion_transformer", FusionCNNTransformer)

ModelFactory.register("resnetTransformer", ResNetTransformer)
ModelFactory.register("fusion_resnetTransformer", FusionResnetTransformer)

ModelFactory.register("transformerWithSelfAttention", CNNTransformerWithSelfAttention)
ModelFactory.register("fusion_transformerWithCrossModelAttention", FusionCNNTransformerWithCrossModalAttention)
