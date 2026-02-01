# trainer_factory.py
from typing import Dict, Type

from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.trainers import BaseTrainer, SingleInputTrainer, \
    FusionInputTrainer


class TrainerFactory:
    """
    TrainerFactory
    ---------------
    Trainer ≠ business logic
    Trainer = input adaptation + training pipeline
    """

    _registry: Dict[str, Type[BaseTrainer]] = {}

    @classmethod
    def register(cls, name: str, trainer_cls: Type[BaseTrainer]):
        cls._registry[name] = trainer_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTrainer:
        if name not in cls._registry:
            raise ValueError(
                f"Trainer '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls):
        return list(cls._registry.keys())


# Default registration
TrainerFactory.register("single_input", SingleInputTrainer)
TrainerFactory.register("fusion_input", FusionInputTrainer)
