import os
import sys
import torch

# 1) Environment setup (MUST run before importing project modules)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project imports
from src.Apps.FusionGait.GaitRecognition.IdentityModel.Utils.constants import (
    ModelNames, Modalities, Phases, TrainerTypes
)
from src.Apps.FusionGait.ConfigUtils.expriment_config import ExperimentConfig
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DopplerBaseLearning.run_config import RunConfig
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DopplerBaseLearning.train_config import get_train_cfg
from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.model_config import get_model_cfg
from src.Apps.FusionGait.GaitRecognition.IdentityModel.Utils.logger import get_logger

from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.model_factory import ModelFactory
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.data_factory import DataFactory
from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.trainer_factory import TrainerFactory
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.dataset_config import (
    get_dataset_cfg, get_finetune_dataset_cfg
)


def run_training_flow(run_cfg, train_cfg, scenario_name, root_dir, data_params):
    """Train on the source scenario and return the trained model/trainer."""
    # Determine the checkpoint/log directory dynamically
    save_dir = ModelFactory.get_model_save_dir(run_cfg.model_cfg, scenario_name)
    train_cfg.save_dir = save_dir

    # Put the training log next to checkpoints for easier debugging
    logger = get_logger(scenario_name, os.path.join(save_dir, f"train_{scenario_name}.log"))
    logger.info(f"--- [Scenario: {scenario_name}] Training started ---")

    # Load source-scenario dataset
    ds_cfg = get_dataset_cfg(root_dir).update(**data_params)
    loaders = DataFactory.create_loaders(run_cfg, ds_cfg, train_cfg)

    # Dynamically adapt model input dimensions from one batch
    sample_batch = next(iter(loaders["train"]))
    sample_x = sample_batch[0]

    if isinstance(sample_x, list):
        # --- Fusion mode ---
        # Infer channels per modality (e.g., WiFi=2, Acoustic=1)
        m_channels = [m.shape[1] for m in sample_x]  # e.g., [2, 1]

        run_cfg.model_cfg.modalities_channels = m_channels
        run_cfg.model_cfg.num_modalities = len(sample_x)

        # Keep input_channels as a legacy/primary reference; internal models should
        # prefer modalities_channels when available
        run_cfg.model_cfg.input_channels = m_channels[0]
    else:
        # --- Single-modality mode ---
        c = sample_x.shape[1]
        run_cfg.model_cfg.input_channels = c
        run_cfg.model_cfg.modalities_channels = [c]
        run_cfg.model_cfg.num_modalities = 1

    model = ModelFactory.create(run_cfg.model_name, run_cfg.model_cfg)
    trainer = TrainerFactory.create(
        name=f"{run_cfg.trainer}_input",
        model=model,
        device=train_cfg.device,
        train_cfg=train_cfg,
        logger=logger,
    )

    best_path = os.path.join(save_dir, f"best_{scenario_name}.pth")
    trainer.fit(loaders["train"], loaders["val"], save_path=best_path)

    # Load best checkpoint
    model.load_state_dict(torch.load(best_path))
    trainer.model = model
    return trainer, save_dir


def run_finetune_and_test_flow(
    trainer, run_cfg, train_cfg, test_scenario_name, target_root_dir, data_params, source_save_dir
):
    """
    Load target-scenario data, run a few epochs of fine-tuning, then evaluate on test set.
    """
    cross_log_path = os.path.join(source_save_dir, f"finetune_on_{test_scenario_name}.log")
    ft_logger = get_logger(f"FT_{test_scenario_name}", cross_log_path)
    ft_logger.info(f"--- Cross-scene fine-tuning started: {test_scenario_name} ---")

    # 1) Load target-scenario dataset (needs train/val splits for fine-tuning)
    ds_cfg = get_finetune_dataset_cfg(target_root_dir).update(**data_params)
    # NOTE: if train/val are empty in data_params, fine-tuning may not be meaningful
    loaders = DataFactory.create_loaders(run_cfg, ds_cfg, train_cfg)

    # 2) Configure fine-tuning strategy
    # Strategy A: reduce learning rate (commonly 1/10 of the original)
    original_lr = train_cfg.lr
    train_cfg.lr = original_lr * 0.1
    # Strategy B: reduce number of epochs for fine-tuning
    train_cfg.epochs = 100

    # Temporarily switch trainer logger to fine-tuning logger
    old_logger = trainer.logger
    trainer.logger = ft_logger

    ft_logger.info("Fine-tuning...")
    # Save fine-tuned best checkpoint (separate from the source best)
    ft_best_path = os.path.join(source_save_dir, f"best_ft_{test_scenario_name}.pth")
    trainer.fit(loaders["train"], loaders["val"], save_path=ft_best_path)

    # 3) Evaluate fine-tuned model
    ft_logger.info("Fine-tuning finished. Running final evaluation...")
    trainer.model.load_state_dict(torch.load(ft_best_path))
    test_loss, acc = trainer.eval_one_epoch(loaders["test"])

    ft_logger.info(f"Finetuned-Scene Acc: {acc:.4f}")

    # Restore configs to avoid affecting later runs
    trainer.logger = old_logger
    train_cfg.lr = original_lr
    return acc


def run_cross_testing_flow(
    trainer, run_cfg, train_cfg, test_scenario_name, target_root_dir, data_params, source_save_dir
):
    """Evaluate on target scenario and write logs into the source training directory."""
    # Append a test log under the source directory for side-by-side comparison
    cross_log_path = os.path.join(source_save_dir, f"cross_test_on_{test_scenario_name}.log")
    test_logger = get_logger(f"Test_{test_scenario_name}", cross_log_path)

    test_logger.info(f"--- Cross-scene testing started: {test_scenario_name} ---")

    ds_cfg = get_dataset_cfg(target_root_dir).update(**data_params)
    loaders = DataFactory.create_loaders(run_cfg, ds_cfg, train_cfg)

    test_set_size = len(loaders["test"].dataset)
    if test_set_size > 0:
        # Temporarily switch trainer logger to the test logger
        old_logger = trainer.logger
        trainer.logger = test_logger

        test_loss, acc = trainer.eval_one_epoch(loaders["test"])

        test_logger.info(f"Cross-Scene Acc: {acc:.4f} (Source: {run_cfg.model_cfg.version})")
        trainer.logger = old_logger
        return acc
    else:
        test_logger.error("Test set is empty!")
        return None


def main():
    # ---------------------------------------------------------
    # Experiment configuration
    # ---------------------------------------------------------
    SOURCE_SCENARIO = "Laboratory"
    TARGET_SCENARIOS = ["Home"]

    TARGET_MODEL = ModelNames.CNN
    TARGET_MODALITY = Modalities.WIFI
    # TARGET_MODALITY = [Modalities.WIFI, Modalities.ACOUSTIC]
    EXP_VERSION = "cross_scene_v1"

    TRAIN_PARAMS = {"num_workers": 10, "epochs": 50, "patience": 20}

    BASE_DATASET_PARAMS = {
        "use_cache": True,
        "dataset_name": "doppler",
        "fusion_dataset_name": "doppler_fusion",
        "traj_sample_percentage": 100,
        "split_strategy": "random",  # NOTE: this is random split, not traj-based split
    }

    SCENARIO_OVERRIDES = {
        "Home": {
            "user_num_classes": 4,
            "selected_users": [1, 3, 12, 14],  # Must match the source scenario's user set for fair evaluation
            "train_trajs": [],
            "val_trajs": [],
            "test_trajs": [1, 3, 4, 5],
        },
        "Laboratory": {
            "user_num_classes": 4,
            "selected_users": [1, 3, 12, 14],
            "train_trajs": [2, 3, 4, 5, 6, 7, 8],
            "val_trajs": [9],
            "test_trajs": [1],
        },
    }

    # ---------------------------------------------------------
    # 1) Initialization
    # ---------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    exp_cfg = ExperimentConfig()
    train_cfg = get_train_cfg(TARGET_MODEL).update(**TRAIN_PARAMS)
    model_cfg = get_model_cfg(TARGET_MODEL).update(name=TARGET_MODEL, version=EXP_VERSION)

    run_cfg = RunConfig(
        trainer=TrainerTypes.SINGLE,
        phase=Phases.TRAIN,
        model_name=TARGET_MODEL,
        model_cfg=model_cfg,
        modality=TARGET_MODALITY,
    )

    # ---------------------------------------------------------
    # 2) Train on the source scenario
    # ---------------------------------------------------------
    exp_cfg.set_scene(SOURCE_SCENARIO)
    src_dir = os.path.join(exp_cfg.exe_path, SOURCE_SCENARIO)
    src_params = BASE_DATASET_PARAMS.copy()
    src_params.update(SCENARIO_OVERRIDES.get(SOURCE_SCENARIO, {}))

    run_cfg.model_cfg.num_classes = src_params["user_num_classes"]

    trainer, source_save_dir = run_training_flow(run_cfg, train_cfg, SOURCE_SCENARIO, src_dir, src_params)

    # ---------------------------------------------------------
    # 3) Cross-scene evaluation (with fine-tuning)
    # ---------------------------------------------------------
    for target_scene in TARGET_SCENARIOS:
        exp_cfg.set_scene(target_scene)
        target_dir = os.path.join(exp_cfg.exe_path, target_scene)
        target_params = BASE_DATASET_PARAMS.copy()
        target_params.update(SCENARIO_OVERRIDES.get(target_scene, {}))

        run_finetune_and_test_flow(
            trainer, run_cfg, train_cfg, target_scene, target_dir, target_params, source_save_dir
        )


if __name__ == "__main__":
    main()
