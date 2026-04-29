import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1) Import constants (key to eliminating magic strings)
from src.Apps.FusionGait.GaitRecognition.IdentityModel.Utils.constants import ModelNames, Modalities, Phases, TrainerTypes

# 2) Import configs and factories
from src.Apps.FusionGait.ConfigUtils.expriment_config import ExperimentConfig
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DopplerBaseLearning.run_config import RunConfig
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DopplerBaseLearning.train_config import TrainConfig, \
    get_train_cfg
from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.model_config import get_model_cfg

from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.data_factory import DataFactory
from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.model_factory import ModelFactory
from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.trainer_factory import TrainerFactory

from src.Apps.FusionGait.GaitRecognition.IdentityModel.Utils.logger import get_logger


def main():  # run_model_suite
    # ---------------------------------------------------------
    # Experiment control panel: all modifications should be made here
    # ---------------------------------------------------------
    TARGET_MODEL = ModelNames.FUSION_CNN  # switch model
    # TARGET_MODEL = target_model          # switch model
    # TARGET_MODALITY = Modalities.ACOUSTIC  # switch modality
    TARGET_MODALITY = [Modalities.WIFI, Modalities.ACOUSTIC]  # dataset with two modalities
    EXP_VERSION = "v6_matrix"  # experiment version tag
    CURRENT_PHASE = Phases.TRAIN  # phase: train or test

    # Training hyper-parameters (injected into TrainConfig)
    TRAIN_PARAMS = {
        "num_workers": 10,
        "epochs": 50,
        "patience": 20
    }

    # Base dataset params (used as a default template)
    BASE_DATASET_PARAMS = {
        "use_cache": True,
        "dataset_name": "doppler", # ppvp
        "fusion_dataset_name": "doppler_fusion", # ppvp_fusion
        "traj_sample_percentage": 100,
        "split_strategy": "traj"
    }

    # Scenario-specific overrides
    SCENARIO_OVERRIDES = {
        "Home": {
            "user_num_classes":19 ,
            "selected_users": [1,2,3,6,7,8,9,10,12,13,14,15,21,22,23,24,25,26,27],
            "selected_trajs": [1, 2, 3, 4, 5],
            "train_trajs": [1, 2, 3],
            "val_trajs": [4],
            "test_trajs": [5]
        },
        "Laboratory": {
            "user_num_classes": 20,
            "selected_users": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # user IDs
            "selected_trajs": list(range(1, 10)),  # trajectory IDs
            "train_trajs": [2, 3, 4, 5, 6, 7, 8],
            "val_trajs": [9],
            "test_trajs": [1]
        },
        "MeetingRoom":{
            "user_num_classes": 20,
            "selected_users": [1,2,3,4,5,6,7,9,10,12,13,15,17,21,22,23,24,25,26,27],
            "selected_trajs": [1, 2, 3, 4, 5],
            "train_trajs": [1, 2, 3],
            "val_trajs": [4],
            "test_trajs": [5]
        }
    }

    # Model hyper-parameters (override presets in model_config)
    MODEL_PARAMS = {
        "name": TARGET_MODEL,
        "version": EXP_VERSION,
        "embedding_dim": 256,
        "version": EXP_VERSION
    }

    # ---------------------------------------------------------

    # 0) Base environment setup and component registration
    torch.backends.cudnn.benchmark = True
    # You may call registration functions here globally, or rely on auto-registration inside modules

    train_cfg = get_train_cfg(TARGET_MODEL).update(**TRAIN_PARAMS)
    exp_cfg = ExperimentConfig()

    # 1) Get the preset model config (includes default embedding_dim etc., can be overridden)
    # model_cfg = get_model_cfg().update(**MODEL_PARAMS)
    model_cfg = get_model_cfg(TARGET_MODEL).update(**MODEL_PARAMS)

    # If Transformer needs special tweaks, you can modify the config object directly with IDE hints
    # if isinstance(model_cfg, TransformerConfig):
    #     model_cfg.nhead = 4  # only applied when TARGET_MODEL is transformer

    # 2) Pack runtime configuration
    run_cfg = RunConfig(
        trainer=TrainerTypes.FUSION,
        phase=CURRENT_PHASE,
        model_name=TARGET_MODEL,
        model_cfg=model_cfg,
        modality=TARGET_MODALITY
    )

    # 3) Iterate over scenarios (driven by ExperimentConfig)
    scenarios = exp_cfg.get_all_scene_configs()

    for scenario_name, _ in scenarios.items():
        if scenario_name == 'Home':
            continue

        # Switch environment to the current scenario
        exp_cfg.set_scene(scenario_name)
        root_dir = os.path.join(exp_cfg.exe_path, scenario_name)

        current_ds_params = BASE_DATASET_PARAMS.copy()
        if scenario_name in SCENARIO_OVERRIDES:
            current_ds_params.update(SCENARIO_OVERRIDES[scenario_name])

        run_cfg.model_cfg.num_classes = current_ds_params["user_num_classes"]

        # Run a single-scenario experiment
        try:
            # Internal logic is fully driven by run_cfg / train_cfg (no hard-coded parameters)
            run_single_experiment(run_cfg, train_cfg, scenario_name, root_dir, current_ds_params)
        except Exception as e:
            print(f"FAILED: Scenario [{scenario_name}] due to: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


def run_single_experiment(run_cfg, train_cfg, scenario_name, root_dir, data_params):
    """
    No hard-coded values here; everything is driven by run_cfg and train_cfg.
    """
    # 1) Dynamically get the save directory (isolated by model_cfg.model_id)
    save_dir = ModelFactory.get_model_save_dir(run_cfg.model_cfg, scenario_name)
    train_cfg.save_dir = save_dir

    # 2) Initialize logger for this scenario run
    logger = get_logger(scenario_name, os.path.join(save_dir, f"{scenario_name}.log"))

    # 3) Load dataset
    from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.dataset_config import get_dataset_cfg
    ds_cfg = get_dataset_cfg(root_dir).update(**data_params)
    loaders = DataFactory.create_loaders(run_cfg, ds_cfg, train_cfg)

    # Print a log to make it clear that the main process is initializing the dataset
    print(f"--- [Scenario: {scenario_name}] Initializing dataset ---")

    # 4) Infer channels dynamically and build the model accordingly
    for batch in loaders['train']:
        sample_x, sample_y = batch[:2]
        break
    # sample_x, sample_y = next(iter(loaders['train']))[:2]

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

    # 5) Create trainer
    trainer = TrainerFactory.create(
        name=f"{run_cfg.trainer}_input",
        model=model,
        device=train_cfg.device,
        train_cfg=train_cfg,
        logger=logger
    )

    # 6) Execute workflow
    best_path = os.path.join(save_dir, f"best_{scenario_name}.pth")

    if run_cfg.phase == Phases.TRAIN:
        trainer.fit(loaders['train'], loaders['val'], save_path=best_path)

        # Force reload the best checkpoint into memory
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint)

        # IMPORTANT: ensure the trainer uses the updated model instance
        trainer.model = model

        test_set_size = len(loaders['test'].dataset)
        print(f"Test dataset size: {test_set_size}")

        if test_set_size > 0:
            # 1) Get predictions
            y_true, y_pred = trainer.predict(loaders['test'])

            # 2) Pack results and save
            eval_results = {
                "y_true": y_true,
                "y_pred": y_pred,
                "scenario": scenario_name,
                "model_name": run_cfg.model_name
            }

            # Save as a .pt file (contains numpy arrays)
            res_save_path = os.path.join(save_dir, f"eval_res_{scenario_name}.pt")
            torch.save(eval_results, res_save_path)

            print(f"✅ Evaluation results saved to: {res_save_path}")

            # (Original logging logic)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_true, y_pred)
            logger.info(f"Final Test Acc: {acc:.4f}")

            # Alternatively, use evaluate method (if trainer has one), or verify eval_one_epoch is correct
            # test_loss, acc = trainer.eval_one_epoch(loaders['test'])
            # logger.info(f"Final Test Acc: {acc:.4f}")
        else:
            logger.error("Test dataset is EMPTY! Check your data split logic.")

    elif run_cfg.phase == Phases.TEST:
        model.load_state_dict(torch.load(best_path))
        _, acc = trainer.eval_one_epoch(loaders['test'])
        logger.info(f"Pure Test Acc: {acc:.4f}")


if __name__ == "__main__":
    main()
