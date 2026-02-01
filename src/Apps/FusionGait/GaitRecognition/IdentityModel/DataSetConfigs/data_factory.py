import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.dataloader import DopplerDataset, \
    DopplerFusionDataset
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.dataset_config import collate_fn_variable_length, \
    collate_fn_fusionDoppler_variable_length
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.dataset_registry import get_dataset_class


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Ensure deterministic convolution behavior (may slightly reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataFactory:
    @classmethod
    def create_loaders(cls, run_cfg, ds_cfg, train_cfg):
        # 1) Normalize modality format: ensure we always handle it as a list
        modalities = run_cfg.modality
        if isinstance(modalities, str):
            modalities = [modalities]

        # Get the base Dataset class (e.g., DopplerDataset)
        base_ds_class = get_dataset_class(ds_cfg.dataset_name)

        # ---------- Branch A: Single Modality ----------
        if run_cfg.trainer == "single":
            # Keep original behavior: only use the first modality in the list
            m = modalities[0]
            full_ds = base_ds_class(ds_cfg.root, m, ds_cfg)

        # ---------- Branch B: Fusion ----------
        elif run_cfg.trainer == "fusion":
            # 1) Dynamically create multiple base dataset instances
            # Each base dataset handles its own modality-specific file reading and user filtering
            ds_dict = {
                m: base_ds_class(ds_cfg.root, m, ds_cfg)
                for m in modalities
            }

            # 2) Instantiate the fusion dataset wrapper
            fusion_ds_class = get_dataset_class(ds_cfg.fusion_dataset_name)
            # Pass the dataset dict in and perform pairing/matching
            full_ds = fusion_ds_class(ds_dict, pair_by="name")

        else:
            raise ValueError(f"Unknown trainer type: {run_cfg.trainer}")

        # Fetch random seed
        seed = getattr(train_cfg, 'seed', 42)
        set_seed(seed)  # Set global seed for reproducibility

        # Create a dedicated RNG for dataset splitting and (optionally) DataLoader randomness
        g = torch.Generator()
        g.manual_seed(seed)

        # 2) Split datasets
        # NOTE: FusionDataset implements split_by_traj as well, so this call is compatible
        if ds_cfg.split_strategy == "traj":
            train_ds, val_ds, test_ds = full_ds.split_by_traj(
                ds_cfg.train_trajs, ds_cfg.val_trajs, ds_cfg.test_trajs
            )
        else:
            r = ds_cfg.split_ratio
            lengths = [int(len(full_ds) * frac) for frac in r]
            lengths[-1] = len(full_ds) - sum(lengths[:-1])
            train_ds, val_ds, test_ds = random_split(full_ds, lengths, generator=g)

        # 3) Build DataLoaders
        loader_args = {
            "batch_size": train_cfg.batch_size,
            "num_workers": train_cfg.num_workers,
            "pin_memory": train_cfg.pin_memory,
            "drop_last": train_cfg.drop_last,
            # Persistent workers avoid re-spawning processes every epoch
            "persistent_workers": True if train_cfg.num_workers > 0 else False,
            "generator": g
        }

        import psutil
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024 ** 3
        print(f"[Memory Check] Main process preloading completed. Current RSS: {mem_gb:.2f} GB")

        collate_fn = full_ds.get_collate_fn()
        if collate_fn is not None:
            loader_args["collate_fn"] = collate_fn

        return {
            "train": DataLoader(train_ds, shuffle=True, **loader_args),
            "val": DataLoader(val_ds, shuffle=False, **loader_args),
            "test": DataLoader(test_ds, shuffle=False, **loader_args)
        }
