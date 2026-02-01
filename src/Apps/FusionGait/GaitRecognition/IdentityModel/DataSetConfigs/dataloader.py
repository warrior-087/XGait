import os
import glob
import random
import re
from abc import abstractmethod, ABC
from typing import Dict, Optional, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.dataset_config import \
    DatasetConfig, collate_fn_variable_length, collate_fn_fusionDoppler_variable_length, \
    collate_fn_fusionPPVP_variable_length
from src.Apps.FusionGait.GaitRecognition.IdentityModel.DataSetConfigs.dataset_registry import register_dataset
from src.Apps.FusionGait.GaitRecognition.IdentityModel.Utils.preprocess import preprocess_doppler, preprocess


def _global_cached_loader(file_path, preprocess_func, target_shape):
    try:
        dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        cache_dir = os.path.join(dir_name, "__cache__")
        cache_path = os.path.join(cache_dir, file_name.replace(".npy", ".bin"))

        # 1. Try reading from the binary cache first
        if os.path.exists(cache_path):
            data = np.fromfile(cache_path, dtype=np.float32)

            # --- Dynamic shape recovery logic ---
            if target_shape is not None:
                total_size = data.size
                # Assume target_shape is (C, T, F)
                # We keep C and F fixed and infer T
                fixed_part = target_shape[0] * target_shape[2]

                if total_size == np.prod(target_shape):
                    return data.reshape(target_shape)
                elif total_size % fixed_part == 0:
                    # Dynamically infer the third dimension (W)
                    dynamic_w = total_size // fixed_part
                    new_shape = (target_shape[0], dynamic_w, target_shape[2])
                    return data.reshape(new_shape)
                elif total_size % ((target_shape[0]-1) * target_shape[2]) == 0:  # Fault-tolerance: try C-1 (special case)
                    dynamic_w = total_size // ((target_shape[0]-1) * target_shape[2])
                    new_shape = (target_shape[0]-1, dynamic_w, target_shape[2])
                    return data.reshape(new_shape)
                else:
                    # If it cannot be divided evenly, the cache is unusable; delete it
                    print(f"Shape mismatch and non-divisible; cleaning cache: {file_name}")
                    os.remove(cache_path)
            else:
                return data

        # 2. Cache miss or corrupted cache: load from the original file
        raw_data = np.load(file_path, mmap_mode='r')
        processed_data = preprocess_func(raw_data).astype(np.float32)

        # 3. Write a new cache file
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Force C-contiguous layout to ensure tofile order matches reshape recovery order exactly
        if not processed_data.flags['C_CONTIGUOUS']:
            processed_data = np.ascontiguousarray(processed_data)

        processed_data.tofile(cache_path)
        return processed_data

    except Exception as e:
        print(f"Load failed {file_path}: {e}")
        return None


class BaseDataset(Dataset, ABC):
    def __init__(self, root, ds_cfg:DatasetConfig = None, files=None, labels=None, cache = None):
        """
        Args:
            Args:
            root: Root directory of the dataset
            ds_cfg: Dataset configuration
            files: List of file paths
            labels: List of labels
            cache: Preprocessed data cache list (for subset splitting)
        """
        self.root = root
        self.ds_cfg = ds_cfg or self.get_default_dataset_cfg(root)
        self.files = []
        self.labels = []
        self.users = []

        # Data cache list
        self.data_cache = cache

        # If files are passed into the constructor, this is a split subset
        if files is not None:
            self.files = files
            self.labels = labels
        else:
            # Otherwise, run the original full loading and filtering logic
            self._load_and_filter_all()

        # If use_cache is enabled, preload data into memory
        if getattr(self.ds_cfg, 'use_cache', False):
            if self.data_cache is not None:
                print("Existing cached data detected (from subset split); skipping duplicate preloading.")
            else:
                self._preload_to_memory()

            # Check current process memory usage
            import psutil
            process = psutil.Process(os.getpid())
            print(f"Current memory usage: {process.memory_info().rss / 1024 ** 3:.2f} GB")

    def _preload_to_memory(self):
        print(f"Starting parallel preloading (dynamic shape recovery enabled)...")

        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        import numpy as np
        from functools import partial

        # For HDD, 4-8 is recommended; for SSD, 16 is recommended
        n_workers = min(multiprocessing.cpu_count(), 8)

        # 1. Load one sample to get a reference shape
        try:
            sample_raw = np.load(self.files[0], mmap_mode='r')
            sample_processed = self.preprocess(sample_raw)
            target_shape = sample_processed.shape
            print(f"Reference shape set to: {target_shape}")
        except Exception as e:
            print(f"Failed to get initial shape: {e}")
            return

        paths = self.files
        prep_func = self.preprocess

        try:
            # 2. Use ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Use partial to bind preprocess_func and target_shape
                worker_func = partial(_global_cached_loader,
                                      preprocess_func=prep_func,
                                      target_shape=target_shape)

                # Note: if self.data_cache already has data, clear it first
                self.data_cache = list(tqdm(
                    executor.map(worker_func, paths, chunksize=40),
                    total=len(paths),
                    desc="Hybrid Loading"
                ))

                # Filter out possible None values (failed samples)
                # self.data_cache = [d for d in self.data_cache if d is not None]

        except Exception as e:
            print(f"Parallel preloading failed: {e}. Switching to single-thread...")
            self.data_cache = []
            for f in tqdm(paths):
                self.data_cache.append(_global_cached_loader(f, prep_func, target_shape))

        # Summarize loading status
        valid_count = sum(1 for d in self.data_cache if d is not None)
        print(f"Dataset loading completed! Success: {valid_count}/{len(paths)}")


    def _load_and_filter_all(self):
        # 1. Get all user directories
        all_user_dirs = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])

        # 2. Build a mapping from directory name -> real ID (assuming names like 'user12', 'user05', etc.)
        # Extract numeric ID, e.g., 'user12' -> 12
        user_to_id = {}
        for d in all_user_dirs:
            try:
                # Extract numeric ID based on your naming rule, e.g., from 'user1' extract '1'
                numeric_id = int(''.join(filter(str.isdigit, d)))
                user_to_id[d] = numeric_id
            except ValueError:
                continue  # Skip folders that do not match the naming rule

        # 3. Determine the final list of selected directories
        selected_dirs = []

        # Case A: A specific list of IDs is provided
        if hasattr(self.ds_cfg, "selected_users") and len(self.ds_cfg.selected_users) > 0:
            target_ids = set(map(int, self.ds_cfg.selected_users))
            selected_dirs = [d for d, uid in user_to_id.items() if uid in target_ids]

        # Case B: Randomly sample a fixed number of users
        elif self.ds_cfg.user_num_classes is not None:
            num = min(self.ds_cfg.user_num_classes, len(all_user_dirs))
            selected_dirs = random.sample(all_user_dirs, num)

        else:
            selected_dirs = all_user_dirs

        # 4. Sort to make results reproducible
        selected_dirs.sort()

        # 5. Load data and generate labels
        # Note: Labels are recommended to be mapped based on ID or based on the selected order.
        # For classification, labels usually need to be mapped to 0 ~ N-1
        for label_idx, u_dir in enumerate(selected_dirs):
            dataset_dir = self.get_dataset_dir(u_dir)
            if not os.path.exists(dataset_dir):
                continue

            files = sorted(glob.glob(os.path.join(dataset_dir, '*.npy')))
            files = self.filter_files(files)

            for f in files:
                if not self.check_file(f): continue
                self.files.append(f)
                self.labels.append(label_idx)  # Continuous labels for classification
            self.users.append(u_dir)

    @abstractmethod
    def create_subset(self, files, labels, cache=None):
        """
        Create a subset with the same configuration but different data.
        Must accept the cache parameter to support in-memory acceleration.
        """
        raise NotImplementedError

    def split_by_traj(self, train_ids, val_ids, test_ids):
        # Record indices rather than just file names so that cache can be aligned as well
        train_indices, val_indices, test_indices = [], [], []

        for idx, (f, l) in enumerate(zip(self.files, self.labels)):
            t_id = self.extract_track_from_filename(f)
            if t_id in train_ids:
                train_indices.append(idx)
            elif t_id in val_ids:
                val_indices.append(idx)
            elif t_id in test_ids:
                test_indices.append(idx)

        # Helper: extract data by indices
        def _extract_by_indices(indices):
            f = [self.files[i] for i in indices]
            l = [self.labels[i] for i in indices]
            # If the main dataset has cache, slice the corresponding cache for the subset
            c = [self.data_cache[i] for i in indices] if self.data_cache else None
            return f, l, c

        tr_f, tr_l, tr_c = _extract_by_indices(train_indices)
        val_f, val_l, val_c = _extract_by_indices(val_indices)
        te_f, te_l, te_c = _extract_by_indices(test_indices)

        # Need to modify create_subset signature so it accepts cache
        return (
            self.create_subset(tr_f, tr_l, cache=tr_c),
            self.create_subset(val_f, val_l, cache=val_c),
            self.create_subset(te_f, te_l, cache=te_c)
        )
        # """Split into three Dataset objects by trajectory ID"""
        # train_f, train_l = [], []
        # val_f, val_l = [], []
        # test_f, test_l = [], []
        #
        # for f, l in zip(self.files, self.labels):
        #     t_id = self.extract_track_from_filename(f)
        #     if t_id in train_ids:
        #         train_f.append(f)
        #         train_l.append(l)
        #     elif t_id in val_ids:
        #         val_f.append(f)
        #         val_l.append(l)
        #     elif t_id in test_ids:
        #         test_f.append(f)
        #         test_l.append(l)
        #
        # # Return sub-datasets of the same type
        # return (
        #     self.create_subset(train_f, train_l),
        #     self.create_subset(val_f, val_l),
        #     self.create_subset(test_f, test_l)
        # )

    @staticmethod
    @abstractmethod
    def get_collate_fn(self):
        """Implemented by subclasses; different datasets require different processing"""
        pass

    @abstractmethod
    def get_dataset_dir(self, user):
        """Implemented by subclasses; return the directory for the given dataset"""
        pass

    def get_default_dataset_cfg(self, root):
        """Get the default dataset configuration; override if needed"""
        return {
            'max_files_per_user': 100,
            'freq_down': 4,
            'time_down': 4,
            'merge_ncnt_as_channel': False
        }

    def filter_files(self, files):
        """Handle file count limits and other filtering operations here"""
        if self.ds_cfg.max_files_per_user is not None:
            files = files[:self.ds_cfg.max_files_per_user]
        elif self.ds_cfg.selected_trajs is not None:
            # Group files by track
            trajs_dict = {}
            for f in files:
                track = self.extract_track_from_filename(f)
                if track is None or track not in self.ds_cfg.selected_trajs:
                    continue

                if track not in trajs_dict:
                    trajs_dict[track] = []
                trajs_dict[track].append(f)

            # Select files per trajectory by percentage
            filtered_files = []
            for track, track_files in trajs_dict.items():
                num_files = len(track_files)
                num_to_select = int(num_files * self.ds_cfg.traj_sample_percentage / 100)

                # If needed, randomly sample a number of files
                selected_files = random.sample(track_files, num_to_select) if num_to_select < num_files else track_files
                filtered_files.extend(selected_files)
            files = filtered_files

        return files

    def extract_track_from_filename(self, filename):
        """Extract the track ID from a filename, e.g., 'track1_rep1_doppler.npy' -> 1"""
        base_name = os.path.basename(filename)

        # Regex explanation:
        # ^track      : starts with 'track'
        # (\d+)       : matches one or more digits and captures them
        # _           : followed by an underscore
        pattern = r'^track(\d+)_'
        match = re.search(pattern, base_name)
        if match:
            # group(1) corresponds to the content captured by the first parentheses (\d+)
            return int(match.group(1))
        else:
            # If the filename does not match the format, return None for later filtering
            return None
        # base_name = os.path.basename(filename)
        # parts = base_name.split('_')  # Assume filename format: trackX_repY_doppler.npy
        # track_part = parts[0]  # Get the track part, e.g., 'track1'
        # track_number = int(track_part[5:])  # Extract the numeric part after 'track'

        # return track_number

    def check_file(self, file):
        return True

    def preprocess(self, arr):
        """Preprocessing method: by default, no processing is applied"""
        return arr

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        # --- 1. Fetch basic data ---
        if self.data_cache is not None:
            out = self.data_cache[idx]
            # Fault-tolerance
            if out is None:
                out = self.preprocess(np.load(self.files[idx]))
        else:
            # Lazy loading logic
            path = self.files[idx]
            arr = np.load(path, mmap_mode='r')
            out = self.preprocess(arr)

        # --- 2. Dynamic channel selection ---
        # If active_channels is configured in ds_cfg, slice here
        m_name = getattr(self, 'modality', 'default').lower()
        specific_key = f"{m_name}_channels"
        target_channels = getattr(self.ds_cfg, specific_key, getattr(self.ds_cfg, 'active_channels', None))

        if target_channels is not None:
            if isinstance(target_channels, int):
                target_channels = [target_channels]

            actual_channels = out.shape[0]
            safe_channels = [c for c in target_channels if c < actual_channels]

            if not safe_channels:
                # You can choose to raise an error here, or fall back to using all channels
                print(
                    f"Warning: modality {m_name} requested channels {target_channels} are out of range (only {actual_channels} available). Falling back to all.")
                safe_channels = slice(None)  # equivalent to [:]
            out = out[safe_channels, :, :]

        # --- 3. Label handling and type conversion ---
        label = self.labels[idx]

        # Ensure a Tensor is returned (required by deep learning models)
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out.copy() if not out.flags['C_CONTIGUOUS'] else out)

        return out, label
    # def __getitem__(self, idx):
    #     # --- Core change: prioritize cache ---
    #     if self.data_cache is not None:
    #         out = self.data_cache[idx]
    #         # Fault-tolerance: if preloading failed, temporarily try lazy loading
    #         if out is None:
    #             out = self.preprocess(np.load(self.files[idx]))
    #         label = self.labels[idx]
    #         return out, label
    #
    #     # --- Original lazy loading logic ---
    #     path = self.files[idx]
    #     # Note: if cache is not used, do NOT .copy() in __getitem__, otherwise swap may be triggered
    #     arr = np.load(path, mmap_mode='r')
    #     out = self.preprocess(arr)
    #     label = self.labels(idx)
    #
    #     return out, label


@register_dataset("doppler")
class DopplerDataset(BaseDataset):
    def __init__(self, root, modality, ds_cfg:DatasetConfig = None, files=None, labels=None, cache=None):
        self.modality = modality
        super().__init__(root, ds_cfg, files,labels,cache)

    def create_subset(self, files, labels, cache=None):
        return DopplerDataset(
            root=self.root,
            modality=self.modality,
            ds_cfg=self.ds_cfg,
            files=files,
            labels=labels,
            cache=cache  # Key: pass cache down
        )

    @staticmethod
    def get_collate_fn():
        return collate_fn_variable_length

    def get_dataset_dir(self, user):
        """Return the data directory for the Doppler dataset"""
        return os.path.join(self.root, user, self.modality, self.ds_cfg.doppler_dir)

    def check_file(self, file):
        # arr = np.load(file)
        # if arr.shape[0] != 3:  # File does not meet requirements; skip
        #     print(f"[DROP] {file}")
        #     return False
        return True

    def preprocess(self, arr):
        """Implement preprocessing logic for the Doppler dataset (example)"""
        out = preprocess_doppler(arr, freq_down=self.ds_cfg.freq_down,
                                 time_down=self.ds_cfg.time_down,
                                 normalize=True,
                                 merge_ncnt=self.ds_cfg.merge_ncnt_as_channel)

        # if hasattr(self.ds_cfg, 'active_channels') and self.ds_cfg.active_channels is not None:
        #     active_idx = self.ds_cfg.active_channels  # Set in config, e.g., [0] or [0, 2]
        #     out = out[active_idx, :, :]

        return out

@register_dataset("gbvp")
class GBVPDataSet(BaseDataset):
    def __init__(self, root, modality, ds_cfg:DatasetConfig = None, files=None, labels=None, cache=None):
        self.modality = modality
        super().__init__(root, ds_cfg, files, labels, cache)

    def create_subset(self, files, labels, cache=None):
        return GBVPDataSet(
            root=self.root,
            modality=self.modality,
            ds_cfg=self.ds_cfg,
            files=files,
            labels=labels,
            cache=cache
        )

    @staticmethod
    def get_collate_fn():
        return None

    def get_dataset_dir(self, user):
        """Return the data directory for the GBVP dataset"""
        return os.path.join(self.root, user, self.modality, self.ds_cfg.GBVP_dir)

    def preprocess(self, arr):
        """Implement preprocessing logic for the GBVP dataset (example)"""
        # todo
        return arr


@register_dataset("fusion_gbvp")
class FusionGBVPDataSet(GBVPDataSet):
    def get_dataset_dir(self, user):
        """Return the data directory for the GBVP dataset"""
        return os.path.join(self.root, user, self.modality, self.ds_cfg.FusionGBVP_dir)

    @staticmethod
    def get_collate_fn():
        return collate_fn_variable_length

    def preprocess(self, arr):
        """Implement preprocessing logic for the PPVP dataset (example)"""
        out = preprocess(arr)
        return out


@register_dataset("ppvp")
class PPVPDataSet(BaseDataset):
    def __init__(self, root, modality, ds_cfg:DatasetConfig = None, files=None, labels=None,cache=None):
        self.modality = modality
        super().__init__(root, ds_cfg, files, labels,cache)

    def create_subset(self, files, labels, cache=None):
        return PPVPDataSet(
            root=self.root,
            modality=self.modality,
            ds_cfg=self.ds_cfg,
            files=files,
            labels=labels,
            cache=cache
        )

    @staticmethod
    def get_collate_fn():
        return collate_fn_variable_length

    def get_dataset_dir(self, user):
        """Return the data directory for the PPVP dataset"""
        return os.path.join(self.root, user, self.modality, self.ds_cfg.PPVP_dir)

    def preprocess(self, arr):
        """Implement preprocessing logic for the PPVP dataset (example)"""
        out = preprocess(arr, direc_down= self.ds_cfg.direc_down)
        return out


@register_dataset("fusion_ppvp")
class FusionPPVPDataSet(PPVPDataSet):
    def get_dataset_dir(self, user):
        """Return the data directory for the PPVP dataset"""
        return os.path.join(self.root, user, self.modality, self.ds_cfg.FusionPPVP_dir)


@register_dataset("doppler_fusion")
class DopplerFusionDataset(Dataset):
    """
    Multi-modal fusion dataset container.
    It does not directly hold file paths. Instead, it holds multiple single-modality Dataset instances
    and maintains index alignment between them.
    """

    def __init__(
            self,
            ds_dict: Dict[str, BaseDataset],
            pair_by: str = "name",
            samples: Optional[List[Tuple[int, ...]]] = None
    ):
        """
        Args:
            ds_dict: Mapping from modality name to Dataset instance, e.g. {"wifi": wifi_ds, "acoustic": ac_ds}
            pair_by: Pairing mode, 'name' (align by filename) or 'index' (simple index alignment)
            samples: Predefined index-pair list [(wifi_idx, ac_idx), ...], used by split_by_traj to create subsets.
        """
        self.ds_dict = ds_dict
        self.modality_names = list(ds_dict.keys())
        self.pair_by = pair_by

        # Core: if initializing, do pairing; if create_subset, use the provided samples directly
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._pair_datasets()

    def _pair_datasets(self) -> List[Tuple[int, ...]]:
        """Pair multiple datasets by filename or by index"""
        # Currently the logic mainly targets two modalities; extend the loop here if more modalities are added
        ds_a = self.ds_dict[self.modality_names[0]]
        ds_b = self.ds_dict[self.modality_names[1]]

        if self.pair_by == "index":
            L = min(len(ds_a), len(ds_b))
            return [(i, i) for i in range(L)]
        else:
            return self._pair_by_filename(ds_a, ds_b)

    def _pair_by_filename(self, ds_a: BaseDataset, ds_b: BaseDataset) -> List[Tuple[int, int]]:
        def build_map(dataset_obj):
            keymap = {}
            # Match the pattern 'user' followed by digits
            user_pattern = re.compile(r'user\d+')

            for idx, f in enumerate(dataset_obj.files):
                match = user_pattern.search(f)
                user_dir = match.group(0) if match else "unknown"

                name = os.path.basename(f)
                parts = name.split('_')
                key = f"{user_dir}_{parts[0]}_{parts[1]}" if len(parts) >= 2 else f"{user_dir}_{name}"
                keymap.setdefault(key, []).append(idx)
            return keymap

        # Pass the entire ds object rather than just the files list, so paths can be extracted
        map_a = build_map(ds_a)
        map_b = build_map(ds_b)

        pairs = []
        # Keep the rest of the logic consistent; since the key is unique, n should theoretically always be 1
        for key in map_a:
            if key in map_b:
                n = min(len(map_a[key]), len(map_b[key]))
                for i in range(n):
                    pairs.append((map_a[key][i], map_b[key][i]))

        pairs.sort()
        return pairs

    def create_subset(self, samples: List[Tuple[int, ...]]):
        """
        Called by split_by_traj to create a new fusion subset.
        It shares the underlying ds_dict but has a different list of index pairs.
        """
        return DopplerFusionDataset(
            ds_dict=self.ds_dict,
            pair_by=self.pair_by,
            samples=samples
        )

    def split_by_traj(self, train_ids: List[int], val_ids: List[int], test_ids: List[int]):
        """
        Split into three subsets by trajectory ID.
        Logic: use the extract_track_from_filename method provided by the first modality's BaseDataset.
        """
        train_s, val_s, test_s = [], [], []

        # Borrow the parsing logic from the first modality
        ref_ds = self.ds_dict[self.modality_names[0]]

        for sample_pair in self.samples:
            # Index in the original list for the first modality
            ref_idx = sample_pair[0]
            f_path = ref_ds.files[ref_idx]

            # Extract trajectory ID
            t_id = ref_ds.extract_track_from_filename(f_path)

            if t_id in train_ids:
                train_s.append(sample_pair)
            elif t_id in val_ids:
                val_s.append(sample_pair)
            elif t_id in test_ids:
                test_s.append(sample_pair)

        return (
            self.create_subset(train_s),
            self.create_subset(val_s),
            self.create_subset(test_s)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return format: ([data_wifi, data_ac], label)
        """
        sample_indices = self.samples[idx]

        datas = []
        labels = []

        for m_idx, m_name in enumerate(self.modality_names):
            ds = self.ds_dict[m_name]
            real_idx = sample_indices[m_idx]

            data, label = ds[real_idx]  # Call the single-modality __getitem__
            datas.append(data)
            labels.append(label)

        # Safety check: labels across modalities must be consistent
        if not all(l == labels[0] for l in labels):
            raise ValueError(f"Inconsistent labels at fusion index {idx}: {labels}")

        return datas, labels[0]

    @staticmethod
    def get_collate_fn():
        """
        Note: fusion mode must use a dedicated fusion_collate
        """
        return collate_fn_fusionDoppler_variable_length




import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any

@register_dataset("ppvp_fusion") # Fusion container tailored for PPVP
class PPVPFusionDataset(Dataset):
    """
    A multi-modal fusion dataset customized for PPVP tasks.
    It manages multiple PPVPDataSet instances and ensures sample-level alignment.
    """
    def __init__(
            self,
            ds_dict: Dict[str, Any],  # Contains {'wifi': PPVPDataSet_inst, 'acoustic': ...}
            pair_by: str = "name",
            samples: Optional[List[Tuple[int, ...]]] = None
    ):
        self.ds_dict = ds_dict
        self.modality_names = list(ds_dict.keys())
        self.pair_by = pair_by

        # If this is a subset created from split_by_traj, directly use the provided index pairs
        if samples is not None:
            self.samples = samples
        else:
            # Run pairing logic during initialization
            self.samples = self._pair_datasets()

    def _pair_datasets(self) -> List[Tuple[int, ...]]:
        """Pair multiple datasets by filename or by index"""
        # Currently the logic mainly targets two modalities; extend the loop here if more modalities are added
        ds_a = self.ds_dict[self.modality_names[0]]
        ds_b = self.ds_dict[self.modality_names[1]]

        if self.pair_by == "index":
            L = min(len(ds_a), len(ds_b))
            return [(i, i) for i in range(L)]
        else:
            return self._pair_by_filename(ds_a, ds_b)

    def _pair_by_filename(self, ds_a: BaseDataset, ds_b: BaseDataset) -> List[Tuple[int, int]]:
        def build_map(dataset_obj):
            keymap = {}
            # Match the pattern 'user' followed by digits
            user_pattern = re.compile(r'user\d+')

            for idx, f in enumerate(dataset_obj.files):
                match = user_pattern.search(f)
                user_dir = match.group(0) if match else "unknown"

                name = os.path.basename(f)
                parts = name.split('_')
                key = f"{user_dir}_{parts[0]}_{parts[1]}" if len(parts) >= 2 else f"{user_dir}_{name}"
                keymap.setdefault(key, []).append(idx)
            return keymap

        # Pass the entire ds object rather than just the files list, so paths can be extracted
        map_a = build_map(ds_a)
        map_b = build_map(ds_b)

        pairs = []
        # Keep the rest of the logic consistent; since the key is unique, n should theoretically always be 1
        for key in map_a:
            if key in map_b:
                n = min(len(map_a[key]), len(map_b[key]))
                for i in range(n):
                    pairs.append((map_a[key][i], map_b[key][i]))

        pairs.sort()
        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return format: ( [data_m1, data_m2, ...], label )
        """
        sample_indices = self.samples[idx]

        datas = []
        labels = []

        for m_idx, m_name in enumerate(self.modality_names):
            ds = self.ds_dict[m_name]
            real_idx = sample_indices[m_idx]

            # Call PPVPDataSet.__getitem__
            data, label = ds[real_idx]
            datas.append(data)
            labels.append(label)

        # Validate label consistency (PPVP: same action across modalities must share the same label)
        if not all(l == labels[0] for l in labels):
            raise ValueError(f"Label mismatch at index {idx}: {labels}")

        return datas, labels[0]

    @staticmethod
    def get_collate_fn():
        """
        Note: fusion mode must use a dedicated fusion_collate
        """
        return collate_fn_fusionPPVP_variable_length

    def create_subset(self, samples: List[Tuple[int, ...]]):
        """Used for splitting datasets"""
        return PPVPFusionDataset(
            ds_dict=self.ds_dict,
            pair_by=self.pair_by,
            samples=samples
        )

    def split_by_traj(self, train_ids, val_ids, test_ids):
        """
        Refer to the Doppler logic to split by trajectory or user ID
        """
        train_s, val_s, test_s = [], [], []
        ref_ds = self.ds_dict[self.modality_names[0]]

        for pair in self.samples:
            ref_idx = pair[0]
            f_path = ref_ds.files[ref_idx]

            # Assume PPVPDataSet also has extract_track_from_filename
            t_id = ref_ds.extract_track_from_filename(f_path)

            if t_id in train_ids:
                train_s.append(pair)
            elif t_id in val_ids:
                val_s.append(pair)
            elif t_id in test_ids:
                test_s.append(pair)

        return (
            self.create_subset(train_s),
            self.create_subset(val_s),
            self.create_subset(test_s)
        )
