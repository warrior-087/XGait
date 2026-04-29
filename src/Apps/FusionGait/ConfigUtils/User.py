import os
import re
from pathlib import Path
from typing import List, Dict

import numpy as np

from src.data_path_cfg import base_paths

class BaseData:
    """
    Base data class: includes doppler, track_los
    """
    def __init__(self, user_id: int):
        self.user_id = user_id

        # Store processed results (trajectory_id -> rep_id -> ndarray)
        self.doppler_spectrum: Dict[int, Dict[int, str]] = {} # Doppler spectrum info
        self.freq_bin: Dict[int, Dict[int, str]] = {}  # frequency bin/window info
        self.locations: Dict[int, Dict[int, str]] = {}  # store location data for each trajectory
        self.speeds: Dict[int, Dict[int, str]] = {}  # store speed data for each trajectory
        self.duration: Dict[int, Dict[int, str]] = {} # timing info

    def release(self):
        self.doppler_spectrum.clear()
        self.freq_bin.clear()
        self.locations.clear()
        self.speeds.clear()
        self.duration.clear()


    def set_doppler_spectrum(self, track_id: int, rep_id: int, spectrum: np.ndarray, base_dir: str = None):
        """
        Save the Doppler spectrum to file and record its path
        """
        if base_dir is None:
            raise ValueError("base_dir must be provided for saving spectra")

        doppler_dir = os.path.join(base_dir, "BaseData/Doppler")
        os.makedirs(doppler_dir, exist_ok=True)

        filename = f"track{track_id}_rep{rep_id}_doppler.npy"
        file_path = os.path.join(doppler_dir, filename)
        np.save(file_path, spectrum)

        self.doppler_spectrum.setdefault(track_id, {})[rep_id] = file_path

    def set_freq_bin(self, track_id: int, rep_id: int, freq_bin: np.ndarray, base_dir: str = None):
        """
        Save the Doppler spectrum to file and record its path
        """
        if base_dir is None:
            raise ValueError("base_dir must be provided for saving spectra")

        freq_dir = os.path.join(base_dir, "BaseData/Freq")
        os.makedirs(freq_dir, exist_ok=True)

        filename = f"track{track_id}_rep{rep_id}_freq.npy"
        file_path = os.path.join(freq_dir, filename)
        np.save(file_path, freq_bin)

        self.freq_bin.setdefault(track_id, {})[rep_id] = file_path

    def set_locations(self, track_id: int, rep_id: int, locations: np.ndarray, base_dir: str = None):
        """
        Save the trajectory location data to file and record its path
        """
        if base_dir is None:
            raise ValueError("base_dir must be provided for saving locations")

        locations_dir = os.path.join(base_dir, "BaseData/Locations")
        os.makedirs(locations_dir, exist_ok=True)

        filename = f"track{track_id}_rep{rep_id}_locations.npy"
        file_path = os.path.join(locations_dir, filename)
        np.save(file_path, locations)

        self.locations.setdefault(track_id, {})[rep_id] = file_path

    def set_speeds(self, track_id: int, rep_id: int, speeds: np.ndarray, base_dir: str = None):
        """
        Save the trajectory speed data to file and record its path
        """
        if base_dir is None:
            raise ValueError("base_dir must be provided for saving speeds")

        speeds_dir = os.path.join(base_dir, "BaseData/Speeds")
        os.makedirs(speeds_dir, exist_ok=True)

        filename = f"track{track_id}_rep{rep_id}_speeds.npy"
        file_path = os.path.join(speeds_dir, filename)
        np.save(file_path, speeds)

        self.speeds.setdefault(track_id, {})[rep_id] = file_path

    def set_duration(self, track_id: int, rep_id: int, duration: np.ndarray, base_dir: str = None):
        """
        Save the trajectory speed data to file and record its path
        """
        if base_dir is None:
            raise ValueError("base_dir must be provided for saving speeds")

        duration_dir = os.path.join(base_dir, "BaseData/Duration")
        os.makedirs(duration_dir, exist_ok=True)

        filename = f"track{track_id}_rep{rep_id}_duration.npy"
        file_path = os.path.join(duration_dir, filename)
        np.save(file_path, duration)

        self.duration.setdefault(track_id, {})[rep_id] = file_path

    def get_doppler_spectrum(self, track_id: int, rep_id: int):
        path = self.doppler_spectrum.get(track_id, {}).get(rep_id)
        if path and os.path.exists(path):
            return np.load(path)
        else:
            #raise FileNotFoundError(f"Doppler file not found: {path}")
            return None

    def get_freq_bin(self, track_id: int, rep_id: int):
        path = self.freq_bin.get(track_id, {}).get(rep_id)
        if path and os.path.exists(path):
            return np.load(path)
        else:
            #raise FileNotFoundError(f"freq file not found: {path}")
            return None

    def get_locations(self, track_id: int, rep_id: int):
        path = self.locations.get(track_id, {}).get(rep_id)
        if path and os.path.exists(path):
            return np.load(path)
        else:
            #raise FileNotFoundError(f"Locations file not found: {path}")
            return None

    def get_speeds(self, track_id: int, rep_id: int):
        path = self.speeds.get(track_id, {}).get(rep_id)
        if path and os.path.exists(path):
            return np.load(path)
        else:
            #raise FileNotFoundError(f"Speeds file not found: {path}")
            return None

    def get_duration(self, track_id: int, rep_id: int):
        path = self.duration.get(track_id, {}).get(rep_id)
        if path and os.path.exists(path):
            return np.load(path)
        else:
            #raise FileNotFoundError(f"Duration file not found: {path}")
            return None

# --------------------------------------------------------------------------------
    # Save/load metadata
    # --------------------------------------------------------------------------------
    def set_all_data(self, track_id: int, rep_id: int,
                     spectrum: np.ndarray, freq_bin: np.ndarray, locations: np.ndarray, speeds: np.ndarray, duration: np.ndarray,
                     base_dir: str = None):
        self.set_doppler_spectrum(track_id, rep_id, spectrum, base_dir = base_dir)
        self.set_freq_bin(track_id, rep_id, freq_bin, base_dir=base_dir)
        self.set_locations(track_id, rep_id, locations, base_dir=base_dir)
        self.set_speeds(track_id, rep_id, speeds, base_dir=base_dir)
        self.set_duration(track_id, rep_id, duration, base_dir=base_dir)

    def get_data_by_id(self, track_id: int, rep_id: int):
        spectrum = self.get_doppler_spectrum(track_id, rep_id)
        freq_bin = self.get_freq_bin(track_id,rep_id)
        locations = self.get_locations(track_id,rep_id)
        speeds = self.get_speeds(track_id,rep_id)
        duration = self.get_duration(track_id,rep_id)

        return spectrum,freq_bin,locations,speeds,duration

    def save_to_file(self, save_dir: str):
        """
        Save the current user's path index and spectrum paths to metadata.npy.
        Do not save ndarray anymore.
        """
        os.makedirs(save_dir, exist_ok=True)

        metadata = {
            "doppler": self.doppler_spectrum,
            "freq": self.freq_bin,
            "locations": self.locations,
            "speeds": self.speeds,
            "duration": self.duration
        }

        np.save(os.path.join(save_dir, "basedata_metadata.npy"), metadata)
        print(f"[User {self.user_id}] base_data saved to {save_dir}")

    def load_from_file(self, save_dir: str):
        """
        Restore path references from metadata.npy
        """
        meta_path = os.path.join(save_dir, "basedata_metadata.npy")

        if not os.path.exists(meta_path):
            #print(f"[User {self.user_id}] No metadata found at {meta_path}")
            return False

        metadata = np.load(meta_path, allow_pickle=True).item()
        self.doppler_spectrum = metadata.get("doppler", {})
        self.freq_bin = metadata.get("freq",{})
        self.locations = metadata.get("locations",{})
        self.speeds = metadata.get("speeds",{})
        self.duration = metadata.get("duration",{})

        #print(f"[User {self.user_id}] base_data loaded from {meta_path}")
        return True

class VelocitySpectrumData:
    """
    Velocity-spectrum data class:
    """
    def __init__(self, user_id: int):
        self.user_id = user_id

        # Store processed results (trajectory_id -> rep_id -> ndarray)
        #self.doppler_spectrum: Dict[int, Dict[int, str]] = {}
        self.processed_spectrum: Dict[int, Dict[int, str]] = {}

    def release(self):
        self.processed_spectrum.clear()

    # Set spectrum: save immediately + store path
    def set_processed_spectrum(self, track_id: int, rep_id: int, spectrum: np.ndarray, base_dir: str = None):
        """
        Save processed spectrum to file and record its path
        """
        if base_dir is None:
            raise ValueError("base_dir must be provided for saving spectra")

        #pro_dir = os.path.join(base_dir, "pro")
        os.makedirs(base_dir, exist_ok=True)

        filename = f"track{track_id}_rep{rep_id}_pro.npy"
        file_path = os.path.join(base_dir, filename)
        np.save(file_path, spectrum)

        self.processed_spectrum.setdefault(track_id, {})[rep_id] = file_path

    # --------------------------------------------------------------------------------
    # Spectrum access interface: lazy loading
    # --------------------------------------------------------------------------------
    def get_processed_spectrum(self, track_id: int, rep_id: int):
        path = self.processed_spectrum.get(track_id, {}).get(rep_id)
        if path and os.path.exists(path):
            return np.load(path)
        else:
            raise FileNotFoundError(f"Processed spectrum not found: {path}")

    # --------------------------------------------------------------------------------
    # Save/load metadata
    # --------------------------------------------------------------------------------
    def save_to_file(self, save_dir: str):
        """
        Save the current user's path index and spectrum paths to metadata.npy.
        Do not save ndarray anymore.
        """
        os.makedirs(save_dir, exist_ok=True)

        metadata = {
            "processed": self.processed_spectrum,
        }
        #print(self.processed_spectrum)
        np.save(os.path.join(save_dir, "pro_metadata.npy"), metadata)
        print(f"[User {self.user_id}] metadata saved to {save_dir}")

    def load_from_file(self, save_dir: str):
        """
        Restore path references from metadata.npy
        """
        meta_path = os.path.join(save_dir, "pro_metadata.npy")

        if not os.path.exists(meta_path):
            print(f"[User {self.user_id}] No metadata found at {meta_path}")
            return False

        metadata = np.load(meta_path, allow_pickle=True).item()
        self.processed_spectrum = metadata.get("processed", {})
        print(f"[User {self.user_id}] metadata loaded from {meta_path}")
        return True

    def check_is_exists(self, track_id: int, rep_id: int):
        path = self.processed_spectrum.get(track_id, {}).get(rep_id)
        if path and os.path.exists(path):
            return True
        else:
            return False

class UserData:
    def __init__(self, user_id: int,
                 modalities=("wifi", "acoustic", "fusion"),
                 methods=("GBVP", "PPVP"),
                 fusion_methods=("FusionGBVP","FusionPPVP")):

        self.user_id = user_id
        self.modalities = modalities
        self.method = methods
        self.fusion_methods = fusion_methods
        self.raw_data: Dict[str, Dict[int, Dict[int, Dict[int, Path]]]]= {}
        self.base_data: Dict[str, BaseData] = {}
        self.velocity_spectrum_data: Dict[(str, str), VelocitySpectrumData] = {}

        self.base_data: Dict[str, BaseData] = {
            mod: BaseData(user_id) for mod in modalities
        }

        for mod in modalities:
            self.raw_data[mod] = {}
            self.base_data[mod] = BaseData(user_id)
            if mod != "fusion":
                for method in methods:
                    self.velocity_spectrum_data[(mod, method)] = VelocitySpectrumData(user_id)
            else:
                for method in fusion_methods:
                    self.velocity_spectrum_data[(mod, method)] = VelocitySpectrumData(user_id)

    def release(self):
        for mod in self.raw_data:
            self.base_data[mod].release()
            for track in self.raw_data[mod]:
                for rep in self.raw_data[mod][track]:
                    self.raw_data[mod][track][rep].clear()
                self.raw_data[mod][track].clear()
            self.raw_data[mod].clear()

        for key, db in self.velocity_spectrum_data.items():
            if hasattr(db, "release") and callable(db.release):
                db.release()
            else:
                # Compatibility with DataBase without release(); clear its contents directly
                if hasattr(db, "pro"):
                    db.pro.clear()
                if hasattr(db, "metadata"):
                    db.metadata.clear()

    def get_base_modalities(self):
        """
        Return the base modality list excluding Fusion
        """
        return tuple(m for m in self.modalities if m.lower() != "fusion")

    def get_base_methods(self):
        return self.method

    def get_fusion_methods(self):
        return self.fusion_methods

    def add_file(self, modality: str, track_id: int, rep_id: int, rx_id: int, path: Path):
        """Add the data file path for a specific trajectory instance"""
        self.raw_data[modality].setdefault(track_id, {}).setdefault(rep_id, {})[rx_id] = path

    def get_rx_group(self, track: int, modality : str) -> List[Dict[int, Path]]:
        """
        Get all receiver data paths for a given trajectory.
        Args:
            track (int): trajectory index. If -1, return all trajectories.
        Returns:
            List[Dict[int, Path]]: each element is {rx_no: Path}, the Rx file paths.
        """
        results = []

        if track == -1:
            # Iterate all trajectories
            for track_no, reps in self.raw_data[modality].items():
                for rep, rx_dict in reps.items():
                    # if len(rx_dict) < 6:
                    #     #raise ValueError(f"Track {track_no}, rep {rep} has only {len(rx_dict)} Rx files, expected 6.")
                    #     continue
                    # Convert to string, then remove suffix like -rX.dat
                    first_file = next(iter(rx_dict.values()))
                    prefix = str(first_file).rsplit("-r", 1)[0]
                    results.append(prefix)
        else:
            if track not in self.raw_data[modality]:
                raise KeyError(f"Track {track} not found in paths_index")

            for rep, rx_dict in self.raw_data[modality][track].items():
                if len(rx_dict) < 6:
                    #raise ValueError(f"Track {track}, rep {rep} has only {len(rx_dict)} Rx files, expected 6.")
                    continue
                # Convert to string, then remove suffix like -rX.dat
                first_file = next(iter(rx_dict.values()))
                prefix = str(first_file).rsplit("-r", 1)[0]
                results.append(prefix)

        return results

    def get_velocity_spectrum(self, modality: str, method: str) -> VelocitySpectrumData:
        return self.velocity_spectrum_data[(modality, method)]

    def get_base_data(self, modality: str) -> BaseData:
        return self.base_data[modality]

    # ---------------- Save data ----------------
    def save_velocity_spectrum(self, save_root: str, modality:str, method:str):
        self.velocity_spectrum_data[modality, method].save_to_file(save_root)

    # ---------------- Load data ----------------
    def load_velocity_spectrum(self, save_root: str, modality:str, method:str):
        return self.velocity_spectrum_data[modality, method].load_from_file(save_root)

    def check_if_velocity_spectrum_exists(self, modality:str, method:str, track_id:int , rep_id:int):
        return self.velocity_spectrum_data[modality, method].check_is_exists(track_id, rep_id)


    def save_base_data(self, save_root: str, modality:str):
        self.base_data[modality].save_to_file(save_root)

    # ---------------- Load data ----------------
    def load_base_data(self, save_root: str, modality:str):
        return self.base_data[modality].load_from_file(save_root)


class User:
    def __init__(self, user_id: int, data_root: str, user_data: 'UserData'):
        """
        :param user_id: user index
        :param data_root: root directory of the dataset (contains userX folders)
        """
        self.user_id = user_id
        self.root_dir = Path(data_root) / f"user{user_id}"

        self.wifi_dir = self.root_dir / "wifi"
        self.acoustic_dir = self.root_dir / "acoustic"

        self.track_groundTruth_dir = self.root_dir / "track_groundTruth"

        self.data = user_data
        self._parse_paths()

        self.wifi_path_groups = self.data.get_rx_group(-1, "wifi")
        self.acoustic_path_groups = self.data.get_rx_group(-1, "acoustic")

    def release(self):
        """Release all data occupying memory"""
        # Clear DataBase data
        if self.data is not None:
            self.data.release()

        # Clear User internal references
        self.data = None
        self.wifi_path_groups = None
        self.acoustic_path_groups = None
        self.root_dir = None
        self.wifi_dir = None
        self.acoustic_dir = None

        print(f"User {self.user_id} data processing finished and released.")

    def __del__(self):
        """Defensively call release (not guaranteed to be invoked)"""
        self.release()

    def get_path_groups(self, modality: str):
        if modality == "wifi":
            return self.wifi_path_groups
        elif modality == "acoustic":
            return self.acoustic_path_groups
        else:
            return

    def _parse_paths(self):
        """
        Compatibility parsing:
        New version: user1-1-11-d1-r1.wav (with tag)
        Old version: user1-1-11-r1.wav    (without tag)
        """

        def parse_files(directory, pattern, dtype):
            for f in directory.glob("*"):
                m = pattern.match(f.name)
                if m:
                    # m.group(1): user_id
                    # m.group(2): trajid
                    # m.group(3): repid
                    # m.group(4): tag (None if not matched)
                    # m.group(5): rx_id
                    u_id, t_id, r_id = map(int, m.groups()[:3])
                    rx_id = int(m.group(5))

                    if u_id == self.user_id:
                        self.data.add_file(dtype, t_id, r_id, rx_id, f)

        # Core regex improvement:
        # (?:-([^-]+))?  --> matches the optional "-tag" part
        wifi_pattern = re.compile(r'^(?:user)?(\d+)-(\d+)-(\d+)(?:-([^-]+))?-r(\d+)\.dat$', re.IGNORECASE)
        acoustic_pattern = re.compile(r'^(?:user)?(\d+)-(\d+)-(\d+)(?:-([^-]+))?-r(\d+)\.wav$', re.IGNORECASE)

        parse_files(self.wifi_dir, wifi_pattern, "wifi")
        parse_files(self.acoustic_dir, acoustic_pattern, "acoustic")


    def save_velocity_spectrum(self, file_path, modality, method):
        self.data.save_velocity_spectrum(file_path, modality, method)

    def load_velocity_spectrum(self, file_path, modality, method):
        self.data.load_velocity_spectrum(file_path,modality, method)

    def save_base_data(self, file_path, modality):
        self.data.save_base_data(file_path,modality)

    def load_base_data(self, file_path, modality):
        self.data.load_base_data(file_path,modality)
