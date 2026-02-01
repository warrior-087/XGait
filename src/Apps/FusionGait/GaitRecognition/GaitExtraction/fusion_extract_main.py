import multiprocessing
import os
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

from src.Apps.FusionGait.MatlabPlugins.init_matlab_path import init_matlab_runtime_path
init_matlab_runtime_path()
from src.Apps.FusionGait.AlgorithmUtils.position_tracker import MultiModalPositionTracker
from src.Apps.FusionGait.ConfigUtils.track_config import GroundTruthTrajectory

from src.Apps.FusionGait.GaitRecognition.GaitExtraction.utils import load_all_users, get_user_in_process, get_tracker_in_process, parse_track_rep, SuppressMCR, get_spectrum_generator, \
    get_dfss_in_process

from src.Apps.FusionGait.ConfigUtils.expriment_config import ExperimentConfig

def worker_wrapper(task_info, expCfg, engine_name_queue):
    """
    task_info: the original task data
    engine_name_queue: a queue storing available engine names
    """
    # 1. Take an available engine name from the queue
    engine_name = engine_name_queue.get()

    try:
        # 2. Connect to the engine (singleton mode)
        from src.Apps.FusionGait.MatlabPlugins.matlab_runtime_singleton import MatlabEngineManager
        # Child process connects to a pre-started engine from the main process
        MatlabEngineManager.connect_to_named_engine(engine_name)

        # 3. Execute the original logic
        # Note: you need to modify your process_user_modality so that it calls MatlabEngineManager.call internally
        result = process_user(task_info, expCfg=expCfg)
        return result
    finally:
        # 4. Put the engine name back for the next task to use
        engine_name_queue.put(engine_name)

def build_instance_map(paths):
    """
    Return:
    {
        (track_id, rep_id): path
    }
    """
    inst_map = {}
    for p in paths:
        track_id, rep_id = parse_track_rep(p)
        inst_map[(track_id, rep_id)] = p
    return inst_map

def get_fusion_tracker_in_process(modalities, expCfg: ExperimentConfig, base_dir):
    """
    Maintain a tracker singleton within each process
    Note: different processes do not share caches
    """
    tracker_dict = {}
    # if "fusion" in tracker_dict:
    #     fusion_tracker = tracker_dict["fusion"]
    #     fusion_tracker._update_base_dir(base_dir)
    #     return fusion_tracker

    for modality in modalities:
        tracker_dict[modality] = get_tracker_in_process(modality, expCfg)

    traj_truth = GroundTruthTrajectory(expCfg.data_root_path)
    fusion_tracker = MultiModalPositionTracker(tracker_dict, traj_truth, base_dir, expCfg.fusion_config)

    #tracker_dict["fusion"] = fusion_tracker

    return fusion_tracker

# ========================================
# Core function executed in child process (one task corresponds to one user)
# ========================================
def process_user(task, expCfg):
    """
    task = (scenario_name, user_id)
    Execute this function in the child process
    """
    scenario_name, user_id = task

    print(f"[Start] Processing User {user_id}, Scenario: {scenario_name}")

    # -------- Per-process singleton: build its own User copy using user_id --------
    data_root = os.path.join(expCfg.data_root_path, scenario_name)
    user = get_user_in_process(user_id, data_root)

    modalities = user.data.get_base_modalities()
    exe_base_dir = os.path.normpath(os.path.join(
        expCfg.exe_path, scenario_name, f"user{user_id}"
    ))
    # Use both WiFi and Acoustic modalities for fusion-based tracking
    fusion_tracker = get_fusion_tracker_in_process(modalities, expCfg, exe_base_dir)

    fusion_instance_map = {}
    for modality in modalities:
        paths_prefix = user.get_path_groups(modality)
        fusion_instance_map[modality] = build_instance_map(paths_prefix)
    # Key set for each modality
    key_sets = [set(inst_map.keys()) for inst_map in fusion_instance_map.values()]
    common_keys = set.intersection(*key_sets) # intersection
    if not common_keys:
        print("[Warning] No common (track_id, rep_id) across modalities")
        return

    fusion_instances = []
    dfss_generators = {}
    for key in sorted(common_keys):
        instance = {
            'track_id': key[0],
            'rep_id': key[1],
        }
        for modality in modalities:
            instance[f'{modality}_path'] = fusion_instance_map[modality][key]
            dfss_generators[modality] = get_dfss_in_process(modality)
        fusion_instances.append(instance)

    for track_instance in fusion_instances:
        track_id = track_instance['track_id']
        rep_id = track_instance['rep_id']
        print(f"[Fusion Tracking] User {user_id}, Track ID: {track_id}, Rep ID: {rep_id}")

        paths = {
            'wifi': track_instance['wifi_path'],
            'acoustic': track_instance['acoustic_path']
        }

        with SuppressMCR():
            modality_params, pos, vs = fusion_tracker._fusion_successive_track(
                dfss_generators = dfss_generators,
                paths=paths,
                traj_id=track_id,
                rep_id=rep_id,
                user=user
            )

        ## Iterate over different fusion methods and save the extracted features
        fusion_methods = user.data.get_fusion_methods()
        print(fusion_methods)
        for method in fusion_methods:
            # if method == "FusionPPVP":
            #     continue

            spectrum_dir = os.path.normpath(
                os.path.join(expCfg.exe_path, scenario_name, f"user{user.user_id}", "fusion", method))
            generator = get_spectrum_generator(method, modalities, expCfg) # for fusion, you need to pass two modalities

            # Load cached files first
            is_load_velocitySpec_success = user.data.load_velocity_spectrum(spectrum_dir, "fusion", method)
            if is_load_velocitySpec_success and user.data.check_if_velocity_spectrum_exists("fusion", method, track_id, rep_id):
                print(
                    f"[Skip] Velocity spectrum already exists for User {user.user_id}, Modality: {modality}, Method: {method}, Track ID: {track_id}, Rep ID: {rep_id}")
                continue
            with SuppressMCR():
                velocity_spectrum_ro = generator.run(modality_params, pos)
            print(
                f"[User {user.user_id} | Fusion-{method} | Track {track_id} | Rep {rep_id}] {method} extract done!")
            user.data.get_velocity_spectrum("fusion", method).set_processed_spectrum(track_id, rep_id, velocity_spectrum_ro,base_dir=spectrum_dir)

            del velocity_spectrum_ro
            # Save the spectrum file path record: when placed here, each sample saves once; it currently doesn't affect logic,
            # but it keeps overwriting, leaving only the last one
            user.save_velocity_spectrum(spectrum_dir, "fusion", method)
            print(f"[Info] Data processed and set for Track ID: {track_id}, Rep ID: {rep_id}")

    print(f"[Done] Data saved for User {user_id}, Modality: Fusion")
    return f"[Done] User {user_id} / Fusion"

def prepare_tasks(expCfg,target_scenario='Home'):
    """
        Wrap the task preparation logic

        :param scenarios: scenario configuration dictionary
        :param expCfg: global experiment configuration object
        :param load_all_users_func: reference to the user-loading function (load_all_users)
        :param target_scenario: the specific scenario name to process, default is 'Home'
        :return: task list [(scenario_name, user_id), ...]
        """
    scenarios = expCfg.get_all_scene_configs()

    tasks = []
    print(f"--- [Task Builder] Scanning for tasks in scenario: {target_scenario} ---")

    for scenario_name, scenario_cfg in scenarios.items():
        # 1. Filter scenarios
        if scenario_name != target_scenario:
            continue

        # 2. Update the scenario inside experiment config
        expCfg.set_scene(scenario_name)

        # 3. Build the data root path and load users
        users = load_all_users(
            os.path.join(expCfg.data_root_path, scenario_name),
            user_range=scenario_cfg.users_list
        )

        for user in users:
            tasks.append((scenario_name, user.user_id))

    print(f"--- [Task Builder] Total tasks identified: {len(tasks)} ---")
    return tasks

def linux_multiprocess():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    expCfg = ExperimentConfig()
    tasks = prepare_tasks(expCfg, target_scenario='Home')

    # 3. Only start if there are tasks
    if tasks:
        manager = multiprocessing.Manager()
        name_queue = manager.Queue()  # Create an empty queue first

        num_parallel = 1
        # [Change 1] Add a timestamp suffix so that even if the last run crashed, this run uses a fresh new name
        import time

        timestamp = int(time.time())
        engine_names = [f"MATLAB_ENG_{i}_{timestamp}" for i in range(num_parallel)]

        print(f"--- Starting up to {num_parallel} MATLAB Engines ---")
        active_engines = []

        import matlab.engine
        import sys
        from functools import partial

        # Start named engines serially
        for name in engine_names:
            print(f"--- [Launcher] Starting new MATLAB instance for {name} ---")
            try:
                # Start a process
                eng = matlab.engine.start_matlab("-nosplash -nodesktop")

                # Execute shared command synchronously
                print(f"--- [Launcher] Requesting share for {name} ---")
                eng.eval(f"matlab.engine.shareEngine('{name}')", nargout=0)

                # Verify visibility
                success = False
                for attempt in range(15):
                    if name in matlab.engine.find_matlab():
                        success = True
                        break
                    time.sleep(2)

                if success:
                    active_engines.append(eng)
                    # [Key change 2] Only put engine names into the Queue after they are truly started and shared successfully
                    name_queue.put(name)
                    print(f"--- [Success] Engine {name} is verified and ready ---")
                else:
                    print(f"--- [Fail] Engine {name} timed out sharing. Skipping. ---")
                    eng.quit()
            except Exception as e:
                print(f"--- [Error] Could not start/share {name}: {e} ---")
                continue

        # [Key change 3] Adjust pool size according to the number of engines actually started
        actual_count = len(active_engines)
        if actual_count == 0:
            print("No MATLAB engines could be started. Check License or Memory.")
            sys.exit(1)

        print(f"--- {actual_count} engines live. Starting Pool with {actual_count} workers ---")

        # Start process pool
        try:
            # Pool size must be equal to or less than the number of successfully started engines
            with multiprocessing.Pool(processes=actual_count) as pool:
                worker_func = partial(worker_wrapper, expCfg=expCfg, engine_name_queue=name_queue)
                for result in tqdm(pool.imap(worker_func, tasks), total=len(tasks)):
                    # You can decide whether to print detailed results as needed
                    pass
        finally:
            print("--- Cleaning up MATLAB Engines ---")
            for eng in active_engines:
                try:
                    eng.quit()
                except:
                    pass
            manager.shutdown()

# ========================================
#              Main process
# ========================================
if __name__ == "__main__":
    import platform

    if platform.system() == "Linux":
        print("--- Detected Linux OS: Running Multiprocess with MATLAB Engine ---")
        linux_multiprocess()
    else:
        print("--- Detected Windows OS: Running Standard/Test Mode ---")
        # win_test logic
        expCfg = ExperimentConfig()
        tasks = prepare_tasks(expCfg, target_scenario='Laboratory')

        if tasks:
            # Run a specific task as in the original code (e.g., index 1)
            # You can change this to a loop if you want to process all tasks on Windows
            task_idx = 1 if len(tasks) > 1 else 0
            print(f"--- Processing task index {task_idx} ---")
            process_user(tasks[task_idx], expCfg=expCfg)
