import multiprocessing
import os
import sys
import time
from functools import partial
from multiprocessing import Pool, Manager

from tqdm import tqdm

from src.Apps.FusionGait.GaitRecognition.GaitExtraction.utils import load_all_users, process_user_modality
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
        result = process_user_modality(task_info, expCfg=expCfg)
        return result
    finally:
        # 4. Put the engine name back for the next task to use
        engine_name_queue.put(engine_name)


def prepare_tasks(expCfg,target_scenario='Home'):
    """
    Wrap the task preparation logic

    :param scenarios: scenario configuration dictionary
    :param expCfg: global experiment configuration object
    :param load_all_users_func: reference to the user-loading function (load_all_users)
    :param target_scenario: the specific scenario name to process, default is 'Home'
    :return: task list [(scenario_name, user_id, modality), ...]
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
        scene_data_path = os.path.join(expCfg.data_root_path, scenario_name)
        if not os.path.exists(scene_data_path):
            print(f"[Warning] Scene path not found: {scene_data_path}")
            continue

        users = load_all_users(
            scene_data_path,
            user_range=scenario_cfg.users_list
        )

        # 4. Iterate users and modalities, extract task tuples
        for user in users:
            # Get all base modalities under this user (e.g., 'wifi', 'acoustic')
            modalities = user.data.get_base_modalities()
            for modality in modalities:
                # Each task is a minimal execution unit parameter package
                tasks.append((scenario_name, user.user_id, modality))

    print(f"--- [Task Builder] Total tasks identified: {len(tasks)} ---")
    return tasks


def linux_multiprocess():
    # 0. Force-kill leftovers (optional, recommended to keep)
    # os.system("pkill -9 -f matlab")

    # 1. Must set start method first
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 2. Prepare tasks
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
                    # You can decide whether to print detailed result as needed
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
            process_user_modality(tasks[task_idx], expCfg=expCfg)
