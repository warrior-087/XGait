import site
import sys
import os
import platform
import threading
import copy
import numpy as np

def _pick_site_packages():
    """
    Pick a site-packages path for the CURRENT interpreter (venv/conda).
    Priority:
      1) site.getsitepackages()
      2) site.getusersitepackages()
      3) any sys.path entry ending with 'site-packages'
    """
    cands = []

    try:
        cands.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        cands.append(site.getusersitepackages())
    except Exception:
        pass

    cands.extend([p for p in sys.path if isinstance(p, str) and p.endswith("site-packages")])

    # keep existing dirs only, de-dup while preserving order
    seen = set()
    out = []
    for p in cands:
        if not p:
            continue
        p = os.path.normpath(p)
        if p in seen:
            continue
        if os.path.isdir(p):
            out.append(p)
            seen.add(p)
    return out[0] if out else None

def bootstrap_matlab_path():
    """Unified path logic: ensure Python only loads the MATLAB engine from the current environment"""
    # Clear old records
    for mod in list(sys.modules.keys()):
        if mod == "matlab" or mod.startswith("matlab."):
            del sys.modules[mod]

    original_path = sys.path[:]

    # Mask MCR / packaged matlab dist paths to avoid interference
    sys.path = [
        p for p in sys.path
        if isinstance(p, str)
        and ("MatlabRunTime" not in p)
        and ("matlab_mod_dist" not in p)
    ]

    # Option A (recommended): do NOT hardcode site-packages.
    # Just make sure CURRENT interpreter's site-packages is on top.
    site_pkg = _pick_site_packages()
    if site_pkg and site_pkg not in sys.path:
        sys.path.insert(0, site_pkg)

    try:
        # Key: perform the real import here
        import matlab.engine
        import matlab
        return matlab, matlab.engine
    finally:
        # Restore paths for other modules to use (append missing ones back)
        sys.path.extend([p for p in original_path if p not in sys.path])

# 2. Run initialization and get module references
# This way, matlab and matlab.engine are already available when the class is defined
# matlab, matlab_engine = bootstrap_matlab_path()

class MatlabEngineManager:
    _engine = None
    _engine_name = None  # Record the engine name bound to the current process

    @classmethod
    def connect_to_named_engine(cls, engine_name):
        """Called in subprocess: connect to the specified named engine"""
        if cls._engine is None:
            if cls._engine is None:
                import matlab.engine
                # For debugging: list engines visible to the subprocess
                available = matlab.engine.find_matlab()
                print(f"--- [PID {os.getpid()}] Available engines: {available} ---")

                if engine_name not in available:
                    # If not found, give it another chance; the main process may have just started it
                    import time
                    time.sleep(3)

                cls._engine = matlab.engine.connect_matlab(engine_name)
            cls._engine_name = engine_name
            # Also need to add paths
            code_root = os.path.join(os.path.dirname(__file__) , "code")
            cls._engine.addpath(cls._engine.genpath(code_root), nargout=0)
        return cls._engine

    @classmethod
    def call(cls, func_name, *args, **kwargs):
        if cls._engine is None:
            raise RuntimeError("Engine not connected. Call connect_to_named_engine first.")

        import matlab
        import copy
        import numpy as np

        converted_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                converted_args.append(matlab.double(arg.tolist()))
            else:
                converted_args.append(copy.deepcopy(arg))

        func = getattr(cls._engine, func_name)
        return func(*converted_args, **kwargs)


## On Windows, MCR runs fine; on Linux there may be underlying memory issues
class MatlabRuntimeManager:
    # Store the current process ID to detect whether a fork has happened
    _owner_pid = None
    _runtime = None
    _lock = threading.Lock()

    @classmethod
    def get_runtime(cls):
        # Get the current process PID
        current_pid = os.getpid()

        # If MCR has not been initialized in this process, or a fork occurred (PID changed)
        if cls._runtime is None or cls._owner_pid != current_pid:
            with cls._lock:
                # Double-check
                if cls._runtime is None or cls._owner_pid != current_pid:
                    print(f"--- Initializing MCR in Process [PID: {current_pid}] ---")

                    # Environment-variable patch needed only on Linux
                    if os.name != 'nt':
                        os.environ['MKL_THREADING_LAYER'] = 'GNU'

                    # Initialize a process-local MCR instance
                    from src.Apps.FusionGait.MatlabPlugins.init_matlab_path import init_matlab_runtime_path
                    init_matlab_runtime_path()
                    import fs_gait
                    cls._runtime = fs_gait.initialize()
                    cls._owner_pid = current_pid

        return cls._runtime

    @classmethod
    def call(cls, func_name, *args, **kwargs):
        # Ensure we get the process-local singleton
        runtime = cls.get_runtime()
        # MCR is not thread-safe internally; keeping a call lock is still recommended
        with cls._lock:
            func = getattr(runtime, func_name)
            return func(*args, **kwargs)

# On Linux, directly calling the MCR (Matlab Runtime) often leads to index length exceeded errors or memory issues.
# In contrast, using the Matlab Engine API to interface with a running Matlab instance is more stable on Linux.
# On Windows, however, MCR works smoothly and is more convenient as it doesn't require launching a separate Matlab process.
# Therefore, we dynamically select the backend manager based on the OS:
# - Windows -> MatlabRuntimeManager (MCR)
# - Linux   -> MatlabEngineManager (Engine API)

# Select the appropriate manager based on the OS
if platform.system() == "Windows":
    # On Windows, using the Runtime (MCR) is convenient and stable
    MatlabManager = MatlabRuntimeManager
else:
    # On Linux, direct MCR usage often causes memory/indexing errors.
    # Therefore, we use the Engine mode (requires starting MATLAB or connecting to shared engine).
    MatlabManager = MatlabEngineManager
