import os
import platform
from pathlib import Path

def init_matlab_runtime_path():
    """
    Initialize MATLAB Runtime using environment variables only.

    Required env (either one):
      - MATLAB_RUNTIME_ROOT : e.g. D:\\SoftwareInstall\\Matlab2025a\\MatlabRunTime\\R2025a
        (recommended)
      - MATLAB_RUNTIME_DIRS : semicolon-separated full override of DLL dirs
        e.g. <root>\\runtime\\win64;<root>\\bin\\win64;<root>\\sys\\os\\win64

    Notes (Windows):
      - We sanitize PATH to remove full MATLAB install paths and other MCR versions.
      - We also call os.add_dll_directory for robust DLL resolution (Py>=3.8).
    """
    system = platform.system()

    # -------- 1) Read env --------
    runtime_root = os.environ.get("MATLAB_RUNTIME_ROOT", "").strip().strip('"')
    override_dirs = os.environ.get("MATLAB_RUNTIME_DIRS", "").strip().strip('"')

    if not runtime_root and not override_dirs:
        raise RuntimeError(
            "Missing environment variable. Please set either:\n"
            "  - MATLAB_RUNTIME_ROOT (recommended), or\n"
            "  - MATLAB_RUNTIME_DIRS (full override)\n"
        )

    # -------- 2) Build runtime dirs --------
    if override_dirs:
        # user fully controls dirs
        runtime_dirs = [p.strip().strip('"') for p in override_dirs.split(";") if p.strip()]
    else:
        if system == "Linux":
            runtime_dirs = [
                os.path.join(runtime_root, "runtime", "glnxa64"),
                os.path.join(runtime_root, "bin", "glnxa64"),
                os.path.join(runtime_root, "sys", "os", "glnxa64"),
            ]
        elif system == "Windows":
            runtime_dirs = [
                os.path.join(runtime_root, "runtime", "win64"),
                os.path.join(runtime_root, "bin", "win64"),
                os.path.join(runtime_root, "sys", "os", "win64"),
            ]
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

    # normalize slashes
    runtime_dirs = [os.path.normpath(p) for p in runtime_dirs]

    # -------- 3) Apply env --------
    if system == "Linux":
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(runtime_dirs) + (":" + ld if ld else "")
        print(f"[MATLAB Runtime] LD_LIBRARY_PATH set to:\n{os.environ['LD_LIBRARY_PATH']}")
        return

    if system != "Windows":
        return

    # -------- 4) Windows: sanitize PATH to avoid MATLAB(full) pollution --------
    # Keep everything that is NOT a MATLAB path, except keep the chosen runtime_root paths.
    def is_matlab_path(p: str) -> bool:
        pl = p.lower()
        return ("\\matlab" in pl) or ("/matlab" in pl)

    def is_under_runtime_root(p: str) -> bool:
        if not runtime_root:
            return False
        rr = os.path.normpath(runtime_root).lower()
        return os.path.normpath(p).lower().startswith(rr)

    old_parts = [p for p in os.environ.get("PATH", "").split(";") if p.strip()]
    cleaned_parts = []
    for p in old_parts:
        pn = os.path.normpath(p)
        if is_matlab_path(pn) and not is_under_runtime_root(pn):
            # drop full MATLAB install paths and other versions
            continue
        cleaned_parts.append(pn)

    # prepend our runtime dirs (highest priority)
    new_parts = runtime_dirs + cleaned_parts
    # de-dup while keeping order
    seen = set()
    uniq_parts = []
    for p in new_parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            uniq_parts.append(p)

    os.environ["PATH"] = ";".join(uniq_parts)

    # -------- 5) Robust DLL search path (Python 3.8+ on Windows) --------
    # This prevents "wrong DLL picked from PATH" issues.
    try:
        for d in runtime_dirs:
            if os.path.isdir(d):
                os.add_dll_directory(d)
    except Exception as e:
        # not fatal, but helpful to know
        print(f"[MATLAB Runtime] os.add_dll_directory skipped: {e}")

    print(f"[MATLAB Runtime] PATH set to:\n{os.environ['PATH']}")



# ------------------------------
# 使用示例：
# ------------------------------

if __name__ == "__main__":
    # 初始化 MATLAB Runtime
    init_matlab_runtime_path()

    import os, sys

    print("MATLAB_RUNTIME_ROOT =", os.environ.get("MATLAB_RUNTIME_ROOT"))
    print("PATH head =", os.environ.get("PATH", "")[:300])
    print("matlab.engine loaded?", "matlab.engine" in sys.modules)

    # 导入 MATLAB 编译包
    try:
        import fs_gait
        h = fs_gait.initialize()
        print("MATLAB package imported successfully!")
    except Exception as e:
        print("Failed to import MATLAB package:", e)
