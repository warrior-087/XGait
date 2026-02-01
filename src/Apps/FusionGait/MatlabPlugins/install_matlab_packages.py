import platform
import sys
import subprocess
from pathlib import Path


def install_matlab_packages(dirs):
    """Install MATLAB-generated Python packages from the given directories using `pip install .`."""

    for d in dirs:
        d = Path(d).expanduser().resolve()
        if not d.exists():
            print(f"[WARN] Directory not found: {d}")
            continue

        print(f"\n=== Installing package from: {d} ===")

        # Use the same Python interpreter that is running this script.
        # This ensures the package is installed into the active environment (venv/conda).
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-build-isolation",
            "--no-deps",
            str(d),
        ]

        try:
            subprocess.check_call(cmd)
            print(f"[OK] Installed package from: {d}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed installing package from {d}")
            print(e)
            continue


if __name__ == "__main__":
    system = platform.system()
    script_dir = Path(__file__).resolve().parent

    if system == "Windows":
        matlab_pkg_dirs = [
            script_dir / "build_py" / "fs_gait_win",
        ]
    elif system == "Linux":
        matlab_pkg_dirs = [
            script_dir / "build_py" / "fs_gait_linux",
        ]
    else:
        # This platform is not supported yet.
        raise RuntimeError(f"Unsupported platform: {system}")

    install_matlab_packages(matlab_pkg_dirs)
