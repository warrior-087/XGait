# To compile MATLAB scripts, you must first install the MATLAB Engine for Python.
# Linux:
#   1) Locate the `engine` folder under your MATLAB installation, e.g.:
#        <MATLABROOT>/extern/engines/python
#   2) Activate your conda environment:
#        conda activate mypyenv
#   3) Install the engine into that environment:
#        python setup.py install
#
# NOTE:
# - This step is required only for the MATLAB Engine workflow (not for MCR).
# - Make sure the `python` used here is the one from the target environment.

import matlab.engine
import platform
import os
from pathlib import Path


def build_unified_package(code_root_dir, matlab_scripts, package_name, out_dir):
    """
    Compiles multiple MATLAB scripts into a single unified Python package
    while automatically handling path dependencies.
    """
    print("Starting MATLAB Engine (this may take a few seconds)...")
    eng = matlab.engine.start_matlab()

    try:
        # 1. Add the code root and all its subdirectories to the MATLAB search path.
        # This ensures the compiler can find all helper functions and dependencies.
        print(f"Adding search paths: {code_root_dir}")
        full_path = eng.genpath(str(code_root_dir))
        eng.addpath(full_path, nargout=0)

        # 2. Filter out non-existent entry point files.
        valid_scripts = [str(s) for s in matlab_scripts if s.exists()]
        if not valid_scripts:
            print("[Error] No valid .m files found for compilation.")
            return

        print(f"\n--- Starting Unified Compilation: {package_name} ---")
        print(f"Number of API functions: {len(valid_scripts)}")
        print(f"Output directory: {out_dir}")

        # Ensure the output directory exists.
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3. Construct the MATLAB compilation command.
        # Passing multiple files as a cell array: {'file1.m', 'file2.m', ...}
        matlab_list = "{" + ", ".join([f"'{s}'" for s in valid_scripts]) + "}"

        # Build the command string.
        cmd = (
            f"compiler.build.pythonPackage({matlab_list}, "
            f"'PackageName', '{package_name}', "
            f"'OutputDir', '{str(out_dir)}')"
        )

        # 4. Execute the build.
        print("MATLAB compiler is analyzing dependencies and building the package, please wait...")
        eng.eval(cmd, nargout=0)
        print(f"\n[Success] Unified package '{package_name}' built successfully.")

    except Exception as e:
        print(f"\n[Error] An exception occurred during compilation: {e}")
    finally:
        # 5. Shut down the engine.
        eng.quit()
        print("MATLAB Engine closed.")


if __name__ == "__main__":
    # Resolve the current directory of this script.
    cur_dir = Path(__file__).resolve().parent

    # Root directory containing all MATLAB source code folders.
    code_root = cur_dir / "code"

    # Define entry-point scripts to be exported as Python APIs.
    matlab_scripts = [
        code_root / "TFA_Wifi" / "timeFrequencyAnalysisWiFi.m",
        code_root / "TFA_Acoustic" / "timeFrequencyAnalysisAcoustic.m",
        code_root / "GBVP" / "segmentOptimizeGBVP.m",
        code_root / "FusionGBVP" / "segmentOptimizeGBVP_Fusion.m",
        code_root / "SuccessiveTrack" / "successiveTrack.m",
        code_root / "FusionSuccessiveTrack" / "fusionSuccessiveTrack.m",
    ]

    # Name of the final Python package to be imported.
    package_name = "fs_gait"

    # Determine output path based on the operating system.
    system = platform.system()
    if system == "Windows":
        out_dir = cur_dir / "build_py" / "fs_gait_win"
    else:
        out_dir = cur_dir / "build_py" / "fs_gait_linux"

    build_unified_package(code_root, matlab_scripts, package_name, out_dir)
