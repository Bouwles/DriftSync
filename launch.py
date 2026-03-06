"""
DriftSync Launcher
==================
Single entry point for the DriftSync application.

This script:
1. Checks for required Python packages.
2. Installs any missing ones automatically via pip.
3. Launches the full interactive Pygame GUI application.

Usage
-----
    python launch.py

No other setup required.
"""

import sys
import os
import subprocess
import importlib

# ---------------------------------------------------------------------------
# Ensure we're in the right directory (DriftSync project root)
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle — __file__ points to the temp
    # extraction folder.  Use the directory that contains the .exe instead.
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Required packages: (import_name, pip_package, description)
# ---------------------------------------------------------------------------
REQUIRED = [
    ("numpy",      "numpy>=1.24.0",       "NumPy  — numerical computing"),
    ("sklearn",    "scikit-learn>=1.3.0", "Scikit-learn — metrics & ML utilities"),
    ("matplotlib", "matplotlib>=3.7.0",   "Matplotlib — plot generation"),
    ("torch",      "torch",               "PyTorch — deep learning framework"),
    ("pygame",     "pygame>=2.5.0",       "Pygame — interactive GUI"),
]


def _pip_install(pip_spec: str, label: str) -> bool:
    """
    Run pip install for one package. Returns True on success.
    """
    print(f"    Installing {label} ...", flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_spec, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: Failed to install {pip_spec}")
        print(f"    {e.stderr.decode('utf-8', errors='replace').strip()}")
        return False


def ensure_dependencies() -> bool:
    """
    Check all required packages; install missing ones.

    Returns:
        True if all packages are available (or were successfully installed).
        False if any installation failed.
    """
    missing = []
    for import_name, pip_spec, label in REQUIRED:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_spec, label))

    if not missing:
        return True

    print()
    print("=" * 60)
    print("  DriftSync — First-time Setup")
    print("=" * 60)
    print(f"  {len(missing)} package(s) need to be installed:")
    for _, _, label in missing:
        print(f"    - {label}")
    print()

    if any("torch" in p for _, p, _ in missing):
        print("  NOTE: PyTorch may take several minutes to download.")
        print("  Please be patient...")
        print()

    all_ok = True
    for import_name, pip_spec, label in missing:
        ok = _pip_install(pip_spec, label)
        if not ok:
            all_ok = False

    if all_ok:
        print()
        print("  All packages installed successfully!")
        print("=" * 60)
        print()
    else:
        print()
        print("  Some packages failed to install.")
        print("  Try: pip install -r requirements.txt")
        print("=" * 60)

    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("  DriftSync: Real-Time Cognitive Drift Prediction")
    print()

    # When running as a PyInstaller bundle all packages are already included —
    # skip the pip installer completely (sys.executable is the .exe, not python).
    if not getattr(sys, "frozen", False):
        if not ensure_dependencies():
            print("Cannot start — dependency installation failed.")
            sys.exit(1)

    # All deps available — launch the application
    try:
        from driftsync.app.application import DriftSyncApplication
    except ImportError as e:
        print(f"Failed to import application: {e}")
        print("Make sure you are running from the DriftSync project directory.")
        sys.exit(1)

    app = DriftSyncApplication()
    app.run()


if __name__ == "__main__":
    main()
