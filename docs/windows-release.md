# Building the Windows release

Build on Windows x64 with Python, the project dependencies and PyInstaller installed:

```powershell
python -m pytest
python scripts/build_windows_release.py
```

The script reads `driftsync.__version__`, builds into `dist/vVERSION/DriftSync`, runs
the frozen executable's self-test, and creates `dist/DriftSync-vVERSION-windows-x64.zip`
plus its `.zip.sha256` checksum. Keep the version in `pyproject.toml` aligned.

The PyInstaller spec collects application code and dependency data through their
hooks. It does not copy the working `driftsync/` directory. Only the checked-in
synthetic experiment summary is included as application data; local recordings,
calibration, prediction logs and trained checkpoints stay outside the package.

The executable's `--self-test REPORT.json` mode renders every workspace screen and
all Mathematics topics, checks the feature contract and executes both neural models
on CPU. Its report must identify the expected version and `frozen: true` before the
builder creates the ZIP. For source verification, use:

```powershell
python launch.py --self-test build/source-release-check.json
```

Users must extract the whole ZIP into a writable folder. The executable stores data
relative to that folder and needs the accompanying `_internal` directory. Python is
bundled, but pretrained model checkpoints are not. The in-app Train models action
creates checkpoints that live mode can discover automatically.
