"""Build and verify a portable Windows ZIP: python scripts/build_windows_release.py.

Requires PyInstaller in the build environment. Artifacts stay in ignored dist/.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driftsync import __version__


def main():
    if sys.platform != "win32":
        raise SystemExit("Build this Windows distribution on Windows.")
    version = __version__
    staging = ROOT / "dist" / f"v{version}"
    work = ROOT / "build" / f"release-v{version}"
    # Explicit fixed descendants; PyInstaller may replace only this release staging.
    for path in (staging, work):
        if not path.resolve().is_relative_to(ROOT):
            raise RuntimeError("Build path escaped the repository")
        path.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(staging), "--workpath", str(work), "DriftSync.spec"],
                   cwd=ROOT, check=True)
    app_dir = staging / "DriftSync"
    results = app_dir / "driftsync" / "results"
    results.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "driftsync/results/experiment_summary.json", results)
    shutil.copy2(ROOT / "LICENSE", app_dir)
    (app_dir / "START_HERE.txt").write_text(
        f"DriftSync {version} - Windows x64\n\n"
        "Extract the ENTIRE ZIP into a writable folder, then run DriftSync.exe.\n"
        "Keep _internal next to the executable. Python is not required.\n"
        "Open Mathematics in the left navigation (Alt+7) for interactive equations.\n"
        "This package contains no personal sessions or pre-trained checkpoints.\n"
        "Use Train models before live predictions; task recording works immediately.\n"
        "Saved data stays in the extracted folder. Keep older recordings when upgrading.\n",
        encoding="utf-8")
    report = work / "packaged-check.json"
    subprocess.run([str(app_dir / "DriftSync.exe"), "--self-test", str(report)],
                   cwd=app_dir, check=True, timeout=180)
    data = json.loads(report.read_text(encoding="utf-8"))
    if not data.get("frozen") or data.get("version") != version or data.get("status") != "passed":
        raise RuntimeError("Packaged executable verification failed")
    archive = ROOT / "dist" / f"DriftSync-v{version}-windows-x64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(f"Verified release: {archive}\nSHA256: {digest}", flush=True)


if __name__ == "__main__":
    main()
