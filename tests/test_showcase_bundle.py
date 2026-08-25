from pathlib import Path
import subprocess
import sys

from scripts.build_showcase_bundle import build_showcase_bundle


def test_build_showcase_bundle_creates_replay_report_and_index(tmp_path):
    out = build_showcase_bundle(output_dir=tmp_path, generate_assets=False)

    assert (out / "index.md").exists()
    assert (out / "replay-report.md").exists()
    assert (out / "sample-session.json").exists()
    assert "Replay Report" in (out / "index.md").read_text(encoding="utf-8")


def test_showcase_bundle_script_runs_from_repo_root(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/build_showcase_bundle.py", "--output-dir", str(tmp_path), "--no-assets"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "index.md").exists()
