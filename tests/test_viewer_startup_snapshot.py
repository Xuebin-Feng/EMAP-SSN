"""EMAPSSN_Viewer.py startup: --delete-settings consumes the per-launch settings
snapshot even when its settings are rejected, so a failed launch leaves no
ssn_viewer_*.json file in the temporary folder."""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER = ROOT / "src" / "EMAPSSN_Viewer.py"


def run_viewer(*arguments):
    environment = {key: value for key, value in os.environ.items() if key != "SSN_VIEWER_SETTINGS_PATH"}
    environment["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, str(VIEWER), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


class RejectedSnapshotTests(unittest.TestCase):
    def test_a_rejected_snapshot_is_still_deleted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = pathlib.Path(temp_dir) / "ssn_viewer_rejected.json"
            # Valid JSON, but not a viewer settings document.
            snapshot.write_text("{}", encoding="utf-8")
            result = run_viewer("--settings", str(snapshot), "--delete-settings")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("ViewerSettingsError", result.stderr)
            self.assertFalse(snapshot.exists())

    def test_without_delete_settings_a_rejected_file_is_kept(self):
        # A user's own settings file is never consumed.
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = pathlib.Path(temp_dir) / "my_settings.json"
            settings.write_text("{}", encoding="utf-8")
            result = run_viewer("--settings", str(settings))
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("ViewerSettingsError", result.stderr)
            self.assertTrue(settings.exists())

    def test_a_missing_snapshot_is_reported_not_masked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = pathlib.Path(temp_dir) / "ssn_viewer_missing.json"
            result = run_viewer("--settings", str(snapshot), "--delete-settings")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            # The settings error, not a FileNotFoundError from the cleanup.
            self.assertIn("ViewerSettingsError: settings_path", result.stderr)
            self.assertNotIn("During handling of the above exception", result.stderr)


if __name__ == "__main__":
    unittest.main()
