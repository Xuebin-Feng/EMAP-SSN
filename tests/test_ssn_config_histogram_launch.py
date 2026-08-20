# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import SSN_Config  # noqa: E402
from SSN_Utils import build_score_histogram_figure  # noqa: E402


class ScoreHistogramFigureTests(unittest.TestCase):
    def test_alignment_histogram_preserves_bins_threshold_and_title(self):
        scores = np.linspace(0.0, 1.0, 201)

        figure = build_score_histogram_figure(
            scores,
            0.625,
            is_evalue=False,
            norm_mode="shorter_sequence",
        )
        self.addCleanup(figure.clear)

        axes = figure.axes[0]
        self.assertEqual(len(axes.patches), 100)
        self.assertEqual(axes.get_title(), "Score Distribution (shorter_sequence)")
        self.assertEqual(list(axes.lines[0].get_xdata()), [0.625, 0.625])
        self.assertEqual(axes.lines[0].get_label(), "Threshold 0.625")

    def test_evalue_histogram_uses_evalue_title(self):
        figure = build_score_histogram_figure(
            np.asarray([1.0, 2.0, 3.0]),
            2.0,
            is_evalue=True,
            norm_mode="ignored",
        )
        self.addCleanup(figure.clear)

        self.assertEqual(figure.axes[0].get_title(), "Score Distribution (E-Value)")


class ViewerHandoffTests(unittest.TestCase):
    def test_macos_launches_viewer_in_terminal_without_replacing_config(self):
        process = object()
        env = {"SSN_TARGET_CACHE_MODE": "new", "SSN_VIEWER_SETTINGS_PATH": "/tmp/a.json"}
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            SSN_Config.subprocess, "Popen", return_value=process
        ) as popen:
            result = SSN_Config._handoff_to_viewer(
                temp_dir, env, platform_name="darwin", executable="/python"
            )

        self.assertIs(result, process)
        self.assertEqual(popen.call_args.args[0][0], "osascript")
        self.assertIn("SSN_VIEWER_SETTINGS_PATH", " ".join(popen.call_args.args[0]))
        self.assertIs(popen.call_args.kwargs["env"], env)

    def test_linux_launches_viewer_in_detected_terminal(self):
        process = object()
        env = {"SSN_TARGET_CACHE_MODE": "existing"}
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            SSN_Config.shutil, "which", side_effect=lambda name: "/terminal" if name == "gnome-terminal" else None
        ), mock.patch.object(SSN_Config.subprocess, "Popen", return_value=process) as popen:
            result = SSN_Config._handoff_to_viewer(
                temp_dir, env, platform_name="linux", executable="/python"
            )

        self.assertIs(result, process)
        self.assertEqual(popen.call_args.args[0][:3], ["gnome-terminal", "--", "bash"])
        self.assertIs(popen.call_args.kwargs["env"], env)

    def test_linux_reports_missing_terminal(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            SSN_Config.shutil, "which", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "No supported terminal emulator"):
                SSN_Config._handoff_to_viewer(
                    temp_dir, {}, platform_name="linux", executable="/python"
                )

    def test_windows_uses_new_console_popen(self):
        process = object()
        env = {"SSN_TARGET_CACHE_MODE": "existing"}
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            SSN_Config.subprocess,
            "Popen",
            return_value=process,
        ) as popen:
            result = SSN_Config._handoff_to_viewer(
                temp_dir,
                env,
                platform_name="win32",
                executable=r"C:\Python\python.exe",
            )

        self.assertIs(result, process)
        _, kwargs = popen.call_args
        self.assertEqual(kwargs["cwd"], os.path.abspath(temp_dir))
        self.assertIs(kwargs["env"], env)
        self.assertNotEqual(kwargs["creationflags"], 0)
        self.assertIn("src\\SSN_Viewer.py", popen.call_args.args[0])

    def test_settings_snapshots_are_unique_and_immutable_copies(self):
        first = SSN_Config._create_viewer_settings_snapshot({"SEQUENCE_SET": "first"})
        second = SSN_Config._create_viewer_settings_snapshot({"SEQUENCE_SET": "second"})
        self.addCleanup(lambda: os.path.exists(first) and os.unlink(first))
        self.addCleanup(lambda: os.path.exists(second) and os.unlink(second))

        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).read_text(encoding="utf-8").count("first"), 1)
        self.assertEqual(Path(second).read_text(encoding="utf-8").count("second"), 1)


class OffscreenConfigIntegrationTests(unittest.TestCase):
    def test_shared_status_tooltip_preserves_statistics_report(self):
        script = textwrap.dedent(
            f"""
            import os
            import pathlib
            import runpy
            import sys
            import tempfile
            from types import SimpleNamespace
            from unittest import mock

            import h5py
            import numpy as np

            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            root = pathlib.Path({str(PROJECT_ROOT)!r})
            src = root / "src"
            sys.path.insert(0, str(src))

            from utilities import Hardware_Utils  # preload torch before PySide6
            from PySide6.QtWidgets import QApplication
            test_app = QApplication.instance() or QApplication([])

            with mock.patch.object(QApplication, "exec", return_value=0), mock.patch.object(
                sys, "exit", return_value=None
            ):
                namespace = runpy.run_path(str(src / "SSN_Config.py"), run_name="__main__")

            app = namespace["app"]
            window = namespace["window"]

            with tempfile.TemporaryDirectory() as temp_dir:
                work = pathlib.Path(temp_dir)
                fasta_path = work / "subset.fasta"
                network_path = work / "network.h5"
                fasta_path.write_text(">a\\nAAAA\\n>b\\nAAAT\\n", encoding="utf-8")
                with h5py.File(network_path, "w") as hf:
                    hf.create_dataset("headers", data=[b"a", b"b"])
                    hf.create_dataset("score", data=np.asarray([4.0], dtype=np.float32))
                    hf.create_dataset("i", data=np.asarray([0], dtype=np.int64))
                    hf.create_dataset("j", data=np.asarray([1], dtype=np.int64))

                window.inputs["FASTA_DIR"].setText(str(work))
                window.inputs["HDF5_DIR"].setText(str(work))
                window.cb_fasta.clear()
                window.cb_fasta.addItem(fasta_path.name)
                window.cb_hdf5.clear()
                window.cb_hdf5.addItem(network_path.name)

                manifest = SimpleNamespace(network_type="blast", model_name="BLAST")
                method_globals = window.run_statistics.__globals__
                cache_manifest = method_globals["cache_manifest"]

                def validate_statistics(_hf):
                    assert "Computing network statistics" in window.tip_panel.text()
                    return manifest

                with mock.patch.object(
                    cache_manifest, "validate_network_schema", side_effect=validate_statistics
                ):
                    window.run_statistics()

                report = window.stat_display.toPlainText()
                assert "====== Network Statistics ======" in report
                assert "Network statistics computed successfully." in window.tip_panel.text()

                class FakeHistogramDialog:
                    def __init__(self, figure, _parent):
                        self.figure = figure

                    def exec(self):
                        assert "Displaying score histogram..." in window.tip_panel.text()
                        return 0

                    def release_figure(self):
                        self.figure.clear()

                    def deleteLater(self):
                        pass

                def validate_histogram(_hf):
                    assert "Computing score distribution" in window.tip_panel.text()
                    return manifest

                with mock.patch.object(
                    cache_manifest, "validate_network_schema", side_effect=validate_histogram
                ), mock.patch.dict(
                    method_globals, {{"ScoreHistogramDialog": FakeHistogramDialog}}
                ):
                    window.run_histogram()

                assert window.stat_display.toPlainText() == report
                assert "Histogram displayed successfully." in window.tip_panel.text()

                with mock.patch("h5py.File", side_effect=OSError("unreadable network")):
                    window.run_histogram()
                assert window.stat_display.toPlainText() == report
                assert "Error during histogram generation: unreadable network" in window.tip_panel.text()

                with mock.patch("h5py.File", side_effect=OSError("unreadable network")):
                    window.run_statistics()
                assert window.stat_display.toPlainText() == report
                assert (
                    "Error during network statistics calculation: unreadable network"
                    in window.tip_panel.text()
                )

            window.close()
            app.processEvents()
            print("SHARED_STATUS_TOOLTIP_OK")
            """
        )
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("SHARED_STATUS_TOOLTIP_OK", completed.stdout)

    def test_dialog_and_save_run_lifecycle(self):
        script = textwrap.dedent(
            f"""
            import os
            import pathlib
            import runpy
            import sys
            import tempfile
            from unittest import mock

            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            root = pathlib.Path({str(PROJECT_ROOT)!r})
            src = root / "src"
            sys.path.insert(0, str(src))

            from utilities import Hardware_Utils  # preload torch before PySide6
            from PySide6.QtCore import QTimer, qInstallMessageHandler
            from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
            from SSN_Utils import build_score_histogram_figure
            test_app = QApplication.instance() or QApplication([])

            with mock.patch.object(QApplication, "exec", return_value=0), mock.patch.object(
                sys, "exit", return_value=None
            ):
                namespace = runpy.run_path(str(src / "SSN_Config.py"), run_name="__main__")

            app = namespace["app"]
            window = namespace["window"]
            dialog_type = namespace["ScoreHistogramDialog"]

            with tempfile.TemporaryDirectory() as temp_dir:
                saved_layout_dir = pathlib.Path(temp_dir)
                cache_folder = saved_layout_dir / "compatible-layout"
                cache_folder.mkdir()
                window.inputs["SAVED_LAYOUT_DIR"].setText(str(saved_layout_dir))
                window.current_cache_folder = str(cache_folder)
                window._cache_launch_allowed = True
                window.check_umap.setChecked(True)
                window.cb_cache_file.clear()
                window.cb_cache_file.addItem("(New Layout Cache)", None)
                window.line_new_cache.setText("launch-test")

                method_globals = window.save_and_run.__globals__
                handoff = mock.Mock(return_value=object())
                with mock.patch.object(window, "save_settings", return_value=True), mock.patch.dict(
                    method_globals, {{"_handoff_to_viewer": handoff}}
                ), mock.patch.object(window, "close") as close:
                    window.save_and_run()
                    close.assert_not_called()
                    launch_env = handoff.call_args.args[1]
                    assert launch_env["SSN_TARGET_CACHE_MODE"] == "new"
                    assert launch_env["SSN_TARGET_CACHE_PATH"] == "compatible-layout/launch-test.h5"
                    snapshot = pathlib.Path(launch_env["SSN_VIEWER_SETTINGS_PATH"])
                    assert snapshot.is_file()
                    snapshot.unlink()

                failed_handoff = mock.Mock(side_effect=OSError("exec failed"))
                with mock.patch.object(window, "save_settings", return_value=True), mock.patch.dict(
                    method_globals, {{"_handoff_to_viewer": failed_handoff}}
                ), mock.patch.object(window, "close") as close, mock.patch.object(
                    QMessageBox, "critical"
                ) as critical:
                    window.save_and_run()
                    close.assert_not_called()
                    critical.assert_called_once()
                    assert "exec failed" in critical.call_args.args[2]

            messages = []
            previous_handler = qInstallMessageHandler(
                lambda mode, context, message: messages.append(message)
            )
            dialog_result = []

            def open_histogram():
                figure = build_score_histogram_figure(
                    [0.1, 0.2, 0.3],
                    0.2,
                    is_evalue=False,
                    norm_mode="alignment_length",
                )
                dialog = dialog_type(figure, window)
                QTimer.singleShot(0, dialog.accept)
                dialog_result.append(dialog.exec())
                dialog.release_figure()
                dialog.deleteLater()
                app.quit()

            try:
                QTimer.singleShot(0, open_histogram)
                app.exec()
            finally:
                qInstallMessageHandler(previous_handler)

            assert dialog_result == [QDialog.DialogCode.Accepted]
            assert not any("event loop is already running" in message.lower() for message in messages), messages

            window.close()
            app.processEvents()
            print("OFFSCREEN_CONFIG_OK")
            """
        )
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("OFFSCREEN_CONFIG_OK", completed.stdout)
        self.assertNotIn("event loop is already running", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
