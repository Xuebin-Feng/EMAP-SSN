# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
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

import EMAPSSN_Config  # noqa: E402
from EMAPSSN_Config import build_score_histogram_figure  # noqa: E402


def _run_config_script(script, *args):
    """Run an offscreen Configuration-window script in a fresh interpreter.

    The scripts open the window through tests.config_gui_loader, so it reads
    temporary directories; SSN_VIEWER_SETTINGS_PATH names a missing file, so
    EMAPSSN_Config's SETTINGS_FILE is not the developer's viewer_settings.json.
    """
    with tempfile.TemporaryDirectory() as settings_dir:
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["SSN_VIEWER_SETTINGS_PATH"] = os.path.join(settings_dir, "missing.json")
        return subprocess.run(
            [sys.executable, "-c", script, *args],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )


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

    def test_the_title_and_legend_take_the_font_families_given(self):
        families = ["sans-serif", "Noto Sans SC"]
        for given, expected in ((None, ["sans-serif"]), (families, families)):
            with self.subTest(font_families=given):
                figure = build_score_histogram_figure(
                    np.asarray([1.0, 2.0, 3.0]), 2.0, is_evalue=True, norm_mode="ignored",
                    font_families=given,
                )
                self.addCleanup(figure.clear)
                axes = figure.axes[0]
                self.assertEqual(axes.title.get_fontfamily(), expected)
                self.assertEqual(axes.get_legend().get_texts()[0].get_fontfamily(), expected)


class ViewerHandoffTests(unittest.TestCase):
    def test_handoff_uses_shared_terminal_launcher_contract(self):
        process = object()
        env = {
            "SSN_TARGET_CACHE_MODE": "new",
            "SSN_VIEWER_SETTINGS_PATH": "/tmp/a.json",
        }
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            EMAPSSN_Config, "launch_in_terminal", return_value=process
        ) as launch:
            result = EMAPSSN_Config._handoff_to_viewer(
                temp_dir, env, platform_name="darwin", executable="/python"
            )

        self.assertIs(result, process)
        command = launch.call_args.args[0]
        self.assertEqual(command[0:2], ["/python", "-u"])
        self.assertEqual(
            command[2],
            os.path.join(os.path.abspath(temp_dir), "src", "EMAPSSN_Viewer.py"),
        )
        self.assertEqual(launch.call_args.kwargs["cwd"], os.path.abspath(temp_dir))
        self.assertIs(launch.call_args.kwargs["env"], env)
        self.assertEqual(
            launch.call_args.kwargs["hold"], EMAPSSN_Config.HoldMode.ON_ERROR
        )
        self.assertEqual(
            launch.call_args.kwargs["title"], EMAPSSN_Config.VIEWER_DISPLAY_NAME
        )
        self.assertEqual(launch.call_args.kwargs["platform_name"], "darwin")

    def test_settings_snapshots_are_unique_and_immutable_copies(self):
        first = EMAPSSN_Config._create_viewer_settings_snapshot({"NODE_FASTA_FILE": "first.fasta"})
        second = EMAPSSN_Config._create_viewer_settings_snapshot({"NODE_FASTA_FILE": "second.fasta"})
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
            import sys
            import tempfile
            import time
            from types import SimpleNamespace
            from unittest import mock

            import h5py
            import numpy as np

            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            root = pathlib.Path({str(PROJECT_ROOT)!r})
            src = root / "src"
            sys.path.insert(0, str(src))
            sys.path.insert(0, str(root))

            from utilities import Hardware_Acceleration as Hardware_Utils  # preload torch before PySide6
            from PySide6.QtWidgets import QApplication
            from tests.config_gui_loader import load_config_namespace, open_config_window
            app = QApplication.instance() or QApplication([])

            # A window over temporary directories, not the developer's viewer_settings.json.
            namespace = load_config_namespace()
            home = tempfile.TemporaryDirectory()
            window = open_config_window(namespace["ConfigGUI"], home.name)

            def settle_cache_discovery():
                # Selecting the inputs starts background input hashing, which also
                # validates the network; it must finish before that is patched below.
                deadline = time.monotonic() + 60
                while window._cache_hash_pending_keys is not None:
                    assert time.monotonic() < deadline, "cache discovery did not finish"
                    app.processEvents()
                    time.sleep(0.01)

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
                settle_cache_discovery()

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
                assert "Stored Edges:" in report
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
            home.cleanup()
            print("SHARED_STATUS_TOOLTIP_OK")
            """
        )
        completed = _run_config_script(script)

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("SHARED_STATUS_TOOLTIP_OK", completed.stdout)

    def test_statistics_read_the_length_column_only_to_divide_by_it(self):
        # The project root arrives as argv[1], so the script needs no f-string escaping.
        script = textwrap.dedent(
            """
            import os
            import pathlib
            import sys
            import tempfile
            from unittest import mock

            import h5py
            import numpy as np

            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            root = pathlib.Path(sys.argv[1])
            src = root / "src"
            sys.path.insert(0, str(src))
            sys.path.insert(0, str(root))

            from utilities import Hardware_Acceleration as Hardware_Utils  # preload torch before PySide6
            from PySide6.QtWidgets import QApplication
            from tests.config_gui_loader import load_config_namespace, open_config_window
            app = QApplication.instance() or QApplication([])

            # A window over temporary directories, not the developer's viewer_settings.json.
            namespace = load_config_namespace()
            home = tempfile.TemporaryDirectory()
            window = open_config_window(namespace["ConfigGUI"], home.name)
            real_file = h5py.File
            indexed = set()

            class RecordingFile:
                # An open network that records which datasets the window reads.
                def __init__(self, *args, **kwargs):
                    self._file = real_file(*args, **kwargs)
                    self.attrs = self._file.attrs

                def __enter__(self):
                    return self

                def __exit__(self, *exc_info):
                    self._file.close()

                def __contains__(self, name):
                    return name in self._file

                def __getitem__(self, name):
                    indexed.add(name)
                    return self._file[name]

            class FakeHistogramDialog:
                def __init__(self, figure, _parent):
                    self.figure = figure

                def exec(self):
                    return 0

                def release_figure(self):
                    self.figure.clear()

                def deleteLater(self):
                    pass

            with tempfile.TemporaryDirectory() as temp_dir:
                work = pathlib.Path(temp_dir)
                fasta_path = work / "subset.fasta"
                network_path = work / "network.h5"
                fasta_path.write_text(">a\\nAAAA\\n>b\\nAAAT\\n>c\\nAATT\\n", encoding="utf-8")
                with h5py.File(network_path, "w") as hf:
                    hf.attrs["model_name"] = "E1_RA"
                    hf.create_dataset("headers", data=[b"a", b"b", b"c"])
                    hf.create_dataset("seq_lens", data=np.asarray([4, 4, 4], dtype=np.uint16))
                    hf.create_dataset("i", data=np.asarray([0, 0, 1], dtype=np.uint16))
                    hf.create_dataset("j", data=np.asarray([1, 2, 2], dtype=np.uint16))
                    for name in ("g_score", "l_score"):
                        hf.create_dataset(name, data=np.asarray([8.0, 6.0, 4.0], dtype=np.float32))
                    for name in ("g_len", "l_len"):
                        hf.create_dataset(name, data=np.asarray([4, 4, 4], dtype=np.uint16))

                window.inputs["FASTA_DIR"].setText(str(work))
                window.inputs["HDF5_DIR"].setText(str(work))
                window.cb_fasta.clear()
                window.cb_fasta.addItem(fasta_path.name)
                window.cb_hdf5.clear()
                window.cb_hdf5.addItem(network_path.name)
                method_globals = window.run_histogram.__globals__

                # global/alignment_length is the positive control for the recorder.
                expected = {
                    ("local", "longer_sequence"): set(),
                    ("global", "shorter_sequence"): set(),
                    ("global", "alignment_length"): {"g_len"},
                }
                for (score, normalization), lengths in expected.items():
                    window.cb_score_mode.setCurrentText(score)
                    window.cb_norm_mode.setCurrentText(normalization)
                    assert window.cb_norm_mode.currentText() == normalization, normalization
                    for run in (window.run_statistics, window.run_histogram):
                        indexed.clear()
                        with mock.patch("h5py.File", RecordingFile), mock.patch.dict(
                            method_globals, {"ScoreHistogramDialog": FakeHistogramDialog}
                        ):
                            run()
                        context = (score, normalization, run.__name__)
                        assert "successfully" in window.tip_panel.text(), (context, window.tip_panel.text())
                        assert indexed & {"g_len", "l_len"} == lengths, (context, sorted(indexed))

            window.close()
            app.processEvents()
            home.cleanup()
            print("LENGTH_COLUMNS_OK")
            """
        )
        completed = _run_config_script(script, str(PROJECT_ROOT))

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("LENGTH_COLUMNS_OK", completed.stdout)

    def test_dialog_and_save_run_lifecycle(self):
        script = textwrap.dedent(
            f"""
            import os
            import pathlib
            import sys
            import tempfile
            import time
            from unittest import mock

            import h5py
            import numpy as np

            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            root = pathlib.Path({str(PROJECT_ROOT)!r})
            src = root / "src"
            sys.path.insert(0, str(src))
            sys.path.insert(0, str(root))

            from utilities import Hardware_Acceleration as Hardware_Utils  # preload torch before PySide6
            from PySide6.QtCore import QTimer, qInstallMessageHandler
            from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
            from EMAPSSN_Config import build_score_histogram_figure
            from tests.config_gui_loader import load_config_namespace, open_config_window
            app = QApplication.instance() or QApplication([])

            # A window over temporary directories, not the developer's viewer_settings.json.
            namespace = load_config_namespace()
            home = tempfile.TemporaryDirectory()
            window = open_config_window(namespace["ConfigGUI"], home.name)
            dialog_type = namespace["ScoreHistogramDialog"]

            def settle_cache_discovery():
                # Background input hashing must finish first, or its late
                # result would replace the cache state set by hand below.
                deadline = time.monotonic() + 60
                while window._cache_hash_pending_keys is not None:
                    assert time.monotonic() < deadline, "cache discovery did not finish"
                    app.processEvents()
                    time.sleep(0.01)

            with tempfile.TemporaryDirectory() as temp_dir:
                saved_layout_dir = pathlib.Path(temp_dir)
                cache_folder = saved_layout_dir / "compatible-layout"
                cache_folder.mkdir()
                # Own inputs instead of whatever viewer_settings.json selects; a
                # fresh clone selects none and layout settings need both paths.
                inputs = saved_layout_dir / "inputs"
                inputs.mkdir()
                fasta_path = inputs / "set.fasta"
                fasta_path.write_text(">a\\nAAAA\\n>b\\nAAAT\\n", encoding="utf-8")
                network_path = inputs / "network.h5"
                with h5py.File(network_path, "w") as hf:
                    hf.attrs["model_name"] = "test_model"
                    hf.create_dataset("headers", data=[b"a", b"b"])
                    hf.create_dataset("seq_lens", data=np.asarray([4, 4], dtype=np.uint16))
                    for name, values in (("i", [0]), ("j", [1]), ("g_len", [4]), ("l_len", [4])):
                        hf.create_dataset(name, data=np.asarray(values, dtype=np.uint16))
                    for name in ("g_score", "l_score"):
                        hf.create_dataset(name, data=np.asarray([3.0], dtype=np.float32))
                window.inputs["SAVED_LAYOUT_DIR"].setText(str(saved_layout_dir))
                window.inputs["FASTA_DIR"].setText(str(inputs))
                window.inputs["HDF5_DIR"].setText(str(inputs))
                window.cb_fasta.clear()
                window.cb_fasta.addItem(fasta_path.name)
                window.cb_hdf5.clear()
                window.cb_hdf5.addItem(network_path.name)
                # Toggling UMAP re-resolves the cache folder, so the manual
                # cache state must come after the toggle and the discovery.
                window.check_umap.setChecked(True)
                settle_cache_discovery()
                window.current_cache_folder = str(cache_folder)
                window._cache_launch_allowed = True
                window.cb_cache_file.clear()
                window.cb_cache_file.addItem("(New Layout Cache)", None)
                window.line_new_cache.setText("launch-test")

                method_globals = window.save_and_run.__globals__
                generator_handoff = mock.Mock(return_value=object())
                with mock.patch.object(window, "save_settings", return_value=True), mock.patch.dict(
                    method_globals, {{"_handoff_to_layout_generator": generator_handoff}}
                ), mock.patch.object(window, "close") as close, mock.patch.object(
                    QMessageBox, "critical"
                ) as critical:
                    window.save_and_run()
                    # Offscreen, a real modal error box would block until the
                    # subprocess timeout; report its text instead.
                    assert not critical.called, critical.call_args
                    close.assert_not_called()
                    generator_handoff.assert_called_once()
                    launch_env = generator_handoff.call_args.args[2]
                    assert launch_env["SSN_TARGET_CACHE_MODE"] == "new"
                    assert launch_env["SSN_TARGET_CACHE_PATH"] == "compatible-layout/launch-test.h5"
                    snapshot = pathlib.Path(launch_env["SSN_VIEWER_SETTINGS_PATH"])
                    assert snapshot.is_file()
                    snapshot.unlink()
                    # The generator would delete its layout snapshot; the mock leaves it.
                    layout_snapshot = pathlib.Path(generator_handoff.call_args.args[1])
                    assert layout_snapshot.name.startswith("ssn_layout_"), layout_snapshot
                    layout_snapshot.unlink()

                failed_generator_handoff = mock.Mock(side_effect=OSError("exec failed"))
                with mock.patch.object(window, "save_settings", return_value=True), mock.patch.dict(
                    method_globals, {{"_handoff_to_layout_generator": failed_generator_handoff}}
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
            home.cleanup()
            print("OFFSCREEN_CONFIG_OK")
            """
        )
        completed = _run_config_script(script)

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("OFFSCREEN_CONFIG_OK", completed.stdout)
        self.assertNotIn("event loop is already running", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
