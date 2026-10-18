# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Compute Network Statistics keeps the Config window usable and never reports for stale inputs.

The statistics run on a StatisticsWorker thread. Most tests here start the worker deferred (its
start() only records, as in CacheDiscoveryRaceTests) and run it on the test's own thread when
they choose, so each interleaving of the run with a click, a selection change or a close is
deterministic; one test uses the real thread, held at a gate, to check the hand-over between
threads.
"""

import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class StatisticsWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401 - load torch before PySide6
        from PySide6.QtWidgets import QApplication
        from tests.config_gui_loader import load_config_namespace

        cls.namespace = load_config_namespace()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tests.config_gui_loader import open_config_window
        from utilities import Network_Score_Statistics as scores

        self.scores = scores
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.root = pathlib.Path(home.name)
        self.window = open_config_window(self.namespace["ConfigGUI"], self.root)
        self.addCleanup(self._close_window)

        # Deferred workers: start() records, the test runs them.
        real = self.namespace["StatisticsWorker"]
        started = self.started = []

        class DeferredWorker(real):
            interrupted = False

            def start(self):
                started.append(self)

            def requestInterruption(self):
                self.interrupted = True

            def isInterruptionRequested(self):
                return self.interrupted

        patcher = mock.patch.dict(self.namespace, {"StatisticsWorker": DeferredWorker})
        patcher.start()
        self.addCleanup(patcher.stop)

        self.work = self.root / "work"
        self.work.mkdir()
        # A BLAST network of 6 nodes and 60 edges, and the FASTA files that select from it.
        rng = np.random.default_rng(1)
        first = rng.integers(0, 5, size=60)
        second = rng.integers(first + 1, 6)
        with h5py.File(self.work / "network.h5", "w") as network:
            network.attrs["model_name"] = "BLAST"
            network.create_dataset("headers", data=[f"n{index}".encode() for index in range(6)])
            network.create_dataset("i", data=first.astype(np.uint16))
            network.create_dataset("j", data=second.astype(np.uint16))
            network.create_dataset("score", data=(rng.random(60) * 40).astype(np.float32))
        self.write_fasta("all.fasta", range(6))
        self.write_fasta("half.fasta", range(0, 6, 2))
        self.write_fasta("none.fasta", [])
        self.write_fasta("foreign.fasta", [0, 1, 2], extra=["x1", "x2"])
        self.select("all.fasta", "network.h5")

    def write_fasta(self, name, nodes, extra=()):
        text = "".join(f">n{index}\nACDE\n" for index in nodes) + "".join(f">{name}\nACDE\n" for name in extra)
        (self.work / name).write_text(text, encoding="utf-8")

    def select(self, fasta, network):
        """Point the window at the files, without the cache discovery a selection also starts."""
        for key in ("FASTA_DIR", "HDF5_DIR"):
            self.window.inputs[key].blockSignals(True)
            self.window.inputs[key].setText(str(self.work))
            self.window.inputs[key].blockSignals(False)
        for combo, name in ((self.window.cb_fasta, fasta), (self.window.cb_hdf5, network)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(name)
            combo.blockSignals(False)
        self.window._update_statistics_button()

    def _close_window(self):
        self.window.close()
        for worker in list(self.window._statistics_workers.values()) + list(self.started):
            worker.wait(1000)
        self.app.processEvents()

    def panel(self):
        from PySide6.QtGui import QTextDocumentFragment

        return QTextDocumentFragment.fromHtml(self.window.tip_panel.text()).toPlainText().strip()

    def report(self):
        return self.window.stat_display.toPlainText()

    def start_run(self):
        self.window.run_statistics()
        self.assertEqual(len(self.started), 1)
        return self.started[-1]

    def finish(self, worker):
        """Run a deferred worker on this thread, as its thread would, then let queued signals land."""
        worker.run()
        self.app.processEvents()

    # --- A run that is left alone ---

    def test_the_window_shows_progress_and_the_button_waits(self):
        window = self.window
        self.assertTrue(window.btn_stats.isEnabled())
        worker = self.start_run()
        self.assertFalse(window.btn_stats.isEnabled())
        self.assertEqual(self.panel(), "Computing network statistics... 0 %")
        self.assertFalse(window.tip_panel.shows_tip())
        # Unrelated edits while it runs do not re-enable the button or stop the run.
        window.update_live_validators()
        window.spin_thresh.setValue(window.spin_thresh.value())
        self.assertFalse(window.btn_stats.isEnabled())
        self.assertFalse(worker.interrupted)

        shown = []
        worker.progress.connect(lambda request, fraction: shown.append(self.panel()))
        with mock.patch.object(self.scores, "CHUNK_EDGES", 10), mock.patch.object(self.scores, "worker_count", lambda: 1):
            self.finish(worker)
        percents = [int(text.split("... ")[1].rstrip(" %")) for text in shown]
        self.assertEqual(percents, sorted(percents))
        self.assertGreater(len(set(percents)), 3)
        self.assertEqual(percents[-1], 100)
        self.assertTrue(window.btn_stats.isEnabled())
        self.assertEqual(self.panel(), "Network statistics computed successfully.")
        self.assertIn("Stored Edges: 60 (Max possible: 15)", self.report())
        self.assertIn("Total Nodes Processed: 6", self.report())

    def test_the_report_is_the_one_the_whole_column_code_gave(self):
        from tests.test_network_score_statistics import legacy_statistics_report

        for fasta in ("all.fasta", "half.fasta"):
            with self.subTest(fasta=fasta):
                self.started.clear()
                self.select(fasta, "network.h5")
                self.finish(self.start_run())
                expected = legacy_statistics_report(str(self.work / "network.h5"), str(self.work / fasta), "global", "alignment_length")
                got = self.window._statistics
                for key in expected:
                    if key != "mean":
                        self.assertEqual(expected[key], got[key], key)
                self.assertAlmostEqual(expected["mean"], got["mean"], places=5)

    def test_a_network_with_no_edge_left_warns_and_keeps_the_report(self):
        self.finish(self.start_run())
        before = self.report()
        self.started.clear()
        self.select("none.fasta", "network.h5")
        self.finish(self.start_run())
        self.assertEqual(self.panel(), "Warning: No valid edges were found in the selected FASTA subset.")
        self.assertEqual(self.report(), before)
        self.assertTrue(self.window.btn_stats.isEnabled())

    def test_errors_show_as_they_always_did(self):
        worker = self.start_run()
        with mock.patch("h5py.File", side_effect=OSError("unreadable network")):
            self.finish(worker)
        self.assertEqual(self.panel(), "Error during network statistics calculation: unreadable network")
        self.assertTrue(self.window.btn_stats.isEnabled())
        self.assertEqual(self.window._statistics, None)

    def test_a_fasta_the_network_does_not_cover_is_an_error(self):
        self.select("foreign.fasta", "network.h5")
        self.finish(self.start_run())
        self.assertEqual(
            self.panel(),
            "Error during network statistics calculation: FASTA file is NOT a strict subset of the "
            "network file. The network lacks 2 sequences.",
        )

    def test_a_missing_fasta_file_counts_every_node(self):
        # run_statistics used to fail here (the node count came from the FASTA's block).
        self.select("absent.fasta", "network.h5")
        self.finish(self.start_run())
        self.assertEqual(self.panel(), "Network statistics computed successfully.")
        self.assertIn("Total Nodes Processed: 6", self.report())
        self.assertIn("Fasta Node Subset: absent.fasta", self.report())

    # --- A run that is interrupted ---

    def stale_run(self):
        """Start a run, and return its worker; the test changes something and then finishes it."""
        self.started.clear()
        return self.start_run()

    def assert_nothing_lands(self, worker, shown_before):
        self.finish(worker)
        self.assertEqual(self.report(), shown_before)
        self.assertIsNone(self.window._statistics)
        self.assertNotIn("successfully", self.panel())
        self.assertEqual(self.window._statistics_workers, {})

    def test_changing_the_selection_or_a_mode_stops_the_run(self):
        window = self.window
        select_combo_value = self.namespace["select_combo_value"]
        shown = self.report()
        for label, change in (
            ("fasta", lambda: self.change_combo(window.cb_fasta, "half.fasta")),
            ("network", lambda: self.change_combo(window.cb_hdf5, "network2.h5")),
            ("score mode", lambda: select_combo_value(window.cb_score_mode, "local")),
            ("normalization", lambda: select_combo_value(window.cb_norm_mode, "longer_sequence")),
            ("fasta folder", lambda: window.inputs["FASTA_DIR"].setText(str(self.root))),
            ("network folder", lambda: window.inputs["HDF5_DIR"].setText(str(self.root))),
        ):
            with self.subTest(change=label):
                self.select("all.fasta", "network.h5")
                select_combo_value(window.cb_score_mode, "global")
                select_combo_value(window.cb_norm_mode, "alignment_length")
                worker = self.stale_run()
                change()
                self.assertTrue(worker.interrupted)
                self.assertEqual(self.panel(), "Network statistics stopped: the inputs changed.")
                self.assertIsNone(window._statistics_request)
                self.assert_nothing_lands(worker, shown)

    def change_combo(self, combo, text):
        if combo.findText(text) < 0:
            combo.addItem(text)
        combo.setCurrentText(text)

    def test_closing_the_window_stops_the_run(self):
        worker = self.start_run()
        self.window.close()
        self.assertTrue(worker.interrupted)
        # Its late result is ignored, whatever the window shows.
        self.finish(worker)
        self.assertEqual(self.window._statistics_workers, {})
        self.assertIsNone(self.window._statistics)

    def test_a_second_run_replaces_the_first(self):
        first = self.start_run()
        self.window.run_statistics()  # the button is disabled, but the method can still be called
        second = self.started[-1]
        self.assertIsNot(first, second)
        self.assertTrue(first.interrupted)
        self.assertFalse(second.interrupted)
        self.finish(first)
        self.assertIsNone(self.window._statistics)
        self.assertFalse(self.window.btn_stats.isEnabled())
        self.finish(second)
        self.assertIn("Stored Edges: 60", self.report())
        self.assertTrue(self.window.btn_stats.isEnabled())

    def test_a_network_rewritten_while_it_ran_is_not_reported(self):
        worker = self.start_run()
        path = self.work / "network.h5"
        before = path.stat()
        with h5py.File(path, "a") as network:
            network.attrs["touched"] = "x" * 500
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))
        self.finish(worker)
        self.assertEqual(self.panel(), "Network statistics stopped: the inputs changed.")
        self.assertIsNone(self.window._statistics)
        self.assertTrue(self.window.btn_stats.isEnabled())

    def test_progress_does_not_take_the_panel_back_from_a_tip_the_user_chose(self):
        worker = self.start_run()
        self.window.tip_panel.setTip("A tip the user asked for.")
        shown = []
        worker.progress.connect(lambda request, fraction: shown.append(self.panel()))
        with mock.patch.object(self.scores, "CHUNK_EDGES", 10), mock.patch.object(self.scores, "worker_count", lambda: 1):
            self.finish(worker)
        self.assertGreater(len(shown), 3)
        self.assertEqual(set(shown), {"A tip the user asked for."})
        # The result itself is news, so it does replace the tip.
        self.assertEqual(self.panel(), "Network statistics computed successfully.")

    def test_progress_is_ignored_after_the_run_ended(self):
        worker = self.start_run()
        self.finish(worker)
        self.window.tip_panel.setTip("Another tip.")
        self.window._statistics_progress(worker.request_id, 0.5)
        self.assertEqual(self.panel(), "Another tip.")


class StatisticsThreadTests(unittest.TestCase):
    """The real thread: its results cross to the window's thread, and a stopped run never lands."""

    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401
        from PySide6.QtWidgets import QApplication
        from tests.config_gui_loader import load_config_namespace

        cls.namespace = load_config_namespace()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tests.config_gui_loader import open_config_window
        from utilities import Network_Score_Statistics as scores

        self.scores = scores
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        root = pathlib.Path(home.name)
        self.window = open_config_window(self.namespace["ConfigGUI"], root)
        self.addCleanup(self.window.close)
        work = root / "work"
        work.mkdir()
        with h5py.File(work / "network.h5", "w") as network:
            network.attrs["model_name"] = "BLAST"
            network.create_dataset("headers", data=[b"a", b"b", b"c"])
            network.create_dataset("i", data=np.asarray([0, 0, 1], dtype=np.uint16))
            network.create_dataset("j", data=np.asarray([1, 2, 2], dtype=np.uint16))
            network.create_dataset("score", data=np.asarray([3.0, 5.0, 8.0], dtype=np.float32))
        (work / "nodes.fasta").write_text(">a\nAC\n>b\nAD\n>c\nAE\n", encoding="utf-8")
        for key in ("FASTA_DIR", "HDF5_DIR"):
            self.window.inputs[key].blockSignals(True)
            self.window.inputs[key].setText(str(work))
            self.window.inputs[key].blockSignals(False)
        for combo, name in ((self.window.cb_fasta, "nodes.fasta"), (self.window.cb_hdf5, "network.h5")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(name)
            combo.blockSignals(False)
        self.window._update_statistics_button()
        # The computation waits at a gate, so the test decides what happens while it runs.
        self.reached = threading.Event()
        self.release = threading.Event()
        real = scores.compute_score_statistics
        reached, release = self.reached, self.release

        def gated(source, **options):
            reached.set()
            release.wait(30)
            if options["cancelled"]():
                raise scores.StatisticsCancelled()
            return real(source, **options)

        patcher = mock.patch.object(scores, "compute_score_statistics", gated)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.release.set)

    def wait_until(self, condition, what, timeout=30):
        deadline = time.monotonic() + timeout
        while not condition():
            self.assertLess(time.monotonic(), deadline, what)
            self.app.processEvents()
            time.sleep(0.005)

    def test_the_result_crosses_to_the_window_thread(self):
        from tests.config_gui_loader import finish_statistics

        self.window.run_statistics()
        self.assertTrue(self.reached.wait(30))
        # The window is free while the run waits: its events are handled.
        self.app.processEvents()
        self.assertFalse(self.window.btn_stats.isEnabled())
        self.release.set()
        finish_statistics(self.window)
        self.assertIn("Stored Edges: 3 (Max possible: 3)", self.window.stat_display.toPlainText())
        self.assertTrue(self.window.btn_stats.isEnabled())
        self.assertEqual(self.window._statistics_workers, {})

    def test_a_stopped_run_never_lands(self):
        self.window.run_statistics()
        self.assertTrue(self.reached.wait(30))
        self.window.cb_fasta.addItem("other.fasta")
        self.window.cb_fasta.setCurrentText("other.fasta")
        self.release.set()
        self.wait_until(lambda: not self.window._statistics_workers, "the stopped worker never finished")
        self.assertIsNone(self.window._statistics)
        self.assertEqual(self.window.stat_display.toPlainText(), "")

    def test_closing_while_it_runs_waits_for_the_thread(self):
        self.window.run_statistics()
        self.assertTrue(self.reached.wait(30))
        worker = next(iter(self.window._statistics_workers.values()))
        threading.Timer(0.3, self.release.set).start()
        self.window.close()  # closeEvent interrupts the run and waits for its thread
        self.assertTrue(worker.isFinished())
        self.assertIsNone(self.window._statistics)


if __name__ == "__main__":
    unittest.main()
