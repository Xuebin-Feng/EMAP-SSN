# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration window cache dropdown, discovery races, and new-cache export."""

import json
import os
import pathlib
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np

from tests.config_cache_fixtures import SharedConfigWindowCase


class CacheDropdownRefreshTests(SharedConfigWindowCase):
    def test_opening_dropdown_finds_new_files_and_preserves_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_layout_dir = pathlib.Path(temp_dir)
            cache_folder = saved_layout_dir / "compatible-layout"
            cache_folder.mkdir()
            older = cache_folder / "older.h5"
            newer = cache_folder / "newer.h5"
            older.touch()
            newer.touch()
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            directory_input = self.window.inputs["SAVED_LAYOUT_DIR"]
            previous_directory = directory_input.text()
            previous_cache_folder = self.window.current_cache_folder
            previous_saved_cache_look = self.window._saved_cache_look
            directory_input.blockSignals(True)
            directory_input.setText(str(saved_layout_dir))
            directory_input.blockSignals(False)
            self.window.current_cache_folder = str(cache_folder)
            self.window._saved_cache_look = None

            try:
                combo = self.window.cb_cache_file
                combo.clear()
                combo.addItem(
                    "older.h5", os.path.join("compatible-layout", "older.h5")
                )
                combo.addItem("(New Layout Cache)", None)
                combo.setCurrentIndex(0)

                combo.showPopup()
                combo.hidePopup()

                self.assertEqual(
                    [combo.itemText(index) for index in range(combo.count())],
                    ["newer.h5", "older.h5", "(New Layout Cache)"],
                )
                self.assertEqual(combo.currentText(), "older.h5")
            finally:
                directory_input.blockSignals(True)
                directory_input.setText(previous_directory)
                directory_input.blockSignals(False)
                self.window.current_cache_folder = previous_cache_folder
                self.window._saved_cache_look = previous_saved_cache_look

    def test_incompatible_canonical_folder_uses_manifest_suffix(self):
        cache_manifest = self.namespace["cache_manifest"]
        current_compatibility = cache_manifest.build_compatibility(
            "a" * 64,
            "b" * 64,
            "alignment",
            alignment_score="global",
            normalization="alignment_length",
            top_edge_percent=5.0,
        )
        incompatible_compatibility = cache_manifest.build_compatibility(
            "c" * 64,
            "b" * 64,
            "alignment",
            alignment_score="global",
            normalization="alignment_length",
            top_edge_percent=5.0,
        )
        incompatible_manifest = cache_manifest.build_manifest(
            {"basename": "set.fasta", "size_bytes": 1, "sha256": "c" * 64},
            {"basename": "network.h5", "size_bytes": 1, "sha256": "b" * 64},
            incompatible_compatibility,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            canonical = root / "readable-cache"
            cache_manifest.write_manifest_atomic(canonical, incompatible_manifest)
            directory_input = self.window.inputs["SAVED_LAYOUT_DIR"]
            previous_directory = directory_input.text()
            previous_cache_folder = self.window.current_cache_folder
            previous_saved_cache_look = self.window._saved_cache_look
            previous_launch_allowed = self.window._cache_launch_allowed
            previous_tracker = self.window.lbl_cache_tracker.text()
            directory_input.blockSignals(True)
            directory_input.setText(str(root))
            directory_input.blockSignals(False)
            try:
                with mock.patch.object(
                    self.window,
                    "_cache_paths_from_inputs",
                    return_value=(str(root / "set.fasta"), str(root / "network.h5")),
                ), mock.patch.object(
                    cache_manifest,
                    "build_compatibility",
                    return_value=current_compatibility,
                ), mock.patch.object(
                    cache_manifest,
                    "build_canonical_cache_name",
                    return_value=canonical.name,
                ), mock.patch.object(
                    cache_manifest,
                    "find_matching_manifest_folders",
                    return_value=[],
                ):
                    self.window._apply_cache_discovery(
                        {
                            "sequence": {"sha256": "a" * 64},
                            "network": {"sha256": "b" * 64},
                            "network_type": "alignment",
                        }
                    )

                suffix = cache_manifest.calculate_manifest_id(
                    current_compatibility
                )[:8]
                expected = root / f"{canonical.name}_[{suffix}]"
                self.assertEqual(
                    pathlib.Path(self.window.current_cache_folder), expected
                )
                self.assertIn(expected.name, self.window.lbl_cache_tracker.text())
                self.assertEqual(
                    cache_manifest.read_manifest(canonical)["manifest_id"],
                    incompatible_manifest["manifest_id"],
                )
            finally:
                directory_input.blockSignals(True)
                directory_input.setText(previous_directory)
                directory_input.blockSignals(False)
                self.window.current_cache_folder = previous_cache_folder
                self.window._saved_cache_look = previous_saved_cache_look
                self.window._cache_launch_allowed = previous_launch_allowed
                self.window.lbl_cache_tracker.setText(previous_tracker)

    def test_folder_dropdown_refresh_does_not_reset_dependent_score_mode(self):
        from PySide6.QtWidgets import QComboBox

        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in ("first.h5", "selected.h5"):
                pathlib.Path(temp_dir, filename).touch()

            network_combo = self.dynamic_combo_class(
                temp_dir, ".h5", include_ext=True
            )
            network_combo.populate()
            network_combo.setCurrentText("selected.h5")

            score_combo = QComboBox()
            score_combo.addItems(["global", "local"])
            score_combo.setCurrentText("local")
            observed_texts = []

            def update_score_mode(network_text):
                observed_texts.append(network_text)
                if not network_text:
                    score_combo.setCurrentIndex(-1)
                elif score_combo.currentIndex() == -1:
                    score_combo.setCurrentText("global")

            network_combo.currentTextChanged.connect(update_score_mode)

            network_combo.showPopup()
            network_combo.hidePopup()

            self.assertEqual(network_combo.currentText(), "selected.h5")
            self.assertEqual(observed_texts, [])
            self.assertEqual(score_combo.currentText(), "local")

    def test_folder_dropdown_emits_final_change_if_selection_disappears(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            remaining = pathlib.Path(temp_dir, "remaining.h5")
            selected = pathlib.Path(temp_dir, "selected.h5")
            remaining.touch()
            selected.touch()

            network_combo = self.dynamic_combo_class(
                temp_dir, ".h5", include_ext=True
            )
            network_combo.populate()
            network_combo.setCurrentText("selected.h5")
            observed_texts = []
            network_combo.currentTextChanged.connect(observed_texts.append)

            selected.unlink()
            network_combo.populate()

            self.assertEqual(network_combo.currentText(), "remaining.h5")
            self.assertEqual(observed_texts, ["remaining.h5"])

    def test_cache_separator_and_conditional_name_field(self):
        from PySide6.QtCore import QPoint

        original_index = self.window.tabs.currentIndex()
        combo = self.window.cb_cache_file
        try:
            self.window.tabs.setCurrentIndex(0)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("existing.h5", "layout/existing.h5")
            combo.addItem("(New Layout Cache)", None)
            combo.setCurrentIndex(0)
            combo.setEnabled(True)
            combo.blockSignals(False)
            self.window._toggle_new_cache_input()
            self.app.processEvents()

            tab = self.window.tabs.widget(0).widget()
            button_container = self.window.btn_stats.parentWidget()
            tracker_container = self.window.lbl_cache_tracker.parentWidget()
            separator = self.window.cache_file_separator
            cache_label = self.window.labels["TARGET_CACHE"]
            target_label = self.window.labels["TARGET_CACHE_FILE"]
            button_position = button_container.mapTo(tab, QPoint(0, 0))
            tracker_position = tracker_container.mapTo(tab, QPoint(0, 0))
            separator_position = separator.mapTo(tab, QPoint(0, 0))
            cache_position = cache_label.mapTo(tab, QPoint(0, 0))
            target_position = target_label.mapTo(tab, QPoint(0, 0))
            expected_margin = self.namespace["CONFIG_TAB_CONTENT_MARGIN"]
            expected_spacing = self.namespace["CONFIG_SEPARATOR_PADDING"]

            self.assertEqual(
                separator.height(), self.namespace["CONFIG_SEPARATOR_THICKNESS"]
            )
            self.assertEqual(separator_position.x(), expected_margin)
            self.assertEqual(
                separator.width(), tab.width() - (2 * expected_margin)
            )
            self.assertEqual(
                separator_position.y()
                - button_position.y()
                - button_container.height(),
                expected_spacing,
            )
            self.assertEqual(
                cache_position.y() - separator_position.y() - separator.height(),
                expected_spacing,
            )
            self.assertEqual(cache_label.text(), "Target Cache:")
            self.assertEqual(tracker_position.y(), cache_position.y())
            self.assertGreater(target_position.y(), tracker_position.y())

            self.assertTrue(self.window.line_new_cache.isHidden())
            combo.setCurrentText("(New Layout Cache)")
            self.app.processEvents()
            self.assertFalse(self.window.line_new_cache.isHidden())
            self.assertTrue(self.window.line_new_cache.isEnabled())
            target_layout = self.window.line_new_cache.parentWidget().layout()
            self.assertEqual(target_layout.stretch(0), target_layout.stretch(1))

            self.window.line_new_cache.setText("temporary-name")
            combo.setCurrentText("existing.h5")
            self.app.processEvents()
            self.assertTrue(self.window.line_new_cache.isHidden())
            self.assertEqual(self.window.line_new_cache.text(), "")
        finally:
            self.window.tabs.setCurrentIndex(original_index)
            self.app.processEvents()

    def test_new_cache_focus_is_user_initiated_and_reference_typing_stays_put(self):
        from PySide6.QtTest import QTest

        combo = self.window.cb_cache_file
        original_tab = self.window.tabs.currentIndex()
        original_focus = self.app.focusWidget()
        original_items = [
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        ]
        original_combo_index = combo.currentIndex()
        original_combo_enabled = combo.isEnabled()
        original_cache_folder = self.window.current_cache_folder
        original_cache_launch_allowed = self.window._cache_launch_allowed
        original_reference = self.window.line_ref.text()
        original_cache_name = self.window.line_new_cache.text()

        try:
            self.window.tabs.setCurrentIndex(0)
            self.window.show()
            self.window.activateWindow()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(New Layout Cache)", None)
            combo.setCurrentIndex(0)
            combo.setEnabled(True)
            combo.blockSignals(False)
            self.window._cache_launch_allowed = True
            self.window._toggle_new_cache_input()
            self.window.line_new_cache.setText("cache-name")

            with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
                self.window, "_request_cache_discovery"
            ) as request_cache_discovery:
                self.window.current_cache_folder = temp_dir
                self.window.line_ref.clear()
                self.window.line_ref.setFocus()
                self.app.processEvents()

                # A programmatic cache refresh may update the conditional field,
                # but it must not redirect keyboard focus to that field.
                self.window._refresh_cache_file_combo()
                self.app.processEvents()
                self.assertIs(self.app.focusWidget(), self.window.line_ref)

                for character in "ref42":
                    QTest.keyClicks(self.app.focusWidget(), character)
                    self.app.processEvents()
                    self.assertIs(self.app.focusWidget(), self.window.line_ref)

                self.assertEqual(self.window.line_ref.text(), "ref42")
                self.assertEqual(self.window.line_new_cache.text(), "cache-name")
                self.assertEqual(combo.currentText(), "(New Layout Cache)")
                self.assertTrue(self.window.spin_alignment_offset.isEnabled())
                self.assertTrue(self.window.lbl_alignment_offset.isEnabled())

                self.window.line_ref.setText("   ")
                self.assertFalse(self.window.spin_alignment_offset.isEnabled())
                self.assertFalse(self.window.lbl_alignment_offset.isEnabled())
                request_cache_discovery.assert_not_called()

                # activated is emitted only for a user selection, so this is
                # the one path that should move focus to the cache-name field.
                combo.activated.emit(combo.currentIndex())
                self.app.processEvents()
                self.assertIs(self.app.focusWidget(), self.window.line_new_cache)
        finally:
            self.window.current_cache_folder = original_cache_folder
            self.window._cache_launch_allowed = original_cache_launch_allowed
            combo.blockSignals(True)
            combo.clear()
            for text, data in original_items:
                combo.addItem(text, data)
            combo.setCurrentIndex(original_combo_index)
            combo.setEnabled(original_combo_enabled)
            combo.blockSignals(False)
            self.window._toggle_new_cache_input()
            if self.window._new_cache_selected():
                self.window.line_new_cache.setText(original_cache_name)
            self.window.line_ref.setText(original_reference)
            self.window.tabs.setCurrentIndex(original_tab)
            if original_focus is not None and original_focus.isVisible():
                original_focus.setFocus()
            self.app.processEvents()

    def test_layout_export_contains_only_generation_settings_and_exact_name(self):
        generation_values = {
            "NODE_FASTA_FILE": "Input_Files/Sequence_Sets/example.fasta",
            "INPUT_HDF5": "Input_Files/Networks_EValues/example.h5",
            "ALIGNMENT_SCORE": "global",
            "NORM_MODE": "alignment_length",
            "UMAP_MODE": False,
            "UMAP_NEIGHBORS": 15,
            "UMAP_MIN_DIST": 0.1,
            "LAYOUT_DEVICE_SELECTION": "auto",
            "SPRING_K": 5.0,
            "COULOMB_K": 10.0,
            "COULOMB_CUTOFF": 30.0,
            "DAMPING": 0.9,
            "DT": 0.005,
            "MAX_STEPS": 10000,
            "RMSD_THRESHOLD": 0.005,
            "PERCENTAGE_DROP_THRESHOLD": 0.1,
            "RMSD_WINDOW": 50,
            "ENABLE_PROGRESSIVE_SIMULATION": False,
            "PACKING_GEOMETRY": "Square",
            "PACKING_GRID_SIZE": 20.0,
            "NODE_SIZE": 10,
            "MSA_FILE": "example.fasta",
            "PRINT_SAVE_DIR": "Analysis_Results/Saved_Images",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            target_json = temp_path / "exported.json"
            method_globals = self.window.export_layout_settings.__func__.__globals__
            with mock.patch.object(
                self.window, "collect_data", return_value=generation_values
            ), mock.patch.object(
                self.window, "_selected_new_cache_filename", return_value="exact.h5"
            ), mock.patch.object(
                self.window.spin_thresh, "optionalValue", return_value=0.1
            ), mock.patch.object(
                self.window.spin_top, "optionalValue", return_value=None
            ), mock.patch.object(
                self.window.cb_layout_device, "currentData", return_value="auto"
            ), mock.patch.object(
                self.window.inputs["SAVED_LAYOUT_DIR"],
                "text",
                return_value=str(temp_path / "layouts"),
            ), mock.patch.object(
                self.window.inputs["SETTING_EXPORT_DIR"],
                "text",
                return_value=str(temp_path / "layout_exports"),
            ), mock.patch.object(
                self.window, "_cache_launch_allowed", True
            ), mock.patch.object(
                self.window, "current_cache_folder", str(temp_path / "target")
            ), mock.patch.dict(
                method_globals, {"PROJECT_ROOT": temp_path}
            ):
                settings = self.window._collect_layout_generation_settings()
                document = settings.to_document(project_root=temp_path)
                from desktop.Viewer_State import decode_document
                payload = decode_document(document, "layout")
                self.assertEqual(document["schema_version"], 2)
                self.assertEqual(document["kind"], "layout")
                self.assertEqual(payload["CACHE_FILENAME"], "exact.h5")
                self.assertIs(payload["UMAP_MODE"], False)
                self.assertIsInstance(payload["MAX_STEPS"], int)
                self.assertNotIn("PHYSICS_ENGINE", payload)
                self.assertFalse(any(key.startswith("MC_") for key in payload))
                self.assertFalse(any(key.startswith("SGLD_") for key in payload))
                self.assertNotIn("NODE_SIZE", payload)
                self.assertNotIn("MSA_FILE", payload)
                self.assertNotIn("PRINT_SAVE_DIR", payload)
                self.assertEqual(payload["SAVED_LAYOUT_DIR"], "layouts")

                with mock.patch.object(
                    self.namespace["QFileDialog"],
                    "getSaveFileName",
                    return_value=(str(target_json), "JSON Files (*.json)"),
                ) as get_save_file_name, mock.patch.object(
                    self.namespace["QMessageBox"], "information"
                ) as information, mock.patch.object(
                    self.namespace["QMessageBox"], "critical"
                ) as critical:
                    self.window.export_layout_settings()
                    suggested_path = pathlib.Path(
                        get_save_file_name.call_args.args[2]
                    )
                    self.assertEqual(
                        suggested_path.parent,
                        temp_path / "layout_exports",
                    )

            exported = json.loads(target_json.read_text(encoding="utf-8"))
            self.assertEqual(exported, document)
            information.assert_called_once()
            self.assertIn(
                "Layout_Cache_Generator.py",
                information.call_args.args[2],
            )
            critical.assert_not_called()

    def test_layout_export_button_is_enabled_only_for_new_cache_generation(self):
        # Save & Run is the one primary action; Export is an outline button.
        self.assertEqual(self.window.btn_save_run.property("role"), "primary")
        self.assertIsNone(self.window.btn_export_layout.property("role"))
        self.assertEqual(self.window.btn_export_layout.styleSheet(), "")
        self.assertEqual(
            self.window.btn_export_layout.sizeHint().height(),
            self.window.btn_save_run.sizeHint().height(),
        )
        self.assertEqual(
            self.window.btn_export_layout.sizeHint().height(),
            self.window.btn_check.sizeHint().height(),
        )
        combo = self.window.cb_cache_file
        original_items = [
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        ]
        original_index = combo.currentIndex()
        try:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("version_00.h5", "layout/version_00.h5")
            combo.addItem("(New Layout Cache)", None)
            combo.blockSignals(False)
            with mock.patch.object(self.window, "_cache_launch_allowed", True):
                combo.setCurrentIndex(1)
                self.assertTrue(self.window.btn_export_layout.isEnabled())
                combo.setCurrentIndex(0)
                self.assertFalse(self.window.btn_export_layout.isEnabled())
        finally:
            combo.blockSignals(True)
            combo.clear()
            for text, data in original_items:
                combo.addItem(text, data)
            combo.setCurrentIndex(original_index)
            combo.blockSignals(False)
            self.window._toggle_new_cache_input()


class CacheDiscoveryRaceTests(unittest.TestCase):
    """Hashing left running for earlier inputs never decides the target cache.

    Discovery hashes the selected FASTA and network in a CacheHashWorker,
    unless both hashes are cached already; then it applies them at once. That
    synchronous path used to leave the running worker's request current, so
    the worker's late result was cached under the newly selected files and
    shown for them. Here each worker hashes only when a test finishes it, on
    the test's thread, so every interleaving is deterministic.

    Nor does a check that ends in Save & Run's pause after a click, which
    keeps a double click from launching twice, cut the pause short or outlast it.

    A cache published after the check, as the Layout Generator publishes one
    (manifest, then .h5), shows at the window's next look without the inputs
    being chosen again; a look that finds what is shown changes nothing.
    """

    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401 - load torch before PySide6
        from PySide6.QtWidgets import QApplication
        from tests.config_gui_loader import load_config_namespace

        cls.namespace = load_config_namespace()
        cls.cache_manifest = cls.namespace["cache_manifest"]
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tests.config_gui_loader import open_config_window

        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        root = pathlib.Path(home.name)
        self._defer_hashing()
        self.window = open_config_window(self.namespace["ConfigGUI"], root)
        self.addCleanup(self._close_window)

        # Two FASTA files share one BLAST network, and each pair has its own
        # compatible layout folder, so the tracker names the hashes it used.
        self.work = root / "work"
        self.work.mkdir()
        (self.work / "a.fasta").write_text(">a1\nMKTAYIAK\n>a2\nMKTAYIAR\n", encoding="utf-8")
        (self.work / "b.fasta").write_text(">b1\nMSDNELKQ\n>b2\nMSDNELKE\n", encoding="utf-8")
        with h5py.File(self.work / "network.h5", "w") as network:
            network.attrs["model_name"] = "BLAST"
            network.create_dataset("headers", data=[b"a1", b"a2"])
            network.create_dataset("i", data=np.asarray([0], dtype=np.uint16))
            network.create_dataset("j", data=np.asarray([1], dtype=np.uint16))
            network.create_dataset("score", data=np.asarray([1e-30]))
        layouts = root / "layouts"
        self.layout_folders = {
            name: self._write_layout_folder(layouts, name)
            for name in ("a.fasta", "b.fasta")
        }
        self.window.inputs["SAVED_LAYOUT_DIR"].setText(str(layouts))
        self.window.inputs["FASTA_DIR"].setText(str(self.work))
        self.window.inputs["HDF5_DIR"].setText(str(self.work))
        self.window.cb_hdf5.setCurrentText("network.h5")

    def _defer_hashing(self):
        """Replace CacheHashWorker with one that hashes only when _finish() runs it."""
        self.workers, self.interrupted = [], set()
        workers, interrupted = self.workers, self.interrupted

        class DeferredHashWorker(self.namespace["CacheHashWorker"]):
            def start(self):
                workers.append(self)

            # A thread that never started ignores requestInterruption(), so
            # keep the request where run() looks for it.
            def requestInterruption(self):
                interrupted.add(self.request_id)

            def isInterruptionRequested(self):
                return self.request_id in interrupted

        patcher = mock.patch.dict(self.namespace, {"CacheHashWorker": DeferredHashWorker})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_layout_folder(self, layouts, fasta_name, folder_name=None):
        """Write a layout folder whose manifest matches this FASTA with the network."""
        sequence = self.cache_manifest.fingerprint_file(self.work / fasta_name)
        network = self.cache_manifest.fingerprint_file(self.work / "network.h5")
        compatibility = self.cache_manifest.build_compatibility(
            sequence["sha256"],
            network["sha256"],
            "blast",
            **self.window._cache_setting_values(),
        )
        folder = layouts / (folder_name or f"{pathlib.Path(fasta_name).stem}_layout")
        self.cache_manifest.write_manifest_atomic(
            folder, self.cache_manifest.build_manifest(sequence, network, compatibility)
        )
        return folder

    def _close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def _select(self, fasta_name):
        self.window.cb_fasta.setCurrentText(fasta_name)
        self.assertEqual(self.window.cb_fasta.currentText(), fasta_name)

    def _finish(self, worker):
        """Hash and report on this thread, as the worker's own thread would."""
        worker.run()
        self.app.processEvents()

    def _tracker(self):
        return self.window.lbl_cache_tracker.text()

    def _shown(self, fasta_name):
        return f"Compatible Folder: {self.layout_folders[fasta_name].name}"

    def _items(self):
        combo = self.window.cb_cache_file
        return [combo.itemText(index) for index in range(combo.count())]

    def _needs_computing(self, fasta_name):
        """Select a FASTA whose layout folder is gone; return the folder the window targets."""
        shutil.rmtree(self.layout_folders[fasta_name])
        self._select(fasta_name)
        self._finish(self.workers[-1])
        target = pathlib.Path(self.window.current_cache_folder)
        self.assertEqual(self._tracker(), f"Target Folder: {target.name} [Needs Computing]")
        self.assertEqual(self._items(), ["(New Layout Cache)"])
        return target

    def _generate(self, fasta_name, folder, cache_name="version_00.h5"):
        """Publish a cache as the Layout Generator does: the manifest, then the .h5 file."""
        if not (folder / self.cache_manifest.MANIFEST_FILENAME).exists():
            self._write_layout_folder(folder.parent, fasta_name, folder.name)
        (folder / cache_name).write_bytes(b"")

    def test_hashing_left_running_for_earlier_inputs_never_overrides_cached_ones(self):
        """a.fasta is still being hashed when b.fasta, already hashed, is chosen again."""
        self._select("b.fasta")
        self._finish(self.workers[-1])
        self.assertEqual(self._tracker(), self._shown("b.fasta"))

        self._select("a.fasta")
        hashing_a = self.workers[-1]
        request_a = hashing_a.request_id
        self.assertEqual(self._tracker(), "Checking input files…")
        self._select("b.fasta")
        self.assertEqual(self._tracker(), self._shown("b.fasta"))

        self._finish(hashing_a)
        self.assertEqual(self._tracker(), self._shown("b.fasta"))
        self.assertEqual(
            pathlib.Path(self.window.current_cache_folder),
            self.layout_folders["b.fasta"],
        )
        # Nor was a.fasta's hash cached for b.fasta: the next refresh agrees.
        self.window.update_live_validators()
        self.assertEqual(self._tracker(), self._shown("b.fasta"))
        # The worker was told to stop, not left to hash for nothing.
        self.assertIn(request_a, self.interrupted)

    def test_inputs_whose_hashing_was_dropped_are_hashed_again(self):
        self._select("b.fasta")
        self._finish(self.workers[-1])
        self._select("a.fasta")
        dropped = self.workers[-1]
        self._select("b.fasta")
        self._select("a.fasta")
        # The dropped worker's result will be ignored, so a.fasta needs another.
        self.assertEqual(self._tracker(), "Checking input files…")
        self.assertEqual(len(self.workers), 3)

        self._finish(dropped)
        self.assertEqual(self._tracker(), "Checking input files…")
        self._finish(self.workers[-1])
        self.assertEqual(self._tracker(), self._shown("a.fasta"))

    def test_a_file_rewritten_while_it_is_hashed_is_hashed_again(self):
        """Hashes are cached for the files as they were when hashing began,
        never for whatever is on disk when the result arrives."""
        a_fasta = self.work / "a.fasta"
        fingerprint_file = self.cache_manifest.fingerprint_file

        def fingerprint_then_rewrite(path, **options):
            record = fingerprint_file(path, **options)
            if os.path.basename(path) == "a.fasta":
                # Replaced right after it was read: it now holds b.fasta's sequences.
                a_fasta.write_bytes((self.work / "b.fasta").read_bytes())
                modified = a_fasta.stat().st_mtime_ns + 5_000_000_000
                os.utime(a_fasta, ns=(modified, modified))
            return record

        self._select("a.fasta")
        with mock.patch.object(
            self.cache_manifest, "fingerprint_file", side_effect=fingerprint_then_rewrite
        ):
            self._finish(self.workers[-1])
        self.window.update_live_validators()  # as the next change in the window would
        self.assertEqual(self._tracker(), "Checking input files…")
        self.assertEqual(len(self.workers), 2)
        self._finish(self.workers[-1])
        self.assertEqual(self._tracker(), self._shown("b.fasta"))

    def test_discovery_names_the_score_kind_of_an_efi_network_as_its_manifest_does(self):
        """An EFI network's manifest carries score_kind, so the window must look for it."""
        efi_network = self.work / "efi.h5"
        shutil.copyfile(self.work / "network.h5", efi_network)
        with h5py.File(efi_network, "a") as network:
            network.attrs["score_kind"] = "efi_alignment_score"
        sequence = self.cache_manifest.fingerprint_file(self.work / "a.fasta")
        network = self.cache_manifest.fingerprint_file(efi_network)
        layouts = pathlib.Path(self.window.inputs["SAVED_LAYOUT_DIR"].text())
        # One folder was made for the EFI network, one as if it were an E-value network.
        for kind, folder in (("evalue", "as_evalue"), ("efi_alignment_score", "as_efi")):
            compatibility = self.cache_manifest.build_compatibility(
                sequence["sha256"], network["sha256"], "blast", score_kind=kind,
                **self.window._cache_setting_values(),
            )
            self.cache_manifest.write_manifest_atomic(
                layouts / folder,
                self.cache_manifest.build_manifest(sequence, network, compatibility),
            )

        self.window.cb_hdf5.refresh_callback()
        self._select("b.fasta")
        self.window.cb_hdf5.setCurrentText("efi.h5")
        self.assertEqual(self.window.cb_hdf5.currentText(), "efi.h5")
        self._select("a.fasta")
        self._finish(self.workers[-1])

        self.assertEqual(self._tracker(), "Compatible Folder: as_efi")
        self.assertEqual(pathlib.Path(self.window.current_cache_folder), layouts / "as_efi")

    def test_giving_up_on_the_selected_inputs_stops_their_hashing(self):
        """However discovery gives up on the selected inputs, it stops the
        worker hashing them and ignores that worker's result."""
        window = self.window

        def clear_the_selection():
            window.cb_fasta.setCurrentText("")

        def fail_to_read_the_inputs():
            with mock.patch.object(
                self.cache_manifest, "file_cache_key", side_effect=OSError("access denied")
            ):
                window.update_live_validators()

        def choose_inputs_that_need_hashing():
            window.cb_fasta.setCurrentText("b.fasta")

        def remove_the_network():
            (self.work / "network.h5").unlink()
            window.update_live_validators()

        for fasta_name, give_up, message in (
            ("a.fasta", clear_the_selection, "Target Cache: Missing FASTA or HDF5"),
            ("b.fasta", fail_to_read_the_inputs, "Cache input error: access denied"),
            ("a.fasta", choose_inputs_that_need_hashing, "Checking input files…"),
            ("a.fasta", remove_the_network, "Target Cache: Selected input file is missing"),
        ):
            with self.subTest(give_up.__name__):
                self._select(fasta_name)
                hashing = self.workers[-1]
                give_up()
                self.assertIn(hashing.request_id, self.interrupted)
                self._finish(hashing)
                self.assertEqual(self._tracker(), message)

    def test_a_double_click_on_save_and_run_launches_once_and_the_button_comes_back(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        self._select("a.fasta")
        self._finish(self.workers[-1])
        self.window.show()
        button = self.window.btn_save_run
        self.assertTrue(button.isEnabled())

        self.assertEqual(self.namespace["CONFIG_RUN_BUTTON_PAUSE_MS"], 1000)
        with mock.patch.object(self.window, "save_and_run") as save_and_run, \
                mock.patch.dict(self.namespace, {"CONFIG_RUN_BUTTON_PAUSE_MS": 150}):
            started = time.monotonic()
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            save_and_run.assert_called_once_with()
            self.assertFalse(button.isEnabled())
            deadline = started + 5.0
            while not button.isEnabled() and time.monotonic() < deadline:
                QTest.qWait(10)
            self.assertTrue(button.isEnabled())
            self.assertGreaterEqual(time.monotonic() - started, 0.14)
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            self.assertEqual(save_and_run.call_count, 2)

    def test_a_cache_check_ending_in_the_pause_neither_ends_it_nor_outlasts_it(self):
        self._select("a.fasta")
        self._finish(self.workers[-1])
        button = self.window.btn_save_run
        timers = []
        # The pause ends when the test says, not on a clock.
        stand_in = SimpleNamespace(singleShot=lambda *args: timers.append(args))
        with mock.patch.object(self.window, "save_and_run"), \
                mock.patch.dict(self.namespace, {"QTimer": stand_in}):
            button.click()
            [(pause, context, end_pause)] = timers
            self.assertEqual((pause, context), (1000, button))
            self.assertFalse(button.isEnabled())

            # b.fasta is hashed, and its cache found, before the pause ends.
            self._select("b.fasta")
            self._finish(self.workers[-1])
            self.assertEqual(self._tracker(), self._shown("b.fasta"))
            self.assertFalse(button.isEnabled())
            end_pause()
            self.assertTrue(button.isEnabled())

            # The inputs are cleared in the next pause: the button stays grey after it.
            button.click()
            end_pause = timers[-1][2]
            self._select("")
            self.assertEqual(self._tracker(), "Target Cache: Missing FASTA or HDF5")
            end_pause()
            self.assertFalse(button.isEnabled())

    def test_a_cache_generated_after_the_check_shows_at_the_next_look(self):
        target = self._needs_computing("a.fasta")
        needs_computing = self._tracker()
        self.window._recheck_saved_caches()  # nothing published yet
        self.assertEqual(self._tracker(), needs_computing)

        self._generate("a.fasta", target)
        self.window._recheck_saved_caches()
        self.assertEqual(self._tracker(), f"Compatible Folder: {target.name}")
        self.assertEqual(self._items(), ["version_00.h5", "(New Layout Cache)"])
        # Selected, so Save & Run opens it rather than computing another.
        self.assertEqual(self.window.cb_cache_file.currentText(), "version_00.h5")
        self.assertTrue(self.window._cache_launch_allowed)
        self.assertEqual(len(self.workers), 1, "the inputs were not hashed again")

    def test_a_manifest_published_before_its_cache_shows_the_folder_then_the_cache(self):
        target = self._needs_computing("a.fasta")
        self._write_layout_folder(target.parent, "a.fasta", target.name)
        self.window._recheck_saved_caches()
        self.assertEqual(self._tracker(), f"Compatible Folder: {target.name}")
        self.assertEqual(self._items(), ["(New Layout Cache)"])

        # Still being written: the Layout Generator's .partial file is no cache yet.
        (target / ".version_00.h5.x1y2.partial").write_bytes(b"")
        with mock.patch.object(self.window, "_request_cache_discovery") as search:
            self.window._recheck_saved_caches()
        search.assert_not_called()

        self._generate("a.fasta", target)
        self.window._recheck_saved_caches()
        self.assertEqual(self.window.cb_cache_file.currentText(), "version_00.h5")

    def test_a_cache_generated_into_the_compatible_folder_is_selected(self):
        self._select("b.fasta")
        self._finish(self.workers[-1])
        self.assertEqual(self._items(), ["(New Layout Cache)"])
        self.window.line_new_cache.setText("mine.h5")

        self._generate("b.fasta", self.layout_folders["b.fasta"], "mine.h5")
        self.window._recheck_saved_caches()
        self.assertEqual(self._items(), ["mine.h5", "(New Layout Cache)"])
        self.assertEqual(self.window.cb_cache_file.currentText(), "mine.h5")
        self.assertEqual(self.window.line_new_cache.text(), "")

    def test_a_look_that_finds_no_new_cache_keeps_the_choice(self):
        folder = self.layout_folders["b.fasta"]
        for age, name in enumerate(("newer.h5", "older.h5")):
            (folder / name).write_bytes(b"")
            os.utime(folder / name, (1_700_000_000 - age, 1_700_000_000 - age))
        self._select("b.fasta")
        self._finish(self.workers[-1])
        combo = self.window.cb_cache_file
        self.assertEqual(self._items(), ["newer.h5", "older.h5", "(New Layout Cache)"])
        combo.setCurrentIndex(combo.findText("older.h5"))

        # A layout folder for other inputs appears: the window is left alone.
        self._write_layout_folder(folder.parent, "a.fasta", "other_inputs_layout")
        with mock.patch.object(combo, "clear", wraps=combo.clear) as clear:
            self.window._recheck_saved_caches()
        clear.assert_not_called()
        self.assertEqual(combo.currentText(), "older.h5")

        # A cache that goes is dropped from the list; the one chosen stays chosen.
        (folder / "newer.h5").unlink()
        self.window._recheck_saved_caches()
        self.assertEqual(self._items(), ["older.h5", "(New Layout Cache)"])
        self.assertEqual(combo.currentText(), "older.h5")

        # So does a new cache being named.
        combo.setCurrentIndex(combo.count() - 1)
        self.window._toggle_new_cache_input()
        self.window.line_new_cache.setText("draft.h5")
        (folder / "older.h5").unlink()
        self.window._recheck_saved_caches()
        self.assertEqual(self._items(), ["(New Layout Cache)"])
        self.assertEqual(self.window.line_new_cache.text(), "draft.h5")

    def test_a_second_compatible_folder_shows_the_duplicate_error_at_the_next_look(self):
        self._select("b.fasta")
        self._finish(self.workers[-1])
        folder = self.layout_folders["b.fasta"]
        shutil.copytree(folder, folder.with_name("b_layout_copy"))
        self.window._recheck_saved_caches()
        self.assertEqual(self._tracker(), "Error: 2 compatible cache folders found")
        self.assertFalse(self.window._cache_launch_allowed)

        shutil.rmtree(folder.with_name("b_layout_copy"))
        self.window._recheck_saved_caches()
        self.assertEqual(self._tracker(), self._shown("b.fasta"))

    def test_a_cache_published_during_the_search_shows_at_the_next_look(self):
        """What a search looks at is read before it runs, so a cache it missed
        still counts as a change at the next look."""
        target = self._needs_computing("a.fasta")
        find_matching = self.cache_manifest.find_matching_manifest_folders

        def search_then_generate(saved_layout_dir, compatibility):
            matches = find_matching(saved_layout_dir, compatibility)
            self._generate("a.fasta", target)
            return matches

        (target.parent / "unrelated").mkdir()  # so the next look searches
        with mock.patch.object(
            self.cache_manifest, "find_matching_manifest_folders", side_effect=search_then_generate
        ):
            self.window._recheck_saved_caches()
        self.assertIn("[Needs Computing]", self._tracker())
        self.window._recheck_saved_caches()
        self.assertEqual(self.window.cb_cache_file.currentText(), "version_00.h5")

    def test_opening_the_cache_list_looks_again(self):
        target = self._needs_computing("a.fasta")
        self._generate("a.fasta", target)
        self.window.show()
        combo = self.window.cb_cache_file
        combo.showPopup()
        combo.hidePopup()
        self.assertEqual(self._tracker(), f"Compatible Folder: {target.name}")
        self.assertEqual(combo.currentText(), "version_00.h5")

    def test_a_shown_window_looks_again_by_itself(self):
        from PySide6.QtTest import QTest

        self.assertEqual(self.namespace["CONFIG_CACHE_RECHECK_MS"], 1000)
        timer = self.window._cache_recheck_timer
        self.assertTrue(timer.isActive())
        self.assertEqual(timer.interval(), 1000)
        target = self._needs_computing("a.fasta")
        self._generate("a.fasta", target)

        # Hidden, it does not look.
        self.window._recheck_saved_caches_while_shown()
        self.assertIn("[Needs Computing]", self._tracker())

        with mock.patch.dict(self.namespace, {"CONFIG_CACHE_RECHECK_MS": 20}):
            timer.setInterval(20)
            self.window.show()
            deadline = time.monotonic() + 5.0
            while "[Needs Computing]" in self._tracker() and time.monotonic() < deadline:
                QTest.qWait(10)
        self.assertEqual(self._tracker(), f"Compatible Folder: {target.name}")

    def test_a_slow_layout_folder_is_looked_at_less_often(self):
        timer = self.window._cache_recheck_timer
        self.window.show()
        clock = iter([10.0, 10.2, 20.0, 20.001])  # a 200 ms look, then a 1 ms one
        with mock.patch.dict(self.namespace, {"time": SimpleNamespace(perf_counter=lambda: next(clock))}), \
                mock.patch.object(self.window, "_recheck_saved_caches"):
            self.window._recheck_saved_caches_while_shown()
            self.assertEqual(timer.interval(), 4000)
            self.window._recheck_saved_caches_while_shown()
            self.assertEqual(timer.interval(), 1000)

    def test_a_folder_removed_before_its_caches_are_listed_shows_an_error(self):
        self._select("b.fasta")
        self._finish(self.workers[-1])
        self.assertIsNotNone(self.window._saved_cache_look)
        gone = str(self.layout_folders["b.fasta"].parent / "removed_layout")
        with mock.patch.object(
            self.cache_manifest, "find_matching_manifest_folders",
            return_value=[{"folder": gone, "manifest": {}}],
        ):
            self.window.update_live_validators()  # b.fasta's hashes are cached
            self.assertTrue(
                self._tracker().startswith("Cache compatibility error: "), self._tracker()
            )
            self.assertIsNone(self.window._saved_cache_look, "nothing left to look at again")

            # Nor does it escape the result of hashing, which nothing else guards.
            self._select("a.fasta")
            self._finish(self.workers[-1])
            self.assertTrue(
                self._tracker().startswith("Cache compatibility error: "), self._tracker()
            )


if __name__ == "__main__":
    unittest.main()
