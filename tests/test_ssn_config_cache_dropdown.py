import os
import pathlib
import runpy
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class CacheDropdownRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Utils  # noqa: F401 - load torch before PySide6
        from SSN_Tools import DynamicComboBox
        from PySide6.QtWidgets import QApplication

        with mock.patch.object(QApplication, "exec", return_value=0), mock.patch.object(
            sys, "exit", return_value=None
        ):
            namespace = runpy.run_path(str(SRC / "SSN_Config.py"), run_name="__main__")

        cls.app = namespace["app"]
        cls.window = namespace["window"]
        cls.dynamic_combo_class = DynamicComboBox
        cls.window._cache_hash_request_id += 1
        for worker in cls.window._cache_hash_workers.values():
            worker.requestInterruption()
            worker.wait()
        cls.window._cache_hash_workers.clear()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.app.processEvents()

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
            directory_input.blockSignals(True)
            directory_input.setText(str(saved_layout_dir))
            directory_input.blockSignals(False)
            self.window.current_cache_folder = str(cache_folder)

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


if __name__ == "__main__":
    unittest.main()
