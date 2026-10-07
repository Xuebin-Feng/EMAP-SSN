# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration window: score-mode and normalization controls follow the network type."""

import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.config_gui_loader import load_config_namespace, open_config_window  # noqa: E402

_WINDOW = None
_APP = None
_DIRECTORY = None


def setUpModule():
    # The window reads temporary directories, never viewer_settings.json, so it
    # selects no inputs and starts no input-hashing worker.
    global _WINDOW, _APP, _DIRECTORY
    namespace = load_config_namespace()
    _APP = namespace["QApplication"].instance() or namespace["QApplication"]([])
    _DIRECTORY = tempfile.TemporaryDirectory()
    _WINDOW = open_config_window(namespace["ConfigGUI"], _DIRECTORY.name)


def tearDownModule():
    for worker in _WINDOW._cache_hash_workers.values():
        worker.requestInterruption()
        worker.wait(10000)
    _WINDOW.close()
    _WINDOW.deleteLater()
    _APP.processEvents()
    _DIRECTORY.cleanup()


def _items(combo):
    return [combo.itemText(index) for index in range(combo.count())]


class NormalizationModeRefreshTests(unittest.TestCase):
    def test_leaving_blast_restores_global_options_and_the_default(self):
        """Signals are blocked while the controls reset, so the list must be refreshed
        explicitly; a list left over from local mode silently rejected the default."""
        window = _WINDOW
        window._set_network_type_controls("alignment")
        window.cb_score_mode.setCurrentText("local")
        self.assertNotIn("alignment_length", _items(window.cb_norm_mode))

        window._set_network_type_controls("blast")
        self.assertEqual(window.cb_score_mode.currentIndex(), -1)
        self.assertEqual(window.cb_norm_mode.currentIndex(), -1)

        window._set_network_type_controls("alignment")
        self.assertEqual(window.cb_score_mode.currentText(), "global")
        self.assertIn("alignment_length", _items(window.cb_norm_mode))
        self.assertEqual(window.cb_norm_mode.currentText(), "alignment_length")
        self.assertFalse(window.cb_score_mode.signalsBlocked())
        self.assertFalse(window.cb_norm_mode.signalsBlocked())


if __name__ == "__main__":
    unittest.main()
