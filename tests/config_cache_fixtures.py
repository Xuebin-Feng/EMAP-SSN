# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Config GUI cache dropdown, input, and tip-panel tests.

Not collected by unittest; import with ``from tests.config_cache_fixtures import ...``.
"""

import os
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def left_press():
    """A left mouse-button press, which always puts an element's tip in the panel."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(1, 1),
        QPointF(1, 1),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def stop_cache_hash_workers(window):
    """Cancel and join any background cache-hash workers started on window creation."""
    window._cache_hash_request_id += 1
    for worker in window._cache_hash_workers.values():
        worker.requestInterruption()
        worker.wait()
    window._cache_hash_workers.clear()


class SharedConfigWindowCase(unittest.TestCase):
    """TestCase base with a shared offscreen ConfigGUI over an isolated folder."""

    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401 - load torch before PySide6
        from EMAPSSN_Tools import DynamicComboBox
        from PySide6.QtWidgets import QApplication
        from tests.config_gui_loader import load_config_namespace, open_config_window
        from tests.theme_fixture import apply_theme_for_class
        from desktop.Studio_Theme import SWITCH_STYLESHEET, TOKENS

        # Keep the developer's viewer_settings.json, and the inputs it selects,
        # out of both the module defaults and the window.
        namespace = load_config_namespace()
        cls.app = QApplication.instance() or QApplication([])
        namespace["configure_qt_application_fonts"](cls.app)
        apply_theme_for_class(cls, cls.app)
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.window = open_config_window(namespace["ConfigGUI"], cls.temp_directory.name)
        cls.window.show()
        cls.namespace = namespace
        cls.switch_stylesheet = SWITCH_STYLESHEET
        cls.tokens = TOKENS
        cls.dynamic_combo_class = DynamicComboBox
        stop_cache_hash_workers(cls.window)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.app.processEvents()
        cls.temp_directory.cleanup()
