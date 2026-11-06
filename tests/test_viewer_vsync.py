"""The Viewer's vertical sync (desktop/Viewer_Vsync.py).

On by default, off with SSN_VIEWER_VSYNC=0 (or off, false, no). The canvas is
made with vsync accordingly, and the Viewer's window, whose swap interval
paces the screen (the canvas is a QOpenGLWidget composed into it), is given
the same interval, for itself alone: the application's default format is
left as it was. canvas.render() is no part of this.
"""
import ast
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402
from vispy import scene  # noqa: E402

from desktop.Viewer_Vsync import (  # noqa: E402
    VSYNC_ENVIRONMENT_VARIABLE, apply_window_swap_interval, vsync_enabled,
)


class VsyncSettingTests(unittest.TestCase):
    def test_on_unless_the_variable_says_off(self):
        self.assertEqual(VSYNC_ENVIRONMENT_VARIABLE, "SSN_VIEWER_VSYNC")
        self.assertTrue(vsync_enabled({}))
        for value in ("1", "on", "true", "yes", "", "anything"):
            with self.subTest(value=value):
                self.assertTrue(vsync_enabled({"SSN_VIEWER_VSYNC": value}))
        for value in ("0", "off", "OFF", "false", "No", " 0 "):
            with self.subTest(value=value):
                self.assertFalse(vsync_enabled({"SSN_VIEWER_VSYNC": value}))

    def test_it_reads_the_environment_by_default(self):
        old = os.environ.get("SSN_VIEWER_VSYNC")
        self.addCleanup(lambda: os.environ.__setitem__("SSN_VIEWER_VSYNC", old) if old is not None
                        else os.environ.pop("SSN_VIEWER_VSYNC", None))
        os.environ["SSN_VIEWER_VSYNC"] = "0"
        self.assertFalse(vsync_enabled())
        os.environ.pop("SSN_VIEWER_VSYNC")
        self.assertTrue(vsync_enabled())

    def test_the_viewer_makes_its_canvas_with_the_setting_and_sets_its_window(self):
        tree = ast.parse((SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8"))
        canvases = [node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "SceneCanvas"]
        self.assertEqual(len(canvases), 1)
        keywords = {keyword.arg: keyword.value for keyword in canvases[0].keywords}
        self.assertEqual(ast.unparse(keywords["vsync"]), "vsync_enabled()")
        calls = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertIn("apply_window_swap_interval(self.main_window, vsync_enabled())", calls)


class VsyncCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def canvas_in_window(self, vsync):
        canvas = scene.SceneCanvas(show=False, vsync=vsync)
        window = QMainWindow()
        window.setCentralWidget(canvas.native)
        self.addCleanup(window.deleteLater)
        return canvas, window

    def test_the_canvas_format_follows_the_argument(self):
        for vsync in (True, False):
            with self.subTest(vsync=vsync):
                canvas, _window = self.canvas_in_window(vsync)
                self.assertEqual(canvas.native.format().swapInterval(), 1 if vsync else 0)

    def test_the_window_gets_the_interval_without_touching_the_default_format(self):
        default = QSurfaceFormat.defaultFormat().swapInterval()
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                _canvas, window = self.canvas_in_window(enabled)
                self.assertTrue(apply_window_swap_interval(window, enabled))
                if window.windowHandle() is not None:
                    self.assertEqual(window.windowHandle().format().swapInterval(), 1 if enabled else 0)
        self.assertEqual(QSurfaceFormat.defaultFormat().swapInterval(), default)

    def test_off_remakes_the_window_surface_once_and_on_leaves_it_alone(self):
        _canvas, window = self.canvas_in_window(True)
        self.assertTrue(apply_window_swap_interval(window, False))
        handle = window.windowHandle()
        self.assertIsNotNone(handle)
        self.assertEqual(handle.format().swapInterval(), 0)
        self.assertTrue(apply_window_swap_interval(window, False))
        self.assertIs(window.windowHandle(), handle)       # already as wanted: nothing is made again
        self.assertTrue(apply_window_swap_interval(window, True))
        self.assertEqual(window.windowHandle().format().swapInterval(), 1)

    def test_a_shown_window_is_left_alone(self):
        _canvas, window = self.canvas_in_window(True)
        window.winId()
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)   # shown, but not on the screen
        window.show()
        self.addCleanup(window.close)
        self.assertFalse(apply_window_swap_interval(window, False))
        self.assertEqual(window.windowHandle().format().swapInterval(), 1)


if __name__ == "__main__":
    unittest.main()
