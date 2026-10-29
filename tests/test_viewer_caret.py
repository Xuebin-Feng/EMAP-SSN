"""The command line's text cursor (desktop/Viewer_Caret.py).

The bar sits between two characters, snapped to whole device pixels at any
display scale; it lights at once and blinks again from there after each key,
at the system's rate (steady when the system sets none); and it is gone, with
its timer stopped, while the command line is closed. It is a widget over the
canvas that never moves or changes size on its own: Qt draws the canvas's
scene again whenever a widget over it moves or resizes, but not when part of
the widget is repainted, so the blink and the cursor keys draw no network.
The GL test counts the scene draws on a real canvas.
"""
import contextlib
import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PySide6.QtCore import QEvent, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFocusEvent, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from desktop import Viewer_Caret  # noqa: E402
from desktop.Studio_Theme import TOKENS  # noqa: E402
from desktop.Viewer_Caret import CommandCaret  # noqa: E402
from tests.gl_window_environment import gl_window_environment  # noqa: E402


class CaretTestCase(unittest.TestCase):
    """A canvas stand-in with a caret on it, on a system whose cursor flash time is 1060 ms."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.canvas = QWidget()
        self.canvas.resize(400, 120)
        self.canvas.show()
        self.addCleanup(self.canvas.close)
        self.hints = self.patch(Viewer_Caret.QGuiApplication, "styleHints")
        self.flash(1060)
        self.caret = CommandCaret(self.canvas)
        self.app.processEvents()

    def flash(self, milliseconds):
        self.hints.return_value.cursorFlashTime.return_value = milliseconds

    def patch(self, target, name, **kwargs):
        patcher = mock.patch.object(target, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def painted(self):
        """What the caret draws, on a transparent image the size of the canvas."""
        image = QImage(self.caret.size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        self.caret.draw(painter)
        painter.end()
        return image

    @contextlib.contextmanager
    def display(self, dpr=1.0, width=1):
        """The caret's place and size worked out on a display of scale dpr, with a bar width px wide."""
        with mock.patch.object(Viewer_Caret, "caret_width", return_value=width), \
                mock.patch.object(CommandCaret, "devicePixelRatioF", return_value=dpr):
            yield


class BarPlacementTests(CaretTestCase):
    def test_the_bar_is_centred_on_the_boundary_between_two_characters(self):
        with self.display(dpr=1.0, width=2):
            self.caret.show_at(100.0, 20.0, 18.0)
        self.assertEqual(self.caret.bar, QRectF(99.0, 20.0, 2.0, 18.0))

    def test_the_bar_is_snapped_to_whole_device_pixels(self):
        # (x, top, height in logical px, display scale) -> the bar's left, top, width and height in device pixels
        cases = [
            (100.25, 20.3, 19.94, 2.0, (200, 41, 2, 39)),    # the user's 200 %
            (100.75, 21.97, 19.94, 2.0, (201, 44, 2, 40)),
            (87.6, 22.0, 19.9, 1.0, (87, 22, 1, 20)),         # 100 %
            (133.37, 10.2, 12.0, 1.5, (199, 15, 2, 18)),      # 150 %
            (50.0, 5.0, 0.001, 2.0, (99, 10, 2, 1)),          # never thinner than a pixel
        ]
        for x, top, height, dpr, expected in cases:
            with self.subTest(x=x, dpr=dpr):
                caret = CommandCaret(self.canvas)
                with self.display(dpr=dpr):
                    caret.show_at(x, top, height)
                bar = caret.bar
                self.assertEqual(
                    tuple(round(value * dpr, 6) for value in (bar.left(), bar.top(), bar.width(), bar.height())),
                    expected)

    def test_the_bar_is_drawn_in_the_text_colour_and_nowhere_else(self):
        with self.display(dpr=1.0, width=1):
            self.caret.show_at(50.0, 10.0, 20.0)
        bar = self.caret.bar
        left, top, bottom = int(bar.left()), int(bar.top()), int(bar.bottom())
        image = self.painted()
        inside = image.pixelColor(left, top + 5)
        self.assertEqual(inside.name(), QColor(TOKENS["text"]).name())
        self.assertEqual(inside.alpha(), 255)
        for x, y in ((left - 1, top + 5), (left + 1, top + 5), (left, top - 1), (left, bottom)):
            self.assertEqual(image.pixelColor(x, y).alpha(), 0, (x, y))
        self.caret._lit = False
        self.assertEqual(self.painted().pixelColor(left, top + 5).alpha(), 0)

    def test_the_bar_takes_no_mouse_events(self):
        self.assertTrue(self.caret.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))

    def test_the_style_sets_the_bar_width_within_limits(self):
        for style_width, width in ((0, 1), (1, 1), (2, 2), (9, Viewer_Caret.MAX_CARET_WIDTH)):
            with self.subTest(style_width=style_width), \
                    mock.patch.object(Viewer_Caret.QApplication, "style") as style:
                style.return_value.pixelMetric.return_value = style_width
                self.assertEqual(Viewer_Caret.caret_width(), width)


class NeverMovesTests(CaretTestCase):
    """The widget keeps the canvas's geometry: moving or resizing it makes Qt draw the scene again."""

    def test_the_widget_covers_the_canvas_and_follows_its_size(self):
        self.assertEqual(self.caret.geometry(), self.canvas.rect())
        self.canvas.resize(520, 200)
        self.app.processEvents()
        self.assertEqual(self.caret.geometry(), self.canvas.rect())

    def test_moving_the_bar_changes_neither_geometry_nor_visibility(self):
        geometry = self.caret.geometry()
        guarded = [self.patch(self.caret, name) for name in
                   ("setGeometry", "move", "resize", "setVisible", "show", "hide", "raise_", "lower")]
        for x in (10.0, 55.5, 300.0):
            self.caret.show_at(x, 12.0, 20.0)
            self.caret.steady()
        self.caret._flip()
        self.caret.away()
        for each in guarded:
            each.assert_not_called()
        self.assertEqual(self.caret.geometry(), geometry)
        self.assertTrue(self.caret.isVisible())

    def test_a_move_repaints_the_old_and_new_bar_only(self):
        self.caret.show_at(10.0, 12.0, 20.0)
        old = self.caret.bar
        with mock.patch.object(self.caret, "update") as update:
            self.caret.show_at(200.0, 12.0, 20.0)
        rectangles = [call.args[0] for call in update.call_args_list]
        self.assertEqual(len(rectangles), 2)
        for rectangle, bar in zip(rectangles, (old, self.caret.bar)):
            self.assertTrue(rectangle.contains(bar.toAlignedRect()))
            self.assertLess(rectangle.width(), 10)
            self.assertLess(rectangle.height(), 30)
        with mock.patch.object(self.caret, "update") as update:
            self.caret.show_at(200.0, 12.0, 20.0)    # laid out again in the same place: nothing to repaint
        update.assert_not_called()


class BlinkTests(CaretTestCase):
    def test_a_bar_that_comes_is_lit_and_blinks_at_half_the_flash_time(self):
        self.assertFalse(self.caret.blinking)
        self.caret.show_at(10.0, 5.0, 20.0)
        self.assertTrue(self.caret.lit)
        self.assertTrue(self.caret.blinking)
        self.assertEqual(self.caret._timer.interval(), 530)

    def test_the_blink_alternates_lit_and_dark(self):
        self.caret.show_at(10.0, 5.0, 20.0)
        states = []
        for _ in range(4):
            self.caret._flip()
            states.append(self.caret.lit)
        self.assertEqual(states, [False, True, False, True])

    def test_each_key_lights_the_bar_and_restarts_the_blink(self):
        self.caret.show_at(10.0, 5.0, 20.0)
        self.caret._flip()
        self.assertFalse(self.caret.lit)
        self.caret._timer.setInterval(1)       # a timer that has been running
        self.caret.steady()
        self.assertTrue(self.caret.lit)
        self.assertTrue(self.caret.blinking)
        self.assertEqual(self.caret._timer.interval(), 530)

    def test_laying_the_text_out_again_does_not_restart_the_blink(self):
        self.caret.show_at(10.0, 5.0, 20.0)
        self.caret._flip()
        self.caret.show_at(40.0, 5.0, 20.0)    # a resize wrapped the line elsewhere
        self.assertFalse(self.caret.lit)
        self.assertTrue(self.caret.blinking)

    def test_a_system_without_blinking_keeps_the_bar_lit(self):
        for flash in (0, -1):
            with self.subTest(flash=flash):
                self.flash(flash)
                caret = CommandCaret(self.canvas)
                caret.show_at(10.0, 5.0, 20.0)
                caret.steady()
                self.assertTrue(caret.lit)
                self.assertFalse(caret.blinking)

    def test_the_flash_time_is_a_whole_cycle_and_the_bar_is_lit_for_half_of_it(self):
        for flash, interval in ((1060, 530), (500, 250), (2, 1), (1, 0), (0, 0), (-1, 0)):
            with self.subTest(flash=flash):
                self.flash(flash)
                self.assertEqual(Viewer_Caret.blink_interval_ms(), interval)

    def test_a_canvas_that_lost_the_keyboard_shows_no_bar_and_no_blink(self):
        # Another window, or a control over the canvas, took the keys: they no longer go to the command line.
        self.caret.show_at(10.0, 5.0, 20.0)
        bar = self.caret.bar
        left, top = int(bar.left()), int(bar.top())
        self.assertEqual(self.painted().pixelColor(left, top + 5).alpha(), 255)
        self.app.sendEvent(self.canvas, QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.ActiveWindowFocusReason))
        self.assertFalse(self.caret.blinking)
        self.assertEqual(self.painted().pixelColor(left, top + 5).alpha(), 0)
        self.caret.steady()            # nor does a late key wake the blink
        self.assertFalse(self.caret.blinking)
        self.app.sendEvent(self.canvas, QFocusEvent(QEvent.Type.FocusIn, Qt.FocusReason.ActiveWindowFocusReason))
        self.assertTrue(self.caret.blinking)
        self.assertTrue(self.caret.lit)
        self.assertEqual(self.painted().pixelColor(left, top + 5).alpha(), 255)

    def test_a_closed_command_line_has_no_bar_and_no_timer(self):
        self.caret.show_at(10.0, 5.0, 20.0)
        self.caret._flip()
        self.caret.away()
        self.assertTrue(self.caret.bar.isNull())
        self.assertFalse(self.caret.blinking)
        self.caret.steady()            # a late key does not wake it
        self.assertFalse(self.caret.blinking)
        self.assertEqual(self.painted().pixelColor(10, 10).alpha(), 0)

    def test_opening_again_lights_the_bar_afresh(self):
        self.caret.show_at(10.0, 5.0, 20.0)
        self.caret._flip()
        self.caret.away()
        self.caret.show_at(10.0, 5.0, 20.0)
        self.assertTrue(self.caret.lit)
        self.assertTrue(self.caret.blinking)


_GL_SCRIPT = textwrap.dedent(
    """
    import sys
    import time
    from unittest import mock
    sys.path.insert(0, sys.argv[1])
    from vispy import app, scene
    app.use_app("pyside6")
    from PySide6 import QtCore
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
    from desktop import Viewer_Caret

    qapp = QApplication.instance() or QApplication([])
    window = QWidget()
    # On a screen, so that it is drawn, but invisible and out of the way of the mouse and keys.
    window.setWindowFlags(QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint
                          | QtCore.Qt.WindowType.WindowDoesNotAcceptFocus
                          | QtCore.Qt.WindowType.WindowTransparentForInput)
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.setWindowOpacity(0.0)
    canvas = scene.SceneCanvas(size=(480, 200), bgcolor="white", show=False)
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(canvas.native)
    window.resize(480, 200)
    window.show()
    draws = [0]
    draw_scene = canvas._draw_scene

    def counted(*args, **kwargs):
        draws[0] += 1
        return draw_scene(*args, **kwargs)

    canvas._draw_scene = counted

    def exposed():
        handle = window.windowHandle()
        return handle is not None and handle.isExposed()

    # Whether the desktop covered, hid, exposed or stalled the window since this was last cleared. It draws a window
    # it covers or hides not at all (a Wayland compositor stops sending it frame callbacks, and on XWayland the
    # event loop then waits a second at a time, so the blink timer ticks a few times), and Qt draws the whole
    # scene again each time the desktop exposes it: a count taken then says nothing about the caret.
    disturbed = [False]

    class ExposeWatcher(QtCore.QObject):
        def eventFilter(self, watched, event):
            if event.type() == QtCore.QEvent.Type.Expose:
                disturbed[0] = True
            return False

    watcher = ExposeWatcher()
    qapp.installEventFilter(watcher)

    def wait(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            began = time.monotonic()
            qapp.processEvents()
            stalled = time.monotonic() - began > 0.25     # a turn of the loop takes a few milliseconds
            disturbed[0] = disturbed[0] or stalled or not exposed()
            time.sleep(0.005)

    wait(1.0)
    if not canvas.native.isValid():
        print("NO CONTEXT", flush=True)
        sys.exit(0)
    print("CONTEXT OK", flush=True)
    wait(1.0)
    out = {}
    flips = [0]

    def phase(name, seconds, action):
        wait(0.3)
        draws[0] = 0
        action()
        wait(seconds)
        out[name] = draws[0]

    with mock.patch.object(Viewer_Caret, "blink_interval_ms", return_value=40):
        caret = Viewer_Caret.CommandCaret(canvas.native)
        caret._timer.timeout.connect(lambda: flips.__setitem__(0, flips[0] + 1))

        def keys():
            for i in range(12):
                caret.show_at(30.0 + 9.6 * i, 12.0 + (i % 2) * 24.0, 20.0)
                caret.steady()
                wait(0.04)

        def measure():
            # Run every phase; return whether the desktop left the window alone all the while, as the counts need,
            # and the canvas kept the keyboard: a caret over a canvas that lost it does not blink.
            caret.away()
            flips[0] = 0
            disturbed[0] = False
            for name, seconds, action in (
                ("opens", 0.3, lambda: caret.show_at(30.0, 12.0, 20.0)),
                ("blinks", 1.2, lambda: None),
                ("moves", 0.3, keys),
                ("closes", 0.3, caret.away),
                ("control", 0.5, canvas.update),
            ):
                phase(name, seconds, action)
                if disturbed[0] or (name == "opens" and not caret.blinking):
                    return False
            return True

        # The desktop covers, uncovers and exposes the window as it likes, and takes the keyboard from it: measure
        # while it leaves it alone, and again when it did not, for as long as that takes.
        give_up = time.monotonic() + 30
        while not (exposed() and measure()):
            if time.monotonic() > give_up:
                print("DISTURBED", flush=True)
                sys.exit(0)
            wait(0.1)
    out["flips"] = flips[0]
    out["covers_canvas"] = int(caret.geometry() == canvas.native.rect())
    print("DRAWS", " ".join(f"{key}={value}" for key, value in out.items()), flush=True)
    sys.exit(0)
    """
)


class SceneDrawTests(unittest.TestCase):
    def test_the_blink_and_the_cursor_keys_draw_no_scene(self):
        environment = gl_window_environment()
        try:
            result = subprocess.run(
                [sys.executable, "-c", _GL_SCRIPT, SRC_DIR],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=environment, timeout=120,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("the GL probe timed out")
        output = result.stdout + result.stderr[-2000:]
        if "CONTEXT OK" not in result.stdout:
            self.skipTest(f"no OpenGL context to draw with (exit {result.returncode})")
        if "DISTURBED" in result.stdout:
            self.skipTest("the desktop kept covering, exposing or stalling the test window, or kept the keyboard from "
                          "it, so its draws cannot be counted")
        self.assertEqual(result.returncode, 0, output)
        line = next(line for line in result.stdout.splitlines() if line.startswith("DRAWS "))
        values = {key: int(value) for key, value in (item.split("=") for item in line.split()[1:])}
        # The control shows the count works: a canvas.update() is a draw.
        self.assertGreaterEqual(values["control"], 1, output)
        for phase in ("opens", "blinks", "moves", "closes"):
            self.assertEqual(values[phase], 0, f"{phase}: {output}")
        self.assertGreater(values["flips"], 10, output)         # it did blink, and nothing was drawn
        self.assertEqual(values["covers_canvas"], 1, output)


if __name__ == "__main__":
    unittest.main()
