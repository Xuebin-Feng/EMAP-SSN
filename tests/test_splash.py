# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The start-up splash (desktop.Splash) and the status the windows report to it (utilities.Startup_Status): the light colours are the Studio theme's, the language follows the saved setting as the windows do, a status file is read line by line as it grows, texts are translated and filled in, the bar fills as the start passes its milestones and never goes back, the window has no taskbar button and opens on the screen under the pointer, and the error view shows the last lines with Copy, Open log and Close. A step of the start that goes wrong without stopping it opens the section by itself with its lines in the warning colour, wrapped to the section's width. A program beside this one adds a window (its name and art) and its own translation catalogs. The splash process hands over to the window it waits for: it closes when the ready file appears, and an error view stays."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QCoreApplication, QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop import Desktop_App, Splash  # noqa: E402
from desktop.Splash import SplashWindow, StatusFeed, tail_lines  # noqa: E402
from desktop.Studio_Theme import TOKENS  # noqa: E402
from utilities import Startup_Status  # noqa: E402


def flush(app, cycles=4):
    for _ in range(cycles):
        app.processEvents()
        time.sleep(0.005)


class StudioColoursTests(unittest.TestCase):
    def test_the_splash_uses_the_studio_themes_light_shades_in_its_default_purple(self):
        pairs = {
            "surface": "surface", "panel": "surface_muted", "border": "border", "text": "text", "text_soft": "text_soft",
            "muted": "text_muted", "subtle": "text_subtle", "track": "border", "accent": "accent",
            "field_border": "field_border", "danger": "danger", "warning": "warning", "accent_soft": "accent_soft",
        }
        for ours, theirs in pairs.items():
            with self.subTest(colour=ours):
                self.assertEqual(Splash.SCHEME[ours], TOKENS[theirs])


class IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])  # QPixmap needs one, whichever test runs first

    def test_the_splash_names_the_product_and_version_the_windows_do(self):
        self.assertEqual(Splash.PRODUCT, Desktop_App.PRODUCT_NAME)
        self.assertEqual(Splash.VERSION, Desktop_App.APPLICATION_VERSION)

    def test_every_art_file_the_splash_names_is_bundled_small(self):
        for kind, name in Splash.LOGO_FILES.items():
            path = Splash.LOGOS / name
            with self.subTest(kind=kind):
                self.assertTrue(path.is_file())
                self.assertLess(path.stat().st_size, 400 * 1024)  # a 512 px copy: the shipped art decodes to 75 MB
                self.assertEqual((QPixmap(str(path)).width(), QPixmap(str(path)).height()), (512, 512))


    def test_the_art_is_the_large_icon_of_the_shortcuts_in_small(self):
        import numpy
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QImage, QImageReader

        def thumbnail(image):
            image = image.convertToFormat(QImage.Format.Format_RGBA8888_Premultiplied)
            image = image.scaled(QSize(96, 96), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            pixels = numpy.frombuffer(image.constBits(), dtype=numpy.uint8).reshape(96, image.bytesPerLine())[:, :96 * 4]
            return pixels.astype(numpy.float64)

        for kind, copy, large, small in (
            ("viewer", "splash_logo_viewer_512.png", "viewer_logo_large.png", "viewer_logo.png"),
            ("tools", "splash_logo_tool_512.png", "tool_logo_large.png", "tool_logo.png"),
        ):
            with self.subTest(kind=kind):
                art = thumbnail(QImage(str(Splash.LOGOS / copy)))
                differences = {}
                for name in (large, small):  # the shortcuts' large picture, and the small one the splash used to be made from
                    reader = QImageReader(str(Splash.LOGOS / name))
                    reader.setAllocationLimit(0)
                    reader.setScaledSize(QSize(512, 512))
                    differences[name] = float(numpy.abs(art - thumbnail(reader.read())).mean())
                self.assertLess(differences[large], 3.0, differences)
                self.assertGreater(differences[small], 3 * differences[large], differences)


class LanguageTests(unittest.TestCase):
    def test_it_picks_the_language_the_windows_pick(self):
        catalogs = Path(Desktop_App.LANGUAGES_DIR)
        cases = [
            ("en", None), ("system", None), ("zh_CN", None), ("xx", None),
        ]
        for setting, ui_languages in cases:
            for languages in ([], ["zh-Hans-CN", "en-US"], ["en-US", "zh-Hans-CN"], ["de-DE", "zh_CN"], ["fr"]):
                with self.subTest(setting=setting, ui=languages):
                    expected = Desktop_App.resolve_language(setting, ui_languages=languages)
                    self.assertEqual(Splash.splash_language(setting, languages), expected)
        self.assertEqual(Splash.splash_language("zh_CN"), "zh_CN")

    def test_the_saved_setting_comes_from_app_settings_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app_settings.json"
            environment = {"SSN_APP_SETTINGS_PATH": str(path)}
            self.assertEqual(Splash.saved_language_setting(environment), "system")  # no file
            path.write_text(json.dumps({"LANGUAGE": "zh_CN"}), encoding="utf-8")
            self.assertEqual(Splash.saved_language_setting(environment), "zh_CN")
            path.write_text("{ not json", encoding="utf-8")
            self.assertEqual(Splash.saved_language_setting(environment), "system")
            path.write_text(json.dumps({"LANGUAGE": ""}), encoding="utf-8")
            self.assertEqual(Splash.saved_language_setting(environment), "system")
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
            self.assertEqual(Splash.saved_language_setting(environment), "system")

    def test_a_language_with_its_own_font_names_its_files(self):
        files = Splash.language_font_files("zh_CN")
        self.assertTrue(files)
        self.assertTrue(all(path.is_file() for path in files), files)
        self.assertEqual(Splash.language_font_files("en"), [])
        self.assertEqual(Splash.language_font_files("xx"), [])


class StatusFileTests(unittest.TestCase):
    def test_new_complete_lines_are_read_as_the_file_grows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "startup.status"
            feed = StatusFeed(path)
            self.assertEqual(feed.poll(), [])  # no file yet
            with open(path, "ab") as handle:
                handle.write(b'{"text": "one", "fraction": 0.1}\n{"text": "tw')
            self.assertEqual([message["text"] for message in feed.poll()], ["one"])
            with open(path, "ab") as handle:
                handle.write('o \u2026", "fraction": null}\nnot json\n{"nothing": 1}\n[1]\n{"text": "three"}\n'.encode("utf-8"))
            self.assertEqual([message["text"] for message in feed.poll()], ["two \u2026", "three"])
            self.assertEqual(feed.poll(), [])

    def test_a_feed_with_no_file_reads_nothing(self):
        self.assertEqual(StatusFeed(None).poll(), [])

    def test_reports_written_by_the_windows_are_what_the_splash_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "startup.status"
            with mock.patch.dict(os.environ, {Startup_Status.STATUS_FILE_VARIABLE: str(path)}):
                self.assertTrue(Startup_Status.report(Startup_Status.LOADING_PYTORCH, 0.45))
                self.assertTrue(Startup_Status.report(Startup_Status.BUILDING_NETWORK, 0.6, nodes="8,000", edges="2,987,354"))
                Startup_Status.finish()
                self.assertFalse(Startup_Status.report(Startup_Status.BUILDING_WINDOW, 0.8))  # a window that is up reports no more
            messages = StatusFeed(path).poll()
        self.assertEqual(messages[0], {"text": "Loading PyTorch…", "fraction": 0.45, "values": {}})
        self.assertEqual(messages[1]["values"], {"nodes": "8,000", "edges": "2,987,354"})
        self.assertEqual(len(messages), 2)

    def test_reporting_without_a_splash_costs_nothing_and_writes_nothing(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(Startup_Status.STATUS_FILE_VARIABLE, None)
            self.assertFalse(Startup_Status.report("anything"))

    def test_a_status_file_that_cannot_be_written_does_not_stop_the_start(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {Startup_Status.STATUS_FILE_VARIABLE: directory}  # a folder, not a file
        ):
            self.assertFalse(Startup_Status.report("anything"))

    def test_the_last_lines_of_a_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "application.log"
            path.write_text("\n".join(f"line {number}" for number in range(40)) + "\n\n  \r\n", encoding="utf-8")
            self.assertEqual(tail_lines(path, 3), ["line 37", "line 38", "line 39"])
            self.assertEqual(len(tail_lines(path)), 14)
            self.assertEqual(tail_lines(Path(directory) / "missing.log"), [])
            big = Path(directory) / "big.log"
            big.write_bytes(b"x" * 200000 + "\nthe end \u03b1\n".encode("utf-8"))
            self.assertEqual(tail_lines(big, 1), ["the end \u03b1"])


class WindowCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        Splash.load_fonts(None)

    def splash(self, kind="config", language=None, **options):
        window = SplashWindow(kind, language, **options)
        self.addCleanup(window.deleteLater)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        flush(self.app, 2)
        return window


class StatusTests(WindowCase):
    def test_the_status_is_translated_and_filled_in(self):
        translator = Splash.install_catalog("zh_CN")
        self.assertIsNotNone(translator)
        self.addCleanup(QCoreApplication.removeTranslator, translator)
        window = self.splash("viewer", "zh_CN")
        window.set_status(Startup_Status.LOADING_PYTORCH)
        self.assertNotEqual(window.status, "Loading PyTorch…")
        self.assertIn("PyTorch", window.status)
        window.set_status(Startup_Status.BUILDING_NETWORK, None, nodes="8,000", edges="2,987,354")
        self.assertIn("8,000", window.status)
        self.assertIn("2,987,354", window.status)
        self.assertNotIn("{", window.status)

    def test_a_language_with_its_own_font_registers_it_and_draws_in_it_after_the_core_face(self):
        from PySide6.QtGui import QFontDatabase

        Splash.load_fonts("zh_CN")
        self.addCleanup(Splash.load_fonts, None)
        self.assertEqual(Splash._language_families, ["Noto Sans SC"])
        self.assertIn("Noto Sans SC", QFontDatabase.families())
        self.assertEqual(Splash._font(12).families()[:2], ["Noto Sans", "Noto Sans SC"])
        Splash.load_fonts(None)
        self.assertEqual(Splash._language_families, [])

    def test_english_shows_the_text_as_written_and_a_missing_value_does_not_break_it(self):
        window = self.splash()
        window.set_status(Startup_Status.BUILDING_NETWORK, None, nodes="5", edges="6")
        self.assertEqual(window.status, "Building network display: 5 nodes, 6 edges.")
        window.set_status(Startup_Status.BUILDING_NETWORK, None, nodes="5")
        self.assertIn("{edges}", window.status)

    def test_a_line_the_dependency_check_printed_is_shown_as_it_is(self):
        window = self.splash()
        window.show_text("Validating runtime backend: NVIDIA CUDA 13.2", 0.2)
        self.assertEqual(window.status, "Validating runtime backend: NVIDIA CUDA 13.2")

    def test_the_status_file_is_followed_by_the_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "startup.status"
            window = self.splash(status_file=path)
            with open(path, "ab") as handle:
                handle.write(b'{"text": "Building the window\\u2026", "fraction": 0.8, "values": {}}\n')
            window._tick()
            self.assertEqual(window.status, "Building the window…")
            self.assertEqual(window.target_fraction, 0.8)

    def test_it_starts_with_a_sweeping_bar_and_the_bar_never_goes_back(self):
        window = self.splash()
        self.assertEqual(window.status, "Starting…")
        self.assertIsNone(window.target_fraction)
        window.set_fraction(0.4)
        window.set_fraction(0.2)
        self.assertEqual(window.target_fraction, 0.4)
        window.set_fraction(7)
        self.assertEqual(window.target_fraction, 1.0)

    def test_the_bar_eases_towards_the_milestone_and_creeps_while_it_waits(self):
        window = self.splash()
        window.set_fraction(0.4)
        for _ in range(60):
            window._tick()
        self.assertGreater(window.shown_fraction, 0.35)
        self.assertLessEqual(window.shown_fraction, 0.4 + 0.005)  # news came a moment ago: no creeping yet
        window._news_time -= 20  # a long wait with no news
        for _ in range(80):
            window._tick()
        # It creeps on, well past the milestone but no further than the limit.
        self.assertGreater(window.shown_fraction, 0.4 + Splash.CREEP_LIMIT / 2)
        self.assertLessEqual(window.shown_fraction, 0.4 + Splash.CREEP_LIMIT + 1e-6)
        self.assertLess(window.shown_fraction, 0.97)

    def test_the_progress_bar_is_drawn_in_the_accent_over_a_light_track(self):
        window = self.splash()
        window.set_fraction(0.5)
        window.shown_fraction = 0.5
        image = window.grab().toImage()
        ratio = window.devicePixelRatioF()

        def pixel(x, y):
            return QColor(image.pixel(int(x * ratio), int(y * ratio))).name()

        y = Splash.BAR_Y + Splash.BAR_HEIGHT / 2
        self.assertEqual(pixel(Splash.TEXT_X + 20, y), Splash.SCHEME["accent"])
        self.assertEqual(pixel(Splash.TEXT_X + Splash.TEXT_WIDTH - 20, y), Splash.SCHEME["track"])

    def test_the_window_is_light(self):
        window = self.splash()
        image = window.grab().toImage()
        ratio = window.devicePixelRatioF()

        def pixel(x, y):
            return QColor(image.pixel(int(x * ratio), int(y * ratio))).name()

        # In the left margin, away from the art's glow: the surface colour. Behind the art: a tint of the accent.
        self.assertEqual(pixel(20, 200), Splash.SCHEME["surface"])
        self.assertNotEqual(pixel(330, 250), Splash.SCHEME["surface"])
        self.assertGreater(QColor(pixel(330, 250)).lightness(), 235)


class PlacementTests(WindowCase):
    def test_it_is_a_splash_screen_without_a_taskbar_button(self):
        window = self.splash()
        self.assertTrue(window.windowFlags() & Qt.WindowType.SplashScreen)
        self.assertTrue(window.windowFlags() & Qt.WindowType.FramelessWindowHint)
        self.assertTrue(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)

    @unittest.skipUnless(sys.platform == "win32", "Windows tool windows have no taskbar button")
    def test_on_windows_it_is_a_tool_window_not_an_app_window(self):
        # The offscreen platform has no native windows: Windows' own platform makes the splash's window in a process
        # of its own and reads the styles it got. The window is created and never shown, so nothing appears.
        program = textwrap.dedent(
            """
            import ctypes, sys
            sys.path.insert(0, sys.argv[1])
            from PySide6.QtWidgets import QApplication
            from desktop.Splash import SplashWindow
            app = QApplication([])
            window = SplashWindow("config")
            window.create()
            app.processEvents()
            user32 = ctypes.windll.user32
            user32.GetWindowLongW.restype = ctypes.c_long
            user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
            print(user32.GetWindowLongW(int(window.winId()), -20) & 0xFFFFFFFF)
            """
        )
        environment = {key: value for key, value in os.environ.items() if key not in {"QT_QPA_PLATFORM", "PYTHONPATH"}}
        done = subprocess.run([sys.executable, "-c", program, str(SRC)], capture_output=True, text=True,
                              env=environment, timeout=60, creationflags=0x08000000)
        if done.returncode != 0 or not done.stdout.strip().isdigit():
            self.skipTest(f"no native window here: {done.stderr.strip()[-200:]}")
        extended = int(done.stdout.strip())
        self.assertTrue(extended & 0x80, "WS_EX_TOOLWINDOW")
        self.assertFalse(extended & 0x40000, "WS_EX_APPWINDOW")

    def test_it_opens_centred_on_the_screen_under_the_pointer(self):
        from PySide6.QtCore import QRect

        class Screen:
            def availableGeometry(self):
                return QRect(1920, 0, 1600, 900)

        with mock.patch.object(Splash.QGuiApplication, "screenAt", return_value=Screen()) as at:
            center = Splash.screen_center(QPoint(2500, 300))
        at.assert_called_once()
        self.assertEqual((center.x(), center.y()), (1920 + 799, 449))
        with mock.patch.object(Splash.QGuiApplication, "screenAt", return_value=None):
            fallback = Splash.screen_center(QPoint(-5000, -5000))  # off every screen: the primary one
        primary = QGuiApplication.primaryScreen().availableGeometry()
        self.assertEqual((fallback.x(), fallback.y()), (primary.center().x(), primary.center().y()))

    def test_the_window_is_centred_on_that_point(self):
        window = self.splash()
        window.move_to(QPoint(800, 500))
        self.assertEqual((window.x() + window.width() // 2, window.y() + window.height() // 2), (800, 500))


class ErrorViewTests(WindowCase):
    def test_it_shows_why_the_last_lines_and_three_buttons(self):
        window = self.splash("tools")
        window.show_error(["Traceback (most recent call last):", "ValueError: boom"], "C:/logs/application.log")
        self.assertTrue(window.error)
        self.assertEqual(window.height(), window._section_height())  # the section opened by itself
        self.assertGreater(window.height(), Splash.HEIGHT)
        self.assertEqual(window.log_view.toPlainText(), "Traceback (most recent call last):\nValueError: boom")
        for widget in (window.log_view, window.copy_button, window.log_button, window.close_button):
            self.assertTrue(widget.isVisibleTo(window))
        self.assertEqual([window.copy_button.text(), window.log_button.text(), window.close_button.text()],
                         ["Copy", "Open log", "Close"])
        self.assertIn("EMAP-SSN Tools", window.error_title)
        self.assertIn("stopped before its window opened", window.error_title)

    def test_the_buttons_fit_inside_the_window_and_do_not_overlap(self):
        window = self.splash()
        window.show_error(["x"], "log")
        buttons = [window.copy_button, window.log_button, window.close_button]
        for button in buttons:
            self.assertTrue(window.rect().contains(button.geometry()), button.text())
        for index, first in enumerate(buttons):
            for second in buttons[index + 1:]:
                self.assertFalse(first.geometry().intersects(second.geometry()))
        self.assertTrue(window.rect().contains(window.log_view.geometry()))

    def test_copy_puts_the_reason_and_the_lines_on_the_clipboard(self):
        window = self.splash()
        window.show_error(["line one", "line two"], "log", reason="It could not start.")
        window.copy_button.click()
        self.assertEqual(QGuiApplication.clipboard().text(), "It could not start.\n\nline one\nline two")
        self.assertEqual(window.copy_button.text(), "Copied")

    def test_open_log_opens_the_file_and_is_off_without_one(self):
        window = self.splash()
        log_path = os.path.join(tempfile.gettempdir(), "logs", "application.log")  # a native path: "C:/..." is not one on POSIX
        window.show_error(["x"], log_path)
        with mock.patch.object(Splash.QDesktopServices, "openUrl") as open_url:
            window.log_button.click()
        opened = open_url.call_args.args[0]
        self.assertTrue(opened.isLocalFile())
        self.assertEqual(os.path.normcase(os.path.normpath(opened.toLocalFile())), os.path.normcase(os.path.normpath(log_path)))
        without = self.splash()
        without.show_error(["x"], None)
        self.assertFalse(without.log_button.isEnabled())

    def test_nothing_printed_says_so(self):
        window = self.splash()
        window.show_error([], None)
        self.assertEqual(window.log_view.toPlainText(), "(It printed nothing.)")

    def test_close_and_escape_end_it_and_tell_whoever_waits(self):
        for how in ("button", "escape"):
            with self.subTest(how=how):
                window = self.splash()
                closed = []
                window.closed.connect(lambda: closed.append(True))
                window.show_error(["x"], "log")
                if how == "button":
                    window.close_button.click()
                else:
                    QTest.keyClick(window, Qt.Key.Key_Escape)
                self.assertEqual(closed, [True])
                self.assertFalse(window.isVisible())

    def test_escape_does_nothing_while_it_is_only_a_splash(self):
        window = self.splash()
        QTest.keyClick(window, Qt.Key.Key_Escape)
        self.assertTrue(window.isVisible())

    def test_the_status_stops_following_the_file_once_it_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "startup.status"
            window = self.splash(status_file=path)
            window.show_error(["x"], "log")
            with open(path, "ab") as handle:
                handle.write(b'{"text": "late news"}\n')
            window._tick()
            self.assertNotEqual(window.status, "late news")


class LightnessTests(unittest.TestCase):
    """The splash is on screen long before the windows' heavy imports: the section must not bring them in."""

    def imported_after(self, statement):
        program = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, sys.argv[1])
            %s
            print(sorted(name for name in sys.modules if name.split(".")[0] in
                         ("numpy", "torch", "vispy", "scipy", "pandas", "h5py", "PySide6WebEngine")
                         or name.startswith(("desktop.Desktop_App", "desktop.Console_View", "desktop.Studio_Theme"))))
            """ % statement
        )
        done = subprocess.run([sys.executable, "-c", program, str(SRC)], capture_output=True, text=True, timeout=120,
                              env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH"}},
                              creationflags=0x08000000 if sys.platform == "win32" else 0)
        self.assertEqual(done.returncode, 0, done.stderr)
        return eval(done.stdout.strip().splitlines()[-1])

    def test_the_splash_imports_nothing_heavy_and_not_the_consoles_window(self):
        self.assertEqual(self.imported_after("import desktop.Splash"), [])

    def test_the_line_classifier_the_section_colours_by_is_the_standard_library_only(self):
        self.assertEqual(self.imported_after("import utilities.Line_Kinds, sys; assert 'PySide6' not in sys.modules"), [])

    def test_the_consoles_window_and_the_splash_classify_alike(self):
        from desktop import Console_View
        from utilities import Line_Kinds

        self.assertIs(Console_View.classify, Line_Kinds.classify)
        self.assertIs(Console_View.Classifier, Line_Kinds.Classifier)
        self.assertIs(Splash.Classifier, Line_Kinds.Classifier)


class FakeScreen:
    """A screen with room for ``height`` pixels below its top, for the section that must stay on it."""

    def __init__(self, top, height):
        from PySide6.QtCore import QRect

        self._rect = QRect(0, top, 1920, height)

    def availableGeometry(self):
        return self._rect


class SectionTests(WindowCase):
    """Show details: the splash grows downward into a section with the start's output."""

    def test_it_starts_collapsed_with_only_the_details_button_added(self):
        window = self.splash()
        self.assertEqual((window.width(), window.height()), (Splash.WIDTH, Splash.HEIGHT))
        self.assertFalse(window.details_open)
        self.assertEqual(window.details_button.text(), "Show details")
        self.assertTrue(window.details_button.isVisibleTo(window))
        for widget in (window.details, window.copy_button, window.log_button, window.close_button):
            self.assertFalse(widget.isVisibleTo(window))
        # The button is inside the window, below the bar, clear of the status line and the bar.
        geometry = window.details_button.geometry()
        self.assertTrue(window.rect().contains(geometry))
        self.assertGreater(geometry.top(), Splash.BAR_Y + Splash.BAR_HEIGHT)

    def test_the_link_and_the_buttons_keep_a_clear_margin_from_the_bottom_edge(self):
        window = self.splash()
        roomy = mock.patch.object(Splash.QGuiApplication, "screenAt", return_value=FakeScreen(0, 3000))
        roomy.start()
        self.addCleanup(roomy.stop)

        def gap(widget):
            return window.height() - (widget.geometry().bottom() + 1)

        self.assertGreaterEqual(Splash.BOTTOM_MARGIN, 24)  # about the margin at the left, not the few pixels it was
        self.assertLessEqual(Splash.BOTTOM_MARGIN, 32)
        self.assertEqual(gap(window.details_button), Splash.BOTTOM_MARGIN)
        window.set_details_open(True)
        for button in (window.copy_button, window.log_button):
            self.assertEqual(gap(button), Splash.BOTTOM_MARGIN)  # as clear of the edge as the link was
        window.show_ready()
        self.assertEqual(gap(window.close_button), Splash.BOTTOM_MARGIN)
        self.assertFalse(window.details_button.isVisibleTo(window))  # Ready leaves the section and its buttons

    def test_the_section_is_below_the_splash_and_leaves_room_for_its_view(self):
        window = self.splash()
        roomy = mock.patch.object(Splash.QGuiApplication, "screenAt", return_value=FakeScreen(0, 3000))
        roomy.start()
        self.addCleanup(roomy.stop)
        window.set_details_open(True)
        view = window.details.geometry()
        self.assertGreaterEqual(view.top(), window.details_button.geometry().bottom())
        self.assertGreaterEqual(view.height(), 180)  # the output has room for a dozen lines
        self.assertLess(view.bottom(), window.copy_button.geometry().top())
        self.assertEqual(Splash.HEIGHT, Splash.LINK_Y + Splash.LINK_HEIGHT + Splash.BOTTOM_MARGIN)

    def test_the_button_expands_the_window_downward_and_collapses_it_again(self):
        window = self.splash()
        top, left = window.y(), window.x()
        roomy = mock.patch.object(Splash.QGuiApplication, "screenAt", return_value=FakeScreen(0, 3000))  # not the 800x600 one
        roomy.start()
        self.addCleanup(roomy.stop)
        window.details_button.click()
        self.assertTrue(window.details_open)
        self.assertEqual(window.height(), Splash.HEIGHT + Splash.DETAILS_HEIGHT)
        self.assertEqual((window.x(), window.y()), (left, top))  # it grew downward: the top did not move
        self.assertEqual(window.details_button.text(), "Hide details")
        for widget in (window.details, window.copy_button, window.log_button):
            self.assertTrue(widget.isVisibleTo(window))
            self.assertTrue(window.rect().contains(widget.geometry()))
        self.assertFalse(window.close_button.isVisibleTo(window))  # nothing to close yet: it is still starting
        self.assertGreaterEqual(window.details.geometry().top(), Splash.HEIGHT)  # below the splash proper
        window.details_button.click()
        self.assertFalse(window.details_open)
        self.assertEqual(window.height(), Splash.HEIGHT)
        self.assertEqual((window.x(), window.y()), (left, top))
        self.assertFalse(window.details.isVisibleTo(window))
        self.assertEqual(window.details_button.text(), "Show details")

    def test_the_section_is_as_tall_as_the_screen_allows_and_the_window_moves_up_if_it_must(self):
        window = self.splash()
        window.move(100, 200)
        # Room for 500 px from the screen's top at 0, so the open window (taller) must come up.
        screen = FakeScreen(0, 560)
        with mock.patch.object(Splash.QGuiApplication, "screenAt", return_value=screen):
            window.set_details_open(True)
            self.assertLessEqual(window.y() + window.height(), screen.availableGeometry().bottom() + 1)
            self.assertLess(window.y(), 200)
            self.assertGreaterEqual(window.height(), Splash.HEIGHT + 120)
            self.assertEqual(window.height(), screen.availableGeometry().height() - 16)  # as tall as the screen allows
            window.set_details_open(False)
        self.assertEqual((window.x(), window.y(), window.height()), (100, 200, Splash.HEIGHT))

    def test_lines_the_dependency_check_prints_appear_in_the_section(self):
        window = self.splash()
        window.add_output("Managed Python: C:\\Users\\x\\python.exe")
        window.add_output("Dependency environment is ready (NVIDIA CUDA 12.6).")
        self.assertEqual(window.details.toPlainText(),
                         "Managed Python: C:\\Users\\x\\python.exe\nDependency environment is ready (NVIDIA CUDA 12.6).")

    def test_what_the_window_prints_is_followed_from_its_log_as_it_grows(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "application.log"
            window = self.splash()
            window.follow_log(log)
            window._tick()
            self.assertEqual(window.details.toPlainText(), "")  # no file yet: nothing, and no error
            log.write_bytes(b"Starting EMAPSSN_Config...\nloading torch\n")
            window._tick()
            self.assertEqual(window.details.toPlainText(), "Starting EMAPSSN_Config...\nloading torch")
            with open(log, "ab") as handle:
                handle.write(b"half a li")
            window._tick()
            self.assertEqual(window.details.toPlainText(), "Starting EMAPSSN_Config...\nloading torch")  # not yet a line
            with open(log, "ab") as handle:
                handle.write(b"ne\n\xce\xb1 and more\n")
            window._tick()
            self.assertEqual(window.details.toPlainText().splitlines()[-2:], ["half a line", "\u03b1 and more"])

    def test_a_progress_line_is_one_line_that_changes_until_its_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "application.log"
            window = self.splash()
            window.follow_log(log)
            log.write_bytes(b"start\nDownloading 10%\rDownloading 50%\r")
            window._tick()
            self.assertEqual(window.details.toPlainText().splitlines(), ["start", "Downloading 50%"])
            with open(log, "ab") as handle:
                handle.write(b"Downloading 90%\r")
            window._tick()
            self.assertEqual(window.details.toPlainText().splitlines(), ["start", "Downloading 90%"])
            with open(log, "ab") as handle:
                handle.write(b"Downloading 100%\nnext\n")
            window._tick()
            self.assertEqual(window.details.toPlainText().splitlines(), ["start", "Downloading 100%", "next"])

    def test_the_last_line_with_no_newline_is_read_when_the_log_is_synced(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "application.log"
            window = self.splash()
            window.follow_log(log)
            log.write_bytes(b"one\nthe last words, with no newline")
            window._tick()
            self.assertEqual(window.details.toPlainText(), "one")
            window.sync_log()
            self.assertEqual(window.details.toPlainText(), "one\nthe last words, with no newline")

    def test_warnings_and_errors_are_in_the_splashs_colours_and_a_tracebacks_frames_with_their_error(self):
        window = self.splash()
        for line in ("plain output", "UserWarning: this is deprecated", "Traceback (most recent call last):",
                     '  File "x.py", line 3, in <module>', "    window = Config()", "ValueError: boom", "plain again"):
            window.add_output(line)
        kinds = [window.details.kind_of(index) for index in range(7)]
        self.assertEqual(kinds, ["info", "warning", "error", "error", "error", "error", "info"])
        self.assertEqual(window.details.formats["error"].foreground().color().name(), Splash.SCHEME["danger"])
        self.assertEqual(window.details.formats["warning"].foreground().color().name(), Splash.SCHEME["warning"])
        self.assertEqual(window.details.formats["info"].foreground().color().name(), Splash.SCHEME["text_soft"])

    def test_the_newest_line_stays_in_view_until_the_reader_scrolls_up_and_again_at_the_end(self):
        window = self.splash()
        window.set_details_open(True)
        bar = window.details.verticalScrollBar()
        window.details.add_lines([(f"line {number}", False) for number in range(300)])
        self.assertGreater(bar.maximum(), 0)
        self.assertEqual(bar.value(), bar.maximum())
        bar.setValue(0)
        window.details.add_lines([(f"more {number}", False) for number in range(50)])
        self.assertEqual(bar.value(), 0)  # the reader's place is kept
        bar.setValue(bar.maximum())
        window.details.add_lines([("the newest", False)])
        self.assertEqual(bar.value(), bar.maximum())

    def test_it_keeps_the_last_lines_only_so_a_flood_cannot_grow_without_end(self):
        window = self.splash()
        window.details.add_lines([(f"line {number}", False) for number in range(Splash.MAX_DETAIL_LINES + 500)])
        self.assertLessEqual(window.details.blockCount(), Splash.MAX_DETAIL_LINES)
        self.assertTrue(window.details.toPlainText().endswith(f"line {Splash.MAX_DETAIL_LINES + 499}"))

    def test_copy_puts_the_whole_section_on_the_clipboard_and_open_log_opens_what_is_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "application.log"
            log.write_bytes(b"alpha\nbeta\n")
            window = self.splash()
            window.add_output("from the check")
            window.follow_log(log)
            window._tick()
            window.set_details_open(True)
            window.copy_button.click()
            self.assertEqual(QGuiApplication.clipboard().text(), "from the check\nalpha\nbeta")
            self.assertEqual(window.copy_button.text(), "Copied")
            with mock.patch.object(Splash.QDesktopServices, "openUrl", return_value=True) as open_url:
                window.log_button.click()
            self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()), log)
            with mock.patch.object(Splash.QDesktopServices, "openUrl", return_value=False) as open_url:
                window.log_button.click()  # nothing opens .log files here: its folder opens instead
            self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()), Path(directory))
            self.assertEqual(open_url.call_count, 2)

    def test_open_log_is_off_until_there_is_a_log_to_open(self):
        window = self.splash()
        window.set_details_open(True)
        self.assertFalse(window.log_button.isEnabled())
        window.follow_log("C:/x/application.log")
        self.assertTrue(window.log_button.isEnabled())

    def test_the_collapsed_splash_looks_as_it_did_apart_from_the_button(self):
        window = self.splash()
        window.set_status(Startup_Status.LOADING_PYTORCH, 0.45)
        window.shown_fraction = 0.45
        with_button = window.grab().toImage()
        window.details_button.hide()
        without = window.grab().toImage()
        ratio = window.devicePixelRatioF()
        box = window.details_button.geometry()
        differing = [(x, y) for y in range(with_button.height()) for x in range(with_button.width())
                     if with_button.pixel(x, y) != without.pixel(x, y)]
        self.assertTrue(differing)  # the button is there
        for x, y in differing:
            self.assertTrue(box.contains(int(x / ratio), int(y / ratio)), (x, y))


class ReadyTests(WindowCase):
    """Ready: a splash that is expanded when the window is ready is being watched, and waits for Close."""

    def test_it_says_ready_fills_the_bar_and_offers_close_with_the_section_still_open(self):
        window = self.splash()
        window.set_details_open(True)
        window.set_status(Startup_Status.LOADING_PYTORCH, 0.45)
        window.show_ready()
        self.assertTrue(window.ready)
        self.assertEqual(window.status, "Ready")
        self.assertEqual(window.target_fraction, 1.0)
        self.assertEqual(window.shown_fraction, 1.0)
        self.assertTrue(window.isVisible())  # it does not close itself
        self.assertTrue(window.details_open)
        self.assertTrue(window.close_button.isVisibleTo(window))
        self.assertFalse(window.details_button.isVisibleTo(window))  # the section stays open
        self.assertFalse(window.error)

    def test_close_and_escape_close_it_and_tell_whoever_waits(self):
        for how in ("button", "escape"):
            with self.subTest(how=how):
                window = self.splash()
                closed = []
                window.closed.connect(lambda: closed.append(True))
                window.set_details_open(True)
                window.show_ready()
                self.assertEqual(closed, [])
                if how == "button":
                    window.close_button.click()
                else:
                    QTest.keyClick(window, Qt.Key.Key_Escape)
                self.assertEqual(closed, [True])
                self.assertFalse(window.isVisible())

    def test_the_section_cannot_be_closed_once_ready_so_close_stays_in_reach(self):
        window = self.splash()
        window.set_details_open(True)
        window.show_ready()
        window.set_details_open(False)
        self.assertTrue(window.details_open)
        self.assertTrue(window.close_button.isVisibleTo(window))

    def test_the_output_goes_on_arriving_and_the_status_no_longer_changes_after_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            status, log = Path(directory) / "startup.status", Path(directory) / "application.log"
            window = self.splash(status_file=status)
            window.follow_log(log)
            window.set_details_open(True)
            window.show_ready()
            with open(status, "ab") as handle:
                handle.write(b'{"text": "late news"}\n')
            log.write_bytes(b"printed after ready\n")
            window._tick()
            self.assertEqual(window.status, "Ready")
            self.assertIn("printed after ready", window.details.toPlainText())

    def test_ready_in_chinese_is_translated(self):
        translator = Splash.install_catalog("zh_CN")
        self.assertIsNotNone(translator)
        self.addCleanup(QCoreApplication.removeTranslator, translator)
        window = self.splash(language="zh_CN")
        self.assertEqual(window.details_button.text(), "显示详情")
        window.set_details_open(True)
        self.assertEqual(window.details_button.text(), "隐藏详情")
        window.show_ready()
        self.assertEqual(window.status, "就绪")


class ErrorOpensTheSectionTests(WindowCase):
    def test_the_section_opens_by_itself_with_what_was_printed_and_the_reason_in_the_danger_colour(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "application.log"
            log.write_bytes(b"Starting\nTraceback (most recent call last):\nValueError: boom\n")
            window = self.splash()
            window.add_output("Dependency environment is ready (fake).")
            window.follow_log(log)
            window.sync_log()
            self.assertFalse(window.details_open)
            window.show_error(None, log, reason="The window stopped.")
            self.assertTrue(window.details_open)
            self.assertEqual(window.height(), window._section_height())
            self.assertEqual(window.details.toPlainText().splitlines(),
                             ["Dependency environment is ready (fake).", "Starting",
                              "Traceback (most recent call last):", "ValueError: boom"])
            self.assertEqual(window.details.kind_of(3), "error")
            self.assertTrue(window.close_button.isVisibleTo(window))
            self.assertFalse(window.details_button.isVisibleTo(window))
            self.assertEqual(window.error_title, "The window stopped.")
            window.copy_button.click()
            self.assertTrue(QGuiApplication.clipboard().text().startswith("The window stopped.\n\nDependency"))

    def test_the_section_cannot_be_closed_in_an_error_view(self):
        window = self.splash()
        window.show_error(["x"], "log")
        window.set_details_open(False)
        self.assertTrue(window.details_open)

    def test_lines_given_to_the_error_view_are_added_only_when_the_section_has_none(self):
        window = self.splash()
        window.add_output("already there")
        window.show_error(["given"], "log")
        self.assertEqual(window.details.toPlainText(), "already there")
        empty = self.splash()
        empty.show_error(["given"], "log")
        self.assertEqual(empty.details.toPlainText(), "given")


class WarningTests(WindowCase):
    """A step that went wrong and did not stop the start: the section opens with its lines as a warning."""

    def test_the_section_opens_by_itself_with_the_lines_in_the_warning_colour(self):
        window = self.splash()
        window.add_output("Dependency environment is ready (fake).")
        self.assertFalse(window.details_open)
        window.show_warning(["VR hardware: a plain line that says nothing of a problem", "  indented", "the last"])
        self.assertTrue(window.details_open)
        self.assertEqual(window.height(), window._section_height())
        self.assertEqual(window.details.toPlainText().splitlines()[1:],
                         ["VR hardware: a plain line that says nothing of a problem", "  indented", "the last"])
        self.assertEqual([window.details.kind_of(index) for index in range(4)], ["info", "warning", "warning", "warning"])
        self.assertEqual(window.details.formats["warning"].foreground().color().name(), Splash.SCHEME["warning"])
        self.assertFalse(window.error)
        self.assertFalse(window.ready)

    def test_unlike_an_error_view_it_is_not_locked_and_has_no_close_button_yet(self):
        window = self.splash()
        window.show_warning(["a warning"])
        self.assertTrue(window.details_button.isVisibleTo(window))
        self.assertFalse(window.close_button.isVisibleTo(window))
        window.set_details_open(False)  # the reader may hide it again
        self.assertFalse(window.details_open)
        self.assertEqual(window.height(), Splash.HEIGHT)

    def test_it_stays_at_ready_with_close_when_the_start_ends_with_the_section_open(self):
        window = self.splash()
        window.show_warning(["a warning"])
        window.show_ready()
        self.assertTrue(window.ready)
        self.assertTrue(window.details_open)
        self.assertEqual(window.status, "Ready")
        self.assertTrue(window.close_button.isVisibleTo(window))
        self.assertEqual(window.details.kind_of(0), "warning")

    def test_long_lines_wrap_to_the_sections_width_with_their_indent_and_nothing_is_cut(self):
        window = self.splash()
        sentence = "Intel integrated graphics cannot drive a headset and neither can anything else here at all. " * 3
        window.show_warning(["  [no] " + sentence.strip(), "no-spaces-" + "x" * 120])  # (a hyphen is no place to break)
        lines = window.details.toPlainText().splitlines()
        self.assertGreater(len(lines), 3)
        for line in lines[:-1]:
            self.assertLessEqual(len(line), Splash.WARNING_COLUMNS, line)
        self.assertTrue(lines[0].startswith("  [no] Intel"))
        self.assertTrue(lines[1].startswith("    "))  # the continuation is indented under it
        self.assertEqual(" ".join(part.strip() for part in lines[:-1]).replace("  ", " "),
                         ("[no] " + sentence.strip()).replace("  ", " "))
        self.assertEqual(lines[-1], "no-spaces-" + "x" * 120)  # a word is never broken
        monospace_columns = Splash.WIDTH - 2 * Splash.DETAILS_MARGIN
        metrics = window.details.fontMetrics()
        self.assertLess(metrics.horizontalAdvance("x" * Splash.WARNING_COLUMNS), monospace_columns - 24)

    def test_a_forced_kind_colours_lines_whatever_they_say_and_plain_lines_still_follow_their_words(self):
        window = self.splash()
        window.add_output("looks plain", kind=Splash.ERROR)
        window.add_output("UserWarning: reads as a warning")
        window.details.add_lines([("an error by decree", False)], kind=Splash.WARNING)
        self.assertEqual([window.details.kind_of(index) for index in range(3)], ["error", "warning", "warning"])


class ButtonLookTests(WindowCase):
    """The Show details link, and the buttons of the section, look the same on every splash and on every platform.

    The Viewer's splash (a session, started from Config) reached the user with its link drawn as a tinted button, the
    Configuration and Tools splashes with a plain link: the session's splash process is started with Config's
    environment, which on a Wayland desktop makes it an X11 window, and an X11 window is activated when it opens,
    which gave the link the keyboard focus, and the focus background of its style sheet.
    """

    def properties(self, button):
        return {
            "text": button.text(), "font": button.font().toString(), "style": button.styleSheet(),
            "geometry": button.geometry().getRect(), "icon": (button.iconSize().width(), button.iconSize().height()), "cursor": button.cursor().shape(),
            "focus_policy": button.focusPolicy(),
        }

    def shown(self, kind):
        """A splash a platform has activated, as an X11 window is: the offscreen platform activates what it shows."""
        window = SplashWindow(kind)
        self.addCleanup(window.deleteLater)
        window.show()
        flush(self.app, 3)
        self.assertTrue(window.isActiveWindow(), "the offscreen platform no longer activates a shown window")
        return window

    def test_every_kind_builds_the_link_and_the_buttons_the_same_way(self):
        windows = {kind: self.splash(kind) for kind in Splash.KINDS}
        first = windows[Splash.KINDS[0]]
        for kind, window in windows.items():
            for name in ("details_button", "copy_button", "log_button", "close_button"):
                with self.subTest(kind=kind, button=name):
                    self.assertEqual(self.properties(getattr(window, name)), self.properties(getattr(first, name)))

    def test_the_link_is_drawn_the_same_whether_or_not_the_platform_activates_the_splash(self):
        for kind in Splash.KINDS:
            with self.subTest(kind=kind):
                quiet = self.splash(kind)
                activated = self.shown(kind)
                self.assertFalse(activated.details_button.hasFocus())
                self.assertIsNone(QApplication.focusWidget())
                self.assertEqual(activated.details_button.grab().toImage(), quiet.details_button.grab().toImage())

    def test_a_click_still_focuses_the_link_and_the_error_view_focuses_close(self):
        window = self.shown("viewer")
        self.assertEqual(window.details_button.focusPolicy(), Qt.FocusPolicy.ClickFocus)
        QTest.mouseClick(window.details_button, Qt.MouseButton.LeftButton)
        flush(self.app, 2)
        self.assertTrue(window.details_open)
        window.show_error(["it stopped"], None)
        flush(self.app, 2)
        self.assertTrue(window.close_button.hasFocus())  # the error view asks for the focus itself


class ArtClearanceTests(WindowCase):
    """The art keeps clear of the text and the bar: at least ART_CLEARANCE logical pixels between what is drawn."""

    LONG_STATUS = "Managed Python: C:\\Users\\someone\\Documents\\EMAP-SSN\\.venv\\Scripts\\python.exe " * 4
    LONG_REASON = "EMAP-SSN Configuration stopped before its window opened, and this is a reason long enough to wrap " * 3

    def test_the_art_is_a_square_of_about_270_centred_vertically_and_as_far_from_the_right_edge_as_set(self):
        art = Splash.art_rect()
        self.assertEqual((art.width(), art.height()), (Splash.ART_SIZE, Splash.ART_SIZE))
        self.assertEqual(Splash.ART_SIZE, 270)
        self.assertEqual(art.top(), Splash.HEIGHT - art.bottom())  # as much room above as below
        self.assertEqual(Splash.WIDTH - art.right(), Splash.ART_RIGHT_MARGIN)
        self.assertGreaterEqual(art.left(), Splash.TEXT_X + Splash.TEXT_WIDTH)  # right of the whole text column

    def layer(self, window, paint, ratio):
        """What ``paint`` draws of the splash, as a mask of the pixels that are visibly there, at a device ratio."""
        import numpy

        from PySide6.QtGui import QImage, QPainter

        image = QImage(int(Splash.WIDTH * ratio), int(Splash.HEIGHT * ratio), QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        paint(painter)
        painter.end()
        pixels = numpy.frombuffer(image.constBits(), dtype=numpy.uint8).reshape(image.height(), image.bytesPerLine())
        alpha = pixels[:, : image.width() * 4].reshape(image.height(), image.width(), 4)[:, :, 3]
        return alpha > 48  # the glow behind the art is fainter than that

    def gap(self, window, ratio):
        import numpy
        from scipy import ndimage

        art = self.layer(window, window._paint_art, ratio)
        text = self.layer(window, window._paint_text, ratio)
        self.assertTrue(art.any() and text.any())
        distance = ndimage.distance_transform_edt(~text)  # from every pixel to the nearest text or bar pixel
        return float(distance[art].min()) / ratio

    def states(self, window):
        yield "starting", lambda: window.set_status("Starting\u2026", 0.4)
        yield "a long status", lambda: window.show_text(self.LONG_STATUS, 0.3)
        yield "ready", lambda: window.show_ready()
        yield "an error", lambda: window.show_error([self.LONG_REASON], None, reason=self.LONG_REASON)

    def test_the_art_of_every_window_is_clear_of_the_text_and_the_bar_in_every_state_language_and_scale(self):
        for language in (None, "zh_CN"):
            translator = Splash.install_catalog(language)
            Splash.load_fonts(language)
            self.addCleanup(Splash.load_fonts, None)
            if translator is not None:
                self.addCleanup(QCoreApplication.removeTranslator, translator)
            for kind in Splash.KINDS:
                for index in range(4):
                    window = self.splash(kind, language)
                    name, change = list(self.states(window))[index]
                    change()
                    for ratio in (1, 2):
                        with self.subTest(kind=kind, language=language, state=name, ratio=ratio):
                            self.assertGreaterEqual(self.gap(window, ratio), Splash.ART_CLEARANCE)

    def test_the_drawn_art_is_as_far_from_the_right_edge_as_the_text_is_from_the_left_and_centred_vertically(self):
        import numpy

        for kind in Splash.KINDS:
            with self.subTest(kind=kind):
                window = self.splash(kind)
                art = self.layer(window, window._paint_art, 1)
                rows, columns = numpy.nonzero(art)
                self.assertLessEqual(abs((Splash.WIDTH - 1 - columns.max()) - Splash.TEXT_X), 8)
                self.assertLessEqual(abs(rows.min() - (Splash.HEIGHT - 1 - rows.max())), 6)

    def test_a_long_status_and_the_error_reason_end_short_of_the_bar(self):
        import numpy

        for index, name in ((1, "a long status"), (3, "an error")):
            window = self.splash("config")
            change = list(self.states(window))[index][1]
            change()
            text = self.layer(window, window._paint_text, 1)
            rows = text[262:318, :] if name == "an error" else text[286:310, :]
            columns = numpy.nonzero(rows.any(axis=0))[0]
            with self.subTest(state=name):
                self.assertLessEqual(columns.max(), Splash.TEXT_X + Splash.STATUS_WIDTH)
        self.assertLess(Splash.STATUS_WIDTH, Splash.TEXT_WIDTH)

    def test_the_old_art_size_would_have_failed_it(self):
        # 300 px, 30 px from the right: the top-left cluster was 6 px from the kicker line, and a long status or the
        # error view's reason touched the art.
        for index in range(4):
            window = self.splash("config")
            name, change = list(self.states(window))[index]
            change()
            with self.subTest(state=name), mock.patch.object(Splash, "ART_SIZE", 300):
                self.assertLess(self.gap(window, 1), Splash.ART_CLEARANCE)


class PluginKindTests(WindowCase):
    """A program beside this one adds a window with register_kind and its own catalogs with install_extra_catalogs."""

    def setUp(self):
        # Cleanups run last in, first out: the windows go first (deleteLater), a flush makes that happen, and only
        # then does the registration go, so no window paints a kind that is no longer there.
        self.addCleanup(Splash.SUBTITLES.pop, "plug", None)
        self.addCleanup(Splash.LOGO_FILES.pop, "plug", None)
        self.addCleanup(self.delete_windows)

    def delete_windows(self):
        from PySide6.QtCore import QEvent

        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def test_a_registered_window_is_named_under_the_product_and_draws_its_own_art(self):
        with self.assertRaises(ValueError):
            SplashWindow("plug")
        art = Splash.LOGOS / "splash_logo_tool_512.png"
        Splash.register_kind("plug", "Plug-in Configuration", art)
        window = self.splash("plug")
        self.assertEqual(Splash.SUBTITLES["plug"], "Plug-in Configuration")
        self.assertEqual(Splash.LOGO_FILES["plug"], str(art))
        self.assertFalse(window.logo.isNull())
        self.assertEqual(window.logo.toImage(), QPixmap(str(art)).toImage())
        self.assertNotIn("plug", Splash.KINDS)  # the built-in windows are what the standalone command line offers

    def test_a_second_registration_replaces_the_first(self):
        Splash.register_kind("plug", "One", Splash.LOGOS / "splash_logo_tool_512.png")
        Splash.register_kind("plug", "Two", Splash.LOGOS / "splash_logo_viewer_512.png")
        self.assertEqual(Splash.SUBTITLES["plug"], "Two")
        self.assertTrue(Splash.LOGO_FILES["plug"].endswith("splash_logo_viewer_512.png"))

    def test_the_name_is_translated_in_the_splash_context_and_a_long_one_shrinks_to_fit(self):
        Splash.register_kind("plug", "Plug-in Configuration", Splash.LOGOS / "splash_logo_tool_512.png")
        window = self.splash("plug")
        from PySide6.QtGui import QFontMetricsF

        text = Splash.tr(Splash.SUBTITLES["plug"])
        font = Splash._fitted_font(text, Splash.TEXT_WIDTH, 30, 20)
        self.assertLess(font.pixelSize(), 30)
        self.assertGreater(font.pixelSize(), 20)
        self.assertLessEqual(QFontMetricsF(font).horizontalAdvance(text), Splash.TEXT_WIDTH)
        smaller = Splash._font(font.pixelSize() + 1, True)  # the largest size that fits
        self.assertGreater(QFontMetricsF(smaller).horizontalAdvance(text), Splash.TEXT_WIDTH)
        self.assertEqual(Splash._fitted_font("Configuration", Splash.TEXT_WIDTH, 30, 20).pixelSize(), 30)  # as ever
        too_long = "A Very Long Name Of A Window Configuration Tool"
        self.assertEqual(Splash._fitted_font(too_long, Splash.TEXT_WIDTH, 30, 20).pixelSize(), 20)  # never smaller
        window.grab()  # it draws

    def test_extra_catalogs_are_installed_after_the_main_one_and_a_missing_one_leaves_english(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            folder = Path(directory) / "zh_CN"
            folder.mkdir()
            import shutil

            shutil.copyfile(Splash.catalog_path("zh_CN"), folder / "extra_zh_CN.qm")
            translators = Splash.install_extra_catalogs("zh_CN", [(directory, "extra"), (directory, "absent")])
            try:
                self.assertEqual(len(translators), 1)
                self.assertEqual(Splash.tr("Ready"), "就绪")  # it came from the extra catalog alone
            finally:
                for translator in translators:
                    QCoreApplication.removeTranslator(translator)
            self.assertEqual(Splash.tr("Ready"), "Ready")
        self.assertEqual(Splash.install_extra_catalogs(None, [(".", "extra")]), [])
        self.assertEqual(Splash.install_extra_catalogs("zh_CN", []), [])


class ProcessHandOffTests(unittest.TestCase):
    """The splash as a process of its own: it follows a status file and ends when the window writes its ready file."""

    def run_splash(self, *options, directory):
        environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        environment["QT_QPA_PLATFORM"] = "offscreen"
        return subprocess.Popen(
            [sys.executable, str(SRC / "desktop" / "Splash.py"), "--hidden", *options],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )

    def test_it_ends_after_the_ready_file_appears_and_not_before(self):
        with tempfile.TemporaryDirectory() as directory:
            status, ready = Path(directory) / "startup.status", Path(directory) / "gui.ready"
            process = self.run_splash("--kind", "viewer", "--language", "zh_CN", "--status-file", str(status),
                                      "--ready-file", str(ready), directory=directory)
            try:
                first = process.stdout.readline()
                self.assertEqual(json.loads(first)["language"], "zh_CN")
                with open(status, "ab") as handle:
                    handle.write(b'{"text": "Loading PyTorch\\u2026", "fraction": 0.4}\n')
                time.sleep(0.6)
                self.assertIsNone(process.poll(), "it ended before the window was ready")
                ready.write_text("ready\n", encoding="utf-8")
                self.assertEqual(process.wait(timeout=20), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_with_the_section_open_it_stays_after_the_ready_file_appears_saying_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            ready, log = Path(directory) / "gui.ready", Path(directory) / "application.log"
            log.write_text("the window prints\n", encoding="utf-8")
            process = self.run_splash("--kind", "config", "--details", "--tail", str(log), "--ready-file", str(ready),
                                      directory=directory)
            try:
                process.stdout.readline()
                time.sleep(0.4)
                ready.write_text("ready\n", encoding="utf-8")
                time.sleep(0.8)
                self.assertIsNone(process.poll(), "the splash closed itself while the user was watching")
            finally:
                process.kill()
                process.communicate()

    def test_the_error_view_stays_after_a_ready_file_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            log, ready = Path(directory) / "application.log", Path(directory) / "gui.ready"
            log.write_text("Traceback\nValueError: boom\n", encoding="utf-8")
            process = self.run_splash("--kind", "tools", "--error-log", str(log), "--ready-file", str(ready),
                                      directory=directory)
            try:
                process.stdout.readline()
                ready.write_text("ready\n", encoding="utf-8")
                time.sleep(0.6)
                self.assertIsNone(process.poll(), "the error view closed itself")
            finally:
                process.kill()
                process.communicate()

    def test_a_shot_of_the_section_open_and_of_ready_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "application.log"
            log.write_text("Starting EMAPSSN_Config...\nUserWarning: a warning\n", encoding="utf-8")
            for name, extra in (("open", ("--details",)), ("ready", ("--details", "--ready"))):
                target = Path(directory) / f"{name}.png"
                process = self.run_splash("--kind", "tools", "--tail", str(log), *extra, "--shot", str(target),
                                          directory=directory)
                out, err = process.communicate(timeout=60)
                self.assertEqual(process.returncode, 0, err)
                self.assertGreater(target.stat().st_size, 5000, name)

    def test_a_shot_renders_each_window_in_each_language(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in Splash.KINDS:
                for language in ("en", "zh_CN"):
                    target = Path(directory) / f"{kind}_{language}.png"
                    process = self.run_splash("--kind", kind, "--language", language, "--status", "Loading PyTorch…",
                                              "--shot", str(target), directory=directory)
                    out, err = process.communicate(timeout=60)
                    self.assertEqual(process.returncode, 0, err)
                    self.assertGreater(target.stat().st_size, 5000, (kind, language))


if __name__ == "__main__":
    unittest.main()
