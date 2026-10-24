# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The start-up splash (desktop.Splash) and the status the windows report to it (utilities.Startup_Status): the light colours are the Studio theme's, the language follows the saved setting as the windows do, a status file is read line by line as it grows, texts are translated and filled in, the bar fills as the start passes its milestones and never goes back, the window has no taskbar button and opens on the screen under the pointer, and the error view shows the last lines with Copy, Open log and Close. The splash process hands over to the window it waits for: it closes when the ready file appears, and an error view stays."""

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
            "field_border": "field_border", "danger": "danger",
        }
        for ours, theirs in pairs.items():
            with self.subTest(colour=ours):
                self.assertEqual(Splash.SCHEME[ours], TOKENS[theirs])


class IdentityTests(unittest.TestCase):
    def test_the_splash_names_the_product_and_version_the_windows_do(self):
        self.assertEqual(Splash.PRODUCT, Desktop_App.PRODUCT_NAME)
        self.assertEqual(Splash.VERSION, Desktop_App.APPLICATION_VERSION)

    def test_every_art_file_the_splash_names_is_bundled_small(self):
        for kind, name in Splash.LOGO_FILES.items():
            path = Splash.LOGOS / name
            with self.subTest(kind=kind):
                self.assertTrue(path.is_file())
                self.assertLess(path.stat().st_size, 200 * 1024)  # a 512 px copy: the shipped art decodes to 75 MB
                self.assertEqual((QPixmap(str(path)).width(), QPixmap(str(path)).height()), (512, 512))


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
        self.assertEqual(window.height(), Splash.ERROR_HEIGHT)
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
        window.show_error(["x"], "C:/logs/application.log")
        with mock.patch.object(Splash.QDesktopServices, "openUrl") as open_url:
            window.log_button.click()
        self.assertEqual(open_url.call_args.args[0].toLocalFile().replace("\\", "/"), "C:/logs/application.log")
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
