# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The start-up splash (desktop.Splash) and the status the windows report to it (utilities.Startup_Status): the light colours are the Studio theme's, its accents (the kicker, bar, link, focus, soft fill and the art's glow) are in the theme colour the window it is for has saved, purple where that is missing, unreadable or unknown, the language follows the saved setting as the windows do, a status file is read line by line as it grows, texts are translated and filled in, the bar fills as the start passes its milestones and never goes back, the window is an ordinary frameless one, not a splash-screen window: it has a taskbar button with the small icon of the window it is for, in that window's taskbar group, goes behind other windows and comes forward when clicked, is dragged by the system's own move from anywhere but its buttons, the link and the section's text, and opens on the screen under the pointer, and the error view shows the last lines with Copy, Open log and Close. A step of the start that goes wrong without stopping it opens the section by itself with its lines in the warning colour, wrapped to the section's width. A program beside this one adds a window (its name and art) and its own translation catalogs. The splash process hands over to the window it waits for: it closes when the ready file appears, and an error view stays."""

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

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSize, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QCloseEvent, QColor, QEnterEvent, QGuiApplication, QIcon, QImage, QMouseEvent, QPainter, QPixmap, QWindow,
)
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QStyleFactory  # noqa: E402

from desktop import Desktop_App, Splash  # noqa: E402
from desktop.Splash import SplashWindow, StatusFeed, tail_lines  # noqa: E402
from desktop.Studio_Theme import TOKENS, studio_stylesheet, theme_tokens  # noqa: E402
from desktop.Theme_Colors import (  # noqa: E402
    BLACK_ACCENTS, DEFAULT_THEME_COLOR, THEME_COLORS, scroll_bar_rules, shade_set,
)
from tests.theme_fixture import contrast  # noqa: E402
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


class ThemeColourSchemeTests(unittest.TestCase):
    """The splash's accents are the family's shades, the windows' own; the greys and the status colours do not move."""

    ACCENT_ROLES = {"accent", "accent_soft"}

    def test_every_family_takes_the_accent_shades_the_windows_take(self):
        for color in THEME_COLORS:
            with self.subTest(color=color):
                scheme, tokens = Splash.scheme_for(color), theme_tokens(color)
                self.assertEqual(scheme["accent"], tokens["accent"])
                self.assertEqual(scheme["accent_soft"], tokens["accent_soft"])

    def test_black_takes_the_windows_black_accents_not_a_hue(self):
        scheme = Splash.scheme_for("black")
        self.assertEqual(scheme["accent"], BLACK_ACCENTS["light"]["accent"])
        self.assertEqual(scheme["accent_soft"], BLACK_ACCENTS["light"]["accent_soft"])
        self.assertEqual(scheme["accent"], theme_tokens("black")["accent"])
        for color in THEME_COLORS:
            if color != "black":
                self.assertNotEqual(Splash.scheme_for(color)["accent"], scheme["accent"])

    def test_only_the_accent_roles_change_with_the_family(self):
        purple = Splash.scheme_for("purple")
        self.assertEqual(purple, Splash.SCHEME)  # the default scheme is purple's
        self.assertEqual(Splash.scheme_for(), purple)
        for color in THEME_COLORS:
            scheme = Splash.scheme_for(color)
            changed = {role for role in scheme if scheme[role] != purple[role]}
            with self.subTest(color=color):
                self.assertEqual(scheme.keys(), purple.keys())
                self.assertEqual(changed, set() if color == "purple" else self.ACCENT_ROLES)

    def test_the_greys_the_text_and_the_status_colours_are_the_windows_in_every_family(self):
        for color in THEME_COLORS:
            scheme, tokens = Splash.scheme_for(color), theme_tokens(color)
            for role, token in Splash.SCHEME_TOKENS.items():
                if role not in self.ACCENT_ROLES:
                    with self.subTest(color=color, role=role):
                        self.assertEqual(scheme[role], tokens[token])
        for role, color in (("danger", "#dc2626"), ("warning", "#b45309"), ("text", "#09090b"), ("surface", "#ffffff")):
            self.assertEqual({Splash.scheme_for(name)[role] for name in THEME_COLORS}, {color})

    def test_the_kicker_and_the_link_stay_readable_in_every_family(self):
        for color in THEME_COLORS:
            scheme = Splash.scheme_for(color)
            with self.subTest(color=color):
                # The windows' own rule (tests/test_theme_colors.py): the accent on white, 4.5:1.
                self.assertGreaterEqual(contrast(scheme["accent"], scheme["surface"]), 4.5)
                # The link under the mouse or focused sits on the soft fill: still four to one in the palest families.
                self.assertGreaterEqual(contrast(scheme["accent"], scheme["accent_soft"]), 4.0)

    def test_a_name_that_is_not_a_family_is_refused(self):
        with self.assertRaises(ValueError):
            Splash.scheme_for("mauve")

    def test_a_scheme_is_cheap_enough_for_the_first_paint(self):
        # Every family's scheme, twenty times over: a few milliseconds (the splash makes two at start-up). The bound only
        # catches the scheme growing into work that belongs after the first paint.
        started = time.perf_counter()
        for _ in range(20):
            for color in THEME_COLORS:
                Splash.scheme_for(color)
        self.assertLess(time.perf_counter() - started, 1.0)


class SavedThemeColourTests(unittest.TestCase):
    """Which family the splash takes: what the window's own setting says, else purple, and never a dialog."""

    def saved(self, content, setting="CONFIG_THEME_COLOR"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app_settings.json"
            if isinstance(content, bytes):
                path.write_bytes(content)
            elif content is not None:
                path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
            return Splash.saved_theme_color(setting, {"SSN_APP_SETTINGS_PATH": str(path)})

    def test_each_setting_gives_the_colour_saved_under_its_own_name(self):
        content = {"CONFIG_THEME_COLOR": "teal", "TOOLS_THEME_COLOR": "orange", "PLUG_THEME_COLOR": "pink",
                   "LANGUAGE": "zh_CN"}
        self.assertEqual(self.saved(content, "CONFIG_THEME_COLOR"), "teal")
        self.assertEqual(self.saved(content, "TOOLS_THEME_COLOR"), "orange")
        self.assertEqual(self.saved(content, "PLUG_THEME_COLOR"), "pink")
        self.assertEqual(self.saved(content, "LANGUAGE"), DEFAULT_THEME_COLOR)  # a setting that is not a colour
        self.assertEqual(self.saved(content, "OTHER_THEME_COLOR"), DEFAULT_THEME_COLOR)  # one nobody saved

    def test_every_family_comes_back_as_saved(self):
        for color in THEME_COLORS:
            with self.subTest(color=color):
                self.assertEqual(self.saved({"TOOLS_THEME_COLOR": color}, "TOOLS_THEME_COLOR"), color)

    def test_no_setting_name_is_the_default_whatever_is_saved(self):
        self.assertEqual(Splash.saved_theme_color(None), DEFAULT_THEME_COLOR)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app_settings.json"
            path.write_text(json.dumps({"CONFIG_THEME_COLOR": "red"}), encoding="utf-8")
            self.assertEqual(Splash.saved_theme_color(None, {"SSN_APP_SETTINGS_PATH": str(path)}), DEFAULT_THEME_COLOR)

    def test_a_missing_unreadable_or_unknown_colour_is_purple(self):
        cases = {
            "no file": None,
            "not JSON": "{ not json",
            "empty file": "",
            "not an object": json.dumps(["CONFIG_THEME_COLOR", "teal"]),
            "a string": json.dumps("teal"),
            "not UTF-8": b"\xff\xfe\x00{",
            "no setting": {},
            "another window's setting only": {"TOOLS_THEME_COLOR": "teal"},
            "unknown name": {"CONFIG_THEME_COLOR": "mauve"},
            "wrong case": {"CONFIG_THEME_COLOR": "Teal"},
            "empty name": {"CONFIG_THEME_COLOR": ""},
            "a number": {"CONFIG_THEME_COLOR": 5},
            "null": {"CONFIG_THEME_COLOR": None},
            "a list": {"CONFIG_THEME_COLOR": ["teal"]},
            "an object": {"CONFIG_THEME_COLOR": {"teal": 1}},
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self.saved(content), DEFAULT_THEME_COLOR)

    def test_a_settings_path_that_is_a_folder_is_purple_too(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(Splash.saved_theme_color("CONFIG_THEME_COLOR", {"SSN_APP_SETTINGS_PATH": directory}),
                             DEFAULT_THEME_COLOR)

    def test_it_never_raises_or_writes_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app_settings.json"
            path.write_text("{ not json", encoding="utf-8")
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            Splash.saved_theme_color("CONFIG_THEME_COLOR", {"SSN_APP_SETTINGS_PATH": str(path)})
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_it_gives_what_the_windows_take_from_the_same_settings(self):
        values = ["teal", "black", "purple", "mauve", "Teal", "", 5, None, "pink"]
        for value in values:
            settings = {"CONFIG_THEME_COLOR": value}
            with self.subTest(value=value), mock.patch.object(Desktop_App, "_warn_once"):
                self.assertEqual(self.saved(settings), Desktop_App.configured_theme_color("CONFIG_THEME_COLOR", settings))
        with mock.patch.object(Desktop_App, "_warn_once"):
            self.assertEqual(Desktop_App.configured_theme_color("CONFIG_THEME_COLOR", {}), self.saved({}))

    def test_the_language_reader_is_unchanged_by_sharing_the_file_reader(self):
        self.assertEqual(Splash.saved_setting("X", {"SSN_APP_SETTINGS_PATH": "no/such/file.json"}), None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app_settings.json"
            environment = {"SSN_APP_SETTINGS_PATH": str(path)}
            path.write_text(json.dumps({"LANGUAGE": "zh_CN", "N": 3}), encoding="utf-8")
            self.assertEqual(Splash.saved_setting("N", environment), 3)
            self.assertEqual(Splash.saved_language_setting(environment), "zh_CN")
            path.write_text(json.dumps({"LANGUAGE": 7}), encoding="utf-8")
            self.assertEqual(Splash.saved_language_setting(environment), "system")


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


class ThemeColourWindowTests(WindowCase):
    """Every part of the splash that is in the accent is in the window's family, and nothing else moves."""

    def picture(self, window):
        image = window.grab().toImage()
        ratio = window.devicePixelRatioF()
        return lambda x, y: QColor(image.pixel(int(x * ratio), int(y * ratio)))

    def chevron_colours(self, window):
        """The colours of the Show/Hide details chevron's fully covered pixels."""
        icon = window.details_button.icon().pixmap(QSize(12, 12)).toImage()
        colours = (QColor.fromRgba(icon.pixel(x, y)) for x in range(icon.width()) for y in range(icon.height()))
        return {colour.name() for colour in colours if colour.alpha() == 255}

    def test_without_a_colour_the_splash_is_purple(self):
        window = self.splash()
        self.assertEqual(window.theme_color, DEFAULT_THEME_COLOR)
        self.assertEqual(window.scheme, Splash.SCHEME)

    def test_a_name_that_is_not_a_family_is_refused(self):
        with self.assertRaises(ValueError):
            SplashWindow("config", theme_color="mauve")

    def test_the_bar_is_drawn_in_the_families_accent(self):
        for color in THEME_COLORS:
            window = self.splash(theme_color=color)
            window.set_fraction(0.5)
            window.shown_fraction = 0.5
            pixel = self.picture(window)
            with self.subTest(color=color):
                self.assertEqual(pixel(Splash.TEXT_X + 20, Splash.BAR_Y + Splash.BAR_HEIGHT / 2).name(),
                                 theme_tokens(color)["accent"])
                self.assertEqual(pixel(Splash.TEXT_X + Splash.TEXT_WIDTH - 20, Splash.BAR_Y + 1).name(),
                                 Splash.SCHEME["track"])  # the track is a grey in every family

    def test_the_kicker_is_drawn_in_the_families_accent(self):
        for color in THEME_COLORS:
            window = self.splash(theme_color=color)
            image = window.grab().toImage()
            ratio = window.devicePixelRatioF()
            accent = theme_tokens(color)["accent"]
            kicker = {QColor(image.pixel(int(x * ratio), int(y * ratio))).name()
                      for x in range(int(Splash.TEXT_X), int(Splash.TEXT_X + 300)) for y in range(36, 58)}
            with self.subTest(color=color):
                self.assertIn(accent, kicker)  # the strokes of its letters, fully covered
                # And it reads on what is drawn behind it (the surface, with a trace of the glow): the windows' 4.5:1.
                self.assertGreaterEqual(contrast(accent, self.picture(window)(330, 47).name()), 4.5)

    def test_the_glow_behind_the_art_is_a_tint_of_the_families_accent(self):
        for color in THEME_COLORS:
            window = self.splash(theme_color=color)
            window.logo = QPixmap()  # the glow alone
            image = QImage(Splash.WIDTH, Splash.HEIGHT, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(QColor(Splash.SCHEME["surface"]))
            painter = QPainter(image)
            window._paint_art(painter)
            painter.end()
            centre = QColor(image.pixel(Splash.WIDTH - 180, Splash.HEIGHT // 2))
            accent, surface = QColor(theme_tokens(color)["accent"]), QColor(Splash.SCHEME["surface"])
            with self.subTest(color=color):
                for channel in ("red", "green", "blue"):
                    expected = getattr(surface, channel)() * 0.86 + getattr(accent, channel)() * 0.14
                    self.assertAlmostEqual(getattr(centre, channel)(), expected, delta=2.5)

    def test_the_link_the_button_focus_and_the_sections_selection_take_the_family(self):
        for color in THEME_COLORS:
            window = self.splash(theme_color=color)
            scheme = Splash.scheme_for(color)
            with self.subTest(color=color):
                link = window.details_button.styleSheet()
                self.assertIn(f"color: {scheme['accent']};", link)
                self.assertEqual(link.count(f"background: {scheme['accent_soft']};"), 2)  # hover and focus
                for button in (window.copy_button, window.log_button, window.close_button):
                    self.assertIn(f"border: 2px solid {scheme['accent']};", button.styleSheet())
                    if color != "purple":
                        self.assertNotIn(Splash.SCHEME["accent"], button.styleSheet())  # no purple left behind
                self.assertIn(f"selection-background-color: {scheme['accent_soft']};", window.details.styleSheet())
                self.assertEqual(self.chevron_colours(window), {scheme["accent"]})  # the chevron is in the accent

    def test_the_open_section_is_in_the_family_too(self):
        window = self.splash(theme_color="teal")
        window.set_details_open(True)
        flush(self.app, 2)
        self.assertEqual(self.chevron_colours(window), {theme_tokens("teal")["accent"]})  # the up chevron, redrawn
        self.assertIn(f"border: 2px solid {theme_tokens('teal')['accent']};", window.close_button.styleSheet())

    def test_what_is_not_an_accent_does_not_follow_the_family(self):
        reference = self.picture(self.splash())
        for color in ("teal", "black"):
            window = self.splash(theme_color=color)
            pixel = self.picture(window)
            with self.subTest(color=color):
                for point in ((20, 200), (20, 20), (Splash.TEXT_X + Splash.TEXT_WIDTH - 20, Splash.BAR_Y + 1)):
                    self.assertEqual(pixel(*point).name(), reference(*point).name())  # surface and track
                self.assertEqual(window.colors["danger"].name(), Splash.SCHEME["danger"])
                self.assertEqual(window.colors["warning"].name(), Splash.SCHEME["warning"])
                self.assertEqual(window.colors["text"].name(), Splash.SCHEME["text"])

    def test_the_warning_and_error_lines_keep_their_colours_in_every_family(self):
        for color in THEME_COLORS:
            window = self.splash(theme_color=color)
            with self.subTest(color=color):
                self.assertEqual(window.details.formats["warning"].foreground().color().name(), Splash.SCHEME["warning"])
                self.assertEqual(window.details.formats["error"].foreground().color().name(), Splash.SCHEME["danger"])
                self.assertEqual(window.details.formats["info"].foreground().color().name(), Splash.SCHEME["text_soft"])


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
    def test_it_is_an_ordinary_frameless_window_not_a_splash_screen_one(self):
        window = self.splash()
        flags = window.windowFlags()
        self.assertEqual(window.windowType(), Qt.WindowType.Window)  # X11: _NET_WM_WINDOW_TYPE_NORMAL, which docks list and group
        for kind in (Qt.WindowType.SplashScreen, Qt.WindowType.Tool, Qt.WindowType.Dialog, Qt.WindowType.Popup, Qt.WindowType.ToolTip):
            self.assertNotEqual(window.windowType(), kind)
        self.assertTrue(flags & Qt.WindowType.FramelessWindowHint)
        self.assertTrue(flags & Qt.WindowType.NoDropShadowWindowHint)
        self.assertTrue(window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))  # the rounded corners
        self.assertEqual((window.width(), window.height()), (Splash.WIDTH, Splash.HEIGHT))

    def test_it_opens_without_taking_the_keyboard_focus_and_a_click_activates_it(self):
        window = self.splash()
        self.assertTrue(window.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating))
        self.assertFalse(window.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus)  # a click activates it as any window's
        self.assertEqual(window.details_button.focusPolicy(), Qt.FocusPolicy.ClickFocus)

    def test_it_stays_above_other_windows_only_for_the_first_moments_where_that_is_needed(self):
        # Linux (X11): a window opened without the focus is stacked below the focused one, so the splash opens above and
        # becomes a normal window after OPEN_ON_TOP_MS. Windows puts it in front by itself; Wayland cannot stack at all.
        with mock.patch.object(Splash, "OPEN_ON_TOP_MS", 0):
            window = self.splash()
            self.assertFalse(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
            self.assertFalse(window.windowHandle().flags() & Qt.WindowType.WindowStaysOnTopHint)
        timers = []
        with mock.patch.object(Splash, "OPEN_ON_TOP_MS", 2500), \
                mock.patch.object(Splash.QTimer, "singleShot", lambda milliseconds, callback: timers.append((milliseconds, callback))):
            window = self.splash()
            self.assertTrue(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
            self.assertTrue(window.windowHandle().flags() & Qt.WindowType.WindowStaysOnTopHint)
            self.assertEqual([milliseconds for milliseconds, _ in timers], [2500])
            window.hide()
            window.show()
            self.assertEqual(len(timers), 1)  # once, though it is shown again
            timers[0][1]()
            self.assertFalse(window.windowHandle().flags() & Qt.WindowType.WindowStaysOnTopHint)
            self.assertTrue(window.isVisible())  # the window manager is told; no hide and show, so nothing flickers

    def test_the_first_moments_are_on_linux_only(self):
        self.assertEqual(Splash.OPEN_ON_TOP_MS > 0, sys.platform.startswith("linux"))

    def test_the_end_of_the_first_moments_after_the_window_has_gone_is_harmless(self):
        import shiboken6

        window = SplashWindow("config")
        window._open_as_normal_window()  # never shown: no window handle yet
        shiboken6.delete(window)
        window._open_as_normal_window()  # the C++ object is gone: the timer fires into nothing

    def test_the_error_view_still_raises_and_activates_the_splash(self):
        window = self.splash()
        with mock.patch.object(window, "raise_") as raised, mock.patch.object(window, "activateWindow") as activated:
            window.show_error(["It printed this."], None)
        raised.assert_called_once_with()
        activated.assert_called_once_with()

    def test_nothing_else_raises_or_activates_it(self):
        window = self.splash()
        with mock.patch.object(window, "raise_") as raised, mock.patch.object(window, "activateWindow") as activated:
            window.set_status("Starting…", 0.2)
            window.show_warning(["A step went wrong."])
            window.show_ready()
            window._tick()
        raised.assert_not_called()
        activated.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "the taskbar button and its identity are Windows'")
    def test_on_windows_it_is_an_app_window_that_goes_behind_others_with_its_windows_taskbar_identity(self):
        # The offscreen platform has no native windows: Windows' own platform makes the splash's window in a process
        # of its own and reads the styles and the taskbar properties it got. The window is created and never shown, so
        # nothing appears.
        program = textwrap.dedent(
            """
            import ctypes, json, sys
            sys.path.insert(0, sys.argv[1])
            from PySide6.QtWidgets import QApplication
            from desktop import Splash, Taskbar_Identity as T
            app = QApplication([])
            Splash.load_fonts(None)
            found = {}
            user32 = ctypes.windll.user32
            user32.GetWindowLongW.restype = ctypes.c_long
            user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
            shell32, ole32 = ctypes.windll.shell32, ctypes.windll.ole32
            shell32.SHGetPropertyStoreForWindow.argtypes = [ctypes.c_void_p, ctypes.POINTER(T._Guid), ctypes.POINTER(ctypes.c_void_p)]
            shell32.SHGetPropertyStoreForWindow.restype = ctypes.c_long
            ole32.PropVariantClear.argtypes = [ctypes.POINTER(T._PropVariant)]
            for kind in sys.argv[2:]:
                window = Splash.SplashWindow(kind)
                window.create()
                app.processEvents()
                hwnd = int(window.winId())
                store = ctypes.c_void_p()
                iid = T._guid(T._PROPERTY_STORE_IID)
                assert shell32.SHGetPropertyStoreForWindow(hwnd, ctypes.byref(iid), ctypes.byref(store)) == 0
                methods = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
                get_value = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(T._PropertyKey),
                                               ctypes.POINTER(T._PropVariant))(methods[5])
                texts = {}
                for property_id in (2, 3, 4, 5):
                    key = T._PropertyKey(T._guid(T._TASKBAR_PROPERTY_FORMAT), property_id)
                    value = T._PropVariant()
                    get_value(store, ctypes.byref(key), ctypes.byref(value))
                    texts[property_id] = ctypes.wstring_at(value.pointer) if value.type == T._VT_LPWSTR else None
                    ole32.PropVariantClear(ctypes.byref(value))
                found[kind] = {"ex": user32.GetWindowLongW(hwnd, -20) & 0xFFFFFFFF, "texts": texts, "title": window.windowTitle()}
            print(json.dumps(found))
            """
        )
        environment = {key: value for key, value in os.environ.items() if key not in {"QT_QPA_PLATFORM", "PYTHONPATH"}}
        done = subprocess.run([sys.executable, "-c", program, str(SRC), "config", "tools", "viewer"], capture_output=True,
                              text=True, env=environment, timeout=60, creationflags=0x08000000)
        if done.returncode != 0 or not done.stdout.strip().startswith("{"):
            self.skipTest(f"no native window here: {done.stderr.strip()[-200:]}")
        found = json.loads(done.stdout.strip().splitlines()[-1])
        from desktop import Taskbar_Identity

        for kind, word, icon, name in (("config", "viewer", "viewer_logo.ico", "EMAP-SSN Configuration"),
                                       ("tools", "tools", "tool_logo.ico", "EMAP-SSN Tools"),
                                       ("viewer", None, "viewer_logo.ico", "EMAP-SSN Viewer")):
            with self.subTest(kind=kind):
                got = found[kind]
                self.assertFalse(got["ex"] & 0x80, "WS_EX_TOOLWINDOW: a tool window has no taskbar button")
                self.assertFalse(got["ex"] & 0x8, "WS_EX_TOPMOST: it would stay above the other windows")
                self.assertFalse(got["ex"] & 0x08000000, "WS_EX_NOACTIVATE: a click would not bring it forward")
                self.assertEqual(got["title"], name)
                texts = {int(key): value for key, value in got["texts"].items()}
                self.assertEqual(texts[5], Taskbar_Identity.APP_USER_MODEL_IDS[kind])  # the window's own ID
                if word is None:  # the Viewer has no way of being started again: its ID only
                    self.assertEqual((texts[2], texts[3], texts[4]), (None, None, None))
                else:
                    self.assertEqual(texts[2], Taskbar_Identity.launcher_relaunch_command(word))
                    self.assertEqual(texts[3], str(Splash.LOGOS / icon) + ",0")
                    self.assertEqual(texts[4], name)

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


class TaskbarTests(WindowCase):
    """The splash's taskbar button: the small icon of the window it is for, in that window's taskbar group, named as the window is."""

    WINDOW_SOURCES = {"config": "EMAPSSN_Config.py", "tools": "EMAPSSN_Tools.py", "viewer": "EMAPSSN_Viewer.py"}

    def test_every_built_in_kind_has_the_small_icon_and_the_identity_of_its_window(self):
        from desktop.Taskbar_Identity import APP_USER_MODEL_IDS, launcher_relaunch_command

        expected = {"config": ("viewer_logo.ico", APP_USER_MODEL_IDS["config"], launcher_relaunch_command("viewer")),
                    "tools": ("tool_logo.ico", APP_USER_MODEL_IDS["tools"], launcher_relaunch_command("tools")),
                    "viewer": ("viewer_logo.ico", APP_USER_MODEL_IDS["viewer"], None)}  # the Viewer cannot be started again
        self.assertEqual(set(Splash.TASKBARS) & set(Splash.KINDS), set(Splash.KINDS))
        for kind, (icon, app_id, relaunch) in expected.items():
            with self.subTest(kind=kind):
                taskbar = Splash.TASKBARS[kind]
                self.assertEqual((taskbar.icon, taskbar.app_id, taskbar.relaunch), (icon, app_id, relaunch))
                self.assertTrue((Splash.LOGOS / icon).is_file())
                self.assertEqual((Splash.LOGOS / icon).suffix, ".ico")

    def test_the_icon_is_the_one_the_window_itself_loads_never_the_large_art_of_the_shortcuts(self):
        for kind, name in self.WINDOW_SOURCES.items():
            with self.subTest(kind=kind):
                icon = Splash.TASKBARS[kind].icon
                self.assertNotIn("large", icon)
                self.assertIn(f'"{icon}"', (SRC / name).read_text(encoding="utf-8"))

    def test_the_splash_sets_the_id_and_the_start_command_the_window_sets(self):
        import ast

        for kind, word in (("config", "viewer"), ("tools", "tools")):
            with self.subTest(kind=kind):
                tree = ast.parse((SRC / self.WINDOW_SOURCES[kind]).read_text(encoding="utf-8"))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "TaskbarIdentity"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(ast.unparse(calls[0].args[1]), f"APP_USER_MODEL_IDS['{kind}']")
                self.assertEqual(ast.unparse(calls[0].args[4]), f"launcher_relaunch_command('{word}')")
        # The Viewer sets only its process's ID, so its splash sets the same ID and nothing to start it again.
        source = (SRC / self.WINDOW_SOURCES["viewer"]).read_text(encoding="utf-8")
        self.assertNotIn("TaskbarIdentity", source)
        self.assertIn("set_app_user_model_id(APP_USER_MODEL_IDS[\"viewer\"])", source)

    def test_the_name_is_the_windows_own_in_english_and_in_chinese(self):
        names = {"config": ("Config", Desktop_App.CONFIG_DISPLAY_NAME), "tools": ("Tools", Desktop_App.TOOLS_DISPLAY_NAME),
                 "viewer": ("Viewer", Desktop_App.VIEWER_DISPLAY_NAME)}
        translator = Splash.install_catalog("zh_CN")
        self.assertIsNotNone(translator)
        self.addCleanup(QCoreApplication.removeTranslator, translator)
        for language in ("zh_CN", None):
            if language is None:
                QCoreApplication.removeTranslator(translator)
            for kind, (context, text) in names.items():
                with self.subTest(kind=kind, language=language):
                    self.assertEqual(Splash.window_name(kind), QCoreApplication.translate(context, text))
        self.assertEqual(Splash.window_name("config"), "EMAP-SSN Configuration")

    def test_the_window_has_that_name_as_its_title_and_the_small_icon(self):
        window = self.splash("tools")
        self.assertEqual(window.windowTitle(), "EMAP-SSN Tools")
        self.assertFalse(window.windowIcon().isNull())
        self.assertEqual(window.windowIcon().pixmap(16).toImage(), QIcon(str(Splash.LOGOS / "tool_logo.ico")).pixmap(16).toImage())

    def test_the_identity_goes_on_the_window_through_the_windows_own_helper(self):
        from desktop.Taskbar_Identity import APP_USER_MODEL_IDS, launcher_relaunch_command

        with mock.patch.object(Splash, "TaskbarIdentity") as identity:
            window = SplashWindow("config")
            self.addCleanup(window.deleteLater)
            viewer = SplashWindow("viewer")
            self.addCleanup(viewer.deleteLater)
        self.assertEqual(identity.call_args_list, [
            mock.call(window, APP_USER_MODEL_IDS["config"], str(Splash.LOGOS / "viewer_logo.ico"), "EMAP-SSN Configuration",
                      launcher_relaunch_command("viewer")),
            mock.call(viewer, APP_USER_MODEL_IDS["viewer"], str(Splash.LOGOS / "viewer_logo.ico"), "EMAP-SSN Viewer", None)])

    def test_off_windows_the_helper_does_nothing_and_the_window_class_is_the_launchers(self):
        # Linux: the window class and the Wayland app id are the process's (Desktop_Launcher.configure_desktop_identity gives
        # them the kind's .desktop name, as the windows have); the splash adds none of its own, and sets no property.
        from desktop import Taskbar_Identity

        with mock.patch.object(Taskbar_Identity.sys, "platform", "linux"), \
                mock.patch.object(Taskbar_Identity, "set_window_taskbar_identity") as setter:
            window = self.splash("config")
            window.winId()
        setter.assert_not_called()


class DragTests(WindowCase):
    """A press on the splash but not on a button, the link or the section's text drags it by the system's own move."""

    def press(self, window, point, button=Qt.MouseButton.LeftButton, started=True):
        with mock.patch.object(QWindow, "startSystemMove", return_value=started) as move:
            QTest.mousePress(window, button, Qt.KeyboardModifier.NoModifier, point)
            QTest.mouseRelease(window, button, Qt.KeyboardModifier.NoModifier, point)
        return move.call_count

    def test_a_press_on_the_text_the_art_or_the_margin_starts_the_move(self):
        window = self.splash()
        art = Splash.art_rect().center().toPoint()
        for point in (QPoint(20, 20), QPoint(int(Splash.TEXT_X) + 30, 85), art, QPoint(Splash.WIDTH - 5, Splash.HEIGHT - 5),
                      QPoint(int(Splash.TEXT_X), int(Splash.BAR_Y))):
            with self.subTest(point=point):
                self.assertIsNone(window.childAt(point))
                self.assertEqual(self.press(window, point), 1)

    def test_it_uses_the_window_system_so_it_works_on_windows_x11_and_wayland(self):
        window = self.splash()
        with mock.patch.object(QWindow, "startSystemMove", return_value=True) as move:
            QTest.mousePress(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(20, 20))
        self.assertEqual(move.call_args.args, ())  # no arguments: the system takes the pointer

    def test_a_press_on_the_details_link_does_not(self):
        window = self.splash()
        self.assertTrue(window.details_button.isVisible())
        self.assertEqual(self.press(window, window.details_button.geometry().center()), 0)
        self.assertEqual(self.press(window, window.details_button.geometry().topLeft() + QPoint(2, 2)), 0)

    def test_a_press_in_the_section_or_on_its_buttons_does_not_and_one_beside_them_does(self):
        window = self.splash()
        window.set_details_open(True)
        flush(self.app)
        self.assertEqual(self.press(window, window.details.geometry().center()), 0)  # its text is selected by dragging
        self.assertEqual(self.press(window, window.details.geometry().topLeft() + QPoint(3, 3)), 0)
        for button in (window.copy_button, window.log_button):  # Open log is off: a disabled button is a button
            with self.subTest(button=button.text()):
                self.assertTrue(button.isVisible())
                self.assertEqual(self.press(window, button.geometry().center()), 0)
        self.assertEqual(self.press(window, QPoint(8, Splash.HEIGHT + 100)), 1)  # the section's margin
        self.assertEqual(self.press(window, QPoint(Splash.WIDTH // 2, window.height() - 6)), 1)

    def test_a_press_on_the_close_button_of_the_error_view_and_the_ready_view_does_not(self):
        for state in ("error", "ready"):
            with self.subTest(state=state):
                window = self.splash()
                if state == "error":
                    window.show_error(["It printed this."], None)
                else:
                    window.set_details_open(True)
                    window.show_ready()
                flush(self.app)
                self.assertTrue(window.close_button.isVisible())
                self.assertEqual(self.press(window, window.close_button.geometry().center()), 0)
                self.assertEqual(self.press(window, QPoint(20, 20)), 1)

    def test_only_the_left_button_drags(self):
        window = self.splash()
        for button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            with self.subTest(button=button):
                self.assertEqual(self.press(window, QPoint(20, 20), button), 0)

    def test_a_system_that_cannot_move_a_window_leaves_the_press_unhandled_and_nothing_breaks(self):
        window = self.splash()
        self.assertEqual(self.press(window, QPoint(20, 20), started=False), 1)
        self.assertTrue(window.isVisible())
        self.assertEqual((window.width(), window.height()), (Splash.WIDTH, Splash.HEIGHT))

    def test_a_press_before_the_window_has_a_native_window_is_ignored(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QMouseEvent, QPointingDevice

        window = SplashWindow("config")
        self.addCleanup(window.deleteLater)
        self.assertIsNone(window.windowHandle())
        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(20, 20), QPointF(20, 20), Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointingDevice.primaryPointingDevice())
        window.mousePressEvent(event)  # no handle to ask the system with: nothing happens


class CloseTests(WindowCase):
    """The taskbar's Close window and Alt+F4 close the splash only once there is something to close it for."""

    class FromTheSystem(QCloseEvent):
        def spontaneous(self):
            return True

    def request(self, window):
        event = self.FromTheSystem()
        window.closeEvent(event)
        return event

    def test_while_the_window_is_starting_the_systems_request_is_refused_and_the_start_is_still_followed(self):
        window = self.splash()
        closed = []
        window.closed.connect(lambda: closed.append(True))
        event = self.request(window)
        self.assertFalse(event.isAccepted())
        self.assertEqual(closed, [])
        self.assertTrue(window.timer.isActive())  # the status goes on: a start that fails still reaches the error view

    def test_the_launchers_own_close_is_not_the_systems_and_goes_through(self):
        window = self.splash()
        closed = []
        window.closed.connect(lambda: closed.append(True))
        window.close()
        self.assertEqual(closed, [True])
        self.assertFalse(window.timer.isActive())

    def test_the_error_view_and_the_ready_view_close_from_the_taskbar_too(self):
        for state in ("error", "ready"):
            with self.subTest(state=state):
                window = self.splash()
                closed = []
                window.closed.connect(lambda: closed.append(True))
                if state == "error":
                    window.show_error(["It printed this."], None)
                else:
                    window.set_details_open(True)
                    window.show_ready()
                event = self.request(window)
                self.assertTrue(event.isAccepted())
                self.assertEqual(closed, [True])


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
                         or name.startswith(("desktop.Desktop_App", "desktop.Console_View", "desktop.Studio_Theme",
                                             "PySide6.QtSvg"))))
            """ % statement
        )
        done = subprocess.run([sys.executable, "-c", program, str(SRC)], capture_output=True, text=True, timeout=120,
                              env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH"}},
                              creationflags=0x08000000 if sys.platform == "win32" else 0)
        self.assertEqual(done.returncode, 0, done.stderr)
        return eval(done.stdout.strip().splitlines()[-1])

    def test_the_splash_imports_nothing_heavy_and_not_the_consoles_window(self):
        self.assertEqual(self.imported_after("import desktop.Splash"), [])

    def test_the_colours_come_from_the_qt_free_module_the_studio_theme_shares_not_from_the_theme_itself(self):
        # The first-paint guard of the theme colour: Studio_Theme (QtSvg, the icon engine) would cost the splash its start,
        # so it takes the colour maths from desktop/Theme_Colors.py, which imports no Qt at all.
        self.assertEqual(self.imported_after("import desktop.Theme_Colors, sys; assert 'PySide6' not in sys.modules"), [])
        self.assertEqual(self.imported_after(
            "import desktop.Splash, sys; assert 'desktop.Theme_Colors' in sys.modules; "
            "assert 'desktop.Studio_Theme' not in sys.modules and 'PySide6.QtSvg' not in sys.modules"), [])

    def test_the_taskbar_identity_is_the_light_module_the_windows_share_not_their_app_module(self):
        # The windows' own module (Desktop_App, with the settings and the theme behind it) would cost the splash a tenth of a
        # second of its start; the identity they share with it is in a module of ctypes and QtCore.
        self.assertEqual(self.imported_after(
            "import desktop.Splash, sys; assert 'desktop.Taskbar_Identity' in sys.modules; "
            "assert 'utilities.App_Settings' not in sys.modules and 'PySide6.QtNetwork' not in sys.modules"), [])

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


class SectionScrollBarTests(WindowCase):
    """The section's scroll bars are the windows': Studio_Theme's rules in its colours, whatever style the platform draws with.

    The splash is a process of its own that does not import Studio_Theme, so the rules are one text in the Qt-free
    Theme_Colors that both take (scroll_bar_rules); the pictures here are what the section's two bars show.
    """

    LINES = [(f"[{i:02d}] " + "a long line of output that runs past the edge " * 5, False) for i in range(80)]
    DARK = shade_set("dark_page", "purple")  # the pages' dark mode: the splash has no dark scheme, the view follows its own

    def scheme(self, dark=False):
        if not dark:
            return Splash.SCHEME
        return {**Splash.SCHEME, "scroll": self.DARK["scroll"], "scroll_hover": self.DARK["text_disabled"]}

    def show(self, view):
        self.addCleanup(view.deleteLater)
        view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        view.resize(420, 170)
        view.show()
        if isinstance(view, Splash.DetailsView):
            view.add_lines(self.LINES)
        else:
            view.setPlainText("\n".join(text for text, _ in self.LINES))
        flush(self.app, 3)
        self.assertTrue(view.verticalScrollBar().isVisible() and view.horizontalScrollBar().isVisible())
        return view

    def pixels(self, view, bar, colour, at="min"):
        """The pixels of ``colour`` the picture of ``view`` has in ``bar``'s box with the bar at its ``at`` end.

        Returns (left, right, top, bottom, count) relative to the box, or None where there is no such pixel.
        """
        bar.setValue(bar.minimum() if at == "min" else bar.maximum())
        flush(self.app, 2)
        self.assertEqual(view.devicePixelRatioF(), 1.0)  # the extents below are in pixels
        image = view.grab().toImage()
        origin = bar.mapTo(view, QPoint(0, 0))
        wanted = QColor(colour).rgb()
        found = [(x - origin.x(), y - origin.y())
                 for x in range(origin.x(), origin.x() + bar.width()) for y in range(origin.y(), origin.y() + bar.height())
                 if image.pixelColor(x, y).rgb() == wanted]
        if not found:
            return None
        return (min(x for x, _ in found), max(x for x, _ in found), min(y for _, y in found), max(y for _, y in found),
                len(found))

    def hover(self, bar, point):
        """Put the mouse over ``point`` (in the bar's own coordinates), as the system's pointer does."""
        centre = QPointF(point)
        self.app.sendEvent(bar, QEnterEvent(centre, centre, QPointF(bar.mapToGlobal(point))))
        self.app.sendEvent(bar, QMouseEvent(QEvent.Type.MouseMove, centre, QPointF(bar.mapToGlobal(point)),
                                            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
        flush(self.app, 2)

    def use_every_style(self):
        """The names of the styles this platform has; the application is back in its own when the test ends."""
        self.addCleanup(self.app.setStyle, self.app.style().objectName())
        return QStyleFactory.keys()

    def test_the_scroll_colours_are_the_windows_in_every_family(self):
        for color in THEME_COLORS:
            scheme, tokens = Splash.scheme_for(color), theme_tokens(color)
            with self.subTest(color=color):
                self.assertEqual(scheme["scroll"], tokens["scroll"])
                self.assertEqual(scheme["scroll_hover"], tokens["text_disabled"])  # what the windows' rule draws under the mouse
                self.assertEqual((scheme["scroll"], scheme["scroll_hover"]), ("#d4d4d8", "#a1a1aa"))

    def test_the_pages_hover_colour_is_the_one_the_windows_draw_in_light_and_dark(self):
        for color in THEME_COLORS:
            with self.subTest(color=color):
                self.assertEqual(shade_set("light_page", color)["scroll_hover"], Splash.scheme_for(color)["scroll_hover"])
                self.assertEqual(shade_set("dark_page", color)["scroll_hover"], shade_set("dark_page", color)["text_disabled"])

    def test_the_view_carries_the_windows_scroll_bar_rules_in_every_family(self):
        for color in THEME_COLORS:
            window = self.splash(theme_color=color)
            tokens = theme_tokens(color)
            rules = scroll_bar_rules(tokens["scroll"], tokens["text_disabled"])
            sheet = window.details.styleSheet()
            with self.subTest(color=color):
                self.assertIn(rules, sheet)
                self.assertIn(rules, studio_stylesheet(tokens))  # the very text the windows' application stylesheet has
                for line in (
                    "QScrollBar:vertical { background: transparent; width: 10px; margin: 6px 2px 6px 0px; }",
                    "QScrollBar:horizontal { background: transparent; height: 10px; margin: 0px 6px 2px 6px; }",
                    "QScrollBar::handle:vertical { background: #d4d4d8; border-radius: 4px; min-height: 32px; }",
                    "QScrollBar::handle:horizontal { background: #d4d4d8; border-radius: 4px; min-width: 32px; }",
                    "QScrollBar::handle:vertical:hover { background: #a1a1aa; }",
                    "QScrollBar::handle:horizontal:hover { background: #a1a1aa; }",
                    "QScrollBar::add-line, QScrollBar::sub-line { height: 0px; width: 0px; }",
                    "QScrollBar::add-page, QScrollBar::sub-page { background: none; }",
                    "QAbstractScrollArea::corner { background: transparent; border: none; }",
                ):
                    self.assertIn(line, sheet)

    def test_the_view_takes_its_bar_colours_from_the_scheme_it_is_given_light_and_dark(self):
        for dark in (False, True):
            scheme = self.scheme(dark)
            sheet = Splash.DetailsView(None, scheme).styleSheet()
            with self.subTest(dark=dark):
                self.assertIn(scroll_bar_rules(scheme["scroll"], scheme["scroll_hover"]), sheet)
                self.assertEqual(sheet.count(f"background: {scheme['scroll']};"), 2)
                self.assertEqual(sheet.count(f"background: {scheme['scroll_hover']};"), 2)
        self.assertEqual(self.scheme(True)["scroll"], "#3f3f46")
        self.assertEqual(self.scheme(True)["scroll_hover"], "#52525b")

    def test_the_handles_are_drawn_in_the_scroll_colour_ten_wide_with_no_arrow_buttons_in_every_style(self):
        for dark in (False, True):
            scheme = self.scheme(dark)
            measured = {}
            for style in self.use_every_style():
                self.app.setStyle(style)
                view = self.show(Splash.DetailsView(None, scheme))
                vertical, horizontal = view.verticalScrollBar(), view.horizontalScrollBar()
                with self.subTest(dark=dark, style=style):
                    # The bars are 10 px thick, as the rules say; the handle fills them but the 2 px margin beside the view.
                    self.assertEqual((vertical.width(), horizontal.height()), (10, 10))
                    top = self.pixels(view, vertical, scheme["scroll"], "min")
                    bottom = self.pixels(view, vertical, scheme["scroll"], "max")
                    left = self.pixels(view, horizontal, scheme["scroll"], "min")
                    right = self.pixels(view, horizontal, scheme["scroll"], "max")
                    for found in (top, bottom, left, right):
                        self.assertIsNotNone(found, "no pixel of the scroll colour: the platform's own bar is drawn")
                    self.assertEqual(top[:2], (0, 7))      # 8 px wide, in the bar's first columns (the margin is on the right)
                    self.assertEqual(left[2:4], (0, 7))    # 8 px thick, in the bar's first rows (the margin is below)
                    # No arrow button: the handle starts a margin (6 px, and the rounding of its corners) from the bar's ends.
                    self.assertLessEqual(top[2], 9)
                    self.assertLessEqual(vertical.height() - 1 - bottom[3], 9)
                    self.assertLessEqual(left[0], 9)
                    self.assertLessEqual(horizontal.width() - 1 - right[1], 9)
                    measured[style] = (top, bottom, left, right)
            self.assertEqual(len({repr(found) for found in measured.values()}), 1, measured)  # the style makes no difference

    def test_the_handle_is_the_one_a_window_draws_pixel_for_pixel(self):
        # A plain text box under the windows' own stylesheet is what Config, Tools and the Viewer show.
        reference = QPlainTextEdit()
        reference.setReadOnly(True)
        reference.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        reference.setStyleSheet(studio_stylesheet())
        self.show(reference)
        view = self.show(Splash.DetailsView(None, Splash.SCHEME))

        def shape(found):  # where its columns are, how many rows it covers and how many pixels: not where it is along the bar
            return found[0], found[1], found[3] - found[2], found[4]

        for at in ("min", "max"):
            with self.subTest(at=at):
                ours = self.pixels(view, view.verticalScrollBar(), TOKENS["scroll"], at)
                theirs = self.pixels(reference, reference.verticalScrollBar(), TOKENS["scroll"], at)
                self.assertIsNotNone(theirs)
                self.assertIsNotNone(ours, "no pixel of the scroll colour: the platform's own bar is drawn")
                self.assertEqual(shape(ours), shape(theirs))

    def test_the_handle_takes_the_hover_colour_under_the_mouse_light_and_dark(self):
        for dark in (False, True):
            scheme = self.scheme(dark)
            view = self.show(Splash.DetailsView(None, scheme))
            for bar, point in ((view.verticalScrollBar(), QPoint(4, 20)), (view.horizontalScrollBar(), QPoint(20, 4))):
                bar.setValue(bar.minimum())
                flush(self.app, 2)
                origin = bar.mapTo(view, QPoint(0, 0))
                with self.subTest(dark=dark, bar=bar.orientation().name):
                    self.assertEqual(view.grab().toImage().pixelColor(origin + point).name(), scheme["scroll"])
                    self.hover(bar, point)
                    self.assertEqual(view.grab().toImage().pixelColor(origin + point).name(), scheme["scroll_hover"])


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
        self.addCleanup(Splash.TASKBARS.pop, "plug", None)
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

    def test_a_registered_window_brings_its_own_small_icon_and_taskbar_identity(self):
        from desktop.Taskbar_Identity import launcher_relaunch_command

        icon = Splash.LOGOS / "tool_logo.ico"
        taskbar = Splash.Taskbar(icon, "ca.test.plug.window", launcher_relaunch_command("plug", Path("plug.bat"), ""))
        Splash.register_kind("plug", "Plug-in Configuration", Splash.LOGOS / "splash_logo_tool_512.png", taskbar=taskbar)
        self.assertIs(Splash.TASKBARS["plug"], taskbar)
        with mock.patch.object(Splash, "TaskbarIdentity") as identity:
            window = self.splash("plug")
        identity.assert_called_once_with(window, "ca.test.plug.window", str(icon), "EMAP-SSN Plug-in Configuration",
                                         taskbar.relaunch)
        self.assertEqual(window.windowTitle(), "EMAP-SSN Plug-in Configuration")
        self.assertFalse(window.windowIcon().isNull())

    def test_a_window_registered_without_a_taskbar_has_a_title_and_no_icon_or_identity(self):
        Splash.register_kind("plug", "Plug-in Configuration", Splash.LOGOS / "splash_logo_tool_512.png")
        self.assertNotIn("plug", Splash.TASKBARS)
        with mock.patch.object(Splash, "TaskbarIdentity") as identity:
            window = self.splash("plug")
        identity.assert_not_called()
        self.assertEqual(window.windowTitle(), "EMAP-SSN Plug-in Configuration")
        self.assertTrue(window.windowIcon().isNull())

    def test_registering_again_without_a_taskbar_takes_the_first_ones_away(self):
        art = Splash.LOGOS / "splash_logo_tool_512.png"
        Splash.register_kind("plug", "One", art, taskbar=Splash.Taskbar("tool_logo.ico", "ca.test.plug.window"))
        Splash.register_kind("plug", "Two", art)
        self.assertNotIn("plug", Splash.TASKBARS)

    def test_an_icon_that_is_missing_or_not_an_ico_never_stops_the_splash_and_sets_no_identity(self):
        art = Splash.LOGOS / "splash_logo_tool_512.png"
        for icon, has_icon in (("no_such_icon.ico", False), (str(art), True)):  # an .ico is what the taskbar reads
            with self.subTest(icon=icon):
                Splash.register_kind("plug", "Plug-in Configuration", art, taskbar=Splash.Taskbar(icon, "ca.test.plug.window"))
                with mock.patch.object(Splash, "TaskbarIdentity") as identity:
                    window = self.splash("plug")
                identity.assert_not_called()
                self.assertEqual(not window.windowIcon().isNull(), has_icon)

    def test_an_icon_with_no_id_is_the_windows_icon_and_nothing_more(self):
        Splash.register_kind("plug", "Plug-in Configuration", Splash.LOGOS / "splash_logo_tool_512.png",
                             taskbar=Splash.Taskbar("tool_logo.ico"))
        with mock.patch.object(Splash, "TaskbarIdentity") as identity:
            window = self.splash("plug")
        identity.assert_not_called()
        self.assertFalse(window.windowIcon().isNull())

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

    def run_splash(self, *options, directory, settings=None):
        environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        environment["QT_QPA_PLATFORM"] = "offscreen"
        if settings is not None:
            environment["SSN_APP_SETTINGS_PATH"] = str(settings)
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

    def test_a_shot_takes_the_colour_saved_under_the_setting_it_is_told(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / "app_settings.json"
            settings.write_text(json.dumps({"CONFIG_THEME_COLOR": "teal", "TOOLS_THEME_COLOR": "black"}), encoding="utf-8")

            def bar(kind, *options, saved=settings):
                target = Path(directory) / f"shot_{len(list(Path(directory).glob('*.png')))}.png"
                process = self.run_splash("--kind", kind, "--language", "en", "--status", "Loading PyTorch…",
                                          "--fraction", "0.5", *options, "--shot", str(target), directory=directory,
                                          settings=saved)
                out, err = process.communicate(timeout=60)
                self.assertEqual(process.returncode, 0, err)
                image = QImage(str(target))
                return image.pixelColor(int(Splash.TEXT_X) + 20, int(Splash.BAR_Y) + 1).name()

            self.assertEqual(bar("config", "--theme-setting", "CONFIG_THEME_COLOR"), theme_tokens("teal")["accent"])
            self.assertEqual(bar("tools", "--theme-setting", "TOOLS_THEME_COLOR"), theme_tokens("black")["accent"])
            # No setting named, one nobody saved, a colour the program does not have, a file that cannot be read: purple.
            purple = theme_tokens("purple")["accent"]
            self.assertEqual(bar("config"), purple)
            self.assertEqual(bar("config", "--theme-setting", "OTHER_THEME_COLOR"), purple)
            settings.write_text(json.dumps({"CONFIG_THEME_COLOR": "mauve"}), encoding="utf-8")
            self.assertEqual(bar("config", "--theme-setting", "CONFIG_THEME_COLOR"), purple)
            settings.write_text("{ not json", encoding="utf-8")
            self.assertEqual(bar("config", "--theme-setting", "CONFIG_THEME_COLOR"), purple)
            self.assertEqual(bar("config", "--theme-setting", "CONFIG_THEME_COLOR", saved=Path(directory) / "missing.json"),
                             purple)


if __name__ == "__main__":
    unittest.main()
