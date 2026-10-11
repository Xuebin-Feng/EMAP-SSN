# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The theme colours: their shades, the colour switch, and redrawing a window in a colour.

A theme colour is a family: every shade has a fixed depth, and the family
gives only the hue, except Black, the look before the theme colours. Config,
VR Config and Tools each keep their own colour in app_settings.json, chosen
with a circle at the end of the language row that opens a palette; choosing
one saves it and redraws that window in it at once, keeping everything it
shows. The Viewer takes Config's colour when it starts.
"""

import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QPoint, Qt  # noqa: E402
import shiboken6  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QTextBrowser, QVBoxLayout, QWidget  # noqa: E402

from desktop import Desktop_App  # noqa: E402
from desktop.Desktop_App import (  # noqa: E402
    CONFIG_THEME_SETTING,
    TOOLS_THEME_SETTING,
    ThemeColorButton,
    ThemeColorPalette,
    choose_theme_color,
    configured_theme_color,
    redraw_in_theme_color,
    theme_color_name,
)
from desktop.Studio_Theme import (  # noqa: E402
    ACCENT_SHADES,
    NEUTRAL_SHADES,
    DEFAULT_THEME_COLOR,
    THEME_COLORS,
    TOKENS,
    apply_studio_theme,
    page_tokens,
    qss_icon_path,
    set_theme_color,
    studio_stylesheet,
    theme_color,
    theme_tokens,
    web_page_properties,
)
from utilities.App_Settings import read_app_settings  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402


def flush(app):
    for _ in range(4):
        app.processEvents()


def contrast(first, second):
    """The WCAG contrast ratio of two #rrggbb colours."""
    def luminance(color):
        channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def oklab_lightness(color):
    """The OKLab lightness of a #rrggbb colour."""
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s


class ThemeColorShadeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_theme_color, theme_color())

    def test_every_colour_keeps_the_contrast_of_the_look_before(self):
        for color in THEME_COLORS:
            tokens = theme_tokens(color)
            with self.subTest(color=color):
                self.assertGreaterEqual(contrast(tokens["primary"], tokens["primary_text"]), 4.5)
                self.assertGreaterEqual(contrast(tokens["accent"], "#ffffff"), 4.5)
                self.assertGreaterEqual(contrast(tokens["text_subtle"], "#ffffff"), 4.5)
                self.assertGreaterEqual(contrast(tokens["text_muted"], tokens["surface_muted"]), 4.5)
                self.assertGreaterEqual(contrast(tokens["text"], "#ffffff"), 15)
                dark = page_tokens(color, dark=True)
                self.assertGreaterEqual(contrast(dark["primary"], dark["primary_text"]), 4.5)
                self.assertGreaterEqual(contrast(dark["text_subtle"], dark["surface"]), 4.5)

    def test_every_colour_has_the_same_tokens(self):
        for color in THEME_COLORS:
            with self.subTest(color=color):
                self.assertEqual(theme_tokens(color).keys(), theme_tokens(DEFAULT_THEME_COLOR).keys())
                self.assertEqual(page_tokens(color).keys(), page_tokens(DEFAULT_THEME_COLOR).keys())
                self.assertEqual(page_tokens(color, dark=True).keys(),
                                 page_tokens(DEFAULT_THEME_COLOR, dark=True).keys())

    def test_only_the_accents_take_the_colour(self):
        # Borders, icons, greyed-out fields and the rest are the same greys in every colour.
        for color in THEME_COLORS:
            with self.subTest(color=color):
                tokens = theme_tokens(color)
                self.assertEqual({token: tokens[token] for token in NEUTRAL_SHADES["light"]},
                                 NEUTRAL_SHADES["light"])
                dark = page_tokens(color, dark=True)
                self.assertEqual({token: dark[token] for token in NEUTRAL_SHADES["dark_page"]},
                                 NEUTRAL_SHADES["dark_page"])
        self.assertEqual(set(ACCENT_SHADES["light"]), {
            "primary", "primary_hover", "primary_pressed", "primary_text", "primary_focus",
            "accent", "accent_soft", "switch_on", "switch_on_hover", "track", "tint", "tint_hover",
            "histogram", "separator",
        })
        self.assertEqual(theme_tokens("teal")["text_subtle"], "#71717a")
        self.assertEqual(theme_tokens("teal")["field_border"], "#dcdce0")

    def test_purple_keeps_the_first_accent_and_black_has_no_colour(self):
        self.assertEqual(theme_tokens("purple")["accent"], "#5b5bd6")
        self.assertEqual(theme_tokens("purple")["primary"], "#5b5bd6")
        black = theme_tokens("black")
        expected = {"window": "#fafafa", "track": "#efeff1", "border": "#e4e4e7", "text_subtle": "#71717a",
                    "primary": "#18181b", "primary_text": "#fafafa", "accent": "#52525b",
                    "accent_soft": "#e4e4e7", "switch_on": "#18181b", "tooltip": "#18181b",
                    "separator": "#d4d4d8"}
        self.assertEqual({token: black[token] for token in expected}, expected)
        self.assertEqual(web_page_properties("black", dark=True)["--primary"], "#fafafa")
        # Every colour Black uses, on the page too, is a grey: red, green and blue equal or nearly.
        colours = list(black.values()) + list(page_tokens("black").values()) \
            + list(page_tokens("black", dark=True).values())
        for colour in colours:
            if isinstance(colour, str) and colour.startswith("#") and colour not in (
                    black["success"], black["warning"], black["danger"]):
                channels = [int(colour[index:index + 2], 16) for index in (1, 3, 5)]
                self.assertLessEqual(max(channels) - min(channels), 12, colour)

    def test_a_colour_family_changes_the_hue_not_the_depth(self):
        # Equal accents of two families differ, at the same OKLCH lightness.
        purple, teal = theme_tokens("purple"), theme_tokens("teal")
        for token in ("primary", "primary_focus", "accent", "accent_soft", "switch_on", "track", "tint",
                      "separator"):
            with self.subTest(token=token):
                self.assertNotEqual(purple[token], teal[token])
                self.assertAlmostEqual(oklab_lightness(purple[token]), oklab_lightness(teal[token]),
                                       delta=0.005)

    def test_setting_a_colour_changes_the_tokens_in_place(self):
        tokens = TOKENS
        set_theme_color("green")
        self.assertIs(TOKENS, tokens)
        self.assertEqual(theme_color(), "green")
        self.assertEqual(TOKENS["primary"], theme_tokens("green")["primary"])
        with self.assertRaisesRegex(ValueError, "no theme colour named 'mauve'"):
            set_theme_color("mauve")
        self.assertEqual(theme_color(), "green")

    def test_every_colour_has_its_stylesheet_arrows(self):
        for color in THEME_COLORS:
            tokens = theme_tokens(color)
            for token in ("text_subtle", "text_disabled"):
                with self.subTest(color=color, token=token):
                    self.assertTrue(qss_icon_path("chevron-down", token, tokens).is_file())
            self.assertIn(tokens["primary"], studio_stylesheet(tokens))

    def test_each_colour_has_a_name(self):
        names = [theme_color_name(color) for color in THEME_COLORS]
        self.assertEqual(names, ["Purple", "Blue", "Teal", "Green", "Orange", "Red", "Pink", "Black"])


class ComboPopupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def test_a_dropdown_shows_every_entry_without_scrolling(self):
        host = QWidget()
        self.addCleanup(host.deleteLater)
        layout = QVBoxLayout(host)
        combos = []
        for count in (2, 3, 6):
            combo = QComboBox()
            combo.addItems([f"entry {index}" for index in range(count)])
            layout.addWidget(combo)
            combos.append(combo)
        host.move(200, 200)
        host.show()
        flush(self.app)
        for combo in combos:
            with self.subTest(entries=combo.count()):
                combo.showPopup()
                flush(self.app)
                view = combo.view()
                popup = view.window()
                rows = sum(view.visualRect(combo.model().index(row, 0)).height()
                           for row in range(combo.count()))
                self.assertGreaterEqual(view.viewport().height(), rows)
                self.assertEqual(view.verticalScrollBar().maximum(), 0)
                # The list opens below its field.
                self.assertGreaterEqual(popup.geometry().top(),
                                        combo.mapToGlobal(QPoint(0, combo.height())).y() - 1)
                # Fusion's own popup showed a scroll arrow; the list makes none.
                scrollers = [child for child in popup.findChildren(QWidget)
                             if shiboken6.isValid(child)
                             and "Scroller" in child.metaObject().className() and child.isVisible()]
                self.assertEqual(scrollers, [])
                combo.hidePopup()
                flush(self.app)


class ThemeColorSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.settings = Path(folder.name) / "app_settings.json"
        patcher = mock.patch.dict(os.environ, {"SSN_APP_SETTINGS_PATH": str(self.settings)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.host = QWidget()
        self.addCleanup(self.host.deleteLater)
        self.host.setGeometry(300, 300, 400, 200)

    def button(self, color=None):
        button = ThemeColorButton(CONFIG_THEME_SETTING, self.host, color=color)
        button.move(300, 150)
        self.host.show()
        flush(self.app)
        return button

    def test_the_saved_colour_is_read_with_a_default(self):
        self.assertEqual(configured_theme_color(CONFIG_THEME_SETTING), DEFAULT_THEME_COLOR)
        self.assertEqual(configured_theme_color(CONFIG_THEME_SETTING, {"CONFIG_THEME_COLOR": "teal"}), "teal")
        self.assertEqual(configured_theme_color(TOOLS_THEME_SETTING, {"CONFIG_THEME_COLOR": "teal"}),
                         DEFAULT_THEME_COLOR)
        with mock.patch.object(Desktop_App, "_warn_once") as warn:
            self.assertEqual(configured_theme_color(CONFIG_THEME_SETTING, {"CONFIG_THEME_COLOR": "mauve"}),
                             DEFAULT_THEME_COLOR)
        warn.assert_called_once()
        self.settings.write_text("{not json", encoding="utf-8")
        with mock.patch.object(Desktop_App, "_warn_once") as warn:
            self.assertEqual(configured_theme_color(CONFIG_THEME_SETTING), DEFAULT_THEME_COLOR)
        warn.assert_called_once()

    def test_the_circle_names_its_colour_and_starts_at_the_saved_one(self):
        self.settings.write_text('{"CONFIG_THEME_COLOR": "orange"}', encoding="utf-8")
        button = self.button()
        self.assertEqual(button.color(), "orange")
        self.assertEqual(button.toolTip(), "Theme color: Orange")
        self.assertEqual(button.accessibleName(), "Theme color: Orange")
        self.assertEqual(button.text(), "")

    def test_a_click_opens_the_palette_above_the_circle(self):
        button = self.button("blue")
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        flush(self.app)
        popup = button.palette_popup
        self.assertIsInstance(popup, ThemeColorPalette)
        self.assertTrue(popup.isVisible())
        self.assertEqual(popup.colors, list(THEME_COLORS))
        self.assertEqual(popup.colors[popup.hovered], "blue")
        card = popup.card_rect().translated(popup.mapToGlobal(QPoint(0, 0)).toPointF())
        circle = button.mapToGlobal(QPoint(0, 0))
        self.assertLessEqual(card.bottom(), circle.y())
        self.assertAlmostEqual(card.right(), circle.x() + button.width(), delta=1)
        popup.close()

    def test_the_arrow_keys_and_enter_choose_a_colour(self):
        button = self.button("purple")
        chosen = []
        button.color_chosen.connect(chosen.append)
        popup = button.open_palette()
        flush(self.app)
        QTest.keyClick(popup, Qt.Key.Key_Right)
        QTest.keyClick(popup, Qt.Key.Key_Right)
        QTest.keyClick(popup, Qt.Key.Key_Left)
        self.assertEqual(popup.colors[popup.hovered], "blue")
        QTest.keyClick(popup, Qt.Key.Key_Return)
        self.assertEqual(chosen, ["blue"])
        self.assertFalse(popup.isVisible())

    def test_a_click_on_a_circle_chooses_it_and_escape_chooses_nothing(self):
        button = self.button("purple")
        chosen = []
        button.color_chosen.connect(chosen.append)
        popup = button.open_palette()
        flush(self.app)
        QTest.keyClick(popup, Qt.Key.Key_Escape)
        self.assertFalse(popup.isVisible())
        self.assertEqual(chosen, [])
        popup = button.open_palette()
        flush(self.app)
        black = popup.colors.index("black")
        QTest.mouseClick(popup, Qt.MouseButton.LeftButton, pos=popup.swatch_rect(black).center().toPoint())
        self.assertEqual(chosen, ["black"])

    def test_the_language_row_ends_in_the_circle(self):
        selector = Desktop_App.LanguageSelector(self.host)
        button = ThemeColorButton(CONFIG_THEME_SETTING, self.host)
        row = Desktop_App.language_selector_row(selector, button)
        widgets = [row.itemAt(index).widget() for index in range(row.count())]
        widgets = [widget for widget in widgets if widget is not None]
        self.assertEqual(widgets[-2:], [selector, button])


class WindowThemeColorTests(unittest.TestCase):
    """Config and Tools: the circle, and a redraw in the chosen colour."""

    @classmethod
    def setUpClass(cls):
        from tests.config_gui_loader import load_config_namespace

        cls.app = QApplication.instance() or QApplication([])
        cls.namespace = load_config_namespace()
        apply_theme_for_class(cls, cls.app)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.settings = Path(folder.name) / "app_settings.json"
        patcher = mock.patch.dict(os.environ, {"SSN_APP_SETTINGS_PATH": str(self.settings)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(apply_studio_theme, self.app, DEFAULT_THEME_COLOR)

    def track(self, window):
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        return window

    def open_config(self):
        from tests.config_gui_loader import open_config_window

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        window = self.track(open_config_window(self.namespace["ConfigGUI"], folder.name))
        window.show()
        flush(self.app)
        return window

    def open_tools(self):
        import EMAPSSN_Tools
        from tests.tools_gui_fixtures import isolated_tools_project

        isolated_tools_project(self)
        patcher = mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser)
        patcher.start()
        self.addCleanup(patcher.stop)
        window = self.track(EMAPSSN_Tools.ToolsGUI())
        window.show()
        flush(self.app)
        return window

    def test_the_circle_sits_after_the_language_dropdown(self):
        for window in (self.open_config(), self.open_tools()):
            with self.subTest(window=type(window).__name__):
                switch, language = window.theme_switch, window.language_selector
                self.assertEqual(switch.color(), theme_color())
                self.assertGreater(switch.mapTo(window, QPoint(0, 0)).x(),
                                   language.mapTo(window, QPoint(language.width(), 0)).x())
                self.assertAlmostEqual(switch.mapTo(window, switch.rect().center()).y(),
                                       language.mapTo(window, language.rect().center()).y(), delta=1)

    def test_choosing_a_colour_saves_it_for_config_and_redraws_keeping_the_fields(self):
        window = self.open_config()
        # The lines between sections take a light shade of the colour, not the cards' grey hairline.
        purple = theme_tokens(DEFAULT_THEME_COLOR)["separator"]
        self.assertNotEqual(purple, theme_tokens(DEFAULT_THEME_COLOR)["border"])
        self.assertEqual(self.section_line_colours(window), {purple})
        window.inputs["ALIGNMENT_REFERENCE"].setText("REF_7")
        window.tabs.setCurrentIndex(1)
        with mock.patch.object(self.namespace["ConfigGUI"], "_read_custom_settings", return_value={}):
            window.theme_switch.color_chosen.emit("teal")
        self.assertEqual(read_app_settings(), {"CONFIG_THEME_COLOR": "teal"})
        self.assertFalse(window.isVisible())
        replacement = self.track(Desktop_App._redrawn_windows[-1])
        self.assertIsNot(replacement, window)
        self.assertEqual(theme_color(), "teal")
        self.assertEqual(replacement.theme_switch.color(), "teal")
        self.assertEqual(replacement.inputs["ALIGNMENT_REFERENCE"].text(), "REF_7")
        self.assertEqual(replacement.tabs.currentIndex(), 1)
        self.assertIn(theme_tokens("teal")["primary"], self.app.styleSheet())
        self.assertEqual(self.section_line_colours(replacement), {theme_tokens("teal")["separator"]})

    def test_tools_keeps_its_own_colour(self):
        window = self.open_tools()
        window.theme_switch.color_chosen.emit("green")
        self.assertEqual(read_app_settings(), {"TOOLS_THEME_COLOR": "green"})
        replacement = self.track(Desktop_App._redrawn_windows[-1])
        self.assertEqual(replacement.theme_switch.color(), "green")
        self.assertEqual(replacement.theme_switch.theme_setting, TOOLS_THEME_SETTING)
        self.assertEqual(self.section_line_colours(replacement), {theme_tokens("green")["separator"]})

    def section_line_colours(self, window):
        """The colours of the lines between the window's shown sections (the "separator" role)."""
        flush(self.app)
        lines = [frame for frame in window.findChildren(QFrame)
                 if frame.property("role") == "separator" and frame.isVisible()]
        self.assertTrue(lines, f"{type(window).__name__} shows no section line")
        return {line.grab().toImage().pixelColor(line.width() // 2, 0).name() for line in lines}

    def test_the_same_colour_needs_no_redraw(self):
        window = self.open_config()
        self.assertIs(window.switch_theme_color(theme_color()), window)
        self.assertTrue(window.isVisible())

    def test_a_colour_that_cannot_be_saved_changes_nothing(self):
        window = self.open_config()
        self.settings.write_text("{not json", encoding="utf-8")
        with mock.patch.object(Desktop_App.QMessageBox, "critical") as critical:
            self.assertIs(choose_theme_color(window, "red"), window)
        critical.assert_called_once()
        self.assertTrue(window.isVisible())
        self.assertEqual(theme_color(), DEFAULT_THEME_COLOR)
        self.assertEqual(self.settings.read_text(encoding="utf-8"), "{not json")

    def test_a_redraw_that_fails_keeps_the_window_and_its_colour(self):
        window = self.open_config()

        def broken():
            raise RuntimeError("cannot build")

        with self.assertRaisesRegex(RuntimeError, "cannot build"):
            redraw_in_theme_color(window, "pink", broken)
        self.assertTrue(window.isVisible())
        self.assertEqual(theme_color(), DEFAULT_THEME_COLOR)
        self.assertNotIn(theme_tokens("pink")["primary"], self.app.styleSheet())


class StartupThemeColorTests(unittest.TestCase):
    """Each window starts in its own saved colour; the Viewer in Config's."""

    def test_each_window_starts_in_its_colour(self):
        for script, setting in (("EMAPSSN_Config.py", "CONFIG_THEME_SETTING"),
                                ("EMAPSSN_Tools.py", "TOOLS_THEME_SETTING"),
                                ("EMAPSSN_Viewer.py", "CONFIG_THEME_SETTING")):
            with self.subTest(script=script):
                tree = ast.parse((SRC / script).read_text(encoding="utf-8"))
                calls = [ast.unparse(node) for node in ast.walk(tree)
                         if isinstance(node, ast.Call) and ast.unparse(node.func) == "apply_studio_theme"]
                self.assertEqual(len(calls), 1, calls)
                self.assertTrue(calls[0].endswith(f", configured_theme_color({setting}))"), calls[0])


if __name__ == "__main__":
    unittest.main()
