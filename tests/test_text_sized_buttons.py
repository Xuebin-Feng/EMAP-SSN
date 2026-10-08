# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Buttons sized by their text (desktop.Desktop_App, "Buttons Sized by Their Text").

Toggle switches, the Config's Pick buttons and the Viewer's sidebar buttons take
their width from their text plus BUTTON_TEXT_PADDING at each end, so a longer
text, such as a translation, widens a button instead of being cut off. A toggle
covers both of its texts, so toggling never resizes it.
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utilities import Hardware_Acceleration as _preload  # noqa: F401 - load torch before PySide6
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

from desktop.Desktop_App import (
    BUTTON_TEXT_PADDING,
    TOGGLE_OFF_STYLESHEET,
    TOGGLE_ON_STYLESHEET,
    TOGGLE_SWITCH_HEIGHT,
    ToggleSwitch,
    clipped_button_text,
    configure_qt_application_fonts,
    fit_buttons_to_text,
)


def flush(app):
    for _ in range(4):
        app.processEvents()


def bold_advance(font, text):
    bold = QFont(font)
    bold.setBold(True)
    return QFontMetrics(bold).horizontalAdvance(text)


class TextSizedTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_qt_application_fonts(cls.app)

    def assert_text_shows_whole(self, button):
        """The room the button's style leaves for its text holds all of it."""
        self.assertEqual(clipped_button_text(button), 0, button.text())

    def assert_switches_fit_both_texts(self, window):
        """Every checkable button is a ToggleSwitch whose size fits both its texts."""
        switches = [button for button in window.findChildren(QPushButton)
                    if button.isCheckable()]
        self.assertTrue(switches)
        for switch in switches:
            with self.subTest(switch=switch.objectName() or switch.text()):
                self.assertIsInstance(switch, ToggleSwitch)
                sizes, texts = set(), []
                for _ in range(2):
                    switch.toggle()
                    flush(self.app)
                    self.assert_text_shows_whole(switch)
                    sizes.add((switch.width(), switch.height()))
                    texts.append(switch.text())
                needed = (max(bold_advance(switch.font(), text) for text in texts)
                          + 2 * BUTTON_TEXT_PADDING)
                self.assertEqual(sizes, {(needed, TOGGLE_SWITCH_HEIGHT)}, texts)


class ToggleSwitchTests(TextSizedTestCase):
    def test_width_fits_the_longer_text_and_never_changes(self):
        cases = (("ON", "OFF"), ("Auto ON", "Auto OFF"),
                 # Translations may make either text the longer one.
                 ("Automatisch eingeschaltet", "AUS"), ("I", "Complètement désactivé"))
        for on_text, off_text in cases:
            with self.subTest(on=on_text, off=off_text):
                switch = ToggleSwitch(on_text, off_text)
                self.addCleanup(switch.deleteLater)
                switch.show()
                expected = (max(bold_advance(switch.font(), text)
                                for text in (on_text, off_text))
                            + 2 * BUTTON_TEXT_PADDING)
                for checked in (False, True, False, True):
                    switch.setChecked(checked)
                    flush(self.app)
                    self.assertEqual(switch.text(), on_text if checked else off_text)
                    self.assertEqual(
                        switch.styleSheet(),
                        TOGGLE_ON_STYLESHEET if checked else TOGGLE_OFF_STYLESHEET,
                    )
                    self.assertEqual(
                        (switch.width(), switch.height()), (expected, TOGGLE_SWITCH_HEIGHT)
                    )
                    # The stylesheet draws the text bold; it keeps its padding.
                    self.assertTrue(switch.font().bold())
                    self.assertGreaterEqual(
                        switch.width() - switch.fontMetrics().horizontalAdvance(switch.text()),
                        2 * BUTTON_TEXT_PADDING,
                    )
                    self.assert_text_shows_whole(switch)

    def test_a_new_switch_reads_off_until_checked(self):
        switch = ToggleSwitch()
        self.addCleanup(switch.deleteLater)
        self.assertTrue(switch.isCheckable())
        self.assertFalse(switch.isChecked())
        self.assertEqual((switch.text(), switch.styleSheet()), ("OFF", TOGGLE_OFF_STYLESHEET))
        switch.setChecked(True)
        self.assertEqual((switch.text(), switch.styleSheet()), ("ON", TOGGLE_ON_STYLESHEET))


class FitButtonsToTextTests(TextSizedTestCase):
    def test_buttons_share_the_widest_text_plus_padding(self):
        buttons = [QPushButton(text) for text in ("Pick", "A label a translation made longer")]
        for button in buttons:
            self.addCleanup(button.deleteLater)
        width = fit_buttons_to_text(*buttons)
        widest = QFontMetrics(buttons[0].font()).horizontalAdvance(buttons[1].text())
        self.assertEqual(width, widest + 2 * BUTTON_TEXT_PADDING)
        for button in buttons:
            self.assertEqual((button.minimumWidth(), button.maximumWidth()), (width, width))
            self.assert_text_shows_whole(button)

    def test_a_style_that_keeps_more_room_than_the_padding_still_shows_all_text(self):
        """Stylesheet padding wider than the padding would clip; the width grows."""
        button = QPushButton("Export")
        self.addCleanup(button.deleteLater)
        button.setStyleSheet("padding: 10px 30px;")
        width = fit_buttons_to_text(button)
        self.assertGreater(
            width, button.fontMetrics().horizontalAdvance("Export") + 2 * BUTTON_TEXT_PADDING
        )
        self.assert_text_shows_whole(button)

    def test_text_is_measured_as_the_button_shows_it(self):
        """'&&' shows as '&', and a lone '&' only marks the shortcut key."""
        from desktop.Desktop_App import text_button_width

        button = QPushButton("Save && Run")
        self.addCleanup(button.deleteLater)
        metrics = QFontMetrics(button.font())
        for text, shown in (("Save && Run", "Save & Run"), ("&Export", "Export")):
            with self.subTest(text=text):
                self.assertEqual(
                    text_button_width([text], button.font()),
                    metrics.horizontalAdvance(shown) + 2 * BUTTON_TEXT_PADDING,
                )
        width = fit_buttons_to_text(button)
        self.assertEqual(width, metrics.horizontalAdvance("Save & Run") + 2 * BUTTON_TEXT_PADDING)

    def test_text_is_measured_in_the_font_a_stylesheet_sets(self):
        window = QMainWindow()
        self.addCleanup(window.deleteLater)
        window.setStyleSheet("QWidget#panel QPushButton { font-size: 24pt; font-weight: bold; }")
        panel = QWidget()
        panel.setObjectName("panel")
        window.setCentralWidget(panel)
        styled = QPushButton("Meta Data", panel)
        plain = QPushButton("Meta Data")
        self.addCleanup(plain.deleteLater)
        self.assertGreater(fit_buttons_to_text(styled), fit_buttons_to_text(plain))
        self.assertTrue(styled.font().bold())
        self.assertEqual(
            styled.width(),
            styled.fontMetrics().horizontalAdvance("Meta Data") + 2 * BUTTON_TEXT_PADDING,
        )


class ViewerSidebarTests(TextSizedTestCase):
    """Sidebar buttons share the widest label's width, and the panel fits them."""

    def make_viewer(self):
        from EMAPSSN_Viewer import MainViewer

        viewer = MainViewer.__new__(MainViewer)
        viewer.main_window = QMainWindow()
        self.addCleanup(viewer.main_window.deleteLater)
        # The Viewer's own rule sets the sidebar buttons' font this way.
        viewer.main_window.setStyleSheet(
            "QWidget#rightPanel QPushButton { font-size: 10pt; font-weight: bold; "
            "padding-left: 10px; padding-right: 10px; }"
        )
        viewer.right_panel = QWidget()
        viewer.right_panel.setObjectName("rightPanel")
        viewer.main_window.setCentralWidget(viewer.right_panel)
        viewer.right_panel_layout = QVBoxLayout(viewer.right_panel)
        viewer.right_panel_layout.setContentsMargins(10, 20, 10, 20)
        viewer.right_panel_layout.addStretch()
        viewer.set_sidebar_visible = lambda visible: None
        return viewer

    def assert_panel_fits(self, viewer):
        buttons = list(viewer.sidebar_buttons.values())
        widest = max(button.fontMetrics().horizontalAdvance(button.text())
                     for button in buttons)
        width = widest + 2 * BUTTON_TEXT_PADDING
        self.assertEqual({button.width() for button in buttons}, {width})
        margins = viewer.right_panel_layout.contentsMargins()
        self.assertEqual(viewer._panel_w, width + margins.left() + margins.right())
        self.assertEqual(viewer.main_window.minimumWidth(), viewer._panel_w)
        for button in buttons:
            self.assertTrue(button.font().bold())
            self.assertEqual(button.font().pointSize(), 10)
            self.assert_text_shows_whole(button)

    def test_buttons_share_the_widest_label_and_the_panel_fits_them(self):
        viewer = self.make_viewer()
        viewer.add_sidebar_button("agentBtn", "🤖 Agent", lambda: None)
        viewer.add_sidebar_button("metaDataBtn", "📊 Meta Data", lambda: None)
        self.assert_panel_fits(viewer)

    def test_a_longer_label_widens_every_button_and_the_panel(self):
        viewer = self.make_viewer()
        viewer.add_sidebar_button("agentBtn", "🤖 Agent", lambda: None)
        before = viewer._panel_w
        viewer.add_sidebar_button(
            "longBtn", "A label as long as a translation might make it", lambda: None
        )
        self.assertGreater(viewer._panel_w, before)
        self.assert_panel_fits(viewer)


class ConfigWindowTests(TextSizedTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from tests.config_gui_loader import load_config_namespace, open_config_window

        cls.namespace = load_config_namespace()
        cls.open_config_window = staticmethod(open_config_window)
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.window = open_config_window(cls.namespace["ConfigGUI"], cls.temp_directory.name)
        cls.window.show()
        flush(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        flush(cls.app)
        cls.temp_directory.cleanup()

    def test_switches_fit_both_texts(self):
        self.assert_switches_fit_both_texts(self.window)

    def test_pick_buttons_fit_their_text(self):
        """Pick fits its text, and so does the longer text a translation may give it."""

        class LongerPick(self.namespace["QPushButton"]):
            def __init__(self, text="", *args, **kwargs):
                super().__init__("Pick a colour" if text == "Pick" else text, *args, **kwargs)

        with patch.dict(self.namespace, {"QPushButton": LongerPick}):
            longer = self.open_config_window(
                self.namespace["ConfigGUI"], self.temp_directory.name
            )
        self.addCleanup(longer.deleteLater)
        self.addCleanup(longer.close)
        for window, text in ((self.window, "Pick"), (longer, "Pick a colour")):
            picks = [button for button in window.findChildren(QPushButton)
                     if button.text() == text]
            self.assertEqual(len(picks), 6)
            for button in picks:
                self.assertEqual(
                    button.width(),
                    button.fontMetrics().horizontalAdvance(text) + 2 * BUTTON_TEXT_PADDING,
                )
                self.assert_text_shows_whole(button)


class ToolsWindowTests(TextSizedTestCase):
    def open_tools_window(self, **replacements):
        """A Tools window built while EMAPSSN_Tools uses ``replacements``."""
        import contextlib
        import EMAPSSN_Tools
        from tests.tools_gui_fixtures import isolated_tools_project

        isolated_tools_project(self)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser))
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(EMAPSSN_Tools, name, replacement))
            window = EMAPSSN_Tools.ToolsGUI()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        window.resize(1800, 1000)
        window.show()
        flush(self.app)
        return window

    def assert_cards_share_one_section_edge(self, window):
        """On every tab, each card's buttons fit their text inside the left
        section, and its title and fields start at one shared x after it.
        Returns that x and the widest row of header buttons."""
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QFormLayout, QLabel
        from desktop.Desktop_App import text_button_width

        starts, action_widths, checked = set(), [], 0
        for index in range(window.tabs.count()):
            window.tabs.setCurrentIndex(index)
            flush(self.app)
            for form in window._tool_form_layouts:
                card = form.parentWidget()
                if not card.isVisible():
                    continue
                title = card.findChild(QLabel, "toolTitle")
                header = title.parentWidget()
                starts.add(header.property("sharedTitleStartX"))
                title_x = title.mapTo(card, QPoint()).x()
                buttons = header.findChildren(QPushButton)
                action_widths.append(buttons[0].parentWidget().sizeHint().width())
                for button in buttons:
                    with self.subTest(tab=index, button=button.text()):
                        self.assertEqual(
                            button.width(), text_button_width([button.text()], button.font())
                        )
                        self.assert_text_shows_whole(button)
                        self.assertLessEqual(
                            button.mapTo(card, QPoint(button.width(), 0)).x(),
                            title_x - form.horizontalSpacing(),
                        )
                for row in range(form.rowCount()):
                    label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                    field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                    if label is None or field is None or field.widget() is None:
                        continue
                    with self.subTest(tab=index, label=label.widget().text()):
                        self.assertEqual(field.widget().mapTo(card, QPoint()).x(), title_x)
                    checked += 1
        self.assertGreater(checked, 20)
        self.assertEqual(len(starts), 1)
        return starts.pop(), max(action_widths)

    def test_switches_fit_both_texts(self):
        self.assert_switches_fit_both_texts(self.open_tools_window())

    def test_cards_split_into_left_and_right_sections_on_every_tab(self):
        self.assert_cards_share_one_section_edge(self.open_tools_window())

    def test_longer_button_texts_widen_every_left_section(self):
        """Header button texts about twice as long, as a translation may make
        them, push every card's title and fields right, on every tab."""
        english_start, _ = self.assert_cards_share_one_section_edge(self.open_tools_window())

        class LongerButton(QPushButton):
            def __init__(self, text="", *args, **kwargs):
                if text in ("Save && Run", "Export", "Save Directories"):
                    text = f"{text} {'~' * len(text)}"
                super().__init__(text, *args, **kwargs)

        window = self.open_tools_window(QPushButton=LongerButton)
        start, widest_actions = self.assert_cards_share_one_section_edge(window)
        spacing = window._tool_form_layouts[0].horizontalSpacing()
        self.assertGreater(start, english_start)
        self.assertEqual(start, widest_actions + spacing)


if __name__ == "__main__":
    unittest.main()
