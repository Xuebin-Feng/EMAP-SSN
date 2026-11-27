# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Console window registered button row, entry points, owner-window lifecycle,
event loop exit, Viewer Console button, and persisted button registration tests
(desktop.Console_View)."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QCloseEvent, QHideEvent  # noqa: E402
from PySide6.QtWidgets import QDialog, QMainWindow, QMessageBox, QPushButton, QWidget  # noqa: E402

from desktop import Console_View, Desktop_App  # noqa: E402
from desktop.Console_View import ButtonRegistry, ConsoleWindow, dark_tokens, show_console  # noqa: E402
from desktop.Studio_Theme import TOKENS  # noqa: E402
from tests.console_view_fixtures import (  # noqa: E402
    QRect_of, THREE_BUTTONS, WindowCase, flush, header_widgets, icon_colour, wait_for,
)
from utilities import Console_Capture  # noqa: E402


class ButtonRegistryTests(unittest.TestCase):
    """What pages and commands register for the Console's second header row."""

    def test_buttons_come_in_the_order_they_were_registered(self):
        registry = ButtonRegistry()
        self.assertEqual(registry.buttons(), [])
        for name in ("b", "a", "c"):
            self.assertTrue(registry.add(name, name.upper(), lambda: None))
        self.assertEqual([spec.name for spec in registry.buttons()], ["b", "a", "c"])

    def test_a_button_is_its_name_label_function_tooltip_and_icon(self):
        registry = ButtonRegistry()

        def callback():
            return "clicked"

        registry.add("metaDataBtn", "Meta Data", callback, "Open the spreadsheet", "chart-column")
        registry.add("plain", "Plain", callback)
        first, second = registry.buttons()
        self.assertEqual((first.name, first.label, first.tooltip, first.icon), ("metaDataBtn", "Meta Data", "Open the spreadsheet", "chart-column"))
        self.assertIs(first.callback, callback)
        self.assertEqual((second.tooltip, second.icon), (None, None))

    def test_a_name_that_is_registered_already_changes_nothing(self):
        registry = ButtonRegistry()
        heard = []
        registry.subscribe(heard.append)
        first = lambda: "first"  # noqa: E731
        self.assertTrue(registry.add("x", "First", first, "tip", "bot"))
        self.assertFalse(registry.add("x", "Second", lambda: "second", "other tip", "dna"))
        self.assertEqual(len(registry.buttons()), 1)
        spec = registry.buttons()[0]
        self.assertEqual((spec.label, spec.tooltip, spec.icon), ("First", "tip", "bot"))
        self.assertIs(spec.callback, first)
        self.assertEqual(len(heard), 1)  # nobody heard of the second call

    def test_a_listener_hears_each_new_button_while_it_is_subscribed(self):
        registry = ButtonRegistry()
        heard = []
        registry.subscribe(heard.append)
        registry.subscribe(heard.append)  # once, however often it asks
        registry.add("a", "A", lambda: None)
        registry.unsubscribe(heard.append)
        registry.add("b", "B", lambda: None)
        registry.unsubscribe(heard.append)  # not subscribed: nothing to do
        self.assertEqual([spec.name for spec in heard], ["a"])

    def test_a_listener_whose_window_is_gone_is_dropped_and_the_registration_goes_on(self):
        registry = ButtonRegistry()
        heard = []

        def gone(_spec):
            raise RuntimeError("Internal C++ object (ConsoleWindow) already deleted.")

        registry.subscribe(gone)
        registry.subscribe(heard.append)
        self.assertTrue(registry.add("a", "A", lambda: None))
        self.assertTrue(registry.add("b", "B", lambda: None))
        self.assertEqual([spec.name for spec in heard], ["a", "b"])
        self.assertEqual(registry._listeners, [heard.append])


class ButtonRowTests(WindowCase):
    """The Console's second header row: the registered buttons, under the search row and above the output."""

    def setUp(self):
        super().setUp()
        self.registry = ButtonRegistry()
        self.clicked = []

    def tearDown(self):
        Console_View._window = None

    def register(self, buttons=THREE_BUTTONS):
        for name, label, icon in buttons:
            def callback(name=name):
                self.clicked.append(name)
            self.registry.add(name, label, callback, f"Open {label}", icon)

    def row_boxes(self, window):
        return [QRect_of(button, window) for button in window.button_controls()]

    def text_top(self, window):
        return window.text.mapTo(window, window.text.rect().topLeft()).y()

    def test_the_row_is_not_shown_while_nothing_is_registered_and_leaves_no_gap(self):
        window = self.window(buttons=self.registry)
        self.assertEqual(window.button_controls(), [])
        self.assertTrue(window.button_row.isHidden())
        without_row = self.text_top(window)
        header_bottom = max(QRect_of(control, window).bottom() for control in window.header_controls())
        self.assertEqual(without_row - header_bottom - 1, window.layout().spacing())  # the header, one gap, the output
        self.register()
        self.drain(window)
        self.assertTrue(window.button_row.isVisibleTo(window))
        # The output moves down by the row and one gap: nothing else changes.
        self.assertEqual(self.text_top(window) - without_row, window.button_row.height() + window.layout().spacing())

    def test_the_row_sits_under_the_search_row_and_above_the_output(self):
        self.register()
        window = self.window(buttons=self.registry)
        header_bottom = max(QRect_of(control, window).bottom() for control in window.header_controls())
        boxes = self.row_boxes(window)
        self.assertEqual(len(boxes), 3)
        for box in boxes:
            self.assertGreater(box.top(), header_bottom)
            self.assertLess(box.bottom(), self.text_top(window))
            self.assertGreaterEqual(box.left(), 12)
        self.assertEqual(boxes[0].left(), QRect_of(window.search, window).left())  # from the same left edge
        layout = window.layout()
        self.assertEqual([layout.itemAt(index).widget() is window.button_row for index in range(layout.count())],
                         [False, True, False, False])  # header, row, output, footer

    def test_a_button_registered_before_the_window_opens_is_there_when_it_opens(self):
        self.register()
        window = self.window(buttons=self.registry)
        buttons = window.button_controls()
        self.assertEqual([button.text() for button in buttons], ["Meta Data", "Agent", "Fold View"])
        self.assertEqual([button.objectName() for button in buttons], [name for name, _, _ in THREE_BUTTONS])
        self.assertEqual([button.toolTip() for button in buttons], ["Open Meta Data", "Open Agent", "Open Fold View"])
        self.assertEqual([button.accessibleName() for button in buttons], ["Meta Data", "Agent", "Fold View"])
        self.assertEqual([button.accessibleDescription() for button in buttons], ["Open Meta Data", "Open Agent", "Open Fold View"])
        for button in buttons:
            self.assertFalse(button.icon().isNull(), button.text())
            self.assertTrue(button.isVisibleTo(window))
            self.assertTrue(button.isEnabled())

    def test_the_row_is_built_with_the_window_so_its_first_layout_has_it(self):
        self.register()
        window = ConsoleWindow(self.recorder, buttons=self.registry)  # not shown yet
        self.addCleanup(window.deleteLater)
        self.assertEqual([button.text() for button in window.button_controls()], ["Meta Data", "Agent", "Fold View"])
        self.assertFalse(window.button_row.isHidden())

    def test_a_button_registered_while_the_window_is_open_appears_at_once(self):
        window = self.window(buttons=self.registry)
        self.assertTrue(window.button_row.isHidden())
        self.registry.add("agentBtn", "Agent", lambda: None, "Open Agent", "bot")
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent"])  # no event loop in between
        self.assertTrue(window.button_row.isVisibleTo(window))
        self.registry.add("metaDataBtn", "Meta Data", lambda: None, "Open Meta Data", "chart-column")
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent", "Meta Data"])
        flush(self.app, 2)
        boxes = self.row_boxes(window)
        self.assertGreater(boxes[1].left(), boxes[0].right())

    def test_a_button_registered_from_another_thread_appears_in_the_window(self):
        import threading

        window = self.window(buttons=self.registry)
        worker = threading.Thread(target=lambda: self.registry.add("agentBtn", "Agent", lambda: None, None, "bot"))
        worker.start()
        worker.join()
        self.assertTrue(wait_for(self.app, lambda: len(window.button_controls()) == 1))

    def test_buttons_registered_while_the_window_is_hidden_are_there_when_it_is_shown_again(self):
        self.register(THREE_BUTTONS[:1])
        window = self.window(buttons=self.registry)
        window.hide()
        self.assertEqual(self.registry._listeners, [])  # a hidden window listens to nothing
        self.registry.add("agentBtn", "Agent", lambda: None, "Open Agent", "bot")
        self.assertEqual(len(window.button_controls()), 1)
        window.show()
        self.assertEqual([button.text() for button in window.button_controls()], ["Meta Data", "Agent"])
        self.assertTrue(window.button_row.isVisibleTo(window))

    def test_a_click_calls_the_function_with_no_argument(self):
        self.register()
        window = self.window(buttons=self.registry)
        for button in window.button_controls():
            button.click()
        self.assertEqual(self.clicked, [name for name, _, _ in THREE_BUTTONS])

    def test_a_name_registered_twice_gives_one_button(self):
        self.register()
        window = self.window(buttons=self.registry)
        self.register()
        self.registry.add("agentBtn", "Another label", lambda: None, None, "dna")
        self.assertEqual([button.text() for button in window.button_controls()], ["Meta Data", "Agent", "Fold View"])
        self.assertEqual(len(window.findChildren(QPushButton, "agentBtn")), 1)

    def test_a_button_without_an_icon_or_tooltip_is_a_label_alone(self):
        self.registry.add("plain", "Plain", lambda: None)
        window = self.window(buttons=self.registry)
        button, = window.button_controls()
        self.assertEqual(button.text(), "Plain")
        self.assertTrue(button.icon().isNull())
        self.assertEqual(button.toolTip(), "")
        self.assertEqual(button.accessibleName(), "Plain")
        self.assertEqual(Desktop_App.clipped_button_text(button), 0)

    def test_the_buttons_have_the_consoles_button_style_and_show_their_whole_label(self):
        self.register()
        window = self.window(buttons=self.registry)
        for button in window.button_controls():
            self.assertEqual(button.property("role"), "compact")  # the outline button the header's others are
            self.assertEqual(button.height(), window.copy_button.height())
            self.assertEqual(Desktop_App.clipped_button_text(button), 0, button.text())
            self.assertEqual(button.width(), button.maximumWidth())  # as wide as its icon and label ask
            self.assertEqual(button.iconSize(), window.copy_button.iconSize())
        widths = [button.width() for button in window.button_controls()]
        self.assertGreater(widths[0], widths[1])  # "Meta Data" asks for more than "Agent": each has its own
        self.assertEqual(window.copy_button.property("role"), "icon")

    def test_the_icons_take_the_windows_colours_in_light_and_dark_like_its_other_icons(self):
        self.register()
        light = self.window(buttons=self.registry)
        for button in light.button_controls():
            self.assertEqual(icon_colour(button), icon_colour(light.copy_button))
        self.assertEqual(icon_colour(light.copy_button), TOKENS["text_subtle"].lower())
        dark = self.window(buttons=self.registry, dark=True)
        muted = dark_tokens()["text_muted"].lower()
        self.assertEqual(icon_colour(dark.copy_button), muted)
        for button in dark.button_controls():
            self.assertEqual(icon_colour(button), muted)
        self.assertNotEqual(muted, TOKENS["text_subtle"].lower())
        # One registered while the dark window is open is drawn in them too.
        self.registry.add("later", "Later", lambda: None, None, "square-terminal")
        self.assertEqual(icon_colour(dark.button_controls()[-1]), muted)
        # And the labels and the buttons' frames are the dark window's.
        self.assertIn(dark_tokens()["surface"], dark.styleSheet())

    def test_the_dark_window_measures_its_buttons_in_its_own_stylesheet(self):
        self.register()
        dark = self.window(buttons=self.registry, dark=True)
        for button in dark.button_controls():
            self.assertEqual(Desktop_App.clipped_button_text(button), 0, button.text())

    def test_the_window_cannot_be_narrower_than_the_wider_of_the_two_rows(self):
        header_only = self.window().minimumWidth()
        self.register()
        window = self.window(buttons=self.registry)
        # Three short labels fit under the header's own minimum, which stays what it was.
        row = sum(button.width() for button in window.button_controls()) + 6 * 2 + 24
        self.assertLess(row, header_only)
        self.assertEqual(window.minimumWidth(), header_only)
        # A row wider than the header sets the minimum, and the window grows to it at once if it is narrower.
        window.resize(header_only, 420)
        flush(self.app, 2)
        for number in range(6):
            self.registry.add(f"long{number}", f"A button with quite a long label {number}", lambda: None, None, "bot")
        flush(self.app, 4)
        wide_row = sum(button.width() for button in window.button_controls()) + 6 * (len(window.button_controls()) - 1) + 24
        self.assertGreater(wide_row, header_only)
        self.assertEqual(window.minimumWidth(), wide_row)
        self.assertGreaterEqual(window.width(), wide_row)
        window.resize(200, 420)
        flush(self.app, 4)
        self.assertEqual(window.width(), wide_row)

    def assert_one_row_of_buttons(self, window):
        boxes = self.row_boxes(window)
        previous = None
        for button, box in zip(window.button_controls(), boxes):
            with self.subTest(button=button.text()):
                self.assertTrue(button.isVisibleTo(window))
                self.assertGreaterEqual(box.left(), 12)
                self.assertLessEqual(box.right(), window.width() - 12)
                if previous is not None:
                    self.assertGreater(box.left(), previous.right())
                    self.assertEqual(box.center().y(), previous.center().y())  # one row, never wrapped
                previous = box

    def test_both_rows_stay_on_one_line_each_at_the_minimum_width_in_each_language(self):
        from PySide6.QtCore import QCoreApplication

        from desktop import Desktop_App

        minimums = {}
        for language in (None, "zh_CN", Desktop_App.PSEUDO_LANGUAGE):
            with self.subTest(language=language), self.in_language(language):
                registry = ButtonRegistry()
                for name, label, icon in THREE_BUTTONS:
                    registry.add(name, QCoreApplication.translate("Viewer", label), lambda: None, None, icon)
                window = self.window(buttons=registry)
                minimums[language] = window.minimumWidth()
                window.resize(window.minimumWidth(), 420)
                flush(self.app, 4)
                self.assertEqual(window.width(), window.minimumWidth())
                self.assertOneRow(window)
                self.assert_one_row_of_buttons(window)
                window.resize(1400, 420)
                flush(self.app, 4)
                self.assertOneRow(window)
                self.assert_one_row_of_buttons(window)
        self.assertEqual(len(set(minimums.values())), 3, minimums)

    def test_a_long_translation_of_the_buttons_widens_the_window_instead_of_wrapping(self):
        from PySide6.QtCore import QCoreApplication

        with self.in_language("pseudo"):
            registry = ButtonRegistry()
            for name, label, icon in THREE_BUTTONS:
                registry.add(name, QCoreApplication.translate("Viewer", label) * 3, lambda: None, None, icon)
            window = self.window(buttons=registry)
            window.resize(200, 420)
            flush(self.app, 4)
            self.assertEqual(window.width(), window.minimumWidth())
            self.assert_one_row_of_buttons(window)
            for button in window.button_controls():
                self.assertEqual(Desktop_App.clipped_button_text(button), 0)
            header_right = max(QRect_of(control, window).right() for control in window.header_controls())
            self.assertLessEqual(header_right, window.width() - 12)

    def test_the_second_row_does_not_change_the_headers_controls(self):
        self.register()
        with_row = self.window(buttons=self.registry)
        without = self.window()
        self.assertEqual(len(with_row.header_controls()), len(without.header_controls()))
        for control in with_row.header_controls():
            self.assertNotIn(control, with_row.button_controls())
        header = with_row.layout().itemAt(0).layout()
        self.assertEqual(header_widgets(header), with_row.header_controls())

    def test_show_console_keeps_one_window_for_one_registry_and_a_new_one_for_another(self):
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            first = show_console(None, buttons=self.registry)
            again = show_console(None, buttons=self.registry)
            nothing_given = show_console(None)  # no registry named: the window as it is
            other = ButtonRegistry()
            second = show_console(None, buttons=other)
        self.addCleanup(first.deleteLater)
        self.addCleanup(second.deleteLater)
        self.assertIs(first, again)
        self.assertIs(first, nothing_given)
        self.assertIsNot(first, second)
        self.assertIs(second.buttons, other)
        self.assertIs(first.buttons, self.registry)
        first.close()
        second.close()


class EntryPointTests(WindowCase):
    def tearDown(self):
        Console_View._window = None

    def test_show_console_opens_one_window_and_raises_it_again(self):
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            first = show_console(None)
            second = show_console(None)
        self.addCleanup(first.deleteLater)
        self.assertIs(first, second)
        self.assertTrue(first.isVisible() or first.testAttribute(Qt.WidgetAttribute.WA_WState_Visible))
        first.close()

    def test_a_program_that_keeps_no_output_opens_a_window_that_says_so_in_a_message_box_no_longer(self):
        with mock.patch.object(Console_Capture, "_capture", None), mock.patch.object(
            Console_View.QMessageBox, "information"
        ) as information:
            window = show_console(None)
        self.addCleanup(window.deleteLater)
        information.assert_not_called()
        self.assertIn("--console", window.text.toPlainText())
        window.close()

    def test_the_window_is_titled_after_the_window_it_belongs_to(self):
        parent = QWidget()
        parent.setWindowTitle("EMAP-SSN Viewer")
        self.addCleanup(parent.deleteLater)
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            window = show_console(parent)
        self.addCleanup(window.deleteLater)
        self.assertEqual(window.windowTitle(), "EMAP-SSN Viewer – Console")
        window.close()

    def test_the_language_row_of_config_tools_and_vr_config_is_as_it_was_with_no_console_button(self):
        import inspect

        from PySide6.QtWidgets import QHBoxLayout

        # The Console is the Viewer's (its command and the button at the foot of its window); Config and Tools
        # show what they print on the start-up splash, and their footers hold no Console button (the high
        # contrast button is the one they have besides the theme switch).
        self.assertEqual(list(inspect.signature(Desktop_App.language_selector_row).parameters),
                         ["selector", "theme_switch", "leading", "contrast_switch"])
        host = QWidget()
        self.addCleanup(host.deleteLater)
        selector = Desktop_App.LanguageSelector(host)
        row = Desktop_App.language_selector_row(selector, QWidget(host))
        self.assertIs(type(row), QHBoxLayout)
        widgets = [row.itemAt(index).widget() for index in range(row.count()) if row.itemAt(index).widget()]
        self.assertEqual([widget.objectName() for widget in widgets[:2]], ["languageGlobe", "languageSelector"])
        self.assertEqual(len(widgets), 3)  # the globe, the dropdown and the theme switch: no other button
        # The dropdown asks for what any dropdown of its choices asks for: no room given away, no own painting.
        for name in ("minimumSizeHint", "paintEvent", "give_way"):
            self.assertNotIn(name, vars(Desktop_App.LanguageSelector))
        self.assertEqual(selector.minimumSizeHint().width(), selector.sizeHint().width())


class OwnerWindowTests(WindowCase):
    """The Console belongs to the window it was opened for (the Viewer's): it closes with it, never the other way round."""

    def setUp(self):
        super().setUp()
        self.owner = QMainWindow()
        self.owner.setWindowTitle("EMAP-SSN Viewer")
        self.owner.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        self.owner.show()
        self.addCleanup(self.owner.deleteLater)
        self.addCleanup(self.owner.close)
        self.recorder.add("stdout", "printed before")

    def tearDown(self):
        Console_View._window = None

    def open_console(self, buttons=None):
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            console = show_console(self.owner, buttons=buttons or ButtonRegistry())
        self.addCleanup(console.deleteLater)
        flush(self.app, 2)
        return console

    def test_closing_the_window_closes_its_console_in_front_behind_or_minimized(self):
        for how in ("in front", "behind", "minimized"):
            with self.subTest(how=how):
                console = self.open_console()
                self.assertTrue(console.isVisible())
                if how == "behind":
                    console.lower()
                elif how == "minimized":
                    console.showMinimized()
                    flush(self.app, 2)
                    self.assertTrue(console.isMinimized())
                self.owner.close()
                flush(self.app, 2)
                self.assertFalse(self.owner.isVisible())
                self.assertFalse(console.isVisible())
                self.owner.show()

    def test_the_console_goes_with_the_window_whichever_way_the_window_closes(self):
        # The title bar's close button asks the window to close (the window handle's close); a program's close() does
        # the same; and a hide() is the Viewer taken off screen.
        for how, close in (("the close button", lambda: self.owner.windowHandle().close()),
                           ("close()", lambda: self.owner.close()), ("hide()", lambda: self.owner.hide())):
            with self.subTest(how=how):
                console = self.open_console()
                close()
                flush(self.app, 2)
                self.assertFalse(console.isVisible())
                self.owner.show()

    def test_a_hide_the_system_makes_leaves_the_console_and_one_the_program_makes_closes_it(self):
        console = self.open_console()
        # A minimized window sends a hide event the system made (spontaneous) and stays open.
        class SystemHide(QHideEvent):
            def spontaneous(self):
                return True

        self.assertFalse(console.eventFilter(self.owner, SystemHide()))
        self.owner.showMinimized()
        flush(self.app, 2)
        self.assertTrue(console.isVisible())
        self.owner.showNormal()
        # The event of any other window changes nothing either.
        self.assertFalse(console.eventFilter(QWidget(), QHideEvent()))
        self.assertTrue(console.isVisible())
        self.assertFalse(console.eventFilter(self.owner, QHideEvent()))
        self.assertFalse(console.isVisible())
        # A close event closes it as well, before the window has finished going.
        console.show()
        self.assertFalse(console.eventFilter(self.owner, QCloseEvent()))
        self.assertFalse(console.isVisible())

    def test_closing_the_console_leaves_the_window_open_and_it_opens_again_with_its_contents(self):
        buttons = ButtonRegistry()
        console = self.open_console(buttons)
        self.assertEqual(self.lines(console), ["printed before"])
        console.close()
        flush(self.app, 2)
        self.assertFalse(console.isVisible())
        self.assertTrue(self.owner.isVisible())
        self.assertFalse(self.owner.isHidden())
        self.recorder.add("stdout", "printed while it was closed")
        again = self.open_console(buttons)
        self.assertIs(again, console)
        self.assertTrue(again.isVisible())
        self.assertTrue(wait_for(self.app, lambda: self.lines(again) == ["printed before", "printed while it was closed"]))

    def test_the_console_is_not_a_window_the_program_waits_for(self):
        # Closing the last window ends the program, whatever else is open (Qt: WA_QuitOnClose), and closing the
        # Console never does.
        self.assertFalse(self.open_console().testAttribute(Qt.WidgetAttribute.WA_QuitOnClose))
        self.assertFalse(self.window().testAttribute(Qt.WidgetAttribute.WA_QuitOnClose))

    def test_a_console_opened_for_no_window_has_nothing_to_follow(self):
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            console = show_console(None)
        self.addCleanup(console.deleteLater)
        self.assertIsNone(console._owner)
        self.owner.close()
        flush(self.app, 2)
        self.assertTrue(console.isVisible())
        console.close()

    def test_a_viewer_started_with_console_has_a_console_that_follows_its_window_too(self):
        with mock.patch.object(Console_Capture, "_capture", None):
            console = show_console(self.owner, buttons=ButtonRegistry())
        self.addCleanup(console.deleteLater)
        self.assertFalse(console.kept)
        self.assertTrue(console.isVisible())
        self.owner.close()
        flush(self.app, 2)
        self.assertFalse(console.isVisible())

    def test_the_save_dialog_closes_with_the_console_and_the_save_ends_with_nothing_saved(self):
        console = self.open_console()
        open_dialogs, left_open = [], []

        def visible_dialogs():
            return [dialog for dialog in console.findChildren(QDialog) if dialog.isVisible()]

        def close_the_window():
            open_dialogs.extend(visible_dialogs())
            self.owner.close()

        def give_up():  # a dialog that the window's closing left open would hold Save here for ever
            for dialog in visible_dialogs():
                left_open.append(dialog)
                dialog.reject()

        # Save waits in the dialog's own event loop, as it does while the person chooses a file.
        QTimer.singleShot(200, close_the_window)
        QTimer.singleShot(5000, give_up)
        saved = console.save_text()
        self.assertEqual(len(open_dialogs), 1)
        self.assertEqual(left_open, [])
        self.assertIsNone(saved)
        self.assertFalse(console.isVisible())
        self.assertEqual(visible_dialogs(), [])

    def test_a_hidden_viewer_that_was_shown_takes_its_console_off_screen_and_closes_it_with_the_window(self):
        from desktop.Viewer_Window import HIDDEN, ViewerWindow

        control = ViewerWindow(self.owner, HIDDEN, platform="linux", alert=lambda *_: None)
        control.open()
        self.assertFalse(control.shows_windows())
        control.show()
        self.assertTrue(control.shows_windows())
        console = self.open_console()
        # Taken off screen again, the Viewer shows no windows, and the Console it opened goes too.
        control.hide()
        flush(self.app, 2)
        self.assertFalse(console.isVisible())
        self.assertFalse(control.shows_windows())
        # Shown again, a Console opens again; closing the shown window ends the Viewer and closes it with the window.
        control.show()
        console = self.open_console()
        self.assertTrue(console.isVisible())
        self.owner.close()
        flush(self.app, 2)
        self.assertFalse(console.isVisible())


# The program's event loop, run in a program of its own so that no other test's windows or hooks are in it. Each
# scenario opens a window and its Console, closes something and says whether the loop ended by itself (it is
# stopped after SECONDS otherwise) and what stayed open.
EVENT_LOOP_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from PySide6.QtCore import QMetaObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow
from desktop import Console_View
from desktop.Console_View import ButtonRegistry, ConsoleWindow, show_console
from utilities.Console_Capture import Recorder

SECONDS = 6.0
app = QApplication([])
recorder = Recorder("viewer", None)
recorder.add("stdout", "printed")
results = {}


def run(name, scenario):
    Console_View._window = None
    main = QMainWindow()
    main.setWindowTitle("EMAP-SSN Viewer")
    main.show()
    state = {"stopped": False, "last_window_closed": 0, "main": main}
    app.lastWindowClosed.connect(lambda: state.update(last_window_closed=state["last_window_closed"] + 1))

    def stop():
        state["stopped"] = True
        app.quit()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(stop)
    watchdog.start(int(SECONDS * 1000))
    scenario(main, state)
    app.exec()
    watchdog.stop()
    console = state.get("console")
    results[name] = {
        "ended_by_itself": not state["stopped"],
        "last_window_closed": state["last_window_closed"],
        "console_visible": None if console is None else console.isVisible(),
        "main_visible": main.isVisible(),
        "main_visible_midway": state.get("main_visible_midway"),
        "last_window_closed_midway": state.get("last_window_closed_midway"),
        "saved": state.get("saved", "not asked"),
    }
    app.lastWindowClosed.disconnect()


def with_console(main, state):
    console = state["console"] = show_console(main, recorder, buttons=ButtonRegistry())
    return console


def window_closed(main, state):
    with_console(main, state)
    QTimer.singleShot(150, main.close)


def window_closed_console_not_following(main, state):
    console = state["console"] = ConsoleWindow(recorder)
    console.show()
    QTimer.singleShot(150, main.close)


def window_closed_in_the_save_dialog(main, state):
    console = with_console(main, state)
    QTimer.singleShot(100, lambda: state.update(saved=console.save_text()))
    QTimer.singleShot(500, main.close)


def application_quit(main, state):
    with_console(main, state)
    # As the shutdown route of the Viewer's web server asks for it.
    QTimer.singleShot(150, lambda: QMetaObject.invokeMethod(app, "quit", Qt.ConnectionType.QueuedConnection))


def console_closed(main, state):
    console = with_console(main, state)
    QTimer.singleShot(100, console.close)
    QTimer.singleShot(700, lambda: state.update(
        main_visible_midway=main.isVisible(), last_window_closed_midway=state["last_window_closed"]))
    QTimer.singleShot(800, app.quit)


run("window_closed", window_closed)
run("window_closed_console_not_following", window_closed_console_not_following)
run("window_closed_in_the_save_dialog", window_closed_in_the_save_dialog)
run("application_quit", application_quit)
run("console_closed", console_closed)
print("RESULT " + json.dumps(results))
"""


class EventLoopTests(unittest.TestCase):
    """A real event loop: what ends the program and what stays open."""

    @classmethod
    def setUpClass(cls):
        environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        done = subprocess.run([sys.executable, "-c", EVENT_LOOP_SCRIPT, str(SRC)], env=environment,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
        if not lines:
            raise AssertionError(f"the event loop program gave no result (exit code {done.returncode}):\n{done.stdout}\n{done.stderr}")
        cls.results = json.loads(lines[-1][len("RESULT "):])

    def test_closing_the_window_with_its_console_open_ends_the_program_and_closes_the_console(self):
        result = self.results["window_closed"]
        self.assertTrue(result["ended_by_itself"], result)
        self.assertEqual(result["last_window_closed"], 1)
        self.assertFalse(result["main_visible"])
        self.assertFalse(result["console_visible"])

    def test_a_console_left_open_cannot_keep_the_program_running(self):
        # One that nothing closes with the window: the program ends all the same, as the window was its last.
        result = self.results["window_closed_console_not_following"]
        self.assertTrue(result["ended_by_itself"], result)
        self.assertEqual(result["last_window_closed"], 1)

    def test_the_program_ends_when_the_window_closes_while_the_save_dialog_is_open(self):
        result = self.results["window_closed_in_the_save_dialog"]
        self.assertTrue(result["ended_by_itself"], result)
        self.assertIsNone(result["saved"])
        self.assertFalse(result["console_visible"])

    def test_the_application_quitting_closes_the_window_and_its_console(self):
        result = self.results["application_quit"]
        self.assertTrue(result["ended_by_itself"], result)
        self.assertFalse(result["main_visible"])
        self.assertFalse(result["console_visible"])

    def test_closing_the_console_ends_nothing_and_leaves_the_window_open(self):
        result = self.results["console_closed"]
        self.assertEqual(result["last_window_closed_midway"], 0)
        self.assertTrue(result["main_visible_midway"])
        self.assertFalse(result["console_visible"])


class ViewerEntryPointTests(WindowCase):
    """The Viewer's Console button, at the foot of its window, and the method the console command calls."""

    def setUp(self):
        super().setUp()
        import EMAPSSN_Viewer

        self.module = EMAPSSN_Viewer
        viewer = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
        viewer.canvas = SimpleNamespace(native=QWidget(), size=(1000, 700))
        self.addCleanup(viewer.canvas.native.deleteLater)
        viewer.hud_layout = {"status_x_offset": 10.0, "status_line_spacing": 38.0}
        viewer._hud_timer = mock.Mock()
        viewer.min_threshold, viewer.max_threshold = 0.0, 9.6
        viewer.current_slider_threshold = 3.74
        viewer.update_edges = mock.Mock()
        viewer._build_slider_overlay()
        viewer._build_main_window()
        self.addCleanup(viewer.main_window.deleteLater)
        self.addCleanup(viewer.main_window.close)
        viewer.window_control = None
        viewer.headless = False
        viewer._update_hud_elements = lambda: None
        self.viewer = viewer

    def tearDown(self):
        Console_View._window = None

    def test_the_row_at_the_foot_gets_a_console_button_with_the_terminal_icon(self):
        button = self.viewer.console_button
        self.assertEqual(button.text(), "Console")
        self.assertFalse(button.icon().isNull())
        self.assertTrue(button.toolTip().startswith("Show everything this Viewer has printed"))
        self.assertIs(button.parentWidget(), self.viewer.slider_overlay)

    def test_the_sidebar_and_the_session_card_have_no_console_entry_any_more(self):
        import ast

        self.assertFalse(hasattr(self.viewer, "_add_console_button"))
        self.assertNotIn("consoleBtn", getattr(self.viewer, "sidebar_buttons", {}))
        # Nothing at the Viewer's start-up adds a sidebar button of that name either.
        source = (SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8")
        self.assertNotIn("consoleBtn", source)
        self.assertNotIn("_add_console_button", source)
        # The card is given no Console function, and has no link for one.
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "ControlsButton"]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("console", [keyword.arg for keyword in calls[0].keywords])
        card = self.viewer.controls_button.toggle_card()
        self.addCleanup(card.close)
        self.assertFalse(hasattr(card, "console_button"))
        self.assertEqual([link.text() for link in card.findChildren(QPushButton) if link.objectName() == "controlsLink"],
                         ["List of commands"])

    def test_the_button_is_given_the_action_the_old_sidebar_button_had(self):
        import ast

        tree = ast.parse((SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "ConsoleButton"]
        self.assertEqual(len(calls), 1)
        opened = [keyword.value for keyword in calls[0].keywords if keyword.arg == "opened"]
        self.assertEqual([ast.unparse(value) for value in opened], ["self.open_console_window"])

    def test_opening_it_says_what_happened(self):
        viewer = self.viewer
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            result = viewer.open_console_window()
            self.addCleanup(lambda: Console_View._window and Console_View._window.deleteLater())
        self.assertEqual(result, "opened")
        # Started with --console, nothing is kept, and the Console opens all the same: for its buttons.
        with mock.patch.object(Console_Capture, "_capture", None):
            self.assertEqual(viewer.open_console_window(), "opened")
        viewer.headless = True
        self.assertEqual(viewer.open_console_window(), "no_window")

    def test_a_viewer_started_with_console_opens_a_console_with_the_buttons_and_no_message_box(self):
        viewer = self.viewer
        viewer.add_console_button("metaDataBtn", "Meta Data", lambda: None, "Open the sheet", "chart-column")
        with mock.patch.object(Console_Capture, "_capture", None), mock.patch.object(QMessageBox, "information") as information:
            viewer.console_button.click()
        information.assert_not_called()
        window = Console_View._window
        self.assertIsNotNone(window)
        self.addCleanup(window.deleteLater)
        self.assertFalse(window.kept)
        self.assertEqual(window.text.toPlainText(), Console_View.not_available_text())
        self.assertEqual([button.text() for button in window.button_controls()], ["Meta Data"])

    def test_the_button_opens_the_window_when_output_is_kept(self):
        self.recorder.add("stdout", "kept by the viewer")
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            self.viewer.console_button.click()
        window = Console_View._window
        self.assertIsNotNone(window)
        self.addCleanup(window.deleteLater)
        self.assertTrue(window.kept)
        self.assertEqual(self.lines(window), ["kept by the viewer"])
        self.assertNotEqual(window.text.toPlainText(), Console_View.not_available_text())

    def test_closing_the_viewers_window_closes_its_console_and_closing_the_console_leaves_the_window_alone(self):
        viewer = self.viewer
        viewer.main_window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        viewer.main_window.show()
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            viewer.console_button.click()
            console = Console_View._window
            self.addCleanup(console.deleteLater)
            self.assertTrue(console.isVisible())
            # The Console closes: the Viewer is as it was, and its button brings the same window back.
            console.close()
            flush(self.app, 2)
            self.assertTrue(viewer.main_window.isVisible())
            viewer.console_button.click()
            self.assertIs(Console_View._window, console)
            self.assertTrue(console.isVisible())
            # The Viewer's window closes: the Console goes with it.
            viewer.main_window.close()
            flush(self.app, 2)
            self.assertFalse(viewer.main_window.isVisible())
            self.assertFalse(console.isVisible())

    def test_the_window_gets_the_viewers_registry_and_a_button_added_later_appears_in_it(self):
        viewer = self.viewer
        viewer.add_console_button("agentBtn", "Agent", lambda: None, "Open the agent", "bot")
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            viewer.open_console_window()
        window = Console_View._window
        self.addCleanup(window.deleteLater)
        self.assertIs(window.buttons, viewer.console_buttons)
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent"])
        viewer.add_console_button("metaDataBtn", "Meta Data", lambda: None, "Open the sheet", "chart-column")
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent", "Meta Data"])
        self.assertFalse(viewer.add_console_button("agentBtn", "Agent", lambda: None))  # known: nothing changes
        self.assertEqual(len(window.button_controls()), 2)

    def test_a_viewer_with_no_window_opens_nothing_and_says_nothing(self):
        # As the sidebar button was: a headless Viewer, or a hidden one not shown, has no window to open or to
        # explain in (the console command refuses with the reason).
        self.viewer.headless = True
        self.viewer.add_console_button("agentBtn", "Agent", lambda: None, "Open the agent", "bot")  # registering is fine
        with mock.patch.object(Console_Capture, "_capture", self.recorder), mock.patch.object(
            QMessageBox, "information"
        ) as information:
            self.viewer.console_button.click()
        information.assert_not_called()
        self.assertIsNone(Console_View._window)
        self.assertEqual(len(self.viewer.console_buttons.buttons()), 1)

    def activate_the_pages(self, viewer):
        import contextlib
        import io

        from web_ui import agent_backend, esmfold_backend, meta_backend

        with contextlib.redirect_stdout(io.StringIO()):
            for plugin in (agent_backend, esmfold_backend, meta_backend):
                plugin.activate(viewer)

    def test_the_pages_register_their_own_buttons_and_the_core_names_none(self):
        viewer = self.viewer
        viewer.open_page = mock.Mock()
        self.activate_the_pages(viewer)
        specs = viewer.console_buttons.buttons()
        self.assertEqual([(spec.name, spec.label, spec.icon) for spec in specs],
                         [("agentBtn", "Agent", "bot"), ("fold_view_btn", "Fold View", "dna"),
                          ("metaDataBtn", "Meta Data", "chart-column")])
        for spec in specs:
            self.assertTrue(spec.tooltip, spec.name)
            self.assertTrue(spec.label.isascii(), spec.label)  # no emoji: the icon is the picture
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        self.assertFalse(hasattr(viewer, "sidebar_buttons"))  # nothing went into the sidebar's code
        # No command or page is named by the Console window or the Viewer's registration.
        for name in ("meta", "agent", "esmfold", "Fold View", "Meta Data"):
            self.assertNotIn(name, inspect.getsource(Console_View.ButtonRegistry))
            self.assertNotIn(name, inspect.getsource(Console_View.ConsoleWindow._sync_buttons))
            self.assertNotIn(name, inspect.getsource(self.module.MainViewer.add_console_button))
            self.assertNotIn(name, inspect.getsource(self.module.MainViewer.console_button_registry))

    def test_repeating_a_pages_command_adds_nothing_and_the_console_shows_each_button_once(self):
        viewer = self.viewer
        self.activate_the_pages(viewer)
        before = [spec.name for spec in viewer.console_buttons.buttons()]
        self.activate_the_pages(viewer)
        self.assertEqual([spec.name for spec in viewer.console_buttons.buttons()], before)
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            viewer.open_console_window()
        window = Console_View._window
        self.addCleanup(window.deleteLater)
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent", "Fold View", "Meta Data"])
        self.activate_the_pages(viewer)
        self.assertEqual(len(window.button_controls()), 3)

    def test_each_button_opens_its_page(self):
        from web_ui import esmfold_backend

        viewer = self.viewer
        viewer.open_page = mock.Mock()
        self.activate_the_pages(viewer)
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            viewer.open_console_window()
        window = Console_View._window
        self.addCleanup(window.deleteLater)
        buttons = {button.objectName(): button for button in window.button_controls()}
        with mock.patch.object(esmfold_backend, "open_esmfold_ui") as open_fold:
            buttons["metaDataBtn"].click()
            buttons["agentBtn"].click()
            buttons["fold_view_btn"].click()
        self.assertEqual(viewer.open_page.call_args_list, [mock.call("meta"), mock.call("agent")])
        open_fold.assert_called_once_with(viewer)

    def test_a_persisted_button_comes_back_as_a_console_button_when_a_session_loads(self):
        from commands import meta as meta_command
        from tests.command_fixtures import run_command

        viewer = self.viewer
        viewer.sidebar_buttons_to_persist = ["meta"]  # what a loaded session holds: the attribute keeps its name
        replayed = []

        def process_command(command, **options):
            name, *arguments = command.split()
            replayed.append((command, options))
            run_command({"meta": meta_command}[name], viewer, arguments)

        viewer.process_command = process_command
        viewer._register_persisted_console_buttons()
        self.assertEqual(replayed, [("meta --register-only", {"record_history": False, "silent": True})])
        self.assertEqual([(spec.name, spec.label) for spec in viewer.console_buttons.buttons()], [("metaDataBtn", "Meta Data")])
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        self.assertFalse(hasattr(viewer, "sidebar_buttons"))

    def test_a_session_with_nothing_persisted_registers_nothing(self):
        viewer = self.viewer
        viewer.process_command = mock.Mock()
        viewer.sidebar_buttons_to_persist = []
        viewer._register_persisted_console_buttons()
        del viewer.sidebar_buttons_to_persist  # an older session file without the attribute
        viewer._register_persisted_console_buttons()
        viewer.process_command.assert_not_called()

    def test_a_viewer_without_the_registration_method_is_skipped_by_the_pages(self):
        # The VR viewer has no Console window: a page's activation finds no add_console_button and does nothing.
        from types import SimpleNamespace

        from web_ui import agent_backend, esmfold_backend, meta_backend

        bare = SimpleNamespace(sidebar_buttons_to_persist=[], open_page=lambda client: None)
        for plugin in (agent_backend, esmfold_backend, meta_backend):
            plugin.activate(bare)
        self.assertEqual(bare.sidebar_buttons_to_persist, [])


class PersistedButtonRegistrationTests(unittest.TestCase):
    """The list of Console buttons comes from a layout cache, a file people share, so it is never typed as it stands.

    Every test's process_command is a recorder: nothing is dispatched, and a line it receives is a line the
    Viewer would have typed at its command line.
    """

    def setUp(self):
        import EMAPSSN_Viewer

        self.viewer = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
        self.typed = []
        self.viewer.process_command = lambda command, **options: self.typed.append((command, options))

    def register(self, persisted):
        """Set the list a loaded session holds and register; return the line printed."""
        import io

        self.viewer.sidebar_buttons_to_persist = persisted
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.viewer._register_persisted_console_buttons()
        return output.getvalue()

    def lines_typed(self):
        return [command for command, _options in self.typed]

    def test_a_command_line_in_the_list_is_not_typed(self):
        # The injection: a cache whose list holds "run C:/some/script.py" made the Viewer type
        # "run C:/some/script.py --register-only" at its command line when it opened.
        printed = self.register(["run C:/some/script.py", "meta"])
        self.assertEqual(self.lines_typed(), ["meta --register-only"])
        self.assertNotIn("run C:/some/script.py --register-only", self.lines_typed())
        self.assertEqual(self.viewer.sidebar_buttons_to_persist, ["meta"])
        self.assertEqual(printed.count("\n"), 1)
        self.assertIn("'run C:/some/script.py' (not a bare command name)", printed)

    def test_the_three_commands_that_declare_a_button_still_register_in_order_without_history(self):
        printed = self.register(["agent", "esmfold", "meta"])
        self.assertEqual(self.typed, [(f"{name} --register-only", {"record_history": False, "silent": True})
                                      for name in ("agent", "esmfold", "meta")])
        self.assertEqual(self.viewer.sidebar_buttons_to_persist, ["agent", "esmfold", "meta"])
        self.assertEqual(printed, "")  # a list the program wrote itself says nothing

    def test_an_entry_that_is_not_a_bare_name_is_not_typed_and_is_dropped_from_the_list(self):
        entries = [5, 1.5, True, None, b"meta", ["meta"], ("meta",), {"meta": 1}, {"meta"},
                   "meta upload x.csv", "meta --register-only", "agent hello", "meta;meta", " meta", "meta ", "meta\n",
                   "", "..", "a.b", "commands.meta", "-h", "run C:/some/script.py", "meta && run x"]
        for entry in entries:
            with self.subTest(entry=entry):
                self.typed.clear()
                printed = self.register([entry, "meta"])
                self.assertEqual(self.lines_typed(), ["meta --register-only"])
                self.assertEqual(self.viewer.sidebar_buttons_to_persist, ["meta"])
                self.assertEqual(printed.count("\n"), 1)
                self.assertIn("(not a bare command name)", printed)

    def test_a_command_without_a_console_button_or_without_a_module_is_not_typed(self):
        entries = ["run", "select", "save", "help", "undo", "console", "zoom", "jobs", "nosuchcommand", "_private",
                   "__init__", "__import__", "os"]
        for entry in entries:
            with self.subTest(entry=entry):
                self.typed.clear()
                printed = self.register([entry, "agent"])
                self.assertEqual(self.lines_typed(), ["agent --register-only"])
                self.assertEqual(self.viewer.sidebar_buttons_to_persist, ["agent"])
                self.assertIn(f"'{entry}' (not a command with a Console button)", printed)

    def test_a_name_listed_twice_is_typed_once(self):
        printed = self.register(["meta", "agent", "meta", "meta", "agent"])
        self.assertEqual(self.lines_typed(), ["meta --register-only", "agent --register-only"])
        self.assertEqual(self.viewer.sidebar_buttons_to_persist, ["meta", "agent"])
        self.assertEqual(printed.count("\n"), 1)
        self.assertEqual(printed.count("(listed twice)"), 3)

    def test_one_line_says_which_entries_were_ignored_and_why(self):
        printed = self.register([7, "run x", "meta", "nosuchcommand", "meta", "select"])
        self.assertEqual(printed.count("\n"), 1)
        self.assertTrue(printed.startswith("Warning: ignoring entries of the cache's list of Console buttons: "))
        for reason in ("7 (not a bare command name)", "'run x' (not a bare command name)",
                       "'nosuchcommand' (not a command with a Console button)", "'meta' (listed twice)",
                       "'select' (not a command with a Console button)"):
            self.assertIn(reason, printed)
        self.assertEqual(self.lines_typed(), ["meta --register-only"])

    def test_an_entry_of_any_length_is_quoted_short(self):
        printed = self.register(["x" * 5000, "meta"])
        self.assertEqual(printed.count("\n"), 1)
        self.assertLess(len(printed), 250)

    def test_what_is_not_a_list_registers_nothing_and_is_replaced_by_an_empty_list(self):
        import numpy as np

        for persisted in ("meta", "run C:/some/script.py", {"meta": 1}, 5, 0, "", np.array(["meta"]),
                          np.array(["meta", "agent"]), np.array([b"meta", b"agent"], dtype=object), b"meta", {"meta"}):
            with self.subTest(persisted=repr(persisted)):
                self.typed.clear()
                printed = self.register(persisted)
                self.assertEqual(self.typed, [])
                self.assertEqual(self.viewer.sidebar_buttons_to_persist, [])
                self.assertEqual(printed.count("\n"), 1)
                self.assertIn("is not a list", printed)

    def test_a_tuple_of_names_registers_like_a_list(self):
        self.register(("meta", "agent"))
        self.assertEqual(self.lines_typed(), ["meta --register-only", "agent --register-only"])

    def test_a_clean_list_keeps_its_identity_so_a_command_appends_to_the_one_the_viewer_saves(self):
        persisted = ["meta"]
        self.assertEqual(self.register(persisted), "")
        self.assertIs(self.viewer.sidebar_buttons_to_persist, persisted)

    def test_an_empty_or_missing_list_says_nothing(self):
        self.assertEqual(self.register([]), "")
        self.assertEqual(self.register(None), "")
        del self.viewer.sidebar_buttons_to_persist
        self.viewer._register_persisted_console_buttons()
        self.assertEqual(self.typed, [])

    def test_a_plugin_that_declares_a_button_is_typed_like_the_three(self):
        from tests.command_fixtures import scratch_commands

        plugins = {"zzbutton_plugin": "CONSOLE_BUTTON = True\ndef run(viewer, args):\n    pass\n",
                   "zzbutton_undeclared": "def run(viewer, args):\n    pass\n"}
        with scratch_commands(plugins):
            self.register(["zzbutton_undeclared", "zzbutton_plugin"])
        self.assertEqual(self.lines_typed(), ["zzbutton_plugin --register-only"])


if __name__ == "__main__":
    unittest.main()
