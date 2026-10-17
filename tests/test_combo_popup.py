# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A combo box's open list scrolls by the wheel, the scroll bar and the keys, never by the mouse.

Qt makes the row under the pointer the list's current row, and a list view
scrolls to its current row. Resting the pointer on the partly visible last row
of a long list therefore scrolled it, row by row, under the pointer, and a
click then picked a row the user had not been pointing at.
desktop.Desktop_App.SteadyPopupComboBox, which every combo box of Config, Tools
and the VR Config derives from, keeps the mouse from scrolling the list.

The lists here run under the Studio theme, which opens them below the combo box
with every entry in view, and are cut off by 12 px at the bottom, as a screen
edge cuts a popup, so the last row is partly visible.
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QCursor, QGuiApplication, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractSlider,
    QApplication,
    QComboBox,
    QVBoxLayout,
    QWidget,
)

from desktop.Desktop_App import (  # noqa: E402
    LanguageSelector,
    SteadyPopupComboBox,
    _ListScrollGuard,
)
from tests.config_gui_loader import load_config_namespace, open_config_window  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from tests.tools_gui_fixtures import isolated_tools_project  # noqa: E402

ROWS = 60  # many more than the ten a list shows
CLIPPED = 12  # px cut off the bottom of the list


def flush(app):
    for _ in range(4):
        app.processEvents()


class OpenList:
    """The open list of a combo box, driven by mouse and keys."""

    def __init__(self, test, combo, current=0, clipped=CLIPPED):
        self.app = test.app
        self.combo = combo
        if combo.count() < ROWS:
            combo.addItems([f"Entry {index}" for index in range(combo.count(), ROWS)])
        combo.setCurrentIndex(current)
        self.host = QWidget()
        QVBoxLayout(self.host).addWidget(combo)
        self.host.resize(300, 120)
        self.host.show()
        flush(self.app)
        test.addCleanup(self.close)
        self.open()
        if clipped:
            container = self.view.window()
            container.resize(container.width(), container.height() - clipped)
            flush(self.app)

    def open(self):
        self.combo.showPopup()
        flush(self.app)

    def close(self):
        self.combo.hidePopup()
        self.host.close()
        self.host.deleteLater()
        flush(self.app)

    @property
    def view(self):
        return self.combo.view()

    @property
    def viewport(self):
        return self.view.viewport()

    @property
    def bar(self):
        return self.view.verticalScrollBar()

    @property
    def scrolled(self):
        return self.bar.value()

    @property
    def current(self):
        return self.view.currentIndex().row()

    @property
    def partial_row(self):
        """The row cut off by the bottom of the list, and a y inside its visible part."""
        y = self.viewport.height() - 3
        return self.view.indexAt(QPoint(20, y)).row(), y

    def row_y(self, row):
        """A y in the middle of the part of ``row`` that shows."""
        rect = self.view.visualRect(self.view.model().index(row, 0))
        visible = rect.intersected(self.viewport.rect())
        return visible.center().y()

    def is_fully_visible(self, row):
        rect = self.view.visualRect(self.view.model().index(row, 0))
        return self.viewport.rect().contains(rect)

    def mouse(self, kind, y, x=20, button=Qt.MouseButton.NoButton, buttons=Qt.MouseButton.NoButton):
        local = QPoint(x, y)
        event = QMouseEvent(
            kind, QPointF(local), QPointF(self.viewport.mapToGlobal(local)),
            button, buttons, Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.viewport, event)
        flush(self.app)

    def hover(self, y, x=20):
        self.mouse(QEvent.Type.MouseMove, y, x)

    def press(self, y, x=20):
        self.mouse(QEvent.Type.MouseButtonPress, y, x, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)

    def drag(self, y, x=20):
        self.mouse(QEvent.Type.MouseMove, y, x, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)

    def release(self, y, x=20):
        self.mouse(QEvent.Type.MouseButtonRelease, y, x, Qt.MouseButton.LeftButton)

    def park_cursor(self, y, x=20):
        """Put the (offscreen) cursor where a held mouse button would have left it."""
        if QGuiApplication.platformName() == "offscreen":
            QCursor.setPos(self.viewport.mapToGlobal(QPoint(x, y)))

    def bar_mouse(self, kind, y, buttons=Qt.MouseButton.NoButton):
        """A mouse event on the scroll bar, ``y`` px down it."""
        local = QPoint(self.bar.width() // 2, y)
        button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
        event = QMouseEvent(
            kind, QPointF(local), QPointF(self.bar.mapToGlobal(local)),
            button, buttons, Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.bar, event)
        flush(self.app)

    def wheel(self, delta):
        local = QPoint(20, 100)
        event = QWheelEvent(
            QPointF(local), QPointF(self.viewport.mapToGlobal(local)), QPoint(), QPoint(0, delta),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
        )
        QApplication.sendEvent(self.viewport, event)
        flush(self.app)

    def key(self, key):
        QTest.keyClick(self.view, key)
        flush(self.app)


class PopupCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)


class SteadyListTests(PopupCase):
    """The behaviour itself, on the shared base class."""

    def make_combo(self):
        return SteadyPopupComboBox()

    def open_list(self, **kwargs):
        return OpenList(self, self.make_combo(), **kwargs)

    def test_the_list_is_cut_off_below_a_whole_row(self):
        # The premise of the scrolling tests: a row is only partly in view.
        popup = self.open_list()
        row, y = popup.partial_row
        self.assertEqual(row, 9)
        self.assertFalse(popup.is_fully_visible(row))
        self.assertGreater(popup.bar.maximum(), 0)

    def test_resting_the_pointer_on_the_partly_visible_row_does_not_scroll_the_list(self):
        popup = self.open_list()
        row, y = popup.partial_row
        for step in range(6):
            popup.hover(y - step % 3, x=20 + step)
        self.assertEqual(popup.scrolled, 0)
        self.assertEqual(popup.current, row)
        # The same on a list that has been scrolled part of the way.
        popup.bar.setValue(20)
        row, y = popup.partial_row
        for step in range(6):
            popup.hover(y - step % 3, x=20 + step)
        self.assertEqual(popup.scrolled, 20)
        self.assertEqual(popup.current, row)

    def test_a_pointer_moved_through_the_window_system_does_not_scroll_the_list(self):
        # QTest moves the cursor, so the move reaches the list the way a mouse's does.
        if QGuiApplication.platformName() != "offscreen":
            self.skipTest("moves the real cursor")
        popup = self.open_list()
        row, y = popup.partial_row
        for step in range(6):
            QTest.mouseMove(popup.viewport, QPoint(20 + step, y - step % 3))
            flush(self.app)
        self.assertEqual(popup.scrolled, 0)
        self.assertEqual(popup.current, row)

    def test_dragging_to_the_bottom_edge_does_not_scroll_the_list(self):
        popup = self.open_list()
        row, y = popup.partial_row
        popup.press(popup.row_y(4))
        popup.park_cursor(y)
        popup.drag(y)
        QTest.qWait(300)  # a held button on the edge would run a timer
        popup.drag(y - 1)
        self.assertEqual(popup.scrolled, 0)
        self.assertEqual(popup.current, row)
        popup.release(y)
        # Letting go picks the row the pointer is on, not one that scrolled under it.
        self.assertEqual(popup.combo.currentIndex(), row)
        self.assertFalse(popup.view.window().isVisible())

    def test_the_wheel_and_the_scroll_bar_still_scroll(self):
        popup = self.open_list()
        popup.wheel(-120)
        self.assertGreater(popup.scrolled, 0)
        popup.bar.setValue(0)
        popup.bar.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
        self.assertEqual(popup.scrolled, 1)
        popup.bar.triggerAction(QAbstractSlider.SliderAction.SliderPageStepAdd)
        self.assertGreater(popup.scrolled, 1)
        popup.bar.setSliderDown(True)  # dragging the handle
        popup.bar.setSliderPosition(30)
        popup.bar.setSliderDown(False)
        self.assertEqual(popup.scrolled, 30)

    def test_they_scroll_after_the_mouse_has_been_over_the_list_too(self):
        popup = self.open_list()
        popup.hover(popup.row_y(3))
        self.assertFalse(popup.view.hasAutoScroll())
        popup.wheel(-120)
        self.assertGreater(popup.scrolled, 0)
        scrolled = popup.scrolled
        popup.wheel(120)
        self.assertLess(popup.scrolled, scrolled)

    def test_a_click_on_the_scroll_bar_track_scrolls_by_a_page(self):
        popup = self.open_list()
        self.assertTrue(popup.bar.isVisible())
        below_the_handle = popup.bar.height() - 12
        popup.bar_mouse(QEvent.Type.MouseButtonPress, below_the_handle, Qt.MouseButton.LeftButton)
        popup.bar_mouse(QEvent.Type.MouseButtonRelease, below_the_handle)
        self.assertGreater(popup.scrolled, 0)

    def test_dragging_the_scroll_bar_handle_scrolls_the_list(self):
        popup = self.open_list()
        popup.hover(popup.row_y(3))  # the mouse has been in the list before
        popup.bar_mouse(QEvent.Type.MouseButtonPress, 20, Qt.MouseButton.LeftButton)
        popup.bar_mouse(QEvent.Type.MouseMove, 120, Qt.MouseButton.LeftButton)
        self.assertGreater(popup.scrolled, 10)
        popup.bar_mouse(QEvent.Type.MouseMove, 60, Qt.MouseButton.LeftButton)
        popup.bar_mouse(QEvent.Type.MouseButtonRelease, 60)
        self.assertGreater(popup.scrolled, 0)
        self.assertLess(popup.scrolled, popup.bar.maximum())

    def test_the_highlight_follows_the_pointer(self):
        popup = self.open_list()
        row, y = popup.partial_row
        for target in (2, 6, 1, row):
            popup.hover(popup.row_y(target))
            self.assertEqual(popup.current, target)
        self.assertEqual(popup.scrolled, 0)

    def test_a_click_on_a_whole_row_selects_it_and_closes_the_list(self):
        popup = self.open_list()
        QTest.mouseClick(popup.viewport, Qt.MouseButton.LeftButton, pos=QPoint(20, popup.row_y(4)))
        flush(self.app)
        self.assertEqual(popup.combo.currentIndex(), 4)
        self.assertFalse(popup.view.window().isVisible())

    def test_a_click_on_the_partly_visible_row_selects_it_and_closes_the_list(self):
        popup = self.open_list()
        row, y = popup.partial_row
        QTest.mouseClick(popup.viewport, Qt.MouseButton.LeftButton, pos=QPoint(20, y))
        flush(self.app)
        self.assertEqual(popup.combo.currentIndex(), row)
        self.assertFalse(popup.view.window().isVisible())

    def test_a_click_after_hovering_selects_the_row_that_was_pointed_at(self):
        popup = self.open_list()
        row, y = popup.partial_row
        popup.hover(y)
        popup.press(y)
        self.assertEqual(popup.scrolled, 0)
        popup.release(y)
        self.assertEqual(popup.combo.currentIndex(), row)

    def test_the_list_opens_at_the_current_entry(self):
        popup = self.open_list(current=45, clipped=0)
        self.assertTrue(popup.is_fully_visible(45))
        self.assertEqual(popup.current, 45)

    def test_it_opens_at_the_current_entry_after_the_mouse_used_the_list_before(self):
        popup = self.open_list()
        popup.hover(popup.row_y(5))
        self.assertFalse(popup.view.hasAutoScroll())
        popup.combo.hidePopup()
        popup.combo.setCurrentIndex(50)
        popup.open()
        self.assertTrue(popup.view.hasAutoScroll())
        self.assertTrue(popup.is_fully_visible(50), popup.scrolled)

    def test_the_keys_scroll_to_keep_the_highlighted_row_in_view(self):
        popup = self.open_list()
        for _ in range(14):
            popup.key(Qt.Key.Key_Down)
            self.assertTrue(popup.is_fully_visible(popup.current), popup.current)
        self.assertEqual(popup.current, 14)
        self.assertGreater(popup.scrolled, 0)
        popup.key(Qt.Key.Key_PageDown)
        self.assertGreater(popup.current, 14)
        self.assertTrue(popup.is_fully_visible(popup.current))
        popup.key(Qt.Key.Key_End)
        self.assertEqual(popup.current, ROWS - 1)
        self.assertTrue(popup.is_fully_visible(ROWS - 1))
        self.assertEqual(popup.scrolled, popup.bar.maximum())
        popup.key(Qt.Key.Key_PageUp)
        self.assertLess(popup.current, ROWS - 1)
        self.assertTrue(popup.is_fully_visible(popup.current))
        popup.key(Qt.Key.Key_Up)
        self.assertTrue(popup.is_fully_visible(popup.current))
        popup.key(Qt.Key.Key_Home)
        self.assertEqual(popup.current, 0)
        self.assertEqual(popup.scrolled, 0)

    def test_the_keys_scroll_after_the_pointer_rested_on_the_partly_visible_row(self):
        popup = self.open_list()
        row, y = popup.partial_row
        popup.hover(y)
        self.assertFalse(popup.view.hasAutoScroll())
        popup.key(Qt.Key.Key_Down)
        self.assertEqual(popup.current, row + 1)
        self.assertTrue(popup.is_fully_visible(row + 1))
        self.assertGreater(popup.scrolled, 0)
        # And the pointer takes over again, without scrolling.
        scrolled = popup.scrolled
        popup.hover(popup.row_y(popup.partial_row[0]))
        popup.hover(popup.partial_row[1])
        self.assertEqual(popup.scrolled, scrolled)

    def test_a_list_opened_with_the_keyboard_still_follows_its_keys(self):
        popup = self.open_list(clipped=0)
        popup.hover(popup.row_y(2))  # the previous time the mouse used it
        popup.combo.hidePopup()
        popup.combo.setCurrentIndex(0)
        popup.open()
        for _ in range(12):
            popup.key(Qt.Key.Key_Down)
        self.assertTrue(popup.is_fully_visible(12))
        self.assertGreater(popup.scrolled, 0)

    def test_the_guard_is_added_to_the_list_once_and_only_when_it_opens(self):
        combo = self.make_combo()
        combo.addItems(["a", "b"])
        self.assertEqual(combo.view().findChildren(_ListScrollGuard), [])
        popup = OpenList(self, combo, clipped=0)
        for _ in range(3):
            popup.combo.hidePopup()
            popup.open()
        self.assertEqual(len(popup.view.findChildren(_ListScrollGuard)), 1)


class DerivedChecks:
    """What every combo box class of the programs must do (a mixin for a test case).

    ``make_combo`` builds one; the checks are the ones a combo box that kept
    its own mouse or popup handling could fail.
    """

    def make_combo(self):
        raise NotImplementedError

    def open_list(self, **kwargs):
        return OpenList(self, self.make_combo(), **kwargs)

    def test_it_is_a_steady_popup_combo_box(self):
        self.assertIsInstance(self.make_combo(), SteadyPopupComboBox)

    def test_resting_the_pointer_on_the_partly_visible_row_does_not_scroll_its_list(self):
        popup = self.open_list()
        row, y = popup.partial_row
        for step in range(5):
            popup.hover(y - step % 3, x=20 + step)
        self.assertEqual(popup.scrolled, 0)
        self.assertEqual(popup.current, row)

    def test_a_click_on_the_partly_visible_row_selects_it(self):
        popup = self.open_list()
        row, y = popup.partial_row
        QTest.mouseClick(popup.viewport, Qt.MouseButton.LeftButton, pos=QPoint(20, y))
        flush(self.app)
        self.assertEqual(popup.combo.currentIndex(), row)
        self.assertFalse(popup.view.window().isVisible())

    def test_the_keys_still_scroll_its_list(self):
        popup = self.open_list()
        for _ in range(12):
            popup.key(Qt.Key.Key_Down)
        self.assertTrue(popup.is_fully_visible(12))
        self.assertGreater(popup.scrolled, 0)


class LanguageSelectorTests(DerivedChecks, PopupCase):

    def make_combo(self):
        return LanguageSelector(setting="en")


class ConfigComboTests(DerivedChecks, PopupCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.namespace = load_config_namespace()

    def make_combo(self):
        return self.namespace["NoScrollComboBox"]()


class ConfigDynamicComboTests(ConfigComboTests):
    def make_combo(self):
        self.refreshed = []
        return self.namespace["DynamicComboBox"](lambda: self.refreshed.append(True))

    def test_it_still_lists_its_entries_again_as_it_opens(self):
        self.open_list()
        self.assertTrue(self.refreshed)


class ToolsComboTests(DerivedChecks, PopupCase):

    def make_combo(self):
        from EMAPSSN_Tools import NoScrollComboBox

        return NoScrollComboBox()


class ToolsDynamicComboTests(DerivedChecks, PopupCase):

    def make_combo(self):
        from EMAPSSN_Tools import DynamicComboBox

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # It lists its folder again as it opens, so the folder holds the entries.
        for index in range(ROWS):
            (Path(directory.name) / f"Entry {index:02d}.fasta").write_text("")
        return DynamicComboBox(directory.name, ".fasta")


class EveryComboBoxTests(PopupCase):
    """No window builds a combo box that lacks the steady list."""

    def assert_all_steady(self, window):
        combos = window.findChildren(QComboBox)
        self.assertGreater(len(combos), 5)
        plain = [combo for combo in combos if not isinstance(combo, SteadyPopupComboBox)]
        self.assertEqual(plain, [])

    def test_the_config_window_has_only_steady_combo_boxes(self):
        namespace = load_config_namespace()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        window = open_config_window(namespace["ConfigGUI"], directory.name)
        self.addCleanup(self.close, window)
        self.assert_all_steady(window)

    def test_the_tools_window_has_only_steady_combo_boxes(self):
        from PySide6.QtWidgets import QTextBrowser
        from unittest import mock
        from EMAPSSN_Tools import ToolsGUI

        isolated_tools_project(self)
        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = ToolsGUI()
        self.addCleanup(self.close, window)
        self.assert_all_steady(window)

    def close(self, window):
        window.close()
        window.deleteLater()
        flush(self.app)


if __name__ == "__main__":
    unittest.main()
