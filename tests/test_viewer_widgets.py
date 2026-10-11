# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's widgets (desktop.Viewer_Widgets): its controls card and the
info button that opens it, its sidebar's rounded card and buttons, and the
View Width box.

The Viewer lists its mouse and keyboard actions in one translated text. The
card reads it, as shipped in English and in Chinese, into key caps beside
what each action does, on a light tint of the theme colour. A sidebar button
centres its label in the room after its icon. The View Width box steps the
view's width by a factor, as the wheel zooms, and reports only what the
user set.
"""

import ast
import os
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ElementTree

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QPoint, QSize, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QRegion  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from desktop.Desktop_App import fit_buttons_to_text  # noqa: E402
from desktop.Studio_Theme import TOKENS, studio_icon  # noqa: E402
from desktop.Viewer_Widgets import (  # noqa: E402
    SIDEBAR_INSET,
    ControlsButton,
    ControlsCard,
    SidebarButton,
    SidebarPanel,
    VIEW_WIDTH_BOX_HEIGHT,
    VIEW_WIDTH_STEP,
    ViewWidthBox,
    ViewWidthSpinBox,
    controls_rows,
)
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

VIEWER = SRC / "EMAPSSN_Viewer.py"
ZH_CATALOG = SRC / "resources" / "languages" / "zh_CN" / "emapssn_zh_CN.ts"


def shipped_actions():
    """The Viewer's list of actions, as its source passes it to translate()."""
    tree = ast.parse(VIEWER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "translate"
                and len(node.args) == 2 and isinstance(node.args[1], ast.Constant)
                and str(node.args[1].value).startswith("[ENTER]")):
            return node.args[1].value
    raise AssertionError("The Viewer no longer lists its actions")


def chinese_actions(source):
    """The Chinese catalog's text for source."""
    for message in ElementTree.parse(ZH_CATALOG).getroot().iter("message"):
        if message.findtext("source") == source:
            return message.findtext("translation")
    raise AssertionError("The Chinese catalog has no text for the Viewer's actions")


def flush(app):
    for _ in range(4):
        app.processEvents()


class ControlsRowsTests(unittest.TestCase):
    def test_the_shipped_actions_read_as_keys_and_what_they_do(self):
        rows = controls_rows(shipped_actions())
        self.assertEqual(rows, [
            (["ENTER"], "Command"),
            (["LeftClick"], "Highlight"),
            (["RightClick"], "Select/Clear"),
            (["Scroll"], "Zoom"),
            (["LeftClick", "Shift/Ctrl"], "Copy Node Header/Sequence"),
            (["LeftClick", "Drag"], "Pan"),
            (["RightClick", "Drag"], "GroupSelect/MoveNodes"),
        ])

    def test_the_chinese_actions_read_the_same_way(self):
        rows = controls_rows(chinese_actions(shipped_actions()))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0], (["回车"], "命令"))
        self.assertEqual(rows[4][0], ["左键", "Shift/Ctrl"])
        self.assertTrue(all(keys and description for keys, description in rows), rows)

    def test_an_action_without_keys_is_all_description(self):
        self.assertEqual(controls_rows("[A] One |  | Two words | [B + C] Three"),
                         [(["A"], "One"), ([], "Two words"), (["B", "C"], "Three")])
        self.assertEqual(controls_rows(""), [])


class ControlsCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def setUp(self):
        self.host = QWidget()
        self.addCleanup(self.host.deleteLater)
        # Clear of the screen's left edge (the offscreen screen is 800 px wide),
        # so the card is not pushed back onto it.
        self.host.setGeometry(10, 400, 400, 200)
        self.button = ControlsButton("Mouse and keyboard controls", shipped_actions(), self.host)
        self.button.move(20, 150)
        self.host.show()
        flush(self.app)

    def open_card(self):
        card = self.button.toggle_card()
        flush(self.app)
        self.assertTrue(card.isVisible())
        return card

    def test_the_button_names_the_card_and_leaves_the_keyboard_with_the_canvas(self):
        self.assertEqual(self.button.toolTip(), "Mouse and keyboard controls")
        self.assertEqual(self.button.accessibleName(), "Mouse and keyboard controls")
        self.assertEqual(self.button.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertFalse(self.button.icon().isNull())
        self.assertEqual(self.button.size(), QSize(ControlsButton.SIZE, ControlsButton.SIZE))

    def test_the_card_shows_each_action_as_key_caps_beside_what_it_does(self):
        card = self.open_card()
        self.assertEqual(card.title_label.text(), "Mouse and keyboard controls")
        self.assertEqual([label.text() for label in card.descriptions],
                         [description for _keys, description in controls_rows(shipped_actions())])
        self.assertEqual([cap.text() for cap in card.key_caps],
                         ["ENTER", "LeftClick", "RightClick", "Scroll", "LeftClick", "Shift/Ctrl",
                          "LeftClick", "Drag", "RightClick", "Drag"])
        # Each description shows whole: no row is cut short.
        for label in card.descriptions:
            self.assertGreaterEqual(label.width(), label.sizeHint().width(), label.text())

    def test_the_card_opens_above_the_button_on_a_tint_of_the_theme_colour(self):
        card = self.open_card()
        rect = card.card_rect().translated(card.mapToGlobal(QPoint(0, 0)).toPointF())
        button = self.button.mapToGlobal(QPoint(0, 0))
        # Well clear of the button and the slider beside it: 6 px read as touching them.
        gap = button.y() - rect.bottom()
        self.assertAlmostEqual(gap, ControlsCard.ANCHOR_GAP, delta=1)
        self.assertGreaterEqual(gap, 12)
        self.assertAlmostEqual(rect.left(), button.x(), delta=1)
        image = card.grab().toImage()
        inside = card.card_rect().center().toPoint() + QPoint(0, 1)
        # Between rows, away from the text: the card's own fill.
        probe = QPoint(round(card.card_rect().right()) - 4, inside.y())
        self.assertEqual(image.pixelColor(probe * image.devicePixelRatio()).name(), TOKENS["tint"])

    def test_escape_and_the_cross_close_the_card(self):
        card = self.open_card()
        QTest.keyClick(card, Qt.Key.Key_Escape)
        flush(self.app)
        self.assertFalse(self.button.card_is_open())
        card = self.open_card()
        QTest.mouseClick(card.close_button, Qt.MouseButton.LeftButton)
        flush(self.app)
        self.assertFalse(self.button.card_is_open())

    def test_a_press_on_the_button_closes_the_card_without_opening_it_again(self):
        card = self.open_card()
        on_button = card.mapFromGlobal(self.button.mapToGlobal(self.button.rect().center()))
        QTest.mousePress(card, Qt.MouseButton.LeftButton, pos=on_button)
        self.assertTrue(card.testAttribute(Qt.WidgetAttribute.WA_NoMouseReplay))
        flush(self.app)
        self.assertFalse(self.button.card_is_open())
        # A press anywhere else outside closes it too, and that press goes on.
        card = self.open_card()
        QTest.mousePress(card, Qt.MouseButton.LeftButton, pos=QPoint(card.width() + 40, 5))
        self.assertFalse(card.testAttribute(Qt.WidgetAttribute.WA_NoMouseReplay))
        flush(self.app)
        self.assertFalse(self.button.card_is_open())

    def test_a_click_while_the_card_is_open_closes_it(self):
        self.open_card()
        self.assertIsNone(self.button.toggle_card())
        flush(self.app)
        self.assertFalse(self.button.card_is_open())


class ViewWidthBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make_spin_box(self, width=242.0, narrowest=0.5, widest=1100.0):
        spin_box = ViewWidthSpinBox()
        self.addCleanup(spin_box.deleteLater)
        spin_box.show_width(width, narrowest, widest)
        sets = []
        spin_box.valueChanged.connect(sets.append)
        return spin_box, sets

    def test_the_view_shows_without_a_signal_and_the_user_sets_with_one(self):
        spin_box, sets = self.make_spin_box()
        self.assertEqual(spin_box.text(), "242.0")
        self.assertEqual((spin_box.minimum(), spin_box.maximum()), (0.5, 1100.0))
        spin_box.show_width(300.04, 0.5, 1200.0)
        self.assertEqual((spin_box.text(), sets), ("300.0", []))
        spin_box.setValue(500.0)
        self.assertEqual(sets, [500.0])
        # A width beyond the limits, as after the nodes were moved together, still shows.
        spin_box.show_width(1500.0, 0.5, 1100.0)
        self.assertEqual(spin_box.value(), 1500.0)
        # With no limit to keep to, a wide view still shows.
        spin_box.show_width(5000.0, 0.5, None)
        self.assertEqual(spin_box.value(), 5000.0)

    def test_a_step_zooms_by_a_factor_and_stops_at_the_limits(self):
        spin_box, sets = self.make_spin_box(width=400.0)
        spin_box.stepBy(1)
        spin_box.stepBy(-2)
        self.assertEqual(sets, [400.0 * VIEW_WIDTH_STEP, round(400.0 / VIEW_WIDTH_STEP, 1)])
        self.assertGreaterEqual(VIEW_WIDTH_STEP, 1.1)
        spin_box.stepBy(20)
        self.assertEqual(sets[-1], 1100.0)
        spin_box.stepBy(-60)
        self.assertEqual(sets[-1], 0.5)

    def test_typing_counts_on_enter_and_esc_puts_back_the_view(self):
        spin_box, sets = self.make_spin_box()
        dismissed = []
        spin_box.dismissed.connect(lambda: dismissed.append(True))
        self.assertFalse(spin_box.keyboardTracking())
        spin_box.lineEdit().selectAll()
        QTest.keyClicks(spin_box, "750")
        self.assertEqual(sets, [])
        QTest.keyClick(spin_box, Qt.Key.Key_Escape)
        self.assertEqual((spin_box.text(), sets, dismissed), ("242.0", [], [True]))
        spin_box.lineEdit().selectAll()
        QTest.keyClicks(spin_box, "750")
        QTest.keyClick(spin_box, Qt.Key.Key_Return)
        self.assertEqual(sets, [750.0])

    def test_the_view_waits_while_the_box_is_typed_in(self):
        spin_box, _sets = self.make_spin_box()
        window = QWidget()
        self.addCleanup(window.deleteLater)
        spin_box.setParent(window)
        window.show()
        spin_box.setFocus()
        self.app.processEvents()
        if not spin_box.hasFocus():
            self.skipTest("this platform gives no window the keyboard")
        spin_box.show_width(900.0, 0.5, 1100.0)
        self.assertEqual(spin_box.text(), "242.0")

    def test_it_never_takes_the_keyboard_from_the_network_unasked(self):
        spin_box, _sets = self.make_spin_box()
        self.assertEqual(spin_box.focusPolicy(), Qt.FocusPolicy.ClickFocus)

    def test_its_width_holds_as_the_limits_change(self):
        spin_box, _sets = self.make_spin_box(widest=100.0)
        narrow = spin_box.sizeHint()
        spin_box.show_width(242.0, 0.5, 250000.0)
        self.assertEqual(spin_box.sizeHint(), narrow)
        self.assertEqual(narrow.height(), VIEW_WIDTH_BOX_HEIGHT)
        self.assertGreaterEqual(narrow.width(), QFontMetrics(spin_box.font()).horizontalAdvance("999999.9"))

    def test_the_text_either_side_of_the_width(self):
        for template, before, after in (("View Width: {width}", "View Width:", ""),
                                        ("{width} units wide", "", "units wide"),
                                        ("视图宽度：{width}", "视图宽度：", "")):
            box = ViewWidthBox(template)
            self.addCleanup(box.deleteLater)
            with self.subTest(template=template):
                self.assertEqual((box.before.text(), box.after.text()), (before, after))
                self.assertEqual(box.before.isVisibleTo(box), bool(before))
                self.assertEqual(box.after.isVisibleTo(box), bool(after))
                layout = box.layout()
                self.assertEqual([layout.itemAt(index).widget() for index in range(layout.count())],
                                 [box.before, box.spin_box, box.after])


class SidebarWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make_sidebar(self, labels=(("Meta Data", "chart-column"), ("Agent", "bot"))):
        """A sidebar as the Viewer builds it, its buttons fitted to the widest label."""
        canvas = QWidget()
        self.addCleanup(canvas.deleteLater)
        canvas.resize(600, 400)
        panel = SidebarPanel(canvas)
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 20, 10 + SIDEBAR_INSET, 20)
        canvas.setStyleSheet("QWidget#rightPanel QPushButton { padding-left: 10px; min-height: 24px; }")
        buttons = []
        for label, icon in labels:
            button = SidebarButton(label, panel)
            button.setIcon(studio_icon(icon))
            button.setIconSize(QSize(16, 16))
            layout.addWidget(button)
            buttons.append(button)
        layout.addStretch()
        width = fit_buttons_to_text(*buttons) + 20 + SIDEBAR_INSET
        panel.setGeometry(600 - width, 0, width, 400)
        canvas.show()
        flush(self.app)
        return panel, buttons

    def test_the_card_has_round_corners_clear_of_the_canvas_edges(self):
        panel, _buttons = self.make_sidebar()
        card = panel.card_rect()
        self.assertAlmostEqual(card.top(), SIDEBAR_INSET, delta=1)
        self.assertAlmostEqual(panel.width() - card.right(), SIDEBAR_INSET, delta=1)
        self.assertAlmostEqual(panel.height() - card.bottom(), SIDEBAR_INSET, delta=1)
        image = QImage(panel.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            # Only what the panel paints: the canvas's background stays out of its corners.
            panel.render(painter, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        finally:
            painter.end()
        # The card's corner is round: its corner pixel is clear, a pixel inside is filled.
        self.assertEqual(image.pixelColor(1, SIDEBAR_INSET + 1).alpha(), 0)
        self.assertEqual(image.pixelColor(panel.width() - SIDEBAR_INSET - 2, panel.height() - SIDEBAR_INSET - 2)
                         .alpha(), 0)
        self.assertGreater(image.pixelColor(panel.width() // 2, panel.height() // 2).alpha(), 200)
        # Above the card, the canvas shows through.
        self.assertEqual(image.pixelColor(panel.width() // 2, SIDEBAR_INSET // 2).alpha(), 0)

    def ink_columns(self, button):
        """The x of every column where the button draws dark text, right of its icon."""
        image = button.grab().toImage()
        ratio = image.devicePixelRatio()
        start = round(button.icon_rect().right() * ratio) + 2
        middle = round(button.height() / 2 * ratio)
        columns = []
        for x in range(start, image.width()):
            for y in range(middle - round(4 * ratio), middle + round(4 * ratio)):
                if QColor(image.pixel(x, y)).lightness() < 110:
                    columns.append(x / ratio)
                    break
        return columns

    def test_a_label_is_centred_in_the_room_after_its_icon(self):
        _panel, (meta, agent) = self.make_sidebar()
        self.assertEqual(meta.width(), agent.width())
        for button in (meta, agent):
            with self.subTest(label=button.text()):
                room = button.label_rect()
                self.assertGreater(room.left(), button.icon_rect().right())
                # The widest label fits its room, so nothing clips.
                self.assertLessEqual(button.fontMetrics().horizontalAdvance(button.text()), room.width())
                ink = self.ink_columns(button)
                self.assertTrue(ink)
                # The text's box is centred in the room: its ink starts where the
                # centred box does, give or take the first letter's bearing.
                metrics = QFontMetrics(button.font())
                box_left = room.left() + (room.width() - metrics.horizontalAdvance(button.text())) / 2
                bounds = metrics.tightBoundingRect(button.text())
                self.assertAlmostEqual(ink[0], box_left + bounds.left(), delta=1.5)
        # The shorter label starts later than the longer: it is centred, not left-aligned.
        self.assertGreater(self.ink_columns(agent)[0], self.ink_columns(meta)[0] + 5)


if __name__ == "__main__":
    unittest.main()
