# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The colour picker (desktop.Desktop_App.ColorPicker), and the Pick buttons that open it.

The picker pops up under its button and fills the colour field at once, as
the user drags in its square or hue strip, types a code or clicks a preset;
Esc puts the field's text back.
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QWidget  # noqa: E402
import shiboken6  # noqa: E402

from desktop.Desktop_App import COLOR_PRESETS, color_from_text, open_color_picker  # noqa: E402
from desktop.Studio_Theme import TOKENS  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402


def flush(app):
    for _ in range(6):
        app.processEvents()


def inside(rect, across, down):
    """The point a fraction across and down rect."""
    return QPoint(round(rect.x() + across * rect.width()), round(rect.y() + down * rect.height()))


def click(widget, point):
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)


def closed(picker):
    return not shiboken6.isValid(picker) or not picker.isVisible()


class ColorPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def open_for(self, text):
        """A field holding text with a Pick button, and the picker the button opened."""
        host = QWidget()
        self.addCleanup(host.deleteLater)
        field = QLineEdit(text, host)
        field.setGeometry(10, 10, 200, 28)
        button = QPushButton("Pick", host)
        button.setGeometry(220, 10, 60, 28)
        host.resize(300, 60)
        host.move(100, 100)
        host.show()
        flush(self.app)
        picker = open_color_picker(button, field)
        self.addCleanup(lambda: shiboken6.isValid(picker) and picker.close())
        flush(self.app)
        return field, button, picker

    def test_it_opens_under_its_button_at_the_fields_colour(self):
        field, button, picker = self.open_for("#4488ff")
        self.assertTrue(picker.isVisible())
        card = picker.card_rect().translated(QPointF(picker.mapToGlobal(QPoint(0, 0))))
        corner = button.mapToGlobal(QPoint(button.width(), button.height()))
        self.assertGreater(card.top(), corner.y())
        self.assertAlmostEqual(card.right(), corner.x(), delta=1)
        self.assertEqual(picker.hex_field.text(), "#4488ff")
        self.assertTrue(picker.hex_field.hasFocus())
        self.assertEqual(field.text(), "#4488ff", "opening changes nothing")

    def test_a_field_with_no_colour_opens_at_white(self):
        for text in ("", "None", "not a colour"):
            with self.subTest(text=text):
                field, _button, picker = self.open_for(text)
                self.assertEqual(picker.color().name(), "#ffffff")
                self.assertEqual(field.text(), text)
                picker.close()

    def test_the_square_and_the_hue_strip_fill_the_field_at_once(self):
        field, _button, picker = self.open_for("#4488ff")
        hue = picker.color().hsvHue()
        # Top right: the hue itself, at full saturation and brightness.
        click(picker, inside(picker.area_rect(), 0.999, 0.0))
        color = color_from_text(field.text())
        self.assertEqual((color.hsvSaturation(), color.value()), (255, 255))
        self.assertAlmostEqual(color.hsvHue(), hue, delta=2)
        # The middle of the strip is cyan, at that saturation and brightness.
        click(picker, inside(picker.hue_rect(), 0.5, 0.5))
        self.assertEqual(field.text(), "#00ffff")
        # A drag follows the mouse; the bottom is black.
        area = picker.area_rect()
        QTest.mousePress(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         inside(area, 0.5, 0.5))
        QTest.mouseMove(picker, inside(area, 0.2, 0.999))
        QTest.mouseRelease(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                           inside(area, 0.2, 0.999))
        self.assertEqual(field.text(), "#000000")
        self.assertTrue(picker.isVisible(), "a choice keeps the card open")

    def test_a_grey_keeps_the_hue_the_strip_shows(self):
        field, _button, picker = self.open_for("#4488ff")
        hue = picker.color().hsvHue()
        click(picker, picker.preset_rect(COLOR_PRESETS.index("#808080")).center().toPoint())
        self.assertEqual(field.text(), "#808080")
        click(picker, inside(picker.area_rect(), 0.999, 0.0))
        self.assertAlmostEqual(color_from_text(field.text()).hsvHue(), hue, delta=2)

    def test_a_preset_fills_the_field_and_is_ringed(self):
        field, _button, picker = self.open_for("#4488ff")
        rect = picker.preset_rect(COLOR_PRESETS.index("#ffaa00"))
        click(picker, rect.center().toPoint())
        self.assertEqual(field.text(), "#ffaa00")
        self.assertEqual(picker.hex_field.text(), "#ffaa00")
        flush(self.app)
        image = picker.grab().toImage()
        ratio = image.devicePixelRatio()
        # The ring, 2.5 to 4.5 px out from the circle's left edge, is the accent.
        accent = QColor(TOKENS["accent"])

        def distance(x):
            pixel = image.pixelColor(round(x * ratio), round(rect.center().y() * ratio))
            return sum(abs(a - b) for a, b in zip(pixel.getRgb()[:3], accent.getRgb()[:3]))

        self.assertLess(min(distance(rect.left() - offset) for offset in (2.5, 3.0, 3.5, 4.0, 4.5)), 60)

    def test_typing_a_code_fills_the_field(self):
        field, _button, picker = self.open_for("#4488ff")
        QTest.keyClicks(picker.hex_field, "ff0000")
        self.assertEqual(field.text(), "#ff0000")
        self.assertEqual(picker.hex_field.text(), "ff0000", "the code isn't rewritten while typed")
        picker.hex_field.selectAll()
        QTest.keyClicks(picker.hex_field, "zz")
        self.assertEqual(field.text(), "#ff0000", "an unreadable code changes nothing")
        picker.hex_field.selectAll()
        QTest.keyClicks(picker.hex_field, "teal")
        self.assertEqual(field.text(), "#008080")

    def test_enter_keeps_the_colour_and_esc_puts_back_the_fields_text(self):
        field, _button, picker = self.open_for("grey")
        click(picker, picker.preset_rect(0).center().toPoint())
        self.assertEqual(field.text(), "#000000")
        QTest.keyClick(picker.hex_field, Qt.Key.Key_Escape)
        flush(self.app)
        self.assertTrue(closed(picker))
        self.assertEqual(field.text(), "grey")

        field, _button, picker = self.open_for("grey")
        click(picker, picker.preset_rect(0).center().toPoint())
        QTest.keyClick(picker.hex_field, Qt.Key.Key_Return)
        flush(self.app)
        self.assertTrue(closed(picker))
        self.assertEqual(field.text(), "#000000")

    def test_a_left_click_outside_the_card_closes_it_keeping_the_colour(self):
        for where in ("outside the window", "on the card's shadow"):
            with self.subTest(where=where):
                field, _button, picker = self.open_for("grey")
                click(picker, picker.preset_rect(0).center().toPoint())
                point = QPoint(-20, -20) if where == "outside the window" else QPoint(1, picker.height() - 1)
                if picker.card_rect().contains(QPointF(point)):
                    picker.close()
                    self.skipTest("the card fills its window where the window system can't draw a shadow")
                click(picker, point)
                flush(self.app)
                self.assertTrue(closed(picker))
                self.assertEqual(field.text(), "#000000")


class PickButtonTests(unittest.TestCase):
    """Config's Pick buttons open the picker; VR Config's code does the same."""

    @classmethod
    def setUpClass(cls):
        from tests.config_gui_loader import load_config_namespace

        cls.app = QApplication.instance() or QApplication([])
        cls.namespace = load_config_namespace()
        apply_theme_for_class(cls, cls.app)

    def test_a_pick_button_opens_the_picker_for_its_field_and_swatch(self):
        from tests.config_gui_loader import open_config_window

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        window = open_config_window(self.namespace["ConfigGUI"], folder.name)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        window.show()
        window.tabs.setCurrentIndex(1)
        flush(self.app)
        field = window.inputs["INITIAL_NODE_COLOR"]
        field.setText("#4488ff")
        button = next(child for child in field.parentWidget().findChildren(QPushButton)
                      if child.text() == "Pick")
        button.click()
        flush(self.app)
        picker = window.color_picker
        self.assertTrue(picker.isVisible())
        self.assertEqual(picker.hex_field.text(), "#4488ff")
        click(picker, picker.preset_rect(COLOR_PRESETS.index("#ffaa00")).center().toPoint())
        self.assertEqual(field.text(), "#ffaa00")
        self.assertIn("#ffaa00", window.color_swatches["INITIAL_NODE_COLOR"].styleSheet())
        QTest.keyClick(picker.hex_field, Qt.Key.Key_Escape)
        flush(self.app)
        self.assertEqual(field.text(), "#4488ff")
        self.assertIn("#4488ff", window.color_swatches["INITIAL_NODE_COLOR"].styleSheet())

    def test_both_config_windows_use_the_picker_not_qts_dialog(self):
        sources = [SRC / "EMAPSSN_Config.py", ROOT / "opt_vr" / "src" / "EMAPSSN_Config_VR.py"]
        for path in sources:
            if not path.is_file():
                continue
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("QColorDialog", text)
                self.assertIn("open_color_picker(button, line_edit)", text)


if __name__ == "__main__":
    unittest.main()
