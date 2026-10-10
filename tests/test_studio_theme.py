# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The shared window theme (desktop.Studio_Theme) and its bundled Lucide icons."""
import importlib.util
import os
from pathlib import Path
import re
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QColor, QIcon, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget  # noqa: E402

from desktop.Studio_Theme import (  # noqa: E402
    HINT_ICON_OFFSET,
    HINT_ICON_SIZE,
    HINT_TEXT_INDENT,
    LUCIDE_DIR,
    TOKENS,
    CardShadows,
    IconLabel,
    add_hint_icon,
    coloured_svg,
    icon_svg,
    set_role,
    studio_icon,
    studio_palette,
    studio_stylesheet,
)
from tests.theme_fixture import apply_theme_for_class  # noqa: E402


def flush(app):
    for _ in range(4):
        app.processEvents()


def load_update_icons():
    path = SRC / "resources" / "icons" / "Update_Icons.py"
    spec = importlib.util.spec_from_file_location("update_icons_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StudioThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def test_the_application_gets_fusion_the_palette_and_the_stylesheet(self):
        self.assertEqual(self.app.styleSheet(), studio_stylesheet())
        # Under a stylesheet, style() is Qt's unnamed wrapper around the base style.
        self.app.setStyleSheet("")
        try:
            self.assertEqual(self.app.style().name().lower(), "fusion")
        finally:
            self.app.setStyleSheet(studio_stylesheet())
        palette = self.app.palette()
        self.assertEqual(palette.color(QPalette.ColorRole.Window).name(), TOKENS["window"])
        self.assertEqual(palette.color(QPalette.ColorRole.Highlight).name(), TOKENS["accent"])
        self.assertEqual(
            palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name(),
            TOKENS["text_disabled"],
        )
        self.assertEqual(studio_palette().color(QPalette.ColorRole.Base).name(), TOKENS["field"])

    def test_every_image_the_stylesheet_names_exists(self):
        urls = re.findall(r'url\("([^"]+)"\)', studio_stylesheet())
        self.assertTrue(urls)
        for url in urls:
            with self.subTest(url=url):
                self.assertTrue(Path(url).is_file())

    def test_a_role_restyles_a_shown_button(self):
        host = QWidget()
        self.addCleanup(host.deleteLater)
        button = QPushButton("Save && Run", host)
        button.resize(120, 30)
        host.resize(140, 40)
        host.show()
        flush(self.app)
        set_role(button, "primary")
        flush(self.app)
        self.assertEqual(button.property("role"), "primary")
        image = button.grab().toImage()
        # Away from the text and the rounded corners, the fill is the primary colour.
        self.assertEqual(image.pixelColor(10, image.height() // 2).name(), TOKENS["primary"])

    def test_the_hint_card_keeps_its_info_icon_clear_of_the_text(self):
        label = QLabel("Click or tab to an input to see helpful tips here.")
        self.addCleanup(label.deleteLater)
        icon = add_hint_icon(label)
        self.assertEqual(label.property("role"), "hint")
        self.assertIsInstance(icon, IconLabel)
        self.assertEqual(icon.parent(), label)
        self.assertEqual((icon.x(), icon.y()), (HINT_ICON_OFFSET, HINT_ICON_OFFSET))
        self.assertGreater(HINT_TEXT_INDENT, HINT_ICON_OFFSET + HINT_ICON_SIZE)

    def test_card_shadows_darken_the_edge_below_a_card(self):
        host = QWidget()
        self.addCleanup(host.deleteLater)
        host.setAutoFillBackground(True)
        palette = host.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
        host.setPalette(palette)
        card = QWidget(host)
        card.setGeometry(20, 20, 100, 60)
        host.resize(140, 110)
        plain = host.grab().toImage().pixelColor(70, 81)
        CardShadows(host, [card])
        host.show()
        flush(self.app)
        shaded = host.grab().toImage().pixelColor(70, 81)
        self.assertEqual(plain.name(), "#ffffff")
        self.assertLess(shaded.lightness(), plain.lightness())


    def test_closing_a_shaded_window_raises_nothing(self):
        from PySide6.QtTest import QTest

        host = QWidget()
        card = QWidget(host)
        card.setGeometry(10, 10, 50, 30)
        CardShadows(host, [card])
        host.show()
        flush(self.app)
        # The cards get Hide and Move events while their host is destroyed.
        host.deleteLater()
        QTest.qWait(50)
        card = host = None


class StudioIconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_the_bundled_icons_are_up_to_date(self):
        self.assertEqual(load_update_icons().check(), [])

    def test_every_icon_the_code_names_is_bundled(self):
        bundled = {path.stem for path in LUCIDE_DIR.glob("*.svg")}
        pattern = re.compile(
            r'(?:studio_icon|IconLabel|icon_button)\(\s*"([a-z0-9-]+)"|\bicon="([a-z0-9-]+)"'
        )
        named = set()
        for path in SRC.rglob("*.py"):
            for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
                named.add(match.group(1) or match.group(2))
        self.assertTrue(named)
        self.assertEqual(sorted(named - bundled), [])

    def test_an_icon_draws_in_the_colour_asked_for_at_the_screens_pixel_ratio(self):
        icon = studio_icon("x", "#ff0000", "#00ff00")
        normal = icon.pixmap(QSize(24, 24), 2.0)
        self.assertEqual(normal.devicePixelRatio(), 2.0)
        self.assertEqual((normal.width(), normal.height()), (48, 48))

        def colours(pixmap):
            image = pixmap.toImage()
            return {image.pixelColor(x, y).name()
                    for x in range(image.width()) for y in range(image.height())
                    if image.pixelColor(x, y).alpha() == 255}

        self.assertIn("#ff0000", colours(normal))
        disabled = icon.pixmap(QSize(24, 24), 2.0, QIcon.Mode.Disabled)
        self.assertIn("#00ff00", colours(disabled))
        self.assertNotIn("#ff0000", colours(disabled))

    def test_current_colour_is_written_into_the_svg(self):
        text = icon_svg("check")
        self.assertIn('stroke="currentColor"', text)
        self.assertNotIn("currentColor", coloured_svg(text, "#123456"))
        self.assertIn('stroke="#123456"', coloured_svg(text, "#123456"))

    def test_a_missing_icon_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            icon_svg("no-such-icon")


if __name__ == "__main__":
    unittest.main()
