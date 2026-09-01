# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont, QFontDatabase, QPalette, QTextLayout  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
)

from utilities.Application_Fonts import (  # noqa: E402
    DESKTOP_FONT_DIR,
    FONT_FILES,
    FONT_MANIFEST_ENTRIES,
    MONOSPACE_BOLD_FILE,
    MONOSPACE_QSS_FONT_STACK,
    MONOSPACE_REGULAR_FILE,
    QT_MONOSPACE_FAMILY,
    QT_UI_FAMILY,
    UI_BOLD_FILE,
    UI_QSS_FONT_STACK,
    UI_REGULAR_FILE,
    VISPY_FALLBACK_FACE,
    VISPY_MONOSPACE_FACE,
    VISPY_UI_FACE,
    configure_qt_application_fonts,
    force_light_palette,
    qt_monospace_font,
    register_vispy_application_fonts,
)


class ApplicationFontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._owns_app = QApplication.instance() is None
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._owns_app:
            from shiboken6 import delete

            cls.app.closeAllWindows()
            cls.app.quit()
            delete(cls.app)
        cls.app = None
        gc.collect()

    def test_light_palette_uses_shared_fusion_colors(self):
        force_light_palette(self.app)

        self.assertEqual(self.app.style().objectName().lower(), "fusion")
        palette = self.app.palette()
        self.assertEqual(
            palette.color(QPalette.ColorRole.Window).getRgb()[:3],
            (240, 240, 240),
        )
        self.assertEqual(
            palette.color(QPalette.ColorRole.Highlight).getRgb()[:3],
            (48, 140, 198),
        )
        self.assertEqual(
            palette.color(QPalette.ColorRole.HighlightedText).getRgb()[:3],
            (255, 255, 255),
        )

    def test_manifest_declares_the_4_68_mib_core_pack_and_hashes_match(self):
        self.assertEqual(len(FONT_FILES), 8)
        self.assertEqual(len(FONT_MANIFEST_ENTRIES), 8)
        bundled_font_files = {
            path.relative_to(DESKTOP_FONT_DIR).as_posix()
            for path in DESKTOP_FONT_DIR.rglob("*")
            if path.suffix.lower() in {".otf", ".ttc", ".ttf"}
        }
        self.assertEqual(bundled_font_files, set(FONT_FILES))
        self.assertEqual(
            sum((DESKTOP_FONT_DIR / relative).stat().st_size for relative in FONT_FILES),
            4_908_576,
        )

        for relative_path, expected_hash in FONT_MANIFEST_ENTRIES:
            font_path = DESKTOP_FONT_DIR / relative_path
            self.assertTrue(font_path.is_file(), relative_path)
            digest = hashlib.sha256(font_path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected_hash, relative_path)

    def test_assets_register_with_primary_families_weights_and_core_scripts(self):
        status = configure_qt_application_fonts(self.app)

        self.assertEqual(set(status.loaded_files), set(FONT_FILES))
        self.assertEqual(status.failed_files, ())
        self.assertTrue(status.ui_family_available)
        self.assertTrue(status.monospace_family_available)
        self.assertIn(QT_UI_FAMILY, status.loaded_families)
        self.assertIn(QT_MONOSPACE_FAMILY, status.loaded_families)

        expected_weights = {
            "Regular": QFont.Weight.Normal,
            "Medium": QFont.Weight.Medium,
            "SemiBold": QFont.Weight.DemiBold,
            "Bold": QFont.Weight.Bold,
        }
        for family in (QT_UI_FAMILY, QT_MONOSPACE_FAMILY):
            styles = set(QFontDatabase.styles(family))
            self.assertTrue(set(expected_weights).issubset(styles))
            for style, weight in expected_weights.items():
                font = QFontDatabase.font(family, style, 12)
                self.assertEqual(font.weight(), weight, f"{family} {style}")

        samples = {
            "Latin": "A",
            "Latin extended": "Ł",
            "Greek": "Ω",
            "Greek extended": "Ἀ",
            "Cyrillic": "Ж",
            "Cyrillic extended": "Ꙁ",
        }
        for label, text in samples.items():
            layout = QTextLayout(text, self.app.font())
            layout.beginLayout()
            layout.createLine()
            layout.endLayout()
            families = [run.rawFont().familyName() for run in layout.glyphRuns()]
            self.assertTrue(families, label)
            self.assertTrue(
                all(family.startswith("Noto") for family in families),
                f"{label}: {families}",
            )

    def test_qt_configuration_is_idempotent_and_preserves_metrics(self):
        original = QFont(self.app.font())
        test_font = QFont(original)
        test_font.setPointSizeF(13.5)
        test_font.setWeight(QFont.Weight.Medium)
        test_font.setItalic(True)
        self.app.setFont(test_font)

        first = configure_qt_application_fonts(self.app)
        second = configure_qt_application_fonts(self.app)
        configured = self.app.font()

        self.assertEqual(first, second)
        self.assertEqual(configured.family(), QT_UI_FAMILY)
        self.assertAlmostEqual(configured.pointSizeF(), 13.5)
        self.assertEqual(configured.weight(), QFont.Weight.Medium)
        self.assertTrue(configured.italic())
        self.app.setFont(original)
        configure_qt_application_fonts(self.app)

    def test_native_widgets_inherit_noto_and_monospace_helper_prefers_noto_mono(self):
        configure_qt_application_fonts(self.app)
        widgets = (
            QLabel("Label"),
            QPushButton("Button"),
            QLineEdit("Input"),
            QComboBox(),
            QTabWidget(),
            QDialog(),
            QSpinBox(),
        )
        for widget in widgets:
            self.assertEqual(widget.font().family(), QT_UI_FAMILY)
            widget.deleteLater()

        mono_font = qt_monospace_font(self.app.font())
        self.assertEqual(mono_font.family(), QT_MONOSPACE_FAMILY)
        self.assertEqual(mono_font.pointSizeF(), self.app.font().pointSizeF())
        self.assertIn("'Noto Sans'", UI_QSS_FONT_STACK)
        self.assertIn("'Noto Sans Mono'", MONOSPACE_QSS_FONT_STACK)

    def test_missing_and_incomplete_assets_warn_and_keep_safe_fallbacks(self):
        before = QFont(self.app.font())
        missing_dir = Path(tempfile.gettempdir()) / "ssn-fonts-do-not-exist"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            missing = configure_qt_application_fonts(self.app, missing_dir)

        self.assertEqual(missing.loaded_files, ())
        self.assertEqual(set(missing.failed_files), set(FONT_FILES))
        self.assertFalse(missing.ui_family_available)
        self.assertFalse(missing.monospace_family_available)
        self.assertEqual(self.app.font().families(), before.families())
        self.assertTrue(caught)

        declared = (
            UI_REGULAR_FILE,
            UI_BOLD_FILE,
            MONOSPACE_REGULAR_FILE,
            MONOSPACE_BOLD_FILE,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            font_dir = Path(temporary_dir)
            for relative_path in declared[:2]:
                target = font_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(DESKTOP_FONT_DIR / relative_path, target)
            (font_dir / "SHA256SUMS").write_text(
                "".join(f"{'0' * 64}  {path}\n" for path in declared),
                encoding="ascii",
            )
            with warnings.catch_warnings(record=True) as incomplete_warnings:
                warnings.simplefilter("always")
                incomplete = configure_qt_application_fonts(self.app, font_dir)
                vispy = register_vispy_application_fonts(incomplete, font_dir)

        self.assertTrue(incomplete.ui_family_available)
        self.assertFalse(incomplete.monospace_family_available)
        self.assertEqual(
            set(incomplete.failed_files),
            {MONOSPACE_REGULAR_FILE, MONOSPACE_BOLD_FILE},
        )
        self.assertEqual(vispy.ui_face, VISPY_UI_FACE)
        self.assertEqual(vispy.monospace_face, VISPY_FALLBACK_FACE)
        self.assertTrue(incomplete_warnings)

    def test_vispy_loads_regular_and_bold_glyphs_from_core_noto_faces(self):
        from vispy.util.fonts import _load_glyph, list_fonts

        qt_status = configure_qt_application_fonts(self.app)
        first = register_vispy_application_fonts(qt_status)
        second = register_vispy_application_fonts(qt_status)

        self.assertEqual(first, second)
        self.assertEqual(first.ui_face, VISPY_UI_FACE)
        self.assertEqual(first.monospace_face, VISPY_MONOSPACE_FACE)
        self.assertIn(VISPY_UI_FACE, list_fonts())
        self.assertIn(VISPY_MONOSPACE_FACE, list_fonts())

        for face in (VISPY_UI_FACE, VISPY_MONOSPACE_FACE):
            for bold in (False, True):
                glyphs = {}
                font = {"face": face, "size": 12, "bold": bold, "italic": False}
                for char in "Ag09":
                    _load_glyph(font, char, glyphs)
                self.assertEqual(set(glyphs), set("Ag09"))

    def test_embedded_web_surfaces_use_only_local_noto_assets(self):
        tools_source = (SRC_DIR / "EMAPSSN_Tools.py").read_text(encoding="utf-8")
        self.assertIn('href="fonts/fonts.css"', tools_source)
        self.assertIn("__UI_FONT_STACK__", tools_source)
        self.assertIn("__MONOSPACE_FONT_STACK__", tools_source)

        src_font_dir = SRC_DIR / "resources" / "fonts"
        docs_font_dir = PROJECT_ROOT / "docs" / "fonts"
        src_css = (src_font_dir / "fonts.css").read_text(encoding="utf-8")
        docs_css = (docs_font_dir / "fonts.css").read_text(encoding="utf-8")
        self.assertEqual(src_css, docs_css)
        self.assertIn("font-family: 'Noto Sans'", src_css)
        self.assertIn("font-family: 'Noto Sans Mono'", src_css)
        self.assertNotIn("http://", src_css)
        self.assertNotIn("https://", src_css)
        self.assertNotIn("Inter", src_css)
        self.assertNotIn("Fira Code", src_css)

        referenced = {
            line.split("url(", 1)[1].split(")", 1)[0]
            for line in src_css.splitlines()
            if "src: url(" in line
        }
        self.assertEqual(len(referenced), 15)
        for filename in referenced:
            self.assertTrue((src_font_dir / filename).is_file(), filename)
            self.assertTrue((docs_font_dir / filename).is_file(), filename)

        agent_html = (SRC_DIR / "web_ui" / "agent.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("--font-family: 'Noto Sans'", agent_html)
        self.assertIn("--font-mono: 'Noto Sans Mono'", agent_html)
        meta_html = (SRC_DIR / "web_ui" / "meta.html").read_text(encoding="utf-8")
        self.assertIn('href="/fonts/fonts.css"', meta_html)
        self.assertIn("--font-family: 'Noto Sans'", meta_html)
        docs_html = (PROJECT_ROOT / "docs" / "list_of_commands.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("font-family: 'Noto Sans'", docs_html)
        self.assertIn("font-family: 'Noto Sans Mono'", docs_html)
        self.assertNotIn("font-family: 'Inter'", docs_html)
        self.assertNotIn("font-family: 'Fira Code'", docs_html)

        embedded_sources = tools_source + agent_html + meta_html + docs_html
        self.assertNotIn("fonts.googleapis.com", embedded_sources)
        self.assertNotIn("fonts.gstatic.com", embedded_sources)


if __name__ == "__main__":
    unittest.main()
