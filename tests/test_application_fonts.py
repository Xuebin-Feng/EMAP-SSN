# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Application fonts (desktop.Desktop_App): the bundled Noto font pack and its
Simplified Chinese font, Qt and VisPy font registration, the light palette,
DPI-independent VisPy text sizes, and the local font assets used by the
embedded web pages."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import re
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

from PySide6.QtGui import QFont, QFontDatabase, QPalette, QRawFont, QTextLayout  # noqa: E402
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

from desktop import Desktop_App  # noqa: E402
from desktop.Desktop_App import (  # noqa: E402
    DESKTOP_FONT_DIR,
    FONT_FILES,
    FONT_MANIFEST_ENTRIES,
    LANGUAGE_FONT_FILES,
    LANGUAGE_FONTS,
    MONOSPACE_BOLD_FILE,
    MONOSPACE_QSS_FONT_STACK,
    MONOSPACE_REGULAR_FILE,
    PSEUDO_LANGUAGE,
    QT_MONOSPACE_FAMILY,
    QT_SIMPLIFIED_CHINESE_FAMILY,
    QT_UI_FAMILY,
    UI_BOLD_FILE,
    UI_QSS_FONT_STACK,
    UI_REGULAR_FILE,
    VISPY_FALLBACK_FACE,
    VISPY_MONOSPACE_FACE,
    VISPY_SIMPLIFIED_CHINESE_FACE,
    VISPY_UI_FACE,
    configure_qt_application_fonts,
    force_light_palette,
    matplotlib_language_families,
    qt_monospace_font,
    register_vispy_application_fonts,
    language_web_font_css,
    vispy_language_face,
    vispy_points_at_reference_dpi,
    vispy_points_for_logical_pixels,
)
from utilities.Help_Pages import HELP_PAGES_DIR  # noqa: E402
from utilities.Localization import LANGUAGES_DIR, read_catalog  # noqa: E402


class VispyTextScalingTests(unittest.TestCase):
    def test_logical_pixel_size_is_independent_of_canvas_dpi(self):
        logical_pixels = 16.0

        for dpi in (72.0, 96.0, 144.0, 192.0, 220.0):
            with self.subTest(dpi=dpi):
                points = vispy_points_for_logical_pixels(
                    logical_pixels, dpi
                )
                rendered_pixels = points / 72.0 * dpi
                self.assertAlmostEqual(rendered_pixels, logical_pixels)

    def test_reference_point_size_preserves_96_dpi_appearance(self):
        for dpi in (96.0, 144.0, 192.0):
            with self.subTest(dpi=dpi):
                points = vispy_points_at_reference_dpi(8.0, dpi)
                rendered_pixels = points / 72.0 * dpi
                self.assertAlmostEqual(rendered_pixels, 8.0 / 72.0 * 96.0)

    def test_invalid_canvas_dpi_uses_reference_dpi(self):
        points = vispy_points_for_logical_pixels(16.0, 0.0)

        self.assertEqual(points, 12.0)


class ApplicationFontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._owns_app = QApplication.instance() is None
        cls.app = QApplication.instance() or QApplication([])
        # These tests register the bundled Noto fonts and switch the application
        # to them and to the Fusion light palette. A borrowed application gets
        # its own look back: while Noto Sans stays registered, stylesheets that
        # name it first make widgets in later test modules measure wider.
        cls._saved_style = cls.app.style().name()
        cls._saved_palette = QPalette(cls.app.palette())
        cls._saved_font = QFont(cls.app.font())
        cls._saved_font_ids = dict(Desktop_App._qt_font_ids)
        cls._saved_app_ref = Desktop_App._qt_app_ref

    @classmethod
    def tearDownClass(cls):
        if cls._owns_app:
            from shiboken6 import delete

            cls.app.closeAllWindows()
            cls.app.quit()
            delete(cls.app)
        else:
            for path, font_id in Desktop_App._qt_font_ids.items():
                if font_id >= 0 and cls._saved_font_ids.get(path) != font_id:
                    QFontDatabase.removeApplicationFont(font_id)
            Desktop_App._qt_font_ids.clear()
            Desktop_App._qt_font_ids.update(cls._saved_font_ids)
            Desktop_App._qt_app_ref = cls._saved_app_ref
            cls.app.setStyle(cls._saved_style)
            cls.app.setPalette(cls._saved_palette)
            cls.app.setFont(cls._saved_font)
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

    def test_manifest_declares_the_core_pack_and_the_chinese_font_and_hashes_match(self):
        self.assertEqual(len(FONT_FILES), 8)
        self.assertEqual(len(FONT_MANIFEST_ENTRIES), 10)
        bundled_font_files = {
            path.relative_to(DESKTOP_FONT_DIR).as_posix()
            for path in DESKTOP_FONT_DIR.rglob("*")
            if path.suffix.lower() in {".otf", ".ttc", ".ttf"}
        }
        self.assertEqual(
            bundled_font_files,
            {relative for relative, _ in FONT_MANIFEST_ENTRIES},
        )
        # Startup registers the 4.68 MiB core; the Chinese font waits for its language.
        self.assertEqual(bundled_font_files - set(FONT_FILES), LANGUAGE_FONT_FILES)
        self.assertEqual(set(LANGUAGE_FONT_FILES), set(LANGUAGE_FONTS["zh_CN"].files))
        self.assertEqual(
            sum((DESKTOP_FONT_DIR / relative).stat().st_size for relative in FONT_FILES),
            4_908_576,
        )
        self.assertEqual(
            sum((DESKTOP_FONT_DIR / relative).stat().st_size for relative in LANGUAGE_FONT_FILES),
            4_583_760,
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

    def test_the_core_faces_stay_registered_after_leaving_a_language_with_its_own_font(self):
        # Removing the Chinese font leaves Qt listing no family for any
        # application font until its font database is read again.
        configure_qt_application_fonts(self.app)
        self.addCleanup(Desktop_App.install_translations, self.app, None)
        Desktop_App.install_translations(self.app, "zh_CN")
        Desktop_App.install_translations(self.app, None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            status = configure_qt_application_fonts(self.app)

        self.assertEqual(set(status.loaded_files), set(FONT_FILES))
        self.assertEqual(status.failed_files, ())
        self.assertTrue(status.ui_family_available)
        self.assertEqual([str(warning.message) for warning in caught], [])

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


def gb2312_characters():
    """Every character GB2312 encodes: its 6,763 hanzi, symbols, kana, Greek and Cyrillic."""
    characters = set()
    for row in range(0xA1, 0xF8):
        for cell in range(0xA1, 0xFF):
            try:
                characters.add(bytes((row, cell)).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return characters


def characters_added(pairs):
    """The characters each (English, translation) pair's translation adds: what the font must draw.

    A translation keeps the English's emoji and symbols (🧬, ⚙), which the
    system's fonts draw in every language, as they do in English.
    """
    return {
        char
        for source, translation in pairs
        for char in translation
        if not char.isspace() and char not in source
    }


class SimplifiedChineseFontTests(unittest.TestCase):
    """The bundled Noto Sans SC, cut down to GB2312: what it covers, and the Viewer's face."""

    @classmethod
    def setUpClass(cls):
        QApplication.instance() or QApplication([])
        cls.font = LANGUAGE_FONTS["zh_CN"]

    def test_regular_and_bold_cover_gb2312_latin_1_and_the_catalog(self):
        gb2312 = gb2312_characters()
        self.assertEqual(len(gb2312), 7445)
        latin = {chr(code) for code in (*range(0x20, 0x7F), *range(0xA0, 0x100))}
        pairs = [
            (message.source, translation)
            for path in LANGUAGES_DIR.glob("emapssn_zh_CN.ts")
            for message in read_catalog(path)
            for translation in message.translations
        ]
        pairs += [
            (page.with_name(page.name.replace(".zh_CN.md", ".md")).read_text(encoding="utf-8"),
             page.read_text(encoding="utf-8"))
            for page in HELP_PAGES_DIR.glob("*.zh_CN.md")
        ]
        catalog = characters_added(pairs)
        self.assertEqual(characters_added([("⚙ Save", "⚙ 保存")]), {"保", "存"}, "the English's own symbols aside")

        for face, style, weight in zip(self.font.files, ("Regular", "Bold"), (400, 700)):
            with self.subTest(face=face):
                font = QRawFont(str(DESKTOP_FONT_DIR / face), 12)
                self.assertEqual(
                    (font.familyName(), font.styleName(), font.weight()),
                    (QT_SIMPLIFIED_CHINESE_FAMILY, style, weight),
                )

                def missing(characters):
                    # PySide6 misreads a non-ASCII str here, so pass the code point.
                    return sorted(c for c in characters if not font.supportsCharacter(ord(c)))

                self.assertEqual(missing({"龘"}), ["龘"], "the check can fail: 龘 is not in GB2312")
                self.assertEqual(missing(gb2312 | latin), [])
                self.assertEqual(missing(catalog), [], "remake the font with these characters")

    def test_the_browser_views_draw_chinese_in_the_bundled_font(self):
        css = language_web_font_css("zh_CN", "/fonts/desktop/")
        faces = re.findall(r"@font-face \{([^}]*)\}", css)
        self.assertEqual(
            [(re.search(r"font-family: '([^']+)'", face).group(1), re.search(r"font-weight: (\d+)", face).group(1),
              re.search(r"url\('([^']+)'\)", face).group(1)) for face in faces],
            [(QT_SIMPLIFIED_CHINESE_FAMILY, weight, "/fonts/desktop/" + path)
             for weight, path in zip(("400", "700"), self.font.files)],
        )
        self.assertTrue(all(f"unicode-range: {self.font.web_range};" in face for face in faces))
        self.assertIn("U+4E00-9FFF", self.font.web_range)
        self.assertNotIn("U+0000", self.font.web_range, "Latin keeps the core faces")
        for language in (None, "en", "de", PSEUDO_LANGUAGE):
            self.assertEqual(language_web_font_css(language, "/fonts/desktop/"), "", language)

        # Each stack names the family right after the core one, as the Qt stacks
        # (and so the Tools help panel's) do; without its rules it is skipped.
        stacks = [UI_QSS_FONT_STACK, MONOSPACE_QSS_FONT_STACK]
        for page in ("agent.html", "meta.html"):
            html = (SRC_DIR / "web_ui" / page).read_text(encoding="utf-8")
            found = re.findall(r"(?:font-family|--font-family|--font-mono):\s*('Noto Sans[^']*'[^;\"]*)", html)
            self.assertGreaterEqual(len(found), 1, page)
            stacks += found
        for stack in stacks:
            self.assertEqual([name.strip() for name in stack.split(",")][1], f"'{QT_SIMPLIFIED_CHINESE_FAMILY}'", stack)

    def test_the_viewer_draws_simplified_chinese_in_its_own_face(self):
        from vispy.util.fonts import _load_glyph

        self.assertEqual(vispy_language_face("zh_CN"), VISPY_SIMPLIFIED_CHINESE_FACE)
        for language in (None, "en", "de", PSEUDO_LANGUAGE):
            self.assertIsNone(vispy_language_face(language), language)

        def bitmaps(face, bold):
            glyphs = {}
            font = {"face": face, "size": 12, "bold": bold, "italic": False}
            # 龘 is in no bundled face, so it draws the empty-glyph box.
            for char in "A中龘":
                _load_glyph(font, char, glyphs)
            return {char: glyph["bitmap"].tobytes() for char, glyph in glyphs.items()}

        latin = bitmaps(VISPY_FALLBACK_FACE, False)
        self.assertEqual(latin["中"], latin["龘"], "the check can fail: a Latin face lacks 中")
        for bold in (False, True):
            with self.subTest(bold=bold):
                chinese = bitmaps(VISPY_SIMPLIFIED_CHINESE_FACE, bold)
                self.assertNotEqual(chinese["中"], chinese["龘"])
                self.assertTrue(chinese["A"])

    def test_a_missing_face_file_leaves_the_viewer_its_core_faces_with_a_warning(self):
        regular = self.font.files[0]
        with tempfile.TemporaryDirectory() as temporary_dir:
            target = Path(temporary_dir) / regular
            target.parent.mkdir(parents=True)
            shutil.copy2(DESKTOP_FONT_DIR / regular, target)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                face = vispy_language_face("zh_CN", temporary_dir)

        self.assertIsNone(face)
        self.assertTrue(any("incomplete" in str(warning.message) for warning in caught))

    def test_figures_draw_simplified_chinese_with_the_bundled_font_after_their_own(self):
        import io

        from matplotlib import font_manager
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        for language in (None, "en", "de", PSEUDO_LANGUAGE):
            self.assertIsNone(matplotlib_language_families(language), language)
        families = matplotlib_language_families("zh_CN")
        self.assertEqual(families, ["sans-serif", QT_SIMPLIFIED_CHINESE_FAMILY])
        matplotlib_language_families("zh_CN")
        registered = [Path(entry.fname).resolve() for entry in font_manager.fontManager.ttflist]
        for face in self.font.files:
            self.assertEqual(registered.count((DESKTOP_FONT_DIR / face).resolve()), 1, "registered once")

        def missing_glyphs(**font):
            figure = Figure()
            figure.add_subplot(111).set_title("分数分布 Score", **font)
            FigureCanvasAgg(figure)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                figure.savefig(io.BytesIO(), format="png")
            return [str(warning.message) for warning in caught if "missing from font" in str(warning.message)]

        self.assertTrue(missing_glyphs(), "the check can fail: matplotlib's own font lacks 分")
        self.assertEqual(missing_glyphs(family=families), [])

    def test_a_missing_face_file_leaves_figures_their_own_fonts_with_a_warning(self):
        regular = self.font.files[0]
        with tempfile.TemporaryDirectory() as temporary_dir:
            target = Path(temporary_dir) / regular
            target.parent.mkdir(parents=True)
            shutil.copy2(DESKTOP_FONT_DIR / regular, target)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                families = matplotlib_language_families("zh_CN", temporary_dir)

        self.assertIsNone(families)
        self.assertTrue(any("figures keep matplotlib's fonts" in str(warning.message) for warning in caught))


if __name__ == "__main__":
    unittest.main()
