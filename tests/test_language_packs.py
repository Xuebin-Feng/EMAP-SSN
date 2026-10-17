# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The languages' folders and their language.json (src/utilities/Language_Packs.py).

Each language has a folder of its own in src/resources/languages, named by
its code, which holds its catalog and, optionally, a language.json with its
name, font and punctuation, its translated help pages and its translated
documentation pages (docs/<page>.html with docs/language.css). Names count only
as the folder lists them, on every system. These tests use made-up folders,
and read the Simplified Chinese folder that ships.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utilities import Language_Packs  # noqa: E402
from utilities.Language_Packs import (  # noqa: E402
    DOCS_FOLDER,
    WEB_LANGUAGE_FAMILY,
    LanguageFont,
    ManifestError,
    docs_folder,
    exact_file,
    exact_path,
    find_catalog,
    language_codes,
    language_folders,
    read_pack,
    read_packs,
    unicode_ranges,
    web_font_css,
)
from utilities.Localization import LANGUAGES_DIR, Punctuation  # noqa: E402

DESKTOP_FONT_DIR = SRC / "resources" / "fonts" / "desktop"


class FolderTestCase(unittest.TestCase):
    def setUp(self):
        self.languages = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.languages)

    def language(self, code, *files, manifest=None):
        folder = self.languages / code
        folder.mkdir()
        for name in files:
            (folder / name).write_bytes(b"")
        if manifest is not None:
            text = manifest if isinstance(manifest, str) else json.dumps(manifest, ensure_ascii=False)
            (folder / "language.json").write_bytes(text.encode("utf-8"))
        return folder


class DiscoveryTests(FolderTestCase):
    def test_a_language_is_a_folder_named_by_its_code_with_its_catalog(self):
        self.language("de", "emapssn_de.ts", "emapssn_de.qm")
        self.language("fr", "emapssn_fr.ts")  # Not compiled yet.
        self.language("pt_BR", "emapssn_pt_BR.ts", "emapssn_pt_BR.qm")
        self.language("sr_Latn_RS", "emapssn_sr_Latn_RS.qm")
        self.language("scripts", "emapssn_scripts.qm")  # Not a language code.
        self.language("es", "emapssn_de.qm")  # Another language's catalog.
        (self.languages / "emapssn_it.qm").write_bytes(b"")  # The old place.
        self.assertEqual(language_folders(self.languages), ["de", "es", "fr", "pt_BR", "sr_Latn_RS"])
        self.assertEqual(language_codes(self.languages), ["de", "pt_BR", "sr_Latn_RS"])
        self.assertEqual(language_codes(self.languages, suffix=".ts"), ["de", "fr", "pt_BR"])
        self.assertEqual(language_codes(self.languages, "emapssn_vr"), [])
        self.assertEqual(language_codes(self.languages / "missing"), [])

    def test_names_count_only_as_the_folder_lists_them(self):
        # pt_BR written pt_br, and de's catalog written with a capital: Windows
        # and macOS would open both, Linux neither, so no system may.
        self.language("pt_br", "emapssn_pt_BR.qm")
        self.language("de", "Emapssn_de.qm")
        self.assertEqual(language_codes(self.languages), [])
        for code in ("pt_BR", "de"):
            with self.subTest(code=code):
                self.assertIsNone(find_catalog(code, languages_dir=self.languages))
        self.assertIsNone(exact_file(self.languages / "de", "emapssn_de.qm"))
        self.assertEqual(exact_file(self.languages / "de", "Emapssn_de.qm"), self.languages / "de" / "Emapssn_de.qm")
        self.assertIsNone(exact_path(self.languages, "PT_BR"))
        self.assertIsNone(exact_path(self.languages, "de", "missing"))

    def test_find_catalog_finds_a_languages_own_catalog(self):
        self.language("de", "emapssn_de.qm", "emapssn_vr_de.qm", "emapssn_de.ts")
        self.assertEqual(find_catalog("de", languages_dir=self.languages), self.languages / "de" / "emapssn_de.qm")
        self.assertEqual(find_catalog("de", "emapssn_vr", languages_dir=self.languages),
                         self.languages / "de" / "emapssn_vr_de.qm")
        self.assertEqual(find_catalog("de", suffix=".ts", languages_dir=self.languages),
                         self.languages / "de" / "emapssn_de.ts")
        for code in ("fr", "../de", "de/..", "", None):
            with self.subTest(code=code):
                self.assertIsNone(find_catalog(code, languages_dir=self.languages))


class ManifestTests(FolderTestCase):
    def test_a_folder_without_language_json_takes_the_defaults(self):
        self.language("de", "emapssn_de.qm")
        pack = read_pack("de", self.languages)
        self.assertEqual((pack.code, pack.folder, pack.name, pack.font, pack.punctuation),
                         ("de", self.languages / "de", None, None, None))

    def test_language_json_gives_the_name_font_and_punctuation(self):
        self.language("ja", manifest={
            "name": "日本語",
            "font": {"family": "Noto Sans JP", "vispy_face": "NotoSansJP",
                     "files": ["noto/NotoSansJP/Regular.ttf", "noto/NotoSansJP/Bold.ttf"],
                     "web_range": "U+3000-30FF, U+4E00-9FFF"},
            "punctuation": {"no_space_after": "。、", "separators": {", ": "、"},
                            "separator_script": "U+3000-30FF, U+4E00-9FFF"},
        })
        pack = read_pack("ja", self.languages)
        self.assertEqual(pack.name, "日本語")
        self.assertEqual(pack.font, LanguageFont("Noto Sans JP", "NotoSansJP",
                                                 ("noto/NotoSansJP/Regular.ttf", "noto/NotoSansJP/Bold.ttf"),
                                                 "U+3000-30FF, U+4E00-9FFF"))
        self.assertEqual(pack.punctuation, Punctuation(frozenset("。、"), ((", ", "、"),),
                                                       ((0x3000, 0x30FF), (0x4E00, 0x9FFF))))
        self.assertTrue(pack.punctuation.in_script("あ"))
        self.assertFalse(pack.punctuation.in_script("abc"))

    def test_a_language_json_that_cant_be_used_names_its_file_and_reason(self):
        font = {"family": "F", "vispy_face": "F", "files": ["noto/F/F.ttf"]}
        cases = {
            "{": "Expecting",
            "[]": "must be an object",
            '{"fonts": {}}': "unknown keys ['fonts']",
            '{"name": ""}': '"name" must be a non-empty string',
            '{"name": 3}': '"name" must be a non-empty string',
            json.dumps({"font": {"family": "F", "files": ["a.ttf"]}}): 'needs "vispy_face"',
            json.dumps({"font": dict(font, files=[])}): "regular face",
            json.dumps({"font": dict(font, files=["../escape.ttf"])}): "inside the desktop font folder",
            json.dumps({"font": dict(font, files=["/abs.ttf"])}): "inside the desktop font folder",
            json.dumps({"font": dict(font, files=["C:/abs.ttf"])}): "inside the desktop font folder",
            json.dumps({"font": dict(font, files=["noto\\F.ttf"])}): "written with /",
            json.dumps({"font": dict(font, web_range="4E00-9FFF")}): "not a range",
            json.dumps({"font": dict(font, web_range="U+9FFF-4E00")}): "ends before it starts",
            json.dumps({"font": dict(font, size=3)}): "unknown keys ['size']",
            json.dumps({"punctuation": {"separators": {", ": "，"}}}): "need a \"separator_script\"",
            json.dumps({"punctuation": {"separators": {" ": ""}, "separator_script": "U+3000"}}): "separators",
            json.dumps({"punctuation": {"no_space_after": ["。"]}}): "string of characters",
        }
        for number, (content, reason) in enumerate(cases.items()):
            code = f"x{chr(ord('a') + number)}"  # A language code of its own for each case.
            self.language(code, manifest=content)
            with self.subTest(content=content):
                with self.assertRaises(ManifestError) as caught:
                    read_pack(code, self.languages)
                self.assertIn(str(self.languages / code / "language.json"), str(caught.exception))
                self.assertIn(reason, str(caught.exception))

    def test_one_broken_language_json_leaves_the_other_languages(self):
        self.language("de", manifest={"name": "Deutsch"})
        self.language("fr", manifest="{")
        self.language("it")
        packs, problems = read_packs(self.languages)
        self.assertEqual(sorted(packs), ["de", "it"])
        self.assertEqual(packs["de"].name, "Deutsch")
        self.assertEqual(len(problems), 1)
        self.assertIn(str(self.languages / "fr" / "language.json"), problems[0])

    def test_a_language_json_counts_only_named_exactly_so(self):
        folder = self.language("de")
        (folder / "Language.json").write_text('{"name": "Deutsch"}', encoding="utf-8")
        self.assertIsNone(read_pack("de", self.languages).name)

    def test_unicode_ranges_read_as_css_writes_them(self):
        self.assertEqual(unicode_ranges("U+3000-303F, U+FF00, u+4e00-9fff"),
                         ((0x3000, 0x303F), (0xFF00, 0xFF00), (0x4E00, 0x9FFF)))
        for text in ("", "U+", "U+4E??", "4E00", "U+1234567"):
            with self.subTest(text=text), self.assertRaises(ManifestError):
                unicode_ranges(text)


class ShippedLanguageTests(unittest.TestCase):
    """The Simplified Chinese folder that ships."""

    def test_simplified_chinese_keeps_what_the_program_had_built_in(self):
        pack = read_pack("zh_CN")
        self.assertEqual(pack.folder, LANGUAGES_DIR / "zh_CN")
        self.assertEqual(pack.name, "简体中文")
        self.assertEqual(pack.font, LanguageFont(
            "Noto Sans SC", "NotoSansSC",
            ("noto/NotoSansSC/NotoSansSC-Regular.ttf", "noto/NotoSansSC/NotoSansSC-Bold.ttf"),
            "U+3000-303F, U+3040-30FF, U+3100-312F, U+3200-33FF, U+4E00-9FFF, U+F900-FAFF, U+FF00-FFEF",
        ))
        # The punctuation joined messages had before each language had its folder.
        self.assertEqual(pack.punctuation, Punctuation(
            frozenset("。！？；：，、）」』】》"), ((", ", "，"), ("; ", "；")),
            ((0x2E80, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF)),
        ))

    def test_every_shipped_language_folder_is_whole(self):
        packs, problems = read_packs()
        self.assertEqual(problems, [])
        self.assertEqual(sorted(packs), language_codes(), "every folder holds a compiled catalog")
        manifest = (DESKTOP_FONT_DIR / "SHA256SUMS").read_text(encoding="ascii")
        for code, pack in packs.items():
            with self.subTest(language=code):
                self.assertIsNotNone(find_catalog(code, suffix=".ts"))
                for path in pack.font.files if pack.font else ():
                    self.assertIsNotNone(exact_path(DESKTOP_FONT_DIR, *path.split("/")), path)
                    self.assertIn(f"  {path}", manifest, "the font manifest lists it")
                if (LANGUAGES_DIR / code / "help").is_dir():
                    self.assertEqual(sorted(p.suffix for p in (LANGUAGES_DIR / code / "help").iterdir()),
                                     [".md"] * len(list((LANGUAGES_DIR / code / "help").iterdir())))
                # docs/ holds a translation of some of the English pages in the project's docs/, and the
                # language.css that Docs_Translation.py makes from language.json when it names a font.
                folder = docs_folder(code, LANGUAGES_DIR)
                if folder.is_dir():
                    english = {page.name for page in (SRC.parent / "docs").glob("*.html")}
                    self.assertEqual(sorted(set(p.name for p in folder.iterdir()) - english - {"language.css"}), [])
                    self.assertTrue(set(p.name for p in folder.glob("*.html")), "a docs/ folder holds a page")
                    has_font = pack.font is not None and bool(pack.font.web_range)
                    self.assertEqual((folder / "language.css").is_file(), has_font, "language.css is made from a font")
        self.assertEqual(Language_Packs.HELP_FOLDER, "help")
        self.assertEqual(Language_Packs.DOCS_FOLDER, "docs")
        self.assertEqual(DOCS_FOLDER, "docs")

    def test_the_docs_folder_of_a_language(self):
        self.assertEqual(docs_folder("zh_CN", LANGUAGES_DIR), LANGUAGES_DIR / "zh_CN" / "docs")
        self.assertEqual(docs_folder("de", Path("x")), Path("x") / "de" / "docs")

    def test_the_web_font_rules_name_the_language_family_with_both_weights(self):
        font = read_pack("zh_CN").font
        css = web_font_css(font, "../../../fonts/desktop/")
        rules = css.splitlines()
        self.assertEqual(len(rules), 2)
        self.assertEqual(WEB_LANGUAGE_FAMILY, "EMAP-SSN Language")  # the family the pages' font stacks name
        for rule, weight, path in zip(rules, (400, 700), font.files):
            self.assertIn("font-family: 'EMAP-SSN Language'", rule)
            self.assertIn(f"font-weight: {weight};", rule)
            self.assertIn(f"url('../../../fonts/desktop/{path}') format('truetype')", rule)
            self.assertIn(f"unicode-range: {font.web_range};", rule)
        # A language without a font, or with a font for no script of its own, has no rules.
        self.assertEqual(web_font_css(None, "x/"), "")
        self.assertEqual(web_font_css(LanguageFont("F", "F", ("a.ttf", "b.ttf")), "x/"), "")


if __name__ == "__main__":
    unittest.main()
