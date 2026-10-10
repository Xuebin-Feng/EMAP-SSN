# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tool help pages, and their translations in each language's folder.

The Tools window shows src/tools/tool_descriptions/<name>.md for a tab. A
translation of it lives in its language's folder (utilities/Language_Packs.py),
under the English page's own name, such as
src/resources/languages/zh_CN/help/Embedding_MSA.md, and its first line
names the English page it was made from by the page's SHA-256:

    <!-- Translation of Embedding_MSA.md, sha256 0123...cdef -->

Tools shows the translation in the window's language while that line
matches the English page, and the English page otherwise. So a page changed
since its translation shows in English until the translation is brought up
to date and its first line remade (translation_marker gives it), as a text
changed in the code shows in English until it is translated again. The
update command (src/resources/languages/Update_Translations.py) lists the
translations that are behind. Nothing here imports Qt.
"""

import hashlib
from pathlib import Path
import re

from utilities.Language_Packs import HELP_FOLDER, LANGUAGE_CODE, exact_file, exact_path, help_folder, language_folders
from utilities.Localization import LANGUAGES_DIR

HELP_PAGES_DIR = Path(__file__).resolve().parents[1] / "tools" / "tool_descriptions"
# The folder of the languages' folders, where translations live. The functions
# read both folders when called, so a test can point them elsewhere.
TRANSLATIONS_DIR = LANGUAGES_DIR
_MARKER = re.compile(
    r"\A<!-- Translation of (?P<name>[^,\s]+), sha256 (?P<digest>[0-9a-f]{64}) -->[ \t]*\r?\n"
)
# Where translations lived before each language had a folder: <name>.<language>.md
# beside the English page.
_OLD_TRANSLATION_NAME = re.compile(r"^(?P<name>[^.]+)\.(?P<language>[^.]+)\.md$")


def english_digest(page):
    """The SHA-256 of an English help page, its line endings read as LF."""
    return hashlib.sha256(Path(page).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def translation_marker(page):
    """The first line a translation of the English page at page needs, without its line break."""
    return f"<!-- Translation of {Path(page).name}, sha256 {english_digest(page)} -->"


def _languages(languages_dir):
    return Path(languages_dir) if languages_dir is not None else TRANSLATIONS_DIR


def _pages(pages_dir):
    return Path(pages_dir) if pages_dir is not None else HELP_PAGES_DIR


def translation_path(page, language, languages_dir=None):
    """Where the translation of the English page at page into language lives."""
    return help_folder(language, _languages(languages_dir)) / Path(page).name


def is_translation(path):
    """Whether the file at path is named as a translation used to be, <name>.<language>.md.

    Translations live in the languages' folders now; Tools skips such a file
    in the English pages' folder, and the update command reports it.
    """
    match = _OLD_TRANSLATION_NAME.match(Path(path).name)
    return match is not None and LANGUAGE_CODE.match(match["language"]) is not None


def _current_text(page, translated):
    """The translation's text after its first line, or None if that line doesn't match page."""
    try:
        text = translated.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = _MARKER.match(text)
    if match is None or match["name"] != page.name or match["digest"] != english_digest(page):
        return None
    return text[match.end():]


def help_page_text(page, language=None, languages_dir=None):
    """The Markdown to show for the English help page at page, in language.

    None or English gives the English page, and so does a translation that
    is missing, unreadable or behind the English page.
    """
    page = Path(page)
    if language:
        translated = exact_path(_languages(languages_dir), language, HELP_FOLDER, page.name)
        text = _current_text(page, translated) if translated is not None else None
        if text is not None:
            return text
    return page.read_text(encoding="utf-8")


def stale_translations(languages_dir=None, pages_dir=None):
    """(translation, its English page) for each translated page that Tools won't show.

    That is one behind its English page, one whose English page is gone, and
    one whose first line names no page.
    """
    languages_dir, pages_dir = _languages(languages_dir), _pages(pages_dir)
    stale = []
    for language in language_folders(languages_dir):
        folder = exact_path(languages_dir, language, HELP_FOLDER)
        for path in sorted(folder.glob("*.md")) if folder is not None else ():
            page = exact_file(pages_dir, path.name)
            if page is None or _current_text(page, path) is None:
                stale.append((path, Path(pages_dir) / path.name))
    return stale


def misplaced_translations(pages_dir=None, languages_dir=None):
    """(file, where it belongs) for each translation still beside the English pages."""
    pages_dir, languages_dir = _pages(pages_dir), _languages(languages_dir)
    misplaced = []
    for path in sorted(pages_dir.glob("*.md")):
        if is_translation(path):
            match = _OLD_TRANSLATION_NAME.match(path.name)
            misplaced.append((path, help_folder(match["language"], languages_dir) / f"{match['name']}.md"))
    return misplaced
