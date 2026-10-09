# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tool help pages, and their translations beside them.

The Tools window shows src/tools/tool_descriptions/<name>.md for a tab. A
translation of it is <name>.<language>.md in the same folder, such as
Embedding_MSA.zh_CN.md, and its first line names the English page it was
made from by the page's SHA-256:

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

HELP_PAGES_DIR = Path(__file__).resolve().parents[1] / "tools" / "tool_descriptions"
_MARKER = re.compile(
    r"\A<!-- Translation of (?P<name>[^,\s]+), sha256 (?P<digest>[0-9a-f]{64}) -->[ \t]*\r?\n"
)
_TRANSLATION_NAME = re.compile(
    r"^(?P<name>[^.]+)\.(?P<language>[a-z]{2,3}(?:_[A-Z][a-z]{3})?(?:_[A-Z]{2})?)\.md$"
)


def english_digest(page):
    """The SHA-256 of an English help page, its line endings read as LF."""
    return hashlib.sha256(Path(page).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def translation_marker(page):
    """The first line a translation of the English page at page needs, without its line break."""
    return f"<!-- Translation of {Path(page).name}, sha256 {english_digest(page)} -->"


def translation_path(page, language):
    """Where the translation of the English page at page into language lives."""
    page = Path(page)
    return page.with_name(f"{page.stem}.{language}.md")


def is_translation(path):
    """Whether the file at path is a help page's translation, by its name."""
    return _TRANSLATION_NAME.match(Path(path).name) is not None


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


def help_page_text(page, language=None):
    """The Markdown to show for the English help page at page, in language.

    None or English gives the English page, and so does a translation that
    is missing, unreadable or behind the English page.
    """
    page = Path(page)
    if language:
        text = _current_text(page, translation_path(page, language))
        if text is not None:
            return text
    return page.read_text(encoding="utf-8")


def stale_translations(folder=HELP_PAGES_DIR):
    """(translation, its English page) for each translation in folder that Tools won't show.

    That is one behind its English page, one whose English page is gone, and
    one whose first line names no page.
    """
    stale = []
    for path in sorted(Path(folder).glob("*.md")):
        match = _TRANSLATION_NAME.match(path.name)
        if match is None:
            continue
        page = path.with_name(f"{match['name']}.md")
        if not page.is_file() or _current_text(page, path) is None:
            stale.append((path, page))
    return stale
