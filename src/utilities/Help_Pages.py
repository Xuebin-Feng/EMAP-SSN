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
translations that are behind. command_list_path finds the Viewer's command
reference, docs/list_of_commands.html, or its translated copy in its
language's folder, src/resources/languages/<language>/docs/list_of_commands.html.
Nothing here imports Qt.
"""

import hashlib
from pathlib import Path
import re

from utilities.Language_Packs import (
    DOCS_FOLDER,
    HELP_FOLDER,
    LANGUAGE_CODE,
    exact_file,
    exact_path,
    help_folder,
    language_folders,
)
from utilities.Localization import LANGUAGES_DIR

HELP_PAGES_DIR = Path(__file__).resolve().parents[1] / "tools" / "tool_descriptions"
# The Viewer's command reference: an HTML page in docs/, which the Viewer's info
# card links to. A translated copy of it is in the language's folder, <language>/docs/
# list_of_commands.html under TRANSLATIONS_DIR.
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
COMMAND_LIST_NAME = "list_of_commands.html"
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


def command_list_path(language=None, docs_dir=None, languages_dir=None):
    """The Viewer's command reference to open for language: a Path, or None if there is none.

    A translated copy, <language>/docs/list_of_commands.html in the language's
    folder, is used where one exists; otherwise, and for English, it is the
    English page docs/list_of_commands.html. None means neither is there, as in
    an install without docs/. Names count only as the folders list them, so
    zh_cn/ is no zh_CN, on Windows and macOS too. The folders are read when
    called, so a test can point them elsewhere.
    """
    docs = Path(docs_dir) if docs_dir is not None else DOCS_DIR
    if language:
        folder = exact_path(_languages(languages_dir), language, DOCS_FOLDER)
        translated = exact_file(folder, COMMAND_LIST_NAME) if folder is not None else None
        if translated is not None:
            return translated
    return exact_file(docs, COMMAND_LIST_NAME)


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


# A page's Parameter tables list one setting per row: its name and key, such
# as "Pre-filtering Strength **`PREFILTER_STRENGTH`**", then what it does.
# Every such table, on every page and in every language, gives the first
# column the same width, room for any name and key in at most three lines
# (tests/test_tool_help_pages.py measures them). A pane narrower than
# PARAMETER_STACK_BELOW has no room for both columns, so there each row
# stacks, its name over what it does, and the page's sides narrow to
# NARROW_PAGE_PADDING, room for the longest key on one line.
PARAMETER_COLUMN_WIDTH = 260
PARAMETER_STACK_BELOW = 580
NARROW_PAGE_PADDING = 12
_TABLE_BOX = re.compile(r'<div class="table-scroll">(<table>.*?</table>)</div>', re.S)
_FIRST_CELL = re.compile(r"<tr>\s*<td[^>]*>(.*?)</td>", re.S)


def is_parameter_table(table_html):
    """Whether a table names a setting in a row: a first-column cell holds a bold key, **`KEY`**."""
    return any("<strong><code>" in cell for cell in _FIRST_CELL.findall(table_html))


def mark_parameter_tables(page_html):
    """page_html with the box of each Parameter table (div.table-scroll) also of class "parameters"."""
    def mark(match):
        if not is_parameter_table(match.group(1)):
            return match.group(0)
        return f'<div class="table-scroll parameters">{match.group(1)}</div>'

    return _TABLE_BOX.sub(mark, page_html)


def parameter_table_css(tokens):
    """The help pane's rules for Parameter tables, in tokens' colours (a theme's tokens)."""
    return f"""
        .parameters table {{ table-layout: fixed; }}
        .parameters th:first-child {{ width: {PARAMETER_COLUMN_WIDTH}px; }}
        @media (max-width: {PARAMETER_STACK_BELOW - 1}px) {{
            .help-page {{ padding-left: {NARROW_PAGE_PADDING}px; padding-right: {NARROW_PAGE_PADDING}px; }}
            .parameters thead {{ display: none; }}
            .parameters table, .parameters tbody, .parameters tr, .parameters td {{
                display: block;
                width: auto;
            }}
            .parameters td {{ border: none; }}
            .parameters tr + tr {{ border-top: 1px solid {tokens['border']}; }}
            .parameters td:first-child {{ background-color: {tokens['window']}; color: {tokens['text']}; }}
        }}
        """


# A fenced code block indented under a list item, as GitHub shows it:
#
#     *   **Structure**:
#         ```text
#         >Header
#         ```
#
# Python-Markdown's fenced_code only sees a fence at the start of a line, so it
# takes this one for an inline code span ("text >Header ..." on one line).
# stash_indented_fences takes such blocks out of the page before it is
# converted, and the caller puts each back as the block it is. The pages stay
# plain Markdown, which GitHub and the MCP agents read as it is.
_FENCE_OPEN = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[*+-]|\d+[.)])[ \t]+\S")
# Four columns of indentation make a code block of a line that no list item holds.
CODE_BLOCK_INDENT = 4


def _fence_opening(line):
    """The match of a line that opens a fenced block (its indent, fence and info string), or None."""
    opening = _FENCE_OPEN.match(line)
    if opening is None or (opening["fence"][0] == "`" and "`" in opening["info"]):
        return None
    return opening


def _closes_fence(line, opening):
    """Whether line closes the fenced block that opening (a _fence_opening match) opens.

    A block at the start of a line is closed as the converter closes it, by
    the same fence at the start of a line. An indented block, by a line of
    only a fence of its character at least as long, indented no more than
    three characters beyond its own fence.
    """
    fence, indent = opening["fence"], opening["indent"]
    if not indent:
        return line.rstrip(" ") == fence
    stripped = line.strip()
    margin = len(line) - len(line.lstrip(" \t"))
    return margin <= len(indent) + 3 and stripped.startswith(fence) and not stripped.strip(fence[0])


def _unindent(line, width):
    cut = 0
    while cut < width and cut < len(line) and line[cut] in " \t":
        cut += 1
    return line[cut:]


def _columns(line):
    """How far in, tabs as four columns, a line's text starts."""
    expanded = line.expandtabs(CODE_BLOCK_INDENT)
    return len(expanded) - len(expanded.lstrip())


def stash_indented_fences(text, marker="<!--FENCE_{}-->"):
    """(text with each indented fenced block replaced by marker, the blocks), each block fenced at column 0.

    The marker's {} is the block's number in the list. A fence at the start
    of a line is left to the converter, as is everything inside one; so is
    an indented fence that is never closed, and a fence of four columns or
    more that no list item holds, which is a code block showing its lines.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    kept, blocks = [], []
    items = []  # the columns of the list items the line being read is inside
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip():
            columns = _columns(line)
            while items and items[-1] >= columns:
                items.pop()
            if _LIST_ITEM.match(line):
                items.append(columns)
        opening = _fence_opening(line)
        closing = None
        if opening is not None and (items or _columns(line) < CODE_BLOCK_INDENT):
            closing = next(
                (end for end in range(index + 1, len(lines)) if _closes_fence(lines[end], opening)),
                None,
            )
        if closing is None:
            kept.append(line)
            index += 1
        elif not opening["indent"]:
            kept.extend(lines[index:closing + 1])
            index = closing + 1
        else:
            width = len(opening["indent"])
            body = [_unindent(body_line, width) for body_line in lines[index + 1:closing]]
            blocks.append("\n".join([opening["fence"] + opening["info"].rstrip(), *body, opening["fence"]]))
            kept.append(opening["indent"] + marker.format(len(blocks) - 1))
            index = closing + 1
    return "\n".join(kept), blocks
