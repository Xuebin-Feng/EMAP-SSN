# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Read a Consistency Check report as the Configuration window shows it.

The report is rich text (EMAPSSN_Config.consistency_report_html): a table with
a row per check, then a paragraph for each problem. parse_report lays it out
as a QTextDocument, as the window's label does, and reads back what a user
sees: each row's mark and its colour, the cells' texts and the verdict's
colour, and the paragraphs below the table.

Not collected by unittest; import with ``from tests.consistency_report import ...``.
"""
import base64
import re
from types import SimpleNamespace

from PySide6.QtGui import QTextCursor, QTextDocument, QTextTable

# Qt turns the spaces of a no-wrap cell into non-breaking ones, and the report
# gives a long header places to break (zero-width spaces). An image is one
# object-replacement character in a block's text.
NO_BREAK_SPACE = chr(0xA0)
ZERO_WIDTH_SPACE = chr(0x200B)
OBJECT_REPLACEMENT = chr(0xFFFC)
LINE_SEPARATOR = chr(0x2028)
PARAGRAPH_SEPARATOR = chr(0x2029)
_INVISIBLE = {ord(NO_BREAK_SPACE): " ", ord(ZERO_WIDTH_SPACE): None}


def clean(text):
    """text as it reads: ordinary spaces, no break opportunities, line breaks as newlines."""
    text = text.translate(_INVISIBLE)
    return text.replace(LINE_SEPARATOR, "\n").replace(PARAGRAPH_SEPARATOR, "\n").strip()


def mark_colour(image_name):
    """The stroke colour of a mark (an SVG in a data URI), such as "#15803d"."""
    prefix = "data:image/svg+xml;base64,"
    assert image_name.startswith(prefix), image_name[:40]
    svg = base64.b64decode(image_name[len(prefix):]).decode("utf-8")
    return re.search(r'stroke="(#[0-9a-fA-F]{6})"', svg).group(1)


def _cell_text(table, row, column):
    cell = table.cellAt(row, column)
    cursor = cell.firstCursorPosition()
    cursor.setPosition(cell.lastCursorPosition().position(), QTextCursor.MoveMode.KeepAnchor)
    return cursor.selectedText()


def _first_colour(table, row, column):
    """The colour of the first text of a cell."""
    block = table.cellAt(row, column).firstCursorPosition().block()
    for piece in block.textFormats():
        if block.text()[piece.start:piece.start + piece.length].strip():
            return piece.format.foreground().color().name()
    return None


def _image_name(block):
    """The name of the first image in block, or None."""
    for piece in block.textFormats():
        if piece.format.isImageFormat():
            return piece.format.toImageFormat().name()
    return None


def _mark(table, row):
    """(name of the image in a row's first cell, or None, its stroke colour)."""
    name = _image_name(table.cellAt(row, 0).firstCursorPosition().block())
    return (name, mark_colour(name)) if name else (None, None)


def lines_per_cell(report_html, width):
    """The lines each row's cells take when the report's text is width pixels wide.

    One tuple per row: (check, result, details), the last None for a row whose
    result takes both cells. A table that could not fit a cell on its line
    keeps it whole and overflows, so a cell of one line is one that fits.
    """
    document = QTextDocument()
    document.setHtml(report_html)
    document.setTextWidth(width)
    document.size()  # lays the text out
    rows = []
    for table in (frame for frame in document.rootFrame().childFrames()
                  if isinstance(frame, QTextTable)):
        for number in range(table.rows()):
            def lines(column):
                block = table.cellAt(number, column).firstCursorPosition().block()
                return block.layout().lineCount()

            spans = table.cellAt(number, 2).columnSpan() > 1
            rows.append((lines(1), lines(2), None if spans else lines(3)))
    return rows


def parse_report(report_html):
    """The report as (rows, notes): rows of SimpleNamespace, and the paragraphs below the table.

    A row has mark and mark_colour, name, result, details (the cell's lines),
    verdict_colour, the colour of the details' first line, and span, how many
    columns its result cell takes. A note is (the colour of its mark, its
    text): a paragraph, with its lines separated by newlines.
    """
    document = QTextDocument()
    document.setHtml(report_html)
    tables = [frame for frame in document.rootFrame().childFrames() if isinstance(frame, QTextTable)]
    rows, notes = [], []
    for table in tables:
        for number in range(table.rows()):
            mark, colour = _mark(table, number)
            result_cell = table.cellAt(number, 2)
            details = (
                clean(_cell_text(table, number, 3)).split("\n")
                if result_cell.columnSpan() == 1 else []
            )
            rows.append(SimpleNamespace(
                mark=mark,
                mark_colour=colour,
                name=clean(_cell_text(table, number, 1)),
                result=clean(_cell_text(table, number, 2)),
                details=details,
                verdict_colour=_first_colour(table, number, 3) if details else None,
                span=result_cell.columnSpan(),
            ))
    block = document.begin()
    while block.isValid():
        text = clean(block.text().replace(OBJECT_REPLACEMENT, ""))
        if QTextCursor(block).currentTable() is None and text:
            name = _image_name(block)
            notes.append((mark_colour(name) if name else None, text))
        block = block.next()
    return SimpleNamespace(rows=rows, notes=notes)
