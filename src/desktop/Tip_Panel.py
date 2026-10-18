# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tip panel: the card under a window's form that describes the field in focus.

Config, VR Config and Tools show a field's description there, and each window's
own messages (a check's result, a failure) too. They share this module's panel,
SpacedTipLabel, and the tip a field gives it. The panel is as tall as so many
lines of its own font (TIP_PANEL_LINES) when its window is first laid out, and
stays so whatever it shows: what does not fit scrolls.

A tip is either plain text, which the panel shows as it always has (escaped, a
line break where the text has one), or a Tip, which the panel lays out:

* a bold title, the field's own label (already translated, without its colon);
* a summary, then muted detail lines;
* a table of options, one row each: the option as the field's dropdown shows it,
  in a code chip (a key cap, like the Viewer's info card keys: a white box with a
  thin border, round corners and room of its own, which the panel draws behind the
  text), then what it does;
* notes with a status icon: "warning" (a cost or a risk) and "info" (a remark);
* a live note, such as what AUTO picks right now, with a swatch when it is a colour;
* a facts footer built from the field itself, not from prose: its default, its
  range and unit (the spin box's own), and the settings key.

A translated text holds no markup. Text between backticks is a typed value, which
shows as a code chip; a placeholder such as {value} takes a value that is never
read as markup (a Code value shows as a chip). Every text is escaped, and the
structure is added here, so a translator sees one clean sentence for each text.
The colours are the theme's (Studio_Theme.TOKENS) at the time a tip is shown.

A window registers FieldTip(tip, Field(...)) wherever it has a plain tip: the
Field names what the tip describes (label, widget, default, key), and the facts
and options are read from it each time the tip shows.
"""

from __future__ import annotations

import base64
import html
import re
import unicodedata
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Callable, Optional

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextOption,
)
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QFrame, QLabel, QTextEdit, QWidget

from desktop.Studio_Theme import (
    HINT_BORDER,
    HINT_PADDING,
    HINT_RIGHT_PADDING,
    HINT_SCROLL_EDGE,
    HINT_TEXT_MARGIN,
    KEY_CAP_BORDER,
    KEY_CAP_BOTTOM,
    KEY_CAP_PADDING,
    KEY_CAP_RADIUS,
    TOKENS,
    IconLabel,
    coloured_svg,
    icon_svg,
    key_cap_colors,
    status_color,
)

# The few words the panel itself adds (the facts' labels, the words that stand
# for the icons) are filed under the context "TipPanel", for every window.


# =====================================================================
# Marks: status icons and colour swatches, as images rich text can lay out
# =====================================================================

# Tell a status by its shape as well as its colour (Studio_Theme.status_color kinds).
STATUS_ICONS = {
    "success": "check",
    "warning": "triangle-alert",
    "danger": "x",
    "subtle": "info",
    "info": "info",
}
STATUS_MARK_SIZE = 14
_marks = {}


def _image(svg, size=STATUS_MARK_SIZE):
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (
        f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}">'
    )


def status_mark(kind):
    """The icon of a status kind, in its colour, as an <img> that rich text can lay out.

    An icon rather than a glyph such as a check mark, which needs a font that has
    it. The SVG is drawn at four times its size, so it stays sharp on a
    high-resolution screen.
    """
    color = status_color("subtle" if kind == "info" else kind)
    if (kind, color) not in _marks:
        svg = coloured_svg(icon_svg(STATUS_ICONS[kind]), color)
        svg = re.sub(
            r'\b(width|height)="24"',
            lambda found: f'{found.group(1)}="{4 * STATUS_MARK_SIZE}"',
            svg,
        )
        _marks[(kind, color)] = _image(svg)
    return _marks[(kind, color)]


def swatch_mark(color_name):
    """A small rounded square in a colour (a name or code QColor reads), or "" for none.

    It carries a hairline in the text colour at low opacity, so a white or
    transparent colour still shows.
    """
    from PySide6.QtGui import QColor

    colour = QColor(str(color_name).strip())
    if not colour.isValid():
        return ""
    key = ("swatch", colour.name(QColor.NameFormat.HexArgb), TOKENS["text_subtle"])
    if key not in _marks:
        size = 4 * STATUS_MARK_SIZE
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 14 14">'
            f'<rect x="0.75" y="0.75" width="12.5" height="12.5" rx="3" '
            f'fill="{colour.name()}" fill-opacity="{colour.alphaF():.3f}" '
            f'stroke="{TOKENS["text_subtle"]}" stroke-opacity="0.55" stroke-width="1.2"/></svg>'
        )
        _marks[key] = _image(svg)
    return _marks[key]


# =====================================================================
# Text: escaping, long words, chips
# =====================================================================

ZERO_WIDTH_SPACE = "\u200b"
NO_BREAK_SPACE = "\u00a0"


def wrappable(text):
    """text with a place to break wherever a long run of characters, such as a header, has none.

    Qt breaks a label's text at spaces and hyphens only, and a word it can't
    break sets the label's minimum width, so one long header widened the whole
    window. A zero-width space (U+200B) after each separator of a run of 50 or
    more characters, and after every 20 characters of what is left, lets it
    wrap. A shorter word stays whole, so an ordinary header moves to the next
    line as one.
    """
    def broken(found):
        run = re.sub(r"([_|/\\.,:;)\]])", "\\1\u200b", found.group())
        return re.sub(r"([^\s\u200b]{20})(?=[^\s\u200b])", "\\1\u200b", run)

    return re.sub(r"[^\s\u200b]{50,}", broken, text)


class Code(str):
    """A value that shows as a code chip wherever a placeholder takes it."""


class CodeList(tuple):
    """Values that show as code chips, in a row, wherever a placeholder takes them."""


# A chip up to this long keeps together, in a table cell of its own too; a
# longer one, such as a device's full name, may wrap, so it never widens the
# panel. A wide character (Chinese, say) counts for two.
CHIP_WHOLE_LIMIT = 28


def is_whole(text):
    """Whether a chip's text is short enough to be kept on one line."""
    width = sum(2 if unicodedata.east_asian_width(character) in "WF" else 1 for character in str(text))
    return width <= CHIP_WHOLE_LIMIT


def chip(text):
    """text as a code chip: monospace on a lighter ground, kept on one line where it is short.

    Rich text can only fill a square and gives a span no padding, so the ground
    is the span's background here, which is how the chip reads back from the
    text. The panel finds every run with that background, takes it off, makes
    room for the key cap's padding on both sides and paints the cap behind the
    text (SpacedTipLabel); this function's text is unchanged by that.
    """
    from desktop.Desktop_App import MONOSPACE_QSS_FONT_STACK

    text = str(text)
    shown = html.escape(wrappable(text))
    if is_whole(text):
        shown = shown.replace(" ", "&nbsp;")
    return (
        f'<span style="font-family: {MONOSPACE_QSS_FONT_STACK}; font-size: smaller; '
        f'background-color: {TOKENS["surface"]}; color: {TOKENS["accent_text"]};">'
        f"{shown}</span>"
    )


# A `code` span, or a {placeholder} with an optional format spec.
_INLINE = re.compile(r"`([^`]+)`|\{(\w+)(?::([^{}]*))?\}")
_INLINE_PLACEHOLDER = re.compile(r"\{(\w+)(?::([^{}]*))?\}")


def _filled(value, spec):
    return format(value, spec) if spec else str(value)


def _pieces(text, values):
    """text as a list of (kind, string): "text" or "code", the placeholders filled in.

    A placeholder takes its value as plain text, or as a chip when the value
    is a Code, and inside a `code` span it takes it as part of the chip. What a
    value holds is never read as markup.
    """
    values = values or {}
    pieces, end = [], 0
    for found in _INLINE.finditer(text):
        if found.start() > end:
            pieces.append(("text", text[end:found.start()]))
        end = found.end()
        if found.group(1) is not None:
            pieces.append(("code", _INLINE_PLACEHOLDER.sub(
                lambda inner: _filled(values[inner.group(1)], inner.group(2))
                if inner.group(1) in values else inner.group(), found.group(1),
            )))
        elif found.group(2) in values:
            value = values[found.group(2)]
            if isinstance(value, CodeList):
                for number, item in enumerate(value):
                    pieces.append(("text", ", ") if number else ("text", ""))
                    pieces.append(("code", str(item)))
            else:
                pieces.append(("code" if isinstance(value, Code) else "text", _filled(value, found.group(3))))
        else:
            pieces.append(("text", found.group()))
    if end < len(text):
        pieces.append(("text", text[end:]))
    return pieces


def rich(text, values=None):
    """text as rich text: escaped, its `code` spans and Code placeholders as chips."""
    return "".join(
        chip(string) if kind == "code" else html.escape(wrappable(string))
        for kind, string in _pieces(text, values)
    )


def plain(text, values=None):
    """text as it reads without the chips: the same, with no backticks."""
    return "".join(string for _kind, string in _pieces(text, values))


# =====================================================================
# The tip a field gives
# =====================================================================

class _Missing:
    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


@dataclass(frozen=True)
class TipOption:
    """A choice the field offers, and what it does.

    value is the stored value of the field's dropdown item; the chip shows the
    item's own text, which is what the user sees there, and a row whose value
    the dropdown lacks (a mode hidden on this device) is left out. label gives
    the chip's text for a row that is not an item of a dropdown.
    """

    value: object
    text: str
    label: Optional[str] = None


@dataclass(frozen=True)
class TipNote:
    """A sentence with a status icon: kind "warning" (a cost or a risk) or "info" (a remark)."""

    kind: str
    text: str


@dataclass(frozen=True)
class LiveNote:
    """What holds right now, worked out when the tip is shown: a sentence, with a swatch for a colour.

    text may hold {placeholders} filled from values (Code values show as chips).
    """

    text: str
    values: Optional[dict] = None
    swatch: Optional[str] = None
    kind: str = "info"


@dataclass(frozen=True)
class Field:
    """What a tip describes: the field's label, its widget, its default and its settings key.

    label is the QLabel (or the text) whose words, without the colon, are the
    tip's title. widget gives the options of a dropdown and the range of a spin
    box (the widget itself or one inside it). default is the value the program
    starts with, when it has one (MISSING when it has none worth showing).
    limits is the (lowest, highest) the setting takes where the program
    defines them, None when it has none; MISSING reads the spin box's own range.
    """

    key: Optional[str] = None
    label: object = None
    widget: Optional[QWidget] = None
    default: object = MISSING
    limits: object = MISSING

    def title(self):
        text = self.label.text() if isinstance(self.label, QLabel) else (self.label or "")
        return str(text).replace("&&", "\0").replace("&", "").replace("\0", "&").strip().rstrip(":：").strip()


@dataclass(frozen=True)
class Tip:
    """A field's description, as texts that are already translated.

    summary opens it; details follow, muted; options become a table; notes and
    live (a function of the Field) follow; the facts footer closes it. unit
    goes after a numeric default and range ("GiB"). facts=False leaves the
    footer out, and title replaces the label's words as the title.
    """

    summary: str
    details: tuple = ()
    options: tuple = ()
    notes: tuple = ()
    live: Optional[Callable] = None
    unit: str = ""
    facts: bool = True
    title: Optional[str] = None

    def texts(self):
        """Every text the tip holds, as a translator sees each."""
        found = [self.summary, *self.details]
        found += [option.text for option in self.options]
        found += [note.text for note in self.notes]
        return found


@dataclass(frozen=True)
class FieldTip:
    """A Tip and the Field it describes: what a window registers for the field's widgets."""

    tip: Tip
    field: Field = dataclass_field(default_factory=Field)


# =====================================================================
# Reading the field
# =====================================================================

def _spin_box(widget):
    if widget is None:
        return None
    if isinstance(widget, QAbstractSpinBox):
        return widget
    return widget.findChild(QAbstractSpinBox)


def _combo(widget):
    if isinstance(widget, QComboBox):
        return widget
    return None


def _is_dropdown(widget):
    """A dropdown, or a widget that holds one as .combo (a file dropdown with its folder button)."""
    return isinstance(widget, QComboBox) or isinstance(getattr(widget, "combo", None), QComboBox)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _shown_default(field, unit):
    """The field's default as the field shows it, or None when it has none worth showing."""
    value = field.default
    if value is MISSING:
        return None
    if isinstance(getattr(field.widget, "combo", None), QComboBox):
        return None  # a file dropdown: its default is a sample file, not a choice
    combo = _combo(field.widget)
    if combo is not None:
        index = combo.findData(value)
        if index >= 0:
            return combo.itemText(index)
    if value is None or (isinstance(value, str) and not value.strip()):
        if _is_dropdown(field.widget):
            return None  # nothing is chosen at first: not a fact worth showing
        return QCoreApplication.translate("TipPanel", "(empty)")
    if isinstance(value, bool):
        texts = getattr(field.widget, "state_texts", None)
        if texts is not None:
            return texts()[value]
        return QCoreApplication.translate("TipPanel", "on") if value else QCoreApplication.translate("TipPanel", "off")
    text = f"{value:g}" if _number(value) else str(value)
    if unit and _number(value):
        text += f" {unit}"
    return text


# A bound this large is "no limit" in practice (a spin box that takes any number), not a fact.
UNBOUNDED = 100000


def _shown_range(field, unit):
    box = _spin_box(field.widget)
    limits = field.limits
    if limits is MISSING:
        if box is None:
            return None
        limits = (box.minimum(), box.maximum())
    if limits is None or any(abs(bound) >= UNBOUNDED for bound in limits):
        return None
    show = box.textFromValue if box is not None else (lambda value: f"{value:g}")
    suffix = (box.suffix() if box is not None else "") or (f" {unit}" if unit else "")
    return f"{show(limits[0])} \u2013 {show(limits[1])}{suffix}"


def _facts(tip, field):
    """The facts footer's (label, value) pairs, read from the field."""
    if not tip.facts:
        return []
    facts = []
    default = _shown_default(field, tip.unit)
    if default is not None:
        facts.append((QCoreApplication.translate("TipPanel", "Default", "a field's default value"), default))
    shown_range = _shown_range(field, tip.unit)
    if shown_range is not None:
        facts.append((QCoreApplication.translate("TipPanel", "Range"), shown_range))
    if field.key:
        facts.append((QCoreApplication.translate("TipPanel", "Setting", "the name of a settings entry"), field.key))
    return facts


def _options(tip, field):
    """The options table's (chip text, text) rows: those the field's dropdown holds, as it names them."""
    combo = _combo(field.widget)
    states = getattr(field.widget, "state_texts", None)
    rows = []
    for option in tip.options:
        label = option.label
        if label is None and states is not None and isinstance(option.value, bool):
            label = states()[option.value]  # a switch's own words for the state
        elif combo is not None and option.label is None:
            index = combo.findData(option.value)
            if index < 0:
                continue
            label = combo.itemText(index)
        rows.append((str(option.value) if label is None else label, option.text))
    return rows


def _live_notes(tip, field):
    """The live notes of a tip as a list: its function may give one LiveNote, a list of them, or None."""
    if tip.live is None:
        return []
    found = tip.live(field)
    if found is None:
        return []
    return [found] if isinstance(found, LiveNote) else [note for note in found if note is not None]


def nearest_existing_folder(path):
    """The folder a path is in, or the nearest of its parents that exists, or None."""
    import os

    folder = os.path.abspath(os.path.expanduser(str(path)))
    while folder and not os.path.isdir(folder):
        parent = os.path.dirname(folder)
        if parent == folder:
            return None
        folder = parent
    return folder or None


def free_bytes(path):
    """The free space on the drive a folder is (or would be) on, or None when it can't be read."""
    import shutil

    folder = nearest_existing_folder(path)
    if folder is None:
        return None
    try:
        return shutil.disk_usage(folder).free
    except OSError:
        return None


@dataclass
class TipModel:
    """A tip resolved against its field: everything that is shown, as data."""

    title: str
    summary: str
    details: list
    options: list
    notes: list
    live: list
    facts: list


def resolve(entry):
    """The TipModel of a FieldTip, or of a bare Tip (no field: no title, options or facts)."""
    if isinstance(entry, Tip):
        entry = FieldTip(entry)
    tip, field = entry.tip, entry.field
    return TipModel(
        title=tip.title if tip.title is not None else field.title(),
        summary=tip.summary,
        details=list(tip.details),
        options=_options(tip, field),
        notes=[(note.kind, note.text) for note in tip.notes],
        live=_live_notes(tip, field),
        facts=_facts(tip, field),
    )


# =====================================================================
# Drawing it
# =====================================================================

LINE_SCALE = 1.2    # a panel line is the font's line, this much taller
LINE = f"line-height: {round(LINE_SCALE * 100)}%;"


def _paragraph(content, margin_top, style=""):
    return f'<p style="margin: {margin_top}px 0px 0px 0px; {LINE} {style}">{content}</p>'


def _row_table(rows, margin_top):
    """A table of (first cell, second cell[, keep whole]) rows, the first as narrow as its content.

    A row with a third value has a chip as its first cell, which gets room
    after it; True keeps it whole (a short chip never wraps, however the
    second cell's text squeezes it: Chinese text may break between any two
    characters, so a spaces-only rule is not enough). A row without one
    holds an icon.
    """
    def cells(first, second, *chip_row):
        keep = " white-space: nowrap;" if chip_row and chip_row[0] else ""
        gap = 14 if chip_row else 8
        return (
            f'<tr><td valign="top" style="padding: 2px {gap}px 2px 0px;{keep}">{first}</td>'
            f'<td valign="top" width="100%" style="padding: 1px 0px 1px 0px;">{second}</td></tr>'
        )

    body = "".join(cells(*row) for row in rows)
    return (
        f'<table width="100%" cellspacing="0" cellpadding="0" style="margin: {margin_top}px 0px 0px 0px;">'
        f"{body}</table>"
    )


def _lines(text):
    """text escaped, with a line break where it has one."""
    return html.escape(wrappable(text)).replace("\r\n", "\n").replace("\n", "<br>")


def status_html(kind, text):
    """A status message with its icon beside it, for a panel that has no corner icon."""
    return _row_table([(status_mark(kind), _lines(text))], 0)


def status_text_html(text):
    """A status message's text, which the panel's corner icon marks."""
    return _paragraph(_lines(text), 0)


def model_html(model):
    """The rich text of a TipModel."""
    soft, muted, subtle, separator = (
        TOKENS[key] for key in ("text_soft", "text_muted", "text_subtle", "separator")
    )
    parts = []
    if model.title:
        parts.append(_paragraph(
            f'<span style="font-weight: 600; color: {TOKENS["text"]};">'
            f"{html.escape(wrappable(model.title))}</span>", 0,
        ))
    if model.summary:
        parts.append(_paragraph(
            f'<span style="color: {soft};">{rich(model.summary)}</span>', 2 if model.title else 0,
        ))
    if model.options:
        parts.append(_row_table(
            [(chip(label), f'<span style="color: {muted};">{rich(text)}</span>', is_whole(label))
             for label, text in model.options], 5,
        ))
    for detail in model.details:
        parts.append(_paragraph(f'<span style="color: {muted};">{rich(detail)}</span>', 3))
    if model.notes or model.live:
        rows = [(status_mark(kind), f'<span style="color: {soft};">{rich(text)}</span>')
                for kind, text in model.notes]
        for note in model.live:
            mark = (swatch_mark(note.swatch) if note.swatch else "") or status_mark(note.kind)
            rows.append((mark, f'<span style="color: {soft};">{rich(note.text, note.values)}</span>'))
        parts.append(_row_table(rows, 5))
    if model.facts:
        items = "&nbsp;&nbsp; ".join(
            f'<span style="color: {subtle};">{html.escape(label)}</span>&nbsp;{chip(value)}'
            for label, value in model.facts
        )
        parts.append(
            f'<table width="100%" cellspacing="0" cellpadding="0" style="margin: 6px 0px 0px 0px;">'
            f'<tr><td style="border-top: 1px solid {separator}; padding: 5px 0px 0px 0px; {LINE}">'
            f"{items}</td></tr></table>"
        )
    return "".join(parts)


def model_plain(model):
    """A TipModel as plain text, for a screen reader: the icons become words."""
    lines = [model.title] if model.title else []
    if model.summary:
        lines.append(plain(model.summary))
    lines += [
        QCoreApplication.translate("TipPanel", "{option}: {text}").format(option=label, text=plain(text))
        for label, text in model.options
    ]
    lines += [plain(detail) for detail in model.details]
    kinds = {"warning": QCoreApplication.translate("TipPanel", "Warning:"), "info": QCoreApplication.translate("TipPanel", "Note:")}
    lines += [f"{kinds.get(kind, kinds['info'])} {plain(text)}" for kind, text in model.notes]
    lines += [plain(note.text, note.values) for note in model.live]
    if model.facts:
        lines.append(" ".join(
            QCoreApplication.translate("TipPanel", "{label}: {value}.", "a field's fact, such as its default")
            .format(label=label, value=value)
            for label, value in model.facts
        ))
    return "\n".join(lines)


def tip_html(entry):
    """The rich text of a Tip or FieldTip."""
    return model_html(resolve(entry))


def tip_plain(entry):
    """The plain text of a Tip or FieldTip, the field's accessible description."""
    return model_plain(resolve(entry))


def tooltip_text(entry):
    """The plain text of a tip_db entry (a string, or a tip), for a widget's tooltip."""
    return tip_plain(entry) if isinstance(entry, (Tip, FieldTip)) else entry


def with_note(entry, kind, note):
    """entry with a note added, as a message about its field's state: a TipNote on a tip, a line of plain text."""
    if isinstance(entry, FieldTip):
        return FieldTip(replace(entry.tip, notes=(*entry.tip.notes, TipNote(kind, note))), entry.field)
    if isinstance(entry, Tip):
        return replace(entry, notes=(*entry.notes, TipNote(kind, note)))
    return f"{entry}\n{note}"


# =====================================================================
# The panel
# =====================================================================

def change_signal(widget):
    """The signal a field's widget emits when the user changes its value, or None."""
    box = _spin_box(widget)
    for candidate in (widget, box):
        for name in ("textChanged", "valueChanged", "currentIndexChanged", "toggled"):
            signal = getattr(candidate, name, None) if candidate is not None else None
            if signal is not None:
                return signal
    return None


# How many lines of the panel's text the tip area shows without scrolling when the window is
# first laid out. A panel line is the font's line at LINE_SCALE (what a paragraph of tip text
# takes, measured by the panel in its own language), and the card's border and padding are
# counted on top. Seven and a half lines hold the title, a two-line summary, a detail line,
# a note or two and the facts footer with its rule and the gaps between them: with the
# windows at their ordinary widths that is 98 % of Config's tips (median 5.6 lines) and 89 %
# of Tools' (median 4.9). A tip with an options table, several notes and a long footer is
# longer, 9 to 11 lines, and scrolls.
TIP_PANEL_LINES = 7.5
# The least it shrinks to when the user drags the splitter: a title line and a little over.
# This keeps the windows' minimum height where it was (the label it replaces asked for 51 px).
TIP_PANEL_MINIMUM_LINES = 1.25

# A code chip is a key cap (Studio_Theme.KEY_CAP_*, the look of the Viewer's info card keys):
# a white box with a thin border, a thicker bottom edge and round corners, with real room
# round its text. Rich text can't give a span padding or a radius, so the panel paints the
# box behind the text and reserves the room in the line: a word joiner (U+2060), which has
# no width of its own, on each side of the chip's text, spaced out by the cap's padding and
# border. The joiners hold the chip to the punctuation next to it ("(e.g. WP_01*)" breaks
# only at its spaces), and since they take their own width, the box covers no neighbouring
# glyph. The box is as tall as the chip's own glyphs and borders, which fits the line.
CHIP_GLUE = "\u2060"


class PanelHeightAtFirstLayout(QObject):
    """Starts a splitter's last pane at the height its tip panel asks for, when the splitter is first shown.

    The pane holds the panel and, in Config, the action buttons under it, which
    wrap onto more rows in a narrow window. The pane's size hint counts one
    row, so a splitter set from that hint at construction would leave the panel
    shorter than its fixed height wherever they wrap. At the first show the
    layout has a width: the pane then takes what its layout needs at that width,
    and the tabs above take the rest. What the panel shows never changes this.
    """

    def __init__(self, split):
        super().__init__(split)
        self._split = split
        split.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._split and event.type() == QEvent.Type.Show:
            self._split.removeEventFilter(self)
            self.apply()
        return False

    def apply(self):
        split = self._split
        pane = split.widget(split.count() - 1)
        layout = pane.layout()
        wanted = (
            layout.totalHeightForWidth(pane.width()) if layout.hasHeightForWidth() and pane.width() > 0
            else layout.totalSizeHint().height()
        )
        sizes = split.sizes()
        split.setSizes([*sizes[:-2], sum(sizes[-2:]) - wanted, wanted])


class SpacedTipLabel(QTextEdit):
    """The tip panel: a read-only card of rich text at a fixed height, which scrolls.

    Config, VR Config and Tools put it under their form. It shows a tip (what
    setTip is given, or the placeholder it starts with) or a message (a
    status, a report, plain text). Its height does not follow what it shows:
    a window starts it at preferred_height() and text that does not fit scrolls,
    with a vertical scroll bar only while it is needed, never a horizontal one.
    What it shows starts at the top, and the mouse wheel over it scrolls.

    The name is the QLabel it once was, and text() still gives the rich text
    last shown (the same string the label held). The code chips (see chip())
    are drawn here with rounded corners, on the theme's lighter ground; the
    document itself draws only their text.
    """

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._html = ""
        self._entry = None
        self._tip_shown = True
        self._chips = []
        self._line_cache = None
        self._sample = str(text) or "Ag"
        self.setReadOnly(True)
        self.document().setUndoRedoEnabled(False)
        # The text keeps a few pixels from the viewport's edges, so glyphs and a chip at the
        # start of a line are not cut by the clip.
        self.document().setDocumentMargin(HINT_TEXT_MARGIN)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setAcceptRichText(False)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        # The card's right padding is a margin of the viewport, so the text keeps
        # that far from the scroll bar, which lies in what the stylesheet leaves.
        self.setViewportMargins(0, 0, HINT_RIGHT_PADDING - HINT_SCROLL_EDGE - HINT_TEXT_MARGIN, 0)
        self._show(self._plain_html(text), "")

    # --- What the panel holds ---

    def text(self):
        """The rich text on display, as setText, setTip, setStatus and setReport made it."""
        return self._html

    def shows_tip(self):
        """Whether the panel holds a tip or its placeholder, not a message.

        A window's hover rule asks this: hovering replaces a tip, and leaves a
        message (a status, a report, plain text) until the user picks a tip.
        """
        return self._tip_shown

    @staticmethod
    def _plain_html(text):
        escaped_text = html.escape(str(text)).replace("\r\n", "\n").replace("\r", "\n")
        return f'<div style="{LINE}">{escaped_text.replace(chr(10), "<br>")}</div>'

    def _corner_icon(self):
        """The icon in the panel's corner (Studio_Theme.add_hint_icon puts it there), or None."""
        return self.findChild(IconLabel)

    def _mark_corner(self, kind):
        """Show a status in the corner icon (a tip's is the info icon). False if there is no corner icon."""
        icon = self._corner_icon()
        if icon is None:
            return False
        if kind in (None, "info", "subtle"):
            icon.set_icon("info")
        else:
            icon.set_icon(STATUS_ICONS[kind], status_color(kind))
        return True

    def setText(self, text):
        """Show plain text as a message: escaped, a line break where the text has one.

        Like any message, it is not a tip (see shows_tip), so hovering leaves it.
        """
        self._tip_shown = False
        self._mark_corner(None)
        self._show(self._plain_html(text), "")

    def setTip(self, entry):
        """Show a tip: plain text as it always was, a Tip or FieldTip laid out as its model says."""
        self._set_tip(entry, keep_place=False)

    def _set_tip(self, entry, keep_place):
        self._tip_shown = True
        if isinstance(entry, (Tip, FieldTip)):
            model = resolve(entry)
            self._show(model_html(model), model_plain(model), keep_place)
            self._entry = entry
        else:
            self._show(self._plain_html(entry), "", keep_place)
        self._mark_corner(None)

    def refresh(self):
        """Lay the tip on display out again, so its live note shows what holds now.

        Returns False, and changes nothing, while the panel shows anything else
        (a message, plain text), so a field changing never replaces one. The
        tip stays where the user scrolled it.
        """
        if not self._tip_shown or self._entry is None:
            return False
        self._set_tip(self._entry, keep_place=True)
        return True

    def setStatus(self, kind, text):
        """Show a status message ("success", "warning", "danger" or "info"), marked by the corner icon.

        The corner icon becomes the status's icon, in its colour, until a tip
        or plain text replaces the message. A panel with no corner icon shows
        the icon beside the text instead.

        A status is a message, not a tip (see shows_tip), so hovering leaves
        it in place.
        """
        self._tip_shown = False
        if self._mark_corner(kind):
            self._show(status_text_html(text), text)
        else:
            self._show(status_html(kind, text), text)

    def setReport(self, report_html, status=None):
        """Show a status report, rich text whose own text is already escaped.

        status is its overall verdict ("success", "warning" or "danger"), which
        the corner icon shows. The report's own rows keep their icons.

        A report is a message, not a tip (see shows_tip), so hovering leaves it in place.
        """
        self._tip_shown = False
        self._mark_corner(status)
        self._show(report_html, None)

    def _show(self, rich_text, plain_text, keep_place=False):
        """Put rich_text in the document, scrolled to the top (or where it was, with keep_place)."""
        self._entry = None
        self.setAccessibleName(plain_text or "")
        if rich_text == self._html:
            return  # already showing it, where the user left it
        place = self.verticalScrollBar().value() if keep_place else 0
        self._html = rich_text
        self.setHtml(rich_text)
        self._take_chips()
        self.verticalScrollBar().setValue(place)

    # --- The size ---

    def line_height(self):
        """The height of one line of the panel's text, as it lays it out, in pixels.

        Measured on the text the panel started with (its placeholder, in the
        window's language), so a font that needs taller lines for that
        language, Chinese for one, counts: the step between two lines.
        """
        font = self.font()
        if self._line_cache is None or self._line_cache[0] != font.key():
            probe = QTextDocument()
            probe.setDefaultFont(font)
            probe.setDocumentMargin(0)
            heights = []
            for lines in (1, 2):
                probe.setHtml(f'<div style="{LINE}">{"<br>".join([html.escape(self._sample)] * lines)}</div>')
                probe.setTextWidth(-1)
                heights.append(probe.size().height())
            self._line_cache = (font.key(), max(1, round(heights[1] - heights[0])))
        return self._line_cache[1]

    def preferred_height(self, lines=TIP_PANEL_LINES):
        """The height of the card for that many lines of text (a number of lines, not only whole ones): the lines, the card's padding and its border."""
        return round(lines * self.line_height()) + 2 * (HINT_BORDER + HINT_PADDING)

    def sizeHint(self):
        return QSize(256, self.preferred_height())

    def minimumSizeHint(self):
        return QSize(64, self.preferred_height(TIP_PANEL_MINIMUM_LINES))

    def setAlignment(self, alignment):
        """Ignored: the text starts at the card's top left, whichever label code asks (add_hint_icon)."""

    # --- Code chips ---

    def _take_chips(self):
        """Find the code chips in the document, make room round each and take the square ground off.

        A chip is a run of text with the chip ground (the theme's surface
        colour) as its background, which the document would paint as a
        rectangle with square corners. The ground comes off. The word
        joiners that reserve the key cap's padding go in on both sides of the run,
        and chip_rects() gives the key caps the panel paints behind them.
        """
        document = self.document()
        ground = QColor(TOKENS["surface"])
        found = []
        block = document.begin()
        while block.isValid():
            for piece in block.textFormats():
                brush = piece.format.background()
                if brush.style() != Qt.BrushStyle.NoBrush and brush.color() == ground:
                    start = block.position() + piece.start
                    found.append((start, start + piece.length, QTextCharFormat(piece.format)))
            block = block.next()
        self._chips = []
        cursor = QTextCursor(document)
        padding = KEY_CAP_PADDING + KEY_CAP_BORDER
        # From the last chip to the first, so the places of those before are not moved by the joiners.
        for start, end, text_format in reversed(found):
            text_format.clearBackground()
            glue = QTextCharFormat(text_format)
            glue.setFontLetterSpacingType(QFont.SpacingType.AbsoluteSpacing)
            glue.setFontLetterSpacing(padding)  # the room after the joiner: before the text, and after it
            cursor.setPosition(end)
            cursor.insertText(CHIP_GLUE, glue)
            cursor.setPosition(start)
            cursor.insertText(CHIP_GLUE, glue)
            cursor.setPosition(start + 1)
            cursor.setPosition(end + 1, QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(text_format)
        fill, border, bottom = (QColor(name) for name in key_cap_colors())
        for number, (start, end, _format) in enumerate(found):
            # Each chip before this one added its two joiners; this chip's box is from its first to past its last.
            self._chips.append((start + 2 * number, end + 2 * number + 2, fill, border, bottom))

    def _chip_boxes(self):
        """(rectangle, fill, border, bottom edge) of every chip's key cap, in document coordinates, one for each line of a wrapped chip."""
        document = self.document()
        document_layout = document.documentLayout()
        boxes = []
        for start, end, fill, border, bottom in self._chips:
            block = document.findBlock(start)
            layout = block.layout()
            if layout is None:
                continue
            # A block's layout is placed within its table cell; the document
            # layout adds the cell's and the table's place.
            origin = document_layout.blockBoundingRect(block).topLeft()
            for number in range(layout.lineCount()):
                line = layout.lineAt(number)
                first = max(start - block.position(), line.textStart())
                last = min(end - block.position(), line.textStart() + line.textLength())
                if last <= first:
                    continue
                # The box is as tall as the glyphs of the chip's text on this line, whatever font
                # (a fallback one for Chinese) drew them, and as wide as the text with its room.
                ascent = descent = 0.0
                for run in line.glyphRuns(first, last - first):
                    raw = run.rawFont()
                    ascent, descent = max(ascent, raw.ascent()), max(descent, raw.descent())
                if not ascent and not descent:
                    continue
                left, right = line.cursorToX(first)[0], line.cursorToX(last)[0]
                baseline = origin.y() + line.y() + line.ascent()
                boxes.append((QRectF(
                    origin.x() + left,
                    baseline - ascent - KEY_CAP_BORDER,
                    right - left,
                    ascent + descent + KEY_CAP_BORDER + KEY_CAP_BOTTOM,
                ), fill, border, bottom))
        return boxes

    def chip_rects(self):
        """(rectangle, fill colour) of every chip's key cap, in document coordinates."""
        return [(rect, fill) for rect, fill, _border, _bottom in self._chip_boxes()]

    def paintEvent(self, event):
        boxes = self._chip_boxes()
        if boxes:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.translate(-self.horizontalScrollBar().value(), -self.verticalScrollBar().value())
            painter.setPen(Qt.PenStyle.NoPen)
            radius, inner_radius = KEY_CAP_RADIUS, max(0, KEY_CAP_RADIUS - KEY_CAP_BORDER)
            for rect, fill, border, bottom in boxes:
                # The bottom edge's colour, the border over all but that edge, the fill inside both.
                painter.setBrush(bottom)
                painter.drawRoundedRect(rect, radius, radius)
                painter.setBrush(border)
                painter.drawRoundedRect(rect.adjusted(0, 0, 0, -KEY_CAP_BOTTOM), radius, radius)
                painter.setBrush(fill)
                painter.drawRoundedRect(
                    rect.adjusted(KEY_CAP_BORDER, KEY_CAP_BORDER, -KEY_CAP_BORDER, -KEY_CAP_BOTTOM),
                    inner_radius, inner_radius,
                )
            painter.end()
        # The document draws its text over the key caps.
        super().paintEvent(event)


__all__ = [
    "CHIP_GLUE",
    "Code",
    "CodeList",
    "Field",
    "FieldTip",
    "LiveNote",
    "MISSING",
    "PanelHeightAtFirstLayout",
    "SpacedTipLabel",
    "TIP_PANEL_LINES",
    "TIP_PANEL_MINIMUM_LINES",
    "Tip",
    "TipModel",
    "UNBOUNDED",
    "TipNote",
    "TipOption",
    "change_signal",
    "chip",
    "free_bytes",
    "model_html",
    "nearest_existing_folder",
    "model_plain",
    "plain",
    "resolve",
    "rich",
    "status_html",
    "status_text_html",
    "status_mark",
    "swatch_mark",
    "tip_html",
    "tip_plain",
    "tooltip_text",
    "with_note",
    "wrappable",
]
