# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tip panel: the card under a window's form that describes the field in focus.

Config, VR Config and Tools show a field's description there, and each window's
own messages (a check's result, a failure) too. They share this module's label,
SpacedTipLabel, and the tip a field gives it.

A tip is either plain text, which the panel shows as it always has (escaped, a
line break where the text has one), or a Tip, which the panel lays out:

* a bold title, the field's own label (already translated, without its colon);
* a summary, then muted detail lines;
* a table of options, one row each: the option as the field's dropdown shows it,
  in a code chip, then what it does;
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

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QLabel, QWidget

from desktop.Studio_Theme import TOKENS, IconLabel, coloured_svg, icon_svg, status_color

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
    """text as a code chip: monospace on a lighter ground, kept on one line where it is short."""
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

LINE = "line-height: 120%;"


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


class SpacedTipLabel(QLabel):
    """Help-panel label with proportional multiline spacing, and the rich layouts above."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._showing_report = False
        self._entry = None
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setText(text)

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
        """Show plain text: escaped, a line break where the text has one."""
        self._mark_corner(None)
        self._showing_report = False
        self._entry = None
        self.setMinimumHeight(0)
        self.setAccessibleName("")
        escaped_text = html.escape(str(text)).replace("\r\n", "\n").replace("\r", "\n")
        escaped_text = escaped_text.replace("\n", "<br>")
        super().setText(f'<div style="line-height: 120%;">{escaped_text}</div>')

    def setTip(self, entry):
        """Show a tip: plain text as setText does, a Tip or FieldTip laid out as its model says."""
        if isinstance(entry, (Tip, FieldTip)):
            model = resolve(entry)
            self._show_rich(model_html(model), model_plain(model))
            self._mark_corner(None)
            self._entry = entry
        else:
            self.setText(entry)

    def refresh(self):
        """Lay the tip on display out again, so its live note shows what holds now.

        Returns False, and changes nothing, while the panel shows anything else
        (a status message, plain text), so a field changing never replaces one.
        """
        if self._entry is None:
            return False
        self.setTip(self._entry)
        return True

    def setStatus(self, kind, text):
        """Show a status message ("success", "warning", "danger" or "info"), marked by the corner icon.

        The corner icon becomes the status's icon, in its colour, until a tip
        or plain text replaces the message. A panel with no corner icon shows
        the icon beside the text instead.

        Like any status message, it is not the tip the panel last showed
        (the window's _tip_panel_tip_text), so hovering leaves it in place.
        """
        if self._mark_corner(kind):
            self._show_rich(status_text_html(text), text)
        else:
            self._show_rich(status_html(kind, text), text)

    def setReport(self, report_html, status=None):
        """Show a status report, rich text whose own text is already escaped.

        status is its overall verdict ("success", "warning" or "danger"), which
        the corner icon shows. The report's own rows keep their icons.

        Like any status message, it is not the tip the panel last showed
        (ConfigGUI._tip_panel_tip_text), so hovering leaves it in place.
        """
        self._mark_corner(status)
        self._show_rich(report_html, None)

    def _show_rich(self, rich_text, plain_text):
        self._entry = None
        if self._showing_report and rich_text == self.text():
            return  # already showing it
        self._showing_report = True
        self.setMinimumHeight(0)
        self.setAccessibleName(plain_text or "")
        super().setText(rich_text)
        self._fit_report()

    def _fit_report(self):
        """Keep the panel as tall as its rich text is at the panel's width.

        A wrapped label's minimum height is only an estimate (the text laid
        out at a width that suits its proportions), and the splitter around
        the panel ignores height-for-width, so a report whose rows wrap
        more than that was cut off at the bottom.
        """
        if self._showing_report and self.wordWrap() and self.width() > 0:
            # heightForWidth never goes below the minimum already set, so
            # clear it first, or the panel could only grow as it narrows.
            self.setMinimumHeight(0)
            self.setMinimumHeight(self.heightForWidth(self.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_report()


__all__ = [
    "Code",
    "CodeList",
    "Field",
    "FieldTip",
    "LiveNote",
    "MISSING",
    "SpacedTipLabel",
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
