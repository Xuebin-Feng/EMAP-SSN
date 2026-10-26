# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Terminal colours: when to print them, how to read them, and how to take them out.

Commands colour part of their output with ANSI escape sequences (``cluster``'s dots,
``spectrum``'s range, the red errors and amber warnings of the alignment and embedding
managers). Two places show those colours: a terminal, and the Console window
(desktop/Console_View.py), which reads the sequences itself. Everywhere else they are
noise, so:

* ``colours_enabled`` is the one question a command asks before it colours: yes for a
  real terminal, and yes for the output the Console captures (utilities/Console_Capture.py
  hands that to the Console window, and passes it on to the stream the process started with
  only as the stream can take it). A plain redirect, a pipe or a buffer says no.
* ``print_coloured`` prints a message in red (an error) or amber (a warning) if it says yes.
* ``strip_ansi`` and ``AnsiStripper`` (for bytes arriving in chunks) take the sequences out,
  for the log file, the splash's details, and a stream that is not a terminal.
* ``parse_ansi`` reads the colour sequences (SGR: reset, bold, italic, underline, reverse,
  the 16 colours, the 256-colour palette and 24-bit colour) into runs of text with a
  ``Style``, and every other sequence, such as a cursor movement, is dropped.
  ``readable_color`` turns a style's colour into one the window's background shows.

Imports only the standard library.
"""

from __future__ import annotations

import colorsys
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

ESC = "\x1b"

# One escape sequence: a CSI (ESC [ parameters intermediates final), a CSI that the text cut short, an
# OSC (ESC ] ... BEL or ESC \), a character-set choice, another escape of one or two characters (ESC c,
# ESC 7, ESC SP F), or an ESC on its own. A CSI ends at the first byte that is not a parameter or an
# intermediate, as in a terminal: a sequence with a letter among its numbers ends at the letter.
_SEQUENCE = re.compile(
    r"\x1b(?:\[(?P<params>[0-?]*)(?P<inter>[ -/]*)(?P<final>[@-~])"
    r"|\[[0-?]*[ -/]*"
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|[()*+][0-~]?"
    r"|[ -/]+[0-~]?"
    r"|[0-~]?)"
)
_SEQUENCE_BYTES = re.compile(_SEQUENCE.pattern.encode("ascii"))
# The end of a chunk that may be the start of a sequence the next chunk finishes.
_UNFINISHED_BYTES = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*|\][^\x07\x1b]*\x1b?|[()*+]|[ -/]+)?\Z")
# A sequence this long is not one: it is passed through the stripper rather than waited for.
MAX_UNFINISHED_BYTES = 1024


# =====================================================================
# Whether to colour
# =====================================================================

def colours_enabled(stream=None) -> bool:
    """Whether output on ``stream`` (standard output by default) may carry colour sequences.

    True for a terminal, and for the stream the Console capture installed: there the Console
    window shows the colours, and the capture passes the output on without them to whatever
    is not a terminal. False for a pipe, a file, a buffer a caller swapped in (the sequences
    would end up in its text) and no stream at all.
    """
    stream = sys.stdout if stream is None else stream
    if stream is None:
        return False
    try:
        if stream.isatty():
            return True
    except (AttributeError, ValueError, OSError):
        pass
    from utilities import Console_Capture

    return Console_Capture.is_capture_stream(stream)


# The colours of the messages a command prints, as the Console's help says: errors are red and warnings
# amber. Red is the bright one, SGR 91. Amber is the palette's yellow, SGR 33, which the Console draws as
# it draws its own warnings (a `[Qt warning]` line): on its dark page it is that very shade, #fbbf24.
RED = "91"
AMBER = "33"


def print_coloured(message: str, colour: str) -> None:
    """Print ``message`` in ``colour`` (RED or AMBER) where colours_enabled(), and as plain text everywhere else.

    The Console shows the colour; the log file and any stream that is not a terminal get the text
    without it (Console_Capture strips the sequences there).
    """
    if colours_enabled():
        print(f"{ESC}[{colour}m{message}{ESC}[0m")
    else:
        print(message)


# =====================================================================
# Taking colours out
# =====================================================================

def strip_ansi(text: str) -> str:
    """``text`` without its escape sequences."""
    if ESC not in text:
        return text
    return _SEQUENCE.sub("", text)


class AnsiStripper:
    """strip_ansi for bytes that arrive in chunks: a sequence cut between two chunks is still taken out whole.

    The end of a chunk that could be the start of a sequence waits for the next chunk (or is dropped
    by ``finish``); the rest is passed on as it comes.
    """

    def __init__(self):
        self._held = b""

    def feed(self, data: bytes) -> bytes:
        data = self._held + data
        self._held = b""
        last = data.rfind(b"\x1b")
        if last < 0:
            return data
        if len(data) - last <= MAX_UNFINISHED_BYTES and _UNFINISHED_BYTES.match(data, last):
            self._held = data[last:]
            data = data[:last]
        return _SEQUENCE_BYTES.sub(b"", data)

    def finish(self) -> bytes:
        """What is left at the end of the stream: nothing, as an unfinished sequence is dropped."""
        self._held = b""
        return b""


# =====================================================================
# Reading colours
# =====================================================================

# A colour is ("index", n) for one of the 16 palette entries, or ("rgb", r, g, b).
Color = tuple


@dataclass(frozen=True)
class Style:
    """What a run of text looks like: colours (None for the window's own), and the text's weight and decoration."""

    fg: Color | None = None
    bg: Color | None = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    reverse: bool = False

    @property
    def has_color(self) -> bool:
        return self.fg is not None or self.bg is not None


DEFAULT_STYLE = Style()


def _channel(text):
    """A colour channel 0 to 255, or None."""
    try:
        value = int(text)
    except ValueError:
        return None
    return value if 0 <= value <= 255 else None


def _xterm_256(number: int) -> Color:
    """Entry ``number`` of the 256-colour palette: the 16 first colours, a 6x6x6 cube, then 24 greys."""
    if number < 16:
        return ("index", number)
    if number < 232:
        number -= 16
        steps = (0, 95, 135, 175, 215, 255)
        return ("rgb", steps[number // 36], steps[number // 6 % 6], steps[number % 6])
    grey = 8 + 10 * (number - 232)
    return ("rgb", grey, grey, grey)


def _extended_color(groups, position):
    """(colour or None, groups used) for the 38 or 48 at ``groups[position]``: ;5;n, ;2;r;g;b, or the colon forms.

    A specification that is cut short or out of range gives None, and takes the groups that belong to it
    with it, so that its numbers are not read as codes of their own.
    """
    group = groups[position]
    if ":" in group:  # 38:5:n, 38:2:r:g:b, 38:2::r:g:b (a colour space in front)
        parts = group.split(":")[1:]
        if parts and parts[0] == "5" and len(parts) == 2:
            number = _channel(parts[1])
            return (_xterm_256(number) if number is not None else None), 1
        if parts and parts[0] == "2" and len(parts) in (4, 5):
            channels = [_channel(part) for part in parts[-3:]]
            return (("rgb", *channels) if None not in channels else None), 1
        return None, 1
    kind = groups[position + 1] if position + 1 < len(groups) else None
    if kind == "5":
        number = _channel(groups[position + 2]) if position + 2 < len(groups) else None
        return (_xterm_256(number) if number is not None else None), min(3, len(groups) - position)
    if kind == "2":
        channels = [_channel(item) for item in groups[position + 2:position + 5]]
        used = min(5, len(groups) - position)
        return (("rgb", *channels) if len(channels) == 3 and None not in channels else None), used
    return None, min(2, len(groups) - position)


def apply_sgr(style: Style, parameters: str) -> Style:
    """``style`` after the SGR sequence whose parameters are ``parameters`` (what stands between ``ESC [`` and ``m``).

    Codes it does not know are ignored; an empty list is a reset.
    """
    groups = parameters.split(";") if parameters else [""]
    position = 0
    while position < len(groups):
        group = groups[position]
        code_text = group.split(":")[0]
        if code_text == "":
            code = 0
        else:
            try:
                code = int(code_text)
            except ValueError:
                position += 1
                continue
        used = 1
        if code == 0:
            style = DEFAULT_STYLE
        elif code == 1:
            style = _replace(style, bold=True)
        elif code == 22:
            style = _replace(style, bold=False)
        elif code == 3:
            style = _replace(style, italic=True)
        elif code == 23:
            style = _replace(style, italic=False)
        elif code == 4:
            style = _replace(style, underline=True)
        elif code == 24:
            style = _replace(style, underline=False)
        elif code == 7:
            style = _replace(style, reverse=True)
        elif code == 27:
            style = _replace(style, reverse=False)
        elif 30 <= code <= 37:
            style = _replace(style, fg=("index", code - 30))
        elif 90 <= code <= 97:
            style = _replace(style, fg=("index", code - 90 + 8))
        elif code == 39:
            style = _replace(style, fg=None)
        elif 40 <= code <= 47:
            style = _replace(style, bg=("index", code - 40))
        elif 100 <= code <= 107:
            style = _replace(style, bg=("index", code - 100 + 8))
        elif code == 49:
            style = _replace(style, bg=None)
        elif code in (38, 48):
            color, used = _extended_color(groups, position)
            if color is not None:
                style = _replace(style, **{"fg" if code == 38 else "bg": color})
        position += max(1, used)
    return style


def _replace(style: Style, **changes) -> Style:
    values = {name: getattr(style, name) for name in ("fg", "bg", "bold", "italic", "underline", "reverse")}
    values.update(changes)
    return Style(**values)


def parse_ansi(text: str, style: Style = DEFAULT_STYLE):
    """(runs, style at the end): ``text`` as [(text, Style), ...] with every escape sequence taken out.

    ``style`` is what the text starts in, as a line that follows one that left its colour on. Colour
    sequences change the style, and every other sequence (cursor movement, a window title, a sequence
    cut short at the end of the text) is dropped without a trace. Runs are never empty, and an empty
    text has no runs.
    """
    if ESC not in text:
        return ([(text, style)] if text else []), style
    runs = []
    position = 0
    for match in _SEQUENCE.finditer(text):
        if match.start() > position:
            runs.append((text[position:match.start()], style))
        position = match.end()
        if match.group("final") == "m" and not match.group("inter"):
            style = apply_sgr(style, match.group("params"))
    if position < len(text):
        runs.append((text[position:], style))
    return runs, style


# =====================================================================
# Colours a window can show
# =====================================================================

# The 16 colours, tuned for a light and for a dark page: the dark ones are lighter, white is a grey on
# the light page, and black a light grey on the dark one, so that nothing disappears into its background.
PALETTE_LIGHT = (
    "#3f3f46", "#dc2626", "#15803d", "#a16207", "#2563eb", "#a21caf", "#0e7490", "#71717a",
    "#52525b", "#ef4444", "#16a34a", "#ca8a04", "#3b82f6", "#c026d3", "#0891b2", "#a1a1aa",
)
PALETTE_DARK = (
    "#71717a", "#f87171", "#4ade80", "#fbbf24", "#60a5fa", "#e879f9", "#22d3ee", "#e4e4e7",
    "#a1a1aa", "#fca5a5", "#86efac", "#fde047", "#93c5fd", "#f0abfc", "#67e8f9", "#fafafa",
)
# The 16 colours as terminals draw them (the xterm-like set VS Code uses): for text that comes with a
# background of its own, where command and terminal chose both colours to go together.
PALETTE_STANDARD = (
    "#000000", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5",
    "#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#ffffff",
)
# The least contrast (WCAG ratio) a coloured mark keeps against what it is drawn on: 3 is the level for
# large text and graphics, which fits the dots and ranges commands colour.
MIN_CONTRAST = 3.0


def _rgb(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(round(channel)))) for channel in rgb))


def _luminance(rgb) -> float:
    linear = [channel / 255 for channel in rgb]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in linear]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    """The WCAG contrast ratio between two "#rrggbb" colours, 1 to 21."""
    one, two = _luminance(_rgb(first)), _luminance(_rgb(second))
    light, dark = max(one, two), min(one, two)
    return (light + 0.05) / (dark + 0.05)


@lru_cache(maxsize=4096)
def readable_color(color: str, background: str, minimum: float = MIN_CONTRAST) -> str:
    """``color`` ("#rrggbb"), moved lighter or darker, with its hue kept, until it shows on ``background``.

    A colour that already has ``minimum`` contrast is returned as it is. Going away from the background,
    which is darker on a light page and lighter on a dark one, always gets there at the latest at black or
    white (the 18 % luminance that tells the two apart is where either has 4:1).
    """
    if contrast_ratio(color, background) >= minimum:
        return color
    hue, lightness, saturation = colorsys.rgb_to_hls(*(channel / 255 for channel in _rgb(color)))
    direction = -1 if _luminance(_rgb(background)) > 0.18 else 1
    candidate = color
    for step in range(1, 51):
        candidate = _hex(tuple(255 * channel for channel in colorsys.hls_to_rgb(
            hue, min(1.0, max(0.0, lightness + direction * step * 0.02)), saturation)))
        if contrast_ratio(candidate, background) >= minimum:
            break
    return candidate


def palette_color(color: Color, *, dark: bool | None) -> str:
    """A style's colour as "#rrggbb", as it is.

    A palette entry is the page's own shade of that colour (light or dark), or, with ``dark`` None, the
    standard terminal shade.
    """
    if color[0] == "index":
        return (PALETTE_STANDARD if dark is None else PALETTE_DARK if dark else PALETTE_LIGHT)[color[1]]
    return _hex(color[1:])


def resolve_color(color: Color, *, dark: bool, background: str) -> str:
    """A style's colour as "#rrggbb" that shows on ``background`` (a page: palette entries in its own shades)."""
    return readable_color(palette_color(color, dark=dark), background)
