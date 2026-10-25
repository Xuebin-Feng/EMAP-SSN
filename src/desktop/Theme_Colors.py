# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The colour maths of the Studio theme, with no Qt: the colour families, the shades each takes, and the OKLCH
conversion that turns a family's hue into them.

desktop/Studio_Theme.py builds the windows' tokens, palette and stylesheet from these (it imports every name
here and keeps them in its own namespace, so it stays the place the windows take them from), and the start-up
splash (desktop/Splash.py) takes the accent shades of the window it starts from here too. The splash is on
screen a third of a second after a click and imports no more of Qt than it draws with: Studio_Theme brings in
QtSvg and the icon engine, which cost it a few dozen milliseconds it does not need. So the colours live in a
module of their own, and there is one copy of them: nothing here is repeated in either importer.
"""

from __future__ import annotations

import math

# The colour families a window can take, by name, with their OKLCH hue, in the
# order the palette shows them. Purple keeps the windows' first accent,
# #5b5bd6, exactly. Black, with no hue, has no colour at all (BLACK_ACCENTS):
# a near-black primary button and switches, and grey sliders, selections and
# focus rings.
THEME_COLORS = {
    "purple": 278.3,
    "blue": 245.0,
    "teal": 190.0,
    "green": 150.0,
    "orange": 48.0,
    "red": 22.0,
    "pink": 350.0,
    "black": None,
}
DEFAULT_THEME_COLOR = "purple"

# The chroma of the theme colour itself. A hue that can't reach it at a
# shade's lightness in sRGB (teal, at the accent's) takes the most it can.
ACCENT_CHROMA = 0.1841

# The colours that are the same in every theme colour, by set: "light" for the
# windows, "light_page" for what the web pages add to them, "dark_page" for the
# pages' dark mode. The greys are Tailwind's zinc, the neutral of shadcn/ui.
NEUTRAL_SHADES = {
    "light": {
        # Surfaces
        "window": "#fafafa",
        "surface": "#ffffff",           # cards: the tab pane, reports
        "surface_muted": "#f4f4f5",     # hover fills
        "border": "#e4e4e7",            # card hairlines
        # Text (contrast on white)
        "text": "#09090b",
        "text_soft": "#3f3f46",         # report body, switch labels
        "text_muted": "#52525b",        # inactive tabs, hints, ghost buttons (7.7:1)
        "text_subtle": "#71717a",       # icons, placeholders, secondary status (4.8:1)
        "text_disabled": "#a1a1aa",
        # Fields
        "field": "#ffffff",
        "field_border": "#dcdce0",      # a touch stronger than a card's hairline
        "field_border_hover": "#d4d4d8",
        "field_disabled": "#fafafa",
        "primary_disabled": "#9b9ba0",  # a primary button greyed out
        # Switches that are off
        "switch_off": "#d4d4d8",
        "switch_off_hover": "#a1a1aa",
        # Tooltips
        "tooltip": "#18181b",
        "tooltip_text": "#fafafa",
        # Status texts
        "success": "#15803d",           # 5.0:1 on white
        "warning": "#b45309",           # 5.0:1 on white
        "danger": "#dc2626",            # 4.8:1 on white
        # Scroll bars
        "scroll": "#d4d4d8",
    },
    "light_page": {
        "inset": "#fafafa", "primary_disabled_text": "#fafafa", "code_text": "#18181b",
        "switch_thumb": "#ffffff", "scroll_hover": "#a1a1aa",
    },
    "dark_page": {
        "window": "#09090b", "surface": "#18181b", "surface_muted": "#27272a",
        "inset": "#111113", "border": "#27272a",
        "text": "#fafafa", "text_soft": "#d4d4d8", "text_muted": "#a1a1aa",
        "text_subtle": "#8e8e96", "text_disabled": "#52525b",
        "field_border": "#3f3f46", "field_border_hover": "#52525b",
        "primary_disabled": "#3f3f46", "primary_disabled_text": "#a1a1aa",
        "code_text": "#e4e4e7",
        "switch_off": "#3f3f46", "switch_off_hover": "#52525b", "switch_thumb": "#fafafa",
        "scroll": "#3f3f46", "scroll_hover": "#52525b",
    },
}

# The accents, which take the theme colour: (OKLCH lightness, chroma) in its
# hue, or a fixed colour; by set, as NEUTRAL_SHADES.
ACCENT_SHADES = {
    "light": {
        # The primary button: the theme colour, with white text
        "primary": (0.540, ACCENT_CHROMA),
        "primary_hover": (0.490, ACCENT_CHROMA),
        "primary_pressed": (0.440, ACCENT_CHROMA),
        "primary_text": "#ffffff",
        "primary_focus": (0.340, ACCENT_CHROMA),  # a focused primary button's ring
        # Focus rings, slider fills, selection
        "accent": (0.540, ACCENT_CHROMA),
        "accent_soft": (0.945, 0.022),
        # Switches that are on
        "switch_on": (0.540, ACCENT_CHROMA),
        "switch_on_hover": (0.490, ACCENT_CHROMA),
        # Light tints: the segmented tab bar's track, and the hint card and muted
        # buttons (tint), darker under the mouse (tint_hover)
        "track": (0.953, 0.016),
        "tint": (0.967, 0.012),
        "tint_hover": (0.920, 0.020),
        # The statistics report's histogram bars, behind its text
        "histogram": (0.925, 0.040),
        # The 1 px lines between a card's sections (the "separator" role), as
        # light as a field's hovered border
        "separator": (0.870, 0.070),
        # Text in the colour on white, as dark as the pressed primary button
        # (7:1 or more): the Viewer's count of hidden nodes, when there are any
        "accent_text": (0.440, ACCENT_CHROMA),
    },
    "light_page": {
        "badge_text": (0.487, ACCENT_CHROMA),
        "badge_border": (0.850, 0.060),
        "switch_thumb_on": "#ffffff",
    },
    "dark_page": {
        "primary": (0.680, ACCENT_CHROMA),
        "primary_hover": (0.740, ACCENT_CHROMA),
        "primary_pressed": (0.800, ACCENT_CHROMA),
        "primary_text": "#18181b",
        "accent": (0.680, ACCENT_CHROMA),
        "accent_soft": (0.290, 0.070),
        "badge_text": (0.780, 0.120),
        "switch_on": (0.680, ACCENT_CHROMA),
        "switch_on_hover": (0.740, ACCENT_CHROMA),
        "switch_thumb_on": "#fafafa",
    },
}

# Black's accents, all zinc: the near-black buttons and switches of the look
# before the theme colours, with grey in place of its indigo.
BLACK_ACCENTS = {
    "light": {
        "primary": "#18181b", "primary_hover": "#27272a", "primary_pressed": "#3f3f46",
        "primary_text": "#fafafa", "primary_focus": "#71717a",
        "accent": "#52525b", "accent_soft": "#e4e4e7",
        "switch_on": "#18181b", "switch_on_hover": "#27272a",
        "track": "#efeff1", "tint": "#f4f4f5", "tint_hover": "#e4e4e7", "histogram": "#e4e4e7",
        "separator": "#d4d4d8", "accent_text": "#52525b",
    },
    "light_page": {"badge_text": "#3f3f46", "badge_border": "#d4d4d8", "switch_thumb_on": "#ffffff"},
    "dark_page": {
        "primary": "#fafafa", "primary_hover": "#e4e4e7", "primary_pressed": "#d4d4d8",
        "primary_text": "#18181b", "accent": "#a1a1aa", "accent_soft": "#3f3f46",
        "badge_text": "#d4d4d8", "switch_on": "#fafafa", "switch_on_hover": "#e4e4e7",
        "switch_thumb_on": "#18181b",
    },
}


def _oklch_to_linear_srgb(lightness, chroma, hue):
    a = chroma * math.cos(math.radians(hue))
    b = chroma * math.sin(math.radians(hue))
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def _in_srgb(channels):
    return all(-1e-6 <= channel <= 1 + 1e-6 for channel in channels)


def _encode_srgb(channel):
    channel = min(1.0, max(0.0, channel))
    return 12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055


def oklch_hex(lightness, chroma, hue):
    """The #rrggbb of an OKLCH colour (Björn Ottosson's Oklab, in polar form).

    Where sRGB can't show that chroma at that lightness and hue, the chroma
    drops to the most it can show, so the lightness, and the contrast with
    it, stays as asked.
    """
    channels = _oklch_to_linear_srgb(lightness, chroma, hue)
    if not _in_srgb(channels):
        low, high = 0.0, chroma
        for _ in range(32):
            middle = (low + high) / 2
            if _in_srgb(_oklch_to_linear_srgb(lightness, middle, hue)):
                low = middle
            else:
                high = middle
        channels = _oklch_to_linear_srgb(lightness, low, hue)
    return "#" + "".join(f"{round(_encode_srgb(channel) * 255):02x}" for channel in channels)


def shade_set(kind, color):
    """One set of colours ("light", "light_page" or "dark_page") in ``color``: the greys and its accents.

    ``color`` is a THEME_COLORS name; any other raises ValueError.
    """
    if color not in THEME_COLORS:
        raise ValueError(
            f"There is no theme colour named {color!r}; the colours are {', '.join(THEME_COLORS)}."
        )
    hue = THEME_COLORS[color]
    if hue is None:
        accents = BLACK_ACCENTS[kind]
    else:
        accents = {
            name: shade if isinstance(shade, str) else oklch_hex(*shade, hue)
            for name, shade in ACCENT_SHADES[kind].items()
        }
    return {**NEUTRAL_SHADES[kind], **accents}
