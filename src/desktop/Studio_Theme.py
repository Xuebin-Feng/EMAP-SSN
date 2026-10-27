# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Studio: the shared look of the EMAP-SSN desktop windows.

Config, VR Config, Tools and the Viewer share one look, after shadcn/ui,
Linear and Vercel: zinc greys, hairline borders, 8 px controls on 12 px
cards, and one theme colour for the primary button, switches that are on,
focus rings, slider fills and selection.

* The theme colour is a colour family (THEME_COLORS), not a shade, and only
  the accents take it (ACCENT_SHADES): the primary button, switches that are
  on, focus rings, slider fills and selection, a light shade of it on the
  lines between a card's sections, a light tint of it on the tab bar,
  the hint card and muted buttons, and a dark shade of it on the Viewer's
  count of hidden nodes. Each has a fixed OKLCH lightness
  and chroma, and the family gives the hue, so no family changes the contrast.
  Everything else, from borders and icons to greyed-out fields, keeps the same
  greys in every colour (NEUTRAL_SHADES). Black has no colour at all
  (BLACK_ACCENTS). Each window keeps its own colour (Desktop_App's theme
  colour option); the Viewer takes Config's when it starts. Tools' help pane
  alone tints its greys in the colour too (tinted_tokens), all but its icons.
* TOKENS names every colour and radius of the theme colour set in place
  (set_theme_color). studio_stylesheet() turns them into the application
  stylesheet, and studio_palette() into the palette, so the parts Qt draws
  natively (popups, selection, disabled text) match. web_theme_css() gives
  the Viewer's Agent and Meta Data pages the same family, light and dark.
* A widget opts into a variant through its "role" property (set_role). The
  stylesheet styles QPushButton[role="primary"] and so on, so no window sets
  colours on its widgets. The roles are listed in ROLES.
* Icons are Lucide line icons (src/resources/icons/lucide, ISC licence; see
  src/resources/icons/README.md). Qt draws an SVG's currentColor as black, so
  StudioIconEngine writes the theme colour into the SVG text and renders it at
  the screen's pixel ratio. The stylesheet's combo and spin-box arrows use the
  pre-coloured copies in src/resources/icons/qss (QSS_ICON_VARIANTS).
* Body text keeps the application font at its size: the layouts size labels,
  toggles and buttons from it. Only weights 400 and 600 are used, because the
  bundled language fonts (Noto Sans SC for Chinese) have no Medium face and would
  draw 500 as bold. The theme changes no font: Qt's setStyle gives some widget
  classes the platform's font again, and apply_studio_theme gives them back
  the fonts they had (WIDGET_FONT_CLASSES).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject, QPoint, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QIconEngine, QPainter, QPalette, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
import shiboken6
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
from desktop.Theme_Colors import (
    ACCENT_CHROMA,
    ACCENT_SHADES,
    BLACK_ACCENTS,
    DEFAULT_THEME_COLOR,
    NEUTRAL_SHADES,
    THEME_COLORS,
    oklch_hex,
    scroll_bar_rules,
    shade_set,
)

ICONS_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"
LUCIDE_DIR = ICONS_DIR / "lucide"
QSS_ICON_DIR = ICONS_DIR / "qss"

# =====================================================================
# Theme Colours
# =====================================================================

# The colour families (THEME_COLORS), the shades each takes (NEUTRAL_SHADES, ACCENT_SHADES, BLACK_ACCENTS) and the
# OKLCH maths that draws them (oklch_hex, shade_set) are in desktop/Theme_Colors.py, with no Qt, so that the
# start-up splash takes them without this module's QtSvg. They are imported above, and every name stays here.

# The greys of Tools' help pane (tinted_tokens): (OKLCH lightness, chroma) in
# the theme colour's hue, at the lightness of the zinc grey each replaces.
TINTED_SHADES = {
    "surface_muted": (0.967, 0.012),
    "border": (0.920, 0.020),
    "text": (0.141, 0.025),
    "text_soft": (0.370, 0.040),
    "text_muted": (0.442, 0.045),
    "text_subtle": (0.552, 0.045),
    "text_disabled": (0.712, 0.035),
    "scroll": (0.871, 0.028),
}

GEOMETRY = {
    "radius": 8,                 # controls
    "radius_card": 12,           # cards
    "card_gap": 8,               # between a tab bar and its card, and between cards
    "separator_thickness": 2,    # the lines between a card's sections
}


@lru_cache(maxsize=None)
def _tokens(color):
    return {**shade_set("light", color), **GEOMETRY}


def theme_tokens(color=DEFAULT_THEME_COLOR):
    """The windows' tokens (as TOKENS holds them) in the theme colour named ``color``."""
    return dict(_tokens(color))


def page_tokens(color=DEFAULT_THEME_COLOR, *, dark=False):
    """The web pages' colours in ``color``: the windows' and a few more, or the dark mode's."""
    if dark:
        return shade_set("dark_page", color)
    return {**shade_set("light", color), **shade_set("light_page", color)}


TOKENS = theme_tokens()
_current_theme_color = DEFAULT_THEME_COLOR


def theme_color():
    """The name of the theme colour TOKENS holds."""
    return _current_theme_color


def set_theme_color(color):
    """Put the theme colour named ``color`` (a THEME_COLORS name) in TOKENS.

    TOKENS changes in place, so every module that imported it sees the new
    colours. What was built before keeps its colours: apply_studio_theme
    restyles the application, and a window redraws itself for the rest.
    """
    global _current_theme_color
    tokens = theme_tokens(color)
    TOKENS.clear()
    TOKENS.update(tokens)
    _current_theme_color = color


def tinted_tokens(color=None):
    """TOKENS in ``color`` (TOKENS' own by default), their greys tinted in its hue (TINTED_SHADES).

    Tools' help pane takes these. Black has no hue, so its greys stay grey.
    """
    color = color or theme_color()
    tokens = theme_tokens(color)
    hue = THEME_COLORS[color]
    if hue is not None:
        tokens.update({name: oklch_hex(*shade, hue) for name, shade in TINTED_SHADES.items()})
    return tokens

# The variants a widget can ask for with set_role(widget, role).
ROLES = {
    "QPushButton": {
        "primary": "the window's main action, in the theme colour (at most one per window)",
        "muted": "a secondary action on a light tint of the theme colour",
        "icon": "a square button showing only an icon",
        "compact": "a short text button beside a field, such as Pick or Clear",
    },
    "QLabel": {
        "hint": "the help card under the tabs; add_hint_icon puts an info icon in it",
        "title": "a panel heading",
        "subtle": "secondary text, such as a status line",
    },
    "QFrame": {
        "separator": "a line between sections, separator_thickness thick, in a light shade of the theme colour",
    },
    "QTextEdit": {
        "report": "a read-only report card",
        "hint": "the help card when it scrolls (the tip panel); add_hint_icon puts an info icon in it",
    },
}

# Copies of Lucide icons coloured for the stylesheet: (icon, token), made for
# every theme colour.
QSS_ICON_VARIANTS = (
    ("chevron-down", "text_subtle"),
    ("chevron-up", "text_subtle"),
    ("chevron-left", "text_subtle"),
    ("chevron-right", "text_subtle"),
    ("chevron-down", "text_disabled"),
    ("chevron-up", "text_disabled"),
    ("chevron-left", "text_disabled"),
    ("chevron-right", "text_disabled"),
)


def qss_icon_path(name, token, tokens=None):
    """The pre-coloured copy of a Lucide icon that the stylesheet uses.

    tokens gives the colours, TOKENS by default.
    """
    return QSS_ICON_DIR / f"{name}-{(TOKENS if tokens is None else tokens)[token].lstrip('#')}.svg"


def coloured_svg(svg_text, color):
    """An SVG whose currentColor strokes and fills are drawn in ``color``."""
    return svg_text.replace("currentColor", color)


def _qss_url(name, token, tokens):
    # Quoted: the program may live in a folder whose name has spaces.
    return f'url("{qss_icon_path(name, token, tokens).as_posix()}")'


# =====================================================================
# Stylesheet & Palette
# =====================================================================

def studio_stylesheet(t=None):
    """The application stylesheet built from the tokens, TOKENS by default."""
    t = TOKENS if t is None else t
    down = _qss_url("chevron-down", "text_subtle", t)
    up = _qss_url("chevron-up", "text_subtle", t)
    down_disabled = _qss_url("chevron-down", "text_disabled", t)
    up_disabled = _qss_url("chevron-up", "text_disabled", t)
    left = _qss_url("chevron-left", "text_subtle", t)
    right = _qss_url("chevron-right", "text_subtle", t)
    left_disabled = _qss_url("chevron-left", "text_disabled", t)
    right_disabled = _qss_url("chevron-right", "text_disabled", t)
    return f"""
QMainWindow, QDialog {{ background: {t['window']}; }}
QWidget {{ color: {t['text']}; }}
QLabel {{ background: transparent; }}
QLabel:disabled {{ color: {t['text_disabled']}; }}
QToolTip {{
    background: {t['tooltip']}; color: {t['tooltip_text']}; border: none;
    padding: 5px 8px; border-radius: 6px;
}}

/* Tabs: a segmented control, the selected tab a raised white pill in a grey track.
   prepare_tab_widget() lets the bar paint its track. */
QTabWidget::tab-bar {{ left: 0px; }}
QTabBar {{ background: {t['track']}; border-radius: 10px; }}
QTabBar::tab {{
    background: transparent; color: {t['text_muted']};
    border: 1px solid transparent; border-radius: 7px;
    padding: 4px 12px; margin: 3px 0px;
}}
QTabBar::tab:first {{ margin-left: 3px; }}
QTabBar::tab:last {{ margin-right: 3px; }}
QTabBar::tab:hover:!selected {{ color: {t['text']}; }}
QTabBar::tab:selected {{
    background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border']}; border-bottom-color: {t['field_border_hover']};
}}
/* The arrows a bar shows when its tabs don't fit. */
QTabBar QToolButton {{
    background: {t['track']}; border: none; border-radius: 6px; margin: 3px 1px;
}}
QTabBar QToolButton:hover {{ background: {t['border']}; }}
QTabBar QToolButton::left-arrow {{ image: {left}; width: 12px; height: 12px; }}
QTabBar QToolButton::right-arrow {{ image: {right}; width: 12px; height: 12px; }}
QTabBar QToolButton::left-arrow:disabled {{ image: {left_disabled}; }}
QTabBar QToolButton::right-arrow:disabled {{ image: {right_disabled}; }}
QTabWidget::pane {{
    border: none; background: transparent; padding: 0px;
    margin: {t['card_gap']}px 0px 0px 0px;
}}
QTabWidget > QStackedWidget {{
    background: {t['surface']}; border: 1px solid {t['border']};
    border-radius: {t['radius_card']}px;
}}

/* Scroll areas let the card's colour show through. */
QScrollArea, QScrollArea > QWidget#qt_scrollarea_viewport, QScrollArea > QWidget > QWidget {{
    background: transparent; border: none;
}}
{scroll_bar_rules(t['scroll'], t['text_disabled'])}

QSplitter::handle {{ background: transparent; }}

/* Fields. Focus draws a 2 px accent ring; the padding shrinks so the text stays put. */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {t['field']}; color: {t['text']};
    border: 1px solid {t['field_border']}; border-radius: {t['radius']}px;
    padding: 3px 8px; min-height: 20px;
    selection-background-color: {t['accent_soft']}; selection-color: {t['text']};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {t['field_border_hover']};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {t['accent']}; padding: 2px 7px;
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {t['field_disabled']}; color: {t['text_disabled']}; border-color: {t['border']};
}}
QLineEdit[readOnly="true"] {{ background: {t['field_disabled']}; }}
QComboBox {{ padding-right: 24px; }}
QComboBox:focus {{ padding-right: 23px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none;
}}
QComboBox::down-arrow {{ image: {down}; width: 12px; height: 12px; }}
QComboBox::down-arrow:disabled {{ image: {down_disabled}; }}
/* A combo opens its list below itself, every entry in view up to its
   maxVisibleItems. Fusion's own popup, which puts the current entry over the
   combo, sized itself without the list's padding and hid half a row behind a
   scroll arrow. The list's window is square, so the list is too. */
QComboBox {{ combobox-popup: 0; }}
QComboBox QAbstractItemView {{
    background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 0px;
    padding: 4px; outline: 0;
    selection-background-color: {t['accent_soft']}; selection-color: {t['text']};
}}
QComboBox QAbstractItemView::item {{
    min-height: 24px; padding: 0px 6px; border: none; border-radius: 6px;
}}
QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected {{
    background: {t['accent_soft']}; color: {t['text']};
}}
QComboBox QAbstractItemView::item:disabled {{ color: {t['text_disabled']}; }}
/* The edit field already stops at the arrow buttons. Fusion adds about 3 px to a
   spin box's height; a QSS height sizes the content box, so 20 px of content plus
   padding and border keep it at the 28 px of the other fields. */
QSpinBox, QDoubleSpinBox {{ padding-right: 0px; max-height: 20px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; width: 16px; border: none; background: transparent;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; margin: 3px 3px 0px 0px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0px 3px 3px 0px; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {t['surface_muted']}; border-radius: 4px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: {up}; width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: {down}; width: 10px; height: 10px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ image: {up_disabled}; }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: {down_disabled}; }}

/* Sliders: a thin zinc track, an accent fill and a white thumb. A styled slider
   draws no tick marks; its height leaves the 16 px thumb room. */
QSlider:horizontal {{ min-height: 18px; }}
QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {t['surface']}; border: 1px solid {t['accent']};
    width: 14px; height: 14px; margin: -6px 0px; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {t['accent_soft']}; }}
QSlider::handle:horizontal:focus {{ border: 2px solid {t['accent']}; width: 12px; height: 12px; }}
QSlider::sub-page:horizontal:disabled {{ background: {t['text_disabled']}; }}
QSlider::handle:horizontal:disabled {{ border-color: {t['text_disabled']}; }}

/* Buttons: an outline button unless a role says otherwise. 13 px of padding and
   the 1 px border leave Desktop_App.BUTTON_TEXT_PADDING (14 px) between a fitted
   button's text and each end; focus trades a pixel of padding for its 2 px ring. */
QPushButton {{
    background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['field_border']}; border-radius: {t['radius']}px;
    padding: 3px 13px; min-height: 20px;
}}
QPushButton:hover {{ background: {t['surface_muted']}; }}
QPushButton:pressed {{ background: {t['border']}; }}
QPushButton:focus {{ border: 2px solid {t['accent']}; padding: 2px 12px; }}
QPushButton:disabled {{
    color: {t['text_disabled']}; background: {t['field_disabled']}; border-color: {t['border']};
}}
QPushButton[role="primary"] {{
    background: {t['primary']}; color: {t['primary_text']};
    border: 1px solid {t['primary']}; font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: {t['primary_hover']}; border-color: {t['primary_hover']}; }}
QPushButton[role="primary"]:pressed {{ background: {t['primary_pressed']}; }}
QPushButton[role="primary"]:focus {{ border: 2px solid {t['primary_focus']}; padding: 2px 12px; }}
QPushButton[role="primary"]:disabled {{
    background: {t['primary_disabled']}; border-color: {t['primary_disabled']}; color: {t['primary_text']};
}}
QPushButton[role="muted"] {{
    background: {t['tint']}; color: {t['text']}; border: 1px solid {t['tint']};
}}
QPushButton[role="muted"]:hover {{ background: {t['tint_hover']}; border-color: {t['tint_hover']}; }}
QPushButton[role="muted"]:focus {{ border: 2px solid {t['accent']}; padding: 2px 12px; }}
QPushButton[role="muted"]:disabled {{
    background: {t['field_disabled']}; border-color: {t['border']}; color: {t['text_disabled']};
}}
QPushButton[role="icon"] {{ padding: 3px; }}
QPushButton[role="icon"]:focus {{ padding: 2px; }}
QPushButton[role="compact"] {{ padding: 3px 9px; }}
QPushButton[role="compact"]:focus {{ padding: 2px 8px; }}

/* Toolbars (the tools' chart windows): a strip in the window's grey over a
   hairline, icon buttons filled under the mouse and on a light tint of the
   theme colour while checked. Only a toolbar's buttons: a tab bar's arrows
   keep their own rules. */
QToolBar {{
    background: {t['window']}; border: none; border-bottom: 1px solid {t['border']};
    padding: 4px 8px; spacing: 2px;
}}
QToolBar::separator {{ background: {t['border']}; width: 1px; margin: 6px 6px; }}
QToolBar QToolButton {{
    background: transparent; border: 1px solid transparent;
    border-radius: {t['radius']}px; padding: 4px;
}}
QToolBar QToolButton:hover {{ background: {t['surface_muted']}; }}
QToolBar QToolButton:pressed {{ background: {t['border']}; }}
QToolBar QToolButton:checked {{ background: {t['accent_soft']}; border-color: {t['accent_soft']}; }}
QToolBar QLabel {{ color: {t['text_muted']}; padding: 0px 4px; }}

/* Panels */
QFrame[role="separator"] {{
    background: {t['separator']}; border: none; max-height: {t['separator_thickness']}px;
}}
QLabel[role="hint"] {{
    background: {t['tint']}; color: {t['text_muted']};
    border: {HINT_BORDER}px solid {t['tint']}; border-radius: {t['radius_card']}px;
    padding: {HINT_PADDING}px {HINT_RIGHT_PADDING}px {HINT_PADDING}px {HINT_TEXT_INDENT}px;
}}
/* The tip panel scrolls: its scroll bar sits in the card's right padding, and the
   viewport's own margin keeps the text off it. The text keeps HINT_TEXT_MARGIN from the
   viewport's edges (the document's margin), so glyphs are not cut by the clip, and the
   padding is that much less. */
QTextEdit[role="hint"] {{
    background: {t['tint']}; color: {t['text_muted']};
    border: {HINT_BORDER}px solid {t['tint']}; border-radius: {t['radius_card']}px;
    padding: {HINT_PADDING - HINT_TEXT_MARGIN}px {HINT_SCROLL_EDGE}px {HINT_PADDING - HINT_TEXT_MARGIN}px {HINT_TEXT_INDENT - HINT_TEXT_MARGIN}px;
}}
QLabel[role="title"] {{ font-weight: 600; font-size: 14px; padding: 0px 2px; }}
QLabel[role="subtle"] {{ color: {t['text_subtle']}; }}
QTextEdit[role="report"] {{
    background: {t['surface']}; color: {t['text_soft']};
    border: 1px solid {t['border']}; border-radius: {t['radius_card']}px; padding: 8px 6px 8px 10px;
}}
"""


# Where the hint card's text starts: its info icon sits in the padding.
HINT_ICON_SIZE = 16
HINT_ICON_OFFSET = 15
HINT_TEXT_INDENT = 40
HINT_BORDER = 1
HINT_PADDING = 11            # above and below the text
HINT_RIGHT_PADDING = 14      # between the text and the card's right edge
HINT_SCROLL_EDGE = 4         # the scrolling card's padding on the right; its scroll bar lies inside it
HINT_TEXT_MARGIN = 3         # the scrolling card's text keeps this far from its viewport's edges

# The key cap: a small white box with a thin border, a thicker bottom edge and round corners,
# which holds a key's name in the Viewer's info card ("ENTER", "LeftClick") and a code chip
# in the tip panel. One definition, so the two look alike: the Viewer's stylesheet is built
# from key_cap_qss and the panel paints from the same numbers and colours.
KEY_CAP_RADIUS = 5
KEY_CAP_PADDING = 6          # left and right of the text, inside the border
KEY_CAP_BORDER = 1           # the border, on the top and the sides
KEY_CAP_BOTTOM = 2           # the bottom edge, thicker and a shade darker


def key_cap_colors(t=None):
    """(fill, border, bottom edge) of a key cap, as colour names, in the tokens (TOKENS by default)."""
    t = TOKENS if t is None else t
    return t["surface"], t["field_border"], t["field_border_hover"]


def key_cap_qss(t=None):
    """The declarations of a key cap's rule: its box and its text colour (the Viewer's controlsKey label)."""
    t = TOKENS if t is None else t
    fill, border, bottom = key_cap_colors(t)
    return (
        f"background: {fill}; color: {t['text']}; "
        f"border: {KEY_CAP_BORDER}px solid {border}; border-bottom: {KEY_CAP_BOTTOM}px solid {bottom}; "
        f"border-radius: {KEY_CAP_RADIUS}px; padding: 0px {KEY_CAP_PADDING}px;"
    )


def studio_palette(t=None):
    """The palette matching the tokens (TOKENS by default), for the parts Qt draws natively."""
    t = TOKENS if t is None else t
    palette = QPalette()
    for role, value in (
        (QPalette.ColorRole.Window, t["window"]),
        (QPalette.ColorRole.WindowText, t["text"]),
        (QPalette.ColorRole.Base, t["field"]),
        (QPalette.ColorRole.AlternateBase, t["surface_muted"]),
        (QPalette.ColorRole.Button, t["surface"]),
        (QPalette.ColorRole.ButtonText, t["text"]),
        (QPalette.ColorRole.Text, t["text"]),
        (QPalette.ColorRole.BrightText, t["danger"]),
        (QPalette.ColorRole.Link, t["accent"]),
        (QPalette.ColorRole.Highlight, t["accent"]),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.PlaceholderText, t["text_subtle"]),
        (QPalette.ColorRole.ToolTipBase, t["tooltip"]),
        (QPalette.ColorRole.ToolTipText, t["tooltip_text"]),
    ):
        palette.setColor(role, QColor(value))
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.PlaceholderText,
    ):
        palette.setColor(disabled, role, QColor(t["text_disabled"]))
    for role in (QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(disabled, role, QColor(t["field_disabled"]))
    return palette


def apply_studio_theme(app, color=None):
    """Give a Qt application the Studio look: Fusion, the palette and the stylesheet.

    color names the theme colour to set first (set_theme_color); None keeps
    the one TOKENS holds. Call it before the first window is built: the
    layouts measure their controls with the stylesheet's padding. The fonts
    stay as they were (widget_class_fonts).
    """
    if color is not None:
        set_theme_color(color)
    fonts = widget_class_fonts()
    app.setStyle("Fusion")
    app.setPalette(studio_palette())
    # The first application stylesheet sets a style too, one that wraps Fusion.
    app.setStyleSheet(studio_stylesheet())
    restore_widget_class_fonts(app, fonts)


# =====================================================================
# Widget-Class Fonts
# =====================================================================

# The widget classes Qt gives a font of their own from the platform theme
# (QApplicationPrivate::initializeWidgetFontHash in Qt 6.11). A class's font
# wins over the application font. Qt fills its table of them when the
# application starts and again in every QApplication.setStyle, which the first
# QApplication.setStyleSheet calls too, while QApplication.setFont(font)
# empties it. Linux's GTK theme names every class here, so a theme applied
# after setFont put the system font back on labels, buttons, check boxes,
# menus and lists; Windows names the menus, message boxes, tooltips
# (QTipLabel), the status bar and the item views. QApplication.font(name)
# gives the font a class gets.
WIDGET_FONT_CLASSES = (
    "QMenu", "QMenuBar", "QMenuItem", "QMessageBox", "QLabel", "QTipLabel",
    "QTitleBar", "QStatusBar", "QMdiSubWindowTitleBar", "QDockWidgetTitle",
    "QPushButton", "QCheckBox", "QRadioButton", "QToolButton",
    "QAbstractItemView", "QListView", "QHeaderView", "QListBox",
    "QComboMenuItem", "QComboLineEdit", "QSmallFont", "QMiniFont",
)


def widget_class_fonts():
    """The font each class in WIDGET_FONT_CLASSES gets, by class name."""
    return {name: QApplication.font(name) for name in WIDGET_FONT_CLASSES}


def restore_widget_class_fonts(app, fonts):
    """Give the widget classes back the fonts widget_class_fonts() returned.

    A class whose font was the application font follows it again.
    """
    if widget_class_fonts() == fonts:
        return
    application_font = app.font()
    app.setFont(application_font)  # empties the table
    for name, font in fonts.items():
        if font != application_font:
            app.setFont(font, name)


# =====================================================================
# The Web Pages' Colours
# =====================================================================

def _rgba(color, alpha):
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _chevron(color):
    # The select boxes' arrow: Lucide's chevron-down, as a data URL.
    stroke = color.replace("#", "%23")
    return (
        "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
        f"fill='none' stroke='{stroke}' stroke-width='2' stroke-linecap='round' "
        "stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")"
    )


# The custom properties of the Agent and Meta Data pages that take the theme
# colour: (property, light value, dark value), each value a page_tokens name or
# a function of the tokens. The pages keep their own status colours, shadows
# and geometry.
WEB_PAGE_PROPERTIES = (
    ("--bg-main", "window", "window"),
    ("--bg-card", "surface", "surface"),
    ("--bg-muted", "surface_muted", "surface_muted"),
    ("--bg-inset", "inset", "inset"),
    ("--bg-table-header", "inset", "inset"),
    ("--border-color", "border", "border"),
    ("--field-border", "field_border", "field_border"),
    ("--field-border-hover", "field_border_hover", "field_border_hover"),
    ("--text-primary", "text", "text"),
    ("--text-soft", "text_soft", "text_soft"),
    ("--text-secondary", "text_muted", "text_muted"),
    ("--text-subtle", "text_subtle", "text_subtle"),
    ("--text-disabled", "text_disabled", "text_disabled"),
    ("--input-bg", "field", "window"),
    ("--input-text", "text", "text"),
    ("--input-disabled-bg", "field_disabled", "surface"),
    ("--primary", "primary", "primary"),
    ("--primary-hover", "primary_hover", "primary_hover"),
    ("--primary-pressed", "primary_pressed", "primary_pressed"),
    ("--primary-text", "primary_text", "primary_text"),
    ("--primary-disabled", "primary_disabled", "primary_disabled"),
    ("--primary-disabled-text", "primary_disabled_text", "primary_disabled_text"),
    ("--accent", "accent", "accent"),
    ("--accent-soft", "accent_soft", "accent_soft"),
    ("--selection", lambda t: _rgba(t["accent"], 0.1), lambda t: _rgba(t["accent"], 0.16)),
    ("--text-badge-bg", "accent_soft", "accent_soft"),
    ("--text-badge-border", "badge_border", lambda t: _rgba(t["accent"], 0.4)),
    ("--text-badge-text", "badge_text", "badge_text"),
    ("--bg-bubble-user", "surface_muted", "surface_muted"),
    ("--text-bubble-user", "text", "text"),
    ("--text-bubble-agent", "text", "text"),
    ("--code-bg", "inset", "inset"),
    ("--code-border", "border", "border"),
    ("--code-text", "code_text", "code_text"),
    ("--code-inline-bg", "surface_muted", "surface_muted"),
    ("--code-inline-text", "code_text", "text"),
    ("--switch-off", "switch_off", "switch_off"),
    ("--switch-off-hover", "switch_off_hover", "switch_off_hover"),
    ("--switch-on", "switch_on", "switch_on"),
    ("--switch-on-hover", "switch_on_hover", "switch_on_hover"),
    ("--switch-thumb", "switch_thumb", "switch_thumb"),
    ("--switch-thumb-on", "switch_thumb_on", "switch_thumb_on"),
    ("--scroll", "scroll", "scroll"),
    ("--scroll-hover", "scroll_hover", "scroll_hover"),
    ("--select-chevron", lambda t: _chevron(t["text_subtle"]), lambda t: _chevron(t["text_muted"])),
)


def web_page_properties(color=None, *, dark=False):
    """The theme colour's values of WEB_PAGE_PROPERTIES, light or dark, by property.

    color names the theme colour, the one TOKENS holds by default.
    """
    tokens = page_tokens(theme_color() if color is None else color, dark=dark)
    column = 2 if dark else 1
    return {
        entry[0]: entry[column](tokens) if callable(entry[column]) else tokens[entry[column]]
        for entry in WEB_PAGE_PROPERTIES
    }


def web_theme_css(color=None):
    """CSS giving the Agent and Meta Data pages the theme colour, light and dark.

    The Viewer's web server adds it at the end of each page's head, after the
    page's own values (which are Purple's), so it wins.
    """
    blocks = []
    for selector, dark in ((":root", False), ('[data-theme="dark"]', True)):
        lines = "\n".join(
            f"    {name}: {value};" for name, value in web_page_properties(color, dark=dark).items()
        )
        blocks.append(f"{selector} {{\n{lines}\n}}")
    return "\n".join(blocks)


def set_role(widget, role):
    """Give ``widget`` one of the stylesheet's variants (ROLES), restyling it if it is shown."""
    widget.setProperty("role", role)
    if widget.isVisible():
        widget.style().unpolish(widget)
        widget.style().polish(widget)
    return widget


def status_color(kind):
    """The colour of a status text: "success", "warning", "danger" or "subtle"."""
    return TOKENS["text_subtle" if kind == "subtle" else kind]


# =====================================================================
# Icons
# =====================================================================

@lru_cache(maxsize=None)
def icon_svg(name):
    """The bundled Lucide SVG named ``name`` (as on lucide.dev), as text."""
    path = LUCIDE_DIR / f"{name}.svg"
    if not path.is_file():
        raise FileNotFoundError(f"No bundled icon named {name!r} in {LUCIDE_DIR}")
    return path.read_text(encoding="utf-8")


class StudioIconEngine(QIconEngine):
    """Draws a Lucide icon in a theme colour, crisp at any device pixel ratio.

    A disabled icon is drawn in the disabled text colour.
    """

    def __init__(self, name, color, disabled_color):
        super().__init__()
        self._name = name
        self._colors = (color, disabled_color)
        self._renderers = {}

    def _renderer(self, mode):
        color = self._colors[mode == QIcon.Mode.Disabled]
        renderer = self._renderers.get(color)
        if renderer is None:
            data = coloured_svg(icon_svg(self._name), color).encode("utf-8")
            renderer = self._renderers[color] = QSvgRenderer(QByteArray(data))
        return renderer

    def paint(self, painter, rect, mode, state):
        self._renderer(mode).render(painter, QRectF(rect))

    def scaledPixmap(self, size, mode, state, scale):
        pixmap = QPixmap(max(1, round(size.width() * scale)), max(1, round(size.height() * scale)))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._renderer(mode).render(painter)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        return pixmap

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def clone(self):
        return StudioIconEngine(self._name, *self._colors)

    def key(self):
        return "studio-lucide"


def studio_icon(name, color=None, disabled_color=None):
    """A QIcon of the bundled Lucide icon ``name``, in ``color`` (default: subtle text)."""
    return QIcon(StudioIconEngine(
        name, color or TOKENS["text_subtle"], disabled_color or TOKENS["text_disabled"]
    ))


class IconLabel(QLabel):
    """A label showing one icon at ``size`` logical pixels, painted at the screen's pixel ratio.

    It stays a QLabel, so it can be the buddy of a field (setBuddy).
    """

    def __init__(self, name, size=16, color=None, parent=None):
        super().__init__(parent)
        self._set(name, color)
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _set(self, name, color):
        self._icon = studio_icon(name, color)
        self.icon_name = name
        self.icon_color = color or TOKENS["text_subtle"]

    def set_icon(self, name, color=None):
        self._set(name, color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
        self._icon.paint(painter, self.rect(), Qt.AlignmentFlag.AlignCenter, mode)
        painter.end()


def icon_button(name, tooltip="", parent=None):
    """A square button showing only the icon ``name``; give it a fixed width where it sits."""
    button = QPushButton(parent)
    button.setIcon(studio_icon(name))
    button.setIconSize(QSize(16, 16))
    if tooltip:
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
    set_role(button, "icon")
    return button


def add_hint_icon(label):
    """Put an info icon in the top-left padding of a "hint" label; returns the icon."""
    set_role(label, "hint")
    label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    icon = IconLabel("info", HINT_ICON_SIZE, parent=label)
    icon.move(HINT_ICON_OFFSET, HINT_ICON_OFFSET)
    return icon


def prepare_tab_widget(tabs):
    """Let a QTabWidget's bar paint the segmented control's grey track."""
    bar = tabs.tabBar()
    bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    bar.setDrawBase(False)
    return tabs


# =====================================================================
# The Painted Switch & Card Shadows
# =====================================================================

SWITCH_WIDTH = 34
SWITCH_HEIGHT = 20
SWITCH_LABEL_GAP = 8


def switch_label(texts):
    """The words a switch's two texts share, shown beside it ("Auto" of "Auto OFF"/"Auto ON")."""
    off, on = (str(text) for text in texts)
    common = os.path.commonprefix([off, on])
    if not common:
        return ""
    if common.rstrip() != common or common in (off, on):
        return common.strip()
    cut = common.rfind(" ")
    if cut > 0:
        return common[:cut].strip()
    # Without spaces, as in Chinese, the shared characters are the label.
    return common.strip() if any(ord(character) > 0x2E80 for character in common) else ""


def _switch_label_font(button):
    font = button.font()
    font.setBold(False)
    return font


def switch_track_rect(button):
    """Where a switch draws its track, in the button's coordinates: after its label, if any."""
    label = switch_label(button.state_texts())
    x = 0.0
    if label:
        x = QFontMetrics(_switch_label_font(button)).horizontalAdvance(label) + SWITCH_LABEL_GAP
    y = (button.height() - SWITCH_HEIGHT) / 2.0
    return QRectF(x + 0.5, y + 0.5, SWITCH_WIDTH - 1, SWITCH_HEIGHT - 1)


def paint_switch(button):
    """Draw a checkable button as a switch: a track and a knob, after the texts' shared label.

    The button keeps its text (screen readers read it); only the drawing changes.
    Its focus ring reaches past the button's edges, so SwitchFocusRing draws
    it, on the button's parent.
    """
    painter = QPainter(button)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    if not button.isEnabled():
        painter.setOpacity(0.45)
    rect = button.rect()
    label = switch_label(button.state_texts())
    if label:
        painter.setFont(_switch_label_font(button))
        painter.setPen(QColor(TOKENS["text_soft"]))
        width = painter.fontMetrics().horizontalAdvance(label)
        painter.drawText(QRectF(0, 0, width + 1, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
    track = switch_track_rect(button)
    x, y = track.x() - 0.5, track.y() - 0.5
    on = button.isChecked()
    hover = button.underMouse() and button.isEnabled()
    if on:
        fill = TOKENS["switch_on_hover" if hover else "switch_on"]
    else:
        fill = TOKENS["switch_off_hover" if hover else "switch_off"]
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(fill))
    painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)
    diameter = SWITCH_HEIGHT - 4.0
    knob = QRectF(x + (SWITCH_WIDTH - 2.0 - diameter if on else 2.0), y + 2.0, diameter, diameter)
    painter.setBrush(QColor(0, 0, 0, 28))  # a soft contact shadow
    painter.drawEllipse(knob.adjusted(0, 0.8, 0, 0.8))
    painter.setBrush(QColor("#ffffff"))
    painter.drawEllipse(knob)
    painter.end()


class SwitchFocusRing(QObject):
    """Draws a focused switch's ring, 2.5 px around its track, on an ancestor of the switch.

    A label-less switch's track starts at the switch's left edge, where the
    switch can't draw the ring's left side, nor can a parent that wraps it
    tightly. So the nearest ancestor that holds the whole ring draws it: an
    ancestor draws before its children, and neither the switch nor the
    widgets between them have a background.
    """

    GAP = 2.5
    PEN = 2.0

    def __init__(self, switch):
        super().__init__(switch)
        self.switch = switch
        self.host = None
        switch.installEventFilter(self)

    def _ring_in(self, widget):
        """The ring's path, in widget's coordinates."""
        offset = QPointF(self.switch.mapTo(widget, QPoint(0, 0)))
        track = switch_track_rect(self.switch).translated(offset)
        return track.adjusted(-self.GAP, -self.GAP, self.GAP, self.GAP)

    def _find_host(self):
        # The ring's pen, and a pixel for antialiasing, reach past its path.
        reach = self.PEN / 2 + 1
        widget = self.switch.parentWidget()
        while widget is not None:
            outer = self._ring_in(widget).adjusted(-reach, -reach, reach, reach)
            if widget.isWindow() or QRectF(widget.rect()).contains(outer):
                return widget
            widget = widget.parentWidget()
        return None

    def _update_host(self):
        host = self._find_host()
        if host is not self.host:
            if self.host is not None and shiboken6.isValid(self.host):
                self.host.removeEventFilter(self)
                self.host.update()
            self.host = host
            if host is not None:
                host.installEventFilter(self)
        if self.host is not None:
            self.host.update()

    def eventFilter(self, watched, event):
        # While a window closes, the switch can go before its ancestors' last
        # events, and Python's garbage collector can clear this filter's
        # attributes before Qt stops calling it.
        switch = getattr(self, "switch", None)
        if switch is None or not shiboken6.isValid(switch):
            return False
        kind = event.type()
        if watched is self.switch:
            if kind in (QEvent.Type.FocusIn, QEvent.Type.FocusOut, QEvent.Type.Move,
                        QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Hide,
                        QEvent.Type.ParentChange):
                self._update_host()
        elif (watched is self.host and kind == QEvent.Type.Paint
              and self.switch.hasFocus() and self.switch.isVisible()):
            painter = QPainter(self.host)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            ring = QPen(QColor(TOKENS["accent"]))
            ring.setWidthF(self.PEN)
            painter.setPen(ring)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = self._ring_in(self.host)
            painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
            painter.end()
        return False


# A switch draws itself, so its stylesheet only clears the button's frame. The
# heights keep a layout from squeezing it below its fixed 28 px.
SWITCH_STYLESHEET = (
    "QPushButton { background: transparent; border: none; padding: 0px; "
    "min-height: 28px; max-height: 28px; }"
)


class CardShadows(QObject):
    """Paints a soft shadow (CSS "0 1px 3px") under each card, on ``host``.

    Every widget between ``host`` and the cards must be transparent, so the
    shadow shows around the cards. Unlike QGraphicsDropShadowEffect it is never
    clipped by a card's parent and never re-renders the card's children.
    """

    # (grow, alpha) layers, outermost first.
    LAYERS = ((3.5, 3), (2.5, 4), (1.5, 5), (0.7, 6))

    def __init__(self, host, cards, radius=None):
        super().__init__(host)
        self.host = host
        self.cards = list(cards)
        self.radius = TOKENS["radius_card"] if radius is None else radius
        host.installEventFilter(self)
        for card in self.cards:
            card.installEventFilter(self)

    def eventFilter(self, watched, event):
        # While a window closes, its cards still get Hide and Move events after
        # the host has gone; there is nothing left to shade then.
        if not shiboken6.isValid(self.host):
            return False
        kind = event.type()
        if watched is self.host and kind == QEvent.Type.Paint:
            painter = QPainter(self.host)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            for card in self.cards:
                if not card.isVisible():
                    continue
                corner = card.mapTo(self.host, QPoint(0, 0))
                rect = QRectF(corner.x(), corner.y() + 1.0, card.width(), card.height())
                for grow, alpha in self.LAYERS:
                    painter.setBrush(QColor(9, 9, 11, alpha))
                    painter.drawRoundedRect(
                        rect.adjusted(-grow, -grow, grow, grow), self.radius + grow, self.radius + grow
                    )
            painter.end()
        elif watched is not self.host and kind in (
            QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show, QEvent.Type.Hide
        ):
            self.host.update()
        return False


__all__ = [
    "ACCENT_CHROMA",
    "ACCENT_SHADES",
    "BLACK_ACCENTS",
    "CardShadows",
    "DEFAULT_THEME_COLOR",
    "HINT_BORDER",
    "HINT_PADDING",
    "HINT_RIGHT_PADDING",
    "HINT_SCROLL_EDGE",
    "HINT_TEXT_INDENT",
    "HINT_TEXT_MARGIN",
    "ICONS_DIR",
    "IconLabel",
    "KEY_CAP_BORDER",
    "KEY_CAP_BOTTOM",
    "KEY_CAP_PADDING",
    "KEY_CAP_RADIUS",
    "NEUTRAL_SHADES",
    "LUCIDE_DIR",
    "QSS_ICON_DIR",
    "QSS_ICON_VARIANTS",
    "ROLES",
    "SWITCH_STYLESHEET",
    "StudioIconEngine",
    "SwitchFocusRing",
    "THEME_COLORS",
    "TINTED_SHADES",
    "TOKENS",
    "WEB_PAGE_PROPERTIES",
    "WIDGET_FONT_CLASSES",
    "add_hint_icon",
    "apply_studio_theme",
    "coloured_svg",
    "icon_button",
    "icon_svg",
    "key_cap_colors",
    "key_cap_qss",
    "oklch_hex",
    "page_tokens",
    "paint_switch",
    "prepare_tab_widget",
    "qss_icon_path",
    "restore_widget_class_fonts",
    "set_role",
    "set_theme_color",
    "status_color",
    "studio_icon",
    "studio_palette",
    "studio_stylesheet",
    "switch_label",
    "switch_track_rect",
    "theme_color",
    "theme_tokens",
    "tinted_tokens",
    "web_page_properties",
    "web_theme_css",
    "widget_class_fonts",
]
