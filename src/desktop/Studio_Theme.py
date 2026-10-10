# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Studio: the shared look of the EMAP-SSN desktop windows.

Config, VR Config, Tools and the Viewer share one look, after shadcn/ui,
Linear and Vercel: zinc greys, hairline borders, 8 px controls on 12 px cards,
a near-black primary button and one indigo accent for focus and slider fills.

* TOKENS names every colour and radius. studio_stylesheet() turns them into
  the application stylesheet, and studio_palette() into the palette, so the
  parts Qt draws natively (popups, selection, disabled text) match.
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
  bundled Noto Sans SC has no Medium face and would draw 500 as bold.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject, QPoint, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QIconEngine, QPainter, QPalette, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
import shiboken6
from PySide6.QtWidgets import QLabel, QPushButton

ICONS_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"
LUCIDE_DIR = ICONS_DIR / "lucide"
QSS_ICON_DIR = ICONS_DIR / "qss"

# Tailwind's zinc scale, the neutral of shadcn/ui.
ZINC = {
    50: "#fafafa", 100: "#f4f4f5", 200: "#e4e4e7", 300: "#d4d4d8", 400: "#a1a1aa",
    500: "#71717a", 600: "#52525b", 700: "#3f3f46", 800: "#27272a", 900: "#18181b",
    950: "#09090b",
}

TOKENS = {
    # Surfaces
    "window": ZINC[50],
    "surface": "#ffffff",        # cards: the tab pane, reports
    "surface_muted": ZINC[100],  # muted buttons, hover fills, the hint card
    "track": "#efeff1",          # the segmented tab bar's track
    "border": ZINC[200],         # card hairlines and separators
    # Text (contrast on white)
    "text": ZINC[950],
    "text_soft": ZINC[700],      # report body, switch labels
    "text_muted": ZINC[600],     # inactive tabs, hints, ghost buttons (7.7:1)
    "text_subtle": ZINC[500],    # icons, placeholders, secondary status (4.8:1)
    "text_disabled": ZINC[400],
    # Fields
    "field": "#ffffff",
    "field_border": "#dcdce0",   # a touch stronger than a card's hairline
    "field_border_hover": ZINC[300],
    "field_disabled": ZINC[50],
    # The primary button: near-black, as shadcn/ui's default button
    "primary": ZINC[900],
    "primary_hover": ZINC[800],
    "primary_pressed": ZINC[700],
    "primary_text": ZINC[50],
    "primary_disabled": "#9b9ba0",
    # The one accent: focus rings, slider fills, selection
    "accent": "#5b5bd6",         # 5.4:1 on white
    "accent_soft": "#ebebfb",
    # Switches
    "switch_off": ZINC[300],
    "switch_off_hover": ZINC[400],
    "switch_on": ZINC[900],
    "switch_on_hover": ZINC[800],
    # Status texts
    "success": "#15803d",        # 5.0:1 on white
    "warning": "#b45309",        # 5.0:1 on white
    "danger": "#dc2626",         # 4.8:1 on white
    # Scroll bars
    "scroll": ZINC[300],
    # Geometry
    "radius": 8,                 # controls
    "radius_card": 12,           # cards
    "card_gap": 8,               # between a tab bar and its card, and between cards
}

# The variants a widget can ask for with set_role(widget, role).
ROLES = {
    "QPushButton": {
        "primary": "the window's main action, near-black (at most one per window)",
        "muted": "a secondary action on a grey fill",
        "ghost": "a text-only action, such as Exit",
        "icon": "a square button showing only an icon",
        "compact": "a short text button beside a field, such as Pick or Clear",
    },
    "QLabel": {
        "hint": "the help card under the tabs; add_hint_icon puts an info icon in it",
        "title": "a panel heading",
        "subtle": "secondary text, such as a status line",
    },
    "QFrame": {"separator": "a 1 px line between sections"},
    "QTextEdit": {"report": "a read-only report card"},
}

# Copies of Lucide icons coloured for the stylesheet: (icon, token).
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


def qss_icon_path(name, token):
    """The pre-coloured copy of a Lucide icon that the stylesheet uses."""
    return QSS_ICON_DIR / f"{name}-{TOKENS[token].lstrip('#')}.svg"


def coloured_svg(svg_text, color):
    """An SVG whose currentColor strokes and fills are drawn in ``color``."""
    return svg_text.replace("currentColor", color)


def _qss_url(name, token):
    # Quoted: the program may live in a folder whose name has spaces.
    return f'url("{qss_icon_path(name, token).as_posix()}")'


# =====================================================================
# Stylesheet & Palette
# =====================================================================

def studio_stylesheet(t=TOKENS):
    """The application stylesheet built from the tokens."""
    down = _qss_url("chevron-down", "text_subtle")
    up = _qss_url("chevron-up", "text_subtle")
    down_disabled = _qss_url("chevron-down", "text_disabled")
    up_disabled = _qss_url("chevron-up", "text_disabled")
    left = _qss_url("chevron-left", "text_subtle")
    right = _qss_url("chevron-right", "text_subtle")
    left_disabled = _qss_url("chevron-left", "text_disabled")
    right_disabled = _qss_url("chevron-right", "text_disabled")
    return f"""
QMainWindow, QDialog {{ background: {t['window']}; }}
QWidget {{ color: {t['text']}; }}
QLabel {{ background: transparent; }}
QLabel:disabled {{ color: {t['text_disabled']}; }}
QToolTip {{
    background: {t['primary']}; color: {t['primary_text']}; border: none;
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
    border: 1px solid {t['border']}; border-bottom-color: {ZINC[300]};
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
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 6px 2px 6px 0px; }}
QScrollBar::handle:vertical {{ background: {t['scroll']}; border-radius: 4px; min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: {t['text_disabled']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0px 6px 2px 6px; }}
QScrollBar::handle:horizontal {{ background: {t['scroll']}; border-radius: 4px; min-width: 32px; }}
QScrollBar::handle:horizontal:hover {{ background: {t['text_disabled']}; }}
/* Where both scroll bars meet: nothing, so a card's rounded corner shows. */
QAbstractScrollArea::corner {{ background: transparent; border: none; }}

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
QComboBox QAbstractItemView {{
    background: {t['surface']}; border: 1px solid {t['border']}; border-radius: {t['radius']}px;
    padding: 4px; outline: 0;
    selection-background-color: {t['surface_muted']}; selection-color: {t['text']};
}}
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
QPushButton[role="primary"]:focus {{ border: 2px solid {t['accent']}; padding: 2px 12px; }}
QPushButton[role="primary"]:disabled {{
    background: {t['primary_disabled']}; border-color: {t['primary_disabled']}; color: {t['primary_text']};
}}
QPushButton[role="muted"] {{
    background: {t['surface_muted']}; color: {t['text']}; border: 1px solid {t['surface_muted']};
}}
QPushButton[role="muted"]:hover {{ background: {t['border']}; border-color: {t['border']}; }}
QPushButton[role="muted"]:focus {{ border: 2px solid {t['accent']}; padding: 2px 12px; }}
QPushButton[role="muted"]:disabled {{
    background: {t['field_disabled']}; border-color: {t['border']}; color: {t['text_disabled']};
}}
QPushButton[role="ghost"] {{ background: transparent; border-color: transparent; color: {t['text_muted']}; }}
QPushButton[role="ghost"]:hover {{ background: {t['surface_muted']}; color: {t['text']}; }}
QPushButton[role="ghost"]:focus {{ border: 2px solid {t['accent']}; padding: 2px 12px; }}
QPushButton[role="icon"] {{ padding: 3px; }}
QPushButton[role="icon"]:focus {{ padding: 2px; }}
QPushButton[role="compact"] {{ padding: 3px 9px; }}
QPushButton[role="compact"]:focus {{ padding: 2px 8px; }}

/* Panels */
QFrame[role="separator"] {{ background: {t['border']}; border: none; max-height: 1px; }}
QLabel[role="hint"] {{
    background: {t['surface_muted']}; color: {t['text_muted']};
    border: 1px solid {t['surface_muted']}; border-radius: {t['radius_card']}px;
    padding: 11px 14px 11px {HINT_TEXT_INDENT}px;
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


def studio_palette(t=TOKENS):
    """The palette matching the tokens, for the parts Qt draws natively."""
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
        (QPalette.ColorRole.ToolTipBase, t["primary"]),
        (QPalette.ColorRole.ToolTipText, t["primary_text"]),
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


def apply_studio_theme(app):
    """Give a Qt application the Studio look: Fusion, the palette and the stylesheet.

    Call it once, before the first window is built: the layouts measure their
    controls with the stylesheet's padding.
    """
    app.setStyle("Fusion")
    app.setPalette(studio_palette())
    app.setStyleSheet(studio_stylesheet())


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
        self._icon = studio_icon(name, color)
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_icon(self, name, color=None):
        self._icon = studio_icon(name, color)
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


def paint_switch(button):
    """Draw a checkable button as a switch: a track and a knob, after the texts' shared label.

    The button keeps its text (screen readers read it); only the drawing changes.
    """
    painter = QPainter(button)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    if not button.isEnabled():
        painter.setOpacity(0.45)
    rect = button.rect()
    label = switch_label(button.state_texts())
    x = 0.0
    if label:
        font = button.font()
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(TOKENS["text_soft"]))
        width = painter.fontMetrics().horizontalAdvance(label)
        painter.drawText(QRectF(0, 0, width + 1, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
        x = width + SWITCH_LABEL_GAP
    y = (rect.height() - SWITCH_HEIGHT) / 2.0
    track = QRectF(x + 0.5, y + 0.5, SWITCH_WIDTH - 1, SWITCH_HEIGHT - 1)
    on = button.isChecked()
    hover = button.underMouse() and button.isEnabled()
    if on:
        fill = TOKENS["switch_on_hover" if hover else "switch_on"]
    else:
        fill = TOKENS["switch_off_hover" if hover else "switch_off"]
    if button.hasFocus():
        ring = QPen(QColor(TOKENS["accent"]))
        ring.setWidthF(2.0)
        painter.setPen(ring)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        radius = SWITCH_HEIGHT / 2 + 2.5
        painter.drawRoundedRect(track.adjusted(-2.5, -2.5, 2.5, 2.5), radius, radius)
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
    "CardShadows",
    "HINT_TEXT_INDENT",
    "ICONS_DIR",
    "IconLabel",
    "LUCIDE_DIR",
    "QSS_ICON_DIR",
    "QSS_ICON_VARIANTS",
    "ROLES",
    "SWITCH_STYLESHEET",
    "StudioIconEngine",
    "TOKENS",
    "ZINC",
    "add_hint_icon",
    "apply_studio_theme",
    "coloured_svg",
    "icon_button",
    "icon_svg",
    "paint_switch",
    "prepare_tab_widget",
    "qss_icon_path",
    "set_role",
    "status_color",
    "studio_icon",
    "studio_palette",
    "studio_stylesheet",
    "switch_label",
]
