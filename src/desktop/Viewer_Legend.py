# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's legend card: what the colors of the network mean, as the color key says (desktop/Node_Colors.py).

It is the legend Tree View shows (web_ui/tree_legend.js), as a card at the canvas's top right, under the top-right
row, opened by the row's Legend button (desktop/Viewer_History.py). It reads only the key's layers, kinds, titles
and labels, so it names no command.

* The layers are listed newest first, counting the nodes that are shown (not hidden) only; a layer or a row with
  no shown node is left out. A layer of categories is a title and a row for each category: a swatch, the label
  and the count. A layer of a gradient is a title, the scheme, a bar of its colors from its minimum to its
  maximum, and the rows of what it leaves out. The shown nodes the key does not cover (not labelled, or with
  no key) are listed last under "Unlabelled colours", grouped by color the way Tree View groups them (4 bits per
  channel), each row a swatch with its hex code and a count; the color nothing has painted is left out.
* A click on a row selects its nodes, replacing the selection; Ctrl+click adds them; a click on the row whose
  nodes already are exactly the selection clears it. Pressing or dragging on a gradient bar selects the nodes
  whose value lies in the range it covers, shown as "{low} to {high}". Hidden nodes are never selected. The
  selection goes the way a box selection's does (MainViewer.select_by_hand): the bar's drag moves the
  selection as it goes and tells the web pages and the status line once, when it is let go.
* The card opens and closes with the Legend button (a second click closes it) or with its x, and only then: it never
  opens by itself when a key appears. It is not a popup: it stays open while the user clicks its rows, the canvas
  or the session card. While no key has been declared it says so in one plain line. It looks like the session
  card (desktop/Viewer_Widgets.py): the same shadow, corners, heading, x button, separators between its sections
  and text styles, on a white body where the session card's is the theme's tint. Both cards can be open: the
  Viewer places them so they never overlap (MainViewer._place_legend), and the card's height is capped, its
  content scrolling.
* Labels and titles come from a layout cache somebody may have shared: they are only ever drawn as plain text.

The card is a child of the canvas, so the network works under the mouse as usual; every mouse event on the card
itself stops at it, and nothing in it takes the keyboard, so [ENTER] still opens the command line.
"""

from __future__ import annotations

import zlib

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from desktop.Node_Colors import KEY_ATTRIBUTES, key_state
from desktop.Studio_Theme import TOKENS
from desktop.Viewer_Widgets import ControlsCard, card_close_button, card_qss, card_separator, paint_card

LEGEND_WIDTH = 260        # the card's width (its body, not its shadow), in px, in a window with room for it
LEGEND_MIN_WIDTH = 140    # the least a card shrinks to in a narrow window
WHITE = "#ffffff"         # the card's body: white, not the theme's tint
SECTION_GAP = 10          # between the sections of the body, the session card's
ROW_HEIGHT = 24
SWATCH = 12
BAR_HEIGHT = 14
BAR_DRAG_PIXELS = 3       # a press that moves less than this is a press, not a drag
CLICK_SPAN = 1 / 32       # the share of the bar a press without a drag covers, each side of the press
CLASS_COUNT = 4096        # colors grouped by their 4 high bits per channel, as Tree View groups them


# ---------------------------------------------------------------------
# What the legend says
# ---------------------------------------------------------------------

def number(value):
    """A bar's end or range as Tree View shows it: 4 significant digits."""
    return f"{float(value):.4g}"


def _byte(channel):
    return int(min(255, max(0, round(float(channel) * 255.0))))


def _rgb(rgba):
    return (_byte(rgba[0]), _byte(rgba[1]), _byte(rgba[2]))


def _shown_nodes(viewer, n):
    """A bool per node: True for the nodes the Viewer shows (not hidden)."""
    mask = getattr(viewer, "visible_mask", None)
    if mask is None or len(mask) != n:
        return np.ones(n, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _color_bytes(viewer):
    colors = np.asarray(viewer.current_colors, dtype=np.float64)
    return np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)


def _color_classes(colors):
    """The class of each color (uint8 rows): its 4 high bits per channel, as Tree View groups colors."""
    return ((colors[:, 0].astype(np.int32) >> 4) << 8) | ((colors[:, 1].astype(np.int32) >> 4) << 4) \
        | (colors[:, 2].astype(np.int32) >> 4)


def _plain_class():
    """The class of the color a node has until something paints it (the Viewer's initial node color)."""
    from matplotlib import colors as mcolors

    import EMAPSSN_Config as cfg

    rgba = mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)
    return int(_color_classes(np.array([[_byte(rgba[0]), _byte(rgba[1]), _byte(rgba[2])]], dtype=np.uint8))[0])


def legend_model(viewer):
    """What the legend shows, as plain data, or None when there is nothing to show.

    {"layers": [...], "unlabelled": [...]}. Each of the layers, newest first and only those with a shown node,
    is {"id", "kind", "title", "scheme", "min", "max", "stops", "rows", "bar"}: "rows" are {"slot", "label",
    "rgb", "count"} of the category or extra with shown nodes, "stops" a gradient's 16 colors as 0-255
    triples, and "bar" how many shown nodes the gradient colors by value. "unlabelled" are {"cls", "hex",
    "rgb", "count"}, the most nodes first. It reads the key (key_state) and the nodes' colors and visibility
    and counts them with numpy, never in a loop over nodes.
    """
    key = key_state(viewer)
    if key is None:
        return None
    n = len(key["owner"])
    shown = _shown_nodes(viewer, n)
    labelled = key["labelled"] & shown
    owner, category, values = key["owner"], key["category"], key["values"]
    layers = []
    for layer in reversed(key["layers"]):
        gradient = layer["kind"] == "gradient"
        slots = layer["extras"] if gradient else layer["categories"]
        mine = labelled & (owner == layer["id"])
        slot_of = category[mine]
        counts = np.bincount(slot_of[(slot_of >= 0) & (slot_of < len(slots))], minlength=len(slots))
        rows = [{"slot": int(slot), "label": slots[slot]["label"], "rgb": _rgb(slots[slot]["rgba"]),
                 "count": int(counts[slot])} for slot in np.flatnonzero(counts)]
        bar = int(np.count_nonzero(mine & (category == -1) & np.isfinite(values))) if gradient else 0
        if not rows and not bar:
            continue
        entry = {"id": layer["id"], "kind": layer["kind"], "title": layer["title"], "scheme": "", "min": None,
                 "max": None, "stops": [], "rows": rows, "bar": bar}
        if gradient:
            entry.update(scheme=layer["scheme"], min=layer["min"], max=layer["max"],
                         stops=[_rgb(stop) for stop in layer["stops"]])
        layers.append(entry)

    unlabelled = []
    nodes = np.flatnonzero(shown & ~key["labelled"])
    if nodes.size:
        colors = _color_bytes(viewer)[nodes]
        classes = _color_classes(colors)
        found, first, counts = np.unique(classes, return_index=True, return_counts=True)
        plain = _plain_class()
        for cls, at, count in sorted(zip(found.tolist(), first.tolist(), counts.tolist()), key=lambda e: (-e[2], e[1])):
            if cls == plain:
                continue
            rgb = tuple(int(channel) for channel in colors[at, :3])
            unlabelled.append({"cls": cls, "hex": "#%02x%02x%02x" % rgb, "rgb": rgb, "count": count})
    if not layers and not unlabelled:
        return None
    return {"layers": layers, "unlabelled": unlabelled}


def _live(viewer):
    """(key, shown nodes, current key) as the Viewer holds them now: what a click picks its nodes from."""
    key = key_state(viewer)
    if key is None:
        return None, None
    return key, _shown_nodes(viewer, len(key["owner"]))


def nodes_of_row(viewer, layer_id, slot):
    """The shown nodes a layer's category (or extra) ``slot`` labels now, as node numbers."""
    key, shown = _live(viewer)
    if key is None:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(shown & key["labelled"] & (key["owner"] == layer_id) & (key["category"] == slot))


def nodes_in_range(viewer, layer_id, low, high):
    """The shown nodes of a gradient layer whose value is from ``low`` to ``high``, as node numbers."""
    key, shown = _live(viewer)
    if key is None:
        return np.empty(0, dtype=np.int64)
    values = key["values"]
    with np.errstate(invalid="ignore"):
        inside = (values >= low) & (values <= high)
    return np.flatnonzero(shown & key["labelled"] & (key["owner"] == layer_id) & (key["category"] == -1) & inside)


def nodes_of_colour(viewer, cls):
    """The shown nodes the key does not cover whose color is in class ``cls``, as node numbers."""
    key, shown = _live(viewer)
    if key is None:
        return np.empty(0, dtype=np.int64)
    nodes = np.flatnonzero(shown & ~key["labelled"])
    return nodes[_color_classes(_color_bytes(viewer)[nodes]) == cls]


# ---------------------------------------------------------------------
# The widgets
# ---------------------------------------------------------------------

class _ElidedLabel(QLabel):
    """A one-line plain-text label that is cut short with an ellipsis, never wider than its room."""

    def __init__(self, text="", parent=None, *, align=Qt.AlignmentFlag.AlignLeft):
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setText(text)
        self._align = align
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        metrics = QFontMetrics(self.font())
        return QSize(metrics.horizontalAdvance(self.text()), metrics.height() + 2)

    def minimumSizeHint(self):
        return QSize(0, QFontMetrics(self.font()).height() + 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(self.palette().color(self.foregroundRole())))
        painter.setFont(self.font())
        text = QFontMetrics(self.font()).elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), int(self._align | Qt.AlignmentFlag.AlignVCenter), text)
        painter.end()


class _RowList(QWidget):
    """The rows of a group, painted by the widget itself: a swatch, a label (plain text) and a count."""

    activated = Signal(int, bool)   # the row clicked, and whether Ctrl was held

    def __init__(self, rows, tooltip, parent=None):
        super().__init__(parent)
        self._rows = list(rows)       # (rgb, label, count)
        self._hover = -1
        self._press = -1
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setFixedHeight(ROW_HEIGHT * len(self._rows))
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def rows(self):
        """The rows as (rgb, label, count text), as drawn."""
        return [(rgb, label, f"{count:,}") for rgb, label, count in self._rows]

    def sizeHint(self):
        return QSize(100, ROW_HEIGHT * len(self._rows))

    def minimumSizeHint(self):
        return QSize(0, ROW_HEIGHT * len(self._rows))

    def _index_at(self, y):
        index = int(y) // ROW_HEIGHT
        return index if 0 <= index < len(self._rows) else -1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        metrics = QFontMetrics(self.font())
        first = max(0, event.rect().top() // ROW_HEIGHT)
        last = min(len(self._rows), event.rect().bottom() // ROW_HEIGHT + 1)
        for index in range(first, last):
            rgb, label, count = self._rows[index]
            top = index * ROW_HEIGHT
            if index == self._hover:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(TOKENS["tint_hover"]))
                painter.drawRoundedRect(QRectF(0, top, self.width(), ROW_HEIGHT), 5, 5)
            painter.setPen(QPen(QColor(TOKENS["border"]), 1.0))
            painter.setBrush(QColor(*rgb))
            painter.drawRoundedRect(QRectF(5.5, top + (ROW_HEIGHT - SWATCH) / 2 + 0.5, SWATCH - 1, SWATCH - 1), 3, 3)
            count_text = f"{count:,}"
            count_width = metrics.horizontalAdvance(count_text)
            painter.setPen(QColor(TOKENS["text_muted"]))
            painter.drawText(QRectF(self.width() - 6 - count_width, top, count_width, ROW_HEIGHT),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), count_text)
            label_left = 5 + SWATCH + 8
            room = max(0, self.width() - label_left - count_width - 14)
            painter.setPen(QColor(TOKENS["text"]))
            painter.drawText(QRectF(label_left, top, room, ROW_HEIGHT),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             metrics.elidedText(label, Qt.TextElideMode.ElideRight, room))
        painter.end()

    def mouseMoveEvent(self, event):
        index = self._index_at(event.position().y())
        if index != self._hover:
            self._hover = index
            self.update()
        event.accept()

    def leaveEvent(self, event):
        self._hover = -1
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = self._index_at(event.position().y())
        event.accept()

    def mouseReleaseEvent(self, event):
        index = self._index_at(event.position().y())
        if event.button() == Qt.MouseButton.LeftButton and index >= 0 and index == self._press:
            self.activated.emit(index, bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier))
        self._press = -1
        event.accept()


class _GradientBar(QWidget):
    """A bar of a gradient's colors; pressing or dragging on it picks a range of it."""

    picked = Signal(float, float, bool)   # the shares (0 to 1, either way round) the pick covers, and Ctrl
    released = Signal()

    def __init__(self, stops, tooltip, parent=None):
        super().__init__(parent)
        self._stops = list(stops)
        self._band = None
        self._start = None
        self._moved = False
        self._ctrl = False
        self.setFixedHeight(BAR_HEIGHT)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip(tooltip)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def show_band(self, band):
        """Mark the range (low share, high share) as picked, or nothing for None."""
        if band != self._band:
            self._band = band
            self.update()

    def _share(self, event):
        # The first pixel is the bar's start and the last its end, so a drag can reach either.
        return min(1.0, max(0.0, event.position().x() / (self.width() - 1))) if self.width() > 1 else 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if len(self._stops) > 1:
            gradient = QLinearGradient(QPointF(rect.left(), 0), QPointF(rect.right(), 0))
            for index, rgb in enumerate(self._stops):
                gradient.setColorAt(index / (len(self._stops) - 1), QColor(*rgb))
            brush = QBrush(gradient)
        else:
            brush = QBrush(QColor(*(self._stops[0] if self._stops else (128, 128, 128))))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(brush)
        painter.drawRoundedRect(rect, 3, 3)
        if self._band is not None:
            low, high = self._band
            left = rect.left() + rect.width() * low
            width = max(2.0, rect.width() * (high - low))
            painter.setBrush(QColor(255, 255, 255, 70))
            painter.setPen(QPen(QColor(TOKENS["text"]), 1.5))
            painter.drawRect(QRectF(left, rect.top() - 0.5, width, rect.height() + 1))
        painter.setPen(QPen(QColor(TOKENS["border"]), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 3, 3)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = self._share(event)
            self._moved = False
            self._ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self.picked.emit(self._start - CLICK_SPAN, self._start + CLICK_SPAN, self._ctrl)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._start is not None:
            here = self._share(event)
            if not self._moved and abs(here - self._start) * self.width() < BAR_DRAG_PIXELS:
                event.accept()
                return
            self._moved = True
            self.picked.emit(self._start, here, self._ctrl)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._start is not None:
            self._start = None
            self.released.emit()
        event.accept()


class LegendCard(QWidget):
    """The card (see the module's description). ``viewer`` is the MainViewer; ``parent`` is the canvas's widget.

    open_card() and close_card() show and hide it as the Legend button asks (toggle()), and call ``toggled`` with
    whether it is open. refresh() reads the Viewer's key and rebuilds what changed; schedule_refresh() asks for
    one refresh at the end of the event loop's turn however often it is called, which is what the Viewer's node
    update listeners call (nothing happens while the card is closed: opening it refreshes).
    place() puts the card's right edge at ``right``, its top at ``top``, and keeps its bottom above ``bottom``.
    """

    def __init__(self, viewer, parent):
        super().__init__(parent)
        self.viewer = viewer
        self.setObjectName("viewerLegendCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setMouseTracking(True)
        self.opened = False
        self.model = None
        self.relayout = None      # called after a refresh changed what the card shows
        self.toggled = None       # called with whether the card is open, after it opened or closed
        self._seen = None         # what the last refresh read: (checksum of colors and shown nodes, the key's objects)
        self._pending = False
        self._style_key = None
        self._range = None        # (layer id, low, high, the selection it made) of the last bar pick
        self._gesture = None
        self._bars = {}
        self._range_labels = {}
        self._range_buttons = {}

        # The session card's margins: the shadow's room, the padding, and room at the right for the scroll bar.
        margin = ControlsCard.SHADOW
        top = margin - 2
        pad = ControlsCard.PAD
        layout = QVBoxLayout(self)
        layout.setContentsMargins(margin + pad, top + pad - 4, margin + pad - 6, 2 * margin - top + pad)
        layout.setSpacing(10)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.heading = QLabel()
        self.heading.setObjectName("controlsHeading")
        header.addWidget(self.heading)
        header.addStretch(1)
        self.close_button = card_close_button(ControlsCard.CLOSE_SIZE, self.close_card)
        header.addWidget(self.close_button)
        layout.addLayout(header)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("controlsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body.setObjectName("controlsBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 6, 0)
        self.body_layout.setSpacing(SECTION_GAP)
        self.scroll.setWidget(self.body)
        layout.addWidget(self.scroll, 1)
        self._style()
        self._retitle()
        self._rebuild(None)
        for child in self.findChildren(QWidget):
            child.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hide()

    # --- looks ---

    def _style(self):
        t = TOKENS
        self._style_key = (t["surface"], t["text"], t["accent_text"])
        self.setStyleSheet(card_qss(t, ControlsCard.CLOSE_SIZE) + f"""
            QLabel#legendTitle {{ color: {t['text']}; font-weight: 600; }}
            QLabel#legendScheme, QLabel#legendEnd {{ color: {t['text_muted']}; }}
            QLabel#legendRange {{ color: {t['text']}; }}
            QLabel#legendEmpty {{ color: {t['text_soft']}; }}
            QPushButton#legendClear {{
                background: transparent; border: none; border-radius: 5px; color: {t['accent_text']};
                font-weight: 600; padding: 1px 6px; min-height: 0px;
            }}
            QPushButton#legendClear:hover {{ background: {t['tint_hover']}; }}
        """)

    def _retitle(self):
        """The heading's text, in the current language."""
        self.heading.setText(QtCore.QCoreApplication.translate("ViewerLegend", "Legend"))
        self.setAccessibleName(self.heading.text())

    def card_rect(self):
        """Where the card is, inside the shadow's margin (the session card's)."""
        margin = ControlsCard.SHADOW
        return QRectF(margin, margin - 2, self.width() - 2 * margin, self.height() - 2 * margin)

    def paintEvent(self, event):
        painter = QPainter(self)
        paint_card(painter, self.card_rect(), ControlsCard.RADIUS, QColor(WHITE))
        painter.end()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self._retitle()
            self.model = None
            self._seen = None
            self.schedule_refresh()
        super().changeEvent(event)

    # A mouse event no widget in the card took stops here, so the canvas never sees one meant for the card;
    # one in the shadow's margin, which is not the card, goes through to the canvas (as the session card's).
    def _take(self, event, position):
        if self.card_rect().contains(position):
            event.accept()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        self._take(event, event.position())

    def mouseReleaseEvent(self, event):
        self._take(event, event.position())

    def mouseDoubleClickEvent(self, event):
        self._take(event, event.position())

    def mouseMoveEvent(self, event):
        self._take(event, event.position())

    def wheelEvent(self, event):
        self._take(event, event.position())

    def contextMenuEvent(self, event):
        self._take(event, QPointF(event.pos()))

    # --- state ---

    def is_open(self):
        return self.opened

    def toggle(self):
        """Open the card, or close it when it is open (the Legend button)."""
        if self.opened:
            self.close_card()
        else:
            self.open_card()

    def open_card(self):
        """Show the card, reading the key as it is now."""
        if self.opened:
            return
        self.opened = True
        self.refresh()
        if self.toggled is not None:
            self.toggled(True)

    def close_card(self):
        """Hide the card (its x, or a second click on the Legend button)."""
        if not self.opened:
            return
        self.opened = False
        self.hide()
        if self.toggled is not None:
            self.toggled(False)
        self._relayout()

    def _relayout(self):
        if self.relayout is not None:
            self.relayout()

    def schedule_refresh(self):
        """Refresh once at the end of this turn of the event loop, however many times this is called."""
        if self._pending or not self.opened:
            return
        self._pending = True
        QTimer.singleShot(0, self._refresh_now)

    def _refresh_now(self):
        self._pending = False
        try:
            self.refresh()
        except RuntimeError:  # the card was deleted with its window
            pass

    def _fingerprint(self):
        """What the legend is made from, cheaply: a checksum of the colors and the shown-nodes mask, and the key's objects.

        Every command that changes the key makes new arrays and a new list (Node_Colors._begin), and undo and
        redo restore copies, so the key's objects stand for their contents; the colors and the mask are
        changed in place, so their bytes are summed (about 0.3 ms at 44,000 nodes).
        """
        crc = 0
        for name in ("current_colors", "visible_mask"):
            array = getattr(self.viewer, name, None)
            if isinstance(array, np.ndarray):
                crc = zlib.crc32(array if array.flags.c_contiguous else np.ascontiguousarray(array), crc)
        return crc, tuple(getattr(self.viewer, name, None) for name in KEY_ATTRIBUTES)

    def refresh(self):
        """Read the key again and show it. Nothing is counted or rebuilt when nothing it is made from changed."""
        self._pending = False
        seen = self._fingerprint()
        t = TOKENS
        style_now = (t["surface"], t["text"], t["accent_text"])
        before = self._seen
        if (before is not None and style_now == self._style_key and seen[0] == before[0]
                and all(a is b for a, b in zip(seen[1], before[1]))):
            self._check_range()
            return
        self._seen = seen
        model = legend_model(self.viewer)
        self._check_range()
        if style_now != self._style_key:
            self._style()
            self._retitle()
            model_changed = True
        else:
            model_changed = model != self.model
        if model_changed:
            self.model = model
            self._rebuild(model)
        self._relayout()

    def _check_range(self):
        """Forget the range a bar showed once the selection is something else than the pick made."""
        if self._range is None or self._gesture is not None:
            return
        if frozenset(int(i) for i in getattr(self.viewer, "selected_indices", ())) != self._range[3]:
            self._range = None
            self._show_ranges()

    # --- building ---

    def _rebuild(self, model):
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._bars = {}
        self._range_labels = {}
        self._range_buttons = {}
        if model is None:
            empty = QLabel(QtCore.QCoreApplication.translate("ViewerLegend", "No colours have been described yet."))
            empty.setObjectName("legendEmpty")
            empty.setWordWrap(True)
            self.body_layout.addWidget(empty)
            self.body_layout.addStretch(1)
            self._show_built()
            return
        for layer in model["layers"]:
            self._add_section(self._layer_group(layer))
        if model["unlabelled"]:
            group = self._group(QtCore.QCoreApplication.translate("ViewerLegend", "Unlabelled colours"))
            rows = [(entry["rgb"], entry["hex"], entry["count"]) for entry in model["unlabelled"]]
            rowlist = _RowList(rows, QtCore.QCoreApplication.translate("ViewerLegend", "Select the nodes of this color"))
            rowlist.activated.connect(lambda index, ctrl, found=model["unlabelled"]:
                                      self._pick_row(nodes_of_colour(self.viewer, found[index]["cls"]), ctrl))
            group.layout().addWidget(rowlist)
            self._add_section(group)
        self.body_layout.addStretch(1)
        self._show_built()
        self._show_ranges()

    def _show_built(self):
        """Show what _rebuild put in the body, now.

        A widget added to a layout whose window is shown stays hidden until a queued event shows it, and a
        layout counts a hidden widget as nothing: until then body.sizeHint() is 0 or only the first
        sections, so natural_height() (and the card's height placed from it) was far too short right after
        a rebuild. Showing the sections here makes the layout see them at once. (The range labels and
        Clear buttons were hidden on purpose and stay so; _show_ranges decides them.)
        """
        for index in range(self.body_layout.count()):
            widget = self.body_layout.itemAt(index).widget()
            if widget is not None:
                widget.show()
        self.body_layout.invalidate()

    def _add_section(self, section):
        """Add a section to the body, a separator before it unless it is the first (as the session card's sections are)."""
        if self.body_layout.count():
            self.body_layout.addWidget(card_separator())
        self.body_layout.addWidget(section)

    @staticmethod
    def _group(title):
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(ControlsCard.ROW_GAP - 2)
        label = _ElidedLabel(title)
        label.setObjectName("legendTitle")
        layout.addWidget(label)
        return group

    def _layer_group(self, layer):
        group = self._group(layer["title"])
        layout = group.layout()
        if layer["kind"] == "gradient":
            if layer["scheme"]:
                scheme = _ElidedLabel(layer["scheme"])
                scheme.setObjectName("legendScheme")
                layout.addWidget(scheme)
            if layer["bar"] and layer["min"] is not None and layer["max"] is not None:
                layout.addWidget(self._bar_block(layer))
        if layer["rows"]:
            rows = [(row["rgb"], row["label"], row["count"]) for row in layer["rows"]]
            rowlist = _RowList(rows, QtCore.QCoreApplication.translate("ViewerLegend", "Select these nodes"))
            rowlist.activated.connect(lambda index, ctrl, layer=layer:
                                      self._pick_row(nodes_of_row(self.viewer, layer["id"], layer["rows"][index]["slot"]), ctrl))
            layout.addWidget(rowlist)
        return group

    def _bar_block(self, layer):
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        bar = _GradientBar(layer["stops"], QtCore.QCoreApplication.translate("ViewerLegend", "Press or drag to select the nodes with a value in that range"))
        bar.picked.connect(lambda a, b, ctrl, layer=layer: self._pick_range(layer, a, b, ctrl))
        bar.released.connect(self._finish_range)
        layout.addWidget(bar)
        ends = QHBoxLayout()
        ends.setContentsMargins(0, 0, 0, 0)
        low = _ElidedLabel(number(layer["min"]))
        high = _ElidedLabel(number(layer["max"]), align=Qt.AlignmentFlag.AlignRight)
        low.setObjectName("legendEnd")
        high.setObjectName("legendEnd")
        ends.addWidget(low, 1)
        ends.addWidget(high, 1)
        layout.addLayout(ends)
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        label = _ElidedLabel()
        label.setObjectName("legendRange")
        clear = QPushButton(QtCore.QCoreApplication.translate("ViewerLegend", "Clear"))
        clear.setObjectName("legendClear")
        clear.setToolTip(QtCore.QCoreApplication.translate("ViewerLegend", "Clear the selection"))
        clear.setCursor(Qt.CursorShape.PointingHandCursor)
        clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear.clicked.connect(self._clear_selection)
        line.addWidget(label, 1)
        line.addWidget(clear)
        layout.addLayout(line)
        label.hide()
        clear.hide()
        self._bars[layer["id"]] = (bar, layer)
        self._range_labels[layer["id"]] = label
        self._range_buttons[layer["id"]] = clear
        return block

    def _show_ranges(self):
        """Show the range of the last bar pick on its bar (and nowhere else)."""
        for layer_id, (bar, layer) in self._bars.items():
            label, clear = self._range_labels[layer_id], self._range_buttons[layer_id]
            picked = self._range if self._range is not None and self._range[0] == layer_id else None
            label.setVisible(picked is not None)
            clear.setVisible(picked is not None)
            if picked is None:
                bar.show_band(None)
                continue
            low, high = picked[1], picked[2]
            span = layer["max"] - layer["min"]
            band = ((low - layer["min"]) / span, (high - layer["min"]) / span) if span > 0 else (0.0, 1.0)
            bar.show_band((min(1.0, max(0.0, band[0])), min(1.0, max(0.0, band[1]))))
            label.setText(QtCore.QCoreApplication.translate("ViewerLegend", "{low} to {high}").format(low=number(low), high=number(high)))
        self._relayout()

    # --- what a click does ---

    def _selected(self):
        return frozenset(int(i) for i in getattr(self.viewer, "selected_indices", ()))

    def _pick_row(self, nodes, ctrl):
        """A click on a row: its nodes replace the selection, Ctrl adds them, the same row again clears it."""
        nodes = frozenset(int(i) for i in nodes)
        current = self._selected()
        if not nodes:
            return
        if ctrl:
            result = current | nodes
        elif nodes == current:
            result = frozenset()
        else:
            result = nodes
        self._range = None
        if result == current:
            self._show_ranges()
            return
        self.viewer.select_by_hand(sorted(result))
        self._show_ranges()

    def _pick_range(self, layer, a, b, ctrl):
        """A press or drag on a bar: the nodes of its range are the selection, as the drag goes."""
        if self._gesture is None or self._gesture["layer"] != layer["id"]:
            start = self._selected()
            self._gesture = {"layer": layer["id"], "start": start, "base": start if ctrl else frozenset()}
        low_share = min(1.0, max(0.0, min(a, b)))
        high_share = min(1.0, max(0.0, max(a, b)))
        span = layer["max"] - layer["min"]
        low, high = layer["min"] + span * low_share, layer["min"] + span * high_share
        result = self._gesture["base"] | frozenset(int(i) for i in nodes_in_range(self.viewer, layer["id"], low, high))
        if result != self._selected():
            self.viewer.select_by_hand(sorted(result), announce=False)
        self._range = (layer["id"], low, high, result)
        self._show_ranges()

    def _finish_range(self):
        """The bar was let go: the selection is told once, if the gesture changed it."""
        gesture, self._gesture = self._gesture, None
        if gesture is not None and self._selected() != gesture["start"]:
            self.viewer.announce_hand_selection(bool(gesture["start"]))
        self._check_range()

    def _clear_selection(self):
        if self._selected():
            self._range = None
            self.viewer.select_by_hand([])
            self._show_ranges()

    # --- where it sits ---

    def natural_height(self, width=None):
        """The height the card would like (its widget's, shadow included) at the card width ``width``: the heading's
        row and everything under it, a line that wraps taking its lines."""
        self.body_layout.activate()
        margins = self.layout().contentsMargins()
        body = self.body.sizeHint().height()
        if width is not None and self.body_layout.hasHeightForWidth():
            inner = width + 2 * ControlsCard.SHADOW - margins.left() - margins.right()
            body = max(body, self.body_layout.totalHeightForWidth(max(1, inner)))
        return margins.top() + ControlsCard.CLOSE_SIZE + self.layout().spacing() + body + margins.bottom()

    def minimum_card_height(self):
        """The least height the card shrinks to, its heading's row alone (card, not widget)."""
        margins = self.layout().contentsMargins()
        return margins.top() + ControlsCard.CLOSE_SIZE + margins.bottom() - 2 * ControlsCard.SHADOW

    def card_width(self, room):
        """The card's width in a window that leaves it ``room`` px: LEGEND_WIDTH, or less down to LEGEND_MIN_WIDTH."""
        return max(LEGEND_MIN_WIDTH, min(LEGEND_WIDTH, int(room)))

    def place(self, right, top, bottom, room):
        """Put the card's right edge at ``right`` and top at ``top`` (canvas px), no lower than ``bottom``, no wider than ``room``.

        These are the card's own edges, inside its shadow: the widget is a little larger all round.
        """
        if not self.opened:
            self.hide()
            return
        shadow = ControlsCard.SHADOW
        width = self.card_width(room)
        wanted = self.natural_height(width) - 2 * shadow
        height = max(self.minimum_card_height(), min(wanted, int(bottom - top)))
        target = QtCore.QRect(int(round(right)) - width - shadow, int(round(top)) - (shadow - 2),
                              width + 2 * shadow, height + 2 * shadow)
        if self.geometry() != target:
            self.setGeometry(target)
        if self.isHidden():
            self.show()
        self.raise_()

    def visible_rect(self):
        """The card's rectangle in the canvas's px (inside its shadow), or None while it is closed."""
        if not self.opened or self.isHidden():
            return None
        return self.card_rect().translated(QPointF(self.pos()))


__all__ = ["LegendCard", "legend_model", "nodes_in_range", "nodes_of_colour", "nodes_of_row", "number"]
