# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Widgets of the Viewer's window: its controls card, and its sidebar's card and buttons.

* The controls. The Viewer lists its mouse and keyboard actions in one
  translated text, "[ENTER] Command | [LeftClick] Highlight | ...": the
  actions are separated by " | ", and each gives its keys in brackets, joined
  by " + ", then what they do. controls_rows() reads that text into rows.
  ControlsCard shows the rows as key caps beside what they do, on a card in a
  light tint of the theme colour. ControlsButton, at the start of the
  threshold slider, opens the card above itself. A click outside the card,
  Esc, the card's × or the button again closes it. The button never takes
  the keyboard, so [ENTER] still opens the command line.
* The sidebar. SidebarPanel is the panel over the canvas's right edge, drawn
  as a card with round corners, SIDEBAR_INSET clear of the canvas's top,
  right and bottom edges. SidebarButton is one of its buttons: the icon at
  the start, the label centred in the room after it.
"""

from __future__ import annotations

import re
import sys

from PySide6 import QtCore
from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from desktop.Desktop_App import button_icon_room
from desktop.Studio_Theme import TOKENS, studio_icon

# What separates the actions in the Viewer's text, and the keys of one action.
ACTION_SEPARATOR = " | "
KEY_SEPARATOR = " + "
_ACTION = re.compile(r"\[([^\]]+)\]\s*(.*)", re.DOTALL)


def controls_rows(text):
    """The actions listed in ``text``, as (keys, what they do) pairs.

    "[LeftClick + Drag] Pan" gives (["LeftClick", "Drag"], "Pan"); an action
    without bracketed keys is a row with no keys, its whole text the description.
    """
    rows = []
    for action in text.split(ACTION_SEPARATOR):
        action = action.strip()
        if not action:
            continue
        match = _ACTION.fullmatch(action)
        if match is None:
            rows.append(([], action))
            continue
        keys = [key.strip() for key in match.group(1).split(KEY_SEPARATOR) if key.strip()]
        rows.append((keys, match.group(2).strip()))
    return rows


class ControlsCard(QWidget):
    """The Viewer's actions on a card that pops up: key caps beside what they do.

    title heads the card; text is the Viewer's list of actions (controls_rows).
    anchor is the button that opened it: a press on it closes the card without
    reaching the button, so a second click closes the card instead of opening
    it again.
    """

    PAD = 14          # inside the card
    RADIUS = 12       # the card's corners, a card's
    ROW_GAP = 8       # between rows
    COLUMN_GAP = 16   # between the keys and what they do
    KEY_GAP = 4       # between a key cap and the "+" beside it
    CLOSE_SIZE = 22   # the × button
    ANCHOR_GAP = 16   # between the card and the button it opens from, clear of the slider
    # Where the window system composes translucent windows, the card has round
    # corners and paints its own shadow in this margin; elsewhere it fills the
    # window (as Desktop_App.ThemeColorPalette).
    SHADOW = 8

    def __init__(self, title, text, anchor=None, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self.setObjectName("viewerControlsCard")
        self.anchor = anchor
        self._translucent = (sys.platform in ("win32", "darwin")
                             or QApplication.platformName() in ("wayland", "offscreen"))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, self._translucent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAccessibleName(title)
        self.rows = controls_rows(text)

        margin = self.SHADOW if self._translucent else 0
        layout = QVBoxLayout(self)
        # The shadow falls a little below the card, so the card sits a little higher.
        top = margin - 2 if self._translucent else 0
        layout.setContentsMargins(margin + self.PAD, top + self.PAD - 4,
                                  margin + self.PAD - 6, 2 * margin - top + self.PAD)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("controlsTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.close_button = QPushButton()
        self.close_button.setObjectName("controlsClose")
        self.close_button.setIcon(studio_icon("x", TOKENS["text_muted"]))
        self.close_button.setIconSize(QSize(14, 14))
        self.close_button.setFixedSize(self.CLOSE_SIZE, self.CLOSE_SIZE)
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_text = QtCore.QCoreApplication.translate("ViewerControls", "Close")
        self.close_button.setToolTip(close_text)
        self.close_button.setAccessibleName(close_text)
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 6, 0)
        grid.setHorizontalSpacing(self.COLUMN_GAP)
        grid.setVerticalSpacing(self.ROW_GAP)
        self.key_caps = []
        self.descriptions = []
        for row, (keys, description) in enumerate(self.rows):
            grid.addWidget(self._keys_widget(keys), row, 0, Qt.AlignmentFlag.AlignLeft)
            label = QLabel(description)
            label.setObjectName("controlsDescription")
            self.descriptions.append(label)
            grid.addWidget(label, row, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        t = TOKENS
        self.setStyleSheet(f"""
            QLabel#controlsTitle {{ font-weight: 600; color: {t['text']}; padding: 0px; }}
            QLabel#controlsDescription {{ color: {t['text_soft']}; }}
            QLabel#controlsPlus {{ color: {t['text_subtle']}; }}
            QLabel#controlsKey {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['field_border']}; border-bottom: 2px solid {t['field_border_hover']};
                border-radius: 5px; padding: 0px 6px;
            }}
            QPushButton#controlsClose {{
                background: transparent; border: none; border-radius: 6px; padding: 0px; min-height: 0px;
            }}
            QPushButton#controlsClose:hover {{ background: {t['tint_hover']}; }}
            QPushButton#controlsClose:pressed {{ background: {t['separator']}; }}
        """)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())

    def _keys_widget(self, keys):
        """The key caps of one action, a "+" between each two."""
        holder = QWidget()
        holder.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(self.KEY_GAP)
        for index, key in enumerate(keys):
            if index:
                plus = QLabel("+")
                plus.setObjectName("controlsPlus")
                row.addWidget(plus)
            cap = QLabel(key)
            cap.setObjectName("controlsKey")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.key_caps.append(cap)
            row.addWidget(cap)
        return holder

    def card_rect(self):
        """Where the card is, inside the shadow's margin when there is one."""
        if not self._translucent:
            return QRectF(self.rect())
        margin = self.SHADOW
        return QRectF(margin, margin - 2, self.width() - 2 * margin, self.height() - 2 * margin)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        card = self.card_rect()
        radius = self.RADIUS if self._translucent else 0
        if self._translucent:
            painter.setPen(Qt.PenStyle.NoPen)
            for grow, alpha in ((7.0, 4), (5.0, 6), (3.0, 9), (1.5, 12)):
                painter.setBrush(QColor(9, 9, 11, alpha))
                painter.drawRoundedRect(card.adjusted(-grow, -grow + 2, grow, grow + 2),
                                        radius + grow, radius + grow)
        painter.setPen(QPen(QColor(TOKENS["tint_hover"]), 1.0))
        painter.setBrush(QColor(TOKENS["tint"]))
        painter.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        painter.end()

    def open_above(self, anchor_widget, gap=ANCHOR_GAP):
        """Show the card above anchor_widget, their left edges lined up, or below it
        where the screen has no room above; returns the card."""
        card = self.card_rect()
        corner = anchor_widget.mapToGlobal(QtCore.QPoint(0, 0))
        x = corner.x() - round(card.left())
        y = corner.y() - gap - round(card.bottom())
        screen = anchor_widget.screen().availableGeometry()
        if y + card.top() < screen.top():
            y = corner.y() + anchor_widget.height() + gap - round(card.top())
        self.move(max(screen.left(), min(x, screen.right() - self.width())), y)
        self.show()
        self.setFocus()
        return self

    def mousePressEvent(self, event):
        if self.card_rect().contains(event.position()):
            super().mousePressEvent(event)
            return
        anchor = self.anchor
        if anchor is not None and anchor.isVisible() and anchor.rect().contains(
                anchor.mapFromGlobal(event.globalPosition().toPoint())):
            # The press would reach the button once the card closes and open it
            # again; the button's press only closes it.
            self.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay, True)
        self.close()


class ControlsButton(QPushButton):
    """The info button that opens the Viewer's controls card above itself.

    title is the card's heading and the button's tooltip; text the Viewer's
    list of actions. A click while the card is open closes it.
    """

    SIZE = 28   # as tall as a field

    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setObjectName("viewerControlsButton")
        self.title = title
        self.text_source = text
        self.card = None
        self.setIcon(studio_icon("info", TOKENS["text_muted"]))
        self.setIconSize(QSize(16, 16))
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setToolTip(title)
        self.setAccessibleName(title)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # The canvas keeps the keyboard: [ENTER] opens the command line.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        t = TOKENS
        self.setStyleSheet(f"""
            QPushButton#viewerControlsButton {{
                background: {t['tint']}; border: 1px solid {t['tint']};
                border-radius: {t['radius']}px; padding: 0px; min-height: 0px;
            }}
            QPushButton#viewerControlsButton:hover {{
                background: {t['tint_hover']}; border-color: {t['tint_hover']};
            }}
            QPushButton#viewerControlsButton:pressed {{
                background: {t['separator']}; border-color: {t['separator']};
            }}
        """)
        self.clicked.connect(self.toggle_card)

    def card_is_open(self):
        card = self.card
        try:
            return card is not None and card.isVisible()
        except RuntimeError:  # deleted on close
            return False

    def toggle_card(self):
        """Open the controls card above the button, or close it if it is open; returns it or None."""
        if self.card_is_open():
            self.card.close()
            return None
        self.card = ControlsCard(self.title, self.text_source, anchor=self, parent=self)
        return self.card.open_above(self)


# =====================================================================
# The Sidebar
# =====================================================================

# How far the sidebar's card stays from the canvas's top, right and bottom edges.
SIDEBAR_INSET = 8
# The sidebar card's fill: the surface, a touch see-through over the network.
SIDEBAR_OPACITY = 0.95


class SidebarPanel(QWidget):
    """The sidebar over the canvas's right edge: a card with round corners.

    The widget fills the sidebar's room down the canvas; the card leaves
    SIDEBAR_INSET of it clear at the top, right and bottom, and its layout's
    margins keep the buttons on the card.
    """

    def card_rect(self):
        """Where the card is in the panel."""
        return QRectF(self.rect()).adjusted(0.5, SIDEBAR_INSET + 0.5, -SIDEBAR_INSET - 0.5,
                                            -SIDEBAR_INSET - 0.5)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fill = QColor(TOKENS["surface"])
        fill.setAlphaF(SIDEBAR_OPACITY)
        painter.setPen(QPen(QColor(TOKENS["border"]), 1.0))
        painter.setBrush(fill)
        radius = TOKENS["radius_card"]
        painter.drawRoundedRect(self.card_rect(), radius, radius)
        painter.end()


class SidebarButton(QPushButton):
    """A sidebar button: its icon at the start, its label centred in the room after the icon.

    The room is the style's contents rectangle less the icon's
    (Desktop_App.button_icon_room), the room fit_buttons_to_text gives every
    label, so a label never clips; a shorter label sits in its middle. The
    button's frame, hover and focus ring are the stylesheet's.
    """

    def label_rect(self):
        """Where the label is centred: the contents after the icon."""
        option = QStyleOptionButton()
        self.initStyleOption(option)
        contents = self.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, self)
        return contents.adjusted(button_icon_room(self), 0, 0, 0)

    def icon_rect(self):
        """Where the icon is: at the start of the contents, centred down the button."""
        option = QStyleOptionButton()
        self.initStyleOption(option)
        contents = self.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, self)
        size = self.iconSize()
        return QRect(contents.left(), contents.top() + (contents.height() - size.height()) // 2,
                     size.width(), size.height())

    def paintEvent(self, event):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        label = option.text
        option.text = ""
        option.icon = QIcon()
        painter = QStylePainter(self)
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)
        if not self.icon().isNull():
            mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
            self.icon().paint(painter, self.icon_rect(), Qt.AlignmentFlag.AlignCenter, mode)
        painter.drawItemText(
            self.label_rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextShowMnemonic,
            option.palette, self.isEnabled(), label, QPalette.ColorRole.ButtonText,
        )
        painter.end()


__all__ = [
    "ControlsButton",
    "ControlsCard",
    "SIDEBAR_INSET",
    "SidebarButton",
    "SidebarPanel",
    "controls_rows",
]
