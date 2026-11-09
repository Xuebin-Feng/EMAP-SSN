# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's Undo and Redo buttons, over the canvas's top right, for a user who does not use the keys.

* **The buttons** are two icon buttons (Lucide undo-2 and redo-2) in the frame-rate
  row, Undo to the left of Redo. When the Viewer gives a ``legend`` action there is a
  third, always there, left of Undo (the Lucide palette icon): it opens and closes the
  legend card (desktop/Viewer_Legend.py) and shows a checked look while the card is open
  (show_legend_open). It is one of the row's buttons, so the row's width, which the frame
  rate and the command line are placed against, counts it. A click does what Ctrl+Z or Ctrl+Y does: the Viewer
  gives them its own undo_by_hand and redo_by_hand, so the status line's messages, and
  the undo that takes back a command that waits or runs, are the keys' too. The frame
  rate, when shown, and the sidebar's toggle keep their places beside them
  (MainViewer.reposition_expand_btn).
* **What is enabled** is what the keys could do now (history_availability): Undo while
  there is an undo step or a command waits or runs, Redo while there is a step to redo
  and no command waits. The Viewer tells the buttons wherever the history or the queue
  changes (MainViewer._refresh_history_buttons), and nothing polls: an idle Viewer runs
  nothing for them. A press on a node saves a snapshot at once, in case a drag follows,
  and takes it back at the release if nothing moved: the buttons leave it out until the
  nodes move, so a plain click does not flash them.
* **They never move** on a state change. Qt draws the whole network again whenever a
  widget over the canvas is moved or resized, but not when a widget only repaints
  (see Viewer_Caret), so place() puts the buttons only where the other overlays are
  placed, and show_state() changes nothing but the enabled state, which repaints a
  button and nothing else.
* **They are widgets over the canvas**, as the frame rate and the jobs badge are, not
  VisPy visuals: canvas.render() (print, MCP captures, the edge cache) draws the
  scene only, so they are never in a picture. They take no keyboard focus, so typing
  still goes to the canvas and the command line.
"""

from __future__ import annotations

from PySide6 import QtCore
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QPushButton

from desktop.Studio_Theme import TOKENS, studio_icon

# A button's side, in px: the height of the frame-rate row, and of the sidebar toggle in it.
HISTORY_BUTTON_SIZE = 30
# Between Undo and Redo, in px.
HISTORY_BUTTON_GAP = 4
ICON_SIZE = 16


def history_availability(viewer, history=None, redo=None):
    """(Undo can act, Redo can act) as Ctrl+Z and Ctrl+Y find the Viewer now.

    Undo takes back the newest command that waits or runs, else the newest history
    step; Redo has nothing to redo while a command waits, since it comes after the
    undone steps (MainViewer.undo_by_hand, redo_by_hand). ``history`` and ``redo``
    are the steps to count in place of the Viewer's own: the Viewer gives them while
    a press on a node holds a snapshot that is no step yet.
    """
    queue = getattr(viewer, 'command_queue', None)
    waiting = queue is not None and bool(queue.pending())
    if history is None:
        history = getattr(viewer, 'position_history', None)
    if redo is None:
        redo = getattr(viewer, 'redo_stack', None)
    can_undo = waiting or bool(history)
    can_redo = not waiting and bool(redo)
    return can_undo, can_redo


class HistoryButton(QPushButton):
    """One icon button of the pair: tinted like the info button, a muted look while it has nothing to do."""

    def __init__(self, parent, object_name, icon_name, text, clicked):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setIcon(studio_icon(icon_name, TOKENS["text_muted"]))
        self.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.setFixedSize(HISTORY_BUTTON_SIZE, HISTORY_BUTTON_SIZE)
        self.setToolTip(text)
        self.setAccessibleName(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # The canvas keeps the keyboard: [ENTER] opens the command line, and typing goes on in it.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        t = TOKENS
        self.setStyleSheet(f"""
            QPushButton#{object_name} {{
                background: {t['tint']}; border: 1px solid {t['tint']};
                border-radius: {t['radius']}px; padding: 0px; min-height: 0px;
            }}
            QPushButton#{object_name}:hover {{
                background: {t['tint_hover']}; border-color: {t['tint_hover']};
            }}
            QPushButton#{object_name}:pressed {{
                background: {t['separator']}; border-color: {t['separator']};
            }}
            QPushButton#{object_name}:checked {{
                background: {t['separator']}; border-color: {t['accent_soft']};
            }}
            QPushButton#{object_name}:disabled {{
                background: {t['field_disabled']}; border-color: {t['border']};
            }}
        """)
        self.clicked.connect(lambda _checked=False: clicked())

    def show_checked(self, checked):
        """Show the checked look (a toggle button whose action is open), changing nothing else."""
        checked = bool(checked)
        if checked != self.isChecked():
            self.setChecked(checked)

    def show_enabled(self, enabled):
        """Enable or disable the button, changing nothing else: it keeps its place and size."""
        enabled = bool(enabled)
        if enabled == self.isEnabled():
            return
        self.setEnabled(enabled)
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor)


class HistoryButtons:
    """The Undo and Redo buttons, and the legend's, on the VisPy canvas ``canvas``'s window.

    undo() and redo() are what a click calls; legend(), when given, is what a click on the legend button calls.
    """

    def __init__(self, canvas, undo, redo, legend=None):
        self.legend = None
        if legend is not None:
            self.legend = HistoryButton(
                canvas.native, "viewerLegendButton", "palette",
                QtCore.QCoreApplication.translate("ViewerLegend", "Legend"), legend)
            self.legend.setCheckable(True)
        self.undo = HistoryButton(
            canvas.native, "viewerUndoButton", "undo-2",
            QtCore.QCoreApplication.translate("ViewerHistory", "Undo (Ctrl+Z)"), undo)
        self.redo = HistoryButton(
            canvas.native, "viewerRedoButton", "redo-2",
            QtCore.QCoreApplication.translate("ViewerHistory", "Redo (Ctrl+Y)"), redo)
        self.show_state(False, False)
        for button in self.buttons():
            button.show()

    def buttons(self):
        """The buttons, left to right: the legend's when there is one, Undo, Redo."""
        return ((self.legend,) if self.legend is not None else ()) + (self.undo, self.redo)

    def width(self):
        """The buttons' width together, in px: the room they take in the row."""
        count = len(self.buttons())
        return count * HISTORY_BUTTON_SIZE + (count - 1) * HISTORY_BUTTON_GAP

    def place(self, right, centre_y):
        """Put the buttons' right edge at right, centred on centre_y (canvas px). A button already there is not moved."""
        right = int(round(right))
        top = int(round(centre_y - HISTORY_BUTTON_SIZE / 2))
        step = HISTORY_BUTTON_SIZE + HISTORY_BUTTON_GAP
        buttons = self.buttons()
        for index, button in enumerate(buttons):
            left = right - HISTORY_BUTTON_SIZE - (len(buttons) - 1 - index) * step
            target = QtCore.QRect(left, top, HISTORY_BUTTON_SIZE, HISTORY_BUTTON_SIZE)
            if button.geometry() != target:
                button.setGeometry(target)

    def show_legend_open(self, is_open):
        """Show the legend button checked while its card is open."""
        if self.legend is not None:
            self.legend.show_checked(is_open)

    def show_state(self, can_undo, can_redo):
        """Enable each button as its action can be done; neither moves."""
        self.undo.show_enabled(can_undo)
        self.redo.show_enabled(can_redo)
