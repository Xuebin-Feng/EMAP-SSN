# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Widgets of the Viewer's window: its info card, and its sidebar's card and buttons.

* The info card. The Viewer lists its mouse and keyboard actions in one
  translated text, "[ENTER] Command | [LeftClick] Highlight | ...": the
  actions are separated by " | ", and each gives its keys in brackets, joined
  by " + ", then what they do. controls_rows() reads that text into rows.
  ControlsCard shows the session, then the rows as key caps beside what they
  do, on a card in a light tint of the theme colour. The session is the
  network's nodes and edges, on one line, and its files, each in the form of
  the Config's Directories tab: a label and a read-only field holding the path
  written with the alias of the folder it is in ($input_file$/...). The sequence
  set, the network and the layout cache are fixed for the session, and their
  fields are quiet and carry no button; the alignment and the embedding can be
  changed, below a small gap, and their button runs the command of that name
  with no argument, as typed, which opens its file dialog. The card's two
  headings, "Session" and "Mouse and keyboard controls", are in the text
  colour; the link to the list of commands is in the first one's row, beside
  the ×. ControlsButton, at the start of the threshold slider, opens the card
  above itself, and ConsoleButton, after it, opens the Console.
  The card is a child of the canvas, so the network, the sliders, the Zoom
  box and the sidebar work as usual while it is open: it closes only
  by its × or by the button again, never by a click elsewhere, and it never
  takes the keyboard, so [ENTER] still opens the command line. Mouse events
  on the card stop at the card. The fields have no keyboard focus to receive
  Copy and Select All, so the card hears those two keys on the canvas, which
  has the keyboard, and answers them for the field that has the selection
  (ControlsCard.eventFilter); every other key, and those two while the
  command line is open, go on to the canvas as before.
* The sidebar. SidebarPanel is the panel over the canvas's right edge, drawn
  as a card with round corners, SIDEBAR_INSET clear of the canvas's top,
  right and bottom edges. SidebarButton is one of its buttons: the icon at
  the start, the label centred in the room after it.
* The view's zoom. ZoomBox, at the canvas's bottom right level with the
  threshold slider, is the translated "Zoom: {zoom}%" with a ZoomSpinBox in
  place of {zoom}: it shows the view's zoom, a percentage, and sets it. The
  percent sign is the text after the box, not part of it.
* The threshold. ThresholdSpinBox, before the threshold slider, shows the
  edge threshold and sets it. Each of its numbers is one of the slider's
  positions, so the two follow each other.

Both boxes are CanvasSpinBoxes: number boxes over the network that take the
keyboard only when clicked.
"""

from __future__ import annotations

import math
import os
import re

from PySide6 import QtCore
from PySide6.QtCore import QT_TRANSLATE_NOOP, QEvent, QRect, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from commands.zoom import MAX_ZOOM_PERCENT
from desktop.Desktop_App import UI_QSS_FONT_STACK, button_icon_room, fit_buttons_to_text, installed_language
from desktop.Studio_Theme import TOKENS, icon_button, key_cap_qss, studio_icon
from desktop.Viewer_Jobs_Card import JobsSection, section_qss
from desktop.Viewer_State import alias_form
from utilities import Desktop_Open
from utilities.Help_Pages import command_list_path

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


# The session card's files, in its order: the key of ViewerInspectionService.session_overview()'s
# "inputs", and the label of its row, as the Config's Directories tab labels a folder. The first
# three are fixed for the session; the last two can be changed from the card.
SESSION_FILES = (
    ("node_fasta", QT_TRANSLATE_NOOP("ViewerControls", "Sequence Set:")),
    ("network_hdf5", QT_TRANSLATE_NOOP("ViewerControls", "Network:")),
    ("layout_cache", QT_TRANSLATE_NOOP("ViewerControls", "Layout Cache:")),
    ("msa", QT_TRANSLATE_NOOP("ViewerControls", "Alignment:")),
    ("embedding", QT_TRANSLATE_NOOP("ViewerControls", "Embedding:")),
)
# The files the card can change: the command its pick button runs, typed with no argument (it opens
# the command's file dialog), and the button's tooltip.
PICK_FILES = {
    "msa": ("alignment", QT_TRANSLATE_NOOP("ViewerControls", "Choose another alignment file…")),
    "embedding": ("embedding", QT_TRANSLATE_NOOP("ViewerControls", "Choose another embedding file…")),
}
# The line the tooltip of a fixed file's field adds to its path.
FIXED_NOTE = QT_TRANSLATE_NOOP("ViewerControls", "Fixed for this Viewer session.")


def display_path(value):
    """A file's full path, as the card keeps it for its tooltip: absolute and normalised, or "" for none.

    The Viewer's settings hold every input already resolved (a $input_file$
    alias is expanded, desktop.Viewer_State.normalize_viewer_paths), so this
    only tidies; an empty setting, such as no alignment, is no path. The
    field shows it written back with the alias (desktop.Viewer_State.alias_form).
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text == "None":
        return ""
    return os.path.normpath(os.path.abspath(text))


class PathField(QLineEdit):
    """A path in a read-only field: it can be selected and copied, never takes the keyboard.

    The mouse selects (a drag, a double or triple click) and the right-click
    menu copies, as in any field. The standard Copy and Select All keys reach
    the field's selection through its card (ControlsCard.eventFilter), not
    through the field, which never has the keyboard to receive them. The
    field starts at the start of its text, as the Config's fields do. The
    wheel over it scrolls a text too long for the field, and the tooltip is
    the whole path, which may be written in full when the field shows it with
    an alias, and then a note. An empty field shows its placeholder, muted. A
    ``fixed`` field, one the user cannot change, is drawn quieter (its
    "fixed" property, for the card's stylesheet) but selects and copies the
    same. ``pressed`` is sent after every mouse press on the field.
    """

    WHEEL_STEP = 4   # characters a notch of the wheel scrolls

    pressed = QtCore.Signal()

    def __init__(self, parent=None, *, fixed=False):
        super().__init__(parent)
        self.setObjectName("controlsPath")
        self.fixed = fixed
        self.setProperty("fixed", fixed)
        self.setReadOnly(True)
        # The keyboard stays with the network, so [ENTER] still opens the
        # command line; the mouse still selects, and the context menu copies.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._scroll = 0
        self.setPlaceholderText(QtCore.QCoreApplication.translate("ViewerControls", "None"))

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.pressed.emit()

    def show_path(self, path, shown=None, note=None):
        """Show ``shown`` (the path itself when there is none) from its start; the tooltip is the path, then ``note``."""
        self.setText(path if shown is None else shown)
        self.setToolTip(f"{path}\n{note}" if path and note else path)
        self.scroll_to(0)

    def scroll_to(self, index):
        """Scroll so character ``index`` is the first shown, or as near the end as the field allows."""
        self._scroll = max(0, min(len(self.text()), index))
        # The line edit scrolls only as far as keeps its cursor in view: go to
        # the end, where the least is hidden, then back to the character.
        self.setCursorPosition(len(self.text()))
        self.setCursorPosition(self._scroll)

    def wheelEvent(self, event):
        delta = event.angleDelta()
        notches = (delta.y() or delta.x()) / 120.0
        if notches and self.text():
            self.scroll_to(self._scroll - round(notches * self.WHEEL_STEP))
        event.accept()


class ControlsCard(QWidget):
    """The Viewer's session and controls on a card over the canvas.

    title names the card (its accessible name) and is not shown: the card
    has two headings, "Session" and controls_title, in the text colour. The
    session section, when ``session`` is given, lists the network's nodes and
    edges on one line and the files it was built from in the form of the
    Config's Directories tab: a label and a read-only field holding the path
    written with the alias of the folder it is in. The sequence set, network
    and layout cache are fixed for the session: their fields are quiet, have no
    button, and their tooltip says so after the full path. The alignment and
    the embedding (PICK_FILES) come after a small gap and can be changed: the
    button after each calls ``command`` (a function taking a Viewer command
    line, as process_command does) with "alignment" or "embedding" and nothing
    else, as if it were typed, which opens that command's file dialog, and then
    leaves the keyboard with the canvas; the row shows the new file when the
    owner refreshes the card. ``session`` is a function returning
    ViewerInspectionService.session_overview()'s dictionary (its "inputs" are
    the files' absolute paths, its "directories" and "project_root" what the
    aliases are written from); refresh() reads it again, and does only the
    work what changed needs. The controls section, headed controls_title, is
    text, the Viewer's list of actions (controls_rows): key caps in the left
    half of the section, what they do in the right half. The link to the list
    of commands is in the row of the first heading, beside the ×, and so is
    Without a session the card has the controls only, headed controls_title
    (or title). The Console has no link on the card: its button is ConsoleButton,
    in the slider row.

    The card is a child of the canvas, not a window of its own, so it leaves
    the network, the sliders and the sidebar working under the mouse and the
    keyboard with the canvas. It closes only by its × or by the button that
    opened it (anchor). Every mouse event on the card itself stops at it, so
    none reaches the canvas, and nothing in it takes the keyboard. place() puts it above
    the anchor inside area, a function returning the rectangle, in the
    canvas's pixels, it must stay in; a card taller than the room scrolls.

    The standard Copy key (QKeySequence.Copy: Ctrl+C, or Cmd+C on macOS)
    copies the selected text of whichever path field has a selection, and
    Select All selects all of the field last clicked or selected in. The
    fields cannot take those keys themselves, having no focus, and a QShortcut
    would take them from the canvas, whose command line copies its own text
    with Copy. So the card listens, through an event filter on its parent, to
    the key presses the canvas receives. It consumes a Copy only when a field
    has a selection, and a Select All only when a field has been clicked in;
    every other key, and those two when there is nothing for them to do or
    when ``command_line_open`` (a function returning whether the command line
    is being typed in) says it is open, go on to the canvas unchanged. Only one
    field shows a selection at a time, as in a window, where only the field
    with the keyboard does: selecting in one clears the others.
    """

    PAD = 14          # inside the card
    RADIUS = 12       # the card's corners, a card's
    ROW_GAP = 8       # between rows
    COLUMN_GAP = 16   # least between the widest keys and what they do, in the controls' two halves
    COUNT_GAP = 24    # between the nodes and the edges on their line
    FIELD_GAP = 8     # between a label, its field and the button after it
    GROUP_GAP = 8     # extra, between the fixed files and the ones that can be changed
    KEY_GAP = 4       # between a key cap and the "+" beside it
    CLOSE_SIZE = 22   # the × button
    PICK_WIDTH = 30   # the pick button, as wide as the Config's folder buttons
    PATH_WIDTH = 260  # the least room a path's field is asked for
    ANCHOR_GAP = 16   # between the card and the button it opens from, clear of the slider
    MIN_HEIGHT = 120  # the least a card shrinks to in a short window
    EDGE = 8          # clear of the canvas's edges when the owner gives no area
    # The card paints its own shadow in this margin around it.
    SHADOW = 8

    def __init__(self, title, text, anchor=None, parent=None, *, session=None, controls_title=None,
                 area=None, command=None, command_line_open=None, jobs=None, confirm_stop=None):
        super().__init__(parent)
        self.setObjectName("viewerControlsCard")
        self.anchor = anchor
        self.session = session
        self.area = area
        self.command = command
        self.command_line_open = command_line_open
        self.jobs = jobs
        self.jobs_section = None
        self._remeasure = QtCore.QTimer(self)
        self._remeasure.setSingleShot(True)
        self._remeasure.timeout.connect(self._measure_again)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMouseTracking(True)
        self.setAccessibleName(title)
        self.rows = controls_rows(text)
        self._current = None   # the field Select All selects in: the last one clicked or selected in
        self.path_fields = {}
        self.pick_buttons = {}
        self.file_labels = {}
        self.count_labels = {}
        self._paths = {}
        self._bases = None
        self._natural = None
        self.key_holders = []
        self.controls_heading = None

        margin = self.SHADOW
        layout = QVBoxLayout(self)
        # The shadow falls a little below the card, so the card sits a little higher.
        top = margin - 2
        layout.setContentsMargins(margin + self.PAD, top + self.PAD - 4,
                                  margin + self.PAD - 6, 2 * margin - top + self.PAD)
        layout.setSpacing(10)

        # The first heading, then the link to the list of commands and the ×.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.session_heading = QLabel(
            QtCore.QCoreApplication.translate("ViewerControls", "Session") if session is not None
            else (controls_title or title))
        self.session_heading.setObjectName("controlsHeading")
        header.addWidget(self.session_heading)
        header.addStretch(1)
        self.commands_button = self._commands_link()
        header.addWidget(self.commands_button)
        self.close_button = QPushButton()
        self.close_button.setObjectName("controlsClose")
        self.close_button.setIcon(studio_icon("x", TOKENS["text_muted"]))
        self.close_button.setIconSize(QSize(14, 14))
        self.close_button.setFixedSize(self.CLOSE_SIZE, self.CLOSE_SIZE)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_text = QtCore.QCoreApplication.translate("ViewerControls", "Close")
        self.close_button.setToolTip(close_text)
        self.close_button.setAccessibleName(close_text)
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        # The sections scroll as one when the window leaves no room for them.
        self.body = QWidget()
        self.body.setObjectName("controlsBody")
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 6, 0)
        body_layout.setSpacing(10)
        if session is not None:
            body_layout.addLayout(self._session_section())
            body_layout.addWidget(self._separator())
            if jobs is not None:
                # The background jobs, between the files and the controls, each section set off by a separator.
                self.jobs_section = JobsSection(jobs, confirm=confirm_stop)
                self.jobs_section.resized.connect(self._jobs_resized)
                if parent is not None:
                    self.jobs_section.keyboard_back = parent.setFocus
                body_layout.addWidget(self.jobs_section)
                body_layout.addWidget(self._separator())
        body_layout.addLayout(self._controls_section(controls_title if session is not None else None))
        self.scroll = _BodyScroll()
        self.scroll.setObjectName("controlsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setWidget(self.body)
        layout.addWidget(self.scroll, 1)

        t = TOKENS
        self.setStyleSheet(f"""
            QLabel#controlsHeading {{ font-weight: 600; color: {t['text']}; padding: 0px; }}
            QLabel#controlsLabel {{ color: {t['text_soft']}; }}
            QLabel#controlsValue {{ color: {t['text']}; }}
            QLabel#controlsDescription {{ color: {t['text_soft']}; }}
            QLabel#controlsPlus {{ color: {t['text_subtle']}; }}
            QLabel#controlsKey {{ {key_cap_qss(t)} }}
            QLineEdit#controlsPath {{
                background: {t['field']}; color: {t['text']};
                border: 1px solid {t['field_border']}; border-radius: {t['radius']}px;
                padding: 3px 8px; min-height: 20px;
                selection-background-color: {t['accent_soft']}; selection-color: {t['text']};
            }}
            QLineEdit#controlsPath:hover {{ border-color: {t['field_border_hover']}; }}
            QLineEdit#controlsPath[fixed="true"] {{
                background: transparent; color: {t['text_soft']}; border-color: {t['separator']};
            }}
            QLineEdit#controlsPath[fixed="true"]:hover {{ border-color: {t['field_border']}; }}
            QPushButton#controlsLink {{
                background: transparent; border: none; border-radius: 6px; color: {t['accent_text']};
                font-weight: 600; text-align: left; padding: 3px 8px 3px 4px; min-height: 0px;
            }}
            QPushButton#controlsLink:hover {{ background: {t['tint_hover']}; }}
            QPushButton#controlsLink:pressed {{ background: {t['separator']}; }}
            QPushButton#controlsLink:disabled {{ color: {t['text_disabled']}; }}
            QFrame#controlsSeparator {{ background: {t['separator']}; border: none; }}
            QPushButton#controlsClose {{
                background: transparent; border: none; border-radius: 6px; padding: 0px;
                min-width: {self.CLOSE_SIZE}px; max-width: {self.CLOSE_SIZE}px;
                min-height: {self.CLOSE_SIZE}px; max-height: {self.CLOSE_SIZE}px;
            }}
            QPushButton#controlsClose:hover {{ background: {t['tint_hover']}; }}
            QPushButton#controlsClose:pressed {{ background: {t['separator']}; }}
            QScrollArea#controlsScroll, QWidget#controlsBody {{ background: transparent; border: none; }}
            QScrollArea#controlsScroll > QWidget#qt_scrollarea_viewport {{ background: transparent; }}
        """ + section_qss(t))
        # Nothing here takes the keyboard: [ENTER] still opens the command line.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for child in self.findChildren(QWidget):
            # The jobs' buttons are reached with Tab (never by a click), so they keep that.
            if child.focusPolicy() != Qt.FocusPolicy.TabFocus:
                child.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if session is not None:
            self.refresh()
        self.fit(1 << 16, 1 << 16)
        if parent is not None:
            # The key presses the canvas receives, for Copy and Select All (eventFilter); the filter
            # ends with the card, as Qt keeps a filter by pointer.
            parent.installEventFilter(self)

    @staticmethod
    def _separator():
        separator = QFrame()
        separator.setObjectName("controlsSeparator")
        separator.setFixedHeight(TOKENS["separator_thickness"])
        return separator

    def _jobs_resized(self):
        """The jobs section gained or lost rows: take the card's size again, and keep it above its button.

        Qt lays the new rows out in the event loop's next turn, and their sizes can only be read after, so
        the card measures itself then.
        """
        self._natural = None
        self._remeasure.start(0)

    def _measure_again(self):
        self._natural = None
        if not self.isHidden():
            self.place()

    def refresh_jobs(self):
        """Show the background jobs as they are now (the scheduler says when they change)."""
        if self.jobs_section is not None:
            self.jobs_section.refresh()

    def _session_section(self):
        """The nodes and edges on one line, and a row for each file: label, path field, and for a file that can be changed a pick button."""
        section = QVBoxLayout()
        section.setContentsMargins(0, 0, 0, 0)
        section.setSpacing(self.ROW_GAP)
        counts = QHBoxLayout()
        counts.setContentsMargins(0, 0, 0, 0)
        counts.setSpacing(self.FIELD_GAP)
        for index, (key, label_text) in enumerate((
                ("nodes", QtCore.QCoreApplication.translate("ViewerControls", "Nodes:")),
                ("edges", QtCore.QCoreApplication.translate("ViewerControls", "Edges:")))):
            if index:
                counts.addSpacing(self.COUNT_GAP - self.FIELD_GAP)
            label = QLabel(label_text)
            label.setObjectName("controlsLabel")
            value = QLabel()
            value.setObjectName("controlsValue")
            self.count_labels[key] = value
            counts.addWidget(label)
            counts.addWidget(value)
        counts.addStretch(1)
        section.addLayout(counts)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(self.FIELD_GAP)
        grid.setVerticalSpacing(self.ROW_GAP)
        grid.setColumnMinimumWidth(1, self.PATH_WIDTH)
        grid.setColumnStretch(1, 1)
        row = 0
        for key, label_text in SESSION_FILES:
            changeable = key in PICK_FILES
            if changeable and not self.pick_buttons:
                # A gap before the files that can be changed.
                grid.setRowMinimumHeight(row, self.GROUP_GAP)
                row += 1
            label = QLabel(QtCore.QCoreApplication.translate("ViewerControls", label_text))
            label.setObjectName("controlsLabel")
            field = PathField(fixed=not changeable)
            field.pressed.connect(lambda field=field: self._field_touched(field))
            field.selectionChanged.connect(lambda field=field: self._field_selected(field))
            self.file_labels[key], self.path_fields[key] = label, field
            self._paths[key] = None
            grid.addWidget(label, row, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            if changeable:
                button = icon_button("file-input", QtCore.QCoreApplication.translate("ViewerControls", PICK_FILES[key][1]))
                button.setFixedWidth(self.PICK_WIDTH)
                button.clicked.connect(lambda checked=False, name=key: self.pick_file(name))
                self.pick_buttons[key] = button
                grid.addWidget(field, row, 1)
                grid.addWidget(button, row, 2)
            else:
                # No button after it, so the field takes that room.
                grid.addWidget(field, row, 1, 1, 2)
            row += 1
        section.addLayout(grid)
        return section

    def _controls_section(self, heading_text):
        """The actions as key caps beside what they do, under heading_text when there is one.

        The keys and what they do are the two halves of the section: equal
        columns, the second starting at the section's middle.
        """
        section = QVBoxLayout()
        section.setContentsMargins(0, 0, 0, 0)
        section.setSpacing(self.ROW_GAP)
        if heading_text:
            self.controls_heading = QLabel(heading_text)
            self.controls_heading.setObjectName("controlsHeading")
            section.addWidget(self.controls_heading)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(self.ROW_GAP)
        self.controls_grid = grid
        self.key_caps = []
        self.descriptions = []
        for row, (keys, description) in enumerate(self.rows):
            holder = self._keys_widget(keys)
            self.key_holders.append(holder)
            grid.addWidget(holder, row, 0, Qt.AlignmentFlag.AlignLeft)
            label = QLabel(description)
            label.setObjectName("controlsDescription")
            self.descriptions.append(label)
            grid.addWidget(label, row, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        section.addLayout(grid)
        return section

    def _commands_link(self):
        """The link that opens the list of commands, for the header's row."""
        link = QPushButton(QtCore.QCoreApplication.translate("ViewerControls", "List of commands"))
        link.setObjectName("controlsLink")
        link.setIcon(studio_icon("file-text", TOKENS["accent_text"]))
        link.setIconSize(QSize(16, 16))
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        link.clicked.connect(self.open_command_list)
        self.commands_button = link
        self.show_command_list_availability()
        return link

    def _share_columns(self):
        """Make the controls' two columns equal, each as wide as the wider of what they must hold.

        The keys need their widest row and a gap after it, what they do their
        widest description; the section is twice the larger, so that nothing
        is cut short and the second column starts at the section's middle.
        Measured once the widgets are polished.
        """
        keys = max((holder.sizeHint().width() for holder in self.key_holders), default=0) + self.COLUMN_GAP
        what = max((label.sizeHint().width() for label in self.descriptions), default=0)
        for column in (0, 1):
            self.controls_grid.setColumnMinimumWidth(column, max(keys, what))

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

    # ---- the session's values ----

    def refresh(self):
        """Show the session as it is now.

        It sets only what changed, so it is cheap to call after every
        threshold step: a path is written again only when it, or what it is
        written from, changed, and nothing is looked up on disk.
        """
        if self.session is None:
            return
        info = self.session() or {}
        total, visible = with_separators(info.get("node_count", 0), info.get("visible_node_count"))
        self._show(self.count_labels["nodes"], total if visible is None else QtCore.QCoreApplication.translate(
            "ViewerControls", "{total} ({visible} visible)").format(total=total, visible=visible))
        total, shown = with_separators(info.get("edge_count", 0), info.get("shown_edge_count"))
        self._show(self.count_labels["edges"], total if shown is None else QtCore.QCoreApplication.translate(
            "ViewerControls", "{total} ({shown} shown)").format(total=total, shown=shown))
        inputs = info.get("inputs") or {}
        directories = info.get("directories") or {}
        root = info.get("project_root") or ""
        # What the paths are written from: when it is the same, so are they.
        bases = (tuple(sorted((str(key), str(value)) for key, value in directories.items())), str(root))
        rewrite = bases != self._bases
        self._bases = bases
        fixed = QtCore.QCoreApplication.translate("ViewerControls", FIXED_NOTE)
        for key, _label in SESSION_FILES:
            path = display_path(inputs.get(key))
            if path == self._paths[key] and not rewrite:
                continue
            self._paths[key] = path
            self.path_fields[key].show_path(path, alias_form(path, directories, root),
                                            None if key in PICK_FILES else fixed)

    @staticmethod
    def _show(label, text):
        if label.text() != text:
            label.setText(text)

    def pick_file(self, key):
        """Run the command that picks another file for ``key``, as typed with no argument; returns whether one was run.

        "alignment" or "embedding" goes to ``command`` alone, which opens that
        command's file dialog and does the rest as it does when typed (history,
        the messages, the rollback of a file that fails). The keyboard stays
        with the canvas afterwards. A file the card cannot change, or a card
        given no command, runs nothing.
        """
        entry = PICK_FILES.get(key)
        if entry is None or self.command is None:
            return False
        self.command(entry[0])
        host = self.parentWidget()
        if host is not None:
            host.setFocus()
        return True

    # ---- the keyboard: Copy and Select All for the path fields ----

    def _field_touched(self, field):
        """A mouse press on ``field``: Select All now selects in it."""
        self._current = field

    def _field_selected(self, field):
        """``field``'s selection changed: when it holds one, it is the only field that does."""
        if not field.hasSelectedText():
            return
        self._current = field
        for other in self.path_fields.values():
            if other is not field and other.hasSelectedText():
                other.deselect()

    def selected_field(self):
        """The path field with a selection, or None when none has one."""
        fields = (self._current, *self.path_fields.values())
        return next((field for field in fields if field is not None and field.hasSelectedText()), None)

    def copy_selection(self):
        """Copy the selected text of the field that has a selection, as its right-click menu does; returns whether one was copied."""
        field = self.selected_field()
        if field is None:
            return False
        field.copy()
        return True

    def select_all(self):
        """Select all the text of the field last clicked or selected in; returns whether it had any."""
        field = self._current
        if field is None or not field.text():
            return False
        field.selectAll()
        return True

    def eventFilter(self, watched, event):
        """Answer Copy and Select All, which the canvas receives since no field has the keyboard.

        Only a key press that matches the standard key (QKeySequence.Copy,
        QKeySequence.SelectAll), when the card is open and the command line is
        not being typed in, and the card has something to do with it, is
        consumed; everything else returns False and reaches the canvas as it
        did before the card was there.
        """
        if (event.type() == QEvent.Type.KeyPress and watched is self.parentWidget()
                and not self.isHidden()
                and not (self.command_line_open is not None and self.command_line_open())):
            if event.matches(QKeySequence.StandardKey.Copy):
                return self.copy_selection()
            if event.matches(QKeySequence.StandardKey.SelectAll):
                return self.select_all()
        return False

    def show_command_list_availability(self):
        """Enable the command list's link when its page is there, else grey it with the reason."""
        found = command_list_path(installed_language()) is not None
        self.commands_button.setEnabled(found)
        self.commands_button.setToolTip(
            QtCore.QCoreApplication.translate("ViewerControls", "Open the list of commands in your web browser")
            if found else
            QtCore.QCoreApplication.translate(
                "ViewerControls", "The list of commands is not installed: docs/list_of_commands.html is missing."))

    def open_command_list(self):
        """Open the list of commands, in the Viewer's language where it is translated, in the web browser.

        The card stays open and the keyboard stays with the canvas. Returns
        whether a page was opened.
        """
        page = command_list_path(installed_language())
        if page is None:
            self.show_command_list_availability()
            return False
        Desktop_Open.open_path(page)
        return True

    # ---- size, place and paint ----

    def card_rect(self):
        """Where the card is, inside the shadow's margin."""
        margin = self.SHADOW
        return QRectF(margin, margin - 2, self.width() - 2 * margin, self.height() - 2 * margin)

    def natural_size(self):
        """The size the card takes when nothing is clipped; measured once, then kept."""
        if self._natural is None:
            # The theme's stylesheet pads a field and a button only once it has
            # polished them, and a widget not shown yet does not tell its
            # layout when its hint changes: polish, then drop every cached hint.
            for widget in (self, *self.findChildren(QWidget)):
                widget.ensurePolished()
            self._share_columns()
            self.scroll.updateGeometry()
            # The jobs section is a widget of its own with a layout of its own, which changes while the card is open.
            sections = (self.jobs_section.layout(),) if self.jobs_section is not None else ()
            for layout in (self.layout(), self.body.layout(), *sections):
                _invalidate(layout)

            hint = self.layout().sizeHint()
            self._natural = QSize(hint.width(), hint.height())
        return self._natural

    def fit(self, max_width, max_height):
        """Size the widget to its sections, or to max_width by max_height where they would not fit.

        A card too tall scrolls its sections, and takes the room for the bar.
        """
        natural = self.natural_size()
        width, height = natural.width(), natural.height()
        if height > max_height:
            height = max_height
            width += self.scroll.verticalScrollBar().sizeHint().width() + 4
        self.setFixedSize(max(1, min(width, max_width)), max(1, height))

    def place(self):
        """Fit the card to its room and put it above its anchor, their left edges lined up.

        The room is the area the owner gives, the canvas's by default; the
        card is moved in if it would leave it.
        """
        host = self.parentWidget()
        if host is None:
            return
        room = self.area() if self.area is not None else host.rect().adjusted(
            self.EDGE, self.EDGE, -self.EDGE, -self.EDGE)
        corner = self.anchor.mapTo(host, QtCore.QPoint(0, 0)) if self.anchor is not None else room.bottomLeft()
        above = corner.y() - self.ANCHOR_GAP - room.top()
        self.fit(room.width() + 2 * self.SHADOW, max(above, self.MIN_HEIGHT) + 2 * self.SHADOW)
        card = self.card_rect()
        x = corner.x() - round(card.left())
        y = corner.y() - self.ANCHOR_GAP - round(card.bottom())
        right = room.left() + room.width()
        x = max(room.left() - round(card.left()), min(x, right - round(card.right())))
        y = max(y, room.top() - round(card.top()))
        self.move(x, y)

    def open(self):
        """Show the card, placed, over everything in the canvas; returns it."""
        self.place()
        self.show()
        self.raise_()
        if self.jobs_section is not None:
            # Rows made before the card was styled and shown change size as they are: measure again after.
            self._jobs_resized()
        return self

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        card = self.card_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        for grow, alpha in ((7.0, 4), (5.0, 6), (3.0, 9), (1.5, 12)):
            painter.setBrush(QColor(9, 9, 11, alpha))
            painter.drawRoundedRect(card.adjusted(-grow, -grow + 2, grow, grow + 2),
                                    self.RADIUS + grow, self.RADIUS + grow)
        painter.setPen(QPen(QColor(TOKENS["tint_hover"]), 1.0))
        painter.setBrush(QColor(TOKENS["tint"]))
        painter.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), self.RADIUS, self.RADIUS)
        painter.end()

    # The card sits over the canvas, a parent that turns the mouse into the
    # network's clicks, drags and zooms: every mouse event on the card is the
    # card's, accepted here when no widget in it took it. The shadow's margin
    # is not the card, so a click there goes through to the canvas.

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
        self._take(event, QtCore.QPointF(event.pos()))


def _invalidate(item):
    """Drop the cached size hints of a layout item and of everything in it."""
    item.invalidate()
    layout = item.layout()
    if layout is not None:
        for index in range(layout.count()):
            _invalidate(layout.itemAt(index))


class _BodyScroll(QScrollArea):
    """The card's scrolling sections: as large as they are, so the card can size itself to them."""

    def sizeHint(self):
        widget = self.widget()
        return widget.sizeHint() if widget is not None else QSize(0, 0)

    def minimumSizeHint(self):
        return QSize(0, 0)


def with_separators(total, part):
    """A count and its part with thousands separators; the part is None when it is the whole."""
    return f"{total:,}", (None if part is None or part == total else f"{part:,}")


class ControlsButton(QPushButton):
    """The info button that opens the session and controls card above itself.

    title is the button's tooltip and the card's accessible name, which the
    card does not show; text the Viewer's list of actions, headed
    controls_title next to the session. session, area, command,
    command_line_open, jobs and confirm_stop are the card's (ControlsCard). The card is a child of host, the
    canvas, which is the button's parent by default. A click while the card
    is open closes it; so does its ×, and nothing else does.
    """

    SIZE = 28   # as tall as a field

    def __init__(self, title, text, parent=None, *, host=None, session=None, controls_title=None,
                 area=None, command=None, command_line_open=None, jobs=None, confirm_stop=None):
        super().__init__(parent)
        self.setObjectName("viewerControlsButton")
        self.title = title
        self.text_source = text
        self.host = host
        self.session = session
        self.controls_title = controls_title
        self.area = area
        self.command = command
        self.command_line_open = command_line_open
        self.jobs = jobs
        self.confirm_stop = confirm_stop
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
            # Not isVisible(): a card in a window not shown yet is open all the same.
            return card is not None and not card.isHidden()
        except RuntimeError:  # deleted on close
            return False

    def toggle_card(self):
        """Open the card above the button, or close it if it is open; returns it or None."""
        if self.card_is_open():
            self.card.close()
            return None
        host = self.host or self.parentWidget() or self.window()
        self.card = ControlsCard(self.title, self.text_source, anchor=self, parent=host,
                                 session=self.session, controls_title=self.controls_title,
                                 area=self.area, command=self.command,
                                 command_line_open=self.command_line_open,
                                 jobs=self.jobs, confirm_stop=self.confirm_stop)
        return self.card.open()

    def refresh_card(self):
        """Bring an open card's session up to date; nothing happens while it is closed."""
        if self.card_is_open():
            self.card.refresh()

    def refresh_card_jobs(self):
        """Bring an open card's background jobs up to date; nothing happens while it is closed."""
        if self.card_is_open():
            self.card.refresh_jobs()

    def place_card(self):
        """Put an open card where the canvas's size and the button now leave it."""
        if self.card_is_open():
            self.card.place()


class ConsoleButton(QPushButton):
    """The Console button of the slider row, after the info button: the terminal icon and the word "Console".

    ``text`` and ``tooltip`` are the owner's translated texts, and ``opened`` is what a
    click calls (the owner opens the Console, or says why it cannot). The button is
    as tall as the row's boxes (SLIDER_BOX_HEIGHT, the info button's SIZE) and as
    wide as its text and icon ask, so a longer translation widens it instead of being
    cut off (Desktop_App.fit_buttons_to_text). Its tint, border and hover are the info
    button's, its text the size of the boxes' numbers. It never takes the keyboard, so
    [ENTER] still opens the command line.
    """

    ICON_SIZE = 16
    PADDING = 10       # between the icon or the text and the button's ends
    FONT_POINTS = 12   # the row's text size, as its number boxes have

    def __init__(self, text, tooltip, parent=None, *, opened=None):
        super().__init__(text, parent)
        self.setObjectName("viewerConsoleButton")
        self.setIcon(studio_icon("square-terminal", TOKENS["text_muted"]))
        self.setIconSize(QSize(self.ICON_SIZE, self.ICON_SIZE))
        self.setFixedHeight(ControlsButton.SIZE)
        self.setToolTip(tooltip)
        self.setAccessibleName(text)
        self.setAccessibleDescription(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        t = TOKENS
        self.setStyleSheet(f"""
            QPushButton#viewerConsoleButton {{
                background: {t['tint']}; border: 1px solid {t['tint']}; color: {t['text_soft']};
                border-radius: {t['radius']}px; padding: 0px;
                min-height: {ControlsButton.SIZE - 2}px; max-height: {ControlsButton.SIZE - 2}px;
                font-family: {UI_QSS_FONT_STACK}; font-size: {self.FONT_POINTS}pt;
            }}
            QPushButton#viewerConsoleButton:hover {{
                background: {t['tint_hover']}; border-color: {t['tint_hover']};
            }}
            QPushButton#viewerConsoleButton:pressed {{
                background: {t['separator']}; border-color: {t['separator']};
            }}
        """)
        fit_buttons_to_text(self, padding=self.PADDING)
        if opened is not None:
            self.clicked.connect(opened)


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


# A step of the Zoom box's arrows, of the wheel over it or of Up/Down zooms in
# (a percentage this many times bigger) or out by this factor, about a notch of
# the mouse wheel over the network.
ZOOM_STEP = 1.25
# The height of the slider row's boxes, the Zoom box's and the threshold's, in
# px: the status texts above them go by it.
SLIDER_BOX_HEIGHT = 28
# Between the box and the text either side of it, in px.
ZOOM_LABEL_GAP = 12
# Between the box and a percent sign after it, in px: the sign is the box's unit,
# so it sits close.
ZOOM_UNIT_GAP = 4
# The labels' text sits this many px lower than centred in the row: the number
# in its box sits a pixel lower than its box's middle (the style's padding), and
# the text beside it takes the same baseline.
ZOOM_LABEL_DROP = 1
# The widest zoom its field shows without scrolling, which is the highest the
# Viewer zooms to (commands.zoom.MAX_ZOOM_PERCENT); with no limit to keep to (no
# nodes), also the highest it takes.
ZOOM_ROOM = MAX_ZOOM_PERCENT
# The lowest zoom the box takes while there is no node to set a limit.
ZOOM_FLOOR = 0.001
# The decimals the box keeps internally, so a typed number is kept as typed; what
# it shows is rounded by zoom_decimals.
ZOOM_DECIMALS = 6
# The threshold box has at least this many steps from the lowest score to the
# highest: the slider's 1000 positions before the box came, so it is never
# coarser than the slider was.
THRESHOLD_STEPS = 1000
# Slack for a number a hair under a whole count of steps, as 0.57 * 100 is.
_STEP_SLACK = 1e-9


class CanvasSpinBox(QDoubleSpinBox):
    """A number box over the network, to read and to set.

    It takes the keyboard only when clicked, never on opening or by Tab, so
    the keyboard stays with the network and [ENTER] opens the command line. A
    typed number counts on Enter, or when the box loses the keyboard; Esc puts
    back the number shown. valueChanged reports only what the user set:
    show_value, for the Viewer's own changes, sends nothing. dismissed is Esc.
    """

    dismissed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setKeyboardTracking(False)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setFixedHeight(SLIDER_BOX_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._shown = self.value()

    def sizeHint(self):
        return super().sizeHint().boundedTo(self.maximumSize())

    def show_value(self, value):
        """Show value without reporting it; Esc then puts it back."""
        blocked = self.blockSignals(True)
        try:
            self.setValue(value)
        finally:
            self.blockSignals(blocked)
        self._shown = self.value()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.show_value(self._shown)  # Its text too, typed or not.
            self.dismissed.emit()
            return
        super().keyPressEvent(event)


def zoom_decimals(percent):
    """How many decimals the Zoom box shows for percent: three significant figures
    as far as they go, and always the whole part: 1235%, 123%, 12.3%, 1.23%, 0.123%."""
    if percent >= 100:
        return 0
    if percent >= 10:
        return 1
    if percent >= 1:
        return 2
    return 3


def shown_zoom(percent):
    """percent as the Zoom box shows it, rounded to zoom_decimals."""
    return round(percent, zoom_decimals(percent))


class ZoomSpinBox(CanvasSpinBox):
    """The view's zoom, a percentage: 100% is a view of commands.zoom.ZOOM_100_PERCENT_WIDTH
    scene units, and a bigger number is closer.

    The arrows, the wheel over it and Up/Down zoom in (Up, wheel forward) or out
    ZOOM_STEP at a time, as the mouse wheel zooms, not by a fixed amount. It
    shows the number alone, to the precision zoom_decimals gives (the % sign is
    ZoomBox's text after it), and keeps the number shown as its value, so
    leaving it as it is changes nothing. A % typed or pasted after the number is
    left out of what the box holds. Esc puts back the zoom the view has.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(ZOOM_DECIMALS)
        self.setRange(ZOOM_FLOOR, ZOOM_ROOM)

    def validate(self, text, pos):
        # "230%" and "230 %" are 230: the sign is dropped as it is typed or pasted.
        stripped = text.rstrip()
        if stripped.endswith("%"):
            stripped = stripped[:-1].rstrip()
            return super().validate(stripped, min(pos, len(stripped)))
        return super().validate(text, pos)

    def textFromValue(self, value):
        locale = self.locale()
        # No group separators, as QDoubleSpinBox shows its numbers: 200000, not 200,000.
        return locale.toString(float(value), "f", zoom_decimals(value)).replace(locale.groupSeparator(), "")

    def sizeHint(self):
        # As wide as ZOOM_ROOM needs in the font it has now, whatever the
        # range is, so the box keeps its width as the limits change.
        hint = super().sizeHint()
        metrics = self.fontMetrics()
        extra = (metrics.horizontalAdvance(self.textFromValue(ZOOM_ROOM))
                 - metrics.horizontalAdvance(self.textFromValue(self.maximum())))
        return QSize(hint.width() + max(0, extra), hint.height()).boundedTo(self.maximumSize())

    def minimumSizeHint(self):
        return self.sizeHint()

    def stepBy(self, steps):
        self.setValue(self.value() * ZOOM_STEP ** steps)

    def show_zoom(self, percent, lowest, highest):
        """Show the view's zoom, with the limits it keeps to as the range.

        lowest is None while there is no node to set one. Nothing changes while
        the box is being typed in. A view beyond a limit, as after the nodes
        were moved closer together, still shows its zoom.
        """
        if self.hasFocus():
            return
        bottom = ZOOM_FLOOR if lowest is None else lowest
        # A hair under a limit is that limit, whatever its last digit says.
        bottom = bottom if percent >= bottom * (1 - 1e-9) else percent
        top = highest if percent <= highest * (1 + 1e-9) else percent
        blocked = self.blockSignals(True)
        try:
            self.setRange(min(bottom, percent), max(top, percent))
        finally:
            self.blockSignals(blocked)
        self.show_value(shown_zoom(percent))


class ThresholdSpinBox(CanvasSpinBox):
    """The edge threshold: edges scoring at least it are drawn.

    set_scale() fits it to the scores. Its step, the smallest change it
    makes, is the largest power of ten that still gives THRESHOLD_STEPS steps
    between the lowest score and the highest, and no more than 1, so every
    number it takes is on a step; it shows as many decimals as the step has.
    Its numbers run from the multiple of the step at or under the lowest
    score, which still draws every edge, to the one at or under the highest,
    which still draws the best. The threshold slider has one position for
    each number: value_at() and position_of() go between them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_scale(0.0, 1.0)

    def set_scale(self, lowest, highest):
        """Fit the box to scores from lowest to highest, without a signal."""
        self._exponent = min(0, math.floor(math.log10((highest - lowest) / THRESHOLD_STEPS)))
        self._first = math.floor(self._in_steps(lowest) + _STEP_SLACK)
        self.positions = max(1, math.floor(self._in_steps(highest) + _STEP_SLACK) - self._first)
        blocked = self.blockSignals(True)
        try:
            self.setDecimals(-self._exponent)
            self.setSingleStep(10.0 ** self._exponent)
            self.setRange(self.value_at(0), self.value_at(self.positions))
        finally:
            self.blockSignals(blocked)

    def _in_steps(self, value):
        """value counted in steps, a whole number when it is on one."""
        return value * 10 ** -self._exponent

    def value_at(self, position):
        """The threshold at a slider position."""
        return (self._first + position) / 10 ** -self._exponent

    def position_of(self, value):
        """The slider position of the highest number at or under value."""
        steps = math.floor(self._in_steps(value) + _STEP_SLACK) - self._first
        return min(max(steps, 0), self.positions)


class ZoomBox(QWidget):
    """The translated "Zoom: {zoom}%", with a ZoomSpinBox for {zoom}.

    The text either side of {zoom} goes either side of the box, so a
    language that puts the number first, or the % sign before it, keeps its
    order. A percent sign after the box sits ZOOM_UNIT_GAP from it, as the
    box's unit; any other text keeps ZOOM_LABEL_GAP clear of it. The labels
    take the stylesheet's font and colour and are as tall as the box, their text
    centred in it as the number is, so the baselines agree.
    """

    def __init__(self, template, parent=None):
        super().__init__(parent)
        self.setObjectName("zoomBox")
        before, _field, after = template.partition("{zoom}")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.before = QLabel(before.strip())
        self.spin_box = ZoomSpinBox()
        self.after = QLabel(after.strip())
        self.before.setContentsMargins(0, 2 * ZOOM_LABEL_DROP, ZOOM_LABEL_GAP, 0)
        unit = after.strip()[:1] in ("%", "％")  # Also the full-width sign.
        self.after.setContentsMargins(ZOOM_UNIT_GAP if unit else ZOOM_LABEL_GAP, 2 * ZOOM_LABEL_DROP, 0, 0)
        for widget in (self.before, self.spin_box, self.after):
            layout.addWidget(widget)
        self.before.setVisible(bool(self.before.text()))
        self.after.setVisible(bool(self.after.text()))


__all__ = [
    "CanvasSpinBox",
    "ConsoleButton",
    "ControlsButton",
    "ControlsCard",
    "PathField",
    "SESSION_FILES",
    "SIDEBAR_INSET",
    "SidebarButton",
    "SidebarPanel",
    "SLIDER_BOX_HEIGHT",
    "THRESHOLD_STEPS",
    "ThresholdSpinBox",
    "ZOOM_LABEL_GAP",
    "ZOOM_ROOM",
    "ZOOM_STEP",
    "ZOOM_UNIT_GAP",
    "ZoomBox",
    "ZoomSpinBox",
    "controls_rows",
    "display_path",
    "shown_zoom",
    "zoom_decimals",
]
