# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""The Session card's Background jobs section: the Viewer's background jobs, by lane, with buttons to arrange them.

The section lists each lane of the scheduler that holds a job under the lane's
title, the running job first, then those waiting, then those that failed lately.
A row has the job's number and description, its state on a second line (Running
with its time, Queued with its place in the lane's line, Stopping, or the reason
it failed), the job's own progress after it and an Open log link when it names a
log, and its buttons: Move up and Move down for a waiting job, which change places
with the waiting job next to it; Remove for a waiting job, Stop and remove for the
running one (after a confirmation: it stops at its next step and what it had not
finished writing is deleted), and Dismiss for a failure. With no job at all one
muted line says so, so the section does not appear and vanish.

Everything a row shows comes from the job through the lanes the section is given
(desktop.Viewer_Jobs.job_lanes): the section names no command. It does not poll:
the card calls refresh() when the scheduler says something changed, and a
one-second timer only moves the running jobs' clocks.

The buttons are reached with Tab (they never take the keyboard from the canvas by
a click, so [ENTER] still opens the command line) and pressed with Space or Enter.
After a keyboard press the focus stays on the button, which moves with its row; a
mouse press leaves the keyboard with the canvas.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from desktop.Desktop_App import translate
from desktop.Studio_Theme import TOKENS, icon_button, studio_icon
from utilities.Localization import display_text


def clock(seconds):
    """A running time as 01:23, or 1:02:03 from an hour on."""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class ElidedLabel(QLabel):
    """A one-line label that cuts its text short with "…" to fit, and shows the whole text when hovered.

    It asks for no width of its own, so a long name never widens the card. It paints its text itself, so
    a screen reader finds none in it: its accessible name is the whole text, as set_full_text last gave it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self._tip = None
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text, tip=None):
        """Show ``text`` as far as it fits; the tooltip is ``tip`` (default: the text) once it is cut."""
        self._full = text
        self._tip = tip
        self.setAccessibleName(text)
        self._show_tip()
        self.update()

    def full_text(self):
        return self._full

    def is_elided(self):
        return self.fontMetrics().horizontalAdvance(self._full) > max(0, self.width() - self.contentsMargins().left()
                                                                       - self.contentsMargins().right())

    def _show_tip(self):
        # A tip is for what is hidden: the cut text's whole, or what the caller said.
        self.setToolTip(self._tip if self._tip is not None else (self._full if self.is_elided() else ""))

    def sizeHint(self):
        return QSize(60, self.fontMetrics().height() + 2)

    def minimumSizeHint(self):
        return QSize(0, self.fontMetrics().height() + 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_tip()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter

        painter = QPainter(self)
        margins = self.contentsMargins()
        room = self.rect().adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        text = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, room.width())
        painter.setPen(self.palette().windowText().color())
        painter.drawText(room, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
        painter.end()


def default_confirm(parent, job_id, description):
    """Ask whether to stop a running job; return whether the user said yes."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(translate("ViewerControls", "Stop job"))
    box.setText(translate("ViewerControls", "Stop job #{job} while it is running?").format(job=job_id))
    box.setInformativeText(
        translate("ViewerControls", "{description}\n\nIt stops at its next step, and the unfinished file it was writing is "
                   "deleted. A file it had already saved stays.").format(description=description))
    stop = box.addButton(translate("ViewerControls", "Stop job"), QMessageBox.ButtonRole.DestructiveRole)
    keep = box.addButton(translate("ViewerControls", "Keep running"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(keep)
    box.setEscapeButton(keep)
    box.exec()
    return box.clickedButton() is stop


class JobRowWidget(QFrame):
    """One job's row: its number and description, its state and log, and its buttons."""

    BUTTON_WIDTH = 28

    def __init__(self, row, section):
        super().__init__()
        self.setObjectName("jobRow")
        self.row = row
        self.section = section
        job = row.job_id
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(1)
        self.lead = ElidedLabel()
        self.lead.setObjectName("jobLead")
        self.lead.set_full_text(f"#{job}  {row.description}", self._lead_tip())
        texts.addWidget(self.lead)
        second = QHBoxLayout()
        second.setContentsMargins(0, 0, 0, 0)
        second.setSpacing(8)
        self.state = ElidedLabel()
        self.state.setObjectName("jobState")
        self.state.setProperty("failed", row.state == "failed")
        second.addWidget(self.state, 1)
        self.log_button = None
        if row.log_path:
            self.log_button = QPushButton(translate("ViewerControls", "Open log"))
            self.log_button.setObjectName("jobLogLink")
            self.log_button.setIcon(studio_icon("file-text", TOKENS["accent_text"]))
            self.log_button.setIconSize(QSize(14, 14))
            self.log_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.log_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            self.log_button.setToolTip(translate("ViewerControls", "Open the log file in your text viewer"))
            self.log_button.setAccessibleName(translate("ViewerControls", "Open the log of #{job}").format(job=job))
            self.log_button.setProperty("job", job)
            self.log_button.setProperty("role", "log")
            self.log_button.clicked.connect(lambda: self.section._act("log", job, self.log_button))
            second.addWidget(self.log_button)
        texts.addLayout(second)
        outer.addLayout(texts, 1)

        self.up = self.down = None
        if row.state == "queued":
            self.up = self._button("chevron-up", "up", translate("ViewerControls", "Move up"),
                                   translate("ViewerControls", "Move #{job} up").format(job=job), row.can_move_up)
            self.down = self._button("chevron-down", "down", translate("ViewerControls", "Move down"),
                                     translate("ViewerControls", "Move #{job} down").format(job=job), row.can_move_down)
            outer.addWidget(self.up)
            outer.addWidget(self.down)
        if row.state == "queued":
            tip, name, enabled = (translate("ViewerControls", "Remove from the queue"),
                                  translate("ViewerControls", "Remove #{job} from the queue").format(job=job), True)
        elif row.state == "running":
            tip, name, enabled = (translate("ViewerControls", "Stop and remove"), translate("ViewerControls", "Stop and remove #{job}").format(job=job), True)
        elif row.state == "stopping":
            tip, name, enabled = (translate("ViewerControls", "Stopping…"), translate("ViewerControls", "Stopping #{job}").format(job=job), False)
        else:
            tip, name, enabled = (translate("ViewerControls", "Dismiss"), translate("ViewerControls", "Dismiss #{job}").format(job=job), True)
        self.remove = self._button("x", "dismiss" if row.state == "failed" else "remove", tip, name, enabled)
        outer.addWidget(self.remove)
        self.refresh_state(section.now())

    def _lead_tip(self):
        row = self.row
        lines = [row.description]
        if row.path:
            lines.append(row.path)
        return "\n".join(lines)

    def _button(self, icon, role, tip, name, enabled):
        button = icon_button(icon, tip)
        button.setAccessibleName(name)
        button.setFixedWidth(self.BUTTON_WIDTH)
        # Reached with Tab, never by a click: the canvas keeps the keyboard.
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setEnabled(enabled)
        button.setProperty("job", self.row.job_id)
        button.setProperty("role", role)
        button.clicked.connect(lambda: self.section._act(role, self.row.job_id, button))
        return button

    def state_text(self, now):
        """The state line: Running with its time, Queued with its place, Stopping, or why it failed; then its progress."""
        row = self.row
        if row.state == "running":
            text = translate("ViewerControls", "Running · {time}").format(time=clock(now - row.started_at if row.started_at else 0))
        elif row.state == "stopping":
            text = translate("ViewerControls", "Stopping…")
        elif row.state == "queued":
            text = translate("ViewerControls", "Queued · position {position}").format(position=row.position)
        else:
            text = translate("ViewerControls", "Failed: {reason}").format(reason=display_text(row.error))
        if row.state in ("running", "stopping", "queued") and row.progress:
            text += " · " + display_text(row.progress)
        return text

    def refresh_state(self, now):
        text = self.state_text(now)
        if text != self.state.full_text():
            tip = display_text(self.row.error) if self.row.state == "failed" else None
            self.state.set_full_text(text, tip)


def section_qss(tokens):
    """The stylesheet of the section's labels, for the card's own to hold."""
    t = tokens
    return f"""
        QLabel#jobLane {{ color: {t['text_muted']}; font-weight: 600; padding: 2px 0px 0px 0px; }}
        QLabel#jobLead {{ color: {t['text']}; }}
        QLabel#jobState {{ color: {t['text_soft']}; }}
        QLabel#jobState[failed="true"] {{ color: {t['danger']}; }}
        QLabel#jobEmpty {{ color: {t['text_muted']}; }}
        QFrame#jobRow {{ background: transparent; border: none; }}
        QPushButton#jobLogLink {{
            background: transparent; border: none; border-radius: 6px; color: {t['accent_text']};
            font-weight: 600; text-align: left; padding: 3px 8px 3px 4px; min-height: 20px;
        }}
        QPushButton#jobLogLink:hover {{ background: {t['tint_hover']}; }}
        QPushButton#jobLogLink:pressed {{ background: {t['separator']}; }}
    """


class JobsSection(QWidget):
    """The Background jobs section: a heading, then the lanes' rows, or one muted line when there are none.

    ``source`` gives lanes() (desktop.Viewer_Jobs.job_lanes) and takes move(job_id, offset),
    remove(job_id), dismiss(job_id) and open_log(job_id); ``confirm(job_id, description)``
    asks before a running job is stopped (default_confirm). ``now`` is the clock the running
    times are read from. ``keyboard_back``, when set, is called after a press with the mouse, to
    return the keyboard to the canvas.
    """

    acted = Signal()
    # The set of rows changed (not just a clock): the card's size must be taken again.
    resized = Signal()

    def __init__(self, source, *, confirm=None, now=time.monotonic, parent=None):
        super().__init__(parent)
        self.setObjectName("jobsSection")
        self.source = source
        self.confirm = confirm
        self.now = now
        self.keyboard_back = None
        self.rows = {}
        self._signature = None
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(8)
        self.heading = QLabel(translate("ViewerControls", "Background jobs"))
        self.heading.setObjectName("controlsHeading")
        self.layout_.addWidget(self.heading)
        self.empty = QLabel(translate("ViewerControls", "No background jobs."))
        self.empty.setObjectName("jobEmpty")
        self.layout_.addWidget(self.empty)
        self.lane_labels = {}
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(6)
        self.layout_.addLayout(self.body)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.refresh()

    # ---- what is listed ----

    @staticmethod
    def _structure(lanes):
        """What a rebuild depends on: everything a row shows but its clock."""
        return tuple(
            (lane.name, display_text(lane.title), tuple(
                (row.job_id, row.state, row.description, row.path, row.position, row.log_path,
                 row.can_move_up, row.can_move_down,
                 display_text(row.progress) if row.progress else "", display_text(row.error) if row.error else "")
                for row in lane.rows))
            for lane in lanes)

    def refresh(self):
        """Show the jobs as they are now; return whether the rows changed (and the card must be sized again).

        Only a change to the rows rebuilds them; otherwise only the clocks move.
        """
        lanes = self.source.lanes()
        signature = self._structure(lanes)
        if signature == self._signature:
            self.tick()
            return False
        focused = QApplication.focusWidget()
        focus_key = (focused.property("job"), focused.property("role")) if (
            focused is not None and self.isAncestorOf(focused)) else None
        self._signature = signature
        self._clear()
        self.empty.setVisible(not lanes)
        for lane in lanes:
            label = QLabel(display_text(lane.title))
            label.setObjectName("jobLane")
            self.lane_labels[lane.name] = label
            self.body.addWidget(label)
            for row in lane.rows:
                widget = JobRowWidget(row, self)
                self.rows[row.job_id] = widget
                self.body.addWidget(widget)
        if any(row.state in ("running", "stopping") for lane in lanes for row in lane.rows):
            self.timer.start()
        else:
            self.timer.stop()
        if focus_key is not None:
            self._restore_focus(*focus_key)
        self.resized.emit()
        return True

    def _clear(self):
        while self.body.count():
            item = self.body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Hidden at once and deleted when the event loop is next idle. It must stay a child until then:
                # a widget taken out of its parent becomes a window, which the layout's pending show would open.
                widget.hide()
                widget.deleteLater()
        self.rows = {}
        self.lane_labels = {}

    def _restore_focus(self, job_id, role):
        """Put the keyboard back on the button that had it, now on the rebuilt row.

        A Move button that can no longer move (its job reached the end of the line)
        passes the keyboard to the other Move button, then to Remove, so pressing
        it again and again never loses the place; a row that is gone keeps nothing.
        """
        widget = self.rows.get(job_id)
        if widget is None:
            return
        buttons = {"up": widget.up, "down": widget.down, "log": widget.log_button,
                   "remove": widget.remove, "dismiss": widget.remove}
        order = {"up": ("up", "down", "remove"), "down": ("down", "up", "remove")}.get(role, (role,))
        for name in order:
            button = buttons.get(name)
            if button is not None and button.isEnabled():
                button.setFocus(Qt.FocusReason.TabFocusReason)
                return

    def tick(self):
        """Move the running jobs' clocks."""
        now = self.now()
        for widget in self.rows.values():
            widget.refresh_state(now)

    # ---- what the buttons do ----

    def _act(self, role, job_id, button):
        """A button was pressed: do what it says, then show the result and put the keyboard where it belongs."""
        row = self.rows[job_id].row if job_id in self.rows else None
        keyboard = button.hasFocus()
        if role == "up":
            self.source.move(job_id, -1)
        elif role == "down":
            self.source.move(job_id, 1)
        elif role == "log":
            self.source.open_log(job_id)
        elif role == "dismiss":
            self.source.dismiss(job_id)
        elif role == "remove" and row is not None:
            if row.state != "running" or self._confirmed(job_id, row.description):
                self.source.remove(job_id)
        self.refresh()
        self.acted.emit()
        if not keyboard and self.keyboard_back is not None:
            self.keyboard_back()

    def _confirmed(self, job_id, description):
        ask = self.confirm or (lambda job, text: default_confirm(self.window(), job, text))
        return bool(ask(job_id, description))
