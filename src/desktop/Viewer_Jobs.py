# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's jobs badge: how many commands wait or run, over the canvas's top right.

It shows only while the command queue (Command_Queue) holds a command or the
background scheduler a file, under the frame-rate display's place, right-aligned
with it; its tooltip lists them. Like the frame rate it is a QLabel on the
canvas, never in a picture the canvas renders.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from desktop.Studio_Theme import TOKENS
from utilities.Localization import Message, display_text

# The badge's height, the frame-rate display's.
BADGE_HEIGHT = 30
# Between the frame-rate row and the badge, in px.
BADGE_GAP = 8


def badge_text(count):
    return display_text(Message("%n job(s)", n=count))


def job_lines(queue, scheduler):
    """One line per command that waits or runs and per file job, for the tooltip."""
    lines = []
    for job in (queue.pending() if queue is not None else []):
        state = 'waiting' if job.status == 'queued' else job.status
        lines.append(f"#{job.job_id} {state}: {job.line}")
    listing = getattr(scheduler, 'listing', None)
    for job, writing in (listing() if listing is not None else []):
        lines.append(f"#{job.job_id} {'writing' if writing else 'waiting'}: {job.description}")
    return lines


class JobsBadge(QLabel):
    """A small card counting the jobs; place() puts it under the frame rate's row."""

    def __init__(self, canvas):
        super().__init__(canvas.native)
        self.setObjectName("jobsBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(BADGE_HEIGHT)
        t = TOKENS
        self.setStyleSheet(f"""
            QLabel#jobsBadge {{
                background: {t['tint']}; color: {t['text']};
                border: 1px solid {t['accent']}; border-radius: {t['radius']}px;
                padding: 0px 8px;
            }}
        """)
        self._right = None
        self.hide()

    def show_jobs(self, lines):
        """Show the count of lines (one per job), or hide when there are none."""
        if not lines:
            self.hide()
            return
        self.setText(badge_text(len(lines)))
        self.setToolTip("\n".join(lines))
        self.adjustSize()
        self._keep_right_edge()
        self.show()
        self.raise_()

    def place(self, right, top):
        """Put the badge's right edge at right and its top at top (canvas px)."""
        self._right = int(round(right))
        self.adjustSize()
        self.setGeometry(self._right - self.width(), int(round(top)), self.width(), BADGE_HEIGHT)

    def _keep_right_edge(self):
        if self._right is not None:
            self.move(self._right - self.width(), self.y())
