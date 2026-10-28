# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's view of its background jobs: the jobs badge, and what the Session card lists.

* **The badge** shows how many background jobs wait or run, over the canvas's top
  right, only while the background scheduler holds a job, under the top-right row
  (the Undo and Redo buttons and the frame rate), right-aligned with it; its
  tooltip lists them. The commands in the command queue (Command_Queue) are not
  background jobs and are not counted. Like the frame rate it is a QLabel on the
  canvas, never in a picture the canvas renders.
* **The rows** (job_lanes) are what the Session card's Background jobs section
  lists: one JobLane for each lane of the scheduler that holds a job, with a
  JobRow for each of its jobs, the running one first, then the ones waiting, then
  the ones that failed lately. Everything a row says comes from the job: its
  description, output, progress and log. Nothing here names a command.
* **ViewerJobs** is the card's side of the scheduler: the lanes it lists, and
  what its buttons do (move, remove, dismiss, open the log).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from desktop.Studio_Theme import TOKENS
from desktop.Viewer_Window import viewer_shows_windows
from utilities import Desktop_Open
from utilities.Localization import Message, display_text

# The badge's height, the frame-rate display's.
BADGE_HEIGHT = 30
# Between the frame-rate row and the badge, in px.
BADGE_GAP = 8


def badge_text(count):
    return display_text(Message("%n job(s)", n=count))


def job_lines(scheduler):
    """One line per background job that waits or runs, for the tooltip."""
    listing = getattr(scheduler, 'listing', None)
    return [f"#{job.job_id} {'running' if running else 'waiting'}: {job.description}"
            for job, running in (listing() if listing is not None else [])]


@dataclass(frozen=True)
class JobRow:
    """One line of the card's list: a job running, waiting, or lately failed.

    ``state`` is "running", "stopping", "queued" or "failed"; ``position`` is the
    place in the lane's line, the running job being 1; ``started_at`` the running
    job's time.monotonic() start; ``progress`` what it says it is doing and
    ``error`` why a failed one failed (each a Message or text); ``log_path`` the
    log it names. ``can_move_up`` and ``can_move_down`` say whether a waiting job
    has a waiting job on that side to change places with.
    """
    job_id: int
    state: str
    description: str
    path: str
    position: int = 0
    started_at: float | None = None
    progress: Any = None
    error: Any = None
    log_path: str | None = None
    can_move_up: bool = False
    can_move_down: bool = False


@dataclass(frozen=True)
class JobLane:
    """A lane of the scheduler and its rows, as the card lists them."""
    name: str
    title: Any
    rows: tuple


def job_lanes(scheduler):
    """The lanes of ``scheduler`` that hold a job or a failure, as JobLane entries; none for no scheduler."""
    snapshot = getattr(scheduler, 'snapshot', None)
    if snapshot is None:
        return []
    entries, failures = snapshot(), scheduler.failures()
    lanes = []
    for name, title in scheduler.lanes():
        active = [entry for entry in entries if entry.lane == name]
        failed = [failure for failure in failures if failure.lane == name]
        if not active and not failed:
            continue
        waiting = [entry for entry in active if not entry.running]
        rows = []
        for entry in active:
            job = entry.job
            state = "queued" if not entry.running else "stopping" if entry.stopping else "running"
            place = waiting.index(entry) if not entry.running else None
            rows.append(JobRow(
                job.job_id, state, job.description, job.output_path, entry.position, entry.started_at,
                entry.progress, None, entry.log_path,
                can_move_up=place is not None and place > 0,
                can_move_down=place is not None and place < len(waiting) - 1,
            ))
        for failure in failed:
            rows.append(JobRow(failure.job_id, "failed", failure.description, "", error=failure.error,
                               log_path=failure.log_path))
        lanes.append(JobLane(name, title, tuple(rows)))
    return lanes


class ViewerJobs:
    """The Session card's side of the Viewer's background jobs: the lanes it lists, and what its buttons do.

    The scheduler is looked up when it is asked, since the card is built before the Viewer has one.
    """

    def __init__(self, viewer):
        self._viewer = viewer

    @property
    def scheduler(self):
        return getattr(self._viewer, 'background_job_scheduler', None)

    def lanes(self):
        return job_lanes(self.scheduler)

    def move(self, job_id, offset):
        scheduler = self.scheduler
        return scheduler is not None and scheduler.move(job_id, offset)

    def remove(self, job_id):
        """Take a waiting job out of its line, or stop the running one; the card asks first for a running one."""
        scheduler = self.scheduler
        return scheduler is not None and scheduler.cancel(job_id)

    def dismiss(self, job_id):
        scheduler = self.scheduler
        return scheduler is not None and scheduler.dismiss_failure(job_id)

    def open_log(self, job_id):
        """Open the log a job names in the system's default text viewer; return whether one was opened.

        The job is found by its number, never by a path.
        """
        scheduler = self.scheduler
        return open_log_file(self._viewer, scheduler.log_path_of(job_id) if scheduler is not None else None)


def open_log_file(viewer, path):
    """Open the log file at path in the system's default text viewer; return whether one was opened.

    A Viewer with no windows (headless, or hidden and not shown) opens nothing, and
    so does a log that is not there.
    """
    if path is None or not os.path.isfile(path) or not viewer_shows_windows(viewer):
        return False
    return Desktop_Open.open_path(path)


class JobsBadge(QLabel):
    """A small card counting the jobs; place() puts it under the top-right row."""

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
