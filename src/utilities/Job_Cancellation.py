# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Stopping a background job part-way: the job asks, at points of its own choosing, whether it was told to stop.

The background job scheduler (Background_Job_Scheduler) runs one job at a time
on one thread and, to stop the job it is running, sets that job's CancelToken.
Nothing interrupts the job: its worker calls raise_if_cancelled() between its
steps, and a call after the token was set raises JobCancelled, which unwinds the
worker through its ``finally`` blocks (where each deletes its partial file) and
ends in the scheduler as a cancelled job. Outside a scheduler job, with no token
bound to the thread, every call here does nothing, so code shared with the
foreground or with the VR scheduler runs as it always did.

JobCancelled derives from BaseException, as asyncio.CancelledError does: the
``except Exception`` blocks of the workers (a subset skipped after an error, an
error reported as a failed command) must not take a cancellation for a failure.

CancellableFile is for the one call that cannot be given a check point, the
encoder or writer of a whole file (a PNG, a figure, a workbook): it wraps the
file the call writes to, and every write checks, so the call stops within one
chunk of output.
"""

from __future__ import annotations

from contextlib import contextmanager
import io
import threading


class JobCancelled(BaseException):
    """Raised at a check point of a background job that was told to stop."""


class CancelToken:
    """One job's request to stop: set by the scheduler's thread, read by the job's."""

    def __init__(self):
        self._event = threading.Event()

    def request(self):
        self._event.set()

    @property
    def requested(self):
        return self._event.is_set()


_bound = threading.local()


@contextmanager
def bound_to(token):
    """Make ``token`` the one this thread's check points read, for the block."""
    previous = getattr(_bound, "token", None)
    _bound.token = token
    try:
        yield token
    finally:
        _bound.token = previous


def cancel_requested():
    """Whether the job running on this thread was told to stop (False outside a job)."""
    token = getattr(_bound, "token", None)
    return token is not None and token.requested


def raise_if_cancelled():
    """The check point: raise JobCancelled if the job running on this thread was told to stop."""
    token = getattr(_bound, "token", None)
    if token is not None and token.requested:
        raise JobCancelled()


class CancellableFile:
    """A file object whose every write is a check point.

    It passes a write on to ``handle`` (binary or text, as the writer wants:
    matplotlib tries a bytes write to find out which) unless the job was told to
    stop, and leaves the file for its owner to close. It reports no file
    descriptor, so an encoder that would write straight to the descriptor
    (Pillow does when it finds one) writes through here instead.

    Once a write has raised JobCancelled the object is finished with: later
    writes go nowhere, and it keeps a position of its own (seek and tell, and
    the length of what was written) so that a writer the exception left half
    done can still finish. A ZipFile does, when it is collected, after its owner
    closed the file: it seeks, writes its directory and reads the position
    back, and must find numbers that make sense instead of raising on a closed file.
    """

    def __init__(self, handle):
        self._handle = handle
        self._stopped = False
        self._position = 0

    def _stop(self):
        self._stopped = True
        try:
            self._position = self._handle.tell()
        except (OSError, ValueError):
            self._position = 0

    def write(self, data):
        if self._stopped:
            self._position += len(data)
            return len(data)
        try:
            raise_if_cancelled()
        except JobCancelled:
            self._stop()
            raise
        return self._handle.write(data)

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

    def flush(self):
        if not self._stopped and not self._handle.closed:
            self._handle.flush()

    def tell(self):
        return self._position if self._stopped else self._handle.tell()

    def seek(self, offset, whence=0):
        if not self._stopped:
            return self._handle.seek(offset, whence)
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        return self._position

    def seekable(self):
        return self._handle.seekable()

    def writable(self):
        return True

    def readable(self):
        return False

    @property
    def closed(self):
        return self._handle.closed

    def close(self):
        # The owner's block closes the file; a library that closes "its" stream must not.
        self.flush()


__all__ = [
    "CancelToken",
    "CancellableFile",
    "JobCancelled",
    "bound_to",
    "cancel_requested",
    "raise_if_cancelled",
]
