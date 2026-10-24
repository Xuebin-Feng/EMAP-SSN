# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""What a background job shares with the scheduler that runs it: a stop request, its partial files, its progress.

The background job scheduler (Background_Job_Scheduler) runs a job's worker on a
thread of its own and hands the worker a JobHandle: the worker's side of the
job's record. Every job-specific thing the scheduler needs to know comes from
the worker through it, so the scheduler names no command:

* **Stopping.** To stop a running job the scheduler calls handle.request().
  Nothing interrupts the job: its worker calls raise_if_cancelled() between its
  steps, and a call after the request raises JobCancelled, which unwinds the
  worker through its ``finally`` blocks (where each deletes its partial file) and
  ends in the scheduler as a cancelled job. A job that waits on something it
  cannot interrupt, a process, registers the function that stops it with
  handle.on_cancel(); request() calls it once.
* **Partial files.** A worker names each temporary or partial file it creates
  with note_partial(path) before it writes, and forget_partial(path) once the
  file is renamed into place or deleted. When the Viewer closes and a job does
  not stop in time, the scheduler deletes the files still named, so that a
  half-written file does not depend on the job's thread to be deleted.
* **Progress and log.** handle.progress(text) is what the card shows of a running
  job beside its time; handle.set_log_path(path) names the log its process writes,
  which the card offers to open.

Outside a scheduler job, with no handle bound to the thread, every module
function here does nothing, so code shared with the foreground or with the VR
scheduler runs as it always did. A worker reaches its handle by taking it as a
second parameter (the scheduler passes it to a worker that has two) or, from deep
inside a helper that cannot take it, with current_job().

JobCancelled derives from BaseException, as asyncio.CancelledError does: the
``except Exception`` blocks of the workers (a subset skipped after an error, an
error reported as a failed command) must not take a cancellation for a failure.

CancellableFile is for the one call that cannot be given check points, the
encoder or writer of a whole file (a PNG, a figure, a workbook): every write to
it is a check point, so the call stops within one chunk of its output.
"""

from __future__ import annotations

from contextlib import contextmanager
import io
import os
import threading


class JobCancelled(BaseException):
    """Raised at a check point of a background job that was told to stop."""


class JobHandle:
    """One running job, as its worker and the scheduler share it. Every method is safe from any thread."""

    def __init__(self, on_change=None):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stoppers = []
        self._partials = []
        self._progress = None
        self._log_path = None
        # Called, with nothing, after the progress or the log path changed: the scheduler tells its card.
        self._on_change = on_change

    # --- stopping ---

    @property
    def requested(self):
        """Whether the job was told to stop."""
        return self._event.is_set()

    cancelled = requested

    def request(self):
        """Tell the job to stop, and call the stop functions it registered, each once; later calls do nothing."""
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            stoppers, self._stoppers = self._stoppers, []
        for stopper in stoppers:
            self._run_stopper(stopper)

    @staticmethod
    def _run_stopper(stopper):
        try:
            stopper()
        except Exception as error:  # A failed stop must not keep the others, or the scheduler, from stopping.
            print(f"Warning: Could not stop a background job's process: {error}")

    def on_cancel(self, function):
        """Register the function that stops what the worker waits on; it is called once, when the job is told to stop.

        It runs on the thread that tells the job to stop (the GUI thread, or the
        one closing the Viewer), so it must be quick: end a process, set an event.
        If the job was told to stop already, it is called now.
        """
        with self._lock:
            if not self._event.is_set():
                self._stoppers.append(function)
                return
        self._run_stopper(function)

    def raise_if_cancelled(self):
        """The check point: raise JobCancelled if the job was told to stop."""
        if self._event.is_set():
            raise JobCancelled()

    # --- partial files ---

    def note_partial(self, path):
        """Name a temporary or partial file this job is about to write."""
        with self._lock:
            self._partials.append(os.fspath(path))

    def forget_partial(self, path):
        """The named file was renamed into place or deleted: it is the job's to clean up no more."""
        path = os.fspath(path)
        with self._lock:
            if path in self._partials:
                self._partials.remove(path)

    def partial_paths(self):
        with self._lock:
            return list(self._partials)

    # --- progress and log ---

    def progress(self, text):
        """Say what the job is doing now (None for nothing), in the Viewer's own words: it is shown beside the job's time."""
        with self._lock:
            if text == self._progress:
                return
            self._progress = text
        self._changed()

    @property
    def progress_text(self):
        return self._progress

    def set_log_path(self, path):
        """Name the log file the job writes, which the Session card offers to open."""
        with self._lock:
            path = None if path is None else os.fspath(path)
            if path == self._log_path:
                return
            self._log_path = path
        self._changed()

    @property
    def log_path(self):
        return self._log_path

    def _changed(self):
        if self._on_change is not None:
            self._on_change()


# The stop request's earlier name; the tests still say it.
CancelToken = JobHandle

_bound = threading.local()


@contextmanager
def bound_to(handle):
    """Make ``handle`` the one this thread's check points read, for the block."""
    previous = getattr(_bound, "handle", None)
    _bound.handle = handle
    try:
        yield handle
    finally:
        _bound.handle = previous


def current_job():
    """The handle of the job running on this thread, or None outside a job."""
    return getattr(_bound, "handle", None)


def cancel_requested():
    """Whether the job running on this thread was told to stop (False outside a job)."""
    handle = getattr(_bound, "handle", None)
    return handle is not None and handle.requested


def raise_if_cancelled():
    """The check point: raise JobCancelled if the job running on this thread was told to stop."""
    handle = getattr(_bound, "handle", None)
    if handle is not None and handle.requested:
        raise JobCancelled()


def note_partial_path(path):
    """Name a partial file the job on this thread is about to write (nothing outside a job)."""
    handle = getattr(_bound, "handle", None)
    if handle is not None:
        handle.note_partial(path)


def forget_partial_path(path):
    """The partial file is renamed into place or deleted (nothing outside a job)."""
    handle = getattr(_bound, "handle", None)
    if handle is not None:
        handle.forget_partial(path)


class CancellableFile:
    """A file object whose every write is a check point.

    ``CancellableFile(handle)`` wraps a file that is open. ``CancellableFile.at(path,
    mode)`` opens the file at its first write instead, and it is a context manager
    that closes the file: until then nothing holds the file open, so the scheduler
    can delete a partial file a job is stuck far from its first write (Windows will
    not delete a file that is open); and a job told to stop never opens it, since
    the write that would is a check point.

    It passes a write on, binary or text as the file is (matplotlib tries a bytes
    write to find out which; a text file refuses bytes as a file does), unless the
    job was told to stop. It reports no file descriptor, so an encoder that would
    write straight to the descriptor (Pillow does when it finds one) writes through
    here instead.

    Once a write has raised JobCancelled the object is finished with: later
    writes go nowhere, and it keeps a position of its own (seek and tell, and
    the length of what was written) so that a writer the exception left half
    done can still finish. A ZipFile does, when it is collected, after its owner
    closed the file: it seeks, writes its directory and reads the position
    back, and must find numbers that make sense instead of raising on a closed file.
    """

    def __init__(self, handle=None, *, path=None, mode="wb", encoding=None):
        self._handle = handle
        self._path, self._mode, self._encoding = path, mode, encoding
        # Finished: told to stop, or the owner closed the file. Writes then go nowhere.
        self._finished = False
        # Where the file is, kept as it is written and sought, for when it is finished.
        try:
            self._position = handle.tell() if handle is not None else 0
        except (OSError, ValueError):
            self._position = 0

    @classmethod
    def at(cls, path, mode="wb", encoding=None):
        return cls(None, path=os.fspath(path), mode=mode, encoding=encoding)

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        if self._handle is not None and self._path is not None:
            self._finished = True
            self._handle.close()

    def _live(self):
        """Whether writes still reach a file: not finished, and the file is not closed under us."""
        if not self._finished and self._handle is not None and self._handle.closed:
            self._finished = True
        return not self._finished

    def _opened(self):
        if self._handle is None:
            self._handle = open(self._path, self._mode, encoding=self._encoding)
        return self._handle

    def write(self, data):
        if self._live():
            try:
                raise_if_cancelled()
            except JobCancelled:
                self._finished = True
                raise
            if self._handle is None:
                binary = "b" in self._mode
                if binary == isinstance(data, str):
                    raise TypeError("a bytes-like object is required, not 'str'" if binary
                                    else "write() argument must be str, not bytes")
                if not len(data):
                    return 0
            written = self._opened().write(data)
            self._position += len(data)
            return written
        self._position += len(data)
        return len(data)

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

    def flush(self):
        if self._live() and self._handle is not None:
            self._handle.flush()

    def tell(self):
        if self._live() and self._handle is not None:
            return self._handle.tell()
        return self._position

    def seek(self, offset, whence=0):
        if self._live():
            if self._handle is None and whence == 0 and offset == 0:
                return 0   # a writer's seek to the start of a file nothing was written to opens nothing
            self._position = self._opened().seek(offset, whence)
            return self._position
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        return self._position

    def seekable(self):
        return True

    def writable(self):
        return True

    def readable(self):
        return False

    @property
    def closed(self):
        return self._handle is not None and self._handle.closed

    def close(self):
        # The owner's block closes the file; a library that closes "its" stream must not.
        self.flush()


__all__ = [
    "CancelToken",
    "CancellableFile",
    "JobCancelled",
    "JobHandle",
    "bound_to",
    "cancel_requested",
    "current_job",
    "forget_partial_path",
    "note_partial_path",
    "raise_if_cancelled",
]
