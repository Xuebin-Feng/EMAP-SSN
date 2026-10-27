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
* **Renaming and deleting a partial file.** replace_file(partial, target) is
  the rename, and remove_partial(partial) the deletion after a failure. On
  Windows another program may hold the file for a moment (a virus scan has
  just opened what was written, a sync client has the folder), and either call
  then waits and tries again, for about 2.5 seconds, instead of failing the
  job; a stop request ends the wait at once. A rename that is refused for the
  whole wait raises FileInUseError, which says another program held the file.
  A partial file that cannot be deleted either is named to leave_partial, and
  sweep_leftover_partials() deletes it later: the scheduler calls it when a
  job ends and when the Viewer closes, and it runs when the process ends.
* **Asking the user.** handle.call_in_gui(function, close=None) runs a function on the GUI
  thread, a dialog for instance, and waits for what it returns; a stop request closes
  the dialog (``close``) and ends the call, so a job never leaves one behind.
* **Progress and log.** handle.progress(text) is what the card shows of a running
  job beside its time; handle.set_log_path(path) names the log its process writes,
  which the card offers to open.

Outside a scheduler job, with no handle bound to the thread, every module
function here that speaks to the job does nothing, so code shared with the
foreground or with the VR scheduler runs as it always did; replace_file and
remove_partial still rename and delete there, and wait for a held file as a
job does, but nothing can stop the wait. A worker reaches its handle by taking
it as a second parameter (the scheduler passes it to a worker that has two) or,
from deep inside a helper that cannot take it, with current_job().

JobCancelled derives from BaseException, as asyncio.CancelledError does: the
``except Exception`` blocks of the workers (a subset skipped after an error, an
error reported as a failed command) must not take a cancellation for a failure.

CancellableFile is for the one call that cannot be given check points, the
encoder or writer of a whole file (a PNG, a figure, a workbook): every write to
it is a check point, so the call stops within one chunk of its output.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import io
import os
import sys
import threading
import time

from utilities.Localization import Message


# How long a stopped job waits for the function it asked the GUI thread to run to end, once told to close it.
GUI_CLOSE_WAIT_SECONDS = 5.0

# The waits, in seconds, before each new try at a file another program holds: 2.55 s in all. Most holds end
# in milliseconds (a scan opens a file just written), so the first waits are short; a scan of a large
# picture, or a sync client, can take a second or two, so the last are half a second each, which polls
# cheaply. A longer wait would only keep a user, whose file is held for good, waiting for the failure.
HELD_FILE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5)

# What Windows says of a file another program has open without letting it be renamed or deleted
# (a virus scan, an indexer, a sync client, a viewer): ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION,
# and ERROR_ACCESS_DENIED, which is what replacing a file that is held open gives.
_HELD_FILE_WINERRORS = frozenset({5, 32, 33})


class JobCancelled(BaseException):
    """Raised at a check point of a background job that was told to stop."""


class FileInUseError(PermissionError):
    """Another program held a file for as long as the job waited for it to let go.

    It is raised with a Message, so a window shows the failure translated, and
    str(error) is English for the terminal; the error the system gave is the
    message's {error} and the exception's __cause__.
    """


class GuiCall:
    """A function posted to the GUI thread, and what it came to; ``done`` is set when it has run."""

    def __init__(self, function):
        self.function = function
        self.done = threading.Event()
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.function()
        except Exception as error:  # Handed to the job's thread, which raises it.
            self.error = error
        finally:
            self.done.set()


class JobHandle:
    """One running job, as its worker and the scheduler share it. Every method is safe from any thread."""

    def __init__(self, on_change=None, call_in_gui=None):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stoppers = []
        self._partials = []
        self._progress = None
        self._log_path = None
        # Called, with nothing, after the progress or the log path changed: the scheduler tells its card.
        self._on_change = on_change
        # The scheduler's way to run a function on the GUI thread: it returns a GuiCall.
        self._call_in_gui = call_in_gui

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

    def wait(self, seconds):
        """Wait for up to ``seconds``, or until the job is told to stop; return whether it was."""
        return self._event.wait(seconds)

    # --- the GUI thread ---

    def call_in_gui(self, function, *, close=None):
        """Run ``function()`` on the GUI thread and return what it returns (or raise what it raises).

        For a dialog the job needs the user for. If the job is told to stop meanwhile, ``close()``
        is run on the GUI thread (to close the dialog), the call is waited for a little longer
        and JobCancelled is raised. Without a scheduler to run it on, the function is just called.
        """
        self.raise_if_cancelled()
        if self._call_in_gui is None:
            return function()
        call = self._call_in_gui(function)
        closing = None
        while not call.done.wait(0.05):
            if self.requested and closing is None:
                closing = time.monotonic()
                if close is not None:
                    self._call_in_gui(close)
            if closing is not None and time.monotonic() - closing > GUI_CLOSE_WAIT_SECONDS:
                raise JobCancelled()
        if call.error is not None:
            raise call.error
        self.raise_if_cancelled()
        return call.result

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


# --- renaming and deleting a partial file another program may hold ---


def _on_windows():
    return sys.platform == "win32"


def _held_by_another_program(error):
    """Whether error is Windows saying that another program has the file open or locked.

    Only Windows holds a file against a rename or a deletion; elsewhere an error is the file's own.
    """
    return _on_windows() and getattr(error, "winerror", None) in _HELD_FILE_WINERRORS


def _pause(seconds):
    """Wait ``seconds``; return whether the job on this thread was told to stop meanwhile, which ends the wait at once."""
    handle = getattr(_bound, "handle", None)
    if handle is None:
        time.sleep(seconds)
        return False
    return handle.wait(seconds)


def replace_file(source, target, delays=HELD_FILE_RETRY_DELAYS):
    """Rename ``source`` over ``target`` as os.replace does, waiting for another program that holds either.

    A hold is the Windows errors of _HELD_FILE_WINERRORS; the rename is tried again after each of ``delays``
    (seconds) and then FileInUseError is raised. Any other error is raised at once, and so is JobCancelled
    if the job on this thread is told to stop during a wait. Off Windows it is os.replace.
    """
    for pause in (*delays, None):
        try:
            os.replace(source, target)
            return
        except OSError as error:
            if not _held_by_another_program(error):
                raise
            if pause is None:
                raise FileInUseError(Message(
                    "Could not save {path}: another program kept the file in use "
                    "(a virus scan or a sync client, for example). The system reported: {error}",
                    path=os.fspath(target), error=error,
                )) from error
        if _pause(pause):
            raise JobCancelled()


def remove_partial(path, delays=HELD_FILE_RETRY_DELAYS):
    """Delete the partial file a failed or stopped job leaves; return whether it is gone. It never raises.

    A file that is gone already counts. One that another program holds is tried again after each of ``delays``,
    except that a job told to stop does not wait: it tries once. A file still there is named to leave_partial.
    """
    for pause in (*delays, None):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as error:
            if pause is None or not _held_by_another_program(error):
                break
        if _pause(pause):
            break
    leave_partial(path)
    return False


# The partial files that could not be deleted, until a later try deletes them.
_leftover_lock = threading.Lock()
_leftover_partials = []
_leftover_sweep_registered = False


def leave_partial(path):
    """Name a partial file that could not be deleted, for sweep_leftover_partials() to delete later."""
    global _leftover_sweep_registered
    path = os.fspath(path)
    with _leftover_lock:
        if path not in _leftover_partials:
            _leftover_partials.append(path)
        if not _leftover_sweep_registered:
            _leftover_sweep_registered = True
            atexit.register(sweep_leftover_partials)


def sweep_leftover_partials():
    """Try once more to delete each partial file that could not be deleted; return how many are still there."""
    with _leftover_lock:
        paths = list(_leftover_partials)
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            continue
        with _leftover_lock:
            if path in _leftover_partials:
                _leftover_partials.remove(path)
    with _leftover_lock:
        return len(_leftover_partials)


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
    "CancellableFile",
    "FileInUseError",
    "HELD_FILE_RETRY_DELAYS",
    "JobCancelled",
    "JobHandle",
    "bound_to",
    "cancel_requested",
    "current_job",
    "forget_partial_path",
    "leave_partial",
    "note_partial_path",
    "raise_if_cancelled",
    "remove_partial",
    "replace_file",
    "sweep_leftover_partials",
]
