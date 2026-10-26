# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The two places the probe stands between the program and the desktop: file dialogs and the desktop openers.

* A native file dialog cannot be driven from inside the process (on GNOME it is the portal's, in another
  process). While the probe is on, the QFileDialog static functions the program calls take their answers from
  a queue the agent fills (the ``dialog_answers`` request). With nothing queued a call answers like Cancel, so
  the program never waits for a dialog nobody can see. Every call is recorded.
* utilities.Desktop_Open (open_path, open_url) is wrapped to record its arguments and still does what it did.

Both are undone when the probe turns itself off. Only the GUI thread calls into this module.
"""

from __future__ import annotations

import sys
from collections import deque

from PySide6 import QtCore, QtWidgets

FILE_DIALOG_FUNCTIONS = (
    "getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory",
    "getOpenFileUrl", "getOpenFileUrls", "getSaveFileUrl", "getExistingDirectoryUrl",
)
# The positional parameters after the parent that the functions share: caption, directory, filter.
POSITIONAL = ("parent", "caption", "dir", "filter")
DESKTOP_OPEN_MODULES = ("utilities.Desktop_Open", "Desktop_Open")
DESKTOP_OPEN_FUNCTIONS = ("open_path", "open_url")
FILES_PLACEHOLDER = "{files}"


class FileDialogAnswers:
    """The queue of answers and the replacements of QFileDialog's static functions."""

    def __init__(self, probe):
        self._probe = probe
        self._queue: deque = deque()
        self._originals: dict = {}

    # --- the queue
    def push(self, answers, *, clear=False) -> int:
        if clear:
            self._queue.clear()
        for answer in answers:
            if not isinstance(answer, dict):
                raise ValueError("each answer is an object: {'path': ...}, {'paths': [...]} or {'cancel': true}")
            self._queue.append(dict(answer))
        return len(self._queue)

    def pending(self) -> list:
        return list(self._queue)

    def _take(self, function: str):
        """The first queued answer for ``function`` (an answer may name one with "method", else it fits any)."""
        for position, answer in enumerate(self._queue):
            method = answer.get("method", "*")
            if method in ("*", function) or method.lower() in function.lower():
                del self._queue[position]
                return answer
        return None

    def _expanded(self, text):
        files = str(self._probe.recorder.out / "files")
        return str(text).replace(FILES_PLACEHOLDER, files)

    # --- the replacements
    def install(self) -> None:
        for name in FILE_DIALOG_FUNCTIONS:
            original = getattr(QtWidgets.QFileDialog, name, None)
            if original is None or name in self._originals:
                continue
            self._originals[name] = original
            setattr(QtWidgets.QFileDialog, name, staticmethod(self._replacement(name)))

    def restore(self) -> None:
        for name, original in self._originals.items():
            try:
                setattr(QtWidgets.QFileDialog, name, original)
            except Exception:
                pass
        self._originals.clear()

    def _replacement(self, function):
        def answer_for(*args, **kwargs):
            call = dict(zip(POSITIONAL, args))
            call.update({key: value for key, value in kwargs.items() if key in POSITIONAL or key == "selectedFilter"})
            caption = str(call.get("caption") or "")
            directory = str(call.get("dir") or "")
            answer = self._take(function)
            cancelled = answer is None or bool(answer.get("cancel"))
            self._probe.recorder.event(
                "file_dialog", function=function, caption=caption, directory=directory,
                filter=str(call.get("filter") or ""), queued=answer is not None,
                answer="cancel" if cancelled else self._paths_of(answer),
            )
            return self._result(function, answer)

        return answer_for

    def _paths_of(self, answer) -> list:
        """The paths an answer gives (``path`` or ``paths``), with {files} expanded."""
        chosen = answer["paths"] if "paths" in answer else [answer.get("path", "")]
        return [self._expanded(path) for path in chosen]

    def _result(self, function, answer):
        cancelled = answer is None or answer.get("cancel")
        selected_filter = "" if cancelled else str(answer.get("filter", ""))
        paths = [] if cancelled else self._paths_of(answer)
        is_url = function.endswith("Url") or function.endswith("Urls")
        if function in ("getExistingDirectory", "getExistingDirectoryUrl"):
            value = paths[0] if paths else ""
            return QtCore.QUrl.fromLocalFile(value) if is_url and value else (QtCore.QUrl() if is_url else value)
        if function.endswith("Names") or function.endswith("Urls"):
            values = [QtCore.QUrl.fromLocalFile(path) for path in paths] if is_url else paths
            return values, selected_filter
        value = paths[0] if paths else ""
        return (QtCore.QUrl.fromLocalFile(value) if is_url and value else QtCore.QUrl() if is_url else value), selected_filter


class DesktopOpenRecorder:
    """Records open_path / open_url calls of utilities.Desktop_Open, then lets them through unchanged."""

    def __init__(self, probe):
        self._probe = probe
        self._wrapped: list = []  # (module, name, original)

    def refresh(self) -> None:
        """Wrap the module once it has been imported (looked at on every tick; a dictionary lookup)."""
        for module_name in DESKTOP_OPEN_MODULES:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for name in DESKTOP_OPEN_FUNCTIONS:
                original = getattr(module, name, None)
                if original is None or getattr(original, "_desktop_probe_original", None) is not None:
                    continue
                setattr(module, name, self._wrapper(name, original))
                self._wrapped.append((module, name, original))

    def _wrapper(self, name, original):
        probe = self._probe

        def recorded(*args, **kwargs):
            result = None
            error = None
            try:
                result = original(*args, **kwargs)
                return result
            except BaseException as caught:
                error = f"{type(caught).__name__}: {caught}"
                raise
            finally:
                try:
                    target = args[0] if args else kwargs.get("path", kwargs.get("url"))
                    options = {key: value for key, value in kwargs.items() if key not in ("path", "url")}
                    probe.recorder.event("desktop_open", function=name, target=str(target), options=options,
                                         result=bool(result) if error is None else None, error=error)
                except Exception:
                    pass

        recorded._desktop_probe_original = original
        return recorded

    def restore(self) -> None:
        for module, name, original in self._wrapped:
            try:
                setattr(module, name, original)
            except Exception:
                pass
        self._wrapped.clear()
