# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Whether a line of output is an error, a warning or plain output, decided by what it says.

The Console window (desktop/Console_View.py) and the start-up splash's details
(desktop/Splash.py) colour the lines of the program's output by this. It is the
standard library only, so the splash, which imports nothing heavy, can use it.

The stream a line came on says little: Python, Qt and CUDA libraries write their
progress and notices to stderr. So a line is an error or a warning because of its
words: Python's own (a traceback, an exception's name, a warning class), Qt's
tagged messages (utilities.Console_Capture.install_qt_message_handler), the
program's "Error:" and "Warning:" messages, and a few phrases libraries use.

A Qt message carries the severity Qt gave it in its tag, and that decides before any word does:
"[Qt warning] ... QDBusError(...)" is a warning, not an error, though "QDBusError(" looks like one.
"""

from __future__ import annotations

import re

ERROR = "error"
WARNING = "warning"
INFO = "info"

ERROR_PATTERN = re.compile(
    r"(Traceback \(most recent call last\)|^\s*File \".*\", line \d+|\w*(?:Error|Exception)\b[:(]"
    r"|\[Qt (?:critical|fatal)\]|\bfatal\b|CUDA error|out of memory)",
    re.IGNORECASE,
)
WARNING_PATTERN = re.compile(
    r"(Warning\b|\[Qt warning\]|\bwarn(?:ing)?:|Failed to|could not|deprecated)", re.IGNORECASE
)


# The tag utilities.Console_Capture.install_qt_message_handler puts first on a Qt message, and the
# kind it stands for. "[Qt message]" (a type Qt has not named) is not here: its words decide.
QT_TAG_PATTERN = re.compile(r"\s*\[Qt (debug|info|warning|critical|fatal)\]", re.IGNORECASE)
QT_TAG_KINDS = {"debug": INFO, "info": INFO, "warning": WARNING, "critical": ERROR, "fatal": ERROR}


def classify(text: str) -> str:
    """ERROR, WARNING or INFO for one line, from what it says (a Qt message: from its tag)."""
    tag = QT_TAG_PATTERN.match(text)
    if tag:
        return QT_TAG_KINDS[tag.group(1).lower()]
    if ERROR_PATTERN.search(text):
        return ERROR
    if WARNING_PATTERN.search(text):
        return WARNING
    return INFO


class Classifier:
    """classify for lines in order: an indented line after an error or warning has its kind.

    A traceback's frames and source lines, and the source line of a Python
    warning, are indented under the line that names the problem.
    """

    def __init__(self):
        self._previous = INFO

    def kind(self, text: str, *, remember: bool = True) -> str:
        found = classify(text)
        if found == INFO and text[:1] in (" ", "\t") and text.strip() and self._previous != INFO:
            found = self._previous
        if remember:
            self._previous = found
        return found
