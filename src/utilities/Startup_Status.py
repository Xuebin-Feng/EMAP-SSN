# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What a starting window tells the splash, line by line, while it loads.

The desktop launcher (desktop/Desktop_Launcher.py) shows a splash while Config,
Tools or a Viewer starts. It gives the window it starts SSN_STARTUP_STATUS_FILE,
and each ``report`` appends one JSON line to that file: the text to show, how far
along the start is (0 to 1, or None when that is not known), and the values the
text names. The splash reads the file as it grows, translates the text in its
own language and shows it, and closes when the window writes SSN_GUI_READY_FILE.

Without that variable, as in a window started by hand, ``report`` does nothing and
costs a lookup. Each text is marked for translation under the "Splash" context,
so the catalogs hold it, and the splash shows the translation of the text it reads.

Imports only the standard library, so an entry point can report before Qt, NumPy or
torch load.
"""

from __future__ import annotations

import json
import os

STATUS_FILE_VARIABLE = "SSN_STARTUP_STATUS_FILE"


def QT_TRANSLATE_NOOP(context, text):
    """Mark a text for translation (what Qt's lupdate looks for); it is the text itself."""
    return text


# The texts the launch chain and the windows report, in the order they come.
STARTING = QT_TRANSLATE_NOOP("Splash", "Starting…")
CHECKING_ENVIRONMENT = QT_TRANSLATE_NOOP("Splash", "Checking the Python environment…")
CHECKING_SETTINGS = QT_TRANSLATE_NOOP("Splash", "Checking the settings and inputs…")
LOADING_PYTORCH = QT_TRANSLATE_NOOP("Splash", "Loading PyTorch…")
BUILDING_WINDOW = QT_TRANSLATE_NOOP("Splash", "Building the window…")
BUILDING_NETWORK = QT_TRANSLATE_NOOP("Splash", "Building network display: {nodes} nodes, {edges} edges.")
OPENING_WINDOW = QT_TRANSLATE_NOOP("Splash", "Opening the window…")


def report(text, fraction=None, **values):
    """Tell the splash, if there is one, what the start is doing now; returns whether a line was written.

    ``values`` fill the placeholders of ``text`` ({nodes}, {edges}) after the splash has translated it.
    """
    path = os.environ.get(STATUS_FILE_VARIABLE)
    if not path:
        return False
    line = json.dumps({"text": text, "fraction": fraction, "values": values}, ensure_ascii=False) + "\n"
    try:
        with open(path, "ab") as handle:
            handle.write(line.encode("utf-8"))
    except OSError:
        return False  # a splash that cannot be told is no reason to stop starting
    return True


def finish():
    """The window is up: report nothing more (a process it starts must not write to the splash's file)."""
    os.environ.pop(STATUS_FILE_VARIABLE, None)
