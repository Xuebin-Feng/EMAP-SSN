# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Suppression of the benign QPainter warning burst Qt emits on fullscreen toggle.

Toggling a window into or out of fullscreen makes Windows destroy and recreate the
native surface. While that happens Qt repaints its own window chrome into a backing
store that is briefly null, producing a burst of:

    QPainter::begin: Paint device returned engine == 0, type: 3
    QPainter::setPen: Painter not active
    ...
    QPainter::end: Painter not active, aborted

This was confirmed by tracing the Python stack at the moment each warning fires: it
runs from the application event loop straight into Qt's C++ and back out again, with
no application or vispy frames in between. The painting happens entirely inside Qt,
is transient, and the window renders correctly on both sides of the transition.

Suppression is deliberately narrow — only this cascade is dropped, and every other
Qt message (including genuine paint errors such as "QPainter::drawImage: Image is
null") is passed through untouched. This is safe because nothing in this project
constructs a QPainter, so these messages can only originate inside Qt. If custom
painting is ever added, revisit that assumption.

Two delivery paths need covering, because the entry points differ:

* SSN_Viewer imports vispy, which installs its own Qt message handler at backend
  import time and routes Qt output into ``logging.getLogger("vispy")``. Any
  qInstallMessageHandler call made before vispy loads is silently replaced, so the
  logging filter is what does the work there.
* SSN_Config and SSN_Tools never import vispy, so Qt's default handler writes
  straight to stderr and only qInstallMessageHandler can intercept it.

Installing both covers all three entry points regardless of import order.

Set SSN_SHOW_QT_PAINT_WARNINGS=1 to disable suppression and see the raw output.
"""

import logging
import os
import sys

_BENIGN_FRAGMENTS = (
    "Painter not active",
    "Paint device returned engine == 0",
)

_installed = False


def _is_benign(message):
    """True for the transient fullscreen-transition paint warnings only."""
    if not message.startswith("QPainter::"):
        return False
    return any(fragment in message for fragment in _BENIGN_FRAGMENTS)


class _TransientPaintFilter(logging.Filter):
    """Drops the benign cascade from vispy's logger (SSN_Viewer path)."""

    def filter(self, record):
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not _is_benign(message)


def _qt_message_handler(mode, context, message):
    """Qt message handler for entry points that never load vispy."""
    if _is_benign(message):
        return
    sys.stderr.write(message + "\n")


def install():
    """Install the suppression. Idempotent; safe to call from any entry point."""
    global _installed
    if _installed or os.environ.get("SSN_SHOW_QT_PAINT_WARNINGS"):
        return
    _installed = True

    # Covers SSN_Viewer, where vispy owns the Qt message handler.
    logging.getLogger("vispy").addFilter(_TransientPaintFilter())

    # Covers SSN_Config and SSN_Tools. In SSN_Viewer this handler is later
    # replaced by vispy's, which is harmless — the filter above already applies.
    try:
        from PySide6 import QtCore
    except ImportError:
        return
    QtCore.qInstallMessageHandler(_qt_message_handler)
