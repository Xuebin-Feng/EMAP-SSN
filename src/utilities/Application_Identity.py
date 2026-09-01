# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Desktop identity helpers shared by the Qt application entry points."""

from __future__ import annotations

import sys


PRODUCT_NAME = "EMAP-SSN"
APPLICATION_VERSION = "0.1.0"
PRODUCT_LONG_NAME = (
    f"{PRODUCT_NAME}: Embedding- and Multiple-Alignment-integrated Protein "
    "Sequence Similarity Network Platform"
)
CONFIG_DISPLAY_NAME = f"{PRODUCT_NAME} Configuration"
VIEWER_DISPLAY_NAME = f"{PRODUCT_NAME} Viewer"
TOOLS_DISPLAY_NAME = f"{PRODUCT_NAME} Tools"

TOOLS_DESKTOP_FILE_NAME = "emapssn_tools"
VIEWER_DESKTOP_FILE_NAME = "emapssn"


def configure_linux_qt_desktop_identity(application, desktop_file_name):
    """Let Linux desktops associate a Qt window with its ``.desktop`` file."""
    if not sys.platform.startswith("linux"):
        return

    # Qt expects the desktop-entry basename without the trailing extension.
    # The application name also supplies the X11 WM_CLASS used by older
    # desktops and by XWayland, which the launchers prefer on Wayland sessions.
    application.setApplicationName(desktop_file_name)
    application.setDesktopFileName(desktop_file_name)
