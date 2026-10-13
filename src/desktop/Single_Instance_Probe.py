# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Lightweight launcher probe for activating an existing SSN Qt window."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtCore import QCoreApplication

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from desktop.Desktop_App import (
    CONFIG_DISPLAY_NAME,
    TOOLS_DISPLAY_NAME,
    instance_handover_message,
    notify_existing_instance,
)
from utilities.Output_Streams import configure_output_streams


APPLICATION_IDS = {
    "viewer": "SSN_Config",
    "tools": "SSN_Tools",
}
WINDOW_NAMES = {
    "viewer": CONFIG_DISPLAY_NAME,
    "tools": TOOLS_DISPLAY_NAME,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("application", choices=sorted(APPLICATION_IDS))
    args = parser.parse_args(argv)

    app = QCoreApplication.instance() or QCoreApplication([])
    if not notify_existing_instance(APPLICATION_IDS[args.application]):
        return 1
    print(instance_handover_message(WINDOW_NAMES[args.application]))
    return 0


if __name__ == "__main__":
    configure_output_streams()
    raise SystemExit(main())
