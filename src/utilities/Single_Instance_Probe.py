# Copyright 2026 Xuebin Feng
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

from utilities.Application_Windows import notify_existing_instance


APPLICATION_IDS = {
    "viewer": "SSN_Config",
    "tools": "SSN_Tools",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("application", choices=sorted(APPLICATION_IDS))
    args = parser.parse_args(argv)

    app = QCoreApplication.instance() or QCoreApplication([])
    return 0 if notify_existing_instance(APPLICATION_IDS[args.application]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
