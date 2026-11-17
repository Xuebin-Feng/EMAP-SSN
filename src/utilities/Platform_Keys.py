# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Key names as the user's system calls them.

The program's texts name the Ctrl key. Qt on macOS reports the Cmd key as the
Control modifier (and the physical Ctrl key as Meta), and the Viewer treats
both alike, so on a Mac the Ctrl gestures and keys are made with Cmd. The texts
stay as they are in the source and in the translation catalogs; this rewrites
them when they are shown, on macOS only. The same rewrite for the pages the
Viewer serves is ``platformKeyText`` in src/web_ui/page_text.js: keep the two
alike.
"""
from __future__ import annotations

import re
import sys

# "Ctrl" as a whole word: a Latin letter, digit or underscore on either side makes it part of a longer word
# ("Ctrl" in "Control" or "Ctrl2" stays). Chinese letters beside it do not (the Simplified Chinese texts write
# "按Ctrl键" as well as "按 Ctrl 键"), which is why this is not \b.
_CTRL = re.compile(r"(?<![A-Za-z0-9_])Ctrl(?![A-Za-z0-9_])")
_REDO = re.compile(r"(?<![A-Za-z0-9_])Ctrl\+Y(?![A-Za-z0-9_])")


def platform_key_text(text: str, platform: str | None = None) -> str:
    """``text`` with its key names in the platform's form: unchanged except on macOS (``platform`` "darwin").

    There "Ctrl+Y" becomes "Cmd+Shift+Z", the Mac's redo key (the Viewer and the Meta Data page redo on
    Cmd+Shift+Z as well as on Cmd+Y), and every other whole word "Ctrl" becomes "Cmd". ``platform`` defaults
    to ``sys.platform`` as it is when called.
    """
    if (sys.platform if platform is None else platform) != "darwin":
        return text
    return _CTRL.sub("Cmd", _REDO.sub("Cmd+Shift+Z", text))
