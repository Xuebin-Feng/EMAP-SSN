# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Give a test class's windows the shipped window theme, and take it away afterwards.

The windows measure their controls under the Studio stylesheet
(desktop.Studio_Theme.apply_studio_theme), so tests that check sizes build them
under it. The stylesheet belongs to the whole QApplication, which every test
module in a run shares, so the class gets the previous style, palette and
stylesheet back when it ends, and later modules measure as before.

Not collected by unittest; import with ``from tests.theme_fixture import ...``.
"""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtGui import QPalette  # noqa: E402

from desktop.Studio_Theme import apply_studio_theme  # noqa: E402


def apply_theme_for_class(test_class, app):
    """Apply the window theme to ``app`` until ``test_class`` finishes."""
    style, palette, sheet = app.style().name(), QPalette(app.palette()), app.styleSheet()

    def restore():
        app.setStyleSheet(sheet)
        app.setStyle(style)
        app.setPalette(palette)

    apply_studio_theme(app)
    test_class.addClassCleanup(restore)
