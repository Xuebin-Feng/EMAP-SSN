# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Give a test class's windows the shipped window theme, and take it away afterwards.

The windows measure their controls under the Studio stylesheet
(desktop.Studio_Theme.apply_studio_theme), so tests that check sizes build them
under it. The stylesheet belongs to the whole QApplication, which every test
module in a run shares, so the class gets the previous style, palette,
stylesheet and theme colour back when it ends, and later modules measure as
before. The class itself runs in the default theme colour.

Not collected by unittest; import with ``from tests.theme_fixture import ...``.
"""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtGui import QPalette  # noqa: E402

from desktop.Studio_Theme import (  # noqa: E402
    DEFAULT_THEME_COLOR,
    apply_studio_theme,
    set_theme_color,
    theme_color,
)


def apply_theme_for_class(test_class, app):
    """Apply the window theme, in the default colour, to ``app`` until ``test_class`` finishes."""
    style, palette, sheet = app.style().name(), QPalette(app.palette()), app.styleSheet()
    color = theme_color()

    def restore():
        set_theme_color(color)
        app.setStyleSheet(sheet)
        app.setStyle(style)
        app.setPalette(palette)

    apply_studio_theme(app, DEFAULT_THEME_COLOR)
    test_class.addClassCleanup(restore)
