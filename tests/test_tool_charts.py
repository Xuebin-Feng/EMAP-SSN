# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tools' pop-up charts' shared look (tools.tool_helpers.Tool_Charts).

The charts themselves are tested with their tools: Sanitize Sequences'
histogram in test_sanitize_sequences, Embedding MSA's regression chart in
test_embedding_msa_regression_chart. Here: the theme colour's setting, the
toolbar's icons, and the window's style, the last in a subprocess because
it restyles the whole QApplication.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.tool_helpers import Tool_Charts  # noqa: E402


# Draws a figure in a Qt window, styles it in teal and prints what it shows.
WINDOW_PROBE = r"""
import json, os, sys
sys.path.insert(0, {src!r})
import PySide6.QtWidgets  # matplotlib takes the binding already imported
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from desktop.Studio_Theme import studio_stylesheet, theme_tokens
from tools.tool_helpers import Tool_Charts

figure = plt.figure(figsize=(10, 6))
manager = figure.canvas.manager
logo = QIcon(os.path.join({root!r}, "src", "bin", "logos", "tool_logo.ico")).pixmap(32).toImage()
icon_before = manager.window.windowIcon().pixmap(32).toImage()
styled = Tool_Charts.style_chart_window(figure, "teal", {root!r})
application = QApplication.instance()
toolbar = manager.toolbar
colours = {{}}
for callback, action in toolbar._actions.items():
    image = action.icon().pixmap(QSize(24, 24), 2.0).toImage()
    colours[callback] = sorted({{
        image.pixelColor(x, y).name() for x in range(image.width()) for y in range(image.height())
        if image.pixelColor(x, y).alpha() == 255
    }})
size = figure.canvas.sizeHint()
print(json.dumps({{
    "styled": styled,
    "stylesheet": application.styleSheet() == studio_stylesheet(theme_tokens("teal")),
    "palette_window": application.palette().window().color().name(),
    "font": application.font().families(),
    "icon_is_logo": manager.window.windowIcon().pixmap(32).toImage() == logo,
    "icon_was_logo": icon_before == logo,
    "movable": toolbar.isMovable(),
    "icon_size": [toolbar.iconSize().width(), toolbar.iconSize().height()],
    "colours": colours,
    "height": [manager.window.height(), size.height() + toolbar.sizeHint().height()],
    "width": [manager.window.width(), size.width()],
}}))
"""


class ToolChartsTests(unittest.TestCase):
    def test_the_setting_name_is_the_tools_windows(self):
        source = (SRC / "desktop" / "Desktop_App.py").read_text(encoding="utf-8")
        self.assertIn(f'TOOLS_THEME_SETTING = "{Tool_Charts.TOOLS_THEME_SETTING}"', source)

    def test_every_toolbar_action_shows_a_bundled_lucide_icon(self):
        from matplotlib.backends.backend_qt import NavigationToolbar2QT
        from desktop.Studio_Theme import LUCIDE_DIR
        from tests.test_studio_theme import load_update_icons

        callbacks = {item[3] for item in NavigationToolbar2QT.toolitems if item[0] is not None}
        self.assertEqual(set(Tool_Charts.TOOLBAR_ICONS), callbacks)
        listed = set(load_update_icons().ICONS)
        for name in Tool_Charts.TOOLBAR_ICONS.values():
            with self.subTest(icon=name):
                self.assertTrue((LUCIDE_DIR / f"{name}.svg").is_file())
                self.assertIn(name, listed)

    def test_a_backend_without_a_qt_window_is_left_alone(self):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure = Figure()
        FigureCanvasAgg(figure)
        self.assertFalse(Tool_Charts.style_chart_window(figure, "teal", str(ROOT)))
        with mock.patch.object(plt, "show") as show, mock.patch.object(plt, "close") as close:
            Tool_Charts.show_chart(figure, "teal", str(ROOT))
        show.assert_called_once_with()
        close.assert_called_once_with(figure)

    def test_a_window_that_cannot_be_styled_still_shows(self):
        import matplotlib.pyplot as plt

        figure = object()
        printed = io.StringIO()
        with mock.patch.object(Tool_Charts, "style_chart_window", side_effect=RuntimeError("no Qt")), \
                mock.patch.object(plt, "show") as show, mock.patch.object(plt, "close") as close, \
                contextlib.redirect_stdout(printed):
            Tool_Charts.show_chart(figure, "teal", str(ROOT))
        show.assert_called_once_with()
        close.assert_called_once_with(figure)
        self.assertIn("the chart window keeps matplotlib's look (no Qt)", printed.getvalue())

    def test_the_qt_window_takes_the_tools_windows_look(self):
        from desktop.Studio_Theme import theme_tokens

        environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "-c", WINDOW_PROBE.format(src=str(SRC), root=str(ROOT))],
            capture_output=True, text=True, encoding="utf-8", env=environment, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(result.stdout.strip().splitlines()[-1])
        tokens = theme_tokens("teal")
        self.assertTrue(seen["styled"])
        self.assertTrue(seen["stylesheet"], "the Studio stylesheet in the theme colour")
        self.assertEqual(seen["palette_window"], tokens["window"])
        self.assertEqual(seen["font"][0], "Noto Sans")
        self.assertFalse(seen["icon_was_logo"], "matplotlib's own icon first")
        self.assertTrue(seen["icon_is_logo"], "then the Tools window's")
        self.assertFalse(seen["movable"])
        self.assertEqual(seen["icon_size"], [Tool_Charts.TOOLBAR_ICON_SIZE] * 2)
        self.assertEqual(set(seen["colours"]), set(Tool_Charts.TOOLBAR_ICONS))
        for callback, colours in seen["colours"].items():
            with self.subTest(action=callback):
                # The Lucide strokes, in the theme's icon grey.
                self.assertIn(tokens["text_subtle"], colours)
        # The window keeps the figure's 10 x 6 inches under the restyled toolbar.
        self.assertEqual(seen["height"][0], seen["height"][1])
        self.assertEqual(seen["width"][0], seen["width"][1])


if __name__ == "__main__":
    unittest.main()
