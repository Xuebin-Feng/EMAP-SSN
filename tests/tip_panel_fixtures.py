# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared rich-text readers and application base for Tip Panel tests.

Not collected by unittest; import with ``from tests.tip_panel_fixtures import ...``.
"""

import base64
import os
import pathlib
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Hardware_Acceleration  # noqa: E402,F401 - load torch before PySide6
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QTextCharFormat, QTextDocument  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.Studio_Theme import TOKENS  # noqa: E402
from tests.consistency_report import ZERO_WIDTH_SPACE  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

LANGUAGES = SRC_DIR / "resources" / "languages"


# ---------------------------------------------------------------------
# Reading rich text back as a user sees it
# ---------------------------------------------------------------------

def document(rich_text):
    parsed = QTextDocument()
    parsed.setHtml(rich_text)
    return parsed


def pieces(rich_text):
    """(text, format) of every run of formatted text in the document, in order."""
    parsed = document(rich_text)
    block = parsed.begin()
    while block.isValid():
        text = block.text()
        for piece in block.textFormats():
            yield text[piece.start:piece.start + piece.length], QTextCharFormat(piece.format)
        block = block.next()


def chips(rich_text):
    """The texts shown as code chips: on the theme's lighter ground, in the monospace face."""
    found = []
    for text, text_format in pieces(rich_text):
        if (
            text_format.background().color().name() == TOKENS["surface"]
            and text_format.background().style() != Qt.BrushStyle.NoBrush
            and "Mono" in " ".join(text_format.fontFamilies())
        ):
            found.append(clean(text))
    return found


def clean(text):
    """text as it reads: no padding, break places or object-replacement characters."""
    for character in ("￼", ZERO_WIDTH_SPACE, " "):
        text = text.replace(character, "")
    return text.replace(" ", " ").strip()


def images(rich_text):
    """The names (data URIs) of the images in the text, in order."""
    return [text_format.toImageFormat().name()
            for _, text_format in pieces(rich_text) if text_format.isImageFormat()]


def svg_of(name):
    return base64.b64decode(name.split(",", 1)[1]).decode("utf-8")


def plain_text(rich_text):
    return clean(document(rich_text).toPlainText())


def squashed(text):
    """text without any space, to compare texts a chip pads."""
    return "".join(clean(text).split())


def left_press():
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(
        QEvent.Type.MouseButtonPress, QPoint(1, 1), Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )


class WithApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

