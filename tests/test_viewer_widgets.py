# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's widgets (desktop.Viewer_Widgets): its info card and the
info button that opens it, its sidebar's rounded card and buttons, and the
Zoom box.

The Viewer lists its mouse and keyboard actions in one translated text. The
card shows the session (the counts, and each input file as a read-only path;
the sequence set, network and layout cache are fixed, the alignment and
embedding have a pick button that types their command), then reads that text, as shipped in English
and in Chinese, into key caps beside what each action does, on a light tint
of the theme colour. It is a child of the canvas: it closes only by its
cross or the button, takes no mouse event from the canvas and none of its
own to the canvas, and never takes the keyboard. A sidebar button
centres its label in the room after its icon. The Zoom box shows the view's
zoom as a percentage, steps it by a factor, as the wheel zooms (a bigger
percentage is closer), and reports only what the user set. The threshold box's step comes from the range of the scores, and
each of its numbers is a position of the threshold slider.
"""

import ast
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ElementTree

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import (  # noqa: E402
    QCoreApplication, QKeyCombination, QPoint, QPointF, QRect, QSize, Qt, QTranslator,
)
from PySide6.QtGui import (  # noqa: E402
    QColor, QContextMenuEvent, QFontMetrics, QGuiApplication, QImage, QKeySequence, QPainter, QRegion, QShortcut,
    QWheelEvent,
)
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget  # noqa: E402

from desktop.Desktop_App import fit_buttons_to_text  # noqa: E402
from desktop.Studio_Theme import TOKENS, key_cap_qss, studio_icon  # noqa: E402
from desktop.Viewer_Widgets import (  # noqa: E402
    FIXED_NOTE,
    PICK_FILES,
    SESSION_FILES,
    SIDEBAR_INSET,
    ControlsButton,
    ControlsCard,
    PathField,
    SidebarButton,
    SLIDER_BOX_HEIGHT,
    SidebarPanel,
    THRESHOLD_STEPS,
    ThresholdSpinBox,
    ZOOM_LABEL_GAP,
    ZOOM_ROOM,
    ZOOM_STEP,
    ZOOM_UNIT_GAP,
    ZoomBox,
    ZoomSpinBox,
    controls_rows,
    display_path,
    shown_zoom,
    zoom_decimals,
)
from desktop.Viewer_State import alias_form  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from utilities import Help_Pages  # noqa: E402

VIEWER = SRC / "EMAPSSN_Viewer.py"
ZH_CATALOG = SRC / "resources" / "languages" / "zh_CN" / "emapssn_zh_CN.ts"


def shipped_actions():
    """The Viewer's list of actions, as its source passes it to translate()."""
    tree = ast.parse(VIEWER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "translate"
                and len(node.args) == 2 and isinstance(node.args[1], ast.Constant)
                and str(node.args[1].value).startswith("[ENTER]")):
            return node.args[1].value
    raise AssertionError("The Viewer no longer lists its actions")


def chinese_actions(source):
    """The Chinese catalog's text for source."""
    for message in ElementTree.parse(ZH_CATALOG).getroot().iter("message"):
        if message.findtext("source") == source:
            return message.findtext("translation")
    raise AssertionError("The Chinese catalog has no text for the Viewer's actions")


def flush(app):
    for _ in range(4):
        app.processEvents()


class ControlsRowsTests(unittest.TestCase):
    def test_the_shipped_actions_read_as_keys_and_what_they_do(self):
        rows = controls_rows(shipped_actions())
        self.assertEqual(rows, [
            (["ENTER"], "Command"),
            (["LeftClick"], "Highlight"),
            (["RightClick"], "Select/Clear"),
            (["Scroll"], "Zoom"),
            (["LeftClick", "Shift/Ctrl"], "Copy Node Header/Sequence"),
            (["LeftClick", "Drag"], "Pan"),
            (["RightClick", "Drag"], "GroupSelect/MoveNodes"),
        ])

    def test_the_chinese_actions_read_the_same_way(self):
        rows = controls_rows(chinese_actions(shipped_actions()))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0], (["回车"], "命令"))
        self.assertEqual(rows[4][0], ["左键", "Shift/Ctrl"])
        self.assertTrue(all(keys and description for keys, description in rows), rows)

    def test_an_action_without_keys_is_all_description(self):
        self.assertEqual(controls_rows("[A] One |  | Two words | [B + C] Three"),
                         [(["A"], "One"), ([], "Two words"), (["B", "C"], "Three")])
        self.assertEqual(controls_rows(""), [])


class RecordingCanvas(QWidget):
    """Stands in for the VisPy canvas the card sits on: it keeps every mouse and key event it is sent.

    ``keys`` are the Qt keys pressed on it, and ``on_copy``, when set, is
    called for each Copy key press it receives, as the Viewer's command line
    copies its own text when it is open.
    """

    def __init__(self):
        super().__init__()
        self.seen = []
        self.keys = []
        self.on_copy = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _keep(self, kind, event):
        self.seen.append(kind)
        event.accept()

    def mousePressEvent(self, event):
        self._keep("press", event)

    def mouseReleaseEvent(self, event):
        self._keep("release", event)

    def mouseMoveEvent(self, event):
        self._keep("move", event)

    def mouseDoubleClickEvent(self, event):
        self._keep("double", event)

    def wheelEvent(self, event):
        self._keep("wheel", event)

    def contextMenuEvent(self, event):
        self._keep("context", event)

    def keyPressEvent(self, event):
        self.keys.append(event.key())
        if self.on_copy is not None and event.matches(QKeySequence.StandardKey.Copy):
            self.on_copy()
        self._keep("key", event)


CTRL = Qt.KeyboardModifier.ControlModifier
NO_MODIFIER = Qt.KeyboardModifier.NoModifier


def chords(standard_key):
    """Every key chord the platform binds to a standard key, as (key, modifiers).

    For Copy that is Ctrl+C (Qt's Control is the Command key on macOS), the
    Ctrl+Insert of Windows and Linux, and a Copy key where there is one.
    """
    return [(sequence[0].key(), sequence[0].keyboardModifiers())
            for sequence in QKeySequence.keyBindings(standard_key)]


def chord_name(chord):
    """A key chord as text, for a failure's message."""
    return QKeySequence(QKeyCombination(chord[1], chord[0])).toString()


def send_wheel(widget, delta=-120, position=None):
    """A wheel turn over widget, delivered as the application does."""
    point = QPointF(widget.rect().center() if position is None else position)
    event = QWheelEvent(point, QPointF(widget.mapToGlobal(point.toPoint())), QPoint(0, 0), QPoint(0, delta),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def send_context_menu(widget):
    event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, widget.rect().center(),
                              widget.mapToGlobal(widget.rect().center()))
    QApplication.sendEvent(widget, event)


def assert_two_halves(case, card):
    """The controls' keys and descriptions are two equal halves of the card's content width, nothing cut short.

    Measured from the geometry of the widgets, in the pixels of the card's body.
    """
    grid = card.controls_grid.geometry()
    middle = grid.x() + grid.width() / 2
    # The card's whole content width: what the body has, less the margin that keeps the scroll bar off the text.
    case.assertEqual(grid.right(), card.body.width() - card.body.layout().contentsMargins().right() - 1)
    # The same width as the session's rows above, which end at the pick buttons and at the fixed fields.
    for widget in (*card.pick_buttons.values(), *(field for field in card.path_fields.values() if field.fixed)):
        case.assertEqual(grid.right(), widget.mapTo(card.body, QPoint(0, 0)).x() + widget.width() - 1)
    for holder, label in zip(card.key_holders, card.descriptions):
        case.assertEqual(holder.geometry().x(), grid.x())
        case.assertAlmostEqual(label.geometry().x(), middle, delta=1)
        # Each fits its half, the keys with the gap after them.
        case.assertLessEqual(holder.geometry().right() + ControlsCard.COLUMN_GAP, middle + 1)
        case.assertLessEqual(label.geometry().right(), grid.right() + 1)
        case.assertGreaterEqual(label.width(), label.sizeHint().width(), label.text())
        case.assertGreaterEqual(holder.width(), holder.sizeHint().width())
    return grid, middle


class DisplayPathTests(unittest.TestCase):
    def test_an_empty_setting_is_no_path_and_any_other_is_absolute_and_normalised(self):
        for empty in (None, "", "   ", "None", " None "):
            self.assertEqual(display_path(empty), "", repr(empty))
        self.assertEqual(display_path("a/b/../c.h5"), os.path.abspath("a/c.h5"))
        absolute = os.path.join(os.path.abspath(os.sep), "data", "x", "..", "set.fasta")
        self.assertEqual(display_path(absolute), os.path.join(os.path.abspath(os.sep), "data", "set.fasta"))


class ControlsCardTests(unittest.TestCase):
    """The session and controls card, a child of the canvas it sits on."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        # The project's folders, as the Config lays them out.
        self.files = {
            "node_fasta": self.root / "Input_Files" / "Sequence_Sets" / "set.fasta",
            "network_hdf5": self.root / "Input_Files" / "Networks_EValues" / "network.h5",
            "msa": self.root / "Input_Files" / "Multiple_Alignments" / "aln.fasta",
            "embedding": self.root / "Embeddings" / "embed.h5",
            "layout_cache": self.root / "Cache_Files" / "Saved_Layouts" / "cache" / "version_00.h5",
        }
        for path in self.files.values():
            path.parent.mkdir(parents=True)
            path.write_text("x")
        self.cache_folder = self.files["layout_cache"].parent
        self.info = {
            "node_count": 1234, "visible_node_count": 1200,
            "edge_count": 98765432, "shown_edge_count": 345678,
            "inputs": {key: str(path) for key, path in self.files.items()},
            "directories": {"INPUT_FILE_DIR": str(self.root / "Input_Files"),
                            "CACHE_FILE_DIR": str(self.root / "Cache_Files"),
                            "ANALYSIS_RESULT_DIR": str(self.root / "Analysis_Results"),
                            "EMBED_DIR": str(self.root / "Embeddings")},
            "project_root": str(self.root),
        }
        self.aliased = {
            "node_fasta": "$input_file$/Sequence_Sets/set.fasta",
            "network_hdf5": "$input_file$/Networks_EValues/network.h5",
            "msa": "$input_file$/Multiple_Alignments/aln.fasta",
            "embedding": "$embedding$/embed.h5",
            "layout_cache": "$cache_file$/Saved_Layouts/cache/version_00.h5",
        }
        self.session_calls = 0
        self.area = None
        # Whether the Viewer's command line is being typed in, which the card asks when a key comes.
        self.command_line_open = False
        # What the pick buttons type; no test runs a command or opens a dialog.
        self.command = mock.Mock()
        # No click in these tests may open a real browser.
        patcher = mock.patch("desktop.Viewer_Widgets.QDesktopServices")
        self.services = patcher.start()
        self.addCleanup(patcher.stop)
        # Clear of the screen's left edge, and wide enough for the card in the offscreen platform's large
        # font, where the counts line alone is 700 px.
        self.host = RecordingCanvas()
        self.addCleanup(self.host.deleteLater)
        self.host.setGeometry(10, 0, 1000, 760)
        self.button = ControlsButton(
            "Session and controls", shipped_actions(), self.host, host=self.host, session=self.read_session,
            controls_title="Mouse and keyboard controls", area=lambda: self.area or self.host.rect().adjusted(8, 8, -8, -8),
            command=self.command, command_line_open=lambda: self.command_line_open)
        self.button.move(20, 700)
        self.host.show()
        flush(self.app)

    def read_session(self):
        self.session_calls += 1
        return self.info

    @staticmethod
    def tip(key, path):
        """A path field's tooltip: the full path, and after it for a fixed file the note that it is."""
        return path if key in PICK_FILES else path + "\n" + FIXED_NOTE

    def open_card(self):
        card = self.button.toggle_card()
        flush(self.app)
        self.assertTrue(card.isVisible())
        return card

    def card_rect_in_host(self, card):
        return card.card_rect().translated(card.pos().toPointF())

    def select_in(self, field):
        """Select in a field with the mouse, as a user does: press near its start, drag to its end, release."""
        y = field.height() // 2
        QTest.mousePress(field, Qt.MouseButton.LeftButton, pos=QPoint(10, y))
        QTest.mouseMove(field, QPoint(field.width() - 10, y))
        QTest.mouseRelease(field, Qt.MouseButton.LeftButton, pos=QPoint(field.width() - 10, y))
        self.assertTrue(field.hasSelectedText())

    def press(self, key, modifiers=NO_MODIFIER):
        """A key press as the keyboard delivers it, to the canvas that has the keyboard; returns the keys the canvas saw.

        Not to a field: no field has the keyboard, and a key sent straight to
        one is copied by the field itself, however the card is wired.
        """
        self.host.keys.clear()
        QTest.keyClick(self.host, key, modifiers)
        return list(self.host.keys)

    def test_the_button_names_the_card_and_leaves_the_keyboard_with_the_canvas(self):
        self.assertEqual(self.button.toolTip(), "Session and controls")
        self.assertEqual(self.button.accessibleName(), "Session and controls")
        self.assertEqual(self.button.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertFalse(self.button.icon().isNull())
        self.assertEqual(self.button.size(), QSize(ControlsButton.SIZE, ControlsButton.SIZE))

    def test_the_card_lists_the_counts_and_files_then_the_actions(self):
        card = self.open_card()
        # Thousands separators; the visible and shown parts when they are less than the whole.
        self.assertEqual(card.count_labels["nodes"].text(), "1,234 (1,200 visible)")
        self.assertEqual(card.count_labels["edges"].text(), "98,765,432 (345,678 shown)")
        # The fixed files, then the two that can be changed.
        self.assertEqual([card.file_labels[key].text() for key, _label in SESSION_FILES],
                         ["Sequence Set:", "Network:", "Layout Cache:", "Alignment:", "Embedding:"])
        self.assertEqual([key for key, _label in SESSION_FILES],
                         ["node_fasta", "network_hdf5", "layout_cache", "msa", "embedding"])
        for key, _label in SESSION_FILES:
            # The field is written with the folder's alias; its tooltip is the whole path.
            self.assertEqual(card.path_fields[key].text(), self.aliased[key], key)
            self.assertEqual(card.path_fields[key].toolTip(), self.tip(key, os.path.normpath(self.info["inputs"][key])), key)
        self.assertEqual([label.text() for label in card.descriptions],
                         [description for _keys, description in controls_rows(shipped_actions())])
        self.assertEqual([cap.text() for cap in card.key_caps],
                         ["ENTER", "LeftClick", "RightClick", "Scroll", "LeftClick", "Shift/Ctrl",
                          "LeftClick", "Drag", "RightClick", "Drag"])
        # Each description shows whole: no row is cut short.
        for label in card.descriptions:
            self.assertGreaterEqual(label.width(), label.sizeHint().width(), label.text())

    def test_the_key_caps_take_their_look_from_the_shared_key_cap_definition(self):
        # The tip panel draws its code chips from the same definition: one look for both.
        card = self.open_card()
        self.assertIn(f"QLabel#controlsKey {{ {key_cap_qss(TOKENS)} }}", card.styleSheet())
        self.assertIn("border-radius: 5px; padding: 0px 6px;", key_cap_qss(TOKENS))
        self.assertIn(f"border-bottom: 2px solid {TOKENS['field_border_hover']}", key_cap_qss(TOKENS))

    def test_the_full_path_is_absolute_and_normalised(self):
        self.info["inputs"]["node_fasta"] = os.path.join(
            str(self.root), "Cache_Files", "..", "Input_Files", "Sequence_Sets", "set.fasta")
        self.info["inputs"]["network_hdf5"] = "relative/network.h5"
        card = self.open_card()
        self.assertEqual(card.path_fields["node_fasta"].toolTip(), self.tip("node_fasta", str(self.files["node_fasta"])))
        self.assertEqual(card.path_fields["node_fasta"].text(), self.aliased["node_fasta"])
        # A relative setting is made absolute (against the working folder) before anything else is done to it.
        absolute = os.path.abspath("relative/network.h5")
        self.assertEqual(card.path_fields["network_hdf5"].toolTip(), self.tip("network_hdf5", absolute))
        self.assertEqual(card.path_fields["network_hdf5"].text(), absolute)

    def test_each_base_folder_writes_its_files_with_its_alias_and_the_rest_by_the_root_or_in_full(self):
        elsewhere = self.root.parent / "other_project" / "version_00.h5"
        inputs = {
            "node_fasta": self.root / "Input_Files" / "a" / "x.fasta",
            "network_hdf5": self.root / "Cache_Files" / "n.h5",
            "msa": self.root / "Analysis_Results" / "m.fasta",
            "embedding": self.root / "Embeddings" / "e" / "e.h5",
            "layout_cache": self.root / "somewhere" / "version_00.h5",
        }
        self.info["inputs"] = {key: str(path) for key, path in inputs.items()}
        card = self.open_card()
        expected = {
            "node_fasta": "$input_file$/a/x.fasta", "network_hdf5": "$cache_file$/n.h5",
            "msa": "$analysis_result$/m.fasta", "embedding": "$embedding$/e/e.h5",
            "layout_cache": "somewhere/version_00.h5",   # under no base: relative to the project root
        }
        for key, text in expected.items():
            self.assertEqual(card.path_fields[key].text(), text, key)
            # The tooltip is the whole path, always.
            self.assertEqual(card.path_fields[key].toolTip(), self.tip(key, str(inputs[key])), key)
        # Not under the project root either: absolute.
        self.info["inputs"]["layout_cache"] = str(elsewhere)
        card.refresh()
        self.assertEqual(card.path_fields["layout_cache"].text(), str(elsewhere))
        self.assertEqual(card.path_fields["layout_cache"].toolTip(), self.tip("layout_cache", str(elsewhere)))
        # An owner that gives no folders (the card then knows no alias and no root) shows the full paths.
        del self.info["directories"], self.info["project_root"]
        card.refresh()
        for key, path in inputs.items():
            shown = elsewhere if key == "layout_cache" else path
            self.assertEqual(card.path_fields[key].text(), str(shown), key)

    def test_a_card_in_a_window_not_shown_yet_is_open_all_the_same(self):
        self.host.hide()
        card = self.button.toggle_card()
        self.assertTrue(self.button.card_is_open())
        self.assertFalse(card.isVisible())
        self.assertIsNone(self.button.toggle_card())
        self.assertFalse(self.button.card_is_open())

    def test_a_whole_count_shows_alone(self):
        self.info.update(visible_node_count=1234, shown_edge_count=98765432)
        card = self.open_card()
        self.assertEqual((card.count_labels["nodes"].text(), card.count_labels["edges"].text()),
                         ("1,234", "98,765,432"))

    def test_a_card_without_a_session_lists_only_the_actions(self):
        button = ControlsButton("Mouse and keyboard controls", shipped_actions(), self.host, host=self.host)
        card = button.toggle_card()
        flush(self.app)
        self.assertEqual(card.path_fields, {})
        self.assertEqual(len(card.descriptions), 7)
        # Its one heading, in the header row with the link and the cross, is the controls'.
        self.assertEqual(card.session_heading.text(), "Mouse and keyboard controls")
        self.assertIsNone(card.controls_heading)
        self.assertEqual([label.text() for label in card.findChildren(QLabel) if label.objectName() == "controlsHeading"],
                         ["Mouse and keyboard controls"])

    @staticmethod
    def darkest_ink(widget):
        """The darkest grey in a widget's picture on white: its text colour where the text is solid."""
        image = QImage(widget.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        widget.render(image)
        return min(image.pixelColor(x, y).lightness() for x in range(image.width()) for y in range(image.height()))

    def test_the_card_has_two_headings_in_the_text_colour_and_no_title(self):
        card = self.open_card()
        self.assertEqual(card.session_heading.text(), "Session")
        self.assertEqual(card.controls_heading.text(), "Mouse and keyboard controls")
        # Nothing on the card is the old title; the name stays for the screen reader, as the button's tooltip.
        self.assertEqual(card.accessibleName(), "Session and controls")
        self.assertEqual(self.button.toolTip(), "Session and controls")
        self.assertNotIn("Session and controls", [label.text() for label in card.findChildren(QLabel)])
        headings = [label for label in card.findChildren(QLabel) if label.objectName() == "controlsHeading"]
        self.assertEqual(headings, [card.session_heading, card.controls_heading])
        text = QColor(TOKENS["text"])
        for heading in headings:
            self.assertEqual(heading.palette().color(heading.foregroundRole()).name(), TOKENS["text"])
            # Painted, not muted: the ink is the text colour, and darker than the labels' soft grey.
            self.assertLessEqual(abs(self.darkest_ink(heading) - text.lightness()), 16, heading.text())
        self.assertGreater(self.darkest_ink(card.file_labels["node_fasta"]), self.darkest_ink(card.session_heading) + 20)
        self.assertNotEqual(TOKENS["text"], TOKENS["text_muted"])

    def test_the_nodes_and_the_edges_share_one_line(self):
        card = self.open_card()
        names = {label.text(): label for label in card.findChildren(QLabel) if label.objectName() == "controlsLabel"}
        row = [names["Nodes:"], card.count_labels["nodes"], names["Edges:"], card.count_labels["edges"]]

        def where(widget):
            return widget.mapTo(card, QPoint(0, 0))

        centres = [where(widget).y() + widget.height() / 2 for widget in row]
        self.assertLessEqual(max(centres) - min(centres), 1.5, centres)
        lefts = [where(widget).x() for widget in row]
        self.assertEqual(lefts, sorted(lefts))
        for before, after in zip(row, row[1:]):
            self.assertLess(where(before).x() + before.width(), where(after).x())
        # A gap between the pairs wider than the one inside a pair, so they read as two.
        inside = where(row[1]).x() - (where(row[0]).x() + row[0].width())
        between = where(row[2]).x() - (where(row[1]).x() + row[1].width())
        self.assertGreater(between, inside + 8)
        # It is above the files, and the whole line fits the card.
        self.assertLess(max(centres), where(card.file_labels["node_fasta"]).y())
        self.assertLessEqual(where(row[3]).x() + row[3].width(), card.body.mapTo(card, QPoint(card.body.width(), 0)).x())
        # The parts show on the line when they apply, and a whole count shows alone.
        self.assertEqual((row[1].text(), row[3].text()), ("1,234 (1,200 visible)", "98,765,432 (345,678 shown)"))

    def test_the_list_of_commands_link_is_in_the_header_row_beside_the_cross(self):
        card = self.open_card()
        link, cross, heading = card.commands_button, card.close_button, card.session_heading

        def where(widget):
            return widget.mapTo(card, QPoint(0, 0))

        for widget in (link, cross):
            self.assertAlmostEqual(where(widget).y() + widget.height() / 2,
                                   where(heading).y() + heading.height() / 2, delta=3)
        # The heading, then the link, then the cross at the card's right edge.
        self.assertLess(where(heading).x() + heading.width(), where(link).x())
        gap = where(cross).x() - (where(link).x() + link.width())
        self.assertGreaterEqual(gap, 0)
        self.assertLessEqual(gap, 12)
        self.assertGreater(where(cross).x(), card.card_rect().center().x())
        # Above the scrolling sections, so it stays in view when they scroll.
        self.assertLessEqual(where(link).y() + link.height(), where(card.scroll).y())
        self.assertFalse(card.body.isAncestorOf(link))
        # It is the link it was: text, icon, style, tooltip and the hand cursor.
        self.assertEqual((link.text(), link.objectName()), ("List of commands", "controlsLink"))
        self.assertFalse(link.icon().isNull())
        self.assertEqual(link.cursor().shape(), Qt.CursorShape.PointingHandCursor)
        found = link.toolTip().startswith("Open the list of commands")
        self.assertTrue(found or link.toolTip().startswith("The list of commands is not installed"))
        self.assertEqual(link.isEnabled(), found)
        # With no card section left holding it, the controls end with the last row.
        last = card.descriptions[-1]
        self.assertGreater(last.mapTo(card.body, QPoint(0, 0)).y() + last.height(), card.body.height() - 20)

    def card_with_console(self, console):
        """A card whose owner gave it the function that opens the Console, as the Viewer does."""
        button = ControlsButton(
            "Session and controls", shipped_actions(), self.host, host=self.host, session=self.read_session,
            controls_title="Mouse and keyboard controls",
            area=lambda: self.host.rect().adjusted(8, 8, -8, -8),
            command=self.command, command_line_open=lambda: False, console=console)
        button.move(60, 700)
        button.show()
        card = button.toggle_card()
        flush(self.app)
        self.assertTrue(card.isVisible())
        return button, card

    def test_the_console_link_is_in_the_header_row_before_the_list_of_commands_link(self):
        button, card = self.card_with_console(lambda: None)
        console, commands, cross, heading = (card.console_button, card.commands_button, card.close_button,
                                             card.session_heading)

        def where(widget):
            return widget.mapTo(card, QPoint(0, 0))

        for widget in (console, commands, cross):
            self.assertAlmostEqual(where(widget).y() + widget.height() / 2,
                                   where(heading).y() + heading.height() / 2, delta=3)
        # The heading, the Console link, the list of commands' link and the cross, in that order, none over another.
        self.assertLess(where(heading).x() + heading.width(), where(console).x())
        self.assertLessEqual(where(console).x() + console.width(), where(commands).x())
        self.assertLessEqual(where(commands).x() + commands.width(), where(cross).x())
        # In the header, above the scrolling sections, and the whole row inside the card.
        self.assertLessEqual(where(console).y() + console.height(), where(card.scroll).y())
        self.assertFalse(card.body.isAncestorOf(console))
        self.assertLessEqual(where(cross).x() + cross.width(), card.card_rect().right())
        # The link looks like its neighbour and says what it opens.
        self.assertEqual((console.text(), console.objectName()), ("Console", "controlsLink"))
        self.assertFalse(console.icon().isNull())
        self.assertEqual(console.cursor().shape(), Qt.CursorShape.PointingHandCursor)
        self.assertTrue(console.toolTip().startswith("Show everything this Viewer has printed"))
        self.assertTrue(console.isEnabled())
        self.assertEqual(console.focusPolicy(), Qt.FocusPolicy.NoFocus)

    def test_the_console_link_does_not_make_the_card_wider(self):
        _button, with_console = self.card_with_console(lambda: None)
        without_console = self.open_card()
        self.assertIsNone(without_console.console_button)
        self.assertEqual(with_console.natural_size().width(), without_console.natural_size().width())

    def test_a_click_on_the_console_link_opens_the_console_and_leaves_the_card_and_the_keyboard_alone(self):
        opened = []
        button, card = self.card_with_console(lambda: opened.append("console"))
        self.host.activateWindow()
        self.host.setFocus()
        flush(self.app)
        focus = QApplication.focusWidget()
        QTest.mouseClick(card.console_button, Qt.MouseButton.LeftButton)
        self.assertEqual(opened, ["console"])
        self.assertTrue(button.card_is_open())
        self.assertIs(QApplication.focusWidget(), focus)
        QTest.mouseClick(card.console_button, Qt.MouseButton.LeftButton)
        self.assertEqual(opened, ["console", "console"])

    def test_a_card_given_no_console_function_has_no_console_link(self):
        card = self.open_card()
        self.assertIsNone(card.console_button)
        self.assertEqual([link.text() for link in card.findChildren(QPushButton) if link.objectName() == "controlsLink"],
                         ["List of commands"])
        card.open_console()  # nothing to open, and nothing breaks

    def assert_halves(self, card):
        return assert_two_halves(self, card)

    def test_the_keys_and_the_descriptions_are_two_equal_halves_of_the_card(self):
        card = self.open_card()
        grid, middle = self.assert_halves(card)
        # The description column starts at the card's middle: 1/2 of the content width, not after the widest key.
        widest = max(holder.sizeHint().width() for holder in card.key_holders)
        self.assertGreater(middle - grid.x(), widest + ControlsCard.COLUMN_GAP + 8)
        self.assertEqual(card.key_holders[0].geometry().x(), grid.x())
        # Every row's description starts at the same x.
        self.assertEqual({label.geometry().x() for label in card.descriptions}, {card.descriptions[0].geometry().x()})

    def test_a_longer_text_widens_the_card_and_the_halves_stay_equal(self):
        # A description much longer than the others does not wrap or shrink the keys: both halves grow to it.
        text = shipped_actions() + " | [Ctrl + LeftClick] Copy every header the selection holds"
        cards = []
        # The third has keys wider than any description: its first half is the keys and the gap after them.
        wide_keys = shipped_actions() + " | [Ctrl + Shift + LeftClick] Go"
        for source in (shipped_actions(), text, wide_keys):
            card = ControlsCard("x", source, parent=self.host, controls_title="Mouse and keyboard controls",
                                area=lambda: self.host.rect())
            self.addCleanup(card.close)
            card.open()
            flush(self.app)
            cards.append(card)
        plain, long, keys = cards
        self.assertGreater(long.natural_size().width(), plain.natural_size().width())
        self.assertGreater(keys.natural_size().width(), plain.natural_size().width())
        for card in cards:
            self.assert_halves(card)
            self.assertEqual(card.scroll.horizontalScrollBar().maximum(), 0)

    def test_a_file_that_is_not_loaded_shows_a_muted_none_and_keeps_its_pick_button(self):
        self.info["inputs"].update(msa=None, embedding="")
        card = self.open_card()
        for key in ("msa", "embedding"):
            field = card.path_fields[key]
            self.assertEqual((field.text(), field.placeholderText(), field.toolTip()), ("", "None", ""), key)
            # Choosing one is what the button is for: it is there when none is loaded.
            self.assertTrue(card.pick_buttons[key].isEnabled(), key)
        # A fixed file with no path has no tooltip either, not a note alone.
        self.info["inputs"]["layout_cache"] = None
        card.refresh()
        self.assertEqual((card.path_fields["layout_cache"].text(), card.path_fields["layout_cache"].toolTip()), ("", ""))

    def row_tops(self, card):
        return {key: card.path_fields[key].mapTo(card, QPoint(0, 0)).y() for key, _label in SESSION_FILES}

    def test_the_fixed_files_come_first_then_a_gap_then_the_two_that_can_be_changed(self):
        card = self.open_card()
        tops = self.row_tops(card)
        keys = [key for key, _label in SESSION_FILES]
        self.assertEqual(sorted(keys, key=tops.get), keys)
        height = card.path_fields["node_fasta"].height()
        steps = [tops[b] - tops[a] - height for a, b in zip(keys, keys[1:])]
        # The same space between the rows of a group, and more where the groups meet.
        self.assertEqual(steps[0], steps[1])
        self.assertEqual(steps[3], steps[0])
        self.assertGreaterEqual(steps[2], steps[0] + ControlsCard.GROUP_GAP - 1)
        self.assertLessEqual(steps[2], steps[0] + ControlsCard.GROUP_GAP + 1)
        # The labels follow their fields.
        for key, _label in SESSION_FILES:
            label = card.file_labels[key]
            self.assertAlmostEqual(label.mapTo(card, QPoint(0, 0)).y() + label.height() / 2,
                                   tops[key] + height / 2, delta=2, msg=key)

    def test_the_fixed_files_have_no_button_and_a_quieter_field_that_still_selects_and_copies(self):
        card = self.open_card()
        fixed, changeable = ("node_fasta", "network_hdf5", "layout_cache"), ("msa", "embedding")
        self.assertEqual(set(card.pick_buttons), set(changeable))
        # No button anywhere in the session but the two pick buttons: no folder button is left.
        self.assertEqual({button for button in card.body.findChildren(QPushButton)}, set(card.pick_buttons.values()))
        for key in fixed:
            field = card.path_fields[key]
            self.assertTrue(field.fixed and field.property("fixed") is True, key)
            self.assertTrue(field.isReadOnly(), key)
            self.assertEqual(field.focusPolicy(), Qt.FocusPolicy.NoFocus)
            # The note after the path says why there is no button.
            self.assertEqual(field.toolTip(), self.tip(key, os.path.normpath(self.info["inputs"][key])))
            self.assertEqual(field.toolTip().splitlines()[1], "Fixed for this Viewer session.")
        for key in changeable:
            self.assertFalse(card.path_fields[key].fixed, key)
            self.assertEqual(card.path_fields[key].toolTip(), os.path.normpath(self.info["inputs"][key]))
        # Quieter: no field fill of its own, a lighter ink, and a lighter border than the fields that can change.
        tint = QColor(TOKENS["tint"])
        for key in fixed + changeable:
            image = QImage(card.path_fields[key].size(), QImage.Format.Format_ARGB32)
            image.fill(tint)
            card.path_fields[key].render(image)
            inside = image.pixelColor(card.path_fields[key].width() // 2, card.path_fields[key].height() - 5)
            if key in fixed:
                self.assertEqual(inside.name(), TOKENS["tint"], key)
            else:
                self.assertEqual(inside.name(), TOKENS["field"], key)
        self.assertNotEqual(TOKENS["field"], TOKENS["tint"])
        self.assertGreater(self.darkest_ink(card.path_fields["node_fasta"]),
                           self.darkest_ink(card.path_fields["msa"]) + 20)
        # They are the same width as the fields beside a button, plus the button's room.
        grid = card.controls_grid.geometry()
        for key in fixed:
            field = card.path_fields[key]
            self.assertEqual(field.mapTo(card.body, QPoint(0, 0)).x() + field.width() - 1, grid.right(), key)
        for key in changeable:
            self.assertLess(card.path_fields[key].mapTo(card.body, QPoint(0, 0)).x() + card.path_fields[key].width(),
                            card.pick_buttons[key].mapTo(card.body, QPoint(0, 0)).x())

    def test_the_two_files_that_can_be_changed_have_a_pick_button_in_place_of_the_folder_button(self):
        card = self.open_card()
        expected = {"msa": "Choose another alignment file…", "embedding": "Choose another embedding file…"}
        for key, text in expected.items():
            button, field = card.pick_buttons[key], card.path_fields[key]
            self.assertEqual((button.toolTip(), button.accessibleName()), (text, text))
            self.assertTrue(button.isEnabled())
            self.assertEqual(button.width(), ControlsCard.PICK_WIDTH)
            self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.NoFocus)
            # In the row of its field, after it.
            self.assertAlmostEqual(button.mapTo(card, QPoint(0, 0)).y() + button.height() / 2,
                                   field.mapTo(card, QPoint(0, 0)).y() + field.height() / 2, delta=2)
            self.assertGreater(button.mapTo(card, QPoint(0, 0)).x(), field.mapTo(card, QPoint(0, 0)).x() + field.width() - 1)
            # A "choose a file" icon of the bundled set, not the folder's.
            icon = button.icon().pixmap(16, 16).toImage()
            self.assertEqual(icon, studio_icon("file-input").pixmap(16, 16).toImage())
            self.assertNotEqual(icon, studio_icon("folder-open").pixmap(16, 16).toImage())
        self.assertEqual({PICK_FILES[key][0] for key in expected}, {"alignment", "embedding"})
        self.assertNotIn("Open Folder", [widget.toolTip() for widget in card.findChildren(QWidget)])

    def test_a_pick_button_types_its_command_with_no_argument_and_leaves_the_keyboard_with_the_canvas(self):
        card = self.open_card()
        self.host.activateWindow()
        self.host.setFocus()
        flush(self.app)
        has_keyboard = self.host.hasFocus()
        for key, command in (("msa", "alignment"), ("embedding", "embedding")):
            self.command.reset_mock()
            QTest.mouseClick(card.pick_buttons[key], Qt.MouseButton.LeftButton)
            # Exactly the command, nothing after it, as when it is typed alone; once.
            self.command.assert_called_once_with(command)
            self.assertEqual(self.host.seen, [], key)
            self.assertTrue(self.button.card_is_open())
            if has_keyboard:
                self.assertIs(QApplication.focusWidget(), self.host, key)
        # Whatever the platform does with focus, the card gives the keyboard to the canvas once the command has run.
        self.command.reset_mock()
        order = []
        self.command.side_effect = lambda text: order.append("command")
        with mock.patch.object(self.host, "setFocus", side_effect=lambda: order.append("focus")) as set_focus:
            QTest.mouseClick(card.pick_buttons["msa"], Qt.MouseButton.LeftButton)
        set_focus.assert_called_once_with()
        self.assertEqual(order, ["command", "focus"])
        self.command.side_effect = None
        # Only the two files that can be changed have such a command; others, and a card with none, run nothing.
        self.command.reset_mock()
        self.assertFalse(card.pick_file("node_fasta"))
        self.assertFalse(card.pick_file("layout_cache"))
        self.command.assert_not_called()
        self.assertTrue(card.pick_file("msa"))
        self.command.assert_called_once_with("alignment")
        bare = ControlsCard("x", shipped_actions(), parent=self.host, session=self.read_session)
        self.addCleanup(bare.close)
        QTest.mouseClick(bare.pick_buttons["embedding"], Qt.MouseButton.LeftButton)
        self.assertFalse(bare.pick_file("embedding"))
        # The click on the bare card's button ran nothing, and neither reached this card's command.
        self.assertEqual(self.command.call_count, 1)

    def test_the_button_hands_its_command_to_the_card(self):
        # The command ControlsButton was given is the one the card it opens calls, nothing wrapped around it.
        card = self.open_card()
        self.assertIs(card.command, self.command)
        self.assertIs(self.button.command, self.command)

    def test_the_path_fields_are_read_only_selectable_and_never_take_the_keyboard(self):
        card = self.open_card()
        field = card.path_fields["node_fasta"]
        self.assertTrue(field.isReadOnly())
        self.assertEqual(field.focusPolicy(), Qt.FocusPolicy.NoFocus)
        # The mouse selects, though the field has no keyboard focus.
        QTest.mousePress(field, Qt.MouseButton.LeftButton, pos=QPoint(10, field.height() // 2))
        QTest.mouseMove(field, QPoint(field.width() - 10, field.height() // 2))
        QTest.mouseRelease(field, Qt.MouseButton.LeftButton, pos=QPoint(field.width() - 10, field.height() // 2))
        self.assertTrue(field.hasSelectedText())
        # The right-click menu copies it, as in any field.
        QGuiApplication.clipboard().clear()
        send_context_menu(field)
        menu = QApplication.activePopupWidget()
        self.addCleanup(lambda: menu.close() if menu is not None else None)
        self.assertIsNotNone(menu)
        actions = {action.text().replace("&", "").split("\t")[0]: action for action in menu.actions()
                   if action.text()}
        copy = next(action for name, action in actions.items() if name.startswith("Copy"))
        self.assertTrue(copy.isEnabled())
        copy.trigger()
        self.assertEqual(QGuiApplication.clipboard().text(), field.selectedText())
        menu.close()
        # No shortcut is taken from the canvas, whose command line copies its own text with Copy: the card
        # answers the Copy key through the key presses the canvas receives (the tests below).
        self.assertEqual(card.findChildren(QShortcut), [])

    def test_the_copy_keys_copy_the_selection_of_whichever_field_has_it(self):
        card = self.open_card()
        clipboard = QGuiApplication.clipboard()
        copy_chords = chords(QKeySequence.StandardKey.Copy)
        self.assertIn((Qt.Key.Key_C, CTRL), copy_chords)
        command_line = mock.Mock()
        self.host.on_copy = command_line
        for key, _label in SESSION_FILES:
            field = card.path_fields[key]
            for chord in copy_chords:
                with self.subTest(field=key, chord=chord_name(chord)):
                    self.select_in(field)
                    clipboard.setText("before")
                    seen = self.press(*chord)
                    # The selected text, as the right-click menu copies it.
                    self.assertEqual(clipboard.text(), field.selectedText())
                    self.assertNotEqual(clipboard.text(), "before")
                    # The card took the Copy key: it never reached the canvas, so nothing there acted on it.
                    self.assertNotIn(chord[0], seen)
                    command_line.assert_not_called()
                    # The selection stays, as after any copy.
                    self.assertTrue(field.hasSelectedText())

    def test_every_other_key_and_a_copy_with_nothing_selected_go_on_to_the_canvas(self):
        card = self.open_card()
        clipboard = QGuiApplication.clipboard()
        field = card.path_fields["node_fasta"]
        shift = Qt.KeyboardModifier.ShiftModifier
        taken = set(chords(QKeySequence.StandardKey.Copy) + chords(QKeySequence.StandardKey.SelectAll))
        others = [chord for chord in (
            (Qt.Key.Key_Z, CTRL), (Qt.Key.Key_Y, CTRL),                   # undo and redo
            (Qt.Key.Key_Return, NO_MODIFIER), (Qt.Key.Key_Enter, NO_MODIFIER), (Qt.Key.Key_Escape, NO_MODIFIER),
            (Qt.Key.Key_Backspace, NO_MODIFIER), (Qt.Key.Key_Left, NO_MODIFIER), (Qt.Key.Key_Up, NO_MODIFIER),
            (Qt.Key.Key_C, CTRL | shift), (Qt.Key.Key_C, NO_MODIFIER),
            (Qt.Key.Key_A, NO_MODIFIER), (Qt.Key.Key_A, CTRL | shift),
            (Qt.Key.Key_X, CTRL), (Qt.Key.Key_V, CTRL), (Qt.Key.Key_Insert, shift),
        ) if chord not in taken]
        self.assertGreaterEqual(len(others), 12)
        self.select_in(field)
        selected = field.selectedText()
        clipboard.setText("before")
        for key, modifiers in others:
            with self.subTest(key=chord_name((key, modifiers))):
                self.assertIn(key, self.press(key, modifiers))
                self.assertEqual(clipboard.text(), "before")
                self.assertEqual(field.selectedText(), selected)
        # A Copy with nothing selected is the canvas's, as it was before the card could copy: its command line copies.
        field.deselect()
        self.assertIsNone(card.selected_field())
        command_line = mock.Mock()
        self.host.on_copy = command_line
        copy_chords = chords(QKeySequence.StandardKey.Copy)
        for chord in copy_chords:
            self.assertIn(chord[0], self.press(*chord))
        self.assertEqual(command_line.call_count, len(copy_chords))
        self.assertEqual(clipboard.text(), "before")

    def test_while_the_command_line_is_open_copy_and_select_all_stay_the_canvas(self):
        card = self.open_card()
        clipboard = QGuiApplication.clipboard()
        command_line = mock.Mock(side_effect=lambda: clipboard.setText("the command"))
        self.host.on_copy = command_line
        field = card.path_fields["msa"]
        self.select_in(field)
        selected = field.selectedText()
        self.command_line_open = True
        clipboard.setText("before")
        # Copy is the command line's own: it copies the command, and the field's selection is left as it is.
        self.assertIn(Qt.Key.Key_C, self.press(Qt.Key.Key_C, CTRL))
        command_line.assert_called_once_with()
        self.assertEqual(clipboard.text(), "the command")
        self.assertEqual(field.selectedText(), selected)
        # Select All is not taken either, even in a field just clicked in.
        QTest.mouseClick(card.path_fields["network_hdf5"], Qt.MouseButton.LeftButton)
        for chord in chords(QKeySequence.StandardKey.SelectAll):
            self.assertIn(chord[0], self.press(*chord))
        self.assertFalse(card.path_fields["network_hdf5"].hasSelectedText())
        # Once the command line is closed, Copy is the card's again.
        self.command_line_open = False
        command_line.reset_mock()
        clipboard.setText("before")
        self.assertNotIn(Qt.Key.Key_C, self.press(Qt.Key.Key_C, CTRL))
        command_line.assert_not_called()
        self.assertEqual(clipboard.text(), selected)

    def test_select_all_selects_the_field_last_clicked_in_and_copy_then_copies_it(self):
        card = self.open_card()
        clipboard = QGuiApplication.clipboard()
        first, second = card.path_fields["network_hdf5"], card.path_fields["msa"]
        select_all_chords = chords(QKeySequence.StandardKey.SelectAll)
        self.assertIn((Qt.Key.Key_A, CTRL), select_all_chords)
        # No field clicked in yet: Select All has nowhere to go, so it is the canvas's.
        for chord in select_all_chords:
            self.assertIn(chord[0], self.press(*chord))
        self.assertIsNone(card.selected_field())
        for chord in select_all_chords:
            for field, other in ((first, second), (second, first)):
                with self.subTest(chord=chord_name(chord), field=field.text()):
                    self.select_in(other)
                    # A plain click makes it the field Select All works on, and the other's selection stays.
                    QTest.mouseClick(field, Qt.MouseButton.LeftButton)
                    self.assertNotIn(chord[0], self.press(*chord))
                    self.assertEqual(field.selectedText(), field.text())
                    self.assertGreater(len(field.text()), 0)
                    # Only one field shows a selection: the other's went when this one took it.
                    self.assertFalse(other.hasSelectedText())
                    # Copy then takes all of it.
                    clipboard.setText("before")
                    self.press(Qt.Key.Key_C, CTRL)
                    self.assertEqual(clipboard.text(), field.text())
                    field.deselect()
        # A field with no text, the embedding not loaded, has nothing to select: the key goes on.
        self.info["inputs"]["embedding"] = None
        card.refresh()
        empty = card.path_fields["embedding"]
        self.assertEqual(empty.text(), "")
        QTest.mouseClick(empty, Qt.MouseButton.LeftButton)
        for chord in select_all_chords:
            self.assertIn(chord[0], self.press(*chord))
        self.assertIsNone(card.selected_field())

    def test_only_one_field_shows_a_selection_and_copy_copies_that_one(self):
        card = self.open_card()
        clipboard = QGuiApplication.clipboard()
        first, second = card.path_fields["node_fasta"], card.path_fields["embedding"]
        self.select_in(first)
        self.select_in(second)
        self.assertFalse(first.hasSelectedText())
        self.assertIs(card.selected_field(), second)
        self.press(Qt.Key.Key_C, CTRL)
        self.assertEqual(clipboard.text(), second.selectedText())
        self.select_in(first)
        self.assertFalse(second.hasSelectedText())
        self.assertIs(card.selected_field(), first)
        self.press(Qt.Key.Key_C, CTRL)
        self.assertEqual(clipboard.text(), first.selectedText())
        # A path that changes loses its selection, as a text that is replaced does, so nothing stale is copied.
        self.info["inputs"]["node_fasta"] = str(self.files["node_fasta"].with_name("other.fasta"))
        card.refresh()
        self.assertFalse(first.hasSelectedText())
        clipboard.setText("before")
        self.assertIn(Qt.Key.Key_C, self.press(Qt.Key.Key_C, CTRL))
        self.assertEqual(clipboard.text(), "before")

    def test_a_card_that_is_closed_or_hidden_takes_no_key(self):
        card = self.open_card()
        clipboard = QGuiApplication.clipboard()
        self.select_in(card.path_fields["node_fasta"])
        card.hide()
        clipboard.setText("before")
        self.assertIn(Qt.Key.Key_C, self.press(Qt.Key.Key_C, CTRL))
        self.assertEqual(clipboard.text(), "before")
        card.show()
        self.assertNotIn(Qt.Key.Key_C, self.press(Qt.Key.Key_C, CTRL))
        self.assertNotEqual(clipboard.text(), "before")
        # Closed by its cross: the card and its filter are gone, and the canvas has every key.
        QTest.mouseClick(card.close_button, Qt.MouseButton.LeftButton)
        flush(self.app)
        self.assertFalse(self.button.card_is_open())
        clipboard.setText("before")
        for chord in chords(QKeySequence.StandardKey.Copy) + chords(QKeySequence.StandardKey.SelectAll):
            self.assertIn(chord[0], self.press(*chord))
        self.assertEqual(clipboard.text(), "before")

    def test_the_button_hands_its_command_line_state_to_the_card(self):
        card = self.open_card()
        self.assertIs(card.command_line_open, self.button.command_line_open)
        self.assertFalse(card.command_line_open())
        self.command_line_open = True
        self.assertTrue(card.command_line_open())

    def test_a_path_too_long_for_its_field_starts_at_its_start_and_scrolls_with_the_wheel(self):
        long_name = "x" * 120 + ".fasta"
        full = self.files["node_fasta"].with_name(long_name)
        self.info["inputs"]["node_fasta"] = str(full)
        card = self.open_card()
        field = card.path_fields["node_fasta"]
        self.assertEqual(field.text(), "$input_file$/Sequence_Sets/" + long_name)
        self.assertEqual(field.toolTip(), self.tip("node_fasta", str(full)))
        # It starts at the start of what it shows, as the Config's fields do, not at the file's name.
        end = len(field.text())
        self.assertEqual((field.cursorPosition(), field._scroll), (0, 0))
        for _ in range(10):
            send_wheel(field, -120)
        self.assertEqual(field._scroll, 10 * PathField.WHEEL_STEP)
        for _ in range(60):
            send_wheel(field, -120)
        self.assertEqual(field._scroll, end)
        for _ in range(60):
            send_wheel(field, 120)
        self.assertEqual(field._scroll, 0)
        self.assertEqual(self.host.seen, [])

    def test_nothing_in_the_card_takes_the_keyboard(self):
        card = self.open_card()
        self.assertEqual(card.focusPolicy(), Qt.FocusPolicy.NoFocus)
        for child in card.findChildren(QWidget):
            self.assertEqual(child.focusPolicy(), Qt.FocusPolicy.NoFocus, child.objectName() or type(child).__name__)
        # A click on the card's fields, buttons and text leaves the keyboard with the canvas.
        self.host.activateWindow()
        self.host.setFocus()
        flush(self.app)
        if not self.host.hasFocus():
            self.skipTest("this platform gives no window the keyboard")
        for widget in (card.path_fields["network_hdf5"], card.pick_buttons["msa"], card.session_heading,
                       card.key_caps[0], card.commands_button, card):
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
            self.assertIs(QApplication.focusWidget(), self.host, type(widget).__name__)
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Return)
        self.assertIn("key", self.host.seen)

    def test_clicks_drags_wheel_and_right_clicks_on_the_canvas_leave_the_card_open(self):
        card = self.open_card()
        away = QPoint(600, 100)
        QTest.mouseClick(self.host, Qt.MouseButton.LeftButton, pos=away)
        QTest.mouseClick(self.host, Qt.MouseButton.RightButton, pos=away)
        QTest.mousePress(self.host, Qt.MouseButton.LeftButton, pos=away)
        QTest.mouseMove(self.host, QPoint(620, 120))
        QTest.mouseRelease(self.host, Qt.MouseButton.LeftButton, pos=QPoint(620, 120))
        send_wheel(self.host, 120, QPointF(away))
        QTest.keyClick(self.host, Qt.Key.Key_Escape)
        flush(self.app)
        # Every one reached the canvas, and the card is still there.
        self.assertEqual(self.host.seen.count("press"), 3)
        self.assertIn("wheel", self.host.seen)
        self.assertTrue(self.button.card_is_open())
        self.assertTrue(card.isVisible())

    def test_events_on_the_card_do_not_reach_the_canvas(self):
        card = self.open_card()
        inside = card.card_rect().center().toPoint()
        targets = [
            (card, inside), (card.session_heading, None), (card.controls_heading, None),
            (card.key_caps[0], None), (card.descriptions[0], None),
            (card.count_labels["nodes"], None), (card.file_labels["network_hdf5"], None),
            (card.path_fields["network_hdf5"], None), (card.scroll.viewport(), None),
            (card.commands_button, None), (card.pick_buttons["msa"], None), (card.pick_buttons["embedding"], None),
            (card.body, QPoint(2, 2)),
        ]
        for widget, position in targets:
            position = widget.rect().center() if position is None else position
            with self.subTest(widget=widget.objectName() or type(widget).__name__):
                QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=position)
                QTest.mouseMove(widget, position + QPoint(3, 0))
                QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=position + QPoint(3, 0))
                QTest.mouseDClick(widget, Qt.MouseButton.LeftButton, pos=position)
                QTest.mouseClick(widget, Qt.MouseButton.RightButton, pos=position)
                send_wheel(widget, -120, QPointF(position))
                # A move with no button down, which the canvas turns into a hover.
                QTest.mouseMove(widget, position)
                if not isinstance(widget, PathField):  # its menu is a popup of its own
                    send_context_menu(widget)
                self.assertEqual(self.host.seen, [], widget.objectName())
        self.assertTrue(self.button.card_is_open())

    def test_the_shadow_around_the_card_is_not_the_card(self):
        card = self.open_card()
        corner = QPoint(2, 2)
        self.assertFalse(card.card_rect().contains(QPointF(corner)))
        QTest.mousePress(card, Qt.MouseButton.LeftButton, pos=corner)
        QTest.mouseRelease(card, Qt.MouseButton.LeftButton, pos=corner)
        # A click there is a click on the canvas beneath.
        self.assertEqual(self.host.seen, ["press", "release"])

    def test_the_cross_and_the_button_close_the_card_and_nothing_else_does(self):
        card = self.open_card()
        QTest.keyClick(card, Qt.Key.Key_Escape)
        QTest.mouseClick(self.host, Qt.MouseButton.LeftButton, pos=QPoint(600, 100))
        flush(self.app)
        self.assertTrue(self.button.card_is_open())
        # The card does not keep the Esc key: it goes on to the canvas, as every key does.
        self.assertEqual(self.host.seen[0], "key")
        QTest.mouseClick(card.close_button, Qt.MouseButton.LeftButton)
        flush(self.app)
        self.assertFalse(self.button.card_is_open())
        # The button opens it, and a second click on the button closes it.
        QTest.mouseClick(self.button, Qt.MouseButton.LeftButton)
        flush(self.app)
        self.assertTrue(self.button.card_is_open())
        QTest.mouseClick(self.button, Qt.MouseButton.LeftButton)
        flush(self.app)
        self.assertFalse(self.button.card_is_open())
        # Only the Esc key and the click away from the card reached the canvas.
        self.assertEqual(self.host.seen, ["key", "press", "release"])

    def test_the_card_is_a_child_of_the_canvas_above_the_button_on_a_tint_of_the_theme_colour(self):
        card = self.open_card()
        self.assertIs(card.parentWidget(), self.host)
        self.assertFalse(card.isWindow())
        rect = self.card_rect_in_host(card)
        button = self.button.mapTo(self.host, QPoint(0, 0))
        # Well clear of the button and the slider beside it: 6 px read as touching them.
        gap = button.y() - rect.bottom()
        self.assertAlmostEqual(gap, ControlsCard.ANCHOR_GAP, delta=1)
        self.assertGreaterEqual(gap, 12)
        self.assertAlmostEqual(rect.left(), button.x(), delta=1)
        image = QImage(card.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            # Only what the card paints: the canvas's background stays out of its corners.
            card.render(painter, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        finally:
            painter.end()
        # In the card's own padding, away from the text: its fill.
        probe = card.card_rect().topLeft().toPoint() + QPoint(5, card.height() // 2)
        self.assertEqual(image.pixelColor(probe).name(), TOKENS["tint"])
        # The corner is round, so the canvas shows through there.
        corner = card.card_rect().topLeft().toPoint() + QPoint(1, 1)
        self.assertLess(image.pixelColor(corner).alpha(), 128)

    def test_the_card_stays_in_its_room_and_a_short_window_scrolls_it(self):
        card = self.open_card()
        natural = card.natural_size()
        self.assertEqual(card.size(), natural)
        self.assertEqual(card.scroll.verticalScrollBar().maximum(), 0)
        # Room above the button for only part of it: the card keeps to the room and scrolls.
        self.area = QRect(8, 330, 400, 340)
        self.button.place_card()
        flush(self.app)
        rect = self.card_rect_in_host(card)
        room = self.area
        self.assertGreaterEqual(rect.left(), room.left())
        self.assertGreaterEqual(rect.top(), room.top())
        self.assertLessEqual(rect.right(), room.left() + room.width())
        button_top = self.button.mapTo(self.host, QPoint(0, 0)).y()
        self.assertAlmostEqual(button_top - rect.bottom(), ControlsCard.ANCHOR_GAP, delta=1)
        self.assertLess(card.height(), natural.height())
        self.assertGreater(card.scroll.verticalScrollBar().maximum(), 0)
        # A narrow room narrows it: the long paths scroll inside their fields instead.
        self.area = QRect(8, 8, 330, 440)
        self.button.place_card()
        flush(self.app)
        rect = self.card_rect_in_host(card)
        self.assertLessEqual(rect.right(), 8 + 330)
        self.assertEqual(card.path_fields["node_fasta"].text(), self.aliased["node_fasta"])
        # Nothing is cut short in it: what does not fit scrolls sideways, and the halves are still equal.
        self.assertGreater(card.scroll.horizontalScrollBar().maximum(), 0)
        self.assertGreaterEqual(card.body.width(), card.body.sizeHint().width())
        self.assert_halves(card)
        # A button at the right edge moves the card back in.
        self.area = None
        self.button.move(self.host.width() - 40, 700)
        self.button.place_card()
        rect = self.card_rect_in_host(card)
        self.assertLessEqual(rect.right(), self.host.width() - 8)

    def test_the_card_follows_its_button(self):
        card = self.open_card()
        first = card.pos()
        self.button.move(20, 650)
        self.button.place_card()
        self.assertEqual(card.pos().y(), first.y() - 50)
        self.assertEqual(card.pos().x(), first.x())
        self.button.move(60, 650)
        self.button.place_card()
        self.assertEqual(card.pos().x(), first.x() + 40)

    def test_refresh_follows_the_session_and_only_when_it_is_open(self):
        self.button.refresh_card()
        self.assertEqual(self.session_calls, 0)  # closed: not even read
        card = self.open_card()
        calls = self.session_calls
        self.info.update(node_count=5, visible_node_count=3, edge_count=10, shown_edge_count=0)
        # Another embedding, in the embedding folder, replaces the first: its row follows.
        other = self.root / "Embeddings" / "other.h5"
        self.info["inputs"] = dict(self.info["inputs"], msa=None, embedding=str(other))
        self.button.refresh_card()
        self.assertEqual(self.session_calls, calls + 1)
        self.assertEqual((card.count_labels["nodes"].text(), card.count_labels["edges"].text()),
                         ("5 (3 visible)", "10 (0 shown)"))
        self.assertEqual(card.path_fields["msa"].text(), "")
        self.assertTrue(card.pick_buttons["msa"].isEnabled())
        self.assertEqual(card.path_fields["embedding"].text(), "$embedding$/other.h5")
        self.assertEqual(card.path_fields["embedding"].toolTip(), str(other))

    def test_a_refresh_that_changes_nothing_touches_nothing(self):
        card = self.open_card()
        with mock.patch.object(PathField, "show_path") as show_path, \
                mock.patch("desktop.Viewer_Widgets.alias_form") as alias, \
                mock.patch.object(card.count_labels["nodes"], "setText") as set_text:
            for _ in range(20):
                self.button.refresh_card()
            show_path.assert_not_called()
            alias.assert_not_called()
            set_text.assert_not_called()
            # The counts changing sets the counts only.
            self.info.update(shown_edge_count=1)
            self.button.refresh_card()
            show_path.assert_not_called()
            alias.assert_not_called()

    def test_a_refresh_writes_only_the_paths_that_changed_unless_the_folders_did(self):
        card = self.open_card()
        with mock.patch("desktop.Viewer_Widgets.alias_form", wraps=alias_form) as alias:
            other = self.root / "Embeddings" / "other.h5"
            self.info["inputs"] = dict(self.info["inputs"], embedding=str(other))
            self.button.refresh_card()
            self.assertEqual([call.args[0] for call in alias.call_args_list], [str(other)])
            self.assertEqual(card.path_fields["embedding"].text(), "$embedding$/other.h5")
            # A base folder that moves rewrites every row.
            alias.reset_mock()
            self.info["directories"] = dict(self.info["directories"], INPUT_FILE_DIR=str(self.root))
            self.button.refresh_card()
            self.assertEqual(alias.call_count, len(SESSION_FILES))
            self.assertEqual(card.path_fields["node_fasta"].text(), "$input_file$/Input_Files/Sequence_Sets/set.fasta")
            self.assertEqual(card.path_fields["embedding"].text(), "$embedding$/other.h5")

    def test_the_command_list_link_opens_the_page_in_the_viewers_language(self):
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "list_of_commands.html").write_text("<html lang='en'>")
        # A translation is in its language's folder, <language>/docs/, with the language's other files.
        languages = self.root / "languages"
        translated = languages / "zh_CN" / "docs"
        translated.mkdir(parents=True)
        (translated / "list_of_commands.html").write_text("<html lang='zh'>")
        with mock.patch.object(Help_Pages, "DOCS_DIR", docs), \
                mock.patch.object(Help_Pages, "TRANSLATIONS_DIR", languages), \
                mock.patch("desktop.Viewer_Widgets.installed_language", return_value=None) as language, \
                mock.patch("desktop.Viewer_Widgets.QDesktopServices") as services:
            card = self.open_card()
            self.host.activateWindow()
            self.host.setFocus()
            flush(self.app)
            focus = QApplication.focusWidget()
            self.assertTrue(card.commands_button.isEnabled())
            self.assertEqual(card.commands_button.text(), "List of commands")
            opened = []
            for code, expected in ((None, docs), ("zh_CN", translated), ("de", docs)):
                language.return_value = code
                services.openUrl.reset_mock()
                QTest.mouseClick(card.commands_button, Qt.MouseButton.LeftButton)
                services.openUrl.assert_called_once()
                opened.append(Path(services.openUrl.call_args.args[0].toLocalFile()))
                self.assertEqual(opened[-1], expected / "list_of_commands.html")
            # The card stays open and the keyboard where it was.
            self.assertTrue(self.button.card_is_open())
            self.assertIs(QApplication.focusWidget(), focus)

    def test_the_command_list_link_opens_the_pages_that_ship(self):
        project = Path(__file__).resolve().parents[1]
        shipped = {
            None: project / "docs" / "list_of_commands.html",
            "en": project / "docs" / "list_of_commands.html",
            "de": project / "docs" / "list_of_commands.html",
            "zh_CN": project / "src" / "resources" / "languages" / "zh_CN" / "docs" / "list_of_commands.html",
        }
        with mock.patch("desktop.Viewer_Widgets.installed_language", return_value=None) as language, \
                mock.patch("desktop.Viewer_Widgets.QDesktopServices") as services:
            card = self.open_card()
            self.assertTrue(card.commands_button.isEnabled())
            for code, expected in shipped.items():
                language.return_value = code
                services.openUrl.reset_mock()
                self.assertTrue(card.open_command_list())
                self.assertEqual(Path(services.openUrl.call_args.args[0].toLocalFile()), expected)
                self.assertTrue(expected.is_file())

    def test_without_the_command_list_the_link_is_grey_and_says_why(self):
        empty = self.root / "empty"
        empty.mkdir()
        with mock.patch.object(Help_Pages, "DOCS_DIR", empty), \
                mock.patch("desktop.Viewer_Widgets.QDesktopServices") as services:
            card = self.open_card()
            self.assertFalse(card.commands_button.isEnabled())
            self.assertIn("docs/list_of_commands.html", card.commands_button.toolTip())
            self.assertFalse(card.open_command_list())
            services.openUrl.assert_not_called()


class ControlsCardLanguageTests(unittest.TestCase):
    """The info card in Simplified Chinese: the catalog's drafts reach every new text."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def test_the_chinese_catalog_translates_the_session_and_its_link(self):
        translator = QTranslator()
        self.assertTrue(translator.load(str(ZH_CATALOG.with_suffix(".qm"))))
        self.assertTrue(self.app.installTranslator(translator))
        self.addCleanup(self.app.removeTranslator, translator)
        host = QWidget()
        self.addCleanup(host.deleteLater)
        host.resize(700, 500)
        info = {"node_count": 1234, "visible_node_count": 1200, "edge_count": 5000, "shown_edge_count": 40,
                "inputs": {"node_fasta": os.path.abspath("set.fasta"), "msa": None}}
        title = QCoreApplication.translate("Viewer", "Session and controls")
        button = ControlsButton(title, shipped_actions(), host, host=host, session=lambda: info,
                                controls_title=QCoreApplication.translate("Viewer", "Mouse and keyboard controls"))
        host.show()
        card = button.toggle_card()
        flush(self.app)
        self.assertEqual(title, "会话与操作")
        self.assertEqual(card.count_labels["nodes"].text(), "1,234（可见 1,200）")
        self.assertEqual(card.count_labels["edges"].text(), "5,000（显示 40）")
        self.assertEqual([card.file_labels[key].text() for key, _label in SESSION_FILES],
                         ["序列集：", "网络：", "布局缓存：", "比对：", "Embedding："])
        self.assertEqual(card.path_fields["msa"].placeholderText(), "无")
        # The pick buttons, and the note after a fixed file's path, are the catalog's.
        self.assertEqual(card.pick_buttons["msa"].toolTip(), "选择其他比对文件…")
        self.assertEqual(card.pick_buttons["embedding"].toolTip(), "选择其他 embedding 文件…")
        self.assertEqual(card.path_fields["node_fasta"].toolTip().splitlines(),
                         [os.path.abspath("set.fasta"), "在本次查看器会话中固定不变。"])
        self.assertEqual(card.close_button.toolTip(), "关闭")
        self.assertEqual(card.commands_button.text(), "命令列表")
        self.assertEqual(card.commands_button.toolTip(), "在网页浏览器中打开命令列表")
        # The two headings are the catalog's, and the title shows nowhere.
        self.assertEqual((card.session_heading.text(), card.controls_heading.text()), ("会话", "鼠标和键盘操作"))
        self.assertEqual(card.accessibleName(), "会话与操作")
        # The nodes and the edges on one line, then the controls in two equal halves, with no text cut short.
        self.assertEqual(card.count_labels["nodes"].mapTo(card, QPoint(0, 0)).y(),
                         card.count_labels["edges"].mapTo(card, QPoint(0, 0)).y())
        assert_two_halves(self, card)
        # Every text of the session comes from the catalog: none is left as the English source.
        for label in (*card.count_labels.values(), *card.file_labels.values(), card.commands_button):
            text = label.text().replace("Embedding", "")
            self.assertEqual(re.findall("[A-Za-z]{3,}", text), [], text)

    def test_every_new_text_has_a_draft_in_the_catalog(self):
        drafted = {(context.findtext("name"), message.findtext("source")): message.findtext("translation")
                   for context in ElementTree.parse(ZH_CATALOG).getroot().iter("context")
                   for message in context.iter("message")}
        for context, source in (
                ("Viewer", "Session and controls"), ("ViewerControls", "Session"), ("ViewerControls", "Nodes:"),
                ("ViewerControls", "Edges:"), ("ViewerControls", "Sequence Set:"), ("ViewerControls", "Network:"),
                ("ViewerControls", "Alignment:"), ("ViewerControls", "Embedding:"), ("ViewerControls", "Layout Cache:"),
                ("ViewerControls", "Choose another alignment file…"),
                ("ViewerControls", "Choose another embedding file…"),
                ("ViewerControls", "Fixed for this Viewer session."),
                ("ViewerControls", "None"), ("ViewerControls", "List of commands"),
                ("ViewerControls", "{total} ({visible} visible)"), ("ViewerControls", "{total} ({shown} shown)"),
                ("ViewerControls", "Open the list of commands in your web browser"),
                ("ViewerControls", "The list of commands is not installed: docs/list_of_commands.html is missing."),
                ("ViewerControls", "Console"),
                ("ViewerControls", "Show everything this Viewer has printed: progress, warnings and errors")):
            with self.subTest(source=source):
                self.assertTrue(drafted.get((context, source)), (context, source))

    def test_the_console_link_is_in_chinese_too(self):
        translator = QTranslator()
        self.assertTrue(translator.load(str(ZH_CATALOG.with_suffix(".qm"))))
        self.assertTrue(self.app.installTranslator(translator))
        self.addCleanup(self.app.removeTranslator, translator)
        host = QWidget()
        self.addCleanup(host.deleteLater)
        host.resize(700, 500)
        info = {"node_count": 1, "visible_node_count": 1, "edge_count": 0, "shown_edge_count": 0,
                "inputs": {"node_fasta": os.path.abspath("set.fasta"), "msa": None}}
        button = ControlsButton("会话与操作", shipped_actions(), host, host=host, session=lambda: info,
                                console=lambda: None)
        host.show()
        card = button.toggle_card()
        flush(self.app)
        self.assertEqual(card.console_button.text(), "控制台")
        self.assertEqual(card.console_button.toolTip(), "显示此查看器打印的全部内容：进度、警告和错误")


class ZoomBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make_spin_box(self, zoom=229.57, lowest=220.0, highest=ZOOM_ROOM):
        spin_box = ZoomSpinBox()
        self.addCleanup(spin_box.deleteLater)
        spin_box.show_zoom(zoom, lowest, highest)
        sets = []
        spin_box.valueChanged.connect(sets.append)
        return spin_box, sets

    def test_the_view_shows_without_a_signal_and_the_user_sets_with_one(self):
        spin_box, sets = self.make_spin_box()
        self.assertEqual(spin_box.text(), "230")
        self.assertEqual((spin_box.minimum(), spin_box.maximum()), (220.0, ZOOM_ROOM))
        spin_box.show_zoom(300.04, 220.0, ZOOM_ROOM)
        self.assertEqual((spin_box.text(), sets), ("300", []))
        spin_box.setValue(500.0)
        self.assertEqual(sets, [500.0])
        # A zoom beyond the limits, as after the nodes were moved together, still shows.
        spin_box.show_zoom(150.0, 220.0, ZOOM_ROOM)
        self.assertEqual(spin_box.value(), 150.0)
        self.assertEqual(spin_box.minimum(), 150.0)
        # With no limit to keep to, a low zoom still shows.
        spin_box.show_zoom(0.4, None, ZOOM_ROOM)
        self.assertEqual(spin_box.value(), 0.4)

    def test_a_limit_a_float_digit_off_is_still_the_limit(self):
        spin_box, sets = self.make_spin_box(zoom=219.99999999999997, lowest=220.0)
        self.assertEqual((spin_box.minimum(), spin_box.value(), spin_box.text()), (220.0, 220.0, "220"))
        spin_box.stepBy(-1)
        self.assertEqual(sets, [])  # Already at the lowest zoom.
        spin_box.show_zoom(ZOOM_ROOM * (1 + 1e-13), 220.0, ZOOM_ROOM)
        self.assertEqual(spin_box.maximum(), ZOOM_ROOM)

    def test_the_precision_reads_from_one_percent_to_the_highest_zoom(self):
        # Three significant figures, and every digit of the whole part.
        for zoom, text in ((0.5, "0.500"), (0.826, "0.826"), (1.0, "1.00"), (1.87, "1.87"),
                           (9.996, "10.0"), (23.26, "23.3"), (99.96, "100"), (123.4, "123"),
                           (1234.5, "1234"), (20000.4, "20000"), (123456.7, "123457"),
                           (200000.0, "200000")):
            with self.subTest(zoom=zoom):
                spin_box, _sets = self.make_spin_box(zoom=zoom, lowest=0.1)
                self.assertEqual(spin_box.text(), text)
        self.assertEqual([zoom_decimals(value) for value in (0.1, 0.99, 1, 9.99, 10, 99.9, 100, 200000)],
                         [3, 3, 2, 2, 1, 1, 0, 0])
        # The number it keeps is the one it shows, so leaving the box as it is changes nothing.
        spin_box, sets = self.make_spin_box(zoom=229.57)
        self.assertEqual((spin_box.value(), shown_zoom(229.57)), (230.0, 230.0))
        self.assertEqual(shown_zoom(23.26), 23.3)
        self.assertEqual(shown_zoom(1.874), 1.87)

    def test_a_typed_number_is_kept_as_typed_with_or_without_the_sign(self):
        spin_box, sets = self.make_spin_box()
        for typed, value in (("250", 250.0), ("300 %", 300.0), ("300%", 300.0), ("1234.5", 1234.5),
                             ("12345", 12345.0), ("0221.4", 221.4)):
            spin_box.lineEdit().selectAll()
            QTest.keyClicks(spin_box, typed)
            QTest.keyClick(spin_box, Qt.Key.Key_Return)
            with self.subTest(typed=typed):
                self.assertEqual(sets[-1], value)
        self.assertEqual(spin_box.text(), "221")

    def test_the_box_holds_only_the_number_and_drops_a_typed_percent_sign(self):
        # The % sign is the text after the box (ZoomBox), so the box has no suffix; a sign typed
        # or pasted after the number is left out of what it holds.
        spin_box, sets = self.make_spin_box()
        self.assertEqual((spin_box.suffix(), spin_box.prefix()), ("", ""))
        spin_box.lineEdit().selectAll()
        QTest.keyClicks(spin_box, "250%")
        self.assertEqual(spin_box.lineEdit().text(), "250")
        QTest.keyClick(spin_box, Qt.Key.Key_Return)
        self.assertEqual((sets, spin_box.text()), ([250.0], "250"))
        spin_box.lineEdit().selectAll()
        spin_box.lineEdit().insert("260 %")  # A paste.
        self.assertEqual(spin_box.lineEdit().text(), "260")
        # A sign with no number, or after text that is no number, gives nothing.
        spin_box.lineEdit().selectAll()
        QTest.keyClicks(spin_box, "%")
        self.assertEqual(spin_box.lineEdit().text(), "")
        spin_box.lineEdit().selectAll()
        QTest.keyClicks(spin_box, "x%")
        self.assertEqual(spin_box.lineEdit().text(), "")

    def test_enter_on_the_number_shown_changes_nothing(self):
        # Clicking in the box and pressing Enter must not nudge a view at 229.57% to 230%.
        spin_box, sets = self.make_spin_box()
        spin_box.lineEdit().selectAll()
        QTest.keyClick(spin_box, Qt.Key.Key_Return)
        spin_box.lineEdit().selectAll()
        QTest.keyClicks(spin_box, "230")
        QTest.keyClick(spin_box, Qt.Key.Key_Return)
        self.assertEqual(sets, [])

    def test_a_number_outside_the_limits_is_not_taken(self):
        spin_box, sets = self.make_spin_box()
        for typed in ("5", "219", "0"):
            spin_box.lineEdit().selectAll()
            QTest.keyClicks(spin_box, typed)
            QTest.keyClick(spin_box, Qt.Key.Key_Return)
        self.assertEqual((sets, spin_box.text()), ([], "230"))

    def test_a_step_zooms_by_a_factor_and_stops_at_the_limits(self):
        spin_box, sets = self.make_spin_box(zoom=400.0)
        # Up and the wheel forward zoom in: a bigger percentage; down zooms out.
        spin_box.stepBy(1)
        spin_box.stepBy(-2)
        self.assertEqual(sets, [400.0 * ZOOM_STEP, 400.0 * ZOOM_STEP / ZOOM_STEP ** 2])
        self.assertGreaterEqual(ZOOM_STEP, 1.1)
        spin_box.stepBy(60)
        self.assertEqual(sets[-1], ZOOM_ROOM)
        spin_box.stepBy(-80)
        self.assertEqual(sets[-1], 220.0)

    def test_the_arrows_the_wheel_and_the_keys_zoom_in_for_up_and_forward(self):
        spin_box, sets = self.make_spin_box(zoom=400.0)
        spin_box.setFocus()
        QTest.keyClick(spin_box, Qt.Key.Key_Up)
        QTest.keyClick(spin_box, Qt.Key.Key_Down)
        QTest.keyClick(spin_box, Qt.Key.Key_Down)
        self.assertEqual(sets, [500.0, 400.0, 320.0])
        # A wheel notch forward (a positive angle) is Up.
        wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        self.app.sendEvent(spin_box, wheel)
        self.assertEqual(sets[-1], 400.0)

    def test_typing_counts_on_enter_and_esc_puts_back_the_view(self):
        spin_box, sets = self.make_spin_box()
        dismissed = []
        spin_box.dismissed.connect(lambda: dismissed.append(True))
        self.assertFalse(spin_box.keyboardTracking())
        spin_box.lineEdit().selectAll()
        QTest.keyClicks(spin_box, "750")
        self.assertEqual(sets, [])
        QTest.keyClick(spin_box, Qt.Key.Key_Escape)
        self.assertEqual((spin_box.text(), sets, dismissed), ("230", [], [True]))
        spin_box.lineEdit().selectAll()
        QTest.keyClicks(spin_box, "750")
        QTest.keyClick(spin_box, Qt.Key.Key_Return)
        self.assertEqual(sets, [750.0])

    def test_the_view_waits_while_the_box_is_typed_in(self):
        spin_box, _sets = self.make_spin_box()
        window = QWidget()
        self.addCleanup(window.deleteLater)
        spin_box.setParent(window)
        window.show()
        spin_box.setFocus()
        self.app.processEvents()
        if not spin_box.hasFocus():
            self.skipTest("this platform gives no window the keyboard")
        spin_box.show_zoom(900.0, 220.0, ZOOM_ROOM)
        self.assertEqual(spin_box.text(), "230")

    def test_it_never_takes_the_keyboard_from_the_network_unasked(self):
        spin_box, _sets = self.make_spin_box()
        self.assertEqual(spin_box.focusPolicy(), Qt.FocusPolicy.ClickFocus)

    def test_its_width_holds_as_the_limits_change(self):
        spin_box, _sets = self.make_spin_box(lowest=0.1)
        narrow = spin_box.sizeHint()
        spin_box.show_zoom(242.0, 3000.0, 4000.0)
        self.assertEqual(spin_box.sizeHint(), narrow)
        self.assertEqual(narrow.height(), SLIDER_BOX_HEIGHT)
        self.assertGreaterEqual(
            narrow.width(), QFontMetrics(spin_box.font()).horizontalAdvance(spin_box.textFromValue(ZOOM_ROOM)))

    def test_the_text_either_side_of_the_zoom(self):
        for template, before, after in (("Zoom: {zoom}%", "Zoom:", "%"),
                                        ("{zoom} closer", "", "closer"),
                                        ("缩放：{zoom}%", "缩放：", "%"),
                                        ("%{zoom}", "%", ""),
                                        ("Zoom: {zoom}", "Zoom:", "")):
            box = ZoomBox(template)
            self.addCleanup(box.deleteLater)
            with self.subTest(template=template):
                self.assertEqual((box.before.text(), box.after.text()), (before, after))
                self.assertEqual(box.before.isVisibleTo(box), bool(before))
                self.assertEqual(box.after.isVisibleTo(box), bool(after))
                layout = box.layout()
                self.assertEqual([layout.itemAt(index).widget() for index in range(layout.count())],
                                 [box.before, box.spin_box, box.after])
                # The text before the box keeps clear of it, not crowding it.
                self.assertGreaterEqual(box.before.contentsMargins().right(), 10)

    def test_a_percent_sign_after_the_box_sits_close_and_other_text_keeps_clear(self):
        close = ZoomBox("Zoom: {zoom}%")
        full_width = ZoomBox("{zoom}\uff05")
        word = ZoomBox("{zoom} closer")
        for box in (close, full_width, word):
            self.addCleanup(box.deleteLater)
        self.assertEqual(close.after.contentsMargins().left(), ZOOM_UNIT_GAP)
        self.assertEqual(full_width.after.contentsMargins().left(), ZOOM_UNIT_GAP)
        self.assertEqual(word.after.contentsMargins().left(), ZOOM_LABEL_GAP)
        self.assertLess(ZOOM_UNIT_GAP, ZOOM_LABEL_GAP)
        self.assertEqual(close.layout().spacing(), 0)  # The gaps are the labels' own.



class ThresholdBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make_box(self, lowest, highest):
        box = ThresholdSpinBox()
        self.addCleanup(box.deleteLater)
        box.set_scale(lowest, highest)
        return box

    def test_its_step_comes_from_the_range_of_the_scores(self):
        # The largest power of ten that still gives THRESHOLD_STEPS steps, and
        # no more than 1; the numbers run between multiples of it at or under
        # the lowest and highest scores.
        for lowest, highest, step, decimals, ends, positions in (
                (0.0, 9.6, 0.001, 3, (0.0, 9.6), 9600),
                (0.0, 1.0, 0.001, 3, (0.0, 1.0), 1000),
                (0.0, 0.999, 0.0001, 4, (0.0, 0.999), 9990),
                (0.31274, 0.98, 0.0001, 4, (0.3127, 0.98), 6673),
                (0.3, 0.9, 0.0001, 4, (0.3, 0.9), 6000),
                (0.2, 0.3, 0.0001, 4, (0.2, 0.3), 1000),  # 0.3 - 0.2 is a hair under 0.1.
                # 0.57 * 100 and 16.06 * 100 are each a hair under a whole number.
                (0.57, 16.06, 0.01, 2, (0.57, 16.06), 1549),
                (5.2, 300.0, 0.1, 1, (5.2, 300.0), 2948),
                (12.0, 2400.0, 1.0, 0, (12.0, 2400.0), 2388),
                (0.0, 100000.0, 1.0, 0, (0.0, 100000.0), 100000),
                (-2.55, 7.45, 0.01, 2, (-2.55, 7.45), 1000)):
            with self.subTest(lowest=lowest, highest=highest):
                box = self.make_box(lowest, highest)
                self.assertEqual((box.singleStep(), box.decimals()), (step, decimals))
                self.assertEqual((box.minimum(), box.maximum()), ends)
                self.assertEqual(box.positions, positions)
                self.assertGreaterEqual(box.positions, THRESHOLD_STEPS)
                self.assertEqual(box.value_at(0), box.minimum())
                self.assertEqual(box.value_at(box.positions), box.maximum())

    def test_each_number_is_a_slider_position(self):
        box = self.make_box(0.3, 0.9)
        for position in range(box.positions + 1):
            value = box.value_at(position)
            self.assertEqual(round(value, box.decimals()), value)
            self.assertEqual(box.position_of(value), position)
        # A number between two positions goes to the one under it; one
        # outside the range, to its end.
        self.assertEqual(box.position_of(0.30019), 1)
        self.assertEqual(box.position_of(0.1), 0)
        self.assertEqual(box.position_of(5.0), box.positions)

    def test_its_ends_show_every_edge_and_the_best(self):
        import numpy

        rng = numpy.random.default_rng(42)
        for scale in (1.0, 9.6, 300.0, 2400.0):
            scores = (rng.random(500) * scale).astype(numpy.float32)
            with self.subTest(scale=scale):
                # The Viewer's range: the lowest and highest scores, as floats.
                box = self.make_box(float(scores.min()), float(scores.max()))
                self.assertTrue((scores >= box.minimum()).all())
                self.assertTrue((scores >= box.maximum()).any())

    def test_typing_counts_on_enter_and_esc_puts_back_the_number(self):
        box = self.make_box(0.0, 9.6)
        sets, dismissed = [], []
        box.valueChanged.connect(sets.append)
        box.dismissed.connect(lambda: dismissed.append(True))
        box.show_value(3.74)
        box.set_scale(0.0, 9.6)
        self.assertEqual((box.text(), sets), ("3.740", []))
        self.assertFalse(box.keyboardTracking())
        self.assertEqual(box.focusPolicy(), Qt.FocusPolicy.ClickFocus)
        box.lineEdit().selectAll()
        QTest.keyClicks(box, "5.5")
        self.assertEqual(sets, [])
        QTest.keyClick(box, Qt.Key.Key_Escape)
        self.assertEqual((box.text(), sets, dismissed), ("3.740", [], [True]))
        box.lineEdit().selectAll()
        QTest.keyClicks(box, "5.5")
        QTest.keyClick(box, Qt.Key.Key_Return)
        self.assertEqual(sets, [5.5])
        # A step is the smallest change.
        box.stepBy(1)
        self.assertEqual(sets[-1], 5.501)

    def test_it_is_as_tall_as_the_zoom_box(self):
        box = self.make_box(0.0, 9.6)
        self.assertEqual(box.sizeHint().height(), SLIDER_BOX_HEIGHT)
        self.assertGreaterEqual(box.sizeHint().width(), QFontMetrics(box.font()).horizontalAdvance("9.600"))


class SidebarWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make_sidebar(self, labels=(("Meta Data", "chart-column"), ("Agent", "bot"))):
        """A sidebar as the Viewer builds it, its buttons fitted to the widest label."""
        canvas = QWidget()
        self.addCleanup(canvas.deleteLater)
        canvas.resize(600, 400)
        panel = SidebarPanel(canvas)
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 20, 10 + SIDEBAR_INSET, 20)
        canvas.setStyleSheet("QWidget#rightPanel QPushButton { padding-left: 10px; min-height: 24px; }")
        buttons = []
        for label, icon in labels:
            button = SidebarButton(label, panel)
            button.setIcon(studio_icon(icon))
            button.setIconSize(QSize(16, 16))
            layout.addWidget(button)
            buttons.append(button)
        layout.addStretch()
        width = fit_buttons_to_text(*buttons) + 20 + SIDEBAR_INSET
        panel.setGeometry(600 - width, 0, width, 400)
        canvas.show()
        flush(self.app)
        return panel, buttons

    def test_the_card_has_round_corners_clear_of_the_canvas_edges(self):
        panel, _buttons = self.make_sidebar()
        card = panel.card_rect()
        self.assertAlmostEqual(card.top(), SIDEBAR_INSET, delta=1)
        self.assertAlmostEqual(panel.width() - card.right(), SIDEBAR_INSET, delta=1)
        self.assertAlmostEqual(panel.height() - card.bottom(), SIDEBAR_INSET, delta=1)
        image = QImage(panel.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            # Only what the panel paints: the canvas's background stays out of its corners.
            panel.render(painter, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        finally:
            painter.end()
        # The card's corner is round: its corner pixel is clear, a pixel inside is filled.
        self.assertEqual(image.pixelColor(1, SIDEBAR_INSET + 1).alpha(), 0)
        self.assertEqual(image.pixelColor(panel.width() - SIDEBAR_INSET - 2, panel.height() - SIDEBAR_INSET - 2)
                         .alpha(), 0)
        self.assertGreater(image.pixelColor(panel.width() // 2, panel.height() // 2).alpha(), 200)
        # Above the card, the canvas shows through.
        self.assertEqual(image.pixelColor(panel.width() // 2, SIDEBAR_INSET // 2).alpha(), 0)

    def ink_columns(self, button):
        """The x of every column where the button draws dark text, right of its icon."""
        image = button.grab().toImage()
        ratio = image.devicePixelRatio()
        start = round(button.icon_rect().right() * ratio) + 2
        middle = round(button.height() / 2 * ratio)
        columns = []
        for x in range(start, image.width()):
            for y in range(middle - round(4 * ratio), middle + round(4 * ratio)):
                if QColor(image.pixel(x, y)).lightness() < 110:
                    columns.append(x / ratio)
                    break
        return columns

    def test_a_label_is_centred_in_the_room_after_its_icon(self):
        _panel, (meta, agent) = self.make_sidebar()
        self.assertEqual(meta.width(), agent.width())
        for button in (meta, agent):
            with self.subTest(label=button.text()):
                room = button.label_rect()
                self.assertGreater(room.left(), button.icon_rect().right())
                # The widest label fits its room, so nothing clips.
                self.assertLessEqual(button.fontMetrics().horizontalAdvance(button.text()), room.width())
                ink = self.ink_columns(button)
                self.assertTrue(ink)
                # The text's box is centred in the room: its ink starts where the
                # centred box does, give or take the first letter's bearing.
                metrics = QFontMetrics(button.font())
                box_left = room.left() + (room.width() - metrics.horizontalAdvance(button.text())) / 2
                bounds = metrics.tightBoundingRect(button.text())
                self.assertAlmostEqual(ink[0], box_left + bounds.left(), delta=1.5)
        # The shorter label starts later than the longer: it is centred, not left-aligned.
        self.assertGreater(self.ink_columns(agent)[0], self.ink_columns(meta)[0] + 5)


if __name__ == "__main__":
    unittest.main()
