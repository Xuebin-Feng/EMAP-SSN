# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tip panel (desktop.Tip_Panel) and the tips the Config and Tools windows give it.

A tip is plain text, shown as it always was, or a Tip the panel lays out: the
field's label as a title, a summary, a table of options, notes with status
icons, a live note and a facts footer read from the field. These tests read the
rich text back as a QTextDocument, as a user sees it: which text is a code chip,
which image is which status icon, in which colour.
"""

import base64
import json
import os
import pathlib
import re
import sys
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Hardware_Acceleration  # noqa: E402,F401 - load torch before PySide6
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QFocusEvent, QTextCharFormat, QTextDocument  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTextBrowser,
    QWidget,
)

from desktop.Desktop_App import MONOSPACE_QSS_FONT_STACK, ToggleSwitch  # noqa: E402
from desktop.Studio_Theme import TOKENS, add_hint_icon, status_color  # noqa: E402
from desktop.Tip_Panel import (  # noqa: E402
    MISSING,
    Code,
    Field,
    FieldTip,
    LiveNote,
    SpacedTipLabel,
    Tip,
    TipNote,
    TipOption,
    change_signal,
    chip,
    is_whole,
    model_html,
    plain,
    resolve,
    rich,
    status_html,
    swatch_mark,
    tip_html,
    tip_plain,
    wrappable,
)
from tests.config_gui_loader import load_config_namespace, open_config_window  # noqa: E402
from tests.consistency_report import ZERO_WIDTH_SPACE, mark_colour  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from tests.tools_gui_fixtures import isolated_tools_project  # noqa: E402

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


# ---------------------------------------------------------------------
# Texts: escaping, chips, placeholders
# ---------------------------------------------------------------------

class InlineTextTests(WithApplication):
    def test_text_is_escaped_so_it_is_never_markup(self):
        shown = rich("a <b>bold</b> & <img src=x> `<i>c</i>`")
        self.assertNotIn("<b>", shown)
        self.assertNotIn("<img", shown)
        self.assertNotIn("<i>", shown)
        self.assertEqual(squashed(plain_text(shown)), squashed("a <b>bold</b> & <img src=x> <i>c</i>"))

    def test_backtick_spans_become_chips_and_nothing_else_does(self):
        shown = rich("A color name or hex code, e.g. `grey` or `#333333`.")
        self.assertEqual(chips(shown), ["grey", "#333333"])
        self.assertEqual(squashed(plain_text(shown)), squashed("A color name or hex code, e.g. grey or #333333."))

    def test_plain_drops_the_backticks_only(self):
        self.assertEqual(
            plain("Uses `{file}` and `-k 0`.", {"file": "viewer_settings.json"}),
            "Uses viewer_settings.json and -k 0.",
        )

    def test_an_unbalanced_backtick_stays_text(self):
        shown = rich("a ` b")
        self.assertEqual(chips(shown), [])
        self.assertIn("`", plain_text(shown))

    def test_a_placeholder_takes_a_value_that_is_never_read_as_markup(self):
        hostile = "`<b>x</b>` {value}"
        shown = rich("Now: {value}", {"value": hostile})
        self.assertEqual(chips(shown), [])
        self.assertEqual(plain_text(shown), f"Now: {hostile}")
        self.assertEqual(chips(rich("Now: {value}", {"value": Code(hostile)})), [hostile])

    def test_a_placeholder_takes_a_format_spec_and_an_unknown_one_stays(self):
        self.assertEqual(plain("{count:,} of {total}", {"count": 12345}), "12,345 of {total}")

    def test_a_chip_keeps_its_spaces_together_and_a_long_one_can_wrap(self):
        self.assertIn("-k&nbsp;0", chip("-k 0"))
        long_chip = chip("sid|18367|pid|10722|hfid|10|sfid|2|gi|914684625|pdb|3WWH(A)|taxonID|1042534")
        self.assertIn(ZERO_WIDTH_SPACE, long_chip)
        self.assertIn(MONOSPACE_QSS_FONT_STACK, long_chip)

    def test_wrappable_leaves_ordinary_words_alone(self):
        self.assertEqual(wrappable("WP_012900.1_hypothetical_protein"), "WP_012900.1_hypothetical_protein")
        self.assertGreater(wrappable("A" * 200).count(ZERO_WIDTH_SPACE), 5)


# ---------------------------------------------------------------------
# A tip resolved against its field
# ---------------------------------------------------------------------

def combo_of(*items):
    combo = QComboBox()
    for text, data in items:
        combo.addItem(text, data)
    return combo


class ResolveTests(WithApplication):
    def test_the_title_is_the_fields_label_without_its_colon(self):
        for text, title in (
            ("CPU Workers:", "CPU Workers"),
            ("主机缓存（GiB）：", "主机缓存（GiB）"),
            ("Save && Run:", "Save & Run"),
            ("&Quick label :", "Quick label"),
        ):
            with self.subTest(text=text):
                model = resolve(FieldTip(Tip("x"), Field(label=QLabel(text))))
                self.assertEqual(model.title, title)
        self.assertEqual(resolve(FieldTip(Tip("x"), Field(label="Plain label:"))).title, "Plain label")
        self.assertEqual(resolve(Tip("x")).title, "")
        self.assertEqual(resolve(FieldTip(Tip("x", title="Own"), Field(label=QLabel("Label:")))).title, "Own")

    def test_options_name_the_choices_as_the_dropdown_does_and_skip_the_ones_it_lacks(self):
        combo = combo_of(("Automatic 32-bit", "automatic_32bit"), ("float32", "float32"), ("BF16 (Low Precision)", "bf16"))
        tip = Tip("x", options=(
            TipOption("automatic_32bit", "one"),
            TipOption("float32", "two"),
            TipOption("tf32", "three"),
            TipOption("bf16", "four"),
        ))
        model = resolve(FieldTip(tip, Field(widget=combo)))
        self.assertEqual(model.options, [
            ("Automatic 32-bit", "one"), ("float32", "two"), ("BF16 (Low Precision)", "four"),
        ])
        # The dropdown follows the device: an option it gains shows, one it loses goes.
        combo.addItem("TF32 (Nvidia GPU Only)", "tf32")
        combo.removeItem(combo.findData("float32"))
        rows = resolve(FieldTip(tip, Field(widget=combo))).options
        # In the tip's order, whatever order the dropdown lists them in.
        self.assertEqual(
            [label for label, _ in rows],
            ["Automatic 32-bit", "TF32 (Nvidia GPU Only)", "BF16 (Low Precision)"],
        )

    def test_an_option_with_a_label_does_not_need_a_dropdown(self):
        tip = Tip("x", options=(TipOption(None, "on", label="AUTO ON"), TipOption("raw", "no label")))
        self.assertEqual(resolve(FieldTip(tip, Field())).options, [("AUTO ON", "on"), ("raw", "no label")])

    def test_the_default_is_shown_as_the_field_shows_it(self):
        combo = combo_of(("auto", "auto"), ("tiled", "tiled"))
        combo.setItemText(0, "Automatisch")
        switch = ToggleSwitch("AUTO ON", "AUTO OFF")
        cases = (
            (Field(widget=combo, default="auto"), "Automatisch"),
            (Field(default=""), "(empty)"),
            (Field(default=None), "(empty)"),
            (Field(default=12), "12"),
            (Field(default=0.1), "0.1"),
            (Field(default=2.0), "2"),
            (Field(widget=switch, default=True), "AUTO ON"),
            (Field(widget=switch, default=False), "AUTO OFF"),
            (Field(default="grey"), "grey"),
        )
        for field, shown in cases:
            with self.subTest(default=field.default):
                self.assertEqual(dict(resolve(FieldTip(Tip("x"), field)).facts).get("Default"), shown)
        self.assertNotIn("Default", dict(resolve(FieldTip(Tip("x"), Field(default=MISSING))).facts))

    def test_a_unit_follows_a_numeric_default_only(self):
        tip = Tip("x", unit="GiB")
        facts = dict(resolve(FieldTip(tip, Field(default=3))).facts)
        self.assertEqual(facts["Default"], "3 GiB")
        self.assertEqual(dict(resolve(FieldTip(tip, Field(default="auto"))).facts)["Default"], "auto")

    def test_the_range_is_read_from_the_spin_box(self):
        box = QSpinBox()
        box.setRange(1, 20)
        decimal = QDoubleSpinBox()
        decimal.setRange(0.0, 100.0)
        decimal.setDecimals(1)
        decimal.setSuffix(" %")
        holder = QWidget()
        QHBoxLayout(holder).addWidget(QSpinBox(holder))
        holder.findChild(QSpinBox).setRange(0, 7)
        for widget, shown in ((box, "1 – 20"), (decimal, "0.0 – 100.0 %"), (holder, "0 – 7")):
            with self.subTest(widget=type(widget).__name__):
                self.assertEqual(dict(resolve(FieldTip(Tip("x"), Field(widget=widget))).facts)["Range"], shown)
        self.assertEqual(
            dict(resolve(FieldTip(Tip("x", unit="GiB"), Field(widget=QSpinBox()))).facts)["Range"],
            "0 – 99 GiB",
        )
        self.assertNotIn("Range", dict(resolve(FieldTip(Tip("x"), Field(widget=QLineEdit()))).facts))

    def test_the_settings_key_is_a_fact_and_facts_can_be_left_out(self):
        field = Field(key="ALIGNMENT_REFERENCE", default="")
        self.assertEqual(
            resolve(FieldTip(Tip("x"), field)).facts,
            [("Default", "(empty)"), ("Setting", "ALIGNMENT_REFERENCE")],
        )
        self.assertEqual(resolve(FieldTip(Tip("x", facts=False), field)).facts, [])
        self.assertEqual(resolve(Tip("x")).facts, [])

    def test_a_live_note_is_worked_out_from_the_field_each_time(self):
        line = QLineEdit("grey")
        tip = Tip("x", live=lambda field: LiveNote("Now {value}", {"value": field.widget.text()}))
        entry = FieldTip(tip, Field(widget=line))
        self.assertEqual(resolve(entry).live.values, {"value": "grey"})
        line.setText("red")
        self.assertEqual(resolve(entry).live.values, {"value": "red"})
        self.assertIsNone(resolve(FieldTip(Tip("x"), Field())).live)

    def test_a_tip_lists_every_text_it_holds(self):
        tip = Tip(
            "summary", details=("detail",), options=(TipOption("a", "option"),),
            notes=(TipNote("info", "note"),),
        )
        self.assertEqual(tip.texts(), ["summary", "detail", "option", "note"])


# ---------------------------------------------------------------------
# Drawing it
# ---------------------------------------------------------------------

FULL = Tip(
    "Summary with `chip`.",
    details=("A detail line.",),
    options=(TipOption("auto", "Does the automatic thing."), TipOption("scalar", "Does it one at a time.")),
    notes=(TipNote("warning", "Careful now."), TipNote("info", "Good to know.")),
    live=lambda field: LiveNote("Right now {value}", {"value": Code("42")}),
)


def full_entry():
    combo = combo_of(("auto", "auto"), ("scalar", "scalar"))
    spin = QSpinBox()
    spin.setRange(1, 20)
    return FieldTip(FULL, Field(key="THE_KEY", label=QLabel("The Field:"), widget=combo, default="auto")), spin


class RenderTests(WithApplication):
    def test_the_layout_reads_title_summary_options_details_notes_live_facts(self):
        entry, _ = full_entry()
        text = plain_text(tip_html(entry))
        order = [
            "The Field", "Summary with chip", "auto", "Does the automatic thing", "scalar",
            "A detail line", "Careful now", "Good to know", "Right now 42", "Default", "Setting", "THE_KEY",
        ]
        positions = [text.index(part) for part in order]
        self.assertEqual(positions, sorted(positions), text)

    def test_options_and_facts_values_are_chips(self):
        entry, _ = full_entry()
        self.assertEqual(
            chips(tip_html(entry)),
            ["chip", "auto", "scalar", "42", "auto", "THE_KEY"],
        )

    def test_notes_and_the_live_note_have_their_status_icons(self):
        entry, _ = full_entry()
        marks = [mark_colour(name) for name in images(tip_html(entry))]
        # warning, info, then the live note (an info icon).
        self.assertEqual(marks, [status_color("warning"), status_color("subtle"), status_color("subtle")])

    def test_the_icons_differ_in_shape_as_well_as_colour(self):
        shapes = {kind: svg_of(images(status_html(kind, "x"))[0]) for kind in ("success", "warning", "danger", "info")}
        self.assertEqual(len(set(shapes.values())), 4)

    def test_a_colour_live_note_shows_a_swatch_of_that_colour(self):
        tip = Tip("x", live=lambda field: LiveNote("Now {value}", {"value": Code("#336699")}, swatch="#336699"))
        shown = tip_html(FieldTip(tip, Field()))
        [name] = images(shown)
        self.assertIn('fill="#336699"', svg_of(name))
        # Not an icon: a colour the program can't read falls back to the info icon.
        tip = Tip("x", live=lambda field: LiveNote("Now", swatch="not a colour"))
        [name] = images(tip_html(FieldTip(tip, Field())))
        self.assertEqual(mark_colour(name), status_color("subtle"))
        self.assertEqual(swatch_mark("nonsense"), "")

    def test_colours_come_from_the_theme_when_the_tip_is_shown(self):
        entry, _ = full_entry()
        colours = {"surface": "#040506", "accent_text": "#0a0b0c", "text": "#070809", "warning": "#0d0e0f",
                   "text_subtle": "#101112"}
        with mock.patch.dict(TOKENS, colours):
            shown = tip_html(entry)
            self.assertEqual(
                {text_format.background().color().name() for _, text_format in pieces(shown)
                 if text_format.background().style() != Qt.BrushStyle.NoBrush},
                {"#040506"},
            )
            self.assertEqual(mark_colour(images(shown)[0]), "#0d0e0f")
            title = next(text_format for text, text_format in pieces(shown) if clean(text) == "The Field")
            self.assertEqual(title.foreground().color().name(), "#070809")
        # Nothing is left of another theme.
        self.assertNotIn("#040506", tip_html(entry))

    def test_a_tip_is_escaped_whatever_it_says(self):
        tip = Tip(
            "<script>alert(1)</script> & `<img src=x>`",
            details=("<b>bold</b>",),
            options=(TipOption("<i>", "<img src=y>"),),
            notes=(TipNote("info", "<h1>big</h1>"),),
        )
        shown = tip_html(FieldTip(tip, Field(label=QLabel("<u>Label</u>:"), key="<K>")))
        text = plain_text(shown)
        for markup in ("<script>alert(1)</script>", "<img src=x>", "<b>bold</b>", "<img src=y>", "<h1>big</h1>",
                       "<u>Label</u>", "<K>"):
            self.assertIn(markup, text)
        self.assertEqual(len(images(shown)), 1)  # only the note's icon

    def test_a_long_token_gets_places_to_wrap_wherever_it_appears(self):
        header = "sid|18367|pid|10722|hfid|10|sfid|2|gi|914684625|pdb|3WWH(A)|taxonID|1042534|Arthrobactersp.KNK168"
        field = Field(label=QLabel("Label:"))
        shown = {
            "title": tip_html(FieldTip(Tip("x"), Field(label=QLabel(header + ":")))),
            "summary": tip_html(FieldTip(Tip(header), field)),
            "detail": tip_html(FieldTip(Tip("x", details=(header,)), field)),
            "option text": tip_html(FieldTip(Tip("x", options=(TipOption("v", header),)), field)),
            "option chip": tip_html(FieldTip(Tip("x", options=(TipOption("v", "y", label=header),)), field)),
            "note": tip_html(FieldTip(Tip("x", notes=(TipNote("info", header),)), field)),
            "live note": tip_html(FieldTip(Tip("x", live=lambda f: LiveNote(header)), field)),
            "fact": tip_html(FieldTip(Tip("x"), Field(key=header))),
            "status": status_html("danger", header),
        }
        for place, rich_text in shown.items():
            with self.subTest(place=place):
                self.assertGreaterEqual(rich_text.count(ZERO_WIDTH_SPACE), 4)

    def test_a_short_option_stays_on_one_line_and_a_long_one_may_wrap(self):
        from PySide6.QtGui import QTextTable

        def first_cell_lines(label, width):
            tip = Tip("x", options=(TipOption("v", "What it does, in a sentence long enough to squeeze the cell."),))
            combo = combo_of((label, "v"))
            parsed = document(tip_html(FieldTip(tip, Field(widget=combo))))
            parsed.setTextWidth(width)
            parsed.size()
            [table] = [frame for frame in parsed.rootFrame().childFrames() if isinstance(frame, QTextTable)]
            return table.cellAt(0, 0).firstCursorPosition().block().layout().lineCount()

        # Chinese may break between any two characters; a short chip still doesn't.
        self.assertEqual(first_cell_lines("TF32（仅限 NVIDIA GPU）", 250), 1)
        self.assertEqual(first_cell_lines("TF32 (Nvidia GPU Only)", 250), 1)
        self.assertGreater(first_cell_lines("NVIDIA GeForce RTX 4090 Laptop GPU with 16 GB", 250), 1)
        self.assertTrue(is_whole("a" * 28))
        self.assertFalse(is_whole("a" * 29))
        self.assertTrue(is_whole("仅限" * 7))
        self.assertFalse(is_whole("仅限" * 8))

    def test_the_plain_text_names_the_icons_in_words(self):
        entry, _ = full_entry()
        text = tip_plain(entry)
        self.assertEqual(text.splitlines(), [
            "The Field",
            "Summary with chip.",
            "auto: Does the automatic thing.",
            "scalar: Does it one at a time.",
            "A detail line.",
            "Warning: Careful now.",
            "Note: Good to know.",
            "Right now 42",
            "Default: auto. Setting: THE_KEY.",
        ])

    def test_a_status_message_is_an_icon_and_its_escaped_text(self):
        shown = status_html("danger", "Error: <b>x</b> & more")
        self.assertEqual(mark_colour(images(shown)[0]), status_color("danger"))
        self.assertEqual(plain_text(shown), "Error: <b>x</b> & more")

    def test_a_long_header_gets_places_to_wrap_so_it_never_widens_the_panel(self):
        header = "sid|18367|pid|10722|hfid|10|sfid|2|gi|914684625|pdb|3WWH(A)|taxonID|1042534|Arthrobactersp.KNK168"
        panel = SpacedTipLabel()
        panel.setWordWrap(True)
        panel.resize(300, 100)
        panel.show()
        panel.setTip(FieldTip(
            Tip(
                f"Names {header} and `{header}`.",
                options=(TipOption("auto", header),),
                notes=(TipNote("warning", header),),
            ),
            Field(label=QLabel(f"{header}:")),  # the title too
        ))
        self.app.processEvents()
        self.assertLessEqual(panel.minimumSizeHint().width(), 300)
        self.assertGreater(panel.minimumHeight(), 50)
        panel.close()


# ---------------------------------------------------------------------
# The panel itself
# ---------------------------------------------------------------------

class PanelTests(WithApplication):
    def panel(self, width=500):
        panel = SpacedTipLabel("start")
        panel.setWordWrap(True)
        panel.resize(width, 100)
        panel.show()
        self.addCleanup(panel.close)
        self.app.processEvents()
        return panel

    def test_plain_text_is_shown_exactly_as_it_always_was(self):
        self.assertEqual(
            SpacedTipLabel("a<b>\nc & d\r\ne").text(),
            '<div style="line-height: 120%;">a&lt;b&gt;<br>c &amp; d<br>e</div>',
        )

    def test_a_plain_tip_goes_through_setTip_unchanged(self):
        panel = self.panel()
        panel.setTip("Plain\ntip")
        self.assertEqual(panel.text(), SpacedTipLabel("Plain\ntip").text())

    def test_a_rich_tip_lays_out_and_sets_a_plain_accessible_name(self):
        entry, _ = full_entry()
        panel = self.panel()
        panel.setTip(entry)
        self.assertEqual(panel.text(), tip_html(entry))
        self.assertEqual(panel.accessibleName(), tip_plain(entry))
        self.assertIn("Warning: Careful now.", panel.accessibleName())
        panel.setTip("plain again")
        self.assertEqual(panel.accessibleName(), "")

    def test_the_panel_is_as_tall_as_a_rich_tip_is_at_its_width_and_plain_text_lets_go(self):
        # Long enough to wrap in a narrow panel, whatever the font.
        entry = FieldTip(Tip(
            "A summary that goes on for a good while, so that it wraps. " * 4 + "`chip`",
            options=(TipOption("auto", "What auto does, in a sentence long enough to wrap at that width. " * 2),),
            notes=(TipNote("warning", "A caution that also goes on for some time, to wrap. " * 3),),
        ), Field(key="K"))
        panel = self.panel()
        panel.setTip(entry)

        def text_height(width):
            # A label's height for a width is never less than its minimum height.
            held = panel.minimumHeight()
            panel.setMinimumHeight(0)
            height = panel.heightForWidth(width)
            panel.setMinimumHeight(held)
            return height

        for width in (500, 250, 900):
            panel.resize(width, 100)
            self.app.processEvents()
            # Whatever width the panel really has, it is as tall as the text is there.
            self.assertEqual(panel.minimumHeight(), text_height(panel.width()), width)
        # The text wraps less when it is wider, so the panel lets go of the height it held.
        self.assertGreater(text_height(250), text_height(900))
        panel.resize(250, 100)
        self.app.processEvents()
        narrow = panel.minimumHeight()
        panel.resize(900, 100)
        self.app.processEvents()
        self.assertLess(panel.minimumHeight(), narrow)
        panel.setTip("plain")
        self.assertEqual(panel.minimumHeight(), 0)

    def test_a_status_without_a_corner_icon_has_its_icon_beside_the_text(self):
        panel = self.panel()
        self.assertIsNone(panel._corner_icon())
        panel.setStatus("success", "Done <ok>.")
        self.assertEqual(mark_colour(images(panel.text())[0]), status_color("success"))
        self.assertEqual(plain_text(panel.text()), "Done <ok>.")
        self.assertEqual(panel.accessibleName(), "Done <ok>.")
        self.assertFalse(panel.refresh())

    def corner_panel(self):
        panel = self.panel()
        add_hint_icon(panel)
        return panel, panel._corner_icon()

    def test_a_status_is_marked_by_the_corner_icon_and_not_twice(self):
        panel, corner = self.corner_panel()
        self.assertEqual((corner.icon_name, corner.icon_color), ("info", TOKENS["text_subtle"]))
        for kind, name, colour in (
            ("success", "check", status_color("success")),
            ("warning", "triangle-alert", status_color("warning")),
            ("danger", "x", status_color("danger")),
            ("info", "info", TOKENS["text_subtle"]),
        ):
            with self.subTest(kind=kind):
                panel.setStatus(kind, "Done <ok>.\nSecond line.")
                self.assertEqual((corner.icon_name, corner.icon_color), (name, colour))
                # One icon: the text has no mark of its own, and keeps its line break.
                self.assertEqual(images(panel.text()), [])
                self.assertEqual(plain_text(panel.text()), "Done <ok>.\nSecond line.")
                self.assertEqual(panel.accessibleName(), "Done <ok>.\nSecond line.")
                self.assertFalse(panel.refresh())

    def test_the_corner_icon_goes_back_to_the_info_icon_when_a_tip_or_text_replaces_the_status(self):
        entry, _ = full_entry()
        panel, corner = self.corner_panel()
        for replace in (lambda: panel.setTip(entry), lambda: panel.setTip("plain tip"),
                        lambda: panel.setText("plain"), lambda: panel.setReport("<p>r</p>")):
            with self.subTest():
                panel.setStatus("danger", "Failed.")
                self.assertEqual(corner.icon_name, "x")
                replace()
                self.assertEqual((corner.icon_name, corner.icon_color), ("info", TOKENS["text_subtle"]))
        # The same status twice (the text unchanged) still marks the corner.
        panel.setStatus("warning", "Same")
        panel.setText("tip")
        panel.setStatus("warning", "Same")
        self.assertEqual(corner.icon_name, "triangle-alert")

    def test_a_report_shows_its_overall_verdict_in_the_corner_and_keeps_its_own_icons(self):
        panel, corner = self.corner_panel()
        report = status_html("success", "row one") + status_html("danger", "row two")
        panel.setReport(report, "danger")
        self.assertEqual((corner.icon_name, corner.icon_color), ("x", status_color("danger")))
        self.assertEqual(len(images(panel.text())), 2)
        panel.setReport(report)
        self.assertEqual(corner.icon_name, "info")

    def test_refresh_shows_a_tip_again_and_leaves_anything_else_alone(self):
        line = QLineEdit("red")
        entry = FieldTip(
            Tip("x", live=lambda field: LiveNote("Now", swatch=field.widget.text())), Field(widget=line)
        )
        panel = self.panel()
        panel.setTip(entry)
        before = panel.text()
        line.setText("#336699")
        self.assertTrue(panel.refresh())
        self.assertNotEqual(panel.text(), before)
        self.assertIn('fill="#336699"', svg_of(images(panel.text())[0]))
        for show in (lambda: panel.setStatus("info", "msg"), lambda: panel.setReport("<p>r</p>"),
                     lambda: panel.setText("plain")):
            show()
            shown = panel.text()
            self.assertFalse(panel.refresh())
            self.assertEqual(panel.text(), shown)

    def test_the_same_rich_text_is_not_laid_out_twice(self):
        entry, _ = full_entry()
        panel = self.panel()
        panel.setTip(entry)
        with mock.patch.object(QLabel, "setText") as set_text:
            panel.setTip(entry)
            panel.setTip(entry)
        set_text.assert_not_called()

    def test_change_signal_finds_the_signal_of_a_field(self):
        spin = QSpinBox()
        holder = QWidget()
        QHBoxLayout(holder).addWidget(QSpinBox(holder))
        for widget in (QLineEdit(), spin, QComboBox(), ToggleSwitch(), holder):
            with self.subTest(widget=type(widget).__name__):
                self.assertIsNotNone(change_signal(widget))
        self.assertIsNone(change_signal(None))
        self.assertIsNone(change_signal(QLabel("x")))

    def test_a_tip_renders_within_a_few_milliseconds(self):
        entry, _ = full_entry()
        panel = self.panel()
        panel.setTip(entry)
        start = time.perf_counter()
        for index in range(40):
            panel.setTip(FieldTip(FULL, Field(key=f"K{index}", label=QLabel("F:"), default=index)))
        mean = (time.perf_counter() - start) / 40
        self.assertLess(mean, 0.05, f"{mean * 1000:.1f} ms per tip")


# ---------------------------------------------------------------------
# The Config window
# ---------------------------------------------------------------------

class ConfigWindowTipTests(WithApplication):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.namespace = load_config_namespace()

    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.window = open_config_window(self.namespace["ConfigGUI"], pathlib.Path(home.name))
        self.addCleanup(self._close)
        self.window.show()
        self.app.processEvents()
        self.panel = self.window.tip_panel

    def _close(self):
        self.window.close()
        self.app.processEvents()

    def hover(self, widget):
        from PySide6.QtCore import QEvent, QPoint
        from PySide6.QtGui import QHelpEvent

        self.app.sendEvent(widget, QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), widget.mapToGlobal(QPoint(1, 1))))

    def click(self, widget):
        self.app.sendEvent(widget, left_press())

    def test_the_converted_tips_are_tips_and_the_rest_stay_plain_text(self):
        keys = self.window.tip_db_keys
        for key in ("SAVED_CONFIG", "ALIGNMENT_REFERENCE", "TEXT_COLOR"):
            self.assertIsInstance(keys[key], Tip, key)
        self.assertIsInstance(keys["NODE_FASTA_FILE"], str)
        # A plain tip is shown exactly as before.
        self.click(self.window.labels["NODE_FASTA_FILE"])
        self.assertEqual(self.panel.text(), SpacedTipLabel(keys["NODE_FASTA_FILE"]).text())

    def test_every_widget_of_a_field_shares_one_entry_with_its_field(self):
        db = self.window.tip_db
        entry = db[self.window.inputs["TEXT_COLOR"]]
        self.assertIsInstance(entry, FieldTip)
        self.assertIs(db[self.window.labels["TEXT_COLOR"]], entry)
        self.assertIs(db[self.window.inputs["TEXT_COLOR"].parentWidget()], entry)
        self.assertEqual(
            (entry.field.key, entry.field.default, entry.field.widget),
            ("TEXT_COLOR", "grey", self.window.inputs["TEXT_COLOR"]),
        )
        reference = db[self.window.inputs["ALIGNMENT_REFERENCE"]]
        self.assertEqual((reference.field.key, reference.field.default), ("ALIGNMENT_REFERENCE", ""))

    def test_a_colour_field_shows_a_swatch_that_follows_what_is_typed(self):
        line = self.window.inputs["TEXT_COLOR"]
        line.setText("#336699")
        self.click(self.window.labels["TEXT_COLOR"])
        self.assertIn("Text Color", plain_text(self.panel.text()))
        self.assertIn('fill="#336699"', svg_of(images(self.panel.text())[0]))
        self.assertIn("#336699", chips(self.panel.text()))
        line.setText("#cc0000")
        self.assertIn('fill="#cc0000"', svg_of(images(self.panel.text())[0]))
        # It is still the tip on display: hovering another field replaces it.
        self.hover(self.window.labels["ALIGNMENT_OFFSET"])
        self.assertIn("Integer offset", plain_text(self.panel.text()))

    def test_an_invalid_colour_shows_no_swatch_and_typing_never_replaces_a_status(self):
        line = self.window.inputs["TEXT_COLOR"]
        self.click(self.window.labels["TEXT_COLOR"])
        line.setText("not a colour")
        self.assertEqual(images(self.panel.text()), [])
        self.panel.setStatus("success", "Saved.")
        status = self.panel.text()
        line.setText("red")
        self.assertEqual(self.panel.text(), status)

    def test_a_status_message_locks_hovering_whether_it_is_plain_report_or_status(self):
        for show in (
            lambda: self.panel.setStatus("danger", "Failed: x"),
            lambda: self.panel.setReport("<p>Report</p>"),
            lambda: self.panel.setText("Plain message"),
        ):
            with self.subTest():
                self.click(self.window.labels["ALIGNMENT_REFERENCE"])  # a tip: hovering works again
                self.hover(self.window.labels["TEXT_COLOR"])
                self.assertIn("Text Color", plain_text(self.panel.text()))
                show()
                message = self.panel.text()
                self.hover(self.window.labels["TEXT_COLOR"])
                self.hover(self.window.labels["ALIGNMENT_REFERENCE"])
                self.assertEqual(self.panel.text(), message)
                self.click(self.window.labels["TEXT_COLOR"])
                self.assertIn("Text Color", plain_text(self.panel.text()))

    def test_the_tip_text_the_lock_compares_is_the_rich_text_of_the_last_tip(self):
        self.click(self.window.labels["ALIGNMENT_REFERENCE"])
        self.assertEqual(self.window._tip_panel_tip_text, self.panel.text())
        self.panel.setStatus("info", "msg")
        self.assertNotEqual(self.window._tip_panel_tip_text, self.panel.text())

    def test_saved_config_lists_the_entries_of_its_own_tabs_selector(self):
        for tab_id, selector in self.window.profile_selectors.items():
            with self.subTest(tab=tab_id):
                self.click(self.window.profile_labels[tab_id])
                names = [data for data in (selector.itemData(i) for i in range(selector.count()))
                         if data in ("(custom)", "(default)", "(new)")]
                # Each entry as the selector names it; the first row's text holds a chip of its own.
                expected = []
                for name in ("(custom)", "(default)", "(new)"):
                    if name in names:
                        expected.append(selector.itemText(selector.findData(name)))
                        if name == "(custom)":
                            expected.append("viewer_settings.json")
                self.assertEqual(chips(self.panel.text()), expected)
        # The Inputs & Outputs tab has no read-only defaults to offer.
        self.assertNotIn("(default)", [selector.itemData(i) for selector in
                                       [self.window.profile_selectors["inputs_outputs"]]
                                       for i in range(selector.count())])

    def test_the_facts_come_from_the_defaults_and_the_widgets(self):
        self.click(self.window.labels["ALIGNMENT_REFERENCE"])
        text = plain_text(self.panel.text())
        self.assertIn("Default (empty)", re.sub(r"\s+", " ", text))
        self.assertIn("ALIGNMENT_REFERENCE", text)
        self.assertEqual(chips(self.panel.text())[:2], ["WP_0123.1", "WP_01*"])

    def test_the_statistics_messages_are_marked_by_the_corner_icon(self):
        corner = self.panel._corner_icon()
        shown = []
        for kind in ("success", "danger"):
            self.panel.setStatus(kind, "m")
            shown.append((corner.icon_name, corner.icon_color))
            self.assertEqual(images(self.panel.text()), [])
        self.assertEqual(shown, [("check", status_color("success")), ("x", status_color("danger"))])
        self.click(self.window.labels["ALIGNMENT_REFERENCE"])
        self.assertEqual(corner.icon_name, "info")


# ---------------------------------------------------------------------
# The Tools window
# ---------------------------------------------------------------------

class ToolsWindowTipTests(WithApplication):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        import EMAPSSN_Tools

        self.tools = EMAPSSN_Tools
        isolated_tools_project(self)
        patcher = mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.window = EMAPSSN_Tools.ToolsGUI()
        self.addCleanup(self._close)
        self.panel = self.window.tip_panel

    def _close(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def widget(self, script, variable):
        for path, data in self.window.script_data.items():
            if os.path.basename(path) == script:
                return data["inputs"][variable]["widget"]
        raise KeyError((script, variable))

    def enter(self, widget):
        self.app.sendEvent(widget, QEvent(QEvent.Type.Enter))

    def shown(self):
        return self.panel.text()

    def test_the_converted_tips_are_tips_and_the_others_are_still_text(self):
        tips = self.window.SCRIPT_TIPS
        for script, key in (
            ("Align_Similarity_Matrix.py", "EXECUTION_MODE"),
            ("Align_Similarity_Matrix.py", "ACCELERATOR_PRECISION"),
            ("Align_Similarity_Matrix.py", "WORKERS"),
            ("Align_Similarity_Matrix.py", "HOST_CACHE_GB"),
            ("Embedding_MSA.py", "WORKERS"),
        ):
            self.assertIsInstance(tips[script][key], Tip, (script, key))
        self.assertIsInstance(tips["Align_Similarity_Matrix.py"]["INPUT_HDF5"], str)

    def test_a_field_gets_its_tip_with_its_field_and_a_plain_tooltip(self):
        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        entry = self.window.tip_db[combo]
        self.assertIsInstance(entry, FieldTip)
        self.assertEqual((entry.field.key, entry.field.default, entry.field.widget), ("EXECUTION_MODE", "auto", combo))
        self.assertEqual(entry.field.title(), "Execution Mode")
        # The widget's accessible description is the tip as plain text.
        self.assertEqual(combo.toolTip(), tip_plain(entry))
        self.assertIn("scalar: Processes one pairwise score matrix at a time.", combo.toolTip())
        # A plain tip keeps its text as the tooltip.
        plain_field = self.widget("Align_Similarity_Matrix.py", "BATCH_SIZE")
        self.assertEqual(plain_field.toolTip(), self.window.SCRIPT_TIPS["Align_Similarity_Matrix.py"]["BATCH_SIZE"])

    def test_the_options_follow_the_dropdown(self):
        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        def listed():
            return [combo.itemText(combo.findData(value)) for value in ("auto", "scalar", "tiled")
                    if combo.findData(value) >= 0]

        # The device decides whether "tiled" is offered: try both.
        if combo.findData("tiled") < 0:
            combo.addItem("tiled", "tiled")
        self.enter(combo)
        # The options as the dropdown names them, then the facts: the default and the key.
        self.assertEqual(chips(self.shown()), listed() + ["auto", "EXECUTION_MODE"])
        self.assertIn("tiled", listed())
        self.assertIn("memory-bounded embedding tiles", plain_text(self.shown()))
        combo.removeItem(combo.findData("tiled"))
        self.enter(combo)
        self.assertEqual(chips(self.shown()), listed() + ["auto", "EXECUTION_MODE"])
        self.assertNotIn("tiled", listed())
        self.assertNotIn("memory-bounded embedding tiles", plain_text(self.shown()))
        self.assertIn("Processes one pairwise score matrix at a time", plain_text(self.shown()))

    def test_the_precision_tip_has_its_caution_and_leaves_out_what_the_device_lacks(self):
        combo = self.widget("Align_Similarity_Matrix.py", "ACCELERATOR_PRECISION")
        self.enter(combo)
        marks = [mark_colour(name) for name in images(self.shown())]
        self.assertEqual(marks, [status_color("warning")])
        self.assertIn("BF16 prints a low-precision warning", plain_text(self.shown()))
        if combo.findData("tf32") >= 0:
            combo.removeItem(combo.findData("tf32"))
        self.enter(combo)
        self.assertNotIn("TF32 (Nvidia GPU Only)", chips(self.shown()))

    def test_the_workers_tip_shows_this_computers_logical_cpus_and_its_range(self):
        spin_holder = self.widget("Align_Similarity_Matrix.py", "WORKERS")
        self.enter(spin_holder)
        text = re.sub(r"\s+", " ", plain_text(self.shown()))
        self.assertIn(f"This computer has {self.tools.MAX_CORES} logical CPU", text)
        self.assertIn(f"Range 1 – {self.tools.MAX_CORES}", text)
        self.assertIn("Default 12", text)
        self.assertIn("WORKERS", text)

    def test_the_memory_caution_is_a_warning_and_the_remark_an_info_note(self):
        self.enter(self.widget("Embedding_MSA.py", "WORKERS"))
        marks = [mark_colour(name) for name in images(self.shown())]
        self.assertEqual(marks, [status_color("warning"), status_color("subtle"), status_color("subtle")])
        self.assertIn("about 16 GB for 44,000 sequences", plain_text(self.shown()))

    def test_the_host_cache_tip_says_what_auto_picks_now(self):
        gib = self.tools.GIB
        control = self.widget("Align_Similarity_Matrix.py", "HOST_CACHE_GB")
        with mock.patch.object(self.tools, "resolve_host_cache_bytes", return_value=8 * gib) as budget, \
                mock.patch.object(self.tools, "system_memory_bytes", return_value=(64 * gib, 24 * gib)):
            self.enter(control)
        budget.assert_called_once_with("auto")
        text = re.sub(r"\s+", " ", plain_text(self.shown()))
        self.assertIn("AUTO would use 8.0 GiB now, with 24.0 GiB of memory available.", text)
        self.assertEqual(
            [chip_text for chip_text in chips(self.shown())][:2], ["AUTO ON", "AUTO OFF"],
        )
        self.assertIn(f"Range 0.0 – {self.tools.HOST_CACHE_MAX_GB:.1f} GiB", text)
        self.assertIn("Default auto", text)
        self.assertIn(f"up to {self.tools.HOST_CACHE_MAX_GB:g} GiB", text)

    def test_hovering_a_child_widget_of_a_field_shows_the_same_tip(self):
        control = self.widget("Align_Similarity_Matrix.py", "HOST_CACHE_GB")
        self.enter(control.spinbox)
        self.assertIn("Host Cache", plain_text(self.shown()))
        self.enter(self.widget("Align_Similarity_Matrix.py", "BATCH_SIZE"))
        self.assertNotIn("Host Cache", plain_text(self.shown()))

    def test_native_tooltips_still_route_to_the_panel_and_the_popup_is_suppressed(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QHelpEvent

        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1))
        self.assertTrue(self.window.eventFilter(combo, event))
        self.assertIn("Execution Mode", plain_text(self.shown()))


# ---------------------------------------------------------------------
# The catalogs
# ---------------------------------------------------------------------

CODE_SPAN = re.compile(r"`[^`]*`")


class CatalogTests(unittest.TestCase):
    def test_a_translation_keeps_the_code_spans_of_its_text(self):
        """Backticks mark a typed value (a chip), so a translation keeps each span exactly."""
        tree = ET.parse(LANGUAGES / "zh_CN" / "emapssn_zh_CN.ts")
        checked = 0
        for message in tree.getroot().iter("message"):
            source = message.findtext("source") or ""
            if "`" not in source:
                continue
            translation = message.find("translation")
            if translation is None or translation.get("type") == "obsolete":
                continue
            translated = "".join(translation.itertext()).strip()
            with self.subTest(source=source[:50]):
                self.assertEqual(CODE_SPAN.findall(translated), CODE_SPAN.findall(source))
            checked += 1
        self.assertGreaterEqual(checked, 3)

    def test_the_panels_own_words_are_in_the_catalog(self):
        tree = ET.parse(LANGUAGES / "zh_CN" / "emapssn_zh_CN.ts")
        context = next(c for c in tree.getroot().iter("context") if c.findtext("name") == "TipPanel")
        sources = {m.findtext("source") for m in context.iter("message")}
        self.assertEqual(sources, {
            "(empty)", "on", "off", "Default", "Range", "Setting", "Warning:", "Note:",
            "{option}: {text}", "{label}: {value}.",
        })


if __name__ == "__main__":
    unittest.main()
