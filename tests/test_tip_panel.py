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
from PySide6.QtGui import QColor, QFocusEvent, QFont, QTextCharFormat, QTextDocument  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QWidget,
)

from desktop.Desktop_App import MONOSPACE_QSS_FONT_STACK, NAME_ITEM_ROLE, ToggleSwitch  # noqa: E402
from desktop.Studio_Theme import (  # noqa: E402
    HINT_BORDER,
    KEY_CAP_BORDER,
    KEY_CAP_BOTTOM,
    KEY_CAP_PADDING,
    KEY_CAP_RADIUS,
    HINT_PADDING,
    HINT_RIGHT_PADDING,
    HINT_SCROLL_EDGE,
    HINT_TEXT_INDENT,
    TOKENS,
    add_hint_icon,
    status_color,
)
from desktop.Tip_Panel import (  # noqa: E402
    CHIP_GLUE,
    MISSING,
    TIP_PANEL_LINES,
    TIP_PANEL_MINIMUM_LINES,
    UNBOUNDED,
    Code,
    CodeList,
    Field,
    FieldTip,
    LiveNote,
    SpacedTipLabel,
    Tip,
    TipNote,
    TipOption,
    change_signal,
    chip,
    free_bytes,
    is_whole,
    model_html,
    nearest_existing_folder,
    plain,
    resolve,
    rich,
    status_html,
    swatch_mark,
    tip_html,
    tip_plain,
    tooltip_text,
    with_note,
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

    def test_a_list_of_codes_is_several_chips_in_a_row(self):
        names = CodeList(["NVIDIA A100", "<b>CPU</b>"])
        shown = rich("Detected devices: {devices}.", {"devices": names})
        self.assertEqual(chips(shown), ["NVIDIA A100", "<b>CPU</b>"])
        self.assertEqual(plain_text(shown), "Detected devices: NVIDIA A100, <b>CPU</b>.")
        self.assertEqual(plain("Detected devices: {devices}.", {"devices": names}), plain_text(shown))
        # One name is one chip, none is nothing.
        self.assertEqual(chips(rich("{devices}", {"devices": CodeList(["Only"])})), ["Only"])
        self.assertEqual(chips(rich("{devices}", {"devices": CodeList()})), [])

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

    def test_a_default_the_dropdown_does_not_offer_is_left_out_not_shown_as_its_stored_value(self):
        # A dropdown that keeps stored values apart from its labels, with a mode hidden for now.
        combo = combo_of(("Shorter Sequence", "shorter_sequence"), ("Longer Sequence", "longer_sequence"))
        combo.setProperty("persistItemData", True)
        facts = dict(resolve(FieldTip(Tip("x"), Field(widget=combo, default="alignment_length"))).facts)
        self.assertNotIn("Default", facts)
        combo.insertItem(0, "Alignment Length", "alignment_length")
        facts = dict(resolve(FieldTip(Tip("x"), Field(widget=combo, default="alignment_length"))).facts)
        self.assertEqual(facts["Default"], "Alignment Length")
        # A dropdown whose items are its values has no label to hide behind: the value shows.
        plain = QComboBox()
        plain.addItems(["one", "two"])
        facts = dict(resolve(FieldTip(Tip("x"), Field(widget=plain, default="three"))).facts)
        self.assertEqual(facts["Default"], "three")

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

    def test_a_bound_that_means_no_limit_is_left_out_and_the_programs_own_limits_win(self):
        def range_of(field):
            return dict(resolve(FieldTip(Tip("x"), field)).facts).get("Range")

        wide = QSpinBox()
        wide.setRange(-999999, 999999)
        self.assertIsNone(range_of(Field(widget=wide)))
        self.assertEqual(range_of(Field(widget=wide, limits=(0, 10))), "0 – 10")
        # Without a spin box the limits are shown as numbers.
        self.assertEqual(range_of(Field(limits=(0.5, 2))), "0.5 – 2")
        self.assertEqual(range_of(Field(limits=(0, UNBOUNDED - 1))), f"0 – {UNBOUNDED - 1}")
        self.assertIsNone(range_of(Field(limits=(0, UNBOUNDED))))
        self.assertIsNone(range_of(Field(limits=(-UNBOUNDED, 5))))
        # None says the program has no limits, whatever the spin box could take.
        narrow = QSpinBox()
        narrow.setRange(1, 5)
        self.assertEqual(range_of(Field(widget=narrow)), "1 – 5")
        self.assertIsNone(range_of(Field(widget=narrow, limits=None)))

    def test_a_file_dropdown_and_an_unset_dropdown_show_no_default(self):
        def default_of(field):
            return dict(resolve(FieldTip(Tip("x"), field)).facts).get("Default")

        files = combo_of(("a_embeddings.h5", "a_embeddings.h5"))
        folder_and_files = QWidget()
        folder_and_files.combo = files  # a file dropdown with its folder button
        self.assertIsNone(default_of(Field(widget=folder_and_files, default="Sample_[E1_RA]_embeddings.h5")))
        self.assertIsNone(default_of(Field(widget=files, default="")))
        self.assertIsNone(default_of(Field(widget=files, default=None)))
        # A text field's empty default is still a fact, and so is a dropdown's real choice.
        self.assertEqual(default_of(Field(widget=QLineEdit(), default="")), "(empty)")
        self.assertEqual(default_of(Field(widget=files, default="a_embeddings.h5")), "a_embeddings.h5")

    def test_a_switch_names_its_options_by_its_own_words_for_the_states(self):
        tip = Tip("x", options=(TipOption(True, "Picks it."), TipOption(False, "Leaves it.")))
        self.assertEqual(
            resolve(FieldTip(tip, Field(widget=ToggleSwitch("AUTO ON", "AUTO OFF")))).options,
            [("AUTO ON", "Picks it."), ("AUTO OFF", "Leaves it.")],
        )
        # Without a switch a state is named as it is stored.
        self.assertEqual(resolve(FieldTip(tip, Field())).options, [("True", "Picks it."), ("False", "Leaves it.")])
        # A dropdown holding a bool names it as it lists it.
        combo = combo_of(("Yes", True), ("No", False))
        self.assertEqual(
            [label for label, _ in resolve(FieldTip(tip, Field(widget=combo))).options], ["Yes", "No"]
        )

    def test_a_tip_without_a_summary_is_its_title_and_facts(self):
        entry = FieldTip(Tip(""), Field(key="SOME_KEY", label=QLabel("Some Field:"), default=3))
        model = resolve(entry)
        self.assertEqual((model.title, model.summary, model.details, model.options), ("Some Field", "", [], []))
        self.assertEqual(model.facts, [("Default", "3"), ("Setting", "SOME_KEY")])
        shown = tip_html(entry)
        self.assertEqual(re.sub(r"\s+", " ", plain_text(shown)), "Some Field Default 3 Setting SOME_KEY")
        self.assertEqual(chips(shown), ["3", "SOME_KEY"])
        self.assertEqual(tip_plain(entry), "Some Field\nDefault: 3. Setting: SOME_KEY.")

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
        self.assertEqual([note.values for note in resolve(entry).live], [{"value": "grey"}])
        line.setText("red")
        self.assertEqual([note.values for note in resolve(entry).live], [{"value": "red"}])
        self.assertEqual(resolve(FieldTip(Tip("x"), Field())).live, [])
        # A hook may give several notes, or none.
        several = Tip("x", live=lambda field: [LiveNote("one"), None, LiveNote("two")])
        self.assertEqual([note.text for note in resolve(FieldTip(several, Field())).live], ["one", "two"])
        nothing = Tip("x", live=lambda field: None)
        self.assertEqual(resolve(FieldTip(nothing, Field())).live, [])

    def test_a_tip_lists_every_text_it_holds(self):
        tip = Tip(
            "summary", details=("detail",), options=(TipOption("a", "option"),),
            notes=(TipNote("info", "note"),),
        )
        self.assertEqual(tip.texts(), ["summary", "detail", "option", "note"])


class FolderAndNoteHelperTests(WithApplication):
    def test_the_nearest_existing_folder_is_the_folder_itself_or_its_nearest_parent(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.abspath(root)
            os.makedirs(os.path.join(folder, "made"))
            self.assertEqual(nearest_existing_folder(os.path.join(folder, "made")), os.path.join(folder, "made"))
            self.assertEqual(nearest_existing_folder(os.path.join(folder, "not", "yet", "there")), folder)
            # A file is not a folder: the folder it is in.
            file_path = os.path.join(folder, "made", "file.txt")
            pathlib.Path(file_path).write_text("x", encoding="utf-8")
            self.assertEqual(nearest_existing_folder(file_path), os.path.join(folder, "made"))
            # A path that starts with ~ is the user's.
            self.assertEqual(
                nearest_existing_folder("~"), os.path.abspath(os.path.expanduser("~")),
            )

    def test_free_bytes_reads_the_drive_of_a_folder_that_does_not_exist_yet(self):
        with tempfile.TemporaryDirectory() as root:
            free = free_bytes(os.path.join(root, "not", "yet"))
            self.assertIsInstance(free, int)
            self.assertGreater(free, 0)
            with mock.patch("shutil.disk_usage", side_effect=OSError("gone")):
                self.assertIsNone(free_bytes(root))
        with mock.patch("desktop.Tip_Panel.nearest_existing_folder", return_value=None):
            self.assertIsNone(free_bytes("anything"))

    def test_a_note_on_a_state_is_added_to_a_tip_a_field_tip_or_a_plain_text(self):
        base = Tip("Summary.", notes=(TipNote("info", "First."),))
        noted = with_note(base, "warning", "Unavailable now.")
        self.assertEqual(noted.notes, (TipNote("info", "First."), TipNote("warning", "Unavailable now.")))
        self.assertEqual(base.notes, (TipNote("info", "First."),))  # the tip itself is untouched
        field = Field(key="K", label=QLabel("Label:"))
        entry = with_note(FieldTip(base, field), "info", "More.")
        self.assertIsInstance(entry, FieldTip)
        self.assertIs(entry.field, field)
        self.assertEqual([text for _, text in resolve(entry).notes], ["First.", "More."])
        self.assertEqual(with_note("Plain tip.", "info", "More."), "Plain tip.\nMore.")

    def test_the_tooltip_of_an_entry_is_its_plain_text(self):
        entry = FieldTip(Tip("Summary with `chip`."), Field(key="K", label=QLabel("Label:")))
        self.assertEqual(tooltip_text(entry), tip_plain(entry))
        self.assertIn("Summary with chip.", tooltip_text(entry))
        self.assertNotIn("`", tooltip_text(entry))
        self.assertEqual(tooltip_text(Tip("x")), tip_plain(Tip("x")))
        self.assertEqual(tooltip_text("As it was."), "As it was.")


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
        # The text wraps at the panel's width, whatever the header: it never needs a horizontal bar.
        self.assertFalse(panel.horizontalScrollBar().isVisible())
        self.assertLessEqual(panel.document().idealWidth(), panel.viewport().width() + 1)
        panel.close()


# ---------------------------------------------------------------------
# The panel itself
# ---------------------------------------------------------------------

def wheel(widget, delta=-120):
    """A real wheel event over the middle of widget (negative scrolls down)."""
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent

    middle = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(
        middle, widget.mapToGlobal(middle), QPoint(0, 0), QPoint(0, delta),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(widget, event)


def long_tip(sentences=12, key="LONG"):
    """A tip too long for the panel's ordinary height, and a short one's field."""
    return FieldTip(
        Tip(
            "A summary that goes on for a good while, so that it wraps. " * 2,
            details=tuple(f"Detail line number {number}, long enough to wrap in a narrow panel." for number in range(sentences)),
        ),
        Field(key=key, label=QLabel("Long field:")),
    )


class PanelTests(WithApplication):
    def panel(self, width=500, hint=False):
        panel = SpacedTipLabel("start")
        if hint:
            add_hint_icon(panel)  # the card the windows put it in
        panel.resize(width, panel.preferred_height())
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

    def test_the_panel_keeps_one_height_whatever_it_shows_and_however_long_it_is(self):
        entry, _ = full_entry()
        panel = self.panel(hint=True)
        height = panel.height()
        held = (panel.minimumHeight(), panel.minimumSizeHint(), panel.sizeHint())
        shows = (
            lambda: panel.setTip(entry),
            lambda: panel.setTip(long_tip()),
            lambda: panel.setTip(long_tip(sentences=60)),
            lambda: panel.setTip("plain\ntip"),
            lambda: panel.setStatus("danger", "Failed.\n" * 30),
            lambda: panel.setReport(status_html("success", "row") * 40, "success"),
            lambda: panel.setText("a plain message\n" * 20),
            lambda: panel.setTip(entry),
        )
        for show in shows:
            show()
            self.app.processEvents()
            with self.subTest(text=panel.text()[:60]):
                self.assertEqual(panel.height(), height)
                self.assertEqual((panel.minimumHeight(), panel.minimumSizeHint(), panel.sizeHint()), held)

    def test_the_height_is_so_many_lines_of_the_panels_own_font(self):
        panel = self.panel(hint=True)
        line = panel.line_height()
        chrome = 2 * (HINT_BORDER + HINT_PADDING)
        self.assertEqual(panel.preferred_height(), round(TIP_PANEL_LINES * line) + chrome)
        self.assertEqual(panel.sizeHint().height(), panel.preferred_height())
        self.assertEqual(panel.minimumSizeHint().height(), panel.preferred_height(TIP_PANEL_MINIMUM_LINES))
        # A line is what the font's lines take in the panel: two lines of text step by that much.
        panel.setTip("one\ntwo")
        doc_height = panel.document().size().height() - 2 * panel.document().documentMargin()
        self.assertAlmostEqual(doc_height, 2 * line, delta=2)
        # The size of the font sets the height, and a font with taller lines for the text does too.
        bigger = QFont(panel.font())
        bigger.setPointSizeF(panel.font().pointSizeF() * 1.5)
        larger = SpacedTipLabel("start")
        larger.setFont(bigger)
        self.assertGreater(larger.preferred_height(), panel.preferred_height() * 1.3)
        self.assertGreater(larger.line_height(), line)
        chinese = SpacedTipLabel("点击某个输入项或其标签，即可在此查看提示。")
        self.assertGreaterEqual(chinese.line_height(), 1)
        # A typical tip fits: the panel shows a tip of seven lines whole.
        self.assertGreaterEqual(TIP_PANEL_LINES, 7)

    def test_a_tip_that_does_not_fit_scrolls_by_a_vertical_bar_that_shows_only_while_it_is_needed(self):
        entry = FieldTip(Tip("A short tip."), Field(key="S", label=QLabel("Short field:")))
        panel = self.panel(width=560, hint=True)
        panel.setTip(entry)
        self.app.processEvents()
        bar = panel.verticalScrollBar()
        wide_text = panel.viewport().width()
        self.assertFalse(bar.isVisible())
        self.assertEqual(bar.maximum(), 0)
        panel.setTip(long_tip())
        self.app.processEvents()
        self.assertTrue(bar.isVisible())
        self.assertGreater(bar.maximum(), 0)
        # The text wraps at what is left of the width: nothing is cut off under the bar.
        self.assertLess(panel.viewport().width(), wide_text)
        self.assertAlmostEqual(panel.document().textWidth(), panel.viewport().width(), delta=1)
        self.assertLessEqual(panel.document().size().height(), panel.verticalScrollBar().maximum() + panel.viewport().height() + 1)
        for tip in (entry, long_tip(), "plain"):
            panel.setTip(tip)
            self.app.processEvents()
            self.assertFalse(panel.horizontalScrollBar().isVisible())
        panel.setTip("short")
        self.app.processEvents()
        self.assertFalse(bar.isVisible())
        self.assertEqual(panel.viewport().width(), wide_text)

    def test_the_bar_appearing_does_not_set_the_text_relaying_in_a_loop(self):
        from PySide6.QtCore import QEvent, QObject

        class Resizes(QObject):
            count = 0

            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Resize:
                    self.count += 1
                return False

        panel = self.panel(width=560, hint=True)
        resizes = Resizes(panel)
        panel.viewport().installEventFilter(resizes)
        counts = []
        for _ in range(3):
            for tip in (long_tip(), "short", long_tip(sentences=3)):
                resizes.count = 0
                panel.setTip(tip)
                self.app.processEvents()
                counts.append(resizes.count)
        # The viewport narrows once when the bar comes and widens once when it goes.
        self.assertLessEqual(max(counts), 2, counts)

    def test_each_new_tip_or_message_starts_at_the_top_and_the_wheel_scrolls(self):
        panel = self.panel(width=560, hint=True)
        bar = panel.verticalScrollBar()
        panel.setTip(long_tip())
        self.app.processEvents()
        self.assertEqual(bar.value(), 0)
        wheel(panel.viewport())
        self.assertGreater(bar.value(), 0)
        # Hovering the same field again leaves the tip where the user scrolled it...
        place = bar.value()
        panel.setTip(long_tip())
        self.assertEqual(bar.value(), place)
        # ...a different tip, a status, a report and plain text start at the top.
        shows = (
            lambda: panel.setTip(long_tip(sentences=13, key="OTHER")),
            lambda: panel.setStatus("warning", "Careful.\n" * 25),
            lambda: panel.setReport(status_html("danger", "row") * 40, "danger"),
            lambda: panel.setText("a plain message\n" * 20),
        )
        for show in shows:
            panel.setTip(long_tip(sentences=20, key="MORE"))
            bar.setValue(bar.maximum())
            show()
            self.app.processEvents()
            with self.subTest(text=panel.text()[:40]):
                self.assertEqual(bar.value(), 0)
                self.assertGreater(bar.maximum(), 0)
        # A wheel over text that fits is left for whatever is behind the panel.
        panel.setTip("short")
        self.app.processEvents()
        before = bar.value()
        wheel(panel.viewport())
        self.assertEqual(bar.value(), before)

    def test_a_live_refresh_keeps_the_place_the_user_scrolled_to(self):
        line = QLineEdit("red")
        entry = FieldTip(
            Tip("x", details=tuple(f"Detail line {number} that goes on for a good while." for number in range(14)),
                live=lambda field: LiveNote("Now", swatch=field.widget.text())),
            Field(widget=line),
        )
        panel = self.panel(width=400, hint=True)
        panel.setTip(entry)
        self.app.processEvents()
        panel.verticalScrollBar().setValue(30)
        line.setText("#336699")
        self.assertTrue(panel.refresh())
        self.assertEqual(panel.verticalScrollBar().value(), 30)

    def test_the_bar_is_the_themes_scroll_bar_inside_the_card(self):
        panel = self.panel(width=560, hint=True)
        panel.setTip(long_tip())
        self.app.processEvents()
        bar = panel.verticalScrollBar()
        self.assertTrue(bar.isVisible())
        # The theme's bar (10 px, at the card's right edge, in the padding), not a native one.
        self.assertEqual(bar.width(), 10)
        bar_right = bar.mapTo(panel, bar.rect().topRight()).x()
        self.assertLessEqual(panel.width() - 1 - bar_right, HINT_BORDER + HINT_SCROLL_EDGE + 2)
        # The text starts where the card's text always did, the document's margin included.
        self.assertEqual(
            panel.viewport().geometry().left() + panel.document().documentMargin(), HINT_BORDER + HINT_TEXT_INDENT
        )
        self.assertEqual(panel.property("role"), "hint")
        # The text keeps clear of the bar by the card's right padding.
        gap = bar.mapTo(panel, bar.rect().topLeft()).x() - panel.viewport().geometry().right() - 1
        self.assertEqual(gap + panel.document().documentMargin(), HINT_RIGHT_PADDING - HINT_SCROLL_EDGE)

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
        with mock.patch.object(QTextEdit, "setHtml") as set_html:
            panel.setTip(entry)
            panel.setTip(entry)
        set_html.assert_not_called()

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

    def test_the_panel_knows_whether_it_holds_a_tip_or_a_message(self):
        entry, _ = full_entry()
        panel = self.panel()
        self.assertTrue(panel.shows_tip())  # its placeholder is a tip's place
        messages = (
            lambda: panel.setStatus("danger", "Failed."),
            lambda: panel.setStatus("info", "Working..."),
            lambda: panel.setReport("<p>Report</p>", "success"),
            lambda: panel.setText("Plain message"),
        )
        for show in messages:
            for tip in (entry, "a plain tip"):
                with self.subTest(tip=type(tip).__name__):
                    panel.setTip(tip)
                    self.assertTrue(panel.shows_tip())
                    show()
                    self.assertFalse(panel.shows_tip())
        # A live refresh of the tip keeps it a tip; it never touches a message.
        panel.setTip(entry)
        self.assertTrue(panel.refresh())
        self.assertTrue(panel.shows_tip())
        panel.setStatus("success", "Done.")
        self.assertFalse(panel.refresh())
        self.assertFalse(panel.shows_tip())
        # The same text again does not turn a message into a tip, or a tip into a message.
        panel.setTip("same")
        panel.setTip("same")
        self.assertTrue(panel.shows_tip())
        panel.setText("same")
        self.assertFalse(panel.shows_tip())


# ---------------------------------------------------------------------
# Code chips: rounded, drawn behind the text
# ---------------------------------------------------------------------

def blocks(document):
    block = document.begin()
    while block.isValid():
        yield block
        block = block.next()


def every_chip_entry():
    """A tip with a code chip in every place the panel puts one."""
    return FieldTip(
        Tip(
            "A summary with `one chip`.",
            details=("A detail with `another` chip.",),
            options=(TipOption("auto", "Does `a thing`."), TipOption("scalar", "Does it alone.")),
            notes=(TipNote("warning", "Careful with `this`."),),
            live=lambda field: LiveNote("Right now {value}", {"value": Code("42")}),
        ),
        Field(key="THE_KEY", label=QLabel("The Field:"), default="auto"),
    )


class ChipTests(WithApplication):
    def panel(self, width=560, height=400):
        panel = SpacedTipLabel("start")
        add_hint_icon(panel)
        panel.resize(width, height)
        panel.show()
        self.addCleanup(panel.close)
        self.app.processEvents()
        return panel

    @staticmethod
    def square_fragments(panel):
        """The text fragments of the panel's document that have the chip ground as a background."""
        found = []
        block = panel.document().begin()
        while block.isValid():
            for piece in block.textFormats():
                brush = piece.format.background()
                if brush.style() != Qt.BrushStyle.NoBrush and brush.color().name() == TOKENS["surface"]:
                    found.append(block.text()[piece.start:piece.start + piece.length])
            block = block.next()
        return found

    def test_a_chip_is_drawn_for_each_place_the_panel_puts_one_and_the_document_draws_no_square(self):
        entry = every_chip_entry()
        panel = self.panel()
        panel.setTip(entry)
        self.app.processEvents()
        # Summary, detail, an option, its text, a note, the live note, the default and the key.
        expected = chips(tip_html(entry))
        self.assertEqual(len(expected), 9, expected)
        self.assertEqual(len(panel.chip_rects()), len(expected))
        self.assertEqual(self.square_fragments(panel), [])
        # The text of the chips is still there, as text.
        shown = panel.toPlainText()
        for name in ("one chip", "another", "a thing", "this", "42", "THE_KEY"):
            self.assertIn(name, shown)
        # What the panel reads back is still the rich text with the chip's ground in it.
        self.assertEqual(chips(panel.text()), expected)

    def test_a_chip_has_rounded_corners_and_only_the_corners_differ_from_a_square(self):
        panel = self.panel()
        panel.setTip(every_chip_entry())
        self.app.processEvents()
        with mock.patch("desktop.Tip_Panel.KEY_CAP_RADIUS", 0):
            square = panel.grab().toImage()
        rounded = panel.grab().toImage()
        offset = panel.viewport().mapTo(panel, panel.viewport().rect().topLeft())
        rects = [rect.translated(offset.x(), offset.y()) for rect, _ in panel.chip_rects()]
        differing = {
            (x, y) for y in range(square.height()) for x in range(square.width())
            if max(abs(square.pixelColor(x, y).red() - rounded.pixelColor(x, y).red()),
                   abs(square.pixelColor(x, y).green() - rounded.pixelColor(x, y).green()),
                   abs(square.pixelColor(x, y).blue() - rounded.pixelColor(x, y).blue())) > 2
        }
        self.assertTrue(differing)
        tint = QColor(TOKENS["tint"])
        reach = KEY_CAP_RADIUS + KEY_CAP_BOTTOM + 1

        def near_corner(x, y, rect):
            return any(
                abs(x + 0.5 - corner_x) <= reach and abs(y + 0.5 - corner_y) <= reach
                for corner_x in (rect.left(), rect.right()) for corner_y in (rect.top(), rect.bottom())
            )

        def distance(image, x, y):
            colour = image.pixelColor(x, y)
            return abs(colour.red() - tint.red()) + abs(colour.green() - tint.green()) + abs(colour.blue() - tint.blue())

        # Every difference is at a corner of a key cap...
        for x, y in differing:
            self.assertTrue(any(near_corner(x, y, rect) for rect in rects), (x, y))
        # ...and where a corner's outermost pixel differs, the rounded one is the card's, not the cap's.
        nearer = 0
        for rect in rects:
            for x, y in ((int(rect.left()), int(rect.top())), (int(rect.right()) - 1, int(rect.top())),
                         (int(rect.left()), int(rect.bottom()) - 1), (int(rect.right()) - 1, int(rect.bottom()) - 1)):
                self.assertLessEqual(distance(rounded, x, y), distance(square, x, y), (x, y))
                nearer += distance(rounded, x, y) < distance(square, x, y)
        self.assertGreaterEqual(nearer, 2 * len(rects))
        # No chip is cut off at the viewport's edge, the first column of a table at the left edge included.
        for rect in panel.chip_rects():
            self.assertGreaterEqual(rect[0].left(), 0)
            self.assertLessEqual(rect[0].right(), panel.viewport().width())
        # Every chip has all four corners rounded: differences on both sides, above and below its middle.
        for rect in rects:
            middle_x, middle_y = rect.center().x(), rect.center().y()
            nearby = [(x, y) for x, y in differing if near_corner(x, y, rect)]
            for left in (True, False):
                for top in (True, False):
                    self.assertTrue(
                        any((x < middle_x) == left and (y < middle_y) == top for x, y in nearby), (rect, left, top)
                    )

    def test_a_chip_is_the_viewers_key_cap_drawn_from_the_same_definition(self):
        import desktop.Studio_Theme as theme
        import desktop.Tip_Panel as module

        self.assertEqual((KEY_CAP_RADIUS, KEY_CAP_PADDING, KEY_CAP_BORDER, KEY_CAP_BOTTOM), (5, 6, 1, 2))
        for name in ("KEY_CAP_RADIUS", "KEY_CAP_PADDING", "KEY_CAP_BORDER", "KEY_CAP_BOTTOM"):
            self.assertEqual(getattr(module, name), getattr(theme, name), name)
        panel = self.panel()
        panel.setTip(every_chip_entry())
        self.app.processEvents()
        fills = {fill.name() for _rect, fill in panel.chip_rects()}
        self.assertEqual(fills, {theme.key_cap_colors()[0]})
        # Its look is a white box with a thin border and a thicker bottom edge, as the card's caps.
        fill, border, bottom = theme.key_cap_colors()
        rect, _ = panel.chip_rects()[0]
        image = panel.grab().toImage()
        offset = panel.viewport().mapTo(panel, panel.viewport().rect().topLeft())
        def near(pixel, name):
            wanted = QColor(name)
            return all(abs(a - b) <= 10 for a, b in ((pixel.red(), wanted.red()), (pixel.green(), wanted.green()),
                                                     (pixel.blue(), wanted.blue())))

        # Below the baseline, where no glyph reaches, the edges and the fill are all there is.
        row = int(offset.y() + rect.bottom()) - KEY_CAP_BOTTOM - 3
        self.assertTrue(near(image.pixelColor(int(offset.x() + rect.left() + 0.5), row), border))  # the border, 1 px
        self.assertTrue(near(image.pixelColor(int(offset.x() + rect.left()) + 1 + 2, row), fill))  # the fill inside
        centre_x = int(offset.x() + rect.left() + rect.width() / 2)
        self.assertTrue(near(image.pixelColor(centre_x, int(offset.y() + rect.bottom()) - 1), bottom))  # the bottom edge

    def test_a_chip_takes_its_own_room_and_covers_no_neighbouring_glyph(self):
        entry = FieldTip(
            Tip("Full header, leading identifier (e.g. `WP_0123.1`), or wildcard (e.g. `WP_01*`) naming it.", facts=False),
            Field(key="K", label=QLabel("L:")),
        )
        panel = self.panel(width=620)
        panel.setTip(entry)
        self.app.processEvents()
        document, padding = panel.document(), KEY_CAP_PADDING + KEY_CAP_BORDER
        block = next(block for block in blocks(document) if "WP_01*" in block.text())
        text = block.text()
        rects = [rect for rect, _ in panel.chip_rects()]
        self.assertEqual(len(rects), 2)
        for chip_text, rect in zip(("WP_0123.1", "WP_01*"), rects):
            first = text.index(chip_text)
            last = first + len(chip_text)
            # A joiner on each side, spaced out by the cap's padding and border: room of its own.
            self.assertEqual((text[first - 1], text[last]), (CHIP_GLUE, CHIP_GLUE))
            line = block.layout().lineForTextPosition(first)
            origin = document.documentLayout().blockBoundingRect(block).topLeft().x()
            before, text_start = line.cursorToX(first - 1)[0], line.cursorToX(first)[0]
            text_end, after = line.cursorToX(last)[0], line.cursorToX(last + 1)[0]
            self.assertAlmostEqual(text_start - before, padding, delta=0.6)
            self.assertAlmostEqual(after - text_end, padding, delta=0.6)
            # The box is exactly that room and the text; the next glyph starts where it ends.
            self.assertAlmostEqual(rect.left(), origin + before, delta=0.6)
            self.assertAlmostEqual(rect.right(), origin + after, delta=0.6)
            self.assertEqual(text[last + 1], ")")
            self.assertEqual(text[first - 2], " ")
        # In the pixels: right of the last chip's box, where the ")" is, there is none of the chip's ground.
        image = panel.grab().toImage()
        offset = panel.viewport().mapTo(panel, panel.viewport().rect().topLeft())
        rect = rects[1]
        ground = QColor(TOKENS["surface"]).name()
        for x in range(int(offset.x() + rect.right()) + 1, int(offset.x() + rect.right()) + 6):
            for y in range(int(offset.y() + rect.top()) + 2, int(offset.y() + rect.bottom()) - 3):
                self.assertNotEqual(image.pixelColor(x, y).name(), ground, (x, y))

    def test_a_chip_does_not_make_its_line_taller(self):
        panel = self.panel(width=700)
        label = QLabel("L:")

        def height(summary):
            panel.setTip(FieldTip(Tip(summary, facts=False), Field(key="K", label=label)))
            self.app.processEvents()
            return panel.document().size().height()

        plain_height = height("A summary with a word and another one.")
        chipped = height("A summary with `a word` and `another one`.")
        self.assertAlmostEqual(chipped, plain_height, delta=0.5)
        # So the panel's line count measures a tip with chips as it does one without.
        footer = FieldTip(Tip("A summary."), Field(key="THE_KEY", label=label, default="auto"))
        panel.setTip(footer)
        self.app.processEvents()
        with_footer = panel.document().size().height()
        panel.setTip(FieldTip(Tip("A summary.", facts=False), Field(key="THE_KEY", label=label)))
        self.app.processEvents()
        self.assertGreater(with_footer, panel.document().size().height())
        self.assertLessEqual(with_footer - panel.document().size().height(), 2 * panel.line_height())

    def test_a_short_chip_never_breaks_and_stays_with_the_punctuation_after_it(self):
        entry = FieldTip(
            Tip("Names a header (e.g. `WP_01*`) and a wildcard (see `WP_0123.1`), as it says here at length.", facts=False),
            Field(key="K", label=QLabel("L:")),
        )
        for width in range(260, 620, 20):  # a chip wider than the panel itself can only break
            with self.subTest(width=width):
                panel = self.panel(width=width)
                panel.setTip(entry)
                self.app.processEvents()
                block = next(block for block in blocks(panel.document()) if "WP_01*" in block.text())
                text = block.text()
                for chip_text, after in (("WP_01*", ")"), ("WP_0123.1", ")")):
                    first = text.index(chip_text)
                    last = first + len(chip_text)
                    layout = block.layout()
                    line_of = lambda position: layout.lineForTextPosition(position).lineNumber()
                    # Joiner, text, joiner and the closing bracket are on one line, and the chip has one box.
                    self.assertEqual({line_of(position) for position in range(first - 1, last + 2)}, {line_of(first)})
                    self.assertEqual(text[last + 1], after)
                self.assertEqual(len(panel.chip_rects()), 2)

    def test_a_chip_that_wraps_has_a_key_cap_on_each_of_its_lines(self):
        token = "sid|18367|pid|10722|hfid|10|sfid|2|gi|914684625|pdb|3WWH(A)|taxonID|1042534|Arthrobactersp.KNK168"
        panel = self.panel(width=300)
        panel.setTip(FieldTip(Tip(f"Names `{token}` here."), Field(key="K", label=QLabel("L:"))))
        self.app.processEvents()
        rects = [rect for rect, _ in panel.chip_rects()]
        self.assertGreaterEqual(len(rects), 2)
        self.assertEqual(len({round(rect.top()) for rect in rects}), len(rects))  # one per line
        for rect in rects:
            self.assertGreaterEqual(rect.left(), 0)
            self.assertLessEqual(rect.right(), panel.viewport().width())

    def test_a_chip_scrolls_with_its_text(self):
        entry = FieldTip(
            Tip("A summary.", details=tuple(f"Detail line number {number} goes on." for number in range(14))
                + ("The last line, with `lastchip`.",)),
            Field(key="K", label=QLabel("L:")),
        )
        panel = self.panel(height=SpacedTipLabel("start").preferred_height(5))  # shorter than the tip
        panel.setTip(entry)
        self.app.processEvents()
        bar = panel.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        bar.setValue(bar.maximum())
        self.app.processEvents()
        rect, colour = panel.chip_rects()[-1]
        image = panel.grab().toImage()
        offset = panel.viewport().mapTo(panel, panel.viewport().rect().topLeft())
        x = int(offset.x() + rect.left() + KEY_CAP_BORDER + KEY_CAP_PADDING / 2)  # inside the fill, left of the first glyph
        y = int(offset.y() + rect.bottom() - KEY_CAP_BOTTOM - 3 - bar.value())  # below the baseline, clear of any glyph
        self.assertEqual(image.pixelColor(x, y).name(), colour.name())
        # At the top, the same place is the card's.
        bar.setValue(0)
        self.app.processEvents()
        self.assertEqual(panel.grab().toImage().pixelColor(x, y).name(), TOKENS["tint"])

    def test_a_chip_takes_the_theme_colour_in_force_when_the_tip_is_shown(self):
        panel = self.panel()
        with mock.patch.dict(TOKENS, {"surface": "#123456"}):
            panel.setTip(every_chip_entry())
            self.app.processEvents()
            self.assertEqual({colour.name() for _, colour in panel.chip_rects()}, {"#123456"})
            self.assertEqual(self.square_fragments(panel), [])
        panel.setTip(every_chip_entry())
        self.assertEqual({colour.name() for _, colour in panel.chip_rects()}, {TOKENS["surface"]})

    def test_a_message_and_plain_text_have_no_chips(self):
        panel = self.panel()
        panel.setTip(every_chip_entry())
        for show in (lambda: panel.setStatus("info", "m `x`"), lambda: panel.setText("`x`"),
                     lambda: panel.setReport("<p>r</p>")):
            show()
            self.assertEqual(panel.chip_rects(), [])


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

    def test_every_tip_is_laid_out_and_a_native_tooltip_still_shows_as_plain_text(self):
        for key, tip in self.window.tip_db_keys.items():
            self.assertIsInstance(tip, Tip, key)
        # A native tooltip (the Clear button's) goes to the panel as plain text, exactly as before.
        button = self.window.btn_clear_top_edge
        self.window.eventFilter(button, left_press())
        self.assertEqual(self.panel.text(), SpacedTipLabel(button.toolTip()).text())

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

    def test_every_colour_field_shows_a_swatch_of_the_colour_typed_and_none_for_a_blank_field(self):
        keys = ("TEXT_COLOR", "INITIAL_NODE_COLOR", "HOVER_COLOR", "CONNECTED_NODE_COLOR",
                "EDGE_COLOR", "NODE_BOUNDARY_COLOR")
        for key in keys:
            with self.subTest(key=key):
                self.assertIsNotNone(self.window.tip_db_keys[key].live, key)
                self.window.inputs[key].setText("#123456")
                self.click(self.window.labels[key])
                self.assertTrue(
                    any('fill="#123456"' in svg_of(name) for name in images(self.panel.text())), key
                )
                self.assertIn("#123456", chips(self.panel.text()))
                self.assertIn("Current color", plain_text(self.panel.text()))
                self.window.inputs[key].setText("")
                self.assertFalse(
                    any('fill="#123456"' in svg_of(name) for name in images(self.panel.text())), key
                )
                self.assertNotIn("Current color", plain_text(self.panel.text()))

    def test_a_directory_field_says_where_it_resolves_and_the_export_folder_how_much_room_is_free(self):
        gib = 1024 ** 3
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        base = os.path.join(home.name, "inputs")
        self.window.inputs["INPUT_FILE_DIR"].setText(base)
        self.window.inputs["FASTA_DIR"].setText("$input_file$/fasta")
        self.window.inputs["SETTING_EXPORT_DIR"].setText(os.path.join(home.name, "exports"))

        def resolved(key):
            self.click(self.window.labels[key])
            text = re.sub(r"\s+", " ", plain_text(self.panel.text()))
            shown = chips(self.panel.text())
            return text, shown

        with mock.patch.dict(self.namespace, {"free_bytes": lambda path: 12.5 * gib}):
            text, shown = resolved("FASTA_DIR")
            self.assertIn("Resolves to", text)
            self.assertIn(os.path.normcase(os.path.join(base, "fasta")),
                          [os.path.normcase(os.path.normpath(chip_text)) for chip_text in shown])
            self.assertNotIn("free on its drive", text)  # only the folders that get large files say so
            text, shown = resolved("SETTING_EXPORT_DIR")
            self.assertIn("Resolves to", text)
            self.assertIn("12.5 GiB free on its drive.", text)
            self.assertIn("12.5 GiB", shown)
            # A drive that can't be read says nothing about room.
            with mock.patch.dict(self.namespace, {"free_bytes": lambda path: None}):
                text, _ = resolved("SETTING_EXPORT_DIR")
                self.assertIn("Resolves to", text)
                self.assertNotIn("free on its drive", text)
        # An empty field resolves to nothing, so it says nothing.
        self.window.inputs["FASTA_DIR"].setText("")
        text, _ = resolved("FASTA_DIR")
        self.assertNotIn("Resolves to", text)

    def test_the_resolved_folder_follows_what_is_typed_while_the_tip_is_on_display(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        field = self.window.inputs["METADATA_DIR"]
        self.click(self.window.labels["METADATA_DIR"])
        first = os.path.join(home.name, "first")
        field.setText(first)
        self.assertIn(os.path.normcase(first),
                      [os.path.normcase(os.path.normpath(chip_text)) for chip_text in chips(self.panel.text())])
        second = os.path.join(home.name, "second")
        field.setText(second)
        shown = [os.path.normcase(os.path.normpath(chip_text)) for chip_text in chips(self.panel.text())]
        self.assertIn(os.path.normcase(second), shown)
        self.assertNotIn(os.path.normcase(first), shown)

    def test_the_layout_device_tip_lists_the_devices_found_by_name(self):
        combo = self.window.inputs["LAYOUT_DEVICE_SELECTION"]
        for index in reversed(range(combo.count())):
            combo.removeItem(index)
        combo.addItem("Automatic", "auto")
        self.click(self.window.labels["LAYOUT_DEVICE_SELECTION"])
        # Only items that show a device's own name count: no device, no note.
        self.assertNotIn("Detected devices", plain_text(self.panel.text()))
        for name, value in (("NVIDIA A100-SXM4-40GB", "cuda:0"), ("AMD Radeon RX 7900 XTX", "cuda:1")):
            combo.addItem(name, value)
            combo.setItemData(combo.count() - 1, True, NAME_ITEM_ROLE)
        self.click(self.window.labels["LAYOUT_DEVICE_SELECTION"])
        self.assertIn("Detected devices: NVIDIA A100-SXM4-40GB, AMD Radeon RX 7900 XTX",
                      plain_text(self.panel.text()))
        self.assertEqual(
            [name for name in chips(self.panel.text()) if "NVIDIA" in name or "AMD" in name],
            ["NVIDIA A100-SXM4-40GB", "AMD Radeon RX 7900 XTX"],
        )

    def test_a_range_the_program_gives_replaces_a_spin_box_that_takes_any_number(self):
        spin_box = self.window.inputs["ALIGNMENT_OFFSET"]
        entry = self.window.tip_db[spin_box]
        self.assertEqual(entry.field.limits, self.namespace["PROFILE_RANGES"].get("ALIGNMENT_OFFSET"))
        for key, (low, high) in self.namespace["PROFILE_RANGES"].items():
            widget = self.window.inputs.get(key)
            if widget is None or key not in self.window.tip_db_keys:
                continue
            with self.subTest(key=key):
                shown = dict(resolve(self.window.tip_db[widget]).facts).get("Range")
                if abs(low) >= UNBOUNDED or abs(high) >= UNBOUNDED:
                    self.assertIsNone(shown)
                else:
                    self.assertIsNotNone(shown)

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

    def test_the_panel_itself_says_whether_a_tip_or_a_message_is_on_display(self):
        self.assertTrue(self.panel.shows_tip())  # the placeholder
        self.click(self.window.labels["ALIGNMENT_REFERENCE"])
        self.assertTrue(self.panel.shows_tip())
        self.hover(self.window.labels["TEXT_COLOR"])
        self.assertTrue(self.panel.shows_tip())
        self.panel.setStatus("info", "msg")
        self.assertFalse(self.panel.shows_tip())
        self.hover(self.window.labels["TEXT_COLOR"])
        self.assertFalse(self.panel.shows_tip())
        self.click(self.window.labels["TEXT_COLOR"])
        self.assertTrue(self.panel.shows_tip())

    def test_a_tip_that_a_live_note_redraws_is_still_a_tip_that_hovering_replaces(self):
        # Typing in a field with a live note lays its tip out again; that is no status message.
        line = self.window.inputs["TEXT_COLOR"]
        self.click(self.window.labels["TEXT_COLOR"])
        for text in ("#336699", "#cc0000", ""):
            line.setText(text)
            self.assertTrue(self.panel.shows_tip(), text)
        self.hover(self.window.labels["ALIGNMENT_OFFSET"])
        self.assertIn("Integer offset", plain_text(self.panel.text()))

    def real_hover(self, widget):
        """Rest the real pointer on widget, and wait until Qt itself sends it the ToolTip event.

        Returns whether the event came, so a test that sees the panel not follow knows the
        window's filter had the event to act on (sendEvent returns True even when the filter
        lets an event through, so it can't tell).
        """
        from PySide6.QtCore import QObject, QPoint

        class Spy(QObject):
            arrived = False

            def eventFilter(spy, watched, event):
                if watched is widget and event.type() == QEvent.Type.ToolTip:
                    spy.arrived = True
                return False

        from PySide6.QtTest import QTest

        spy = Spy()
        self.app.installEventFilter(spy)
        try:
            self.window.activateWindow()
            QTest.mouseMove(self.window, QPoint(2, 2))
            QTest.qWait(30)
            QTest.mouseMove(widget, QPoint(5, 5))
            QTest.qWait(30)
            QTest.mouseMove(widget, QPoint(7, 6))
            waited = 0
            while not spy.arrived and waited < 3000:
                QTest.qWait(50)
                waited += 50
            self.app.processEvents()
        finally:
            self.app.removeEventFilter(spy)
        return spy.arrived

    def real_click(self, widget, button=Qt.MouseButton.LeftButton):
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        QTest.mouseClick(widget, button, pos=QPoint(5, 5))
        self.app.processEvents()

    def test_the_real_pointer_resting_on_fields_shows_their_tips_until_a_message_and_a_left_click(self):
        # The path a user takes: Qt's own ToolTip event after its delay, then a status message,
        # then a left click. Config and VR Config share the rule; Tools has none by design.
        first, second = self.window.labels["NODE_FASTA_FILE"], self.window.labels["MSA_FILE"]
        self.window.tabs.setCurrentIndex(0)
        self.assertTrue(first.isVisible() and second.isVisible())

        def title():
            return plain_text(self.panel.text()).split("\n")[0].strip()

        self.assertTrue(self.panel.shows_tip())  # the placeholder
        self.assertTrue(self.real_hover(first))
        self.assertEqual(title(), "Sequence Set / Subset (.fasta)")
        self.assertTrue(self.real_hover(second))
        self.assertEqual(title(), "MSA Input (.fasta / _sparse.h5)")
        self.assertTrue(self.real_hover(first))
        self.assertEqual(title(), "Sequence Set / Subset (.fasta)")
        # A status message: hovering leaves it where it is...
        self.panel.setStatus("success", "Settings saved.")
        message = self.panel.text()
        self.assertTrue(self.real_hover(second))
        self.assertEqual(self.panel.text(), message)
        self.assertTrue(self.real_hover(first))
        self.assertEqual(self.panel.text(), message)
        # (a right click counts as hovering, and so does Qt moving the focus by itself)
        self.real_click(first, Qt.MouseButton.RightButton)
        self.assertEqual(self.panel.text(), message)
        # ...a left click on an element with a tip shows that tip and gives hovering back...
        self.real_click(first)
        self.assertEqual(title(), "Sequence Set / Subset (.fasta)")
        self.assertTrue(self.real_hover(second))
        self.assertEqual(title(), "MSA Input (.fasta / _sparse.h5)")
        # ...and a report is a message too.
        self.panel.setReport("<p>Report</p>", "warning")
        report = self.panel.text()
        self.assertTrue(self.real_hover(first))
        self.assertEqual(self.panel.text(), report)
        self.real_click(second)
        self.assertEqual(title(), "MSA Input (.fasta / _sparse.h5)")

    def test_the_tip_area_is_laid_out_at_its_fixed_height_and_keeps_it_whatever_a_tip_or_message_holds(self):
        window, panel = self.window, self.panel
        # The window as it first appears: the panel is so many lines of its own font tall.
        self.assertEqual(panel.height(), panel.preferred_height())
        for width in (self.window.minimumSizeHint().width(), 1000, 1400):
            with self.subTest(width=width):
                window.resize(width, 650)
                self.app.processEvents()
                # (The action buttons under it wrap onto another row in a narrow window, as they
                # always did, and the panel takes up the difference; its content never does.)
                held = (panel.height(), window.left_split.sizes(), window.tabs.height(),
                        window.minimumSizeHint(), window.size())
                shows = [lambda tip=tip: panel.setTip(tip) for tip in
                         (window._field_tips.get(key, entry) for key, entry in window.tip_db_keys.items())]
                shows += [
                    lambda: panel.setStatus("danger", "Failed.\n" * 30),
                    lambda: panel.setReport(status_html("success", "row") * 40, "success"),
                    lambda: panel.setText("Plain message"),
                ]
                for show in shows:
                    show()
                    self.app.processEvents()
                    self.assertEqual(
                        (panel.height(), window.left_split.sizes(), window.tabs.height(),
                         window.minimumSizeHint(), window.size()), held,
                    )

    def test_a_tip_scrolls_exactly_when_it_does_not_fit_and_the_longest_ones_do(self):
        window, panel = self.window, self.panel
        window.resize(1000, 650)
        self.app.processEvents()
        bar = panel.verticalScrollBar()
        heights, viewports = {}, {}
        for key, entry in window.tip_db_keys.items():
            panel.setTip(window._field_tips.get(key, entry))
            self.app.processEvents()
            needed, available = panel.document().size().height(), panel.viewport().height()
            self.assertEqual(bar.isVisible(), needed > available, (key, needed, available))
            self.assertFalse(panel.horizontalScrollBar().isVisible(), key)
            heights[key], viewports[key] = needed, available
        # Which tip is the tallest depends on the font the run measures with (the Layout
        # Method dropdown's tip and Fewer Edges Drawn While Moving's are about the same
        # height), so none is named: the tallest scrolls and the shortest fits, and the
        # tips span the viewport, so that both halves of the rule above were exercised.
        longest = max(heights, key=heights.get)
        shortest = min(heights, key=heights.get)
        self.assertGreater(heights[longest], viewports[longest], "no tip is tall enough to scroll")
        self.assertLess(heights[shortest], viewports[shortest], "no tip is short enough to fit")
        panel.setTip(window._field_tips.get(longest, window.tip_db_keys[longest]))
        self.app.processEvents()
        self.assertTrue(bar.isVisible(), longest)
        panel.setTip(window._field_tips.get(shortest, window.tip_db_keys[shortest]))
        self.app.processEvents()
        self.assertFalse(bar.isVisible(), shortest)

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

    def test_every_tool_tip_is_laid_out(self):
        for script, tips in self.window.SCRIPT_TIPS.items():
            for key, tip in tips.items():
                self.assertIsInstance(tip, Tip, (script, key))
        # ...and every field's entry holds the field it describes, whether or not it has a tip of its own.
        for path, data in self.window.script_data.items():
            for variable, info in data["inputs"].items():
                entry = self.window.tip_db[info["widget"]]
                self.assertIsInstance(entry, FieldTip, (path, variable))
                self.assertEqual(entry.field.key, variable)

    def test_a_field_gets_its_tip_with_its_field_and_a_plain_tooltip(self):
        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        entry = self.window.tip_db[combo]
        self.assertIsInstance(entry, FieldTip)
        self.assertEqual((entry.field.key, entry.field.default, entry.field.widget), ("EXECUTION_MODE", "auto", combo))
        self.assertEqual(entry.field.title(), "Execution Mode")
        # The widget's accessible description is the tip as plain text.
        self.assertEqual(combo.toolTip(), tip_plain(entry))
        self.assertIn("Scalar: Processes one pairwise score matrix at a time.", combo.toolTip())
        batch = self.widget("Align_Similarity_Matrix.py", "BATCH_SIZE")
        self.assertEqual(batch.toolTip(), tip_plain(self.window.tip_db[batch]))
        self.assertIn("Warning: Larger values improve throughput but require more RAM.", batch.toolTip())

    def test_the_options_follow_the_dropdown(self):
        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        def listed():
            return [combo.itemText(combo.findData(value)) for value in ("auto", "scalar", "tiled")
                    if combo.findData(value) >= 0]

        # The device decides whether "tiled" is offered: try both.
        if combo.findData("tiled") < 0:
            combo.addItem("Tiled", "tiled")
        self.enter(combo)
        # The options as the dropdown names them, then the facts: the default and the key.
        self.assertEqual(chips(self.shown()), listed() + ["Auto", "EXECUTION_MODE"])
        self.assertIn("Tiled", listed())
        self.assertIn("memory-bounded embedding tiles", plain_text(self.shown()))
        combo.removeItem(combo.findData("tiled"))
        self.enter(combo)
        self.assertEqual(chips(self.shown()), listed() + ["Auto", "EXECUTION_MODE"])
        self.assertNotIn("Tiled", listed())
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

    WORKER_FIELDS = (
        ("Align_Similarity_Matrix.py", "WORKERS"),
        ("Align_Substitution_Matrix.py", "NUM_THREADS"),
        ("Embedding_MSA.py", "WORKERS"),
        ("Network_Injection.py", "WORKERS"),
        ("Embedding_SSEARCH.py", "WORKERS"),
    )
    DEVICE_FIELDS = (
        "Generate_Embeddings.py", "Align_Similarity_Matrix.py", "Embedding_MSA.py",
        "Network_Injection.py", "Embedding_SSEARCH.py",
    )

    def test_every_cpu_worker_field_says_how_many_logical_cpus_this_computer_has(self):
        for script, variable in self.WORKER_FIELDS:
            for cores in (self.tools.MAX_CORES, 3, 1):
                with self.subTest(script=script, variable=variable, cores=cores), \
                        mock.patch.object(self.tools, "MAX_CORES", cores):
                    self.enter(self.widget(script, variable))
                    text = re.sub(r"\s+", " ", plain_text(self.shown()))
                    self.assertIn(f"This computer has {cores} logical CPU", text)
                    # The count is worked out when the tip is shown, not when the window opened.
                    self.assertEqual(
                        text.count("This computer has"), 1,
                    )

    def test_the_host_cache_tip_of_both_tools_that_have_one_says_what_auto_picks_now(self):
        gib = self.tools.GIB
        for script in ("Align_Similarity_Matrix.py", "Network_Injection.py"):
            for budget, available in ((8, 24), (2, 5)):
                with self.subTest(script=script, budget=budget), \
                        mock.patch.object(self.tools, "resolve_host_cache_bytes", return_value=budget * gib), \
                        mock.patch.object(self.tools, "system_memory_bytes", return_value=(64 * gib, available * gib)):
                    self.enter(self.widget(script, "HOST_CACHE_GB"))
                    text = re.sub(r"\s+", " ", plain_text(self.shown()))
                    self.assertIn(
                        f"AUTO would use {budget:.1f} GiB now, with {available:.1f} GiB of memory available.", text
                    )
                    self.assertIn(f"{budget:.1f} GiB", chips(self.shown()))
                    self.assertIn("HOST_CACHE_GB", text)

    def test_every_device_field_lists_the_devices_found_by_name_and_says_nothing_without_one(self):
        for script in self.DEVICE_FIELDS:
            with self.subTest(script=script):
                combo = self.widget(script, "DEVICE_SELECTION")
                for index in reversed(range(combo.count())):
                    if combo.itemData(index, NAME_ITEM_ROLE):
                        combo.removeItem(index)
                self.enter(combo)
                self.assertNotIn("Detected devices", plain_text(self.shown()))
                for name, value in (("NVIDIA A100-SXM4-40GB", "cuda:0"), ("Apple M3 Max", "mps")):
                    combo.addItem(name, value)
                    combo.setItemData(combo.count() - 1, True, NAME_ITEM_ROLE)
                self.enter(combo)
                self.assertIn("Detected devices: NVIDIA A100-SXM4-40GB, Apple M3 Max", plain_text(self.shown()))
                self.assertEqual(
                    [name for name in chips(self.shown()) if name in ("NVIDIA A100-SXM4-40GB", "Apple M3 Max")],
                    ["NVIDIA A100-SXM4-40GB", "Apple M3 Max"],
                )

    def test_a_directory_tip_says_where_the_folder_is_and_two_of_them_how_much_room_is_free(self):
        gib = self.tools.GIB
        project = pathlib.Path(self.tools._PROJECT_ROOT)
        absolute = os.path.join(tempfile.gettempdir(), "tip_panel_probe")
        for key, typed, folder, with_room in (
            ("FASTA_DIR", "Input_Files/fasta", str(project / "Input_Files" / "fasta"), False),
            ("MSA_DIR", absolute, absolute, False),
            ("EMBED_DIR", absolute, absolute, False),
            ("NETWORK_DIR", absolute, absolute, False),
            ("REPORT_DIR", absolute, absolute, True),
            ("SETTING_EXPORT_DIR", "Exports", str(project / "Exports"), True),
        ):
            with self.subTest(key=key), mock.patch.object(self.tools, "free_bytes", return_value=7.5 * gib) as free:
                field = self.window.dir_inputs[key]
                field.setText(typed)
                self.enter(field)
                text = re.sub(r"\s+", " ", plain_text(self.shown()))
                self.assertIn("Resolves to", text)
                self.assertIn(
                    os.path.normcase(os.path.normpath(folder)),
                    [os.path.normcase(os.path.normpath(shown)) for shown in chips(self.shown())],
                )
                self.assertEqual("7.5 GiB free on its drive." in text, with_room)
                self.assertEqual(free.called, with_room)
                self.assertIn(key, text)
        # An empty field resolves to nothing: no note, and no disk is asked.
        with mock.patch.object(self.tools, "free_bytes") as free:
            self.window.dir_inputs["REPORT_DIR"].setText("   ")
            self.enter(self.window.dir_inputs["REPORT_DIR"])
            text = plain_text(self.shown())
            self.assertNotIn("Resolves to", text)
            self.assertNotIn("free on its drive", text)
            free.assert_not_called()
        # A drive that can't be read says nothing about room, but still names the folder.
        with mock.patch.object(self.tools, "free_bytes", return_value=None):
            self.window.dir_inputs["REPORT_DIR"].setText(absolute)
            self.enter(self.window.dir_inputs["REPORT_DIR"])
            text = plain_text(self.shown())
            self.assertIn("Resolves to", text)
            self.assertNotIn("free on its drive", text)

    def test_the_directory_tips_show_the_labels_title_and_the_key_and_the_default(self):
        for key, field in self.window.dir_inputs.items():
            with self.subTest(key=key):
                entry = self.window.tip_db[field]
                self.assertIsInstance(entry, FieldTip)
                self.assertEqual(entry.field.key, key)
                self.assertEqual(self.window.tip_db[field.parentWidget()], entry)
                self.assertEqual(field.parentWidget().toolTip(), tip_plain(entry))
                self.enter(field)
                text = re.sub(r"\s+", " ", plain_text(self.shown()))
                self.assertTrue(text.startswith(entry.field.title()), text)
                self.assertIn("Default", text)
                self.assertIn(key, chips(self.shown()))

    def test_a_tool_setting_without_a_tip_of_its_own_still_shows_its_title_default_and_key(self):
        entry = self.tools.tool_tip_entry(None, "SOME_SETTING", QLabel("Some Setting:"), QSpinBox(), 4)
        self.assertIsInstance(entry, FieldTip)
        self.assertEqual(entry.tip.summary, "")
        self.assertEqual(
            re.sub(r"\s+", " ", plain_text(tip_html(entry))), "Some Setting Default 4 Range 0 – 99 Setting SOME_SETTING"
        )
        # A plain-text tip stays as it was.
        self.assertEqual(self.tools.tool_tip_entry("Plain.", "K", QLabel("L:"), QSpinBox()), "Plain.")

    def test_a_message_about_a_fields_state_is_a_note_on_its_tip_and_goes_with_the_state(self):
        from types import SimpleNamespace

        from Cache_Manifest import NetworkCompletenessInfo

        incomplete = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
        )
        complete = NetworkCompletenessInfo(
            status="complete", sequence_count=10, edge_count=45, expected_edge_count=45,
        )
        network = self.widget("Embedding_MSA.py", "INPUT_NETWORK").combo
        switch = self.widget("Embedding_MSA.py", "SHOW_REGRESSION_PLOT")
        network_type = SimpleNamespace(network_type="embedding")
        with mock.patch.object(self.tools, "validate_network_schema", return_value=network_type), \
                mock.patch.object(self.tools, "inspect_network_completeness", return_value=incomplete) as inspect:
            network.addItem("sparse.h5")
            network.setCurrentText("sparse.h5")
            self.assertTrue(switch.isEnabled())
            self.enter(switch)
            text = re.sub(r"\s+", " ", plain_text(self.shown()))
            self.assertIn("Incomplete network: 10 sequences and 20/45 observed pairs", text)
            # The state's note joins the tip's own text; the tooltip is that tip as plain text.
            self.assertIn("Displays a diagnostic scatter plot for sparse networks.", text)
            self.assertEqual(switch.toolTip(), tip_plain(self.window.tip_db[switch]))
            self.assertEqual(switch.property("registeredTooltip"), switch.toolTip())
            self.assertEqual(
                [mark_colour(name) for name in images(self.shown())][-1], status_color("subtle"),
            )
            inspect.return_value = complete
            network.addItem("complete.h5")
            network.setCurrentText("complete.h5")
            self.assertFalse(switch.isEnabled())
            self.enter(switch)
            text = re.sub(r"\s+", " ", plain_text(self.shown()))
            self.assertIn("Complete network: 10 sequences", text)
            self.assertNotIn("Incomplete network", text)  # one state at a time, not a pile of notes
            self.assertEqual(text.count("Complete network"), 1)
            self.assertEqual(
                [mark_colour(name) for name in images(self.shown())][-1], status_color("warning"),
            )
        # A network that can't be read: a warning, and the base tip is what it was.
        with mock.patch.object(self.tools, "validate_network_schema", side_effect=ValueError("not a network")):
            network.addItem("broken.h5")
            network.setCurrentText("broken.h5")
            self.enter(switch)
            text = re.sub(r"\s+", " ", plain_text(self.shown()))
            self.assertIn("Unable to determine network type: not a network", text)
            self.assertEqual(text.count("Unable to determine network type"), 1)
        for combo in (self.widget("Embedding_MSA.py", "ALIGNMENT_SCORE"), self.widget("Embedding_MSA.py", "NORMALIZATION_MODE")):
            self.enter(combo)
            self.assertIn("Unable to determine network type", plain_text(self.shown()))
        # Back to a network that can be read, the earlier states' notes are gone, not kept under it.
        with mock.patch.object(self.tools, "validate_network_schema", return_value=network_type), \
                mock.patch.object(self.tools, "inspect_network_completeness", return_value=incomplete):
            network.setCurrentText("sparse.h5")
            self.enter(switch)
            text = re.sub(r"\s+", " ", plain_text(self.shown()))
            self.assertIn("Incomplete network", text)
            for message in ("Unable to determine", "Complete network:"):
                self.assertNotIn(message, text)

    def test_a_tooltip_changed_after_the_tip_was_registered_shows_instead_of_the_tip(self):
        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        self.assertEqual(combo.property("registeredTooltip"), tip_plain(self.window.tip_db[combo]))
        self.enter(combo)
        self.assertIn("Execution Mode", plain_text(self.shown()))
        combo.setToolTip("Something is wrong with this device.")
        self.enter(combo)
        self.assertEqual(self.shown(), SpacedTipLabel("Something is wrong with this device.").text())
        # Cleared (as a widget with no message has it), the field's own tip shows again.
        combo.setToolTip("")
        self.enter(combo)
        self.assertIn("Execution Mode", plain_text(self.shown()))
        # Back to what was registered: the same.
        combo.setToolTip(combo.property("registeredTooltip"))
        self.enter(combo)
        self.assertIn("Execution Mode", plain_text(self.shown()))

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

    def test_the_tip_area_keeps_its_height_whatever_a_tip_holds_and_the_window_around_it_stays_put(self):
        window, panel = self.window, self.panel
        window.resize(1000, 700)
        window.show()
        self.app.processEvents()
        self.assertEqual(panel.height(), panel.preferred_height())
        held = (panel.height(), window.left_split.sizes(), window.tabs.height(), window.minimumSizeHint())
        seen = {}
        for widget, entry in window.tip_db.items():
            if isinstance(entry, FieldTip):
                seen.setdefault(id(entry), entry)
        self.assertGreater(len(seen), 50)
        shows = [lambda entry=entry: panel.setTip(entry) for entry in seen.values()]
        shows += [lambda: panel.setStatus("danger", "Failed.\n" * 30), lambda: panel.setText("Plain\n" * 30)]
        scrolled = 0
        for show in shows:
            show()
            self.app.processEvents()
            needed, available = panel.document().size().height(), panel.viewport().height()
            self.assertEqual(panel.verticalScrollBar().isVisible(), needed > available)
            scrolled += panel.verticalScrollBar().isVisible()
            self.assertFalse(panel.horizontalScrollBar().isVisible())
            self.assertEqual(
                (panel.height(), window.left_split.sizes(), window.tabs.height(), window.minimumSizeHint()), held,
            )
        # What does not fit scrolls (the longest are an options table with notes), and not everything does.
        self.assertGreater(scrolled, 0)
        self.assertLess(scrolled, len(shows))
        # Tools hovers by Enter and has no lock: a message gives way to the next field at once.
        panel.setStatus("info", "message")
        self.enter(self.widget("Align_Similarity_Matrix.py", "BATCH_SIZE"))
        self.assertIn("Batch Size", plain_text(self.shown()))


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
