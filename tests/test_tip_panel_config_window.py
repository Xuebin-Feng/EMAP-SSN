# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tip panel in the Configuration window (EMAPSSN_Config.ConfigGUI)."""

import os
import pathlib
import re
import tempfile
import unittest
from unittest import mock

from tests.tip_panel_fixtures import (  # noqa: E402
    WithApplication,
    chips,
    images,
    left_press,
    plain_text,
    svg_of,
)
from PySide6.QtCore import QEvent, Qt  # noqa: E402

from desktop.Desktop_App import NAME_ITEM_ROLE  # noqa: E402
from desktop.Studio_Theme import status_color  # noqa: E402
from desktop.Tip_Panel import (  # noqa: E402
    UNBOUNDED,
    FieldTip,
    SpacedTipLabel,
    Tip,
    resolve,
    status_html,
)
from tests.config_gui_loader import load_config_namespace, open_config_window  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
