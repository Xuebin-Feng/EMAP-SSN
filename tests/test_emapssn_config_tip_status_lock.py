# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration window tip-panel status-lock behavior."""

import pathlib
import tempfile
import unittest
from unittest import mock

import h5py

import tests.config_cache_fixtures  # noqa: F401 - configure SRC path and offscreen Qt


class TipPanelStatusLockTests(unittest.TestCase):
    """Hovering never replaces a status message in the tip panel.

    The panel shows both tips and status messages, such as a consistency-check
    result. Hovering (Qt's ToolTip event, after its wake-up delay) shows tips
    only while the panel holds a tip. A left click, or focus by Tab, Shift+Tab
    or a shortcut, on an element with a tip always shows it, and then hovering
    works again. Focus that Qt moves by itself and right clicks count as hovering.
    """

    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401 - load torch before PySide6
        from PySide6.QtWidgets import QApplication
        from tests.config_gui_loader import load_config_namespace

        cls.namespace = load_config_namespace()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tests.config_gui_loader import open_config_window

        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.root = pathlib.Path(home.name)
        self.window = open_config_window(self.namespace["ConfigGUI"], self.root)
        self.addCleanup(self._close_window)
        self.window.show()
        self.app.processEvents()
        self.labels = self.window.labels
        self.tips = self.window.tip_db_keys

    def _close_window(self):
        self.window.close()
        self.app.processEvents()

    def panel(self):
        from PySide6.QtGui import QTextDocumentFragment
        text = QTextDocumentFragment.fromHtml(self.window.tip_panel.text()).toPlainText()
        # A status message's icon is an image: one object-replacement character.
        return text.replace("\ufffc", "").strip()

    def assert_panel_shows_tip(self, key):
        """The panel shows the tip of key: a plain tip as it is, a rich tip with every text it holds."""
        from desktop.Tip_Panel import Tip, plain

        tip = self.tips[key]
        if not isinstance(tip, Tip):
            self.assertEqual(self.panel(), tip)
            return
        shown = self.panel()

        def squashed(text):
            # A chip pads its text with spaces, which are not in the tip's own text.
            return "".join(text.split())

        for text in tip.texts():
            self.assertIn(squashed(plain(text)), squashed(shown))
        # The title is the field's own label, without its colon.
        self.assertTrue(shown.startswith(self.labels[key].text().rstrip(":")), shown)

    def hover(self, widget):
        from PySide6.QtCore import QEvent, QPoint
        from PySide6.QtGui import QHelpEvent
        from PySide6.QtWidgets import QToolTip
        self.app.sendEvent(widget, QEvent(QEvent.Type.Enter))
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), widget.mapToGlobal(QPoint(1, 1)))
        self.app.sendEvent(widget, event)
        # Tips go to the panel, never to Qt's popup, even while hovering
        # leaves the panel alone (the Clear button has a tooltip of its own).
        visible = QToolTip.isVisible()
        QToolTip.hideText()
        self.assertFalse(visible)

    def click(self, widget, button=None):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        QTest.mouseClick(widget, button or Qt.MouseButton.LeftButton)

    def focus(self, widget, reason):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QFocusEvent
        self.app.sendEvent(widget, QFocusEvent(QEvent.Type.FocusIn, reason))

    def run_consistency_check(self):
        """Run the real check on a FASTA that the network covers."""
        window = self.window
        work = self.root / "work"
        work.mkdir()
        (work / "nodes.fasta").write_text(">a\nMKTAYIAK\n>b\nMSEQNNTE\n", encoding="utf-8")
        with h5py.File(work / "network.h5", "w") as network:
            network.create_dataset("headers", data=[b"a", b"b"])
        for combo, name in ((window.cb_fasta, "nodes.fasta"), (window.cb_hdf5, "network.h5")):
            combo.blockSignals(True)
            combo.addItem(name)
            combo.setCurrentText(name)
            combo.blockSignals(False)
        window.cb_msa.blockSignals(True)
        window.cb_msa.clear()
        window.cb_msa.blockSignals(False)
        with mock.patch.object(window, "_resolved_directory_input", return_value=str(work)):
            window.run_consistency_check()
        result = self.panel()
        # The report is a table of rich text, not a plain message.
        self.assertIn("FASTA vs HDF5", result)
        self.assertIn("strict subset", result)
        self.assertIn("<table", self.window.tip_panel.text())
        return result

    def save(self):
        with mock.patch.object(
            self.window,
            "_prepare_profile_writes",
            return_value=([], dict(self.window._custom_settings), [], []),
        ):
            self.window.save_only()
        result = self.panel()
        self.assertTrue(result.startswith("Settings saved successfully."), result)
        return result

    def fail_statistics(self):
        from tests.config_gui_loader import finish_statistics

        with mock.patch("h5py.File", side_effect=OSError("unreadable network")):
            self.window.run_statistics()
            finish_statistics(self.window)
        result = self.panel()
        self.assertEqual(
            result, "Error during network statistics calculation: unreadable network"
        )
        return result

    def test_hovering_shows_tips_while_the_panel_holds_a_tip(self):
        from PySide6.QtCore import QEvent

        placeholder = "Click or tab to an input or its label to see helpful tips here."
        self.assertEqual(self.panel(), placeholder)
        # Unlike Tools, Config waits for Qt's tooltip delay: entering is not enough.
        self.app.sendEvent(self.labels["ALIGNMENT_REFERENCE"], QEvent(QEvent.Type.Enter))
        self.assertEqual(self.panel(), placeholder)
        self.hover(self.labels["ALIGNMENT_REFERENCE"])
        self.assert_panel_shows_tip("ALIGNMENT_REFERENCE")
        self.hover(self.labels["ALIGNMENT_OFFSET"])
        self.assert_panel_shows_tip("ALIGNMENT_OFFSET")
        self.hover(self.window.btn_clear_top_edge)
        # The Clear button's own tooltip replaces the offset tip in the panel.
        clear_tip = self.window.btn_clear_top_edge.toolTip()
        self.assertTrue(clear_tip)
        self.assertEqual(self.panel(), clear_tip)

    def test_a_consistency_check_result_stays_until_a_left_click_on_a_tip(self):
        from PySide6.QtCore import Qt

        result = self.run_consistency_check()
        reference_label = self.labels["ALIGNMENT_REFERENCE"]
        self.hover(reference_label)
        self.hover(self.window.line_ref)
        self.hover(self.window.btn_clear_top_edge)
        self.click(reference_label, Qt.MouseButton.RightButton)
        for reason in (
            Qt.FocusReason.ActiveWindowFocusReason,
            Qt.FocusReason.PopupFocusReason,
            Qt.FocusReason.MouseFocusReason,
            Qt.FocusReason.OtherFocusReason,
        ):
            self.focus(self.window.line_ref, reason)
        self.assertEqual(self.panel(), result)

        self.click(reference_label)
        self.assert_panel_shows_tip("ALIGNMENT_REFERENCE")
        self.hover(self.labels["ALIGNMENT_OFFSET"])
        self.assert_panel_shows_tip("ALIGNMENT_OFFSET")

    def test_tab_shift_tab_and_shortcut_focus_show_the_tip_and_reenable_hovering(self):
        from PySide6.QtCore import Qt

        for reason in (
            Qt.FocusReason.TabFocusReason,
            Qt.FocusReason.BacktabFocusReason,
            Qt.FocusReason.ShortcutFocusReason,
        ):
            with self.subTest(reason=reason.name):
                self.save()
                self.focus(self.window.line_ref, reason)
                self.assert_panel_shows_tip("ALIGNMENT_REFERENCE")
                self.hover(self.labels["ALIGNMENT_OFFSET"])
                self.assert_panel_shows_tip("ALIGNMENT_OFFSET")

    def test_statistics_and_save_messages_also_stay_while_hovering(self):
        for show_status in (self.fail_statistics, self.save):
            with self.subTest(show_status.__name__):
                result = show_status()
                self.hover(self.labels["ALIGNMENT_OFFSET"])
                self.hover(self.window.btn_clear_top_edge)
                self.assertEqual(self.panel(), result)
                self.click(self.labels["ALIGNMENT_REFERENCE"])
                self.assert_panel_shows_tip("ALIGNMENT_REFERENCE")


if __name__ == "__main__":
    unittest.main()
