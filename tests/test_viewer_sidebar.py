# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's sidebar and slider overlay, as its main window builds them.

The sidebar is a card with round corners (desktop.Viewer_Widgets) whose
buttons centre their labels after their icons. In a shown window it slides
open and closed, its toggle, the slider and the HUD following it at each
frame; a hidden or headless window switches it at once, and the toggle turns
a slide around where it is. The info button at the threshold slider's start
opens the mouse and keyboard controls.
"""

import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import EMAPSSN_Viewer  # noqa: E402
from EMAPSSN_Viewer import SIDEBAR_SLIDE_MS, MainViewer, _sidebar_room  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from desktop.Viewer_Widgets import SIDEBAR_INSET, ControlsButton, SidebarButton, SidebarPanel  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

CANVAS_SIZE = (1200, 800)


class SidebarSlideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make_viewer(self, shown=True):
        """A Viewer's main window with two sidebar buttons, the sidebar closed."""
        viewer = MainViewer.__new__(MainViewer)
        viewer.canvas = SimpleNamespace(native=QWidget(), size=CANVAS_SIZE)
        viewer._build_main_window()
        self.addCleanup(viewer.main_window.deleteLater)
        self.addCleanup(viewer.main_window.close)
        viewer.hud_updates = []
        viewer._update_hud_elements = lambda: viewer.hud_updates.append(_sidebar_room(viewer))
        viewer.add_sidebar_button("metaDataBtn", "Meta Data", lambda: None, icon="chart-column")
        viewer.add_sidebar_button("agentBtn", "Agent", lambda: None, icon="bot")
        viewer.set_sidebar_visible(False, animate=False)
        if shown:
            viewer.main_window.resize(*CANVAS_SIZE)
            viewer.main_window.show()
            self.app.processEvents()
        viewer.hud_updates.clear()
        return viewer

    def wait_for_slide(self, viewer, until=None, timeout=3.0):
        """Run the event loop until the slide ends, or until until() holds; returns the widths seen."""
        seen = []
        deadline = time.monotonic() + timeout
        while viewer._sidebar_slide_width is not None and time.monotonic() < deadline:
            seen.append(viewer._sidebar_slide_width)
            if until is not None and until():
                return seen
            self.app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(viewer._sidebar_slide_width, "the slide never ended")
        return seen

    def assert_placed(self, viewer, room):
        width, height = CANVAS_SIZE
        panel, toggle = viewer.right_panel.geometry(), viewer.toggle_sidebar_btn.geometry()
        self.assertEqual((panel.x(), panel.width(), panel.height()), (width - room, viewer._panel_w, height))
        self.assertEqual((toggle.x(), toggle.y()), (width - room - 40, 10))

    def test_the_sidebar_is_a_round_card_of_centred_buttons(self):
        viewer = self.make_viewer(shown=False)
        self.assertIsInstance(viewer.right_panel, SidebarPanel)
        margins = viewer.right_panel_layout.contentsMargins()
        self.assertEqual(margins.right() - margins.left(), SIDEBAR_INSET)
        self.assertTrue(all(isinstance(button, SidebarButton) for button in viewer.sidebar_buttons.values()))
        # The panel's own room holds its buttons and the card's inset.
        button = viewer.sidebar_buttons["agentBtn"]
        self.assertEqual(viewer._panel_w, button.width() + margins.left() + margins.right())

    def test_a_hidden_or_headless_window_switches_the_sidebar_at_once(self):
        viewer = self.make_viewer(shown=False)
        viewer.set_sidebar_visible(True)
        self.assertIsNone(viewer._sidebar_slide_width)
        self.assertFalse(viewer.right_panel.isHidden())
        viewer.headless = True
        viewer.main_window.show()
        self.app.processEvents()
        viewer.set_sidebar_visible(False)
        self.assertIsNone(viewer._sidebar_slide_width)
        self.assertTrue(viewer.right_panel.isHidden())
        self.assert_placed(viewer, 0)

    def test_low_resource_mode_switches_the_sidebar_at_once(self):
        import EMAPSSN_Config as cfg

        viewer = self.make_viewer()
        self.assertTrue(viewer._sidebar_can_slide())
        with mock.patch.object(cfg, "LOW_RESOURCE_MODE", True):
            self.assertFalse(viewer._sidebar_can_slide())
            viewer.set_sidebar_visible(True)
            self.assertIsNone(viewer._sidebar_slide_width)
            self.assertFalse(viewer.right_panel.isHidden())
            self.assert_placed(viewer, viewer._panel_w)
            viewer.toggle_sidebar()
            self.assertIsNone(viewer._sidebar_slide_width)
            self.assertTrue(viewer.right_panel.isHidden())
            self.assert_placed(viewer, 0)

    def test_opening_slides_the_sidebar_in_and_the_hud_follows(self):
        viewer = self.make_viewer()
        started = time.monotonic()
        viewer.set_sidebar_visible(True)
        # Shown from the first frame, still clear of the canvas.
        self.assertTrue(viewer.right_panel.isVisible())
        self.assertEqual(viewer._sidebar_slide_width, 0.0)
        seen = self.wait_for_slide(viewer)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, SIDEBAR_SLIDE_MS / 1000 * 0.8)
        self.assertTrue(any(0 < width < viewer._panel_w for width in seen), seen)
        self.assertEqual(seen, sorted(seen))
        self.assertTrue(viewer.right_panel.isVisible())
        self.assert_placed(viewer, viewer._panel_w)
        # The HUD moved with it, frame by frame, and ended at the open width.
        self.assertGreater(len(viewer.hud_updates), 2)
        self.assertTrue(any(0 < room < viewer._panel_w for room in viewer.hud_updates), viewer.hud_updates)
        self.assertEqual(viewer.hud_updates[-1], viewer._panel_w)

    def test_closing_slides_the_sidebar_out_then_hides_it(self):
        viewer = self.make_viewer()
        viewer.set_sidebar_visible(True, animate=False)
        viewer.set_sidebar_visible(False)
        self.assertTrue(viewer.right_panel.isVisible())
        seen = self.wait_for_slide(viewer)
        self.assertEqual(seen, sorted(seen, reverse=True))
        self.assertFalse(viewer.right_panel.isVisible())
        self.assert_placed(viewer, 0)

    def test_the_toggle_turns_a_slide_around_where_it_is(self):
        viewer = self.make_viewer()
        viewer.toggle_sidebar()
        self.wait_for_slide(viewer, until=lambda: viewer._sidebar_slide_width > viewer._panel_w / 4)
        midway = viewer._sidebar_slide_width
        self.assertIsNotNone(midway)
        # Part of the panel shows: its left edge, and the toggle, are where the slide has got to.
        self.assertEqual(viewer.right_panel.x(), round(CANVAS_SIZE[0] - midway))
        self.assertEqual(viewer.toggle_sidebar_btn.x(), round(CANVAS_SIZE[0] - midway) - 40)
        # Still opening, so the toggle closes it from there.
        viewer.toggle_sidebar()
        self.assertFalse(viewer._sidebar_open)
        self.assertEqual(viewer._sidebar_slide_width, midway)
        self.assertLess(viewer._sidebar_animation.duration(), SIDEBAR_SLIDE_MS)
        seen = self.wait_for_slide(viewer)
        self.assertTrue(all(width <= midway for width in seen), seen)
        self.assertFalse(viewer.right_panel.isVisible())
        self.assert_placed(viewer, 0)

    def test_the_toggle_reopens_a_sidebar_while_it_closes(self):
        viewer = self.make_viewer()
        viewer.set_sidebar_visible(True, animate=False)
        viewer.toggle_sidebar()
        self.assertFalse(viewer._sidebar_open)
        self.wait_for_slide(viewer, until=lambda: viewer._sidebar_slide_width < viewer._panel_w * 3 / 4)
        # Still shown while it closes, yet heading closed: the toggle opens it again from there.
        self.assertTrue(viewer.right_panel.isVisible())
        viewer.toggle_sidebar()
        self.assertTrue(viewer._sidebar_open)
        self.wait_for_slide(viewer)
        self.assertTrue(viewer.right_panel.isVisible())
        self.assert_placed(viewer, viewer._panel_w)

    def test_a_viewer_without_sidebar_buttons_hides_the_sidebar_and_its_toggle(self):
        viewer = self.make_viewer()
        viewer.sidebar_buttons = {}
        viewer.set_sidebar_visible(True)
        self.assertIsNone(viewer._sidebar_slide_width)
        self.assertFalse(viewer.right_panel.isVisible())
        self.assertFalse(viewer.toggle_sidebar_btn.isVisible())


class SliderOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make_viewer(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.canvas = SimpleNamespace(native=QWidget(), size=CANVAS_SIZE)
        self.addCleanup(viewer.canvas.native.deleteLater)
        viewer.hud_layout = {"status_x_offset": 10.0, "status_line_spacing": 38.0}
        viewer._hud_timer = mock.Mock()
        viewer.current_slider_threshold = 3.74
        with mock.patch.object(MainViewer, "on_slider_value_changed"):
            viewer._build_slider_overlay()
        return viewer

    def test_the_view_width_box_ends_the_slider_row(self):
        viewer = self.make_viewer()
        for room in (0, 160):
            with self.subTest(sidebar=room), mock.patch.object(EMAPSSN_Viewer, "_sidebar_room", return_value=room):
                viewer.position_slider_overlay()
                box, row = viewer.view_width_box.geometry(), viewer.slider_overlay.geometry()
                # At the right, 10 px in from the sidebar's room, as the status texts above it.
                self.assertEqual(box.x() + box.width(), CANVAS_SIZE[0] - room - 10)
                # Level with the slider: the same middle, to the pixel.
                self.assertAlmostEqual(box.y() + box.height() / 2, row.y() + row.height() / 2, delta=0.5)
                # The slider ends a clear gap before it.
                self.assertGreaterEqual(box.x() - (row.x() + row.width()), 16)
                # Hidden Nodes and the metadata property stack above it, the
                # box's middle and theirs evenly spaced.
                room_size = (CANVAS_SIZE[0] - room, CANVAS_SIZE[1])
                hidden = viewer._status_hud_position(0, room_size)[1]
                meta = viewer._status_hud_position(1, room_size)[1]
                self.assertAlmostEqual((box.y() + box.height() / 2) - hidden, hidden - meta, delta=0.5)
                self.assertGreaterEqual(hidden - meta, 30)
        self.assertEqual(viewer.view_width_box.before.text(), "View Width:")
        self.assertFalse(viewer.view_width_box.after.isVisibleTo(viewer.view_width_box))

    def test_the_view_width_box_zooms_as_zoom_does(self):
        viewer = self.make_viewer()
        viewer.canvas.native.setFocus = mock.Mock()
        spin_box = viewer.view_width_box.spin_box
        spin_box.show_width(242.0, 0.5, 1100.0)
        with mock.patch("commands.zoom.set_view_width") as set_view_width:
            spin_box.setValue(500.0)
        set_view_width.assert_called_once_with(viewer, 500.0)
        viewer._hud_timer.start.assert_called()
        # A width zoom refuses shows its error on the console line.
        viewer._show_click_status = mock.Mock()
        error = ValueError(EMAPSSN_Viewer.Message("Error: Zoom width is too large for the current view."))
        with mock.patch("commands.zoom.set_view_width", side_effect=error), mock.patch("builtins.print"):
            spin_box.setValue(600.0)
        viewer._show_click_status.assert_called_once_with(error.args[0])

    def test_low_resource_mode_redraws_the_edges_when_the_slider_is_let_go(self):
        import EMAPSSN_Config as cfg

        for low_resource in (False, True):
            viewer = MainViewer.__new__(MainViewer)
            viewer.canvas = SimpleNamespace(native=QWidget(), size=CANVAS_SIZE)
            self.addCleanup(viewer.canvas.native.deleteLater)
            viewer.hud_layout = {"status_x_offset": 10.0, "status_line_spacing": 38.0}
            viewer._hud_timer = mock.Mock()
            viewer.min_threshold, viewer.max_threshold = 0.0, 100.0
            viewer.current_slider_threshold = 0.0
            viewer.update_edges = mock.Mock()
            viewer._build_slider_overlay()
            slider = viewer.slider
            with self.subTest(low_resource=low_resource), \
                    mock.patch.object(cfg, "LOW_RESOURCE_MODE", low_resource):
                # A drag: the number follows it; the edges follow it, or wait for the release.
                slider.setSliderDown(True)
                for value in (100, 200, 300):
                    slider.setValue(value)
                self.assertEqual(viewer.slider_label.text(), viewer.format_sig_figs(30.0))
                self.assertEqual(viewer.update_edges.call_count, 0 if low_resource else 3)
                slider.setSliderDown(False)
                self.assertEqual(viewer.update_edges.call_count, 1 if low_resource else 3)
                self.assertEqual(viewer.current_slider_threshold, 30.0)
                # A click on the track or a key moves it at once either way.
                viewer.update_edges.reset_mock()
                slider.setValue(400)
                viewer.update_edges.assert_called_once_with()

    def test_the_controls_button_starts_the_threshold_slider(self):
        viewer = self.make_viewer()
        layout = viewer.slider_overlay.layout()
        widgets = [layout.itemAt(index).widget() for index in range(layout.count())]
        self.assertEqual([widget for widget in widgets if widget is not None],
                         [viewer.controls_button, viewer.slider_label, viewer.slider])
        button = viewer.controls_button
        self.assertIsInstance(button, ControlsButton)
        self.assertEqual(button.toolTip(), "Mouse and keyboard controls")
        self.assertTrue(button.text_source.startswith("[ENTER] Command | "))
        self.assertEqual(viewer.slider_label.text(), "3.74")
        # The number starts well clear of the button, so the button doesn't
        # read as a mark on it; the label's text starts at its left edge.
        layout.activate()
        gap = viewer.slider_label.x() - (button.x() + button.width())
        self.assertGreaterEqual(gap, 24)
        self.assertLessEqual(gap, 40)
        self.assertEqual(viewer.slider_label.contentsMargins().left(), 0)
        self.assertTrue(viewer.slider_label.alignment() & Qt.AlignmentFlag.AlignLeft)


if __name__ == "__main__":
    unittest.main()
