# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Console window lifecycle, header layout, search, output-follow scrolling,
action buttons, and no-capture mode tests (desktop.Console_View)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from desktop import Console_View, Desktop_App  # noqa: E402
from desktop.Console_View import (  # noqa: E402
    ERROR, WARNING, ButtonRegistry, ConsoleWindow, dark_tokens, show_console,
)
from desktop.Studio_Theme import TOKENS  # noqa: E402
from tests.console_view_fixtures import (  # noqa: E402
    QRect_of, THREE_BUTTONS, WindowCase, flush, fragments, header_widgets, wait_for,
)
from utilities import Console_Capture  # noqa: E402
from utilities.Console_Capture import Recorder  # noqa: E402


class WindowTests(WindowCase):
    def test_it_shows_what_the_capture_holds_with_the_time_of_each_line(self):
        started = time.mktime((2026, 10, 24, 11, 47, 44, 0, 0, -1))
        self.recorder.add("stdout", "first")
        self.recorder.lines[-1].time = started
        self.recorder.add("stderr", "second")
        self.recorder.lines[-1].time = started + 7
        window = self.window()
        self.assertEqual(self.lines(window), ["first", "second"])  # the times are in the gutter, not the text
        self.assertEqual([window.text.time_of_block(number) for number in range(2)], ["11:47:44", "11:47:51"])

    def test_new_lines_arrive_while_it_is_shown(self):
        window = self.window()
        self.recorder.add("stdout", "late")
        self.assertTrue(wait_for(self.app, lambda: self.lines(window)[-1:] and self.lines(window)[-1].endswith("late")))

    def test_lines_do_not_pile_up_while_it_is_hidden_and_come_back_when_shown(self):
        self.recorder.add("stdout", "before")
        window = self.window()
        window.hide()
        self.assertEqual(self.recorder._listeners, [])
        self.recorder.add("stdout", "while hidden")
        flush(self.app)
        self.assertEqual(window._pending, [])  # not queued, so nothing piles up
        self.assertEqual(self.lines(window), ["before"])
        window.show()
        self.drain(window)
        self.assertEqual(len(self.recorder._listeners), 1)
        self.assertEqual(self.lines(window), ["before", "while hidden"])

    def test_errors_and_warnings_have_the_themes_danger_and_warning_colours(self):
        for stream, text in (("stdout", "plain"), ("stderr", "Warning: careful"), ("stderr", "Error: broken")):
            self.recorder.add(stream, text)
        window = self.window()
        self.assertEqual([fragments(window, number)[0][1] for number in range(3)],
                         [TOKENS["text_soft"], TOKENS["warning"], TOKENS["danger"]])

    def test_the_time_is_drawn_in_the_subtle_text_colour(self):
        self.recorder.add("stdout", "x")
        window = self.window()
        self.assertEqual(window.text._time_color.name(), TOKENS["text_subtle"])
        dark = self.window(dark=True)
        self.assertEqual(dark.text._time_color.name(), dark_tokens()["text_subtle"])

    def test_a_progress_line_changes_in_place_and_the_finished_line_replaces_it(self):
        window = self.window()
        self.recorder.add("stdout", "start")
        self.recorder.add("stdout", "10%", partial=True)
        self.drain(window)
        self.assertEqual(self.lines(window), ["start", "10%"])
        time.sleep(0.06)
        self.recorder.add("stdout", "60%", partial=True)
        self.drain(window)
        self.assertEqual(self.lines(window), ["start", "60%"])
        self.recorder.add("stdout", "100% done")
        self.drain(window)
        self.assertEqual(self.lines(window), ["start", "100% done"])

    def test_a_progress_line_keeps_the_time_of_its_latest_state(self):
        window = self.window()
        first = time.mktime((2026, 10, 24, 9, 0, 0, 0, 0, -1))
        self.recorder.add("stdout", "start")
        self.recorder.lines[-1].time = first
        self.recorder.add("stdout", "10%", partial=True)
        self.recorder.lines[-1].time = first + 5
        self.drain(window)
        self.assertEqual([window.text.time_of_block(number) for number in range(2)], ["09:00:00", "09:00:05"])
        time.sleep(0.06)
        self.recorder.add("stdout", "100% done")
        self.recorder.lines[-1].time = first + 61
        self.drain(window)
        self.assertEqual([window.text.time_of_block(number) for number in range(2)], ["09:00:00", "09:01:01"])

    def test_the_filter_shows_warnings_and_errors_or_errors_only(self):
        for text in ("plain one", "Warning: careful", "Error: broken", "plain two"):
            self.recorder.add("stdout", text)
        window = self.window()
        window.filter_box.setCurrentIndex(window.filter_box.findData(WARNING))
        self.assertEqual(self.lines(window), ["Warning: careful", "Error: broken"])
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        self.assertEqual(self.lines(window), ["Error: broken"])
        self.recorder.add("stdout", "Warning: later")
        self.recorder.add("stdout", "Error: later too")
        self.drain(window)
        self.assertEqual(self.lines(window), ["Error: broken", "Error: later too"])
        window.filter_box.setCurrentIndex(window.filter_box.findData("all"))
        self.assertEqual(len(self.lines(window)), 6)

    def test_a_traceback_stays_whole_under_the_errors_only_filter(self):
        for text in ("before", "Traceback (most recent call last):", '  File "a.py", line 1', "ValueError: bad", "after"):
            self.recorder.add("stderr", text)
        window = self.window()
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        self.assertEqual(len(self.lines(window)), 3)

    def test_clear_view_empties_the_window_but_not_the_capture(self):
        self.recorder.add("stdout", "old")
        window = self.window()
        window.clear_view()
        self.assertEqual(self.lines(window), [])
        self.recorder.add("stdout", "new")
        self.drain(window)
        self.assertEqual([line for line in self.lines(window) if line], ["new"])
        window.filter_box.setCurrentIndex(window.filter_box.findData(WARNING))
        window.filter_box.setCurrentIndex(window.filter_box.findData("all"))
        self.assertEqual([line for line in self.lines(window) if line], ["new"])  # not "old" again
        self.assertEqual([line.text for line in self.recorder.snapshot()], ["old", "new"])

    def test_the_footer_counts_lines_and_names_the_log(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        recorder = Recorder("viewer", directory.name, ring_lines=3)
        self.addCleanup(recorder.close_log)  # before the folder goes
        for number in range(5):
            recorder.add("stdout", f"line {number}")
        window = ConsoleWindow(recorder)
        self.addCleanup(window.deleteLater)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        flush(self.app, 2)
        self.assertIn("3 lines shown", window.count_label.text())
        self.assertIn("5 written since start-up", window.count_label.text())
        self.assertIn("older lines are in the log file", window.count_label.text())
        self.assertIn(recorder.log_path, window.log_label.toolTip())
        self.assertTrue(window.log_button.isEnabled())

    def test_a_capture_with_no_log_disables_the_log_buttons_and_says_why(self):
        recorder = Recorder("viewer", None)
        recorder.log_error = "disk full"
        window = ConsoleWindow(recorder)
        self.addCleanup(window.deleteLater)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        flush(self.app, 2)
        self.assertFalse(window.log_button.isEnabled())
        self.assertFalse(window.folder_button.isEnabled())
        self.assertIn("disk full", window.log_label.toolTip())

    def test_the_window_keeps_at_most_the_rings_lines(self):
        window = self.window()
        self.assertEqual(window.text.maximumBlockCount(), self.recorder.lines.maxlen)


class HeaderTests(WindowCase):
    """The header is one row that never wraps; the window is never narrower than that row needs."""

    def test_every_control_sits_on_one_row_at_the_minimum_width_in_each_language(self):
        from desktop import Desktop_App
        minimums = {}
        for language in (None, "zh_CN", Desktop_App.PSEUDO_LANGUAGE):
            with self.subTest(language=language), self.in_language(language):
                window = self.window()
                minimums[language] = window.minimumWidth()
                window.resize(window.minimumWidth(), 420)
                flush(self.app, 4)
                self.assertEqual(window.width(), window.minimumWidth())
                self.assertOneRow(window)
                window.resize(1400, 420)
                flush(self.app, 4)
                self.assertOneRow(window)
        # Each language has the row its own words make: the minimum follows them (so the languages were applied).
        self.assertEqual(len(set(minimums.values())), 3, minimums)

    def test_the_window_cannot_be_made_narrower_than_the_row(self):
        window = self.window()
        minimum = window.minimumWidth()
        window.resize(200, 420)
        flush(self.app, 4)
        self.assertEqual(window.width(), minimum)
        row = sum(control.minimumSizeHint().width() for control in window.header_controls())
        gaps = 6 * (len(window.header_controls()) - 1) + 2 * Console_View.SWITCH_EXTRA_GAP
        self.assertGreaterEqual(minimum, row + gaps + 24)  # the controls, the gaps (the switch's two wider), the margins

    def test_the_show_times_switch_has_wider_gaps_on_both_sides_and_the_minimum_holds_them(self):
        from desktop import Desktop_App
        for language in (None, "zh_CN", Desktop_App.PSEUDO_LANGUAGE):
            with self.subTest(language=language), self.in_language(language):
                window = self.window()
                controls = window.header_controls()
                switch_at = controls.index(window.times_switch)
                for width in (window.minimumWidth(), 1400):
                    window.resize(width, 420)
                    flush(self.app, 4)
                    boxes = [QRect_of(control, window) for control in controls]
                    gaps = [boxes[index + 1].left() - boxes[index].right() - 1 for index in range(len(boxes) - 1)]
                    self.assertEqual(gaps[switch_at - 1], 18, gaps)  # between the filter and the switch
                    self.assertEqual(gaps[switch_at], 18, gaps)      # between the switch and Copy
                    others = gaps[:switch_at - 1] + gaps[switch_at + 1:]
                    self.assertEqual(set(others), {6}, gaps)         # the rest of the row keeps its 6 px
                # At the minimum the row just fits: the margins, the controls and every gap, the search field at its least.
                window.resize(window.minimumWidth(), 420)
                flush(self.app, 4)
                self.assertEqual(window.width(), window.minimumWidth())
                self.assertEqual(window.search.width(), Console_View.SEARCH_MINIMUM_WIDTH)
                boxes = [QRect_of(control, window) for control in controls]
                self.assertEqual(boxes[0].left(), 12)
                self.assertEqual(boxes[-1].right(), window.width() - 12 - 1)
                self.assertEqual(window.minimumWidth(), 24 + sum(box.width() for box in boxes) + 6 * (len(boxes) - 3) + 2 * 18)
                self.assertOneRow(window)

    def test_the_search_field_is_the_one_that_gives_way_and_the_one_that_takes_the_room(self):
        window = self.window()
        window.resize(window.minimumWidth(), 420)
        flush(self.app, 4)
        narrow = {id(control): control.width() for control in window.header_controls()}
        self.assertEqual(window.search.width(), Console_View.SEARCH_MINIMUM_WIDTH)
        window.resize(window.minimumWidth() + 300, 420)
        flush(self.app, 4)
        for control in window.header_controls()[1:]:
            self.assertEqual(control.width(), narrow[id(control)], control.accessibleName())
        self.assertEqual(window.search.width(), Console_View.SEARCH_MINIMUM_WIDTH + 300)

    def test_the_match_count_is_in_the_footer_and_changing_it_does_not_change_the_minimum(self):
        for text in ("alpha beta", "beta again"):
            self.recorder.add("stdout", text)
        window = self.window()
        minimum = window.minimumWidth()
        header = [control for control in window.header_controls()]
        self.assertNotIn(window.match_label, header)
        self.assertGreater(window.match_label.mapTo(window, window.match_label.rect().topLeft()).y(),
                           window.text.mapTo(window, window.text.rect().bottomLeft()).y())  # under the text, with the counts
        for needle in ("beta", "zzz", ""):
            window.search.setText(needle)
            flush(self.app, 3)
            self.assertEqual(window.minimumWidth(), minimum)
        with mock.patch.object(Console_View, "MAX_MATCHES", 1):
            window.search.setText("beta")
        flush(self.app, 3)
        self.assertIn("or more", window.match_label.text())
        self.assertEqual(window.minimumWidth(), minimum)

    def test_the_header_has_no_wrapping_layout(self):
        window = self.window()
        layout = window.layout().itemAt(0).layout()
        self.assertEqual(type(layout).__name__, "QHBoxLayout")
        self.assertEqual(header_widgets(layout), window.header_controls())
        # The only other items are the two blank gaps beside the Show Times switch.
        spacers = [index for index in range(layout.count()) if layout.itemAt(index).spacerItem() is not None]
        switch_index = [layout.itemAt(index).widget() for index in range(layout.count())].index(window.times_switch)
        self.assertEqual(spacers, [switch_index - 1, switch_index + 1])


class SearchTests(WindowCase):
    def setUp(self):
        super().setUp()
        for text in ("alpha beta", "Gamma", "beta again", "none here", "BETA last"):
            self.recorder.add("stdout", text)
        self.window_ = self.window()

    def test_it_highlights_and_counts_every_match_ignoring_case(self):
        window = self.window_
        window.search.setText("beta")
        self.assertEqual(len(window.text.extraSelections()), 3)
        self.assertEqual(window.match_label.text(), "3 match(es)")
        window.search.setText("nothing like it")
        self.assertEqual(window.text.extraSelections(), [])
        self.assertEqual(window.match_label.text(), "No matches")
        window.search.setText("")
        self.assertEqual(window.match_label.text(), "")

    def test_next_and_previous_walk_the_matches_and_wrap_round(self):
        window = self.window_
        window.search.setText("beta")
        found = []
        for _ in range(4):
            found.append(window.text.textCursor().block().blockNumber())
            window.find(True)
        self.assertEqual(found, [0, 2, 4, 0])
        window.find(False)  # back from the second match to the first, then round to the last
        self.assertEqual(window.text.textCursor().block().blockNumber(), 0)
        window.find(False)
        self.assertEqual(window.text.textCursor().block().blockNumber(), 4)

    def test_the_arrow_buttons_enter_and_the_function_keys_do_the_same(self):
        window = self.window_
        window.search.setText("beta")
        window.next_button.click()
        self.assertEqual(window.text.textCursor().block().blockNumber(), 2)
        window.previous_button.click()
        self.assertEqual(window.text.textCursor().block().blockNumber(), 0)
        QTest.keyClick(window.search, Qt.Key.Key_Return)
        self.assertEqual(window.text.textCursor().block().blockNumber(), 2)

    def test_matches_are_found_in_lines_that_arrive_later(self):
        window = self.window_
        window.search.setText("beta")
        self.recorder.add("stdout", "beta once more")
        self.drain(window)
        self.assertEqual(window.match_label.text(), "4 match(es)")

    def test_the_count_stops_at_a_limit(self):
        with mock.patch.object(Console_View, "MAX_MATCHES", 2):
            self.window_.search.setText("beta")
        self.assertIn("or more", self.window_.match_label.text())


class FollowTests(WindowCase):
    """The window follows the output with no control: it stays at the last line while that is in view, and a
    view that is higher up never moves by itself."""

    def fill(self, count=400, prefix="line"):
        for number in range(count):
            self.recorder.add("stdout", f"{prefix} {number}")

    def opened(self, count=400):
        """A window of 700 x 300 px holding ``count`` lines, and its vertical scroll bar."""
        self.fill(count)
        window = self.window()
        window.resize(700, 300)
        flush(self.app)
        return window, window.text.verticalScrollBar()

    def scroll_up(self, window):
        """A page up by hand: the scroll bar's own action, as a wheel notch or a key press triggers it."""
        scroll = window.text.verticalScrollBar()
        scroll.triggerAction(scroll.SliderAction.SliderPageStepSub)
        flush(self.app)
        return scroll.value()

    def test_there_is_no_follow_control_in_the_header(self):
        window = self.window()
        self.assertFalse(hasattr(window, "follow_switch"))
        self.assertEqual(window.findChildren(Desktop_App.ToggleSwitch), [window.times_switch])
        for control in window.header_controls():
            words = " ".join((getattr(control, "text", lambda: "")(), control.accessibleName(), control.toolTip()))
            self.assertNotIn("follow", words.lower())

    def test_a_fresh_window_follows_from_the_start(self):
        window = self.window()
        window.resize(700, 300)
        flush(self.app)
        self.fill()
        self.drain(window)
        scroll = window.text.verticalScrollBar()
        self.assertGreater(scroll.maximum(), 0)
        self.assertEqual(scroll.value(), scroll.maximum())
        self.recorder.add("stdout", "one more")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_following_keeps_the_newest_line_in_view(self):
        window, scroll = self.opened()
        self.assertEqual(scroll.value(), scroll.maximum())
        self.recorder.add("stdout", "one more")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_scrolling_up_leaves_the_view_where_it_is_as_output_arrives(self):
        window, scroll = self.opened()
        value = self.scroll_up(window)
        self.assertLess(value, scroll.maximum())
        for text in ("arrives below", "and more"):
            self.recorder.add("stdout", text)
            self.drain(window)
            self.assertEqual(scroll.value(), value)
        self.assertLess(scroll.value(), scroll.maximum())

    def test_reaching_the_bottom_again_follows_again(self):
        window, scroll = self.opened()
        self.scroll_up(window)
        scroll.triggerAction(scroll.SliderAction.SliderToMaximum)
        flush(self.app)
        self.recorder.add("stdout", "arrives below")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def press(self, window, key, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(window.text, key, modifier)
        flush(self.app)

    def test_home_end_and_control_home_and_end_go_to_the_top_and_the_bottom(self):
        window, scroll = self.opened()
        control = Qt.KeyboardModifier.ControlModifier
        for key, modifier in ((Qt.Key.Key_Home, Qt.KeyboardModifier.NoModifier), (Qt.Key.Key_Home, control)):
            with self.subTest(key=key, modifier=modifier):
                self.press(window, Qt.Key.Key_End, control)
                self.assertEqual(scroll.value(), scroll.maximum())
                self.press(window, key, modifier)
                self.assertEqual(scroll.value(), 0)
                self.assertGreater(scroll.maximum(), 0)
        for key, modifier in ((Qt.Key.Key_End, Qt.KeyboardModifier.NoModifier), (Qt.Key.Key_End, control)):
            with self.subTest(key=key, modifier=modifier):
                self.press(window, Qt.Key.Key_Home, control)
                self.assertEqual(scroll.value(), 0)
                self.press(window, key, modifier)
                self.assertEqual(scroll.value(), scroll.maximum())

    def test_control_home_stops_the_following_and_control_end_follows_again(self):
        window, scroll = self.opened()
        control = Qt.KeyboardModifier.ControlModifier
        self.press(window, Qt.Key.Key_Home, control)
        for text in ("arrives below", "and more"):
            self.recorder.add("stdout", text)
            self.drain(window)
            self.assertEqual(scroll.value(), 0)  # the view stays at the top while output arrives
        self.press(window, Qt.Key.Key_End, control)
        self.recorder.add("stdout", "arrives at the end")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())
        self.assertIn("arrives at the end", self.lines(window)[-1])

    def test_select_all_and_copy_still_work_from_the_keyboard(self):
        window, scroll = self.opened(count=20)
        self.press(window, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(window.text.textCursor().selectedText().count(" "), 19)
        self.press(window, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(QGuiApplication.clipboard().text().splitlines(), [f"line {n}" for n in range(20)])

    def test_the_wheel_scrolls_it_off_the_end_and_back(self):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent

        window, scroll = self.opened()

        def wheel(delta):
            point = QPointF(40, 40)
            event = QWheelEvent(point, point, QPoint(0, 0), QPoint(0, delta), Qt.MouseButton.NoButton,
                                Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
            QApplication.sendEvent(window.text.viewport(), event)
            flush(self.app)

        wheel(480)
        self.assertLess(scroll.value(), scroll.maximum())
        value = scroll.value()
        self.recorder.add("stdout", "arrives below")
        self.drain(window)
        self.assertEqual(scroll.value(), value)
        wheel(-480 * 200)
        self.assertEqual(scroll.value(), scroll.maximum())
        self.recorder.add("stdout", "arrives again")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_a_window_resize_keeps_a_following_view_at_the_end_and_a_scrolled_one_in_place(self):
        window, scroll = self.opened()
        for height in (200, 420, 300):
            window.resize(700, height)
            flush(self.app, 4)
            self.assertEqual(scroll.value(), scroll.maximum(), height)
        value = self.scroll_up(window)
        for height in (200, 420, 300):
            window.resize(700, height)
            flush(self.app, 4)
            self.assertEqual(scroll.value(), value, height)
            self.assertLess(scroll.value(), scroll.maximum())

    def test_a_window_grown_to_show_every_line_follows_again(self):
        window, scroll = self.opened(count=30)
        self.scroll_up(window)
        self.assertLess(scroll.value(), scroll.maximum())
        window.resize(700, 900)
        flush(self.app, 4)
        self.assertEqual(scroll.maximum(), 0)  # every line is in view now, though nothing was scrolled
        self.fill(200, "later")
        self.drain(window)
        self.assertGreater(scroll.maximum(), 0)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_a_search_jump_up_stops_the_following_and_one_to_the_end_keeps_it(self):
        window, scroll = self.opened()
        window.search.setText("line 5")
        self.assertEqual(window.text.textCursor().block().blockNumber(), 5)
        self.assertLess(scroll.value(), scroll.maximum())
        value = scroll.value()
        self.recorder.add("stdout", "arrives below")
        self.drain(window)
        self.assertEqual(scroll.value(), value)  # the view stays where the search left it
        window.search.setText("line 399")  # a match in the last lines leaves the end in view
        self.assertEqual(scroll.value(), scroll.maximum())
        self.recorder.add("stdout", "arrives below again")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())
        window.search.setText("")  # clearing the search moves nothing
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_a_match_at_the_end_of_a_following_view_keeps_following(self):
        window, scroll = self.opened()
        window.search.setText("line 399")
        self.assertEqual(scroll.value(), scroll.maximum())
        self.recorder.add("stdout", "arrives below")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def mixed(self, count=300, every=10):
        """``count`` lines of which every ``every``-th is an error, in a window of 700 x 300 px."""
        for number in range(count):
            error = number % every == 0
            self.recorder.add("stderr" if error else "stdout", f"{'Error: ' if error else ''}line {number}")
        window = self.window()
        window.resize(700, 300)
        flush(self.app)
        return window, window.text.verticalScrollBar()

    def test_a_filter_change_keeps_a_following_view_at_the_end(self):
        window, scroll = self.mixed(1000, 2)
        for key in (ERROR, "all", WARNING):
            window.filter_box.setCurrentIndex(window.filter_box.findData(key))
            flush(self.app, 3)
            self.assertEqual(scroll.value(), scroll.maximum(), key)
            self.assertGreater(scroll.maximum(), 0)
        self.recorder.add("stderr", "Error: one more")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_a_filter_change_leaves_a_scrolled_view_in_place_and_not_following(self):
        window, scroll = self.mixed(1000, 2)
        scroll.setSliderPosition(200)  # a drag of the scroll bar's slider, which triggers its move action
        flush(self.app)
        self.assertEqual(scroll.value(), 200)
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        flush(self.app, 3)
        self.assertLess(200, scroll.maximum())
        self.assertEqual(scroll.value(), 200)  # a redraw is not a scroll
        self.recorder.add("stderr", "Error: arrives below")
        self.drain(window)
        self.assertEqual(scroll.value(), 200)
        window.filter_box.setCurrentIndex(window.filter_box.findData("all"))
        flush(self.app, 3)
        self.assertEqual(scroll.value(), 200)
        self.assertLess(scroll.value(), scroll.maximum())

    def test_a_filter_that_leaves_a_few_lines_follows_again(self):
        window, scroll = self.mixed()
        self.scroll_up(window)
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        flush(self.app, 3)
        window.resize(700, 900)
        flush(self.app, 4)
        self.assertEqual(scroll.maximum(), 0)
        for number in range(100):
            self.recorder.add("stderr", f"Error: more {number}")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_clear_view_follows_again_whatever_the_view_was(self):
        window, scroll = self.opened()
        self.scroll_up(window)
        window.clear_view()
        flush(self.app, 3)
        self.assertEqual(scroll.maximum(), 0)
        self.fill(400, "after")
        self.drain(window)
        self.assertGreater(scroll.maximum(), 0)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_a_window_hidden_and_shown_again_keeps_a_scrolled_view_in_place(self):
        window, scroll = self.opened()
        value = self.scroll_up(window)
        window.hide()
        flush(self.app, 3)
        window.show()
        flush(self.app, 4)
        self.assertEqual(scroll.value(), value)
        self.recorder.add("stdout", "arrives below")
        self.drain(window)
        self.assertEqual(scroll.value(), value)

    def test_a_progress_line_rewriting_the_last_line_stays_in_view_when_following(self):
        window, scroll = self.opened()
        for state in ("10%", "40%", "70%"):
            time.sleep(Console_Capture.PARTIAL_NOTIFY_SECONDS + 0.01)  # listeners hear of a progress state only this often
            self.recorder.add("stdout", state, partial=True)
            self.drain(window)
            self.assertEqual(window.text.document().lastBlock().text(), state)
            self.assertEqual(scroll.value(), scroll.maximum())
        self.recorder.add("stdout", "100% done")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_a_progress_line_leaves_a_scrolled_view_alone(self):
        window, scroll = self.opened()
        self.recorder.add("stdout", "10%", partial=True)
        self.drain(window)
        value = self.scroll_up(window)
        for state in ("40%", "70%"):
            time.sleep(Console_Capture.PARTIAL_NOTIFY_SECONDS + 0.01)
            self.recorder.add("stdout", state, partial=True)
            self.drain(window)
            self.assertEqual(window.text.document().lastBlock().text(), state)
            self.assertEqual(scroll.value(), value)
        self.assertLess(scroll.value(), scroll.maximum())


class ButtonTests(WindowCase):
    def test_copy_puts_the_selection_or_everything_on_the_clipboard(self):
        for text in ("one", "two", "three"):
            self.recorder.add("stdout", text)
        window = self.window()
        everything = window.copy_text()
        self.assertEqual(QGuiApplication.clipboard().text().splitlines(), everything.splitlines())
        self.assertEqual(len(everything.splitlines()), 3)
        cursor = window.text.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(window.text.document().findBlockByNumber(1).position() + 3, cursor.MoveMode.KeepAnchor)
        window.text.setTextCursor(cursor)
        selected = window.copy_text()
        self.assertEqual(selected, "one\ntwo")
        self.assertNotIn("\u2029", selected)

    def test_save_writes_the_shown_lines_to_the_chosen_file(self):
        for text in ("one", "Error: two"):
            self.recorder.add("stdout", text)
        window = self.window()
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "out.txt")
            with mock.patch.object(Console_View.QFileDialog, "getSaveFileName", return_value=(target, "")) as dialog:
                self.assertEqual(window.save_text(), target)
            self.assertRegex(Path(dialog.call_args.args[2]).name, r"^viewer-console-\d{8}-\d{6}\.txt$")
            text = Path(target).read_text(encoding="utf-8")
        self.assertTrue(text.strip().endswith("Error: two"))
        self.assertNotIn("one", text)

    def test_a_cancelled_save_writes_nothing_and_a_failed_one_says_so(self):
        window = self.window()
        with mock.patch.object(Console_View.QFileDialog, "getSaveFileName", return_value=("", "")):
            self.assertIsNone(window.save_text())
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            Console_View.QFileDialog, "getSaveFileName", return_value=(directory, "")
        ), mock.patch.object(Console_View.QMessageBox, "critical") as critical:
            self.assertIsNone(window.save_text())
        critical.assert_called_once()

    def test_the_log_buttons_open_the_file_and_show_its_folder(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        recorder = Recorder("viewer", directory.name)
        self.addCleanup(recorder.close_log)  # before the folder goes
        window = ConsoleWindow(recorder)
        self.addCleanup(window.deleteLater)
        with mock.patch.object(Console_View.Desktop_Open, "open_path", return_value=True) as open_path:
            window.log_button.click()
        self.assertEqual(Path(open_path.call_args.args[0]), Path(recorder.log_path))
        with mock.patch.object(Console_View, "open_in_file_manager") as reveal:
            window.folder_button.click()
        reveal.assert_called_once_with(str(Path(recorder.log_path).parent))
        with mock.patch.object(Console_View.Desktop_Open, "open_path", return_value=False), mock.patch.object(
            Console_View, "open_in_file_manager"
        ) as reveal:
            window.log_button.click()  # no program for .log files: the folder opens instead
        reveal.assert_called_once()

    def test_clear_view_button_clears(self):
        self.recorder.add("stdout", "old")
        window = self.window()
        window.clear_button.click()
        self.assertEqual(window.text.toPlainText(), "")

    def test_every_button_has_a_tooltip_and_an_accessible_name(self):
        window = self.window()
        for button in (window.previous_button, window.next_button, window.copy_button, window.save_button,
                       window.clear_button, window.log_button, window.folder_button):
            self.assertTrue(button.toolTip())
            self.assertEqual(button.accessibleName(), button.toolTip())


class NotKeptTests(WindowCase):
    """A program started with --console keeps no output: the Console opens all the same, for the buttons."""

    def tearDown(self):
        Console_View._window = None

    def setUp(self):
        super().setUp()
        self.registry = ButtonRegistry()
        self.clicked = []
        for name, label, icon in THREE_BUTTONS:
            self.registry.add(name, label, lambda name=name: self.clicked.append(name), f"Open {label}", icon)

    def test_it_says_where_the_output_is_in_place_of_the_output(self):
        window = self.window(capture=None, buttons=self.registry)
        self.assertFalse(window.kept)
        self.assertEqual(window.text.toPlainText(), Console_View.not_available_text())
        self.assertIn("--console", window.text.toPlainText())
        # Readable whole at the window's width: it wraps, and nothing is cut off at the side.
        self.assertEqual(window.text.lineWrapMode(), window.text.LineWrapMode.WidgetWidth)
        self.assertFalse(window.text.horizontalScrollBar().isVisible())

    def test_the_buttons_work_and_what_only_works_on_output_is_off(self):
        window = self.window(capture=None, buttons=self.registry)
        self.assertEqual([button.text() for button in window.button_controls()], ["Meta Data", "Agent", "Fold View"])
        for button in window.button_controls():
            self.assertTrue(button.isEnabled(), button.text())
            button.click()
        self.assertEqual(self.clicked, [name for name, _, _ in THREE_BUTTONS])
        for control in window.header_controls():
            self.assertFalse(control.isEnabled(), control.accessibleName() or type(control).__name__)
        self.assertFalse(window.text.times_visible())
        self.assertFalse(window.text._gutter.isVisible())
        self.assertEqual(window.count_label.text(), "")
        self.assertEqual(window.log_label.text(), "")

    def test_showing_it_changes_no_saved_setting(self):
        # The time switch's choice is for the Consoles that have output.
        self.settings_path.write_text('{"CONSOLE_SHOW_TIMES": true}', encoding="utf-8")
        self.window(capture=None, buttons=self.registry)
        self.assertEqual(self.settings_path.read_text(encoding="utf-8"), '{"CONSOLE_SHOW_TIMES": true}')

    def test_it_shows_and_hides_again_and_takes_a_late_button(self):
        window = self.window(capture=None, buttons=self.registry)
        window.hide()
        window.show()
        flush(self.app, 2)
        self.registry.add("later", "Later", lambda: None, None, "bot")
        self.assertEqual(len(window.button_controls()), 4)
        self.assertEqual(window.text.toPlainText(), Console_View.not_available_text())
        # Nothing on the keyboard or in the dark does anything to the missing output.
        window.focus_search()
        self.assertFalse(window.find(True))
        self.assertEqual(window.shown_text(), Console_View.not_available_text())

    def test_the_dark_one_says_it_too_and_it_is_one_row_at_the_least_width(self):
        window = self.window(capture=None, buttons=self.registry, dark=True)
        window.resize(100, 420)
        flush(self.app, 4)
        self.assertEqual(window.width(), window.minimumWidth())
        self.assertEqual(window.text.toPlainText(), Console_View.not_available_text())
        self.assertOneRow(window)

    def test_show_console_opens_it_for_a_program_with_no_capture(self):
        with mock.patch.object(Console_Capture, "_capture", None), mock.patch.object(QMessageBox, "information") as information:
            window = show_console(None, buttons=self.registry)
            again = show_console(None, buttons=self.registry)
        self.addCleanup(window.deleteLater)
        information.assert_not_called()  # a window, not a message box
        self.assertIsNotNone(window)
        self.assertIs(window, again)
        self.assertFalse(window.kept)
        self.assertEqual(len(window.button_controls()), 3)
        window.close()

    def test_a_window_with_output_and_one_without_are_not_mixed_up(self):
        with mock.patch.object(Console_Capture, "_capture", None):
            without = show_console(None, buttons=self.registry)
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            with_output = show_console(None, buttons=self.registry)
        self.addCleanup(without.deleteLater)
        self.addCleanup(with_output.deleteLater)
        self.assertIsNot(without, with_output)
        self.assertTrue(with_output.kept)
        self.assertNotEqual(with_output.text.toPlainText(), Console_View.not_available_text())
        with_output.close()
        without.close()


if __name__ == "__main__":
    unittest.main()
