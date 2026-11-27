# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Console window line classification, ANSI colouring, timestamps, copy/paste
formatting, and light/dark theme tests (desktop.Console_View)."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import shiboken6  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop import Console_View, Desktop_App  # noqa: E402
from desktop.Console_View import (  # noqa: E402
    ERROR, INFO, WARNING, Classifier, ConsoleWindow, classify, dark_tokens,
)
from desktop.Studio_Theme import TOKENS, apply_studio_theme, set_theme_color, theme_color  # noqa: E402
from tests.console_view_fixtures import (  # noqa: E402
    ESC, RECORDER, WindowCase, flush, fragments, output_lines,
)
from utilities import Terminal_Colors as TC  # noqa: E402
from utilities.Console_Capture import Recorder  # noqa: E402


class ClassifyTests(unittest.TestCase):
    def test_python_and_library_text_is_told_by_what_it_says(self):
        cases = {
            ERROR: [
                "Traceback (most recent call last):",
                '  File "C:\\x\\y.py", line 3, in f',
                "ZeroDivisionError: division by zero",
                "OSError: [WinError 740] The requested operation requires elevation",
                "Error: Could not launch the ESMFold terminal.",
                "Command Error: nothing selected",
                "[Qt critical] boom",
                "[Qt fatal] gone",
                "CUDA error: out of memory",
                "torch.OutOfMemoryError: CUDA out of memory",
            ],
            WARNING: [
                "C:\\x\\typed_passes.py:336: NumbaPerformanceWarning: The keyword argument",
                "Warning: could not signal GUI readiness",
                "[Qt warning] QWindowsContext: OleInitialize() failed",
                "warning: something odd",
                "Failed to load viewer settings",
                "this API is deprecated",
            ],
            INFO: [
                "Loading PyTorch...",
                "Selected 128 node(s).",
                "Viewer Ready. Press [ENTER] to type commands.",
                "WebServer started at http://127.0.0.1:5000",
                "",
            ],
        }
        for expected, lines in cases.items():
            for line in lines:
                with self.subTest(line=line):
                    self.assertEqual(classify(line), expected)

    PORTAL_WARNING = (
        '[Qt warning] qt.qpa.services: Failed to register with host portal QDBusError('
        '"org.freedesktop.portal.Error.Failed", "Could not register app ID: Connection already '
        'associated with an application ID")'
    )

    def test_a_qt_message_is_told_by_its_tag_not_by_the_words_in_it(self):
        # The line GNOME starts print: "QDBusError(" reads as an exception's name, but Qt called it a warning.
        cases = {
            self.PORTAL_WARNING: WARNING,
            "[Qt warning] GPUInfo not initialized on GpuInfoUpdate": WARNING,
            "[Qt warning] qt.qpa.fonts: Error: could not load the font": WARNING,
            "[Qt warning] qt.qpa.xcb: out of memory": WARNING,
            "[Qt warning] CUDA error: not really": WARNING,
            "[Qt warning] a fatal misunderstanding": WARNING,
            self.PORTAL_WARNING.replace("[Qt warning]", "[Qt critical]"): ERROR,
            "[Qt critical] qt.qpa.services: no portal": ERROR,
            "[Qt fatal] This application failed to start": ERROR,
            '[Qt info] QDBusError("x", "y")': INFO,
            "[Qt info] Failed to open the cache, a new one is made": INFO,
            "[Qt debug] Traceback (most recent call last)": INFO,
            "  [Qt warning] indented, still tagged": WARNING,
            "[qt WARNING] the tag's case does not matter": WARNING,
        }
        for line, expected in cases.items():
            with self.subTest(line=line):
                self.assertEqual(classify(line), expected)

    def test_a_qt_message_without_a_known_tag_is_told_by_its_words(self):
        for line, expected in (
            ('[Qt message] QDBusError("x", "y")', ERROR),
            ("[Qt message] Failed to open the cache", WARNING),
            ("[Qt message] fine", INFO),
            ("[Qt] CUDA error: out of memory", ERROR),
            ("see [Qt warning] mentioned in the middle of a QDBusError(...) line", ERROR),
        ):
            with self.subTest(line=line):
                self.assertEqual(classify(line), expected)

    def test_a_line_under_a_tagged_qt_message_has_its_kind(self):
        classifier = Classifier()
        lines = [
            (self.PORTAL_WARNING, WARNING),
            ("    continued under the warning", WARNING),
            ('[Qt info] QDBusError("x", "y")', INFO),
            ("    continued under plain output", INFO),
        ]
        self.assertEqual([classifier.kind(text) for text, _ in lines], [kind for _, kind in lines])

    def test_a_line_on_stderr_is_not_an_error_for_that_reason(self):
        # classify takes no stream: libraries write progress to stderr.
        self.assertEqual(classify("100%|##########| 4/4 [00:01<00:00]"), INFO)

    def test_an_indented_line_under_a_problem_shares_its_colour(self):
        classifier = Classifier()
        lines = [
            ("Loading", INFO),
            ("Traceback (most recent call last):", ERROR),
            ('  File "a.py", line 1, in <module>', ERROR),
            ("    1 / 0", ERROR),
            ("    ~~^~~", ERROR),
            ("ZeroDivisionError: division by zero", ERROR),
            ("after", INFO),
            ("x.py:3: UserWarning: careful", WARNING),
            ("  warnings.warn('careful')", WARNING),
            ("not indented", INFO),
            ("  indented under plain output", INFO),
        ]
        self.assertEqual([classifier.kind(text) for text, _ in lines], [kind for _, kind in lines])

    def test_a_blank_indented_line_does_not_carry_the_colour(self):
        classifier = Classifier()
        classifier.kind("Error: x")
        self.assertEqual(classifier.kind("   "), INFO)

    def test_a_progress_state_is_classified_without_changing_what_follows(self):
        classifier = Classifier()
        classifier.kind("Error: x")
        classifier.kind("  progress", remember=False)
        self.assertEqual(classifier.kind("  still under the error"), ERROR)


class ColourTests(WindowCase):
    """A command's ANSI colours show, as colour, on a page they can be read on; they are not text."""

    def surface(self, window):
        return window._tokens()["surface"]

    def test_a_sequence_colours_its_run_and_is_no_part_of_the_text(self):
        self.recorder.add("stdout", f"| {ESC}[38;2;230;57;70m\u25cf{ESC}[0m Cluster 0")
        window = self.window()
        self.assertEqual("".join(text for text, *_ in fragments(window, 0)), "| \u25cf Cluster 0")
        self.assertNotIn(ESC, window.text.toPlainText())
        self.assertNotIn("38;2", window.text.toPlainText())
        runs = fragments(window, 0)
        self.assertEqual([text for text, *_ in runs], ["| ", "\u25cf", " Cluster 0"])
        expected = TC.resolve_color(("rgb", 230, 57, 70), dark=False, background=self.surface(window))
        self.assertEqual(runs[1][1], expected)
        self.assertEqual(runs[0][1], TOKENS["text_soft"])
        self.assertEqual(runs[2][1], TOKENS["text_soft"])

    def test_the_sixteen_colours_are_the_pages_own_shades_in_light_and_dark(self):
        self.recorder.add("stdout", f"{ESC}[31mred{ESC}[0m {ESC}[92mbright{ESC}[0m")
        for dark, palette in ((False, TC.PALETTE_LIGHT), (True, TC.PALETTE_DARK)):
            with self.subTest(dark=dark):
                window = self.window(dark=dark)
                runs = fragments(window, 0)
                surface = self.surface(window)
                self.assertEqual(runs[0][1], TC.readable_color(palette[1], surface))
                self.assertEqual(runs[2][1], TC.readable_color(palette[10], surface))

    def test_every_colour_of_the_palette_and_a_vanishing_24_bit_colour_can_be_read_on_either_page(self):
        for index in range(16):
            self.recorder.add("stdout", f"{ESC}[38;5;{index}mpalette {index}{ESC}[0m")
        self.recorder.add("stdout", f"{ESC}[38;2;255;255;255mwhite on the light page{ESC}[0m")
        self.recorder.add("stdout", f"{ESC}[38;2;0;0;0mblack on the dark page{ESC}[0m")
        for dark in (False, True):
            window = self.window(dark=dark)
            surface = self.surface(window)
            for number in range(18):
                with self.subTest(dark=dark, number=number):
                    (_, colour, *_), = fragments(window, number)
                    self.assertGreaterEqual(TC.contrast_ratio(colour, surface), TC.MIN_CONTRAST - 0.01)

    def test_bold_underline_italic_and_a_reset_in_the_middle_of_a_line(self):
        self.recorder.add("stdout", f"{ESC}[1;4;3mloud{ESC}[0m quiet {ESC}[1mbold{ESC}[22m normal")
        runs = fragments(self.window(), 0)
        self.assertEqual([(text, bold, underline, italic) for text, _, _, bold, underline, italic in runs], [
            ("loud", True, True, True), (" quiet ", False, False, False),
            ("bold", True, False, False), (" normal", False, False, False),
        ])

    def test_a_background_of_its_own_keeps_a_text_that_shows_on_it(self):
        self.recorder.add("stdout", f"{ESC}[48;5;196;97m on red {ESC}[0m{ESC}[47m grey bg {ESC}[0m")
        window = self.window()
        (_, red_text, red_bg, *_), (_, grey_text, grey_bg, *_) = fragments(window, 0)
        self.assertEqual(red_bg, "#ff0000")
        self.assertEqual(grey_bg, TC.PALETTE_STANDARD[7])
        self.assertGreaterEqual(TC.contrast_ratio(red_text, red_bg), TC.MIN_CONTRAST)
        self.assertGreaterEqual(TC.contrast_ratio(grey_text, grey_bg), TC.MIN_CONTRAST)  # the page's text, moved to show

    def test_reverse_video_swaps_the_text_and_its_background(self):
        self.recorder.add("stdout", f"{ESC}[31;7m flagged {ESC}[0m")
        window = self.window()
        (_, text, background, *_), = fragments(window, 0)
        self.assertEqual(text.lower(), self.surface(window).lower())
        self.assertEqual(background, TC.resolve_color(("index", 1), dark=False, background=self.surface(window)))

    def test_a_colour_lasts_over_the_lines_of_one_message_until_its_reset(self):
        for text in (f"{ESC}[93mWARNING: first", "  second", f"  third{ESC}[0m", "after"):
            self.recorder.add("stdout", text)
        window = self.window()
        colours = [fragments(window, number)[0][1] for number in range(4)]
        surface = self.surface(window)
        yellow = TC.readable_color(TC.PALETTE_LIGHT[11], surface)
        self.assertEqual(colours, [yellow, yellow, yellow, TOKENS["text_soft"]])
        self.assertEqual(output_lines(window), ["WARNING: first", "  second", "  third", "after"])

    def test_an_amber_warning_block_is_drawn_in_the_warning_colour_and_a_red_error_is_not_amber(self):
        # What print_coloured writes (utilities.Terminal_Colors): a warning in AMBER, an error in RED.
        for text in (f"{ESC}[{TC.AMBER}mWARNING: Incomplete MSA coverage", "Aligned network nodes: 2/3 (66.7%)",
                     f"  - node2_full{ESC}[0m", f"{ESC}[{TC.RED}mERROR: MSA rejected{ESC}[0m"):
            self.recorder.add("stdout", text)
        for dark, palette in ((False, TC.PALETTE_LIGHT), (True, TC.PALETTE_DARK)):
            with self.subTest(dark=dark):
                window = self.window(dark=dark)
                surface = self.surface(window)
                amber = TC.readable_color(palette[3], surface)
                self.assertEqual([fragments(window, number)[0][1] for number in range(4)],
                                 [amber, amber, amber, TC.readable_color(palette[9], surface)])
                self.assertGreaterEqual(TC.contrast_ratio(amber, surface), TC.MIN_CONTRAST)
        # On the dark page this is the colour the window gives its own warnings, a [Qt warning] line.
        self.assertEqual(TC.palette_color(("index", 3), dark=True), dark_tokens()["warning"])

    def test_a_colour_left_on_by_one_stream_does_not_tint_the_other(self):
        self.recorder.add("stdout", f"{ESC}[31mred and not reset")
        self.recorder.add("stderr", "plain on stderr")
        self.recorder.add("stdout", "still red")
        window = self.window()
        red = TC.readable_color(TC.PALETTE_LIGHT[1], self.surface(window))
        self.assertEqual([fragments(window, number)[0][1] for number in range(3)], [red, TOKENS["text_soft"], red])

    def test_a_progress_state_does_not_change_the_colour_the_next_line_starts_in(self):
        self.recorder.add("stdout", f"{ESC}[31mstate one", partial=True)
        self.recorder.add("stdout", f"{ESC}[31mstate two", partial=True)
        self.recorder.add("stdout", "finished plainly")  # replaces the progress line
        self.recorder.add("stdout", "after")
        window = self.window()
        self.assertEqual(output_lines(window), ["finished plainly", "after"])
        self.assertEqual([fragments(window, number)[0][1] for number in range(2)], [TOKENS["text_soft"]] * 2)
        self.recorder.add("stdout", f"{ESC}[31mstate", partial=True)
        self.recorder.add("stdout", f"{ESC}[31mdone, red stays on")
        self.recorder.add("stdout", "next")
        self.drain(window)
        red = TC.readable_color(TC.PALETTE_LIGHT[1], self.surface(window))
        self.assertEqual(output_lines(window), ["finished plainly", "after", "done, red stays on", "next"])
        self.assertEqual([fragments(window, number)[0][1] for number in (2, 3)], [red, red])

    def test_a_line_with_colours_of_its_own_keeps_them_and_is_not_coloured_by_what_it_says(self):
        self.recorder.add("stderr", f"Error: {ESC}[32mall good{ESC}[0m really")
        self.recorder.add("stderr", f"{ESC}[1mError: only bold{ESC}[0m")
        self.recorder.add("stderr", "Error: no escape at all")
        window = self.window()
        coloured, bold_only, plain = (fragments(window, number) for number in range(3))
        green = TC.readable_color(TC.PALETTE_LIGHT[2], self.surface(window))
        self.assertEqual([(text, colour) for text, colour, *_ in coloured],
                         [("Error: ", TOKENS["text_soft"]), ("all good", green), (" really", TOKENS["text_soft"])])
        # Without colour of its own a line still has the colour of what it says, bold or not.
        self.assertEqual(bold_only[0][1], TOKENS["danger"])
        self.assertTrue(bold_only[0][3])
        self.assertEqual(plain[0][1], TOKENS["danger"])

    def test_what_a_line_says_decides_the_filter_not_its_escape_codes(self):
        for text in (f"{ESC}[91mError: red and an error{ESC}[0m", f"{ESC}[32mgreen output{ESC}[0m",
                     f"{ESC}[0mTraceback (most recent call last):", f'{ESC}[2m  File "a.py", line 1{ESC}[0m'):
            self.recorder.add("stdout", text)
        window = self.window()
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        self.assertEqual(output_lines(window), ["Error: red and an error", "Traceback (most recent call last):",
                                              '  File "a.py", line 1'])

    def test_a_colour_sequence_in_a_line_the_filter_hides_still_colours_what_follows(self):
        self.recorder.add("stdout", f"{ESC}[31mhidden by the filter, leaves red on")
        self.recorder.add("stderr", "Error: shown")
        self.recorder.add("stdout", "Error: shown too, in red")
        window = self.window()
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        red = TC.readable_color(TC.PALETTE_LIGHT[1], self.surface(window))
        self.assertEqual([fragments(window, number)[0][1] for number in range(2)], [TOKENS["danger"], red])

    def test_the_search_finds_the_text_that_shows_and_not_the_codes(self):
        self.recorder.add("stdout", f"{ESC}[31mred text{ESC}[0m and {ESC}[38;2;1;2;3mdot{ESC}[0m")
        window = self.window()
        for needle, expected in (("31m", "No matches"), ("38;2", "No matches"), ("[", "No matches"),
                                 ("red text", "1 match(es)"), ("text and dot", "1 match(es)")):
            with self.subTest(needle=needle):
                window.search.setText(needle)
                self.assertEqual(window.match_label.text(), expected)

    def test_copy_and_save_hold_the_text_and_no_colour_codes(self):
        self.recorder.add("stdout", f"{ESC}[31mred{ESC}[0m plain {ESC}[38;2;1;2;3m\u25cf{ESC}[0m")
        window = self.window()
        self.assertEqual(window.copy_text(), "red plain \u25cf")
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "out.txt")
            with mock.patch.object(Console_View.QFileDialog, "getSaveFileName", return_value=(target, "")):
                window.save_text()
            saved = Path(target).read_text(encoding="utf-8")
        self.assertNotIn(ESC, saved)
        self.assertNotIn("[31m", saved)
        self.assertIn("red plain \u25cf", saved)

    def test_a_theme_change_gives_the_runs_their_colours_again(self):
        self.recorder.add("stdout", f"{ESC}[31mred{ESC}[0m")
        window = self.window()
        before = fragments(window, 0)
        window._apply_theme()
        window._render_all()
        self.assertEqual(fragments(window, 0), before)
        self.assertTrue(window._run_formats)

    def test_an_empty_line_is_a_line_and_its_neighbours_stay_apart(self):
        for text in ("", "second", "", ""):
            self.recorder.add("stdout", text)
        window = self.window()
        self.assertEqual(output_lines(window), ["", "second", "", ""])
        self.assertIn("4 lines shown", window.count_label.text())

    def test_a_line_that_is_only_a_colour_sequence_is_an_empty_line_that_sets_the_colour(self):
        for text in (f"{ESC}[31m", "red now"):
            self.recorder.add("stdout", text)
        window = self.window()
        self.assertEqual(output_lines(window), ["", "red now"])
        self.assertEqual(fragments(window, 1)[0][1], TC.readable_color(TC.PALETTE_LIGHT[1], self.surface(window)))


class TimesTests(WindowCase):
    """The time of each line is drawn in a gutter beside the text, never in it, and the Show Times switch shows or hides it."""

    STARTED = time.mktime((2026, 10, 24, 11, 47, 44, 0, 0, -1))

    def fill(self, texts=("first", "Error: second", "third")):
        for number, text in enumerate(texts):
            self.recorder.add("stdout", text)
            self.recorder.lines[-1].time = self.STARTED + 7 * number

    def ink(self, widget):
        """Whether anything is drawn on a widget: a pixel that differs from its corner's."""
        image = widget.grab().toImage()
        background = image.pixel(0, 0)
        return any(image.pixel(x, y) != background for x in range(image.width()) for y in range(image.height()))

    def test_no_time_is_in_the_text_whatever_is_selected_copied_saved_or_searched(self):
        self.fill()
        window = self.window()
        window.text.selectAll()
        for text in (window.text.toPlainText(), window.shown_text(), window.copy_text()):
            self.assertEqual(text, "first\nError: second\nthird")
            self.assertNotRegex(text, r"\d\d:\d\d")
        self.assertEqual(QGuiApplication.clipboard().text(), "first\nError: second\nthird")
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "out.txt")
            with mock.patch.object(Console_View.QFileDialog, "getSaveFileName", return_value=(target, "")):
                window.save_text()
            self.assertEqual(Path(target).read_text(encoding="utf-8"), "first\nError: second\nthird\n")
        for needle in ("11:47", "11:47:44", "47:51", "00:00"):
            window.search.setText(needle)
            self.assertEqual(window.match_label.text(), "No matches", needle)

    def test_each_line_keeps_its_own_time_for_the_gutter(self):
        self.fill()
        window = self.window()
        self.assertEqual([window.text.time_of_block(number) for number in range(3)], ["11:47:44", "11:47:51", "11:47:58"])
        self.assertEqual(window.text.time_of_block(99), "")
        self.assertEqual(Console_View.clock_text(-1), "")
        self.assertEqual(Console_View.clock_text(86399), "23:59:59")
        self.assertEqual(Console_View.clock_text(0), "00:00:00")

    def test_a_time_stays_with_its_line_when_older_lines_are_trimmed_and_when_the_filter_hides_some(self):
        self.fill(("a", "b", "c", "d", "Error: e"))
        window = self.window()
        window.text.setMaximumBlockCount(3)
        window._render_all()
        self.assertEqual(output_lines(window), ["c", "d", "Error: e"])
        self.assertEqual([window.text.time_of_block(number) for number in range(3)], ["11:47:58", "11:48:05", "11:48:12"])
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        self.assertEqual(output_lines(window), ["Error: e"])
        self.assertEqual(window.text.time_of_block(0), "11:48:12")

    def test_new_lines_get_their_time_as_they_arrive(self):
        self.fill()
        window = self.window()
        self.recorder.add("stdout", "late")
        self.recorder.lines[-1].time = self.STARTED + 3600
        self.drain(window)
        self.assertEqual(window.text.time_of_block(3), "12:47:44")
        self.assertEqual(output_lines(window)[-1], "late")

    def test_the_times_show_by_default_in_a_gutter_left_of_the_text_and_the_switch_hides_them(self):
        self.fill()
        window = self.window()
        gutter, text = window.text._gutter, window.text
        self.assertTrue(window.times_switch.isChecked())
        self.assertTrue(text.times_visible())
        self.assertTrue(gutter.isVisible())
        self.assertEqual(text.gutter_width(), text.fontMetrics().horizontalAdvance("00:00:00") + text.GAP)
        self.assertEqual(text.viewportMargins().left(), text.gutter_width())
        self.assertLessEqual(gutter.geometry().right() + 1, text.viewport().geometry().left())  # beside the text, not over it
        self.assertEqual(gutter.geometry().top(), text.viewport().geometry().top())
        self.assertEqual(gutter.geometry().height(), text.viewport().geometry().height())
        self.assertTrue(self.ink(gutter))
        window.times_switch.click()
        self.assertFalse(window.times_switch.isChecked())
        self.assertFalse(text.times_visible())
        self.assertFalse(gutter.isVisible())
        self.assertEqual(text.gutter_width(), 0)
        self.assertEqual(text.viewportMargins().left(), 0)
        window.times_switch.click()
        self.assertTrue(gutter.isVisible())
        self.assertEqual(text.viewportMargins().left(), text.gutter_width())
        self.assertGreater(text.gutter_width(), 0)
        self.assertEqual(output_lines(window), ["first", "Error: second", "third"])  # the text never changed

    def test_the_gutter_is_empty_for_a_window_with_nothing_in_it(self):
        window = self.window()
        self.assertFalse(self.ink(window.text._gutter))

    def test_the_switch_is_a_labelled_header_control_with_a_tooltip(self):
        window = self.window()
        self.assertEqual(window.times_switch.text(), "Show Times")
        self.assertEqual(window.times_switch.toolTip(), "Show the time of each line, beside the text")
        self.assertIn("time", window.times_switch.toolTip().lower())
        self.assertTrue(window.times_switch.isCheckable())

    def test_the_choice_is_kept_for_the_next_console_beside_the_other_settings(self):
        self.settings_path.write_text('{"LANGUAGE": "en", "CONFIG_THEME_COLOR": "teal"}', encoding="utf-8")
        window = self.window()
        self.assertTrue(window.times_switch.isChecked())  # shown unless turned off
        window.times_switch.click()
        saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, {"LANGUAGE": "en", "CONFIG_THEME_COLOR": "teal", "CONSOLE_SHOW_TIMES": False})
        again = self.window()
        self.assertFalse(again.times_switch.isChecked())
        self.assertFalse(again.text.times_visible())
        self.assertFalse(again.text._gutter.isVisible())
        self.assertEqual(again.text.viewportMargins().left(), 0)
        again.times_switch.click()
        self.assertEqual(json.loads(self.settings_path.read_text(encoding="utf-8"))["CONSOLE_SHOW_TIMES"], True)
        self.assertTrue(self.window().times_switch.isChecked())

    def test_a_missing_odd_or_unreadable_setting_leaves_the_times_shown_and_the_file_alone(self):
        for content in (None, "{}", '{"CONSOLE_SHOW_TIMES": "no"}', '{"CONSOLE_SHOW_TIMES": null}', "not json at all"):
            with self.subTest(content=content):
                if content is None:
                    self.settings_path.unlink(missing_ok=True)
                else:
                    self.settings_path.write_text(content, encoding="utf-8")
                window = self.window()
                self.assertTrue(window.times_switch.isChecked())
                self.assertTrue(window.text._gutter.isVisible())
        # A file that cannot be read is left alone, as the other settings writers do: the choice lasts for the window.
        window = self.window()
        window.times_switch.click()
        self.assertFalse(window.text.times_visible())
        self.assertEqual(self.settings_path.read_text(encoding="utf-8"), "not json at all")

    def test_the_gutter_follows_the_font_and_the_theme(self):
        self.fill()
        window = self.window()
        before = window.text.gutter_width()
        bigger = window.text.font()
        bigger.setPointSize(bigger.pointSize() + 8)
        window.text.setFont(bigger)
        flush(self.app, 2)
        self.assertGreater(window.text.gutter_width(), before)
        self.assertEqual(window.text.viewportMargins().left(), window.text.gutter_width())
        self.assertEqual(window.text._gutter.width(), window.text.gutter_width())
        dark = self.window(dark=True)
        self.assertEqual(dark.text._time_color.name(), dark_tokens()["text_subtle"])

    def test_the_wheel_over_the_gutter_scrolls_the_text(self):
        for number in range(300):
            self.recorder.add("stdout", f"line {number}")
        window = self.window()
        window.resize(700, 300)
        flush(self.app)
        scroll = window.text.verticalScrollBar()
        scroll.setValue(scroll.maximum())
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent
        gutter = window.text._gutter
        event = QWheelEvent(QPointF(5, 20), QPointF(5, 20), QPoint(0, 0), QPoint(0, 480), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        before = scroll.value()
        QApplication.sendEvent(gutter, event)
        self.assertLess(scroll.value(), before)


class CopyFormatTests(WindowCase):
    """Copy and Ctrl+C put the same plain text and the same HTML (one monospace <pre>) on the clipboard."""

    FAMILY = "Test Mono"

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(Console_View, "system_monospace_family", return_value=self.FAMILY)
        patch.start()
        self.addCleanup(patch.stop)

    def clipboard(self):
        data = QGuiApplication.clipboard().mimeData()
        return data.text(), data.html()

    def pre(self, html):
        found = re.fullmatch(r'<pre style="([^"]*)">(.*)</pre>', html, re.S)
        self.assertIsNotNone(found, html)
        return found.group(1), found.group(2)

    def select(self, window, start, end):
        """Select from (block, offset) to (block, offset)."""
        document = window.text.document()
        cursor = window.text.textCursor()
        cursor.setPosition(document.findBlockByNumber(start[0]).position() + start[1])
        cursor.setPosition(document.findBlockByNumber(end[0]).position() + end[1], cursor.MoveMode.KeepAnchor)
        window.text.setTextCursor(cursor)

    def test_the_html_is_one_pre_naming_the_system_family_first_then_monospace(self):
        self.recorder.add("stdout", "one")
        window = self.window()
        window.copy_text()
        text, html = self.clipboard()
        self.assertEqual(text, "one")
        style, body = self.pre(html)
        self.assertEqual(body, "one")
        self.assertEqual(style, f"font-family:'{self.FAMILY}',monospace;font-size:10pt;margin:0")
        self.assertNotIn("mso-", html)
        self.assertNotIn("text-autospace", html)
        self.assertNotIn("Noto Sans Mono", html)

    def test_with_no_installed_family_only_monospace_is_named(self):
        self.recorder.add("stdout", "one")
        window = self.window()
        with mock.patch.object(Console_View, "system_monospace_family", return_value=None):
            window.copy_text()
        html = self.clipboard()[1]
        self.assertEqual(self.pre(html)[0], "font-family:monospace;font-size:10pt;margin:0")
        self.assertNotIn("mso-", html)
        self.assertNotIn("text-autospace", html)
        self.assertNotIn("Noto Sans Mono", html)

    def test_the_real_chooser_never_names_the_bundled_family(self):
        self.recorder.add("stdout", "one")
        window = self.window()
        with mock.patch.object(Console_View, "system_monospace_family", Desktop_App.system_monospace_family):
            window.copy_text()
        self.assertNotIn("Noto Sans Mono", self.clipboard()[1])

    def test_angle_brackets_and_ampersands_are_escaped_in_the_html_only(self):
        self.recorder.add("stdout", "a <b> & c")
        window = self.window()
        window.copy_text()
        text, html = self.clipboard()
        self.assertEqual(text, "a <b> & c")
        self.assertEqual(self.pre(html)[1], "a &lt;b&gt; &amp; c")

    def test_a_plain_line_has_no_span_and_problem_lines_carry_their_colours(self):
        for line in ("plain", "Error: boom", "Warning: odd"):
            self.recorder.add("stdout", line)
        window = self.window()
        window.copy_text()
        lines = self.pre(self.clipboard()[1])[1].split("\n")
        self.assertEqual(lines[0], "plain")
        self.assertNotIn("<span", lines[0])
        for number, line in ((1, "Error: boom"), (2, "Warning: odd")):
            colour = fragments(window, number)[0][1]
            self.assertEqual(lines[number], f'<span style="color:{colour}">{line}</span>')
        self.assertNotEqual(fragments(window, 1)[0][1], fragments(window, 2)[0][1])

    def test_a_coloured_dot_is_alone_in_its_span(self):
        self.recorder.add("stdout", f"| {ESC}[38;2;200;30;30m●{ESC}[0m Cluster 1")
        window = self.window()
        window.copy_text()
        colour = fragments(window, 0)[1][1]
        self.assertEqual(self.pre(self.clipboard()[1])[1], f'| <span style="color:{colour}">●</span> Cluster 1')

    def test_bold_italic_underline_and_a_background_each_appear_when_set(self):
        self.recorder.add("stdout", f"{ESC}[1mb{ESC}[0m {ESC}[3mi{ESC}[0m {ESC}[4mu{ESC}[0m {ESC}[48;5;196;97mg{ESC}[0m")
        window = self.window()
        window.copy_text()
        body = self.pre(self.clipboard()[1])[1]
        self.assertRegex(body, r'<span style="[^"]*font-weight:bold[^"]*">b</span>')
        self.assertRegex(body, r'<span style="[^"]*font-style:italic[^"]*">i</span>')
        self.assertRegex(body, r'<span style="[^"]*text-decoration:underline[^"]*">u</span>')
        self.assertRegex(body, r'<span style="[^"]*background-color:#ff0000[^"]*">g</span>')

    def test_several_lines_are_joined_by_a_line_feed_in_both_forms(self):
        for line in ("one", "two", "three"):
            self.recorder.add("stdout", line)
        window = self.window()
        window.copy_text()
        text, html = self.clipboard()
        self.assertEqual(text, "one\ntwo\nthree")
        self.assertEqual(self.pre(html)[1], "one\ntwo\nthree")
        self.assertNotIn("\u2029", text + html)

    def test_a_selection_gives_exactly_the_selected_characters_in_both_forms(self):
        self.recorder.add("stdout", f"ab {ESC}[38;2;200;30;30mCOLOUR{ESC}[0m cd")
        self.recorder.add("stdout", "second line")
        window = self.window()
        colour = fragments(window, 0)[1][1]
        self.select(window, (0, 5), (1, 3))  # "LOUR cd" and "sec"
        window.copy_text()
        text, html = self.clipboard()
        self.assertEqual(text, "LOUR cd\nsec")
        self.assertEqual(self.pre(html)[1], f'<span style="color:{colour}">LOUR</span> cd\nsec')
        self.select(window, (0, 4), (0, 7))  # inside the coloured run
        window.copy_text()
        text, html = self.clipboard()
        self.assertEqual(text, "OLO")
        self.assertEqual(self.pre(html)[1], f'<span style="color:{colour}">OLO</span>')

    def test_ctrl_c_gives_what_the_button_gives(self):
        self.recorder.add("stdout", f"ab {ESC}[38;2;200;30;30mCOLOUR{ESC}[0m <cd>")
        self.recorder.add("stdout", "Error: second")
        window = self.window()
        self.select(window, (0, 1), (1, 4))
        window.copy_text()
        by_button = self.clipboard()
        QGuiApplication.clipboard().clear()
        window.text.copy()
        self.assertEqual(self.clipboard(), by_button)
        direct = window.text.createMimeDataFromSelection()
        self.assertEqual((direct.text(), direct.html()), by_button)
        gc.collect()
        self.assertEqual(self.clipboard(), by_button)  # the data is owned by Qt, not freed with a wrapper
        self.assertEqual(self.clipboard(), by_button)

    def mortal(self, capture=RECORDER):
        """A window this test destroys itself, so no cleanup touches it afterwards."""
        window = ConsoleWindow(self.recorder if capture is RECORDER else capture)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        flush(self.app, 2)
        return window

    def destroy(self, window):
        window.close()
        shiboken6.delete(window)
        gc.collect()
        self.app.processEvents()

    def test_the_clipboard_outlives_the_window_after_the_button(self):
        self.recorder.add("stdout", f"a <b> {ESC}[38;2;200;30;30mred{ESC}[0m")
        window = self.mortal()
        window.copy_text()
        before = self.clipboard()
        self.destroy(window)
        self.assertEqual(self.clipboard(), before)
        self.assertEqual(self.clipboard(), before)

    def test_the_clipboard_outlives_the_window_after_ctrl_c(self):
        self.recorder.add("stdout", f"a <b> {ESC}[38;2;200;30;30mred{ESC}[0m")
        window = self.mortal()
        window.text.selectAll()
        window.text.copy()
        before = self.clipboard()
        self.assertEqual(before[0], "a <b> red")
        self.destroy(window)
        self.assertEqual(self.clipboard(), before)
        self.assertEqual(self.clipboard(), before)

    def test_a_second_window_copies_after_the_first_is_gone(self):
        self.recorder.add("stdout", "first text")
        window = self.mortal()
        window.text.selectAll()
        window.text.copy()
        self.destroy(window)
        other = Recorder("viewer", None)
        other.add("stdout", "second <text>")
        window = self.window(capture=other)
        window.text.selectAll()
        window.text.copy()
        gc.collect()
        text, html = self.clipboard()
        self.assertEqual(text, "second <text>")
        self.assertEqual(self.pre(html)[1], "second &lt;text&gt;")

    def test_an_empty_first_line_keeps_its_line_break_after_the_pre_tag(self):
        # An HTML reader drops one line break right after <pre>, so the HTML holds one more than the plain text.
        self.recorder.add("stdout", "")
        self.recorder.add("stdout", "second")
        window = self.window()
        self.assertEqual(self.lines(window), ["", "second"])
        window.copy_text()
        text, html = self.clipboard()
        self.assertEqual(text, "\nsecond")
        self.assertEqual(self.pre(html)[1], "\n\nsecond")

    def test_a_console_text_alone_copies_a_selection(self):
        text = Console_View.ConsoleText()
        self.addCleanup(text.deleteLater)
        text.setPlainText("alpha\nbeta")
        cursor = text.textCursor()
        cursor.setPosition(2)
        cursor.setPosition(8, cursor.MoveMode.KeepAnchor)
        text.setTextCursor(cursor)
        data = text.createMimeDataFromSelection()
        self.assertEqual(data.text(), "pha\nbe")
        self.assertEqual(self.pre(data.html())[1], "pha\nbe")
        text.copy()
        self.assertEqual(QGuiApplication.clipboard().text(), "pha\nbe")


class ThemeTests(WindowCase):
    def test_a_theme_change_recolours_the_window_and_its_lines(self):
        self.recorder.add("stderr", "Error: broken")
        window = self.window()
        before = window.formats[ERROR].foreground().color().name()
        self.assertEqual(before, TOKENS["danger"])
        original = theme_color()
        self.addCleanup(set_theme_color, original)
        old_selection = TOKENS["accent_soft"]
        self.assertIn(old_selection, window.text.styleSheet())
        apply_studio_theme(self.app, "blue")
        flush(self.app)
        # Only the accent shades change with the theme colour: the selection colour is the one the window shows.
        self.assertNotEqual(TOKENS["accent_soft"], old_selection)
        self.assertIn(TOKENS["accent_soft"], window.text.styleSheet())
        self.assertNotIn(old_selection, window.text.styleSheet())
        self.assertEqual(window.formats[INFO].foreground().color().name(), TOKENS["text_soft"])
        self.assertIn(TOKENS["border"], window.text.styleSheet())

    def test_the_dark_window_uses_the_dark_shades_and_its_own_error_colours(self):
        self.recorder.add("stderr", "Error: broken")
        self.recorder.add("stdout", "plain")
        window = self.window(dark=True)
        tokens = dark_tokens()
        self.assertNotEqual(tokens["surface"], TOKENS["surface"])
        self.assertEqual(tokens["surface"], "#18181b")
        self.assertIn(tokens["surface"], window.text.styleSheet())
        self.assertEqual(window.formats[ERROR].foreground().color().name(), tokens["danger"])
        self.assertEqual(window.formats[INFO].foreground().color().name(), tokens["text_soft"])
        self.assertNotEqual(window.formats[ERROR].foreground().color().name(), TOKENS["danger"])
        self.assertEqual(window.palette().color(window.backgroundRole()).name(), tokens["window"])

    def test_the_dark_windows_arrow_files_do_not_outlive_the_process_that_wrote_them(self):
        import subprocess
        import textwrap

        program = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, sys.argv[1])
            from desktop import Console_View
            Console_View._dark_stylesheet(Console_View.dark_tokens())
            print(Console_View._dark_icon_folder)
            """
        )
        with tempfile.TemporaryDirectory() as private:
            environment = dict(os.environ, TEMP=private, TMP=private, TMPDIR=private)
            done = subprocess.run([sys.executable, "-c", program, str(SRC)], capture_output=True, text=True,
                                  env=environment, timeout=120, creationflags=0x08000000 if sys.platform == "win32" else 0)
            self.assertEqual(done.returncode, 0, done.stderr)
            folder = Path(done.stdout.strip().splitlines()[-1])
            self.assertEqual(folder.parent.resolve(), Path(private).resolve())  # it was written in this TEMP...
            self.assertFalse(folder.exists())  # ...and removed when the process ended
            self.assertEqual(list(Path(private).iterdir()), [])

    def test_the_dark_windows_arrows_are_files_in_the_dark_colours(self):
        window = self.window(dark=True)
        tokens = dark_tokens()
        urls = re.findall(r'url\("([^"]*chevron[^"]*)"\)', window.styleSheet())
        self.assertTrue(urls)
        for url in urls:
            with self.subTest(url=Path(url).name):
                self.assertTrue(Path(url).is_file(), url)
        # The theme's own bundled arrows have none in these colours: the window made them.
        bundled = Console_View.qss_icon_path("chevron-down", "text_subtle", tokens)
        self.assertFalse(bundled.is_file())
        self.assertIn(bundled.name, {Path(url).name for url in urls})

    def test_dark_text_is_readable_on_the_dark_surface(self):
        def luminance(name):
            from PySide6.QtGui import QColor
            colour = QColor(name)
            def channel(value):
                value /= 255
                return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
            return 0.2126 * channel(colour.red()) + 0.7152 * channel(colour.green()) + 0.0722 * channel(colour.blue())
        tokens = dark_tokens()
        for name in ("text_soft", "danger", "warning", "text_subtle"):
            lighter, darker = sorted((luminance(tokens[name]), luminance(tokens["surface"])), reverse=True)
            self.assertGreaterEqual((lighter + 0.05) / (darker + 0.05), 4.5, name)


if __name__ == "__main__":
    unittest.main()
