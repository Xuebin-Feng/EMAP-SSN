# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Console window (desktop.Console_View) and its entry points: lines are classified as errors, warnings or output from what they say, an indented line under a problem shares its colour, the window shows the capture's lines with their times, feeds live while shown and not while hidden, searches, filters, follows the output with no control, clears, copies and saves, replaces a progress line in place, shows its footer, takes the theme's colours in light and dark and again when the theme changes; its second header row shows the buttons that pages and commands registered (the Viewer's add_console_button); the Viewer's Console button, at the foot of its window, opens it, and a Viewer started with --console gets a Console that has the buttons and says where the output is."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QWidget  # noqa: E402

from desktop import Console_View, Desktop_App  # noqa: E402
from desktop.Console_View import (  # noqa: E402
    ERROR, INFO, WARNING, ButtonRegistry, Classifier, ConsoleWindow, classify, dark_tokens, show_console,
)
from desktop.Studio_Theme import TOKENS, apply_studio_theme, set_theme_color, theme_color  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from utilities import Console_Capture  # noqa: E402
from utilities import Terminal_Colors as TC  # noqa: E402
from utilities.Console_Capture import Recorder  # noqa: E402


# The capture a WindowCase gives its windows, unless a test names another (None: a program that keeps no output).
RECORDER = object()


def flush(app, cycles=6):
    for _ in range(cycles):
        app.processEvents()
        time.sleep(0.01)


def wait_for(app, condition, seconds=3.0):
    """Run the event loop until condition() holds (the window draws what arrives every FLUSH_MS)."""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    return condition()


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


class WindowCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def setUp(self):
        self.recorder = Recorder("viewer", None)
        # The window keeps its choice of showing the times in app_settings.json: not the user's own.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.settings_path = Path(directory.name) / "app_settings.json"
        patch = mock.patch.dict(os.environ, {"SSN_APP_SETTINGS_PATH": str(self.settings_path)})
        patch.start()
        self.addCleanup(patch.stop)

    def window(self, capture=RECORDER, **options):
        window = ConsoleWindow(self.recorder if capture is RECORDER else capture, **options)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        flush(self.app, 2)
        return window

    def lines(self, window):
        return window.text.toPlainText().splitlines()

    def drain(self, window):
        """Draw what has arrived now, instead of waiting for the window's timer."""
        window._flush()
        self.app.processEvents()


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


ESC = "\x1b"


def fragments(window, number):
    """The runs of text of one line, as (text, foreground, background or None, bold, underline, italic)."""
    found = []
    iterator = window.text.document().findBlockByNumber(number).begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        fmt = fragment.charFormat()
        background = fmt.background().color().name() if fmt.background().style() != Qt.BrushStyle.NoBrush else None
        found.append((fragment.text(), fmt.foreground().color().name(), background,
                      fmt.fontWeight() >= 700, fmt.fontUnderline(), fmt.fontItalic()))
        iterator += 1
    return found


def output_lines(window):
    """The window's lines, blank lines included (a line of ``splitlines`` is lost when it is the last and empty)."""
    if window._empty:
        return []
    return window.text.toPlainText().split("\n")


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


class HeaderTests(WindowCase):
    """The header is one row that never wraps; the window is never narrower than that row needs."""

    @contextlib.contextmanager
    def in_language(self, language):
        """Windows built inside are in ``language`` (None: English as written). The bundled CJK font is not
        registered: its registering and removing would make Qt look fonts up again for every later window."""
        if language is None:
            yield
            return
        from desktop import Desktop_App
        with mock.patch.dict(Desktop_App.LANGUAGE_FONTS, clear=True):
            installed = Desktop_App.install_translations(self.app, language)
        try:
            yield
        finally:
            installed.remove()

    def assertOneRow(self, window):
        controls = window.header_controls()
        self.assertEqual(len(controls), 10)  # search, previous, next, filter, times, copy, save, clear, log, folder
        root = window.rect()
        centres, previous_right = [], None
        for control in controls:
            with self.subTest(control=control.accessibleName() or getattr(control, 'text', lambda: '')() or type(control).__name__):
                self.assertTrue(control.isVisibleTo(window))
                box = QRect_of(control, window)
                centres.append(box.center().y())
                self.assertGreaterEqual(box.left(), 12)                   # inside the window's margins
                self.assertLessEqual(box.right(), root.width() - 12)
                self.assertGreater(box.width(), 0)
                if previous_right is not None:
                    self.assertGreater(box.left(), previous_right)        # beside the one before it, never over or under it
                previous_right = box.right()
        self.assertLessEqual(max(centres) - min(centres), 1, centres)       # one row, on one centre line
        header_bottom = max(QRect_of(control, window).bottom() for control in controls)
        self.assertLess(header_bottom, window.text.mapTo(window, window.text.rect().topLeft()).y())

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
        self.assertGreaterEqual(minimum, row + 6 * (len(window.header_controls()) - 1) + 24)  # the controls, the gaps, the margins

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
        self.assertEqual([layout.itemAt(index).widget() for index in range(layout.count())], window.header_controls())


def QRect_of(widget, window):
    """A widget's rectangle in the window's coordinates."""
    from PySide6.QtCore import QRect
    return QRect(widget.mapTo(window, widget.rect().topLeft()), widget.size())


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


def icon_colour(button):
    """The colour a button's icon is drawn in: the commonest of its (nearly) opaque pixels."""
    image = button.icon().pixmap(16, 16).toImage()
    counts = {}
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            if colour.alpha() > 200:
                counts[colour.name()] = counts.get(colour.name(), 0) + 1
    return max(counts, key=counts.get) if counts else None


class ButtonRegistryTests(unittest.TestCase):
    """What pages and commands register for the Console's second header row."""

    def test_buttons_come_in_the_order_they_were_registered(self):
        registry = ButtonRegistry()
        self.assertEqual(registry.buttons(), [])
        for name in ("b", "a", "c"):
            self.assertTrue(registry.add(name, name.upper(), lambda: None))
        self.assertEqual([spec.name for spec in registry.buttons()], ["b", "a", "c"])

    def test_a_button_is_its_name_label_function_tooltip_and_icon(self):
        registry = ButtonRegistry()

        def callback():
            return "clicked"

        registry.add("metaDataBtn", "Meta Data", callback, "Open the spreadsheet", "chart-column")
        registry.add("plain", "Plain", callback)
        first, second = registry.buttons()
        self.assertEqual((first.name, first.label, first.tooltip, first.icon), ("metaDataBtn", "Meta Data", "Open the spreadsheet", "chart-column"))
        self.assertIs(first.callback, callback)
        self.assertEqual((second.tooltip, second.icon), (None, None))

    def test_a_name_that_is_registered_already_changes_nothing(self):
        registry = ButtonRegistry()
        heard = []
        registry.subscribe(heard.append)
        first = lambda: "first"  # noqa: E731
        self.assertTrue(registry.add("x", "First", first, "tip", "bot"))
        self.assertFalse(registry.add("x", "Second", lambda: "second", "other tip", "dna"))
        self.assertEqual(len(registry.buttons()), 1)
        spec = registry.buttons()[0]
        self.assertEqual((spec.label, spec.tooltip, spec.icon), ("First", "tip", "bot"))
        self.assertIs(spec.callback, first)
        self.assertEqual(len(heard), 1)  # nobody heard of the second call

    def test_a_listener_hears_each_new_button_while_it_is_subscribed(self):
        registry = ButtonRegistry()
        heard = []
        registry.subscribe(heard.append)
        registry.subscribe(heard.append)  # once, however often it asks
        registry.add("a", "A", lambda: None)
        registry.unsubscribe(heard.append)
        registry.add("b", "B", lambda: None)
        registry.unsubscribe(heard.append)  # not subscribed: nothing to do
        self.assertEqual([spec.name for spec in heard], ["a"])

    def test_a_listener_whose_window_is_gone_is_dropped_and_the_registration_goes_on(self):
        registry = ButtonRegistry()
        heard = []

        def gone(_spec):
            raise RuntimeError("Internal C++ object (ConsoleWindow) already deleted.")

        registry.subscribe(gone)
        registry.subscribe(heard.append)
        self.assertTrue(registry.add("a", "A", lambda: None))
        self.assertTrue(registry.add("b", "B", lambda: None))
        self.assertEqual([spec.name for spec in heard], ["a", "b"])
        self.assertEqual(registry._listeners, [heard.append])


THREE_BUTTONS = (("metaDataBtn", "Meta Data", "chart-column"), ("agentBtn", "Agent", "bot"),
                 ("fold_view_btn", "Fold View", "dna"))


class ButtonRowTests(WindowCase):
    """The Console's second header row: the registered buttons, under the search row and above the output."""

    def setUp(self):
        super().setUp()
        self.registry = ButtonRegistry()
        self.clicked = []

    def tearDown(self):
        Console_View._window = None

    def register(self, buttons=THREE_BUTTONS):
        for name, label, icon in buttons:
            def callback(name=name):
                self.clicked.append(name)
            self.registry.add(name, label, callback, f"Open {label}", icon)

    def row_boxes(self, window):
        return [QRect_of(button, window) for button in window.button_controls()]

    def text_top(self, window):
        return window.text.mapTo(window, window.text.rect().topLeft()).y()

    def test_the_row_is_not_shown_while_nothing_is_registered_and_leaves_no_gap(self):
        window = self.window(buttons=self.registry)
        self.assertEqual(window.button_controls(), [])
        self.assertTrue(window.button_row.isHidden())
        without_row = self.text_top(window)
        header_bottom = max(QRect_of(control, window).bottom() for control in window.header_controls())
        self.assertEqual(without_row - header_bottom - 1, window.layout().spacing())  # the header, one gap, the output
        self.register()
        self.drain(window)
        self.assertTrue(window.button_row.isVisibleTo(window))
        # The output moves down by the row and one gap: nothing else changes.
        self.assertEqual(self.text_top(window) - without_row, window.button_row.height() + window.layout().spacing())

    def test_the_row_sits_under_the_search_row_and_above_the_output(self):
        self.register()
        window = self.window(buttons=self.registry)
        header_bottom = max(QRect_of(control, window).bottom() for control in window.header_controls())
        boxes = self.row_boxes(window)
        self.assertEqual(len(boxes), 3)
        for box in boxes:
            self.assertGreater(box.top(), header_bottom)
            self.assertLess(box.bottom(), self.text_top(window))
            self.assertGreaterEqual(box.left(), 12)
        self.assertEqual(boxes[0].left(), QRect_of(window.search, window).left())  # from the same left edge
        layout = window.layout()
        self.assertEqual([layout.itemAt(index).widget() is window.button_row for index in range(layout.count())],
                         [False, True, False, False])  # header, row, output, footer

    def test_a_button_registered_before_the_window_opens_is_there_when_it_opens(self):
        self.register()
        window = self.window(buttons=self.registry)
        buttons = window.button_controls()
        self.assertEqual([button.text() for button in buttons], ["Meta Data", "Agent", "Fold View"])
        self.assertEqual([button.objectName() for button in buttons], [name for name, _, _ in THREE_BUTTONS])
        self.assertEqual([button.toolTip() for button in buttons], ["Open Meta Data", "Open Agent", "Open Fold View"])
        self.assertEqual([button.accessibleName() for button in buttons], ["Meta Data", "Agent", "Fold View"])
        self.assertEqual([button.accessibleDescription() for button in buttons], ["Open Meta Data", "Open Agent", "Open Fold View"])
        for button in buttons:
            self.assertFalse(button.icon().isNull(), button.text())
            self.assertTrue(button.isVisibleTo(window))
            self.assertTrue(button.isEnabled())

    def test_a_button_registered_while_the_window_is_open_appears_at_once(self):
        window = self.window(buttons=self.registry)
        self.assertTrue(window.button_row.isHidden())
        self.registry.add("agentBtn", "Agent", lambda: None, "Open Agent", "bot")
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent"])  # no event loop in between
        self.assertTrue(window.button_row.isVisibleTo(window))
        self.registry.add("metaDataBtn", "Meta Data", lambda: None, "Open Meta Data", "chart-column")
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent", "Meta Data"])
        flush(self.app, 2)
        boxes = self.row_boxes(window)
        self.assertGreater(boxes[1].left(), boxes[0].right())

    def test_a_button_registered_from_another_thread_appears_in_the_window(self):
        import threading

        window = self.window(buttons=self.registry)
        worker = threading.Thread(target=lambda: self.registry.add("agentBtn", "Agent", lambda: None, None, "bot"))
        worker.start()
        worker.join()
        self.assertTrue(wait_for(self.app, lambda: len(window.button_controls()) == 1))

    def test_buttons_registered_while_the_window_is_hidden_are_there_when_it_is_shown_again(self):
        self.register(THREE_BUTTONS[:1])
        window = self.window(buttons=self.registry)
        window.hide()
        self.assertEqual(self.registry._listeners, [])  # a hidden window listens to nothing
        self.registry.add("agentBtn", "Agent", lambda: None, "Open Agent", "bot")
        self.assertEqual(len(window.button_controls()), 1)
        window.show()
        self.assertEqual([button.text() for button in window.button_controls()], ["Meta Data", "Agent"])
        self.assertTrue(window.button_row.isVisibleTo(window))

    def test_a_click_calls_the_function_with_no_argument(self):
        self.register()
        window = self.window(buttons=self.registry)
        for button in window.button_controls():
            button.click()
        self.assertEqual(self.clicked, [name for name, _, _ in THREE_BUTTONS])

    def test_a_name_registered_twice_gives_one_button(self):
        self.register()
        window = self.window(buttons=self.registry)
        self.register()
        self.registry.add("agentBtn", "Another label", lambda: None, None, "dna")
        self.assertEqual([button.text() for button in window.button_controls()], ["Meta Data", "Agent", "Fold View"])
        self.assertEqual(len(window.findChildren(QPushButton, "agentBtn")), 1)

    def test_a_button_without_an_icon_or_tooltip_is_a_label_alone(self):
        self.registry.add("plain", "Plain", lambda: None)
        window = self.window(buttons=self.registry)
        button, = window.button_controls()
        self.assertEqual(button.text(), "Plain")
        self.assertTrue(button.icon().isNull())
        self.assertEqual(button.toolTip(), "")
        self.assertEqual(button.accessibleName(), "Plain")
        self.assertEqual(Desktop_App.clipped_button_text(button), 0)

    def test_the_buttons_have_the_consoles_button_style_and_show_their_whole_label(self):
        self.register()
        window = self.window(buttons=self.registry)
        for button in window.button_controls():
            self.assertEqual(button.property("role"), "compact")  # the outline button the header's others are
            self.assertEqual(button.height(), window.copy_button.height())
            self.assertEqual(Desktop_App.clipped_button_text(button), 0, button.text())
            self.assertEqual(button.width(), button.maximumWidth())  # as wide as its icon and label ask
            self.assertEqual(button.iconSize(), window.copy_button.iconSize())
        widths = [button.width() for button in window.button_controls()]
        self.assertGreater(widths[0], widths[1])  # "Meta Data" asks for more than "Agent": each has its own
        self.assertEqual(window.copy_button.property("role"), "icon")

    def test_the_icons_take_the_windows_colours_in_light_and_dark_like_its_other_icons(self):
        self.register()
        light = self.window(buttons=self.registry)
        for button in light.button_controls():
            self.assertEqual(icon_colour(button), icon_colour(light.copy_button))
        self.assertEqual(icon_colour(light.copy_button), TOKENS["text_subtle"].lower())
        dark = self.window(buttons=self.registry, dark=True)
        muted = dark_tokens()["text_muted"].lower()
        self.assertEqual(icon_colour(dark.copy_button), muted)
        for button in dark.button_controls():
            self.assertEqual(icon_colour(button), muted)
        self.assertNotEqual(muted, TOKENS["text_subtle"].lower())
        # One registered while the dark window is open is drawn in them too.
        self.registry.add("later", "Later", lambda: None, None, "square-terminal")
        self.assertEqual(icon_colour(dark.button_controls()[-1]), muted)
        # And the labels and the buttons' frames are the dark window's.
        self.assertIn(dark_tokens()["surface"], dark.styleSheet())

    def test_the_dark_window_measures_its_buttons_in_its_own_stylesheet(self):
        self.register()
        dark = self.window(buttons=self.registry, dark=True)
        for button in dark.button_controls():
            self.assertEqual(Desktop_App.clipped_button_text(button), 0, button.text())

    def test_the_window_cannot_be_narrower_than_the_wider_of_the_two_rows(self):
        header_only = self.window().minimumWidth()
        self.register()
        window = self.window(buttons=self.registry)
        # Three short labels fit under the header's own minimum, which stays what it was.
        row = sum(button.width() for button in window.button_controls()) + 6 * 2 + 24
        self.assertLess(row, header_only)
        self.assertEqual(window.minimumWidth(), header_only)
        # A row wider than the header sets the minimum, and the window grows to it at once if it is narrower.
        window.resize(header_only, 420)
        flush(self.app, 2)
        for number in range(6):
            self.registry.add(f"long{number}", f"A button with quite a long label {number}", lambda: None, None, "bot")
        flush(self.app, 4)
        wide_row = sum(button.width() for button in window.button_controls()) + 6 * (len(window.button_controls()) - 1) + 24
        self.assertGreater(wide_row, header_only)
        self.assertEqual(window.minimumWidth(), wide_row)
        self.assertGreaterEqual(window.width(), wide_row)
        window.resize(200, 420)
        flush(self.app, 4)
        self.assertEqual(window.width(), wide_row)

    def assert_one_row_of_buttons(self, window):
        boxes = self.row_boxes(window)
        previous = None
        for button, box in zip(window.button_controls(), boxes):
            with self.subTest(button=button.text()):
                self.assertTrue(button.isVisibleTo(window))
                self.assertGreaterEqual(box.left(), 12)
                self.assertLessEqual(box.right(), window.width() - 12)
                if previous is not None:
                    self.assertGreater(box.left(), previous.right())
                    self.assertEqual(box.center().y(), previous.center().y())  # one row, never wrapped
                previous = box

    def test_both_rows_stay_on_one_line_each_at_the_minimum_width_in_each_language(self):
        from PySide6.QtCore import QCoreApplication

        from desktop import Desktop_App

        minimums = {}
        for language in (None, "zh_CN", Desktop_App.PSEUDO_LANGUAGE):
            with self.subTest(language=language), HeaderTests.in_language(self, language):
                registry = ButtonRegistry()
                for name, label, icon in THREE_BUTTONS:
                    registry.add(name, QCoreApplication.translate("Viewer", label), lambda: None, None, icon)
                window = self.window(buttons=registry)
                minimums[language] = window.minimumWidth()
                window.resize(window.minimumWidth(), 420)
                flush(self.app, 4)
                self.assertEqual(window.width(), window.minimumWidth())
                HeaderTests.assertOneRow(self, window)
                self.assert_one_row_of_buttons(window)
                window.resize(1400, 420)
                flush(self.app, 4)
                HeaderTests.assertOneRow(self, window)
                self.assert_one_row_of_buttons(window)
        self.assertEqual(len(set(minimums.values())), 3, minimums)

    def test_a_long_translation_of_the_buttons_widens_the_window_instead_of_wrapping(self):
        from PySide6.QtCore import QCoreApplication

        with HeaderTests.in_language(self, "pseudo"):
            registry = ButtonRegistry()
            for name, label, icon in THREE_BUTTONS:
                registry.add(name, QCoreApplication.translate("Viewer", label) * 3, lambda: None, None, icon)
            window = self.window(buttons=registry)
            window.resize(200, 420)
            flush(self.app, 4)
            self.assertEqual(window.width(), window.minimumWidth())
            self.assert_one_row_of_buttons(window)
            for button in window.button_controls():
                self.assertEqual(Desktop_App.clipped_button_text(button), 0)
            header_right = max(QRect_of(control, window).right() for control in window.header_controls())
            self.assertLessEqual(header_right, window.width() - 12)

    def test_the_second_row_does_not_change_the_headers_controls(self):
        self.register()
        with_row = self.window(buttons=self.registry)
        without = self.window()
        self.assertEqual(len(with_row.header_controls()), len(without.header_controls()))
        for control in with_row.header_controls():
            self.assertNotIn(control, with_row.button_controls())
        header = with_row.layout().itemAt(0).layout()
        self.assertEqual([header.itemAt(index).widget() for index in range(header.count())], with_row.header_controls())

    def test_show_console_keeps_one_window_for_one_registry_and_a_new_one_for_another(self):
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            first = show_console(None, buttons=self.registry)
            again = show_console(None, buttons=self.registry)
            nothing_given = show_console(None)  # no registry named: the window as it is
            other = ButtonRegistry()
            second = show_console(None, buttons=other)
        self.addCleanup(first.deleteLater)
        self.addCleanup(second.deleteLater)
        self.assertIs(first, again)
        self.assertIs(first, nothing_given)
        self.assertIsNot(first, second)
        self.assertIs(second.buttons, other)
        self.assertIs(first.buttons, self.registry)
        first.close()
        second.close()


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
        HeaderTests.assertOneRow(self, window)

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


class EntryPointTests(WindowCase):
    def tearDown(self):
        Console_View._window = None

    def test_show_console_opens_one_window_and_raises_it_again(self):
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            first = show_console(None)
            second = show_console(None)
        self.addCleanup(first.deleteLater)
        self.assertIs(first, second)
        self.assertTrue(first.isVisible() or first.testAttribute(Qt.WidgetAttribute.WA_WState_Visible))
        first.close()

    def test_a_program_that_keeps_no_output_opens_a_window_that_says_so_in_a_message_box_no_longer(self):
        with mock.patch.object(Console_Capture, "_capture", None), mock.patch.object(
            Console_View.QMessageBox, "information"
        ) as information:
            window = show_console(None)
        self.addCleanup(window.deleteLater)
        information.assert_not_called()
        self.assertIn("--console", window.text.toPlainText())
        self.assertFalse(Console_View.console_available())
        window.close()

    def test_the_window_is_titled_after_the_window_it_belongs_to(self):
        parent = QWidget()
        parent.setWindowTitle("EMAP-SSN Viewer")
        self.addCleanup(parent.deleteLater)
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            window = show_console(parent)
        self.addCleanup(window.deleteLater)
        self.assertEqual(window.windowTitle(), "EMAP-SSN Viewer – Console")
        window.close()

    def test_the_language_row_of_config_tools_and_vr_config_is_as_it_was_with_no_console_button(self):
        import inspect

        from PySide6.QtWidgets import QHBoxLayout

        # The Console is the Viewer's (its command and the button at the foot of its window); Config and Tools
        # show what they print on the start-up splash, and their windows keep the footer they always had.
        self.assertEqual(list(inspect.signature(Desktop_App.language_selector_row).parameters),
                         ["selector", "theme_switch", "leading"])
        host = QWidget()
        self.addCleanup(host.deleteLater)
        selector = Desktop_App.LanguageSelector(host)
        row = Desktop_App.language_selector_row(selector, QWidget(host))
        self.assertIs(type(row), QHBoxLayout)
        widgets = [row.itemAt(index).widget() for index in range(row.count()) if row.itemAt(index).widget()]
        self.assertEqual([widget.objectName() for widget in widgets[:2]], ["languageGlobe", "languageSelector"])
        self.assertEqual(len(widgets), 3)  # the globe, the dropdown and the theme switch: no other button
        # The dropdown asks for what any dropdown of its choices asks for: no room given away, no own painting.
        for name in ("minimumSizeHint", "paintEvent", "give_way"):
            self.assertNotIn(name, vars(Desktop_App.LanguageSelector))
        self.assertEqual(selector.minimumSizeHint().width(), selector.sizeHint().width())


class ViewerEntryPointTests(WindowCase):
    """The Viewer's Console button, at the foot of its window, and the method the console command calls."""

    def setUp(self):
        super().setUp()
        import EMAPSSN_Viewer

        self.module = EMAPSSN_Viewer
        viewer = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
        viewer.canvas = SimpleNamespace(native=QWidget(), size=(1000, 700))
        self.addCleanup(viewer.canvas.native.deleteLater)
        viewer.hud_layout = {"status_x_offset": 10.0, "status_line_spacing": 38.0}
        viewer._hud_timer = mock.Mock()
        viewer.min_threshold, viewer.max_threshold = 0.0, 9.6
        viewer.current_slider_threshold = 3.74
        viewer.update_edges = mock.Mock()
        viewer._build_slider_overlay()
        viewer._build_main_window()
        self.addCleanup(viewer.main_window.deleteLater)
        self.addCleanup(viewer.main_window.close)
        viewer.window_control = None
        viewer.headless = False
        viewer._update_hud_elements = lambda: None
        self.viewer = viewer

    def tearDown(self):
        Console_View._window = None

    def test_the_row_at_the_foot_gets_a_console_button_with_the_terminal_icon(self):
        button = self.viewer.console_button
        self.assertEqual(button.text(), "Console")
        self.assertFalse(button.icon().isNull())
        self.assertTrue(button.toolTip().startswith("Show everything this Viewer has printed"))
        self.assertIs(button.parentWidget(), self.viewer.slider_overlay)

    def test_the_sidebar_and_the_session_card_have_no_console_entry_any_more(self):
        import ast

        self.assertFalse(hasattr(self.viewer, "_add_console_button"))
        self.assertNotIn("consoleBtn", getattr(self.viewer, "sidebar_buttons", {}))
        # Nothing at the Viewer's start-up adds a sidebar button of that name either.
        source = (SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8")
        self.assertNotIn("consoleBtn", source)
        self.assertNotIn("_add_console_button", source)
        # The card is given no Console function, and has no link for one.
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "ControlsButton"]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("console", [keyword.arg for keyword in calls[0].keywords])
        card = self.viewer.controls_button.toggle_card()
        self.addCleanup(card.close)
        self.assertFalse(hasattr(card, "console_button"))
        self.assertEqual([link.text() for link in card.findChildren(QPushButton) if link.objectName() == "controlsLink"],
                         ["List of commands"])

    def test_the_button_is_given_the_action_the_old_sidebar_button_had(self):
        import ast

        tree = ast.parse((SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "ConsoleButton"]
        self.assertEqual(len(calls), 1)
        opened = [keyword.value for keyword in calls[0].keywords if keyword.arg == "opened"]
        self.assertEqual([ast.unparse(value) for value in opened], ["self.open_console_window"])

    def test_opening_it_says_what_happened(self):
        viewer = self.viewer
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            result = viewer.open_console_window()
            self.addCleanup(lambda: Console_View._window and Console_View._window.deleteLater())
        self.assertEqual(result, "opened")
        # Started with --console, nothing is kept, and the Console opens all the same: for its buttons.
        with mock.patch.object(Console_Capture, "_capture", None):
            self.assertEqual(viewer.open_console_window(), "opened")
        viewer.headless = True
        self.assertEqual(viewer.open_console_window(), "no_window")

    def test_a_viewer_started_with_console_opens_a_console_with_the_buttons_and_no_message_box(self):
        viewer = self.viewer
        viewer.add_console_button("metaDataBtn", "Meta Data", lambda: None, "Open the sheet", "chart-column")
        with mock.patch.object(Console_Capture, "_capture", None), mock.patch.object(QMessageBox, "information") as information:
            viewer.console_button.click()
        information.assert_not_called()
        window = Console_View._window
        self.assertIsNotNone(window)
        self.addCleanup(window.deleteLater)
        self.assertFalse(window.kept)
        self.assertEqual(window.text.toPlainText(), Console_View.not_available_text())
        self.assertEqual([button.text() for button in window.button_controls()], ["Meta Data"])

    def test_the_button_opens_the_window_when_output_is_kept(self):
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            self.viewer.console_button.click()
        self.assertIsNotNone(Console_View._window)
        self.addCleanup(Console_View._window.deleteLater)

    def test_the_window_gets_the_viewers_registry_and_a_button_added_later_appears_in_it(self):
        viewer = self.viewer
        viewer.add_console_button("agentBtn", "Agent", lambda: None, "Open the agent", "bot")
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            viewer.open_console_window()
        window = Console_View._window
        self.addCleanup(window.deleteLater)
        self.assertIs(window.buttons, viewer.console_buttons)
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent"])
        viewer.add_console_button("metaDataBtn", "Meta Data", lambda: None, "Open the sheet", "chart-column")
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent", "Meta Data"])
        self.assertFalse(viewer.add_console_button("agentBtn", "Agent", lambda: None))  # known: nothing changes
        self.assertEqual(len(window.button_controls()), 2)

    def test_a_viewer_with_no_window_opens_nothing_and_says_nothing(self):
        # As the sidebar button was: a headless Viewer, or a hidden one not shown, has no window to open or to
        # explain in (the console command refuses with the reason).
        self.viewer.headless = True
        self.viewer.add_console_button("agentBtn", "Agent", lambda: None, "Open the agent", "bot")  # registering is fine
        with mock.patch.object(Console_Capture, "_capture", self.recorder), mock.patch.object(
            QMessageBox, "information"
        ) as information:
            self.viewer.console_button.click()
        information.assert_not_called()
        self.assertIsNone(Console_View._window)
        self.assertEqual(len(self.viewer.console_buttons.buttons()), 1)

    def activate_the_pages(self, viewer):
        import contextlib
        import io

        from web_ui import agent_backend, esmfold_backend, meta_backend

        with contextlib.redirect_stdout(io.StringIO()):
            for plugin in (agent_backend, esmfold_backend, meta_backend):
                plugin.activate(viewer)

    def test_the_pages_register_their_own_buttons_and_the_core_names_none(self):
        viewer = self.viewer
        viewer.open_metadata_ui, viewer.open_agent_ui = mock.Mock(), mock.Mock()
        self.activate_the_pages(viewer)
        specs = viewer.console_buttons.buttons()
        self.assertEqual([(spec.name, spec.label, spec.icon) for spec in specs],
                         [("agentBtn", "Agent", "bot"), ("fold_view_btn", "Fold View", "dna"),
                          ("metaDataBtn", "Meta Data", "chart-column")])
        for spec in specs:
            self.assertTrue(spec.tooltip, spec.name)
            self.assertTrue(spec.label.isascii(), spec.label)  # no emoji: the icon is the picture
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        self.assertFalse(hasattr(viewer, "sidebar_buttons"))  # nothing went into the sidebar's code
        # No command or page is named by the Console window or the Viewer's registration.
        for name in ("meta", "agent", "esmfold", "Fold View", "Meta Data"):
            self.assertNotIn(name, inspect.getsource(Console_View.ButtonRegistry))
            self.assertNotIn(name, inspect.getsource(Console_View.ConsoleWindow._sync_buttons))
            self.assertNotIn(name, inspect.getsource(self.module.MainViewer.add_console_button))
            self.assertNotIn(name, inspect.getsource(self.module.MainViewer.console_button_registry))

    def test_repeating_a_pages_command_adds_nothing_and_the_console_shows_each_button_once(self):
        viewer = self.viewer
        self.activate_the_pages(viewer)
        before = [spec.name for spec in viewer.console_buttons.buttons()]
        self.activate_the_pages(viewer)
        self.assertEqual([spec.name for spec in viewer.console_buttons.buttons()], before)
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            viewer.open_console_window()
        window = Console_View._window
        self.addCleanup(window.deleteLater)
        self.assertEqual([button.text() for button in window.button_controls()], ["Agent", "Fold View", "Meta Data"])
        self.activate_the_pages(viewer)
        self.assertEqual(len(window.button_controls()), 3)

    def test_each_button_opens_its_page(self):
        from web_ui import esmfold_backend

        viewer = self.viewer
        viewer.open_metadata_ui, viewer.open_agent_ui = mock.Mock(), mock.Mock()
        self.activate_the_pages(viewer)
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            viewer.open_console_window()
        window = Console_View._window
        self.addCleanup(window.deleteLater)
        buttons = {button.objectName(): button for button in window.button_controls()}
        with mock.patch.object(esmfold_backend, "open_esmfold_ui") as open_fold:
            buttons["metaDataBtn"].click()
            buttons["agentBtn"].click()
            buttons["fold_view_btn"].click()
        viewer.open_metadata_ui.assert_called_once_with()
        viewer.open_agent_ui.assert_called_once_with()
        open_fold.assert_called_once_with(viewer)

    def test_a_persisted_button_comes_back_as_a_console_button_when_a_session_loads(self):
        from commands import meta as meta_command
        from tests.command_fixtures import run_command

        viewer = self.viewer
        viewer.sidebar_buttons_to_persist = ["meta"]  # what a loaded session holds: the attribute keeps its name
        replayed = []

        def process_command(command, **options):
            name, *arguments = command.split()
            replayed.append((command, options))
            run_command({"meta": meta_command}[name], viewer, arguments)

        viewer.process_command = process_command
        viewer._register_persisted_console_buttons()
        self.assertEqual(replayed, [("meta --register-only", {"record_history": False, "silent": True})])
        self.assertEqual([(spec.name, spec.label) for spec in viewer.console_buttons.buttons()], [("metaDataBtn", "Meta Data")])
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        self.assertFalse(hasattr(viewer, "sidebar_buttons"))

    def test_a_session_with_nothing_persisted_registers_nothing(self):
        viewer = self.viewer
        viewer.process_command = mock.Mock()
        viewer.sidebar_buttons_to_persist = []
        viewer._register_persisted_console_buttons()
        del viewer.sidebar_buttons_to_persist  # an older session file without the attribute
        viewer._register_persisted_console_buttons()
        viewer.process_command.assert_not_called()

    def test_a_viewer_without_the_registration_method_is_skipped_by_the_pages(self):
        # The VR viewer has no Console window: a page's activation finds no add_console_button and does nothing.
        from types import SimpleNamespace

        from web_ui import agent_backend, esmfold_backend, meta_backend

        bare = SimpleNamespace(sidebar_buttons_to_persist=[], open_metadata_ui=lambda: None)
        for plugin in (agent_backend, esmfold_backend, meta_backend):
            plugin.activate(bare)
        self.assertEqual(bare.sidebar_buttons_to_persist, [])


if __name__ == "__main__":
    unittest.main()
