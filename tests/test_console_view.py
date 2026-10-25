# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Console window (desktop.Console_View) and its entry points: lines are classified as errors, warnings or output from what they say, an indented line under a problem shares its colour, the window shows the capture's lines with their times, feeds live while shown and not while hidden, searches, filters, follows, clears, copies and saves, replaces a progress line in place, shows its footer, takes the theme's colours in light and dark and again when the theme changes; the Viewer's Console button, at the foot of its window, opens it, and says why when nothing is kept."""

from __future__ import annotations

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
from PySide6.QtWidgets import QApplication, QPushButton, QWidget  # noqa: E402

from desktop import Console_View, Desktop_App  # noqa: E402
from desktop.Console_View import (  # noqa: E402
    ERROR, INFO, WARNING, Classifier, ConsoleWindow, classify, dark_tokens, show_console,
)
from desktop.Studio_Theme import TOKENS, apply_studio_theme, set_theme_color, theme_color  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from utilities import Console_Capture  # noqa: E402
from utilities.Console_Capture import Recorder  # noqa: E402


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

    def window(self, **options):
        window = ConsoleWindow(self.recorder, **options)
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
        self.assertEqual(self.lines(window), ["11:47:44  first", "11:47:51  second"])

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
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window)], ["before"])
        window.show()
        self.drain(window)
        self.assertEqual(len(self.recorder._listeners), 1)
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window)], ["before", "while hidden"])

    def test_errors_and_warnings_have_the_themes_danger_and_warning_colours(self):
        for stream, text in (("stdout", "plain"), ("stderr", "Warning: careful"), ("stderr", "Error: broken")):
            self.recorder.add(stream, text)
        window = self.window()
        document = window.text.document()

        def text_colour(number):
            """The colour of a block's text, the second fragment after the time."""
            iterator = document.findBlockByNumber(number).begin()
            iterator += 1
            return iterator.fragment().charFormat().foreground().color().name()

        self.assertEqual(text_colour(0), TOKENS["text_soft"])
        self.assertEqual(text_colour(1), TOKENS["warning"])
        self.assertEqual(text_colour(2), TOKENS["danger"])

    def test_the_time_is_in_the_subtle_text_colour(self):
        self.recorder.add("stdout", "x")
        window = self.window()
        first = window.text.document().findBlockByNumber(0).begin()
        self.assertEqual(first.fragment().charFormat().foreground().color().name(), TOKENS["text_subtle"])

    def test_a_progress_line_changes_in_place_and_the_finished_line_replaces_it(self):
        window = self.window()
        self.recorder.add("stdout", "start")
        self.recorder.add("stdout", "10%", partial=True)
        self.drain(window)
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window)], ["start", "10%"])
        time.sleep(0.06)
        self.recorder.add("stdout", "60%", partial=True)
        self.drain(window)
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window)], ["start", "60%"])
        self.recorder.add("stdout", "100% done")
        self.drain(window)
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window)], ["start", "100% done"])

    def test_the_filter_shows_warnings_and_errors_or_errors_only(self):
        for text in ("plain one", "Warning: careful", "Error: broken", "plain two"):
            self.recorder.add("stdout", text)
        window = self.window()
        window.filter_box.setCurrentIndex(window.filter_box.findData(WARNING))
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window)], ["Warning: careful", "Error: broken"])
        window.filter_box.setCurrentIndex(window.filter_box.findData(ERROR))
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window)], ["Error: broken"])
        self.recorder.add("stdout", "Warning: later")
        self.recorder.add("stdout", "Error: later too")
        self.drain(window)
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window)], ["Error: broken", "Error: later too"])
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
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window) if line], ["new"])
        window.filter_box.setCurrentIndex(window.filter_box.findData(WARNING))
        window.filter_box.setCurrentIndex(window.filter_box.findData("all"))
        self.assertEqual([line.split("  ", 1)[1] for line in self.lines(window) if line], ["new"])  # not "old" again
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

    def test_going_to_a_match_turns_follow_off(self):
        window = self.window_
        self.assertTrue(window.follow_switch.isChecked())
        window.search.setText("gamma")
        self.assertFalse(window.follow_switch.isChecked())

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
    def fill(self, count=400):
        for number in range(count):
            self.recorder.add("stdout", f"line {number}")

    def test_following_keeps_the_newest_line_in_view(self):
        self.fill()
        window = self.window()
        window.resize(700, 300)
        flush(self.app)
        scroll = window.text.verticalScrollBar()
        self.assertEqual(scroll.value(), scroll.maximum())
        self.recorder.add("stdout", "one more")
        self.drain(window)
        self.assertEqual(scroll.value(), scroll.maximum())

    def test_scrolling_up_turns_follow_off_and_the_view_stays_put(self):
        self.fill()
        window = self.window()
        window.resize(700, 300)
        flush(self.app)
        scroll = window.text.verticalScrollBar()
        scroll.triggerAction(scroll.SliderAction.SliderPageStepSub)
        flush(self.app)
        self.assertFalse(window.follow_switch.isChecked())
        value = scroll.value()
        self.recorder.add("stdout", "arrives below")
        self.drain(window)
        self.assertEqual(scroll.value(), value)

    def test_reaching_the_bottom_again_turns_follow_on(self):
        self.fill()
        window = self.window()
        window.resize(700, 300)
        flush(self.app)
        scroll = window.text.verticalScrollBar()
        scroll.triggerAction(scroll.SliderAction.SliderPageStepSub)
        flush(self.app)
        self.assertFalse(window.follow_switch.isChecked())
        scroll.triggerAction(scroll.SliderAction.SliderToMaximum)
        flush(self.app)
        self.assertTrue(window.follow_switch.isChecked())

    def test_switching_follow_on_jumps_to_the_end(self):
        self.fill()
        window = self.window()
        window.resize(700, 300)
        flush(self.app)
        window.follow_switch.setChecked(False)
        scroll = window.text.verticalScrollBar()
        scroll.setValue(0)
        window.follow_switch.setChecked(True)
        self.assertEqual(scroll.value(), scroll.maximum())


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
        cursor.setPosition(window.text.document().findBlockByNumber(1).position() + 8, cursor.MoveMode.KeepAnchor)
        window.text.setTextCursor(cursor)
        selected = window.copy_text()
        self.assertEqual(selected.count("\n"), 1)
        self.assertNotIn("\u2029", selected)
        self.assertIn("one", selected)

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

    def test_a_program_that_keeps_no_output_says_so_instead_of_opening_a_window(self):
        with mock.patch.object(Console_Capture, "_capture", None), mock.patch.object(
            Console_View.QMessageBox, "information"
        ) as information:
            self.assertIsNone(show_console(None))
        information.assert_called_once()
        self.assertIn("--console", information.call_args.args[2])
        self.assertFalse(Console_View.console_available())

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
        self.assertEqual([ast.unparse(value) for value in opened], ["self._console_button_clicked"])

    def test_opening_it_says_what_happened(self):
        viewer = self.viewer
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            result = viewer.open_console_window()
            self.addCleanup(lambda: Console_View._window and Console_View._window.deleteLater())
        self.assertEqual(result, "opened")
        with mock.patch.object(Console_Capture, "_capture", None):
            self.assertEqual(viewer.open_console_window(), "not_kept")
        viewer.headless = True
        self.assertEqual(viewer.open_console_window(), "no_window")

    def test_the_button_explains_when_nothing_is_kept(self):
        with mock.patch.object(Console_Capture, "_capture", None), mock.patch.object(
            self.module.QtWidgets.QMessageBox, "information"
        ) as information:
            self.viewer.console_button.click()
        information.assert_called_once()
        self.assertIn("--console", information.call_args.args[2])

    def test_the_button_opens_the_window_when_output_is_kept(self):
        with mock.patch.object(Console_Capture, "_capture", self.recorder):
            self.viewer.console_button.click()
        self.assertIsNotNone(Console_View._window)
        self.addCleanup(Console_View._window.deleteLater)

    def test_a_viewer_with_no_window_opens_nothing_and_says_nothing(self):
        # As the sidebar button was: a headless Viewer, or a hidden one not shown, has no window to open or to
        # explain in (the console command refuses with the reason).
        self.viewer.headless = True
        with mock.patch.object(Console_Capture, "_capture", self.recorder), mock.patch.object(
            self.module.QtWidgets.QMessageBox, "information"
        ) as information:
            self.viewer.console_button.click()
        information.assert_not_called()
        self.assertIsNone(Console_View._window)


if __name__ == "__main__":
    unittest.main()
