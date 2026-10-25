# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Terminal colours (utilities.Terminal_Colors): the one question commands ask before they colour output (a terminal, or the output the Console captures), the reader of colour sequences (reset, bold, the 16, 256-colour and 24-bit colours, a reset in the middle of a line, malformed and unknown sequences, other escapes dropped), taking sequences out of text and out of bytes that arrive in chunks, and colours that show on a light or a dark page. The commands that colour their output ask the shared question."""

from __future__ import annotations

import ast
import io
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Console_Capture  # noqa: E402
from utilities import Terminal_Colors as T  # noqa: E402
from utilities.Terminal_Colors import DEFAULT_STYLE, Style  # noqa: E402

ESC = "\x1b"


def runs(text, style=DEFAULT_STYLE):
    return T.parse_ansi(text, style)[0]


def end_style(text, style=DEFAULT_STYLE):
    return T.parse_ansi(text, style)[1]


class WhetherToColourTests(unittest.TestCase):
    class Stream(io.StringIO):
        def __init__(self, terminal=False):
            super().__init__()
            self._terminal = terminal

        def isatty(self):
            return self._terminal

    def test_a_terminal_is_coloured_and_a_buffer_or_a_pipe_is_not(self):
        self.assertTrue(T.colours_enabled(self.Stream(terminal=True)))
        self.assertFalse(T.colours_enabled(self.Stream()))
        with mock.patch.object(sys, "stdout", self.Stream(terminal=True)):
            self.assertTrue(T.colours_enabled())
        with mock.patch.object(sys, "stdout", self.Stream()):
            self.assertFalse(T.colours_enabled())

    def test_the_stream_the_capture_installed_is_coloured_though_it_is_a_pipe(self):
        own, buffer = self.Stream(), self.Stream()
        capture = mock.Mock(text_streams=(own, self.Stream()))
        with mock.patch.object(Console_Capture, "_capture", capture):
            self.assertTrue(T.colours_enabled(own))
            with mock.patch.object(sys, "stdout", own):
                self.assertTrue(T.colours_enabled())
            # A buffer a caller swapped in is not the capture's stream: its text would hold the sequences.
            with mock.patch.object(sys, "stdout", buffer):
                self.assertFalse(T.colours_enabled())
        with mock.patch.object(Console_Capture, "_capture", None):
            self.assertFalse(T.colours_enabled(own))

    def test_no_stream_or_a_closed_one_is_not_coloured(self):
        closed = self.Stream()
        closed.close()
        closed.isatty = mock.Mock(side_effect=ValueError("I/O operation on closed file"))
        self.assertFalse(T.colours_enabled(closed))
        with mock.patch.object(sys, "stdout", None):
            self.assertFalse(T.colours_enabled())
        self.assertFalse(T.colours_enabled(object()))  # no isatty at all

    def test_the_commands_that_colour_their_output_ask_this_question_and_have_no_copy_of_it(self):
        users = ("commands/cluster.py", "commands/subcluster.py", "commands/spectrum.py", "Alignment_Manager.py",
                 "Embedding_Manager.py", "utilities/Sequence_Utils.py")
        for name in users:
            with self.subTest(module=name):
                source = (SRC_DIR / name).read_text(encoding="utf-8")
                tree = ast.parse(source)
                imported = any(isinstance(node, ast.ImportFrom) and node.module == "utilities.Terminal_Colors"
                               and any(alias.name == "colours_enabled" for alias in node.names)
                               for node in ast.walk(tree))
                self.assertTrue(imported)
                self.assertNotIn("isatty", source)
        # Output that nothing but a terminal or the Console shows has no colour in it.
        for path in (SRC_DIR / "commands").glob("*.py"):
            self.assertNotIn("isatty", path.read_text(encoding="utf-8"), path.name)

    def test_a_command_colours_for_the_capture_and_not_for_a_buffer(self):
        from commands import cluster, spectrum

        own = self.Stream()
        capture = mock.Mock(text_streams=(own, self.Stream()))
        with mock.patch.object(Console_Capture, "_capture", capture):
            with mock.patch.object(sys, "stdout", own):
                self.assertIn(ESC + "[38;2;204;204;204m", cluster.get_colored_cluster_name(-1, "Noise"))
                self.assertIn(ESC + "[38;2;1;2;3m", spectrum._terminal_endpoint_text("min", 1, (1 / 255, 2 / 255, 3 / 255)))
            with mock.patch.object(sys, "stdout", self.Stream()):
                self.assertEqual(cluster.get_colored_cluster_name(-1, "Noise"), "● Noise")
                self.assertEqual(spectrum._terminal_endpoint_text("min", 1, (0, 0, 0)), "min: 1")


class ReadingColoursTests(unittest.TestCase):
    def test_text_with_no_sequence_is_one_run_and_nothing_is_one_run_fewer(self):
        self.assertEqual(runs("plain"), [("plain", DEFAULT_STYLE)])
        self.assertEqual(runs(""), [])
        self.assertEqual(T.parse_ansi("plain")[1], DEFAULT_STYLE)

    def test_the_sixteen_colours_and_bright_colours_and_their_defaults(self):
        found = runs(f"{ESC}[31mred{ESC}[92mbright green{ESC}[39mdefault{ESC}[44;37mon blue{ESC}[49mno bg{ESC}[104mbright bg")
        self.assertEqual([style.fg for _, style in found], [("index", 1), ("index", 10), None, ("index", 7), ("index", 7), ("index", 7)])
        self.assertEqual([style.bg for _, style in found], [None, None, None, ("index", 4), None, ("index", 12)])
        self.assertEqual("".join(text for text, _ in found), "redbright greendefaulton blueno bgbright bg")

    def test_24_bit_colour_foreground_and_background(self):
        found = runs(f"a{ESC}[38;2;10;20;30mb{ESC}[48;2;200;100;50mc{ESC}[0md")
        self.assertEqual([(text, style.fg, style.bg) for text, style in found], [
            ("a", None, None), ("b", ("rgb", 10, 20, 30), None),
            ("c", ("rgb", 10, 20, 30), ("rgb", 200, 100, 50)), ("d", None, None),
        ])

    def test_256_colour_palette_has_the_sixteen_a_cube_and_greys(self):
        self.assertEqual(runs(f"{ESC}[38;5;9mx")[0][1].fg, ("index", 9))
        self.assertEqual(runs(f"{ESC}[38;5;16mx")[0][1].fg, ("rgb", 0, 0, 0))
        self.assertEqual(runs(f"{ESC}[38;5;196mx")[0][1].fg, ("rgb", 255, 0, 0))
        self.assertEqual(runs(f"{ESC}[38;5;21mx")[0][1].fg, ("rgb", 0, 0, 255))
        self.assertEqual(runs(f"{ESC}[38;5;231mx")[0][1].fg, ("rgb", 255, 255, 255))
        self.assertEqual(runs(f"{ESC}[38;5;232mx")[0][1].fg, ("rgb", 8, 8, 8))
        self.assertEqual(runs(f"{ESC}[48;5;255mx")[0][1].bg, ("rgb", 238, 238, 238))

    def test_the_colon_forms_of_extended_colours(self):
        self.assertEqual(runs(f"{ESC}[38:2:1:2:3mx")[0][1].fg, ("rgb", 1, 2, 3))
        self.assertEqual(runs(f"{ESC}[38:2::4:5:6mx")[0][1].fg, ("rgb", 4, 5, 6))  # an empty colour space
        self.assertEqual(runs(f"{ESC}[48:5:196mx")[0][1].bg, ("rgb", 255, 0, 0))

    def test_bold_italic_underline_reverse_and_their_ends(self):
        style = end_style(f"{ESC}[1;3;4;7mx")
        self.assertEqual((style.bold, style.italic, style.underline, style.reverse), (True, True, True, True))
        style = end_style(f"{ESC}[1;3;4;7;22;23;24;27mx")
        self.assertEqual(style, DEFAULT_STYLE)

    def test_a_reset_in_the_middle_of_a_line_ends_every_attribute_at_once(self):
        found = runs(f"{ESC}[1;31;4mloud{ESC}[0m quiet {ESC}[32mgreen{ESC}[m plain")
        self.assertEqual([(text, style.fg, style.bold, style.underline) for text, style in found], [
            ("loud", ("index", 1), True, True), (" quiet ", None, False, False),
            ("green", ("index", 2), False, False), (" plain", None, False, False),
        ])

    def test_a_reset_with_no_number_or_an_empty_one_is_a_reset(self):
        for sequence in (f"{ESC}[m", f"{ESC}[0m", f"{ESC}[;m", f"{ESC}[00m"):
            with self.subTest(sequence=sequence):
                self.assertEqual(end_style(f"{ESC}[31;1mx{sequence}"), DEFAULT_STYLE)
        self.assertEqual(end_style(f"{ESC}[;1mx"), Style(bold=True))  # an empty first number is 0: reset, then bold

    def test_the_style_a_line_leaves_is_the_one_the_next_starts_in(self):
        _, left = T.parse_ansi(f"{ESC}[93mWARNING first line")
        found, left = T.parse_ansi("  second line", left)
        self.assertEqual(found, [("  second line", Style(fg=("index", 11)))])
        found, left = T.parse_ansi(f"  last line{ESC}[0m", left)
        self.assertEqual(found, [("  last line", Style(fg=("index", 11)))])
        self.assertEqual(left, DEFAULT_STYLE)

    def test_an_unknown_code_changes_nothing_and_keeps_the_codes_around_it(self):
        found = runs(f"{ESC}[1;9;53;999;31mx")
        self.assertEqual(found, [("x", Style(fg=("index", 1), bold=True))])
        self.assertEqual(runs(f"{ESC}[99mx"), [("x", DEFAULT_STYLE)])

    def test_malformed_sequences_are_dropped_and_their_numbers_are_not_read_as_codes(self):
        cases = {
            f"{ESC}[38;2;300;0;0mx": DEFAULT_STYLE,           # a channel out of range
            f"{ESC}[38;2;1;2mx": DEFAULT_STYLE,               # one channel short
            f"{ESC}[38;5mx": DEFAULT_STYLE,                   # no number
            f"{ESC}[38;5;256mx": DEFAULT_STYLE,               # a number out of range
            f"{ESC}[38mx": DEFAULT_STYLE,                     # nothing after it
            f"{ESC}[38;2;1;1;1;1mx": Style(fg=("rgb", 1, 1, 1), bold=True),  # a colour, then a code 1
            f"{ESC}[38;2;;0;0mx": DEFAULT_STYLE,              # an empty channel
            f"{ESC}[1;;31mx": Style(fg=("index", 1)),         # an empty number is a reset, in the middle too
            f"{ESC}[1;mx": Style(),                           # and at the end
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                found = runs(text)
                self.assertEqual("".join(part for part, _ in found), "x")
                self.assertEqual(found[0][1], expected)

    def test_a_sequence_with_a_letter_among_its_numbers_ends_at_the_letter_as_in_a_terminal(self):
        # "ESC [ 38 ; 2 ; x" is a complete (unknown) sequence ending in x: what follows is text.
        self.assertEqual(runs(f"{ESC}[38;2;x;0;0mq"), [(";0;0mq", DEFAULT_STYLE)])
        self.assertEqual(runs(f"{ESC}[abcmq"), [("bcmq", DEFAULT_STYLE)])

    def test_sequences_that_are_not_colours_are_dropped_without_a_trace(self):
        for text in (f"a{ESC}[2Kb", f"a{ESC}[1Ab", f"a{ESC}[10;20Hb", f"a{ESC}[?25lb", f"a{ESC}]0;a window title\x07b",
                     f"a{ESC}]8;;http://example.org{ESC}\\b", f"a{ESC}(Bb", f"a{ESC}cb", f"a{ESC}[0 mb"):
            with self.subTest(text=text):
                self.assertEqual("".join(part for part, _ in runs(text)), "ab")

    def test_a_sequence_cut_short_at_the_end_or_a_bare_escape_shows_nothing(self):
        for text in (f"ab{ESC}", f"ab{ESC}[", f"ab{ESC}[38;2;1", f"ab{ESC}]0;title", f"ab{ESC}(", f"a{ESC}b"):
            with self.subTest(text=text):
                self.assertNotIn(ESC, "".join(part for part, _ in runs(text)))
        self.assertEqual("".join(part for part, _ in runs(f"ab{ESC}[38;2;1")), "ab")

    def test_text_with_unicode_and_runs_are_never_empty(self):
        found = runs(f"{ESC}[31m●{ESC}[0m {ESC}[1m{ESC}[0m蛋白质")
        self.assertEqual("".join(text for text, _ in found), "● 蛋白质")
        self.assertTrue(all(text for text, _ in found))
        self.assertEqual(runs(f"{ESC}[31m"), [])


class TakingColoursOutTests(unittest.TestCase):
    SAMPLE = (
        f"plain {ESC}[1;31mred{ESC}[0m ● {ESC}[38;2;1;2;3mdot{ESC}[0m{ESC}[2K{ESC}]0;title\x07 end"
        f" {ESC}(B{ESC}c tail{ESC}["
    )

    def test_sequences_come_out_of_text_and_other_text_is_untouched(self):
        self.assertEqual(T.strip_ansi(self.SAMPLE), "plain red ● dot end  tail")
        self.assertEqual(T.strip_ansi("no escapes: [31m stays 蛋"), "no escapes: [31m stays 蛋")
        self.assertEqual(T.strip_ansi(""), "")

    def test_stripping_agrees_with_what_the_reader_shows(self):
        for text in (self.SAMPLE, f"{ESC}[31mx", f"{ESC}[38;2;1;2m{ESC}[0mq"):
            with self.subTest(text=text):
                self.assertEqual(T.strip_ansi(text), "".join(part for part, _ in runs(text)))

    def test_bytes_in_chunks_give_the_same_text_whatever_the_cuts(self):
        data = self.SAMPLE.encode("utf-8")
        expected = T.strip_ansi(self.SAMPLE).encode("utf-8")
        for size in (1, 2, 3, 5, 7, 64):
            with self.subTest(size=size):
                stripper = T.AnsiStripper()
                out = b"".join(stripper.feed(data[i:i + size]) for i in range(0, len(data), size)) + stripper.finish()
                self.assertEqual(out, expected)

    def test_every_single_cut_of_one_chunk_in_two(self):
        data = f"x{ESC}[38;2;10;20;30mdot{ESC}[0m●y{ESC}]0;t\x07z".encode("utf-8")
        for cut in range(len(data) + 1):
            stripper = T.AnsiStripper()
            out = stripper.feed(data[:cut]) + stripper.feed(data[cut:]) + stripper.finish()
            self.assertEqual(out, "xdot●yz".encode("utf-8"), cut)

    def test_a_sequence_that_waits_for_its_end_is_held_and_one_that_never_comes_is_dropped(self):
        stripper = T.AnsiStripper()
        self.assertEqual(stripper.feed(f"ab{ESC}[3".encode()), b"ab")
        self.assertEqual(stripper.feed(b"1mcd"), b"cd")
        self.assertEqual(stripper.feed(f"ef{ESC}[3".encode()), b"ef")
        self.assertEqual(stripper.finish(), b"")
        self.assertEqual(stripper.feed(b"gh"), b"gh")  # nothing is left over from before

    def test_a_long_run_of_junk_after_an_escape_is_not_held_for_ever(self):
        stripper = T.AnsiStripper()
        junk = f"{ESC}]" + "x" * (T.MAX_UNFINISHED_BYTES + 10)
        out = stripper.feed(junk.encode())
        self.assertNotIn(b"\x1b", out)  # passed through the stripper rather than waited for

    def test_text_with_no_escape_is_passed_on_whole(self):
        stripper = T.AnsiStripper()
        self.assertEqual(stripper.feed(b"line one\nline two\n"), b"line one\nline two\n")


class SplashDetailsTests(unittest.TestCase):
    def test_the_start_up_details_show_the_text_and_not_the_sequences(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from desktop.Splash import DetailsView

        app = QApplication.instance() or QApplication([])
        view = DetailsView()
        self.addCleanup(view.deleteLater)
        view.add_lines([(f"{ESC}[31mred{ESC}[0m text {ESC}[2K{ESC}[38;2;1;2;3m●{ESC}[0m", False), ("plain", False)])
        self.assertEqual(view.toPlainText(), "red text ●\nplain")
        self.assertIs(app, QApplication.instance())


class ReadableColoursTests(unittest.TestCase):
    PAGES = {"light": "#ffffff", "dark": "#18181b"}

    def test_contrast_ratio_is_the_wcag_one(self):
        self.assertAlmostEqual(T.contrast_ratio("#000000", "#ffffff"), 21.0, places=2)
        self.assertAlmostEqual(T.contrast_ratio("#777777", "#ffffff"), 4.48, places=2)
        self.assertEqual(T.contrast_ratio("#123456", "#123456"), 1.0)

    def test_a_colour_that_shows_is_left_alone(self):
        self.assertEqual(T.readable_color("#dc2626", "#ffffff"), "#dc2626")
        self.assertEqual(T.readable_color("#f87171", "#18181b"), "#f87171")

    def test_a_colour_that_vanishes_moves_away_from_the_page_and_keeps_its_hue(self):
        light = T.readable_color("#fde68a", "#ffffff")  # a pale amber on white
        self.assertGreaterEqual(T.contrast_ratio(light, "#ffffff"), T.MIN_CONTRAST)
        red, green, blue = (int(light[i:i + 2], 16) for i in (1, 3, 5))
        self.assertGreater(red, blue)  # still amber, only darker
        self.assertLess(red, 0xfd)
        dark = T.readable_color("#27272a", "#18181b")  # near-black on the dark page
        self.assertGreaterEqual(T.contrast_ratio(dark, "#18181b"), T.MIN_CONTRAST)
        self.assertGreater(int(dark[1:3], 16), 0x27)

    def test_the_noise_grey_the_cluster_commands_use_shows_on_both_pages(self):
        for page, background in self.PAGES.items():
            with self.subTest(page=page):
                shown = T.resolve_color(("rgb", 204, 204, 204), dark=page == "dark", background=background)
                self.assertGreaterEqual(T.contrast_ratio(shown, background), T.MIN_CONTRAST)

    def test_every_palette_entry_and_a_sweep_of_24_bit_colours_show_on_each_page(self):
        for page, background in self.PAGES.items():
            dark = page == "dark"
            for index in range(16):
                with self.subTest(page=page, index=index):
                    shown = T.resolve_color(("index", index), dark=dark, background=background)
                    self.assertGreaterEqual(T.contrast_ratio(shown, background), T.MIN_CONTRAST)
            for red in range(0, 256, 51):
                for green in range(0, 256, 51):
                    for blue in range(0, 256, 51):
                        shown = T.resolve_color(("rgb", red, green, blue), dark=dark, background=background)
                        self.assertGreaterEqual(T.contrast_ratio(shown, background), T.MIN_CONTRAST - 0.01, (page, red, green, blue))

    def test_a_colour_equal_to_its_page_goes_as_far_as_it_must(self):
        for page in ("#ffffff", "#000000", "#777777", "#18181b"):
            with self.subTest(page=page):
                shown = T.readable_color(page, page)
                self.assertGreaterEqual(T.contrast_ratio(shown, page), T.MIN_CONTRAST)

    def test_a_palette_entry_is_the_pages_own_shade_or_the_standard_one(self):
        self.assertEqual(T.palette_color(("index", 1), dark=False), T.PALETTE_LIGHT[1])
        self.assertEqual(T.palette_color(("index", 1), dark=True), T.PALETTE_DARK[1])
        self.assertEqual(T.palette_color(("index", 1), dark=None), T.PALETTE_STANDARD[1])
        self.assertEqual(T.palette_color(("rgb", 1, 2, 255), dark=False), "#0102ff")


if __name__ == "__main__":
    unittest.main()
