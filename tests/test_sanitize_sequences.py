"""Sanitize_Sequences tool: its own rules, output writing and command-line runs.

The header and sequence rules it re-imports from utilities.Sequence_Utils are
covered in test_fasta_sanitization.py.
"""
import contextlib
import io
import json
import os
import pathlib
import re
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = PROJECT_ROOT / "src" / "tools" / "Sanitize_Sequences.py"
# A package import, so the tests-package hook keeps <project>/tools_settings.json
# out of the module's import-time settings.
from tools import Sanitize_Sequences as sanitize_sequences  # noqa: E402


def _tool_environment(temp_dir):
    """Subprocess environment whose import-time settings read finds no file.

    A tool reads <project>/tools_settings.json when it is imported unless
    SSN_TOOL_SETTINGS_SCRIPT/FILE name another file; main() then applies the
    file given on the command line.
    """
    return {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "SSN_TOOL_SETTINGS_SCRIPT": "Sanitize_Sequences.py",
        "SSN_TOOL_SETTINGS_FILE": os.path.join(temp_dir, "no-shared-settings.json"),
    }


class SanitizeSequencesTests(unittest.TestCase):
    @staticmethod
    def printed_table(lengths):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sanitize_sequences.print_length_distribution(lengths)
        lines = output.getvalue().splitlines()
        rows = lines[lines.index("-" * 57) + 1:-1]
        return lines, [(row.split("|")[0].strip(), int(row.split("|")[1])) for row in rows]

    def test_length_table_matches_histogram_counts(self):
        # Lengths spanning more than 50 values: 50 bins of equal width.
        for lengths in ([1, 2, 2, 26, 51], [1, 51]):
            with self.subTest(lengths=lengths):
                lines, rows = self.printed_table(lengths)
                counts, edges = sanitize_sequences.np.histogram(lengths, bins=50)
                self.assertEqual(len(rows), 50)
                self.assertEqual([count for _, count in rows], counts.tolist())
                self.assertTrue(rows[0][0].startswith(f"[{edges[0]:.12g}, "))
                self.assertIn(f", {edges[-1]:.12g}]", rows[-1][0])
                self.assertIn("Length bin (aa)", "\n".join(lines))

    def test_lengths_spanning_50_values_or_fewer_get_a_row_each(self):
        # One row per length, empty lengths between included, up to a span of 50.
        cases = (
            ([7, 8, 8, 9, 9, 9, 10, 10, 12, 12, 12, 12],
             [("7", 1), ("8", 2), ("9", 3), ("10", 2), ("11", 0), ("12", 4)]),
            ([10, 10], [("10", 2)]),
            ([7], [("7", 1)]),
            (list(range(1, 51)), [(str(length), 1) for length in range(1, 51)]),
        )
        for lengths, expected in cases:
            with self.subTest(lengths=lengths[:6]):
                lines, rows = self.printed_table(lengths)
                self.assertEqual(rows, expected)
                self.assertIn("one row per length", "\n".join(lines))
                counts, edges, one_per_length = sanitize_sequences.length_bins(lengths)
                self.assertTrue(one_per_length)
                self.assertEqual(counts.tolist(), [count for _, count in expected])
                # Each bin runs from half a residue below its length to half above.
                self.assertEqual(edges[0], int(expected[0][0]) - 0.5)
                self.assertEqual(edges[-1], int(expected[-1][0]) + 0.5)
        self.assertEqual(sanitize_sequences.HISTOGRAM_BINS, 50)

    def test_mcp_headless_invocation_never_imports_matplotlib(self):
        from tools.tool_helpers.Tool_Pipeline import prepare_headless_invocation

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = pathlib.Path(temp_dir) / "input.fasta"
            input_path.write_text(">short\nAC\n>long\nACDEF\n", encoding="utf-8")
            invocation = prepare_headless_invocation(
                "sanitize_sequences",
                {
                    "DIRECTORIES": {"FASTA_DIR": temp_dir},
                    "Sanitize_Sequences.py": {
                        "INPUT_FASTA": str(input_path),
                        "OVER_WRITE": False,
                        "ENABLE_LENGTH_FILTER": False,
                        "REMOVE_BY_HEADER_STRING": "",
                    },
                },
                PROJECT_ROOT,
                snapshot_directory=temp_dir,
            )
            # Block even importing the plotting package and Qt in a fresh interpreter.
            code = (
                "import runpy, sys; "
                "sys.modules['matplotlib'] = None; sys.modules['PySide6'] = None; "
                f"sys.path.insert(0, {str(MODULE_PATH.parent)!r}); "
                f"sys.argv = {list(invocation.argv[2:])!r}; "
                f"runpy.run_path({str(MODULE_PATH)!r}, run_name='__main__')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=invocation.cwd,
                env=_tool_environment(temp_dir),
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Frequency (sequences)", result.stdout)
            self.assertNotIn("Opening length distribution histogram", result.stdout)
            self.assertEqual(
                (pathlib.Path(temp_dir) / "input_sanitized.fasta").read_text(),
                ">short\nAC\n>long\nACDEF\n",
            )

    def test_utf8_inputs_match_downstream_and_write_without_bom(self):
        from utilities.Sequence_Utils import load_sanitized_fasta

        for encoding in ("utf-8", "utf-8-sig"):
            for overwrite in (False, True):
                with self.subTest(encoding=encoding, overwrite=overwrite):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        input_path = pathlib.Path(temp_dir) / "input.fasta"
                        input_path.write_text(
                            ">long descriptive header\nACDE\n"
                            ">short\nACDE\n>unique\nFGHI\n",
                            encoding=encoding,
                        )
                        expected_headers, expected_sequences, _ = load_sanitized_fasta(
                            input_path, report=False,
                        )
                        settings_path = pathlib.Path(temp_dir) / "settings.json"
                        settings_path.write_text(json.dumps({
                            "DIRECTORIES": {"FASTA_DIR": temp_dir},
                            "Sanitize_Sequences.py": {
                                "INPUT_FASTA": str(input_path),
                                "OVER_WRITE": overwrite,
                                "ENABLE_LENGTH_FILTER": False,
                                "REMOVE_BY_HEADER_STRING": "",
                            },
                        }), encoding="utf-8")

                        result = subprocess.run(
                            [sys.executable, "-B", str(MODULE_PATH), str(settings_path)],
                            cwd=PROJECT_ROOT,
                            env=_tool_environment(temp_dir),
                            capture_output=True, text=True, encoding="utf-8", timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        output_path = (
                            input_path if overwrite
                            else pathlib.Path(temp_dir) / "input_sanitized.fasta"
                        )
                        output_bytes = output_path.read_bytes()
                        self.assertFalse(output_bytes.startswith(b"\xef\xbb\xbf"))
                        self.assertEqual(
                            output_bytes.decode("utf-8"),
                            ">long_descriptive_header\nACDE\n>unique\nFGHI\n",
                        )
                        self.assertEqual(
                            sanitize_sequences.read_fasta(output_path),
                            (expected_headers, expected_sequences),
                        )

    def test_header_filter_is_case_sensitive_and_none_is_literal(self):
        self.assertTrue(
            sanitize_sequences.should_remove_by_header(
                "protein with None assigned", "None"
            )
        )
        self.assertFalse(
            sanitize_sequences.should_remove_by_header(
                "protein with none assigned", "None"
            )
        )
        self.assertTrue(
            sanitize_sequences.should_remove_by_header(
                "protein with None assigned", None
            )
        )
        self.assertFalse(
            sanitize_sequences.should_remove_by_header("protein", "   ")
        )

    def test_configuration_requires_an_explicit_output_directory(self):
        with self.assertRaisesRegex(ValueError, "output directory"):
            sanitize_sequences.validate_configuration(
                None,
                "input.fasta",
                "output.fasta",
            )

    def test_empty_destructive_overwrite_is_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "input.fasta"
            output_path.write_text(">original\nAAAA\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "zero output sequences"):
                sanitize_sequences.write_fasta_atomic(
                    output_path,
                    [],
                    [],
                    refuse_empty=True,
                )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                ">original\nAAAA\n",
            )

    def test_failed_atomic_replace_preserves_existing_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "input.fasta"
            output_path.write_text(">original\nAAAA\n", encoding="utf-8")

            with mock.patch.object(
                sanitize_sequences.os,
                "replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    sanitize_sequences.write_fasta_atomic(
                        output_path,
                        ["replacement"],
                        ["CCCC"],
                        refuse_empty=True,
                    )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                ">original\nAAAA\n",
            )
            self.assertEqual(list(pathlib.Path(temp_dir).glob("*.tmp")), [])


class LengthHistogramSettingTests(unittest.TestCase):
    """SHOW_LENGTH_HISTOGRAM turns off the pop-up and the printed table alike.

    The length statistics are printed either way. Each run executes the
    tool's module code afresh as if the script lived in a stand-in project,
    as a script run starts. An interactive run reads
    <project>/tools_settings.json; a headless run is given a settings file.
    """

    SKIPPED = "Length histogram skipped (SHOW_LENGTH_HISTOGRAM is off)."
    NOTHING_LEFT = "No sequences passed the length filters. Histogram skipped."
    TABLE = "Frequency (sequences)"
    OPENING = "Opening length distribution histogram"
    STATISTICS = (
        "Length statistics: 2 sequences · min 2 · median 3.5 · mean 3.5 · SD 1.5 · max 5 aa"
    )

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = pathlib.Path(temp_dir.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.fasta_dir = self.root / "data"
        self.fasta_dir.mkdir()
        (self.fasta_dir / "input.fasta").write_text(
            ">short\nAC\n>long\nACDEF\n", encoding="utf-8"
        )

    def write_settings(self, path, show=None, **tool_settings):
        """Write a settings document; show=None leaves the switch at its default."""
        settings = {"INPUT_FASTA": "input.fasta", **tool_settings}
        if show is not None:
            settings["SHOW_LENGTH_HISTOGRAM"] = show
        path.write_text(json.dumps({
            "DIRECTORIES": {"FASTA_DIR": str(self.fasta_dir)},
            "Sanitize_Sequences.py": settings,
        }), encoding="utf-8")
        return path

    def load_tool(self):
        """The tool's module code, run afresh as a copy in the stand-in project.

        It then reads no settings file when it is run, as in a script run.
        """
        script = self.project / "src" / "tools" / "Sanitize_Sequences.py"
        namespace = {"__name__": "sanitize_sequences_run", "__file__": str(script)}
        code = compile(MODULE_PATH.read_text(encoding="utf-8"), str(script), "exec")
        with mock.patch.dict(os.environ, _tool_environment(str(self.root))):
            exec(code, namespace)
        return namespace

    def run_tool(self, argv):
        """Run main(argv) on a fresh copy; return its stdout and the plot stand-in."""
        namespace = self.load_tool()
        plot = mock.Mock()
        namespace["plot_length_distribution"] = plot
        output = io.StringIO()
        with mock.patch.dict(os.environ, _tool_environment(str(self.root))), \
                contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(namespace["main"](argv), 0)
        return output.getvalue(), plot

    def test_an_interactive_run_shows_the_pop_up_unless_switched_off(self):
        settings = self.project / "tools_settings.json"
        for show in (None, True, False):
            with self.subTest(show=show):
                self.write_settings(settings, show)
                output, plot = self.run_tool([])
                self.assertNotIn(self.TABLE, output)
                self.assertIn(self.STATISTICS, output)
                if show is False:
                    plot.assert_not_called()
                    self.assertIn(self.SKIPPED, output)
                    self.assertNotIn(self.OPENING, output)
                else:
                    plot.assert_called_once_with([2, 5], [])
                    self.assertIn(self.OPENING, output)
                    self.assertNotIn(self.SKIPPED, output)

    def test_a_headless_run_prints_the_table_unless_switched_off(self):
        settings = self.root / "settings.json"
        for show in (None, True, False):
            with self.subTest(show=show):
                self.write_settings(settings, show)
                output, plot = self.run_tool([str(settings)])
                plot.assert_not_called()
                self.assertNotIn(self.OPENING, output)
                self.assertIn(self.STATISTICS, output)
                if show is False:
                    self.assertNotIn(self.TABLE, output)
                    self.assertIn(self.SKIPPED, output)
                else:
                    self.assertLess(output.index(self.STATISTICS), output.index(self.TABLE))
                    self.assertNotIn(self.SKIPPED, output)

    def test_no_sequences_left_keeps_its_own_message(self):
        settings = self.root / "settings.json"
        for show in (True, False):
            with self.subTest(show=show):
                self.write_settings(
                    settings, show, ENABLE_LENGTH_FILTER=True, MIN_SEQ_LENGTH=10,
                )
                output, plot = self.run_tool([str(settings)])
                plot.assert_not_called()
                self.assertIn(self.NOTHING_LEFT, output)
                self.assertNotIn("Length statistics", output)
                self.assertNotIn(self.SKIPPED, output)
                self.assertNotIn(self.TABLE, output)

    def test_the_pop_up_marks_the_length_filters_bounds_only_while_it_is_on(self):
        settings = self.project / "tools_settings.json"
        cases = (
            ({"ENABLE_LENGTH_FILTER": True, "MIN_SEQ_LENGTH": 1, "MAX_SEQ_LENGTH": 10},
             [("min", 1), ("max", 10)]),
            ({"ENABLE_LENGTH_FILTER": True, "MIN_SEQ_LENGTH": 0, "MAX_SEQ_LENGTH": 10},
             [("max", 10)]),
            ({"ENABLE_LENGTH_FILTER": False, "MIN_SEQ_LENGTH": 1, "MAX_SEQ_LENGTH": 10}, []),
        )
        for tool_settings, bounds in cases:
            with self.subTest(**tool_settings):
                self.write_settings(settings, **tool_settings)
                _, plot = self.run_tool([])
                plot.assert_called_once_with([2, 5], bounds)

    def test_the_pop_up_falls_back_to_matplotlibs_font_without_the_bundled_one(self):
        # The stand-in project has no src/resources/fonts.
        self.assertEqual(self.load_tool()["_plot_font_families"](), ["DejaVu Sans"])
        self.assertEqual(sanitize_sequences._plot_font_families(), ["Noto Sans"])


class LengthStatisticsTests(unittest.TestCase):
    def test_the_line_gives_count_min_median_mean_sd_and_max(self):
        for lengths, line in (
            ([2, 5], "2 sequences · min 2 · median 3.5 · mean 3.5 · SD 1.5 · max 5 aa"),
            ([1, 2, 2, 26, 51], "5 sequences · min 1 · median 2 · mean 16.4 · SD 19.7 · max 51 aa"),
            ([1000, 3000], "2 sequences · min 1,000 · median 2,000 · mean 2,000.0 · "
                           "SD 1,000.0 · max 3,000 aa"),
            ([7], "1 sequence · min 7 · median 7 · mean 7.0 · SD 0.0 · max 7 aa"),
        ):
            with self.subTest(lengths=lengths):
                self.assertEqual(str(sanitize_sequences.length_statistics(lengths)), line)


class LengthHistogramChartTests(unittest.TestCase):
    """The pop-up chart, drawn under Agg with plt.show standing in for the window."""

    LENGTHS = [1, 2, 2, 26, 51, 51, 51, 40, 12, 12]

    @classmethod
    def setUpClass(cls):
        import matplotlib
        import matplotlib.pyplot as plt

        cls.plt = plt
        previous = matplotlib.get_backend(auto_select=False)
        plt.switch_backend("agg")
        # A process that had chosen no backend yet keeps Agg.
        if previous:
            cls.addClassCleanup(plt.switch_backend, previous)

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.app_settings = pathlib.Path(temp_dir.name) / "app_settings.json"

    def draw(self, lengths=LENGTHS, bounds=(), color=None, saved=None, language="en",
             environment=None):
        """The figure plt.show was given, drawn; color is the Tools window's saved one.

        language is the saved LANGUAGE, English unless a test asks, whatever the
        system's; environment adds variables, as SSN_PSEUDO_TRANSLATION. saved,
        when given, is the whole app_settings.json text instead. The warnings
        raised while drawing are left in self.warnings.
        """
        if saved is None:
            settings = {"LANGUAGE": language}
            if color is not None:
                settings["TOOLS_THEME_COLOR"] = color
            saved = json.dumps(settings)
        self.app_settings.write_text(saved, encoding="utf-8")
        shown = []

        def show(*args, **kwargs):
            figure = self.plt.gcf()
            figure.canvas.draw()
            shown.append(figure)
            # Closing the figure takes its window (manager) away.
            self.window_title = figure.canvas.manager.get_window_title()

        variables = {"SSN_APP_SETTINGS_PATH": str(self.app_settings), **(environment or {})}
        with mock.patch.dict(os.environ, variables), \
                mock.patch.object(self.plt, "show", side_effect=show), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sanitize_sequences.plot_length_distribution(lengths, bounds)
        self.warnings = [str(warning.message) for warning in caught]
        self.assertEqual(len(shown), 1)
        self.assertFalse(self.plt.fignum_exists("Sequence Length Distribution"))
        return shown[0]

    @staticmethod
    def bars(figure):
        from matplotlib.patches import FancyBboxPatch

        return [patch for patch in figure.axes[0].patches if isinstance(patch, FancyBboxPatch)]

    @staticmethod
    def corner_pixels(bar):
        """How far the bar's lower-left rounded corner reaches along its base, in pixels."""
        transform = bar.axes.transData.transform
        start = transform(bar.get_path().vertices[0])
        corner = transform((bar.get_x(), bar.get_y()))
        return start[0] - corner[0]

    def test_a_few_lengths_get_a_bar_each_over_its_length(self):
        lengths = [7, 8, 8, 9, 9, 9, 10, 10, 12, 12, 12, 12]
        bars = self.bars(self.draw(lengths))
        self.assertEqual([bar.get_height() for bar in bars], [1, 2, 3, 2, 4])
        for bar, length in zip(bars, (7, 8, 9, 10, 12)):
            self.assertAlmostEqual(bar.get_x() + bar.get_width() / 2, length)
            self.assertTrue(0.7 < bar.get_width() < 1)

    def test_one_rounded_bar_per_filled_bin_with_a_gap_between(self):
        from matplotlib.patches import BoxStyle
        from matplotlib.path import Path

        counts, edges = sanitize_sequences.np.histogram(self.LENGTHS, bins=50)
        figure = self.draw()
        bars = self.bars(figure)
        self.assertEqual(len(bars), sanitize_sequences.np.count_nonzero(counts))
        self.assertEqual([bar.get_height() for bar in bars], [int(c) for c in counts if c])
        width = edges[1] - edges[0]
        for bar in bars:
            self.assertIsInstance(bar.get_boxstyle(), BoxStyle.Round)
            self.assertIn(Path.CURVE3, bar.get_path().codes)
            self.assertTrue(0.7 * width < bar.get_width() < width)
        # The corners are the report's 3 px (2.25 pt) on screen, whatever the
        # figure's size, but never more than half a bar's side.
        tallest = max(bars, key=lambda bar: bar.get_height())
        expected = 2.25 * figure.dpi / 72
        self.assertAlmostEqual(self.corner_pixels(tallest), expected, delta=0.05)
        figure.set_size_inches(5, 3)
        figure.canvas.draw()
        self.assertAlmostEqual(self.corner_pixels(tallest), expected, delta=0.05)
        for bar in bars:
            self.assertLessEqual(self.corner_pixels(bar), expected + 0.05)

    def test_the_bars_take_the_tools_windows_theme_colour(self):
        from matplotlib.colors import to_rgba
        from desktop.Studio_Theme import theme_tokens

        colours = {}
        for color in ("teal", "purple", "black"):
            with self.subTest(color=color):
                bar = self.bars(self.draw(color=color))[0]
                tokens = theme_tokens(color)
                self.assertEqual(bar.get_facecolor(), to_rgba(tokens["separator"]))
                self.assertEqual(bar.get_edgecolor(), to_rgba(tokens["accent"]))
                colours[color] = bar.get_facecolor()
        self.assertEqual(len(set(colours.values())), 3)
        # No saved colour, a colour the program lacks, or an unreadable file: the default.
        for saved in ("{}", '{"TOOLS_THEME_COLOR": "plaid"}', '{"TOOLS_THEME_COLOR": ["teal"]}', "{"):
            with self.subTest(saved=saved):
                bar = self.bars(self.draw(saved=saved))[0]
                self.assertEqual(bar.get_facecolor(), colours["purple"])

    def test_the_setting_name_is_the_tools_windows(self):
        source = (SRC_DIR / "desktop" / "Desktop_App.py").read_text(encoding="utf-8")
        self.assertIn(
            f'TOOLS_THEME_SETTING = "{sanitize_sequences._TOOLS_THEME_SETTING}"', source
        )

    def test_light_axes_a_titled_window_and_the_statistics_under_the_title(self):
        from matplotlib.colors import to_rgba
        from matplotlib.text import Text
        from desktop.Studio_Theme import theme_tokens

        tokens = theme_tokens("teal")
        figure = self.draw(color="teal")
        axes = figure.axes[0]
        self.assertEqual(self.window_title, "Sequence Length Distribution")
        for side in ("top", "right"):
            self.assertFalse(axes.spines[side].get_visible(), side)
        self.assertTrue(axes.spines["bottom"].get_visible())
        self.assertTrue(all(line.get_visible() for line in axes.yaxis.get_gridlines()))
        self.assertFalse(any(line.get_visible() for line in axes.xaxis.get_gridlines()))
        self.assertTrue(axes.get_axisbelow())
        texts = {text.get_text(): text for text in figure.findobj(Text) if text.get_text()}
        title = texts["Sequence Length Distribution"]
        self.assertEqual(to_rgba(title.get_color()), to_rgba(tokens["text"]))
        self.assertEqual(title.get_fontname(), "Noto Sans")
        statistics = texts[str(sanitize_sequences.length_statistics(self.LENGTHS))]
        self.assertEqual(to_rgba(statistics.get_color()), to_rgba(tokens["text_muted"]))
        self.assertEqual(to_rgba(axes.xaxis.label.get_color()), to_rgba(tokens["text_muted"]))

    def test_filter_bounds_are_dashed_lines_with_labels_only_when_given(self):
        figure = self.draw(bounds=[("min", -10), ("max", 80)])
        axes = figure.axes[0]
        self.assertEqual(
            sorted(tuple(line.get_xdata()) for line in axes.lines), [(-10, -10), (80, 80)]
        )
        self.assertTrue(all(line.get_linestyle() == "--" for line in axes.lines))
        low, high = axes.get_xlim()
        self.assertLess(low, -10)
        self.assertGreater(high, 80)
        labels = {text.get_text() for text in axes.texts}
        self.assertLessEqual({"min -10", "max 80"}, labels)

        axes = self.draw().axes[0]
        self.assertEqual(list(axes.lines), [])
        self.assertFalse(any(text.get_text().startswith(("min ", "max ")) for text in axes.texts))


    # The chart's texts: its window's name, title, statistics line, axis labels and
    # the filter bounds' labels, six in all; the rest are the axes' numbers.
    @staticmethod
    def words(figure):
        from matplotlib.text import Text

        shown = [text.get_text() for text in figure.findobj(Text) if text.get_text()]
        return [text for text in shown if not re.fullmatch(r"[\d,.\-]+", text)]

    def test_every_text_the_chart_shows_goes_through_the_catalog(self):
        # The pseudo-language brackets every catalog text, so one left plain skipped it.
        from utilities.Localization import is_pseudo_translated

        figure = self.draw(bounds=[("min", 1), ("max", 50)],
                           environment={"SSN_PSEUDO_TRANSLATION": "1"})
        words = self.words(figure)
        self.assertEqual(len(words), 6, words)
        for text in words + [self.window_title]:
            self.assertTrue(is_pseudo_translated(text), text)
        self.assertEqual(self.warnings, [])

    def test_simplified_chinese_draws_in_its_bundled_font_with_no_glyph_missing(self):
        # Nor a face: matplotlib logs one a family lacks, as Noto Sans SC lacks semibold.
        with self.assertNoLogs("matplotlib", level="WARNING"):
            figure = self.draw(bounds=[("min", 1), ("max", 50)], language="zh_CN")
        title = figure.axes[0].get_title(loc="left")
        self.assertEqual(self.window_title, title)
        for text in self.words(figure):
            self.assertRegex(text, "[\u4e00-\u9fff]", "every text shows in Chinese")
        self.assertEqual(sanitize_sequences._plot_font_families("zh_CN"), ["Noto Sans", "Noto Sans SC"])
        # matplotlib warns of every character its fonts lack; with Noto Sans alone it does.
        self.assertEqual(self.warnings, [])
        with mock.patch.object(sanitize_sequences, "_plot_font_families", return_value=["Noto Sans"]):
            self.draw(language="zh_CN")
        self.assertTrue(any("missing from font" in warning for warning in self.warnings), self.warnings)

    def test_the_terminal_line_stays_english_while_the_chart_shows_the_language(self):
        from desktop.Desktop_App import message_translator
        from utilities import Localization

        statistics = sanitize_sequences.length_statistics(self.LENGTHS)
        previous = Localization.set_translator(message_translator("zh_CN"))
        try:
            self.assertEqual(
                str(statistics),
                "10 sequences · min 1 · median 19 · mean 24.8 · SD 20.6 · max 51 aa",
            )
            self.assertRegex(statistics.display(), "^10 [\u4e00-\u9fff]")
        finally:
            Localization.set_translator(previous)
        # The chart puts back whatever translator it found.
        self.draw(language="zh_CN")
        self.assertIs(Localization._translator, previous)

    def test_a_language_that_cannot_load_leaves_the_chart_english(self):
        from desktop import Desktop_App

        english = "Sequence Length Distribution"
        with mock.patch.object(Desktop_App, "message_translator", side_effect=LookupError("no catalog")), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            figure = self.draw(language="zh_CN")
        self.assertEqual(figure.axes[0].get_title(loc="left"), english)
        self.assertIn("stays English (no catalog)", printed.getvalue())
        # A language without a catalog shows English, as the windows do.
        self.assertEqual(self.draw(language="xx").axes[0].get_title(loc="left"), english)

if __name__ == "__main__":
    unittest.main()
