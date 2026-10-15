"""Embedding_MSA's isotonic regression chart: when a run shows it, and how it looks."""

import contextlib
import gc
import io
import json
import os
import pathlib
import re
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
from sklearn.isotonic import IsotonicRegression

from tests.msa_fixtures import (
    GuideTreeBuilt,
    InProcessPool,
    NetworkFixture,
    builder_settings,
)
import Embedding_MSA  # noqa: E402  msa_fixtures puts src/tools on the path


OPENING = "Displaying Isotonic Regression plot."
FAILED = "Could not open plot window: "


class RegressionPlotRunTests(unittest.TestCase):
    """When run_msa_builder shows the chart, and what it gives it."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = pathlib.Path(temp_dir.name)

    def run_builder(self, fixture, plot, **settings):
        """Run the builder up to its guide tree with plot standing in; return its stdout."""
        output = io.StringIO()
        with mock.patch.multiple(
            Embedding_MSA,
            benchmark_msa_devices=mock.Mock(side_effect=GuideTreeBuilt),
            plot_regression_fit=plot,
            **builder_settings(fixture, **settings),
        ), mock.patch.object(Embedding_MSA.mp, "set_start_method"), mock.patch.object(
            Embedding_MSA.mp, "Pool", InProcessPool
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(GuideTreeBuilt):
                Embedding_MSA.run_msa_builder()
        # Release the builder's HDF5 handles before the folder is removed.
        gc.collect()
        return output.getvalue()

    def test_a_sparse_run_shows_the_fit_over_the_pairs_it_was_made_on(self):
        fixture = NetworkFixture(self.directory, complete=False, seed=2)
        plot = mock.Mock()
        output = self.run_builder(fixture, plot, SHOW_REGRESSION_PLOT=True)

        plot.assert_called_once()
        iso_reg, x, y, rho, r2, pairs = plot.call_args.args
        self.assertIsInstance(iso_reg, IsotonicRegression)
        # Under 100,000 pairs, the fit takes them all, and so does the chart.
        self.assertEqual(len(x), pairs)
        self.assertEqual(len(y), pairs)
        self.assertGreater(pairs, 0)
        # The fit was made on these very pairs.
        self.assertEqual(float(iso_reg.X_min_), float(np.min(x)))
        self.assertEqual(float(iso_reg.X_max_), float(np.max(x)))
        np.testing.assert_array_equal(
            iso_reg.predict(x), IsotonicRegression(out_of_bounds="clip").fit(x, y).predict(x)
        )
        self.assertIn(f"Spearman Rank Correlation (rho): {rho:.4f}", output)
        self.assertIn(f"Coefficient of Determination (R^2): {r2:.4f}", output)
        self.assertIn(OPENING, output)

    def test_the_switch_off_or_a_complete_network_shows_nothing(self):
        for complete, show in ((False, False), (True, True)):
            with self.subTest(complete=complete, show=show):
                fixture = NetworkFixture(self.directory / f"{complete}{show}", complete=complete, seed=2)
                plot = mock.Mock()
                output = self.run_builder(fixture, plot, SHOW_REGRESSION_PLOT=show)
                plot.assert_not_called()
                self.assertNotIn(OPENING, output)

    def test_a_chart_that_fails_leaves_the_run_going(self):
        fixture = NetworkFixture(self.directory, complete=False, seed=2)
        plot = mock.Mock(side_effect=RuntimeError("no display"))
        # The run still reaches its guide tree (run_builder asserts it).
        output = self.run_builder(fixture, plot, SHOW_REGRESSION_PLOT=True)
        self.assertIn(FAILED + "no display", output)


class RegressionStatisticsTests(unittest.TestCase):
    def test_the_line_gives_rho_r2_and_the_pairs(self):
        line = Embedding_MSA.regression_statistics(0.33541, 0.16638, 2_345_678)
        self.assertEqual(str(line), "Spearman ρ 0.3354 · R² 0.1664 · 2,345,678 observed pairs")
        line = Embedding_MSA.regression_statistics(float("nan"), 1.0, 1)
        self.assertEqual(str(line), "Spearman ρ nan · R² 1.0000 · 1 observed pair")

    def test_the_colour_bar_ticks(self):
        for top, ticks in ((2, [1, 2]), (9, [1, 9]), (10, [1, 10]), (99, [1, 10]),
                           (5000, [1, 10, 100, 1000])):
            with self.subTest(top=top):
                self.assertEqual(Embedding_MSA._count_ticks(top), ticks)


class RegressionChartTests(unittest.TestCase):
    """The pop-up chart, drawn under Agg with plt.show standing in for the window."""

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
        rng = np.random.default_rng(3)
        cls.x = rng.uniform(0.6, 1.0, 400).astype(np.float32)
        cls.y = (2 + 6 * (cls.x - 0.6) ** 2 + rng.normal(0, 0.5, 400)).astype(np.float32)
        cls.iso = IsotonicRegression(out_of_bounds="clip").fit(cls.x, cls.y)

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.app_settings = pathlib.Path(temp_dir.name) / "app_settings.json"

    def draw(self, pairs=400, color=None, saved=None, language="en", environment=None):
        """The figure plt.show was given, drawn; color is the Tools window's saved one.

        pairs counts the observed pairs, of which the class's 400 are a sample
        when it is larger. language is the saved LANGUAGE, English unless a
        test asks, whatever the system's; environment adds variables, as
        SSN_PSEUDO_TRANSLATION. saved, when given, is the whole
        app_settings.json text instead. The warnings raised while drawing
        are left in self.warnings.
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
            Embedding_MSA.plot_regression_fit(self.iso, self.x, self.y, 0.5, 0.25, pairs)
        self.warnings = [str(warning.message) for warning in caught]
        self.assertEqual(len(shown), 1)
        self.assertFalse(self.plt.fignum_exists("Isotonic Regression Fit"))
        return shown[0]

    @staticmethod
    def hexagons(figure):
        from matplotlib.collections import PolyCollection

        cells = [item for item in figure.axes[0].collections if isinstance(item, PolyCollection)]
        assert len(cells) == 1, cells
        return cells[0]

    def test_the_hexagons_count_every_pair_given(self):
        cells = self.hexagons(self.draw())
        counts = np.asarray(cells.get_array())
        self.assertEqual(int(counts.sum()), len(self.x))
        self.assertGreaterEqual(counts.min(), 1)
        self.assertEqual(cells.get_clim(), (1, max(2, int(counts.max()))))
        self.assertGreater(len(counts), 50, "the pairs spread over many hexagons")

    def test_the_shades_take_the_tools_windows_theme_colour(self):
        from matplotlib.colors import to_rgba
        from desktop.Studio_Theme import theme_tokens

        shades = {}
        for color in ("teal", "purple", "black"):
            with self.subTest(color=color):
                cells = self.hexagons(self.draw(color=color))
                tokens = theme_tokens(color)
                self.assertEqual(cells.cmap(0.0), to_rgba(tokens["histogram"]))
                self.assertEqual(cells.cmap(1.0), to_rgba(tokens["accent_text"]))
                shades[color] = cells.cmap(0.5)
        self.assertEqual(len(set(shades.values())), 3)
        # No saved colour, a colour the program lacks, or an unreadable file: the default.
        for saved in ("{}", '{"TOOLS_THEME_COLOR": "plaid"}', "{"):
            with self.subTest(saved=saved):
                self.assertEqual(self.hexagons(self.draw(saved=saved)).cmap(0.5), shades["purple"])

    def test_the_fit_is_drawn_through_its_breakpoints_over_a_ring(self):
        from matplotlib import patheffects
        from matplotlib.colors import to_rgba
        from desktop.Studio_Theme import theme_tokens

        tokens = theme_tokens("teal")
        figure = self.draw(color="teal")
        (line,) = figure.axes[0].lines
        np.testing.assert_array_equal(line.get_xdata(), self.iso.X_thresholds_)
        np.testing.assert_array_equal(line.get_ydata(), self.iso.y_thresholds_)
        # Between its breakpoints the fit predicts linearly, as the line runs.
        middle = (self.iso.X_thresholds_[:-1] + self.iso.X_thresholds_[1:]) / 2
        np.testing.assert_allclose(
            np.interp(middle, line.get_xdata(), line.get_ydata()), self.iso.predict(middle), rtol=1e-5
        )
        self.assertEqual(to_rgba(line.get_color()), to_rgba(tokens["text"]))
        ring, plain = line.get_path_effects()
        self.assertIsInstance(ring, patheffects.Stroke)
        self.assertIsInstance(plain, patheffects.Normal)
        self.assertEqual(to_rgba(ring._gc["foreground"]), to_rgba(tokens["surface"]))
        self.assertGreater(ring._gc["linewidth"], line.get_linewidth())
        self.assertGreater(line.get_zorder(), self.hexagons(figure).get_zorder())

    def test_title_statistics_legend_and_window_name(self):
        from matplotlib.colors import to_rgba
        from matplotlib.text import Text
        from desktop.Studio_Theme import theme_tokens

        tokens = theme_tokens("teal")
        figure = self.draw(pairs=2_000, color="teal")
        axes = figure.axes[0]
        self.assertEqual(self.window_title, "Isotonic Regression Fit")
        texts = {text.get_text(): text for text in figure.findobj(Text) if text.get_text()}
        title = texts["Isotonic Regression Fit"]
        self.assertEqual(to_rgba(title.get_color()), to_rgba(tokens["text"]))
        self.assertEqual(title.get_fontname(), "Noto Sans")
        statistics = texts[str(Embedding_MSA.regression_statistics(0.5, 0.25, 2_000))]
        self.assertEqual(to_rgba(statistics.get_color()), to_rgba(tokens["text_muted"]))
        for side in ("top", "right", "left"):
            self.assertFalse(axes.spines[side].get_visible(), side)
        legend = [text.get_text() for text in axes.get_legend().get_texts()]
        self.assertEqual(legend, ["Observed pairs (random sample of 400)", "Isotonic fit"])
        # The legend sits above the plot, under the statistics line. A closed
        # figure's canvas draws nothing, so an Agg canvas lays it out again.
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        legend_box = axes.get_legend().get_window_extent(renderer)
        self.assertGreaterEqual(legend_box.y0, axes.get_window_extent(renderer).y1)
        self.assertLess(legend_box.y1, statistics.get_window_extent(renderer).y0)
        self.assertLess(statistics.get_window_extent(renderer).y1, title.get_window_extent(renderer).y0)

        axes = self.draw(pairs=len(self.x)).axes[0]
        self.assertEqual(axes.get_legend().get_texts()[0].get_text(), "Observed pairs")

    def test_the_colour_bar_counts_pairs_per_hexagon(self):
        figure = self.draw()
        colour_bar = figure.axes[1]
        self.assertEqual(colour_bar.get_ylabel(), "Pairs per hexagon")
        top = int(self.hexagons(figure).get_clim()[1])
        self.assertEqual(list(colour_bar.get_yticks()), Embedding_MSA._count_ticks(top))
        self.assertEqual([label.get_text() for label in colour_bar.get_yticklabels()],
                         [f"{tick:,}" for tick in Embedding_MSA._count_ticks(top)])

    # The chart's texts: its title, statistics line, axis labels, colour bar
    # label and two legend entries, seven in all; the rest are the axes' numbers.
    @staticmethod
    def words(figure):
        from matplotlib.text import Text

        shown = [text.get_text() for text in figure.findobj(Text) if text.get_text()]
        return [text for text in shown if not re.fullmatch(r"[\d,.\-−]+", text)]

    def test_every_text_the_chart_shows_goes_through_the_catalog(self):
        # The pseudo-language brackets every catalog text, so one left plain skipped it.
        from utilities.Localization import is_pseudo_translated

        figure = self.draw(pairs=2_000, environment={"SSN_PSEUDO_TRANSLATION": "1"})
        words = self.words(figure)
        self.assertEqual(len(words), 7, words)
        for text in words + [self.window_title]:
            self.assertTrue(is_pseudo_translated(text), text)
        self.assertEqual(self.warnings, [])

    def test_simplified_chinese_draws_in_its_bundled_font_with_no_glyph_missing(self):
        # Nor a face: matplotlib logs one a family lacks, as Noto Sans SC lacks semibold.
        with self.assertNoLogs("matplotlib", level="WARNING"):
            figure = self.draw(pairs=2_000, language="zh_CN")
        title = figure.axes[0].get_title(loc="left")
        self.assertEqual(self.window_title, title)
        for text in self.words(figure):
            self.assertRegex(text, "[一-鿿]", "every text shows in Chinese")
        self.assertEqual(self.warnings, [])

    def test_the_pop_up_opens_in_a_window_in_the_tools_windows_look(self):
        from tools.tool_helpers import Tool_Charts

        with mock.patch.object(Tool_Charts, "show_chart", wraps=Tool_Charts.show_chart) as show:
            figure = self.draw(color="teal")
        show.assert_called_once_with(figure, "teal", Embedding_MSA.PROJECT_ROOT)

    def test_the_chart_puts_back_the_translator_it_found(self):
        from utilities import Localization

        previous = Localization._translator
        self.draw(language="zh_CN")
        self.assertIs(Localization._translator, previous)

    def test_a_language_that_cannot_load_leaves_the_chart_english(self):
        from desktop import Desktop_App

        english = "Isotonic Regression Fit"
        with mock.patch.object(Desktop_App, "message_translator", side_effect=LookupError("no catalog")), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            figure = self.draw(language="zh_CN")
        self.assertEqual(figure.axes[0].get_title(loc="left"), english)
        self.assertIn("the regression plot stays English (no catalog)", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
