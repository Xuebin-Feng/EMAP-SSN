"""Embedding_MSA's isotonic regression: how a sparse run fits and checks it
against a network's calibration sample, when the run shows its chart, and how
the chart looks."""

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
UNCHECKED = "⚠️ No calibration sample was used, so the imputed scores are unchecked."
UNIFORM = "Uniform fill: every missing pair below the fit's range gets the fit's lowest value, {value:.4f}."


class ImputationFitTests(unittest.TestCase):
    """fit_imputation, held_out_check, outside_fit_fraction and the texts that report them."""

    def setUp(self):
        rng = np.random.default_rng(11)
        # Observed pairs above a cut at 0.5, and a calibration sample below it.
        self.above_x = rng.uniform(0.5, 1.0, 600)
        self.below_x = rng.uniform(0.0, 0.5, 40)
        self.x = np.concatenate([self.above_x, self.below_x])
        self.y = 3 * self.x + rng.normal(0, 0.1, self.x.size)
        # The sample's edges, listed in sample order (not edge order), each
        # standing for a different number of skipped pairs.
        self.positions = np.arange(600, 640)[::-1]
        self.weights = rng.uniform(5.0, 50.0, 40)

    def keys_for(self, size):
        """Distinct pair keys in no particular order, one per observed pair."""
        return np.random.default_rng(21).permutation(size * 3)[:size].astype(np.int64)

    def permuted(self, with_sample, seed):
        """x, y, keys and sample positions with the observed pairs in another order; the sample stays at the end in its order."""
        keys = self.keys_for(self.x.size)
        order = np.random.default_rng(seed).permutation(600)
        full = np.concatenate([order, np.arange(600, 640)])
        return self.x[full], self.y[full], keys[full]

    def test_the_pick_does_not_follow_the_order_of_the_edges_when_keys_are_given(self):
        for with_sample in (False, True):
            fits = []
            for seed in (1, 2, 3):
                with self.subTest(with_sample=with_sample, seed=seed):
                    x, y, keys = self.permuted(with_sample, seed)
                    positions = np.arange(600, 640)[::-1] if with_sample else None
                    weights = self.weights if with_sample else None
                    fits.append(Embedding_MSA.fit_imputation(x, y, positions, weights, fit_pairs=100, pair_keys=keys))
            for fit in fits[1:]:
                np.testing.assert_array_equal(fit.fit_x, fits[0].fit_x)
                np.testing.assert_array_equal(fit.fit_y, fits[0].fit_y)
                np.testing.assert_array_equal(fit.iso.X_thresholds_, fits[0].iso.X_thresholds_)
                np.testing.assert_array_equal(fit.iso.y_thresholds_, fits[0].iso.y_thresholds_)
        # Without keys the pick does follow the order (what the keys fix).
        x1, y1, _ = self.permuted(False, 1)
        x2, y2, _ = self.permuted(False, 2)
        one = Embedding_MSA.fit_imputation(x1, y1, fit_pairs=100)
        two = Embedding_MSA.fit_imputation(x2, y2, fit_pairs=100)
        self.assertFalse(np.array_equal(one.fit_x, two.fit_x))

    def test_edges_already_in_canonical_order_give_the_fit_they_always_gave(self):
        keys = np.arange(self.x.size, dtype=np.int64) * 5
        for positions, weights in ((None, None), (np.arange(600, 640)[::-1], self.weights)):
            for fit_pairs in (100, 1_000):
                with self.subTest(sample=positions is not None, fit_pairs=fit_pairs):
                    old = Embedding_MSA.fit_imputation(self.x, self.y, positions, weights, fit_pairs=fit_pairs)
                    new = Embedding_MSA.fit_imputation(self.x, self.y, positions, weights, fit_pairs=fit_pairs, pair_keys=keys)
                    np.testing.assert_array_equal(new.fit_x, old.fit_x)
                    np.testing.assert_array_equal(new.iso.X_thresholds_, old.iso.X_thresholds_)
                    np.testing.assert_array_equal(new.iso.y_thresholds_, old.iso.y_thresholds_)

    def test_the_sample_pairs_stay_out_of_the_pick_and_keep_their_order(self):
        keys = self.keys_for(self.x.size)
        fit = Embedding_MSA.fit_imputation(
            self.x, self.y, self.positions, self.weights, fit_pairs=100, pair_keys=keys)
        np.testing.assert_array_equal(fit.calibration_x, self.x[self.positions])
        self.assertEqual(fit.fit_x.size, 100)
        self.assertFalse(np.isin(fit.fit_x, self.below_x).any())

    def test_the_key_is_the_ranks_of_the_code_point_sorted_headers_and_ignores_every_order(self):
        headers = ["b", "B", "a", "é", "Z", "10", "9"]  # code points: "10" < "9" < "B" < "Z" < "a" < "b" < "é"
        edge_i = np.array([0, 3, 6, 2])
        edge_j = np.array([1, 2, 0, 3])
        keys = Embedding_MSA.canonical_pair_keys(edge_i, edge_j, headers)
        rank = {name: k for k, name in enumerate(sorted(headers))}
        n = len(headers)
        for key, a, b in zip(keys, edge_i, edge_j):
            ra, rb = rank[headers[a]], rank[headers[b]]
            self.assertEqual(int(key), min(ra, rb) * n + max(ra, rb))
        self.assertEqual(sorted(rank, key=rank.get), ["10", "9", "B", "Z", "a", "b", "é"])
        # The same pairs with the sequences listed in another order and the ends turned round: the same keys.
        shuffled = [headers[k] for k in (4, 0, 6, 2, 5, 1, 3)]
        where = {name: k for k, name in enumerate(shuffled)}
        again = Embedding_MSA.canonical_pair_keys(
            np.array([where[headers[b]] for b in edge_j]), np.array([where[headers[a]] for a in edge_i]), shuffled)
        np.testing.assert_array_equal(again, keys)

    def test_without_a_sample_the_fit_is_the_one_it_always_was(self):
        fit = Embedding_MSA.fit_imputation(self.x, self.y, fit_pairs=100)
        np.random.seed(42)
        pick = np.random.choice(self.x.size, size=100, replace=False)
        expected = IsotonicRegression(out_of_bounds="clip").fit(self.x[pick], self.y[pick])
        np.testing.assert_array_equal(fit.fit_x, self.x[pick])
        np.testing.assert_array_equal(fit.iso.X_thresholds_, expected.X_thresholds_)
        np.testing.assert_array_equal(fit.iso.y_thresholds_, expected.y_thresholds_)
        self.assertIsNone(fit.check)
        self.assertIsNone(fit.calibration_x)
        # Too few sample pairs to check: the same fit.
        few = Embedding_MSA.fit_imputation(
            self.x, self.y, self.positions[:Embedding_MSA.MIN_CALIBRATION_PAIRS - 1],
            self.weights[:Embedding_MSA.MIN_CALIBRATION_PAIRS - 1], fit_pairs=100,
        )
        np.testing.assert_array_equal(few.iso.X_thresholds_, expected.X_thresholds_)
        self.assertIsNone(few.check)

    def test_a_sample_extends_the_fit_below_the_cut_weighted(self):
        fit = Embedding_MSA.fit_imputation(self.x, self.y, self.positions, self.weights)
        self.assertAlmostEqual(fit.iso.X_min_, self.below_x.min())
        np.testing.assert_array_equal(fit.calibration_x, self.x[self.positions])
        np.testing.assert_array_equal(np.sort(fit.fit_x), np.sort(self.above_x))
        # The fit weights the sample by the skipped pairs each stands for.
        expected = IsotonicRegression(out_of_bounds="clip").fit(
            np.concatenate([self.above_x, self.x[self.positions]]),
            np.concatenate([self.y[:600], self.y[self.positions]]),
            sample_weight=np.concatenate([np.ones(600), self.weights]),
        )
        np.testing.assert_allclose(fit.iso.predict(self.x), expected.predict(self.x))
        # Below the cut the estimates now follow the scores instead of one value.
        self.assertGreater(np.ptp(fit.iso.predict(self.below_x)), 1.0)

    def test_the_check_scores_every_second_pair_on_a_fit_without_them(self):
        fit = Embedding_MSA.fit_imputation(self.x, self.y, self.positions, self.weights)
        first = self.positions[0::2]
        held = self.positions[1::2]
        half = IsotonicRegression(out_of_bounds="clip").fit(
            np.concatenate([self.above_x, self.x[first]]),
            np.concatenate([self.y[:600], self.y[first]]),
            sample_weight=np.concatenate([np.ones(600), self.weights[0::2] * 2]),
        )
        expected = Embedding_MSA.held_out_check(half.predict(self.x[held]), self.y[held])
        self.assertEqual(fit.check.pairs, 20)
        self.assertAlmostEqual(fit.check.spearman, expected.spearman)
        self.assertAlmostEqual(fit.check.mae, expected.mae)
        self.assertAlmostEqual(fit.check.r2, expected.r2)
        self.assertGreater(fit.check.r2, 0.8)

    def test_a_sampled_fit_takes_a_random_share_of_the_pairs_above_the_cut(self):
        fit = Embedding_MSA.fit_imputation(self.x, self.y, self.positions, self.weights, fit_pairs=50)
        self.assertEqual(fit.fit_x.size, 50)
        self.assertTrue(np.all(fit.fit_x >= 0.5))
        self.assertEqual(fit.calibration_x.size, 40)

    def test_the_fit_takes_up_to_500000_observed_pairs_by_default(self):
        import inspect

        self.assertEqual(Embedding_MSA.IMPUTATION_FIT_PAIRS, 500_000)
        default = inspect.signature(Embedding_MSA.fit_imputation).parameters["fit_pairs"].default
        self.assertEqual(default, Embedding_MSA.IMPUTATION_FIT_PAIRS)

    def test_with_more_than_500000_observed_pairs_the_default_picks_exactly_500000_and_with_fewer_uses_every_pair(self):
        rng = np.random.default_rng(5)
        limit = Embedding_MSA.IMPUTATION_FIT_PAIRS
        total = limit + 21
        x = rng.random(total)
        y = (3.0 * x + 0.1 * rng.random(total)).astype(np.float32)
        positions, weights = np.arange(total - 20, total), np.full(20, 3.0)
        # 500,001 observed pairs without a sample, 500,001 besides the sample with one: 500,000 are picked.
        fit = Embedding_MSA.fit_imputation(x[:limit + 1], y[:limit + 1])
        self.assertEqual(fit.fit_x.size, limit)
        self.assertEqual(fit.fit_y.size, limit)
        fit = Embedding_MSA.fit_imputation(x, y, positions, weights)
        self.assertEqual(fit.fit_x.size, limit)
        self.assertEqual(fit.fit_y.size, limit)
        # Exactly 500,000 observed pairs (besides the sample, if any): every pair is used and nothing is picked.
        with mock.patch.object(Embedding_MSA.np.random, "choice", side_effect=AssertionError("a pick was made")):
            fit = Embedding_MSA.fit_imputation(x[:limit], y[:limit])
            self.assertEqual(fit.fit_x.size, limit)
            fit = Embedding_MSA.fit_imputation(x[:limit + 20], y[:limit + 20], positions - 1, weights)
            self.assertEqual(fit.fit_x.size, limit)

    def test_too_few_sample_pairs_stay_among_the_observed_pairs(self):
        few = self.positions[:Embedding_MSA.MIN_CALIBRATION_PAIRS - 1]
        on = Embedding_MSA.fit_imputation(self.x, self.y, few, self.weights[:len(few)])
        self.assertEqual(on.fit_x.size, self.x.size)
        self.assertIsNone(on.calibration_x)
        self.assertIsNone(on.check)
        self.assertTrue(on.uniform)

    def test_held_out_check_numbers(self):
        check = Embedding_MSA.held_out_check([1.0, 2.0, 4.0], [1.0, 3.0, 3.0])
        self.assertEqual(check.pairs, 3)
        self.assertAlmostEqual(check.mae, 2 / 3)
        centred = np.array([1.0, 3.0, 3.0]) - 7 / 3
        self.assertAlmostEqual(check.r2, 1 - 2 / float(np.dot(centred, centred)))
        self.assertAlmostEqual(check.spearman, 0.8660254, places=6)
        # One predicted value, or one actual value: no rank correlation, and no R² for the latter.
        self.assertTrue(np.isnan(Embedding_MSA.held_out_check([2, 2, 2], [1, 2, 3]).spearman))
        flat = Embedding_MSA.held_out_check([1, 2, 3], [2, 2, 2])
        self.assertTrue(np.isnan(flat.spearman))
        self.assertTrue(np.isnan(flat.r2))

    def test_outside_fraction_counts_the_missing_pairs_beyond_the_fit(self):
        rng = np.random.default_rng(2)
        n = 1_300  # more rows than one block
        matrix = rng.uniform(0, 1, (n, n))
        matrix = (matrix + matrix.T) / 2
        rows, cols = np.triu_indices(n, k=1)
        observed = rng.random(rows.size) < 0.3
        observed_x = matrix[rows[observed], cols[observed]]
        iso = IsotonicRegression(out_of_bounds="clip").fit([0.2, 0.9], [0.0, 1.0])
        missing = int(np.count_nonzero(~observed))
        values = matrix[rows[~observed], cols[~observed]]
        expected = np.count_nonzero((values < 0.2) | (values > 0.9)) / missing
        got = Embedding_MSA.outside_fit_fraction(matrix, iso, observed_x, missing)
        self.assertAlmostEqual(got, expected)
        self.assertEqual(Embedding_MSA.outside_fit_fraction(matrix, iso, observed_x, 0), 0.0)

    def test_the_guide_tree_note_says_how_missing_pairs_were_filled(self):
        check = Embedding_MSA.CalibrationCheck(1234, 0.81234, 0.41234, 0.63456)
        cases = (
            (False, None, "None. Every pair is in the network."),
            (True, None, "Regression-imputed from embedding distances"),
            (True, Embedding_MSA.ImputationSummary(9_000, 0.001, check, 2_468),
             "Regression-imputed from embedding distances; checked on 1,234 held-out "
             "calibration pairs: Spearman 0.8123, MAE 0.4123, R^2 0.6346; 0.1% of missing "
             "pairs outside the fit's range"),
            (True, Embedding_MSA.ImputationSummary(9_000, 1.0),
             "Regression-imputed from embedding distances; no calibration sample was used, "
             "so the estimates are unchecked; 100.0% of missing pairs outside the fit's range"),
            (True, Embedding_MSA.ImputationSummary(
                9_000, 0.5, calibration="calibration not applied: 6.6% of pairs missing, below 10%"),
             "Regression-imputed from embedding distances; calibration not applied: 6.6% of pairs "
             "missing, below 10%, so the estimates are unchecked; 50.0% of missing pairs outside the fit's range"),
        )
        for imputed, summary, note in cases:
            with self.subTest(note=note):
                self.assertEqual(Embedding_MSA.missing_pairs_note(imputed, summary), note)

    def test_the_report_gives_the_check_or_warns_without_one(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 1, 200)
        y = 2 * x
        sampled = Embedding_MSA.fit_imputation(x, y, np.arange(20), np.ones(20))
        summary = Embedding_MSA.ImputationSummary(5_000, 0.002, sampled.check, 20)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            Embedding_MSA.report_imputation(sampled, summary)
        text = printed.getvalue()
        self.assertIn("Calibration check (20 directly computed pairs from this run's missing pairs):", text)
        self.assertIn("A fit without every second one predicts those 10: Spearman rho", text)
        self.assertIn("Missing pairs outside the fit's range: 0.2% of 5,000", text)
        self.assertNotIn("⚠️", text)
        # With the sample used there is no uniform value to report.
        self.assertNotIn("Uniform fill:", text)

        plain = Embedding_MSA.fit_imputation(x, y)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            Embedding_MSA.report_imputation(plain, Embedding_MSA.ImputationSummary(5_000, 0.97))
        text = printed.getvalue()
        self.assertIn("Missing pairs outside the fit's range: 97.0% of 5,000", text)
        self.assertIn(UNCHECKED, text)
        self.assertNotIn("Calibration check", text)
        # The uniform value is printed, as one line, for a fit without the sample.
        value = Embedding_MSA.uniform_fill_value(plain.iso)
        self.assertIn(UNIFORM.format(value=value), text)
        self.assertEqual(text.count("Uniform fill:"), 1)

        # The reason the run had no sample is the warning's first words.
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            Embedding_MSA.report_imputation(plain, Embedding_MSA.ImputationSummary(
                5_000, 0.97, calibration="calibration not applied: 6.6% of pairs missing, below 10%"))
        self.assertIn(
            "⚠️ Calibration not applied: 6.6% of pairs missing, below 10%, so the imputed scores are unchecked.",
            printed.getvalue())



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

    def test_a_network_that_records_no_scoring_warns_that_its_estimates_are_unchecked(self):
        fixture = NetworkFixture(self.directory, complete=False, seed=2)
        plot = mock.Mock()
        output = self.run_builder(fixture, plot, SHOW_REGRESSION_PLOT=True)
        self.assertIn(
            "⚠️ Calibration unavailable: the network does not record the gap penalties its scores were made "
            "with, so the imputed scores are unchecked.", output)
        self.assertIn("Missing pairs outside the fit's range: ", output)
        self.assertNotIn("Calibration check", output)
        fit = plot.call_args.kwargs["calibration"]
        self.assertIsNone(fit.check)

    def run_for_baseline(self, fixture, **settings):
        """Run to the guide tree with perturbed trees off; return the output and the matrix SciPy got."""
        seen = {}
        real_linkage = Embedding_MSA.sch.linkage

        def record(matrix, *args, **kwargs):
            seen["matrix"] = np.array(matrix)
            return real_linkage(matrix, *args, **kwargs)

        plot = mock.Mock()
        with mock.patch.object(Embedding_MSA.sch, "linkage", record):
            output = self.run_builder(fixture, plot, SHOW_REGRESSION_PLOT=True, **settings)
        return output, seen["matrix"], plot

    def test_a_network_that_records_no_scoring_is_filled_the_same_with_the_switch_on_or_off(self):
        fixture = NetworkFixture(self.directory, complete=False, seed=2)
        results = {
            switch: self.run_for_baseline(fixture, CALIBRATED_IMPUTATION=switch) for switch in (True, False)
        }
        self.assertEqual(results[True][1].tobytes(), results[False][1].tobytes())
        for switch, (output, _, plot) in results.items():
            with self.subTest(switch=switch):
                self.assertEqual(output.count("Uniform fill: every missing pair below the fit's range"), 1)
                self.assertIn("so the imputed scores are unchecked", output)
                self.assertNotIn("Measured over", output)
        uniform = [line for line in results[True][0].splitlines() if line.startswith("Uniform fill:")]
        self.assertEqual(uniform, [line for line in results[False][0].splitlines() if line.startswith("Uniform fill:")])
        self.assertEqual(len(uniform), 1)

    def test_the_imputation_uses_the_shared_pooling_and_power_whatever_an_old_network_recorded(self):
        fixture = NetworkFixture(self.directory, complete=False, seed=2)
        fixture.add_calibration_sample(count=12, pooling="mean", length_power=1.0)
        calibration = Embedding_MSA.Prefilter_Calibration
        real_pooled = calibration.pooled_vectors
        real_similarity = calibration.length_adjusted_similarity
        used = {}

        # Only the settings are kept: holding the HDF5 group would keep the file open.
        def pooled(embeddings, names, pooling, progress=None):
            used["pooling"] = pooling
            return real_pooled(embeddings, names, pooling, progress)

        def similarity(vectors, lengths, power):
            used["power"] = power
            return real_similarity(vectors, lengths, power)

        with mock.patch.object(calibration, "pooled_vectors", pooled), \
                mock.patch.object(calibration, "length_adjusted_similarity", similarity):
            output = self.run_builder(fixture, mock.Mock())
        # An old network recorded mean pooling and power 1.0; nothing reads that any more.
        self.assertEqual(used, {"pooling": calibration.POOLING, "power": calibration.LENGTH_RATIO_POWER})
        self.assertEqual((calibration.POOLING, calibration.LENGTH_RATIO_POWER), ("max", 2.0))
        self.assertNotIn("The network's similarity was made with", output)


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

    def draw(self, pairs=400, color=None, saved=None, language="en", environment=None,
             calibration=None):
        """The figure plt.show was given, drawn; color is the Tools window's saved one.

        pairs counts the observed pairs, of which the class's 400 are a sample
        when it is larger. language is the saved LANGUAGE, English unless a
        test asks, whatever the system's; environment adds variables, as
        SSN_PSEUDO_TRANSLATION. saved, when given, is the whole
        app_settings.json text instead. calibration is passed on to the
        chart. The warnings raised while drawing are left in self.warnings.
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
            Embedding_MSA.plot_regression_fit(
                self.iso, self.x, self.y, 0.5, 0.25, pairs, calibration=calibration
            )
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

    def calibrated(self):
        """A fit over the class's pairs and a 60-pair sample below them, as a run makes it."""
        rng = np.random.default_rng(8)
        below = rng.uniform(0.1, 0.6, 60).astype(np.float32)
        x = np.concatenate([self.x, below])
        y = np.concatenate([self.y, (1 + 2 * below + rng.normal(0, 0.3, 60)).astype(np.float32)])
        return Embedding_MSA.fit_imputation(x, y, np.arange(400, 460), np.full(60, 50.0))

    def test_a_calibration_sample_shows_as_dots_with_its_check(self):
        from matplotlib.collections import PathCollection
        from matplotlib.colors import to_rgba
        from matplotlib.text import Text
        from desktop.Studio_Theme import theme_tokens

        tokens = theme_tokens("teal")
        fit = self.calibrated()
        figure = self.draw(pairs=2_060, color="teal", calibration=fit)
        axes = figure.axes[0]
        (dots,) = [item for item in axes.collections if isinstance(item, PathCollection)]
        np.testing.assert_array_equal(
            dots.get_offsets(), np.column_stack([fit.calibration_x, fit.calibration_y])
        )
        self.assertEqual(to_rgba(dots.get_facecolor()[0])[:3], to_rgba(tokens["text_muted"])[:3])
        # Over the hexagons, under the fit's line.
        self.assertGreater(dots.get_zorder(), self.hexagons(figure).get_zorder())
        self.assertLess(dots.get_zorder(), axes.lines[0].get_zorder())
        legend = [text.get_text() for text in axes.get_legend().get_texts()]
        self.assertEqual(legend, [
            "Observed pairs (random sample of 400)",
            "Calibration pairs, a sample of the missing pairs",
            "Isotonic fit",
        ])
        texts = {text.get_text(): text for text in figure.findobj(Text) if text.get_text()}
        check = texts[str(Embedding_MSA.calibration_statistics(fit.check))]
        self.assertTrue(check.get_text().startswith("Held-out check on 30 calibration pairs · Spearman ρ "))
        statistics = texts[str(Embedding_MSA.regression_statistics(0.5, 0.25, 2_060))]
        title = texts["Isotonic Regression Fit"]
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        # Bottom up: the legend, the check line, the statistics line, the title.
        boxes = [axes.get_legend(), check, statistics, title]
        extents = [item.get_window_extent(renderer) for item in boxes]
        for lower, upper in zip(extents, extents[1:]):
            self.assertLess(lower.y1, upper.y0)
        # 400 hexagon pairs stand for all 2,000 besides the sample: a random sample.
        axes = self.draw(pairs=460, calibration=fit).axes[0]
        self.assertEqual(axes.get_legend().get_texts()[0].get_text(), "Observed pairs")

    def test_a_fit_without_a_sample_draws_as_before(self):
        from matplotlib.collections import PathCollection

        plain = Embedding_MSA.fit_imputation(self.x, self.y)
        axes = self.draw(calibration=plain).axes[0]
        self.assertFalse([item for item in axes.collections if isinstance(item, PathCollection)])
        self.assertEqual(len(axes.get_legend().get_texts()), 2)

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
        # A calibration sample adds its legend entry and its check line.
        figure = self.draw(pairs=2_060, environment={"SSN_PSEUDO_TRANSLATION": "1"},
                           calibration=self.calibrated())
        words = self.words(figure)
        self.assertEqual(len(words), 9, words)
        for text in words:
            self.assertTrue(is_pseudo_translated(text), text)
        self.assertEqual(self.warnings, [])

    def test_simplified_chinese_draws_in_its_bundled_font_with_no_glyph_missing(self):
        # Nor a face: matplotlib logs one a family lacks, as Noto Sans SC lacks semibold.
        with self.assertNoLogs("matplotlib", level="WARNING"):
            figure = self.draw(pairs=2_060, language="zh_CN", calibration=self.calibrated())
        title = figure.axes[0].get_title(loc="left")
        self.assertEqual(self.window_title, title)
        words = self.words(figure)
        self.assertEqual(len(words), 9, words)
        for text in words:
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
