"""Embedding_MSA's calibration sample: drawn from the run's own missing pairs, aligned with the network's
recorded scoring at the start of the run, checked on some of the network's own edges, and added to the
observed pairs before the scores are normalized."""

import contextlib
import gc
import glob
import io
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

from tests.msa_fixtures import (
    GuideTreeBuilt,
    InProcessPool,
    NetworkFixture,
    builder_settings,
    fake_calibration_alignment,
)
import Embedding_MSA  # noqa: E402  msa_fixtures puts src/tools on the path
from utilities import Prefilter_Calibration  # noqa: E402

SAMPLE = 12  # the draw's bounds in these tests: a sample of 12 pairs


class CalibrationRunTestCase(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = pathlib.Path(temp_dir.name)

    def run_builder(self, fixture, *, bounds=True, reproduce=True, **settings):
        """Run to the guide tree with the calibration alignment replaced; return (output, state, plot, kernel).

        bounds sets the draw's bounds to SAMPLE pairs. kernel lists what the normalization kernel was given.
        """
        output = io.StringIO()
        plot = mock.Mock()
        patches, state = fake_calibration_alignment(reproduce)
        kernel = []
        real_kernel = Embedding_MSA.calculate_normalized_scores_kernel

        def spy(**arguments):
            kernel.append({name: np.array(value) for name, value in arguments.items() if isinstance(value, np.ndarray)})
            return real_kernel(**arguments)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.multiple(
                Embedding_MSA,
                benchmark_msa_devices=mock.Mock(side_effect=GuideTreeBuilt),
                plot_regression_fit=plot,
                calculate_normalized_scores_kernel=spy,
                **builder_settings(fixture, SHOW_REGRESSION_PLOT=True, **settings),
            ))
            stack.enter_context(patches)
            stack.enter_context(mock.patch.object(Embedding_MSA.mp, "set_start_method"))
            stack.enter_context(mock.patch.object(Embedding_MSA.mp, "Pool", InProcessPool))
            if bounds:
                stack.enter_context(mock.patch.object(Prefilter_Calibration, "DRAW_MIN_PAIRS", SAMPLE))
                stack.enter_context(mock.patch.object(Prefilter_Calibration, "DRAW_MAX_PAIRS", SAMPLE))
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            with self.assertRaises(GuideTreeBuilt):
                Embedding_MSA.run_msa_builder()
        gc.collect()  # release the builder's HDF5 handles before the folder is removed
        return output.getvalue(), state, plot, kernel

    def fixture(self, name="run", **options):
        options.setdefault("sequences", 27)
        options.setdefault("ghosts", 0)
        options.setdefault("seed", 3)
        fixture = NetworkFixture(self.directory / name, **options)
        fixture.record_scoring((-3.0, -0.5), "ieee_fp32")
        return fixture

    def guide_tree_notes(self, fixture):
        (path,) = glob.glob(str(fixture.msa_dir / "**" / "*guide_tree.nex"), recursive=True)
        return pathlib.Path(path).read_text(encoding="utf-8")


class MissingShareRuleTests(CalibrationRunTestCase):
    """Calibration needs at least 10% of the run's pairs missing: judged on the run, not on the network."""

    def test_just_below_exactly_at_and_just_above_the_limit(self):
        # 25 sequences make 300 pairs: 29 missing is 9.67%, 30 exactly 10%, 31 more.
        for missing, calls in ((29, 0), (30, 1), (31, 1)):
            with self.subTest(missing=missing):
                fixture = self.fixture(f"m{missing}", drop=missing)
                output, state, plot, _ = self.run_builder(fixture)
                self.assertEqual(len(state.calls), calls)
                if calls:
                    self.assertIn(f"Calibration sample of {SAMPLE} of the {missing} missing pairs", output)
                else:
                    self.assertIn("Calibration not applied: 9.7% of pairs missing, below 10%", output)
                    self.assertIn("the missing pairs get the uniform fill", output)
                    self.assertIsNone(plot.call_args.kwargs["calibration"].check)
                    notes = self.guide_tree_notes(fixture)
                    self.assertIn("CALIBRATION", notes)
                    self.assertIn("Calibration not applied: 9.7% of pairs missing, below 10%", notes)

    def test_a_subset_run_is_judged_by_its_own_completeness(self):
        # 20 sequences have embeddings; 20 more are in the network only. The network has 780 pairs, the run 190.
        # Below the limit for the run while the whole network is above it, and the other way round.
        cases = (
            ("run_complete_network_sparse", dict(drop=100, drop_from="ghost"), 0),
            ("run_sparse_network_complete", dict(drop=20, drop_from="run"), 1),
        )
        for name, options, calls in cases:
            with self.subTest(name=name):
                fixture = self.fixture(name, sequences=20, missing=(), ghosts=20, out_of_range=False, **options)
                pairs, run_size = fixture.run_pairs()
                self.assertEqual(run_size, 20)
                output, state, *_ = self.run_builder(fixture)
                self.assertEqual(len(state.calls), calls, output)

    def test_the_switch_off_draws_and_aligns_nothing_whatever_the_share(self):
        for missing in (0, 29, 60):
            with self.subTest(missing=missing):
                fixture = self.fixture(f"off{missing}", drop=missing or None)
                output, state, plot, _ = self.run_builder(fixture, CALIBRATED_IMPUTATION=False)
                self.assertEqual(state.calls, [])
                if missing:
                    self.assertIn("Calibration switched off: no sample is drawn or aligned", output)
                    self.assertIn("so the imputed scores are unchecked", output)


class SampleTests(CalibrationRunTestCase):
    def test_the_sample_is_drawn_among_the_missing_pairs_and_added_before_the_normalization(self):
        fixture = self.fixture(drop=60)
        edges, size = fixture.run_pairs()
        output, state, plot, kernel = self.run_builder(fixture, ALIGNMENT_SCORE="global")
        self.assertEqual(len(state.calls), 1)
        pairs = state.calls[0]
        sample, check = pairs[:SAMPLE], pairs[SAMPLE:]
        self.assertEqual(len(sample), SAMPLE)
        self.assertEqual(len(set(map(tuple, sample.tolist()))), SAMPLE)
        self.assertTrue(all(tuple(pair) not in edges for pair in sample.tolist()), "a sample pair has an edge")
        self.assertTrue(all(a < b for a, b in sample.tolist()))
        # The check edges are the network's own, in the run, at most 32.
        self.assertTrue(0 < len(check) <= Embedding_MSA.CALIBRATION_CHECK_EDGES)
        self.assertTrue(all((min(pair), max(pair)) in edges for pair in check.tolist()))
        # The kernel got the sample as its last pairs, with their aligned scores and lengths.
        (given,) = kernel
        self.assertEqual(given["edge_i"][-SAMPLE:].tolist(), sample[:, 0].tolist())
        self.assertEqual(given["edge_j"][-SAMPLE:].tolist(), sample[:, 1].tolist())
        expected = (40.0 + (sample[:, 0] * 7 + sample[:, 1] * 3) % 50).astype(np.float32)
        np.testing.assert_array_equal(given["raw_scores"][-SAMPLE:], expected)
        np.testing.assert_array_equal(given["align_lens"][-SAMPLE:], (10 + (sample[:, 0] + sample[:, 1]) % 30).astype(np.float32))
        self.assertEqual(len(given["edge_i"]), len(given["raw_scores"]))
        # The fit takes them as its calibration pairs, weighted by the missing pairs each stands for.
        fit = plot.call_args.kwargs["calibration"]
        self.assertEqual(fit.calibration_x.size, SAMPLE)
        self.assertEqual(fit.check.pairs, SAMPLE // 2)
        self.assertIn("Calibration check (12 directly computed pairs from this run's missing pairs):", output)
        self.assertNotIn("No calibration sample", output)
        # The note names the sample, its bounds and seed, the device, and the check.
        notes = self.guide_tree_notes(fixture)
        for text in ("Calibration sample of 12 of the 60 missing pairs", "seed 42", "Fake CPU [cpu] (scalar, ieee_fp32)",
                     "self-check on", "reproduced"):
            self.assertIn(text, notes)

    def test_the_networks_gap_penalties_and_precision_go_to_the_alignment_unswapped(self):
        fixture = self.fixture(drop=60)
        fixture.record_scoring((-3.0, -0.5), "tf32")
        _, state, *_ = self.run_builder(fixture)
        self.assertEqual(state.arguments, [([-3.0, -0.5], "tf32")])

    def test_all_missing_pairs_are_aligned_when_no_more_than_the_sample_size_and_nothing_is_imputed(self):
        fixture = self.fixture(drop=40)
        output, state, plot, kernel = self.run_builder(fixture, bounds=False)
        sample = state.calls[0][:40]
        self.assertEqual(len(sample), 40)
        self.assertIn("All 40 missing pairs (13.3% of the run's pairs) are no more than the sample size, so all were aligned", output)
        self.assertNotIn("Isotonic Regression Fit Diagnostics", output)
        plot.assert_not_called()
        notes = self.guide_tree_notes(fixture)
        self.assertIn("None. Every pair is in the network.", notes)
        self.assertIn("All 40 missing pairs", notes)
        _, _, _, without = self.run_builder(fixture, CALIBRATED_IMPUTATION=False)
        self.assertEqual(len(kernel[0]["edge_i"]) - len(without[0]["edge_i"]), 40)

    def test_a_stored_sample_of_an_older_network_is_ignored_and_its_edges_are_ordinary(self):
        fixture = self.fixture(drop=60)
        fixture.add_calibration_sample(count=12)  # the record an earlier version stored, with its pairs among the edges
        _, state, plot, kernel = self.run_builder(fixture)
        fit = plot.call_args.kwargs["calibration"]
        # The sample is the new draw (12 pairs), not the stored one; the stored 12 are among the other observed pairs.
        self.assertEqual(len(state.calls[0]) - len(state.check.rows), SAMPLE)
        self.assertEqual(fit.calibration_x.size, SAMPLE)
        self.assertEqual(plot.call_args.args[5], len(plot.call_args.args[1]) + SAMPLE)

    def test_the_networks_own_edges_must_be_reproduced(self):
        fixture = self.fixture(drop=60)
        output, state, plot, kernel = self.run_builder(fixture, reproduce=False)
        self.assertEqual(len(state.calls), 1)
        self.assertIn("Calibration unavailable: the network's own edges were not reproduced", output)
        self.assertIn("differ by more than the tolerance", output)
        # Nothing was added, and the fit is the plain one.
        fit = plot.call_args.kwargs["calibration"]
        self.assertIsNone(fit.calibration_x)
        self.assertTrue(fit.uniform)
        (given,) = kernel
        self.assertIn("Calibration unavailable", self.guide_tree_notes(fixture))
        _, _, _, without = self.run_builder(fixture, CALIBRATED_IMPUTATION=False)
        self.assertEqual(len(given["edge_i"]), len(without[0]["edge_i"]))

    def test_a_network_that_does_not_record_its_gap_penalties_is_filled_uniformly(self):
        fixture = NetworkFixture(self.directory / "plain", sequences=27, ghosts=0, seed=3, drop=60)
        output, state, plot, _ = self.run_builder(fixture)
        self.assertEqual(state.calls, [])
        self.assertIn("Calibration unavailable: the network does not record the gap penalties", output)
        self.assertIn("so the imputed scores are unchecked", output)

    def test_a_failing_alignment_leaves_the_run_going_with_the_uniform_fill(self):
        fixture = self.fixture(drop=60)
        patches, state = fake_calibration_alignment()
        output = io.StringIO()
        with mock.patch.multiple(
            Embedding_MSA,
            benchmark_msa_devices=mock.Mock(side_effect=GuideTreeBuilt),
            plot_regression_fit=mock.Mock(),
            align_calibration_pairs=mock.Mock(side_effect=RuntimeError("no device")),
            **builder_settings(fixture),
        ), mock.patch.object(Embedding_MSA.mp, "Pool", InProcessPool), mock.patch.object(
            Embedding_MSA.mp, "set_start_method"
        ), mock.patch.object(Prefilter_Calibration, "DRAW_MIN_PAIRS", SAMPLE), mock.patch.object(
            Prefilter_Calibration, "DRAW_MAX_PAIRS", SAMPLE
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(GuideTreeBuilt):
                Embedding_MSA.run_msa_builder()
        gc.collect()
        self.assertIn("Calibration unavailable: the sample could not be aligned (RuntimeError: no device)", output.getvalue())


class CheckEdgeTests(unittest.TestCase):
    def network(self, directory, count=500, seed=5):
        rng = np.random.default_rng(seed)
        path = os.path.join(directory, "n.h5")
        with h5py.File(path, "w") as hf:
            hf.create_dataset("i", data=rng.integers(0, 40, count).astype(np.uint16))
            hf.create_dataset("j", data=rng.integers(40, 80, count).astype(np.uint16))
            for name in Embedding_MSA.CHECK_FIELDS:
                values = rng.uniform(0, 300, count)
                hf.create_dataset(name, data=values.astype(np.uint16 if name.endswith("_len") else np.float32))
        return path

    def test_they_are_edges_of_the_run_spread_over_the_stored_scores_and_the_same_every_time(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.network(temp)
            with h5py.File(path, "r") as hf:
                edge_i, edge_j = hf["i"][:], hf["j"][:]
                # Network sequences 0-39 are in the run (as run indices 0-39); 40-79 are not, except 40-59.
                mapping = np.full(80, -1, dtype=np.int32)
                mapping[:40] = np.arange(40)
                mapping[40:60] = np.arange(40, 60)
                first = Embedding_MSA.choose_check_edges(hf, edge_i, edge_j, mapping, "g_score")
                again = Embedding_MSA.choose_check_edges(hf, edge_i, edge_j, mapping, "g_score")
        np.testing.assert_array_equal(first.rows, again.rows)
        np.testing.assert_array_equal(first.cols, again.cols)
        self.assertEqual(len(first.rows), Embedding_MSA.CALIBRATION_CHECK_EDGES)
        self.assertTrue(np.all(first.rows < 40) and np.all(first.cols < 60) and np.all(first.cols >= 40))
        scores = np.sort(first.stored["g_score"])
        # An even spread: the lowest and highest of the pool are among them, and the gaps stay small.
        self.assertLess(scores[0], 20)
        self.assertGreater(scores[-1], 280)
        self.assertLess(np.diff(scores).max(), 60)
        self.assertIsNone(Embedding_MSA.choose_check_edges(None, edge_i[:0], edge_j[:0], mapping, "g_score"))

    def test_a_score_may_differ_by_the_tolerance_and_a_length_not_at_all(self):
        stored = {"l_score": np.array([1.0, 100.0], np.float32), "l_len": np.array([5, 6], np.uint16),
                  "g_score": np.array([2.0, 200.0], np.float32), "g_len": np.array([7, 8], np.uint16)}

        def aligned(**changes):
            fields = {name: np.concatenate([[0], values]) for name, values in stored.items()}
            for name, change in changes.items():
                fields[name] = fields[name].astype(np.float64) + np.concatenate([[0], change])
            return SimpleNamespace(**fields)

        absolute, relative = Embedding_MSA.CALIBRATION_CHECK_ABSOLUTE, Embedding_MSA.CALIBRATION_CHECK_RELATIVE
        ok, worst, text = Embedding_MSA.compare_with_stored(
            aligned(l_score=[absolute / 2, 0.5 * relative * 100.0]), 1, stored)
        self.assertTrue(ok, text)
        ok, worst, text = Embedding_MSA.compare_with_stored(aligned(g_score=[0.0, 3 * relative * 200.0 + absolute]), 1, stored)
        self.assertFalse(ok)
        self.assertIn("g_score", text)
        ok, worst, text = Embedding_MSA.compare_with_stored(aligned(l_len=[1, 0]), 1, stored)
        self.assertFalse(ok)
        self.assertIn("l_len", text)


if __name__ == "__main__":
    unittest.main()
