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
    EMBED_NAME,
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

    def run_builder(self, fixture, *, bounds=True, reproduce=True, wrap_align=None, **settings):
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
            if wrap_align is not None:  # wrap the replaced alignment, so the test can do what the real one does first
                stack.enter_context(mock.patch.object(
                    Embedding_MSA, "align_calibration_pairs", wrap_align(Embedding_MSA.align_calibration_pairs)))
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
        with mock.patch.object(Embedding_MSA, "fit_imputation", wraps=Embedding_MSA.fit_imputation) as fitted:
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
        _, _, positions, weights = fitted.call_args.args
        self.assertEqual(positions.tolist(), list(range(len(given["edge_i"]) - SAMPLE, len(given["edge_i"]))))
        # Each pair stands for the missing pairs of its similarity band: 60 pairs in 20 equal bands of 3, one drawn from each.
        self.assertEqual(weights.tolist(), [3.0] * SAMPLE)
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


class SharedSimilarityTests(CalibrationRunTestCase):
    def test_align_prefilter_and_the_msa_get_the_same_matrix_from_the_same_inputs(self):
        # Both real code paths run: the MSA's builder up to its guide tree, and Align's prefilter_pairs on the
        # embeddings file and the very headers and lengths the MSA gave the rule. A spy on the shared function
        # keeps the matrix each of them was handed.
        from utilities import Align_Library

        fixture = self.fixture(drop=60)
        real = Prefilter_Calibration.pair_similarity
        seen = []

        def spy(embeddings, headers, lengths, progress=None):
            result = real(embeddings, headers, lengths, progress)
            seen.append((list(headers), np.array(lengths), result))
            return result

        with mock.patch.object(Prefilter_Calibration, "pair_similarity", spy):
            self.run_builder(fixture)
            self.assertEqual(len(seen), 1)
            headers, lengths, from_msa = seen[0]
            align = Align_Library.load()
            with mock.patch.multiple(
                align, FULL_INPUT_HDF5=str(fixture.embed_dir / EMBED_NAME), PREFILTER_STRENGTH=50
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                required, record = align.prefilter_pairs(headers, lengths.tolist())
        self.assertEqual(len(seen), 2)
        from_align = seen[1][2]
        self.assertEqual(from_align.shape, from_msa.shape)
        self.assertEqual(from_align.tobytes(), from_msa.tobytes())
        # Align's cut is that matrix's 50th percentile of the upper triangle.
        upper = np.triu(np.ones(from_msa.shape, dtype=bool), k=1)
        self.assertEqual(record.threshold, float(np.percentile(from_msa[upper], 50)))


class OpenEmbeddingsTests(CalibrationRunTestCase):
    def test_the_accelerator_paths_opens_succeed_on_the_embeddings_the_run_holds_open(self):
        # In a real run Embedding MSA holds its embeddings file open while the calibration alignment runs, and
        # as soon as an accelerator is a candidate the alignment opens the same file in this process (the tile
        # store, the scalar pipeline) with SWMR; HDF5 refuses that when the first open had no SWMR flag. The
        # alignment itself is replaced here, so the wrapper makes those two opens, without a GPU.
        from Embedding_Alignment_Engine import EmbeddingTileStore

        def accelerator_opens(real):
            def wrapper(pairs, embeddings_path, names, *rest, **kwargs):
                EmbeddingTileStore(embeddings_path, list(names), 0)
                with h5py.File(embeddings_path, "r", libver="latest", swmr=True) as hf:
                    hf["embeddings"][names[0]][:]
                return real(pairs, embeddings_path, names, *rest, **kwargs)
            return wrapper

        fixture = self.fixture(drop=60)
        output, state, plot, _ = self.run_builder(fixture, wrap_align=accelerator_opens)
        self.assertNotIn("OSError", output)
        self.assertNotIn("Calibration unavailable", output)
        self.assertIn("Calibration sample of 12 of the 60 missing pairs", output)
        self.assertEqual(len(state.calls), 1)

    def test_a_file_opened_plainly_blocks_those_opens_and_the_shared_open_does_not_for_any_file(self):
        from Embedding_Alignment_Engine import EmbeddingTileStore
        from utilities.HDF5_Storage import open_embeddings_shared

        first = NetworkFixture(self.directory / "a", sequences=6, ghosts=0, seed=1)
        second = NetworkFixture(self.directory / "b", sequences=6, ghosts=0, seed=2)
        first_path, second_path = (str(f.embed_dir / EMBED_NAME) for f in (first, second))
        with h5py.File(first_path, "r") as plain:
            with self.assertRaises(OSError):
                EmbeddingTileStore(first_path, first.emb_headers, 0)
        # Two files held open at once, each opened the shared way: the alignment path can open both.
        with open_embeddings_shared(first_path), open_embeddings_shared(second_path):
            EmbeddingTileStore(first_path, first.emb_headers, 0)
            EmbeddingTileStore(second_path, second.emb_headers, 0)
            with h5py.File(second_path, "r", libver="latest", swmr=True) as hf:
                self.assertEqual(len(hf["embeddings"]), 6)


class AlignmentCallTests(unittest.TestCase):
    def test_the_real_call_hands_align_the_networks_values_and_this_runs_device_choice(self):
        seen = {}

        class FakeAlign:
            def align_pairs_in_memory(self, *args, **kwargs):
                seen["args"], seen["kwargs"] = args, kwargs
                return "aligned"

        pairs = np.array([[0, 1], [1, 2]])
        names = ["a", "b", "c"]
        with mock.patch.object(Embedding_MSA.Align_Library, "load", return_value=FakeAlign()), mock.patch.object(
            Embedding_MSA, "DEVICE_SELECTION", "cuda:1"
        ), mock.patch.object(Embedding_MSA.Numba_Threads, "default_thread_count", return_value=5):
            # Two pairs of 3,000 x 3,000 cells: 18 million cells, three workers of 8 million.
            result = Embedding_MSA.align_calibration_pairs(
                pairs, "embeddings.h5", names, [-3.0, -0.5], "tf32", np.array([3000, 3000, 3000]))
        self.assertEqual(result, "aligned")
        given = seen["args"]
        self.assertIs(given[0], pairs)
        # The embeddings path, the dataset names, then the gap penalties as [local, global] and the precision.
        self.assertEqual(list(given[1:]), ["embeddings.h5", names, [-3.0, -0.5], "tf32"])
        self.assertEqual(seen["kwargs"], {"device_selection": "cuda:1", "workers": 3})

    def call_with(self, selection, cells_pairs, default=5):
        """What align_pairs_in_memory is handed, and what was printed, for a selection and a number of 1,000,000-cell pairs."""
        seen = {}

        class FakeAlign:
            def align_pairs_in_memory(self, *args, **kwargs):
                seen["kwargs"] = kwargs
                return "aligned"

        lengths = np.full(2, 1000)
        pairs = np.zeros((cells_pairs, 2), dtype=int)
        pairs[:, 1] = 1
        printed = io.StringIO()
        with mock.patch.object(Embedding_MSA.Align_Library, "load", return_value=FakeAlign()), mock.patch.object(
            Embedding_MSA, "DEVICE_SELECTION", selection
        ), mock.patch.object(
            Embedding_MSA.Numba_Threads, "default_thread_count", return_value=default
        ), contextlib.redirect_stdout(printed):
            Embedding_MSA.align_calibration_pairs(pairs, "e.h5", ["a", "b"], [-3.0, -0.5], "ieee_fp32", lengths)
        return seen["kwargs"], printed.getvalue()

    def test_auto_uses_the_cpu_without_timing_while_the_sample_needs_fewer_workers_than_the_default(self):
        # 32 pairs of 1,000,000 cells: 32 million cells, four workers of 8 million, fewer than the default five.
        kwargs, text = self.call_with("auto", 32)
        self.assertEqual(kwargs, {"device_selection": "cpu", "workers": 4})
        self.assertIn("on the CPU, without timing the devices", text)
        self.assertIn("needs 4 CPU worker(s), fewer than the 5", text)

    def test_auto_times_the_devices_once_the_sample_needs_every_default_worker(self):
        # 33 pairs: 33 million cells, five workers: the default count, so the planner decides.
        kwargs, text = self.call_with("auto", 33)
        self.assertEqual(kwargs, {"device_selection": "auto", "workers": 5})
        self.assertIn("large enough for all 5 CPU workers, so the devices are timed", text)
        kwargs, _ = self.call_with("auto", 5000)
        self.assertEqual(kwargs, {"device_selection": "auto", "workers": 5})

    def test_a_named_device_is_used_as_named_for_a_small_sample_too(self):
        for selection in ("cuda:0", "cpu"):
            kwargs, text = self.call_with(selection, 4)
            self.assertEqual(kwargs, {"device_selection": selection, "workers": 1})
            self.assertEqual(text, "")

    def test_the_pool_has_one_child_per_eight_million_cells_up_to_the_default_count(self):
        lengths = np.full(2000, 1000)
        pairs = lambda count: np.column_stack([np.zeros(count, dtype=int), np.ones(count, dtype=int)])  # 1,000,000 cells each
        with mock.patch.object(Embedding_MSA.Numba_Threads, "default_thread_count", return_value=18):
            workers = lambda count: Embedding_MSA.calibration_workers(pairs(count), lengths)
            self.assertEqual(workers(1), 1)       # a few cells: one child
            self.assertEqual(workers(8), 1)       # exactly 8 million cells: one
            self.assertEqual(workers(9), 2)       # one cell more than 8 million: two
            self.assertEqual(workers(30), 4)      # kringle's 5,000 pairs are about 30 million cells: four
            self.assertEqual(workers(10_000), 18)  # a large sample: the default count
        with mock.patch.object(Embedding_MSA.Numba_Threads, "default_thread_count", return_value=2):
            self.assertEqual(Embedding_MSA.calibration_workers(pairs(10_000), lengths), 2)


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
