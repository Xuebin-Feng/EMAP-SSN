"""The `logo` command (commands/logo.py).

Covers position parsing and compact plot coordinates, the y axis, letter
heights with and without identity weighting, the identity kernels and their
thread budget, the background SVG renderer, and the job `logo` enqueues.
"""

import io
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
from tests.sparse_alignment import sparse_alignment
from commands import logo as logo_command
from commands.logo import (
    _configure_logo_y_axis,
    _calculate_identity_neighbour_counts_numpy,
    _encode_standard_amino_acids,
    calculate_identity_weights,
    calculate_logo_matrix,
    extract_identity_threshold,
    get_compact_logo_coordinates,
    parse_logo_positions,
)


class LogoYAxisTests(unittest.TestCase):
    def test_percentage_mode_uses_zero_to_one_with_percent_labels(self):
        from matplotlib.figure import Figure

        ax = Figure().subplots()
        _configure_logo_y_axis(ax, "pcts", "with_gap")

        self.assertEqual(ax.get_ylim(), (0.0, 1.0))
        self.assertEqual(ax.get_ylabel(), "Percentage")
        labels = [label.get_text() for label in ax.get_yticklabels()]
        self.assertEqual(labels, ["0%", "20%", "40%", "60%", "80%", "100%"])

    def test_bits_mode_uses_theoretical_protein_maximum(self):
        from matplotlib.figure import Figure

        ax = Figure().subplots()
        _configure_logo_y_axis(ax, "bits", "no_gap")

        self.assertAlmostEqual(ax.get_ylim()[0], 0.0)
        self.assertAlmostEqual(ax.get_ylim()[1], np.log2(20))
        self.assertEqual(ax.get_ylabel(), "Bits")


class DisconnectedLogoPositionTests(unittest.TestCase):
    def test_disconnected_residue_labels_receive_adjacent_plot_coordinates(self):
        labels = [58, 62, 86, 152, 221, 282]

        coordinates = get_compact_logo_coordinates(labels)

        self.assertEqual(coordinates, [0, 1, 2, 3, 4, 5])
        self.assertEqual(labels, [58, 62, 86, 152, 221, 282])

    def test_empty_and_single_position_inputs_are_supported(self):
        self.assertEqual(get_compact_logo_coordinates([]), [])
        self.assertEqual(get_compact_logo_coordinates([282]), [0])


class LogoPositionParsingTests(unittest.TestCase):
    def test_explicit_fractional_positions_are_parsed_in_alignment_order(self):
        positions = parse_logo_positions("[11,10.2,10,10.1]")

        self.assertEqual(positions, [10, "10.1", "10.2", 11])

    def test_integer_ranges_retain_integer_only_behavior(self):
        positions = parse_logo_positions("[10-12,10.1]")

        self.assertEqual(positions, [10, "10.1", 11, 12])

    def test_explicit_plus_prefix_remains_supported(self):
        self.assertEqual(parse_logo_positions("[+1,+1.1]"), [1, "1.1"])

    def test_parenthesized_negative_offset_labels_are_supported(self):
        positions = parse_logo_positions("[(-1.1),(-1),0]")

        self.assertEqual(positions, [-1, "-1.1", 0])

    def test_parenthesized_negative_ranges_are_supported(self):
        self.assertEqual(
            parse_logo_positions("[(-3)-(-1),0]"),
            [-3, -2, -1, 0],
        )
        self.assertEqual(
            parse_logo_positions("[(-2)-1]"),
            [-2, -1, 0, 1],
        )

    def test_descending_parenthesized_negative_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lower to higher"):
            parse_logo_positions("[(-1)-(-3)]")

    def test_bare_negative_positions_are_rejected_with_correction(self):
        for position_spec, correction in (
            ("[-1]", "(-1)"),
            ("[-1.1,0]", "(-1.1)"),
            ("[-3--1]", "(-3)"),
            ("[1--1]", "(-1)"),
        ):
            with self.subTest(position_spec=position_spec):
                with self.assertRaisesRegex(ValueError, re.escape(correction)):
                    parse_logo_positions(position_spec)

    def test_fractional_ranges_require_explicit_insertion_labels(self):
        with self.assertRaisesRegex(ValueError, "list insertion positions explicitly"):
            parse_logo_positions("[10.1-11.2]")

    def test_ordinary_ranges_are_unchanged_by_the_size_cap(self):
        self.assertEqual(parse_logo_positions("[1-5]"), [1, 2, 3, 4, 5])
        self.assertEqual(
            parse_logo_positions("[2-3, 7, 1.1]"), ["1.1", 2, 3, 7]
        )
        with mock.patch.object(logo_command, "_MAX_LOGO_RANGE_POSITIONS", 10):
            self.assertEqual(
                parse_logo_positions("[1-10]"), list(range(1, 11))
            )

    def test_a_range_over_the_cap_is_refused_before_it_is_expanded(self):
        with mock.patch.object(logo_command, "_MAX_LOGO_RANGE_POSITIONS", 10):
            with self.assertRaisesRegex(ValueError, "too large.*at most 10 positions"):
                parse_logo_positions("[1-11]")
        # The real cap refuses a huge range at once instead of expanding it.
        for position_spec in ("[1-5000000]", "[1-99999999999999999999999]"):
            with self.subTest(position_spec=position_spec):
                with self.assertRaisesRegex(ValueError, "too large"):
                    parse_logo_positions(position_spec)

    def test_a_list_of_ranges_over_the_cap_is_refused(self):
        with mock.patch.object(logo_command, "_MAX_LOGO_RANGE_POSITIONS", 10):
            with self.assertRaisesRegex(ValueError, "position list is too large"):
                parse_logo_positions("[1-6, 7-12]")
            # Overlapping ranges name each position once.
            self.assertEqual(
                parse_logo_positions("[1-8, 3-9]"), list(range(1, 10))
            )


class LogoFilenameTests(unittest.TestCase):
    def test_a_name_that_is_only_an_extension_is_refused(self):
        for name in (".svg", ".png", ".SVG", " .png "):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "needs a name before its extension"):
                    logo_command._normalize_logo_filename(name)

    def test_ordinary_names_keep_their_extension_rule(self):
        normalize = logo_command._normalize_logo_filename
        self.assertEqual(normalize("motif"), "motif.svg")
        self.assertEqual(normalize("motif.png"), "motif.png")
        self.assertEqual(normalize("a.b.svg"), "a.b.svg")
        self.assertEqual(normalize(".hidden"), ".hidden.svg")


class LogoIdentityThresholdParsingTests(unittest.TestCase):
    def test_fraction_and_percentage_forms_are_equivalent(self):
        for token in ("0.9", "90", "90%"):
            with self.subTest(token=token):
                threshold, remaining = extract_identity_threshold(["[1]", token])
                self.assertAlmostEqual(threshold, 0.9)
                self.assertEqual(remaining, ["[1]"])

    def test_weighting_is_disabled_when_threshold_is_omitted(self):
        threshold, remaining = extract_identity_threshold(
            ["#cluster_1#", "[1]", "bits"]
        )

        self.assertIsNone(threshold)
        self.assertEqual(remaining, ["#cluster_1#", "[1]", "bits"])

    def test_invalid_or_repeated_threshold_is_rejected(self):
        for arguments in (["[1]", "0"], ["[1]", "101%"], ["[1]", "90", "80"]):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    extract_identity_threshold(arguments)


class LogoMatrixTests(unittest.TestCase):
    """Letter heights without identity weighting (the default)."""

    A_INDEX = "ACDEFGHIKLMNPQRSTVWY".index("A")

    def test_percentages_count_unknown_residues_in_n_but_not_occupancy(self):
        sequences = ["A", "A", "-", "X"]

        with_gap, _ = calculate_logo_matrix(sequences, [0], mode="pcts", gap_mode="with_gap")
        no_gap, _ = calculate_logo_matrix(sequences, [0], mode="pcts", gap_mode="no_gap")

        self.assertAlmostEqual(with_gap[0, self.A_INDEX], 0.5)
        self.assertAlmostEqual(no_gap[0, self.A_INDEX], 1.0)
        self.assertAlmostEqual(float(with_gap.sum()), 0.5)
        self.assertAlmostEqual(float(no_gap.sum()), 1.0)

    def test_bits_correct_for_the_non_gap_count(self):
        # log2(20) minus the small-sample correction 19 / (2 ln 2 N), N = 4.
        expected_bits = np.log2(20) - 19.0 / (8.0 * np.log(2))
        self.assertAlmostEqual(expected_bits, 0.8955, places=4)

        conserved, _ = calculate_logo_matrix(["A"] * 4, [0], mode="bits", gap_mode="no_gap")
        self.assertAlmostEqual(conserved[0, self.A_INDEX], expected_bits)

        # Two residues and two gaps: N is 2, not 4. The correction
        # 19 / (2 ln 2 * 2) = 6.85 exceeds log2(20) = 4.32, so the column
        # carries no information in either gap mode (it used to read 0.4478
        # with_gap, which counted the two gaps).
        for gap_mode in ("with_gap", "no_gap"):
            with self.subTest(gap_mode=gap_mode):
                half_gapped, _ = calculate_logo_matrix(
                    ["A", "A", "-", "-"], [0], mode="bits", gap_mode=gap_mode
                )
                self.assertEqual(float(half_gapped[0, self.A_INDEX]), 0.0)

    def test_gapped_column_is_corrected_for_its_non_gap_count(self):
        # Sixteen A and four gaps: N = 16.
        #   correction = 19 / (2 * 0.693147 * 16) = 0.8566
        #   bits       = 4.321928 - 0 - 0.8566    = 3.4653
        # Counting the gaps (N = 20) would give 4.321928 - 0.6853 = 3.6366.
        sequences = ["A"] * 16 + ["-"] * 4

        no_gap, _ = calculate_logo_matrix(sequences, [0], mode="bits", gap_mode="no_gap")
        with_gap, _ = calculate_logo_matrix(sequences, [0], mode="bits", gap_mode="with_gap")

        self.assertAlmostEqual(no_gap[0, self.A_INDEX], 3.4653, places=4)
        # with_gap then scales by the 16/20 occupancy, a separate step.
        self.assertAlmostEqual(with_gap[0, self.A_INDEX], 3.4653 * 0.8, places=4)
        self.assertAlmostEqual(with_gap[0, self.A_INDEX], 2.7723, places=4)

    def test_mixed_gapped_column_uses_the_non_gap_count(self):
        # Twelve A, four G, four gaps: N = 16, p = 0.75 / 0.25.
        #   entropy = 0.811278, correction = 0.856600
        #   information = 4.321928 - 0.811278 - 0.856600 = 2.654050
        #   A = 0.75 * 2.654050 = 1.9905, G = 0.25 * 2.654050 = 0.6635
        sequences = ["A"] * 12 + ["G"] * 4 + ["-"] * 4

        heights, _ = calculate_logo_matrix(sequences, [0], mode="bits", gap_mode="no_gap")

        self.assertAlmostEqual(heights[0, self.A_INDEX], 1.9905, places=4)
        self.assertAlmostEqual(heights[0, "ACDEFGHIKLMNPQRSTVWY".index("G")], 0.6635, places=4)

    def test_correction_counts_only_standard_residues_on_every_code_path(self):
        # The ASCII fast path, a column past the shortest sequence, and a
        # non-ASCII sequence (which forces the per-sequence loop) all give
        # the hand-computed N = 16 value.
        paths = {
            "ascii": ["A"] * 16 + ["-"] * 2 + ["X"] * 2,
            "short_rows": ["AAAA"] * 16 + ["AA"] * 4,
            "non_ascii": ["A"] * 16 + ["-"] * 3 + ["\u00e9"],
        }
        for name, sequences in paths.items():
            with self.subTest(path=name):
                column = 3 if name == "short_rows" else 0
                heights, _ = calculate_logo_matrix(
                    sequences, [column], mode="bits", gap_mode="no_gap"
                )
                self.assertAlmostEqual(heights[0, self.A_INDEX], 3.4653, places=4)


class LogoSequenceWeightingTests(unittest.TestCase):
    def test_identical_sequences_share_one_effective_observation(self):
        weights = calculate_identity_weights(["AAAA", "AAAA", "GGGG"], 0.9)

        np.testing.assert_allclose(weights, [0.5, 0.5, 1.0])
        self.assertAlmostEqual(float(weights.sum()), 2.0)

    def test_missing_coverage_does_not_create_high_identity(self):
        weights = calculate_identity_weights(["AAAA", "AA--"], 0.9)

        np.testing.assert_allclose(weights, [1.0, 1.0])

    def test_percentage_mode_uses_weighted_amino_acid_frequencies(self):
        weighted, weights = calculate_logo_matrix(
            ["AAAA", "AAAA", "GGGG"],
            [0],
            mode="pcts",
            gap_mode="no_gap",
            identity_threshold=0.9,
        )
        unweighted, _ = calculate_logo_matrix(
            ["AAAA", "AAAA", "GGGG"],
            [0],
            mode="pcts",
            gap_mode="no_gap",
        )

        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        self.assertAlmostEqual(weighted[0, aa_order.index("A")], 0.5)
        self.assertAlmostEqual(weighted[0, aa_order.index("G")], 0.5)
        self.assertAlmostEqual(unweighted[0, aa_order.index("A")], 2.0 / 3.0)
        self.assertAlmostEqual(float(weights.sum()), 2.0)

    def test_bits_mode_uses_effective_count_for_small_sample_correction(self):
        sequences = ["AAAA"] * 63
        weighted, weights = calculate_logo_matrix(
            sequences,
            [0],
            mode="bits",
            gap_mode="no_gap",
            identity_threshold=0.9,
        )
        unweighted, _ = calculate_logo_matrix(
            sequences,
            [0],
            mode="bits",
            gap_mode="no_gap",
        )

        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        a_index = aa_order.index("A")
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(weighted[0, a_index], 0.0)
        self.assertGreater(unweighted[0, a_index], 4.0)

    def test_bits_match_with_and_without_identity_when_no_sequence_repeats(self):
        # Twenty-five distinct sequences. Column 0 holds 13 A, 4 G, 3 L and
        # five gaps, so it is gapped; columns 1 and 2 are fully occupied and
        # make every sequence unique (their pair of letters never repeats).
        aa = "ACDEFGHIKLMNPQRSTVWY"
        first = list("A" * 13 + "G" * 4 + "L" * 3 + "-" * 5)
        sequences = [first[i] + aa[i // 5] + aa[i % 5] for i in range(25)]
        self.assertEqual(len(set(sequences)), 25)

        # The gapped column is corrected for its 20 residues, not 25 rows:
        # p = 13/20, 4/20, 3/20 and N = 20, in both modes. with_gap then
        # scales by the 20/25 occupancy.
        probabilities = np.array([13, 4, 3]) / 20.0
        entropy = -np.sum(probabilities * np.log2(probabilities))
        information = np.log2(20) - entropy - 19.0 / (2.0 * np.log(2) * 20)
        expected_a = 13 / 20.0 * information
        a_index = "ACDEFGHIKLMNPQRSTVWY".index("A")

        for gap_mode, scale in (("with_gap", 20 / 25.0), ("no_gap", 1.0)):
            with self.subTest(gap_mode=gap_mode):
                plain, _ = calculate_logo_matrix(
                    sequences, [0, 1, 2], mode="bits", gap_mode=gap_mode
                )
                weighted, weights = calculate_logo_matrix(
                    sequences, [0, 1, 2], mode="bits", gap_mode=gap_mode,
                    identity_threshold=1.0,
                )
                # Identity 100% merges only identical sequences: all weights 1.
                np.testing.assert_allclose(weights, np.ones(25))
                np.testing.assert_allclose(weighted, plain)
                self.assertAlmostEqual(float(plain[0, a_index]), expected_a * scale)

    def test_identity_correction_counts_the_summed_weight_of_non_gap_sequences(self):
        # Twenty distinct sequences with A at column 0, five copies of one
        # more (together they weigh 1), and four distinct gapped rows. The
        # residue weight at column 0 is 20 + 1 = 21 and the gap rows add none.
        #   correction = 19 / (2 * 0.693147 * 21) = 0.6527
        #   bits       = 4.321928 - 0 - 0.6527    = 3.6693
        aa = "ACDEFGHIKLMNPQRSTVWY"
        sequences = (
            ["A" + aa[i // 5] + aa[i % 5] for i in range(20)]
            + ["AWW"] * 5
            + ["-CC", "-DD", "-EE", "-FF"]
        )
        weighted, weights = calculate_logo_matrix(
            sequences, [0], mode="bits", gap_mode="no_gap", identity_threshold=1.0
        )

        np.testing.assert_allclose(weights[20:25], np.full(5, 0.2))
        self.assertAlmostEqual(float(weights.sum()), 25.0)
        self.assertAlmostEqual(
            float(weighted[0, aa.index("A")]), 3.6693, places=4
        )

    def test_numba_and_numpy_counts_match_for_edge_cases_and_thresholds(self):
        sequences = [
            "AAAAAAAAAA",
            "AAAAAAAAAA",
            "AAAAAAAAAG",
            "AAAAAAAA--",
            "AAAAAAAAXX",
            "GGGGGGGGGG",
            "----------",
            "XXXXXXXXXX",
        ]
        encoded = _encode_standard_amino_acids(sequences)
        multiplicities = np.array([2, 1, 3, 1, 2, 1, 4, 2], dtype=np.int64)

        for threshold in (0.8, 0.9, 1.0):
            with self.subTest(threshold=threshold):
                expected = _calculate_identity_neighbour_counts_numpy(
                    encoded,
                    multiplicities,
                    threshold,
                    block_size=3,
                )
                actual, _ = logo_command.run_identity_neighbour_counts(
                    encoded,
                    multiplicities,
                    threshold,
                )
                np.testing.assert_array_equal(actual, expected)

    def test_numba_and_numpy_weights_match_for_seeded_alignment(self):
        rng = np.random.default_rng(20260812)
        amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        rows = rng.choice(amino_acids, size=(60, 80))
        sequences = ["".join(row) for row in rows]
        sequences.extend([sequences[0], sequences[0], sequences[1]])

        accelerated = calculate_identity_weights(sequences, 0.9)
        with mock.patch.object(logo_command, "NUMBA_AVAILABLE", False):
            fallback = calculate_identity_weights(sequences, 0.9)

        np.testing.assert_array_equal(accelerated, fallback)

    def test_empty_and_singleton_weighting(self):
        self.assertEqual(calculate_identity_weights([], 0.9).size, 0)
        np.testing.assert_array_equal(
            calculate_identity_weights(["ACDE"], 0.9),
            [1.0],
        )

    def test_numba_unavailable_uses_exact_numpy_fallback(self):
        with mock.patch.object(logo_command, "NUMBA_AVAILABLE", False):
            weights, metadata = calculate_identity_weights(
                ["AAAA", "AAAA", "GGGG"],
                0.9,
                return_metadata=True,
            )

        np.testing.assert_array_equal(weights, [0.5, 0.5, 1.0])
        self.assertEqual(metadata["backend"], "numpy")
        self.assertIn("not available", metadata["fallback_reason"])

    def test_numba_error_uses_exact_numpy_fallback(self):
        with mock.patch.object(
            logo_command,
            "run_identity_neighbour_counts",
            side_effect=RuntimeError("forced kernel error"),
        ):
            weights, metadata = calculate_identity_weights(
                ["AAAA", "AAAA", "GGGG"],
                0.9,
                return_metadata=True,
            )

        np.testing.assert_array_equal(weights, [0.5, 0.5, 1.0])
        self.assertEqual(metadata["backend"], "numpy")
        self.assertEqual(metadata["fallback_reason"], "forced kernel error")


class LogoIdentityKernelThreadTests(unittest.TestCase):
    def test_balanced_thread_selection_reserves_capacity(self):
        cases = (
            (1, 1, 1),
            (2, 2, 1),
            (4, 4, 2),
            (20, 20, 18),
        )
        for configured, logical, expected in cases:
            with self.subTest(configured=configured, logical=logical):
                self.assertEqual(
                    logo_command.choose_balanced_thread_count(
                        configured,
                        logical,
                    ),
                    expected,
                )

    def test_worker_restores_previous_numba_thread_setting(self):
        encoded = np.array([[0, 0], [1, 1]], dtype=np.int8)
        multiplicities = np.ones(2, dtype=np.int64)
        expected_counts = np.ones(2, dtype=np.int64)

        with (
            mock.patch.object(
                logo_command,
                "get_num_threads",
                return_value=20,
            ),
            mock.patch.object(
                logo_command,
                "set_num_threads",
            ) as set_threads,
            mock.patch.object(
                logo_command.os,
                "cpu_count",
                return_value=20,
            ),
            mock.patch.object(
                logo_command,
                "_identity_neighbour_counts_kernel",
                return_value=expected_counts,
            ),
        ):
            counts, selected_threads = (
                logo_command.run_identity_neighbour_counts(
                    encoded,
                    multiplicities,
                    0.9,
                )
            )

        np.testing.assert_array_equal(counts, expected_counts)
        self.assertEqual(selected_threads, 18)
        self.assertEqual(
            set_threads.call_args_list,
            [mock.call(18), mock.call(20)],
        )

    def test_worker_restores_threads_after_kernel_error(self):
        with (
            mock.patch.object(
                logo_command,
                "get_num_threads",
                return_value=4,
            ),
            mock.patch.object(
                logo_command,
                "set_num_threads",
            ) as set_threads,
            mock.patch.object(
                logo_command.os,
                "cpu_count",
                return_value=4,
            ),
            mock.patch.object(
                logo_command,
                "_identity_neighbour_counts_kernel",
                side_effect=RuntimeError("kernel failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                logo_command.run_identity_neighbour_counts(
                    np.zeros((1, 1), dtype=np.int8),
                    np.ones(1, dtype=np.int64),
                    0.9,
                )

        self.assertEqual(
            set_threads.call_args_list,
            [mock.call(2), mock.call(4)],
        )


class LogoSynchronousArtifactTests(unittest.TestCase):
    @staticmethod
    def make_payload(
        directory,
        filename="background_logo.svg",
        allow_overwrite=False,
    ):
        return {
            "selected_seqs": ("AAAA", "AAAA", "GGGG"),
            "valid_cols": (0,),
            "plot_positions": (1,),
            "mode": "pcts",
            "gap_mode": "no_gap",
            "identity_threshold": 0.9,
            "filename": filename,
            "color_scheme": "chemistry",
            "logo_dir": directory,
            "allow_overwrite": allow_overwrite,
            "ref_id": "reference",
        }

    def test_artifact_renderer_writes_svg_without_qt_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.make_payload(directory, filename="artifact.svg")

            result = logo_command._generate_logo_artifact(payload)

            self.assertTrue(os.path.isfile(result["save_path"]))
            self.assertGreater(os.path.getsize(result["save_path"]), 0)
            self.assertIn("effective N 2.00", str(result["message"]))

    def test_atomic_renderer_removes_partial_file_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.make_payload(directory, filename="atomic.svg")

            logo_command._generate_logo_artifact(payload)

            self.assertEqual(
                [name for name in os.listdir(directory) if ".partial" in name],
                [],
            )

    def test_artifact_renderer_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.make_payload(
                directory,
                filename="overwrite.svg",
                allow_overwrite=True,
            )
            out_file = os.path.join(directory, "overwrite.svg")
            with open(out_file, "wb") as f:
                f.write(b"old content")

            result = logo_command._generate_logo_artifact(payload)

            self.assertEqual(result["save_path"], out_file)
            self.assertTrue(os.path.isfile(out_file))
            with open(out_file, "rb") as f:
                content = f.read()
            self.assertNotEqual(content, b"old content")
            self.assertGreater(len(content), 0)

    def test_artifact_renderer_rejects_existing_file_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.make_payload(directory, filename="protected.svg")
            out_file = os.path.join(directory, "protected.svg")
            with open(out_file, "wb") as output:
                output.write(b"existing content")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                logo_command._generate_logo_artifact(payload)

            with open(out_file, "rb") as output:
                self.assertEqual(output.read(), b"existing content")

    def test_renderer_preserves_file_created_during_protected_render(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.make_payload(directory, filename="protected.svg")
            out_file = os.path.join(directory, "protected.svg")

            def save_and_create_competing_output(_figure, partial_path, **_kwargs):
                with open(partial_path, "wb") as output:
                    output.write(b"rendered content")
                with open(out_file, "wb") as output:
                    output.write(b"competing content")

            with mock.patch(
                "matplotlib.figure.Figure.savefig",
                autospec=True,
                side_effect=save_and_create_competing_output,
            ):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    logo_command._generate_logo_artifact(payload)

            with open(out_file, "rb") as output:
                self.assertEqual(output.read(), b"competing content")
            self.assertEqual(
                [name for name in os.listdir(directory) if ".partial" in name],
                [],
            )

class CapturingScheduler:
    def __init__(self):
        self.job = None

    def is_output_path_reserved(self, _path):
        return False

    def enqueue(self, **job):
        self.job = job
        return 1


class LogoSnapshotTests(unittest.TestCase):
    def test_automatic_filename_uses_deterministic_numeric_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "logo_stamp.svg")
            second = os.path.join(directory, "logo_stamp_2.svg")
            with open(first, "wb") as output:
                output.write(b"existing")
            scheduler = CapturingScheduler()
            scheduler.is_output_path_reserved = lambda path: (
                os.path.normcase(os.path.abspath(path))
                == os.path.normcase(os.path.abspath(second))
            )

            filename, path = logo_command._available_automatic_filename(
                scheduler,
                directory,
                "logo_stamp.svg",
            )

            self.assertEqual(filename, "logo_stamp_3.svg")
            self.assertEqual(path, os.path.join(directory, filename))

    def test_enqueued_logo_keeps_invocation_time_selection_and_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            alignment = SimpleNamespace(
                aln=sparse_alignment([("node0", "AAAA"), ("node1", "CCCC")]),
                viewer_to_aln=np.array([0, 1]),
                col_to_label={0: "1", 1: "2", 2: "3", 3: "4"},
                label_to_col={"1": 0, "2": 1, "3": 2, "4": 3},
                has_reference=True,
            )
            scheduler = CapturingScheduler()
            viewer = SimpleNamespace(
                alignment=alignment,
                full_headers=["node0", "node1"],
                selected_indices=[0],
                cluster_labels=None,
                group_labels=None,
                active_reference="node0",
                console_text=SimpleNamespace(text=""),
                background_job_scheduler=scheduler,
            )

            with mock.patch.object(logo_command, "LOGO_DIRECTORY", directory), \
                    mock.patch.object(
                        logo_command.cfg,
                        "HEADER_LIST_DIR",
                        directory,
                    ):
                logo_command.run(viewer, ["[1]", "snapshot.svg"])

            viewer.selected_indices[:] = [1]
            # Rows are materialized from the sparse matrix, so mutate the matrix.
            live = alignment.aln.matrix
            live.data[live.indptr[0]:live.indptr[1]] = Alignment_Manager.AA_TO_INT["G"]
            self.assertEqual(str(alignment.aln[0].seq), "GGGG")
            alignment.label_to_col["1"] = 3

            payload = scheduler.job["payload"]
            self.assertEqual(payload["selected_seqs"], ("AAAA",))
            self.assertEqual(payload["valid_cols"], (0,))
            self.assertEqual(payload["plot_positions"], (1,))

    def test_run_submits_explicit_fractional_position_to_logo_job(self):
        with tempfile.TemporaryDirectory() as directory:
            alignment = SimpleNamespace(
                aln=sparse_alignment([("node0", "A-AA"), ("node1", "ACAA")]),
                viewer_to_aln=np.array([0, 1]),
                col_to_label={0: "1", 1: "1.1", 2: "2", 3: "3"},
                label_to_col={"1": 0, "1.1": 1, "2": 2, "3": 3},
                has_reference=True,
            )
            scheduler = CapturingScheduler()
            viewer = SimpleNamespace(
                alignment=alignment,
                full_headers=["node0", "node1"],
                selected_indices=[0, 1],
                cluster_labels=None,
                group_labels=None,
                active_reference="node0",
                console_text=SimpleNamespace(text=""),
                background_job_scheduler=scheduler,
            )

            with mock.patch.object(logo_command, "LOGO_DIRECTORY", directory), \
                    mock.patch.object(
                        logo_command.cfg,
                        "HEADER_LIST_DIR",
                        directory,
                    ):
                logo_command.run(viewer, ["[1,1.1,2]", "insertions.svg"])

            payload = scheduler.job["payload"]
            self.assertEqual(payload["valid_cols"], (0, 1, 2))
            self.assertEqual(payload["plot_positions"], (1, "1.1", 2))

    def test_run_uses_occupancy_labels_without_active_reference(self):
        # Occupancy mode: the first row's gap in column 1 must not shift the
        # plotted columns away from the labels that query reports.
        with tempfile.TemporaryDirectory() as directory:
            alignment = SimpleNamespace(
                aln=sparse_alignment([("node0", "M-KCD"), ("node1", "MAKCD")]),
                viewer_to_aln=np.array([0, 1]),
                col_to_label={0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},
                label_to_col={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
                has_reference=False,
            )
            scheduler = CapturingScheduler()
            viewer = SimpleNamespace(
                alignment=alignment,
                full_headers=["node0", "node1"],
                selected_indices=[0, 1],
                cluster_labels=None,
                group_labels=None,
                active_reference="",
                console_text=SimpleNamespace(text=""),
                background_job_scheduler=scheduler,
            )
            output = io.StringIO()

            with mock.patch.object(logo_command, "LOGO_DIRECTORY", directory), \
                    mock.patch.object(
                        logo_command.cfg,
                        "HEADER_LIST_DIR",
                        directory,
                    ), \
                    redirect_stdout(output):
                logo_command.run(viewer, ["[2-5]", "occupancy.svg"])

            payload = scheduler.job["payload"]
            self.assertEqual(payload["valid_cols"], (1, 2, 3, 4))
            self.assertEqual(payload["plot_positions"], (2, 3, 4, 5))
            self.assertEqual(payload["numbering"], "occupancy")

    def run_logo_on(self, alignment, active_reference, args):
        """Run `logo` over both nodes of ALIGNMENT and return the viewer and scheduler."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        scheduler = CapturingScheduler()
        viewer = SimpleNamespace(
            alignment=alignment,
            full_headers=[str(record.id) for record in alignment.aln],
            selected_indices=[0, 1],
            cluster_labels=None,
            group_labels=None,
            active_reference=active_reference,
            console_text=SimpleNamespace(text=""),
            background_job_scheduler=scheduler,
        )
        with mock.patch.object(logo_command, "LOGO_DIRECTORY", directory.name), \
                mock.patch.object(logo_command.cfg, "HEADER_LIST_DIR", directory.name), \
                redirect_stdout(io.StringIO()):
            logo_command.run(viewer, args)
        return viewer, scheduler

    def test_run_names_the_anchored_header_on_the_axis(self):
        # The reference may be requested by identifier or wildcard; the axis
        # names the header the alignment anchored numbering on.
        alignment = SimpleNamespace(
            aln=sparse_alignment([("WP_0123.1_protein_A", "MAKCD"), ("node1", "MAKCD")]),
            viewer_to_aln=np.array([0, 1]),
            col_to_label={0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},
            label_to_col={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
            has_reference=True,
            resolved_ref_full="WP_0123.1_protein_A",
        )

        _, scheduler = self.run_logo_on(alignment, "WP_0123.1", ["[2-3]", "anchor.svg"])

        payload = scheduler.job["payload"]
        self.assertEqual(payload["numbering"], "reference")
        self.assertEqual(payload["ref_id"], "WP_0123.1_protein_A")

    def test_run_without_retained_columns_finds_no_positions(self):
        # No column passes the occupancy filter, so there are no displayed
        # positions. logo reports them missing, as query does, instead of
        # numbering the residues of some row.
        alignment = SimpleNamespace(
            aln=sparse_alignment([("node0", "M-KCD"), ("node1", "MAKCD")]),
            viewer_to_aln=np.array([0, 1]),
            col_to_label={},
            label_to_col={},
            has_reference=False,
        )

        viewer, scheduler = self.run_logo_on(alignment, "node0", ["[1-3]", "empty.svg"])

        self.assertIsNone(scheduler.job)
        self.assertEqual(
            viewer.console_text.text,
            "Error: Requested positions are outside the sequence bounds.",
        )

    def test_run_scopes_overwrite_to_explicit_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            existing_file = os.path.join(directory, "custom_logo.svg")
            with open(existing_file, "wb") as f:
                f.write(b"previous_content")

            alignment = SimpleNamespace(
                aln=sparse_alignment([("node0", "AAAA")]),
                viewer_to_aln=np.array([0]),
                col_to_label={0: "1", 1: "2", 2: "3", 3: "4"},
                label_to_col={"1": 0, "2": 1, "3": 2, "4": 3},
                has_reference=True,
            )
            scheduler = CapturingScheduler()
            viewer = SimpleNamespace(
                alignment=alignment,
                full_headers=["node0"],
                selected_indices=[0],
                cluster_labels=None,
                group_labels=None,
                active_reference="node0",
                console_text=SimpleNamespace(text=""),
                background_job_scheduler=scheduler,
            )

            with mock.patch.object(logo_command, "LOGO_DIRECTORY", directory), \
                    mock.patch.object(
                        logo_command.cfg,
                        "HEADER_LIST_DIR",
                        directory,
                    ):
                logo_command.run(viewer, ["[1]", "custom_logo.svg"])
                explicit_job = scheduler.job
                logo_command.run(viewer, ["[1]"])
                automatic_job = scheduler.job

            self.assertIsNotNone(explicit_job)
            self.assertEqual(explicit_job["command_name"], "logo")
            self.assertEqual(explicit_job["output_path"], existing_file)
            self.assertTrue(explicit_job["allow_overwrite"])
            self.assertTrue(explicit_job["payload"]["allow_overwrite"])
            self.assertIsNotNone(automatic_job)
            self.assertFalse(automatic_job["allow_overwrite"])
            self.assertFalse(automatic_job["payload"]["allow_overwrite"])
            self.assertEqual(viewer.console_text.text, "")


class LogoArgumentTests(unittest.TestCase):
    """How `logo` reads its arguments before it queues the job."""

    HEADERS = ["Escherichia_a", "Escherichia_b", "Bacillus_c"]

    def run_logo(self, args, selected=(0, 1, 2), existing_files=()):
        """Run `logo` with ARGS; return the viewer, queued job and the terminal text.

        EXISTING_FILES are created in the logo directory first, holding b"previous";
        self.directory names that directory.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        for name in existing_files:
            with open(os.path.join(self.directory, name), "wb") as existing:
                existing.write(b"previous")
        alignment = SimpleNamespace(
            aln=sparse_alignment(zip(self.HEADERS, ["AAAA", "ACAA", "GGGG"])),
            viewer_to_aln=np.array([0, 1, 2]),
            col_to_label={0: "1", 1: "2", 2: "3", 3: "4"},
            label_to_col={"1": 0, "2": 1, "3": 2, "4": 3},
            has_reference=True,
        )
        scheduler = CapturingScheduler()
        viewer = SimpleNamespace(
            alignment=alignment,
            full_headers=list(self.HEADERS),
            selected_indices=list(selected),
            cluster_labels=np.array([1, 2, 3]),
            group_labels=None,
            metadata={
                "Organism": {
                    "type": "text",
                    "values": ["Escherichia coli", "Escherichia coli", "Bacillus subtilis"],
                }
            },
            active_reference="Escherichia_a",
            console_text=SimpleNamespace(text=""),
            background_job_scheduler=scheduler,
        )
        output = io.StringIO()
        with mock.patch.object(logo_command, "LOGO_DIRECTORY", directory.name), \
                mock.patch.object(logo_command.cfg, "HEADER_LIST_DIR", directory.name), \
                redirect_stdout(output):
            logo_command.run(viewer, list(args))
        return viewer, scheduler.job, output.getvalue()

    def test_a_second_bracket_argument_is_refused_not_used_as_the_filename(self):
        for args in (
            ["[1-2]", "[3-4]"],
            ["[1-2]", "#cluster_1#", "[3-4]"],
            ["[1-2]", "[3-4]", "out.svg"],
            ["[1-2]", "[3,", "out.svg"],
        ):
            with self.subTest(args=args):
                viewer, job, _ = self.run_logo(args)
                self.assertIsNone(job)
                self.assertIn("second one", viewer.console_text.text)

    def test_bracket_looking_filenames_with_an_extension_remain_filenames(self):
        viewer, job, _ = self.run_logo(["[1-2]", "[3-4].svg"])

        self.assertEqual(job["payload"]["filename"], "[3-4].svg")
        self.assertEqual(job["payload"]["plot_positions"], (1, 2))

    def test_positions_split_by_spaces_are_joined(self):
        for args in (["[1,", "3]"], ["[1", ",", "3", "]"], ["[", "1,3", "]"]):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args + ["pcts", "out.svg"])
                self.assertEqual(job["payload"]["plot_positions"], (1, 3))
                self.assertEqual(job["payload"]["mode"], "pcts")
                self.assertEqual(job["payload"]["filename"], "out.svg")
                self.assertIsNone(job["payload"]["identity_threshold"])

    def test_a_number_inside_split_brackets_is_not_an_identity_threshold(self):
        _, job, _ = self.run_logo(["[1,", "5", "]", "out.svg"])

        # Position 5 is not in the alignment; 5 is not read as "5%" either.
        self.assertEqual(job["payload"]["plot_positions"], (1,))
        self.assertIsNone(job["payload"]["identity_threshold"])

    def test_expression_tokens_are_joined_with_spaces(self):
        # Selects the two Escherichia nodes; "Escherichiacoli" matches none.
        _, job, _ = self.run_logo(
            ["{Organism=Escherichia", "coli}", "[1]", "out.svg"], selected=()
        )

        self.assertEqual(len(job["payload"]["selected_seqs"]), 2)

    def test_operators_between_separate_tokens_parse_as_before(self):
        for args in (
            ["#cluster_1#", "|", "#cluster_2#", "[1]", "out.svg"],
            ["#cluster_1#", "|", "#cluster_2#", "out.svg", "[1]"],
            ["!", "#cluster_3#", "[1]", "out.svg"],
            ["(#cluster_1#", "|", "#cluster_2#)", "&", "!", "#cluster_3#", "[1]", "out.svg"],
        ):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args, selected=())
                self.assertEqual(len(job["payload"]["selected_seqs"]), 2)

    def test_a_mistyped_keyword_is_refused_and_never_becomes_a_filename(self):
        # Each used to be the filename (nogap.svg, ...), leaving the mode at
        # its default and replacing an existing file of that name.
        for typo in ("nogap", "no-gap", "percent", "colourblind"):
            for args in (
                ["#cluster_1#", "[1]", typo],
                ["#cluster_1#", "[1]", "pcts", typo],
                [typo, "[1]", "#cluster_1#", "&", "#cluster_2#", typo],
            ):
                with self.subTest(args=args):
                    viewer, job, _ = self.run_logo(args, existing_files=[f"{typo}.svg"])
                    self.assertIsNone(job)
                    self.assertEqual(
                        viewer.console_text.text,
                        f"Error: Unrecognized logo argument '{typo}'. "
                        "A filename must end in .svg or .png.",
                    )
                    self.assertEqual(os.listdir(self.directory), [f"{typo}.svg"])
                    with open(os.path.join(self.directory, f"{typo}.svg"), "rb") as kept:
                        self.assertEqual(kept.read(), b"previous")

    def test_an_extensionless_last_word_may_end_a_valid_multi_word_expression(self):
        for args, selected_rows in (
            (["{Organism=Escherichia", "coli}", "[1]"], 2),
            (["#cluster_1#", "|", "#cluster_2#", "[1]"], 2),
            (["#cluster_1#", "&", "!", "#cluster_2#", "[1]"], 1),
            (["[1]", "!", "#cluster_3#"], 2),
        ):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args, selected=())
                self.assertEqual(len(job["payload"]["selected_seqs"]), selected_rows)
                self.assertRegex(job["payload"]["filename"], r"^logo_\d{8}_\d{6}\.svg$")
                self.assertFalse(job["allow_overwrite"])

    def test_explicit_png_and_svg_names_end_the_arguments_in_any_case(self):
        cases = (
            (["#cluster_1#", "[1]", "target_logo.png"], "target_logo.png", 1),
            (["[1]", "plain.svg"], "plain.svg", 3),
            (["[1]", "UPPER.SVG"], "UPPER.SVG", 3),
            (["#cluster_1#", "[1]", "Mixed.PnG"], "Mixed.PnG", 1),
            (["#cluster_1#", "|", "#cluster_2#", "[1]", "pair.png"], "pair.png", 2),
            (["#cluster_1#", "[1,2]", "pcts", "no_gap", "modes.svg"], "modes.svg", 1),
        )
        for args, filename, selected_rows in cases:
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args, selected=(0, 1, 2))
                self.assertEqual(job["payload"]["filename"], filename)
                self.assertEqual(len(job["payload"]["selected_seqs"]), selected_rows)
                self.assertTrue(job["allow_overwrite"])

    def test_documented_examples_still_work(self):
        _, job, _ = self.run_logo(["#cluster_1#", "[1,2]", "pcts", "no_gap"])
        self.assertEqual(
            (job["payload"]["mode"], job["payload"]["gap_mode"]), ("pcts", "no_gap")
        )
        self.assertEqual(len(job["payload"]["selected_seqs"]), 1)
        self.assertRegex(job["payload"]["filename"], r"^logo_\d{8}_\d{6}\.svg$")

        _, job, _ = self.run_logo(["#cluster_1#", "[1]", "target_logo.png"])
        self.assertEqual(job["payload"]["filename"], "target_logo.png")

    def test_a_last_string_that_belongs_to_an_expression_keeps_the_selection_error(self):
        # Only an extensionless plain word is named as unrecognized; a broken
        # expression, or one followed by a real filename, reports its own error.
        for args in (
            ["#cluster_1#", "&", "(#cluster_2#", "[1]"],
            ["#cluster_1#", "&", "[1]"],
            ["#cluster_1#", "nogap", "[1]", "out.svg"],
            ["[1]", "nogap"],
        ):
            with self.subTest(args=args):
                viewer, job, _ = self.run_logo(args)
                self.assertIsNone(job)
                self.assertTrue(viewer.console_text.text.startswith("Logo error: "))
                self.assertNotIn("Unrecognized logo argument", viewer.console_text.text)

    def test_color_values_are_validated_before_the_job_is_queued(self):
        for value in ("nonsense", "", "classicc", "#12", "notacolor"):
            with self.subTest(value=value):
                viewer, job, _ = self.run_logo(["[1]", f"color={value}", "out.svg"])
                self.assertIsNone(job)
                self.assertIn("Unknown color scheme", viewer.console_text.text)
                self.assertIn(f"'{value}'", viewer.console_text.text)

    def test_preset_names_match_any_case_and_other_colors_pass_through(self):
        cases = (
            ("color=Classic", "classic"),
            ("color=NAJAFABADIETAL2017", "NajafabadiEtAl2017"),
            ("scheme=najafabadietal2017", "NajafabadiEtAl2017"),
            ("colors=Colorblind_Safe", "colorblind_safe"),
            ("color=classic", "classic"),
            ("color=red", "red"),
            ("color=Red", "Red"),
            ("color=#ff0000", "#ff0000"),
            ("color=0.5", "0.5"),
            ("Classic", "classic"),
            ("NajafabadiEtAl2017", "NajafabadiEtAl2017"),
        )
        for argument, expected in cases:
            with self.subTest(argument=argument):
                _, job, _ = self.run_logo(["[1]", argument, "out.svg"])
                self.assertEqual(job["payload"]["color_scheme"], expected)

    def test_default_color_scheme_is_unchanged(self):
        _, job, _ = self.run_logo(["[1]", "out.svg"])

        self.assertEqual(job["payload"]["color_scheme"], "chemistry")

    def test_every_accepted_color_value_renders_in_logomaker(self):
        # What the command accepts must be what the job's logomaker accepts.
        import logomaker
        import pandas as pd
        from matplotlib.figure import Figure

        frame = pd.DataFrame([[0.5] * 20], columns=list(logo_command.STANDARD_AAS))
        for value in ("classic", "NajafabadiEtAl2017", "red", "#ff0000", "0.5"):
            with self.subTest(value=value):
                logomaker.Logo(frame, ax=Figure().subplots(), color_scheme=value)

    def test_a_filename_that_is_only_an_extension_is_refused(self):
        for name in (".svg", ".png"):
            with self.subTest(name=name):
                viewer, job, _ = self.run_logo(["[1]", name])
                self.assertIsNone(job)
                self.assertEqual(
                    viewer.console_text.text,
                    f"Error: Filename '{name}' needs a name before its extension.",
                )

    def test_a_range_too_large_to_expand_is_refused(self):
        viewer, job, _ = self.run_logo(["[1-5000000]", "out.svg"])

        self.assertIsNone(job)
        self.assertIn("too large", viewer.console_text.text)

    def test_ordinary_range_still_warns_about_positions_not_found(self):
        _, job, output = self.run_logo(["[3-6]", "out.svg"])

        self.assertEqual(job["payload"]["plot_positions"], (3, 4))
        self.assertIn("Warning: Position 5 was not found", output)
        self.assertIn("Warning: Position 6 was not found", output)

    def test_help_documents_hidden_nodes_and_bare_threshold_numbers(self):
        output = io.StringIO()
        with redirect_stdout(output):
            logo_command.print_help()
        text = " ".join(output.getvalue().split())

        self.assertIn("Hidden nodes are included", text)
        self.assertIn("1 means 100%", text)
        self.assertIn("5 means 5%", text)
        self.assertIn("not case-sensitive", text)

    def test_help_states_the_filename_rule(self):
        output = io.StringIO()
        with redirect_stdout(output):
            logo_command.print_help()
        text = " ".join(output.getvalue().split())

        self.assertIn("A filename must end in .svg or .png", text)
        self.assertNotIn("the LAST is the filename", text)


class LogoPositionAxisLabelTests(unittest.TestCase):
    def test_axis_label_names_the_numbering_actually_plotted(self):
        label = logo_command._position_axis_label
        self.assertEqual(
            label({"numbering": "occupancy", "ref_id": "WP_1"}),
            "Position (occupancy numbering)",
        )
        self.assertEqual(
            label({"numbering": "reference", "ref_id": "WP_1"}),
            "Position (relative to WP_1)",
        )


if __name__ == "__main__":
    unittest.main()
