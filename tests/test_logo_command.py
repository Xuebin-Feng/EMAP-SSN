"""The `logo` command (commands/logo.py).

Covers position parsing and compact plot coordinates, the y axis, letter
heights with and without identity weighting, the identity kernels and their
thread budget, the background SVG renderer, the size limit of a PNG logo, the
x axis label, and the job `logo` enqueues and the outcome it reports.
"""

import io
import os
import re
import subprocess
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
import Command_Engine
from tests.sparse_alignment import sparse_alignment
from commands import logo as logo_command
from utilities.Job_Cancellation import cancel_requested
from utilities import Alignment_Profiles
from utilities.Alignment_Profiles import (
    _calculate_identity_neighbour_counts_numpy,
    encode_standard_amino_acids,
)
from commands.logo import (
    _configure_logo_y_axis,
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


class LogoSpaceSeparatedPositionTests(unittest.TestCase):
    """A position list is comma-separated or space-separated, and never both."""

    def test_both_forms_give_the_same_positions(self):
        for commas, spaces, expected in (
            ("[1,5,9-12]", "[1 5 9-12]", [1, 5, 9, 10, 11, 12]),
            ("[ 1, 5 ]", "[ 1 5 ]", [1, 5]),
            ("[10,10.1,10.2,11]", "[10 10.1 10.2 11]", [10, "10.1", "10.2", 11]),
            ("[(-3)-(-1),0,5]", "[(-3)-(-1) 0 5]", [-3, -2, -1, 0, 5]),
            ("[(-1.1),(-1),0]", "[(-1.1) (-1) 0]", [-1, "-1.1", 0]),
            ("[+1,+1.1]", "[+1 +1.1]", [1, "1.1"]),
            ("[11,10.2,10,10.1]", "[11 10.2 10 10.1]", [10, "10.1", "10.2", 11]),
            ("[3,1,3]", "[3 1 3]", [1, 3]),
        ):
            with self.subTest(commas=commas):
                self.assertEqual(parse_logo_positions(commas), expected)
            with self.subTest(spaces=spaces):
                self.assertEqual(parse_logo_positions(spaces), expected)

    def test_spaces_just_inside_the_brackets_are_ignored(self):
        for position_spec in ("[1 5]", "[ 1 5]", "[1 5 ]", "[   1   5   ]", " [ 1 5 ] "):
            with self.subTest(position_spec=position_spec):
                self.assertEqual(parse_logo_positions(position_spec), [1, 5])
        for position_spec in ("[1,5]", "[ 1,5]", "[1,5 ]", "[   1 ,  5   ]", " [ 1, 5 ] "):
            with self.subTest(position_spec=position_spec):
                self.assertEqual(parse_logo_positions(position_spec), [1, 5])

    def test_a_range_may_have_spaces_around_its_dash_in_either_form(self):
        for position_spec, expected in (
            ("[9 - 12]", [9, 10, 11, 12]),
            ("[9- 12]", [9, 10, 11, 12]),
            ("[1, 9 - 12]", [1, 9, 10, 11, 12]),
            ("[1 9 - 12]", [1, 9, 10, 11, 12]),
            ("[1 9- 12 20]", [1, 9, 10, 11, 12, 20]),
            ("[9 - 12 1]", [1, 9, 10, 11, 12]),
            ("[(-3) - (-1) 0 5]", [-3, -2, -1, 0, 5]),
            ("[(-3) - 1]", [-3, -2, -1, 0, 1]),
        ):
            with self.subTest(position_spec=position_spec):
                self.assertEqual(parse_logo_positions(position_spec), expected)

    def test_a_list_that_mixes_commas_and_spaces_is_refused_naming_the_item(self):
        for position_spec, item in (
            ("[1, 5 9]", "5 9"),
            ("[1 5, 9]", "1 5"),
            ("[1 5,]", "1 5"),
            ("[1,5 ,9 12]", "9 12"),
            ("[10 10.1, 11]", "10 10.1"),
            ("[1, (-3) 0]", "(-3) 0"),
        ):
            with self.subTest(position_spec=position_spec):
                with self.assertRaises(ValueError) as caught:
                    parse_logo_positions(position_spec)
                self.assertIsInstance(caught.exception.args[0], logo_command.Message)
                self.assertIn(f"'{item}'", str(caught.exception))
                self.assertIn("commas or with spaces, not both", str(caught.exception))

    def test_a_space_before_a_dash_and_none_after_it_is_still_a_bare_negative(self):
        for position_spec, correction in (
            ("[1 -5]", "(-5)"),
            ("[1 5 -3]", "(-3)"),
            ("[9 -12]", "(-12)"),
            ("[1, -5]", "(-5)"),
            ("[-1 2]", "(-1)"),
        ):
            with self.subTest(position_spec=position_spec):
                with self.assertRaisesRegex(ValueError, re.escape(correction)):
                    parse_logo_positions(position_spec)

    def test_ranges_keep_their_rules_in_a_space_separated_list(self):
        with self.assertRaisesRegex(ValueError, "lower to higher"):
            parse_logo_positions("[1 12 - 9]")
        with self.assertRaisesRegex(ValueError, "list insertion positions explicitly"):
            parse_logo_positions("[1 10.1 - 11]")
        with self.assertRaisesRegex(ValueError, "Invalid position label"):
            parse_logo_positions("[1 x 3]")

    def test_an_empty_list_gives_no_positions(self):
        for position_spec in ("[]", "[ ]", "[,]"):
            with self.subTest(position_spec=position_spec):
                self.assertEqual(parse_logo_positions(position_spec), [])

    def test_the_size_cap_counts_a_space_separated_list_too(self):
        with mock.patch.object(logo_command, "_MAX_LOGO_RANGE_POSITIONS", 10):
            self.assertEqual(parse_logo_positions("[1 - 10]"), list(range(1, 11)))
            with self.assertRaisesRegex(ValueError, "too large"):
                parse_logo_positions("[1 - 6 7 - 12]")


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
        encoded = encode_standard_amino_acids(sequences)
        multiplicities = np.array([2, 1, 3, 1, 2, 1, 4, 2], dtype=np.int64)

        for threshold in (0.8, 0.9, 1.0):
            with self.subTest(threshold=threshold):
                expected = _calculate_identity_neighbour_counts_numpy(
                    encoded,
                    multiplicities,
                    threshold,
                    block_size=3,
                )
                actual, _ = Alignment_Profiles.run_identity_neighbour_counts(
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
        with mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", False):
            fallback = calculate_identity_weights(sequences, 0.9)

        np.testing.assert_array_equal(accelerated, fallback)

    def test_empty_and_singleton_weighting(self):
        self.assertEqual(calculate_identity_weights([], 0.9).size, 0)
        np.testing.assert_array_equal(
            calculate_identity_weights(["ACDE"], 0.9),
            [1.0],
        )

    def test_numba_unavailable_uses_exact_numpy_fallback(self):
        with mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", False):
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
            Alignment_Profiles,
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
                    Alignment_Profiles.choose_balanced_thread_count(
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
                Alignment_Profiles,
                "get_num_threads",
                return_value=20,
            ),
            mock.patch.object(
                Alignment_Profiles,
                "set_num_threads",
            ) as set_threads,
            mock.patch.object(
                Alignment_Profiles.os,
                "cpu_count",
                return_value=20,
            ),
            mock.patch.object(
                Alignment_Profiles,
                "_identity_neighbour_counts_kernel",
                return_value=expected_counts,
            ),
        ):
            counts, selected_threads = (
                Alignment_Profiles.run_identity_neighbour_counts(
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
                Alignment_Profiles,
                "get_num_threads",
                return_value=4,
            ),
            mock.patch.object(
                Alignment_Profiles,
                "set_num_threads",
            ) as set_threads,
            mock.patch.object(
                Alignment_Profiles.os,
                "cpu_count",
                return_value=4,
            ),
            mock.patch.object(
                Alignment_Profiles,
                "_identity_neighbour_counts_kernel",
                side_effect=RuntimeError("kernel failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                Alignment_Profiles.run_identity_neighbour_counts(
                    np.zeros((1, 1), dtype=np.int8),
                    np.ones(1, dtype=np.int64),
                    0.9,
                )

        self.assertEqual(
            set_threads.call_args_list,
            [mock.call(2), mock.call(4)],
        )


class LogoColumnHeightTests(unittest.TestCase):
    """Alignment_Profiles.logo_column_heights is calculate_logo_matrix's default column height, from counts."""

    @staticmethod
    def counts_of(sequences):
        encoded = Alignment_Profiles.encode_standard_amino_acids(sequences)
        counts = np.zeros((encoded.shape[1], 20), dtype=np.int64)
        for column in range(encoded.shape[1]):
            counts[column] = np.bincount(encoded[:, column][encoded[:, column] >= 0], minlength=20)
        return counts

    def test_it_equals_the_matrix_functions_column_sums_on_random_alignments(self):
        rng = np.random.default_rng(42)
        alphabet = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "-", "-", "X", "x", "a"]
        for rows, columns in ((1, 6), (2, 9), (7, 12), (60, 25), (200, 40)):
            with self.subTest(rows=rows, columns=columns):
                sequences = ["".join(rng.choice(alphabet, size=columns)) for _ in range(rows)]
                # One column of gaps only, and one conserved column.
                sequences = [row[:1] + "-" + row[2:3] + "A" + row[4:] for row in sequences]
                matrix, _ = calculate_logo_matrix(sequences, list(range(columns)), mode="bits", gap_mode="with_gap")
                heights = Alignment_Profiles.logo_column_heights(self.counts_of(sequences), rows)
                np.testing.assert_allclose(heights, matrix.sum(axis=1), rtol=1e-12, atol=1e-12)
                self.assertEqual(heights[1], 0.0)

    def test_hand_computed_columns(self):
        log2_20 = np.log2(20)
        # 40 of 40 rows have A: no entropy, only the small-sample correction.
        conserved = Alignment_Profiles.logo_column_heights([[40] + [0] * 19], 40)
        self.assertAlmostEqual(float(conserved[0]), log2_20 - 19.0 / (2.0 * np.log(2) * 40), places=12)
        # 20 of 40 rows have A and the rest have gaps: the same information, half as tall.
        gapped = Alignment_Profiles.logo_column_heights([[20] + [0] * 19], 40)
        self.assertAlmostEqual(float(gapped[0]), (log2_20 - 19.0 / (2.0 * np.log(2) * 20)) / 2, places=12)
        # Two sequences carry less than the correction takes away.
        self.assertEqual(float(Alignment_Profiles.logo_column_heights([[2] + [0] * 19], 2)[0]), 0.0)
        # An empty column, or a count table with no columns.
        self.assertEqual(float(Alignment_Profiles.logo_column_heights([[0] * 20], 5)[0]), 0.0)
        self.assertEqual(Alignment_Profiles.logo_column_heights(np.zeros((0, 20)), 5).shape, (0,))
        # Equal shares of all twenty residues: the entropy is log2(20) and nothing is left.
        self.assertEqual(float(Alignment_Profiles.logo_column_heights([[10] * 20], 200)[0]), 0.0)


class LogoUtilityAliasTests(unittest.TestCase):
    """The computation lives in utilities.Alignment_Profiles; logo calls it through the names it imports."""

    def test_patching_the_utility_reaches_logos_aliases(self):
        with mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", False):
            _, metadata = logo_command.calculate_identity_weights(
                ["AAAA", "AAAA", "GGGG"], 0.9, return_metadata=True
            )
        self.assertEqual(metadata["backend"], "numpy")
        _, metadata = logo_command.calculate_identity_weights(
            ["AAAA", "AAAA", "GGGG"], 0.9, return_metadata=True
        )
        self.assertEqual(metadata["backend"], "numba" if Alignment_Profiles.NUMBA_AVAILABLE else "numpy")

    def test_the_utility_needs_no_viewer_settings_or_qt(self):
        # Fresh interpreter: what the import itself pulls in.
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import utilities.Alignment_Profiles;"
            "bad = sorted(name for name in sys.modules if name.split('.')[0] in "
            "('PySide6', 'Command_Engine', 'EMAPSSN_Config', 'commands'));"
            "print(bad)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, SRC_DIR], capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "[]")


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

    def flat_payload(self, directory, sequences, mode="bits", filename="flat.svg"):
        positions = tuple(range(len(sequences[0])))
        return {**self.make_payload(directory, filename=filename), "selected_seqs": tuple(sequences),
                "valid_cols": positions, "plot_positions": tuple(position + 10 for position in positions),
                "mode": mode, "gap_mode": "with_gap", "identity_threshold": None}

    def render_with_warnings(self, payload):
        """The result of rendering, the warnings it raised, and the limits of the figure's y axis."""
        import warnings

        limits = []

        def save(figure, target, **_kwargs):
            limits.append(figure.axes[0].get_ylim())
            target.write("rendered content")

        with mock.patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=save), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = logo_command._generate_logo_artifact(payload)
        return result, [str(item.message) for item in caught], limits[0]

    def test_a_logo_of_two_sequences_is_flat_and_says_so_without_a_matplotlib_warning(self):
        # With two sequences the small-sample correction (19 / (2 ln 2 n) = 6.8 bits) takes all of the
        # log2(20) = 4.3 bits, so every letter is 0 tall and logomaker limited the axes to (0, 0).
        with tempfile.TemporaryDirectory() as directory:
            payload = self.flat_payload(directory, ("ACDEFGHIKL", "ACDEFGHIKL"), filename="r6_logo.svg")
            result, warned, ylim = self.render_with_warnings(payload)
            self.assertTrue(os.path.isfile(result["save_path"]))
        self.assertEqual(warned, [])
        # The fixed scale the other logos have, so a flat one reads against them.
        self.assertAlmostEqual(ylim[0], 0.0)
        self.assertAlmostEqual(ylim[1], np.log2(20))
        text = str(result["message"])
        self.assertTrue(text.startswith("Saved with_gap bits logo for 2 aligned nodes to r6_logo.svg "), text)
        self.assertIn("The logo is flat: every position has 0 bits, so it draws no letters.", text)
        self.assertIn("2 aligned sequences are too few", text)
        self.assertIn("pcts", text)

    def test_a_logo_with_no_residue_at_its_positions_is_flat_in_pcts_mode_too(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.flat_payload(directory, ("----", "----", "----"), mode="pcts")
            result, warned, ylim = self.render_with_warnings(payload)
        self.assertEqual(warned, [])
        self.assertEqual(ylim, (0.0, 1.0))
        self.assertTrue(str(result["message"]).endswith(
            "The logo is flat: no aligned node has a residue at any of these positions, so it draws no letters."))

    def test_a_logo_with_letters_has_no_note(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.flat_payload(directory, ("ACDE",) * 12 + ("ACDF",))
            result, warned, ylim = self.render_with_warnings(payload)
        self.assertEqual(warned, [])
        self.assertEqual(str(result["message"]), "Saved with_gap bits logo for 13 aligned nodes to flat.svg")
        self.assertAlmostEqual(ylim[1], np.log2(20))

    def test_the_warning_is_left_out_for_the_one_call_only(self):
        import warnings
        from matplotlib.figure import Figure

        before = list(warnings.filters)
        with tempfile.TemporaryDirectory() as directory:
            self.render_with_warnings(self.flat_payload(directory, ("ACDE", "ACDE")))
        self.assertEqual(warnings.filters, before)
        # matplotlib still warns about the same range anywhere else, in this process.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Figure().subplots().set_ylim(0, 0)
        self.assertEqual(len(caught), 1)
        self.assertIn("identical low and high ylims", str(caught[0].message))

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

            def save_and_create_competing_output(_figure, target, **_kwargs):
                # The figure is saved into the partial file through the renderer's cancellable
                # file, which for an SVG is a text file, as matplotlib opens a path for one.
                target.write("rendered content")
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

    def test_a_space_separated_list_is_one_argument_however_the_line_split_it(self):
        for args in (
            ["[1", "2", "4]"],
            ["[", "1", "2", "4", "]"],
            ["[1", "2", "4", "]"],
            ["[", "1", "2", "4]"],
            ["[1 2 4]"],
            ["[ 1 2 4 ]"],
        ):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args + ["pcts", "no_gap", "out.svg"])
                self.assertEqual(job["payload"]["plot_positions"], (1, 2, 4))
                self.assertEqual(job["payload"]["mode"], "pcts")
                self.assertEqual(job["payload"]["gap_mode"], "no_gap")
                self.assertEqual(job["payload"]["filename"], "out.svg")
                self.assertIsNone(job["payload"]["identity_threshold"])

    def test_a_number_in_a_space_separated_list_is_never_an_identity_threshold(self):
        for args in (
            ["[1", "2", "4]", "out.svg"],
            ["[", "4", "]", "out.svg"],
            ["[1", "2", "3", "4", "]", "out.svg"],
            ["[1", "-", "4]", "out.svg"],
        ):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args)
                self.assertIsNotNone(job)
                self.assertIsNone(job["payload"]["identity_threshold"])

    def test_the_identity_threshold_beside_a_space_separated_list_is_unchanged(self):
        for args, threshold in (
            (["[1", "2", "4]", "5"], 0.05),
            (["5", "[1", "2", "4]"], 0.05),
            (["[1", "2", "4]", "90%", "out.svg"], 0.9),
            (["[1", "2", "4]", "0.5", "out.svg"], 0.5),
            (["[1", "2", "4]", "90", "out.svg"], 0.9),
        ):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args)
                self.assertEqual(job["payload"]["plot_positions"], (1, 2, 4))
                self.assertEqual(job["payload"]["identity_threshold"], threshold)

    def test_the_filename_and_the_expression_beside_a_space_separated_list_are_unchanged(self):
        # Cluster 1 is the first node alone and cluster 2 the second.
        for args, filename, sequences in (
            (["#cluster_1#", "[1", "2]", "out.svg"], "out.svg", 1),
            (["[1", "2]", "#cluster_1#", "|", "#cluster_2#", "out.svg"], "out.svg", 2),
            (["#cluster_1#", "|", "#cluster_2#", "[1", "2]", "out.png"], "out.png", 2),
            (["{Organism=Escherichia", "coli}", "[1", "2]", "out.svg"], "out.svg", 2),
            (["[1", "2]", "out.svg", "pcts"], "out.svg", 3),
        ):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args, selected=())
                self.assertEqual(job["payload"]["filename"], filename)
                self.assertEqual(len(job["payload"]["selected_seqs"]), sequences)
                self.assertEqual(job["payload"]["plot_positions"], (1, 2))
                self.assertIsNone(job["payload"]["identity_threshold"])

    def test_a_space_separated_list_with_no_filename_keeps_the_automatic_name(self):
        _, job, _ = self.run_logo(["[1", "2]", "#cluster_1#"], selected=())

        self.assertRegex(job["payload"]["filename"], r"^logo_\d{8}_\d{6}.*\.svg$")
        self.assertEqual(len(job["payload"]["selected_seqs"]), 1)

    def test_a_spaced_range_dash_in_a_command_line_is_a_range(self):
        for args, positions in (
            (["[1", "3", "-", "4]"], (1, 3, 4)),
            (["[1", "-", "3]"], (1, 2, 3)),
            (["[1", "-", "3", "4]"], (1, 2, 3, 4)),
            (["[1,", "3", "-", "4]"], (1, 3, 4)),
            (["[3", "-", "4", "1]"], (1, 3, 4)),
        ):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args + ["out.svg"])
                self.assertEqual(job["payload"]["plot_positions"], positions)
                self.assertIsNone(job["payload"]["identity_threshold"])

    def test_positions_the_alignment_lacks_are_warned_about_in_either_form(self):
        for args in (["[(-1)", "1", "2.1", "3]"], ["[(-1),", "1,", "2.1,", "3]"]):
            with self.subTest(args=args):
                _, job, output = self.run_logo(args + ["out.svg"])
                self.assertEqual(job["payload"]["plot_positions"], (1, 3))
                self.assertIn("Position -1 was not found", output)
                self.assertIn("Position 2.1 was not found", output)

    def test_a_mixed_list_is_refused_and_nothing_is_queued_or_written(self):
        for args, item in (
            (["[1,", "2", "4]", "out.svg"], "2 4"),
            (["[1", "2,", "4]", "out.svg"], "1 2"),
            (["[1", "2,", "4,]", "out.svg"], "1 2"),
            (["[1,", "(-1)", "4]", "out.svg"], "(-1) 4"),
        ):
            with self.subTest(args=args):
                viewer, job, _ = self.run_logo(args, existing_files=["out.svg"])
                self.assertIsNone(job)
                self.assertEqual(
                    viewer.console_text.text,
                    "Error: Separate positions with commas or with spaces, not both: "
                    f"'{item}' has a space inside a comma-separated list. "
                    "Write [1,5,9-12] or [1 5 9-12].",
                )
                with open(os.path.join(self.directory, "out.svg"), "rb") as kept:
                    self.assertEqual(kept.read(), b"previous")
                self.assertEqual(os.listdir(self.directory), ["out.svg"])

    def test_a_bare_negative_in_a_space_separated_list_is_refused(self):
        for args, correction in (
            (["[1", "-2]", "out.svg"], "(-2)"),
            (["[1", "2", "-3]", "out.svg"], "(-3)"),
            (["[1", "-2", "3]", "out.svg"], "(-2)"),
        ):
            with self.subTest(args=args):
                viewer, job, _ = self.run_logo(args)
                self.assertIsNone(job)
                self.assertIn(correction, viewer.console_text.text)
                self.assertIn("Parentheses are required", viewer.console_text.text)

    def test_a_list_that_is_never_closed_queues_nothing(self):
        for args in (["[1", "2", "4", "out.svg"], ["[1", "2"], ["[1", "2", "4", "5"]):
            with self.subTest(args=args):
                _, job, _ = self.run_logo(args)
                self.assertIsNone(job)

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

    def test_help_shows_both_position_forms_and_refuses_mixing(self):
        output = io.StringIO()
        with redirect_stdout(output):
            logo_command.print_help()
        text = " ".join(output.getvalue().split())

        self.assertIn("separated by commas or by spaces", text)
        self.assertIn("[1,5,9-12] and [1 5 9-12] both work", text)
        self.assertIn("mixes commas and spaces, such as [1, 5 9], is refused", text)
        self.assertIn("logo [10 10.1 10.2 11]", text)

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


class LogoPngSizeTests(unittest.TestCase):
    """A PNG logo is one raster at 600 dpi, so its size is limited; an SVG is not."""

    def test_raster_estimate_reads_the_figure_and_dpi_the_renderer_uses(self):
        # Capture the figure size and dpi of a real render instead of restating them.
        seen = {}

        def save(figure, target, **kwargs):
            seen["inches"] = tuple(figure.get_size_inches())
            seen["dpi"] = kwargs["dpi"]
            target.write(b"x")

        for positions in (1, 11, 25):
            with self.subTest(positions=positions), tempfile.TemporaryDirectory() as directory:
                payload = LogoSynchronousArtifactTests.make_payload(directory, "size.png")
                payload["selected_seqs"] = ("A" * positions, "C" * positions)
                payload["valid_cols"] = tuple(range(positions))
                payload["plot_positions"] = tuple(range(1, positions + 1))
                with mock.patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=save):
                    logo_command._generate_logo_artifact(payload)

                width, height, raster_bytes = logo_command._png_logo_raster(positions)
                self.assertEqual(
                    (width, height),
                    (round(seen["inches"][0] * seen["dpi"]), round(seen["inches"][1] * seen["dpi"])),
                )
                self.assertEqual(raster_bytes, width * height * 4)

    def test_raster_estimate_bounds_the_png_that_is_saved(self):
        # Agg draws the whole figure, then bbox_inches='tight' crops it by about
        # 0.1 inch a side, so the estimate is a little over the saved size.
        with tempfile.TemporaryDirectory() as directory:
            payload = LogoSynchronousArtifactTests.make_payload(directory, "measured.png")
            payload["selected_seqs"] = ("A" * 12, "C" * 12)
            payload["valid_cols"] = tuple(range(12))
            payload["plot_positions"] = tuple(range(1, 13))
            path = logo_command._generate_logo_artifact(payload)["save_path"]
            with open(path, "rb") as image:
                header = image.read(24)
        saved = (int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big"))

        estimate = logo_command._png_logo_raster(12)[:2]
        margin = 0.25 * logo_command._LOGO_DPI
        for estimated, actual in zip(estimate, saved):
            self.assertGreaterEqual(estimated, actual)
            self.assertLess(estimated - actual, margin)

    def test_the_agg_limit_is_the_one_matplotlib_enforces(self):
        from matplotlib.backends.backend_agg import RendererAgg

        limit = logo_command._AGG_MAX_SIDE_PIXELS
        RendererAgg(limit - 1, 1, 100)
        RendererAgg(1, limit - 1, 100)
        with self.assertRaisesRegex(ValueError, "too large"):
            RendererAgg(limit, 1, 100)
        with self.assertRaisesRegex(ValueError, "too large"):
            RendererAgg(1, limit, 100)

    def test_a_raster_just_under_the_memory_budget_fits_and_just_over_does_not(self):
        budget = logo_command._png_logo_raster(20)[2]

        with mock.patch.object(logo_command, "_MAX_PNG_RASTER_BYTES", budget):
            self.assertTrue(logo_command._png_logo_fits(20))
            self.assertFalse(logo_command._png_logo_fits(21))
            self.assertEqual(logo_command._max_png_logo_positions(), 20)
        with mock.patch.object(logo_command, "_MAX_PNG_RASTER_BYTES", budget - 1):
            self.assertFalse(logo_command._png_logo_fits(20))
            self.assertTrue(logo_command._png_logo_fits(19))

    def test_a_side_just_under_the_agg_limit_fits_and_just_over_does_not(self):
        width, height, _ = logo_command._png_logo_raster(20)
        # Leave the memory budget out of it.
        with mock.patch.object(logo_command, "_MAX_PNG_RASTER_BYTES", 1 << 60):
            with mock.patch.object(logo_command, "_AGG_MAX_SIDE_PIXELS", width + 1):
                self.assertTrue(logo_command._png_logo_fits(20))
                self.assertFalse(logo_command._png_logo_fits(21))
            with mock.patch.object(logo_command, "_AGG_MAX_SIDE_PIXELS", width):
                self.assertFalse(logo_command._png_logo_fits(20))
            # The height counts too.
            with mock.patch.object(logo_command, "_AGG_MAX_SIDE_PIXELS", height):
                self.assertFalse(logo_command._png_logo_fits(1))

    def test_the_real_limits_keep_ordinary_logos_and_stop_the_audits_case(self):
        largest = logo_command._max_png_logo_positions()

        self.assertTrue(logo_command._png_logo_fits(largest))
        self.assertFalse(logo_command._png_logo_fits(largest + 1))
        self.assertTrue(logo_command._png_logo_fits(1))
        self.assertTrue(logo_command._png_logo_fits(100))
        self.assertFalse(logo_command._png_logo_fits(1000))
        self.assertLessEqual(logo_command._png_logo_raster(largest)[2], 1 << 30)


class LogoOutcomeTests(unittest.TestCase):
    """What `logo` reports, and queues, for the nodes an expression matches."""

    HEADERS = ["node0", "node1", "node2"]

    def run_logo(self, args, columns=4):
        """Run `logo` over a viewer whose node2 is not in the alignment.

        Returns the viewer, the queued job (or None), and the mocks of
        command_failed and command_succeeded.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        alignment = SimpleNamespace(
            aln=sparse_alignment([("node0", "A" * columns), ("node1", "C" * columns)]),
            viewer_to_aln=np.array([0, 1, -1]),
            col_to_label={index: str(index + 1) for index in range(columns)},
            label_to_col={str(index + 1): index for index in range(columns)},
            has_reference=True,
        )
        scheduler = CapturingScheduler()
        viewer = SimpleNamespace(
            alignment=alignment,
            full_headers=list(self.HEADERS),
            selected_indices=[],
            cluster_labels=np.array([1, 1, 2]),
            group_labels=None,
            active_reference="node0",
            console_text=SimpleNamespace(text=""),
            background_job_scheduler=scheduler,
        )
        with mock.patch.object(logo_command, "LOGO_DIRECTORY", directory.name), \
                mock.patch.object(logo_command.cfg, "HEADER_LIST_DIR", directory.name), \
                mock.patch.object(Command_Engine, "command_failed") as failed, \
                mock.patch.object(Command_Engine, "command_succeeded") as succeeded, \
                redirect_stdout(io.StringIO()):
            logo_command.run(viewer, list(args))
        return viewer, scheduler.job, failed, succeeded

    def test_a_valid_expression_that_matches_no_node_succeeds_and_says_so(self):
        viewer, job, failed, succeeded = self.run_logo(['"zzz"', "[1]", "out.svg"])

        message = "No nodes matched the criteria for logo generation."
        self.assertIsNone(job)
        failed.assert_not_called()
        succeeded.assert_called_once()
        self.assertEqual(str(succeeded.call_args.args[1]), message)
        self.assertEqual(viewer.console_text.text, message)

    def test_nodes_that_match_but_are_not_in_the_alignment_are_a_failure(self):
        # node2 is the only member of cluster 2, and the alignment lacks it.
        viewer, job, failed, succeeded = self.run_logo(["#cluster_2#", "[1]", "out.svg"])

        message = "Error: No aligned nodes matched the logo selection criteria."
        self.assertIsNone(job)
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertEqual(str(failed.call_args.args[1]), message)
        self.assertEqual(viewer.console_text.text, message)

    def test_a_match_that_is_partly_aligned_is_queued(self):
        # Everything, including node2, matches; the aligned two are drawn.
        _, job, failed, _ = self.run_logo(['"*"', "[1]", "out.svg"])

        failed.assert_not_called()
        self.assertEqual(len(job["payload"]["selected_seqs"]), 2)

    def test_a_png_over_the_size_limit_is_refused_before_a_job_is_queued(self):
        largest = logo_command._max_png_logo_positions()
        columns = largest + 1

        for name in ("big.png", "BIG.PNG"):
            with self.subTest(name=name):
                viewer, job, failed, succeeded = self.run_logo(
                    [f"[1-{columns}]", name], columns=columns
                )

                message = (
                    f"Error: A PNG logo of {columns} positions is too large to draw; "
                    f"at most {largest} positions fit. Use an .svg filename, which has "
                    "no such limit, or fewer positions."
                )
                self.assertIsNone(job)
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertEqual(str(failed.call_args.args[1]), message)
                self.assertEqual(viewer.console_text.text, message)

    def test_the_most_a_png_holds_is_queued(self):
        largest = logo_command._max_png_logo_positions()

        _, job, failed, _ = self.run_logo([f"[1-{largest}]", "full.png"], columns=largest)

        failed.assert_not_called()
        self.assertEqual(job["payload"]["filename"], "full.png")
        self.assertEqual(len(job["payload"]["plot_positions"]), largest)

    def test_the_limit_counts_the_positions_that_are_plotted(self):
        # Positions the alignment lacks are not drawn, so they do not count.
        largest = logo_command._max_png_logo_positions()

        _, job, failed, _ = self.run_logo([f"[1-{largest + 50}]", "full.png"], columns=largest)

        failed.assert_not_called()
        self.assertEqual(len(job["payload"]["plot_positions"]), largest)

    def test_an_svg_is_never_refused_for_its_size(self):
        largest = logo_command._max_png_logo_positions()
        columns = largest + 100

        for args in (
            [f"[1-{columns}]", "wide.svg"],
            [f"[1-{columns}]"],
            [f"[1-{columns}]", "WIDE.SVG"],
        ):
            with self.subTest(args=args):
                _, job, failed, _ = self.run_logo(args, columns=columns)

                failed.assert_not_called()
                self.assertEqual(len(job["payload"]["plot_positions"]), columns)

    def test_help_states_the_png_limit_and_the_outcomes(self):
        output = io.StringIO()
        with redirect_stdout(output):
            logo_command.print_help()
        text = " ".join(output.getvalue().split())

        self.assertIn(
            f"a PNG logo may have at most {logo_command._max_png_logo_positions()} positions",
            text,
        )
        self.assertIn("An SVG has no such limit", text)
        self.assertIn("the command reports that none matched and queues no job", text)
        self.assertIn("leave nothing to analyse, and the command fails", text)
        self.assertIn("cut short with an ellipsis", text)


class LogoLongHeaderTests(unittest.TestCase):
    """A reference header too long for the x axis is shortened, never widening the image."""

    LONG_HEADER = "WP_0123456789.1_" + "very_long_protein_description_" * 12

    @staticmethod
    def svg_width(path):
        with open(path, encoding="utf-8") as svg:
            match = re.search(r'<svg[^>]*?width="([\d.]+)pt"', svg.read(2000), re.S)
        return float(match.group(1))

    def render(self, header, numbering="reference", filename="header.svg"):
        """Render a three-position logo; return its saved width and the x label drawn."""
        from matplotlib.figure import Figure

        drawn = {}
        real_savefig = Figure.savefig

        def save(figure, partial_path, **kwargs):
            drawn["label"] = figure.axes[0].get_xlabel()
            return real_savefig(figure, partial_path, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            payload = LogoSynchronousArtifactTests.make_payload(directory, filename)
            payload["selected_seqs"] = ("ACD", "ACD")
            payload["valid_cols"] = (0, 1, 2)
            payload["plot_positions"] = (1, 2, 3)
            payload["ref_id"] = header
            payload["numbering"] = numbering
            with mock.patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=save):
                path = logo_command._generate_logo_artifact(payload)["save_path"]
            with open(path, "rb") as image:
                head = image.read(24)
            width = (
                int.from_bytes(head[16:20], "big") if filename.endswith(".png")
                else self.svg_width(path)
            )
        return width, drawn["label"]

    def test_a_300_character_header_gives_the_same_width_as_a_short_one(self):
        short_width, short_label = self.render("WP_1.1")
        long_width, long_label = self.render("A" * 300)
        wordy_width, wordy_label = self.render(self.LONG_HEADER)

        self.assertEqual(short_label, "Position (relative to WP_1.1)")
        self.assertEqual(long_width, short_width)
        self.assertEqual(wordy_width, short_width)
        # The header was cut and ends in the ellipsis, inside the closing parenthesis.
        for label in (long_label, wordy_label):
            self.assertTrue(label.startswith("Position (relative to "), label)
            self.assertTrue(label.endswith("…)"), label)
        self.assertLess(len(long_label), 100)

    def test_a_header_that_fits_is_left_whole(self):
        for header in ("WP_1.1", "Escherichia_coli_K12_protein_A", "x"):
            with self.subTest(header=header):
                _, label = self.render(header)
                self.assertEqual(label, f"Position (relative to {header})")

    def test_the_png_is_as_wide_with_a_long_header_as_with_a_short_one(self):
        long_width, _ = self.render("A" * 300, filename="header.png")
        short_width, _ = self.render("WP_1.1", filename="header.png")

        self.assertEqual(long_width, short_width)

    def test_the_shortened_label_is_no_wider_than_the_axes_and_as_long_as_it_can_be(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(6, 4))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        fig.tight_layout()
        payload = {"numbering": "reference", "ref_id": self.LONG_HEADER}

        logo_command._fit_position_axis_label(ax, payload)

        renderer = fig.canvas.get_renderer()
        label = ax.xaxis.label
        axes_width = ax.get_window_extent(renderer).width
        self.assertLessEqual(label.get_window_extent(renderer).width, axes_width)
        # One more character of the header would not fit.
        kept = len(label.get_text()) - len("Position (relative to ") - len("…)")
        label.set_text(
            logo_command._position_axis_label(payload, self.LONG_HEADER[:kept + 1] + "…")
        )
        self.assertGreater(
            label.get_window_extent(renderer).width,
            axes_width * logo_command._AXIS_LABEL_FIT,
        )

    def test_the_occupancy_label_is_never_shortened(self):
        _, label = self.render("ignored", numbering="occupancy")

        self.assertEqual(label, "Position (occupancy numbering)")


class LogoJobCancellationTests(unittest.TestCase):
    """A logo job that is told to stop while it runs ends as cancelled and leaves no file.

    The job runs on the real background job scheduler's thread; each test asks for
    the stop from inside the stage named, and for both file types.
    """

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtCore

        cls.application = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def setUp(self):
        import time
        from Background_Job_Scheduler import BackgroundJobScheduler

        class StatusLine:
            def set_background_job_status(self, message):
                pass

        self.status = StatusLine()
        self.scheduler = BackgroundJobScheduler(self.status)
        self.addCleanup(self.scheduler.shutdown)
        self.reports = []
        self.scheduler.job_succeeded.connect(lambda job, result, elapsed: self.reports.append("succeeded"))
        self.scheduler.job_failed.connect(lambda job, error, detail, elapsed: self.reports.append("failed"))
        self.scheduler.job_cancelled.connect(
            lambda job, started: self.reports.append("stopped" if started else "removed"))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.time = time

    def run_job(self, directory, filename, allow_overwrite=False):
        payload = LogoSynchronousArtifactTests.make_payload(directory, filename, allow_overwrite)
        self.scheduler.enqueue("logo", f"logo -> {filename}", payload, logo_command._generate_logo_artifact,
                               os.path.join(directory, filename), allow_overwrite=allow_overwrite)
        deadline = self.time.monotonic() + 60
        while self.time.monotonic() < deadline:
            self.application.processEvents()
            if self.scheduler.queue_depth == 0 and self.reports:
                break
            self.time.sleep(0.005)
        self.application.processEvents()

    def stop_job(self):
        (running,) = [entry for entry in self.scheduler.snapshot() if entry.running]
        self.assertTrue(self.scheduler.cancel(running.job.job_id))

    def stop_when_called(self, owner, name):
        real = getattr(owner, name)
        calls = []

        def stopping(*args, **kwargs):
            if not calls:
                self.stop_job()
            calls.append(name)
            return real(*args, **kwargs)

        # autospec: a method patched on a class is still called with its instance.
        return mock.patch.object(owner, name, autospec=True, side_effect=stopping), calls

    def test_a_stop_during_each_stage_ends_the_job_with_no_file(self):
        import logomaker
        from matplotlib.figure import Figure

        for filename in ("stopped.svg", "stopped.png"):
            for owner, name in (
                (logo_command, "calculate_logo_matrix"),
                (logomaker, "Logo"),
                (Figure, "tight_layout"),
            ):
                with self.subTest(filename=filename, stage=name), tempfile.TemporaryDirectory() as directory:
                    self.reports.clear()
                    patch, calls = self.stop_when_called(owner, name)
                    with patch, mock.patch.object(logo_command, "new_partial_file",
                                                  wraps=logo_command.new_partial_file) as made:
                        self.run_job(directory, filename)
                    self.assertTrue(calls, "the stage ran")
                    self.assertEqual(self.reports, ["stopped"])
                    self.assertEqual(os.listdir(directory), [])
                    self.assertEqual(self.scheduler.snapshot(), [])
                    # The job stopped at its next check point, before it made its partial file to draw into.
                    made.assert_not_called()

    def test_a_stop_while_the_image_is_written_deletes_the_partial_file(self):
        for filename in ("midway.svg", "midway.png"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                self.reports.clear()
                seen, writes = [], []
                test = self

                class StopsOnWrite(logo_command.CancellableFile):
                    def write(inner, data):
                        if data:
                            writes.append(1)
                            if len(writes) == 1:
                                seen.extend(name for name in os.listdir(directory))
                                test.stop_job()
                        return super().write(data)

                with mock.patch.object(logo_command, "CancellableFile", StopsOnWrite):
                    self.run_job(directory, filename)
                self.assertEqual(len(seen), 1)
                self.assertIn(".partial.", seen[0], "the partial file existed when the stop came")
                self.assertEqual(self.reports, ["stopped"])
                self.assertEqual(os.listdir(directory), [])

    def close_the_viewer_during_the_save(self, directory, filename, *, job_obeys, wait=2.5):
        """Run a logo job and close the scheduler, as closing the Viewer does, at the first write of its image.

        With ``job_obeys`` the job waits at that write until it is told to stop; without, it
        holds there until the test lets it go, as a job stuck in one long call does. Returns
        (how long closing took, what the folder held when closing returned, which is before
        the job is let go).
        """
        import threading

        entered, release = threading.Event(), threading.Event()

        class HoldsAtFirstWrite(logo_command.CancellableFile):
            held = False

            def write(inner, data):
                if data and not HoldsAtFirstWrite.held:
                    HoldsAtFirstWrite.held = True
                    entered.set()
                    if job_obeys:
                        while not cancel_requested():
                            self.time.sleep(0.005)
                    else:
                        release.wait(30)
                return super().write(data)

        payload = LogoSynchronousArtifactTests.make_payload(directory, filename)
        with mock.patch.object(logo_command, "CancellableFile", HoldsAtFirstWrite):
            self.scheduler.enqueue("logo", f"logo -> {filename}", payload, logo_command._generate_logo_artifact,
                                   os.path.join(directory, filename))
            self.assertTrue(entered.wait(60))
            started = self.time.monotonic()
            self.scheduler.shutdown(wait=wait)
            took = self.time.monotonic() - started
            left_when_closed = sorted(os.listdir(directory))
            release.set()
            deadline = self.time.monotonic() + 30
            while self.scheduler._thread.is_alive() and self.time.monotonic() < deadline:
                self.time.sleep(0.005)
        return took, left_when_closed

    def test_closing_the_viewer_during_the_save_leaves_neither_a_partial_file_nor_the_logo(self):
        from Background_Job_Scheduler import BackgroundJobScheduler

        for filename in ("closing.svg", "closing.png"):
            for job_obeys in (True, False):
                with self.subTest(filename=filename, job_obeys=job_obeys), tempfile.TemporaryDirectory() as directory:
                    self.scheduler = BackgroundJobScheduler(self.status)
                    self.addCleanup(self.scheduler.shutdown)
                    took, left_when_closed = self.close_the_viewer_during_the_save(
                        directory, filename, job_obeys=job_obeys, wait=0.4)
                    self.assertLess(took, 3.0, "closing returns within the bound")
                    # Before the job that ignores the stop is let go: the scheduler deleted the partial file itself.
                    self.assertEqual(left_when_closed, [])
                    self.assertEqual(os.listdir(directory), [])

    def test_closing_the_viewer_during_a_logo_leaves_an_earlier_finished_one_as_it_was(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_job(directory, "first.svg")
            self.assertEqual(self.reports, ["succeeded"])
            with open(os.path.join(directory, "first.svg"), "rb") as handle:
                before = handle.read()
            self.close_the_viewer_during_the_save(directory, "second.svg", job_obeys=False, wait=0.3)
            self.assertEqual(os.listdir(directory), ["first.svg"])
            with open(os.path.join(directory, "first.svg"), "rb") as handle:
                self.assertEqual(handle.read(), before)

    def test_a_stop_leaves_an_existing_logo_of_that_name_as_it_was(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = os.path.join(directory, "again.svg")
            with open(existing, "wb") as handle:
                handle.write(b"previous logo")
            patch, calls = self.stop_when_called(logo_command, "calculate_logo_matrix")
            with patch:
                self.run_job(directory, "again.svg", allow_overwrite=True)
            self.assertEqual(self.reports, ["stopped"])
            self.assertEqual(os.listdir(directory), ["again.svg"])
            with open(existing, "rb") as handle:
                self.assertEqual(handle.read(), b"previous logo")

    def test_the_output_is_free_again_and_the_same_logo_can_be_made_after_a_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            patch, _calls = self.stop_when_called(logo_command, "calculate_logo_matrix")
            with patch:
                self.run_job(directory, "retry.svg")
            self.assertFalse(self.scheduler.is_output_path_reserved(os.path.join(directory, "retry.svg")))
            self.reports.clear()
            self.run_job(directory, "retry.svg")
            self.assertEqual(self.reports, ["succeeded"])
            self.assertEqual(os.listdir(directory), ["retry.svg"])


if __name__ == "__main__":
    unittest.main()
