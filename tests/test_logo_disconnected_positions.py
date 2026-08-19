import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

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

    def test_negative_offset_insertion_labels_are_supported(self):
        positions = parse_logo_positions("[-1.1,-1,0]")

        self.assertEqual(positions, [-1, "-1.1", 0])

    def test_fractional_ranges_require_explicit_insertion_labels(self):
        with self.assertRaisesRegex(ValueError, "list insertion positions explicitly"):
            parse_logo_positions("[10.1-11.2]")


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
            self.assertIn("effective N 2.00", result["message"])

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
            alignment_rows = MultipleSeqAlignment(
                [
                    SeqRecord(Seq("AAAA"), id="node0"),
                    SeqRecord(Seq("CCCC"), id="node1"),
                ]
            )
            alignment = SimpleNamespace(
                aln=alignment_rows,
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

            with mock.patch.object(logo_command.cfg, "LOGO_DIR", directory), \
                    mock.patch.object(
                        logo_command.cfg,
                        "HEADER_LIST_DIR",
                        directory,
                    ):
                logo_command.run(viewer, ["[1]", "snapshot.svg"])

            viewer.selected_indices[:] = [1]
            alignment_rows[0].seq = Seq("GGGG")
            alignment.label_to_col["1"] = 3

            payload = scheduler.job["payload"]
            self.assertEqual(payload["selected_seqs"], ("AAAA",))
            self.assertEqual(payload["valid_cols"], (0,))
            self.assertEqual(payload["plot_positions"], (1,))

    def test_run_submits_explicit_fractional_position_to_logo_job(self):
        with tempfile.TemporaryDirectory() as directory:
            alignment_rows = MultipleSeqAlignment(
                [
                    SeqRecord(Seq("A-AA"), id="node0"),
                    SeqRecord(Seq("ACAA"), id="node1"),
                ]
            )
            alignment = SimpleNamespace(
                aln=alignment_rows,
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

            with mock.patch.object(logo_command.cfg, "LOGO_DIR", directory), \
                    mock.patch.object(
                        logo_command.cfg,
                        "HEADER_LIST_DIR",
                        directory,
                    ):
                logo_command.run(viewer, ["[1,1.1,2]", "insertions.svg"])

            payload = scheduler.job["payload"]
            self.assertEqual(payload["valid_cols"], (0, 1, 2))
            self.assertEqual(payload["plot_positions"], (1, "1.1", 2))

    def test_run_scopes_overwrite_to_explicit_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            existing_file = os.path.join(directory, "custom_logo.svg")
            with open(existing_file, "wb") as f:
                f.write(b"previous_content")

            alignment_rows = MultipleSeqAlignment(
                [
                    SeqRecord(Seq("AAAA"), id="node0"),
                ]
            )
            alignment = SimpleNamespace(
                aln=alignment_rows,
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

            with mock.patch.object(logo_command.cfg, "LOGO_DIR", directory), \
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


if __name__ == "__main__":
    unittest.main()
