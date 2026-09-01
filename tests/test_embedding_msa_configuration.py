import io
import os
import pathlib
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILITIES_DIR = PROJECT_ROOT / "src" / "utilities"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import Embedding_MSA


class EmbeddingMsaConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.fasta_dir = os.path.join("project", "Input_Files", "Sequence_Sets")
        self.embed_dir = os.path.join("project", "Embeddings")
        self.network_dir = os.path.join(
            "project", "Input_Files", "Networks_EValues"
        )
        self.msa_dir = os.path.join("project", "Input_Files", "Multiple_Alignments")
        self.input_embed = "uniprotkb_IPR011343_90_[esmc_6b]_embeddings.h5"
        self.input_network = "uniprotkb_IPR011343_90_[esmc_6b]_network.h5"

    def resolve(self, *, input_fasta="", input_embed=None, input_network=None, use_filter=False):
        return Embedding_MSA.resolve_msa_configuration(
            self.fasta_dir,
            self.embed_dir,
            self.network_dir,
            input_fasta,
            self.input_embed if input_embed is None else input_embed,
            self.input_network if input_network is None else input_network,
            use_filter,
        )

    def test_empty_fasta_is_not_joined_when_filter_is_disabled(self):
        original_join = os.path.join
        with mock.patch.object(
            Embedding_MSA.os.path, "join", wraps=original_join
        ) as join_mock:
            resolved = self.resolve()

        self.assertEqual(resolved["full_input_fasta"], "")
        self.assertNotIn(
            (self.fasta_dir, ""),
            [call.args for call in join_mock.call_args_list],
        )
        self.assertEqual(resolved["sequence_set"], "uniprotkb_IPR011343_90")

    def test_current_gui_settings_produce_expected_output_name(self):
        resolved = self.resolve(input_fasta="", use_filter=False)
        output_path = Embedding_MSA.build_msa_output_path(
            self.msa_dir,
            resolved["sequence_set"],
            "esmc_6b",
        )

        self.assertEqual(
            os.path.basename(output_path),
            "uniprotkb_IPR011343_90_[esmc_6b]_alignment.fasta",
        )

    def test_enabled_filter_requires_a_selected_fasta(self):
        with self.assertRaisesRegex(
            Embedding_MSA.MSAConfigurationError,
            "INPUT_FASTA must select a FASTA file",
        ):
            self.resolve(input_fasta="", use_filter=True)

    def test_enabled_filter_uses_selected_fasta_name_and_path(self):
        resolved = self.resolve(
            input_fasta="Selected_subset.fasta",
            use_filter=True,
        )

        self.assertEqual(
            resolved["full_input_fasta"],
            os.path.join(self.fasta_dir, "Selected_subset.fasta"),
        )
        self.assertEqual(resolved["sequence_set"], "Selected_subset")

    def test_required_hdf5_settings_have_field_specific_errors(self):
        cases = (
            ("", self.input_network, "INPUT_EMBED"),
            (self.input_embed, "", "INPUT_NETWORK"),
        )
        for input_embed, input_network, expected_field in cases:
            with self.subTest(expected_field=expected_field):
                with self.assertRaisesRegex(
                    Embedding_MSA.MSAConfigurationError,
                    expected_field,
                ):
                    self.resolve(
                        input_embed=input_embed,
                        input_network=input_network,
                    )

    def test_noncanonical_embedding_name_uses_stem_fallback(self):
        resolved = self.resolve(input_embed="custom_embeddings.h5")
        self.assertEqual(resolved["sequence_set"], "custom")

    def test_processing_time_summary_is_human_readable(self):
        output = io.StringIO()
        with redirect_stdout(output):
            Embedding_MSA.report_processing_times(
                total_processing_seconds=3661.5,
                tree_building_seconds=65.25,
                cluster_merging_seconds=0.5,
            )

        self.assertEqual(
            output.getvalue().strip().splitlines(),
            [
                "--- Processing Time Summary ---",
                "Total processing time: 1h 1m 1.50s",
                "Tree building time: 1m 5.25s",
                "Cluster merging time: 0.50s",
            ],
        )

    def test_bootstrap_seeds_are_reproducible_and_use_fixed_seed(self):
        first = Embedding_MSA.generate_bootstrap_seeds(5)
        second = Embedding_MSA.generate_bootstrap_seeds(5)

        self.assertEqual(first.tolist(), second.tolist())
        with mock.patch.object(Embedding_MSA, "RANDOM_SEED", 43):
            changed = Embedding_MSA.generate_bootstrap_seeds(5)
        self.assertNotEqual(first.tolist(), changed.tolist())

    def test_representative_leaf_merges_use_cost_percentiles(self):
        headers = [f"h{index}" for index in range(10)]
        lengths = [10, 10, 20, 20, 30, 30, 40, 40, 50, 50]
        embeddings = {
            header: SimpleNamespace(shape=(length, 8))
            for header, length in zip(headers, lengths)
        }
        linkage = np.asarray(
            [
                [0, 1, 0.0, 2],
                [2, 3, 0.0, 2],
                [4, 5, 0.0, 2],
                [6, 7, 0.0, 2],
                [8, 9, 0.0, 2],
                [10, 11, 0.0, 4],
            ],
            dtype=np.float64,
        )

        selected = Embedding_MSA.representative_leaf_merge_pairs(
            linkage,
            headers,
            embeddings,
        )

        self.assertEqual(
            selected,
            [
                (2, 3, 20, 20),
                (4, 5, 30, 30),
                (8, 9, 50, 50),
            ],
        )

    def test_representative_leaf_merges_use_every_pair_when_fewer_than_three(self):
        headers = ["a", "b", "c", "d"]
        embeddings = {
            "a": SimpleNamespace(shape=(10, 8)),
            "b": SimpleNamespace(shape=(20, 8)),
            "c": SimpleNamespace(shape=(30, 8)),
            "d": SimpleNamespace(shape=(40, 8)),
        }
        linkage = np.asarray(
            [[0, 1, 0.0, 2], [2, 3, 0.0, 2]],
            dtype=np.float64,
        )

        selected = Embedding_MSA.representative_leaf_merge_pairs(
            linkage,
            headers,
            embeddings,
        )

        self.assertEqual(selected, [(0, 1, 10, 20), (2, 3, 30, 40)])

    def test_representative_leaf_merges_allow_a_single_sequence(self):
        selected = Embedding_MSA.representative_leaf_merge_pairs(
            np.zeros((0, 4), dtype=np.float64),
            ["only"],
            {"only": SimpleNamespace(shape=(10, 8))},
        )
        self.assertEqual(selected, [])

    def test_manual_device_skips_msa_benchmark(self):
        cpu = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cpu",
            "CPU",
            torch.device("cpu"),
            "cpu",
        )
        with mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu],
        ), mock.patch.object(
            Embedding_MSA,
            "representative_leaf_merge_pairs",
            side_effect=AssertionError("manual selection sampled benchmark pairs"),
        ):
            ranked = Embedding_MSA.benchmark_msa_devices(
                np.zeros((1, 4)),
                ["a", "b"],
                {},
                "cpu",
            )

        self.assertEqual([result.candidate.spec for result in ranked], ["cpu"])

    def test_unavailable_manual_device_is_rejected_before_sampling(self):
        cpu = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cpu",
            "CPU",
            torch.device("cpu"),
            "cpu",
        )
        with mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu],
        ), mock.patch.object(
            Embedding_MSA,
            "representative_leaf_merge_pairs",
            side_effect=AssertionError("unavailable device sampled benchmark pairs"),
        ), self.assertRaisesRegex(ValueError, "not available"):
            Embedding_MSA.benchmark_msa_devices(
                np.zeros((1, 4)),
                ["a", "b"],
                {},
                "cuda:99",
            )

    def test_auto_single_device_skips_msa_benchmark(self):
        cpu = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cpu",
            "CPU",
            torch.device("cpu"),
            "cpu",
        )
        with mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu],
        ), mock.patch.object(
            Embedding_MSA,
            "representative_leaf_merge_pairs",
            side_effect=AssertionError("single device sampled benchmark pairs"),
        ):
            ranked = Embedding_MSA.benchmark_msa_devices(
                np.zeros((1, 4)),
                ["a", "b"],
                {},
                "auto",
            )

        self.assertEqual([result.candidate.spec for result in ranked], ["cpu"])

    def test_auto_benchmark_uses_warmup_and_median_of_three_runs(self):
        cpu = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cpu",
            "CPU",
            torch.device("cpu"),
            "cpu",
        )
        cuda = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cuda:0",
            "Test CUDA",
            torch.device("cuda:0"),
            "cuda",
            0,
            True,
        )
        sample = (
            0,
            1,
            2,
            3,
            np.ones((2, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
        )
        perf_values = [
            0.0,
            1.0,
            10.0,
            13.0,
            20.0,
            22.0,
            30.0,
            30.5,
            40.0,
            40.6,
            50.0,
            50.4,
        ]

        with mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu, cuda],
        ), mock.patch.object(
            Embedding_MSA,
            "representative_leaf_merge_pairs",
            return_value=[(0, 1, 2, 3)],
        ), mock.patch.object(
            Embedding_MSA,
            "_load_benchmark_embedding_pairs",
            return_value=[sample],
        ), mock.patch.object(
            Embedding_MSA,
            "compute_score_matrix_torch",
            return_value=np.zeros((2, 3), dtype=np.float32),
        ) as score, mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "synchronize_device",
        ) as synchronize, mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "release_device_cache",
        ), mock.patch.object(
            Embedding_MSA.time,
            "perf_counter",
            side_effect=perf_values,
        ):
            ranked = Embedding_MSA.benchmark_msa_devices(
                np.zeros((1, 4)),
                ["a", "b"],
                {},
                "auto",
            )

        self.assertEqual(ranked[0].candidate.spec, "cuda:0")
        self.assertEqual(ranked[0].value, 0.5)
        self.assertEqual(ranked[1].value, 2.0)
        self.assertEqual(score.call_count, 8)
        self.assertEqual(synchronize.call_count, 14)

    def test_auto_benchmark_reports_when_every_device_fails(self):
        cpu = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cpu",
            "CPU",
            torch.device("cpu"),
            "cpu",
        )
        cuda = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cuda:0",
            "Test CUDA",
            torch.device("cuda:0"),
            "cuda",
            0,
            True,
        )
        sample = (
            0,
            1,
            2,
            3,
            np.ones((2, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
        )

        with mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu, cuda],
        ), mock.patch.object(
            Embedding_MSA,
            "representative_leaf_merge_pairs",
            return_value=[(0, 1, 2, 3)],
        ), mock.patch.object(
            Embedding_MSA,
            "_load_benchmark_embedding_pairs",
            return_value=[sample],
        ), mock.patch.object(
            Embedding_MSA,
            "compute_score_matrix_torch",
            side_effect=RuntimeError("simulated backend failure"),
        ), mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "release_device_cache",
        ), self.assertRaisesRegex(
            RuntimeError,
            "No device completed the MSA benchmark",
        ):
            Embedding_MSA.benchmark_msa_devices(
                np.zeros((1, 4)),
                ["a", "b"],
                {},
                "auto",
            )

    def test_auto_score_failure_retries_and_promotes_fallback(self):
        cpu = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cpu",
            "CPU",
            torch.device("cpu"),
            "cpu",
        )
        cuda = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cuda:0",
            "Test CUDA",
            torch.device("cuda:0"),
            "cuda",
            0,
            True,
        )
        ranked = [
            Embedding_MSA.Hardware_Utils.BenchmarkResult(cuda, 1.0),
            Embedding_MSA.Hardware_Utils.BenchmarkResult(cpu, 2.0),
        ]
        expected = np.zeros((2, 3), dtype=np.float32)

        with mock.patch.object(
            Embedding_MSA,
            "compute_score_matrix_torch",
            side_effect=[RuntimeError("simulated OOM"), expected],
        ) as score, mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "release_device_cache",
        ):
            actual = Embedding_MSA.compute_score_matrix_with_fallback(
                np.ones((2, 4), dtype=np.float32),
                np.ones((3, 4), dtype=np.float32),
                ranked,
                "auto",
            )

        self.assertIs(actual, expected)
        self.assertEqual(score.call_count, 2)
        self.assertEqual([result.candidate.spec for result in ranked], ["cpu"])

    def test_manual_score_failure_does_not_fall_back(self):
        cuda = Embedding_MSA.Hardware_Utils.DeviceCandidate(
            "cuda:0",
            "Test CUDA",
            torch.device("cuda:0"),
            "cuda",
            0,
            True,
        )
        ranked = [Embedding_MSA.Hardware_Utils.BenchmarkResult(cuda, 0.0)]

        with mock.patch.object(
            Embedding_MSA,
            "compute_score_matrix_torch",
            side_effect=RuntimeError("simulated failure"),
        ), mock.patch.object(
            Embedding_MSA.Hardware_Utils,
            "release_device_cache",
        ), self.assertRaisesRegex(RuntimeError, "manually selected device"):
            Embedding_MSA.compute_score_matrix_with_fallback(
                np.ones((2, 4), dtype=np.float32),
                np.ones((3, 4), dtype=np.float32),
                ranked,
                "cuda:0",
            )

    def test_msa_gui_exposes_persisted_device_selection(self):
        tools_source = (PROJECT_ROOT / "src" / "emapssn_tools.py").read_text(
            encoding="utf-8"
        )
        panel = tools_source.split('"Embedding_MSA": {', 1)[1].split(
            '"Sparse_MSA_Converter.py": [', 1
        )[0]
        self.assertIn('"var_name": "DEVICE_SELECTION"', panel)
        self.assertIn('"type": "device_dropdown"', panel)
        self.assertIn('("WORKERS", "DEVICE_SELECTION")', tools_source)
        self.assertIn('DEVICE_SELECTION = "auto"', pathlib.Path(
            Embedding_MSA.__file__
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
