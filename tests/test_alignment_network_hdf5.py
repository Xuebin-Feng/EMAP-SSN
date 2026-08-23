import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from utilities.Alignment_Network_HDF5 import (  # noqa: E402
    AlignmentNetworkIdentity,
    CanonicalAlignmentNetworkReader,
    ResumableAlignmentNetworkWriter,
    SparsityProfile,
    build_sparsity_profile,
    discover_compatible_alignment_networks,
    exact_top_k_mask,
)
from tools import Align_Similarity_Matrix as similarity_matrix  # noqa: E402
from Cache_Manifest import validate_network_schema  # noqa: E402


def identity(sequence_count=6):
    return AlignmentNetworkIdentity(
        headers=tuple(f"seq-{index}" for index in range(sequence_count)),
        sequence_lengths=tuple(10 + index for index in range(sequence_count)),
        embedding_checksum="embedding-checksum",
        model_name="test-model",
        gap_penalties=(-2.0, 0.0),
        saving_mode="float32",
    )


def records_for_mask(mask):
    records = []
    for left in range(mask.shape[0]):
        for right in np.flatnonzero(mask[left, left + 1:]) + left + 1:
            score = np.float32(left * 10 + right)
            records.append(
                (
                    left,
                    int(right),
                    score,
                    np.uint16(left + right + 1),
                    np.float32(score + 0.5),
                    np.uint16(left + right + 2),
                )
            )
    return records


class AlignmentNetworkHDF5Tests(unittest.TestCase):
    def setUp(self):
        self.scores = np.ones((6, 6), dtype=np.float32)
        np.fill_diagonal(self.scores, 2.0)
        self.identity = identity()

    def profile(self, sparsity):
        return build_sparsity_profile(
            self.scores,
            embedding_checksum=self.identity.embedding_checksum,
            enabled=True,
            sparsity_percent=sparsity,
            pooling_method="max",
            length_ratio_power=2.0,
        )

    def test_sparse_runtime_path_uses_density_suffix(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            similarity_matrix, "INPUT_HDF5", os.path.join(folder, "set_[model]_embeddings.h5")
        ), mock.patch.object(
            similarity_matrix, "EMBED_DIR", folder
        ), mock.patch.object(
            similarity_matrix, "NETWORK_DIR", folder
        ), mock.patch.object(
            similarity_matrix, "EDGE_PREFILTERING", True
        ), mock.patch.object(
            similarity_matrix, "PREFILTER_STRENGTH", 62.5
        ):
            similarity_matrix.configure_runtime_paths()
            self.assertTrue(
                similarity_matrix.FINAL_OUTPUT_NET.endswith(
                    "set_[model]_network_sparse62p5pct.h5"
                )
            )

    def test_exact_sparse_masks_are_nested_with_deterministic_ties(self):
        mask_80, keep_80, _cutoff_80 = exact_top_k_mask(
            self.scores, 80, enabled=True
        )
        mask_60, keep_60, _cutoff_60 = exact_top_k_mask(
            self.scores, 60, enabled=True
        )
        self.assertEqual(keep_80, 3)
        self.assertEqual(keep_60, 6)
        self.assertTrue(np.all(mask_80 <= mask_60))
        self.assertEqual(keep_60 - keep_80, 3)
        self.assertEqual(
            list(zip(*np.nonzero(mask_80))),
            [(0, 1), (0, 2), (0, 3)],
        )

    def test_writer_resumes_and_final_file_has_no_indicator_or_journal(self):
        mask, profile = self.profile(60)
        records = records_for_mask(mask)
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "network_sparse60pct.h5")
            writer = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
            )
            writer.commit(records[:2], next_pair=(0, 3))
            writer.close()

            writer = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
            )
            self.assertEqual(writer.committed_count, 2)
            writer.commit(records[2:], next_pair=(6, 6))
            writer.finalize()

            with h5py.File(target, "r") as network:
                self.assertNotIn("_resume", network)
                self.assertNotIn("format_version", network.attrs)
                self.assertNotIn("source_type", network.attrs)
                self.assertNotIn("reuse_origin", network.attrs)
                self.assertEqual(len(network["i"]), profile.keep_count)
                np.testing.assert_array_equal(
                    network["l_score"][:],
                    np.asarray([record[2] for record in records], np.float32),
                )

    def test_corrupted_latest_chunk_falls_back_to_previous_checkpoint(self):
        mask, profile = self.profile(60)
        records = records_for_mask(mask)
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "network_sparse60pct.h5")
            writer = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
            )
            writer.commit(records[:2], next_pair=(0, 3))
            writer.commit(records[2:4], next_pair=(0, 5))
            writer.close()
            with h5py.File(target + ".partial", "r+") as partial:
                partial["l_score"][3] += np.float32(1000.0)
                partial.flush()

            writer = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
            )
            self.assertEqual(writer.committed_count, 2)
            writer.commit(records[2:], next_pair=(6, 6))
            writer.finalize()

    def test_interruption_uses_only_a_flushed_checkpoint(self):
        mask, profile = self.profile(60)
        records = records_for_mask(mask)
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "network_sparse60pct.h5")

            def fail_after_data(stage):
                if stage == "after_data_flush":
                    raise RuntimeError("interrupted")

            writer = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
                failure_hook=fail_after_data,
            )
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                writer.commit(records[:2], next_pair=(0, 3))
            writer.close()
            resumed = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
            )
            self.assertEqual(resumed.committed_count, 0)
            resumed.close()

            os.remove(target + ".partial")

            def fail_after_checkpoint(stage):
                if stage == "after_checkpoint_flush":
                    raise RuntimeError("interrupted")

            writer = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
                failure_hook=fail_after_checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                writer.commit(records[:2], next_pair=(0, 3))
            writer.close()
            resumed = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
            )
            self.assertEqual(resumed.committed_count, 2)
            resumed.close()

    def test_streaming_reader_retains_unused_rows_between_calls(self):
        mask, profile = self.profile(60)
        records = records_for_mask(mask)
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "source.h5")
            writer = ResumableAlignmentNetworkWriter(
                target,
                identity=self.identity,
                sparsity_profile=profile,
                matmul_precision="ieee_fp32",
                chunk_edges=2,
            )
            writer.commit(records, next_pair=(6, 6))
            writer.finalize()
            with CanonicalAlignmentNetworkReader(
                target, self.identity.sequence_count, chunk_edges=2
            ) as reader:
                first = reader.lookup_many([(0, 1)])
                second = reader.lookup_many([(0, 2)])
            self.assertEqual(first[0][2], records[0][2])
            self.assertEqual(second[0][2], records[1][2])

    def test_discovery_accepts_indicator_free_network_and_ranks_coverage(self):
        target_mask, target_profile = self.profile(60)
        source_mask, _source_profile = self.profile(80)
        source_records = records_for_mask(source_mask)
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name, count in (("injected.h5", 3), ("reused.h5", 4)):
                path = os.path.join(folder, name)
                paths.append(path)
                rows = records_for_mask(target_mask)[:count]
                with h5py.File(path, "w") as network:
                    network.attrs["embedding_checksum"] = self.identity.embedding_checksum
                    network.attrs["model_name"] = self.identity.model_name
                    network.attrs["gap_penalties"] = np.asarray(
                        self.identity.gap_penalties, np.float32
                    )
                    network.attrs["matmul_precision"] = "ieee_fp32"
                    text = h5py.string_dtype("utf-8")
                    network.create_dataset(
                        "headers",
                        data=np.asarray(self.identity.headers, dtype=object),
                        dtype=text,
                    )
                    network.create_dataset(
                        "seq_lens",
                        data=np.asarray(self.identity.sequence_lengths, np.uint16),
                    )
                    for column, dataset_name in enumerate(
                        ("i", "j", "l_score", "l_len", "g_score", "g_len")
                    ):
                        dtype = {
                            "i": np.uint16,
                            "j": np.uint16,
                            "l_score": np.float32,
                            "l_len": np.uint16,
                            "g_score": np.float32,
                            "g_len": np.uint16,
                        }[dataset_name]
                        network.create_dataset(
                            dataset_name,
                            data=np.asarray([row[column] for row in rows], dtype=dtype),
                        )
                with h5py.File(path, "r") as network:
                    self.assertNotIn("format_version", network.attrs)
                    self.assertNotIn("source_type", network.attrs)

            candidates = discover_compatible_alignment_networks(
                folder,
                identity=self.identity,
                target_mask=target_mask,
                required_precision="ieee_fp32",
            )
            self.assertEqual(candidates[0].path, os.path.abspath(paths[1]))
            self.assertEqual(candidates[0].reusable_edge_count, 4)
            self.assertEqual(
                target_profile.keep_count - len(source_records),
                target_profile.keep_count - 3,
            )

    def test_align_expansion_calculates_only_the_exact_sparse_delta(self):
        mask_80, profile_80 = self.profile(80)
        mask_60, profile_60 = self.profile(60)
        manifest = SimpleNamespace(model_name="test-model", saving_mode="float32")
        plan = SimpleNamespace(
            candidate=SimpleNamespace(
                is_cpu=True,
                spec="cpu",
                device="cpu",
                display_name="Test CPU",
            ),
            variant="scalar",
            lanes=1,
            execution_plan=None,
        )
        calculated_by_run = []

        def calculate(tasks, plans, plan_index, **kwargs):
            pairs = [(int(task[0]), int(task[1])) for task in tasks]
            calculated_by_run[-1].extend(pairs)
            return (
                [
                    (
                        left,
                        right,
                        np.float32(left * 10 + right),
                        np.uint16(left + right + 1),
                        np.float32(left * 10 + right + 0.5),
                        np.uint16(left + right + 2),
                    )
                    for left, right in pairs
                ],
                plan_index,
            )

        with tempfile.TemporaryDirectory() as folder:
            input_path = os.path.join(folder, "embeddings.h5")
            with h5py.File(input_path, "w"):
                pass
            target_80 = os.path.join(folder, "set_[test-model]_network_sparse80pct.h5")
            target_60 = os.path.join(folder, "set_[test-model]_network_sparse60pct.h5")

            def run_target(target, mask, profile):
                calculated_by_run.append([])
                with mock.patch.object(
                    similarity_matrix, "FULL_INPUT_HDF5", input_path
                ), mock.patch.object(
                    similarity_matrix, "NETWORK_DIR", folder
                ), mock.patch.object(
                    similarity_matrix, "FINAL_OUTPUT_NET", target
                ), mock.patch.object(
                    similarity_matrix, "configure_runtime_paths"
                ), mock.patch.object(
                    similarity_matrix,
                    "load_embedding_metadata",
                    return_value=(
                        list(self.identity.headers),
                        list(self.identity.headers),
                        list(self.identity.sequence_lengths),
                        manifest,
                    ),
                ), mock.patch.object(
                    similarity_matrix,
                    "calculate_file_hash",
                    return_value=self.identity.embedding_checksum,
                ), mock.patch.object(
                    similarity_matrix,
                    "_compute_target_selection",
                    return_value=(mask, profile),
                ), mock.patch.object(
                    similarity_matrix,
                    "EmbeddingTileStore",
                    return_value=SimpleNamespace(
                        fully_cached=False, cached_bytes=0
                    ),
                ), mock.patch.object(
                    similarity_matrix,
                    "_resolve_active_matmul_precision",
                    return_value="ieee_fp32",
                ), mock.patch.object(
                    similarity_matrix,
                    "_benchmark_processing_plans",
                    return_value=[plan],
                ), mock.patch.object(
                    similarity_matrix,
                    "_calculate_result_chunk_with_ranked_plans",
                    side_effect=calculate,
                ), mock.patch.object(
                    similarity_matrix, "set_start_method"
                ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    similarity_matrix.run_job_distributor()

            run_target(target_80, mask_80, profile_80)
            run_target(target_60, mask_60, profile_60)

            self.assertEqual(len(calculated_by_run[0]), profile_80.keep_count)
            self.assertEqual(
                len(calculated_by_run[1]),
                profile_60.keep_count - profile_80.keep_count,
            )
            with h5py.File(target_80, "r") as source, h5py.File(
                target_60, "r"
            ) as expanded:
                self.assertEqual(len(expanded["i"]), profile_60.keep_count)
                np.testing.assert_array_equal(
                    expanded["l_score"][: profile_80.keep_count],
                    source["l_score"][:],
                )
                source_pairs = list(zip(source["i"][:], source["j"][:]))
                expanded_pairs = list(zip(expanded["i"][:], expanded["j"][:]))
                self.assertEqual(expanded_pairs, sorted(expanded_pairs))
                self.assertEqual(expanded_pairs[: len(source_pairs)], source_pairs)
                metadata = validate_network_schema(
                    expanded, expected_network_type="alignment"
                )
                self.assertEqual(metadata.model_name, self.identity.model_name)
            self.assertEqual(
                [name for name in os.listdir(folder) if name.startswith("batch_")],
                [],
            )


if __name__ == "__main__":
    unittest.main()
