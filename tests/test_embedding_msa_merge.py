"""Embedding_MSA merges column matrices exactly as it once merged strings.

The reference functions restate the merge stage before the change (commit
96dbeb6): aligned sequences grown one character at a time and the profile
built one column at a time. The array merge must reproduce them bit for bit,
down to the sign of zero in float16 profiles.
"""

import gc
import io
import pathlib
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILITIES_DIR = PROJECT_ROOT / "src" / "utilities"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import Embedding_MSA
from tests.msa_fixtures import (
    EMBED_NAME,
    InProcessPool,
    NetworkFixture,
    builder_settings,
)


ALIGNMENT_CODES = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY-", dtype=np.uint8)


class StringCluster:
    """A cluster as the reference merge sees it: one string per sequence."""

    def __init__(self, sequences, ids, embedding=None):
        self.sequences = sequences
        self.ids = ids
        self.embedding = embedding


def reference_merge(cluster_a, cluster_b, path, emb_a, emb_b):
    """merge_clusters at 96dbeb6, verbatim apart from the cluster it returns."""
    path = path[::-1]
    new_seqs_a = ["" for _ in cluster_a.sequences]
    new_seqs_b = ["" for _ in cluster_b.sequences]
    merged_vecs = []

    idx_a, idx_b = 0, 0

    w_a = float(len(cluster_a.ids))
    w_b = float(len(cluster_b.ids))
    total_w = w_a + w_b

    for move in path:
        if move == 1:
            for i, s in enumerate(cluster_a.sequences): new_seqs_a[i] += s[idx_a]
            for i, s in enumerate(cluster_b.sequences): new_seqs_b[i] += s[idx_b]
            vec = (emb_a[idx_a].astype(np.float32) * w_a + emb_b[idx_b].astype(np.float32) * w_b) / total_w
            merged_vecs.append(vec)
            idx_a += 1; idx_b += 1

        elif move == 2:
            for i, s in enumerate(cluster_a.sequences): new_seqs_a[i] += s[idx_a]
            for i, s in enumerate(cluster_b.sequences): new_seqs_b[i] += "-"
            vec = (emb_a[idx_a].astype(np.float32) * w_a) / total_w
            merged_vecs.append(vec)
            idx_a += 1

        elif move == 3:
            for i, s in enumerate(cluster_a.sequences): new_seqs_a[i] += "-"
            for i, s in enumerate(cluster_b.sequences): new_seqs_b[i] += s[idx_b]
            vec = (emb_b[idx_b].astype(np.float32) * w_b) / total_w
            merged_vecs.append(vec)
            idx_b += 1

    return StringCluster(
        new_seqs_a + new_seqs_b,
        cluster_a.ids + cluster_b.ids,
        np.stack(merged_vecs, axis=0).astype(np.float16),
    )


def reference_alignment_fasta(fixture, linkage_matrix, valid_headers, gap_open, gap_extend):
    """Return the FASTA bytes the 96dbeb6 merge stage writes for a guide tree.

    Every leaf's sequence length equals its embedding's row count here, as
    the manifest check guarantees, so the old trim/pad step never applies.
    """
    sequence_by_header = dict(zip(fixture.emb_headers, fixture.emb_sequences))
    num_seqs = len(valid_headers)
    clusters = {
        index: StringCluster([sequence_by_header[header]], [index])
        for index, header in enumerate(valid_headers)
    }
    with h5py.File(fixture.embed_dir / EMBED_NAME, "r") as embeddings:
        group = embeddings["embeddings"]

        def profile(cluster):
            if cluster.embedding is not None:
                return cluster.embedding
            header = valid_headers[cluster.ids[0]]
            safe = header.replace("/", "_").replace("\\", "_")
            return Embedding_MSA._normalize_residue_embeddings(group[safe][:])

        for iteration, link in enumerate(linkage_matrix):
            cluster_a = clusters.pop(int(link[0]))
            cluster_b = clusters.pop(int(link[1]))
            emb_a = profile(cluster_a)
            emb_b = profile(cluster_b)
            score = Embedding_MSA.compute_score_matrix_torch(emb_a, emb_b, "cpu")
            path = Embedding_MSA.run_global_traceback(score, gap_open, gap_extend)
            clusters[num_seqs + iteration] = reference_merge(
                cluster_a, cluster_b, path, emb_a, emb_b
            )
    final = clusters[num_seqs + len(linkage_matrix) - 1]
    return "".join(
        f">{valid_headers[index]}\n{sequence}\n"
        for index, sequence in zip(final.ids, final.sequences)
    ).encode("utf-8")


def random_rows(rng, count, width):
    """Aligned rows of residues and gaps."""
    codes = rng.choice(ALIGNMENT_CODES, size=(count, width))
    return [row.tobytes().decode("ascii") for row in codes]


def array_cluster(rows, ids, embedding=None):
    """The MSACluster holding these aligned rows, one column per matrix row."""
    by_sequence = np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8)
    aligned = np.ascontiguousarray(by_sequence.reshape(len(rows), -1).T)
    return Embedding_MSA.MSACluster(-1, aligned, ids, embedding)


def rows_of(cluster):
    return [row.tobytes().decode("ascii") for row in np.ascontiguousarray(cluster.aligned.T)]


def random_path(rng, width_a, width_b, matches=None):
    """A complete traceback path, last column first, as the kernel returns it."""
    if matches is None:
        matches = int(rng.integers(0, min(width_a, width_b) + 1))
    moves = np.asarray(
        [1] * matches + [2] * (width_a - matches) + [3] * (width_b - matches),
        dtype=np.int8,
    )
    rng.shuffle(moves)
    return moves[::-1].copy()


def random_profile(rng, width, sequences, dims=16):
    """Leaves carry float32 unit vectors, merged clusters float16 averages.

    About one value in twenty is set to -0.0 or +0.0. Real float16 profiles
    hold -0.0 wherever a small negative average underflows, and the sign of
    zero is what a +0.0-initialized sum gets wrong.
    """
    profile = rng.standard_normal((width, dims)).astype(np.float32)
    profile /= np.linalg.norm(profile, axis=1, keepdims=True)
    if sequences > 1:
        profile = profile.astype(np.float16)
    zeros = rng.random(profile.shape) < 0.05
    profile[zeros] = np.where(rng.random(int(zeros.sum())) < 0.5, -0.0, 0.0)
    return profile


class MergeEquivalenceTests(unittest.TestCase):
    def assertSameMerge(self, actual, expected):
        self.assertEqual(rows_of(actual), expected.sequences)
        self.assertEqual(actual.ids, expected.ids)
        self.assertEqual(actual.embedding.dtype, np.float16)
        self.assertEqual(actual.embedding.shape, expected.embedding.shape)
        self.assertEqual(
            actual.embedding.view(np.uint16).tobytes(),
            expected.embedding.view(np.uint16).tobytes(),
        )
        self.assertEqual(actual.aligned.dtype, np.uint8)
        self.assertFalse(actual.is_leaf)

    def merge_both(self, rows_a, rows_b, ids_a, ids_b, path, emb_a, emb_b):
        actual = Embedding_MSA.merge_clusters(
            array_cluster(rows_a, ids_a), array_cluster(rows_b, ids_b), path, emb_a, emb_b
        )
        expected = reference_merge(
            StringCluster(rows_a, ids_a), StringCluster(rows_b, ids_b), path, emb_a, emb_b
        )
        return actual, expected

    def test_random_merges_match_reference_bit_for_bit(self):
        rng = np.random.default_rng(11)
        negative_zeros = 0
        for trial in range(300):
            count_a, count_b = (int(value) for value in rng.integers(1, 9, size=2))
            width_a, width_b = (int(value) for value in rng.integers(1, 80, size=2))
            ids_a = list(range(count_a))
            ids_b = list(range(100, 100 + count_b))
            emb_a = random_profile(rng, width_a, count_a)
            emb_b = random_profile(rng, width_b, count_b)
            path = random_path(rng, width_a, width_b)
            with self.subTest(trial=trial):
                actual, expected = self.merge_both(
                    random_rows(rng, count_a, width_a),
                    random_rows(rng, count_b, width_b),
                    ids_a, ids_b, path, emb_a, emb_b,
                )
                self.assertSameMerge(actual, expected)
                negative_zeros += int(np.sum(np.signbit(expected.embedding) & (expected.embedding == 0)))
        # The fixtures must exercise the sign of zero, or the bit check is weak.
        self.assertGreater(negative_zeros, 1000)

    def test_paths_from_the_traceback_kernel_match_reference(self):
        rng = np.random.default_rng(12)
        for gap_open, gap_extend in ((-0.5, 0.0), (-1.0, -0.1), (0.0, 0.0)):
            for trial in range(10):
                count_a, count_b = (int(value) for value in rng.integers(1, 6, size=2))
                width_a, width_b = (int(value) for value in rng.integers(1, 60, size=2))
                emb_a = random_profile(rng, width_a, count_a)
                emb_b = random_profile(rng, width_b, count_b)
                score = Embedding_MSA.compute_score_matrix_torch(emb_a, emb_b, "cpu")
                path = Embedding_MSA.run_global_traceback(score, gap_open, gap_extend)
                with self.subTest(gap_open=gap_open, gap_extend=gap_extend, trial=trial):
                    actual, expected = self.merge_both(
                        random_rows(rng, count_a, width_a),
                        random_rows(rng, count_b, width_b),
                        list(range(count_a)), list(range(50, 50 + count_b)),
                        path, emb_a, emb_b,
                    )
                    self.assertSameMerge(actual, expected)

    def test_negative_zero_in_a_b_only_column_keeps_its_sign(self):
        emb_a = np.ones((1, 4), dtype=np.float32)
        emb_b = np.asarray([[-0.0, 0.0, -0.5, 0.5]], dtype=np.float16)
        # Last column first: A's column, preceded by B's inserted column.
        path = np.asarray([2, 3], dtype=np.int8)
        actual, expected = self.merge_both(["A"], ["C"], [0], [1], path, emb_a, emb_b)
        self.assertSameMerge(actual, expected)
        self.assertEqual(rows_of(actual), ["-A", "C-"])
        self.assertTrue(np.signbit(actual.embedding[0, 0]))
        self.assertFalse(np.signbit(actual.embedding[0, 1]))

    def test_path_that_misplaces_columns_is_rejected(self):
        one_column = np.ones((1, 4), dtype=np.float32)
        two_columns = np.ones((2, 4), dtype=np.float32)
        cases = {
            # Masked assignment would broadcast A's single column into none.
            "skips A's column": (np.asarray([3, 3], np.int8), one_column, two_columns, ["CD"]),
            # ...or copy it into two.
            "repeats A's column": (np.asarray([2, 2, 3, 3], np.int8), one_column, two_columns, ["CD"]),
            # A profile row count that disagrees with the alignment width.
            "profile too long": (np.asarray([1], np.int8), two_columns, one_column, ["C"]),
        }
        for name, (path, emb_a, emb_b, rows_b) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "does not place every column"):
                Embedding_MSA.merge_clusters(
                    array_cluster(["A"], [0]), array_cluster(rows_b, [1]), path, emb_a, emb_b
                )


class LeafAlignmentTests(unittest.TestCase):
    def test_sequences_encode_as_one_sequence_columns(self):
        leaves = Embedding_MSA.encode_leaf_alignments(
            {"first": "ACDXU", "second": "", "third": "W"},
            ["third", "first", "second"],
        )
        self.assertEqual([leaf.shape for leaf in leaves], [(1, 1), (5, 1), (0, 1)])
        self.assertTrue(all(leaf.dtype == np.uint8 for leaf in leaves))
        self.assertEqual(leaves[1][:, 0].tobytes(), b"ACDXU")

    def test_non_ascii_sequences_are_rejected_by_name(self):
        sequences = {f"seq{index}": "ACD" for index in range(9)}
        sequences["seq2"] = "AÇD"
        sequences["seq8"] = "ΑCD"  # Greek capital alpha
        with self.assertRaises(Embedding_MSA.NonAsciiSequenceError) as raised:
            Embedding_MSA.encode_leaf_alignments(sequences, sorted(sequences))
        message = str(raised.exception)
        self.assertIn("2 sequence(s)", message)
        self.assertIn("seq2", message)
        self.assertIn("seq8", message)
        self.assertNotIn("seq1", message)

    def test_fitting_trims_and_gap_pads_like_the_string_code(self):
        for sequence, length in (("ACDEF", 3), ("AC", 5), ("ACD", 3), ("", 2), ("A", 0)):
            with self.subTest(sequence=sequence, length=length):
                leaf = Embedding_MSA.encode_leaf_alignments({"s": sequence}, ["s"])[0]
                fitted = Embedding_MSA.fit_leaf_to_embedding(leaf, length)
                expected = (
                    sequence[:length] if len(sequence) > length
                    else sequence.ljust(length, "-")
                )
                self.assertEqual(fitted.shape, (length, 1))
                self.assertEqual(fitted.dtype, np.uint8)
                self.assertEqual(fitted[:, 0].tobytes().decode("ascii"), expected)


class LeafEmbeddingReadTests(unittest.TestCase):
    def test_leaves_are_read_through_the_open_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "embeddings.h5"
            with h5py.File(path, "w") as embeddings:
                embeddings.create_group("embeddings").create_dataset(
                    "sp_P1_X", data=np.asarray([[3.0, 4.0], [0.0, 0.0]], np.float16)
                )
            with h5py.File(path, "r") as embeddings, mock.patch.object(
                Embedding_MSA, "h5py"
            ) as h5py_module:
                leaf = Embedding_MSA.MSACluster(0, np.zeros((2, 1), np.uint8), [0])
                profile = leaf.get_embedding(embeddings["embeddings"], ["sp/P1\\X"])
            h5py_module.File.assert_not_called()
            gc.collect()
        self.assertEqual(profile.dtype, np.float32)
        np.testing.assert_array_equal(
            profile, np.asarray([[0.6, 0.8], [0.0, 0.0]], dtype=np.float32)
        )


class FullRunTests(unittest.TestCase):
    def test_aligned_fasta_matches_the_string_merge_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NetworkFixture(directory, seed=4)
            values = builder_settings(fixture, GAP_OPEN=-0.5, GAP_EXTEND=0.0)
            seen = {}
            real_benchmark = Embedding_MSA.benchmark_msa_devices

            def record_tree(linkage_matrix, valid_headers, *args, **kwargs):
                seen["tree"] = np.array(linkage_matrix)
                seen["headers"] = list(valid_headers)
                return real_benchmark(linkage_matrix, valid_headers, *args, **kwargs)

            with mock.patch.multiple(
                Embedding_MSA, benchmark_msa_devices=record_tree, **values
            ), mock.patch.object(Embedding_MSA.mp, "set_start_method"), mock.patch.object(
                Embedding_MSA.mp, "Pool", InProcessPool
            ), mock.patch.object(
                Embedding_MSA, "h5py", wraps=h5py
            ) as h5py_module, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                Embedding_MSA.run_msa_builder()
                output = pathlib.Path(Embedding_MSA.OUTPUT_FASTA)
            actual = output.read_bytes()
            expected = reference_alignment_fasta(
                fixture, seen["tree"], seen["headers"], -0.5, 0.0
            )
            embedding_opens = [
                call for call in h5py_module.File.call_args_list
                if pathlib.Path(call.args[0]).name == EMBED_NAME
            ]
            # Release the builder's HDF5 handles before the folder is removed.
            gc.collect()

        self.assertEqual(len(seen["headers"]), 22)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.count(b">"), 22)
        # Leaves are read through the handle opened at the start.
        self.assertEqual(len(embedding_opens), 1)


class MergeSpeedTests(unittest.TestCase):
    def test_four_thousand_sequence_merge_beats_the_string_merge_per_sequence(self):
        # Wall-clock limits fail on a busy machine, so the array merge is
        # compared with the 96dbeb6 string merge timed in this process on 200
        # of the 4000 sequences. Appending one character per sequence and
        # column cost about 80 times more per sequence than the array merge
        # (it made a 44k-sequence MSA spend hours merging). Requiring a tenth
        # leaves room for load spikes, and the string merge itself fails.
        rng = np.random.default_rng(13)
        clusters = [
            Embedding_MSA.MSACluster(
                -1, rng.choice(ALIGNMENT_CODES, size=(4000, 2000)), list(range(2000)),
                rng.standard_normal((4000, 768)).astype(np.float16),
            )
            for _ in range(2)
        ]
        path = random_path(rng, 4000, 4000, matches=3000)
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            merged = Embedding_MSA.merge_clusters(
                clusters[0], clusters[1], path,
                clusters[0].embedding, clusters[1].embedding,
            )
            timings.append(time.perf_counter() - started)
        self.assertEqual(merged.aligned.shape, (5000, 4000))

        subset = 100
        string_clusters = [
            StringCluster(
                [row.tobytes().decode("ascii")
                 for row in np.ascontiguousarray(cluster.aligned[:, :subset].T)],
                cluster.ids[:subset],
            )
            for cluster in clusters
        ]
        started = time.perf_counter()
        reference_merge(
            string_clusters[0], string_clusters[1], path,
            clusters[0].embedding, clusters[1].embedding,
        )
        reference_seconds = time.perf_counter() - started

        per_sequence = min(timings) / 4000
        reference_per_sequence = reference_seconds / (2 * subset)
        self.assertLess(
            per_sequence * 10,
            reference_per_sequence,
            f"array merge of 4000 sequences: {timings} s; "
            f"string merge of {2 * subset}: {reference_seconds:.3f} s",
        )


if __name__ == "__main__":
    unittest.main()
