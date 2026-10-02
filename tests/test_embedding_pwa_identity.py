"""Percent identity and exact traceback coverage for Embedding_PWA."""

import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import h5py
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import tools.Embedding_PWA as embedding_pwa
from tools.Embedding_SSEARCH import (
    QUERY_UNMATCHED_CODE,
    TARGET_UNMATCHED_CODE,
    encode_identity_residues,
)
from utilities.Network_Kernels import (
    global_score_length_identity,
    local_score_length_identity,
)


def path_score(idx_1, idx_2, score_matrix, gap_penalty, score_shift=0.0):
    return sum(
        float(score_matrix[i, j]) - score_shift
        if i != -1 and j != -1
        else gap_penalty
        for i, j in zip(idx_1, idx_2)
    )


def kernel_length_identities(mode, score_matrix, gap_penalty, seq_ref, seq_tar):
    query_codes = encode_identity_residues(seq_ref, QUERY_UNMATCHED_CODE)
    target_codes = encode_identity_residues(seq_tar, TARGET_UNMATCHED_CODE)
    kernel = (
        global_score_length_identity
        if mode == "global"
        else local_score_length_identity
    )
    _, length, identities = kernel(
        score_matrix, gap_penalty, query_codes, target_codes
    )
    return int(length), int(identities)


class PairwiseTracebackTests(unittest.TestCase):
    def test_global_traceback_follows_recorded_moves_at_near_ties(self):
        # The diagonal into the last cell falls 0.01 short of the free gaps,
        # within np.isclose's relative tolerance at a score of 5000.
        matrix = np.array([[5000.0, -10.0], [-10.0, -0.01]], dtype=np.float32)
        idx_1, idx_2, score = embedding_pwa.needleman_wunsch_custom(matrix, 0.0)

        self.assertEqual(idx_1, [0, -1, 1])
        self.assertEqual(idx_2, [0, 1, -1])
        self.assertEqual(float(score), 5000.0)
        self.assertAlmostEqual(path_score(idx_1, idx_2, matrix, 0.0), 5000.0)
        _, identities = embedding_pwa.build_alignment_columns(
            idx_1, idx_2, "AC", "AD", set()
        )
        self.assertEqual(
            (len(idx_1), identities),
            kernel_length_identities("global", matrix, 0.0, "AC", "AD"),
        )

    def test_tracebacks_match_ssearch_kernels(self):
        rng = np.random.default_rng(20261002)
        alphabet = np.array(list("ACDX"))
        for case in range(30):
            rows, cols = (int(value) for value in rng.integers(1, 13, size=2))
            matrix = rng.normal(size=(rows, cols)).astype(np.float32)
            seq_ref = "".join(rng.choice(alphabet, size=rows))
            seq_tar = "".join(rng.choice(alphabet, size=cols))
            for mode, gap, shift in (
                ("global", 0.0, 0.0),
                ("global", -1.0, 0.0),
                ("local", -1.0, 2.0),
                ("local", -2.0, 2.0),
            ):
                with self.subTest(case=case, mode=mode, gap=gap):
                    if mode == "global":
                        idx_1, idx_2, score = embedding_pwa.needleman_wunsch_custom(
                            matrix, gap
                        )
                    else:
                        idx_1, idx_2, score = embedding_pwa.smith_waterman_custom(
                            matrix, gap
                        )
                    _, identities = embedding_pwa.build_alignment_columns(
                        idx_1, idx_2, seq_ref, seq_tar, set()
                    )
                    self.assertEqual(
                        (len(idx_1), identities),
                        kernel_length_identities(
                            mode, matrix, gap, seq_ref, seq_tar
                        ),
                    )
                    self.assertAlmostEqual(
                        path_score(idx_1, idx_2, matrix, gap, shift),
                        float(score),
                        places=4,
                    )

    def test_columns_mark_only_identical_standard_residues(self):
        columns, identities = embedding_pwa.build_alignment_columns(
            [0, 1, 2, 3, -1],
            [0, 1, 2, -1, 3],
            "AXCK",
            "AXDE",
            {1},
        )
        self.assertEqual(
            columns,
            [
                ("A", "A", "|", True),
                ("X", "X", ".", False),
                ("C", "D", ".", False),
                ("K", "-", " ", False),
                ("-", "E", " ", False),
            ],
        )
        self.assertEqual(identities, 1)


class PairwiseIdentityReportTests(unittest.TestCase):
    def run_pairwise(self, directory, mode):
        rng = np.random.default_rng(7)
        database_path = os.path.join(directory, "database.h5")
        headers = ["reference", "target"]
        sequences = ["MKTAYIAKQRXW", "MKSAYIARQW"]
        # Target residues copy reference embeddings, so the alignment pairs
        # them in order and leaves the reference's internal "RX" unpaired.
        reference = rng.normal(size=(12, 6)).astype(np.float32)
        target = reference[[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]] + np.float32(0.01) * (
            rng.normal(size=(10, 6)).astype(np.float32)
        )
        with h5py.File(database_path, "w") as hf:
            hf.attrs["model_name"] = "test_model"
            hf.attrs["saving_mode"] = "float32"
            hf.attrs["num_sequences"] = 2
            hf.attrs["generation_complete"] = True
            hf.create_dataset("headers", data=np.asarray(headers, dtype="S"))
            hf.create_dataset("sequences", data=np.asarray(sequences, dtype="S"))
            embeddings = hf.create_group("embeddings")
            embeddings.create_dataset(headers[0], data=reference)
            embeddings.create_dataset(headers[1], data=target.astype(np.float32))
        database = embedding_pwa.prepare_embedding_database(database_path)
        with (
            mock.patch.object(embedding_pwa, "GENERATE_REPORT", True),
            mock.patch.object(embedding_pwa, "REPORT_DIR", directory),
            redirect_stdout(StringIO()) as output,
        ):
            embedding_pwa.run_alignment(
                "reference",
                "target",
                "",
                "",
                database_path,
                database.sequence_by_header,
                mode,
                -2.0,
                0.0,
                "",
                database.model_name,
            )
        reports = [name for name in os.listdir(directory) if name.endswith(".html")]
        self.assertEqual(len(reports), 1)
        with open(os.path.join(directory, reports[0]), encoding="utf-8") as handle:
            html = handle.read()
        return output.getvalue(), html

    def test_console_and_html_report_identity_with_mode_denominator(self):
        # Global pairs all ten target residues around the reference's
        # unpaired "RX"; local stops at Q because the final W/W pair cannot
        # pay for two gap columns. T/S and K/R are the aligned mismatches.
        cases = {
            "global": ("Identical standard residues / full alignment length "
                       "incl. end gaps", 8, 12),
            "local": ("Identical standard residues / alignment length incl. "
                      "internal gaps", 7, 9),
        }
        for mode, (definition, expected_identities, expected_length) in cases.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                console, html = self.run_pairwise(directory, mode)
                align_len = int(re.search(r"^Align Len : (\d+)$", console, re.M).group(1))
                match = re.search(
                    r"^Identity  : (\d+)/(\d+) \((\d+\.\d)%\)\n {12}(.+)$",
                    console,
                    re.M,
                )
                self.assertIsNotNone(match)
                identities, denominator = int(match.group(1)), int(match.group(2))
                self.assertEqual(
                    (identities, align_len),
                    (expected_identities, expected_length),
                )
                self.assertEqual(denominator, align_len)
                self.assertEqual(
                    match.group(3), f"{100.0 * identities / align_len:.1f}"
                )
                self.assertEqual(match.group(4), definition)

                lines = console.splitlines()
                marks = [
                    lines[index + 1]
                    for index, line in enumerate(lines)
                    if line.startswith("Ref: ")
                ]
                self.assertEqual(sum(line.count("|") for line in marks), identities)
                self.assertIn(match.group(0), html)


if __name__ == "__main__":
    unittest.main()
