import contextlib
import io
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import BLAST_Tabular as blast_tabular  # noqa: E402


def standard_row(query, subject, evalue):
    return "\t".join(
        [
            query,
            subject,
            "90.0",
            "10",
            "1",
            "0",
            "1",
            "10",
            "1",
            "10",
            str(evalue),
            "20",
        ]
    )


class ParseBlastOutputTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = pathlib.Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def write_fasta(self, records, name="input.fasta"):
        path = self.temp_path / name
        text = "".join(f">{header}\n{sequence}\n" for header, sequence in records)
        path.write_text(text, encoding="utf-8")
        return path

    def write_blast(self, text, name="input.tabular"):
        path = self.temp_path / name
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def build(self, fasta, blast, name="network.h5", **kwargs):
        output = self.temp_path / name
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            summary = blast_tabular.build_blast_network(
                blast,
                fasta,
                output,
                batch_size=kwargs.pop("batch_size", 2),
                show_progress=False,
                **kwargs,
            )
        return output, summary, stdout.getvalue()

    def test_standard_outfmt6_sanitizes_both_sources_and_persists_counts(self):
        fasta = self.write_fasta([("A?", "aa**"), ("B*", "BBBB")])
        blast = self.write_blast(standard_row("A?", "B*", "1e-5") + "\n")

        output, summary, diagnostics = self.build(fasta, blast)

        self.assertIn("FASTA headers sanitized: 2 of 2", diagnostics)
        self.assertIn(
            "BLAST headers sanitized: 2 of 2 distinct headers", diagnostics
        )
        self.assertEqual(summary.unique_edges, 1)
        with h5py.File(output, "r") as network:
            self.assertEqual(network["headers"].asstr()[:].tolist(), ["A_", "B_"])
            self.assertEqual(network.attrs["fasta_headers_sanitized"], 2)
            self.assertEqual(network.attrs["blast_headers_sanitized"], 2)
            self.assertEqual(network.attrs["matrix"], "Unknown")

    def test_fasta_only_and_blast_only_header_modifications_are_reported(self):
        cases = (
            (
                "A?",
                "A_",
                "FASTA headers sanitized: 1 of 2",
                "BLAST headers sanitized: 0 of 2 distinct headers",
            ),
            (
                "A_",
                "A?",
                "FASTA headers sanitized: 0 of 2",
                "BLAST headers sanitized: 1 of 2 distinct headers",
            ),
        )
        for index, case in enumerate(cases):
            fasta_header, blast_header, fasta_report, blast_report = case
            with self.subTest(fasta_header=fasta_header, blast_header=blast_header):
                fasta = self.write_fasta(
                    [(fasta_header, "AAAA"), ("B", "BBBB")],
                    f"source_{index}.fasta",
                )
                blast = self.write_blast(
                    standard_row(blast_header, "B", "1e-5") + "\n",
                    f"source_{index}.tabular",
                )
                output, _, diagnostics = self.build(
                    fasta, blast, name=f"source_{index}.h5"
                )
                self.assertIn(fasta_report, diagnostics)
                self.assertIn(blast_report, diagnostics)
                with h5py.File(output, "r") as network:
                    self.assertEqual(network["headers"].asstr()[0], "A_")

    def test_shared_header_rule_is_used_for_complex_changes(self):
        raw_header = "  Alpha[one]??__    Beta  "
        expected_header, modified = blast_tabular.sanitize_header(raw_header)
        self.assertTrue(modified)
        self.assertEqual(expected_header, "Alpha(one)_Beta")
        fasta = self.write_fasta([(raw_header, "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(standard_row(raw_header, "B", "1e-5") + "\n")

        output, _, diagnostics = self.build(fasta, blast)

        self.assertIn("FASTA headers sanitized: 1 of 2", diagnostics)
        self.assertIn(
            "BLAST headers sanitized: 1 of 2 distinct headers", diagnostics
        )
        with h5py.File(output, "r") as network:
            self.assertEqual(network["headers"].asstr()[0], expected_header)

    def test_outfmt7_uses_full_query_comment_and_subject_title(self):
        fasta = self.write_fasta(
            [("A? real protein one", "AAAA"), ("B* real protein two", "BBBB")]
        )
        blast = self.write_blast(
            "# BLASTP 2.17.0+\n"
            "# Query: A? real protein one\n"
            "# Database: real_db\n"
            "# Fields: subject title, evalue, % identity, bit score\n"
            "# 2 hits found\n"
            "A? real protein one\t0.0\t100\t30\n"
            "B* real protein two\t1e-5\t90\t20\n"
            "# BLASTP 2.17.0+\n"
            "# Query: B* real protein two\n"
            "# Database: real_db\n"
            "# Fields: subject title, evalue, % identity, bit score\n"
            "# 2 hits found\n"
            "B* real protein two\t0.0\t100\t30\n"
            "A? real protein one\t1e-7\t90\t25\n"
        )

        output, summary, diagnostics = self.build(
            fasta, blast, layout="outfmt7_fields", matrix="BLOSUM62"
        )

        self.assertEqual(summary.self_rows, 2)
        self.assertIn("FASTA headers sanitized: 2 of 2", diagnostics)
        self.assertIn(
            "BLAST headers sanitized: 2 of 2 distinct headers", diagnostics
        )
        with h5py.File(output, "r") as network:
            self.assertEqual(
                network["headers"].asstr()[:].tolist(),
                ["A_real_protein_one", "B_real_protein_two"],
            )
            np.testing.assert_array_equal(network["i"][:], [0])
            np.testing.assert_array_equal(network["j"][:], [1])
            np.testing.assert_allclose(network["score"][:], [7.0])
            self.assertEqual(network.attrs["blast_program"], "BLASTP")
            self.assertEqual(network.attrs["blast_version"], "2.17.0+")
            self.assertEqual(network.attrs["blast_database"], "real_db")
            self.assertEqual(network.attrs["query_column_1based"], 0)
            self.assertEqual(network.attrs["subject_column_1based"], 1)
            self.assertEqual(network.attrs["evalue_column_1based"], 2)

    def test_first_token_only_does_not_match_full_fasta_header(self):
        fasta = self.write_fasta([("A protein description", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(standard_row("A", "B", "1e-5") + "\n")

        with self.assertRaisesRegex(
            blast_tabular.BlastParseError, "is not present in the FASTA manifest"
        ):
            self.build(fasta, blast)

    def test_duplicate_and_colliding_fasta_headers_fail(self):
        cases = [
            [("A", "AAAA"), ("A", "BBBB")],
            [("A?", "AAAA"), ("A*", "BBBB")],
        ]
        blast = self.write_blast("")
        for index, records in enumerate(cases):
            with self.subTest(records=records):
                fasta = self.write_fasta(records, f"collision_{index}.fasta")
                with self.assertRaises(blast_tabular.BlastParseError):
                    self.build(fasta, blast, name=f"collision_{index}.h5")

    def test_distinct_blast_headers_that_sanitize_together_fail(self):
        fasta = self.write_fasta([("A_", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(
            standard_row("A?", "B", "1e-5")
            + "\n"
            + standard_row("A*", "B", "1e-6")
            + "\n"
        )

        with self.assertRaisesRegex(blast_tabular.BlastParseError, "both sanitize"):
            self.build(fasta, blast)

    def test_custom_columns_accept_decimal_scores_and_sort_deduplicated_edges(self):
        fasta = self.write_fasta(
            [("A", "AAAA"), ("B", "BBBB"), ("C", "CCCC"), ("D", "DDDD")]
        )
        blast = self.write_blast(
            "A\tB\t0.1\textra\n"
            "C\tD\t0.01\textra\n"
            "A\tC\t10\textra\n"
            "B\tA\t0.001\textra\n"
        )

        output, summary, _ = self.build(
            fasta,
            blast,
            layout="custom_columns",
            query_column=1,
            subject_column=2,
            evalue_column=3,
            batch_size=1,
        )

        self.assertEqual(summary.unique_edges, 3)
        with h5py.File(output, "r") as network:
            np.testing.assert_array_equal(network["i"][:], [0, 0, 2])
            np.testing.assert_array_equal(network["j"][:], [1, 2, 3])
            np.testing.assert_allclose(network["score"][:], [3.0, -1.0, 2.0])
            self.assertNotIn("_sorted_runs", network)

    def test_orphans_and_zero_hit_outfmt7_produce_empty_network(self):
        fasta = self.write_fasta(
            [("A", "AAAA"), ("B", "BBBB"), ("orphan", "CCCC")]
        )
        blast = self.write_blast(
            "# BLASTP 2.17.0+\n"
            "# Query: A\n"
            "# Database: db\n"
            "# 0 hits found\n"
            "# BLAST processed 1 queries\n"
        )

        output, summary, diagnostics = self.build(
            fasta, blast, layout="outfmt7_fields"
        )

        self.assertEqual(summary.unique_edges, 0)
        self.assertIn(
            "BLAST headers sanitized: 0 of 1 distinct headers", diagnostics
        )
        with h5py.File(output, "r") as network:
            self.assertEqual(network["headers"].shape, (3,))
            self.assertEqual(network["i"].shape, (0,))
            self.assertEqual(network.attrs["subject_column_1based"], 0)
            self.assertEqual(network.attrs["evalue_column_1based"], 0)

    def test_invalid_evalues_fail_strictly_and_preserve_existing_output(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        output = self.temp_path / "existing.h5"
        output.write_bytes(b"existing-result")

        for index, value in enumerate(("-1e-5", "nan", "inf", "bad")):
            blast = self.write_blast(
                standard_row("A", "B", value) + "\n", f"invalid_{index}.tabular"
            )
            with self.subTest(value=value), self.assertRaises(
                blast_tabular.BlastParseError
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    blast_tabular.build_blast_network(
                        blast,
                        fasta,
                        output,
                        batch_size=2,
                        show_progress=False,
                    )
            self.assertEqual(output.read_bytes(), b"existing-result")
            self.assertFalse(pathlib.Path(str(output) + ".partial").exists())

    def test_malformed_and_inconsistent_rows_fail_with_line_numbers(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        malformed = self.write_blast("A B 1e-5\n", "malformed.tabular")
        with self.assertRaisesRegex(blast_tabular.BlastParseError, "line 1"):
            self.build(fasta, malformed)

        inconsistent = self.write_blast(
            "A\tB\t1e-5\textra\nA\tB\t1e-6\n", "inconsistent.tabular"
        )
        with self.assertRaisesRegex(blast_tabular.BlastParseError, "line 2"):
            self.build(
                fasta,
                inconsistent,
                layout="custom_columns",
                query_column=1,
                subject_column=2,
                evalue_column=3,
            )

    def test_outfmt7_requires_consistent_fields_per_query_block(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(
            "# Query: A\n"
            "# Fields: subject title, evalue\n"
            "B\t1e-5\n"
            "# Query: B\n"
            "# Fields: subject id, evalue\n"
            "A\t1e-5\n"
        )
        with self.assertRaisesRegex(blast_tabular.BlastParseError, "schema differs"):
            self.build(fasta, blast, layout="outfmt7_fields")

    def test_outfmt7_prefers_subject_title_when_id_is_also_present(self):
        fasta = self.write_fasta([("A full", "AAAA"), ("B full", "BBBB")])
        blast = self.write_blast(
            "# Query: A full\n"
            "# Fields: subject id, subject title, evalue\n"
            "B\tB full\t1e-5\n"
        )

        output, _, _ = self.build(fasta, blast, layout="outfmt7_fields")
        with h5py.File(output, "r") as network:
            np.testing.assert_array_equal(network["i"][:], [0])
            np.testing.assert_array_equal(network["j"][:], [1])

    def test_invalid_utf8_is_rejected(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        blast = self.temp_path / "invalid_utf8.tabular"
        blast.write_bytes(b"A\tB\t90\t10\t1\t0\t1\t10\t1\t10\t1e-5\t20\xff\n")
        with self.assertRaisesRegex(blast_tabular.BlastParseError, "not valid UTF-8"):
            self.build(fasta, blast)

    def test_validation_failure_does_not_replace_existing_output(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(standard_row("A", "B", "1e-5") + "\n")
        output = self.temp_path / "protected.h5"
        output.write_bytes(b"protected")

        with mock.patch.object(
            blast_tabular,
            "validate_final_output",
            return_value=(False, "injected validation failure"),
        ), self.assertRaisesRegex(RuntimeError, "injected validation failure"):
            with contextlib.redirect_stdout(io.StringIO()):
                blast_tabular.build_blast_network(
                    blast,
                    fasta,
                    output,
                    batch_size=1,
                    show_progress=False,
                )

        self.assertEqual(output.read_bytes(), b"protected")
        self.assertFalse(pathlib.Path(str(output) + ".partial").exists())

    def test_provenance_hashes_and_manifest_are_stable(self):
        fasta = self.write_fasta([("A", "aaaa"), ("B", "B*B*")])
        blast = self.write_blast(standard_row("A", "B", "1e-5") + "\n")
        first, _, _ = self.build(fasta, blast, name="first.h5")
        second, _, _ = self.build(fasta, blast, name="second.h5")

        with h5py.File(first, "r") as left, h5py.File(second, "r") as right:
            self.assertEqual(
                left.attrs["source_fasta_sha256"],
                right.attrs["source_fasta_sha256"],
            )
            self.assertEqual(
                left.attrs["source_blast_sha256"],
                right.attrs["source_blast_sha256"],
            )
            self.assertEqual(
                left.attrs["manifest_sha256"], right.attrs["manifest_sha256"]
            )
            self.assertEqual(left.attrs["score_transform"], "-log10(E + 1e-300)")


if __name__ == "__main__":
    unittest.main()
