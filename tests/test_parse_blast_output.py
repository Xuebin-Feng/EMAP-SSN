import contextlib
import importlib.util
import io
import json
import os
import pathlib
import random
import shutil
import subprocess
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


DIAMOND_FIELDS = (
    "Query Seq - id, Subject Seq - id, Percentage of identical matches, "
    "Alignment length, Number of mismatches, Number of gap openings, "
    "Start of alignment in query, End of alignment in query, "
    "Start of alignment in subject, End of alignment in subject, "
    "Expect value, Bit score"
)
DIAMOND_K0 = "diamond blastp -d db -q input.fasta -o hits.tsv -k 0 --header verbose"


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


def diamond_header(invocation=DIAMOND_K0, fields=DIAMOND_FIELDS):
    """The comment lines DIAMOND 2.1 writes with --header verbose."""
    return (
        "# DIAMOND v2.1.23. http://github.com/bbuchfink/diamond\n"
        f"# Invocation: {invocation}\n"
        f"# Fields: {fields}\n"
    )


def ranked_rows(headers, limit=None):
    """All-vs-all rows in which every query ranks targets in FASTA order.

    With ``limit``, each query keeps only its self hit and the first targets,
    as a per-query target limit does, so early records are reported by many
    more queries than any query reports.
    """
    rows = []
    for query in headers:
        targets = [query] + [header for header in headers if header != query]
        rows.extend(
            standard_row(query, target, "1e-30") for target in targets[:limit]
        )
    return "\n".join(rows) + "\n"


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
            self.assertEqual(network.attrs["matrix"], "Imported")

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
            self.assertEqual(network.attrs["search_program"], "BLASTP")
            self.assertEqual(network.attrs["search_version"], "2.17.0+")
            self.assertEqual(network.attrs["query_column_1based"], 0)
            self.assertEqual(network.attrs["subject_column_1based"], 1)
            self.assertEqual(network.attrs["evalue_column_1based"], 2)

    def test_first_token_only_does_not_match_full_fasta_header(self):
        fasta = self.write_fasta([("A protein description", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(standard_row("A", "B", "1e-5") + "\n")

        with self.assertRaisesRegex(
            blast_tabular.BlastParseError, "is not present in the FASTA manifest"
        ) as raised:
            self.build(fasta, blast)
        message = str(raised.exception)
        self.assertIn(
            "matches the first word of the FASTA header 'A_protein_description'",
            message,
        )
        self.assertIn("DIAMOND --outfmt 6 qtitle stitle evalue", message)

    def test_unmatched_header_mentions_fasta_headers_with_spaces(self):
        fasta = self.write_fasta([("A protein description", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(standard_row("Z", "B", "1e-5") + "\n")

        with self.assertRaises(blast_tabular.BlastParseError) as raised:
            self.build(fasta, blast)

        message = str(raised.exception)
        self.assertNotIn("matches the first word", message)
        self.assertIn("1 FASTA header(s) contain spaces", message)

        plain = self.write_fasta([("A", "AAAA"), ("B", "BBBB")], "plain.fasta")
        with self.assertRaises(blast_tabular.BlastParseError) as raised:
            self.build(plain, blast)
        self.assertNotIn("first word", str(raised.exception))

    def test_diamond_titles_import_descriptive_headers_with_custom_columns(self):
        fasta = self.write_fasta(
            [("A protein one", "AAAA"), ("B protein two", "BBBB")]
        )
        blast = self.write_blast(
            diamond_header(fields="Query title, Subject title, Expect value")
            + "A protein one\tA protein one\t1e-50\n"
            + "A protein one\tB protein two\t1e-9\n"
            + "B protein two\tB protein two\t1e-50\n"
            + "B protein two\tA protein one\t1e-8\n"
        )

        output, summary, _ = self.build(
            fasta,
            blast,
            layout="custom_columns",
            query_column=1,
            subject_column=2,
            evalue_column=3,
        )

        self.assertEqual(summary.unique_edges, 1)
        self.assertEqual(summary.warnings, ())
        with h5py.File(output, "r") as network:
            self.assertEqual(
                network["headers"].asstr()[:].tolist(),
                ["A_protein_one", "B_protein_two"],
            )
            np.testing.assert_allclose(network["score"][:], [9.0])

    def test_diamond_verbose_header_is_recorded_as_search_provenance(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(diamond_header() + ranked_rows(["A", "B"]))

        output, summary, _ = self.build(fasta, blast)

        self.assertEqual(summary.search.program, "DIAMOND")
        self.assertEqual(summary.search.version, "2.1.23")
        self.assertEqual(summary.search.invocation, DIAMOND_K0)
        self.assertEqual(summary.search.network_tag, "DIAMOND")
        self.assertEqual(summary.warnings, ())
        with h5py.File(output, "r") as network:
            # The network type stays BLAST so viewers load it as E-values.
            self.assertEqual(network.attrs["model_name"], "BLAST")
            self.assertEqual(network.attrs["search_program"], "DIAMOND")
            self.assertEqual(network.attrs["search_version"], "2.1.23")
            self.assertEqual(network.attrs["search_invocation"], DIAMOND_K0)
            self.assertEqual(network.attrs["blast_program"], "Unknown")
            self.assertEqual(network.attrs["queries_observed"], 2)
            self.assertEqual(network.attrs["max_targets_per_query"], 2)
            self.assertEqual(json.loads(network.attrs["import_warnings"]), [])

    def test_headerless_output_has_unknown_search_provenance(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(standard_row("A", "B", "1e-5") + "\n")

        output, summary, _ = self.build(fasta, blast)

        self.assertEqual(summary.search, blast_tabular.SearchHeader())
        self.assertEqual(summary.search.network_tag, "BLAST")
        with h5py.File(output, "r") as network:
            for name in ("search_program", "search_version", "search_invocation"):
                self.assertEqual(network.attrs[name], "Unknown")

    def test_search_header_reads_only_the_leading_comment_block(self):
        cases = (
            ("# BLASTP 2.17.0+\n# Query: A\n", ("BLASTP", "2.17.0+", "Unknown")),
            (
                "# BLAST processed 2 queries\n# BLASTP 2.16.0+\n",
                ("BLASTP", "2.16.0+", "Unknown"),
            ),
            (
                "\ufeff# DIAMOND v2.1.9.163 http://example\n# Invocation: x blastp\n",
                ("DIAMOND", "2.1.9.163", "x blastp"),
            ),
            (
                standard_row("A", "B", "1e-5") + "\n" + diamond_header(),
                ("Unknown", "Unknown", "Unknown"),
            ),
        )
        for index, (text, expected) in enumerate(cases):
            with self.subTest(text=text):
                path = self.temp_path / f"header_{index}.tsv"
                path.write_text(text, encoding="utf-8", newline="\n")
                header = blast_tabular.read_search_header(path)
                self.assertEqual(
                    (header.program, header.version, header.invocation), expected
                )
        self.assertEqual(
            blast_tabular.read_search_header(self.temp_path / "missing.tsv"),
            blast_tabular.SearchHeader(),
        )

    def test_diamond_target_limit_follows_diamond_option_parsing(self):
        cases = (
            ("diamond blastp -d db -q q.fasta", (25, False, None)),
            ("diamond blastp -d db -k 0", (0, True, None)),
            ("diamond blastp -d db -k0", (0, True, None)),
            ("diamond blastp -d db -k=0", (0, True, None)),
            ("diamond blastp --max-target-seqs 100", (100, True, None)),
            ("diamond blastp -k 5 -k 0", (0, True, None)),
            ("diamond blastp -d db --top 10", (25, False, "10")),
            (r"C:\Program Files\NCBI\diamond.exe blastp -d db", (25, False, None)),
            ("diamond blastx -d db -k 3", (3, True, None)),
            ("diamond view -a hits.daa -o hits.tsv", (None, False, None)),
            ("Unknown", (None, False, None)),
        )
        for invocation, expected in cases:
            with self.subTest(invocation=invocation):
                self.assertEqual(
                    blast_tabular.diamond_target_limit(invocation), expected
                )

    def test_diamond_default_target_limit_is_reported_from_its_invocation(self):
        headers = [f"S{index:02d}" for index in range(30)]
        fasta = self.write_fasta([(header, "ACDE" + header) for header in headers])
        default_run = "diamond blastp -d db -q input.fasta -o hits.tsv --header verbose"
        blast = self.write_blast(
            diamond_header(default_run) + ranked_rows(headers, limit=25)
        )

        _, summary, _ = self.build(fasta, blast, batch_size=1000)

        self.assertEqual(len(summary.warnings), 1)
        self.assertIn(
            "--max-target-seqs 25 (its default) on 30 sequences, and 30 queries "
            "reached that limit",
            summary.warnings[0],
        )
        self.assertEqual(summary.max_targets_per_query, 25)
        self.assertEqual(summary.queries_at_max_targets, 30)

    def test_declared_unlimited_diamond_search_is_not_flagged(self):
        headers = [f"S{index:02d}" for index in range(12)]
        fasta = self.write_fasta([(header, "ACDE" + header) for header in headers])
        # The invocation is authoritative: -k 0 means no per-query limit, even
        # though these rows would otherwise look capped at four targets.
        blast = self.write_blast(diamond_header() + ranked_rows(headers, limit=4))

        _, summary, _ = self.build(fasta, blast, batch_size=1000)

        self.assertEqual(summary.warnings, ())

    def test_diamond_top_option_is_reported(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        blast = self.write_blast(
            diamond_header("diamond blastp -d db -q input.fasta --top 10")
            + standard_row("A", "A", "1e-50")
            + "\n"
            + standard_row("B", "B", "1e-50")
            + "\n"
        )

        _, summary, _ = self.build(fasta, blast)

        self.assertEqual(len(summary.warnings), 1)
        self.assertIn("--top 10", summary.warnings[0])

    def test_headerless_target_limit_is_inferred_from_reporting_asymmetry(self):
        headers = [f"S{index:02d}" for index in range(40)]
        fasta = self.write_fasta([(header, "ACDE" + header) for header in headers])
        blast = self.write_blast(ranked_rows(headers, limit=25))

        output, summary, _ = self.build(fasta, blast, batch_size=1000)

        self.assertEqual(len(summary.warnings), 1)
        self.assertIn(
            "limited to 25 target sequences per query: 40 queries report exactly "
            "25 targets",
            summary.warnings[0],
        )
        self.assertIn(
            "DIAMOND reports at most 25 targets per query unless it is run with -k 0",
            summary.warnings[0],
        )
        with h5py.File(output, "r") as network:
            self.assertEqual(
                json.loads(network.attrs["import_warnings"]), list(summary.warnings)
            )

        small = self.write_fasta(
            [(header, "ACDE" + header) for header in headers[:12]], "small.fasta"
        )
        capped = self.write_blast(ranked_rows(headers[:12], limit=4), "capped.tsv")
        _, summary, _ = self.build(small, capped, name="capped.h5")
        self.assertEqual(len(summary.warnings), 1)
        self.assertIn("limited to 4 target sequences per query", summary.warnings[0])
        self.assertIn("Rerun without a per-query target limit", summary.warnings[0])

    def test_complete_family_tied_at_the_largest_target_count_is_not_flagged(self):
        family = [f"F{index}" for index in range(10)]
        singletons = [f"single{index}" for index in range(5)]
        fasta = self.write_fasta(
            [(header, "ACDE" + header) for header in family + singletons]
        )
        # Every family member reports all ten members: ten queries tie at the
        # largest target count, but nobody is reported more often than that.
        rows = ranked_rows(family) + "".join(
            standard_row(header, header, "1e-40") + "\n" for header in singletons
        )
        blast = self.write_blast(rows)

        _, summary, _ = self.build(fasta, blast, batch_size=1000)

        self.assertEqual(summary.warnings, ())
        self.assertEqual(summary.max_targets_per_query, 10)
        self.assertEqual(summary.queries_at_max_targets, 10)
        self.assertEqual(summary.queries_observed, 15)
        self.assertTrue(summary.query_rows_grouped)

    def test_isolated_reporting_asymmetry_is_not_flagged(self):
        family = ["A", "B", "C", "D", "E", "F"]
        fasta = self.write_fasta(
            [(header, "ACDE" + header) for header in family + ["G"]]
        )
        # Without a limit, G finds A and B but they miss G, as weak hits near
        # the E-value cutoff do. A and B then sit at the largest target count
        # and are reported by more queries, yet two such queries are noise.
        rows = ranked_rows(family) + "".join(
            standard_row("G", subject, "1e-3") + "\n" for subject in ("G", "A", "B")
        )
        blast = self.write_blast(rows)

        _, summary, _ = self.build(fasta, blast, batch_size=1000)

        self.assertEqual(summary.max_targets_per_query, 6)
        self.assertEqual(summary.queries_at_max_targets, 6)
        self.assertEqual(summary.warnings, ())

    def test_ungrouped_rows_skip_the_inferred_target_limit_check(self):
        headers = [f"S{index:02d}" for index in range(12)]
        fasta = self.write_fasta([(header, "ACDE" + header) for header in headers])
        rows = ranked_rows(headers, limit=4).splitlines()
        # Sorting by subject interleaves the query blocks.
        rows.sort(key=lambda row: row.split("\t")[1])
        blast = self.write_blast("\n".join(rows) + "\n")

        _, summary, _ = self.build(fasta, blast, batch_size=1000)

        self.assertFalse(summary.query_rows_grouped)
        self.assertEqual(summary.warnings, ())

    def test_records_missing_as_queries_are_reported_when_self_hits_exist(self):
        records = [(name, "ACDE" + name) for name in ("A", "B", "C", "D", "E")]
        fasta = self.write_fasta(records)
        blast = self.write_blast(
            "".join(
                standard_row(query, subject, "1e-20") + "\n"
                for query, subject in (("A", "A"), ("A", "B"), ("B", "B"), ("C", "C"))
            )
        )

        _, summary, _ = self.build(fasta, blast)

        self.assertEqual(summary.queries_observed, 3)
        self.assertEqual(len(summary.warnings), 1)
        self.assertIn(
            "2 of 5 FASTA records never appear as a query (for example 'D', 'E')",
            summary.warnings[0],
        )

        without_self = self.write_blast(
            standard_row("A", "B", "1e-20") + "\n", "no_self.tsv"
        )
        _, summary, _ = self.build(fasta, without_self, name="no_self.h5")
        self.assertEqual(summary.warnings, ())

    def test_declared_fields_must_put_an_evalue_in_the_selected_column(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        fields = "Query Seq - id, Subject Seq - id, Bit score, Expect value"
        blast = self.write_blast(
            diamond_header(fields=fields) + "A\tB\t40.5\t1e-9\n"
        )
        custom = {
            "layout": "custom_columns",
            "query_column": 1,
            "subject_column": 2,
        }

        with self.assertRaisesRegex(
            blast_tabular.BlastParseError,
            r"line 3: # Fields declares column 3 as 'Bit score', not an E-value",
        ):
            self.build(fasta, blast, evalue_column=3, **custom)

        output, summary, _ = self.build(fasta, blast, evalue_column=4, **custom)
        self.assertEqual(summary.unique_edges, 1)

        short = self.write_blast(
            diamond_header(fields="Query Seq - id, Subject Seq - id, Expect value")
            + standard_row("A", "B", "1e-5")
            + "\n",
            "short.tsv",
        )
        with self.assertRaisesRegex(
            blast_tabular.BlastParseError, "declares only 3 columns"
        ):
            self.build(fasta, short, name="short.h5")

    def test_diamond_simple_header_row_is_rejected_with_advice(self):
        fasta = self.write_fasta([("A", "AAAA"), ("B", "BBBB")])
        names = (
            "qseqid sseqid pident length mismatch gapopen qstart qend sstart send "
            "evalue bitscore"
        )
        blast = self.write_blast(
            "\t".join(names.split()) + "\n" + standard_row("A", "B", "1e-5") + "\n"
        )

        with self.assertRaisesRegex(
            blast_tabular.BlastParseError, r"line 1: .*'--header simple'"
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


def load_parse_tool():
    """Import the tool script afresh so a test can set its module settings."""
    spec = importlib.util.spec_from_file_location(
        "parse_blast_output_under_test", SRC_DIR / "tools" / "Parse_BLAST_Output.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParseBlastToolTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = pathlib.Path(self._temp_dir.name)
        self.tool = load_parse_tool()

    def tearDown(self):
        self._temp_dir.cleanup()

    def write(self, name, text):
        (self.temp_path / name).write_text(text, encoding="utf-8", newline="\n")

    def test_output_name_follows_the_declared_search_program(self):
        self.write("input.fasta", ">A\nAAAA\n>B\nBBBB\n")
        cases = (
            ("diamond.tsv", diamond_header() + ranked_rows(["A", "B"]), "[DIAMOND]"),
            ("plain.tsv", ranked_rows(["A", "B"]), "[BLAST]"),
            ("missing.tsv", None, "[BLAST]"),
        )
        for name, text, tag in cases:
            with self.subTest(name=name):
                if text is not None:
                    self.write(name, text)
                self.tool.INPUT_BLAST_TABULAR = name
                self.tool.INPUT_FASTA = "input.fasta"
                self.tool.FASTA_DIR = str(self.temp_path)
                self.tool.NETWORK_DIR = str(self.temp_path)
                self.tool.configure_runtime_paths()
                stem = pathlib.Path(name).stem
                self.assertEqual(
                    self.tool.OUTPUT_HDF5,
                    str(self.temp_path / f"{stem}_{tag}_EValue.h5"),
                )

    def test_main_reports_search_provenance_and_warnings(self):
        headers = [f"S{index:02d}" for index in range(30)]
        self.write(
            "set.fasta", "".join(f">{header}\nACDE{header}\n" for header in headers)
        )
        default_run = "diamond blastp -d set -q set.fasta -o hits.tsv --header verbose"
        self.write("hits.tsv", diamond_header(default_run) + ranked_rows(headers, 25))
        settings = self.temp_path / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "DIRECTORIES": {
                        "FASTA_DIR": str(self.temp_path),
                        "NETWORK_DIR": str(self.temp_path),
                    },
                    "Parse_BLAST_Output.py": {
                        "INPUT_BLAST_TABULAR": "hits.tsv",
                        "INPUT_FASTA": "set.fasta",
                    },
                }
            ),
            encoding="utf-8",
        )

        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(
            stdout
        ):
            self.assertEqual(self.tool.main([str(settings)]), 0)

        report = stdout.getvalue()
        self.assertIn("Search Program:       DIAMOND 2.1.23", report)
        self.assertIn(f"Search Command:       {default_run}", report)
        self.assertIn("Queries Observed:     30 of 30", report)
        self.assertIn("Most Targets/Query:   25 (30 queries)", report)
        self.assertIn(
            "WARNING: DIAMOND was run with --max-target-seqs 25 (its default)", report
        )
        self.assertLess(report.index("WARNING:"), report.index("Conversion complete."))
        self.assertTrue((self.temp_path / "hits_[DIAMOND]_EValue.h5").is_file())


class DiamondIntegrationTests(unittest.TestCase):
    """Import real DIAMOND output; skipped unless ``diamond`` is on PATH."""

    @classmethod
    def setUpClass(cls):
        cls.diamond = shutil.which("diamond")
        if cls.diamond is None:
            raise unittest.SkipTest("DIAMOND is not on PATH.")
        usage = subprocess.run(
            [cls.diamond, "blastp"], capture_output=True, text=True, timeout=120
        )
        if "verbose" not in usage.stdout + usage.stderr:
            raise unittest.SkipTest("This DIAMOND has no '--header verbose'.")
        cls.version = cls.run_diamond("version").split()[-1]
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = pathlib.Path(cls._temp_dir.name)

        # Two families of 40 exceed DIAMOND's default 25 targets per query.
        rng = random.Random(1234)
        residues = "ACDEFGHIKLMNPQRSTVWY"
        records = []
        for family in range(2):
            root = "".join(rng.choice(residues) for _ in range(160))
            for member in range(40):
                records.append((
                    f"fam{family}_member{member:02d}",
                    "".join(
                        rng.choice(residues) if rng.random() < 0.12 else residue
                        for residue in root
                    ),
                ))
        for index in range(5):
            records.append((
                f"single{index}",
                "".join(rng.choice(residues) for _ in range(160)),
            ))
        (cls.temp_path / "set.fasta").write_text(
            "".join(f">{header}\n{sequence}\n" for header, sequence in records),
            encoding="utf-8",
            newline="\n",
        )
        cls.run_diamond("makedb", "--in", "set.fasta", "-d", "set", "--quiet")

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    @classmethod
    def run_diamond(cls, *arguments):
        completed = subprocess.run(
            [cls.diamond, *arguments],
            cwd=getattr(cls, "temp_path", None),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode:
            raise AssertionError(
                f"diamond {' '.join(arguments)} failed: {completed.stderr}"
            )
        return completed.stdout

    def search(self, name, *options):
        self.run_diamond(
            "blastp", "-d", "set", "-q", "set.fasta", "-o", name,
            "--threads", "1", "--quiet", *options,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return blast_tabular.build_blast_network(
                self.temp_path / name,
                self.temp_path / "set.fasta",
                self.temp_path / f"{name}.h5",
                show_progress=False,
            )

    def test_verbose_header_records_provenance_and_the_default_limit(self):
        capped = self.search("capped.tsv", "--header", "verbose")
        complete = self.search("complete.tsv", "--header", "verbose", "-k", "0")

        self.assertEqual(capped.search.program, "DIAMOND")
        self.assertEqual(capped.search.version, self.version)
        self.assertIn("blastp", complete.search.invocation)
        self.assertEqual(len(capped.warnings), 1)
        self.assertIn("--max-target-seqs 25 (its default)", capped.warnings[0])
        self.assertEqual(complete.warnings, ())
        self.assertEqual(complete.queries_observed, 85)
        self.assertGreater(complete.unique_edges, capped.unique_edges)

    def test_headerless_rows_reveal_the_default_limit(self):
        capped = self.search("plain.tsv")

        self.assertEqual(capped.search, blast_tabular.SearchHeader())
        self.assertEqual(len(capped.warnings), 1)
        self.assertIn("limited to 25 target sequences per query", capped.warnings[0])


if __name__ == "__main__":
    unittest.main()
