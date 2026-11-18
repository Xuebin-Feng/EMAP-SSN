"""The identity command (commands/identity.py): the pairwise % identity matrix of one or two selections, its
orientation, the positions it counts, the files it writes and what it refuses."""

import csv
import io
import os
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

# Command_Engine.command_succeeded() imports Viewer_Command_Portal (and with it
# PySide6) on first use; imported here, as the other command tests do.
import Viewer_Command_Portal  # noqa: F401,E402
from commands import identity as identity_command  # noqa: E402
from tests.command_fixtures import network_viewer, run_command  # noqa: E402
from tests.sparse_alignment import sparse_alignment  # noqa: E402
from tests.test_alignment_profiles import pair_identity  # noqa: E402

# Cluster 1 is nodes 0-2, cluster 2 nodes 3-5; node 5 has no MSA row.
ROWS = ["AAAA", "AAAC", "AACC", "AAD-", "ACDD", None]
CLUSTERS = [1, 1, 1, 2, 2, 2]


class CapturingScheduler:
    def __init__(self):
        self.job = None

    def is_output_path_reserved(self, _path):
        return False

    def enqueue(self, **job):
        self.job = job
        return 1


def aligned_viewer(rows=ROWS, clusters=CLUSTERS):
    """A viewer whose MSA gives node i the row rows[i] (None: no row), with occupancy numbering 1..L."""
    viewer = network_viewer(len(rows), (), cluster_labels=clusters)
    viewer.full_headers = [f"seq_{index}" for index in range(len(rows))]
    viewer.selected_indices = []
    length = max(len(row) for row in rows if row is not None)
    present = [row is not None for row in rows]
    viewer.alignment = SimpleNamespace(
        aln=sparse_alignment([(f"seq_{i}", row) for i, row in enumerate(rows) if row is not None]),
        viewer_to_aln=np.where(present, np.cumsum(present) - 1, -1),
        label_to_col={str(column + 1): column for column in range(length)},
        col_to_label={column: str(column + 1) for column in range(length)},
        has_reference=False,
        msa_file="test.fasta",
    )
    viewer.background_job_scheduler = CapturingScheduler()
    return viewer


class IdentityCommandCase(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = mock.patch.object(identity_command, "IDENTITY_DIRECTORY", self.folder.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def queue(self, args, viewer=None):
        """Run the command; return (viewer, succeeded, failed, output, job or None)."""
        viewer = viewer or aligned_viewer()
        succeeded, failed, output = run_command(identity_command, viewer, args)
        return viewer, succeeded, failed, output, viewer.background_job_scheduler.job

    def finish(self, job):
        """Run the queued job as the scheduler would; return (result, printed text)."""
        printed = io.StringIO()
        with redirect_stdout(printed):
            result = job["worker"](job["payload"])
        return result, printed.getvalue()

    def run_identity(self, args, viewer=None):
        viewer, succeeded, failed, _output, job = self.queue(args, viewer)
        failed.assert_not_called()
        succeeded.assert_called_once()
        self.assertIsNotNone(job)
        result, printed = self.finish(job)
        return viewer, job, result, printed

    def workbook(self, path):
        import openpyxl
        return openpyxl.load_workbook(path, read_only=True)

    def sheet_rows(self, path, name="Identity"):
        workbook = self.workbook(path)
        try:
            return [list(row) for row in workbook[name].iter_rows(values_only=True)]
        finally:
            workbook.close()

    def expected(self, rows_nodes, col_nodes, rows=ROWS):
        return [[round(pair_identity(rows[i], rows[j]), 2) if pair_identity(rows[i], rows[j]) is not None else None
                 for j in col_nodes] for i in rows_nodes]


class MatrixTests(IdentityCommandCase):
    def test_one_expression_compares_every_sequence_with_every_other(self):
        _viewer, job, result, printed = self.run_identity(["#cluster_1#"])
        path = result["save_path"]
        self.assertTrue(path.endswith(".xlsx") and os.path.basename(path).startswith("identity_"))
        rows = self.sheet_rows(path)
        self.assertEqual(rows[0], ["% identity", "seq_0", "seq_1", "seq_2"])
        self.assertEqual([row[0] for row in rows[1:]], ["seq_0", "seq_1", "seq_2"])
        self.assertEqual([row[1:] for row in rows[1:]], [[100, 75, 50], [75, 100, 75], [50, 75, 100]])
        self.assertIn("Pairwise % identity of 3 sequences (#cluster_1#), all against all: 3 pairs.", printed)
        self.assertIn("Mean 66.7%, median 75.0%, minimum 50.0%, maximum 75.0%.", printed)
        # Each row's mean is over the other two.
        means = self.sheet_rows(path, "Row means")
        self.assertEqual(means[1:], [["seq_0", 62.5, 2], ["seq_1", 75, 2], ["seq_2", 62.5, 2]])
        self.assertIn("Highest mean identity to the others: seq_1 (75.0%).", printed)
        self.assertEqual(str(result["message"]), f"Saved the pairwise identity of 3 x 3 sequences to "
                                                 f"{os.path.basename(path)} (mean 66.7%).")

    def test_the_first_expression_gives_the_rows_and_the_second_the_columns(self):
        _viewer, job, result, printed = self.run_identity(["#cluster_2#", "#cluster_1#"])
        rows = self.sheet_rows(result["save_path"])
        self.assertEqual(rows[0], ["% identity", "seq_0", "seq_1", "seq_2"])
        self.assertEqual([row[0] for row in rows[1:]], ["seq_3", "seq_4"])
        self.assertEqual([row[1:] for row in rows[1:]], self.expected([3, 4], [0, 1, 2]))
        self.assertIn("rows #cluster_2# (2 sequences), columns #cluster_1# (3 sequences): 6 pairs.", printed)
        # Node 5 matched cluster 2 but has no MSA row.
        self.assertIn("1 node(s) '#cluster_2#' matched have no MSA row and are left out.", printed)
        summary = [row[0] for row in self.sheet_rows(result["save_path"], "Summary") if row]
        self.assertIn("seq_5", summary)
        self.assertEqual(len(self.sheet_rows(result["save_path"], "Column means")), 4)
        # The other order turns the matrix round.
        _viewer, job, swapped, _printed = self.run_identity(["#cluster_1#", "#cluster_2#"])
        rows = self.sheet_rows(swapped["save_path"])
        self.assertEqual([row[0] for row in rows[1:]], ["seq_0", "seq_1", "seq_2"])
        self.assertEqual([row[1:] for row in rows[1:]], self.expected([0, 1, 2], [3, 4]))

    def test_a_node_both_expressions_match_is_not_compared_with_itself_in_the_figures(self):
        _viewer, _job, result, printed = self.run_identity(["#cluster_1#", '"seq_0"'])
        rows = self.sheet_rows(result["save_path"])
        self.assertEqual([row[1:] for row in rows[1:]], [[100], [75], [50]])
        self.assertIn("2 pairs.", printed)
        self.assertIn("1 node(s) both expressions match are not compared with themselves", printed)
        self.assertIn("Mean 62.5%, median 62.5%, minimum 50.0%, maximum 75.0%.", printed)

    def test_positions_count_only_their_columns_and_every_one_either_sequence_fills(self):
        rows = ["AAAA", "A-AC", "AC-C", "--DD"]
        viewer = aligned_viewer(rows, [1, 1, 1, 1])
        _viewer, _job, result, printed = self.run_identity(["#cluster_1#", "[2-3]"], viewer)
        matrix = [row[1:] for row in self.sheet_rows(result["save_path"])[1:]]
        # Columns 2-3: AA, -A, C-, -D. A sequence missing part of the region differs there.
        self.assertEqual(matrix, [[100, 50, 0, 0], [50, 100, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]])
        self.assertIn("over positions [2-3] (2 columns)", printed)
        self.assertNotIn("incl. internal gaps", printed)
        # Without positions the same rows are compared as local alignments.
        _viewer, _job, whole, printed = self.run_identity(["#cluster_1#"], aligned_viewer(rows, [1, 1, 1, 1]))
        matrix = [row[1:] for row in self.sheet_rows(whole["save_path"])[1:]]
        self.assertEqual(matrix, self.expected(range(4), range(4), rows))
        self.assertIn("incl. internal gaps", printed)

    def test_positions_missing_from_the_mapping_are_named_and_skipped(self):
        viewer, succeeded, failed, output, job = self.queue(["#cluster_1#", "[3-6]"])
        failed.assert_not_called()
        self.assertIn("Warning: Position 5 was not found in the active alignment mapping.", output)
        self.assertEqual(job["payload"]["columns"], [2, 3])

    def test_a_pair_without_an_alignment_has_an_empty_cell_and_is_counted(self):
        rows = ["AA--", "--AA", "AAAA"]
        _viewer, _job, result, printed = self.run_identity(["#cluster_1#"], aligned_viewer(rows, [1, 1, 1]))
        matrix = [row[1:] for row in self.sheet_rows(result["save_path"])[1:]]
        self.assertEqual(matrix[0], [100, None, 100])
        self.assertIn("1 pair(s) share no aligned residue and have no value.", printed)
        self.assertIn("     -", printed)


class FileTests(IdentityCommandCase):
    def test_csv_and_tsv_hold_the_matrix_alone(self):
        rows = ["AA--", "--AA", "AAAA"]
        for name, delimiter in (("out.csv", ","), ("OUT.TSV", "\t")):
            with self.subTest(name=name):
                _viewer, _job, result, _printed = self.run_identity(
                    ["#cluster_1#", name], aligned_viewer(rows, [1, 1, 1]))
                self.assertEqual(result["save_path"], os.path.join(self.folder.name, name))
                with open(result["save_path"], encoding="utf-8", newline="") as handle:
                    text = handle.read()
                self.assertNotIn("\r", text)
                table = list(csv.reader(io.StringIO(text), delimiter=delimiter))
                self.assertEqual(table, [
                    ["% identity", "seq_0", "seq_1", "seq_2"],
                    ["seq_0", "100.00", "", "100.00"],
                    ["seq_1", "", "100.00", "100.00"],
                    ["seq_2", "100.00", "100.00", "100.00"],
                ])

    def test_an_explicit_name_replaces_its_file_and_leaves_no_partial_file(self):
        path = os.path.join(self.folder.name, "c1.xlsx")
        with open(path, "wb") as handle:
            handle.write(b"old")
        _viewer, job, result, _printed = self.run_identity(["#cluster_1#", "c1.xlsx"])
        self.assertTrue(job["allow_overwrite"])
        self.assertEqual(result["save_path"], path)
        self.assertEqual(self.sheet_rows(path)[0][0], "% identity")
        self.assertEqual(os.listdir(self.folder.name), ["c1.xlsx"])

    def test_the_console_shows_at_most_20_by_20_cells(self):
        rng = np.random.default_rng(42)
        rows = ["".join(rng.choice(list("ACDE"), 12)) for _ in range(25)]
        _viewer, _job, result, printed = self.run_identity(["#cluster_1#"], aligned_viewer(rows, [1] * 25))
        lines = printed.splitlines()
        header = next(line for line in lines if line.strip().startswith("1 ") and line.rstrip().endswith("20"))
        self.assertEqual(header.split(), [str(number) for number in range(1, 21)])
        self.assertIn("(The first 20 of 25 rows and 20 of 25 columns; the file holds them all.)", printed)
        self.assertIn("20 = seq_19", printed)
        self.assertNotIn("21 = seq_20", printed)
        self.assertEqual(len(self.sheet_rows(result["save_path"])), 26)


class RefusalTests(IdentityCommandCase):
    def assert_refused(self, args, text, viewer=None):
        viewer, succeeded, failed, output, job = self.queue(args, viewer)
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn(text, str(failed.call_args.args[1]))
        self.assertIsNone(job)

    def test_arguments_it_cannot_use_are_refused_before_anything_runs(self):
        cases = [
            (["#cluster_1#", "#cluster_2#", '"seq_0"'], "identity takes one or two expressions; found 3"),
            (["#cluster_1#", "[1]", "[2]"], "found a second one: '[2]'"),
            (["#cluster_1#", "a.csv", "b.xlsx"], "Give one output filename; found a second one: 'b.xlsx'"),
            (["#cluster_1#", "report"], "Unrecognized identity argument 'report'. A filename must end in .xlsx"),
            (["#cluster_1#", "[1-x]"], "Error:"),
            (["#cluster_1#", "[40-50]"], "None of the requested positions is in the active alignment mapping."),
            (["#cluster_1#", ".csv"], "needs a name before its extension"),
        ]
        for args, text in cases:
            with self.subTest(args=args):
                self.assert_refused(args, text)

    def test_a_matrix_too_large_for_its_file_is_refused(self):
        with mock.patch.dict(identity_command.MAX_CELLS, {".xlsx": 8, ".csv": 9}):
            self.assert_refused(["#cluster_1#"], "A 3 x 3 matrix (9 cells) is too large for a .xlsx file")
            viewer, succeeded, failed, _output, job = self.queue(["#cluster_1#", "c.csv"])
            failed.assert_not_called()
            self.assertIsNotNone(job)
        with mock.patch.object(identity_command, "XLSX_MAX_COLUMNS", 2):
            self.assert_refused(["#cluster_1#"], "at most 4,000,000 cells and 2 columns")

    def test_no_alignment_or_no_aligned_match_fails(self):
        viewer = aligned_viewer()
        viewer.alignment = None
        self.assert_refused(["#cluster_1#"], "No alignment loaded", viewer)
        self.assert_refused(['"seq_5"'], "matched 1 node, but none is in the alignment")

    def test_nothing_to_compare_succeeds_with_a_note_and_writes_nothing(self):
        for args, text in ((["#cluster_1#&#cluster_2#"], "No sequences matched the expression"),
                           ([], "No nodes are selected")):
            with self.subTest(args=args):
                viewer, succeeded, failed, output, job = self.queue(args)
                failed.assert_not_called()
                succeeded.assert_called_once()
                self.assertIn(text, str(succeeded.call_args.args[1]))
                self.assertIsNone(job)
        self.assertEqual(os.listdir(self.folder.name), [])

    def test_the_selection_is_used_without_an_expression(self):
        viewer = aligned_viewer()
        viewer.selected_indices = [0, 2]
        _viewer, job, result, printed = self.run_identity([], viewer)
        self.assertEqual([row[0] for row in self.sheet_rows(result["save_path"])[1:]], ["seq_0", "seq_2"])
        self.assertIn("($sele$)", printed)


if __name__ == "__main__":
    unittest.main()
