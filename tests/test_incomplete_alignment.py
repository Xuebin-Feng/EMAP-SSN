"""MSAs whose rows do not match the network one to one.

Covers loading a partial or reordered MSA (coverage, reference fallback,
warnings), three-state AA expressions for unaligned nodes, and the commands
(alignment, query, logo, label, select) that map network nodes to alignment
rows through viewer_to_aln.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
import Command_Engine
import EMAPSSN_Config as cfg
from commands import alignment as alignment_command
from commands import label as label_command
from commands import logo as logo_command
from commands import query as query_command
from commands import select as select_command
from tests.sparse_alignment import load_manager, write_fasta


class TTYBuffer(io.StringIO):
    def isatty(self):
        return True


class IncompleteAlignmentLoaderTests(unittest.TestCase):
    def test_reference_modes_preserve_occupancy_and_offset_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "alignment.fasta")
            write_fasta(fasta_path, [("reference", "A-C"), ("other", "ATC")])
            sparse_path = os.path.join(directory, "alignment.h5")
            with h5py.File(sparse_path, "w") as hf:
                matrix = hf.create_group("matrix")
                matrix.attrs["shape"] = (2, 3)
                matrix.create_dataset("data", data=np.array([1, 2, 1, 3, 2], dtype=np.uint8))
                matrix.create_dataset("indices", data=np.array([0, 2, 0, 1, 2], dtype=np.int32))
                matrix.create_dataset("indptr", data=np.array([0, 2, 5], dtype=np.int32))
                hf.create_dataset("headers", data=[b"reference", b"other"])
                hf.create_dataset("int_to_aa", data=json.dumps({"1": "A", "2": "C", "3": "T"}).encode())

            for path in (fasta_path, sparse_path):
                for reference in ("reference", "absent", ""):
                    with self.subTest(path=path, reference=reference):
                        with mock.patch.object(Alignment_Manager.cfg, "FILTER_MIN_OCCUPANCY", 75), redirect_stdout(io.StringIO()) as output:
                            manager = Alignment_Manager.Alignment_Manager(
                                path, full_headers=["reference", "other"],
                                active_reference=reference, alignment_offset=10,
                            )
                        self.assertEqual(len(manager.aln), 2)
                        self.assertEqual(set(manager.valid_cols), {0, 2})
                        matched = reference == "reference"
                        self.assertEqual(manager.has_reference, matched)
                        self.assertEqual(manager.offset, 10 if matched else 0)
                        self.assertEqual(manager.col_to_label, {0: "11", 2: "12"} if matched else {0: "1", 2: "2"})
                        if reference == "absent":
                            self.assertIn("was not found", output.getvalue())
                            self.assertIn("pure occupancy mode", output.getvalue())
                            self.assertIn("alignment offsets are inactive", output.getvalue())
                        else:
                            self.assertNotIn("was not found", output.getvalue())

    def test_partial_fasta_load_tracks_exact_coverage_and_warns_red(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "partial.fasta")
            write_fasta(
                fasta_path,
                [("node1 full", "AC"), ("node3 full", "TC")],
            )
            output = TTYBuffer()
            with redirect_stdout(output):
                manager = Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=["node1_full", "node2_full", "node3_full"],
                    active_reference="node1",
                )

        self.assertEqual(manager.matched_headers, ["node1_full", "node3_full"])
        self.assertEqual(manager.missing_headers, ["node2_full"])
        np.testing.assert_array_equal(manager.viewer_to_aln, np.array([0, -1, 1]))
        np.testing.assert_array_equal(
            manager.aligned_node_mask,
            np.array([True, False, True]),
        )
        self.assertTrue(manager.has_reference)
        warning = output.getvalue()
        self.assertIn("\033[91mWARNING: Incomplete MSA coverage", warning)
        self.assertIn("Aligned network nodes: 2/3 (66.7%)", warning)
        self.assertIn("  - node2_full", warning)
        self.assertIn("\033[0m", warning)

    def test_exact_full_header_matching_does_not_accept_shortened_description(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "shortened.fasta")
            write_fasta(fasta_path, [("node1", "AC")])
            with redirect_stdout(io.StringIO()):
                manager = Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=["node1_full_description"],
                )

        self.assertEqual(len(manager.aln), 0)
        self.assertEqual(manager.matched_headers, [])
        self.assertEqual(manager.missing_headers, ["node1_full_description"])

    def test_zero_overlap_is_a_loaded_empty_alignment_state(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "zero.fasta")
            write_fasta(fasta_path, [("other", "AC")])
            with redirect_stdout(io.StringIO()) as output:
                manager = Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=["node1", "node2"],
                    active_reference="node1",
                    alignment_offset=10,
                )

        self.assertIsNotNone(manager.aln)
        self.assertEqual(len(manager.aln), 0)
        self.assertEqual(manager.valid_cols, set())
        self.assertFalse(manager.has_reference)
        self.assertEqual(manager.offset, 0)
        np.testing.assert_array_equal(manager.viewer_to_aln, np.array([-1, -1]))
        self.assertIn("Aligned network nodes: 0/2 (0.0%)", output.getvalue())
        self.assertIn("pure occupancy mode", output.getvalue())

    def test_missing_reference_falls_back_without_disabling_partial_msa(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "missing_ref.fasta")
            write_fasta(fasta_path, [("node1", "AC"), ("node3", "TC")])
            with redirect_stdout(io.StringIO()):
                manager = Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=["node1", "node2", "node3"],
                    active_reference="node2",
                    alignment_offset=10,
                )

        self.assertIsNotNone(manager.aln)
        self.assertEqual(len(manager.aln), 2)
        self.assertFalse(manager.has_reference)
        self.assertEqual(manager.offset, 0)
        self.assertEqual(manager.resolved_ref_full, "None")
        self.assertTrue(manager.col_to_label)

    def test_sparse_hdf5_partial_load_preserves_extra_row_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            h5_path = os.path.join(directory, "partial.h5")
            with h5py.File(h5_path, "w") as hf:
                matrix = hf.create_group("matrix")
                matrix.attrs["shape"] = (2, 2)
                matrix.create_dataset("data", data=np.array([1, 2, 2, 1], dtype=np.uint8))
                matrix.create_dataset("indices", data=np.array([0, 1, 0, 1], dtype=np.int32))
                matrix.create_dataset("indptr", data=np.array([0, 2, 4], dtype=np.int32))
                hf.create_dataset("headers", data=np.array([b"node1", b"extra"]))
                hf.create_dataset("int_to_aa", data=json.dumps({"1": "A", "2": "C"}).encode("utf-8"))

            with redirect_stdout(io.StringIO()):
                manager = Alignment_Manager.Alignment_Manager(
                    h5_path,
                    full_headers=["node1", "node2"],
                )

        self.assertEqual(len(manager.aln), 1)
        self.assertEqual(manager.aln.headers, ["node1"])
        self.assertEqual(manager.missing_headers, ["node2"])
        np.testing.assert_array_equal(manager.viewer_to_aln, np.array([0, -1]))

    def test_warning_lists_ten_examples_and_reports_omitted_count(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "truncated_warning.fasta")
            write_fasta(fasta_path, [("matched", "AC")])
            headers = ["matched"] + [f"missing_{i}" for i in range(12)]
            with redirect_stdout(io.StringIO()) as output:
                Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=headers,
                )

        warning = output.getvalue()
        self.assertIn("showing 10 of 12", warning)
        self.assertIn("  - missing_9", warning)
        self.assertNotIn("  - missing_10", warning)
        self.assertIn("... and 2 more.", warning)


class ThreeStateExpressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        fasta_path = os.path.join(self.temp_dir.name, "partial.fasta")
        write_fasta(fasta_path, [("node1", "AC"), ("node3", "TC")])
        with redirect_stdout(io.StringIO()):
            self.alignment = Alignment_Manager.Alignment_Manager(
                fasta_path,
                full_headers=["node1", "node2", "node3"],
                active_reference="node1",
            )
        self.headers = ["node1", "node2", "node3"]
        self.viewer = SimpleNamespace(
            full_headers=self.headers,
            alignment=self.alignment,
        )
        self.viewer_to_aln, self.valid_indices = Command_Engine.get_alignment_mapping(
            self.viewer
        )
        self.metadata = {
            "Length": {
                "type": "number",
                "values": np.array([600.0, 600.0, 600.0]),
            }
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def evaluate(self, expression):
        return Command_Engine.parse_advanced_expression(
            expression,
            self.viewer_to_aln,
            self.valid_indices,
            self.headers,
            alignment=self.alignment,
            metadata=self.metadata,
        )

    def test_aa_and_negated_aa_are_both_false_for_unaligned_nodes(self):
        np.testing.assert_array_equal(self.evaluate("A1"), [True, False, False])
        np.testing.assert_array_equal(self.evaluate("!A1"), [False, False, True])

    def test_grouped_aa_and_negation_preserve_unknown_unaligned_nodes(self):
        np.testing.assert_array_equal(self.evaluate("(AT)1"), [True, False, True])
        np.testing.assert_array_equal(self.evaluate("!(AC)1"), [False, False, True])

    def test_independent_known_or_clause_can_match_unaligned_node(self):
        np.testing.assert_array_equal(
            self.evaluate("A1|{Length>500}"),
            [True, True, True],
        )

    def test_unknown_propagates_through_and_xor_and_parenthesized_not(self):
        np.testing.assert_array_equal(
            self.evaluate("A1&{Length>500}"),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            self.evaluate("A1^{Length>500}"),
            [False, False, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("!(A1|{Length<500})"),
            [False, False, True],
        )

    def test_non_aa_expression_still_matches_unaligned_nodes(self):
        np.testing.assert_array_equal(
            self.evaluate("{Length>500}"),
            [True, True, True],
        )


class IncompleteAlignmentViewerSmokeTests(unittest.TestCase):
    def test_viewer_load_method_accepts_partial_and_zero_coverage(self):
        import EMAPSSN_Config as cfg
        from EMAPSSN_Viewer import MainViewer
        old_msa = cfg.MSA_FILE
        try:
            with tempfile.TemporaryDirectory() as directory:
                for filename, records, expected_count in [
                    ("partial.fasta", [("node1", "AC")], 1),
                    ("zero.fasta", [("other", "AC")], 0),
                ]:
                    msa_path = os.path.join(directory, filename)
                    write_fasta(msa_path, records)
                    cfg.MSA_FILE = msa_path

                    viewer = MainViewer.__new__(MainViewer)
                    viewer.full_headers = ["node1", "node2"]
                    viewer.active_reference = "node1"
                    viewer.alignment_offset = 0
                    with redirect_stdout(io.StringIO()):
                        viewer.load_global_alignment()

                    self.assertIsNotNone(viewer.alignment.aln)
                    self.assertEqual(len(viewer.alignment.aln), expected_count)
        finally:
            cfg.MSA_FILE = old_msa


class IncompleteAlignmentCommandTests(unittest.TestCase):
    def test_alignment_command_accepts_zero_overlap_and_reports_coverage(self):
        from EMAPSSN_Viewer import MainViewer

        with tempfile.TemporaryDirectory() as directory:
            msa_path = os.path.join(directory, "zero.fasta")
            write_fasta(msa_path, [("other", "AC")])
            headers = ["node1", "node2"]
            viewer = SimpleNamespace(
                full_headers=headers,
                active_reference="node1",
                alignment_offset=0,
                alignment=SimpleNamespace(aln=None),
                console_text=SimpleNamespace(text=""),
            )
            # The command points cfg.MSA_FILE at the new MSA and reloads
            # through the Viewer's own method.
            viewer.load_global_alignment = lambda: MainViewer.load_global_alignment(viewer)
            old_msa = cfg.MSA_FILE
            try:
                with redirect_stdout(io.StringIO()) as output:
                    alignment_command.run(viewer, [msa_path])
            finally:
                cfg.MSA_FILE = old_msa

        self.assertIsNotNone(viewer.alignment.aln)
        self.assertEqual(len(viewer.alignment.aln), 0)
        self.assertIn("0/2 network nodes aligned", output.getvalue())
        self.assertIn("0/2 aligned", viewer.console_text.text)

    def test_alignment_command_accepts_a_quoted_path_with_spaces(self):
        from EMAPSSN_Viewer import MainViewer

        with tempfile.TemporaryDirectory() as directory:
            folder = os.path.join(directory, "My Files")
            os.mkdir(folder)
            msa_path = os.path.join(folder, "x.fasta")
            write_fasta(msa_path, [("node1", "AC"), ("node2", "AC")])
            viewer = SimpleNamespace(
                full_headers=["node1", "node2"],
                active_reference="",
                alignment_offset=0,
                alignment=SimpleNamespace(aln=None),
                console_text=SimpleNamespace(text=""),
            )
            viewer.load_global_alignment = lambda: MainViewer.load_global_alignment(viewer)
            old_msa = cfg.MSA_FILE
            try:
                # The dispatcher splits the command line on whitespace and the
                # command joins the pieces back, quotes included.
                with redirect_stdout(io.StringIO()) as output:
                    alignment_command.run(viewer, f'"{msa_path}"'.split())
            finally:
                cfg.MSA_FILE = old_msa

        self.assertEqual(len(viewer.alignment.aln), 2)
        self.assertIn("Success: Loaded alignment 'x.fasta' (2/2 network nodes aligned)", output.getvalue())

    def failed_alignment_command(self, directory, name, content):
        """Run `alignment` on a file the loader rejects: (viewer, previous alignment, failures, console line)."""
        from EMAPSSN_Viewer import MainViewer

        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            handle.write(content)
        previous = SimpleNamespace(aln=None)
        viewer = SimpleNamespace(
            full_headers=["node1", "node2"],
            active_reference="node1",
            alignment_offset=0,
            alignment=previous,
            console_text=SimpleNamespace(text=""),
        )
        viewer.load_global_alignment = lambda: MainViewer.load_global_alignment(viewer)
        old_msa = cfg.MSA_FILE
        try:
            with mock.patch.object(alignment_command.Command_Engine, "command_failed") as failed, \
                    mock.patch.object(alignment_command.Command_Engine, "command_succeeded") as succeeded, \
                    redirect_stdout(io.StringIO()):
                alignment_command.run(viewer, [path])
            self.assertEqual(cfg.MSA_FILE, old_msa)
        finally:
            cfg.MSA_FILE = old_msa
        succeeded.assert_not_called()
        self.assertIs(viewer.alignment, previous)
        return viewer, [str(call.args[1]) for call in failed.call_args_list]

    def test_a_failed_load_reports_the_loaders_reason(self):
        cases = (
            ("unequal.fasta", b">node1\nAC-\n>node2\nAC\n",
             "MSA rejected: MSA sequences must have equal aligned lengths; expected 3, found 'node2' (2)."),
            ("empty.fasta", b"", "MSA rejected: MSA FASTA contains no records."),
            ("duplicate.fasta", b">node1\nAC\n>node1\nAC\n", "MSA rejected: Duplicate MSA header: 'node1'."),
            ("notes.txt", b">node1\nAC\n", "MSA rejected: Unsupported alignment extension '.txt'. Expected .fasta or .h5."),
            ("broken.h5", b"not an HDF5 file", "Error loading HDF5: "),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, content, reason in cases:
                with self.subTest(file=name):
                    viewer, failures = self.failed_alignment_command(directory, name, content)

                    self.assertTrue(failures[0].startswith(f"\nFailed to load alignment '{name}': {reason}"), failures)
                    self.assertNotIn("failed to return an alignment", failures[0])
                    # The record of the failure keeps its one summary, and the console line adds the reason.
                    self.assertEqual(failures[1:], ["Load failed. Reverted to previous alignment."])
                    self.assertTrue(
                        viewer.console_text.text.startswith("Load failed. Reverted to previous alignment. " + reason),
                        viewer.console_text.text,
                    )

    def test_the_failure_reason_on_the_console_line_is_translated(self):
        from utilities import Localization

        previous = Localization.set_translator(lambda template, n: f"<{template}>")
        self.addCleanup(Localization.set_translator, previous)
        with tempfile.TemporaryDirectory() as directory:
            viewer, failures = self.failed_alignment_command(directory, "empty.fasta", b"")

        # The terminal and the command record stay English.
        self.assertEqual(failures[0], "\nFailed to load alignment 'empty.fasta': MSA rejected: MSA FASTA contains no records.")
        self.assertEqual(
            viewer.console_text.text,
            "<Load failed. Reverted to previous alignment.> <MSA rejected: <MSA FASTA contains no records.>>",
        )

    def test_success_report_treats_the_word_none_as_no_reference(self):
        from EMAPSSN_Viewer import MainViewer

        with tempfile.TemporaryDirectory() as directory:
            msa_path = os.path.join(directory, "toy.fasta")
            write_fasta(msa_path, [("node1", "AC"), ("node2", "AC")])
            for configured, inactive in (("None", False), ("missing", True)):
                with self.subTest(active_reference=configured):
                    viewer = SimpleNamespace(
                        full_headers=["node1", "node2"],
                        active_reference=configured,
                        alignment_offset=0,
                        alignment=SimpleNamespace(aln=None),
                        console_text=SimpleNamespace(text=""),
                    )
                    viewer.load_global_alignment = lambda: MainViewer.load_global_alignment(viewer)
                    old_msa = cfg.MSA_FILE
                    try:
                        with mock.patch.object(alignment_command.Command_Engine, "command_succeeded") as succeeded, \
                                redirect_stdout(io.StringIO()):
                            alignment_command.run(viewer, [msa_path])
                    finally:
                        cfg.MSA_FILE = old_msa

                    text = str(succeeded.call_args.args[1])
                    self.assertEqual("reference inactive (occupancy mode)" in text, inactive)

    def test_query_logo_and_label_reject_loaded_zero_coverage_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            msa_path = os.path.join(directory, "zero.fasta")
            write_fasta(msa_path, [("other", "AC")])
            alignment = load_manager(msa_path, ["node1"], "node1")
            viewer = SimpleNamespace(
                alignment=alignment,
                full_headers=["node1"],
                active_reference="node1",
                console_text=SimpleNamespace(text=""),
            )

            with redirect_stdout(io.StringIO()):
                query_command.run(viewer, ["[1]"])
            self.assertIn("no aligned rows", viewer.console_text.text)

            logo_command.run(viewer, ["[1]"])
            self.assertIn("no aligned rows", viewer.console_text.text)

            with redirect_stdout(io.StringIO()):
                label_command.run(viewer, [])
            self.assertIn("no aligned rows", viewer.console_text.text)

    def test_select_not_aa_excludes_unaligned_node(self):
        with tempfile.TemporaryDirectory() as directory:
            msa_path = os.path.join(directory, "partial.fasta")
            write_fasta(msa_path, [("node1", "AC"), ("node3", "TC")])
            headers = ["node1", "node2", "node3"]
            alignment = load_manager(msa_path, headers, "node1")
            viewer = SimpleNamespace(
                alignment=alignment,
                full_headers=headers,
                visible_mask=np.ones(3, dtype=bool),
                selected_indices=[],
                console_text=SimpleNamespace(text=""),
                update_selection_visual=lambda: None,
            )

            with mock.patch.object(cfg, "HEADER_LIST_DIR", directory):
                select_command.run(viewer, ["!A1"])

        self.assertEqual(viewer.selected_indices, [2])


class CapturingScheduler:
    def __init__(self):
        self.job = None

    def is_output_path_reserved(self, _path):
        return False

    def enqueue(self, **job):
        self.job = job
        return 1


class AlignmentRowOrderTests(unittest.TestCase):
    """MSA rows keep the FASTA order, so viewer_to_aln is a permutation.

    Network order is n1, n2, n3 but the FASTA lists n3 (W), n1 (A), n2 (C).
    A command that indexed rows by network position would read W for n1.
    """

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        fasta_path = os.path.join(self.directory, "permuted.fasta")
        write_fasta(fasta_path, [("n3", "W"), ("n1", "A"), ("n2", "C")])
        self.headers = ["n1", "n2", "n3"]
        with mock.patch.object(Alignment_Manager.cfg, "FILTER_MIN_OCCUPANCY", 50):
            self.alignment = load_manager(fasta_path, self.headers, "n1")

    def test_loader_maps_network_nodes_to_their_fasta_rows(self):
        np.testing.assert_array_equal(self.alignment.viewer_to_aln, [1, 2, 0])
        self.assertEqual(self.alignment.resolved_ref_full, "n1")

    def test_query_reports_the_named_nodes_residue(self):
        viewer = SimpleNamespace(
            alignment=self.alignment,
            alignment_offset=0,
            full_headers=self.headers,
            selected_indices=[],
            cluster_labels=None,
            group_labels=None,
            metadata=None,
            console_text=SimpleNamespace(text=""),
        )
        output = io.StringIO()

        with redirect_stdout(output):
            query_command.run(viewer, ['"n1"', "[1]"])

        position_lines = [
            line for line in output.getvalue().splitlines() if line.startswith("Pos ")
        ]
        self.assertEqual(position_lines, ["Pos 1       \tGap   0.0% | A 100.0%"])

    def test_logo_plots_the_selected_nodes_row(self):
        scheduler = CapturingScheduler()
        viewer = SimpleNamespace(
            alignment=self.alignment,
            full_headers=self.headers,
            selected_indices=[0],
            cluster_labels=None,
            group_labels=None,
            active_reference="n1",
            console_text=SimpleNamespace(text=""),
            background_job_scheduler=scheduler,
        )

        with mock.patch.object(logo_command, "LOGO_DIRECTORY", self.directory), \
                mock.patch.object(logo_command.cfg, "HEADER_LIST_DIR", self.directory), \
                redirect_stdout(io.StringIO()):
            logo_command.run(viewer, ["[1]", "permuted.svg"])

        self.assertEqual(scheduler.job["payload"]["selected_seqs"], ("A",))

    def test_label_subsets_hold_their_nodes_rows(self):
        viewer = SimpleNamespace(
            alignment=self.alignment,
            full_headers=self.headers,
            cluster_labels=np.asarray([1, 2, 3]),
            group_labels=None,
        )
        viewer_to_aln, _ = Command_Engine.get_alignment_mapping(viewer)

        tasks = label_command._build_label_tasks(viewer, "clusters", viewer_to_aln)

        self.assertEqual(
            [
                (int(task[1]), [str(record.seq) for record in task[2]], task[4].tolist())
                for task in tasks
            ],
            [(1, ["A"], [1]), (2, ["C"], [2]), (3, ["W"], [0])],
        )


if __name__ == "__main__":
    unittest.main()
