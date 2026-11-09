"""The `distribution` command (commands/distribution.py): nodes colored by the residue at one position."""

import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import numpy as np
import matplotlib.colors as mcolors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
import Command_Engine
from commands.distribution import RESIDUE_COLORS, print_help, run
from tests.sparse_alignment import load_manager, write_fasta

START = (0.5, 0.25, 0.75, 0.9)  # a color no residue has, so a recolored node is easy to tell
NOISE = (0.8, 0.8, 0.8, 0.4)


def rgba(residue):
    return mcolors.to_rgba(RESIDUE_COLORS[residue])


class FakeViewer(SimpleNamespace):
    """The parts of the Viewer `distribution` uses; _save_state keeps what one undo restores."""

    def _save_state(self):
        self.history.append(self.current_colors.copy())

    def promote_nodes(self, mask):
        pass

    def update_nodes(self):
        pass

    def undo(self):
        self.current_colors = self.history.pop()


class DistributionTests(unittest.TestCase):
    def run_distribution(self, residues, args, hidden=(), selected=(), cluster_labels=None, unaligned=0):
        """Run the command on one node per entry of RESIDUES (one MSA column), then UNALIGNED nodes."""
        headers = [f"s{i}" for i in range(len(residues))]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "msa.fasta")
        write_fasta(path, [(h, r) for h, r in zip(headers, residues)])
        with mock.patch.object(Alignment_Manager.cfg, "FILTER_MIN_OCCUPANCY", 0):
            alignment = load_manager(path, headers + [f"extra{i}" for i in range(unaligned)])
        n = len(headers) + unaligned
        viewer = FakeViewer(
            alignment=alignment, alignment_offset=0,
            full_headers=headers + [f"extra{i}" for i in range(unaligned)],
            selected_indices=list(selected), n_nodes=n,
            visible_mask=np.array([i not in hidden for i in range(n)], dtype=bool),
            cluster_labels=None if cluster_labels is None else np.asarray(cluster_labels),
            group_labels=None, metadata=None, console_text=SimpleNamespace(text=""),
            current_colors=np.tile(START, (n, 1)), history=[],
        )
        outcomes = []
        output = StringIO()
        with (
            mock.patch.object(Command_Engine, "command_succeeded", lambda v, m=None, artifact=None: outcomes.append(("ok", str(m)))),
            mock.patch.object(Command_Engine, "command_failed", lambda v, m: outcomes.append(("failed", str(m)))),
            redirect_stdout(output),
        ):
            run(viewer, args)
        return viewer, output.getvalue(), outcomes

    def assert_color(self, viewer, node, color):
        np.testing.assert_allclose(viewer.current_colors[node], color, err_msg=f"node {node}")

    def test_each_residue_gets_its_fixed_color_in_either_case(self):
        viewer, output, outcomes = self.run_distribution(list("KRDEwHA"), ["[1]"])
        for node, residue in enumerate("KRDEWHA"):
            self.assert_color(viewer, node, rgba(residue))
        self.assertEqual(RESIDUE_COLORS["K"], "#1f77b4")
        self.assertEqual(outcomes, [("ok", "Colored 7 nodes by the residue at position 1.")])
        self.assertEqual(viewer.console_text.text, "Colored 7 nodes by the residue at position 1.")

    def test_gap_and_other_symbols_take_the_noise_color(self):
        viewer, output, _ = self.run_distribution(list("A-XB*"), ["[1]"])
        self.assert_color(viewer, 0, rgba("A"))
        for node in range(1, 5):
            self.assert_color(viewer, node, NOISE)
        self.assertIn("gap", output)
        self.assertIn("other", output)

    def test_the_table_lists_residues_by_count_with_percentages(self):
        _, output, _ = self.run_distribution(list("KKKRRG-"), ["[1]"])
        rows = [line for line in output.splitlines() if line.startswith("| ") and "Node Count" not in line]
        text = [" ".join(row.replace("|", " ").split()[1:]) for row in rows]
        self.assertEqual(text, ["K 3 42.86%", "R 2 28.57%", "G 1 14.29%", "gap 1 14.29%"])

    def test_expression_and_bracket_in_either_order(self):
        for args in (['#cluster_1#', "[1]"], ["[1]", '#cluster_1#']):
            with self.subTest(args=args):
                viewer, _, _ = self.run_distribution(list("KRDE"), args, cluster_labels=[1, 1, 2, 2])
                self.assert_color(viewer, 0, rgba("K"))
                self.assert_color(viewer, 1, rgba("R"))
                self.assert_color(viewer, 2, START)
                self.assert_color(viewer, 3, START)

    def test_end_names_the_last_position(self):
        for bracket in ("[E]", "[END]", "[e]"):
            with self.subTest(bracket=bracket):
                viewer, _, _ = self.run_distribution(["KR", "RK"], [bracket])
                self.assert_color(viewer, 0, rgba("R"))
                self.assert_color(viewer, 1, rgba("K"))

    def test_default_target_is_the_selection_then_all_visible_nodes(self):
        viewer, _, _ = self.run_distribution(list("KRDE"), ["[1]"], selected=[1, 2])
        self.assert_color(viewer, 0, START)
        self.assert_color(viewer, 1, rgba("R"))
        self.assert_color(viewer, 2, rgba("D"))
        self.assert_color(viewer, 3, START)

        viewer, output, _ = self.run_distribution(list("KRDE"), ["[1]"])
        for node, residue in enumerate("KRDE"):
            self.assert_color(viewer, node, rgba(residue))
        self.assertIn("Defaulting to all visible nodes", output)

    def test_hidden_nodes_are_skipped_and_keep_their_colors(self):
        viewer, output, outcomes = self.run_distribution(list("KRDE"), ["[1]"], hidden={1}, selected=[0, 1])
        self.assert_color(viewer, 0, rgba("K"))
        self.assert_color(viewer, 1, START)
        self.assertEqual(outcomes[0][0], "ok")
        viewer, output, _ = self.run_distribution(list("KRDE"), ["[1]"], hidden={1})
        self.assert_color(viewer, 1, START)
        self.assertIn("(3 sequences mapped)", output)

    def test_an_expression_matching_only_hidden_nodes_succeeds_with_a_note(self):
        viewer, output, outcomes = self.run_distribution(list("KRD"), ['"s2"', "[1]"], hidden={2})
        note = "The expression '\"s2\"' matched only hidden nodes, which distribution skips. Nothing was colored."
        self.assertEqual(outcomes, [("ok", note)])
        self.assertEqual(viewer.history, [])
        self.assert_color(viewer, 2, START)

    def test_an_expression_matching_nothing_succeeds_with_a_note(self):
        viewer, output, outcomes = self.run_distribution(list("KRD"), ['"nomatch"', "[1]"])
        self.assertEqual(outcomes, [("ok", "No sequences matched the expression '\"nomatch\"'. Nothing was colored.")])
        self.assertEqual(viewer.history, [])

    def test_matched_nodes_that_are_all_outside_the_alignment_fail(self):
        viewer, output, outcomes = self.run_distribution(list("KR"), ['"extra*"', "[1]"], unaligned=2)
        self.assertEqual(outcomes, [(
            "failed", "The expression '\"extra*\"' matched 2 nodes, but none is in the alignment. Nothing was colored.")])
        self.assertEqual(viewer.history, [])

    def test_nodes_outside_the_alignment_keep_their_colors_and_are_counted(self):
        viewer, output, outcomes = self.run_distribution(list("KR"), ['"*"', "[1]"], unaligned=2)
        self.assert_color(viewer, 0, rgba("K"))
        self.assert_color(viewer, 2, START)
        self.assert_color(viewer, 3, START)
        self.assertEqual(outcomes, [("ok", "Colored 2 nodes by the residue at position 1. "
                                           "2 matched nodes not in the alignment kept their colors.")])

    def test_more_than_one_position_fails_and_changes_nothing(self):
        for bracket in ("[1,2]", "[1 2]", "[1-2]", "[1-E]", "[K>10%]", "[(K>0.1)&(R>0.05)]", "[]"):
            with self.subTest(bracket=bracket):
                viewer, output, outcomes = self.run_distribution(["KR", "RK"], [bracket])
                self.assertEqual([kind for kind, _ in outcomes], ["failed"])
                self.assertIn("only one alignment position", outcomes[0][1])
                self.assertEqual(viewer.history, [])
                np.testing.assert_array_equal(viewer.current_colors, np.tile(START, (2, 1)))

    def test_a_position_not_in_the_alignment_and_a_missing_bracket_fail(self):
        viewer, _, outcomes = self.run_distribution(list("KR"), ["[99]"])
        self.assertEqual(outcomes, [("failed", "Error: Position 99 is not in the active alignment mapping.")])
        viewer, _, outcomes = self.run_distribution(list("KR"), ['"*"'])
        self.assertEqual([kind for kind, _ in outcomes], ["failed"])
        self.assertIn("No bracketed argument", outcomes[0][1])
        self.assertEqual(viewer.history, [])

    def test_undo_restores_the_previous_colors(self):
        viewer, _, _ = self.run_distribution(list("KRDE"), ["[1]"], selected=[0, 2])
        self.assertEqual(len(viewer.history), 1)
        viewer.undo()
        np.testing.assert_array_equal(viewer.current_colors, np.tile(START, (4, 1)))

    def test_the_table_columns_line_up_with_the_header(self):
        _, output, _ = self.run_distribution(list("KKKRRG-X"), ["[1]"])
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        lines = [ansi.sub("", line) for line in output.splitlines() if line.startswith("| ")]
        self.assertEqual(len(lines), 6)  # header, K, R, G, gap, other
        bars = [[i for i, c in enumerate(line) if c == "|"] for line in lines]
        for line, positions in zip(lines, bars):
            self.assertEqual(positions, bars[0], line)

    def test_the_residues_gap_and_other_are_the_categories_in_the_tables_order(self):
        from desktop.Node_Colors import key_state
        viewer, _, _ = self.run_distribution(list("KKKRRG-X"), ["[1]"], hidden=(7,))

        state = key_state(viewer)
        (layer,) = state["layers"]
        self.assertEqual(layer["title"], "Residue at 1")
        self.assertEqual([entry["label"] for entry in layer["categories"]], ["K", "R", "G", "Gap"])  # the hidden X is not counted
        np.testing.assert_allclose(layer["categories"][0]["rgba"], rgba("K"))
        np.testing.assert_allclose(layer["categories"][3]["rgba"], NOISE)
        self.assertEqual(state["owner"].tolist(), [1] * 7 + [-1])
        self.assertEqual(state["category"].tolist(), [0, 0, 0, 1, 1, 2, 3, -1])
        self.assertTrue(state["labelled"][:7].all())

    def test_other_symbols_have_a_category_of_their_own(self):
        from desktop.Node_Colors import key_state
        viewer, _, _ = self.run_distribution(list("AX*-"), ["[1]"])

        labels = [entry["label"] for entry in key_state(viewer)["layers"][0]["categories"]]
        self.assertEqual(labels, ["A", "Gap", "Other"])
        self.assertEqual(key_state(viewer)["category"].tolist(), [0, 2, 2, 1])

    def test_nothing_colored_declares_nothing(self):
        from desktop.Node_Colors import key_state
        viewer, _, _ = self.run_distribution(list("KRDE"), ['"nobody"', "[1]"])
        self.assertIsNone(key_state(viewer))

    def test_help_prints(self):
        output = StringIO()
        with redirect_stdout(output):
            print_help()
        self.assertIn("distribution [EXPRESSION] [POSITION]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
