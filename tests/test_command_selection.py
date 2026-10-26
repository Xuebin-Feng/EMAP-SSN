"""The selection a command uses: queued commands keep the one they were given, and no command bypasses the engine.

A command typed while a heavy command computes waits in the Viewer's command queue. It uses the
selection that stood when it was given; a selection made by hand while it waits never reaches it
(Command_Engine.command_selection and set_command_selection, Command_Queue.Job). The first classes
queue `logo`, `esmfold` and `hide` behind a clustering whose compute step the test finishes when it
likes, change the selection by hand while they wait, and check what each one then used. The last
class scans every command module, the Viewer's and the VR viewer's, and fails on a command that
reads or writes the Viewer's own selected_indices instead of going through the engine.
"""
import ast
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(PROJECT_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Viewer_Worker_Tracking  # noqa: E402
from commands import esmfold as esmfold_command  # noqa: E402
from commands import logo as logo_command  # noqa: E402
from tests.sparse_alignment import sparse_alignment  # noqa: E402
from tests.test_command_queue import CLUSTER, QueueTestCase  # noqa: E402


class RecordingScheduler:
    """What a command asks of the background scheduler: each job it queues is kept, and no path is held."""

    def __init__(self):
        self.jobs = []

    def is_output_path_reserved(self, path):
        return False

    def reservation_holder(self, path):
        return None

    def enqueue(self, **job):
        self.jobs.append(job)
        return len(self.jobs)

    def snapshot(self):
        return []


class QueuedSelectionCase(QueueTestCase):
    """A queue whose first job is a clustering that computes until the test finishes it."""

    def finish_under(self, *patches):
        """Let the clustering end, and the commands waiting behind it run, with patches in force.

        The patches go on after the commands were typed: typing a command reloads its module, which
        would undo a patch of the module's own names.
        """
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            self.finish_compute()
        self.assertEqual(self.viewer.queue.pending(), [])


class LogoSelectionTests(QueuedSelectionCase):
    RESIDUES = "ACDEFGHIKL"

    def setUp(self):
        super().setUp()
        viewer = self.viewer
        # Node i is the only one with residue RESIDUES[i], so a logo's sequences name its nodes.
        viewer.alignment = SimpleNamespace(
            aln=sparse_alignment([(header, self.RESIDUES[i] * 3) for i, header in enumerate(viewer.full_headers)]),
            viewer_to_aln=np.arange(viewer.n_nodes),
            col_to_label={column: str(column + 1) for column in range(3)},
            label_to_col={str(column + 1): column for column in range(3)},
            has_reference=True,
        )
        viewer.active_reference = "n0"
        self.scheduler = viewer.background_job_scheduler = RecordingScheduler()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def finish_logo(self):
        self.finish_under(
            mock.patch.object(logo_command, "LOGO_DIRECTORY", self.folder),
            mock.patch.object(logo_command.cfg, "HEADER_LIST_DIR", self.folder),
        )

    def logoed_nodes(self):
        """The nodes the queued logo draws, found by their residues; None when it queued none."""
        if not self.scheduler.jobs:
            return None
        sequences = self.scheduler.jobs[-1]["payload"]["selected_seqs"]
        return sorted(self.RESIDUES.index(sequence[0]) for sequence in sequences)

    def test_a_selection_cleared_by_hand_while_the_logo_waits_does_not_make_it_logo_every_node(self):
        viewer = self.viewer
        viewer.select_by_hand([3])
        viewer.type(CLUSTER)
        viewer.type("logo [1] out.svg")
        viewer.select_by_hand([])
        self.finish_logo()
        self.assertEqual(self.logoed_nodes(), [3])
        self.assertNotIn("Defaulting to ALL nodes", self.output.getvalue())

    def test_a_selection_made_by_hand_while_the_logo_waits_does_not_stop_it_logoing_every_node(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("logo [1] out.svg")
        viewer.select_by_hand([3])
        self.finish_logo()
        self.assertEqual(self.logoed_nodes(), list(range(10)))
        self.assertIn("No nodes selected. Defaulting to ALL nodes", self.output.getvalue())
        self.assertEqual(viewer.queue.finished[-1].status, 'succeeded')

    def test_a_logo_after_a_queued_select_logos_the_nodes_that_select_leaves(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("select #cluster_2#")
        viewer.type("logo [1] out.svg")
        viewer.select_by_hand([])
        self.finish_logo()
        self.assertEqual(self.logoed_nodes(), [4, 5, 6, 7])


class EsmfoldSelectionTests(QueuedSelectionCase):
    def setUp(self):
        super().setUp()
        viewer = self.viewer
        viewer.sequences_map = {header: "ACDE" for header in viewer.full_headers}
        viewer.get_web_url = mock.Mock(return_value="http://127.0.0.1:49123/api/action")
        self.scheduler = viewer.background_job_scheduler = RecordingScheduler()
        self.addCleanup(esmfold_command.RUNS.clear)
        self.addCleanup(self.discard_inputs)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def discard_inputs(self):
        for job in self.scheduler.jobs:
            job["payload"].discard_input()

    def finish_fold(self):
        """Run the waiting fold with no model, no worker and no page: only the nodes it queues are looked at."""
        backend = esmfold_command.esmfold_backend
        self.finish_under(
            mock.patch.object(esmfold_command, "launch_without_window", side_effect=AssertionError("no worker")),
            mock.patch.object(Viewer_Worker_Tracking, "track_worker"),
            mock.patch.object(esmfold_command, "_credentials_state", return_value=("saved", None)),
            mock.patch.object(esmfold_command, "_model_suffix", return_value="esm3-large-2024-03"),
            mock.patch.object(backend, "get_structures_directory", return_value=self.folder),
            mock.patch.object(backend, "register"),
            mock.patch.object(backend, "open_esmfold_ui", return_value=True),
        )

    def folded_nodes(self):
        """The names of the nodes the queued fold folds; None when it queued none."""
        if not self.scheduler.jobs:
            return None
        return [record[0] for record in self.scheduler.jobs[-1]["payload"].records]

    def test_the_fold_folds_the_selection_it_was_given_not_one_made_by_hand_since(self):
        viewer = self.viewer
        viewer.select_by_hand([3])
        viewer.type(CLUSTER)
        viewer.type("esmfold large")
        viewer.select_by_hand([6])
        self.finish_fold()
        self.assertEqual(self.folded_nodes(), ["n3"])
        self.assertEqual(viewer.selected_indices, [6])

    def test_a_selection_cleared_by_hand_while_the_fold_waits_does_not_change_what_it_folds(self):
        viewer = self.viewer
        viewer.select_by_hand([3])
        viewer.type(CLUSTER)
        viewer.type("esmfold large")
        viewer.select_by_hand([])
        self.finish_fold()
        self.assertEqual(self.folded_nodes(), ["n3"])
        self.assertEqual(viewer.queue.finished[-1].status, 'succeeded')

    def test_a_selection_made_by_hand_while_the_fold_waits_does_not_give_a_fold_given_with_none_a_node(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("esmfold large")
        viewer.select_by_hand([6])
        self.finish_fold()
        self.assertIsNone(self.folded_nodes())
        self.assertEqual(viewer.queue.finished[-1].status, 'failed')
        self.assertIn("No nodes selected", self.output.getvalue())

    def test_the_fold_after_a_queued_select_folds_the_node_that_select_leaves(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type('select "n7"')
        viewer.type("esmfold large")
        viewer.select_by_hand([1, 2])
        self.finish_fold()
        self.assertEqual(self.folded_nodes(), ["n7"])

    def test_with_nothing_selected_the_node_clicked_when_the_fold_runs_is_folded(self):
        # The click is a transient highlight the Viewer does not date, so it is not bound at the command's
        # dispatch: the node clicked while the fold waits is the one it folds, and says so.
        viewer = self.viewer
        viewer.selected_node_idx = 2
        viewer.type(CLUSTER)
        viewer.type("esmfold large")
        viewer.selected_node_idx = 5
        self.finish_fold()
        self.assertEqual(self.folded_nodes(), ["n5"])
        self.assertIn("the last clicked node is folded: n5", self.output.getvalue())

    def test_a_selection_given_wins_over_a_node_clicked_while_the_fold_waits(self):
        viewer = self.viewer
        viewer.select_by_hand([3])
        viewer.type(CLUSTER)
        viewer.type("esmfold large")
        viewer.selected_node_idx = 5
        self.finish_fold()
        self.assertEqual(self.folded_nodes(), ["n3"])
        self.assertNotIn("Nothing is selected", self.output.getvalue())


class HideSelectionTests(QueuedSelectionCase):
    def test_a_hide_hides_the_nodes_it_was_given_and_leaves_a_newer_hand_selection_less_those(self):
        viewer = self.viewer
        viewer.select_by_hand([3, 4])
        viewer.type(CLUSTER)
        viewer.type("hide")
        viewer.select_by_hand([4, 5])
        self.finish_compute()
        self.assertEqual([i for i in range(10) if not viewer.visible_mask[i]], [3, 4])
        # The hand selection stays, as it is newer, but no hidden node stays selected.
        self.assertEqual(viewer.selected_indices, [5])
        self.assertEqual(viewer.queue.finished[-1].status, 'succeeded')

    def test_a_hide_with_a_selection_made_after_it_was_given_does_not_hide_that_selection(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("hide")
        viewer.select_by_hand([8, 9])
        self.finish_compute()
        self.assertTrue(viewer.visible_mask.all())
        self.assertEqual(viewer.selected_indices, [8, 9])
        self.assertEqual(viewer.queue.finished[-1].status, 'failed')

    def test_a_hide_leaves_the_next_command_the_selection_less_the_hidden_nodes(self):
        viewer = self.viewer
        viewer.select_by_hand([3, 4])
        viewer.type(CLUSTER)
        viewer.type('hide "n3"')
        viewer.type("color red")
        viewer.select_by_hand([8])
        self.finish_compute()
        self.assertEqual(self.red_nodes(), [4])
        self.assertEqual(viewer.selected_indices, [8])


class Bypasses:
    """The direct uses of a Viewer's selected_indices in a module's source."""

    NAME = "selected_indices"

    def __init__(self, source):
        self.found = []
        self._functions = []
        self._visit(ast.parse(source))

    def _visit(self, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._functions.append(node.name)
        # viewer.selected_indices, read, assigned or deleted.
        if isinstance(node, ast.Attribute) and node.attr == self.NAME:
            self._note(node, "an attribute")
        # getattr(viewer, 'selected_indices', ...), hasattr, setattr, delattr, viewer.__dict__['selected_indices'].
        if isinstance(node, ast.Constant) and node.value == self.NAME:
            self._note(node, "the attribute's name as a string")
        for child in ast.iter_child_nodes(node):
            self._visit(child)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._functions.pop()

    def _note(self, node, how):
        self.found.append((self._functions[-1] if self._functions else None, node.lineno, how))


# Direct uses a command may keep, as (file, function): why each is not a bypass.
ALLOWED = {
    ("src/commands/hide.py", "_drop_hidden_from_selection"): (
        "The Viewer never keeps a hidden node selected (undo, box selection, picking and the select commands "
        "leave them out), so hiding a node takes it out of the live selection whichever selection that is. "
        "That is upkeep of an invariant, not a choice of selection; the command's own result goes through "
        "set_command_selection."
    ),
}


class CommandModuleScanTests(unittest.TestCase):
    """Commands read the selection through Command_Engine.command_selection and change it through set_command_selection."""

    def test_the_scan_finds_a_read_a_write_and_a_probe_by_name(self):
        source = (
            "def read(viewer):\n"
            "    return viewer.selected_indices\n"
            "def write(viewer):\n"
            "    viewer.selected_indices = [1]\n"
            "def grow(viewer):\n"
            "    viewer.selected_indices += [1]\n"
            "def probe(viewer):\n"
            "    return getattr(viewer, 'selected_indices', [])\n"
            "def has(viewer):\n"
            "    return hasattr(viewer, \"selected_indices\")\n"
            "def put(viewer):\n"
            "    setattr(viewer, 'selected_indices', [])\n"
            "def table(viewer):\n"
            "    return vars(viewer)['selected_indices']\n"
            "class Window:\n"
            "    def method(self):\n"
            "        return self.viewer.selected_indices\n"
        )
        found = Bypasses(source).found
        self.assertEqual(
            [(function, line) for function, line, _ in found],
            [("read", 2), ("write", 4), ("grow", 6), ("probe", 8), ("has", 10), ("put", 12), ("table", 14),
             ("method", 17)],
        )

    def test_the_scan_leaves_local_variables_comments_and_text_alone(self):
        source = (
            "def run(viewer):\n"
            "    # viewer.selected_indices is what a hand selection sets\n"
            "    selected_indices = command_selection(viewer)\n"
            "    for index in selected_indices:\n"
            "        print(index)\n"
            "    return 'the selected_indices of the Viewer'\n"
            "def other(selected_indices):\n"
            "    \"\"\"Reads viewer.selected_indices to say so.\"\"\"\n"
            "    return len(selected_indices)\n"
        )
        self.assertEqual(Bypasses(source).found, [])

    def command_modules(self):
        """(path relative to the project, path) of every command module of the Viewer and the VR viewer."""
        folders = [PROJECT_ROOT / "src" / "commands"]
        vr_folder = PROJECT_ROOT / "opt_vr" / "src" / "commands"
        # The VR viewer is a submodule: a checkout without it has none to scan.
        if vr_folder.is_dir():
            folders.append(vr_folder)
        modules = [(path.relative_to(PROJECT_ROOT).as_posix(), path)
                   for folder in folders for path in sorted(folder.glob("*.py"))]
        self.assertGreater(len(modules), 20)
        return modules

    def test_no_command_reads_or_writes_the_selection_of_the_viewer_directly(self):
        used = set()
        for name, path in self.command_modules():
            with self.subTest(module=name):
                source = path.read_text(encoding="utf-8")
                bypasses = [(function, line, how) for function, line, how in Bypasses(source).found
                            if (name, function) not in ALLOWED]
                used.update((name, function) for function, _line, _how in Bypasses(source).found)
                self.assertEqual(
                    bypasses, [],
                    f"{name} uses the Viewer's selected_indices: a command waiting in the queue must read the "
                    "selection through Command_Engine.command_selection (or get_selected_mask) and change it "
                    "through Command_Engine.set_command_selection",
                )
        for entry in ALLOWED:
            if entry[0].startswith("opt_vr/") and not (PROJECT_ROOT / "opt_vr" / "src" / "commands").is_dir():
                continue
            with self.subTest(allowed=entry):
                self.assertIn(entry, used, f"{entry} no longer uses selected_indices: drop it from ALLOWED")

    def test_every_allowed_use_says_why(self):
        for entry, reason in ALLOWED.items():
            with self.subTest(entry=entry):
                self.assertGreater(len(reason.split()), 10)
                self.assertTrue((PROJECT_ROOT / entry[0]).is_file(), entry)


if __name__ == "__main__":
    unittest.main()
