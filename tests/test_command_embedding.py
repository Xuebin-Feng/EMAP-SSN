# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The `embedding` command: switch the Viewer's active embedding, as `alignment` switches its MSA.

Covers each argument form and how the file is found, the file dialog (its
cancel and the command portal's user_interaction), what the command reports
(coverage, model, width and saving mode, nodes whose sequence differs, a
network built with another model), the rollback of a file the loader refuses,
and everything that shows the active embedding following a switch: the
Viewer's info card, require_embedding and the embedding mapping and vectors,
MCP snapshots and the settings the Viewer reports, and the command catalog.
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

import Command_Engine  # noqa: E402
import EMAPSSN_Config as cfg  # noqa: E402
from commands import embedding as embedding_command  # noqa: E402
from desktop.Command_Metadata import agent_policy  # noqa: E402
from desktop.Viewer_Inspection import ViewerInspectionService, command_catalog, embedding_overview  # noqa: E402
from desktop.Viewer_Widgets import display_path  # noqa: E402
from EMAPSSN_Viewer import MainViewer  # noqa: E402
from tests.command_fixtures import reported_outcomes  # noqa: E402
from tests.embedding_fixtures import write_embedding_file  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from tests.viewer_fixtures import PortalFixture  # noqa: E402
from utilities import Localization  # noqa: E402

NODES = ["node1", "node2", "node3", "node4"]
SEQUENCES = {"node1": "AC", "node2": "AD", "node3": "AE", "node4": "AF"}
# Two embeddings that cover different nodes, with another model, width and saving mode.
FIRST = [("node1", "AC"), ("node2", "AD")]
SECOND = [("node2", "AD"), ("node3", "AE"), ("node4", "AF")]


def slashed(path):
    """A path as the command stores it in EMBED_FILE."""
    return os.path.abspath(path).replace("\\", "/")


class EmbeddingFixture:
    """A folder of small embedding files, and the settings the command reads pointed at it."""

    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.folder = os.path.join(self.root, "Embeddings")
        os.mkdir(self.folder)
        for name, value in (("EMBED_FILE", ""), ("EMBED_DIR", self.folder), ("INPUT_HDF5", "")):
            patcher = mock.patch.object(cfg, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.first = self.write("first.h5", FIRST)
        self.second = self.write("second.h5", SECOND, model_name="model_b", saving_mode="float32", width=4)

    def write(self, name, records, folder=None, **options):
        return write_embedding_file(os.path.join(folder or self.folder, name), records, **options)

    def write_network(self, model_name):
        path = os.path.join(self.root, "network.h5")
        with h5py.File(path, "w") as handle:
            handle.attrs["model_name"] = model_name
        cfg.INPUT_HDF5 = path
        return path

    def write_unusable(self, name, kind):
        """A file the loader refuses: bytes that are no HDF5 file, an incomplete database, or an HDF5 file of no embeddings."""
        path = os.path.join(self.folder, name)
        if kind == "text":
            Path(path).write_bytes(b"not an HDF5 file")
        elif kind == "incomplete":
            write_embedding_file(path, FIRST, complete=False)
        else:
            with h5py.File(path, "w") as handle:
                handle.create_dataset("unrelated", data=np.arange(3))
        return path


def make_viewer(**attributes):
    """A Viewer of NODES that loads embeddings through the real MainViewer.load_embedding."""
    viewer = SimpleNamespace(
        full_headers=list(NODES), sequences_map=dict(SEQUENCES), embedding=None,
        console_text=SimpleNamespace(text=""), canvas=SimpleNamespace(native="canvas"),
        _save_state=mock.Mock(), _refresh_session_card=mock.Mock(), position_history=[],
    )
    viewer.load_embedding = lambda: MainViewer.load_embedding(viewer)
    vars(viewer).update(attributes)
    with contextlib.redirect_stdout(io.StringIO()):
        viewer.load_embedding()
    return viewer


def run(viewer, *args):
    """Run the command quietly: (the text it succeeded with or None, the texts it failed with, what it printed)."""
    with reported_outcomes() as (succeeded, failed), contextlib.redirect_stdout(io.StringIO()) as output:
        embedding_command.run(viewer, list(args))
    return (
        succeeded.call_args.args[1] if succeeded.called else None,
        [call.args[1] for call in failed.call_args_list],
        output.getvalue(),
    )


class ArgumentFormTests(EmbeddingFixture, unittest.TestCase):
    def test_a_file_is_found_as_given_in_its_folder_and_without_its_extension(self):
        spaced = os.path.join(self.root, "My Files")
        os.mkdir(spaced)
        with_spaces = self.write("spaced.h5", SECOND, folder=spaced)
        inside = self.write("inside.h5", FIRST, folder=os.path.join(self.folder))
        elsewhere = self.write("elsewhere.h5", FIRST, folder=self.root)
        cases = (
            ("absolute", [self.first], self.first),
            ("name in the folder", ["inside.h5"], inside),
            ("name without the extension", ["inside"], inside),
            ("path without the extension", [self.second[:-3]], self.second),
            ("a quoted path with spaces", [f'"{with_spaces}"'], with_spaces),
            ("single quotes", [f"'{with_spaces}'"], with_spaces),
            ("a path the dispatcher split at its spaces", f'"{with_spaces}"'.split(), with_spaces),
            ("outside the folder", [elsewhere], elsewhere),
        )
        for label, args, expected in cases:
            with self.subTest(label):
                viewer = make_viewer()
                succeeded, failed, _ = run(viewer, *args)
                self.assertEqual(failed, [])
                self.assertIsNotNone(succeeded)
                self.assertEqual(cfg.EMBED_FILE, slashed(expected))
                self.assertEqual(viewer.embedding.embed_file, slashed(expected))
                self.assertIsNotNone(viewer.embedding.source)

    def test_a_relative_path_is_taken_from_the_working_directory_first(self):
        shown = os.path.join(self.root, "nearby")
        os.mkdir(shown)
        near = self.write("near.h5", FIRST, folder=shown)
        # The same name in the embedding folder loses to the file next to the working directory.
        self.write("near.h5", SECOND)
        previous = os.getcwd()
        os.chdir(shown)
        self.addCleanup(os.chdir, previous)
        for name in ("near.h5", "near"):
            with self.subTest(name=name):
                viewer = make_viewer()
                self.assertEqual(run(viewer, name)[1], [])
                self.assertEqual(cfg.EMBED_FILE, slashed(near))

    def test_a_name_that_is_no_file_fails_naming_where_it_looked_and_changes_nothing(self):
        directory = os.path.join(self.folder, "a_directory.h5")
        os.mkdir(directory)
        for name in ("missing.h5", "missing", "a_directory.h5", "a_directory"):
            with self.subTest(name=name):
                viewer = make_viewer()
                before = viewer.embedding
                succeeded, failed, _ = run(viewer, name)
                self.assertIsNone(succeeded)
                self.assertEqual(failed, [
                    f"Error: Embedding file '{name}' not found "
                    f"(checked absolute, relative, and {self.folder})."
                ])
                self.assertIs(viewer.embedding, before)
                self.assertEqual(cfg.EMBED_FILE, "")
                self.assertEqual(viewer.console_text.text, failed[0])

    def test_the_folder_defaults_to_the_embeddings_folder(self):
        del cfg.EMBED_DIR
        self.addCleanup(setattr, cfg, "EMBED_DIR", self.folder)
        failed = run(make_viewer(), "missing")[1]
        self.assertIn("checked absolute, relative, and Embeddings).", failed[0])

    def test_help_prints_its_text_and_loads_nothing(self):
        for word in ("help", "-h", "--help", "HELP", "Help"):
            with self.subTest(word=word):
                viewer = make_viewer()
                succeeded, failed, printed = run(viewer, word)
                self.assertEqual((succeeded, failed), ("Help information printed to the terminal.", []))
                self.assertEqual(viewer.console_text.text, "Help information printed to the terminal")
                for line in ("Embedding Switcher Tool", "embedding [filename.h5]", "embedding help",
                             "is not an undo step"):
                    self.assertIn(line, printed)
                self.assertIsNone(viewer.embedding.source)
                self.assertEqual(cfg.EMBED_FILE, "")
        # Only its first word asks for help: a file of that name is loaded.
        self.write("help.h5", FIRST)
        succeeded, failed, _ = run(make_viewer(), "help.h5")
        self.assertEqual(failed, [])
        self.assertIn("Loaded embedding 'help.h5'", succeeded)

    def test_a_switch_is_no_undo_step(self):
        viewer = make_viewer()
        run(viewer, self.first)
        run(viewer, self.second)
        viewer._save_state.assert_not_called()
        self.assertEqual(viewer.position_history, [])


class ReportTests(EmbeddingFixture, unittest.TestCase):
    def test_a_loaded_file_reports_its_coverage_model_width_and_saving_mode(self):
        viewer = make_viewer()
        succeeded, failed, printed = run(viewer, self.first)
        self.assertEqual(failed, [])
        record = (
            "Success: Loaded embedding 'first.h5' (2/4 network nodes embedded). "
            "Embedding model: test_model (width 3, float16)."
        )
        self.assertEqual(succeeded, record)
        self.assertIn("Attempting to load embedding: first.h5...", printed)
        self.assertIn("\nSuccess: Loaded embedding 'first.h5' (2/4 network nodes embedded).\n", printed)
        self.assertIn("\nEmbedding model: test_model (width 3, float16).\n", printed)
        self.assertEqual(
            viewer.console_text.text,
            "Loaded first.h5: 2/4 embedded. Embedding model: test_model (width 3, float16).",
        )
        # The warning of an incomplete coverage is the manager's, and stays.
        self.assertIn("WARNING: Incomplete embedding coverage", printed)
        self.assertIn("Embedded network nodes: 2/4 (50.0%)", printed)
        run(viewer, self.second)
        self.assertEqual(
            viewer.console_text.text,
            "Loaded second.h5: 3/4 embedded. Embedding model: model_b (width 4, float32).",
        )

    def test_full_coverage_has_no_warning_and_zero_coverage_is_accepted(self):
        full = self.write("full.h5", [(node, SEQUENCES[node]) for node in NODES])
        succeeded, failed, printed = run(make_viewer(), full)
        self.assertIn("(4/4 network nodes embedded)", succeeded)
        self.assertNotIn("WARNING", printed)
        other = self.write("other.h5", [("elsewhere", "AC")])
        viewer = make_viewer()
        succeeded, failed, printed = run(viewer, other)
        self.assertEqual(failed, [])
        self.assertIn("(0/4 network nodes embedded)", succeeded)
        self.assertIn("WARNING: Incomplete embedding coverage", printed)
        self.assertEqual(viewer.embedding.embed_file, slashed(other))

    def test_nodes_whose_sequence_differs_are_counted(self):
        viewer = make_viewer(sequences_map={**SEQUENCES, "node1": "WW", "node2": "WW"})
        succeeded, failed, printed = run(viewer, self.first)
        self.assertEqual(failed, [])
        self.assertEqual(succeeded, (
            "Success: Loaded embedding 'first.h5' (0/4 network nodes embedded). "
            "2 nodes not embedded: the file holds a different sequence. "
            "Embedding model: test_model (width 3, float16)."
        ))
        self.assertIn("2 nodes not embedded: the file holds a different sequence.", printed)
        self.assertEqual(
            viewer.console_text.text,
            "Loaded first.h5: 0/4 embedded. 2 nodes not embedded: the file holds a different sequence. "
            "Embedding model: test_model (width 3, float16).",
        )
        viewer = make_viewer(sequences_map={**SEQUENCES, "node2": "WW"})
        self.assertIn("1 node not embedded: the file holds", run(viewer, self.first)[0])

    def test_a_network_built_with_another_model_is_noted_and_accepted(self):
        note = (
            "NOTE: The network was built with esm2_t12_35M_UR50D, not test_model. The Viewer "
            "accepts this; embedding commands use the embedding's model."
        )
        self.write_network("esm2_t12_35M_UR50D")
        viewer = make_viewer()
        succeeded, failed, printed = run(viewer, self.first)
        self.assertEqual(failed, [])
        self.assertTrue(succeeded.endswith(" " + note), succeeded)
        self.assertIn("\n" + note + "\n", printed)
        self.assertTrue(viewer.console_text.text.endswith(" " + note), viewer.console_text.text)
        self.assertEqual(viewer.embedding.model_name, "test_model")  # Accepted: the embedding's model is used.

    def test_no_note_without_a_different_model(self):
        # The same model, a BLAST network (which has no model), no network, and one that cannot be read.
        for label, setup in (
            ("same model", lambda: self.write_network("test_model")),
            ("BLAST network", lambda: self.write_network("BLAST")),
            ("no network", lambda: setattr(cfg, "INPUT_HDF5", "")),
            ("a network file that is gone", lambda: setattr(cfg, "INPUT_HDF5", os.path.join(self.root, "gone.h5"))),
            ("a file that is no network", lambda: Path(os.path.join(self.root, "x.h5")).write_bytes(b"x")
             or setattr(cfg, "INPUT_HDF5", os.path.join(self.root, "x.h5"))),
        ):
            with self.subTest(label):
                setup()
                succeeded, failed, printed = run(make_viewer(), self.first)
                self.assertEqual(failed, [])
                self.assertNotIn("NOTE", succeeded + printed)

    def test_the_console_line_is_translated_and_the_terminal_and_record_stay_english(self):
        previous = Localization.set_translator(lambda template, n: f"<{template}>")
        self.addCleanup(Localization.set_translator, previous)
        self.write_network("esm2_t12_35M_UR50D")
        viewer = make_viewer(sequences_map={**SEQUENCES, "node2": "WW"})
        succeeded, failed, printed = run(viewer, self.first)
        self.assertEqual(failed, [])
        self.assertEqual(succeeded, (
            "Success: Loaded embedding 'first.h5' (1/4 network nodes embedded). "
            "1 node not embedded: the file holds a different sequence. "
            "Embedding model: test_model (width 3, float16). "
            "NOTE: The network was built with esm2_t12_35M_UR50D, not test_model. The Viewer "
            "accepts this; embedding commands use the embedding's model."
        ))
        self.assertIn("\nEmbedding model: test_model (width 3, float16).\n", printed)
        self.assertEqual(viewer.console_text.text, (
            "<Loaded first.h5: 1/4 embedded.> "
            "<%n node(s) not embedded: the file holds a different sequence.> "
            "<Embedding model: test_model (width 3, float16).> "
            "<NOTE: The network was built with esm2_t12_35M_UR50D, not test_model. The Viewer "
            "accepts this; embedding commands use the embedding's model.>"
        ))
        with contextlib.redirect_stdout(io.StringIO()):
            Command_Engine.show_status(viewer, Localization.Message("Loading {file}...", file="first.h5"))
        self.assertEqual(viewer.console_text.text, "<Loading first.h5...>")
        # An error the command reports keeps its English in the record and translates on the console line.
        failed = run(make_viewer(), "missing")[1]
        self.assertEqual(
            failed, [f"Error: Embedding file 'missing' not found (checked absolute, relative, and {self.folder})."]
        )

    def test_a_file_without_sequences_has_no_width_to_report(self):
        empty = self.write("empty.h5", [])
        succeeded, failed, _ = run(make_viewer(), empty)
        self.assertEqual(failed, [])
        self.assertTrue(succeeded.endswith("Embedding model: test_model (float16)."), succeeded)

    def test_choosing_the_same_file_again_reads_it_afresh(self):
        viewer = make_viewer()
        run(viewer, self.first)
        before = viewer.embedding
        # Rewritten with another model: the switch sees the new file, not the manifest read before.
        self.write("first.h5", FIRST + [("node3", "AE")], model_name="rewritten", width=5)
        succeeded, failed, _ = run(viewer, self.first)
        self.assertEqual(failed, [])
        self.assertIsNot(viewer.embedding, before)
        self.assertEqual((viewer.embedding.model_name, viewer.embedding.width), ("rewritten", 5))
        self.assertIn("(3/4 network nodes embedded)", succeeded)


class RollbackTests(EmbeddingFixture, unittest.TestCase):
    def switched(self):
        """A viewer already holding the first embedding."""
        viewer = make_viewer()
        run(viewer, self.first)
        return viewer

    def test_a_file_the_loader_refuses_restores_the_embedding_and_its_path(self):
        cases = (
            ("text", "Embedding file rejected: "),
            ("incomplete", "Embedding file rejected: "),
            ("empty", "Embedding file rejected: "),
        )
        for kind, reason in cases:
            with self.subTest(kind):
                viewer = self.switched()
                previous = viewer.embedding
                path = self.write_unusable(f"{kind}.h5", kind)
                succeeded, failed, printed = run(viewer, path)
                self.assertIsNone(succeeded)
                self.assertIs(viewer.embedding, previous)
                self.assertEqual(cfg.EMBED_FILE, slashed(self.first))
                # One report of the failure, with the loader's reason; the terminal says what it did.
                self.assertEqual(len(failed), 1, failed)
                self.assertTrue(failed[0].startswith(f"Failed to load embedding '{kind}.h5': {reason}"), failed[0])
                self.assertTrue(failed[0].endswith("\nReverted to previous embedding."), failed[0])
                self.assertIn(f"Failed to load embedding '{kind}.h5': {reason}", printed)
                self.assertIn("Reverting to previous embedding state...", printed)
                self.assertTrue(viewer.console_text.text.startswith(
                    "Load failed. Reverted to previous embedding. " + reason), viewer.console_text.text)
                # The previous embedding still reads.
                np.testing.assert_array_equal(
                    viewer.embedding.pooled([0]), previous.pooled([0]),
                )
                self.assertEqual(viewer.embedding.viewer_to_emb.tolist(), [0, 1, -1, -1])

    def test_the_loaders_reasons_are_the_ones_it_gives(self):
        viewer = self.switched()
        incomplete = self.write_unusable("incomplete.h5", "incomplete")
        failed = run(viewer, incomplete)[1][0]
        self.assertIn("incomplete", failed.lower())
        failed = run(viewer, self.write_unusable("text.h5", "text"))[1][0]
        self.assertIn("file signature not found", failed)

    def test_a_refused_file_without_a_previous_embedding_leaves_none(self):
        viewer = make_viewer()
        previous = viewer.embedding
        succeeded, failed, _ = run(viewer, self.write_unusable("text.h5", "text"))
        self.assertEqual(len(failed), 1)
        self.assertIs(viewer.embedding, previous)
        self.assertIsNone(viewer.embedding.source)
        self.assertEqual(cfg.EMBED_FILE, "")

    def test_the_reason_on_the_console_line_is_translated(self):
        previous = Localization.set_translator(lambda template, n: f"<{template}>")
        self.addCleanup(Localization.set_translator, previous)
        viewer = make_viewer()
        run(viewer, self.write_unusable("incomplete.h5", "incomplete"))
        self.assertTrue(viewer.console_text.text.startswith(
            "<Load failed. Reverted to previous embedding.> <Embedding file rejected: "), viewer.console_text.text)

    def test_a_loader_that_raises_is_rolled_back_too(self):
        viewer = self.switched()
        previous = viewer.embedding

        def fails():
            viewer.embedding = None  # Half done: the manager was replaced before the error.
            raise RuntimeError("out of memory")

        viewer.load_embedding = fails
        succeeded, failed, printed = run(viewer, self.second)
        self.assertIsNone(succeeded)
        self.assertIs(viewer.embedding, previous)
        self.assertEqual(cfg.EMBED_FILE, slashed(self.first))
        self.assertEqual(failed, ["Failed to load embedding 'second.h5': out of memory\nReverted to previous embedding."])
        self.assertEqual(viewer.console_text.text, "Load failed. Reverted to previous embedding.")

    def test_a_viewer_left_without_a_manager_is_rolled_back_too(self):
        viewer = self.switched()
        previous = viewer.embedding
        viewer.load_embedding = lambda: setattr(viewer, "embedding", None)
        failed = run(viewer, self.second)[1]
        self.assertIs(viewer.embedding, previous)
        self.assertIn("Embedding loader failed to return an embedding.", failed[0])

    def test_the_session_card_is_refreshed_after_a_switch_and_after_a_rollback(self):
        viewer = make_viewer()
        run(viewer, self.first)
        self.assertEqual(viewer._refresh_session_card.call_count, 1)
        run(viewer, self.write_unusable("text.h5", "text"))
        self.assertEqual(viewer._refresh_session_card.call_count, 2)


class DialogTests(EmbeddingFixture, unittest.TestCase):
    def choose(self, result):
        return mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(result, ""))

    def test_no_argument_opens_a_dialog_in_the_embedding_folder_and_loads_the_choice(self):
        viewer = make_viewer()
        with self.choose(self.first) as dialog:
            succeeded, failed, _ = run(viewer)
        self.assertEqual(failed, [])
        self.assertIn("Loaded embedding 'first.h5'", succeeded)
        self.assertEqual(cfg.EMBED_FILE, self.first.replace("\\", "/"))
        parent, title, folder, filters = dialog.call_args.args
        self.assertEqual((parent, title, folder), ("canvas", "Select Embedding File", self.folder))
        self.assertEqual(filters, "Embedding Files (*.h5);;All Files (*)")

    def test_a_folder_that_is_not_there_is_not_offered(self):
        shutil.rmtree(self.folder)
        with self.choose("") as dialog:
            run(make_viewer())
        self.assertEqual(dialog.call_args.args[2], "")

    def test_the_dialog_is_translated(self):
        translator = mock.Mock(side_effect=lambda context, text, *rest: f"<{context}:{text}>")
        with mock.patch.object(embedding_command, "translate", translator), self.choose("") as dialog:
            run(make_viewer())
        _, title, _, filters = dialog.call_args.args
        self.assertEqual(title, "<Viewer:Select Embedding File>")
        self.assertEqual(filters, "<Viewer:Embedding Files (*.h5)>;;<Viewer:All Files (*)>")

    def test_cancelling_the_dialog_is_a_cancellation_and_changes_nothing(self):
        viewer = self.switched_viewer()
        previous = viewer.embedding
        with self.choose(""), mock.patch.object(Command_Engine, "command_cancelled") as cancelled, \
                reported_outcomes() as (succeeded, failed), contextlib.redirect_stdout(io.StringIO()):
            embedding_command.run(viewer, [])
        cancelled.assert_called_once()
        self.assertEqual(str(cancelled.call_args.args[1]), "Embedding selection cancelled.")
        succeeded.assert_not_called()
        failed.assert_not_called()
        self.assertIs(viewer.embedding, previous)
        self.assertEqual(cfg.EMBED_FILE, slashed(self.first))
        self.assertEqual(viewer.console_text.text, "Embedding selection cancelled.")

    def switched_viewer(self):
        viewer = make_viewer()
        run(viewer, self.first)
        return viewer

    def test_a_choice_that_is_gone_fails(self):
        gone = os.path.join(self.folder, "gone.h5")
        with self.choose(gone):
            succeeded, failed, _ = run(make_viewer())
        self.assertIsNone(succeeded)
        self.assertEqual(failed, [f"Error: File '{gone}' does not exist."])

    def test_a_dialog_that_cannot_open_fails(self):
        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", side_effect=RuntimeError("no display")):
            succeeded, failed, _ = run(make_viewer())
        self.assertEqual((succeeded, failed), (None, ["Error opening file dialog: no display"]))

    def test_a_chosen_file_the_loader_refuses_is_rolled_back(self):
        viewer = self.switched_viewer()
        previous = viewer.embedding
        with self.choose(self.write_unusable("text.h5", "text")):
            succeeded, failed, _ = run(viewer)
        self.assertIsNone(succeeded)
        self.assertIs(viewer.embedding, previous)
        self.assertEqual(len(failed), 1)


class PortalTests(EmbeddingFixture, PortalFixture, unittest.TestCase):
    """The command through the real command portal, as MCP clients and the agent page run it."""

    def setUp(self):
        super().setUp()
        viewer = self.viewer
        viewer.full_headers = list(NODES)
        viewer.n_nodes = len(NODES)
        viewer.visible_mask = np.ones(len(NODES), bool)
        viewer.group_labels = [set() for _ in NODES]
        viewer.cluster_labels = np.zeros(len(NODES), int)
        viewer.sequences_map = dict(SEQUENCES)
        viewer.canvas = SimpleNamespace(native=None)
        viewer.embedding = None
        viewer.load_embedding = lambda: MainViewer.load_embedding(viewer)
        viewer._refresh_session_card = mock.Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            viewer.load_embedding()

    def submit(self, command, expected="succeeded"):
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.finish(self.portal.submit(str(len(self.portal.requests)), command)["request_id"])
        self.assertEqual(result["status"], expected, result)
        return result["commands"][0]

    def test_a_file_loads_with_one_summary_and_help_reports_success(self):
        record = self.submit(f'embedding "{self.first}"')
        summaries = [m["text"] for m in record["messages"] if m["status"] == "succeeded"]
        self.assertEqual(len(summaries), 1, record)
        self.assertEqual(summaries[0], (
            "Success: Loaded embedding 'first.h5' (2/4 network nodes embedded). "
            "Embedding model: test_model (width 3, float16)."
        ))
        self.assertEqual(self.viewer.embedding.embed_file, slashed(self.first))
        record = self.submit("embedding help")
        self.assertEqual([m["text"] for m in record["messages"] if m["status"] == "succeeded"],
                         ["Help information printed to the terminal."])

    def test_a_failure_is_reported_once_with_the_reason_and_nothing_succeeds(self):
        # The dialog must never be reached here; were it, a real one would block this test forever.
        unreachable = mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=("", ""))
        unreachable.start()
        self.addCleanup(unreachable.stop)
        for command, text in (
            ("embedding", "Error opening file dialog"),
            ("embedding missing", "Error: Embedding file 'missing' not found"),
            (f"embedding {self.write_unusable('text.h5', 'text')}", "Failed to load embedding 'text.h5'"),
        ):
            with self.subTest(command=command):
                record = self.submit(command, "failed")
                self.assertEqual([m["status"] for m in record["messages"] if m["status"]], ["failed"], record)
                self.assertTrue(any(text in m["text"] for m in record["messages"]), record)

    def test_headless_the_dialog_fails_before_it_opens(self):
        with mock.patch.dict(os.environ, {"SSN_VIEWER_HEADLESS": "1"}), \
                mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName") as dialog:
            record = self.submit("embedding", "failed")
        dialog.assert_not_called()
        self.assertIn("requires a visible Viewer and user input", str(record["messages"]))

    def test_a_visible_viewer_waits_for_the_dialog_then_loads_the_choice(self):
        seen = []

        def choose(*args):
            request_id = list(self.portal.requests)[-1]
            seen.append(self.portal.get(request_id)["status"])
            return self.first, ""

        with mock.patch.dict(os.environ, {"SSN_VIEWER_HEADLESS": "0", "QT_QPA_PLATFORM": ""}), \
                mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", side_effect=choose):
            record = self.submit("embedding")
        self.assertEqual(seen, ["awaiting_user_input"])
        self.assertIn("Select embedding file in the Viewer dialog", str(record["messages"]))
        self.assertEqual(self.viewer.embedding.embed_file, slashed(self.first))

    def test_a_visible_viewer_reports_a_cancelled_dialog(self):
        with mock.patch.dict(os.environ, {"SSN_VIEWER_HEADLESS": "0", "QT_QPA_PLATFORM": ""}), \
                mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=("", "")):
            record = self.submit("embedding", "cancelled")
        self.assertEqual([m["status"] for m in record["messages"] if m["status"]], ["cancelled"])
        self.assertIsNone(self.viewer.embedding.source)


class FollowTheSwitchTests(EmbeddingFixture, unittest.TestCase):
    """What shows the active embedding shows the new one after a switch, and the old one after a rollback."""

    def test_require_embedding_and_the_mapping_follow_each_switch(self):
        viewer = make_viewer()
        # Before any file: nothing to require, and every node unmapped.
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(Command_Engine, "command_failed") as failed:
            self.assertIsNone(Command_Engine.require_embedding(viewer))
        self.assertIn("No embedding loaded in the viewer", str(failed.call_args.args[1]))
        self.assertEqual(Command_Engine.get_embedding_mapping(viewer)[0].tolist(), [-1, -1, -1, -1])

        run(viewer, self.first)
        self.assertIs(Command_Engine.require_embedding(viewer), viewer.embedding)
        mapping, embedded = Command_Engine.get_embedding_mapping(viewer)
        self.assertEqual((mapping.tolist(), embedded.tolist()), ([0, 1, -1, -1], [0, 1]))
        first_vectors = viewer.embedding.pooled([1])

        # Another file covering other nodes: the mapping, the vectors and the model are the new file's.
        run(viewer, self.second)
        self.assertIs(Command_Engine.require_embedding(viewer), viewer.embedding)
        mapping, embedded = Command_Engine.get_embedding_mapping(viewer)
        self.assertEqual((mapping.tolist(), embedded.tolist()), ([-1, 0, 1, 2], [1, 2, 3]))
        self.assertEqual((viewer.embedding.model_name, viewer.embedding.width), ("model_b", 4))
        second_vectors = viewer.embedding.pooled([1])
        self.assertEqual(second_vectors.shape, (1, 4))
        self.assertNotEqual(first_vectors.shape, second_vectors.shape)
        np.testing.assert_array_equal(viewer.embedding.residues(1), write_residues(self.second, "node2"))

        # A file the loader refuses leaves the second one in place.
        run(viewer, self.write_unusable("text.h5", "text"))
        self.assertEqual(Command_Engine.get_embedding_mapping(viewer)[0].tolist(), [-1, 0, 1, 2])

    def test_a_switch_to_a_file_covering_no_node_makes_commands_refuse(self):
        viewer = make_viewer()
        run(viewer, self.first)
        other = self.write("other.h5", [("elsewhere", "AC")])
        self.assertEqual(run(viewer, other)[1], [])
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(Command_Engine, "command_failed") as failed:
            self.assertIsNone(Command_Engine.require_embedding(viewer))
        self.assertIn("covers no node of the current network", str(failed.call_args.args[1]))

    def test_inspection_reports_the_new_embedding(self):
        viewer = make_viewer(
            n_nodes=4, visible_mask=np.ones(4, bool), selected_indices=[], cluster_labels=None,
            group_labels=[set() for _ in NODES], metadata={}, edges=[], current_slider_threshold=0.5,
        )
        service = ViewerInspectionService(viewer, cfg)

        def overview():
            return embedding_overview(viewer, 4)

        self.assertEqual((overview()["available"], overview()["file"]), (False, None))
        run(viewer, self.first)
        self.assertEqual(overview(), {
            "available": True, "file": slashed(self.first), "model": "test_model", "dimension": 3,
            "saving_mode": "float16", "mapped_nodes": 2, "unmapped_nodes": 2, "sequence_mismatches": 0,
            "load_failure": None,
        })
        # A snapshot is frozen as taken; the next one is the new state.
        before = service.snapshots.execute("get_summary", service.capture_snapshot())
        run(viewer, self.second)
        after = service.snapshots.execute("get_summary", service.capture_snapshot())
        self.assertEqual(before["embedding"]["file"], slashed(self.first))
        self.assertEqual(before["inputs"]["embedding"], slashed(self.first))
        self.assertEqual(after["embedding"], {
            "available": True, "file": slashed(self.second), "model": "model_b", "dimension": 4,
            "saving_mode": "float32", "mapped_nodes": 3, "unmapped_nodes": 1, "sequence_mismatches": 0,
            "load_failure": None,
        })
        self.assertEqual(after["inputs"]["embedding"], slashed(self.second))
        self.assertIn("embedding_input_v1", after["capabilities"])
        # The settings the Viewer reports to agents, and the info card reads.
        self.assertEqual(service.get_summary()["inputs"]["embedding"], slashed(self.second))
        self.assertEqual(service.session_overview()["inputs"]["embedding"], slashed(self.second))
        # A refused file leaves the report as it was.
        run(viewer, self.write_unusable("text.h5", "text"))
        self.assertEqual(service.get_summary()["inputs"]["embedding"], slashed(self.second))
        self.assertEqual(overview()["file"], slashed(self.second))
        self.assertIsNone(overview()["load_failure"])


def write_residues(path, header):
    """The matrix a fixture file holds for HEADER, read back as float32."""
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["embeddings"][header][()], dtype=np.float32)


class InfoCardTests(EmbeddingFixture, unittest.TestCase):
    """The Viewer's info card, whose Embedding row _refresh_session_card keeps current."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make_card_viewer(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.canvas = SimpleNamespace(native=QWidget(), size=(1200, 800))
        self.addCleanup(viewer.canvas.native.deleteLater)
        viewer.hud_layout = {"status_x_offset": 10.0, "status_line_spacing": 38.0}
        viewer._hud_timer = mock.Mock()
        viewer.min_threshold, viewer.max_threshold = 0.0, 10.0
        viewer.current_slider_threshold = 1.0
        viewer.update_edges = mock.Mock()
        viewer.n_nodes = len(NODES)
        viewer.visible_mask = np.ones(len(NODES), bool)
        viewer.edges = np.zeros((0, 2), dtype=np.int32)
        viewer.full_headers = list(NODES)
        viewer.sequences_map = dict(SEQUENCES)
        viewer.embedding = None
        viewer.console_text = SimpleNamespace(text="")
        viewer._build_slider_overlay()
        viewer.canvas.native.resize(1200, 800)
        viewer.canvas.native.show()
        with contextlib.redirect_stdout(io.StringIO()):
            viewer.load_embedding()
        self.app.processEvents()
        return viewer

    def row(self, card):
        """The Embedding row: the path it shows, and its tooltip, the full path."""
        return card.path_fields["embedding"].text(), card.path_fields["embedding"].toolTip()

    @staticmethod
    def shows(path):
        """What the row holds for a file in the embedding folder: its name under $embedding$, and the full path."""
        return "$embedding$/" + os.path.basename(path), display_path(path.replace("\\", "/"))

    def test_the_embedding_row_shows_the_new_file_after_a_switch_and_keeps_the_old_one_after_a_rollback(self):
        viewer = self.make_card_viewer()
        card = viewer.controls_button.toggle_card()
        self.app.processEvents()
        self.assertEqual(self.row(card), ("", ""))

        self.assertEqual(run(viewer, self.first)[1], [])
        self.assertEqual(self.row(card), self.shows(self.first))

        self.assertEqual(run(viewer, self.second)[1], [])
        self.assertEqual(self.row(card), self.shows(self.second))

        # A refused file leaves the row on the file in use, and the card too says which.
        run(viewer, self.write_unusable("text.h5", "text"))
        self.assertEqual(self.row(card), self.shows(self.second))

    def test_the_pick_button_runs_embedding_through_the_viewers_command_path_and_the_row_follows(self):
        viewer = self.make_card_viewer()
        viewer.command_history = []
        viewer.history_file = os.path.join(self.root, "history.txt")
        viewer.broadcast_metadata_state = lambda: None
        viewer.update_console_background = lambda: None
        card = viewer.controls_button.toggle_card()
        self.app.processEvents()
        self.assertEqual(self.row(card), ("", ""))

        def pick(chosen):
            with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(chosen, "")) as dialog, \
                    reported_outcomes() as (succeeded, failed), contextlib.redirect_stdout(io.StringIO()):
                QTest.mouseClick(card.pick_buttons["embedding"], Qt.MouseButton.LeftButton)
            self.app.processEvents()
            dialog.assert_called_once()
            return succeeded, failed

        # The button typed "embedding" alone: its dialog opened, the file loaded, the row shows it.
        succeeded, failed = pick(self.first)
        self.assertTrue(succeeded.called)
        self.assertFalse(failed.called)
        self.assertEqual(viewer.command_history, ["embedding"])
        self.assertEqual(self.row(card), self.shows(self.first))
        # Another file replaces it; a refused one, and a cancelled dialog, leave the row on the file in use.
        pick(self.second)
        self.assertEqual(self.row(card), self.shows(self.second))
        succeeded, failed = pick(self.write_unusable("text.h5", "text"))
        self.assertTrue(failed.called)
        self.assertEqual(self.row(card), self.shows(self.second))
        pick("")
        self.assertEqual(self.row(card), self.shows(self.second))
        # The same command typed twice in a row is in the history once, as for any typed command.
        self.assertEqual(viewer.command_history, ["embedding"])

    def test_a_card_opened_after_a_switch_shows_it_too(self):
        viewer = self.make_card_viewer()
        run(viewer, self.first)
        card = viewer.controls_button.toggle_card()
        self.app.processEvents()
        self.assertEqual(self.row(card), self.shows(self.first))

    def test_the_command_costs_a_closed_card_nothing(self):
        viewer = self.make_card_viewer()
        with mock.patch.object(viewer.controls_button, "session", side_effect=AssertionError("read")):
            self.assertEqual(run(viewer, self.first)[1], [])


class CatalogTests(unittest.TestCase):
    def test_the_catalog_lists_the_command_like_alignment(self):
        entries = {entry["command"]: entry for entry in command_catalog()["commands"]}
        entry, alignment = entries["embedding"], entries["alignment"]
        self.assertEqual(entry["syntax"], ["embedding", "embedding [filename.h5]", "embedding help"])
        self.assertTrue(entry["opens_interface"])
        self.assertFalse(entry["writes_files"])
        self.assertFalse(entry["background_work"])
        self.assertEqual(entry["requires"], [])
        self.assertEqual(entry["agent_policy"], alignment["agent_policy"])
        self.assertEqual([argument["name"] for argument in entry["arguments"]], ["filename", "help"])
        self.assertEqual(
            entry["arguments"][-1]["choices"], [{"value": "help", "aliases": ["-h", "--help"]}]
        )
        detailed = command_catalog("embedding")["commands"][0]
        self.assertIn("Embedding Switcher Tool", detailed["help"])

    def test_the_agent_page_needs_approval_to_run_it(self):
        # It reads a file and may open a dialog, as alignment does.
        for command in ("embedding", "embedding first.h5", "EMBEDDING help"):
            with self.subTest(command=command):
                self.assertEqual(agent_policy(command), "approve")


if __name__ == "__main__":
    unittest.main()
