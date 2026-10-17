# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's optional embedding: reading the manifest, matching nodes, reading rows.

Embedding_Manager reads only the file's manifest when it loads and matches it
to the network's nodes by exact header and identical sequence; per-residue
rows and pooled vectors are read when asked for.
"""

import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Embedding_Manager
from tests.embedding_fixtures import residue_values, write_embedding_file


RECORDS = [("node_a", "ACDE"), ("node_b", "FGH"), ("node_c", "KLMNP")]
SEQUENCES = dict(RECORDS)


def load(path, headers, sequences=SEQUENCES, reuse=None):
    with redirect_stdout(io.StringIO()) as output:
        manager = Embedding_Manager.Embedding_Manager(
            path, full_headers=headers, sequences=sequences, reuse=reuse
        )
    return manager, output.getvalue()


class EmbeddingFileTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "set_[test_model]_embeddings.h5")
        write_embedding_file(self.path, RECORDS, width=4)


class NoEmbeddingTests(unittest.TestCase):
    def test_an_empty_path_loads_nothing_and_says_so(self):
        for path in ("", None, "  ", "None"):
            manager, output = load(path, ["node_a"])
            self.assertIsNone(manager.source)
            self.assertIsNone(manager.load_failure)
            self.assertEqual(manager.viewer_to_emb.tolist(), [-1])
            self.assertIn("An embedding file is not selected", output)
            with self.assertRaises(ValueError):
                manager.residues(0)


class ManifestTests(EmbeddingFileTestCase):
    def test_summary_reads_the_manifest_and_width_only(self):
        manifest, width = Embedding_Manager.read_embedding_summary(self.path)
        self.assertEqual(manifest.headers, ["node_a", "node_b", "node_c"])
        self.assertEqual(manifest.sequences, ["ACDE", "FGH", "KLMNP"])
        self.assertEqual(manifest.model_name, "test_model")
        self.assertEqual(manifest.saving_mode, "float16")
        self.assertEqual(width, 4)

    def test_an_incomplete_file_is_rejected(self):
        incomplete = os.path.join(self.directory.name, "incomplete.h5")
        write_embedding_file(incomplete, RECORDS, complete=False)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            Embedding_Manager.read_embedding_summary(incomplete)
        manager, output = load(incomplete, ["node_a"])
        self.assertIsNone(manager.source)
        self.assertIn("incomplete", str(manager.load_failure))
        self.assertIn("Embedding file rejected", output)

    def test_a_file_without_the_required_attributes_is_rejected(self):
        legacy = os.path.join(self.directory.name, "legacy.h5")
        with h5py.File(legacy, "w") as handle:
            handle.create_dataset("node_a", data=np.zeros((4, 2), dtype=np.float32))
        manager, _ = load(legacy, ["node_a"])
        self.assertIsNone(manager.source)
        self.assertIn("missing required attribute", str(manager.load_failure))

    def test_a_file_that_is_not_hdf5_is_rejected(self):
        text = os.path.join(self.directory.name, "not_hdf5.h5")
        with open(text, "w", encoding="utf-8") as handle:
            handle.write(">node_a\nACDE\n")
        manager, _ = load(text, ["node_a"])
        self.assertIsNone(manager.source)
        self.assertIsNotNone(manager.load_failure)

    def test_a_missing_file_is_rejected(self):
        manager, _ = load(os.path.join(self.directory.name, "absent.h5"), ["node_a"])
        self.assertIsNone(manager.source)
        self.assertIsNotNone(manager.load_failure)


class CoverageTests(EmbeddingFileTestCase):
    def test_full_coverage_maps_nodes_in_network_order(self):
        headers = ["node_c", "node_a", "node_b"]
        manager, output = load(self.path, headers)
        self.assertEqual(manager.viewer_to_emb.tolist(), [2, 0, 1])
        self.assertTrue(manager.embedded_node_mask.all())
        self.assertEqual(manager.model_name, "test_model")
        self.assertEqual(manager.width, 4)
        self.assertEqual(manager.sequence_count, 3)
        self.assertEqual(manager.source.node_headers, ("node_c", "node_a", "node_b"))
        self.assertIn("3/3 network nodes embedded", output)
        self.assertNotIn("WARNING", output)

    def test_partial_coverage_is_accepted_with_a_warning(self):
        manager, output = load(self.path, ["node_a", "absent_node"])
        self.assertEqual(manager.viewer_to_emb.tolist(), [0, -1])
        self.assertEqual(manager.missing_headers, ["absent_node"])
        self.assertIsNotNone(manager.source)
        self.assertIn("WARNING: Incomplete embedding coverage", output)
        self.assertIn("Embedded network nodes: 1/2 (50.0%)", output)
        self.assertIn("  - absent_node", output)

    def test_zero_coverage_is_accepted_with_a_warning(self):
        manager, output = load(self.path, ["other_1", "other_2"])
        self.assertIsNotNone(manager.source)
        self.assertFalse(manager.embedded_node_mask.any())
        self.assertIn("Embedded network nodes: 0/2 (0.0%)", output)

    def test_a_node_whose_sequence_differs_has_no_embedding(self):
        # An embedding of full-length sequences used with a cropped FASTA:
        # the rows would not line up with the node's residues.
        cropped = {**SEQUENCES, "node_b": "GH"}
        manager, output = load(self.path, ["node_a", "node_b"], sequences=cropped)
        self.assertEqual(manager.viewer_to_emb.tolist(), [0, -1])
        self.assertEqual(manager.mismatched_headers, ["node_b"])
        self.assertEqual(manager.missing_headers, [])
        self.assertIn("stored sequence differs from the FASTA", output)
        with self.assertRaisesRegex(ValueError, "no embedding"):
            manager.residues(1)

    def test_a_sequence_known_only_by_first_word_is_still_checked(self):
        # The Viewer's sequences_map also indexes each header's first word.
        path = os.path.join(self.directory.name, "spaced.h5")
        write_embedding_file(path, [("node_x description", "ACD")])
        manager, _ = load(path, ["node_x description"], sequences={"node_x": "ACE"})
        self.assertEqual(manager.mismatched_headers, ["node_x description"])
        manager, _ = load(path, ["node_x description"], sequences={"node_x": "ACD"})
        self.assertEqual(manager.matched_headers, ["node_x description"])

    def test_a_node_without_a_known_sequence_is_matched_by_header(self):
        manager, _ = load(self.path, ["node_a"], sequences={})
        self.assertEqual(manager.viewer_to_emb.tolist(), [0])

    def test_order_comes_from_headers_not_from_the_group(self):
        # HDF5 lists group members alphabetically; /headers keeps FASTA order.
        records = [("zeta", "AC"), ("alpha", "DEF")]
        path = os.path.join(self.directory.name, "ordered.h5")
        write_embedding_file(path, records, width=2)
        manager, _ = load(path, ["alpha", "zeta"], sequences=dict(records))
        self.assertEqual(manager.viewer_to_emb.tolist(), [1, 0])
        np.testing.assert_array_equal(manager.residues(0), residue_values(1, 3, 2))


class ReadTests(EmbeddingFileTestCase):
    def test_residues_are_float32_rows_of_the_nodes_matrix(self):
        manager, _ = load(self.path, ["node_b", "node_c"])
        rows = manager.residues(1)
        self.assertEqual(rows.dtype, np.float32)
        self.assertEqual(rows.shape, (5, 4))
        np.testing.assert_array_equal(rows, residue_values(2, 5, 4))

    def test_pooled_vectors_are_the_mean_or_maximum_over_residues(self):
        manager, _ = load(self.path, ["node_a", "node_b", "node_c"])
        for method, reduce in (("mean", np.mean), ("max", np.max)):
            vectors = manager.pooled([2, 0], method)
            self.assertEqual(vectors.dtype, np.float32)
            expected = np.stack([
                reduce(residue_values(2, 5, 4), axis=0),
                reduce(residue_values(0, 4, 4), axis=0),
            ])
            np.testing.assert_allclose(vectors, expected, rtol=0, atol=1e-6)

    def test_pooled_vectors_are_kept_once_read(self):
        manager, _ = load(self.path, ["node_a", "node_b"])
        first = manager.pooled([0, 1, 0])
        self.assertEqual(sorted(manager._pooled["mean"]), [0, 1])
        os.remove(self.path)  # nothing left to read from
        np.testing.assert_array_equal(manager.pooled([1, 0]), first[[1, 0]])

    def test_an_unknown_pooling_method_is_refused(self):
        manager, _ = load(self.path, ["node_a"])
        with self.assertRaisesRegex(ValueError, "Unknown pooling method 'median'"):
            manager.pooled([0], "median")

    def test_a_float32_file_reads_the_same_values(self):
        path = os.path.join(self.directory.name, "float32.h5")
        write_embedding_file(path, RECORDS, saving_mode="float32", width=4)
        manager, _ = load(path, ["node_a"])
        self.assertEqual(manager.saving_mode, "float32")
        np.testing.assert_array_equal(manager.residues(0), residue_values(0, 4, 4))

    def test_the_source_reads_on_its_own_in_one_open_file(self):
        manager, _ = load(self.path, ["node_a", "node_c"])
        source = manager.source
        with source.open() as handle:
            vectors = source.pooled([0, 1], "max", handle)
            rows = source.residues(1, handle)
        np.testing.assert_array_equal(rows, residue_values(2, 5, 4))
        np.testing.assert_array_equal(vectors[1], residue_values(2, 5, 4).max(axis=0))

    def test_a_file_changed_after_loading_is_not_read(self):
        manager, _ = load(self.path, ["node_a"])
        stamp = time.time() + 10
        os.utime(self.path, (stamp, stamp))
        with self.assertRaisesRegex(ValueError, "changed after the Viewer loaded it"):
            manager.residues(0)


class ReuseTests(EmbeddingFileTestCase):
    def test_an_unchanged_file_reuses_its_manifest(self):
        first, _ = load(self.path, ["node_a"])
        second, output = load(self.path, ["node_b", "node_a"], reuse=first)
        self.assertIn("Reusing the loaded embedding manifest", output)
        self.assertIs(second._manifest, first._manifest)
        self.assertEqual(second.viewer_to_emb.tolist(), [1, 0])

    def test_a_changed_file_is_read_again(self):
        first, _ = load(self.path, ["node_a"])
        stamp = time.time() + 10
        os.utime(self.path, (stamp, stamp))
        second, output = load(self.path, ["node_a"], reuse=first)
        self.assertNotIn("Reusing", output)
        self.assertIsNot(second._manifest, first._manifest)


class ViewerLoadingTests(EmbeddingFileTestCase):
    """MainViewer.load_embedding matches cfg.EMBED_FILE to the Viewer's nodes."""

    def viewer(self):
        return SimpleNamespace(
            full_headers=["node_b", "node_a"],
            sequences_map={"node_a": "ACDE", "node_b": "XXX"},
        )

    def test_the_viewer_loads_the_configured_file_against_its_nodes(self):
        from EMAPSSN_Viewer import MainViewer
        import EMAPSSN_Config as cfg

        viewer = self.viewer()
        with mock.patch.object(cfg, "EMBED_FILE", self.path, create=True), \
                redirect_stdout(io.StringIO()):
            MainViewer.load_embedding(viewer)
        self.assertEqual(viewer.embedding.viewer_to_emb.tolist(), [-1, 0])
        self.assertEqual(viewer.embedding.mismatched_headers, ["node_b"])
        first = viewer.embedding
        with mock.patch.object(cfg, "EMBED_FILE", self.path, create=True), \
                redirect_stdout(io.StringIO()) as output:
            MainViewer.load_embedding(viewer, reuse_loaded=True)
        self.assertIn("Reusing the loaded embedding manifest", output.getvalue())
        self.assertIs(viewer.embedding._manifest, first._manifest)

    def test_the_viewer_loads_the_embedding_after_the_alignment_at_startup(self):
        # Building a MainViewer needs OpenGL, so its constructor is read instead:
        # the nodes and FASTA sequences exist by then, as for the alignment.
        import inspect
        from EMAPSSN_Viewer import MainViewer

        source = inspect.getsource(MainViewer.__init__)
        self.assertIn("self.embedding = None", source)
        self.assertLess(source.index("self.load_and_simulate()"),
                        source.index("self.load_global_alignment()"))
        self.assertLess(source.index("self.load_global_alignment()"),
                        source.index("self.load_embedding()"))
        self.assertLess(source.index("self.embedding = None"),
                        source.index("self.load_embedding()"))

    def test_no_configured_file_leaves_a_manager_without_a_source(self):
        from EMAPSSN_Viewer import MainViewer
        import EMAPSSN_Config as cfg

        viewer = self.viewer()
        with mock.patch.object(cfg, "EMBED_FILE", "", create=True), \
                redirect_stdout(io.StringIO()):
            MainViewer.load_embedding(viewer)
        self.assertIsNone(viewer.embedding.source)
        self.assertIsNone(viewer.embedding.load_failure)


class DescribeTests(EmbeddingFileTestCase):
    """Embedding_Manager.describe: what the embedding command reports after the coverage."""

    def network(self, model_name):
        path = os.path.join(self.directory.name, "network.h5")
        with h5py.File(path, "w") as handle:
            handle.attrs["model_name"] = model_name
        return path

    def sentences(self, manager, network_path=None):
        return [str(message) for message in manager.describe(network_path)]

    def test_the_model_width_and_saving_mode_are_reported(self):
        manager, _ = load(self.path, ["node_a", "node_b"])
        self.assertEqual(self.sentences(manager), ["Embedding model: test_model (width 4, float16)."])

    def test_nodes_whose_sequence_differs_come_first_and_are_counted(self):
        for differing, expected in ((["node_a"], "1 node not embedded"),
                                    (["node_a", "node_b"], "2 nodes not embedded")):
            with self.subTest(differing=differing):
                sequences = {**SEQUENCES, **dict.fromkeys(differing, "WWW")}
                manager, _ = load(self.path, ["node_a", "node_b", "node_c"], sequences=sequences)
                self.assertEqual(self.sentences(manager), [
                    f"{expected}: the file holds a different sequence.",
                    "Embedding model: test_model (width 4, float16).",
                ])

    def test_a_file_without_sequences_has_no_width(self):
        empty = os.path.join(self.directory.name, "empty.h5")
        write_embedding_file(empty, [])
        manager, _ = load(empty, ["node_a"])
        self.assertEqual(self.sentences(manager), ["Embedding model: test_model (float16)."])

    def test_a_network_built_with_another_model_is_noted(self):
        manager, _ = load(self.path, ["node_a"])
        note = ("NOTE: The network was built with esm2_t12_35M_UR50D, not test_model. The Viewer "
                "accepts this; embedding commands use the embedding's model.")
        self.assertEqual(self.sentences(manager, self.network("esm2_t12_35M_UR50D"))[-1], note)
        # None for the same model, a BLAST network, no network and one that cannot be read.
        for label, path in (
            ("same model", self.network("test_model")),
            ("BLAST", self.network("BLAST")),
            ("none", None),
            ("empty", ""),
            ("missing", os.path.join(self.directory.name, "gone.h5")),
            ("not hdf5", self.path.replace(".h5", ".txt")),
        ):
            with self.subTest(label):
                if label == "not hdf5":
                    with open(path, "wb") as handle:
                        handle.write(b"x")
                self.assertEqual(
                    self.sentences(manager, path), ["Embedding model: test_model (width 4, float16)."]
                )

    def test_the_messages_translate_and_keep_their_english(self):
        from utilities import Localization

        previous = Localization.set_translator(lambda template, n: f"<{template}>")
        self.addCleanup(Localization.set_translator, previous)
        manager, _ = load(self.path, ["node_a"])
        message = manager.describe(self.network("other"))[-2]
        self.assertEqual(message.display(), "<Embedding model: {model} (width {width}, {saving_mode}).>".format(
            model="test_model", width=4, saving_mode="float16"))
        self.assertEqual(str(message), "Embedding model: test_model (width 4, float16).")


class CommandHelperTests(EmbeddingFileTestCase):
    """Command_Engine.require_embedding and get_embedding_mapping, for commands that need it."""

    def viewer(self, embedding, headers=("node_a", "node_b", "node_c")):
        return SimpleNamespace(
            full_headers=list(headers), embedding=embedding,
            console_text=SimpleNamespace(text=""),
        )

    def require(self, viewer):
        import Command_Engine

        with Command_Engine.recorded_outcome() as outcome, \
                redirect_stdout(io.StringIO()) as output:
            result = Command_Engine.require_embedding(viewer)
        return result, outcome["status"], output.getvalue()

    def test_a_usable_embedding_is_returned_without_a_report(self):
        manager, _ = load(self.path, ["node_c", "absent"])
        viewer = self.viewer(manager, ["node_c", "absent"])
        result, status, output = self.require(viewer)
        self.assertIs(result, manager)
        self.assertIsNone(status)
        self.assertEqual(output, "")
        self.assertEqual(viewer.console_text.text, "")

    def test_each_reason_the_embedding_is_unusable_fails_the_command(self):
        no_file, _ = load("", ["node_a"])
        incomplete = os.path.join(self.directory.name, "incomplete.h5")
        write_embedding_file(incomplete, RECORDS, complete=False)
        refused, _ = load(incomplete, ["node_a"])
        unrelated, _ = load(self.path, ["other_1", "other_2"])
        cases = (
            (None, "Error: No embedding loaded in the viewer. Select an embedding file in "
                   "Config (Embedding Input)."),
            (no_file, "Error: No embedding loaded in the viewer. Select an embedding file in "
                      "Config (Embedding Input)."),
            (refused, "Error: The embedding file failed to load: Embedding file rejected: "
                      "Embedding generation is incomplete."),
            (unrelated, "Error: The selected embedding covers no node of the current network."),
        )
        for embedding, expected in cases:
            with self.subTest(expected=expected):
                viewer = self.viewer(embedding, ["other_1", "other_2"] if embedding is unrelated else ["node_a"])
                result, status, output = self.require(viewer)
                self.assertIsNone(result)
                self.assertEqual(status, "failed")
                self.assertIn(expected, output)
                self.assertEqual(viewer.console_text.text, expected)

    def test_a_viewer_without_an_embedding_attribute_fails_too(self):
        viewer = SimpleNamespace(full_headers=["node_a"], console_text=SimpleNamespace(text=""))
        result, status, _ = self.require(viewer)
        self.assertIsNone(result)
        self.assertEqual(status, "failed")

    def test_the_mapping_names_each_nodes_row_or_minus_one(self):
        import Command_Engine

        manager, _ = load(self.path, ["node_c", "absent", "node_a"])
        mapping, embedded = Command_Engine.get_embedding_mapping(
            self.viewer(manager, ["node_c", "absent", "node_a"])
        )
        self.assertEqual(mapping.tolist(), [2, -1, 0])
        self.assertEqual(embedded.tolist(), [0, 2])
        for embedding, headers in ((None, ["node_a"]), (load("", ["node_a"])[0], ["node_a"]),
                                   # A mapping left on a manager with no source,
                                   # and a stale one of another length.
                                   (SimpleNamespace(source=None, viewer_to_emb=np.array([0])), ["node_a"]),
                                   (manager, ["node_c"])):
            with self.subTest(embedding=embedding, headers=headers):
                mapping, embedded = Command_Engine.get_embedding_mapping(self.viewer(embedding, headers))
                self.assertEqual(mapping.tolist(), [-1] * len(headers))
                self.assertEqual(embedded.size, 0)


if __name__ == "__main__":
    unittest.main()
