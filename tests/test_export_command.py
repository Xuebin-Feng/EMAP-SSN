"""The export command (commands/export.py): FASTA subsets per cluster, group or
#label#, the folders they are written to, and what it refuses."""

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

import EMAPSSN_Config as cfg
from commands import export as export_command
from tests.command_fixtures import one_node_viewer, reported_outcomes


class ExportCommandTests(unittest.TestCase):
    def make_viewer(self):
        return one_node_viewer()

    def test_export_uses_configured_sequence_export_directory(self):
        viewer = self.make_viewer()
        viewer.cluster_labels = np.array([0])
        viewer._selected_fasta_records = [("node", "CCCC")]

        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = os.path.join(temp_dir, "source.fasta")
            with open(fasta_path, "w", encoding="utf-8") as handle:
                handle.write(">node\nAAAA\n")

            export_root = os.path.join(temp_dir, "exports")
            metadata = SimpleNamespace(model_name="test", network_type="blast")
            patches = (
                mock.patch.object(cfg, "NODE_FASTA_FILE", fasta_path),
                mock.patch.object(cfg, "INPUT_HDF5", "network.h5"),
                mock.patch.object(
                    export_command,
                    "SEQUENCE_EXPORT_DIRECTORY",
                    export_root,
                ),
                mock.patch.object(cfg, "TOP_EDGE_PERCENT", None),
                mock.patch.object(cfg, "SIMILARITY_THRESHOLD", 0.5),
                mock.patch.object(
                    export_command.cache_manifest,
                    "validate_network_schema",
                    return_value=metadata,
                ),
            )
            # Without this patch the real file manager opens the export folder,
            # which TemporaryDirectory has already deleted when it appears.
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                    mock.patch.object(export_command, "open_in_file_manager") as open_folder:
                with redirect_stdout(io.StringIO()):
                    export_command.run(viewer, ["clusters"])

            output_dir = os.path.join(export_root, "source_[test]", "Score0.5")
            open_folder.assert_called_once()
            self.assertEqual(
                os.path.normpath(open_folder.call_args.args[0]),
                os.path.normpath(output_dir),
            )
            output_path = os.path.join(output_dir, "Cluster_0.fasta")
            self.assertTrue(os.path.isfile(output_path))
            with open(output_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), ">node\nCCCC\n")

    def test_export_specific_labels_support_groups_clusters_noise_and_overlap(self):
        viewer = self.make_viewer()
        viewer.n_nodes = 4
        viewer.full_headers = ["node_0", "node_1", "node_2", "node_3"]
        viewer.cluster_labels = np.array([0, 1, -1, 1])
        viewer.group_labels = [
            {"alpha"},
            {"cluster_001"},
            {"alpha", "beta"},
            set(),
        ]
        viewer._selected_fasta_records = list(
            zip(viewer.full_headers, ["AAAA", "CCCC", "DDDD", "EEEE"])
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = SimpleNamespace(model_name="test", network_type="blast")
            with mock.patch.object(cfg, "NODE_FASTA_FILE", os.path.join(temp_dir, "source.fasta")), mock.patch.object(
                cfg, "INPUT_HDF5", "network.h5"
            ), mock.patch.object(
                cfg, "TOP_EDGE_PERCENT", None
            ), mock.patch.object(
                cfg, "SIMILARITY_THRESHOLD", 0.5
            ), mock.patch.object(
                export_command, "SEQUENCE_EXPORT_DIRECTORY", temp_dir
            ), mock.patch.object(
                export_command.cache_manifest,
                "validate_network_schema",
                return_value=metadata,
            ), mock.patch.object(
                export_command, "open_in_file_manager"
            ):
                with redirect_stdout(io.StringIO()):
                    export_command.run(
                        viewer,
                        ["#alpha#", "#cluster_1#", "#noise#", "#alpha#"],
                    )

            output_dir = os.path.join(
                temp_dir,
                "source_[test]",
                "Score0.5_LABELS",
            )
            expected = {
                "alpha.fasta": (">node_0\nAAAA\n>node_2\nDDDD\n"),
                "Cluster_1.fasta": (">node_1\nCCCC\n>node_3\nEEEE\n"),
                "Noise.fasta": (">node_2\nDDDD\n"),
            }
            self.assertEqual(set(os.listdir(output_dir)), set(expected))
            for filename, content in expected.items():
                with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), content)

    def test_export_noncanonical_cluster_name_resolves_as_group(self):
        viewer = self.make_viewer()
        viewer.cluster_labels = np.array([1])
        viewer.group_labels = [{"cluster_001"}]
        viewer._selected_fasta_records = [("node", "AAAA")]

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = SimpleNamespace(model_name="test", network_type="blast")
            with mock.patch.object(cfg, "NODE_FASTA_FILE", os.path.join(temp_dir, "source.fasta")), mock.patch.object(
                cfg, "INPUT_HDF5", "network.h5"
            ), mock.patch.object(cfg, "TOP_EDGE_PERCENT", None), mock.patch.object(
                cfg, "SIMILARITY_THRESHOLD", 0.5
            ), mock.patch.object(
                export_command, "SEQUENCE_EXPORT_DIRECTORY", temp_dir
            ), mock.patch.object(
                export_command.cache_manifest,
                "validate_network_schema",
                return_value=metadata,
            ), mock.patch.object(export_command, "open_in_file_manager"):
                with redirect_stdout(io.StringIO()):
                    export_command.run(viewer, ["#cluster_001#"])

            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        temp_dir,
                        "source_[test]",
                        "Score0.5_GROUPS",
                        "cluster_001.fasta",
                    )
                )
            )

    def test_export_rejects_legacy_mixed_missing_and_ambiguous_targets_preflight(self):
        cases = (
            (["group:alpha"], "Legacy export"),
            (["groups", "#alpha#"], "cannot be combined"),
            (["#missing#"], "does not exist"),
            (["#cluster_0#"], "ambiguous"),
        )
        for args, message in cases:
            with self.subTest(args=args):
                viewer = self.make_viewer()
                viewer.cluster_labels = np.array([0])
                viewer.group_labels = [{"alpha", "cluster_0"}]
                viewer._selected_fasta_records = [("node", "AAAA")]
                with mock.patch.object(
                    export_command.cache_manifest, "validate_network_schema"
                ) as schema, mock.patch.object(
                    export_command, "write_fasta_atomic"
                ) as writer:
                    with redirect_stdout(io.StringIO()):
                        export_command.run(viewer, args)

                self.assertIn(message, viewer.console_text.text)
                schema.assert_not_called()
                writer.assert_not_called()


class ExportBranchTests(unittest.TestCase):
    """Missing prerequisites, folder naming, and an export with nothing to write."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = directory.name
        self.network = SimpleNamespace(model_name="test", network_type="blast")
        for patcher in (
            mock.patch.object(cfg, "NODE_FASTA_FILE", os.path.join(self.root, "source.fasta")),
            mock.patch.object(cfg, "INPUT_HDF5", "network.h5"),
            mock.patch.object(cfg, "TOP_EDGE_PERCENT", None),
            mock.patch.object(cfg, "SIMILARITY_THRESHOLD", 0.5),
            mock.patch.object(cfg, "NORM_MODE", "alignment_length"),
            mock.patch.object(cfg, "ALIGNMENT_SCORE", "global"),
            mock.patch.object(export_command, "SEQUENCE_EXPORT_DIRECTORY", self.root),
            mock.patch.object(export_command.cache_manifest, "validate_network_schema",
                              side_effect=lambda path: self.network),
            # A successful export opens its folder in the system file manager.
            mock.patch.object(export_command, "open_in_file_manager"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_viewer(self, cluster_id=1):
        """One node, "node" (sequence AAAA), in cluster `cluster_id`."""
        viewer = one_node_viewer()
        viewer.cluster_labels = None if cluster_id is None else np.array([cluster_id])
        viewer._selected_fasta_records = [("node", "AAAA")]
        return viewer

    def export(self, viewer, *args):
        engine = export_command.Command_Engine
        with mock.patch.object(engine, "command_artifact"), \
                reported_outcomes() as (succeeded, failed), \
                redirect_stdout(io.StringIO()):
            export_command.run(viewer, list(args))
        return succeeded, failed

    def written(self):
        """Every file under the export folder, relative to it."""
        return sorted(
            os.path.relpath(os.path.join(folder, name), self.root).replace(os.sep, "/")
            for folder, _, names in os.walk(self.root) for name in names
        )

    def test_cluster_export_requires_a_clustering(self):
        for args in ([], ["clusters"]):
            with self.subTest(args=args):
                viewer = self.make_viewer(cluster_id=None)
                succeeded, failed = self.export(viewer, *args)
                failed.assert_called_with(viewer, "Error: Run 'cluster' first to export clusters.")
                succeeded.assert_not_called()
                self.assertEqual(viewer.console_text.text, "Error: Run 'cluster' first.")
                self.assertEqual(os.listdir(self.root), [])

    def test_export_requires_the_sequences_in_memory(self):
        viewer = self.make_viewer()
        del viewer._selected_fasta_records
        succeeded, failed = self.export(viewer, "clusters")
        failed.assert_called_once_with(
            viewer, "Error: No in-memory sequence set is available for export."
        )
        succeeded.assert_not_called()
        self.assertEqual(os.listdir(self.root), [])

    def test_top_edge_percent_names_the_threshold_folder(self):
        for top, folder in ((5, "Top5.0Pct"), ("12.5", "Top12.5Pct"), ("None", "Score0.5")):
            with self.subTest(top=top), mock.patch.object(cfg, "TOP_EDGE_PERCENT", top):
                succeeded, failed = self.export(self.make_viewer(), "clusters")
                failed.assert_not_called()
                self.assertIn(f"source_[test]/{folder}/Cluster_1.fasta", self.written())

    def test_only_alignment_networks_name_their_scoring_in_the_folder(self):
        for network_type, folder in (
            ("alignment", "source_[test]_alignment_length_global"),
            ("blast", "source_[test]"),
        ):
            with self.subTest(network_type=network_type):
                self.network.network_type = network_type
                succeeded, failed = self.export(self.make_viewer(), "clusters")
                failed.assert_not_called()
                self.assertIn(f"{folder}/Score0.5/Cluster_1.fasta", self.written())

    def test_all_noise_clustering_exports_nothing(self):
        viewer = self.make_viewer(cluster_id=-1)
        succeeded, failed = self.export(viewer, "clusters")
        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, "No valid subsets found to export.")
        self.assertEqual(viewer.console_text.text, "No valid subsets found to export.")
        self.assertEqual(os.listdir(self.root), [])
        export_command.open_in_file_manager.assert_not_called()


if __name__ == "__main__":
    unittest.main()
