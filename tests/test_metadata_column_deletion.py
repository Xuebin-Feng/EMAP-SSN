import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import h5py


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emapssn_viewer import MainViewer
from commands import meta as meta_command
from commands import save as save_command
from web_ui import meta_backend
from web_ui.Plugin_Manager import WebPluginRegistry


class FakeDisplay:
    def __init__(self):
        self.hidden = False
        self.messages = []

    def hide(self):
        self.hidden = True

    def show(self, message):
        self.hidden = False
        self.messages.append(message)


def make_viewer():
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = 2
    viewer.full_headers = ["node-1", "node-2"]
    viewer.visible_mask = np.array([True, True], dtype=bool)
    viewer.selected_indices = []
    viewer.selected_node_idx = 0
    viewer.metadata = {
        "Length": {
            "type": "number",
            "values": np.array([100, 110], dtype=np.int32),
        },
        "Organism": {
            "type": "text",
            "values": np.array(["alpha", "beta"], dtype=object),
        },
        "Host": {
            "type": "text",
            "values": np.array(["plant", "soil"], dtype=object),
        },
    }
    viewer.position_history = []
    viewer.redo_stack = []
    viewer.console_text = SimpleNamespace(text="")
    viewer.hud_displays = {"meta_display": FakeDisplay()}
    viewer.meta_display_prop = None
    viewer.update_nodes = mock.Mock()
    viewer.canvas = SimpleNamespace(update=mock.Mock())
    viewer.broadcast_event = mock.Mock()
    return viewer


class MetadataColumnDeletionTests(unittest.TestCase):
    def test_multiple_case_insensitive_names_are_atomic_and_deduplicated(self):
        viewer = make_viewer()

        deleted = meta_backend.delete_metadata_columns(
            viewer, ["organism", "HOST", "Organism"]
        )

        self.assertEqual(deleted, ["Organism", "Host"])
        self.assertEqual(list(viewer.metadata), ["Length"])
        self.assertEqual(len(viewer.position_history), 1)
        event = viewer.broadcast_event.call_args.args[0]
        self.assertEqual(event["columns"], ["Node ID", "Length"])
        self.assertEqual(event["types"], {"Length": "number"})

    def test_missing_or_protected_name_aborts_without_mutation(self):
        for requested in (["Organism", "Missing"], ["Organism", "Node ID"]):
            with self.subTest(requested=requested):
                viewer = make_viewer()
                original = list(viewer.metadata)

                with self.assertRaises(meta_backend.MetadataColumnDeleteError):
                    meta_backend.delete_metadata_columns(viewer, requested)

                self.assertEqual(list(viewer.metadata), original)
                self.assertEqual(viewer.position_history, [])
                viewer.broadcast_event.assert_not_called()

    def test_all_keyword_is_not_supported(self):
        viewer = make_viewer()
        with self.assertRaisesRegex(
            meta_backend.MetadataColumnDeleteError, "not supported"
        ):
            meta_backend.delete_metadata_columns(viewer, ["all"])

    def test_length_is_a_normal_deletable_metadata_column(self):
        viewer = make_viewer()

        deleted = meta_backend.delete_metadata_columns(viewer, ["length"])

        self.assertEqual(deleted, ["Length"])
        self.assertNotIn("Length", viewer.metadata)

    def test_saved_metadata_schema_does_not_regenerate_deleted_length(self):
        viewer = make_viewer()
        viewer.metadata = {}
        viewer._metadata_loaded_from_cache = True
        viewer.sequences_map = {"node-1": "AAAA", "node-2": "AAAAA"}

        viewer._init_colors()

        self.assertNotIn("Length", viewer.metadata)

    def test_delete_undo_redo_restores_order_values_and_active_hud(self):
        viewer = make_viewer()
        display = viewer.hud_displays["meta_display"]
        viewer.meta_display_prop = "Organism"

        meta_backend.delete_metadata_columns(viewer, ["Organism", "Host"])
        self.assertEqual(list(viewer.metadata), ["Length"])
        self.assertIsNone(viewer.meta_display_prop)
        self.assertTrue(display.hidden)

        self.assertTrue(viewer._do_undo())
        self.assertEqual(list(viewer.metadata), ["Length", "Organism", "Host"])
        np.testing.assert_array_equal(
            viewer.metadata["Organism"]["values"], ["alpha", "beta"]
        )
        self.assertEqual(viewer.meta_display_prop, "Organism")
        self.assertEqual(display.messages[-1], "Organism: alpha")

        self.assertTrue(viewer._do_redo())
        self.assertEqual(list(viewer.metadata), ["Length"])
        self.assertIsNone(viewer.meta_display_prop)
        self.assertTrue(display.hidden)

    def test_cell_edit_uses_compact_shared_history(self):
        viewer = make_viewer()

        self.assertTrue(meta_backend.handle_edit_cell(
            viewer,
            {"row": 1, "column": "Organism", "value": "gamma"},
        ))
        self.assertEqual(viewer.metadata["Organism"]["values"][1], "gamma")
        self.assertEqual(
            viewer.position_history[-1]["_history_kind"], "metadata_cell"
        )

        viewer._do_undo()
        self.assertEqual(viewer.metadata["Organism"]["values"][1], "beta")
        viewer._do_redo()
        self.assertEqual(viewer.metadata["Organism"]["values"][1], "gamma")

    def test_subsequent_save_excludes_deleted_columns(self):
        viewer = make_viewer()
        viewer.pos = np.zeros((2, 2), dtype=np.float32)
        viewer.cache_manifest_id = "test-manifest"
        meta_backend.delete_metadata_columns(
            viewer, ["Length", "Organism"], broadcast=False
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            default_path = os.path.join(temp_dir, "version_00.h5")
            with mock.patch.object(
                save_command,
                "resolve_selected_cache",
                return_value=(default_path, None),
            ), mock.patch.object(
                save_command.cache_manifest,
                "read_manifest",
                return_value={"manifest_id": "test-manifest"},
            ), mock.patch.object(
                save_command.cache_manifest, "validate_cache_filename"
            ), mock.patch.object(
                save_command.Command_Engine, "print_help"
            ):
                save_command.run(viewer, ["deleted_columns.h5"])

            saved_path = os.path.join(temp_dir, "deleted_columns.h5")
            with h5py.File(saved_path, "r") as handle:
                self.assertEqual(list(handle["metadata"].keys()), ["Host"])

    def test_web_actions_include_delete_and_server_history(self):
        viewer = make_viewer()
        registry = WebPluginRegistry(viewer)

        meta_backend.register_backend(registry, viewer)

        self.assertIn("delete_columns", registry.actions)
        self.assertIn("metadata_undo", registry.actions)
        self.assertIn("metadata_redo", registry.actions)


class MetadataCliDeletionTests(unittest.TestCase):
    def test_delete_remove_and_clear_dispatch_the_same_multiple_column_helper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for alias in ("delete", "remove", "clear"):
                with self.subTest(alias=alias), mock.patch.object(
                    meta_command.cfg, "METADATA_DIR", temp_dir
                ), mock.patch.object(
                    meta_command, "register"
                ) as register_mock, mock.patch.object(
                    meta_command,
                    "delete_metadata_columns",
                    return_value=["Organism", "Host"],
                ) as delete_mock, mock.patch.object(
                    meta_command.Command_Engine, "print_help"
                ):
                    viewer = SimpleNamespace()

                    meta_command.run(viewer, [alias, "organism", "HOST"])

                    register_mock.assert_not_called()
                    delete_mock.assert_called_once_with(
                        viewer, ["organism", "HOST"], broadcast=False
                    )

    def test_delete_without_names_prints_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            meta_command.cfg, "METADATA_DIR", temp_dir
        ), mock.patch.object(
            meta_command.Command_Engine, "print_help"
        ) as print_help:
            meta_command.run(SimpleNamespace(), ["delete"])

        self.assertIn("Usage: meta delete", print_help.call_args.args[1])


class MetadataWebMarkupTests(unittest.TestCase):
    def test_web_headers_have_confirmed_delete_and_server_undo_actions(self):
        html = (SRC_DIR / "web_ui" / "meta.html").read_text(encoding="utf-8")

        self.assertIn('className = "metadata-delete-column"', html)
        self.assertIn('window.confirm(`Delete metadata column', html)
        self.assertIn('action: "delete_columns"', html)
        self.assertIn('action: "metadata_undo"', html)
        self.assertIn('action: "metadata_redo"', html)
        self.assertIn('title: "Sequence Header"', html)
        self.assertNotIn('columnName.toLowerCase() !== "length"', html)
        self.assertIn('event.stopPropagation()', html)
        self.assertIn('.metadata-typed-column .tabulator-header-filter', html)
        self.assertIn('content: "NUM"', html)
        self.assertIn('content: "TXT"', html)
        self.assertIn('cssClass: `metadata-typed-column metadata-${type', html)
        self.assertIn('justify-content: center;', html)
        self.assertIn('className = "metadata-column-name"', html)
        self.assertIn('className = "metadata-column-drag-handle"', html)
        self.assertIn('right: -4px;', html)
        self.assertIn('#spreadsheet-table .tabulator-col-sorter', html)
        self.assertIn('right: 10px !important;', html)


if __name__ == "__main__":
    unittest.main()
