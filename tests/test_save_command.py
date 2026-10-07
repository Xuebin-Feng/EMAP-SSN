"""The Viewer's save command (commands/save.py): snapshots bound to the active
folder manifest and the cache's original provenance, written atomically under
a validated plain filename, reopenable with the generation settings they came
from, and never replacing a destination when the provenance is invalid."""
import copy
import pathlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Cache_Manifest
from commands import save as save_command
from desktop.Viewer_State import resolve_viewer_document
from utilities.Cache_Metadata import read_cache_metadata, validate_cache_provenance
from tests.layout_fixtures import (
    HeadlessSettingsFixture, make_compatibility, make_manifest, make_provenance,
)


class InteractiveSaveTests(unittest.TestCase):
    def test_saved_snapshot_is_atomically_bound_to_active_manifest(self):
        compatibility = make_compatibility()
        manifest = make_manifest(compatibility)
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = pathlib.Path(temp_dir) / "cache-folder"
            Cache_Manifest.write_manifest_atomic(folder, manifest)
            default_path = folder / "version_00.h5"
            viewer = SimpleNamespace(
                cache_manifest_id=manifest["manifest_id"],
                _cache_provenance=make_provenance(manifest["manifest_id"]),
                full_headers=["A", "B"],
                pos=np.zeros((2, 2), dtype=np.float32),
                original_pos=np.ones((2, 2), dtype=np.float32),
                node_render_order=np.array([1, 0], dtype=np.int32),
            )

            with mock.patch.object(
                save_command,
                "resolve_selected_cache",
                return_value=str(default_path),
            ), mock.patch.object(save_command.Command_Engine, "print_help"):
                save_command.run(viewer, [])

            self.assertTrue(default_path.exists())
            self.assertFalse(pathlib.Path(str(default_path) + ".partial").exists())
            with h5py.File(default_path, "r") as cache:
                self.assertEqual(
                    cache.attrs["cache_manifest_id"], manifest["manifest_id"]
                )
                np.testing.assert_array_equal(cache["positions"][:], viewer.pos)
                np.testing.assert_array_equal(
                    cache["node_render_order"][:], viewer.node_render_order
                )
            np.testing.assert_array_equal(viewer.original_pos, viewer.pos)

    def test_unsafe_interactive_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = pathlib.Path(temp_dir) / "cache-folder"
            folder.mkdir()
            viewer = SimpleNamespace(
                cache_manifest_id="d" * 64,
                full_headers=["A"],
                pos=np.zeros((1, 2), dtype=np.float32),
            )
            messages = []
            with mock.patch.object(
                save_command,
                "resolve_selected_cache",
                return_value=str(folder / "version_00.h5"),
            ), mock.patch.object(
                save_command.Command_Engine,
                "print_help",
                side_effect=lambda _viewer, message: messages.append(message),
            ):
                save_command.run(viewer, ["../escape"])

            self.assertTrue(messages)
            self.assertIn("Error saving layout state", messages[-1])
            self.assertFalse((pathlib.Path(temp_dir) / "escape.h5").exists())


class SnapshotProvenanceTests(HeadlessSettingsFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.saved = self.saved_config()

    def snapshot_viewer(self, path):
        with h5py.File(path, "r") as cache:
            manifest_id = cache.attrs["cache_manifest_id"]
            return SimpleNamespace(
                cache_manifest_id=manifest_id,
                _cache_provenance=validate_cache_provenance(cache.attrs, manifest_id),
                full_headers=cache["headers"].asstr()[:].tolist(),
                pos=cache["positions"][:],
            )

    def test_generated_snapshot_reopens_and_preserves_original_provenance(self):
        for changes in ({"SIMILARITY_THRESHOLD": 0.2},
                        {"TOP_EDGE_PERCENT": 50.0, "SIMILARITY_THRESHOLD": None},
                        {"UMAP_MODE": True, "UMAP_NEIGHBORS": 2, "SIMILARITY_THRESHOLD": None}):
            with self.subTest(changes=changes):
                source, document = self.cache_and_viewer(**changes)
                expected = resolve_viewer_document(document, self.root)
                from EMAPSSN_Viewer import MainViewer
                import EMAPSSN_Viewer
                viewer = MainViewer.__new__(MainViewer)
                # load_and_simulate also sets CACHE_MANIFEST_ID and INPUT_IS_EVALUE on cfg.
                with mock.patch.multiple(EMAPSSN_Viewer.cfg, create=True, CACHE_MANIFEST_ID=None,
                                         INPUT_IS_EVALUE=False, **expected):
                    viewer.load_and_simulate()
                original = dict(viewer._cache_provenance)
                for filename in ("snapshot.h5", "snapshot.h5", Path(source).name):
                    with mock.patch.object(save_command, "resolve_selected_cache", return_value=source), \
                         mock.patch.object(save_command.cfg, "BOX_SCALE", 999), \
                         mock.patch.object(save_command.Command_Engine, "print_help"), \
                         mock.patch.object(save_command.Command_Engine, "command_failed") as failed:
                        save_command.run(viewer, [filename])
                    failed.assert_not_called()
                    saved = Path(source).parent / filename
                    metadata = read_cache_metadata(str(saved))
                    self.assertEqual(metadata["status"], "complete")
                    self.assertEqual(metadata["attributes"], original)
                    reopened = copy.deepcopy(document)
                    reopened["inputs"]["TARGET_CACHE_PATH"] = str(saved)
                    actual = resolve_viewer_document(reopened, self.root)
                    for key in ("BOX_SCALE", "UMAP_MODE", "UMAP_NEIGHBORS", "SIMILARITY_THRESHOLD", "TOP_EDGE_PERCENT"):
                        self.assertEqual(actual[key], expected[key])
                    viewer = self.snapshot_viewer(saved)

    def test_invalid_snapshot_provenance_does_not_replace_destination(self):
        source, _ = self.cache_and_viewer()
        destination = Path(source).parent / "protected.h5"
        destination.write_bytes(b"existing destination")
        for alteration in ("absent", "missing", "hash", "binding", "folder"):
            with self.subTest(alteration=alteration):
                viewer = self.snapshot_viewer(source)
                if alteration == "absent":
                    del viewer._cache_provenance
                elif alteration == "missing":
                    del viewer._cache_provenance["layout_compatibility_json"]
                elif alteration == "hash":
                    viewer._cache_provenance["layout_compatibility_id"] = "bad"
                elif alteration == "binding":
                    viewer._cache_provenance["cache_manifest_id"] = "bad"
                else:
                    viewer.cache_manifest_id = "bad"
                    viewer._cache_provenance["cache_manifest_id"] = "bad"
                with mock.patch.object(save_command, "resolve_selected_cache", return_value=source), \
                     mock.patch.object(save_command.Command_Engine, "print_help"), \
                     mock.patch.object(save_command.Command_Engine, "command_failed") as failed:
                    save_command.run(viewer, [destination.name])
                failed.assert_called_once()
                self.assertEqual(destination.read_bytes(), b"existing destination")
                self.assertFalse(Path(str(destination) + ".partial").exists())


if __name__ == "__main__":
    unittest.main()
