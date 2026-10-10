"""The Viewer's save command (commands/save.py): snapshots bound to the active
folder manifest and the cache's original provenance, written atomically under
a validated plain filename, reopenable with the generation settings they came
from, and never replacing a destination when the provenance is invalid."""
import copy
import io
import json
import os
import pathlib
from contextlib import redirect_stdout
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

    def test_the_sidebar_button_list_is_written_into_the_cache(self):
        # The list the viewer registers its sidebar buttons from at startup is a
        # cacheable attribute: undo leaves it alone, and `save` still writes it.
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
                _cacheable_attrs={"sidebar_buttons_to_persist"},
                sidebar_buttons_to_persist=["meta"],
            )

            with mock.patch.object(
                save_command,
                "resolve_selected_cache",
                return_value=str(default_path),
            ), mock.patch.object(save_command.Command_Engine, "print_help"):
                save_command.run(viewer, [])

            with h5py.File(default_path, "r") as cache:
                dataset = cache["sidebar_buttons_to_persist"]
                self.assertTrue(dataset.attrs["is_json"])
                self.assertEqual(json.loads(dataset[()]), ["meta"])

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
                side_effect=lambda _viewer, message: messages.append(str(message)),
            ):
                save_command.run(viewer, ["../escape"])

            self.assertTrue(messages)
            self.assertIn("Error saving layout state", messages[-1])
            self.assertFalse((pathlib.Path(temp_dir) / "escape.h5").exists())

    def test_uppercase_h5_name_keeps_its_suffix(self):
        compatibility = make_compatibility()
        manifest = make_manifest(compatibility)
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = pathlib.Path(temp_dir) / "cache-folder"
            Cache_Manifest.write_manifest_atomic(folder, manifest)
            viewer = SimpleNamespace(
                cache_manifest_id=manifest["manifest_id"],
                _cache_provenance=make_provenance(manifest["manifest_id"]),
                full_headers=["A", "B"],
                pos=np.zeros((2, 2), dtype=np.float32),
            )
            with mock.patch.object(
                save_command,
                "resolve_selected_cache",
                return_value=str(folder / "version_00.h5"),
            ), mock.patch.object(save_command.Command_Engine, "print_help"):
                save_command.run(viewer, ["Snap.H5"])

            self.assertTrue((folder / "Snap.H5").exists())
            self.assertFalse((folder / "Snap.H5.h5").exists())

    @unittest.skipUnless(os.name == "nt", "Only Windows refuses ':' (NTFS reads it as a data stream).")
    def test_colon_in_name_is_refused_before_any_file_is_made(self):
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
                side_effect=lambda _viewer, message: messages.append(str(message)),
            ):
                save_command.run(viewer, ["bad:name"])

            self.assertIn("Filename cannot contain ':'", messages[-1])
            self.assertEqual(list(folder.iterdir()), [])

    def test_help_says_what_a_saved_name_replaces_and_what_reset_restores(self):
        output = io.StringIO()
        with redirect_stdout(output):
            save_command.run(SimpleNamespace(), ["--help"])

        self.assertIn("save version_00.h5 replaces the original layout file", output.getvalue())
        self.assertIn("reset network returns to the saved layout", output.getvalue())


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
