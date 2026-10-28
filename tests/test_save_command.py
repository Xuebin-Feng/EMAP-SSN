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
from tests.held_file_fixtures import transient_hold
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

    def run_save_in_a_cache_folder(self, held_renames):
        """Run `save` for a two-node viewer in a fresh cache folder, whose first ``held_renames`` renames fail as a
        held file's do; return (the folder, command_failed's mock, the renames asked for, the waits made)."""
        compatibility = make_compatibility()
        manifest = make_manifest(compatibility)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = pathlib.Path(directory.name) / "cache-folder"
        Cache_Manifest.write_manifest_atomic(folder, manifest)
        viewer = SimpleNamespace(
            cache_manifest_id=manifest["manifest_id"],
            _cache_provenance=make_provenance(manifest["manifest_id"]),
            full_headers=["A", "B"],
            pos=np.zeros((2, 2), dtype=np.float32),
            original_pos=np.ones((2, 2), dtype=np.float32),
            node_render_order=np.array([1, 0], dtype=np.int32),
        )
        with mock.patch.object(save_command, "resolve_selected_cache", return_value=str(folder / "version_00.h5")), \
                mock.patch.object(save_command.Command_Engine, "print_help"), \
                mock.patch.object(save_command.Command_Engine, "command_failed") as failed, \
                transient_hold(held_renames) as (attempts, pauses):
            save_command.run(viewer, [])
        return folder, failed, attempts, pauses

    def test_a_virus_scan_holding_the_finished_file_for_a_moment_does_not_fail_the_save(self):
        # Windows refuses the rename (WinError 32) while a scan has the file just written open.
        folder, failed, attempts, pauses = self.run_save_in_a_cache_folder(3)

        failed.assert_not_called()
        self.assertEqual(len(attempts), 4)
        self.assertEqual(len(pauses), 3)
        with h5py.File(folder / "version_00.h5", "r") as cache:
            np.testing.assert_array_equal(cache["node_render_order"][:], [1, 0])
        self.assertFalse((folder / "version_00.h5.partial").exists())

    def test_a_hold_that_never_ends_fails_the_save_saying_so_and_leaves_no_partial_file(self):
        folder, failed, attempts, pauses = self.run_save_in_a_cache_folder(10 ** 6)

        failed.assert_called_once()
        message = str(failed.call_args.args[1])
        self.assertIn("Error saving layout state", message)
        self.assertIn("another program kept the file in use", message)
        self.assertFalse((folder / "version_00.h5").exists())
        self.assertFalse((folder / "version_00.h5.partial").exists())

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

    def test_a_snapshot_keeps_the_alpha_of_every_node_color(self):
        # Colors are saved and loaded as RGBA, so a translucent node reopens translucent.
        source, document = self.cache_and_viewer()
        from EMAPSSN_Viewer import MainViewer
        import EMAPSSN_Viewer
        expected = resolve_viewer_document(document, self.root)
        viewer = MainViewer.__new__(MainViewer)
        with mock.patch.multiple(EMAPSSN_Viewer.cfg, create=True, CACHE_MANIFEST_ID=None,
                                 INPUT_IS_EVALUE=False, **expected):
            viewer.load_and_simulate()
        self.assertEqual(viewer.current_colors.shape[1], 4)
        viewer.current_colors[0] = [1.0, 0.0, 0.0, 0.25]
        viewer.current_colors[1, 3] = 0.0
        with mock.patch.object(save_command, "resolve_selected_cache", return_value=source), \
             mock.patch.object(save_command.Command_Engine, "print_help"), \
             mock.patch.object(save_command.Command_Engine, "command_failed") as failed:
            save_command.run(viewer, ["alpha.h5"])
        failed.assert_not_called()
        saved = Path(source).parent / "alpha.h5"
        with h5py.File(saved, "r") as cache:
            np.testing.assert_array_equal(cache["colors"][:2], viewer.current_colors[:2])
        reopened = copy.deepcopy(document)
        reopened["inputs"]["TARGET_CACHE_PATH"] = str(saved)
        again = MainViewer.__new__(MainViewer)
        with mock.patch.multiple(EMAPSSN_Viewer.cfg, create=True, CACHE_MANIFEST_ID=None,
                                 INPUT_IS_EVALUE=False, **resolve_viewer_document(reopened, self.root)):
            again.load_and_simulate()
        np.testing.assert_array_equal(again.current_colors[:2], viewer.current_colors[:2])
        self.assertEqual(float(again.current_colors[0, 3]), 0.25)
        self.assertEqual(float(again.current_colors[1, 3]), 0.0)

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


class CustomDatasetLoadingTests(HeadlessSettingsFixture, unittest.TestCase):
    """A layout cache is a file people share. Its extra root datasets (what `save` writes for the Viewer's custom
    attributes) become custom attributes of the Viewer and nothing else: no method, no private name and no state
    the Viewer already has is replaced, and the list of Console buttons never types a command line."""

    def setUp(self):
        super().setUp()
        self.saved_config()
        self.source, self.document = self.cache_and_viewer()

    def add_datasets(self, path, datasets):
        with h5py.File(path, "a") as cache:
            for name, value in datasets.items():
                if isinstance(value, np.ndarray):
                    cache.create_dataset(name, data=value)
                else:
                    cache.create_dataset(name, data=json.dumps(value)).attrs["is_json"] = True

    def load(self, path=None, before=None):
        """Open the cache as the Viewer does, on a Viewer that already holds ``before``; return it and what it printed."""
        import EMAPSSN_Viewer

        document = copy.deepcopy(self.document)
        if path is not None:
            document["inputs"]["TARGET_CACHE_PATH"] = str(path)
        viewer = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
        for name, value in (before or {}).items():
            setattr(viewer, name, value)
        output = io.StringIO()
        with mock.patch.multiple(EMAPSSN_Viewer.cfg, create=True, CACHE_MANIFEST_ID=None, INPUT_IS_EVALUE=False,
                                 **resolve_viewer_document(document, self.root)), redirect_stdout(output):
            viewer.load_and_simulate()
        return viewer, output.getvalue()

    def save(self, viewer, name):
        with mock.patch.object(save_command, "resolve_selected_cache", return_value=self.source), \
                mock.patch.object(save_command.Command_Engine, "print_help"), \
                mock.patch.object(save_command.Command_Engine, "command_failed") as failed:
            save_command.run(viewer, [name])
        failed.assert_not_called()
        return Path(self.source).parent / name

    def test_a_crafted_list_of_console_buttons_types_nothing_but_the_command_that_declares_one(self):
        # A cache whose list held "run C:/some/script.py" made the Viewer type "run C:/some/script.py
        # --register-only" at its command line as it opened, which runs that script. process_command is a recorder.
        self.add_datasets(self.source, {"sidebar_buttons_to_persist": ["run C:/some/script.py", "meta", "meta", 7]})
        viewer, printed = self.load()
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["run C:/some/script.py", "meta", "meta", 7])
        typed = []
        viewer.process_command = lambda command, **options: typed.append((command, options))
        with redirect_stdout(io.StringIO()) as note:
            viewer._register_persisted_console_buttons()
        self.assertEqual(typed, [("meta --register-only", {"record_history": False, "silent": True})])
        self.assertEqual(note.getvalue().count("\n"), 1)
        self.assertIn("'run C:/some/script.py' (not a bare command name)", note.getvalue())
        # The refused entries are gone from the list, so that a later `save` does not write them back.
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        saved = self.save(viewer, "clean.h5")
        with h5py.File(saved, "r") as cache:
            self.assertEqual(json.loads(cache["sidebar_buttons_to_persist"][()]), ["meta"])

    def test_a_cache_dataset_cannot_replace_a_method_a_private_name_or_state_the_viewer_has(self):
        sentinel = object()
        refused = {"process_command": "not a method", "_cacheable_attrs": ["x"], "_private_value": 3,
                   "command_queue": "not a queue", "full_headers": ["x"], "cache_manifest_id": "0" * 64,
                   "not a name": 1, "a-b": 2, "1st": 3}
        accepted = {"custom_scores": np.array([0.5, 1.5], dtype=np.float32), "custom_notes": {"a": [1, 2], "b": "text"}}
        self.add_datasets(self.source, {**refused, **accepted})
        with h5py.File(self.source, "a") as cache:
            cache.create_group("payload").create_dataset("x", data=1)  # a group is not a dataset
        viewer, printed = self.load(before={"command_queue": sentinel})
        from EMAPSSN_Viewer import MainViewer
        self.assertIs(viewer.process_command.__func__, MainViewer.process_command)
        self.assertIs(viewer.command_queue, sentinel)
        self.assertIsInstance(viewer._cacheable_attrs, set)
        self.assertEqual(len(viewer.full_headers), 2)
        self.assertNotEqual(viewer.cache_manifest_id, "0" * 64)
        for name in ("_private_value", "not a name", "a-b", "1st", "payload"):
            self.assertNotIn(name, vars(viewer))
        # The datasets that are custom attributes load, and are registered for `save`.
        np.testing.assert_array_equal(viewer.custom_scores, accepted["custom_scores"])
        self.assertEqual(viewer.custom_notes, accepted["custom_notes"])
        self.assertTrue({"custom_scores", "custom_notes", "sidebar_buttons_to_persist"} <= viewer._cacheable_attrs)
        self.assertFalse((set(refused) | {"payload"}) & viewer._cacheable_attrs)
        # One line says which were left out and why.
        lines = [line for line in printed.splitlines() if line.startswith("Warning: ignoring extra datasets")]
        self.assertEqual(len(lines), 1)
        for name in (*refused, "payload"):
            self.assertIn(repr(name), lines[0])
        self.assertIn("'process_command' (the Viewer already has that name)", lines[0])
        self.assertIn("'_private_value' (not a plain public name)", lines[0])
        self.assertIn("'payload' (not a dataset)", lines[0])
        self.assertNotIn("custom_scores", lines[0])

    def test_a_name_registered_as_cacheable_or_listed_as_a_custom_attribute_loads_though_the_viewer_has_it(self):
        self.add_datasets(self.source, {"plugin_scores": np.array([1.0, 2.0]), "sidebar_buttons_to_persist": ["agent"]})
        viewer, printed = self.load(before={"_cacheable_attrs": {"plugin_scores"}, "plugin_scores": np.zeros(2),
                                            "sidebar_buttons_to_persist": ["meta"]})
        np.testing.assert_array_equal(viewer.plugin_scores, [1.0, 2.0])
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["agent"])
        self.assertNotIn("ignoring", printed)

    def test_a_name_listed_as_a_custom_attribute_is_accepted_before_it_is_registered_too(self):
        from EMAPSSN_Viewer import MainViewer

        viewer = MainViewer.__new__(MainViewer)
        viewer._cacheable_attrs = set()
        viewer.sidebar_buttons_to_persist = ["meta"]
        with h5py.File("scratch.h5", "w", driver="core", backing_store=False) as cache:
            dataset = cache.create_dataset("sidebar_buttons_to_persist", data=1)
            self.assertIsNone(viewer._custom_dataset_problem("sidebar_buttons_to_persist", dataset))
            self.assertEqual(viewer._custom_dataset_problem("process_command", dataset),
                             "the Viewer already has that name")

    def test_a_cache_saved_with_custom_attributes_loads_them_back_as_it_always_did(self):
        # The round trip, with a custom attribute that is not in CUSTOM_ATTRIBUTES_INIT, an array and a JSON one.
        viewer, _printed = self.load()
        viewer.custom_scores = np.array([0.25, 0.75], dtype=np.float32)
        viewer.custom_notes = {"a": [1, 2], "b": "text"}
        viewer._cacheable_attrs.update({"custom_scores", "custom_notes"})
        viewer.sidebar_buttons_to_persist = ["agent", "meta"]
        saved = self.save(viewer, "roundtrip.h5")
        again, printed = self.load(path=saved)
        self.assertNotIn("ignoring", printed)
        np.testing.assert_array_equal(again.custom_scores, viewer.custom_scores)
        self.assertEqual(again.custom_notes, {"a": [1, 2], "b": "text"})
        self.assertEqual(again.sidebar_buttons_to_persist, ["agent", "meta"])
        self.assertTrue({"custom_scores", "custom_notes", "sidebar_buttons_to_persist"} <= again._cacheable_attrs)
        # Saving that again writes the same datasets.
        with h5py.File(self.save(again, "roundtrip_again.h5"), "r") as second, h5py.File(saved, "r") as first:
            self.assertEqual(sorted(second.keys()), sorted(first.keys()))
            np.testing.assert_array_equal(second["custom_scores"][:], first["custom_scores"][:])

    def test_undo_snapshots_hold_only_names_the_loader_registered(self):
        # Undo restores the names of a snapshot, and a snapshot holds the cacheable names, so a name the loader refused
        # is never in one.
        self.add_datasets(self.source, {"process_command": "not a method", "_private_value": 3,
                                        "custom_scores": np.array([1.0])})
        viewer, _printed = self.load()
        snapshot = viewer._get_custom_attributes_snapshot()
        self.assertEqual(set(snapshot), {"custom_scores"})
        viewer._apply_custom_attributes_snapshot(snapshot)
        from EMAPSSN_Viewer import MainViewer
        self.assertIs(viewer.process_command.__func__, MainViewer.process_command)
        self.assertNotIn("_private_value", vars(viewer))


if __name__ == "__main__":
    unittest.main()
