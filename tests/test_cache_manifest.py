import json
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import Cache_Manifest
from commands import save as save_command
from utilities.Cache_Selection import resolve_selected_cache


def make_compatibility(sequence_hash="a" * 64, network_hash="b" * 64, **overrides):
    settings = {
        "alignment_score": "global",
        "normalization": "alignment_length",
        "umap_mode": False,
        "umap_neighbors": 15,
        "top_edge_percent": None,
        "similarity_threshold": 0.4,
    }
    settings.update(overrides)
    return Cache_Manifest.build_compatibility(
        sequence_hash,
        network_hash,
        "alignment",
        **settings,
    )


def make_manifest(compatibility, sequence_name="set.fasta", network_name="network.h5"):
    return Cache_Manifest.build_manifest(
        {
            "basename": sequence_name,
            "size_bytes": 10,
            "sha256": compatibility["sequence_sha256"],
        },
        {
            "basename": network_name,
            "size_bytes": 20,
            "sha256": compatibility["network_sha256"],
        },
        compatibility,
    )


class CacheSelectionTests(unittest.TestCase):
    def test_default_and_legacy_cache_paths_preserve_canonical_naming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            network_path = root / "network.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "model"
                network.create_dataset("headers", data=[b"A"])
                network.create_dataset("seq_lens", data=[1])
                for dataset in (
                    "i",
                    "j",
                    "l_score",
                    "l_len",
                    "g_score",
                    "g_len",
                ):
                    network.create_dataset(dataset, data=[])

            msa_path = root / "alignment.fasta"
            msa_path.write_text(
                ">Alpha description\nA\n>Beta description\nB\n",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                ALIGNMENT_REFERENCE="beta",
                MSA_FILE=str(msa_path),
                SAVED_LAYOUT_DIR=str(root / "layouts"),
                TARGET_CACHE_PATH=None,
                TARGET_CACHE_FILE=None,
                NODE_FASTA_FILE=str(root / "set.fasta"),
                SEQUENCES_FILE="",
                INPUT_HDF5=str(network_path),
                ALIGNMENT_SCORE="global",
                NORM_MODE="alignment_length",
                UMAP_MODE=False,
                UMAP_NEIGHBORS=15,
                TOP_EDGE_PERCENT=None,
                SIMILARITY_THRESHOLD=0.4,
            )

            cache_path, reference = resolve_selected_cache(settings)
            self.assertEqual(reference, "Beta description")
            self.assertEqual(
                pathlib.Path(cache_path).name,
                "version_00.h5",
            )
            self.assertEqual(
                pathlib.Path(cache_path).parent.name,
                "set_[model]_alignment_length_global_Score0.4",
            )
            self.assertFalse(settings.INPUT_IS_EVALUE)

            settings.TARGET_CACHE_FILE = "saved_layout.h5"
            legacy_path, _ = resolve_selected_cache(settings)
            self.assertEqual(pathlib.Path(legacy_path).name, "saved_layout.h5")

    def test_explicit_relative_path_resolves_hdf5_reference_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            msa_path = root / "alignment.h5"
            with h5py.File(msa_path, "w") as alignment:
                alignment.create_dataset("headers", data=[b"Alpha", b"Beta full"])

            settings = SimpleNamespace(
                ALIGNMENT_REFERENCE="beta",
                MSA_FILE=str(msa_path),
                SAVED_LAYOUT_DIR=str(root / "layouts"),
                TARGET_CACHE_PATH="chosen/layout.h5",
            )
            cache_path, reference = resolve_selected_cache(settings)
            self.assertEqual(reference, "Beta full")
            self.assertEqual(
                pathlib.Path(cache_path),
                root / "layouts" / "chosen" / "layout.h5",
            )

            settings.TARGET_CACHE_PATH = "../escape.h5"
            with self.assertRaises(Cache_Manifest.CacheManifestError):
                resolve_selected_cache(settings)

    def test_none_reference_is_inactive(self):
        settings = SimpleNamespace(
            ALIGNMENT_REFERENCE=None,
            MSA_FILE="missing.fasta",
            SAVED_LAYOUT_DIR="layouts",
            TARGET_CACHE_PATH="chosen/layout.h5",
        )
        _, reference = resolve_selected_cache(settings)
        self.assertIsNone(reference)


class ManifestIdentityTests(unittest.TestCase):
    def test_file_hash_uses_contents_not_name_or_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = pathlib.Path(temp_dir) / "first.bin"
            second_dir = pathlib.Path(temp_dir) / "elsewhere"
            second_dir.mkdir()
            second = second_dir / "renamed.bin"
            first.write_bytes(b"identical bytes")
            second.write_bytes(b"identical bytes")

            self.assertEqual(
                Cache_Manifest.calculate_file_sha256(first),
                Cache_Manifest.calculate_file_sha256(second),
            )
            second.write_bytes(b"changed bytes")
            self.assertNotEqual(
                Cache_Manifest.calculate_file_sha256(first),
                Cache_Manifest.calculate_file_sha256(second),
            )

    def test_manifest_id_excludes_informational_fields(self):
        compatibility = make_compatibility()
        first = make_manifest(compatibility, "first.fasta", "first.h5")
        second = make_manifest(compatibility, "renamed.fasta", "renamed.h5")

        self.assertEqual(first["manifest_id"], second["manifest_id"])
        self.assertNotEqual(first["created_at_utc"], "")
        self.assertNotIn("schema_version", first)

    def test_only_active_edge_selection_affects_identity(self):
        threshold = make_compatibility(similarity_threshold=0.4)
        different_threshold = make_compatibility(similarity_threshold=0.5)
        top_percent = make_compatibility(top_edge_percent=1.0)
        umap = make_compatibility(
            umap_mode=True,
            umap_neighbors=25,
            top_edge_percent=2.0,
            similarity_threshold=0.8,
        )
        same_umap = make_compatibility(
            umap_mode=True,
            umap_neighbors=25,
            top_edge_percent=None,
            similarity_threshold=0.1,
        )

        self.assertNotEqual(
            Cache_Manifest.calculate_manifest_id(threshold),
            Cache_Manifest.calculate_manifest_id(different_threshold),
        )
        self.assertEqual(top_percent["edge_filter"]["mode"], "top_edge_percent")
        self.assertEqual(umap, same_umap)

    def test_blast_ignores_alignment_score_and_normalization(self):
        first = Cache_Manifest.build_compatibility(
            "a" * 64,
            "b" * 64,
            "blast",
            alignment_score="global",
            normalization="alignment_length",
            similarity_threshold=10,
        )
        second = Cache_Manifest.build_compatibility(
            "a" * 64,
            "b" * 64,
            "blast",
            alignment_score="local",
            normalization="longer_sequence",
            similarity_threshold=10.0,
        )
        self.assertEqual(first, second)

    def test_canonical_name_uses_authoritative_network_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "example_[wrong]_network.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "model"
                network.create_dataset("headers", data=[b"Alpha"])
                network.create_dataset("seq_lens", data=[1])
                for dataset in (
                    "i",
                    "j",
                    "l_score",
                    "l_len",
                    "g_score",
                    "g_len",
                ):
                    network.create_dataset(dataset, data=[])

            name = Cache_Manifest.build_canonical_cache_name(
                "Input_Files/Sequence_Sets/example.fasta",
                network_path,
                "alignment",
                alignment_score="global",
                normalization="alignment_length",
                umap_mode=False,
                top_edge_percent=None,
                similarity_threshold=0.4,
            )
        self.assertEqual(
            name,
            "example_[model]_alignment_length_global_Score0.4",
        )

    def test_next_cache_version_filename_uses_simple_numbering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = pathlib.Path(temp_dir)
            self.assertEqual(
                Cache_Manifest.next_cache_version_filename(folder),
                "version_00.h5",
            )
            (folder / "version_00.h5").touch()
            (folder / "VERSION_03.H5").touch()
            (folder / "old-folder_ver.99.h5").touch()
            (folder / "version_notes.h5").touch()
            self.assertEqual(
                Cache_Manifest.next_cache_version_filename(folder),
                "version_04.h5",
            )

    def test_default_folder_adds_identity_suffix_for_canonical_collision(self):
        current_compatibility = make_compatibility()
        current_manifest = make_manifest(current_compatibility)
        incompatible_manifest = make_manifest(
            make_compatibility(sequence_hash="c" * 64)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            canonical = root / "readable-cache-name"
            Cache_Manifest.write_manifest_atomic(canonical, incompatible_manifest)

            resolved = pathlib.Path(
                Cache_Manifest.resolve_default_cache_folder(
                    root, canonical.name, current_compatibility
                )
            )
            expected_suffix = current_manifest["manifest_id"][:8]
            self.assertEqual(
                resolved.name, f"{canonical.name}_[{expected_suffix}]"
            )
            self.assertEqual(
                Cache_Manifest.read_manifest(canonical)["manifest_id"],
                incompatible_manifest["manifest_id"],
            )

            Cache_Manifest.write_manifest_atomic(resolved, current_manifest)
            self.assertEqual(
                pathlib.Path(
                    Cache_Manifest.resolve_default_cache_folder(
                        root, canonical.name, current_compatibility
                    )
                ),
                resolved,
            )


class ManifestDiscoveryTests(unittest.TestCase):
    def test_zero_one_renamed_and_duplicate_matches(self):
        compatibility = make_compatibility()
        manifest = make_manifest(compatibility)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            self.assertEqual(
                Cache_Manifest.find_matching_manifest_folders(root, compatibility),
                [],
            )

            renamed = root / "user-renamed-folder"
            Cache_Manifest.write_manifest_atomic(renamed, manifest)
            matches = Cache_Manifest.find_matching_manifest_folders(
                root, compatibility
            )
            self.assertEqual([pathlib.Path(item["folder"]).name for item in matches], ["user-renamed-folder"])

            duplicate = root / "another-copy"
            Cache_Manifest.write_manifest_atomic(duplicate, manifest)
            matches = Cache_Manifest.find_matching_manifest_folders(
                root, compatibility
            )
            self.assertEqual(len(matches), 2)

    def test_nested_malformed_and_mismatching_manifests_are_ignored(self):
        compatibility = make_compatibility()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            nested = root / "outer" / "nested"
            Cache_Manifest.write_manifest_atomic(
                nested, make_manifest(compatibility)
            )
            malformed = root / "malformed"
            malformed.mkdir()
            (malformed / Cache_Manifest.MANIFEST_FILENAME).write_text(
                "{not json", encoding="utf-8"
            )
            mismatch = root / "mismatch"
            Cache_Manifest.write_manifest_atomic(
                mismatch,
                make_manifest(make_compatibility(network_hash="c" * 64)),
            )

            self.assertEqual(
                Cache_Manifest.find_matching_manifest_folders(root, compatibility),
                [],
            )

    def test_manifest_tampering_is_rejected(self):
        manifest = make_manifest(make_compatibility())
        manifest["compatibility"]["edge_filter"]["value"] = 0.9
        with self.assertRaisesRegex(
            Cache_Manifest.CacheManifestError, "Manifest ID"
        ):
            Cache_Manifest.validate_manifest(manifest)


class CachePathAndHdf5Tests(unittest.TestCase):
    def test_node_render_order_requires_complete_integer_permutation(self):
        np.testing.assert_array_equal(
            Cache_Manifest.validate_node_render_order([2, 0, 1], 3),
            [2, 0, 1],
        )
        for invalid in ([0, 0, 2], [0, 1, 3], [0, 1], [0.0, 1.0, 2.0]):
            with self.subTest(invalid=invalid), self.assertRaises(
                Cache_Manifest.CacheManifestError
            ):
                Cache_Manifest.validate_node_render_order(invalid, 3)

    def test_relative_cache_path_round_trip_and_traversal_rejection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            folder = root / "cache-folder"
            folder.mkdir()
            relative = Cache_Manifest.relative_cache_path(
                root, folder, "layout.h5"
            )
            self.assertEqual(
                pathlib.Path(
                    Cache_Manifest.resolve_relative_cache_path(root, relative)
                ),
                folder / "layout.h5",
            )
            for unsafe in ("../layout.h5", "folder/../layout.h5", "layout.h5"):
                with self.assertRaises(Cache_Manifest.CacheManifestError):
                    Cache_Manifest.resolve_relative_cache_path(root, unsafe)

    def test_valid_cache_requires_binding_headers_and_finite_positions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = pathlib.Path(temp_dir) / "layout.h5"
            headers = ["A", "B"]
            with h5py.File(cache_path, "w") as cache:
                cache.attrs["cache_manifest_id"] = "d" * 64
                cache.create_dataset(
                    "headers",
                    data=np.asarray(headers, dtype=object),
                    dtype=h5py.string_dtype("utf-8"),
                )
                cache.create_dataset(
                    "positions", data=np.asarray([[0.0, 1.0], [2.0, 3.0]])
                )

            with h5py.File(cache_path, "r") as cache:
                cached_headers, positions = Cache_Manifest.validate_cache_hdf5(
                    cache, headers, "d" * 64
                )
            self.assertEqual(cached_headers, headers)
            self.assertEqual(positions.shape, (2, 2))

            with h5py.File(cache_path, "r+") as cache:
                cache["positions"][1, 0] = np.nan
            with h5py.File(cache_path, "r") as cache:
                with self.assertRaisesRegex(
                    Cache_Manifest.CacheManifestError, "invalid coordinates"
                ):
                    Cache_Manifest.validate_cache_hdf5(
                        cache, headers, "d" * 64
                    )

    def test_legacy_and_wrong_header_cache_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = pathlib.Path(temp_dir) / "layout.h5"
            with h5py.File(cache_path, "w") as cache:
                cache.create_dataset(
                    "headers",
                    data=np.asarray(["B", "A"], dtype=object),
                    dtype=h5py.string_dtype("utf-8"),
                )
                cache.create_dataset("positions", data=np.zeros((2, 2)))
            with h5py.File(cache_path, "r") as cache:
                with self.assertRaisesRegex(
                    Cache_Manifest.CacheManifestError, "folder manifest"
                ):
                    Cache_Manifest.validate_cache_hdf5(
                        cache, ["A", "B"], "d" * 64
                    )

            with h5py.File(cache_path, "r+") as cache:
                cache.attrs["cache_manifest_id"] = "d" * 64
            with h5py.File(cache_path, "r") as cache:
                with self.assertRaisesRegex(
                    Cache_Manifest.CacheManifestError, "node order"
                ):
                    Cache_Manifest.validate_cache_hdf5(
                        cache, ["A", "B"], "d" * 64
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
                full_headers=["A", "B"],
                pos=np.zeros((2, 2), dtype=np.float32),
                original_pos=np.ones((2, 2), dtype=np.float32),
                node_render_order=np.array([1, 0], dtype=np.int32),
            )

            with mock.patch.object(
                save_command,
                "resolve_selected_cache",
                return_value=(str(default_path), None),
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
                return_value=(str(folder / "version_00.h5"), None),
            ), mock.patch.object(
                save_command.Command_Engine,
                "print_help",
                side_effect=lambda _viewer, message: messages.append(message),
            ):
                save_command.run(viewer, ["../escape"])

            self.assertTrue(messages)
            self.assertIn("Error saving layout state", messages[-1])
            self.assertFalse((pathlib.Path(temp_dir) / "escape.h5").exists())


class ViewerCacheIntegrationTests(unittest.TestCase):
    def test_new_cache_and_manifest_are_reloaded_only_when_bound(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import Layout_Engine_SSN_MolecularDynamics as layout_engine
        import emapssn_viewer

        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = pathlib.Path(temp_dir) / "set.fasta"
            network_path = pathlib.Path(temp_dir) / "network_[model]_network.h5"
            layout_root = pathlib.Path(temp_dir) / "layouts"
            relative_cache = os.path.join("target", "layout.h5")
            fasta_path.write_text(
                ">Alpha??__Beta\nAA\n>Gamma##Delta\nCC\n",
                encoding="utf-8",
            )
            with h5py.File(network_path, "w") as network:
                string_dtype = h5py.string_dtype("utf-8")
                network.attrs["model_name"] = "model"
                network.create_dataset(
                    "headers",
                    data=np.asarray(
                        ["Alpha_Beta", "Gamma_Delta"], dtype=object
                    ),
                    dtype=string_dtype,
                )
                network.create_dataset("i", data=np.asarray([0], dtype=np.uint16))
                network.create_dataset("j", data=np.asarray([1], dtype=np.uint16))
                network.create_dataset(
                    "seq_lens", data=np.asarray([2, 2], dtype=np.uint16)
                )
                for name in ("g_score", "l_score"):
                    network.create_dataset(
                        name, data=np.asarray([10], dtype=np.float32)
                    )
                for name in ("g_len", "l_len"):
                    network.create_dataset(
                        name, data=np.asarray([2], dtype=np.uint16)
                    )

            settings = {
                "SAVED_LAYOUT_DIR": str(layout_root),
                "TARGET_CACHE_PATH": relative_cache,
                "TARGET_CACHE_MODE": "new",
                "NODE_FASTA_FILE": str(fasta_path),
                "SEQUENCES_FILE": str(fasta_path),
                "INPUT_HDF5": str(network_path),
                "ALIGNMENT_REFERENCE": "",
                "MSA_FILE": "",
                "ALIGNMENT_SCORE": "global",
                "NORM_MODE": "alignment_length",
                "UMAP_MODE": False,
                "UMAP_NEIGHBORS": 15,
                "TOP_EDGE_PERCENT": None,
                "SIMILARITY_THRESHOLD": 0.1,
                "BOX_SCALE": 2.0,
            }
            expected_positions = np.asarray(
                [[0.0, 0.0], [1.0, 1.0]], dtype=np.float32
            )
            preexisting_cache_folder = layout_root / "target"
            preexisting_cache_folder.mkdir(parents=True)
            (preexisting_cache_folder / fasta_path.name).write_bytes(
                fasta_path.read_bytes()
            )
            with mock.patch.multiple(emapssn_viewer.cfg, **settings), mock.patch.object(
                layout_engine,
                "calculate_layout",
                return_value=(expected_positions, 10.0),
            ):
                created = emapssn_viewer.MainViewer.__new__(emapssn_viewer.MainViewer)
                created.load_and_simulate()

            self.assertEqual(
                created.full_headers, ["Alpha_Beta", "Gamma_Delta"]
            )
            self.assertEqual(created.sequences_map["Alpha_Beta"], "AA")
            np.testing.assert_array_equal(created.node_render_order, [0, 1])

            cache_path = layout_root / "target" / "layout.h5"
            manifest_path = (
                layout_root / "target" / Cache_Manifest.MANIFEST_FILENAME
            )
            self.assertTrue(cache_path.exists())
            self.assertTrue(manifest_path.exists())
            fasta_backup_path = layout_root / "target" / fasta_path.name
            self.assertTrue(fasta_backup_path.exists())
            self.assertEqual(
                fasta_backup_path.read_text(encoding="utf-8"),
                ">Alpha_Beta\nAA\n>Gamma_Delta\nCC\n",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with h5py.File(cache_path, "r") as cache:
                self.assertEqual(
                    cache.attrs["cache_manifest_id"], manifest["manifest_id"]
                )

            with h5py.File(cache_path, "r+") as cache:
                cache.create_dataset(
                    "node_render_order", data=np.asarray([1, 0], dtype=np.int32)
                )

            settings["TARGET_CACHE_MODE"] = "existing"
            with mock.patch.multiple(emapssn_viewer.cfg, **settings):
                loaded = emapssn_viewer.MainViewer.__new__(emapssn_viewer.MainViewer)
                loaded.load_and_simulate()
            np.testing.assert_allclose(loaded.pos, expected_positions)
            np.testing.assert_array_equal(loaded.node_render_order, [1, 0])


if __name__ == "__main__":
    unittest.main()
