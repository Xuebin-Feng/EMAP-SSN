"""Cache_Manifest: cache selection from settings, manifest identity and
compatibility hashing, canonical cache-folder names, network metadata and
schema validation, manifest discovery, cache paths, HDF5 cache validation,
and the Viewer reloading a generated cache only when its manifest binds it."""
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
from desktop.Viewer_State import resolve_selected_cache
from tests.layout_fixtures import (  # noqa: E402
    HeadlessSettingsFixture,
    make_compatibility,
    make_manifest,
    settings_document,
    write_inputs,
)


def write_network(path, model_name, network_type="alignment"):
    """An empty network with the datasets ``network_type`` requires."""
    required = {
        "alignment": ("seq_lens", "i", "j", "l_score", "l_len", "g_score", "g_len"),
        "blast": ("i", "j", "score"),
    }[network_type]
    with h5py.File(path, "w") as network:
        network.attrs["model_name"] = model_name
        network.create_dataset("headers", data=[b"A"])
        for dataset in required:
            network.create_dataset(dataset, data=[])


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

            settings = SimpleNamespace(
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

            cache_path = resolve_selected_cache(settings)
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
            legacy_path = resolve_selected_cache(settings)
            self.assertEqual(pathlib.Path(legacy_path).name, "saved_layout.h5")

            # LAYOUT_DIMENSIONS selects the generator's "_3D" folder, and the
            # keyword overrides it, which is how opt_vr finds the 2D fallback.
            settings.TARGET_CACHE_FILE = None
            settings.LAYOUT_DIMENSIONS = 3
            self.assertEqual(
                pathlib.Path(resolve_selected_cache(settings)).parent.name,
                "set_[model]_alignment_length_global_Score0.4_3D",
            )
            self.assertEqual(
                pathlib.Path(
                    resolve_selected_cache(settings, layout_dimensions=2)
                ).parent.name,
                "set_[model]_alignment_length_global_Score0.4",
            )

    def test_explicit_relative_path_resolves_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            settings = SimpleNamespace(
                SAVED_LAYOUT_DIR=str(root / "layouts"),
                TARGET_CACHE_PATH="chosen/layout.h5",
            )
            cache_path = resolve_selected_cache(settings)
            self.assertEqual(
                pathlib.Path(cache_path),
                root / "layouts" / "chosen" / "layout.h5",
            )

            settings.TARGET_CACHE_PATH = "../escape.h5"
            with self.assertRaises(Cache_Manifest.CacheManifestError):
                resolve_selected_cache(settings)


class ManifestIdentityTests(unittest.TestCase):
    def test_umap_revision_rejects_old_caches_and_reuses_corrected_caches(self):
        current = make_compatibility(umap_mode=True)
        self.assertEqual(current['umap_knn_revision'], 2)
        self.assertEqual(current['edge_filter']['value'], 15)
        for revision in (None, 1):
            old = dict(current)
            if revision is None:
                old.pop('umap_knn_revision')
            else:
                old['umap_knn_revision'] = revision
            with self.assertRaises(Cache_Manifest.CacheManifestError):
                Cache_Manifest.validate_manifest(make_manifest(old), current)
        manifest = make_manifest(current)
        self.assertEqual(Cache_Manifest.validate_manifest(manifest, current), manifest)

    def test_physics_compatibility_has_no_umap_revision(self):
        expected = {
            'sequence_sha256': 'a' * 64, 'network_sha256': 'b' * 64,
            'network_type': 'alignment', 'alignment_score': 'global',
            'normalization': 'alignment_length', 'layout_mode': 'physics',
            'edge_filter': {'mode': 'similarity_threshold', 'value': 0.4},
        }
        self.assertEqual(make_compatibility(), expected)

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

    def test_version_gaps_and_occupied_directory_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            (root / "version_03.h5").write_bytes(b"cache")
            (root / "version_04.h5").mkdir()
            self.assertEqual(Cache_Manifest.next_cache_version_filename(root), "version_05.h5")

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


class CanonicalCacheNameTests(unittest.TestCase):
    """build_canonical_cache_name gives the folder that the desktop Viewer and
    opt_vr look up (resolve_selected_cache), so existing caches are found only
    while these names stay exactly the same. (DIAMOND networks' [DIAMOND] label
    is the one change; resolve_selected_cache also tries their old [BLAST] name.)"""

    def canonical_name(self, model_name, network_type="alignment",
                       sequence_path="d/my set.fasta", **settings):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "network.h5"
            write_network(network_path, model_name, network_type)
            return Cache_Manifest.build_canonical_cache_name(
                sequence_path, network_path, network_type, **settings
            )

    def test_alignment_name_lists_the_score_settings_and_the_top_filter(self):
        self.assertEqual(
            self.canonical_name(
                "esm/c:300m", alignment_score="local",
                normalization="average_sequence", top_edge_percent=12.5,
                similarity_threshold=0.3,
            ),
            "my set_[esm_c_300m]_average_sequence_local_Top12.5Pct",
        )

    def test_umap_name_records_neighbors_and_ignores_edge_filters(self):
        self.assertEqual(
            self.canonical_name(
                "esm/c:300m", alignment_score="local",
                normalization="average_sequence", umap_mode=True,
                umap_neighbors=15, top_edge_percent=12.5,
                similarity_threshold=0.3,
            ),
            "my set_[esm_c_300m]_average_sequence_local_UMAP_k15",
        )

    def test_blast_name_omits_score_settings_and_formats_thresholds_as_floats(self):
        for threshold, expected in ((1e-5, "set_[BLAST]_Score1e-05"),
                                    (10, "set_[BLAST]_Score10.0")):
            with self.subTest(threshold=threshold):
                self.assertEqual(
                    self.canonical_name(
                        "BLAST", "blast", sequence_path="set.fasta",
                        alignment_score="global",
                        normalization="alignment_length",
                        similarity_threshold=threshold,
                    ),
                    expected,
                )

    def test_no_active_edge_filter_adds_no_suffix(self):
        for threshold in (None, "", "None"):
            with self.subTest(threshold=threshold):
                self.assertEqual(
                    self.canonical_name(
                        "BLAST", "blast", sequence_path="set.fasta",
                        top_edge_percent=None, similarity_threshold=threshold,
                    ),
                    "set_[BLAST]",
                )


def write_blast_network(path, search_program=None):
    """An empty E-value network, with search_program recorded when given."""
    write_network(path, "BLAST", "blast")
    if search_program is not None:
        with h5py.File(path, "a") as network:
            network.attrs["search_program"] = search_program


class DiamondCacheLabelTests(unittest.TestCase):
    """Imported DIAMOND networks keep model_name BLAST but name their caches [DIAMOND]."""

    def name(self, search_program, *, legacy=False, network_type="blast"):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "network.h5"
            if network_type == "blast":
                write_blast_network(network_path, search_program)
            else:
                write_network(network_path, "esm2", "alignment")
                with h5py.File(network_path, "a") as network:
                    network.attrs["search_program"] = search_program
            return Cache_Manifest.build_canonical_cache_name(
                "set.fasta", network_path, network_type, similarity_threshold=10,
                legacy_model_label=legacy,
            )

    def test_only_a_recorded_diamond_search_changes_the_label(self):
        # h5py reads plain bytes back as str; only fixed-length strings stay bytes.
        for program, label in (
            ("DIAMOND", "DIAMOND"), (np.bytes_(b"DIAMOND"), "DIAMOND"), (" DIAMOND ", "DIAMOND"),
            ("BLASTP", "BLAST"), ("Unknown", "BLAST"), (None, "BLAST"),
            ("diamond", "BLAST"), (np.int64(7), "BLAST"),
        ):
            with self.subTest(program=program):
                self.assertEqual(self.name(program), f"set_[{label}]_Score10.0")

    def test_legacy_label_and_alignment_networks_keep_the_model_name(self):
        self.assertEqual(self.name("DIAMOND", legacy=True), "set_[BLAST]_Score10.0")
        self.assertEqual(self.name("DIAMOND", network_type="alignment"), "set_[esm2]_Score10.0")

    def test_label_helper_reads_paths_and_open_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "network.h5"
            write_blast_network(network_path, "DIAMOND")
            self.assertEqual(Cache_Manifest.network_name_label(network_path), "DIAMOND")
            with h5py.File(network_path, "r") as network:
                self.assertEqual(Cache_Manifest.network_name_label(network), "DIAMOND")
            write_network(network_path, "esm/c:300m")
            self.assertEqual(Cache_Manifest.network_name_label(network_path), "esm_c_300m")

    def test_label_matches_what_the_importer_records(self):
        from utilities.BLAST_Tabular import DIAMOND_PROGRAM, SearchHeader
        self.assertEqual(Cache_Manifest.DIAMOND_SEARCH_PROGRAM, DIAMOND_PROGRAM)
        self.assertEqual(SearchHeader(program=DIAMOND_PROGRAM).network_tag, "DIAMOND")

    def test_selection_by_name_still_finds_folders_named_before_the_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            write_blast_network(root / "network.h5", "DIAMOND")
            layouts = root / "layouts"
            settings = SimpleNamespace(
                SAVED_LAYOUT_DIR=str(layouts), TARGET_CACHE_PATH=None, TARGET_CACHE_FILE=None,
                NODE_FASTA_FILE=str(root / "set.fasta"), SEQUENCES_FILE="", INPUT_HDF5=str(root / "network.h5"),
                ALIGNMENT_SCORE="global", NORM_MODE="alignment_length", UMAP_MODE=False, UMAP_NEIGHBORS=15,
                TOP_EDGE_PERCENT=None, SIMILARITY_THRESHOLD=10.0,
            )

            def folder(**kwargs):
                return pathlib.Path(resolve_selected_cache(settings, **kwargs)).parent.name

            self.assertEqual(folder(), "set_[DIAMOND]_Score10.0")
            (layouts / "set_[BLAST]_Score10.0").mkdir(parents=True)
            (layouts / "set_[BLAST]_Score10.0_3D").mkdir()
            self.assertEqual(folder(), "set_[BLAST]_Score10.0")
            self.assertEqual(folder(layout_dimensions=3), "set_[BLAST]_Score10.0_3D")
            (layouts / "set_[DIAMOND]_Score10.0").mkdir()
            self.assertEqual(folder(), "set_[DIAMOND]_Score10.0")
            self.assertEqual(folder(layout_dimensions=3), "set_[BLAST]_Score10.0_3D")


class DiamondCacheReuseTests(HeadlessSettingsFixture, unittest.TestCase):
    """A layout job finds compatible caches by manifest, whatever their folder is called."""

    def test_new_diamond_caches_get_the_label_and_old_folders_are_reused(self):
        write_inputs(self.root)
        network = self.root / "network.h5"
        network.unlink()
        with h5py.File(network, "w") as handle:
            handle.attrs["model_name"] = "BLAST"
            handle.attrs["search_program"] = "DIAMOND"
            handle.create_dataset("headers", data=np.asarray(["Alpha_Beta", "Gamma_Delta"], dtype=object),
                                  dtype=h5py.string_dtype("utf-8"))
            handle.create_dataset("i", data=np.asarray([0], dtype=np.uint16))
            handle.create_dataset("j", data=np.asarray([1], dtype=np.uint16))
            handle.create_dataset("score", data=np.asarray([50.0], dtype=np.float32))
        document = settings_document(self.root)
        document["output"]["CACHE_NAME_MODE"] = "auto"

        first = pathlib.Path(self.generate(document).cache_path)
        self.assertEqual(first.parent.name, "set_[DIAMOND]_Score0.1")

        # A folder made before the label existed holds the same manifest.
        legacy = first.parent.with_name("set_[BLAST]_Score0.1")
        first.parent.rename(legacy)
        second = pathlib.Path(self.generate(document).cache_path)
        self.assertEqual(second, legacy / "version_01.h5")
        self.assertFalse(first.parent.exists())


class NetworkMetadataTests(unittest.TestCase):
    """model_name selects the network type, and the type the required datasets."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = pathlib.Path(self.temp.name) / "network.h5"

    def test_model_name_must_be_present_nonblank_utf8_text(self):
        # A fixed-length string attribute reads back as bytes. h5py decodes a
        # variable-length one with surrogateescape, so its invalid UTF-8 reads
        # back as lone surrogates (b"\xff" as "\udcff") rather than failing.
        for model_name, message in ((None, "missing the required root attribute 'model_name'"),
                                    ("  ", "blank required root attribute 'model_name'"),
                                    (np.bytes_(b"\xff"), "'model_name' attribute that is not valid UTF-8"),
                                    (7, "non-text 'model_name' attribute")):
            with self.subTest(model_name=model_name):
                write_network(self.path, "model")
                with h5py.File(self.path, "a") as network:
                    del network.attrs["model_name"]
                    if model_name is not None:
                        network.attrs["model_name"] = model_name
                with self.assertRaisesRegex(Cache_Manifest.NetworkMetadataError, message):
                    Cache_Manifest.read_network_metadata(self.path)
        with self.subTest(model_name="variable-length b'\\xff'"):
            write_network(self.path, "model")
            with h5py.File(self.path, "a") as network:
                network.attrs.create("model_name", b"\xff", dtype=h5py.string_dtype(encoding="utf-8"))
            with h5py.File(self.path, "r") as network:
                self.assertEqual(network.attrs["model_name"], "\udcff")
                for source in (self.path, network):
                    with self.assertRaisesRegex(Cache_Manifest.NetworkMetadataError,
                                                "'model_name' attribute that is not valid UTF-8"):
                        Cache_Manifest.read_network_metadata(source)

    def test_blast_model_name_is_trimmed_and_case_insensitive(self):
        write_network(self.path, " Blast ", "blast")
        metadata = Cache_Manifest.validate_network_schema(self.path, "blast")
        self.assertEqual(metadata, Cache_Manifest.NetworkMetadata("Blast", "blast"))
        write_network(self.path, "blast-like model")
        self.assertEqual(
            Cache_Manifest.validate_network_schema(self.path).network_type, "alignment"
        )

    def test_missing_datasets_of_the_declared_type_are_listed(self):
        write_network(self.path, "model")
        with h5py.File(self.path, "a") as network:
            del network["l_len"]
        with self.assertRaisesRegex(
            Cache_Manifest.NetworkMetadataError,
            r"model_name='model' \(alignment\) but is missing required dataset\(s\): l_len\.",
        ):
            Cache_Manifest.validate_network_schema(self.path)

    def test_expected_type_must_match_the_declared_type(self):
        write_network(self.path, "model")
        with self.assertRaisesRegex(
            Cache_Manifest.NetworkMetadataError,
            "declares network type 'alignment' through model_name='model', "
            "not the expected type 'blast'",
        ):
            Cache_Manifest.validate_network_schema(self.path, "blast")


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


class ViewerCacheIntegrationTests(unittest.TestCase):
    def test_new_cache_and_manifest_are_reloaded_only_when_bound(self):
        import Layout_Engine_SSN as layout_engine
        import EMAPSSN_Viewer

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
            from Layout_Cache_Generator import LayoutGenerationSettings, generate_layout_cache
            from types import SimpleNamespace
            gen_settings = LayoutGenerationSettings.from_namespace(
                SimpleNamespace(**settings),
                cache_filename="layout.h5",
                target_cache_path=str(layout_root / "target" / "layout.h5"),
            )
            with mock.patch.object(
                layout_engine,
                "calculate_layout",
                return_value=(expected_positions, 10.0),
            ):
                generate_layout_cache(gen_settings)

            settings["TARGET_CACHE_MODE"] = "existing"
            # load_and_simulate also sets CACHE_MANIFEST_ID and INPUT_IS_EVALUE.
            with mock.patch.multiple(EMAPSSN_Viewer.cfg, create=True, CACHE_MANIFEST_ID=None,
                                     INPUT_IS_EVALUE=False, **settings):
                created = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
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
            # load_and_simulate also sets CACHE_MANIFEST_ID and INPUT_IS_EVALUE.
            with mock.patch.multiple(EMAPSSN_Viewer.cfg, create=True, CACHE_MANIFEST_ID=None,
                                     INPUT_IS_EVALUE=False, **settings):
                loaded = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
                loaded.load_and_simulate()
            np.testing.assert_allclose(loaded.pos, expected_positions)
            np.testing.assert_array_equal(loaded.node_render_order, [1, 0])


if __name__ == "__main__":
    unittest.main()
