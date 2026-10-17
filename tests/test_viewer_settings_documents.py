"""Viewer settings documents (desktop.Viewer_State): resolve_viewer_document
and validate_viewer_document take every generation setting from the selected
cache's verified provenance, keep Viewer preferences (alignment, visualization)
as written, and reject legacy, misplaced, mismatched or tampered inputs; layout
exports leave visual preferences out."""
import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

import h5py

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Cache_Manifest
from desktop.Viewer_State import (
    ALIASES,
    DEFAULTS,
    DIRECTORY_PROFILE_DEFAULTS,
    EMBEDDING_ALIAS,
    VIEWER_SECTIONS,
    ViewerSettingsError,
    decode_document,
    encode_document,
    get_viewer_settings_schema,
    normalize_viewer_paths,
    resolve_viewer_document,
    validate_viewer_document,
)
from utilities.Headless_Settings import export_config_settings
from tests.embedding_fixtures import write_embedding_file
from tests.layout_fixtures import HeadlessSettingsFixture


class ViewerSettingsDocumentTests(HeadlessSettingsFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.saved = self.saved_config()

    def test_selected_cache_controls_runtime_without_leaking_into_json(self):
        path, viewer = self.cache_and_viewer(BOX_SCALE=3.5, SIMILARITY_THRESHOLD=0.2)
        self.saved.update(SIMILARITY_THRESHOLD=999, BOX_SCALE=8, UMAP_MODE=True)
        (self.root / "viewer_settings.json").write_text(json.dumps(self.saved))
        runtime = resolve_viewer_document(viewer, self.root)
        self.assertEqual(runtime["BOX_SCALE"], 3.5)
        self.assertEqual(runtime["SIMILARITY_THRESHOLD"], 0.2)
        self.assertFalse(runtime["UMAP_MODE"])
        normalized = validate_viewer_document(viewer, self.root)
        self.assertEqual(set(normalized), {"schema_version", "kind", *VIEWER_SECTIONS})
        self.assertEqual(encode_document("viewer", runtime), normalized)
        flat = decode_document(normalized, "viewer")
        for key in ("BOX_SCALE", "UMAP_MODE", "ALIGNMENT_SCORE", "NORM_MODE", "SIMILARITY_THRESHOLD", "CACHE_FILENAME"):
            self.assertNotIn(key, flat)
        # A second selected cache supplies a different configuration.
        self.saved.update(SIMILARITY_THRESHOLD=0.1, UMAP_MODE=False)
        (self.root / "viewer_settings.json").write_text(json.dumps(self.saved))
        _, second = self.cache_and_viewer(BOX_SCALE=4.5, SIMILARITY_THRESHOLD=0.3)
        self.assertEqual(resolve_viewer_document(second, self.root)["BOX_SCALE"], 4.5)

    def test_layout_export_ignores_unrelated_visual_preferences(self):
        self.saved.update(NODE_SIZE=999, EDGE_COLOR="invalid color", ALIGNMENT_REFERENCE="absent")
        (self.root / "viewer_settings.json").write_text(json.dumps(self.saved))
        document = export_config_settings("layout", self.root)["settings_document"]
        self.assertNotIn("visualization", document)
        self.assertNotIn("alignment", document)
        self.assertNotIn("embedding", document)
        self.assertEqual(document["simulation"]["MAX_STEPS"], self.saved["MAX_STEPS"])

    def test_empty_and_old_export_overlays_fail(self):
        path = self.root / "bad-overlay.json"
        for kind in ("viewer", "layout"):
            for content in ({}, {"TARGET_CACHE_PATH": "other.h5"},
                            {"schema_version": 1, "kind": kind}):
                path.write_text(json.dumps(content))
                with self.assertRaisesRegex(ValueError, "Re-export"):
                    export_config_settings(kind, self.root, settings_path=path)

    def test_top_filter_and_umap(self):
        for overrides in (dict(TOP_EDGE_PERCENT=50.0, SIMILARITY_THRESHOLD=None),
                          dict(UMAP_MODE=True, SIMILARITY_THRESHOLD=None, UMAP_NEIGHBORS=2)):
            with self.subTest(overrides=overrides):
                _, viewer = self.cache_and_viewer(**overrides)
                runtime = resolve_viewer_document(viewer, self.root)
                for key, value in overrides.items():
                    self.assertEqual(runtime[key], value)

    def test_blast_metadata(self):
        with h5py.File(self.root / "network.h5", "a") as network:
            for key in ("g_score", "l_score", "g_len", "l_len", "seq_lens"):
                del network[key]
            network.attrs["model_name"] = "blast"
            network.create_dataset("score", data=[10.0])
        _, viewer = self.cache_and_viewer(SIMILARITY_THRESHOLD=1e-5)
        self.assertEqual(resolve_viewer_document(viewer, self.root)["SIMILARITY_THRESHOLD"], 1e-5)

    def test_alignment_preferences_fasta_and_sparse(self):
        _, viewer = self.cache_and_viewer()
        for suffix in (".fasta", ".h5"):
            path = self.root / ("msa" + suffix)
            if suffix == ".fasta":
                path.write_text(">reference\nAA\n")
            else:
                with h5py.File(path, "w") as msa:
                    msa.create_dataset("headers", data=[b"reference"])
            viewer["alignment"].update(MSA_FILE=str(path), ALIGNMENT_REFERENCE="reference",
                                      FILTER_MIN_OCCUPANCY=25, ALIGNMENT_OFFSET=7)
            runtime = resolve_viewer_document(viewer, self.root)
            for key, value in viewer["alignment"].items():
                self.assertEqual(runtime[key], value)
            viewer["alignment"]["ALIGNMENT_REFERENCE"] = "absent"
            original = copy.deepcopy(viewer)
            normalized = validate_viewer_document(viewer, self.root)
            runtime = resolve_viewer_document(normalized, self.root)
            self.assertEqual(viewer, original)
            self.assertEqual(normalized["alignment"], original["alignment"])
            for key, value in original["alignment"].items():
                self.assertEqual(runtime[key], value)

    def test_embedding_preference_is_kept_and_resolved_in_its_folder(self):
        _, viewer = self.cache_and_viewer()
        folder = self.root / "embeddings"
        folder.mkdir()
        write_embedding_file(folder / "set_[model]_embeddings.h5",
                             [("Alpha_Beta", "AA"), ("Gamma_Delta", "CC")])
        viewer["directories"]["EMBED_DIR"] = str(folder)
        viewer["embedding"]["EMBED_FILE"] = "set_[model]_embeddings.h5"
        original = copy.deepcopy(viewer)
        runtime = resolve_viewer_document(viewer, self.root)
        self.assertEqual(pathlib.Path(runtime["EMBED_FILE"]), folder / "set_[model]_embeddings.h5")
        normalized = validate_viewer_document(viewer, self.root)
        self.assertEqual(viewer, original)
        self.assertEqual(pathlib.Path(normalized["embedding"]["EMBED_FILE"]),
                         folder / "set_[model]_embeddings.h5")
        self.assertEqual(resolve_viewer_document(normalized, self.root)["EMBED_FILE"],
                         runtime["EMBED_FILE"])
        # Empty selects no embedding.
        viewer["embedding"]["EMBED_FILE"] = ""
        self.assertEqual(resolve_viewer_document(viewer, self.root)["EMBED_FILE"], "")

    def test_documents_exported_before_the_embedding_input_still_decode(self):
        _, viewer = self.cache_and_viewer()
        del viewer["embedding"]
        del viewer["directories"]["EMBED_DIR"]
        runtime = resolve_viewer_document(viewer, self.root)
        self.assertEqual(runtime["EMBED_FILE"], "")
        self.assertEqual(pathlib.Path(runtime["EMBED_DIR"]), self.root / "Embeddings")
        normalized = validate_viewer_document(viewer, self.root)
        self.assertEqual(normalized["embedding"], {"EMBED_FILE": ""})
        self.assertIn("EMBED_DIR", normalized["directories"])
        # The schema agents read lets them leave both out, as these documents do.
        schema = get_viewer_settings_schema()
        self.assertNotIn("embedding", schema["required"])
        self.assertIn("alignment", schema["required"])
        self.assertEqual(schema["properties"]["embedding"]["required"], [])
        self.assertNotIn("EMBED_DIR", schema["properties"]["directories"]["required"])
        self.assertIn("MSA_DIR", schema["properties"]["directories"]["required"])
        self.assertIn("EMBED_FILE", schema["properties"]["embedding"]["properties"])

    def test_an_embedding_file_breaking_the_interface_refuses_launch(self):
        _, viewer = self.cache_and_viewer()
        incomplete = self.root / "incomplete_embeddings.h5"
        write_embedding_file(incomplete, [("Alpha_Beta", "AA")], complete=False)
        not_hdf5 = self.root / "not_hdf5.h5"
        not_hdf5.write_text(">Alpha_Beta\nAA\n")
        for path, pattern in ((incomplete, "EMBED_FILE: Embedding generation is incomplete"),
                              (not_hdf5, "EMBED_FILE: "),
                              (self.root / "absent.h5", "EMBED_FILE: file does not exist")):
            with self.subTest(path=path.name):
                viewer["embedding"]["EMBED_FILE"] = str(path)
                with self.assertRaisesRegex(ViewerSettingsError, pattern):
                    validate_viewer_document(viewer, self.root)

    def test_partial_embedding_coverage_is_accepted_at_launch(self):
        _, viewer = self.cache_and_viewer()
        partial = self.root / "partial_embeddings.h5"
        write_embedding_file(partial, [("Alpha_Beta", "AA"), ("Unrelated", "KK")])
        viewer["embedding"]["EMBED_FILE"] = str(partial)
        self.assertEqual(pathlib.Path(resolve_viewer_document(viewer, self.root)["EMBED_FILE"]), partial)

    def test_corrupt_missing_conflicting_provenance(self):
        path, viewer = self.cache_and_viewer()
        with h5py.File(path, "r") as cache:
            original = dict(cache.attrs)
        for alteration in ("missing", "hash", "conflict", "missing_box"):
            with self.subTest(alteration=alteration):
                with h5py.File(path, "a") as cache:
                    for key, value in original.items():
                        cache.attrs[key] = value
                    if alteration == "missing":
                        del cache.attrs["layout_compatibility_json"]
                    elif alteration == "hash":
                        cache.attrs["layout_compatibility_id"] = "bad"
                    else:
                        values = json.loads(cache.attrs["layout_compatibility_json"])
                        if alteration == "conflict":
                            values["UMAP_MODE"] = True
                        else:
                            del values["BOX_SCALE"]
                        text = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                        cache.attrs["layout_compatibility_json"] = text
                        cache.attrs["layout_compatibility_id"] = hashlib.sha256(text.encode()).hexdigest()
                with self.assertRaisesRegex(ViewerSettingsError, "Cache provenance"):
                    validate_viewer_document(viewer, self.root)

    def test_local_scores_normalized_by_alignment_length_are_refused(self):
        # Generation refuses this pair, so the cache that claims it is edited:
        # its folder manifest is rebound to local scores.
        path, viewer = self.cache_and_viewer()
        folder = pathlib.Path(path).parent
        manifest = Cache_Manifest.read_manifest(folder)
        manifest["compatibility"]["alignment_score"] = "local"
        manifest["manifest_id"] = Cache_Manifest.calculate_manifest_id(manifest["compatibility"])
        Cache_Manifest.write_manifest_atomic(folder, manifest)
        with h5py.File(path, "a") as cache:
            cache.attrs["cache_manifest_id"] = manifest["manifest_id"]
        with self.assertRaisesRegex(
            ViewerSettingsError, "NORM_MODE: alignment_length is unavailable for local scores"
        ):
            validate_viewer_document(viewer, self.root)

    def test_input_mismatch_and_legacy_documents(self):
        _, viewer = self.cache_and_viewer()
        for document in (decode_document(viewer, "viewer"), {**viewer, "kind": "layout"},
                         {**viewer, "schema_version": 1}, {**viewer, "network": {}}):
            with self.assertRaises((ValueError, ViewerSettingsError)):
                validate_viewer_document(document, self.root)
        malformed = copy.deepcopy(viewer)
        malformed["alignment"]["BOX_SCALE"] = 2
        with self.assertRaisesRegex(ViewerSettingsError, "misplaced"):
            validate_viewer_document(malformed, self.root)
        (self.root / "set.fasta").write_text(">different\nAA\n")
        with self.assertRaisesRegex(ViewerSettingsError, "[Mm]anifest"):
            validate_viewer_document(viewer, self.root)

    def test_manifest_binding_and_cache_headers_are_verified(self):
        path, viewer = self.cache_and_viewer()
        with h5py.File(path, "a") as cache:
            manifest_id = cache.attrs["cache_manifest_id"]
            cache.attrs["cache_manifest_id"] = "different-manifest"
        with self.assertRaisesRegex(ViewerSettingsError, "Cache provenance"):
            validate_viewer_document(viewer, self.root)
        with h5py.File(path, "a") as cache:
            cache.attrs["cache_manifest_id"] = manifest_id
            del cache["headers"]
            cache.create_dataset("headers", data=[b"wrong", b"identities"])
        with self.assertRaisesRegex(ViewerSettingsError, "[Hh]eader"):
            validate_viewer_document(viewer, self.root)


class EmbeddingRootAliasTests(unittest.TestCase):
    """$embedding$ is the fourth reusable root, resolved as the other three are."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name).resolve()

    def resolve(self, **changes):
        values = copy.deepcopy(DEFAULTS)
        # Placeholders: the two required inputs, which are not resolved against disk.
        values.update(NODE_FASTA_FILE="set.fasta", INPUT_HDF5="network.h5", **changes)
        return normalize_viewer_paths(values, self.root, require_cache=False)

    def test_the_alias_names_the_embedding_directory_and_roots_lead_the_defaults(self):
        self.assertEqual(EMBEDDING_ALIAS, "$embedding$")
        self.assertEqual(ALIASES[EMBEDDING_ALIAS], "EMBED_DIR")
        self.assertEqual(
            list(DIRECTORY_PROFILE_DEFAULTS)[:len(ALIASES)], list(ALIASES.values())
        )
        self.assertEqual(
            list(ALIASES.values()),
            ["INPUT_FILE_DIR", "CACHE_FILE_DIR", "ANALYSIS_RESULT_DIR", "EMBED_DIR"],
        )
        # The roots are folders of the project, not under one another.
        self.assertEqual(DIRECTORY_PROFILE_DEFAULTS["EMBED_DIR"], "Embeddings")

    def test_a_folder_or_file_under_the_alias_resolves_under_the_embedding_directory(self):
        resolved = self.resolve(
            EMBED_DIR="my_embeddings",
            HDF5_DIR="$embedding$\\sub",
            MSA_DIR="$embedding$/other/deeper",
            EMBED_FILE="$embedding$/set_[model]_embeddings.h5",
        )
        folder = self.root / "my_embeddings"
        self.assertEqual(pathlib.Path(resolved["EMBED_DIR"]), folder)
        self.assertEqual(pathlib.Path(resolved["HDF5_DIR"]), folder / "sub")
        self.assertEqual(pathlib.Path(resolved["MSA_DIR"]), folder / "other" / "deeper")
        self.assertEqual(
            pathlib.Path(resolved["EMBED_FILE"]), folder / "set_[model]_embeddings.h5"
        )
        # A bare file name is looked up in the embedding directory, as before.
        bare = self.resolve(EMBED_DIR="my_embeddings", EMBED_FILE="set_[model]_embeddings.h5")
        self.assertEqual(
            pathlib.Path(bare["EMBED_FILE"]), folder / "set_[model]_embeddings.h5"
        )
        # An absolute embedding directory stays where it is.
        elsewhere = self.root / "elsewhere"
        resolved = self.resolve(EMBED_DIR=str(elsewhere), HDF5_DIR="$embedding$/sub")
        self.assertEqual(pathlib.Path(resolved["EMBED_DIR"]), elsewhere)
        self.assertEqual(pathlib.Path(resolved["HDF5_DIR"]), elsewhere / "sub")

    def test_old_settings_still_resolve(self):
        # The default, as every settings file written so far stores it.
        resolved = self.resolve(EMBED_DIR="Embeddings")
        self.assertEqual(pathlib.Path(resolved["EMBED_DIR"]), self.root / "Embeddings")
        # A folder an old file placed under the input root.
        resolved = self.resolve(
            INPUT_FILE_DIR="inputs", EMBED_DIR="$input_file$/Embeddings"
        )
        self.assertEqual(
            pathlib.Path(resolved["EMBED_DIR"]), self.root / "inputs" / "Embeddings"
        )

    def test_every_alias_is_refused_anywhere_but_at_the_front(self):
        for alias, key in ALIASES.items():
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(ViewerSettingsError, "Invalid directory alias"):
                    self.resolve(HEADER_LIST_DIR=f"prefix/{alias}/folder")
                # A root that names itself is circular, and so refused too.
                with self.assertRaisesRegex(ViewerSettingsError, "Invalid directory alias"):
                    self.resolve(**{key: f"{alias}/folder"})


if __name__ == "__main__":
    unittest.main()
