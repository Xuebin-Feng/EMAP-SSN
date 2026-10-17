"""Viewer settings documents (desktop.Viewer_State): resolve_viewer_document
and validate_viewer_document take every generation setting from the selected
cache's verified provenance, keep Viewer preferences (alignment, visualization)
as written, and reject legacy, misplaced, mismatched or tampered inputs; layout
exports leave visual preferences out."""
import copy
import hashlib
import json
import os
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
    alias_form,
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


class AliasFormTests(unittest.TestCase):
    """alias_form writes a path the way the Config's Directories tab does: under the deepest base folder it
    is in, as that folder's alias; failing that relative to the project root; failing that in full."""

    def setUp(self):
        # Only strings are compared; nothing here is read from the disk.
        self.root = pathlib.Path(os.path.abspath(os.sep)) / "alias_project"
        self.bases = {
            "INPUT_FILE_DIR": str(self.root / "Input_Files"),
            "CACHE_FILE_DIR": str(self.root / "Cache_Files"),
            "ANALYSIS_RESULT_DIR": str(self.root / "Analysis_Results"),
            "EMBED_DIR": str(self.root / "Embeddings"),
        }

    def form(self, path, bases=None, root="default"):
        return alias_form(str(path), self.bases if bases is None else bases,
                          str(self.root) if root == "default" else root)

    def test_each_base_folder_names_the_files_under_it_and_the_rest_is_written_with_slashes(self):
        for alias, key in ALIASES.items():
            with self.subTest(alias=alias):
                folder = pathlib.Path(self.bases[key])
                self.assertEqual(self.form(folder / "x.h5"), f"{alias}/x.h5")
                self.assertEqual(self.form(folder / "Networks_EValues" / "deep" / "x.h5"),
                                 f"{alias}/Networks_EValues/deep/x.h5")
                # The folder itself is the bare alias, as the tab writes it.
                self.assertEqual(self.form(folder), alias)

    def test_the_deepest_base_names_a_file_and_the_first_alias_when_two_are_one_folder(self):
        nested = dict(self.bases, EMBED_DIR=str(self.root / "Input_Files" / "Embeddings"),
                      ANALYSIS_RESULT_DIR=str(self.root / "Input_Files" / "Embeddings" / "Results"))
        inside = self.root / "Input_Files"
        self.assertEqual(self.form(inside / "Embeddings" / "e.h5", nested), "$embedding$/e.h5")
        self.assertEqual(self.form(inside / "Embeddings" / "Results" / "r.csv", nested), "$analysis_result$/r.csv")
        self.assertEqual(self.form(inside / "Sequence_Sets" / "s.fasta", nested), "$input_file$/Sequence_Sets/s.fasta")
        # The deeper one wins whichever order the folders are listed in.
        reordered = dict(reversed(list(nested.items())))
        self.assertEqual(self.form(inside / "Embeddings" / "e.h5", reordered), "$embedding$/e.h5")
        # One folder under two aliases: the first of them.
        same = dict(self.bases, EMBED_DIR=self.bases["INPUT_FILE_DIR"])
        self.assertEqual(self.form(inside / "x.h5", same), "$input_file$/x.h5")

    def test_a_folder_whose_name_only_starts_like_a_base_is_not_under_it(self):
        self.assertEqual(self.form(self.root / "Input_Files2" / "x.h5"), "Input_Files2/x.h5")
        self.assertEqual(self.form(self.root / "Input_Files_backup" / "x.h5"), "Input_Files_backup/x.h5")

    def test_case_and_slashes_do_not_matter_where_the_platform_ignores_them(self):
        path = str(self.root / "INPUT_files" / "Networks_EValues" / "X.h5")
        if os.path.normcase("A") == "a":  # Windows: case-insensitive, and either slash is one
            self.assertEqual(self.form(path), "$input_file$/Networks_EValues/X.h5")
            self.assertEqual(self.form(path.replace("\\", "/")), "$input_file$/Networks_EValues/X.h5")
            slashes = {key: value.replace("\\", "/").upper() for key, value in self.bases.items()}
            self.assertEqual(self.form(path, slashes), "$input_file$/Networks_EValues/X.h5")
            # The file's own spelling is kept, not folded.
            self.assertEqual(self.form(str(self.root / "input_files" / "ReadMe.TXT")), "$input_file$/ReadMe.TXT")
        else:  # a case-sensitive file system: another case is another folder
            self.assertEqual(self.form(path), "INPUT_files/Networks_EValues/X.h5")
        self.assertEqual(self.form(self.root / "Input_Files" / "Networks_EValues" / "X.h5"),
                         "$input_file$/Networks_EValues/X.h5")

    @unittest.skipUnless(os.name == "nt", "drive letters and UNC shares are Windows paths")
    def test_another_drive_is_not_under_a_base(self):
        other = "D:" if self.root.drive.upper() != "D:" else "E:"
        self.assertEqual(self.form(other + "\\alias_project\\Input_Files\\x.h5"),
                         other + "\\alias_project\\Input_Files\\x.h5")
        self.assertEqual(self.form("\\\\server\\share\\Input_Files\\x.h5", {"INPUT_FILE_DIR": "\\\\server\\share\\Input_Files"},
                                   None), "$input_file$/x.h5")

    def test_without_a_base_a_file_in_the_project_is_relative_to_its_root_and_any_other_is_absolute(self):
        self.assertEqual(self.form(self.root / "somewhere" / "deep" / "c.h5"), "somewhere/deep/c.h5")
        self.assertEqual(self.form(self.root / "c.h5"), "c.h5")
        outside = self.root.parent / "other_project" / "Input_Files" / "x.h5"
        self.assertEqual(self.form(outside), str(outside))
        # The root itself is no file's relative path, and a root that is not given leaves the path in full.
        self.assertEqual(self.form(self.root), str(self.root))
        self.assertEqual(self.form(self.root / "c.h5", root=None), str(self.root / "c.h5"))
        self.assertEqual(self.form(self.root / "c.h5", root=""), str(self.root / "c.h5"))
        # A path with ".." in it is the path it leads to; a relative one is made absolute first.
        self.assertEqual(self.form(self.root / "Cache_Files" / ".." / "Input_Files" / "x.h5"), "$input_file$/x.h5")
        self.assertEqual(self.form("relative.h5", {}, None), os.path.abspath("relative.h5"))
        self.assertEqual(self.form("", self.bases), "")

    def test_the_base_folders_are_resolved_as_the_viewer_resolves_them(self):
        # As the settings hold them: a folder of the project root, or an alias of another base.
        raw = {"INPUT_FILE_DIR": "inputs", "CACHE_FILE_DIR": "Cache_Files", "ANALYSIS_RESULT_DIR": "Analysis_Results",
               "EMBED_DIR": "$input_file$/Embeddings"}
        self.assertEqual(self.form(self.root / "inputs" / "Embeddings" / "e.h5", raw), "$embedding$/e.h5")
        self.assertEqual(self.form(self.root / "inputs" / "s.fasta", raw), "$input_file$/s.fasta")
        self.assertEqual(self.form(self.root / "Cache_Files" / "v.h5", raw), "$cache_file$/v.h5")
        # A base that is missing, empty or cannot be resolved is left out, and the others still work.
        for broken in ({"INPUT_FILE_DIR": None}, {"INPUT_FILE_DIR": ""}, {"INPUT_FILE_DIR": "$cache_file$/$input_file$"},
                       {"INPUT_FILE_DIR": "$input_file$/loop"}):
            with self.subTest(broken=broken):
                bases = dict(self.bases, **broken)
                self.assertEqual(self.form(self.root / "Cache_Files" / "v.h5", bases), "$cache_file$/v.h5")
                self.assertEqual(self.form(self.root / "Input_Files" / "s.fasta", bases), "Input_Files/s.fasta")
        self.assertEqual(self.form(self.root / "Cache_Files" / "v.h5", {"CACHE_FILE_DIR": self.bases["CACHE_FILE_DIR"]}),
                         "$cache_file$/v.h5")
        # A base that needs a missing one cannot be written, and costs nothing else.
        dependent = {"EMBED_DIR": "$input_file$/Embeddings", "CACHE_FILE_DIR": self.bases["CACHE_FILE_DIR"]}
        self.assertEqual(self.form(self.root / "Cache_Files" / "v.h5", dependent), "$cache_file$/v.h5")
        self.assertEqual(self.form(self.root / "Input_Files" / "Embeddings" / "e.h5", dependent),
                         "Input_Files/Embeddings/e.h5")
        # No base folders at all (None or an empty mapping): the root, then the full path.
        self.assertEqual(alias_form(str(self.root / "c.h5"), None, str(self.root)), "c.h5")
        self.assertEqual(alias_form(str(self.root / "c.h5"), {}, str(self.root)), "c.h5")

    def test_resolving_the_alias_form_gives_the_path_back(self):
        values = copy.deepcopy(DEFAULTS)
        values.update(self.bases, NODE_FASTA_FILE="set.fasta")
        for path in (self.root / "Input_Files" / "Networks_EValues" / "net.h5",
                     self.root / "Cache_Files" / "Saved_Layouts" / "c" / "version_00.h5",
                     self.root / "Analysis_Results" / "r" / "x.h5", self.root / "Embeddings" / "e.h5"):
            with self.subTest(path=path.name):
                values["INPUT_HDF5"] = self.form(path)
                self.assertIn("$", values["INPUT_HDF5"])
                resolved = normalize_viewer_paths(values, self.root, require_cache=False)
                self.assertEqual(pathlib.Path(resolved["INPUT_HDF5"]), path)


if __name__ == "__main__":
    unittest.main()
