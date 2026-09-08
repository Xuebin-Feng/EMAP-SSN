import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from tests.test_headless_settings import HeadlessSettingsTests
from desktop.Viewer_State import (
    LAYOUT_SECTIONS,
    VIEWER_SECTIONS,
    ViewerSettingsError,
    decode_document,
    encode_document,
    resolve_viewer_document,
    validate_viewer_document,
)
from utilities.Headless_Settings import export_config_settings


class ExecutionV2Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = HeadlessSettingsTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.saved = self.fixture.saved_config()

    def cache_and_viewer(self, **changes):
        layout = export_config_settings("layout", self.root)["settings_document"]
        for section, fields in LAYOUT_SECTIONS.items():
            for key in fields:
                if key in changes:
                    layout[section][key] = changes[key]
        result = self.fixture.generate(layout)
        overlay = self.root / "overlay.json"
        overlay.write_text(json.dumps({"schema_version": 2, "kind": "viewer",
                                      "inputs": {"TARGET_CACHE_PATH": result.cache_path}}))
        viewer = export_config_settings("viewer", self.root, settings_path=overlay)["settings_document"]
        return result.cache_path, viewer

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
            with self.assertRaisesRegex(ViewerSettingsError, "not found"):
                validate_viewer_document(viewer, self.root)

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


if __name__ == "__main__":
    unittest.main()
