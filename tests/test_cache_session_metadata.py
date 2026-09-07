import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import h5py
import Cache_Manifest
from mcp_server.Cache_Metadata import read_cache_metadata
from mcp_server.Viewer_Sessions import ensure_viewer_identity, session_alias, select_viewer_session
from tests.test_layout_cache_generator import _write_inputs


class MetadataTests(unittest.TestCase):
    def test_attributes_manifest_validation_and_refresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_inputs(root)
            manifest = Cache_Manifest.build_manifest_for_files(str(root / "set.fasta"), str(root / "network.h5"),
                alignment_score="global", normalization="alignment_length", similarity_threshold=0.1)
            folder = root / "cache"
            Cache_Manifest.write_manifest_atomic(folder, manifest)
            path = folder / "version_00.h5"
            params = '{"SPRING_K":5.0}'
            with h5py.File(path, "w") as cache:
                cache.attrs["cache_manifest_id"] = manifest["manifest_id"]
                cache.attrs["layout_compatibility_json"] = params
                cache.attrs["layout_compatibility_id"] = hashlib.sha256(params.encode()).hexdigest()
            result = read_cache_metadata(str(path))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["generation_parameters"], {"SPRING_K": 5.0})
            self.assertEqual(result["folder_manifest"], manifest)
            result["generation_parameters"]["SPRING_K"] = 99
            self.assertEqual(read_cache_metadata(str(path))["generation_parameters"]["SPRING_K"], 5.0)
            with h5py.File(path, "r+") as cache:
                cache.attrs["layout_compatibility_json"] = '{"SPRING_K":6.0}'
            self.assertEqual(read_cache_metadata(str(path))["status"], "invalid")
            with h5py.File(path, "r+") as cache:
                del cache.attrs["layout_compatibility_json"]
                del cache.attrs["layout_compatibility_id"]
            self.assertEqual(read_cache_metadata(str(path))["status"], "partial")
            (folder / "cache_manifest.json").write_text("{")
            self.assertEqual(read_cache_metadata(str(path))["status"], "invalid")

    def test_missing_files_and_absent_path(self):
        self.assertEqual(read_cache_metadata(None)["status"], "unavailable")
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(read_cache_metadata(str(Path(temp) / "missing.h5"))["status"], "partial")

    def test_identity_is_stable_and_alias_selection_rejects_collisions(self):
        viewer = SimpleNamespace()
        first = ensure_viewer_identity(viewer)
        self.assertEqual(ensure_viewer_identity(viewer), first)
        self.assertEqual(len(viewer.inspection_session_alias), 8)
        one = SimpleNamespace(session_id="a7c92f10-0000-4000-8000-000000000001")
        two = SimpleNamespace(session_id="a7c92f10-0000-4000-8000-000000000002")
        with mock.patch("mcp_server.Viewer_Sessions.discover_viewer_sessions", return_value=[one]):
            self.assertIs(select_viewer_session("a7c92f10"), one)
        with mock.patch("mcp_server.Viewer_Sessions.discover_viewer_sessions", return_value=[one, two]):
            with self.assertRaisesRegex(LookupError, "Ambiguous"):
                select_viewer_session("A7C92F10")
            self.assertIs(select_viewer_session(two.session_id), two)


if __name__ == "__main__":
    unittest.main()
