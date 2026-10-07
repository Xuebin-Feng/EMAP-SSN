"""Cache metadata (utilities.Cache_Metadata.read_cache_metadata): cache
attributes, generation parameters and the folder manifest, cached per file
state and re-read when the cache or manifest changes, and the cache settings
the desktop Viewer (2D only) and opt_vr (3D too) resolve from them."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import h5py
import Cache_Manifest
from utilities.Cache_Metadata import read_cache_metadata
from tests.layout_fixtures import write_inputs


def _build_settings_cache(root, dimensions):
    """Publish a minimal cache of `dimensions` for the inputs write_inputs wrote."""
    manifest = Cache_Manifest.build_manifest_for_files(
        str(root / "set.fasta"), str(root / "network.h5"),
        alignment_score="global", normalization="alignment_length",
        similarity_threshold=0.1, layout_dimensions=dimensions)
    folder = root / f"cache_{dimensions}d"
    Cache_Manifest.write_manifest_atomic(folder, manifest)
    params = json.dumps({"UMAP_MODE": False, "UMAP_NEIGHBORS": 15, "UMAP_MIN_DIST": 0.1,
                         "BOX_SCALE": 2.0, "SIMILARITY_THRESHOLD": 0.1},
                        sort_keys=True, separators=(",", ":"))
    path = folder / "version_00.h5"
    with h5py.File(path, "w") as cache:
        cache.attrs["cache_manifest_id"] = manifest["manifest_id"]
        cache.attrs["layout_compatibility_json"] = params
        cache.attrs["layout_compatibility_id"] = hashlib.sha256(params.encode()).hexdigest()
    return str(path)


class MetadataTests(unittest.TestCase):
    def test_attributes_manifest_validation_and_refresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_inputs(root)
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
            # Metadata is cached per file signature: give every rewrite its own
            # mtime, as a rewrite within one clock tick may keep size and mtime.
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
            self.assertEqual(read_cache_metadata(str(path))["status"], "invalid")
            with h5py.File(path, "r+") as cache:
                del cache.attrs["layout_compatibility_json"]
                del cache.attrs["layout_compatibility_id"]
            os.utime(path, ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(read_cache_metadata(str(path))["status"], "partial")
            (folder / "cache_manifest.json").write_text("{")
            self.assertEqual(read_cache_metadata(str(path))["status"], "invalid")

    def test_desktop_viewer_names_a_3d_cache_as_one(self):
        from desktop.Viewer_State import ViewerSettingsError, resolve_cache_settings

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_inputs(root)
            # The 2D twin resolves, so the fixture itself is a valid cache.
            self.assertEqual(
                resolve_cache_settings(_build_settings_cache(root, 2))["SIMILARITY_THRESHOLD"], 0.1)
            with self.assertRaises(ViewerSettingsError) as caught:
                resolve_cache_settings(_build_settings_cache(root, 3))
            message = str(caught.exception)
            self.assertIn("is a 3D layout cache (layout_mode 'physics_3d')", message)
            self.assertIn("LAYOUT_DIMENSIONS=2", message)
            self.assertNotIn("Cache provenance", message)

    def test_the_vr_viewer_reads_a_3d_cache_like_its_2d_twin(self):
        """allow_3d, which opt_vr passes, accepts the 3D caches the VR viewer opens."""
        from desktop.Viewer_State import resolve_cache_settings

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_inputs(root)
            self.assertEqual(
                resolve_cache_settings(_build_settings_cache(root, 3), allow_3d=True),
                resolve_cache_settings(_build_settings_cache(root, 2)),
            )

    def test_missing_files_and_absent_path(self):
        self.assertEqual(read_cache_metadata(None)["status"], "unavailable")
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(read_cache_metadata(str(Path(temp) / "missing.h5"))["status"], "partial")

if __name__ == "__main__":
    unittest.main()
