import json
import copy
import os
import pathlib
import subprocess
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
import SSN_Utils  # noqa: F401 - keep the authoritative preparer loaded across mocks
from Layout_Cache_Generator import (
    LayoutGenerationError,
    LayoutGenerationSettings,
    generate_layout_cache,
)


def _settings_document(temp_path, *, cache_filename="version_00.h5"):
    return {
        "DIRECTORIES": {"SAVED_LAYOUT_DIR": str(temp_path / "layouts")},
        "Layout_Cache_Generator.py": {
            "NODE_FASTA_FILE": str(temp_path / "set.fasta"),
            "INPUT_HDF5": str(temp_path / "network.h5"),
            "CACHE_FILENAME": cache_filename,
            "ALIGNMENT_SCORE": "global",
            "NORM_MODE": "alignment_length",
            "SIMILARITY_THRESHOLD": 0.1,
            "TOP_EDGE_PERCENT": None,
            "UMAP_MODE": False,
            "UMAP_NEIGHBORS": 15,
            "UMAP_MIN_DIST": 0.1,
            "PHYSICS_ENGINE": "Molecular Dynamics (Style)",
            "LAYOUT_DEVICE_SELECTION": "auto",
            "SPRING_K": 5.0,
            "COULOMB_K": 10.0,
            "COULOMB_CUTOFF": 30.0,
            "DAMPING": 0.9,
            "DT": 0.005,
            "MAX_STEPS": 10000,
            "RMSD_THRESHOLD": 0.005,
            "PERCENTAGE_DROP_THRESHOLD": 0.1,
            "RMSD_WINDOW": 50,
            "ENABLE_PROGRESSIVE_SIMULATION": False,
            "PACKING_GEOMETRY": "Square",
            "PACKING_GRID_SIZE": 20.0,
            "SGLD_MIN_K": 20,
            "SGLD_K_PERCENT": 0.01,
            "SGLD_START_TEMP": 1.5,
            "SGLD_NOISE_SCALE": 1.0,
        },
    }


def _write_inputs(temp_path):
    (temp_path / "set.fasta").write_text(
        ">Alpha?? Beta\nAA\n>Gamma##Delta\nCC\n", encoding="utf-8"
    )
    with h5py.File(temp_path / "network.h5", "w") as network:
        string_dtype = h5py.string_dtype("utf-8")
        network.attrs["model_name"] = "model"
        network.create_dataset(
            "headers",
            data=np.asarray(["Alpha_Beta", "Gamma_Delta"], dtype=object),
            dtype=string_dtype,
        )
        network.create_dataset("i", data=np.asarray([0], dtype=np.uint16))
        network.create_dataset("j", data=np.asarray([1], dtype=np.uint16))
        network.create_dataset("seq_lens", data=np.asarray([2, 2], dtype=np.uint16))
        for name in ("g_score", "l_score"):
            network.create_dataset(name, data=np.asarray([10], dtype=np.float32))
        for name in ("g_len", "l_len"):
            network.create_dataset(name, data=np.asarray([2], dtype=np.uint16))


class LayoutSettingsTests(unittest.TestCase):
    def test_schema_is_strict_and_hidden_coordinate_defaults_are_exported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            document = _settings_document(temp_path)
            settings = LayoutGenerationSettings.from_document(document)
            exported = settings.to_document(project_root=ROOT)

            payload = exported["Layout_Cache_Generator.py"]
            self.assertEqual(payload["CACHE_FILENAME"], "version_00.h5")
            self.assertEqual(payload["BOX_SCALE"], 2.0)
            self.assertEqual(payload["PACKING_PADDING"], 10.0)
            self.assertEqual(payload["MAX_FORCE_LIMIT"], 20.0)
            self.assertEqual(payload["MAX_TOTAL_REPULSION_FORCE"], 0.0)
            self.assertNotIn("NODE_SIZE", payload)
            self.assertNotIn("MSA_FILE", payload)
            self.assertNotIn("TARGET_CACHE_PATH", payload)

            document["Layout_Cache_Generator.py"]["NODE_SIZE"] = 10
            with self.assertRaisesRegex(
                LayoutGenerationError, "Unknown layout-generation setting"
            ):
                LayoutGenerationSettings.from_document(document)

    def test_missing_unsafe_and_wrong_typed_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            for key, value, message in (
                ("MAX_STEPS", None, "Missing layout-generation"),
                ("CACHE_FILENAME", "../escape.h5", "plain basename"),
                ("UMAP_MODE", "false", "JSON boolean"),
            ):
                with self.subTest(key=key):
                    document = _settings_document(temp_path)
                    if value is None:
                        del document["Layout_Cache_Generator.py"][key]
                    else:
                        document["Layout_Cache_Generator.py"][key] = value
                    with self.assertRaisesRegex(Exception, message):
                        LayoutGenerationSettings.from_document(document)


class LayoutCacheGenerationTests(unittest.TestCase):
    def test_duplicate_and_incompatible_manifest_folders_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            _write_inputs(temp_path)
            settings = LayoutGenerationSettings.from_document(
                _settings_document(temp_path)
            )
            manifest = Cache_Manifest.build_manifest_for_files(
                settings.NODE_FASTA_FILE,
                settings.INPUT_HDF5,
                alignment_score=settings.ALIGNMENT_SCORE,
                normalization=settings.NORM_MODE,
                umap_mode=settings.UMAP_MODE,
                umap_neighbors=settings.UMAP_NEIGHBORS,
                top_edge_percent=settings.TOP_EDGE_PERCENT,
                similarity_threshold=settings.SIMILARITY_THRESHOLD,
            )
            layout_root = pathlib.Path(settings.SAVED_LAYOUT_DIR)
            for folder_name in ("duplicate_a", "duplicate_b"):
                Cache_Manifest.write_manifest_atomic(
                    layout_root / folder_name, manifest
                )
            with self.assertRaisesRegex(
                LayoutGenerationError, "Multiple compatible"
            ):
                generate_layout_cache(settings)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            _write_inputs(temp_path)
            settings = LayoutGenerationSettings.from_document(
                _settings_document(temp_path)
            )
            manifest = Cache_Manifest.build_manifest_for_files(
                settings.NODE_FASTA_FILE,
                settings.INPUT_HDF5,
                alignment_score=settings.ALIGNMENT_SCORE,
                normalization=settings.NORM_MODE,
                umap_mode=settings.UMAP_MODE,
                umap_neighbors=settings.UMAP_NEIGHBORS,
                top_edge_percent=settings.TOP_EDGE_PERCENT,
                similarity_threshold=settings.SIMILARITY_THRESHOLD,
            )
            incompatible = copy.deepcopy(manifest)
            incompatible["compatibility"]["sequence_sha256"] = "0" * 64
            incompatible["inputs"]["sequence"]["sha256"] = "0" * 64
            incompatible["manifest_id"] = Cache_Manifest.calculate_manifest_id(
                incompatible["compatibility"]
            )
            folder_name = Cache_Manifest.build_canonical_cache_name(
                settings.NODE_FASTA_FILE,
                settings.INPUT_HDF5,
                manifest["compatibility"]["network_type"],
                alignment_score=settings.ALIGNMENT_SCORE,
                normalization=settings.NORM_MODE,
                umap_mode=settings.UMAP_MODE,
                umap_neighbors=settings.UMAP_NEIGHBORS,
                top_edge_percent=settings.TOP_EDGE_PERCENT,
                similarity_threshold=settings.SIMILARITY_THRESHOLD,
            )
            Cache_Manifest.write_manifest_atomic(
                pathlib.Path(settings.SAVED_LAYOUT_DIR) / folder_name,
                incompatible,
            )
            with self.assertRaisesRegex(
                LayoutGenerationError, "incompatible cache manifest"
            ):
                generate_layout_cache(settings)

    def test_generation_publishes_only_minimal_cache_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            _write_inputs(temp_path)
            settings = LayoutGenerationSettings.from_document(
                _settings_document(temp_path)
            )
            captured = {}

            def calculate(connectivity, node_count, params):
                captured["connectivity"] = connectivity.copy()
                captured["params"] = dict(params)
                return np.asarray([[0, 0], [1, 1]], dtype=np.float32), 12.0

            fake_engine = SimpleNamespace(calculate_layout=calculate)
            with mock.patch.dict(
                sys.modules, {"Layout_Engine_SSN_MolecularDynamics": fake_engine}
            ):
                result = generate_layout_cache(settings)

            self.assertEqual(result.full_headers, ["Alpha_Beta", "Gamma_Delta"])
            self.assertEqual(captured["connectivity"].shape, (1, 3))
            self.assertEqual(captured["params"]["BOX_SCALE"], 2.0)
            self.assertEqual(captured["params"]["PACKING_PADDING"], 10.0)
            cache_path = pathlib.Path(result.cache_path)
            self.assertTrue(cache_path.exists())
            with h5py.File(cache_path, "r") as cache:
                self.assertEqual(set(cache.keys()), {"headers", "positions"})
                self.assertEqual(
                    cache.attrs["cache_manifest_id"], result.manifest["manifest_id"]
                )
            manifest_path = cache_path.parent / Cache_Manifest.MANIFEST_FILENAME
            self.assertTrue(manifest_path.exists())
            self.assertEqual(
                (cache_path.parent / "set.fasta").read_text(encoding="utf-8"),
                ">Alpha_Beta\nAA\n>Gamma_Delta\nCC\n",
            )
            self.assertEqual(list(cache_path.parent.glob("*.partial")), [])

            with mock.patch.dict(
                sys.modules, {"Layout_Engine_SSN_MolecularDynamics": fake_engine}
            ), self.assertRaises(FileExistsError):
                generate_layout_cache(settings)

    def test_engine_dispatch_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            _write_inputs(temp_path)

            cases = (
                (
                    "Monte Carlo (Style)",
                    False,
                    "mc.h5",
                    "Layout_Engine_SSN_MonteCarlo",
                ),
                (
                    "Molecular Dynamics (Style)",
                    True,
                    "umap.h5",
                    "Layout_Engine_UMAP",
                ),
            )
            for engine, umap_mode, filename, module_name in cases:
                with self.subTest(module=module_name):
                    document = _settings_document(
                        temp_path, cache_filename=filename
                    )
                    payload = document["Layout_Cache_Generator.py"]
                    payload["PHYSICS_ENGINE"] = engine
                    payload["UMAP_MODE"] = umap_mode
                    settings = LayoutGenerationSettings.from_document(document)
                    fake_engine = SimpleNamespace(
                        calculate_layout=lambda connectivity, count, params: (
                            np.zeros((count, 2), dtype=np.float32),
                            5.0,
                        )
                    )
                    with mock.patch.dict(sys.modules, {module_name: fake_engine}):
                        result = generate_layout_cache(settings)
                    self.assertTrue(pathlib.Path(result.cache_path).exists())

            failed_document = _settings_document(
                temp_path, cache_filename="failed.h5"
            )
            failed_settings = LayoutGenerationSettings.from_document(failed_document)
            failed_engine = SimpleNamespace(
                calculate_layout=mock.Mock(side_effect=RuntimeError("engine failed"))
            )
            with mock.patch.dict(
                sys.modules,
                {"Layout_Engine_SSN_MolecularDynamics": failed_engine},
            ), self.assertRaisesRegex(RuntimeError, "engine failed"):
                generate_layout_cache(failed_settings)
            layout_root = temp_path / "layouts"
            self.assertEqual(list(layout_root.rglob("failed.h5")), [])
            self.assertEqual(list(layout_root.rglob("*.partial")), [])

    def test_import_is_headless_and_cli_requires_valid_explicit_json(self):
        import_check = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import Layout_Cache_Generator; "
                    "assert 'SSN_Viewer' not in sys.modules; "
                    "assert 'vispy' not in sys.modules; "
                    "assert 'PySide6' not in sys.modules"
                ),
            ],
            cwd=SRC,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(import_check.returncode, 0, import_check.stderr)

        with tempfile.TemporaryDirectory() as temp_dir:
            malformed = pathlib.Path(temp_dir) / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            process = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "Layout_Cache_Generator.py"),
                    str(malformed),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(process.returncode, 1)
        self.assertIn("Could not read layout settings", process.stderr)


if __name__ == "__main__":
    unittest.main()
