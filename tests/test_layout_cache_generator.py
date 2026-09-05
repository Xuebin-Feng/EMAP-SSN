import copy
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Cache_Manifest
from Layout_Cache_Generator import (
    LayoutGenerationError,
    LayoutGenerationSettings,
    generate_layout_cache,
)
from utilities.Network_Preparation import prepare_network


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


def _preparation_settings(**overrides):
    values = {
        "NODE_FASTA_FILE": "",
        "ALIGNMENT_SCORE": "global",
        "NORM_MODE": "alignment_length",
        "SIMILARITY_THRESHOLD": 0.0,
        "TOP_EDGE_PERCENT": None,
        "UMAP_MODE": False,
        "UMAP_NEIGHBORS": 15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class NetworkPreparationTests(unittest.TestCase):
    def test_alignment_normalization_modes_and_fasta_subset(self):
        expected_scores = {
            "alignment_length": 4.0,
            "shorter_sequence": 4.0,
            "longer_sequence": 2.0,
            "average_sequence": 8.0 / 3.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "model"
                network.create_dataset("headers", data=[b"A", b"B", b"C"])
                network.create_dataset("i", data=[0, 0, 1])
                network.create_dataset("j", data=[1, 2, 2])
                network.create_dataset("seq_lens", data=[2, 4, 8])
                for name in ("g_score", "l_score"):
                    network.create_dataset(name, data=[8.0, 8.0, 8.0])
                for name in ("g_len", "l_len"):
                    network.create_dataset(name, data=[2, 2, 2])

            for normalization, expected_score in expected_scores.items():
                with self.subTest(normalization=normalization), h5py.File(
                    network_path, "r"
                ) as network:
                    settings = _preparation_settings(NORM_MODE=normalization)
                    headers, edges, scores = prepare_network(
                        network,
                        settings=settings,
                        selected_fasta_headers=["A", "B"],
                    )
                    self.assertEqual(headers, ["A", "B"])
                    np.testing.assert_array_equal(edges, [[0, 1]])
                    np.testing.assert_allclose(scores, [expected_score])
                    self.assertFalse(settings.INPUT_IS_EVALUE)

    def test_top_percent_updates_effective_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "model"
                network.create_dataset("headers", data=[b"A", b"B", b"C"])
                network.create_dataset("i", data=[0, 0, 1])
                network.create_dataset("j", data=[1, 2, 2])
                network.create_dataset("seq_lens", data=[2, 2, 2])
                for name in ("g_score", "l_score"):
                    network.create_dataset(name, data=[2.0, 6.0, 4.0])
                for name in ("g_len", "l_len"):
                    network.create_dataset(name, data=[2, 2, 2])

            settings = _preparation_settings(TOP_EDGE_PERCENT=34.0)
            with h5py.File(network_path, "r") as network:
                headers, edges, scores = prepare_network(
                    network,
                    settings=settings,
                    selected_fasta_headers=["A", "B", "C"],
                )

            self.assertEqual(headers, ["A", "B", "C"])
            self.assertEqual(settings.SIMILARITY_THRESHOLD, 3.0)
            np.testing.assert_array_equal(edges, [[0, 2]])
            np.testing.assert_allclose(scores, [3.0])

    def test_empty_blast_network_returns_empty_connectivity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "blast.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "blast"
                network.create_dataset("headers", data=[b"A", b"B"])
                network.create_dataset("i", data=np.asarray([], dtype=np.int32))
                network.create_dataset("j", data=np.asarray([], dtype=np.int32))
                network.create_dataset("score", data=np.asarray([], dtype=np.float32))

            settings = _preparation_settings()
            with h5py.File(network_path, "r") as network:
                headers, edges, scores = prepare_network(
                    network,
                    settings=settings,
                    selected_fasta_headers=None,
                )

            self.assertEqual(headers, ["A", "B"])
            self.assertEqual(edges.shape, (0, 2))
            self.assertEqual(scores.shape, (0,))
            self.assertTrue(settings.INPUT_IS_EVALUE)


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
            self.assertNotIn("PHYSICS_ENGINE", payload)
            self.assertFalse(any(key.startswith("MC_") for key in payload))
            self.assertFalse(any(key.startswith("SGLD_") for key in payload))
            self.assertNotIn("NODE_SIZE", payload)
            self.assertNotIn("MSA_FILE", payload)
            self.assertNotIn("TARGET_CACHE_PATH", payload)

            document["Layout_Cache_Generator.py"]["NODE_SIZE"] = 10
            with self.assertRaisesRegex(
                LayoutGenerationError, "Unknown layout-generation setting"
            ):
                LayoutGenerationSettings.from_document(document)

    def test_obsolete_layout_engine_settings_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            document = _settings_document(temp_path)
            payload = document["Layout_Cache_Generator.py"]
            payload.update(
                {
                    "PHYSICS_ENGINE": "historical engine value",
                    "MC_SWEEPS": "ignored",
                    "MC_QUENCH_SWEEPS": -1,
                    "MC_TELEPORT_PROBABILITY": object(),
                    "MC_RANDOM_SEED": None,
                    "SGLD_MIN_K": 20,
                    "SGLD_K_PERCENT": 0.01,
                    "SGLD_START_TEMP": 1.5,
                    "SGLD_NOISE_SCALE": 1.0,
                }
            )
            settings = LayoutGenerationSettings.from_document(document)
            exported = settings.to_document(project_root=ROOT)[
                "Layout_Cache_Generator.py"
            ]
            self.assertNotIn("PHYSICS_ENGINE", exported)
            self.assertFalse(any(key.startswith("MC_") for key in exported))
            self.assertFalse(any(key.startswith("SGLD_") for key in exported))

    def test_strict_layout_numeric_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            invalid_cases = (
                ("SPRING_K", float("nan")),
                ("COULOMB_K", -1.0),
                ("COULOMB_CUTOFF", float("inf")),
                ("MAX_FORCE_LIMIT", -1.0),
                ("PACKING_PADDING", -1.0),
                ("BOX_SCALE", 0.0),
                ("PACKING_GRID_SIZE", 0.0),
            )
            for key, value in invalid_cases:
                with self.subTest(key=key, value=value):
                    document = _settings_document(temp_path)
                    payload = document["Layout_Cache_Generator.py"]
                    payload[key] = value
                    with self.assertRaises(LayoutGenerationError):
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
    def test_duplicate_folders_are_rejected_and_canonical_collision_is_renamed(self):
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
            fake_engine = SimpleNamespace(
                calculate_layout=lambda _connectivity, _node_count, _params: (
                    np.asarray([[0, 0], [1, 1]], dtype=np.float32),
                    12.0,
                )
            )
            with mock.patch.dict(
                sys.modules, {"Layout_Engine_SSN": fake_engine}
            ):
                result = generate_layout_cache(settings)

            expected_folder = (
                pathlib.Path(settings.SAVED_LAYOUT_DIR)
                / f"{folder_name}_[{manifest['manifest_id'][:8]}]"
            )
            self.assertEqual(pathlib.Path(result.cache_path).parent, expected_folder)
            self.assertEqual(
                Cache_Manifest.read_manifest(
                    pathlib.Path(settings.SAVED_LAYOUT_DIR) / folder_name
                )["manifest_id"],
                incompatible["manifest_id"],
            )
            self.assertEqual(
                Cache_Manifest.read_manifest(expected_folder)["manifest_id"],
                manifest["manifest_id"],
            )

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
                sys.modules, {"Layout_Engine_SSN": fake_engine}
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
                layout_metadata = json.loads(
                    cache.attrs["layout_compatibility_json"]
                )
                self.assertNotIn("PHYSICS_ENGINE", layout_metadata)
                self.assertFalse(
                    any(key.startswith("MC_") for key in layout_metadata)
                )
                self.assertFalse(
                    any(key.startswith("SGLD_") for key in layout_metadata)
                )
                canonical = json.dumps(
                    layout_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                self.assertEqual(
                    cache.attrs["layout_compatibility_id"],
                    hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                )
            manifest_path = cache_path.parent / Cache_Manifest.MANIFEST_FILENAME
            self.assertTrue(manifest_path.exists())
            self.assertEqual(
                (cache_path.parent / "set.fasta").read_text(encoding="utf-8"),
                ">Alpha_Beta\nAA\n>Gamma_Delta\nCC\n",
            )
            self.assertEqual(list(cache_path.parent.glob("*.partial")), [])

            with mock.patch.dict(
                sys.modules, {"Layout_Engine_SSN": fake_engine}
            ), self.assertRaises(FileExistsError):
                generate_layout_cache(settings)

    def test_top_edge_percent_generates_cache_without_float32_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            _write_inputs(temp_path)
            doc = _settings_document(temp_path, cache_filename="top_edge.h5")
            doc["Layout_Cache_Generator.py"]["TOP_EDGE_PERCENT"] = 50.0
            doc["Layout_Cache_Generator.py"]["SIMILARITY_THRESHOLD"] = None
            settings = LayoutGenerationSettings.from_document(doc)

            fake_engine = SimpleNamespace(
                calculate_layout=lambda _connectivity, node_count, _params: (
                    np.zeros((node_count, 2), dtype=np.float32),
                    10.0,
                )
            )
            with mock.patch.dict(
                sys.modules, {"Layout_Engine_SSN": fake_engine}
            ), redirect_stdout(io.StringIO()):
                result = generate_layout_cache(settings)

            self.assertTrue(pathlib.Path(result.cache_path).exists())
            self.assertIsInstance(result.effective_similarity_threshold, float)
            with h5py.File(result.cache_path, "r") as cache:
                self.assertIn("layout_compatibility_json", cache.attrs)
                self.assertIn("layout_compatibility_id", cache.attrs)

    def test_engine_dispatch_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            _write_inputs(temp_path)

            cases = (
                (False, "md.h5", "Layout_Engine_SSN", {}),
                (
                    False,
                    "legacy_mc.h5",
                    "Layout_Engine_SSN",
                    {
                        "PHYSICS_ENGINE": "Monte Carlo (Style)",
                        "MC_RANDOM_SEED": None,
                    },
                ),
                (True, "umap.h5", "Layout_Engine_UMAP", {}),
            )
            for umap_mode, filename, module_name, legacy in cases:
                with self.subTest(module=module_name):
                    document = _settings_document(
                        temp_path, cache_filename=filename
                    )
                    payload = document["Layout_Cache_Generator.py"]
                    payload["UMAP_MODE"] = umap_mode
                    payload.update(legacy)
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
                {"Layout_Engine_SSN": failed_engine},
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
                    "import utilities.Cache_Selection; "
                    "import utilities.Network_Clustering; "
                    "import utilities.Network_Preparation; "
                    "assert 'EMAPSSN_Viewer' not in sys.modules; "
                    "assert 'vispy' not in sys.modules; "
                    "assert 'PySide6' not in sys.modules; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'Bio' not in sys.modules; "
                    "assert 'matplotlib' not in sys.modules"
                ),
            ],
            cwd=SRC,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(import_check.returncode, 0, import_check.stderr)
        self.assertEqual(import_check.stdout, "")

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
