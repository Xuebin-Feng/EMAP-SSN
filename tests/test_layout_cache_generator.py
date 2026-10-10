import copy
import hashlib
import io
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
# prepare_network imports pandas lazily on the UMAP path, and pandas' isna
# then loads numpy.rec. Loading both here keeps the patch.dict(sys.modules)
# windows below from dropping them on exit, which leaves a second pandas
# whose C parser rejects its own "str" dtype.
import pandas  # noqa: F401
import numpy.rec  # noqa: F401


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Cache_Manifest
import Layout_Cache_Generator
from Layout_Cache_Generator import (
    LayoutGenerationError,
    LayoutGenerationSettings,
    generate_layout_cache,
)
from desktop.Viewer_State import decode_document, LAYOUT_SECTIONS
from utilities.Sequence_Utils import derive_node_metadata
from tests.layout_fixtures import (  # noqa: E402
    make_compatibility,
    make_manifest,
    settings_document as _settings_document,
    write_inputs as _write_inputs,
)

FIELD_SECTION = {key: section for section, keys in LAYOUT_SECTIONS.items() for key in keys}


class LayoutSettingsTests(unittest.TestCase):
    def test_schema_is_strict_and_hidden_coordinate_defaults_are_exported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            document = _settings_document(temp_path)
            settings = LayoutGenerationSettings.from_document(document)
            exported = settings.to_document(project_root=ROOT)

            payload = decode_document(exported, "layout")
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
            self.assertIsNone(payload["TARGET_CACHE_PATH"])
            self.assertEqual(payload["CACHE_NAME_MODE"], "explicit")
            self.assertNotIn("TARGET_CACHE_PATH", settings.engine_params())
            self.assertNotIn("CACHE_NAME_MODE", settings.engine_params())

            document["physics"]["NODE_SIZE"] = 10
            with self.assertRaisesRegex(
                LayoutGenerationError, "unknown or misplaced"
            ):
                LayoutGenerationSettings.from_document(document)

    def test_obsolete_and_legacy_layout_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = _settings_document(pathlib.Path(temp_dir))
            legacy = {"DIRECTORIES": {}, "Layout_Cache_Generator.py": decode_document(document, "layout")}
            with self.assertRaisesRegex(LayoutGenerationError, "Re-export"):
                LayoutGenerationSettings.from_document(legacy)
            document["physics"]["PHYSICS_ENGINE"] = "obsolete"
            with self.assertRaisesRegex(LayoutGenerationError, "unknown or misplaced"):
                LayoutGenerationSettings.from_document(document)

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
                    document[FIELD_SECTION[key]][key] = value
                    with self.assertRaises(LayoutGenerationError):
                        LayoutGenerationSettings.from_document(document)

    def test_auto_dt_is_an_optional_json_boolean(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            document = _settings_document(temp_path)
            # Documents exported before AUTO_DT existed leave it out.
            del document["simulation"]["AUTO_DT"]
            settings = LayoutGenerationSettings.from_document(document)
            self.assertIs(settings.AUTO_DT, False)
            self.assertIs(settings.engine_params()["AUTO_DT"], False)

            document = _settings_document(temp_path)
            document["simulation"]["AUTO_DT"] = True
            settings = LayoutGenerationSettings.from_document(document)
            self.assertIs(settings.engine_params()["AUTO_DT"], True)
            self.assertIs(settings.to_document()["simulation"]["AUTO_DT"], True)

            document["simulation"]["AUTO_DT"] = "true"
            with self.assertRaisesRegex(LayoutGenerationError, "AUTO_DT must be a JSON boolean"):
                LayoutGenerationSettings.from_document(document)

    def test_missing_unsafe_and_wrong_typed_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            for key, value, message in (
                ("MAX_STEPS", None, "missing fields"),
                ("CACHE_FILENAME", "../escape.h5", "plain basename"),
                ("UMAP_MODE", "false", "JSON boolean"),
            ):
                with self.subTest(key=key):
                    document = _settings_document(temp_path)
                    if value is None:
                        del document[FIELD_SECTION[key]][key]
                    else:
                        document[FIELD_SECTION[key]][key] = value
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

    def test_generation_publishes_initial_metadata_and_never_overwrites(self):
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
                self.assertEqual(
                    set(cache.keys()), {"headers", "positions", "metadata"}
                )
                self.assertEqual(
                    set(cache["metadata"].keys()),
                    {"Length", "kDa", "pI", "GRAVY"},
                )
                for name in ("Length", "kDa", "pI", "GRAVY"):
                    column = cache["metadata"][name]
                    self.assertEqual(column.attrs["type"], "number")
                    self.assertEqual(column.dtype, np.float64)
                    self.assertEqual(column.shape, (2,))
                np.testing.assert_array_equal(
                    cache["metadata"]["Length"][:], [2, 2]
                )
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
            doc["network"]["TOP_EDGE_PERCENT"] = 50.0
            doc["network"]["SIMILARITY_THRESHOLD"] = None
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
                (True, "umap.h5", "Layout_Engine_UMAP", {}),
            )
            for umap_mode, filename, module_name, legacy in cases:
                with self.subTest(module=module_name):
                    document = _settings_document(
                        temp_path, cache_filename=filename
                    )
                    document["layout"]["UMAP_MODE"] = umap_mode
                    if umap_mode:
                        document["network"]["SIMILARITY_THRESHOLD"] = None
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
                    "import desktop.Viewer_State; "
                    "import utilities.Network_Kernels; "
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

    def test_cli_launch_viewer_and_delete_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            _write_inputs(temp_path)
            doc = _settings_document(temp_path, cache_filename="cli_test.h5")
            settings_file = temp_path / "settings.json"
            settings_file.write_text(json.dumps(doc), encoding="utf-8")

            captured = {}
            def capture_launch(command, **kwargs):
                captured.update(json.loads(pathlib.Path(command[4]).read_text()))
                return 42
            fake_result = SimpleNamespace(
                cache_path=str(temp_path / "layouts" / "folder" / "cli_test.h5"),
            )
            stale_target = {
                "SSN_TARGET_CACHE_PATH": "stale.h5",
                "SSN_TARGET_CACHE_MODE": "stale",
                "SSN_TARGET_CACHE": "stale",
            }
            with mock.patch(
                "Layout_Cache_Generator.generate_layout_cache",
                return_value=fake_result,
            ) as mock_gen, mock.patch("desktop.Viewer_State.validate_viewer_document", side_effect=lambda doc, root: doc), mock.patch(
                "subprocess.call",
                side_effect=capture_launch,
            ) as mock_call, mock.patch.dict(os.environ, stale_target):
                # main() reads an inherited SSN_VIEWER_SETTINGS_PATH and then deletes it.
                os.environ.pop("SSN_VIEWER_SETTINGS_PATH", None)
                code = Layout_Cache_Generator.main(
                    [str(settings_file), "--launch-viewer", "--delete-settings"]
                )
                self.assertEqual(code, 42)
                mock_gen.assert_called_once()
                mock_call.assert_called_once()
                called_cmd, called_kwargs = mock_call.call_args
                self.assertIn("EMAPSSN_Viewer.py", called_cmd[0][2])
                for key in stale_target:
                    self.assertNotIn(key, called_kwargs["env"])
                snapshot = pathlib.Path(called_cmd[0][4])
                snapshot_document = captured
                self.assertEqual(snapshot_document["inputs"]["TARGET_CACHE_PATH"], fake_result.cache_path)
                self.assertEqual(snapshot_document["alignment"]["MSA_FILE"], "")
                self.assertIn("--delete-settings", called_cmd[0])
                self.assertFalse(snapshot.exists())
                self.assertFalse(settings_file.exists())

    def _launch_with_config_snapshot(self, temp_path, snapshot_text, generate):
        """Run main() as the Config does: --launch-viewer and a snapshot in the environment."""
        _write_inputs(temp_path)
        settings_file = temp_path / "settings.json"
        settings_file.write_text(json.dumps(_settings_document(temp_path)), encoding="utf-8")
        snapshot = temp_path / "ssn_viewer_config.json"
        snapshot.write_text(snapshot_text, encoding="utf-8")
        launched = {}

        def capture_launch(command, **kwargs):
            launched.update(json.loads(pathlib.Path(command[4]).read_text(encoding="utf-8")))
            return 42

        errors = io.StringIO()
        with mock.patch(
            "Layout_Cache_Generator.generate_layout_cache", side_effect=generate,
        ) as mock_gen, mock.patch(
            "desktop.Viewer_State.validate_viewer_document", side_effect=lambda doc, root: doc,
        ), mock.patch("subprocess.call", side_effect=capture_launch) as mock_call, mock.patch.dict(
            os.environ, {"SSN_VIEWER_SETTINGS_PATH": str(snapshot)},
        ), redirect_stdout(io.StringIO()), redirect_stderr(errors):
            code = Layout_Cache_Generator.main(
                [str(settings_file), "--launch-viewer", "--delete-settings"]
            )
        self.assertFalse(snapshot.exists())
        self.assertFalse(settings_file.exists())
        return SimpleNamespace(code=code, generate=mock_gen, call=mock_call,
                               launched=launched, errors=errors.getvalue(), snapshot=snapshot)

    def test_launch_viewer_consumes_the_config_snapshot_before_generating(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            snapshot = temp_path / "ssn_viewer_config.json"
            cache_path = str(temp_path / "layouts" / "folder" / "cli_test.h5")

            def generate(settings):
                # Closing the window from here on must leave no snapshot behind.
                self.assertFalse(snapshot.exists())
                return SimpleNamespace(cache_path=cache_path)

            run = self._launch_with_config_snapshot(
                temp_path, json.dumps({"inputs": {}, "alignment": {"MSA_FILE": "chosen.fasta"}}), generate,
            )
        self.assertEqual(run.code, 42)
        run.generate.assert_called_once()
        # The Viewer gets the Config's choices, with the published cache.
        self.assertEqual(run.launched["alignment"], {"MSA_FILE": "chosen.fasta"})
        self.assertEqual(run.launched["inputs"]["TARGET_CACHE_PATH"], cache_path)

    def test_a_failed_generation_leaves_no_config_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run = self._launch_with_config_snapshot(
                pathlib.Path(temp_dir), json.dumps({"inputs": {}}),
                LayoutGenerationError("no edges above the threshold"),
            )
        self.assertEqual(run.code, 1)
        self.assertIn("no edges above the threshold", run.errors)
        run.call.assert_not_called()

    def test_an_unreadable_config_snapshot_stops_before_generating(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run = self._launch_with_config_snapshot(pathlib.Path(temp_dir), "{", None)
        self.assertEqual(run.code, 1)
        self.assertIn("settings_path", run.errors)
        run.generate.assert_not_called()
        run.call.assert_not_called()

    def test_cli_output_is_utf8_whatever_the_stream_encoding(self):
        """A run on its own whose output goes to a file or a pipe prints UTF-8.

        Python opens those in the ANSI code page on Windows, so printing the
        FASTA path of a project folder such as "Projekt-α" raised
        UnicodeEncodeError and the run ended before writing its cache.
        """
        sample = "Résistance α-amylase 中文"
        script = SRC / "Layout_Cache_Generator.py"
        # The script's real startup, up to --help, which exits before any work.
        probe = "\n".join([
            "import runpy, sys",
            f"sys.path.insert(0, {str(SRC)!r})",
            f"sys.argv = [{str(script)!r}, '--help']",
            "try:",
            "    runpy.run_path(sys.argv[0], run_name='__main__')",
            "except SystemExit:",
            "    pass",
            f"print('stdout:', {ascii(sample)})",
            f"print('stderr:', {ascii(sample)}, file=sys.stderr)",
        ])
        # cp1252 is a file's or pipe's encoding on Western Windows; the
        # variable sets it on any platform.
        environment = {
            key: value for key, value in os.environ.items() if key != "PYTHONUTF8"
        }
        environment["PYTHONIOENCODING"] = "cp1252"
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = pathlib.Path(temp_dir) / "generator.log"
            with log_path.open("wb") as log:  # as with `> generator.log 2>&1`
                process = subprocess.run(
                    [sys.executable, "-u", "-c", probe],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    timeout=120,
                )
            output = log_path.read_bytes().decode("utf-8", errors="replace")

        self.assertEqual(process.returncode, 0, output)
        self.assertIn(f"stdout: {sample}", output)
        self.assertIn(f"stderr: {sample}", output)

    def test_import_leaves_the_process_streams_alone(self):
        """Config, the VR Config and the MCP server import this module, and the
        MCP server's stdout carries its protocol: only a run of the script
        itself may switch the output streams."""
        check = "\n".join([
            "import sys",
            "before = [(s.encoding, s.errors) for s in (sys.stdout, sys.stderr)]",
            "import Layout_Cache_Generator",
            "assert [(s.encoding, s.errors) for s in (sys.stdout, sys.stderr)] == before",
        ])
        environment = {
            key: value for key, value in os.environ.items() if key != "PYTHONUTF8"
        }
        environment["PYTHONIOENCODING"] = "cp1252"
        process = subprocess.run(
            [sys.executable, "-c", check],
            cwd=SRC,
            capture_output=True,
            env=environment,
            timeout=120,
        )
        self.assertEqual(
            process.returncode, 0, process.stderr.decode("utf-8", errors="replace")
        )


class LayoutTargetProtectionTests(unittest.TestCase):
    """resolve_layout_selection refuses a Viewer-chosen target folder that
    belongs to another cache, and creates nothing while it checks."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = pathlib.Path(temp_dir.name)
        _write_inputs(self.root)
        self.layouts = self.root / "layouts"
        self.target = self.layouts / "foo"

    def settings(self, target="foo/version_00.h5"):
        document = _settings_document(self.root)
        document["output"]["TARGET_CACHE_PATH"] = target
        return LayoutGenerationSettings.from_document(document, project_root=self.root)

    def resolve(self, target="foo/version_00.h5"):
        settings = self.settings(target)
        manifest = Layout_Cache_Generator.resolve_layout_selection(settings)
        return settings, manifest

    def resolve_error(self, target="foo/version_00.h5"):
        with self.assertRaises(LayoutGenerationError) as raised:
            self.resolve(target)
        return str(raised.exception)

    def test_free_target_resolves_without_creating_it(self):
        settings, manifest = self.resolve()
        self.assertEqual(
            settings.TARGET_CACHE_PATH,
            os.path.join(settings.SAVED_LAYOUT_DIR, "foo", "version_00.h5"),
        )
        self.assertEqual(manifest["compatibility"]["network_type"], "alignment")
        self.assertFalse(self.layouts.exists())

    def test_target_holding_another_inputs_manifest_is_refused(self):
        Cache_Manifest.write_manifest_atomic(
            self.target, make_manifest(make_compatibility())
        )
        self.assertEqual(
            self.resolve_error(),
            "The target folder contains an incompatible cache manifest.",
        )

    def test_target_with_an_unreadable_manifest_is_refused(self):
        self.target.mkdir(parents=True)
        (self.target / Cache_Manifest.MANIFEST_FILENAME).write_text("{", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError) as parse_error:
            json.loads("{")
        self.assertEqual(
            self.resolve_error(),
            f"The target folder contains an invalid cache manifest: {parse_error.exception}",
        )

    def test_target_with_stray_files_and_no_manifest_is_refused(self):
        self.target.mkdir(parents=True)
        # The FASTA backup and staged .partial files may be there already.
        for name in ("notes.txt", "set.fasta", "version_00.h5.partial"):
            (self.target / name).write_text("x", encoding="utf-8")
        self.assertEqual(
            self.resolve_error(),
            "The canonical target folder already contains files but no compatible "
            "manifest: notes.txt",
        )

        (self.target / "notes.txt").unlink()
        settings, _manifest = self.resolve()
        self.assertEqual(
            settings.TARGET_CACHE_PATH,
            os.path.join(settings.SAVED_LAYOUT_DIR, "foo", "version_00.h5"),
        )

    def test_target_away_from_the_compatible_manifest_folder_is_refused(self):
        _settings, manifest = self.resolve()
        Cache_Manifest.write_manifest_atomic(self.layouts / "bar", manifest)
        self.assertEqual(
            self.resolve_error(),
            "The viewer target folder differs from the compatible manifest folder.",
        )

        settings, _manifest = self.resolve("bar/version_00.h5")
        self.assertEqual(
            settings.TARGET_CACHE_PATH,
            os.path.join(settings.SAVED_LAYOUT_DIR, "bar", "version_00.h5"),
        )

    def test_target_filename_must_match_cache_filename(self):
        self.assertEqual(
            self.resolve_error("foo/other.h5"),
            "The viewer target filename does not match CACHE_FILENAME.",
        )


class NodeMetadataDerivationTests(unittest.TestCase):
    def test_columns_follow_network_node_order(self):
        records = [("Gamma_Delta", "CCC"), ("Alpha_Beta", "AA")]

        metadata = derive_node_metadata(["Alpha_Beta", "Gamma_Delta"], records)

        self.assertEqual(set(metadata), {"Length", "kDa", "pI", "GRAVY"})
        for name, entry in metadata.items():
            with self.subTest(column=name):
                self.assertEqual(entry["type"], "number")
                self.assertEqual(entry["values"].dtype, np.float64)
        np.testing.assert_array_equal(metadata["Length"]["values"], [2, 3])
        np.testing.assert_allclose(metadata["GRAVY"]["values"], [1.8, 2.5])

    def test_derived_columns_match_expasy_protparam(self):
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        random.seed(20260912)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        sequences = [
            "".join(random.choice(alphabet) for _ in range(random.randint(30, 400)))
            for _ in range(50)
        ]
        headers = [f"node_{index}" for index in range(len(sequences))]

        metadata = derive_node_metadata(headers, list(zip(headers, sequences)))

        for index, sequence in enumerate(sequences):
            analysis = ProteinAnalysis(sequence)
            # The isoelectric point is bisected to 0.0001 pH units; mass and
            # hydropathy are closed forms and agree to machine precision.
            self.assertAlmostEqual(
                metadata["pI"]["values"][index],
                analysis.isoelectric_point(),
                places=3,
            )
            self.assertAlmostEqual(
                metadata["kDa"]["values"][index],
                analysis.molecular_weight() / 1000.0,
                places=9,
            )
            self.assertAlmostEqual(
                metadata["GRAVY"]["values"][index], analysis.gravy(), places=12
            )

    def test_ambiguity_codes_average_the_residues_they_stand_for(self):
        def derived(sequence):
            return derive_node_metadata(["n"], [("n", sequence)])

        aspartate = derived("ADK")
        asparagine = derived("ANK")
        ambiguous = derived("ABK")

        # Mass is linear in composition, so B lands exactly halfway.
        self.assertAlmostEqual(
            ambiguous["kDa"]["values"][0],
            (aspartate["kDa"]["values"][0] + asparagine["kDa"]["values"][0]) / 2,
            places=9,
        )
        # The titration curve is not linear, so averaging half of aspartate's
        # charge places the isoelectric point between the two, not at their mean.
        self.assertLess(aspartate["pI"]["values"][0], ambiguous["pI"]["values"][0])
        self.assertLess(ambiguous["pI"]["values"][0], asparagine["pI"]["values"][0])

    def test_unscored_residues_leave_gravy_undefined(self):
        metadata = derive_node_metadata(["n"], [("n", "XXUO")])

        self.assertTrue(np.isnan(metadata["GRAVY"]["values"][0]))
        self.assertEqual(metadata["Length"]["values"][0], 4)
        self.assertGreater(metadata["kDa"]["values"][0], 0)

    def test_non_residue_characters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-residue character"):
            derive_node_metadata(["n"], [("n", "AC-DE")])

    def test_node_absent_from_records_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Gamma_Delta"):
            derive_node_metadata(
                ["Alpha_Beta", "Gamma_Delta"], [("Alpha_Beta", "AA")]
            )

    def test_duplicate_record_headers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate headers"):
            derive_node_metadata(
                ["Alpha_Beta"], [("Alpha_Beta", "AA"), ("Alpha_Beta", "CCC")]
            )


if __name__ == "__main__":
    unittest.main()
