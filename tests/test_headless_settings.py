import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from utilities.Headless_Settings import (
    export_pipeline_settings, export_config_settings, build_pipeline_export,
)
from utilities.Viewer_Settings import DEFAULTS, normalize_viewer_settings
from utilities.Execution_Settings import decode_document
from utilities.Tool_Execution import list_tool_specs
from Layout_Cache_Generator import LayoutGenerationSettings, generate_layout_cache
from tests.test_layout_cache_generator import _write_inputs, _settings_document


class HeadlessSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def saved_config(self):
        _write_inputs(self.root)
        values = copy.deepcopy(DEFAULTS)
        values.update(NODE_FASTA_FILE="set.fasta", INPUT_HDF5="network.h5", FASTA_DIR=str(self.root),
                      HDF5_DIR=str(self.root), CACHE_FILE_DIR=str(self.root / "custom-cache"),
                      SIMILARITY_THRESHOLD=0.1, NODE_SIZE="13", MAX_STEPS=1,
                      LAYOUT_DEVICE_SELECTION="cpu")
        (self.root / "viewer_settings.json").write_text(json.dumps(values))
        return values

    def generate(self, document):
        engine = SimpleNamespace(calculate_layout=lambda *a: (np.array([[0, 0], [1, 1]], dtype=np.float32), 12.0))
        with mock.patch.dict(sys.modules, {"Layout_Engine_SSN": engine}):
            return generate_layout_cache(LayoutGenerationSettings.from_document(document, project_root=self.root))

    def test_pipeline_saved_values_and_only_selected_section(self):
        saved = {"DIRECTORIES": {"FASTA_DIR": "my sequences", "SETTING_EXPORT_DIR": "exports"},
                 "Sanitize_Sequences.py": {"INPUT_FASTA": "chosen.fasta"},
                 "Generate_Embeddings.py": {"MODEL_NAME": "unrelated"}}
        original = json.dumps(saved)
        (self.root / "tools_settings.json").write_text(original)
        result = export_pipeline_settings("sanitize_sequences", self.root)
        document = result["settings_document"]
        self.assertEqual(set(document), {"DIRECTORIES", "Sanitize_Sequences.py"})
        self.assertEqual(document["DIRECTORIES"], {"FASTA_DIR": str(self.root / "my sequences")})
        self.assertEqual(document["Sanitize_Sequences.py"]["INPUT_FASTA"], "chosen.fasta")
        self.assertEqual((self.root / "tools_settings.json").read_text(), original)
        self.assertEqual(document, build_pipeline_export("sanitize_sequences", document["DIRECTORIES"], document["Sanitize_Sequences.py"], self.root, absolute=True))
        with self.assertRaises(FileExistsError):
            export_pipeline_settings("sanitize_sequences", self.root, result["settings_path"])

    def test_all_pipeline_defaults_export_as_editable_templates(self):
        for spec in list_tool_specs():
            with self.subTest(tool=spec.tool_id):
                result = export_pipeline_settings(spec.tool_id, self.root)
                self.assertEqual(set(result["settings_document"]), {"DIRECTORIES", spec.settings_section})

    def test_malformed_settings_fail_without_export(self):
        (self.root / "tools_settings.json").write_text("{")
        with self.assertRaisesRegex(ValueError, "Could not read"):
            export_pipeline_settings("sanitize_sequences", self.root)
        self.assertFalse((self.root / "Cache_Files").exists())

    def test_layout_preview_and_reexport_after_edits(self):
        self.saved_config()
        result = export_config_settings("layout", self.root)
        section = decode_document(result["settings_document"], "layout")
        self.assertEqual(section["CACHE_NAME_MODE"], "auto")
        self.assertEqual(section["CACHE_FILENAME"], "version_00.h5")
        self.assertFalse(Path(section["TARGET_CACHE_PATH"]).parent.exists())
        self.assertNotIn("NODE_SIZE", section)
        self.assertNotIn("MSA_FILE", section)
        self.assertIn(str(self.root / "custom-cache"), section["TARGET_CACHE_PATH"])
        before = section["TARGET_CACHE_PATH"]
        result["settings_document"]["network"]["SIMILARITY_THRESHOLD"] = 0.2
        Path(result["settings_path"]).write_text(json.dumps(result["settings_document"]))
        updated = export_config_settings("layout", self.root, settings_path=result["settings_path"])
        self.assertNotEqual(updated["cache_path"], before)

    def test_config_exports_preserve_saved_preferences_and_explicit_overrides(self):
        values = self.saved_config()
        preferences = dict(NODE_SIZE=17, EDGE_ALPHA=0.35, TEXT_SIZE=14,
                           SPRING_K=8.0, COULOMB_K=16.0, DAMPING=0.7,
                           DT=0.003, MAX_STEPS=1234, RMSD_WINDOW=75,
                           PACKING_GEOMETRY="Circle", PACKING_GRID_SIZE=35.0)
        values.update(preferences)
        saved = self.root / "viewer_settings.json"
        saved.write_text(json.dumps(values))
        original = saved.read_bytes()
        layout = export_config_settings("layout", self.root)
        section = decode_document(layout["settings_document"], "layout")
        for key in preferences.keys() - {"NODE_SIZE", "EDGE_ALPHA", "TEXT_SIZE"}:
            self.assertEqual(section[key], preferences[key], key)
        generated = self.generate(layout["settings_document"])
        overlay = self.root / "viewer-overlay.json"
        overlay.write_text(json.dumps({"schema_version": 2, "kind": "viewer", "inputs": {"TARGET_CACHE_PATH": generated.cache_path}, "visualization": {"TEXT_SIZE": 15}}))
        viewer = export_config_settings("viewer", self.root, settings_path=overlay)
        for key in ("NODE_SIZE", "EDGE_ALPHA", "TEXT_SIZE"):
            self.assertEqual(viewer["settings_document"]["visualization"][key], 15 if key == "TEXT_SIZE" else preferences[key], key)
        self.assertEqual(viewer["cache_path"], generated.cache_path)
        layout["settings_document"]["simulation"]["MAX_STEPS"] = 4321
        Path(layout["settings_path"]).write_text(json.dumps(layout["settings_document"]))
        edited = export_config_settings("layout", self.root, settings_path=layout["settings_path"])
        self.assertEqual(edited["settings_document"]["simulation"]["MAX_STEPS"], 4321)
        self.assertEqual(edited["settings_document"]["physics"]["SPRING_K"], 8.0)
        self.assertEqual(saved.read_bytes(), original)

    def test_viewer_full_snapshot_newest_and_explicit_selection(self):
        self.saved_config()
        with self.assertRaisesRegex(ValueError, "No compatible"):
            export_config_settings("viewer", self.root)
        layout = export_config_settings("layout", self.root)["settings_document"]
        first = self.generate(layout)
        second = self.generate(layout)
        os.utime(first.cache_path, ns=(1_000_000_000, 1_000_000_000))
        exported = export_config_settings("viewer", self.root)
        document = exported["settings_document"]
        self.assertEqual(set(document), {"schema_version", "kind", "inputs", "alignment", "visualization", "directories"})
        self.assertEqual(document["visualization"]["NODE_SIZE"], 13)
        self.assertEqual(document["inputs"]["TARGET_CACHE_PATH"], second.cache_path)
        self.assertEqual(Path(document["inputs"]["TARGET_CACHE_PATH"]).name, "version_01.h5")
        overlay = self.root / "selection.json"
        overlay.write_text(json.dumps({"schema_version": 2, "kind": "viewer", "inputs": {"TARGET_CACHE_PATH": first.cache_path}}))
        selected = export_config_settings("viewer", self.root, settings_path=overlay)
        self.assertEqual(selected["cache_path"], first.cache_path)
        stamp = 2_000_000_000
        os.utime(first.cache_path, ns=(stamp, stamp))
        os.utime(second.cache_path, ns=(stamp, stamp))
        self.assertEqual(export_config_settings("viewer", self.root)["cache_path"], first.cache_path)

    def test_explicit_collision_and_filename_mismatch(self):
        self.saved_config()
        document = export_config_settings("layout", self.root)["settings_document"]
        self.generate(document)
        section = document["output"]
        section["CACHE_NAME_MODE"] = "explicit"
        with self.assertRaises(FileExistsError):
            self.generate(document)
        section["CACHE_FILENAME"] = "custom.h5"
        with self.assertRaisesRegex(ValueError, "filename does not match"):
            self.generate(document)

    def test_version_gaps_and_occupied_directory_names(self):
        from Cache_Manifest import next_cache_version_filename
        (self.root / "version_03.h5").write_bytes(b"cache")
        (self.root / "version_04.h5").mkdir()
        self.assertEqual(next_cache_version_filename(self.root), "version_05.h5")

    def test_auto_publish_race_reuses_calculation(self):
        self.saved_config()
        document = export_config_settings("layout", self.root)["settings_document"]
        import Layout_Cache_Generator as generator
        original = generator._publish_cache_without_overwrite
        attempts = []
        def publish(staged, destination):
            attempts.append(destination)
            if len(attempts) == 1:
                Path(destination).write_bytes(b"other writer")
            original(staged, destination)
        with mock.patch.object(generator, "_publish_cache_without_overwrite", side_effect=publish):
            result = self.generate(document)
        self.assertEqual(Path(result.cache_path).name, "version_01.h5")
        self.assertEqual(Path(attempts[0]).read_bytes(), b"other writer")

    def test_two_processes_share_folder_and_distinct_versions(self):
        _write_inputs(self.root)
        document = _settings_document(self.root)
        document["output"]["CACHE_NAME_MODE"] = "auto"
        path = self.root / "layout.json"
        path.write_text(json.dumps(document))
        script = """import sys,time,json
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, sys.argv[1])
from Layout_Cache_Generator import LayoutGenerationSettings,generate_layout_cache
def calculate(*args):
    time.sleep(0.3)
    return np.array([[0,0],[1,1]],dtype=np.float32),12.0
sys.modules['Layout_Engine_SSN']=SimpleNamespace(calculate_layout=calculate)
result=generate_layout_cache(LayoutGenerationSettings.from_json_file(sys.argv[2]))
print(result.cache_path)
"""
        processes = [subprocess.Popen([sys.executable, "-c", script, str(ROOT / "src"), str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        try:
            paths = []
            for proc in processes:
                out, err = proc.communicate(timeout=60)
                self.assertEqual(proc.returncode, 0, err)
                paths.append(Path(out.strip().splitlines()[-1]))
            self.assertEqual(paths[0].parent, paths[1].parent)
            self.assertEqual({p.name for p in paths}, {"version_00.h5", "version_01.h5"})
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()

    def test_cli_export_does_not_import_qt_or_hardware(self):
        target = self.root / "cli.json"
        code = """import sys,runpy
class BlockGUI:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'PySide6','torch','vispy'}:
            raise RuntimeError('Forbidden GUI/hardware import: '+fullname)
sys.meta_path.insert(0,BlockGUI())
sys.path.insert(0,sys.argv[1])
sys.argv=[sys.argv[1]+'/EMAPSSN_Tools.py','--headless','export','--tool','sanitize_sequences','--output',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "src"), str(target)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["settings_path"], str(target))

    def test_config_cli_export_does_not_import_qt_or_hardware(self):
        _write_inputs(self.root)
        document = _settings_document(self.root)
        document["output"]["CACHE_NAME_MODE"] = "auto"
        source = self.root / "source.json"
        source.write_text(json.dumps(document))
        target = self.root / "config-export.json"
        code = """import sys,runpy
class BlockGUI:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'PySide6','torch','vispy'}:
            raise RuntimeError('Forbidden GUI/hardware import: '+fullname)
sys.meta_path.insert(0,BlockGUI())
sys.path.insert(0,sys.argv[1])
sys.argv=[sys.argv[1]+'/EMAPSSN_Config.py','--headless','export','--kind','layout','--settings',sys.argv[2],'--output',sys.argv[3]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "src"), str(source), str(target)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["settings_path"], str(target))


class HeadlessLayoutJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_cpu_job_reports_actual_path_and_keeps_original_snapshot(self):
        from mcp_server.MCP_Pipeline_Jobs import PipelineJobManager
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_inputs(root)
            document = _settings_document(root)
            section = document["output"]
            section["CACHE_NAME_MODE"] = "auto"
            document["simulation"].update(MAX_STEPS=1, LAYOUT_DEVICE_SELECTION="cpu")
            manager = PipelineJobManager(ROOT, temporary_parent=root / "jobs")
            try:
                jobs = [await manager.submit_layout_job(copy.deepcopy(document)) for _ in range(2)]
                snapshots = {j["job_id"]: Path(j["settings_snapshot"]).read_bytes() for j in jobs}
                deadline = asyncio.get_running_loop().time() + 90
                for job in jobs:
                    while True:
                        state = await manager.get_job(job["job_id"])
                        if state["status"] in {"succeeded", "failed", "cancelled"}:
                            break
                        self.assertLess(asyncio.get_running_loop().time(), deadline)
                        await asyncio.sleep(0.1)
                    self.assertEqual(state["status"], "succeeded", Path(state["stderr_log"]).read_text())
                    self.assertEqual(Path(state["settings_snapshot"]).read_bytes(), snapshots[job["job_id"]])
                    actual = state["output_locations"]
                    self.assertTrue(Path(actual["TARGET_CACHE_PATH"]).is_file())
                    executed = json.loads(Path(actual["EXECUTED_SETTINGS"]).read_text())["output"]
                    self.assertEqual(executed["TARGET_CACHE_PATH"], actual["TARGET_CACHE_PATH"])
                    self.assertEqual(executed["CACHE_FILENAME"], actual["CACHE_FILENAME"])
                states = [await manager.get_job(j["job_id"]) for j in jobs]
                self.assertEqual({s["output_locations"]["CACHE_FILENAME"] for s in states}, {"version_00.h5", "version_01.h5"})
            finally:
                await manager.close()


if __name__ == "__main__":
    unittest.main()
