import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from utilities.Tool_Execution import (  # noqa: E402
    format_invocation_command,
    get_tool_spec,
    list_tool_specs,
    prepare_gui_invocation,
    prepare_headless_invocation,
)
from utilities.Viewer_Inspection import (  # noqa: E402
    ViewerInspectionError,
    ViewerInspectionService,
)
from utilities.Viewer_Sessions import (  # noqa: E402
    SESSION_DIRECTORY_ENV,
    discover_viewer_sessions,
    publish_viewer_session,
    select_viewer_session,
)


class ToolExecutionTests(unittest.TestCase):
    def test_transport_neutral_modules_import_without_stdout(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(SRC_DIR)!r}); "
            "import utilities.Tool_Execution; "
            "import utilities.Viewer_Inspection; "
            "import utilities.Viewer_Sessions"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_catalog_is_complete_stable_and_quiet(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            specs = list_tool_specs()
        self.assertEqual(captured.getvalue(), "")
        self.assertEqual(len(specs), 14)
        self.assertEqual(len({spec.tool_id for spec in specs}), 14)
        self.assertEqual(
            get_tool_spec("sanitize_sequences").script_name,
            "Sanitize_Sequences.py",
        )
        with self.assertRaises(KeyError):
            get_tool_spec("../arbitrary.py")

    def test_gui_invocation_preserves_the_existing_command(self):
        script = SRC_DIR / "tools" / "Sanitize_Sequences.py"
        invocation = prepare_gui_invocation(
            script,
            PROJECT_ROOT,
            python_executable="managed-python",
        )
        self.assertEqual(
            invocation.argv,
            ("managed-python", "-u", str(script.resolve())),
        )
        self.assertEqual(invocation.cwd, str(script.parent.resolve()))
        self.assertFalse(invocation.owns_settings_snapshot)
        self.assertEqual(
            format_invocation_command(invocation),
            f'"managed-python" -u "{script.resolve()}"',
        )

    def test_object_and_file_settings_create_equivalent_snapshots(self):
        document = {
            "DIRECTORIES": {"FASTA_DIR": "Input_Files/Sequence_Sets"},
            "Sanitize_Sequences.py": {
                "INPUT_FASTA": None,
                "OVER_WRITE": False,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = pathlib.Path(temp_dir) / "source.json"
            source_path.write_text(json.dumps(document), encoding="utf-8")
            object_call = prepare_headless_invocation(
                "sanitize_sequences",
                document,
                PROJECT_ROOT,
                python_executable="managed-python",
                snapshot_directory=temp_dir,
            )
            file_call = prepare_headless_invocation(
                "sanitize_sequences",
                source_path,
                PROJECT_ROOT,
                python_executable="managed-python",
                snapshot_directory=temp_dir,
            )
            object_payload = json.loads(
                pathlib.Path(object_call.settings_path).read_text(encoding="utf-8")
            )
            file_payload = json.loads(
                pathlib.Path(file_call.settings_path).read_text(encoding="utf-8")
            )
            self.assertEqual(object_payload, document)
            self.assertEqual(file_payload, document)
            self.assertIsNone(
                object_payload["Sanitize_Sequences.py"]["INPUT_FASTA"]
            )
            self.assertEqual(object_call.argv[-1], object_call.settings_path)
            self.assertTrue(object_call.owns_settings_snapshot)


class ViewerInspectionTests(unittest.TestCase):
    def setUp(self):
        self.viewer = SimpleNamespace(
            n_nodes=4,
            full_headers=["A", "B", "C", "D"],
            visible_mask=[True, False, True, True],
            selected_indices=[3, 0, 3, 99],
            edges=[(0, 1), (2, 3)],
            metadata={
                "score": {"type": "numeric", "values": [1.0, float("nan"), 3.5, 4.0]},
                "family": {"type": "text", "values": ["x", "y", "x", "z"]},
            },
            cluster_labels=[1, 1, 2, 2],
            group_labels=[{"g1"}, set(), {"g2"}, {"g1", "g2"}],
            current_slider_threshold=0.75,
        )
        self.config = SimpleNamespace(
            NODE_FASTA_FILE=None,
            INPUT_HDF5="network.h5",
            MSA_FILE="",
            TARGET_CACHE_PATH=None,
            TARGET_CACHE_FILE=None,
            SIMILARITY_THRESHOLD=0.1,
        )
        self.service = ViewerInspectionService(self.viewer, self.config)

    def test_summary_is_bounded_and_json_ready(self):
        summary = self.service.get_summary()
        self.assertEqual(summary["node_count"], 4)
        self.assertEqual(summary["edge_count"], 2)
        self.assertEqual(summary["visible_node_count"], 3)
        self.assertEqual(summary["selected_node_count"], 2)
        self.assertIsNone(summary["inputs"]["node_fasta"])
        self.assertEqual(summary["clusters"]["count"], 2)
        self.assertEqual(summary["groups"]["count"], 2)

    def test_query_nodes_filters_pages_and_normalizes_values(self):
        page = self.service.query_nodes(
            scope="visible",
            offset=1,
            limit=2,
            columns=["score"],
        )
        self.assertEqual(page["total"], 3)
        self.assertEqual([row["index"] for row in page["nodes"]], [2, 3])
        self.assertEqual(page["nodes"][0]["metadata"], {"score": 3.5})
        selected = self.service.query_nodes(scope="selected", columns=["score"])
        self.assertEqual([row["index"] for row in selected["nodes"]], [0, 3])

    def test_query_rejects_unbounded_or_unknown_requests(self):
        with self.assertRaises(ViewerInspectionError):
            self.service.query_nodes(limit=501)
        with self.assertRaises(ViewerInspectionError):
            self.service.query_nodes(columns=["missing"])
        with self.assertRaises(ViewerInspectionError):
            self.service.query_nodes(scope="mutating")


class ViewerDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def _request_with_qt_pump(self, url, token=None):
        outcome = {}

        def request():
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=3
                ) as response:
                    outcome["status"] = response.status
                    outcome["payload"] = json.loads(response.read().decode("utf-8"))
            except Exception as error:
                outcome["error"] = error

        thread = threading.Thread(target=request)
        thread.start()
        deadline = time.monotonic() + 4
        while thread.is_alive() and time.monotonic() < deadline:
            self.application.processEvents()
            thread.join(0.01)
        thread.join(timeout=0.1)
        if "error" in outcome:
            raise outcome["error"]
        return outcome

    def test_authenticated_endpoints_and_descriptor_lifecycle(self):
        from web_ui import Web_Server

        viewer = SimpleNamespace(
            n_nodes=1,
            full_headers=["A"],
            visible_mask=[True],
            selected_indices=[],
            edges=[],
            metadata={},
            cluster_labels=None,
            group_labels=[set()],
            web_plugin_registry=None,
        )
        viewer.viewer_inspection = ViewerInspectionService(viewer)
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {SESSION_DIRECTORY_ENV: temp_dir},
        ):
            server = Web_Server.start_server(viewer, preferred_port=0)
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                with self.assertRaises(urllib.error.HTTPError) as unauthorized:
                    urllib.request.urlopen(
                        f"{base_url}/api/mcp/v1/session", timeout=2
                    )
                self.assertEqual(unauthorized.exception.code, 401)
                with self.assertRaises(urllib.error.HTTPError) as wrong_token:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            f"{base_url}/api/mcp/v1/session",
                            headers={"Authorization": "Bearer incorrect"},
                        ),
                        timeout=2,
                    )
                self.assertEqual(wrong_token.exception.code, 401)

                session = self._request_with_qt_pump(
                    f"{base_url}/api/mcp/v1/session",
                    server.inspection_token,
                )
                self.assertEqual(session["status"], 200)
                self.assertEqual(
                    session["payload"]["session_id"],
                    server.inspection_session_id,
                )
                self.assertNotIn("token", session["payload"])
                summary = self._request_with_qt_pump(
                    f"{base_url}/api/mcp/v1/summary",
                    server.inspection_token,
                )
                self.assertEqual(summary["payload"]["node_count"], 1)

                sessions = discover_viewer_sessions(timeout=1)
                self.assertEqual(len(sessions), 1)
                self.assertEqual(
                    select_viewer_session(timeout=1).session_id,
                    server.inspection_session_id,
                )
                descriptor_path = pathlib.Path(server.inspection_descriptor.descriptor_path)
                self.assertTrue(descriptor_path.is_file())
            finally:
                descriptor_path = pathlib.Path(server.inspection_descriptor.descriptor_path)
                Web_Server.stop_server(server)
            self.assertFalse(descriptor_path.exists())

    def test_multiple_sessions_require_explicit_selection(self):
        from web_ui import Web_Server

        viewer = SimpleNamespace(
            n_nodes=0,
            full_headers=[],
            visible_mask=[],
            selected_indices=[],
            edges=[],
            metadata={},
            web_plugin_registry=None,
        )
        viewer.viewer_inspection = ViewerInspectionService(viewer)
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {SESSION_DIRECTORY_ENV: temp_dir},
        ):
            first = Web_Server.start_server(viewer, preferred_port=0)
            second = Web_Server.start_server(viewer, preferred_port=0)
            try:
                sessions = discover_viewer_sessions(timeout=1)
                self.assertEqual(len(sessions), 2)
                with self.assertRaises(LookupError):
                    select_viewer_session(timeout=1)
                selected = select_viewer_session(
                    first.inspection_session_id,
                    timeout=1,
                )
                self.assertEqual(selected.port, first.server_address[1])
            finally:
                Web_Server.stop_server(first)
                Web_Server.stop_server(second)

    def test_definitively_stale_descriptor_is_pruned(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {SESSION_DIRECTORY_ENV: temp_dir},
        ):
            descriptor = publish_viewer_session(
                session_id="stale",
                pid=999999,
                port=9,
                token="unused",
            )
            with mock.patch(
                "utilities.Viewer_Sessions._validate_live_session",
                return_value=False,
            ), mock.patch(
                "utilities.Viewer_Sessions._process_is_running",
                return_value=False,
            ):
                self.assertEqual(discover_viewer_sessions(), [])
            self.assertFalse(pathlib.Path(descriptor.descriptor_path).exists())


if __name__ == "__main__":
    unittest.main()
