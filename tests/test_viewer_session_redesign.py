import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from unittest import mock

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tests import layout_fixtures as fixtures
from Layout_Cache_Generator import LayoutGenerationSettings, generate_layout_cache
from desktop.Viewer_State import (
    validate_viewer_document,
    ViewerSettingsError,
    normalize_viewer_settings,
    DEFAULTS,
    encode_document,
    VIEWER_SECTIONS,
)
from mcp_server.viewer.Viewer_Client import MCPViewerClient, MCPViewerError
from EMAPSSN_MCP_Server import mcp
from mcp import Client, StdioServerParameters


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixtures.write_inputs(self.root)
        settings = LayoutGenerationSettings.from_document(fixtures.settings_document(self.root), project_root=fixtures.ROOT)
        with mock.patch("Layout_Engine_SSN.calculate_layout", return_value=(np.array([[0, 0], [1, 1]], dtype=np.float32), 10.0)):
            self.cache = generate_layout_cache(settings).cache_path
        self.document = dict(TARGET_CACHE_PATH=self.cache, NODE_FASTA_FILE=str(self.root / "set.fasta"),
            INPUT_HDF5=str(self.root / "network.h5"), MSA_FILE="", UMAP_MODE=False,
            ALIGNMENT_SCORE="global", NORM_MODE="alignment_length", SIMILARITY_THRESHOLD=0.1,
            TOP_EDGE_PERCENT=None)
        self.document = encode_document("viewer", {**DEFAULTS, **self.document})

    def test_complete_document_and_no_personal_settings(self):
        with mock.patch.dict(os.environ, {"SSN_VIEWER_SETTINGS_PATH": "unrelated.json"}):
            result = validate_viewer_document(self.document, fixtures.ROOT)
        self.assertEqual(result["alignment"]["MSA_FILE"], "")
        self.assertEqual(result["visualization"]["NODE_SIZE"], 10)
        self.assertNotIn("network", result)

    def test_missing_files_fields_and_manifest_conflicts(self):
        field_sections = {key: section for section, keys in VIEWER_SECTIONS.items() for key in keys}
        for key in ("NODE_FASTA_FILE", "INPUT_HDF5", "MSA_FILE", "TARGET_CACHE_PATH", "ALIGNMENT_REFERENCE"):
            doc = deepcopy(self.document)
            doc[field_sections[key]].pop(key)
            with self.subTest(key=key), self.assertRaises(ViewerSettingsError):
                validate_viewer_document(doc, fixtures.ROOT)
        for override in ({"INPUT_HDF5": "missing.h5"}, {"SIMILARITY_THRESHOLD": 100},
                         {"ALIGNMENT_REFERENCE": "missing"}, {"NODE_SIZE": True}, {"TYPO": 1}):
            with self.subTest(override=override), self.assertRaises(ViewerSettingsError):
                doc = deepcopy(self.document)
                for key, value in override.items():
                    doc[field_sections.get(key, "inputs")][key] = value
                validate_viewer_document(doc, fixtures.ROOT)

    def test_aliases_and_header_order(self):
        doc = deepcopy(self.document)
        doc["directories"]["INPUT_FILE_DIR"] = str(self.root)
        doc["inputs"].update(NODE_FASTA_FILE="$input_file$/set.fasta", INPUT_HDF5="$input_file$/network.h5")
        self.assertEqual(validate_viewer_document(doc, fixtures.ROOT)["inputs"]["INPUT_HDF5"], str(self.root / "network.h5"))
        import h5py
        with h5py.File(self.cache, "r+") as cache:
            cache["headers"][:] = cache["headers"][:][::-1]
        with self.assertRaises(ViewerSettingsError):
            validate_viewer_document(doc, fixtures.ROOT)

    def test_missing_reference_does_not_bypass_msa_readability_checks(self):
        import h5py
        doc = deepcopy(self.document)
        doc["alignment"]["ALIGNMENT_REFERENCE"] = "absent"
        for filename, data in (("missing.fasta", None), ("invalid.fasta", b"\xff"), ("invalid.h5", b"not HDF5")):
            with self.subTest(filename=filename):
                path = self.root / filename
                if data is not None:
                    path.write_bytes(data)
                doc["alignment"]["MSA_FILE"] = str(path)
                with self.assertRaises(ViewerSettingsError):
                    validate_viewer_document(doc, fixtures.ROOT)
        path = self.root / "no_headers.h5"
        with h5py.File(path, "w"):
            pass
        doc["alignment"]["MSA_FILE"] = str(path)
        with self.assertRaises(ViewerSettingsError):
            validate_viewer_document(doc, fixtures.ROOT)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_validation_preserves_missing_reference_preferences(self):
        fixture = SettingsTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        msa_path = fixture.root / "alignment.fasta"
        msa_path.write_text(">Alpha_Beta\nAA\n>Gamma_Delta\nCC\n", encoding="utf-8")
        fixture.document["alignment"].update(
            MSA_FILE=str(msa_path), ALIGNMENT_REFERENCE="absent",
            FILTER_MIN_OCCUPANCY=75, ALIGNMENT_OFFSET=10,
        )
        original = deepcopy(fixture.document)
        settings_path = fixture.root / "viewer.json"
        settings_path.write_text(json.dumps(original), encoding="utf-8")
        original_bytes = settings_path.read_bytes()
        with mock.patch.object(tempfile, "tempdir", str(fixture.root)):
            async with Client(mcp) as client:
                for arguments in ({"settings_document": fixture.document}, {"settings_path": str(settings_path)}):
                    response = await client.call_tool("emapssn_viewer_control", {
                        "action": "validate_settings", "arguments": arguments,
                    })
                    self.assertFalse(response.is_error, str(response))
                    result = response.structured_content
                    self.assertEqual(set(result), {"valid", "settings_document"})
                    self.assertTrue(result["valid"])
                    self.assertEqual(result["settings_document"]["schema_version"], 2)
                    self.assertEqual(result["settings_document"]["alignment"], original["alignment"])
                    self.assertEqual(fixture.document, original)
                    self.assertEqual(settings_path.read_bytes(), original_bytes)

    async def test_visible_viewer_launch_through_headless_config(self):
        fixture = SettingsTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(fixture.root / "sessions"),
                                         "PYTHONIOENCODING": "utf-8"}):
            client = MCPViewerClient(fixtures.ROOT)
            info = None
            try:
                info = await client.launch_session(settings_document=fixture.document, mode="normal", timeout=45)
                self.assertEqual(info["mode"], "normal")
                summary = await client.get_summary(info["session_id"])
                self.assertEqual(summary["node_count"], 2)
                output = await client.read_log(info["session_id"])
                self.assertTrue(output["text"], "Visible Viewer must retain startup terminal output")
                self.assertIn(f'[{info["session_alias"]}]', summary["window_title"])
                self.assertEqual(summary["provenance"]["status"], "complete")
                connected = await client.connect_session(info["session_alias"].lower())
                self.assertEqual(connected["session_id"], info["session_id"])
                listed = await client.list_sessions()
                item = next(s for s in listed["sessions"] if s["session_id"] == info["session_id"])
                self.assertEqual(item["inputs"]["layout_cache"], fixture.cache)
                self.assertEqual(item["session_alias"], info["session_alias"])
                if sys.platform == "win32":
                    import ctypes
                    import subprocess
                    from ctypes import wintypes
                    identity = json.loads((Path(info["stdout_log"]).parent / "terminal-process.json").read_text())
                    console = subprocess.run([sys.executable, "-c",
                        "import ctypes,sys; k=ctypes.windll.kernel32; k.FreeConsole(); "
                        "attached=k.AttachConsole(int(sys.argv[1])); k.GetConsoleWindow.restype=ctypes.c_void_p; "
                        "h=k.GetConsoleWindow(); print(bool(attached and h)); k.FreeConsole()",
                        str(identity["pid"])], capture_output=True, text=True, timeout=5)
                    self.assertEqual(console.stdout.strip(), "True", console.stderr)
                    user32 = ctypes.windll.user32
                    windows = []
                    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                    @callback_type
                    def collect(hwnd, _):
                        pid = wintypes.DWORD()
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        title = ctypes.create_unicode_buffer(512)
                        user32.GetWindowTextW(hwnd, title, 512)
                        if pid.value == info["pid"] and info["session_alias"] in title.value:
                            windows.append(hwnd)
                        return True
                    user32.EnumWindows(collect, 0)
                    self.assertTrue(windows, "Expected a native Viewer window")
                    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                    self.assertTrue(user32.PostMessageW(windows[0], 0x0010, 0, 0))  # WM_CLOSE, as with X
                    deadline = asyncio.get_running_loop().time() + 10
                    while client.connected_session_id and asyncio.get_running_loop().time() < deadline:
                        await asyncio.sleep(0.1)
                    self.assertIsNone(client.connected_session_id)
                    self.assertTrue((await client.read_log(info["session_id"]))["text"])
                    self.assertFalse(psutil.pid_exists(info["pid"]))
                    info = None
            finally:
                if info:
                    await client.close_session(info["session_id"])
                    self.assertFalse(psutil.pid_exists(info["pid"]))

    async def test_timeout_and_cancellation_leave_a_loading_viewer_running(self):
        # A Viewer that is still loading when the wait ends, or when the MCP
        # call is cancelled (as a client-side tool timeout does), must survive.
        # On 2026-10-02 a fixed 30 s deadline killed a healthy Foldtype IV
        # Viewer while it built its display, seconds from publishing a session.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "EMAPSSN_Config.py").write_text(
                "import time,sys\nprint('x'*1000000,flush=True)\nprint('Loading phase two',flush=True)\ntime.sleep(60)\n"
            )
            with mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(root / "sessions")}), mock.patch(
                "mcp_server.viewer.Viewer_Client.validate_viewer_document", return_value={"inputs": {"TARGET_CACHE_PATH": "test"}}
            ):
                client = MCPViewerClient(root)
                started = await client.launch_session(settings_document={}, mode="headless", timeout=0.3)
                self.assertEqual(started["status"], "starting")
                self.assertEqual(started["next_step"], {"tool": "emapssn_viewer_control", "action": "wait_session",
                                                        "arguments": {"launch_id": started["launch_id"]}})
                self.assertIsNone(client.connected_session_id)
                launcher = psutil.Process(started["launcher_pid"])
                self.assertTrue(launcher.is_running())
                deadline = asyncio.get_running_loop().time() + 10
                while (await client.wait_for_launch(started["launch_id"], timeout=0.1))["phase"] != "Loading phase two":
                    self.assertLess(asyncio.get_running_loop().time(), deadline)
                self.assertTrue(launcher.is_running())
                closed = await client.close_session(launch_id=started["launch_id"])
                self.assertEqual((closed["closed"], closed["session_id"], closed["launch_id"]),
                                 (True, None, started["launch_id"]))
                self.assertFalse(launcher.is_running())
                with self.assertRaisesRegex(MCPViewerError, "exited before readiness.*Loading phase two"):
                    await client.wait_for_launch(started["launch_id"], timeout=1)

                task = asyncio.create_task(client.launch_session(settings_document={}, mode="headless"))
                deadline = asyncio.get_running_loop().time() + 10
                while not any(p.stat().st_size >= 1000000 for p in (root / "sessions").rglob("stdout.log")
                              if p.parent.name != started["launch_id"]):
                    if task.done() or asyncio.get_running_loop().time() >= deadline:
                        break
                    await asyncio.sleep(0.1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                cancelled = next(p for p in (root / "sessions" / "launches").iterdir() if p.name != started["launch_id"])
                identity = json.loads((cancelled / "process.json").read_text())
                self.assertTrue(psutil.pid_exists(identity["pid"]))
                await client.close_session(launch_id=cancelled.name)
                self.assertFalse(psutil.pid_exists(identity["pid"]))

    async def test_exit_before_readiness_reports_phase_and_stops_the_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            # Outlive the launch broker so the exit is observed while waiting.
            (root / "src" / "EMAPSSN_Config.py").write_text(
                "import sys,time\nprint('Reading the network',flush=True)\ntime.sleep(1.5)\nsys.exit(3)\n"
            )
            with mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(root / "sessions")}), mock.patch(
                "mcp_server.viewer.Viewer_Client.validate_viewer_document", return_value={"inputs": {"TARGET_CACHE_PATH": "test"}}
            ):
                with self.assertRaisesRegex(MCPViewerError, "exited before readiness.*'Reading the network'.*terminated"):
                    await MCPViewerClient(root).launch_session(settings_document={}, mode="headless", timeout=30)
            directory = next((root / "sessions" / "launches").iterdir())
            self.assertFalse((directory / "settings.json").exists())

    async def test_readiness_probe_failures_are_retried_until_ready_or_handed_off(self):
        # A published Viewer whose Qt thread is still busy (first paint, startup
        # commands) answers the probe with 503; that is not a failed launch.
        from types import SimpleNamespace
        import time
        session = SimpleNamespace(session_id="s", launch_id="abc", pid=1, base_url="http://127.0.0.1:9",
                                  started_at="now", token="t", process_created_at=None)
        launch = {"launch_id": "abc", "mode": "headless", "cache_path": "cache.h5", "started_epoch": time.time()}
        with tempfile.TemporaryDirectory() as temporary:
            for outcomes, expected in (([MCPViewerError("busy"), MCPViewerError("busy"), {}], "ready"),
                                       ([MCPViewerError("busy")] * 1000, "starting")):
                with self.subTest(expected=expected):
                    client = MCPViewerClient(temporary)
                    remaining = list(outcomes)
                    def probe(*_arguments):
                        outcome = remaining.pop(0)
                        if isinstance(outcome, Exception):
                            raise outcome
                        return outcome
                    with mock.patch("mcp_server.viewer.Viewer_Client.discover_viewer_sessions", return_value=[session]), \
                            mock.patch.object(client, "_request", side_effect=probe), \
                            mock.patch.object(client, "_remember_session"):
                        result = await client._await_launch(Path(temporary), launch, lambda: True, 1.0)
                    self.assertEqual(result["status"], expected)
                    if expected == "ready":
                        self.assertEqual(client.connected_session_id, "s")
                    else:
                        self.assertEqual(result["readiness_probe_error"], "busy")

    async def test_wait_and_close_reject_unknown_or_malformed_launch_ids(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": temporary}):
            client = MCPViewerClient(temporary)
            for launch_id in ("../launches", "not-a-uuid", ""):
                with self.subTest(launch_id=launch_id), self.assertRaisesRegex(MCPViewerError, "launch_id"):
                    await client.wait_for_launch(launch_id)
            unknown = "0" * 32
            with self.assertRaisesRegex(MCPViewerError, "No MCP launch"):
                await client.wait_for_launch(unknown)
            damaged = Path(temporary) / "launches" / ("1" * 32)
            damaged.mkdir(parents=True)
            (damaged / "launch.json").write_text(json.dumps({"launch_id": damaged.name, "mode": "headless"}))
            with self.assertRaisesRegex(MCPViewerError, "Invalid launch record"):
                await client.wait_for_launch(damaged.name)
            with self.assertRaisesRegex(MCPViewerError, "No MCP launch"):
                await client.close_session(launch_id=unknown)
            with self.assertRaisesRegex(MCPViewerError, "not both"):
                await client.close_session("session", launch_id=unknown)

    async def test_slow_launch_hands_off_to_wait_session_over_mcp(self):
        fixture = SettingsTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        launch_id = None
        with mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(fixture.root / "sessions"),
                                         "PYTHONIOENCODING": "utf-8"}), \
                mock.patch.object(tempfile, "tempdir", str(fixture.root)):
            async with Client(mcp) as client:
                try:
                    described = await client.call_tool("emapssn_viewer_control", {
                        "action": "describe", "arguments": {"action": "start_session"}})
                    timeout_schema = described.structured_content["arguments_schema"]["properties"]["ready_timeout"]
                    self.assertEqual((timeout_schema["default"], timeout_schema["maximum"]), (45.0, 600))
                    started = await client.call_tool("emapssn_viewer_control", {"action": "start_session", "arguments": {
                        "mode": "headless", "settings_document": fixture.document, "ready_timeout": 0.01}})
                    self.assertFalse(started.is_error, str(started))
                    self.assertEqual(started.structured_content["status"], "starting")
                    launch_id = started.structured_content["launch_id"]
                    self.assertTrue((await client.call_tool("emapssn_viewer_data", {
                        "action": "get_summary", "arguments": {}})).is_error)
                    ready = await client.call_tool("emapssn_viewer_control", {"action": "wait_session", "arguments": {
                        "launch_id": launch_id, "ready_timeout": 120}})
                    self.assertFalse(ready.is_error, str(ready))
                    self.assertEqual(ready.structured_content["status"], "ready")
                    self.assertEqual(ready.structured_content["launch_id"], launch_id)
                    summary = await client.call_tool("emapssn_viewer_data", {"action": "get_summary", "arguments": {}})
                    self.assertFalse(summary.is_error, str(summary))
                    self.assertEqual(summary.structured_content["node_count"], 2)
                    sessions = await client.call_tool("emapssn_viewer_data", {"action": "list_sessions", "arguments": {}})
                    self.assertFalse(sessions.is_error, str(sessions))
                    self.assertIn(ready.structured_content["session_id"],
                                  [entry["session_id"] for entry in sessions.structured_content["sessions"]])
                    nodes = await client.call_tool("emapssn_viewer_data", {"action": "query_nodes", "arguments": {
                        "snapshot_id": summary.structured_content["snapshot_id"], "limit": 10}})
                    self.assertFalse(nodes.is_error, str(nodes))
                    self.assertEqual(len(nodes.structured_content["rows"]), 2)
                    output = await client.call_tool("emapssn_viewer_data", {"action": "read_log", "arguments": {}})
                    self.assertIn("Building network display: 2 nodes", output.structured_content["text"])
                    closed = await client.call_tool("emapssn_viewer_control", {"action": "close_session", "arguments": {
                        "launch_id": launch_id}})
                    self.assertFalse(closed.is_error, str(closed))
                    self.assertFalse(psutil.pid_exists(ready.structured_content["pid"]))
                    launch_id = None
                finally:
                    if launch_id is not None:
                        await MCPViewerClient(fixtures.ROOT).close_session(launch_id=launch_id)

    @unittest.skipUnless(sys.platform == "win32", "Windows console creation flags")
    async def test_windows_launch_keeps_headless_viewers_off_screen(self):
        # A venv's python.exe redirector started with DETACHED_PROCESS has no
        # console, so Windows opened a new, visible terminal for the interpreter it
        # started. The broker and headless children must use a hidden console;
        # normal mode still opens its terminal deliberately.
        expected_child_flag = {"headless": "CREATE_NO_WINDOW", "normal": "CREATE_NEW_CONSOLE"}
        for mode, child_flag in expected_child_flag.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, \
                    mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(Path(temporary) / "sessions")}), \
                    mock.patch(
                        "mcp_server.viewer.Viewer_Client.validate_viewer_document",
                        return_value={"inputs": {"TARGET_CACHE_PATH": "test"}},
                    ), \
                    mock.patch("mcp_server.viewer.Viewer_Client._identity", return_value=None), \
                    mock.patch("mcp_server.viewer.Viewer_Client.subprocess.Popen") as popen:
                popen.return_value.wait.return_value = 1  # broker fails; nothing real starts
                with self.assertRaisesRegex(MCPViewerError, "broker failed"):
                    await MCPViewerClient(Path(temporary)).launch_session(settings_document={}, mode=mode)

                flags = popen.call_args.kwargs["creationflags"]
                broker_code = popen.call_args.args[0][2]
                self.assertTrue(flags & subprocess.CREATE_NO_WINDOW)
                self.assertFalse(flags & subprocess.DETACHED_PROCESS)
                self.assertIn(f"creationflags=subprocess.{child_flag}", broker_code)
                self.assertNotIn("DETACHED_PROCESS", broker_code)

    def _use_posix_terminal(self, script):
        """Route normal launches through the Linux/macOS terminal path to ``script``.

        The stand-in terminal command runs ``script`` and never the Viewer, as an
        emulator without a display or osascript refused control of Terminal does.
        Returns the client and its session root.
        """
        from types import SimpleNamespace
        from mcp_server.viewer import Viewer_Client

        terminals = []

        def launch_in_terminal(command, *, cwd, env, **streams):
            terminals.append(subprocess.Popen([sys.executable, "-c", script], cwd=cwd, env=env, **streams))
            return terminals[-1]

        def stop_terminals():
            for terminal in terminals:
                if terminal.poll() is None:
                    terminal.kill()
                terminal.wait(timeout=10)

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.addCleanup(stop_terminals)  # before the directory is removed
        self.enterContext(mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(root / "sessions")}))
        self.enterContext(mock.patch.object(Viewer_Client, "validate_viewer_document",
                                            return_value={"inputs": {"TARGET_CACHE_PATH": "test"}}))
        # A stand-in sys selects the POSIX branch on every host, leaving the real one alone.
        self.enterContext(mock.patch.object(Viewer_Client, "sys",
                                            SimpleNamespace(platform="linux", executable=sys.executable)))
        self.enterContext(mock.patch("utilities.Terminal_Launcher.launch_in_terminal", launch_in_terminal))
        return MCPViewerClient(root), root

    async def test_posix_terminal_that_fails_is_an_error_with_its_output(self):
        # Until 2026-10-04 such a launch was handed off as "starting", and every
        # wait_session said the same, although nothing was running.
        client, root = self._use_posix_terminal(
            "import sys,time; print('cannot open display', file=sys.stderr, flush=True); "
            "time.sleep(0.5); sys.exit(1)"
        )
        with self.assertRaisesRegex(MCPViewerError,
                                    r"(?s)exited with code 1 before starting the Viewer.*cannot open display"):
            await client.launch_session(settings_document={}, mode="normal", timeout=10)
        directory = next((root / "sessions" / "launches").iterdir())
        self.assertFalse((directory / "settings.json").exists())

    async def test_posix_terminal_that_exits_cleanly_without_the_viewer_times_out(self):
        # Emulators that hand their window to a running server exit 0 at once,
        # so only the wrapper's identity file can show the Viewer started.
        from mcp_server.viewer import Viewer_Client

        client, _root = self._use_posix_terminal("pass")
        started = await client.launch_session(settings_document={}, mode="normal", timeout=0.3)
        self.assertEqual(started["status"], "starting")
        with mock.patch.object(Viewer_Client, "TERMINAL_START_GRACE", 0), \
                self.assertRaisesRegex(MCPViewerError, "No terminal started the Viewer within 0 s"):
            await client.wait_for_launch(started["launch_id"], timeout=10)

    async def test_posix_terminal_still_running_keeps_the_launch_until_closed(self):
        # macOS's osascript waits while the user is asked to allow control of
        # Terminal; that wait is not a failure, and close_session must stop it.
        from mcp_server.viewer import Viewer_Client

        client, root = self._use_posix_terminal("import time; time.sleep(60)")
        with mock.patch.object(Viewer_Client, "TERMINAL_START_GRACE", 0):
            started = await client.launch_session(settings_document={}, mode="normal", timeout=0.5)
            self.assertEqual(started["status"], "starting")
            waited = await client.wait_for_launch(started["launch_id"], timeout=0.5)
            self.assertEqual(waited["status"], "starting")
        directory = root / "sessions" / "launches" / started["launch_id"]
        terminal = json.loads((directory / "process.json").read_text())
        closed = await client.close_session(launch_id=started["launch_id"])
        self.assertEqual((closed["closed"], closed["pid"]), (True, terminal["pid"]))
        self.assertFalse(psutil.pid_exists(terminal["pid"]))

    async def test_stdio_survival_connection_isolation_and_verified_close(self):
        fixture = SettingsTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        msa_path = fixture.root / "alignment.fasta"
        msa_path.write_text(">Alpha_Beta\nAA\n>Gamma_Delta\nCC\n", encoding="utf-8")
        fixture.document["alignment"].update(
            MSA_FILE=str(msa_path), ALIGNMENT_REFERENCE="absent",
            FILTER_MIN_OCCUPANCY=75, ALIGNMENT_OFFSET=10,
        )
        directory = fixture.root / "sessions"
        environment = dict(os.environ, SSN_VIEWER_SESSION_DIR=str(directory), PYTHONIOENCODING="utf-8",
                           TEMP=str(fixture.root), TMP=str(fixture.root))
        parameters = StdioServerParameters(command=sys.executable,
            args=["-u", str(fixtures.SRC / "EMAPSSN_MCP_Server.py")], cwd=str(fixtures.ROOT), env=environment)
        info = None
        with mock.patch.dict(os.environ, environment), mock.patch.object(tempfile, "tempdir", str(fixture.root)), mock.patch("mcp.os.win32.utilities._create_job_object", return_value=None):
            try:
                async with Client(parameters, read_timeout_seconds=60) as first:
                    started = await first.call_tool("emapssn_viewer_control", {"action": "start_session", "arguments": {"mode": "headless", "settings_document": fixture.document}})
                    self.assertFalse(started.is_error, str(started))
                    info = started.structured_content
                    output = await first.call_tool("emapssn_viewer_data", {"action": "read_log", "arguments": {}})
                    self.assertFalse(output.is_error, str(output))
                    self.assertTrue(output.structured_content["text"])
                    self.assertIn("pure occupancy mode", output.structured_content["text"])
                    self.assertIn("alignment offsets are inactive", output.structured_content["text"])
                    summary = await first.call_tool("emapssn_viewer_data", {"action": "get_summary", "arguments": {"include_alignment": True}})
                    self.assertFalse(summary.is_error, str(summary))
                    alignment = summary.structured_content["alignment"]
                    self.assertTrue(alignment["available"])
                    self.assertEqual(alignment["mapped_nodes"], 2)
                    self.assertEqual(alignment["requested_reference"], "absent")
                    self.assertEqual(alignment["reference"], "None")
                    residues = await first.call_tool("emapssn_viewer_data", {
                        "action": "get_residue_distribution", "arguments": {
                            "snapshot_id": summary.structured_content["snapshot_id"],
                            "positions": ["1", "2"],
                        },
                    })
                    self.assertFalse(residues.is_error, str(residues))
                    command = await first.call_tool("emapssn_viewer_control", {
                        "action": "execute_commands", "arguments": {
                            "submission_id": "missing-reference-help", "commands": ["offset help"],
                        },
                    })
                    self.assertFalse(command.is_error, str(command))
                    request_id = command.structured_content["request_id"]
                    deadline = asyncio.get_running_loop().time() + 10
                    while True:
                        state = await first.call_tool("emapssn_viewer_data", {
                            "action": "get_command_request", "arguments": {"request_id": request_id},
                        })
                        self.assertFalse(state.is_error, str(state))
                        if state.structured_content["status"] in {"succeeded", "failed", "cancelled"}:
                            break
                        self.assertLess(asyncio.get_running_loop().time(), deadline, str(state))
                        await asyncio.sleep(0.1)
                    self.assertEqual(state.structured_content["status"], "succeeded", str(state))
                    async with Client(mcp) as second:
                        self.assertTrue((await second.call_tool("emapssn_viewer_data", {"action": "get_summary", "arguments": {}})).is_error)
                        connected = await second.call_tool("emapssn_viewer_control", {"action": "connect_session", "arguments": {"session_id": info["session_id"]}})
                        self.assertFalse(connected.is_error, str(connected))
                        self.assertFalse((await second.call_tool("emapssn_viewer_control", {"action": "disconnect_session", "arguments": {}})).is_error)
                        self.assertTrue((await second.call_tool("emapssn_viewer_data", {"action": "get_summary", "arguments": {}})).is_error)
                        response = await first.call_tool("emapssn_viewer_data", {"action": "get_summary", "arguments": {}})
                        self.assertFalse(response.is_error, str(response))
                # The STDIO server is gone; the independent Viewer must still answer.
                async with Client(parameters, read_timeout_seconds=60) as third:
                    connected = await third.call_tool("emapssn_viewer_control", {"action": "connect_session", "arguments": {"session_id": info["session_id"]}})
                    self.assertFalse(connected.is_error, str(connected))
                    summary = await third.call_tool("emapssn_viewer_data", {"action": "get_summary", "arguments": {}})
                    self.assertEqual(summary.structured_content["node_count"], 2)
                    closed = await third.call_tool("emapssn_viewer_control", {"action": "close_session", "arguments": {}})
                    self.assertFalse(closed.is_error, str(closed))
                    self.assertFalse(psutil.pid_exists(info["pid"]))
                    info = None
            finally:
                if info is not None:
                    if psutil.pid_exists(info["pid"]):
                        try:
                            await MCPViewerClient().close_session(info["session_id"])
                        except MCPViewerError:
                            try:
                                process = psutil.Process(info["pid"]); process.terminate(); process.wait(5)
                            except psutil.NoSuchProcess:
                                pass

    async def test_disconnect_and_failed_close_preserve_state(self):
        client = MCPViewerClient()
        client.connected_session_id = "existing"
        with mock.patch("mcp_server.viewer.Viewer_Client.select_viewer_session", side_effect=LookupError("unreachable")):
            with self.assertRaises(MCPViewerError):
                await client.close_session()
        self.assertEqual(client.connected_session_id, "existing")
        self.assertEqual((await client.disconnect_session())["session_id"], "existing")
        self.assertIsNone((await client.disconnect_session())["session_id"])
        with self.assertRaises(MCPViewerError):
            await client.get_summary()

    async def test_failed_termination_does_not_remove_descriptor(self):
        from types import SimpleNamespace
        session = SimpleNamespace(session_id="test", pid=123, process_created_at=None,
                                  base_url="http://127.0.0.1:9", token="test")
        client = MCPViewerClient()
        client.connected_session_id = "test"
        with mock.patch("mcp_server.viewer.Viewer_Client.select_viewer_session", return_value=session), mock.patch(
            "mcp_server.viewer.Viewer_Client._identity", return_value=mock.Mock()
        ), mock.patch("mcp_server.viewer.Viewer_Client._alive", return_value=True), mock.patch(
            "mcp_server.viewer.Viewer_Client.urllib.request.urlopen", side_effect=OSError("offline")
        ), mock.patch("mcp_server.viewer.Viewer_Client._terminate_tree", side_effect=psutil.AccessDenied(123)), mock.patch(
            "mcp_server.viewer.Viewer_Client.remove_viewer_session"
        ) as remove:
            with self.assertRaises(MCPViewerError):
                await client.close_session(timeout=0)
            remove.assert_not_called()
        self.assertEqual(client.connected_session_id, "test")


if __name__ == "__main__":
    unittest.main()
