"""Viewer session identity, discovery and selection (utilities.Viewer_Sessions).

Covers alias selection, the authenticated session endpoints a live web server
publishes, and which session descriptors discovery trusts with their bearer
token or prunes.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error
import urllib.request

import psutil

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from desktop.Viewer_Inspection import ViewerInspectionService  # noqa: E402
from utilities import Viewer_Sessions as sessions  # noqa: E402
from utilities.Viewer_Sessions import (  # noqa: E402
    SESSION_DIRECTORY_ENV,
    discover_viewer_sessions,
    ensure_viewer_identity,
    publish_viewer_session,
    select_viewer_session,
)


def _exited_pid():
    """The PID of a process that has exited and whose process object is gone."""
    result = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                            capture_output=True, text=True, check=True, timeout=60)
    pid = int(result.stdout)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            psutil.Process(pid)
        except psutil.NoSuchProcess:
            return pid
        time.sleep(0.05)
    raise AssertionError(f"Process {pid} still exists after it exited")


class ViewerSessionSelectionTests(unittest.TestCase):
    def test_identity_is_stable_and_alias_selection_rejects_collisions(self):
        viewer = SimpleNamespace()
        first = ensure_viewer_identity(viewer)
        self.assertEqual(ensure_viewer_identity(viewer), first)
        self.assertEqual(len(viewer.inspection_session_alias), 8)
        one = SimpleNamespace(session_id="a7c92f10-0000-4000-8000-000000000001")
        two = SimpleNamespace(session_id="a7c92f10-0000-4000-8000-000000000002")
        with mock.patch("utilities.Viewer_Sessions.discover_viewer_sessions", return_value=[one]):
            self.assertIs(select_viewer_session("a7c92f10"), one)
        with mock.patch("utilities.Viewer_Sessions.discover_viewer_sessions", return_value=[one, two]):
            with self.assertRaisesRegex(LookupError, "Ambiguous"):
                select_viewer_session("A7C92F10")
            self.assertIs(select_viewer_session(two.session_id), two)


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
            other_viewer = SimpleNamespace(**{k: v for k, v in vars(viewer).items() if not k.startswith("inspection_session")})
            other_viewer.viewer_inspection = ViewerInspectionService(other_viewer)
            second = Web_Server.start_server(other_viewer, preferred_port=0)
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
                pid=os.getpid(),
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


class SessionDescriptorTrustTests(unittest.TestCase):
    """Which descriptors discovery sends their bearer token to, and which it deletes.

    Any local process can write a descriptor file, so discovery may probe only
    the IPv4 loopback with the current protocol, accept only a responder that
    echoes the descriptor's identity, and delete only descriptors whose Viewer
    process has exited.
    """

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        environment = mock.patch.dict(os.environ, {SESSION_DIRECTORY_ENV: directory.name})
        environment.start()
        self.addCleanup(environment.stop)

    def descriptor(self, name="trusted", **overrides):
        """Write a descriptor that is valid for this live process unless overridden."""
        payload = dict(protocol_version=1, session_id=name, pid=os.getpid(), host="127.0.0.1",
                       port=8765, token="secret-token", started_at="2026-10-07T00:00:00Z",
                       launch_id=None, process_created_at=psutil.Process().create_time())
        payload.update(overrides)
        path = self.root / f"viewer-{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def answering(payload):
        """Patch urlopen so that every session probe is answered with ``payload``."""
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
        return mock.patch("urllib.request.urlopen", return_value=response)

    def test_untrusted_descriptors_never_receive_the_token(self):
        cases = {"non-loopback host": {"host": "192.0.2.1"}, "newer protocol": {"protocol_version": 2},
                 "port 0": {"port": 0}, "empty token": {"token": ""}}
        for label, overrides in cases.items():
            with self.subTest(label):
                path = self.descriptor("untrusted", **overrides)
                # The responder would confirm this very session, so only the
                # descriptor checks can refuse it.
                answer = {"protocol_version": overrides.get("protocol_version", 1),
                          "session_id": "untrusted", "pid": os.getpid()}
                with self.answering(answer) as urlopen:
                    self.assertEqual(discover_viewer_sessions(), [])
                urlopen.assert_not_called()
                self.assertTrue(path.is_file())  # ignored, not deleted
                path.unlink()
        self.descriptor()
        with self.answering({"protocol_version": 1, "session_id": "trusted", "pid": os.getpid()}) as urlopen:
            self.assertEqual([session.session_id for session in discover_viewer_sessions()], ["trusted"])
        probe = urlopen.call_args.args[0]
        self.assertEqual((probe.full_url, probe.get_header("Authorization")),
                         ("http://127.0.0.1:8765/api/mcp/v1/session", "Bearer secret-token"))

    def test_publish_refuses_hosts_other_than_the_ipv4_loopback(self):
        for host in ("0.0.0.0", "192.0.2.1", "localhost", "::1"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "loopback"):
                publish_viewer_session(session_id="refused", pid=os.getpid(), port=8765, token="t", host=host)
        self.assertEqual(list(self.root.iterdir()), [])
        published = publish_viewer_session(session_id="published", pid=os.getpid(), port=8765, token="t")
        written = json.loads(pathlib.Path(published.descriptor_path).read_text(encoding="utf-8"))
        self.assertEqual((written["host"], written["protocol_version"]), ("127.0.0.1", 1))

    def test_probe_must_echo_the_descriptor_identity(self):
        launch_id = "a" * 32
        descriptor = sessions._load_descriptor(str(self.descriptor(launch_id=launch_id)))
        answer = {"protocol_version": 1, "session_id": "trusted", "pid": os.getpid(), "launch_id": launch_id}
        with self.answering(answer):
            self.assertTrue(sessions._validate_live_session(descriptor, 1))
        for field, value in (("session_id", "other"), ("pid", os.getpid() + 1), ("launch_id", "b" * 32),
                             ("launch_id", None), ("protocol_version", 2)):
            with self.subTest(field=field, value=value), self.answering({**answer, field: value}):
                self.assertFalse(sessions._validate_live_session(descriptor, 1))
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            self.assertFalse(sessions._validate_live_session(descriptor, 1))

    def test_only_descriptors_of_exited_viewers_are_pruned(self):
        dead = self.descriptor("exited", pid=_exited_pid(), port=8766)
        live = self.descriptor("unanswered")
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")) as urlopen:
            self.assertEqual(discover_viewer_sessions(prune_stale=False), [])
            self.assertTrue(dead.is_file())
            self.assertEqual(discover_viewer_sessions(), [])
        self.assertFalse(dead.exists())
        self.assertTrue(live.is_file())  # a live Viewer that did not answer may just be busy
        # Only the live Viewer was probed: an exited Viewer's port may belong to anyone now.
        self.assertEqual({call.args[0].full_url for call in urlopen.call_args_list},
                         {"http://127.0.0.1:8765/api/mcp/v1/session"})


if __name__ == "__main__":
    unittest.main()
