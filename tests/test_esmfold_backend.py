"""The ESMFold web backend (src/web_ui/esmfold_backend.py): the Mol* session
that esmfold.html saves into and restores from the selected layout-cache
folder, the actions that reach those handlers, and the folded-structure event.
Opening the page itself is covered in test_browser_page_opening."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_ui import esmfold_backend  # noqa: E402


class RecordingViewer:
    def __init__(self):
        self.events = []

    def broadcast_event(self, event):
        self.events.append(event)


class RecordingRegistry:
    def __init__(self):
        self.actions = {}
        self.routes = {}

    def register_action(self, plugin, name, handler):
        self.actions[(plugin, name)] = handler

    def register_static_route(self, plugin, prefix, directory):
        self.routes[(plugin, prefix)] = directory


class ConsoleButtonTests(unittest.TestCase):
    """The Fold View page declares its own button for the Console's button row."""

    def test_activation_registers_the_button_with_its_icon_and_page(self):
        viewer = mock.Mock(spec=["add_console_button", "_open_web_ui"])
        esmfold_backend.activate(viewer)
        viewer.add_console_button.assert_called_once()
        name, label, callback, tooltip = viewer.add_console_button.call_args.args
        self.assertEqual((name, label), ("fold_view_btn", "Fold View"))
        self.assertEqual(tooltip, "Open ESMFold & Mol* structure viewer")
        self.assertEqual(viewer.add_console_button.call_args.kwargs, {"icon": "dna"})
        callback()  # a click opens the Mol* page, from the Viewer
        viewer._open_web_ui.assert_called_once()
        self.assertEqual(viewer._open_web_ui.call_args.args[0], "/esmfold.html")

    def test_a_viewer_with_no_console_button_registration_is_left_alone(self):
        esmfold_backend.activate(mock.Mock(spec=[]))

    def test_the_backend_no_longer_reaches_for_the_sidebar(self):
        self.assertNotIn("add_sidebar_button", (SRC_DIR / "web_ui" / "esmfold_backend.py").read_text(encoding="utf-8"))


class MolstarSessionTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.layout_dir = os.path.join(temp_dir.name, "layouts", "set_[model]")
        self.session_file = os.path.join(self.layout_dir, "molstar_session.json")
        patcher = mock.patch.object(
            esmfold_backend,
            "resolve_selected_cache",
            return_value=os.path.join(self.layout_dir, "version_00.h5"),
        )
        self.resolve = patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer = RecordingViewer()

    def test_saved_session_is_restored_by_a_later_load(self):
        session = {"camera": {"radius": 12.5}, "structures": ["/structures/a.pdb"]}
        esmfold_backend.handle_save_session(self.viewer, {"session": session})

        with open(self.session_file, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), session)
        self.resolve.assert_called_with(esmfold_backend.cfg)
        self.assertEqual(self.viewer.events, [])

        esmfold_backend.handle_load_session(self.viewer, {})
        self.assertEqual(
            self.viewer.events, [{"type": "restore_session", "session": session}]
        )

    def test_save_without_a_session_writes_nothing(self):
        esmfold_backend.handle_save_session(self.viewer, {})
        esmfold_backend.handle_save_session(self.viewer, {"session": None})
        self.assertFalse(os.path.exists(self.layout_dir))

    def test_load_without_a_saved_session_restores_an_empty_scene(self):
        esmfold_backend.handle_load_session(self.viewer, {})
        self.assertEqual(
            self.viewer.events, [{"type": "restore_session", "session": None}]
        )

    def test_unreadable_session_restores_an_empty_scene(self):
        os.makedirs(self.layout_dir)
        with open(self.session_file, "w", encoding="utf-8") as handle:
            handle.write("{")
        with redirect_stdout(io.StringIO()) as output:
            esmfold_backend.handle_load_session(self.viewer, {})

        self.assertEqual(
            self.viewer.events, [{"type": "restore_session", "session": None}]
        )
        self.assertTrue(output.getvalue().startswith("Error loading Mol* session: "))

    def test_failed_save_is_reported_without_raising(self):
        # A file stands where the cache folder should be.
        os.makedirs(os.path.dirname(self.layout_dir))
        Path(self.layout_dir).write_text("not a folder", encoding="utf-8")
        with redirect_stdout(io.StringIO()) as output:
            esmfold_backend.handle_save_session(self.viewer, {"session": {"a": 1}})

        self.assertTrue(output.getvalue().startswith("Error saving Mol* session: "))
        self.assertEqual(Path(self.layout_dir).read_text(encoding="utf-8"), "not a folder")

    def test_page_actions_reach_the_session_handlers(self):
        registry = RecordingRegistry()
        with mock.patch.object(
            esmfold_backend, "get_structures_directory", return_value="structures"
        ):
            esmfold_backend.register_backend(registry, self.viewer)

        session = {"structures": []}
        registry.actions[("esmfold", "save_molstar_session")]({"session": session})
        registry.actions[("esmfold", "load_molstar_session")]({})
        self.assertEqual(
            self.viewer.events, [{"type": "restore_session", "session": session}]
        )
        self.assertEqual(registry.routes[("esmfold", "/structures/")], "structures")


class JobEventTests(unittest.TestCase):
    """How a folding run is going, as the worker posts it, and what the page is told of it."""

    def test_the_fields_the_page_shows_are_passed_on_bounded(self):
        viewer = RecordingViewer()
        esmfold_backend.handle_esmfold_job(viewer, {
            "event": "structure_failed", "name": "n" * 500, "reason": "r" * 900, "log": "L" * 900, "index": 2, "total": 5,
            "secret": "not for the page", "names": ["a", "b"] + ["x"] * 80, "succeeded": True, "failed": True,
        })
        (message,) = viewer.events
        self.assertEqual(message["type"], "esmfold_job")
        self.assertEqual(message["event"], "structure_failed")
        self.assertEqual((len(message["name"]), len(message["reason"]), len(message["log"])), (200, 500, 500))
        self.assertEqual((message["index"], message["total"]), (2, 5))
        self.assertNotIn("secret", message)
        # A boolean is not a count, and a list of names is cut short.
        self.assertNotIn("succeeded", message)
        self.assertEqual(len(message["names"]), 50)
        self.assertIs(message["failed"], True)

    def test_an_event_the_page_does_not_know_is_ignored(self):
        viewer = RecordingViewer()
        for data in ({}, {"event": "explode"}, {"event": 3}, {"event": None, "total": 2}):
            esmfold_backend.handle_esmfold_job(viewer, data)
        self.assertEqual(viewer.events, [])

    def test_the_action_reaches_the_handler(self):
        registry = RecordingRegistry()
        viewer = RecordingViewer()
        with mock.patch.object(esmfold_backend, "get_structures_directory", return_value="structures"):
            esmfold_backend.register_backend(registry, viewer)
        registry.actions[("esmfold", "esmfold_job")]({"event": "started", "total": 3})
        self.assertEqual(viewer.events, [{"type": "esmfold_job", "event": "started", "total": 3}])


class ViewerInfoTests(unittest.TestCase):
    """The page asks for its session when it connects; the Viewer says then whether it has a window."""

    class Viewer(RecordingViewer):
        def __init__(self, shows):
            super().__init__()
            self.shows = shows

        def shows_windows(self):
            return self.shows

    def load(self, viewer):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            esmfold_backend, "resolve_selected_cache", return_value=os.path.join(folder, "version_00.h5")
        ):
            esmfold_backend.handle_load_session(viewer, {})

    def test_a_viewer_with_no_window_says_so_before_the_session(self):
        viewer = self.Viewer(shows=False)
        self.load(viewer)
        self.assertEqual(viewer.events, [{"type": "viewer_info", "hidden": True}, {"type": "restore_session", "session": None}])

    def test_a_viewer_with_a_window_says_it_has_one(self):
        viewer = self.Viewer(shows=True)
        self.load(viewer)
        self.assertEqual(viewer.events[0], {"type": "viewer_info", "hidden": False})

    def test_a_viewer_that_cannot_say_sends_nothing_of_it(self):
        viewer = RecordingViewer()
        self.load(viewer)
        self.assertEqual([event["type"] for event in viewer.events], ["restore_session"])


class FoldedStructureEventTests(unittest.TestCase):
    def test_folded_structure_is_announced_with_its_url(self):
        viewer = RecordingViewer()
        with redirect_stdout(io.StringIO()):
            esmfold_backend.handle_structure_folded(
                viewer, {"node_id": "WP_1", "pdb_filename": "WP_1.pdb"}
            )
        self.assertEqual(
            viewer.events,
            [{"type": "esmfold_pdb", "node_id": "WP_1", "pdb_url": "/structures/WP_1.pdb"}],
        )

    def test_incomplete_fold_reports_are_ignored(self):
        viewer = RecordingViewer()
        for data in ({"node_id": "WP_1"}, {"pdb_filename": "WP_1.pdb"}, {}):
            esmfold_backend.handle_structure_folded(viewer, data)
        self.assertEqual(viewer.events, [])


class OpenLogTests(unittest.TestCase):
    """The page's Open log button names a run; the Viewer finds the log it made for that run, and opens it only where a person is."""

    class Viewer(RecordingViewer):
        def __init__(self, shows=True):
            super().__init__()
            self.shows = shows

        def shows_windows(self):
            return self.shows

    def setUp(self):
        import Viewer_Worker_Tracking
        from commands import esmfold
        self.esmfold = esmfold
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        patcher = mock.patch.object(Viewer_Worker_Tracking, "__file__", str(self.root / "src" / "Viewer_Worker_Tracking.py"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.run = esmfold.FoldRun([], [], mode="large", device=None, structures_dir=str(self.root), action_url="", portal=False)
        Path(self.run.log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.run.log_path).write_text("worker output", encoding="utf-8")
        esmfold.RUNS[self.run.run_id] = self.run
        self.addCleanup(esmfold.RUNS.clear)
        patcher = mock.patch("desktop.Viewer_Jobs.Desktop_Open.open_path", return_value=True)
        self.open_url = patcher.start()
        self.addCleanup(patcher.stop)

    def not_opened(self, viewer):
        return [event for event in viewer.events if event == {"type": "esmfold_job", "event": "log_not_opened"}]

    def test_the_log_of_the_run_the_page_names_is_opened_and_nothing_else(self):
        viewer = self.Viewer()
        esmfold_backend.handle_open_log(viewer, {
            "run": self.run.run_id, "path": str(self.root / "elsewhere.txt"), "log": r"C:\Windows\win.ini"})
        self.open_url.assert_called_once()
        opened = self.open_url.call_args.args[0]
        self.assertEqual(os.path.normcase(opened), os.path.normcase(os.path.abspath(self.run.log_path)))
        self.assertEqual(viewer.events, [])

    def test_a_run_this_viewer_did_not_start_or_a_bad_id_opens_nothing_and_the_page_is_told(self):
        for data in ({}, {"run": "nosuchrun"}, {"run": "../../etc/passwd"}, {"run": 5}, {"path": self.run.log_path}):
            with self.subTest(data=data):
                self.open_url.reset_mock()
                viewer = self.Viewer()
                esmfold_backend.handle_open_log(viewer, data)
                self.open_url.assert_not_called()
                self.assertEqual(len(self.not_opened(viewer)), 1)

    def test_a_viewer_with_no_window_opens_nothing_and_says_so(self):
        viewer = self.Viewer(shows=False)
        esmfold_backend.handle_open_log(viewer, {"run": self.run.run_id})
        self.open_url.assert_not_called()
        self.assertEqual(len(self.not_opened(viewer)), 1)

    def test_a_log_that_is_gone_opens_nothing_and_says_so(self):
        os.unlink(self.run.log_path)
        viewer = self.Viewer()
        esmfold_backend.handle_open_log(viewer, {"run": self.run.run_id})
        self.open_url.assert_not_called()
        self.assertEqual(len(self.not_opened(viewer)), 1)

    def test_a_log_outside_the_workers_folder_is_never_named(self):
        outside = self.root / "outside.log"
        outside.write_text("not a worker log", encoding="utf-8")
        self.run.log_path = str(outside)
        self.assertIsNone(self.esmfold.log_of(self.run.run_id))
        viewer = self.Viewer()
        esmfold_backend.handle_open_log(viewer, {"run": self.run.run_id})
        self.open_url.assert_not_called()
        self.assertEqual(len(self.not_opened(viewer)), 1)

    def test_the_action_reaches_the_handler(self):
        registry = RecordingRegistry()
        viewer = self.Viewer()
        with mock.patch.object(esmfold_backend, "get_structures_directory", return_value="structures"):
            esmfold_backend.register_backend(registry, viewer)
        registry.actions[("esmfold", "esmfold_open_log")]({"run": self.run.run_id})
        self.open_url.assert_called_once()

    def test_an_event_of_a_known_run_carries_the_log_the_viewer_made_not_the_workers_own(self):
        viewer = self.Viewer()
        esmfold_backend.handle_esmfold_job(viewer, {
            "event": "failed", "reason": "stopped", "run": self.run.run_id, "log": r"C:\temp\esmfold-x.json.stdout"})
        [message] = viewer.events
        self.assertEqual(message["log"], self.run.log_path)
        self.assertEqual(message["run"], self.run.run_id)
        viewer = self.Viewer()
        esmfold_backend.handle_esmfold_job(viewer, {"event": "failed", "run": "unknownrun", "log": "worker.stdout"})
        self.assertEqual(viewer.events[0]["log"], "worker.stdout")


if __name__ == "__main__":
    unittest.main()
