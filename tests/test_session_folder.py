# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The private folder a test run works in (tests/session_folder.py and the start of
tests/__init__.py): its temp folder, how it is removed, how the folders of runs that
were killed are found and swept, and what a process started by hand does with it."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

import psutil

from tests import session_folder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ended_process_id():
    """The process ID of a process that has ended."""
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait(timeout=60)
    return done.pid


class FolderTestCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.base = folder.name

    def session(self, name="emapssn-test-sessions-x", owner=None, age=None):
        path = os.path.join(self.base, name)
        os.makedirs(os.path.join(path, "temp"))
        if owner is not None:
            Path(path, session_folder.OWNER_FILE).write_text(json.dumps(owner), encoding="utf-8")
        if age is not None:
            past = time.time() - age
            os.utime(path, (past, past))
        return path


class TempFolderTests(FolderTestCase):
    def test_adopting_a_folder_makes_its_temp_folder_and_names_this_process_as_its_owner(self):
        path = self.session()
        shutil.rmtree(os.path.join(path, "temp"))
        temp = session_folder.adopt(path)
        self.assertEqual(temp, os.path.join(path, "temp"))
        self.assertTrue(os.path.isdir(temp))
        owner = json.loads(Path(path, session_folder.OWNER_FILE).read_text(encoding="utf-8"))
        self.assertEqual(owner["pid"], os.getpid())
        self.assertAlmostEqual(owner["started"], psutil.Process().create_time(), delta=1)

    def test_what_is_left_in_the_temp_folder_is_listed(self):
        path = self.session()
        self.assertEqual(session_folder.leaked(path), [])
        Path(path, "temp", "b.txt").write_text("", encoding="utf-8")
        os.mkdir(os.path.join(path, "temp", "a"))
        self.assertEqual(session_folder.leaked(path), ["a", "b.txt"])
        self.assertEqual(session_folder.leaked(os.path.join(path, "missing")), [])

    def test_the_environment_names_the_temp_folder_under_every_variable_python_reads(self):
        posix = {"TMP": "t", "TEMP": "t", "TMPDIR": "t"}
        with mock.patch.object(session_folder.os, "name", "posix"):
            self.assertEqual(session_folder.temp_environment("t", {}), posix)
        with mock.patch.object(session_folder.os, "name", "nt"):
            self.assertEqual(session_folder.temp_environment("t", {}), {"TMP": "t", "TEMP": "t"})
            # Python reads TMPDIR before TEMP, so a TMPDIR that is there would win over the others.
            self.assertEqual(session_folder.temp_environment("t", {"TMPDIR": "elsewhere"}), posix)

    def test_using_a_temp_folder_moves_this_process_and_its_children(self):
        with mock.patch.dict(os.environ), mock.patch.object(tempfile, "tempdir", None):
            session_folder.use_temp(self.base)
            self.assertEqual(tempfile.gettempdir(), self.base)
            self.assertEqual((os.environ["TMP"], os.environ["TEMP"]), (self.base, self.base))
            done = subprocess.run([sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
                                  capture_output=True, text=True, check=True)
            self.assertEqual(os.path.normcase(done.stdout.strip()), os.path.normcase(self.base))


class RemoveTreeTests(FolderTestCase):
    def test_a_folder_with_a_read_only_file_is_removed(self):
        path = self.session()
        locked = Path(path, "temp", "locked.txt")
        locked.write_text("x", encoding="utf-8")
        os.chmod(locked, stat.S_IREAD)
        self.assertIsNone(session_folder.remove_tree(path))
        self.assertFalse(os.path.exists(path))

    def test_a_missing_folder_counts_as_removed(self):
        self.assertIsNone(session_folder.remove_tree(os.path.join(self.base, "missing")))

    def test_a_folder_windows_still_holds_is_retried_until_it_lets_go(self):
        path = self.session()
        real = shutil.rmtree
        attempts = []

        def holds_twice(target, **keywords):
            attempts.append(target)
            if len(attempts) <= 2:
                raise PermissionError(13, "in use", target)
            return real(target, **keywords)

        with mock.patch.object(session_folder.shutil, "rmtree", holds_twice):
            self.assertIsNone(session_folder.remove_tree(path, seconds=30))
        self.assertEqual(len(attempts), 3)
        self.assertFalse(os.path.exists(path))

    def test_a_folder_that_is_never_let_go_gives_the_error_after_the_time_is_up(self):
        path = self.session()

        def always_held(target, **keywords):
            raise PermissionError(13, "in use", target)

        started = time.monotonic()
        with mock.patch.object(session_folder.shutil, "rmtree", always_held):
            problem = session_folder.remove_tree(path, seconds=0.3)
        self.assertIsInstance(problem, PermissionError)
        self.assertGreaterEqual(time.monotonic() - started, 0.3)
        self.assertTrue(os.path.isdir(path))


class StaleTests(FolderTestCase):
    def test_the_folder_of_a_process_that_ended_is_stale(self):
        path = self.session(owner={"pid": ended_process_id(), "started": 1.0})
        self.assertTrue(session_folder.stale(path))

    def test_the_folder_of_a_running_process_is_not(self):
        me = psutil.Process()
        path = self.session(owner={"pid": me.pid, "started": round(me.create_time(), 2)})
        self.assertFalse(session_folder.stale(path))

    def test_a_process_ID_that_another_process_took_over_does_not_keep_the_folder(self):
        me = psutil.Process()
        path = self.session(owner={"pid": me.pid, "started": round(me.create_time(), 2) - 3600})
        self.assertTrue(session_folder.stale(path))

    def test_a_folder_that_names_no_start_time_is_judged_by_its_process_alone(self):
        self.assertFalse(session_folder.stale(self.session(owner={"pid": os.getpid(), "started": None})))

    def test_an_owner_file_that_cannot_be_read_keeps_the_folder(self):
        path = self.session()
        Path(path, session_folder.OWNER_FILE).write_text("{not json", encoding="utf-8")
        self.assertFalse(session_folder.stale(path))
        Path(path, session_folder.OWNER_FILE).write_text("[1, 2]", encoding="utf-8")
        self.assertFalse(session_folder.stale(path))

    def test_a_folder_from_before_owner_files_is_stale_once_nothing_has_touched_it_for_a_day(self):
        self.assertFalse(session_folder.stale(self.session("emapssn-test-sessions-new", age=3600)))
        self.assertTrue(session_folder.stale(self.session("emapssn-test-sessions-old", age=2 * 24 * 3600)))


class SweepTests(FolderTestCase):
    def test_only_the_folders_of_ended_processes_are_swept(self):
        me = psutil.Process()
        live = self.session("emapssn-test-sessions-live", owner={"pid": me.pid, "started": round(me.create_time(), 2)})
        dead = self.session("emapssn-test-sessions-dead", owner={"pid": ended_process_id(), "started": 1.0})
        fresh = self.session("emapssn-test-sessions-fresh", age=60)
        old = self.session("emapssn-test-sessions-old", age=3 * 24 * 3600)
        kept = self.session("emapssn-test-sessions-kept", owner={"pid": ended_process_id(), "started": 1.0})
        other = os.path.join(self.base, "emapssn-other")
        os.mkdir(other)
        note = os.path.join(self.base, "emapssn-test-sessions-note.txt")
        Path(note).write_text("", encoding="utf-8")

        removed = session_folder.sweep(self.base, keep=[kept])

        self.assertEqual(sorted(removed), ["emapssn-test-sessions-dead", "emapssn-test-sessions-old"])
        self.assertEqual([os.path.exists(path) for path in (live, dead, fresh, old, kept, other, note)],
                         [True, False, True, False, True, True, True])

    def test_a_base_folder_that_cannot_be_listed_sweeps_nothing(self):
        self.assertEqual(session_folder.sweep(os.path.join(self.base, "missing")), [])


class StartedByHandTests(FolderTestCase):
    """A process that imports the tests package with no runner behind it."""

    def child(self, code, **variables):
        environment = {name: value for name, value in os.environ.items()
                       if name not in ("EMAPSSN_TEST_ENVIRONMENT", session_folder.SESSION_ENV)}
        environment.update(session_folder.temp_environment(self.base, {"TMPDIR": ""}))
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        environment.update(variables)
        return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                              cwd=PROJECT_ROOT, env=environment, timeout=120, check=False)

    def sessions(self):
        return sorted(name for name in os.listdir(self.base) if name.startswith(session_folder.PREFIX))

    def test_it_works_in_a_session_folder_with_its_own_temp_folder_and_removes_it_at_exit(self):
        done = self.child("""
            import os, tempfile, tests
            print(os.environ["SSN_VIEWER_SESSION_DIR"])
            print(tempfile.gettempdir())
            print(os.path.exists(os.path.join(os.environ["SSN_VIEWER_SESSION_DIR"], "owner.json")))
            """)
        session, temp, owned = done.stdout.split("\n")[:3]
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(os.path.normcase(os.path.dirname(session)), os.path.normcase(os.path.normpath(self.base)))
        self.assertEqual(os.path.normcase(temp), os.path.normcase(os.path.join(session, "temp")))
        self.assertEqual(owned, "True")
        self.assertEqual(self.sessions(), [], "removed as the process ended")

    def test_what_the_tests_left_in_the_temp_folder_is_named_and_goes_with_the_session(self):
        done = self.child("""
            import os, tempfile, tests
            os.mkdir(os.path.join(tempfile.gettempdir(), "left-behind"))
            """)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Left in the test temp folder, now removed: left-behind", done.stderr)
        self.assertEqual(self.sessions(), [])

    def test_a_folder_the_runner_made_is_used_and_left_for_the_runner_to_remove(self):
        given = os.path.join(self.base, "emapssn-test-sessions-given")
        os.mkdir(given)
        done = self.child("""
            import os, tempfile, tests
            print(os.environ["SSN_VIEWER_SESSION_DIR"])
            print(tempfile.gettempdir())
            """, **{session_folder.SESSION_ENV: given})
        session, temp = done.stdout.split("\n")[:2]
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(os.path.normcase(session), os.path.normcase(given))
        self.assertEqual(os.path.normcase(temp), os.path.normcase(os.path.join(given, "temp")))
        self.assertTrue(os.path.isfile(os.path.join(given, session_folder.OWNER_FILE)))
        self.assertEqual(self.sessions(), ["emapssn-test-sessions-given"], "the runner removes it")

    def test_a_process_that_never_gets_to_exit_cleanly_leaves_a_folder_the_next_one_sweeps(self):
        killed = self.child("""
            import os, tests
            print(os.environ["SSN_VIEWER_SESSION_DIR"], flush=True)
            os._exit(0)  # no exit handlers, as when a process is killed
            """)
        first = killed.stdout.strip()
        self.assertTrue(os.path.isdir(first), "the killed process left its folder")
        later = self.child("""
            import os, tests
            print(os.environ["SSN_VIEWER_SESSION_DIR"])
            """)
        self.assertEqual(later.returncode, 0, later.stderr)
        self.assertNotEqual(later.stdout.strip(), first)
        self.assertFalse(os.path.exists(first), "the next process swept it")
        self.assertEqual(self.sessions(), [])


if __name__ == "__main__":
    unittest.main()
