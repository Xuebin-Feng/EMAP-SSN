# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The section runner (tests/run_sections.py): every test module sits in exactly one section of tests/sections.toml, changed files map to the sections whose tests use them, and each section runs as its own process whose log decides the verdict."""

from __future__ import annotations

import ast
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

import psutil

from tests import run_sections
from tests.run_sections import (
    UNASSIGNED,
    ReferenceIndex,
    Section,
    Selector,
    check_assignment,
    child_environment,
    load_config,
    parse_log,
    path_matches,
    runnable_sections,
    section_command,
    test_modules as list_test_modules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write(root, files):
    for relative, text in files.items():
        path = Path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")


def stop_processes_in(folder, grace=2.0):
    """Name and stop the processes still running in folder after grace seconds,
    so that a stop the runner missed fails an assertion instead of the folder's
    removal. (A venv's python.exe redirector reports another folder, but it ends
    with the interpreter it started, which reports this one.)"""
    found = []
    for process in psutil.process_iter():
        try:
            if os.path.samefile(process.cwd(), folder):
                found.append(process)
        except (psutil.Error, OSError):
            pass
    _, alive = psutil.wait_procs(found, timeout=grace)
    names = []
    for process in alive:
        try:
            names.append(f"{process.name()} {process.pid}")
            process.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=10)
    return names


class SectionsFileTests(unittest.TestCase):
    """The real tests/sections.toml."""

    def test_every_test_module_is_in_exactly_one_section(self):
        _, problems = check_assignment(load_config(), list_test_modules())
        self.assertEqual(problems, [], "fix tests/sections.toml")

    def test_every_section_runs_something(self):
        for section in load_config().sections.values():
            with self.subTest(section=section.name):
                self.assertTrue(section.discover or section.modules)

    def test_every_repository_check_names_a_test_that_exists(self):
        for test_id in load_config().repo_checks:
            with self.subTest(test=test_id):
                package, module, *names = test_id.split(".")
                self.assertEqual(package, "tests")
                tree = ast.parse((PROJECT_ROOT / "tests" / f"{module}.py").read_text(encoding="utf-8"))
                scope = tree.body
                for name in names:
                    found = [node for node in scope
                             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
                    self.assertTrue(found, f"{name} is not defined in tests/{module}.py")
                    scope = found[0].body

    def test_the_runner_and_its_file_are_not_collected_as_tests(self):
        modules = list_test_modules()
        self.assertNotIn("run_sections", modules)
        self.assertIn("test_run_sections", modules)


class FoldViewSectionsTests(unittest.TestCase):
    """The Fold View page's tests have sections of their own, and what changes them selects those sections.

    They open esmfold.html in Qt WebEngine and wait on Mol* and the page's saves, which made them most of the
    time of the whole suite when they shared a section with the Agent and metadata tests; sections run in
    parallel, so the page tests are spread over several.
    """

    PAGE_SECTIONS = {"fold_color", "fold_struct", "fold_canvas", "fold_session", "fold_layout", "fold_text", "fold_panel", "fold_webgl"}

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.selector = Selector(cls.config, root=PROJECT_ROOT)

    def test_every_test_module_that_opens_the_page_is_in_a_page_section(self):
        # A module opens the page when it uses the harness or the WebGL scenarios, or imports one that does.
        sources = {name: (PROJECT_ROOT / "tests" / f"{name}.py").read_text(encoding="utf-8")
                   for name in list_test_modules() if name != "test_run_sections"}
        opening = {name for name, text in sources.items() if "esmfold_page_harness" in text or "esmfold_gl_scenarios" in text}
        grew = True
        while grew:
            grew = False
            for name, text in sources.items():
                if name not in opening and any(f"tests.{other} import" in text or f"from tests import {other}" in text
                                               for other in opening):
                    opening.add(name)
                    grew = True
        self.assertIn("test_esmfold_shell", opening)
        owner, _ = check_assignment(self.config, list_test_modules())
        for name in sorted(opening):
            with self.subTest(module=name):
                self.assertIn(owner[name], self.PAGE_SECTIONS)

    def test_the_page_sections_hold_only_fold_view_modules_and_each_holds_one(self):
        for name in self.PAGE_SECTIONS:
            with self.subTest(section=name):
                self.assertIn(name, self.config.sections)
                modules = self.config.sections[name].modules
                self.assertTrue(modules)
                for module in modules:
                    self.assertTrue(module.startswith("test_esmfold_") or module == "test_molstar_api_surface", module)

    def test_the_page_its_scripts_and_the_helpers_that_open_it_select_every_page_section(self):
        for path in ("src/web_ui/esmfold.html", "src/web_ui/esmfold_shell.js", "src/web_ui/esmfold_residue.js",
                     "src/web_ui/page_text.js", "src/web_ui/residue_layers.py", "tests/esmfold_page_harness.py",
                     "tests/esmfold_layer_fixtures.py", "tests/esmfold_gl_scenarios.py", "tests/molstar_api_probe.py"):
            with self.subTest(path=path):
                self.assertLessEqual(self.PAGE_SECTIONS, self.selector.explain(path)[0])

    def test_the_vendored_mol_star_selects_them_too(self):
        for path in ("src/resources/esmfold/molstar.js", "src/resources/esmfold/molstar.css",
                     "src/resources/esmfold/NOTICES.molstar-bundle", "src/resources/esmfold/LICENSE.molstar"):
            with self.subTest(path=path):
                self.assertLessEqual(self.PAGE_SECTIONS | {"plugins", "viewer"}, self.selector.explain(path)[0])

    def test_the_command_the_worker_and_the_backend_select_plugins_and_not_every_page_section(self):
        for path in ("src/commands/esmfold.py", "src/resources/esmfold/esmfold_worker.py", "src/web_ui/esmfold_backend.py"):
            with self.subTest(path=path):
                chosen = self.selector.explain(path)[0]
                self.assertIn("plugins", chosen)
                self.assertNotEqual(self.PAGE_SECTIONS - chosen, set(), "a change here would run every page section")


class AssignmentTests(unittest.TestCase):
    def config(self, **sections):
        return run_sections.Config(
            sections={name: Section(name, modules=modules) for name, modules in sections.items()},
            rules=[], full_suite=[], repo_checks=[])

    def test_duplicates_missing_and_unlisted_modules_are_reported(self):
        config = self.config(a=["test_one", "test_two"], b=["test_two", "test_gone"])
        owner, problems = check_assignment(config, ["test_one", "test_two", "test_new"])
        self.assertEqual(owner, {"test_one": "a", "test_two": "a"})
        self.assertEqual(problems, [
            "test_two is listed in both a and b",
            "test_gone (section b) is not a test module in tests/",
            "test_new is in no section of tests/sections.toml",
        ])

    def test_a_full_run_still_runs_modules_the_file_forgot(self):
        config = self.config(a=["test_one", "test_gone"])
        sections = runnable_sections(config, ["test_one", "test_new"])
        self.assertEqual(sections["a"].modules, ["test_one"])
        self.assertEqual(sections[UNASSIGNED].modules, ["test_new"])


class GlobTests(unittest.TestCase):
    def test_a_single_star_stays_in_its_folder_and_a_double_star_does_not(self):
        self.assertTrue(path_matches("src/commands/hide.py", ["src/commands/*.py"]))
        self.assertFalse(path_matches("src/commands/sub/hide.py", ["src/commands/*.py"]))
        self.assertTrue(path_matches("opt_vr/src/a/b.py", ["opt_vr/**"]))
        self.assertTrue(path_matches("src/resources/languages/zh_CN/help/x.md",
                                     ["src/resources/languages/*/help/**"]))
        self.assertFalse(path_matches("opt_vr_old/x.py", ["opt_vr/**"]))
        self.assertTrue(path_matches("opt_vr", ["opt_vr"]))


class SelectionTests(unittest.TestCase):
    """A small project: which sections each kind of change selects."""

    FILES = {
        "tests/__init__.py": "",
        "tests/sections.toml": """\
            [sections.alpha]
            modules = ["test_alpha"]
            [sections.beta]
            modules = ["test_beta", "test_gamma"]
            [sections.opt_vr]
            cwd = "opt_vr"
            discover = true
            [full_suite]
            paths = ["tests/__init__.py"]
            [[rules]]
            paths = ["assets/**"]
            sections = ["beta"]
            [[rules]]
            paths = ["opt_vr", "opt_vr/**"]
            sections = ["opt_vr"]
            [repo_checks]
            tests = ["tests.test_alpha.AlphaTests", "tests.test_beta.BetaTests.test_scan"]
            """,
        "tests/test_alpha.py": "import core\n",
        "tests/test_beta.py": "from tests.helper import thing\n",
        "tests/helper.py": "from widgets import widget\nthing = 1\n",
        "tests/test_gamma.py": "PAGE = 'page.html'\nTARGET = 'Named_Module.attribute'\n",
        "tests/test_new.py": "",
        "src/core.py": "import deep\n",
        "src/deep.py": "",
        "src/widgets/__init__.py": "",
        "src/widgets/widget.py": "",
        "src/page.html": "",
        "src/Named_Module.py": "",
        "src/lonely.py": "",
        "src/shared.py": "",
        "assets/logo.png": "",
        "docs/guide.md": "",
        "opt_vr/tests/test_vr.py": "import shared\n",
        "opt_vr/src/client.py": "",
    }

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        write(self.root, self.FILES)
        files = sorted(self.FILES)
        self.selector = Selector(load_config(self.root), root=self.root,
                                 index=ReferenceIndex(self.root, files=files))

    def sections(self, path):
        return self.selector.explain(path)[0]

    def test_a_test_module_selects_its_own_section(self):
        self.assertEqual(self.sections("tests/test_gamma.py"), {"beta"})
        self.assertEqual(self.sections("tests/test_new.py"), {UNASSIGNED})

    def test_a_file_a_test_imports_selects_that_tests_section(self):
        self.assertEqual(self.sections("src/core.py"), {"alpha"})

    def test_a_file_a_helper_module_imports_selects_the_helpers_users(self):
        self.assertEqual(self.sections("src/widgets/widget.py"), {"beta"})
        self.assertEqual(self.sections("tests/helper.py"), {"beta"})

    def test_a_file_only_source_uses_selects_its_nearest_users(self):
        sections, reasons = self.selector.explain("src/deep.py")
        self.assertEqual(sections, {"alpha"})
        self.assertIn("through src/core.py", reasons[0])

    def test_file_names_and_module_names_in_strings_count(self):
        self.assertEqual(self.sections("src/page.html"), {"beta"})
        self.assertEqual(self.sections("src/Named_Module.py"), {"beta"})

    def test_path_rules_add_sections(self):
        self.assertEqual(self.sections("assets/logo.png"), {"beta"})
        self.assertEqual(self.sections("opt_vr/src/client.py"), {"opt_vr"})

    def test_main_code_the_vr_suite_uses_selects_it(self):
        self.assertEqual(self.sections("src/shared.py"), {"opt_vr"})

    def test_a_full_suite_file_selects_everything(self):
        self.assertIsNone(self.sections("tests/__init__.py"))
        self.assertTrue(self.selector.select(["src/core.py", "tests/__init__.py"]).everything)

    def test_a_file_no_test_uses_selects_nothing_and_is_named(self):
        selection = self.selector.select(["docs/guide.md", "src/lonely.py"])
        self.assertEqual(selection.sections, set())
        self.assertEqual(selection.unreferenced, ["docs/guide.md", "src/lonely.py"])

    def test_python_source_changes_add_the_repository_checks_not_already_running(self):
        self.assertFalse(self.selector.select(["docs/guide.md", "assets/logo.png"]).repo_checks)
        selection = self.selector.select([".\\src\\lonely.py"])
        self.assertTrue(selection.repo_checks)
        self.assertEqual(self.selector.pending_repo_checks({"beta"}), ["tests.test_alpha.AlphaTests"])
        self.assertEqual(self.selector.pending_repo_checks({"alpha", "beta"}), [])


class LogTests(unittest.TestCase):
    def test_a_passing_log(self):
        log = b"....s\n------\nRan 5 tests in 1.250s\n\nOK (skipped=1)\n"
        self.assertEqual(parse_log(log),
                         ("ok", {"tests": 5, "failures": 0, "errors": 0, "skipped": 1}, []))

    def test_a_failing_log_names_its_failures(self):
        log = (b"FE.\n======\nFAIL: test_a (tests.test_x.T.test_a)\n------\nTraceback\n"
               b"======\nERROR: test_b (tests.test_x.T.test_b)\n------\n"
               b"Ran 3 tests in 0.1s\n\nFAILED (failures=1, errors=1, unexpected successes=1)\n")
        status, counts, problems = parse_log(log)
        self.assertEqual(status, "failed")
        self.assertEqual(counts, {"tests": 3, "failures": 2, "errors": 1, "skipped": 0})
        self.assertEqual(problems, ["FAIL: test_a (tests.test_x.T.test_a)",
                                    "ERROR: test_b (tests.test_x.T.test_b)"])

    def test_no_tests_and_a_crash(self):
        self.assertEqual(parse_log(b"\nRan 0 tests in 0.000s\n\nNO TESTS RAN\n")[0], "empty")
        self.assertEqual(parse_log(b"Fatal Python error: Segmentation fault\n")[0], "crashed")


class CommandTests(unittest.TestCase):
    def test_module_sections_name_their_modules_and_pass_filters_first(self):
        section = Section("a", modules=["test_one", "test_two"])
        self.assertEqual(section_command(section, "py", verbose=True, patterns=["fast"]),
                         ["py", "-m", "unittest", "-v", "-k", "fast", "tests.test_one", "tests.test_two"])

    def test_the_vr_suite_runs_by_discovery_and_checks_by_test_id(self):
        self.assertEqual(section_command(Section("opt_vr", discover=True), "py"),
                         ["py", "-m", "unittest", "discover", "-s", "tests", "-t", "."])
        self.assertEqual(section_command(Section("r", tests=["tests.test_x.T"]), "py"),
                         ["py", "-m", "unittest", "tests.test_x.T"])

    def test_each_section_sets_up_its_own_private_test_folders(self):
        # The marker tells tests/__init__.py that a parent already chose the folders;
        # parallel sections must not share them.
        environment = child_environment({"EMAPSSN_TEST_ENVIRONMENT": "123", "PATH": "x"})
        self.assertEqual(environment, {"PATH": "x"})


class RunTests(unittest.TestCase):
    """Real child processes on a small project."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        write(self.root, {
            "tests/__init__.py": "",
            "tests/sections.toml": """\
                [sections.good]
                modules = ["test_pass"]
                [sections.bad]
                modules = ["test_fail"]
                [sections.slow]
                modules = ["test_sleep"]
                """,
            "tests/test_pass.py": """\
                import os, unittest
                class PassTests(unittest.TestCase):
                    def test_private_folders(self):
                        self.assertNotIn("EMAPSSN_TEST_ENVIRONMENT", os.environ)
                """,
            "tests/test_fail.py": """\
                import unittest
                class FailTests(unittest.TestCase):
                    def test_fails(self):
                        self.fail("on purpose")
                """,
            "tests/test_sleep.py": """\
                import subprocess, sys, time, unittest
                class SleepTests(unittest.TestCase):
                    def test_sleeps(self):
                        # A process of its own, so that stopping the section means
                        # stopping a tree.
                        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                        time.sleep(60)
                """,
            "tests/test_forgotten.py": """\
                import unittest
                class ForgottenTests(unittest.TestCase):
                    def test_runs(self):
                        pass
                """,
        })
        patcher = mock.patch.object(run_sections, "LOG_BASE", self.root / "logs")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.logs = self.root / "run"
        # As in a real test process, whose tests/__init__.py set the marker.
        marker = mock.patch.dict(os.environ, {"EMAPSSN_TEST_ENVIRONMENT": "1"})
        marker.start()
        self.addCleanup(marker.stop)

    def run_main(self, *arguments):
        out = io.StringIO()
        code = run_sections.main([*arguments, "--log-dir", str(self.logs)], root=self.root, out=out)
        return code, out.getvalue()

    def test_a_passing_section_exits_zero(self):
        code, out = self.run_main("good")
        self.assertEqual(code, 0, out)
        self.assertIn("1 tests in", out)
        self.assertTrue((self.logs / "good.log").is_file())

    def test_a_failing_section_exits_one_and_names_the_test(self):
        code, out = self.run_main("good", "bad", "-j", "2")
        self.assertEqual(code, 1, out)
        self.assertIn("bad: failed", out)
        self.assertIn("FAIL: test_fails (tests.test_fail.FailTests.test_fails)", out)

    def test_a_full_run_includes_modules_missing_from_the_file(self):
        code, out = self.run_main("--all", "--skip", "bad", "slow")
        self.assertEqual(code, 0, out)
        self.assertIn("warning: test_forgotten is in no section", out)
        self.assertIn(UNASSIGNED, out)

    def test_a_section_past_its_timeout_is_stopped(self):
        self.addCleanup(stop_processes_in, self.root, grace=0)
        code, out = self.run_main("slow", "--timeout", "3")
        self.assertEqual(code, 1, out)
        self.assertIn("slow: timeout", out)
        self.assertNotIn("not stopped", out)
        self.assertEqual(stop_processes_in(self.root), [], out)

    def test_a_stop_that_failed_is_reported(self):
        self.addCleanup(stop_processes_in, self.root, grace=0)
        kill_tree = run_sections._kill_tree
        with mock.patch.object(run_sections, "_kill_tree",
                               side_effect=lambda process: kill_tree(process) or "process 1 is still running"):
            code, out = self.run_main("slow", "--timeout", "1")
        self.assertEqual(code, 1, out)
        self.assertIn("slow: timeout", out)
        self.assertIn("not stopped: process 1 is still running", out)

    def test_an_unknown_section_is_a_usage_error(self):
        code, out = self.run_main("nonexistent")
        self.assertEqual(code, 2)
        self.assertIn("unknown section(s): nonexistent", out)


class SessionFolderTests(unittest.TestCase):
    """Each section works in a session folder of its own, which the runner removes whatever became of the process."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        write(self.root, {
            "tests/__init__.py": "",
            "tests/sections.toml": """\
                [sections.temp]
                modules = ["test_temp"]
                [sections.leaky]
                modules = ["test_leaky"]
                [sections.crash]
                modules = ["test_crash"]
                [sections.slow]
                modules = ["test_sleep"]
                """,
            "tests/test_temp.py": """\
                import os, tempfile, unittest
                class TempTests(unittest.TestCase):
                    def test_the_temp_folder_is_the_sessions(self):
                        session = os.environ["EMAPSSN_TEST_SESSION_DIR"]
                        self.assertTrue(os.path.isdir(session))
                        temp = os.path.join(session, "temp")
                        self.assertEqual(os.path.normcase(tempfile.gettempdir()), os.path.normcase(temp))
                        self.assertEqual(os.path.normcase(os.environ["TMP"]), os.path.normcase(temp))
                        self.assertEqual(os.path.normcase(os.environ["TEMP"]), os.path.normcase(temp))
                        self.assertNotIn("EMAPSSN_TEST_ENVIRONMENT", os.environ)
                """,
            "tests/test_leaky.py": """\
                import os, tempfile, unittest
                class LeakyTests(unittest.TestCase):
                    def test_forgets_a_folder(self):
                        folder = tempfile.mkdtemp(prefix="left-behind-")
                        open(os.path.join(folder, "note.txt"), "w").close()
                        open("leaked-session.txt", "w").write(os.environ["EMAPSSN_TEST_SESSION_DIR"])
                """,
            "tests/test_crash.py": """\
                import os, unittest
                class CrashTests(unittest.TestCase):
                    def test_ends_the_process_without_exit_handlers(self):
                        open("crashed-session.txt", "w").write(os.environ["EMAPSSN_TEST_SESSION_DIR"])
                        os._exit(3)
                """,
            "tests/test_sleep.py": """\
                import os, subprocess, sys, time, unittest
                class SleepTests(unittest.TestCase):
                    def test_sleeps(self):
                        open("slept-session.txt", "w").write(os.environ["EMAPSSN_TEST_SESSION_DIR"])
                        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                        time.sleep(60)
                """,
        })
        patcher = mock.patch.object(run_sections, "LOG_BASE", self.root / "logs")
        patcher.start()
        self.addCleanup(patcher.stop)
        marker = mock.patch.dict(os.environ, {"EMAPSSN_TEST_ENVIRONMENT": "1"})
        marker.start()
        self.addCleanup(marker.stop)
        self.logs = self.root / "run"
        self.before = self.sessions()

    def sessions(self):
        """The session folders in this process's temp folder (private to the run, so these are the runner's)."""
        return sorted(name for name in os.listdir(tempfile.gettempdir()) if name.startswith("emapssn-test-sessions-"))

    def run_main(self, *arguments):
        out = io.StringIO()
        code = run_sections.main([*arguments, "--log-dir", str(self.logs)], root=self.root, out=out)
        return code, out.getvalue()

    def recorded(self, name):
        return (self.root / name).read_text(encoding="utf-8")

    def test_a_section_runs_in_a_session_folder_whose_temp_folder_is_its_temp_folder(self):
        code, out = self.run_main("temp")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.sessions(), self.before, "removed when the section ended")
        self.assertNotIn("temp folder", out)

    def test_the_session_folders_of_sections_that_ran_together_are_different_ones(self):
        code, out = self.run_main("temp", "leaky", "-j", "2")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.sessions(), self.before)

    def test_what_a_sections_tests_left_in_their_temp_folder_is_named_and_removed(self):
        code, out = self.run_main("leaky")
        self.assertEqual(code, 0, "a leak is reported, not a failure: " + out)
        self.assertRegex(out, r"leaky: its tests left 1 item\(s\) in their temp folder.*left-behind-")
        self.assertFalse(os.path.exists(self.recorded("leaked-session.txt")))
        self.assertEqual(self.sessions(), self.before)

    def test_a_section_that_ends_without_exit_handlers_leaves_no_session_folder(self):
        code, out = self.run_main("crash")
        self.assertEqual(code, 1, out)
        self.assertIn("crash: crashed", out)
        session = self.recorded("crashed-session.txt")
        self.assertEqual(os.path.normcase(os.path.dirname(session)), os.path.normcase(tempfile.gettempdir()))
        self.assertFalse(os.path.exists(session), "the process never ran its exit handlers")
        self.assertEqual(self.sessions(), self.before)

    def test_a_section_stopped_at_its_timeout_leaves_no_session_folder(self):
        self.addCleanup(stop_processes_in, self.root, grace=0)
        code, out = self.run_main("slow", "--timeout", "3")
        self.assertEqual(code, 1, out)
        self.assertIn("slow: timeout", out)
        self.assertFalse(os.path.exists(self.recorded("slept-session.txt")))
        self.assertEqual(self.sessions(), self.before)

    def test_a_folder_that_cannot_be_removed_is_reported(self):
        # The folder the stand-in leaves is this test's to remove, not the outer run's to report.
        self.addCleanup(lambda: [shutil.rmtree(os.path.join(tempfile.gettempdir(), name), ignore_errors=True)
                                 for name in set(self.sessions()) - set(self.before)])
        with mock.patch.object(run_sections.session_folder, "remove_tree",
                               return_value=PermissionError(13, "in use")):
            code, out = self.run_main("temp")
        self.assertEqual(code, 0, out)
        self.assertIn("temp: its session folder could not be removed", out)
        self.assertIn("in use", out)


class RealTreeTestCase(unittest.TestCase):
    """Stopping a real tree: a process whose child started a process of its own."""

    def start_tree(self, *child_options):
        """Start the tree as the runner starts a section, and wait for the child.
        "ignore-sigterm" makes the child ignore SIGTERM; "slow-cleanup" makes it
        take half a second to leave, and write "cleaned-up" as it goes."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        write(self.root, {"sleeper.py": """\
            import signal, subprocess, sys, time
            if sys.argv[1:2] == ["child"]:
                if "ignore-sigterm" in sys.argv:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                if "slow-cleanup" in sys.argv:
                    def clean_up(*_):
                        time.sleep(0.5)
                        open("cleaned-up", "w").close()
                        sys.exit()
                    signal.signal(signal.SIGTERM, clean_up)
                open("child-started", "w").close()
            else:
                subprocess.Popen([sys.executable, __file__, "child", *sys.argv[1:]])
            time.sleep(60)
            """})
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                   else {"start_new_session": True})
        self.process = subprocess.Popen([sys.executable, "sleeper.py", *child_options], cwd=self.root,
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, **options)
        self.addCleanup(self.process.wait, 10)
        self.addCleanup(stop_processes_in, self.root, grace=0)
        deadline = time.monotonic() + 30
        while not (self.root / "child-started").exists():
            self.assertLess(time.monotonic(), deadline, "the sleeper's child did not start")
            time.sleep(0.05)


@unittest.skipUnless(os.name == "posix", "process groups are POSIX")
class PosixStopTests(RealTreeTestCase):
    def test_a_member_that_ignores_sigterm_is_killed(self):
        self.start_tree("ignore-sigterm")
        with mock.patch.object(run_sections, "STOP_WAIT", 0.5):
            self.assertEqual(run_sections._kill_tree(self.process), "")
        self.assertEqual(stop_processes_in(self.root), [])

    def test_members_get_the_grace_period_to_clean_up(self):
        # As multiprocessing's resource tracker does after the leader is gone.
        self.start_tree("slow-cleanup")
        self.assertEqual(run_sections._kill_tree(self.process), "")
        self.assertTrue((self.root / "cleaned-up").exists())

    def test_a_group_that_outlives_sigkill_is_reported(self):
        self.start_tree("ignore-sigterm")  # so that SIGKILL still finds the child
        killpg = os.killpg

        def members_never_leave(group, signal_number):
            if signal_number:  # signal 0 only asks whether the group has members
                killpg(group, signal_number)

        with mock.patch.object(run_sections.os, "killpg", side_effect=members_never_leave), \
                mock.patch.object(run_sections, "STOP_WAIT", 0.2):
            problem = run_sections._kill_tree(self.process)
        self.assertEqual(problem, f"process group {self.process.pid} still has members after SIGKILL")


@unittest.skipUnless(os.name == "nt", "Windows stops trees without process groups")
class WindowsStopTests(RealTreeTestCase):
    FAILED_TASKKILL = subprocess.CompletedProcess([], 1, b"ERROR: cancelled\r\n", b"")

    def setUp(self):
        self.start_tree()

    def test_the_stop_does_not_need_taskkill(self):
        # taskkill lists the tree through WMI, which cancels queries under load.
        with mock.patch.object(run_sections.subprocess, "run", return_value=self.FAILED_TASKKILL):
            self.assertEqual(run_sections._kill_tree(self.process), "")
        self.assertEqual(stop_processes_in(self.root), [])

    def test_processes_that_outlive_the_stop_are_named(self):
        with mock.patch.object(psutil, "wait_procs", side_effect=lambda processes, timeout: ([], processes)):
            problem = run_sections._kill_tree(self.process)
        self.assertRegex(problem, rf"^still running after a forced stop: .*\b{self.process.pid}\b")

    def test_without_psutil_taskkill_stops_the_tree(self):
        with mock.patch.object(run_sections, "psutil", None):
            self.assertEqual(run_sections._kill_tree(self.process), "")
        self.assertEqual(stop_processes_in(self.root), [])

    def test_without_psutil_a_failing_taskkill_is_tried_again_and_reported(self):
        with mock.patch.object(run_sections, "psutil", None), \
                mock.patch.multiple(run_sections, STOP_WAIT=0.1, TASKKILL_PAUSE=0), \
                mock.patch.object(run_sections.subprocess, "run", return_value=self.FAILED_TASKKILL) as run:
            problem = run_sections._kill_tree(self.process)
        self.assertEqual(run.call_count, run_sections.TASKKILL_TRIES)
        self.assertEqual(problem, "taskkill exited with 1 on all 3 tries: ERROR: cancelled")


if __name__ == "__main__":
    unittest.main()
