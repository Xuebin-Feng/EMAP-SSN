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
from tests.git_availability import git_executable_for, git_problem
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

    def test_full_suite_files_cover_test_startup_helpers_and_explicit_rules_hold(self):
        selector = Selector(load_config(), root=PROJECT_ROOT)
        for path in ("tests/__init__.py", "tests/session_folder.py", "tests/desktop_openers.py"):
            with self.subTest(full_suite=path):
                self.assertIsNone(selector.explain(path)[0])
        contracts = selector.explain("src/mcp_server/pipeline/Pipeline_Settings_Contracts.json")[0]
        self.assertLessEqual({"mcp", "tools"}, contracts)
        for path in ("tests/sections.toml", "tests/run_sections.py"):
            with self.subTest(runner_file=path):
                self.assertIn("platform", selector.explain(path)[0])
        # git_availability.py names Run_Benchmark.py only in a docstring, so it must not pull
        # unrelated sections that import git_availability into benchmark edits.
        self.assertEqual(selector.explain("src/resources/benchmark/Run_Benchmark.py")[0], {"tools"})


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

    def test_the_command_and_the_worker_select_plugins_and_not_every_page_section(self):
        for path in ("src/commands/esmfold.py", "src/resources/esmfold/esmfold_worker.py"):
            with self.subTest(path=path):
                chosen = self.selector.explain(path)[0]
                self.assertIn("plugins", chosen)
                self.assertNotIn("platform", chosen)
                self.assertNotEqual(self.PAGE_SECTIONS - chosen, set(), "a change here would run every page section")

    def test_the_backend_selects_plugins_and_every_page_section_and_not_platform(self):
        chosen = self.selector.explain("src/web_ui/esmfold_backend.py")[0]
        self.assertLessEqual(self.PAGE_SECTIONS | {"plugins"}, chosen)
        self.assertNotIn("platform", chosen)

    def test_residue_layers_selects_plugins_tools_and_every_page_section_and_not_platform(self):
        chosen = self.selector.explain("src/web_ui/residue_layers.py")[0]
        self.assertLessEqual(self.PAGE_SECTIONS | {"plugins", "tools"}, chosen)
        self.assertNotIn("platform", chosen)


class TreeViewSectionsTests(unittest.TestCase):
    """Tree View's page tests have sections of their own, and what changes them selects those sections.

    They open tree.html in Qt WebEngine and load it afresh for every test, which made them one section of 470 tests
    and the time of the whole suite; sections run in parallel, so the page tests are spread over several.
    """

    PAGE_SECTIONS = {"tree_core", "tree_detail", "tree_color", "tree_select", "tree_look", "tree_open"}

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.selector = Selector(cls.config, root=PROJECT_ROOT)

    def test_every_test_module_that_uses_the_harness_is_in_a_page_section(self):
        owner, _ = check_assignment(self.config, list_test_modules())
        using = [name for name in list_test_modules()
                 if name != "test_run_sections"
                 and "tree_page_harness" in (PROJECT_ROOT / "tests" / f"{name}.py").read_text(encoding="utf-8")]
        self.assertIn("test_tree_page", using)
        self.assertIn("test_tree_open_page", using)
        for name in using:
            with self.subTest(module=name):
                self.assertIn(owner[name], self.PAGE_SECTIONS)

    def test_the_page_sections_hold_only_tree_view_page_modules(self):
        for name in self.PAGE_SECTIONS:
            with self.subTest(section=name):
                self.assertIn(name, self.config.sections)
                self.assertLessEqual(len(name), 12, "the runner's tables fit names of 12 characters")
                modules = self.config.sections[name].modules
                self.assertTrue(modules)
                for module in modules:
                    self.assertTrue(module.startswith("test_tree_") and module.endswith("_page"), module)

    def test_the_page_its_scripts_and_the_helpers_that_open_it_select_every_page_section(self):
        for path in ("src/web_ui/tree.html", "src/web_ui/tree_layout.js", "src/web_ui/tree_select.js", "src/web_ui/page_text.js",
                     "tests/tree_page_harness.py", "tests/tree_page_helpers.py", "tests/tree_fixtures.py"):
            with self.subTest(path=path):
                self.assertLessEqual(self.PAGE_SECTIONS, self.selector.explain(path)[0])

    def test_the_backend_selects_plugins_and_every_page_section(self):
        for path in ("src/web_ui/tree_backend.py", "src/commands/tree.py"):
            with self.subTest(path=path):
                self.assertLessEqual(self.PAGE_SECTIONS | {"plugins"}, self.selector.explain(path)[0])

    def test_the_snapshot_selects_commands_plugins_and_every_page_section(self):
        chosen = self.selector.explain("src/web_ui/tree_snapshot.py")[0]
        self.assertLessEqual(self.PAGE_SECTIONS | {"commands", "plugins"}, chosen)
        self.assertNotIn("platform", chosen)


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
        self.assertFalse(self.selector.select(["src/page.html", "src/other/script.js"]).repo_checks)
        for path in (".\\src\\lonely.py", "src/web_ui/tree.js", "src/web_ui/tree.html",
                     "tests/test_alpha.py", "tests/helper.py", "tests/__init__.py"):
            with self.subTest(path=path):
                self.assertTrue(self.selector.select([path]).repo_checks)
        self.assertEqual(self.selector.pending_repo_checks({"beta"}), ["tests.test_alpha.AlphaTests"])
        self.assertEqual(self.selector.pending_repo_checks({"alpha", "beta"}), [])

    def test_docstrings_do_not_create_false_references_while_code_strings_still_do(self):
        files = dict(self.FILES)
        files["tests/test_alpha.py"] = '''\
            """Module docstring naming src/lonely.py and page.html and Named_Module."""
            import core

            class AlphaCase:
                """Class docstring naming src/lonely.py and page.html."""

                def method(self):
                    """Function docstring naming src/lonely.py and Named_Module."""
                    return "src/shared.py"

                async def async_method(self):
                    """Async docstring naming src/lonely.py."""
                    return None
            '''
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root, files)
            index = ReferenceIndex(root, files=sorted(files))
            refs = index.refs("tests/test_alpha.py")
            self.assertEqual(refs, {"src/core.py", "src/shared.py"})
            selector = Selector(load_config(root), root=root, index=index)
            self.assertEqual(selector.explain("src/lonely.py")[0], set())

    def test_test_run_sections_is_excluded_from_reverse_indexing(self):
        files = dict(self.FILES)
        files["tests/sections.toml"] += "\n[sections.platform]\nmodules = [\"test_run_sections\"]\n"
        files["tests/test_run_sections.py"] = 'PATHS = ("src/core.py", "src/lonely.py", "page.html")\n'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root, files)
            selector = Selector(load_config(root), root=root,
                                index=ReferenceIndex(root, files=sorted(files)))
            self.assertEqual(selector.explain("tests/test_run_sections.py")[0], {"platform"})
            self.assertEqual(selector.explain("src/core.py")[0], {"alpha"})
            self.assertEqual(selector.explain("src/lonely.py")[0], set())
            self.assertEqual(selector.explain("src/page.html")[0], {"beta"})


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

    def test_skipping_a_section_keeps_its_repository_checks(self):
        write(self.root, {
            "tests/sections.toml": """\
                [sections.good]
                modules = ["test_pass"]
                [sections.guard]
                modules = ["test_guard"]
                [repo_checks]
                tests = ["tests.test_guard.GuardTests.test_scan"]
                """,
            "src/core.py": "VALUE = 1\n",
            "tests/test_pass.py": """\
                import unittest
                TARGET = "src/core.py"
                class PassTests(unittest.TestCase):
                    def test_ok(self):
                        pass
                """,
            "tests/test_guard.py": """\
                import unittest
                TARGET = "src/core.py"
                class GuardTests(unittest.TestCase):
                    def test_scan(self):
                        pass
                    def test_other(self):
                        self.fail("only test_scan should run when guard is skipped")
                """,
        })
        code, out = self.run_main("--files", "src/core.py", "--skip", "guard")
        self.assertEqual(code, 0, out)
        self.assertIn("repo-checks", out)
        self.assertIn("2 tests in", out)
        self.assertTrue((self.logs / "repo-checks.log").is_file())
        self.assertFalse((self.logs / "guard.log").exists())

        with mock.patch.object(run_sections, "changed_files", return_value=["src/core.py"]):
            code, out = self.run_main("--changed", "--skip", "guard", "--dry-run")
        self.assertEqual(code, 0, out)
        self.assertIn("tests.test_guard.GuardTests.test_scan", out)
        self.assertNotIn("tests.test_guard\n", out)


class PlatformTests(unittest.TestCase):
    """A section can declare the systems it runs on; elsewhere the runner skips it and says why."""

    SKIP_LINE = "skip  vr_only"

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        passing = """\
            import unittest
            class PassTests(unittest.TestCase):
                def test_passes(self):
                    pass
            """
        write(self.root, {
            "tests/__init__.py": "",
            "tests/sections.toml": """\
                [sections.vr_only]
                modules = ["test_vr"]
                platforms = ["win32"]
                [sections.everywhere]
                modules = ["test_plain"]
                [sections.gone]
                cwd = "nowhere"
                discover = true
                """,
            "tests/test_vr.py": passing,
            "tests/test_plain.py": passing,
        })
        patcher = mock.patch.object(run_sections, "LOG_BASE", self.root / "logs")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.logs = self.root / "run"
        marker = mock.patch.dict(os.environ, {"EMAPSSN_TEST_ENVIRONMENT": "1"})
        marker.start()
        self.addCleanup(marker.stop)

    def run_main(self, platform, *arguments):
        out = io.StringIO()
        code = run_sections.main([*arguments, "--log-dir", str(self.logs)], root=self.root,
                                 out=out, platform=platform)
        return code, out.getvalue()

    def test_a_windows_section_is_skipped_on_linux_and_macos_and_never_started(self):
        for platform, name in (("linux", "Linux"), ("darwin", "macOS")):
            with self.subTest(platform=platform), \
                    mock.patch.object(run_sections.subprocess, "Popen") as popen:
                code, out = self.run_main(platform, "vr_only")
                self.assertEqual(code, 0, out)
                self.assertIn(f"skip  vr_only      runs on Windows only (this is {name})", out)
                self.assertRegex(out, r"vr_only\s+0\s+0\s+0\s+0\s+0\.0\s+platform")
                self.assertIn("0 tests in", out)
                popen.assert_not_called()
                self.assertFalse((self.logs / "vr_only.log").exists())

    def test_a_windows_section_runs_on_windows(self):
        code, out = self.run_main("win32", "vr_only")
        self.assertEqual(code, 0, out)
        self.assertNotIn(self.SKIP_LINE, out)
        self.assertIn("1 tests in", out)
        self.assertTrue((self.logs / "vr_only.log").is_file())

    def test_a_section_without_the_key_runs_on_every_system(self):
        for platform in ("win32", "linux", "darwin"):
            with self.subTest(platform=platform):
                code, out = self.run_main(platform, "everywhere")
                self.assertEqual(code, 0, out)
                self.assertNotIn("skip  everywhere", out)
                self.assertIn("1 tests in", out)

    def test_the_skip_holds_however_the_section_was_chosen(self):
        for arguments in (("vr_only",), ("--all", "--skip", "gone", "everywhere"),
                          ("--files", "tests/test_vr.py")):
            with self.subTest(arguments=arguments), \
                    mock.patch.object(run_sections.Runner, "run_one") as run_one:
                code, out = self.run_main("linux", *arguments)
                self.assertEqual(code, 0, out)
                self.assertIn("skip  vr_only      runs on Windows only (this is Linux)", out)
                run_one.assert_not_called()

    def test_a_skipped_section_takes_no_lane_and_the_others_still_run(self):
        code, out = self.run_main("linux", "vr_only", "everywhere")
        self.assertEqual(code, 0, out)
        self.assertIn("Running 1 section(s), 1 at a time: everywhere", out)
        self.assertIn("1 tests in", out)

    def test_a_run_of_only_skipped_sections_exits_as_one_of_only_missing_ones_does(self):
        # The rule: neither a "missing" nor a "platform" section fails; with nothing else
        # chosen the run prints "No tests ran." and exits 0.
        skipped_code, skipped = self.run_main("linux", "vr_only")
        missing_code, missing = self.run_main("linux", "gone")
        self.assertIn("is not here", missing)
        self.assertEqual((skipped_code, missing_code), (0, 0), skipped + missing)
        self.assertIn("No tests ran.", skipped)
        self.assertIn("No tests ran.", missing)

    def test_the_listing_marks_the_section_and_the_dry_run_starts_nothing(self):
        code, out = self.run_main("linux", "--list")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"vr_only .*\(Windows only\)")
        code, out = self.run_main("linux", "vr_only", "--dry-run")
        self.assertIn("skip  vr_only      runs on Windows only (this is Linux)", out)
        self.assertNotIn("unittest", out)

    def test_an_unlisted_system_is_shown_as_it_is(self):
        reason = Section("a", platforms=["win32", "darwin"]).platform_skip("freebsd14")
        self.assertEqual(reason, "runs on Windows and macOS only (this is freebsd14)")

    def test_a_bad_platforms_value_is_refused_at_load(self):
        bad = ('"win32"', "[]", "[1]", '["Windows"]', '["win32", "linux2"]', "true")
        for value in bad:
            with self.subTest(value=value):
                write(self.root, {"tests/sections.toml": f"""\
                    [sections.vr_only]
                    modules = ["test_vr"]
                    platforms = {value}
                    """})
                with self.assertRaisesRegex(run_sections.UsageError, "section 'vr_only': platforms"):
                    load_config(self.root)
                code, out = self.run_main("linux", "--list")
                self.assertEqual(code, 2, out)
                self.assertIn("error: tests/sections.toml: section 'vr_only': platforms", out)

    def test_only_the_vr_section_of_the_real_file_declares_platforms(self):
        declared = {name: section.platforms for name, section in load_config().sections.items()
                    if section.platforms}
        self.assertEqual(declared, {"opt_vr": ["win32"]})


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
                [sections.holder]
                modules = ["test_holder"]
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
            "tests/test_holder.py": """\
                import os, subprocess, sys, tempfile, unittest
                class HolderTests(unittest.TestCase):
                    def test_starts_a_process_that_outlives_the_section_and_holds_a_file(self):
                        # Not in the section's process tree to stop, as a Viewer launched by a test is not.
                        code = ("import os, sys, time; handle = open(os.path.join(sys.argv[1], 'held.txt'), 'w'); "
                                "print('ready', flush=True); time.sleep(60)")
                        child = subprocess.Popen([sys.executable, "-c", code, tempfile.gettempdir()],
                                                 stdout=subprocess.PIPE, text=True)
                        self.assertEqual(child.stdout.readline().strip(), "ready")
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

    @unittest.skipUnless(os.name == "nt", "an open file blocks removing its folder on Windows only")
    def test_a_process_that_outlived_the_section_and_holds_its_folder_is_stopped(self):
        self.addCleanup(stop_processes_in, self.root, grace=0)
        code, out = self.run_main("holder")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"holder: its tests left \d+ process\(es\) running, which held its session folder, "
                              r"and the runner stopped them: python")
        self.assertNotIn("could not be removed", out)
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

    def test_runner_startup_sweeps_stale_session_folders_and_ignores_sweep_errors(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait(timeout=10)
        stale = Path(run_sections.session_folder.create("dead"))
        self.addCleanup(shutil.rmtree, stale, True)
        (stale / run_sections.session_folder.OWNER_FILE).write_text(
            f'{{"pid": {dead.pid}, "started": 1.0}}', encoding="utf-8")
        code, out = self.run_main("temp")
        self.assertEqual(code, 0, out)
        self.assertFalse(stale.exists(), "startup sweep should remove dead session folders")

        with mock.patch.object(run_sections.session_folder, "sweep", side_effect=OSError("cannot sweep")) as sweep:
            code, out = self.run_main("temp")
        self.assertEqual(code, 0, out)
        sweep.assert_called_once()


class LogFolderTests(unittest.TestCase):
    """The runner's own logs are kept for the newest ten runs."""

    def test_a_new_run_keeps_the_newest_ten_run_folders_and_touches_nothing_else(self):
        with tempfile.TemporaryDirectory() as base, mock.patch.object(run_sections, "LOG_BASE", Path(base)):
            names = [f"20260101-0000{number:02d}-{number}" for number in range(12)]
            for number, name in enumerate(names):
                folder = Path(base, name)
                folder.mkdir()
                (folder / "commands.log").write_text("log", encoding="utf-8")
                os.utime(folder, (1_000_000 + number, 1_000_000 + number))  # in order of age
            Path(base, "durations.json").write_text("{}", encoding="utf-8")
            Path(base, "notes").mkdir()
            Path(base, "20260101-000000").mkdir()  # not a run folder: no process ID
            new = run_sections.new_log_dir()
            self.assertTrue(new.is_dir())
            self.assertRegex(new.name, r"\d{8}-\d{6}-\d+")
            runs = sorted(entry.name for entry in Path(base).iterdir() if run_sections.RUN_FOLDER.fullmatch(entry.name))
            self.assertEqual(runs, sorted([*names[3:], new.name]), "the new one and the nine newest before it")
            self.assertEqual(sorted(entry.name for entry in Path(base).iterdir() if entry.name not in runs),
                             ["20260101-000000", "durations.json", "notes"])

    def test_save_durations_prunes_sections_no_longer_in_config(self):
        with tempfile.TemporaryDirectory() as base, mock.patch.object(run_sections, "LOG_BASE", Path(base)):
            Path(base, "durations.json").write_text(
                '{"alpha": 10.0, "beta": 20.0, "removed_section": 99.0}', encoding="utf-8")
            results = [
                run_sections.Result("alpha", "ok", seconds=12.34),
                run_sections.Result("repo-checks", "ok", seconds=3.21),
                run_sections.Result("beta", "timeout", seconds=60.0),
            ]
            run_sections.save_durations(results, {"alpha", "beta"})
            self.assertEqual(run_sections.load_durations(), {"alpha": 12.3, "beta": 20.0})


class MacGroupAnswerTests(unittest.TestCase):
    """killpg as macOS answers it: EPERM for a group whose only members are
    zombies not yet reaped (Linux answers success for them), then ESRCH. Such a
    group runs nothing of ours, so it is not reported as having members; a group
    that answers success to signal 0 has a live member and still is."""

    def setUp(self):
        self.signals = []
        patcher = mock.patch.object(run_sections.signal, "SIGKILL", 9, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.process = mock.Mock(pid=4242)
        self.process.poll.return_value = 0

    def stand_in(self, probe, send=None):
        def killpg(group, signal_number):
            if signal_number:
                self.signals.append(signal_number)
                if send is not None:
                    raise send
            elif probe is not None:
                raise probe
        return mock.patch.object(run_sections.os, "killpg", killpg, create=True)

    def test_a_probe_answering_eperm_means_the_group_ended(self):
        with self.stand_in(PermissionError(1, "Operation not permitted")):
            self.assertTrue(run_sections._group_ended(self.process, 4242, time.monotonic() + 5))

    def test_a_probe_answering_eperm_with_the_leader_alive_is_not_an_end(self):
        self.process.poll.return_value = None
        with self.stand_in(PermissionError(1, "Operation not permitted")):
            self.assertFalse(run_sections._group_ended(self.process, 4242, time.monotonic() + 0.1))

    def test_eperm_everywhere_with_a_live_leader_is_reported_not_stopped(self):
        # A sandbox that refuses signals.
        self.process.poll.return_value = None
        refused = PermissionError(1, "Operation not permitted")
        with self.stand_in(refused, send=refused), mock.patch.object(run_sections, "STOP_WAIT", 0.1):
            problem = run_sections._kill_process_group(self.process)
        self.assertEqual(problem, "process group 4242 still has members after SIGKILL")
        self.assertEqual(self.signals, [15, 9])

    def test_a_probe_answering_success_means_a_member_lives(self):
        with self.stand_in(None):
            self.assertFalse(run_sections._group_ended(self.process, 4242, time.monotonic() + 0.1))

    def test_the_stop_ends_cleanly_when_the_members_are_unreaped_zombies(self):
        with self.stand_in(PermissionError(1, "Operation not permitted")),                 mock.patch.object(run_sections, "STOP_WAIT", 0.2):
            self.assertEqual(run_sections._kill_process_group(self.process), "")
        self.assertEqual(self.signals, [15], "no SIGKILL for a group nothing can be signalled in")

    def test_eperm_on_a_send_ends_the_stop(self):
        with self.stand_in(None, send=PermissionError(1, "Operation not permitted")):
            self.assertEqual(run_sections._kill_process_group(self.process), "")

    def test_a_live_member_is_still_reported(self):
        with self.stand_in(None), mock.patch.object(run_sections, "STOP_WAIT", 0.1):
            problem = run_sections._kill_process_group(self.process)
        self.assertEqual(problem, "process group 4242 still has members after SIGKILL")
        self.assertEqual(self.signals, [15, 9])


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


class GitAvailabilityTests(unittest.TestCase):
    """Tests run git only where it works: on a Mac without Apple's command-line tools /usr/bin/git is a stub that opens an install dialog."""

    # Files that may name git without the helper, and why.
    ALLOWED = {
        "tests/git_availability.py": "the helper itself",
    }
    # What a file shows once it asks first: the test helper, or the benchmark's copy of it.
    ASKS_FIRST = ("git_availability", "git_problem")

    def test_the_runner_does_not_start_git_where_git_cannot_be_used(self):
        stub = dict(platform="darwin", which=lambda name: "/usr/bin/git",
                    run=lambda command, **keywords: subprocess.CompletedProcess(command, 2))
        real = run_sections.git_availability.git_problem
        with mock.patch.object(run_sections, "_GIT_EXECUTABLE", []), \
                mock.patch.object(run_sections.git_availability, "git_problem", lambda root=None: real(root, **stub)), \
                mock.patch.object(run_sections.subprocess, "run", side_effect=AssertionError("git started")) as run:
            self.assertIsNone(run_sections._git_lines(PROJECT_ROOT, "ls-files"))
            self.assertIsNone(run_sections._git_lines(PROJECT_ROOT, "diff", "--name-only"))
            run.assert_not_called()
            self.assertEqual(run_sections._GIT_EXECUTABLE, [None])  # asked once
            self.assertTrue(run_sections.list_files(PROJECT_ROOT))  # falls back to walking the folder

    def test_the_runner_runs_the_git_it_found(self):
        with mock.patch.object(run_sections, "_GIT_EXECUTABLE", []), \
                mock.patch.object(run_sections.git_availability, "git_problem", return_value=None), \
                mock.patch.object(run_sections.shutil, "which", return_value="/opt/x/git"), \
                mock.patch.object(run_sections.subprocess, "run",
                                  return_value=subprocess.CompletedProcess([], 0, stdout="a\nb\n")) as run:
            self.assertEqual(run_sections._git_lines("/p", "ls-files"), ["a", "b"])
        self.assertEqual(run.call_args.args[0][0], "/opt/x/git")

    def problem(self, platform="darwin", found="/usr/bin/git", tools=0, git_entry=True, root="/project"):
        def which(name):
            return found

        def run(command, **keywords):
            self.assertEqual(command, ["xcode-select", "-p"])
            self.assertIn("timeout", keywords)
            if isinstance(tools, BaseException):
                raise tools
            return subprocess.CompletedProcess(command, tools)

        return git_problem(root, platform=platform, which=which, run=run, has_git_entry=git_entry)

    def test_a_missing_executable_is_a_problem_everywhere(self):
        for platform in ("win32", "linux", "darwin"):
            with self.subTest(platform=platform):
                self.assertEqual(self.problem(platform, found=None), "git is not installed")

    def test_apples_stub_needs_the_command_line_tools(self):
        self.assertIsNone(self.problem(tools=0))
        self.assertIn("command-line tools", self.problem(tools=2))
        self.assertIn("command-line tools", self.problem(tools=subprocess.TimeoutExpired("xcode-select", 10)))
        self.assertIn("command-line tools", self.problem(tools=FileNotFoundError()))

    def test_only_the_stub_on_a_mac_is_asked_about_the_tools(self):
        # xcode-select exits 2 here, so it would be a problem if it were asked.
        self.assertIsNone(self.problem(found="/opt/homebrew/bin/git", tools=2))
        self.assertIsNone(self.problem(found="/usr/local/bin/git", tools=2))
        for platform in ("win32", "linux"):
            with self.subTest(platform=platform):
                self.assertIsNone(self.problem(platform, tools=2))

    def test_a_folder_without_a_git_entry_is_not_a_work_tree(self):
        self.assertEqual(self.problem(git_entry=False), "this folder is not a git work tree")
        self.assertIsNone(git_problem(None, platform="linux", which=lambda name: "/usr/bin/git", has_git_entry=False))

    def test_a_git_entry_is_looked_for_in_the_folder_without_running_git(self):
        with tempfile.TemporaryDirectory() as folder:
            found = dict(platform="linux", which=lambda name: "/usr/bin/git", run=mock.Mock(side_effect=AssertionError))
            self.assertEqual(git_problem(folder, **found), "this folder is not a git work tree")
            Path(folder, ".git").write_text("gitdir: elsewhere", encoding="utf-8")  # a linked work tree's file
            self.assertIsNone(git_problem(folder, **found))

    def test_a_test_that_cannot_use_git_is_skipped_with_the_reason(self):
        with mock.patch("tests.git_availability.git_problem", return_value="no git here"):
            with self.assertRaisesRegex(unittest.SkipTest, "no git here"):
                git_executable_for(PROJECT_ROOT)

    @staticmethod
    def runs_git(tree):
        """Whether a module names the git executable as a command: ["git", ...] or which("git")."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
                first = node.elts[0]
                if isinstance(first, ast.Constant) and first.value in ("git", "git.exe"):
                    return True
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
                if name in ("which", "run", "Popen", "check_output", "check_call", "call") and node.args[0].value in ("git", "git.exe"):
                    return True
        return False

    def test_the_scan_sees_the_ways_a_test_names_git(self):
        for text in ('subprocess.run(["git", "status"])', "shutil.which('git')", 'x = ("git", "-C", folder)',
                     'subprocess.check_output(["git.exe", "log"])'):
            with self.subTest(text=text):
                self.assertTrue(self.runs_git(ast.parse(text)))
        for text in ('subprocess.run(["gitk"])', 'print("git is not installed")', 'x = ["a", "git"]'):
            with self.subTest(text=text):
                self.assertFalse(self.runs_git(ast.parse(text)))

    def test_every_python_file_that_runs_git_asks_first(self):
        found = []
        for folder in ("tests", "src"):
            for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                if relative in self.ALLOWED or ".venv" in path.parts:
                    continue
                source = path.read_text(encoding="utf-8")
                if "git" not in source:
                    continue
                if self.runs_git(ast.parse(source)) and not any(mark in source for mark in self.ASKS_FIRST):
                    found.append(relative)
        self.assertEqual(found, [], "these run git without asking tests.git_availability or git_problem first")

    def test_the_scan_covers_the_runner_and_the_benchmark(self):
        for relative in ("tests/run_sections.py", "tests/git_availability.py", "src/resources/benchmark/Run_Benchmark.py"):
            source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(relative=relative):
                self.assertTrue(any(mark in source for mark in self.ASKS_FIRST))


if __name__ == "__main__":
    unittest.main()
