# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The program runs with any one command taken away, and a command that is renamed is found under its new name.

Commands are plug-ins: a module of src/commands says what the program needs to know about it (Command_Declarations,
desktop.Command_Metadata, commands/reset.py), and nothing central names it. CentralCodeNamesNoCommandTests
(test_command_declarations.py) reads the code for that; this shows it by running.

RemovingAnyOneCommandTests runs tests/command_removal_probe.py in a process of its own, offscreen. It imports every
central module while no command can be imported, then reads what depends on the commands from a copy of the commands
folder that has them all and from a copy that lacks one, for each command in turn: the dispatch declarations, the
Console buttons, the Session card's pick buttons, the reset targets, the catalog, the Agent's review policy and
instructions, what help lists, what typing the missing command says, and the Viewer's Session card, Console buttons and
Zoom box. Only the removed command may be gone from any of them, with the commands whose modules import it (they cannot
be imported without it), and what the commands import of each other is read here from their source.

RenamedCommandTests copies real modules under another name into a scratch commands folder, and every declaration
follows the new name.
"""
import ast
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import Command_Declarations as declarations  # noqa: E402
from desktop import Command_Metadata  # noqa: E402
from desktop.Viewer_Inspection import command_catalog  # noqa: E402
from tests.command_fixtures import one_node_viewer, only_commands, reported_outcomes, scratch_commands  # noqa: E402

PROJECT_ROOT = SRC_DIR.parent
COMMANDS_DIR = SRC_DIR / "commands"
PROBE = Path(__file__).resolve().with_name("command_removal_probe.py")
COMMAND_NAMES = sorted(path.stem for path in COMMANDS_DIR.glob("*.py") if not path.stem.startswith("_"))
# The probe takes about 20 s when the machine is quiet, and the first import of what it loads can take longer.
PROBE_TIMEOUT = 600
BLOW_UP = 240


def module_level_command_imports(name):
    """The commands that the module of command ``name`` imports when it is imported (not those it imports in a function)."""
    tree = ast.parse((COMMANDS_DIR / f"{name}.py").read_text(encoding="utf-8"))
    found = set()

    def visit(statements):
        for statement in statements:
            if isinstance(statement, ast.ImportFrom):
                module = (statement.module or "").split(".")
                if module == ["commands"]:
                    found.update(alias.name for alias in statement.names)
                elif module[0] == "commands" and len(module) > 1:
                    found.add(module[1])
            elif isinstance(statement, ast.Import):
                found.update(alias.name.split(".")[1] for alias in statement.names if alias.name.startswith("commands."))
            elif isinstance(statement, (ast.Try, ast.If, ast.With)):  # a block at the top level runs when it is imported
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(statement, field, []))
                for handler in getattr(statement, "handlers", []):
                    visit(handler.body)

    visit(tree.body)
    return found & set(COMMAND_NAMES)


def cannot_be_imported_without(name):
    """The commands whose modules cannot be imported when ``name`` is missing: itself, and those that import it, and theirs."""
    needs = {command: module_level_command_imports(command) for command in COMMAND_NAMES}
    gone = {name}
    while True:
        more = {command for command, imports in needs.items() if imports & gone} - gone
        if not more:
            return gone
        gone |= more


class RemovingAnyOneCommandTests(unittest.TestCase):
    """Every command taken away in turn: the rest of the program works, and nothing of it names the one that is gone."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory(prefix="emapssn-removal-") as scratch:
            result = subprocess.run(
                [sys.executable, str(PROBE), str(SRC_DIR), scratch], capture_output=True, text=True, cwd=PROJECT_ROOT,
                timeout=PROBE_TIMEOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        cls.probe_returncode, cls.probe_stderr = result.returncode, result.stderr
        lines = result.stdout.strip().splitlines()
        cls.probe = json.loads(lines[-1]) if result.returncode == 0 and lines else None
        cls.probe_output = result.stdout[-3000:]

    def setUp(self):
        self.assertEqual(self.probe_returncode, 0, self.probe_stderr + self.probe_output)
        self.assertIsNotNone(self.probe, self.probe_output)

    def test_the_probe_read_the_real_commands(self):
        baseline = self.probe["baseline"]
        self.assertEqual(baseline["names"], COMMAND_NAMES)
        self.assertEqual(sorted(self.probe["removed"]), COMMAND_NAMES)
        self.assertEqual(baseline["unimportable"], [])
        self.assertEqual(baseline["console_buttons"], ["agent", "esmfold", "meta"])
        self.assertEqual(baseline["pickers"], {"embedding": "embedding", "msa": "alignment"})
        self.assertEqual(baseline["help"], COMMAND_NAMES)
        self.assertEqual(baseline["reset_targets"], ["cluster", "color", "group", "hidden", "hide", "layer", "network", "order",
                                                     "shape", "size"])

    def test_every_central_module_imports_while_no_command_can_be_imported(self):
        imported = self.probe["imported"]
        self.assertGreater(len(imported), 90)
        for module in ("Command_Engine", "EMAPSSN_Viewer", "desktop.Viewer_Widgets", "desktop.Viewer_Zoom",
                       "desktop.Command_Metadata", "Viewer_Command_Portal", "web_ui.agent_prompt", "web_ui.Web_Server",
                       "web_ui.esmfold_backend", "web_ui.agent_backend", "web_ui.meta_backend"):
            self.assertIn(module, imported)
        self.assertEqual({module: error for module, error in imported.items() if error}, {})

    def test_the_rest_of_the_program_works_without_any_one_command(self):
        for name in COMMAND_NAMES:
            with self.subTest(removed=name):
                self.assertEqual(self.probe["problems"][name], [])

    def test_a_removed_command_is_in_none_of_the_lists_made_from_the_modules(self):
        for name in COMMAND_NAMES:
            removed = self.probe["removed"][name]
            with self.subTest(removed=name):
                self.assertNotIn(name, removed["prompt_items"])
                self.assertNotIn(name, removed["pickers"].values())
                self.assertNotIn(name, removed["console_buttons"])
                if removed["listed_by_help"] is not None:
                    self.assertNotIn(name, removed["listed_by_help"])
                self.assertEqual(removed["typed"][name], [f"Unknown command: {name}"])

    def test_the_commands_that_cannot_be_imported_without_one_are_those_that_import_it(self):
        # label, subcluster and cluster need the module of cluster, and group is needed by cluster, and so on; the
        # others stand alone, so any of them can be taken away without another command going.
        dependent = {}
        for name in COMMAND_NAMES:
            expected = cannot_be_imported_without(name)
            with self.subTest(removed=name):
                self.assertEqual(set(self.probe["removed"][name]["unimportable"]), expected)
            if expected != {name}:
                dependent[name] = sorted(expected - {name})
        self.assertEqual(dependent, {"cluster": ["label", "subcluster"], "group": ["cluster", "label", "subcluster"],
                                     "logo": ["label"]})

    def test_a_command_that_cannot_be_imported_without_another_reports_the_dependency_when_typed(self):
        for name, removed in self.probe["removed"].items():
            for dependent in set(removed["unimportable"]) - {name}:
                with self.subTest(removed=name, typed=dependent):
                    [report] = removed["typed"][dependent]
                    self.assertRegex(report, r"^(Command dependency unavailable|Unknown command)")

    def test_the_probe_ran_in_a_reasonable_time(self):
        seconds = self.probe["seconds"]
        print(f"\ncommand removal probe: {seconds['total']} s in all ({seconds['import with no command importable']} s to "
              f"import every central module, {seconds['baseline']} s for the baseline, about "
              f"{max(r['seconds'] for r in self.probe['removed'].values())} s at most for each command taken away)", file=sys.stderr)
        self.assertLess(seconds["total"], BLOW_UP)


REAL_COPIES = ("meta", "alignment", "color", "jobs", "agent", "zoom", "esmfold")


def renamed(name):
    return f"zz_{name}"


def copies(names=REAL_COPIES):
    """{zz_<name>: the source of the real module}: the same declarations, under another name."""
    return {renamed(name): (COMMANDS_DIR / f"{name}.py").read_text(encoding="utf-8") for name in names}


class RenamedCommandTests(unittest.TestCase):
    """A real module copied under another name is described, reviewed, dispatched and reset by that name."""

    def folder(self):
        import commands

        return Path(commands.__path__[-1])  # scratch_commands appends its folder last

    def test_the_catalog_lists_the_new_name_with_what_the_module_declares(self):
        with scratch_commands(copies()):
            entries = {entry["command"]: entry for entry in command_catalog(commands_dir=self.folder())["commands"]}
        real = {entry["command"]: entry for entry in command_catalog()["commands"]}
        self.assertEqual(set(entries), {renamed(name) for name in REAL_COPIES})
        for name in REAL_COPIES:
            with self.subTest(command=name):
                for key in ("summary", "arguments", "requires", "agent_policy", "writes_files", "opens_interface", "background_work"):
                    self.assertEqual(entries[renamed(name)][key], real[name][key], key)
                self.assertNotIn("metadata_declared", entries[renamed(name)])

    def test_the_review_policy_follows_the_name(self):
        lines = {"meta": ["meta", "meta upload a.csv", "meta show x"], "agent": ["agent", "agent hello", "agent off"],
                 "jobs": ["jobs", "jobs cancel all"], "zoom": ["zoom 200"], "color": ["color red"],
                 "esmfold": ["esmfold", "esmfold stop"]}
        with scratch_commands(copies()):
            for name, commands in lines.items():
                for line in commands:
                    renamed_line = renamed(name) + line[len(name):]
                    with self.subTest(line=renamed_line):
                        self.assertEqual(Command_Metadata.agent_policy(renamed_line), Command_Metadata.agent_policy(line))
            plain, by_form = Command_Metadata.agent_policy_summary()
            names = {name for kind in plain.values() for name in kind} | {name for name, _default, _exceptions in by_form}
        self.assertTrue({renamed(name) for name in REAL_COPIES} <= names)

    def test_the_agents_instructions_get_the_modules_item_under_the_new_name(self):
        from web_ui.agent_prompt import render_system_prompt

        template = (SRC_DIR / "resources" / "agent" / "system_prompt.md").read_text(encoding="utf-8")
        with scratch_commands(copies(("meta", "color", "jobs"))):
            items = {item.name: item for item in Command_Metadata.agent_prompt_items()}
            text = render_system_prompt(template)
        for name in ("meta", "color", "jobs"):
            with self.subTest(command=name):
                real = Command_Metadata.declarations(name)
                self.assertEqual((items[renamed(name)].text, items[renamed(name)].order), (real.agent_prompt, real.agent_prompt_order))
                self.assertIn(items[renamed(name)].text.strip().splitlines()[0], text)
        self.assertTrue({"zz_meta", "zz_color", "zz_jobs"} <= set(items))

    def test_the_dispatch_declarations_follow_the_name(self):
        import Command_Engine as engine

        with scratch_commands(copies()):
            self.assertEqual(engine.queue_kind("zz_zoom", ["200"], "typed"), "instant")
            self.assertEqual(engine.queue_kind("zz_meta", ["upload", "a.csv"], "typed"), "ordered")
            self.assertEqual(engine.queue_kind("zz_meta", [], "portal"), "instant")
            self.assertTrue(declarations.acts_on_jobs("zz_jobs", []))
            self.assertTrue(declarations.acts_on_jobs("zz_esmfold", ["stop"]))
            self.assertFalse(declarations.acts_on_jobs("zz_esmfold", []))
            self.assertTrue(declarations.starts_model_request("zz_agent", ["hello"]))
            self.assertFalse(declarations.starts_model_request("zz_agent", ["off"]))
            self.assertFalse(declarations.starts_model_request("zz_meta", ["hello"]))

    def test_the_console_button_follows_the_name(self):
        with scratch_commands(copies()):
            buttons = {name for name in (renamed(name) for name in REAL_COPIES) if declarations.registers_console_button(name)}
        self.assertEqual(buttons, {"zz_agent", "zz_esmfold", "zz_meta"})

    def test_the_session_cards_pick_button_follows_the_name(self):
        with only_commands(copies(("alignment", "embedding", "meta"))):
            self.assertEqual(declarations.session_file_pickers(), {"msa": "zz_alignment", "embedding": "zz_embedding"})

    def test_a_reset_target_follows_the_name_and_is_listed_by_the_help(self):
        from commands import reset

        with only_commands(copies(("color",))):
            owners = reset.declared_targets()
            self.assertEqual([module.__name__ for module in owners["color"]], ["commands.zz_color"])
            self.assertIn("colors", reset.usage())
            self.assertIn("\n      colors\n          Restores all node colors to the configured default color.", reset.targets_help())
            viewer = one_node_viewer()
            viewer.current_colors = np.array([[0.9, 0.9, 0.9, 1.0]])
            with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
                reset.run(viewer, ["colors"])
            failed.assert_not_called()
            succeeded.assert_called_once_with(viewer, "Reset successful: colors.")
            self.assertFalse(np.allclose(viewer.current_colors, [[0.9, 0.9, 0.9, 1.0]]))
            viewer._save_state.assert_called_once()
        # With no module that declares it, "color" is no target.
        with only_commands({}):
            self.assertNotIn("color", reset.target_names())
            self.assertNotIn("colors", reset.usage())

    def test_help_lists_a_renamed_command_under_its_new_name(self):
        from commands import help as help_command

        with scratch_commands(copies(("meta",))):
            folder = self.folder()
            with mock.patch("desktop.Viewer_Inspection.command_catalog",
                            lambda command=None, commands_dir=None: command_catalog(command, commands_dir=folder)):
                output = io.StringIO()
                with reported_outcomes(), redirect_stdout(output):
                    help_command._list_commands(SimpleNamespace(console_text=SimpleNamespace(text="")))
        listed = [line.split()[0] for line in output.getvalue().splitlines() if line.startswith("  ") and line[2:3] != " "]
        self.assertIn("zz_meta", listed)
        self.assertNotIn("meta", listed)


if __name__ == "__main__":
    unittest.main()
