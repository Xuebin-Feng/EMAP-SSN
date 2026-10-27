# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The help command (commands/help.py) and the failure line of a command that cannot be dispatched.

help lists the commands from the MCP command catalog, shows one command's help exactly as ``<command> help``
does without running the command, says that a word that is no command is unknown, runs at once, imports no Qt,
and is catalogued as a command that writes nothing and opens nothing. Command_Engine prints one red failure line
for an unknown command, a dependency that is missing and a module without run(), so the Console and the log have
it (it was on the status line only), and a request from the command portal gets its outcome as before.
"""

from __future__ import annotations

import ast
import io
import re
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import Command_Engine  # noqa: E402
from commands import help as help_command  # noqa: E402
from desktop.Command_Metadata import AGENT_POLICY, COMMAND_METADATA, agent_policy  # noqa: E402
from desktop.Viewer_Inspection import ViewerInspectionService, command_catalog  # noqa: E402
from utilities.Line_Kinds import ERROR, classify  # noqa: E402
from Viewer_Command_Portal import bind  # noqa: E402
from tests.command_fixtures import run_command, scratch_commands  # noqa: E402
from tests.viewer_fixtures import PortalFixture  # noqa: E402

PRINTED = "Help information printed to the Console."
HINT = "Type help for the list of commands."

# A command that records how it is run, and answers help as every command does.
RECORDING = (
    "import Command_Engine\n"
    "def run(viewer, args):\n"
    "    Command_Engine.SCRATCH_CALLS.append(list(args))\n"
    "    if args and args[0].lower() == 'help':\n"
    "        print('ZZ-HELP-TEXT')\n"
    "        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')\n"
    "        return\n"
    "    print('ZZ-RAN')\n"
    "    Command_Engine.command_succeeded(viewer, 'ran')\n"
)


def plain_viewer():
    return SimpleNamespace(console_text=SimpleNamespace(text=""))


def listed(text):
    """{command: its summary, wrapped lines joined} from the text of a listing."""
    found, current = {}, None
    for line in text.splitlines():
        entry = re.match(r"^  (\S+) {2,}(\S.*)$", line)
        more = re.match(r"^ {3,}(\S.*)$", line)
        if entry:
            current = entry.group(1)
            found[current] = entry.group(2)
        elif more and current:
            found[current] += " " + more.group(1)
        else:
            current = None
    return found


class ListingTests(unittest.TestCase):
    def test_bare_help_lists_every_command_with_the_first_sentence_of_its_catalog_summary(self):
        succeeded, failed, printed = run_command(help_command, plain_viewer(), [])
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], PRINTED)
        entries = command_catalog()["commands"]
        self.assertEqual(len(entries), len(COMMAND_METADATA))
        shown = listed(printed)
        self.assertEqual(list(shown), [entry["command"] for entry in entries])
        for entry in entries:
            with self.subTest(command=entry["command"]):
                self.assertEqual(shown[entry["command"]], help_command.one_line(entry["summary"]))
                self.assertTrue(shown[entry["command"]])
        self.assertIn("help", shown)

    def test_the_status_line_says_the_help_was_printed(self):
        fake = plain_viewer()
        run_command(help_command, fake, [])
        self.assertEqual(fake.console_text.text, "Help information printed to the Console")

    def test_the_listing_is_read_from_the_catalog_the_mcp_server_serves(self):
        served = [{"command": "alpha", "summary": "Does alpha things. More."},
                  {"command": "beta", "summary": "Does the beta things now; needs a thing."}]
        with mock.patch("desktop.Viewer_Inspection.command_catalog", return_value={"commands": served}):
            _, _, printed = run_command(help_command, plain_viewer(), [])
        self.assertIn("  alpha  Does alpha things\n", printed)
        self.assertIn("  beta   Does the beta things now\n", printed)
        self.assertNotIn("zoom", printed)

    def test_a_command_dropped_into_the_commands_folder_is_listed_without_any_change(self):
        with tempfile.TemporaryDirectory() as folder:
            commands = Path(folder) / "commands"
            shutil.copytree(SRC_DIR / "commands", commands, ignore=shutil.ignore_patterns("__pycache__"))
            (commands / "zzlisted.py").write_text(
                '"""Scratch command that is only listed."""\ndef run(viewer, args):\n    return None\n', encoding="utf-8")
            # The catalog the MCP server serves, read from the folder with the plugin in it.
            with mock.patch("desktop.Viewer_Inspection.command_catalog",
                            side_effect=lambda *args, **kwargs: command_catalog(*args, commands_dir=commands, **kwargs)):
                _, _, printed = run_command(help_command, plain_viewer(), [])
        shown = listed(printed)
        self.assertEqual(shown["zzlisted"], "Scratch command that is only listed")
        self.assertEqual(set(shown) - {"zzlisted"}, {entry["command"] for entry in command_catalog()["commands"]})

    def test_a_summary_is_reduced_to_what_the_command_does(self):
        one_line = help_command.one_line
        self.assertEqual(one_line("Queue an XLSX conservation report; requires a nonempty MSA. It ends."), "Queue an XLSX conservation report")
        self.assertEqual(one_line("Queue an XLSX conservation report; requires a nonempty MSA"), "Queue an XLSX conservation report")
        self.assertEqual(one_line("Does a thing, then another. Needs more."), "Does a thing, then another")
        # A clause too short to say what the command does is not cut at its semicolon.
        self.assertEqual(one_line("Do it; then more."), "Do it; then more")
        self.assertEqual(one_line("Two\n   lines  here."), "Two lines here")

    def test_a_long_summary_hangs_under_its_column(self):
        long = " ".join(["word"] * 60)
        text = help_command.listing([{"command": "ab", "summary": long}, {"command": "abcd", "summary": "Short."}])
        lines = text.splitlines()
        first = next(index for index, line in enumerate(lines) if line.startswith("  ab  "))
        self.assertTrue(lines[first + 1].startswith(" " * 8 + "word"))
        self.assertTrue(all(len(line) <= help_command.LISTING_WIDTH for line in lines))


class OfACommandTests(PortalFixture, unittest.TestCase):
    """help <command> is <command> help, to the letter, and the command is not run."""

    def setUp(self):
        super().setUp()
        import EMAPSSN_Config as cfg
        # Some commands create their output folders before anything else, even for help.
        for name, folder in (("METADATA_DIR", "metadata"), ("ANALYSIS_RESULT_DIR", "results")):
            patcher = mock.patch.object(cfg, name, str(Path(self.directory.name) / folder))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.viewer.alignment = SimpleNamespace(aln=[object()], has_reference=True)
        for target in ("web_ui.agent_backend.register", "web_ui.meta_backend.register"):
            patcher = mock.patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def execute(self, command, expected=None):
        """The request's single command record and what it printed."""
        output = io.StringIO()
        with redirect_stdout(output):
            result = self.finish(self.portal.submit(str(len(self.portal.requests)), command)["request_id"])
        if expected is not None:
            self.assertEqual(result["status"], expected, result)
        return result["commands"][0], output.getvalue()

    @staticmethod
    def outcome(record):
        return [(message["status"], message["text"]) for message in record["messages"]]

    def test_every_commands_help_is_shown_as_the_command_shows_it(self):
        for entry in command_catalog()["commands"]:
            name = entry["command"]
            with self.subTest(command=name):
                direct, direct_text = self.execute(f"{name} help", "succeeded")
                asked, asked_text = self.execute(f"help {name}", "succeeded")
                self.assertTrue(direct_text.strip())
                self.assertEqual(asked_text, direct_text)
                self.assertEqual(self.outcome(asked), self.outcome(direct))
                self.assertEqual(self.outcome(asked), [("succeeded", PRINTED)])

    def test_the_command_is_not_run(self):
        with scratch_commands({"zzrecorded": RECORDING}):
            Command_Engine.SCRATCH_CALLS = []
            record, text = self.execute("help zzrecorded", "succeeded")
            self.assertEqual(Command_Engine.SCRATCH_CALLS, [["help"]])
            self.assertIn("ZZ-HELP-TEXT", text)
            self.assertNotIn("ZZ-RAN", text)
            self.assertEqual(self.outcome(record), [("succeeded", PRINTED)])
            # What the command does with its own arguments is no part of help.
            Command_Engine.SCRATCH_CALLS = []
            self.execute("help zzrecorded now", "failed")
            self.assertEqual(Command_Engine.SCRATCH_CALLS, [])

    def test_a_real_command_changes_nothing_when_its_help_is_asked_for(self):
        self.viewer._do_undo = mock.Mock()
        self.viewer.visible_mask[:] = False
        for line in ("help undo", "help hide", "help reset", "help select"):
            with self.subTest(line=line):
                self.execute(line, "succeeded")
        self.viewer._do_undo.assert_not_called()
        self.assertEqual(self.viewer.saved, 0)
        self.assertFalse(self.viewer.visible_mask.any())
        self.assertEqual(self.viewer.selected_indices, [])

    def test_the_name_is_not_case_sensitive(self):
        _, lower = self.execute("help zoom", "succeeded")
        _, upper = self.execute("HELP ZoOm", "succeeded")
        self.assertEqual(upper, lower)

    def test_help_help_and_the_help_words_print_its_own_help(self):
        for line in ("help help", "help -h", "help --help", "help -?", "HELP HELP"):
            with self.subTest(line=line):
                record, text = self.execute(line, "succeeded")
                self.assertRegex(text, r"Usage: help\n")
                self.assertIn("help <command>", text)
                self.assertIn("Examples:", text)
                self.assertNotIn("EMAP-SSN Viewer commands", text)
                self.assertEqual(self.outcome(record), [("succeeded", PRINTED)])

    def test_a_word_that_is_no_command_fails_and_says_so(self):
        for word in ("nosuch", "NoSuch", "__init__", "_private", "zoom.py", "../zoom", "commands.zoom", "-x"):
            with self.subTest(word=word):
                record, text = self.execute(f"help {word}", "failed")
                shown = word.lower()
                self.assertEqual(self.outcome(record), [("failed", f"Unknown command: {shown}")])
                self.assertIn(f"Error: Unknown command: {shown}. {HINT}", text)
                self.assertEqual(self.viewer.console_text.text, f"Unknown command: {shown}")
                self.assertEqual(classify(f"Error: Unknown command: {shown}. {HINT}"), ERROR)

    def test_a_module_without_run_is_no_command(self):
        with scratch_commands({"zzhelper": "VALUE = 1\n"}):
            record, text = self.execute("help zzhelper", "failed")
        self.assertEqual(self.outcome(record), [("failed", "Unknown command: zzhelper")])
        self.assertIn("Error: Unknown command: zzhelper.", text)

    def test_more_than_one_word_is_refused_and_nothing_is_shown(self):
        for line in ("help color red", "help color help", "help help help"):
            with self.subTest(line=line):
                record, text = self.execute(line, "failed")
                self.assertEqual(len(record["messages"]), 1, record)
                self.assertIn("takes at most one command name", record["messages"][0]["text"])
                self.assertIn("Usage: help [command]", text)
                self.assertNotIn("Advanced Coloring", text)

    def test_a_command_that_cannot_be_imported_fails_as_typing_it_would(self):
        with scratch_commands({"zzbroken": "import zz_missing_dependency\ndef run(viewer, args):\n    return None\n"}):
            record, text = self.execute("help zzbroken", "failed")
        self.assertEqual(self.outcome(record), [("failed", "Command dependency unavailable: zz_missing_dependency")])
        self.assertIn("Error: Command dependency unavailable: zz_missing_dependency", text)

    def test_it_runs_at_once_even_while_other_commands_wait(self):
        for args in ([], ["zoom"], ["help"], ["nosuch"]):
            for source in ("typed", "script", "portal"):
                with self.subTest(args=args, source=source):
                    self.assertEqual(Command_Engine.queue_kind("help", args, source), Command_Engine.INSTANT)
        self.assertEqual(help_command.QUEUE, "instant")

    def test_a_module_a_waiting_command_still_has_to_run_is_not_reloaded_for_its_help(self):
        waiting = SimpleNamespace(uses_module=lambda name: name == "zoom")
        viewer = SimpleNamespace(command_queue=waiting)
        with mock.patch("importlib.reload") as reload:
            Command_Engine.load_command_module(viewer, "zoom")
            Command_Engine.load_command_module(viewer, "color")
        self.assertEqual([call.args[0].__name__ for call in reload.call_args_list], ["commands.color"])


class UnknownCommandLineTests(PortalFixture, unittest.TestCase):
    """The failure of a command that cannot be dispatched reaches the Console and the log, in red."""

    def type(self, line, silent=False):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()), bind(None):
            Command_Engine.execute_command(self.viewer, line, record_history=False, silent=silent)
        return output.getvalue()

    def test_an_unknown_command_prints_one_red_line_with_a_hint(self):
        text = self.type("blahblah")
        lines = [line for line in text.splitlines() if line.strip()]
        self.assertEqual(lines, [f"Error: Unknown command: blahblah. {HINT}"])
        self.assertEqual(classify(lines[0]), ERROR)
        # The status line says what it always did.
        self.assertEqual(self.viewer.console_text.text, "Unknown command: blahblah")

    def test_the_name_is_printed_as_the_dispatch_reads_it(self):
        self.assertIn("Error: Unknown command: blahblah.", self.type("BlahBlah with words"))

    def test_silent_keeps_the_status_line_quiet_and_not_the_log(self):
        # silent only spares the status line, as in _report_command_error, which prints whatever silent is.
        text = self.type("blahblah", silent=True)
        self.assertIn(f"Error: Unknown command: blahblah. {HINT}", text)
        self.assertEqual(self.viewer.console_text.text, "")

    def test_a_request_from_the_portal_gets_its_outcome_as_before_and_the_line_is_printed(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = self.finish(self.portal.submit("unknown", "blahblah")["request_id"])
        self.assertEqual(result["status"], "failed")
        record = result["commands"][0]
        self.assertEqual([(m["status"], m["text"]) for m in record["messages"]],
                         [("failed", "Unknown command: blahblah")])
        self.assertIn("Error: Unknown command: blahblah.", output.getvalue())

    def test_a_missing_dependency_prints_its_own_line_without_the_hint(self):
        with scratch_commands({"zzbroken": "import zz_missing_dependency\ndef run(viewer, args):\n    return None\n"}):
            text = self.type("zzbroken")
        self.assertEqual([line for line in text.splitlines() if line.strip()],
                         ["Error: Command dependency unavailable: zz_missing_dependency"])
        self.assertEqual(classify("Error: Command dependency unavailable: zz_missing_dependency"), ERROR)

    def test_a_module_without_run_prints_its_failure_too(self):
        with scratch_commands({"zzhelper": "VALUE = 1\n"}):
            text = self.type("zzhelper")
        self.assertEqual([line for line in text.splitlines() if line.strip()], ["Error: No run entry point in zzhelper"])
        self.assertEqual(self.viewer.console_text.text, "Error: No 'run' in zzhelper")

    def test_a_failure_inside_a_command_still_prints_as_it_did(self):
        with scratch_commands({"zzraises": "def run(viewer, args):\n    raise RuntimeError('broken')\n"}):
            text = self.type("zzraises")
        self.assertEqual(text.strip(), "Command Error: broken")  # the traceback goes to stderr


class CatalogTests(unittest.TestCase):
    def test_it_is_catalogued_and_writes_nothing_and_opens_nothing(self):
        entry = command_catalog("help")["commands"][0]
        self.assertTrue(entry["summary"])
        self.assertEqual((entry["writes_files"], entry["opens_interface"], entry["background_work"]), (False, False, False))
        self.assertEqual(entry["requires"], [])
        self.assertNotIn("metadata_declared", entry)
        self.assertEqual([argument["name"] for argument in entry["arguments"]], ["command", "help"])

    def test_the_mcp_catalog_serves_it_with_its_syntax_and_help(self):
        service = ViewerInspectionService(mock.Mock())
        general = service.command_action("get_command_catalog", {})
        self.assertIn("help", {entry["command"] for entry in general["commands"]})
        detailed = service.command_action("get_command_catalog", {"command": "help"})["commands"][0]
        self.assertEqual(detailed["syntax"], ["help", "help <command>", "help help"])
        self.assertIn("Description:", detailed["help"])
        self.assertEqual(detailed["agent_policy"], {"default": "refuse", "first_argument": {}})

    def test_an_agent_reply_may_not_run_it_because_the_prompt_already_holds_every_commands_usage(self):
        self.assertEqual(AGENT_POLICY["help"]["default"], "refuse")
        for command in ("help", "help zoom", "HELP help"):
            self.assertEqual(agent_policy(command), "refuse")

    def test_the_vr_viewer_could_load_it_because_it_imports_no_qt(self):
        tree = ast.parse((SRC_DIR / "commands" / "help.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add((node.module or "").split(".")[0])
        self.assertTrue(imported <= {"re", "textwrap", "Command_Engine", "utilities", "desktop"}, imported)


if __name__ == "__main__":
    unittest.main()
