"""What a command module declares about how the Viewer dispatches it (Command_Declarations).

The engine, the command queue and the command portal never name a command: the
modules of src/commands say which run at once, which start a model request, which
act on jobs and which bring a Console button back when a session loads, and a
module that says nothing gets the cautious defaults.
The first class writes out the cases the modules declare, and the next ones show
that a plugin dropped into the commands package is dispatched by what it
declares with no change to the program. CentralCodeNamesNoCommandTests reads the
source of every file of central code (all of src but the commands, the tools and
the resources) and fails when one names a command, imports one or holds a table
of command names, apart from the few cases ALLOWED lists with their reasons.
tests/test_command_removal.py shows the same from the other side: the program
runs with any one command taken away.
"""
import ast
import collections
import inspect
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import Command_Declarations as declarations  # noqa: E402
import Command_Engine as ce  # noqa: E402
from Viewer_Command_Portal import ViewerCommandPortal  # noqa: E402
from desktop.Viewer_Inspection import command_catalog  # noqa: E402
from web_ui.Plugin_Manager import discover_plugin_descriptors  # noqa: E402
from tests.command_fixtures import only_commands, scratch_commands  # noqa: E402
from tests.viewer_fixtures import PortalFixture  # noqa: E402

COMMANDS_DIR = SRC_DIR / "commands"
COMMAND_NAMES = sorted(path.stem for path in COMMANDS_DIR.glob("*.py") if not path.stem.startswith("_"))
REFUSAL = "Model-generated agent commands cannot start another model request."
INSTANT, ORDERED = "instant", "ordered"
SOURCES = ("typed", "script", "portal")

ARGUMENT_LISTS = [
    [], ["help"], ["Help"], ["-h"], ["--help"], ["--HELP"], ["-?"], ["--register-only"], ["--register-only", "x"],
    ["stop"], ["STOP"], ["Stop"], ["stop", "large"], ["stop", "now"], ["large"], ["multi"], ["large", "multi"],
    ["off"], ["OFF"], ["deactivate"], ["Deactivate"], ["off", "now"], ["deactivate", "now"], ["help", "me"],
    ["<Model>"], ["<My", "Model>"], ["<x"], ["x>"], ["<>"], ["<", ">"],
    ["hello"], ["hello", "world"], ['"make', "cluster", "1", 'red"'], ["3"], ["0"], ["cancel"], ["cancel", "all"],
    ["cancel", "14"], ["upload", "a.csv"], ["import", "a.csv"], ["red"], ["{Length>10}"],
]
NAMES_TO_TRY = COMMAND_NAMES + ["nosuchcommand", "help"]


class DeclaredCasesTests(unittest.TestCase):
    """The cases the modules declare, written out."""

    def test_queue(self):
        cases = [
            # (command, args, sources it runs at once from, the others wait)
            ("jobs", [], SOURCES), ("jobs", ["cancel", "all"], SOURCES),
            ("zoom", ["200"], SOURCES), ("run", [], SOURCES), ("run", ["stop"], SOURCES),
            ("esmfold", ["stop"], SOURCES), ("esmfold", ["STOP", "large"], SOURCES),
            ("esmfold", [], ()), ("esmfold", ["large"], ()), ("esmfold", ["multi"], ()),
            ("undo", [], ("typed",)), ("undo", ["3"], ("typed",)), ("redo", [], ("typed",)),
            ("meta", [], SOURCES), ("meta", ["upload", "a.csv"], ()),
            ("agent", [], SOURCES), ("agent", ["off"], SOURCES), ("agent", ["Deactivate"], SOURCES),
            ("agent", ["<My", "Model>"], SOURCES), ("agent", ["hello", "there"], ()),
            ("select", [], ()), ("color", ["red"], ()), ("cluster", [], ()), ("nosuchcommand", [], ()),
        ]
        for name, args, instant_from in cases:
            for source in SOURCES:
                with self.subTest(command=name, args=args, source=source):
                    expected = INSTANT if source in instant_from else ORDERED
                    self.assertEqual(ce.queue_kind(name, args, source), expected)
                    # The dispatch hands over the module it has just loaded; it decides the same.
                    module = declarations.command_module(name)
                    self.assertEqual(ce.queue_kind(name, args, source, module), expected)

    def test_help_and_registration_run_at_once_for_every_command(self):
        for name in NAMES_TO_TRY:
            for word in ("help", "-h", "--help", "-?", "--register-only", "HELP"):
                for source in SOURCES:
                    with self.subTest(command=name, word=word, source=source):
                        self.assertEqual(ce.queue_kind(name, [word], source), INSTANT)

    def test_model_requests(self):
        for args, expected in [
                ([], False), (["off"], False), (["Deactivate"], False), (["help"], False), (["--HELP"], False),
                (["-h"], False), (["--register-only"], False), (["<Gemini", "API>"], False),
                (["hello"], True), (["hello", "there"], True), (["-?"], True), (["help", "me"], True),
                (["off", "now"], True), (["--register-only", "x"], True), (["<unclosed"], True)]:
            with self.subTest(args=args):
                self.assertIs(declarations.starts_model_request("agent", args), expected)
        for name in COMMAND_NAMES:
            if name != "agent":
                with self.subTest(command=name):
                    self.assertIs(declarations.starts_model_request(name, ["hello", "world"]), False)

    def test_commands_that_act_on_jobs(self):
        for name, args, expected in [
                ("jobs", [], True), ("jobs", ["cancel"], True), ("JOBS", ["cancel", "all"], False),
                ("esmfold", ["stop"], True), ("esmfold", ["STOP"], True), ("esmfold", ["stop", "large"], False),
                ("esmfold", [], False), ("run", ["stop"], True), ("run", [], False), ("run", ["stop", "x"], False),
                ("select", ["stop"], False), ("undo", [], False), ("nosuchcommand", ["stop"], False)]:
            with self.subTest(command=name, args=args):
                self.assertIs(declarations.acts_on_jobs(name, args), expected)

    def test_a_request_of_stop_commands_runs_ahead_of_the_head_unless_it_awaits_review(self):
        for command, expected in [
                ("esmfold stop", True), ("  ESMFOLD   stop  ", True), ("run stop", True), ("Run Stop", True),
                ("jobs", True), ("JOBS cancel all", True),
                ("esmfold stop large", False), ("esmfold", False), ("run", False), ("select stop", False),
                ("undo", False), ("nosuchcommand stop", False), ("", False), ("   ", False)]:
            with self.subTest(command=command):
                self.assertIs(ViewerCommandPortal._is_stop({"command": command}), expected)
        for approval, expected in ((None, True), ("approved", True), ("pending", False), ("refused", False),
                                   ("discarded", False)):
            with self.subTest(approval=approval):
                self.assertIs(ViewerCommandPortal._is_stop({"command": "esmfold stop", "approval": approval}),
                              expected)

    def test_every_declaration_of_the_real_commands_is_well_formed(self):
        # A typo in a declaration would silently fall back to the default, so the real modules are read here.
        for name in COMMAND_NAMES:
            module = declarations.command_module(name)
            self.assertIsNotNone(module, name)
            with self.subTest(command=name):
                if hasattr(module, "QUEUE"):
                    self.assertIn(module.QUEUE, (INSTANT, ORDERED))
                for constant in ("STARTS_MODEL_REQUEST", "ACTS_ON_JOBS", "CONSOLE_BUTTON"):
                    if hasattr(module, constant):
                        self.assertIsInstance(getattr(module, constant), bool, constant)
                for function, parameters, valid in (("queue_kind", ["args", "source"], {INSTANT, ORDERED}),
                                                    ("starts_model_request", ["args"], {True, False}),
                                                    ("acts_on_jobs", ["args"], {True, False})):
                    declared = getattr(module, function, None)
                    if declared is None:
                        continue
                    self.assertEqual(list(inspect.signature(declared).parameters), parameters, function)
                    for args in ARGUMENT_LISTS:
                        for source in SOURCES:
                            result = declared(list(args), source) if function == "queue_kind" else declared(list(args))
                            self.assertIn(result, valid, (function, args, source))
                            self.assertIs(type(result), type(next(iter(valid))), (function, args, source))

    def test_exactly_these_commands_declare_how_they_are_dispatched(self):
        declaring = {name for name in COMMAND_NAMES
                     if any(hasattr(declarations.command_module(name), attribute)
                            for attribute in ("QUEUE", "queue_kind", "STARTS_MODEL_REQUEST", "starts_model_request",
                                              "ACTS_ON_JOBS", "acts_on_jobs", "CONSOLE_BUTTON"))}
        self.assertEqual(declaring, {"jobs", "zoom", "run", "esmfold", "undo", "redo", "meta", "agent", "console", "help"})

    def test_exactly_these_commands_declare_a_console_button(self):
        # Only a command that declares one is run by name when a layout cache lists it: a cache cannot extend this.
        buttons = {name for name in COMMAND_NAMES if declarations.registers_console_button(name)}
        self.assertEqual(buttons, {"agent", "esmfold", "meta"})
        for name in COMMAND_NAMES:
            with self.subTest(command=name):
                self.assertIs(getattr(declarations.command_module(name), "CONSOLE_BUTTON", False), name in buttons)


# --- Throwaway plugins -----------------------------------------------------------------
RECORD = ("import Command_Engine\n"
          "def run(viewer, args):\n"
          "    Command_Engine.SCRATCH_CALLS.append((__name__.split('.')[-1], list(args)))\n"
          "    Command_Engine.command_succeeded(viewer)\n")
PLUGINS = {
    "zzdecl_plain": RECORD,
    "zzdecl_instant": "QUEUE = 'instant'\n" + RECORD,
    "zzdecl_ordered": "QUEUE = 'ordered'\n" + RECORD,
    "zzdecl_by_source": ("def queue_kind(args, source):\n"
                         "    return 'instant' if source == 'script' and args == ['now'] else 'ordered'\n" + RECORD),
    "zzdecl_function_wins": "QUEUE = 'ordered'\ndef queue_kind(args, source):\n    return 'instant'\n" + RECORD,
    "zzdecl_model": "STARTS_MODEL_REQUEST = True\n" + RECORD,
    "zzdecl_no_model": "STARTS_MODEL_REQUEST = False\n" + RECORD,
    "zzdecl_model_by_args": "def starts_model_request(args):\n    return args != ['info']\n" + RECORD,
    "zzdecl_jobs": "ACTS_ON_JOBS = True\n" + RECORD,
    "zzdecl_jobs_by_args": "def acts_on_jobs(args):\n    return args[:1] == ['halt']\n" + RECORD,
    "zzdecl_raises": ("def queue_kind(args, source):\n    raise RuntimeError('broken')\n"
                      "def starts_model_request(args):\n    raise RuntimeError('broken')\n"
                      "def acts_on_jobs(args):\n    raise RuntimeError('broken')\n" + RECORD),
    "zzdecl_wrong_kinds": ("QUEUE = 'Instant'\nSTARTS_MODEL_REQUEST = 'no'\nACTS_ON_JOBS = 'yes'\n" + RECORD),
    "zzdecl_wrong_results": ("def queue_kind(args, source):\n    return True\n"
                             "def starts_model_request(args):\n    return None\n"
                             "def acts_on_jobs(args):\n    return 1\n" + RECORD),
    "zzdecl_not_callable": ("queue_kind = 'instant'\nstarts_model_request = False\nacts_on_jobs = True\n" + RECORD),
    "zzdecl_mutates": ("def queue_kind(args, source):\n    args.clear()\n    return 'instant'\n"
                       "def starts_model_request(args):\n    args.append('x')\n    return False\n"
                       "def acts_on_jobs(args):\n    args.clear()\n    return False\n" + RECORD),
    "zzdecl_button": "CONSOLE_BUTTON = True\n" + RECORD,
    "zzdecl_no_button": "CONSOLE_BUTTON = False\n" + RECORD,
    "zzdecl_button_wrong_kinds": "CONSOLE_BUTTON = 'yes'\n" + RECORD,
    "zzdecl_button_one": "CONSOLE_BUTTON = 1\n" + RECORD,
    "zzdecl_button_function": "def console_button():\n    return True\n" + RECORD,
    "zzdecl_button_import_error": "CONSOLE_BUTTON = True\nraise RuntimeError('this plugin cannot be imported')\n",
    "_zzdecl_button_private": "CONSOLE_BUTTON = True\n" + RECORD,
    "zzdecl_import_error": "raise RuntimeError('this plugin cannot be imported')\n",
    "zzdecl_syntax_error": "def run(viewer, args:\n",
}


class DefaultsTests(unittest.TestCase):
    """A module that declares nothing, or cannot be imported, is dispatched cautiously."""

    def setUp(self):
        self.enterContext(scratch_commands(PLUGINS))

    def assertDefaults(self, name):
        for args in ([], ["stop"], ["hello", "world"]):
            for source in SOURCES:
                self.assertEqual(ce.queue_kind(name, list(args), source), ORDERED, (name, args, source))
            self.assertFalse(declarations.starts_model_request(name, list(args)), (name, args))
            self.assertFalse(declarations.acts_on_jobs(name, list(args)), (name, args))
        self.assertIs(declarations.registers_console_button(name), False, name)

    def test_an_undeclared_module_waits_its_turn_starts_no_model_request_and_acts_on_no_jobs(self):
        self.assertDefaults("zzdecl_plain")
        self.assertDefaults("select")

    def test_a_module_that_cannot_be_imported_declares_nothing(self):
        for name in ("zzdecl_import_error", "zzdecl_syntax_error", "nosuchcommand", "a.b", "", "..", "__init__x y"):
            with self.subTest(command=name):
                self.assertIsNone(declarations.command_module(name))
                self.assertDefaults(name)
        self.assertIsNone(declarations.command_module(None))
        self.assertIsNone(declarations.command_module(7))

    def test_the_declared_forms_say_what_they_say(self):
        self.assertEqual(ce.queue_kind("zzdecl_instant", [], "typed"), INSTANT)
        self.assertEqual(ce.queue_kind("zzdecl_ordered", [], "typed"), ORDERED)
        for source in SOURCES:
            self.assertEqual(ce.queue_kind("zzdecl_by_source", ["now"], source), INSTANT if source == "script" else ORDERED)
            self.assertEqual(ce.queue_kind("zzdecl_by_source", ["later"], source), ORDERED)
        self.assertTrue(declarations.starts_model_request("zzdecl_model", []))
        self.assertFalse(declarations.starts_model_request("zzdecl_no_model", ["hello"]))
        self.assertTrue(declarations.starts_model_request("zzdecl_model_by_args", ["hello"]))
        self.assertFalse(declarations.starts_model_request("zzdecl_model_by_args", ["info"]))
        self.assertTrue(declarations.acts_on_jobs("zzdecl_jobs", ["anything"]))
        self.assertTrue(declarations.acts_on_jobs("zzdecl_jobs_by_args", ["halt", "now"]))
        self.assertFalse(declarations.acts_on_jobs("zzdecl_jobs_by_args", ["go"]))

    def test_a_function_wins_over_the_constant_of_the_same_declaration(self):
        self.assertEqual(ce.queue_kind("zzdecl_function_wins", [], "portal"), INSTANT)

    def test_the_help_words_run_at_once_for_a_plugin_that_waits(self):
        for word in ("help", "--help", "-h", "-?", "--register-only"):
            self.assertEqual(ce.queue_kind("zzdecl_ordered", [word], "portal"), INSTANT)

    def test_a_declaration_that_raises_or_has_the_wrong_kind_of_value_is_not_obeyed(self):
        for name in ("zzdecl_raises", "zzdecl_wrong_kinds", "zzdecl_wrong_results", "zzdecl_not_callable"):
            for source in SOURCES:
                with self.subTest(command=name, source=source):
                    self.assertEqual(ce.queue_kind(name, ["x"], source), ORDERED)
                    self.assertFalse(declarations.acts_on_jobs(name, ["x"]))
                    # A model rule that cannot be read refuses the command instead of letting it through.
                    self.assertTrue(declarations.starts_model_request(name, ["x"]))

    def test_a_console_button_is_declared_by_the_constant_true_and_nothing_else(self):
        self.assertIs(declarations.registers_console_button("zzdecl_button"), True)
        for name in ("zzdecl_plain", "zzdecl_no_button", "zzdecl_button_wrong_kinds", "zzdecl_button_one",
                     "zzdecl_button_function", "zzdecl_button_import_error", "zzdecl_import_error",
                     "zzdecl_syntax_error", "nosuchcommand", "select", "run", "help", "console"):
            with self.subTest(command=name):
                self.assertIs(declarations.registers_console_button(name), False)

    def test_what_is_not_a_bare_command_name_has_no_console_button(self):
        # A name a layout cache lists may be anything. A module of the commands folder that starts with "_" is not a
        # command, so it is not imported by name even though it declares one.
        for name in ("zzdecl_button x", " zzdecl_button", "zzdecl_button ", "zzdecl_button --register-only",
                     "zzdecl_button\n", "commands.zzdecl_button", "zzdecl_button.py", "..", "", "a.b", "-h",
                     "_zzdecl_button_private", "__init__", "run C:/script.py"):
            with self.subTest(command=name):
                self.assertIs(declarations.registers_console_button(name), False)
        for name in (None, 7, 1.5, True, b"zzdecl_button", ["zzdecl_button"], ("zzdecl_button",),
                     {"zzdecl_button"}, {"zzdecl_button": 1}):
            with self.subTest(command=name):
                self.assertIs(declarations.registers_console_button(name), False)

    def test_a_declaration_cannot_change_the_arguments_that_are_dispatched(self):
        args = ["keep", "me"]
        self.assertEqual(ce.queue_kind("zzdecl_mutates", args, "typed"), INSTANT)
        self.assertFalse(declarations.starts_model_request("zzdecl_mutates", args))
        self.assertFalse(declarations.acts_on_jobs("zzdecl_mutates", args))
        self.assertEqual(args, ["keep", "me"])

    def test_the_module_the_caller_passes_is_the_one_asked(self):
        stand_in = type("Module", (), {"QUEUE": "instant"})
        self.assertEqual(ce.queue_kind("select", [], "typed", stand_in), INSTANT)
        self.assertEqual(ce.queue_kind("jobs", [], "typed", type("Module", (), {})), ORDERED)
        self.assertTrue(declarations.starts_model_request("select", [], type("M", (), {"STARTS_MODEL_REQUEST": True})))
        self.assertTrue(declarations.acts_on_jobs("select", [], type("M", (), {"ACTS_ON_JOBS": True})))


class FakeQueue:
    """A command queue that records what it is given and finishes each command at once."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.given = []

    def uses_module(self, command_name):
        return False

    def dispatch(self, module, command_name, args, line, source='typed', silent=False):
        self.given.append((command_name, list(args), source))
        ce.command_succeeded(self.viewer)


class EngineDispatchTests(PortalFixture, unittest.TestCase):
    """Command_Engine and the portal dispatch a dropped-in plugin by what it declares, with no change to either."""

    def setUp(self):
        super().setUp()
        self.enterContext(scratch_commands(PLUGINS))
        self.calls = []
        self.enterContext(mock.patch.object(ce, "SCRATCH_CALLS", self.calls, create=True))
        self.viewer.command_queue = self.queue = FakeQueue(self.viewer)

    def typed(self, line, source="typed"):
        if source == "script":
            with ce.typed_script():
                self.viewer.process_command(line, record_history=False)
        else:
            self.viewer.process_command(line, record_history=False)

    def test_a_plugin_that_declares_instant_runs_at_once_while_others_wait(self):
        self.typed("zzdecl_instant")
        self.assertEqual((self.calls, self.queue.given), ([("zzdecl_instant", [])], []))
        self.typed("zzdecl_plain 1")
        self.typed("zzdecl_ordered 2")
        self.assertEqual(self.calls, [("zzdecl_instant", [])])
        self.assertEqual(self.queue.given, [("zzdecl_plain", ["1"], "typed"), ("zzdecl_ordered", ["2"], "typed")])

    def test_a_plugin_that_decides_by_source_and_arguments(self):
        self.typed("zzdecl_by_source now", "script")
        self.typed("zzdecl_by_source now", "typed")
        self.typed("zzdecl_by_source later", "script")
        self.assertEqual(self.calls, [("zzdecl_by_source", ["now"])])
        self.assertEqual(self.queue.given, [("zzdecl_by_source", ["now"], "typed"), ("zzdecl_by_source", ["later"], "script")])
        result = self.finish(self.portal.submit("by-source", "zzdecl_by_source now")["request_id"])
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(self.queue.given[-1], ("zzdecl_by_source", ["now"], "portal"))
        self.assertEqual(len(self.calls), 1)

    def test_a_real_command_that_declares_nothing_or_waits_is_given_to_the_queue(self):
        # The real run() is not reached: the queue is a stand-in that records what it is given.
        for line, source in (("reset", "typed"), ("select", "script"), ("undo", "script"), ("undo 2", "script"),
                             ("meta upload a.csv", "typed"), ("esmfold large", "typed"), ("agent hello", "typed")):
            before = len(self.queue.given)
            self.typed(line, source)
            with self.subTest(line=line, source=source):
                self.assertEqual(len(self.queue.given) - before, 1, self.queue.given)
                self.assertEqual(self.queue.given[-1][0], line.split()[0])

    def test_a_plugin_that_starts_a_model_request_is_refused_from_the_portal(self):
        for line in ("zzdecl_model", "zzdecl_model hello", "zzdecl_model_by_args hello", "zzdecl_raises x",
                     "zzdecl_wrong_kinds x", "ZZDECL_MODEL hello"):
            with self.subTest(command=line):
                result = self.finish(self.portal.submit(line, line)["request_id"])
                self.assertEqual(result["status"], "failed", result)
                self.assertEqual(result["commands"][0]["messages"][-1]["text"], REFUSAL)
        self.assertEqual((self.calls, self.queue.given), ([], []))

    def test_a_plugin_that_does_not_start_one_runs_from_the_portal(self):
        for line in ("zzdecl_model_by_args info", "zzdecl_no_model hello", "zzdecl_plain hello"):
            with self.subTest(command=line):
                result = self.finish(self.portal.submit(line, line)["request_id"])
                self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(len(self.queue.given), 3)

    def test_typing_a_command_that_starts_a_model_request_is_allowed(self):
        self.typed("zzdecl_model hello")
        self.assertEqual(self.queue.given, [("zzdecl_model", ["hello"], "typed")])

    def test_the_agent_command_is_still_refused_when_it_would_message_a_model(self):
        for line in ("agent hello", "agent\thello", "AGENT make cluster 1 red", "agent -?", "agent help me"):
            with self.subTest(command=line):
                result = self.finish(self.portal.submit(line, line)["request_id"])
                self.assertEqual(result["status"], "failed", result)
                self.assertEqual(result["commands"][0]["messages"][-1]["text"], REFUSAL)
        self.assertEqual(self.queue.given, [])

    def test_a_refused_model_request_is_not_kept_in_the_history(self):
        self.finish(self.portal.submit("history", "zzdecl_model hello")["request_id"])
        self.assertEqual(self.viewer.command_history, [])


AGENT_PLUGINS = {
    # Declares nothing about the Agent.
    "zagentless": "def run(viewer, args):\n    return None\n",
    # Declares its policy, where it goes in the Agent's instructions and the text of its item.
    "zagentful": '''AGENT_POLICY = {'default': 'run', 'first_argument': {'wipe': 'approve', 'nuke': 'refuse'}}
AGENT_PROMPT_ORDER = 25
AGENT_PROMPT = """
`zagentful [wipe]`
- Does the declared thing.
"""


def run(viewer, args):
    return None
''',
}


class AgentDeclarationsOfAPluginTests(PortalFixture, unittest.TestCase):
    """A plugin dropped into the commands package is reviewed and described for the Agent by what it declares."""

    def test_the_portal_reviews_an_agent_reply_by_the_policy_the_plugin_declares(self):
        with scratch_commands(AGENT_PLUGINS):
            submitted = self.portal.submit("agent plugin", ["zagentful", "zagentful wipe", "ZAGENTFUL Nuke", "zagentless x"],
                                           source="web_agent")
        self.assertEqual([record["approval"] for record in submitted["commands"]], [None, "pending", "refused", "pending"])

    def test_without_the_plugin_the_same_commands_wait_for_approval(self):
        # A command no module declares is unknown: it waits, as a command added without a policy does.
        submitted = self.portal.submit("agent gone", ["zagentful", "zagentful wipe"], source="web_agent")
        self.assertEqual([record["approval"] for record in submitted["commands"]], ["pending", "pending"])

    def test_the_agents_instructions_describe_the_plugin_and_its_review(self):
        from web_ui.agent_prompt import render_system_prompt

        template = (SRC_DIR / "resources" / "agent" / "system_prompt.md").read_text(encoding="utf-8")
        with scratch_commands(AGENT_PLUGINS):
            text = render_system_prompt(template)
        self.assertIn("\n3. `zagentful [wipe]`\n   - Does the declared thing.\n", text)
        self.assertIn("\n27. `zagentless`\n    - The zagentless command; its module declares no description.\n", text)
        self.assertIn("`zagentful` waits for the user to press Run only with the first word `wipe`", text)
        self.assertIn("; is refused only with the first word `nuke`; any other form runs at once.", text)
        self.assertNotIn("zagent", render_system_prompt(template))


# --- No central file names a command -----------------------------------------------------
# Central code is every .py file of src except the commands themselves, the tools (scripts run from a terminal) and
# the bundled resources. It may not name a command. Four rules read its source (command_words_in):
#   1. a string constant that is a command's name (a name that is also an ordinary word counts only where
#      a command is meant, rule 4);
#   2. an import of a command module, in any form: from, import, an attribute of the package, a module path in a
#      string such as importlib.import_module('commands.zoom');
#   3. a dict, set, list or tuple literal, up to two levels deep, that holds two or more command names: a table;
#   4. the names that are ordinary words (AMBIGUOUS_NAMES) count only as the first argument of a call that takes a
#      command's name or line (LINE_CALLS, NAME_CALLS), or compared with a variable called command* or cmd*, or in a
#      table (rule 3); every other name counts wherever a string equals it.
# What a file may hold anyway is in ALLOWED, with the reason. The files that dispatch commands (DISPATCH_FILES) are
# read more strictly: there an ordinary word counts wherever it stands.
NOT_CENTRAL = ("commands", "tools", "resources")
# The names of commands that are ordinary words as well: nouns of the network and the alignment, and the verbs of what
# is done to them. "help" is the word every command answers (the engine handles it for all of them, as it does -h,
# --help and -?), so it counts only where a command is meant, even in the files that dispatch.
AMBIGUOUS_NAMES = frozenset({"alignment", "embedding", "cluster", "group", "label", "color", "hide", "offset", "reference",
                             "select", "reset", "print", "save", "export", "query", "help"})
# The calls that take a command's line, and those that take a command's name, as their first argument.
LINE_CALLS = frozenset({"process_command", "execute_command", "agent_policy"})
NAME_CALLS = frozenset({"queue_kind", "command_module", "starts_model_request", "acts_on_jobs", "registers_console_button",
                        "get_command_metadata", "import_module"})
COMMAND_VARIABLE = re.compile(r"^(command|cmd)", re.IGNORECASE)
COMMAND_MODULE_PATH = re.compile(r"^commands\.([A-Za-z_]\w*)")
CONTAINERS = (ast.Dict, ast.List, ast.Tuple, ast.Set)
# The files that dispatch, queue, run, review and describe commands.
DISPATCH_FILES = ("src/Command_Engine.py", "src/Viewer_Command_Portal.py", "src/Command_Queue.py",
                  "src/Background_Job_Scheduler.py", "src/Viewer_Worker_Tracking.py", "src/Command_Declarations.py",
                  "src/desktop/Command_Metadata.py", "src/web_ui/agent_prompt.py")

SELECTION_GRAMMAR = "the selection grammar's own vocabulary: what a #label# or an amino-acid target is read from"
RESET_DELEGATE = ("the one place the engine hands a reset to the reset command (commands/reset.py reads each module's "
                  "RESET_TARGETS and does the rest); the other commands call this and the VR viewer aliases it")
JOBS_KEY = "the 'jobs' entry of a command request's record"
ENTRY_POINT = "'run', the entry point every command module has: hasattr(module, 'run')"
ALIGNMENT_SETTINGS = "the keys of an alignment's settings (its reference and its offset), nouns that are not commands"
LABEL_KINDS = "the two kinds of label a node can have (a cluster and a group), nouns that are not commands"
ALLOWED = {
    "src/Command_Engine.py": {
        ("resolve_label_target", "string", "cluster"): (1, SELECTION_GRAMMAR),
        ("resolve_label_target", "string", "group"): (1, SELECTION_GRAMMAR),
        ("get_alignment_mapping", "string", "alignment"): (1, SELECTION_GRAMMAR),
        ("get_embedding_mapping", "string", "embedding"): (1, SELECTION_GRAMMAR),
        ("embedding_unavailable_message", "string", "embedding"): (1, SELECTION_GRAMMAR),
        ("evaluate_label_mask", "string", "cluster"): (1, SELECTION_GRAMMAR),
        ("_unterminated_target_message", "string", "label"): (1, SELECTION_GRAMMAR),
        ("_SelectionSyntaxParser._parse_primary", "string", "label"): (1, SELECTION_GRAMMAR),
        ("_evaluate_selection_node", "string", "label"): (1, SELECTION_GRAMMAR),
        ("execute_reset", "import", "reset"): (1, RESET_DELEGATE),
        ("_dispatch_user_command", "string", "run"): (1, ENTRY_POINT),
    },
    "src/Viewer_Command_Portal.py": {
        ("<module>", "string", "run"): (1, "a value of APPROVAL, the review's verdict (run at once), not a command"),
        ("agent_page_connected", "string", "agent"): (1, "the client id of the Agent web page, which is a web plugin's id"),
        ("ExecutionContext.add_job", "string", "jobs"): (1, JOBS_KEY),
        ("ExecutionContext.job_event", "string", "jobs"): (1, JOBS_KEY),
        ("ExecutionContext.job_progress", "string", "jobs"): (1, JOBS_KEY),
        ("ViewerCommandPortal._advance", "string", "jobs"): (1, JOBS_KEY),
        ("ViewerCommandPortal._has_work_left", "string", "jobs"): (1, JOBS_KEY),
        ("ViewerCommandPortal._frees_the_line", "string", "jobs"): (1, JOBS_KEY),
        ("ViewerCommandPortal.read_output", "string", "offset"): (1, "a key of an output page"),
    },
    "src/Background_Job_Scheduler.py": {
        ("BackgroundJobScheduler._update_records", "string", "jobs"): (1, JOBS_KEY),
        ("BackgroundJobScheduler._report_succeeded", "string", "jobs"): (1, JOBS_KEY),
    },
    "src/Viewer_Worker_Tracking.py": {
        ("WorkerTracker.poll", "string", "jobs"): (1, JOBS_KEY),
    },
    "src/desktop/Command_Metadata.py": {
        ("<module>", "table", "alignment, embedding"): (1, "REQUIRABLE_INPUTS: the optional inputs a command can need, not commands"),
        ("<module>", "string", "run"): (1, "RUN, a class of the Agent policy: a reply's command runs at once"),
        ("read_declarations", "string", "run"): (1, ENTRY_POINT),
    },
    "src/desktop/Viewer_Inspection.py": {
        ("command_catalog_entry", "string", "run"): (1, ENTRY_POINT + ", read from the module's source"),
        ("SnapshotStore._execute", "table", "cluster, group"): (1, LABEL_KINDS),
        ("SnapshotStore._execute", "table", "offset, reference"): (1, ALIGNMENT_SETTINGS),
        ("SnapshotStore.capture", "table", "alignment, offset, reference"): (1, ALIGNMENT_SETTINGS),
    },
    "src/desktop/Alignment_Snapshot.py": {
        ("freeze_alignment", "table", "offset, reference"): (1, ALIGNMENT_SETTINGS),
    },
    "src/desktop/Viewer_State.py": {
        ("<module>", "table", "alignment, embedding"): (1, "VIEWER_SECTIONS: the sections of a Viewer settings document, "
                                                           "named for the files they set"),
    },
    "src/mcp_server/viewer/Viewer_Operations.py": {
        ("_wait_line", "string", "jobs"): (1, JOBS_KEY),
        ("get_residue_distribution", "table", "cluster, group"): (1, "group_by's choices: " + LABEL_KINDS),
    },
    "src/web_ui/Page_Texts.py": {
        ("<module>", "string", "meta"): (1, "HTML's <meta> element, which has no end tag: the name of a tag, not of a command"),
    },
    "src/web_ui/agent_backend.py": {
        ("handle_decide_agent_command", "string", "run"): (1, "the 'run' field of the Agent page's message: the user's decision on a command"),
    },
    "src/web_ui/esmfold_backend.py": {
        ("handle_esmfold_job", "string", "run"): (2, "the 'run' field of the Fold View's messages: the id of a fold run"),
        ("handle_open_log", "string", "run"): (1, "the 'run' field of the Fold View's messages: the id of a fold run"),
        ("handle_structure_folded", "string", "run"): (1, "the 'run' field of the Fold View's messages: the id of a fold run"),
    },
    "src/web_ui/residue_layers.py": {
        ("plddt_layer", "table", "color, label"): (1, "the keys of a legend band (its colour and its label), nouns that are not commands"),
    },
}


def central_files(src_dir=SRC_DIR):
    """Every .py file of central code under ``src_dir``: all but the commands, the tools and the resources."""
    return sorted(path for path in Path(src_dir).rglob("*.py") if path.relative_to(src_dir).parts[0] not in NOT_CENTRAL)


def plugin_pairs():
    """{name: relative paths of its two files} for each web plugin whose id is also a command's name.

    A page and its command are one plugin pair (the Agent page and the agent command, ...): the page's manifest and
    its backend say the name as the id of the page, and the backend reaches its own command. Read from the manifests
    on disk, so the pair of a plugin added later is covered without a change here.
    """
    descriptors, diagnostics = discover_plugin_descriptors(str(SRC_DIR / "web_ui" / "plugins"))
    assert not diagnostics, diagnostics
    pairs = {}
    for descriptor in descriptors:
        if descriptor.plugin_id in COMMAND_NAMES:
            backend = Path(*descriptor.backend.split(".")).with_suffix(".py")
            pairs[descriptor.plugin_id] = (Path(descriptor.source_path).resolve().relative_to(SRC_DIR.parent).as_posix(),
                                           (SRC_DIR / backend).relative_to(SRC_DIR.parent).as_posix())
    return pairs


def _held_strings(node, depth):
    """The string constants a dict, list, tuple or set literal holds, and those of the literals inside it, ``depth`` levels."""
    children = ([child for child in node.keys if child is not None] + list(node.values)
                if isinstance(node, ast.Dict) else list(node.elts))
    held = []
    for child in children:
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            held.append(child)
        elif depth > 1 and isinstance(child, CONTAINERS):
            held.extend(_held_strings(child, depth - 1))
    return held


def _call_name(call):
    function = call.func
    return function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else None


def _leading_string(node):
    """The string constant an expression that builds a string starts with: 'a', f'a{x}', 'a' + x, 'a %s' % x."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return node.values[0]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return _leading_string(node.left)
    return None


def _is_command_variable(node):
    while (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
           and node.func.attr in ("lower", "upper", "strip", "casefold")):
        node = node.func.value
    return bool(COMMAND_VARIABLE.match(node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""))


def _constants_of(node):
    """The string constants of a comparand: one, or the elements of a tuple, list or set of them."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [element for element in node.elts if isinstance(element, ast.Constant) and isinstance(element.value, str)]
    return []


def command_words_in(source, names, strict=False, exempt=frozenset()):
    """Counter of (enclosing scope, 'string' | 'import' | 'table', word) for the code of a module's source.

    'string': a string constant that names a command (docstrings and the generic help words are not counted; a name
    that is an ordinary word counts only where a command is meant, or anywhere when ``strict``). 'import': an import
    of a command module (from commands import x, from commands.x import y, import commands.x, commands.x, and a
    module path in a string, 'commands.x'). 'table': a dict, set, list or tuple literal that holds two or more
    command names, up to two levels deep, the names it holds then counted once, in it, and the word is those names.
    ``exempt`` names are not counted at all: the pair of a page and its command, in the page's own files.
    """
    names = set(names) - set(exempt)
    counted = names - {"help"}
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))

    def scope(node):
        parts = []
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                parts.append(node.name)
        return ".".join(reversed(parts)) or "<module>"

    def names_a_command(node):
        return (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
                and node.value in counted)

    found = collections.Counter()

    # Tables, outermost first: a literal that holds two names takes them, and a literal inside it only what is left.
    in_table = set()

    def find_tables(node):
        if isinstance(node, CONTAINERS):
            held = [constant for constant in _held_strings(node, 2) if names_a_command(constant) and id(constant) not in in_table]
            if len({constant.value for constant in held}) >= 2:
                found[(scope(node), "table", ", ".join(sorted({constant.value for constant in held})))] += 1
                in_table.update(id(constant) for constant in held)
        elif isinstance(node, ast.Call) and _call_name(node) == "dict":
            keywords = {keyword.arg for keyword in node.keywords if keyword.arg in counted}
            if len(keywords) >= 2:
                found[(scope(node), "table", ", ".join(sorted(keywords)))] += 1
        for child in ast.iter_child_nodes(node):
            find_tables(child)

    find_tables(tree)

    # Where a command is meant: {id of the string constant: the name it holds}.
    meant = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and _call_name(node) in LINE_CALLS | NAME_CALLS:
            constant = _leading_string(node.args[0])
            if constant is not None:
                word = (constant.value.split() or [""])[0].lower() if _call_name(node) in LINE_CALLS else constant.value
                if word in names:
                    meant[id(constant)] = word
        elif isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(_is_command_variable(side) for side in sides):
                for side in sides:
                    for constant in _constants_of(side):
                        if constant.value in names:
                            meant[id(constant)] = constant.value

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if id(node) in in_table:
                continue
            module = COMMAND_MODULE_PATH.match(node.value)
            if module and module.group(1) not in exempt:
                found[(scope(node), "import", module.group(1))] += 1
            word = meant.get(id(node))
            if word is None and node.value in counted and (strict or node.value not in AMBIGUOUS_NAMES):
                word = node.value
            if word is not None:
                found[(scope(node), "string", word)] += 1
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")
            if module == ["commands"]:
                found.update((scope(node), "import", alias.name) for alias in node.names if alias.name not in exempt)
            elif module[0] == "commands" and len(module) > 1 and module[1] not in exempt:
                found[(scope(node), "import", module[1])] += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "commands" and len(parts) > 1 and parts[1] not in exempt:
                    found[(scope(node), "import", parts[1])] += 1
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "commands"
              and node.attr in names):
            found[(scope(node), "import", node.attr)] += 1
    return found


def central_findings(src_dir=SRC_DIR):
    """{relative path: the command_words_in of the file} for every central file, and the paths read, in order."""
    pairs = plugin_pairs()
    names = set(COMMAND_NAMES)
    findings, scanned = {}, []
    for path in central_files(src_dir):
        relative = path.relative_to(Path(src_dir).parent).as_posix()
        exempt = {name for name, files in pairs.items() if relative in files}
        scanned.append(relative)
        findings[relative] = command_words_in(path.read_text(encoding="utf-8"), names, relative in DISPATCH_FILES, exempt)
    return findings, scanned


class CentralCodeNamesNoCommandTests(unittest.TestCase):
    """No file of central code names a command, imports one or holds a table of them (the rules above)."""

    NAMES = {"jobs", "zoom", "run", "undo", "agent", "redo", "meta", "esmfold", "select", "reset", "cluster", "group",
             "label", "alignment", "offset", "hide", "color", "print", "save", "export", "help", "embedding"}

    def scan(self, source, **options):
        return command_words_in(source, self.NAMES, **options)

    def test_the_scanner_finds_a_command_named_in_code_and_an_import_of_one_in_every_form(self):
        # Positive controls: without them a scanner that finds nothing would pass every file.
        source = ("from commands import zoom\n"
                  "import commands.run\n"
                  "from commands.undo import steps\n"
                  "'''docstring jobs'''\n"
                  "def kind(name, args):\n"
                  "    '''A docstring that says jobs.'''\n"
                  "    if name in ('jobs', 'zoom'):\n"
                  "        return 'instant'\n"
                  "    if name == 'agent' and args == ['--help', '-?']:\n"
                  "        return commands.redo\n"
                  "    return 'a jobs sentence'\n"
                  "def load(name):\n"
                  "    importlib.import_module('commands.meta')\n"
                  "    importlib.import_module(f'commands.{name}')\n"
                  "    __import__('commands.esmfold')\n"
                  "    return sys.modules['commands.reset'], sys.modules.get('commands.' + name)\n")
        self.assertEqual(self.scan(source), collections.Counter({
            ("<module>", "import", "zoom"): 1, ("<module>", "import", "run"): 1, ("<module>", "import", "undo"): 1,
            ("kind", "table", "jobs, zoom"): 1, ("kind", "string", "agent"): 1, ("kind", "import", "redo"): 1,
            ("load", "import", "meta"): 1, ("load", "import", "esmfold"): 1, ("load", "import", "reset"): 1}))

    def test_a_name_that_is_an_ordinary_word_counts_only_where_a_command_is_meant(self):
        source = ("def f(command, cmd_name, words, viewer, name):\n"
                  "    word = 'select'\n"                                    # a noun
                  "    print('save', 'export')\n"                            # two arguments, no table
                  "    if words[0] == 'reset' or name == 'label':\n"         # not a variable of a command
                  "        a = ['cluster', 'x']\n"                           # one name in a literal is no table
                  "        b = {'group': 1}\n"
                  "    if command == 'cluster':\n"
                  "        pass\n"
                  "    if cmd_name.lower() in ('hide', 'x'):\n"
                  "        pass\n"
                  "    viewer.process_command('color red')\n"                # a line: its first word
                  "    viewer.process_command(f'group {words}')\n"
                  "    viewer.process_command('select ' + words)\n"
                  "    viewer.process_command(words)\n"
                  "    queue_kind('print', [], 'typed')\n"                   # a name
                  "    Command_Declarations.command_module('help')\n"
                  "    return agent_policy('Export 1')\n")
        self.assertEqual(self.scan(source), collections.Counter({
            ("f", "string", "cluster"): 1, ("f", "string", "hide"): 1, ("f", "string", "color"): 1,
            ("f", "string", "group"): 1, ("f", "string", "select"): 1, ("f", "string", "print"): 1,
            ("f", "string", "help"): 1, ("f", "string", "export"): 1}))
        # In the files that dispatch, an ordinary word counts wherever it stands (but not "help", the generic word).
        self.assertEqual(self.scan("x = ['select', 'help']\ny = 'label'\n", strict=True), collections.Counter({
            ("<module>", "string", "select"): 1, ("<module>", "string", "label"): 1}))
        # A name that is no ordinary word counts wherever a string equals it, and only then.
        self.assertEqual(self.scan("x = 'zoom'\ny = 'zoom 200'\nz = 'a zoom'\n"), collections.Counter({
            ("<module>", "string", "zoom"): 1}))

    def test_a_table_is_a_dict_set_list_or_tuple_of_two_names_up_to_two_levels_deep(self):
        source = ("A = {'alignment': 1, 'offset': 2}\n"
                  "B = ['cluster', 'group', 'other']\n"
                  "C = ('select', 'x')\n"
                  "D = {'a': ('hide', 'x'), 'b': {'color', 'y'}}\n"           # the names are one level down
                  "E = [[['print', 'save']]]\n"                                # the inner literal is two levels deep
                  "F = dict(export=1, label=2, other=3)\n"
                  "G = {'meta': 1}\n"
                  "def f():\n"
                  "    return frozenset({'embedding', 'reset'})\n")
        self.assertEqual(self.scan(source), collections.Counter({
            ("<module>", "table", "alignment, offset"): 1, ("<module>", "table", "cluster, group"): 1,
            ("<module>", "table", "color, hide"): 1, ("<module>", "table", "print, save"): 1,
            ("<module>", "table", "export, label"): 1, ("<module>", "string", "meta"): 1,
            ("f", "table", "embedding, reset"): 1}))
        # The names of a table are counted once, in it, not again as strings.
        self.assertEqual(self.scan("T = {'zoom': 'undo', 'run': ['agent']}\n"), collections.Counter({
            ("<module>", "table", "agent, run, undo, zoom"): 1}))

    def test_a_pair_name_is_not_counted_in_its_own_files(self):
        source = ("from commands import esmfold\n"
                  "import commands.esmfold as pair\n"
                  "ID = 'esmfold'\n"
                  "LIST = {'esmfold': 1, 'label': 2}\n"
                  "OTHER = 'zoom'\n"
                  "def f():\n"
                  "    return importlib.import_module('commands.esmfold'), commands.esmfold, 'commands.undo'\n")
        self.assertEqual(self.scan(source, exempt={"esmfold"}), collections.Counter({
            ("<module>", "string", "zoom"): 1, ("f", "import", "undo"): 1}))
        self.assertEqual(self.scan(source), collections.Counter({
            ("<module>", "import", "esmfold"): 2, ("<module>", "string", "esmfold"): 1,
            ("<module>", "table", "esmfold, label"): 1, ("<module>", "string", "zoom"): 1,
            ("f", "import", "esmfold"): 2, ("f", "import", "undo"): 1}))

    def test_the_central_files_are_all_of_src_but_the_commands_the_tools_and_the_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ("a.py", "pkg/b.py", "pkg/sub/c.py", "commands/d.py", "tools/e.py", "resources/languages/f.py",
                             "pkg/commands/g.py", "pkg/readme.md"):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text("x = 1\n", encoding="utf-8")
            # Only the top folders are not central: a folder of that name further down is.
            self.assertEqual([path.relative_to(root).as_posix() for path in central_files(root)],
                             ["a.py", "pkg/b.py", "pkg/commands/g.py", "pkg/sub/c.py"])

    def test_the_scan_reads_every_file_of_src_but_the_commands_the_tools_and_the_resources(self):
        # The list is made again here, by walking the folders, so a scan that left a file out would differ from it.
        expected = set()
        for folder, _dirs, files in os.walk(SRC_DIR):
            relative = Path(folder).relative_to(SRC_DIR)
            if relative.parts[:1] in (("commands",), ("tools",), ("resources",)):
                continue
            expected |= {(Path("src") / relative / name).as_posix() for name in files if name.endswith(".py")}
        findings, scanned = central_findings()
        self.assertEqual(len(scanned), len(set(scanned)))
        self.assertEqual(set(scanned), expected)
        self.assertEqual(set(findings), expected)
        self.assertTrue(all(path.startswith("src/") for path in scanned))   # nothing of opt_vr
        for path in ("src/EMAPSSN_Viewer.py", "src/Command_Engine.py", "src/desktop/Viewer_Widgets.py",
                     "src/web_ui/Web_Server.py", "src/mcp_server/viewer/Viewer_Operations.py",
                     "src/utilities/Localization.py", "src/Layout_Engine_SSN.py"):
            self.assertIn(path, expected)
        for path in ("src/commands/zoom.py", "src/tools/Sanitize_FASTA.py", "src/resources/languages/Docs_Translation.py"):
            self.assertNotIn(path, expected)
        self.assertEqual(NOT_CENTRAL, ("commands", "tools", "resources"))
        self.assertTrue(set(DISPATCH_FILES) <= expected)

    def test_the_command_names_are_the_modules_of_src_commands(self):
        # A command is a module of src/commands that does not start with "_" and defines run: the rule of the
        # Viewer's command catalog, which lists the folder, and of the dispatch, which imports commands.<name>.
        self.assertTrue({"agent", "jobs", "run", "esmfold", "undo", "redo", "meta", "zoom"} <= set(COMMAND_NAMES))
        self.assertEqual(set(COMMAND_NAMES), {entry["command"] for entry in command_catalog()["commands"]})
        for name in COMMAND_NAMES:
            with self.subTest(command=name):
                self.assertTrue(callable(getattr(declarations.command_module(name), "run", None)), name)
        # The names that are ordinary words are commands; one that stopped being a command would be a stale entry.
        self.assertTrue(AMBIGUOUS_NAMES <= set(COMMAND_NAMES), AMBIGUOUS_NAMES - set(COMMAND_NAMES))

    def test_the_pairs_of_a_page_and_its_command_are_these(self):
        # A new plugin whose page and command share a name is exempt in its own files; it is added here on purpose.
        pairs = plugin_pairs()
        self.assertEqual(set(pairs), {"agent", "esmfold", "meta"})
        for name, files in pairs.items():
            self.assertEqual(files, (f"src/web_ui/plugins/{name}.py", f"src/web_ui/{name}_backend.py"))
            for relative in files:
                self.assertTrue((SRC_DIR.parent / relative).is_file(), relative)

    def test_no_central_file_names_a_command_imports_one_or_holds_a_table_of_them(self):
        findings, scanned = central_findings()
        self.assertTrue(set(ALLOWED) <= set(scanned), set(ALLOWED) - set(scanned))
        for relative in sorted({relative for relative, found in findings.items() if found} | set(ALLOWED)):
            expected = collections.Counter({key: count for key, (count, _reason) in ALLOWED.get(relative, {}).items()})
            with self.subTest(file=relative):
                named = findings[relative] - expected
                gone = expected - findings[relative]
                self.assertFalse(named, f"{relative} names a command, imports one or holds a table of them: {dict(named)}. "
                                        "A command is dispatched, described and reset by what its module declares "
                                        "(Command_Declarations, desktop.Command_Metadata, commands/reset.py), not by name here.")
                self.assertFalse(gone, f"{relative} no longer holds {dict(gone)}: remove these entries from ALLOWED.")

    def test_every_allowed_entry_has_a_reason(self):
        for relative, allowed in ALLOWED.items():
            for key, (count, reason) in allowed.items():
                with self.subTest(file=relative, entry=key):
                    self.assertGreaterEqual(count, 1)
                    self.assertGreater(len(reason.split()), 3)
                    self.assertIn(key[1], ("string", "import", "table"))

    def test_the_viewers_console_button_code_names_no_command(self):
        # The Viewer brings back the Console buttons a session had by asking each command's module (CONSOLE_BUTTON);
        # these three functions of EMAPSSN_Viewer.py are read as strictly as the files that dispatch.
        import textwrap

        source = (SRC_DIR / "EMAPSSN_Viewer.py").read_text(encoding="utf-8")
        wanted = {"_register_persisted_console_buttons", "_custom_dataset_problem", "_quoted_for_log"}
        found = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                found.add(node.name)
                text = textwrap.dedent(" " * node.col_offset + ast.get_source_segment(source, node))
                with self.subTest(function=node.name):
                    self.assertEqual(command_words_in(text, set(COMMAND_NAMES), strict=True), collections.Counter())
        self.assertEqual(found, wanted)


# --- The Session card's pick buttons ---------------------------------------------------
PICK_PLUGINS = {
    "zzpick_plain": RECORD,
    "zzpick_msa": "PICKS_SESSION_FILE = 'msa'\n" + RECORD,
    "zzpick_embedding": "PICKS_SESSION_FILE = 'embedding'\n" + RECORD,
    "zzpick_msa_too": "PICKS_SESSION_FILE = 'msa'\n" + RECORD,
    "zzpick_fixed_file": "PICKS_SESSION_FILE = 'node_fasta'\n" + RECORD,
    "zzpick_unknown_file": "PICKS_SESSION_FILE = 'nosuchfile'\n" + RECORD,
    "zzpick_empty": "PICKS_SESSION_FILE = ''\n" + RECORD,
    "zzpick_none": "PICKS_SESSION_FILE = None\n" + RECORD,
    "zzpick_number": "PICKS_SESSION_FILE = 7\n" + RECORD,
    "zzpick_true": "PICKS_SESSION_FILE = True\n" + RECORD,
    "zzpick_list": "PICKS_SESSION_FILE = ['msa']\n" + RECORD,
    "zzpick_bytes": "PICKS_SESSION_FILE = b'msa'\n" + RECORD,
    "zzpick_function": "def PICKS_SESSION_FILE():\n    return 'msa'\n" + RECORD,
    "zzpick_import_error": "PICKS_SESSION_FILE = 'msa'\nraise RuntimeError('this plugin cannot be imported')\n",
    "zzpick_syntax_error": "PICKS_SESSION_FILE = 'msa'\ndef run(viewer, args:\n",
    "_zzpick_private": "PICKS_SESSION_FILE = 'msa'\n" + RECORD,
}
# A module that does not mention the declaration is never imported by the lookup; one that does is.
IMPORT_RECORD = "import Command_Engine\nCommand_Engine.SCRATCH_IMPORTS.append(__name__.split('.')[-1])\n"
WRONG_VALUES = ("zzpick_empty", "zzpick_none", "zzpick_number", "zzpick_true", "zzpick_list", "zzpick_bytes",
                "zzpick_function", "zzpick_import_error", "zzpick_syntax_error", "_zzpick_private")


def pickers_with(plugins, real=False):
    """session_file_pickers() with only the plugins (or the real commands as well), and what it wrote to stderr."""
    import contextlib
    import io

    manager = scratch_commands(plugins) if real else only_commands(plugins)
    errors = io.StringIO()
    with manager, contextlib.redirect_stderr(errors):
        return declarations.session_file_pickers(), errors.getvalue()


class SessionFilePickerTests(unittest.TestCase):
    """The Session card's pick buttons type the command that declares the file it picks, and no code names one."""

    def test_exactly_these_commands_declare_a_file_they_pick(self):
        # The card's two buttons type the same commands as before; only the two declare it.
        picking = {name: declarations.command_module(name).PICKS_SESSION_FILE for name in COMMAND_NAMES
                   if hasattr(declarations.command_module(name), "PICKS_SESSION_FILE")}
        self.assertEqual(picking, {"alignment": "msa", "embedding": "embedding"})
        for key in picking.values():
            self.assertIsInstance(key, str)
        pickers, note = pickers_with({}, real=True)
        self.assertEqual(pickers, {"msa": "alignment", "embedding": "embedding"})
        self.assertEqual(note, "")

    def test_a_command_that_declares_nothing_picks_no_file(self):
        self.assertEqual(pickers_with({"zzpick_plain": PICK_PLUGINS["zzpick_plain"]}), ({}, ""))
        self.assertEqual(pickers_with({}), ({}, ""))

    def test_a_plugin_that_declares_a_file_is_read_with_no_change_to_the_program(self):
        plugins = {name: PICK_PLUGINS[name] for name in ("zzpick_plain", "zzpick_msa", "zzpick_embedding")}
        self.assertEqual(pickers_with(plugins), ({"msa": "zzpick_msa", "embedding": "zzpick_embedding"}, ""))
        # With only one of them, the other file has none.
        self.assertEqual(pickers_with({"zzpick_msa": PICK_PLUGINS["zzpick_msa"]}), ({"msa": "zzpick_msa"}, ""))

    def test_the_file_key_is_the_viewers_to_judge_the_lookup_passes_any_string_on(self):
        # A file that cannot change, and a key the card does not have, are the card's to refuse (no button).
        plugins = {name: PICK_PLUGINS[name] for name in ("zzpick_fixed_file", "zzpick_unknown_file")}
        self.assertEqual(pickers_with(plugins),
                         ({"node_fasta": "zzpick_fixed_file", "nosuchfile": "zzpick_unknown_file"}, ""))

    def test_a_value_that_is_not_a_non_empty_string_or_a_module_that_cannot_be_used_picks_no_file(self):
        for name in WRONG_VALUES:
            with self.subTest(command=name):
                self.assertEqual(pickers_with({name: PICK_PLUGINS[name]}), ({}, ""))
        # Together, none of them spoils a good declaration beside them.
        plugins = {name: PICK_PLUGINS[name] for name in WRONG_VALUES + ("zzpick_msa",)}
        self.assertEqual(pickers_with(plugins), ({"msa": "zzpick_msa"}, ""))

    def test_two_commands_that_declare_the_same_file_leave_it_with_no_button_and_say_so(self):
        plugins = {name: PICK_PLUGINS[name] for name in ("zzpick_msa", "zzpick_msa_too", "zzpick_embedding")}
        pickers, note = pickers_with(plugins)
        self.assertEqual(pickers, {"embedding": "zzpick_embedding"})
        self.assertIn("zzpick_msa", note)
        self.assertIn("zzpick_msa_too", note)
        self.assertIn("'msa'", note)
        self.assertNotIn("zzpick_embedding", note)
        # A plugin that claims what a built-in command picks is a conflict too: the built-in one is not preferred.
        pickers, note = pickers_with({"zzpick_msa": PICK_PLUGINS["zzpick_msa"]}, real=True)
        self.assertEqual(pickers, {"embedding": "embedding"})
        self.assertIn("alignment", note)
        self.assertIn("zzpick_msa", note)

    def test_only_a_module_that_mentions_the_declaration_is_imported(self):
        # The lookup runs each time the card opens, so it does not import every command (that takes seconds).
        imported = []
        plugins = {"zzpick_quiet": IMPORT_RECORD, "zzpick_loud": "PICKS_SESSION_FILE = 'msa'\n" + IMPORT_RECORD}
        with mock.patch.object(ce, "SCRATCH_IMPORTS", imported, create=True):
            self.assertEqual(pickers_with(plugins), ({"msa": "zzpick_loud"}, ""))
        self.assertEqual(imported, ["zzpick_loud"])

    def test_it_is_read_afresh_each_time(self):
        with only_commands({}):
            self.assertEqual(declarations.session_file_pickers(), {})
        with only_commands({"zzpick_msa": PICK_PLUGINS["zzpick_msa"]}):
            self.assertEqual(declarations.session_file_pickers(), {"msa": "zzpick_msa"})
        with only_commands({}):
            self.assertEqual(declarations.session_file_pickers(), {})

    def test_the_viewers_window_code_imports_no_command(self):
        # The Session card, the Zoom box and the Viewer take what they need from the Viewer's own modules and from
        # what commands declare; none of them imports a command module (EMAPSSN_Viewer.py and Viewer_Widgets.py hold
        # many strings that are also command names, so only their imports are read).
        names = set(COMMAND_NAMES)
        for relative in ("src/EMAPSSN_Viewer.py", "src/desktop/Viewer_Widgets.py", "src/desktop/Viewer_Zoom.py"):
            source = (SRC_DIR.parent / relative).read_text(encoding="utf-8")
            imports = [key for key in command_words_in(source, names) if key[1] == "import"]
            with self.subTest(file=relative):
                self.assertEqual(imports, [])


if __name__ == "__main__":
    unittest.main()
