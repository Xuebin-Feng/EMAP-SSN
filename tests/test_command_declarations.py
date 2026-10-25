"""What a command module declares about how the Viewer dispatches it (Command_Declarations).

The engine, the command queue and the command portal never name a command: the
modules of src/commands say which run at once, which start a model request and
which act on jobs, and a module that says nothing gets the cautious defaults.
The first classes show that every case that used to be hard-coded behaves as it
did, against copies of the old rules, and that a plugin dropped into the
commands package is dispatched by what it declares with no change to the
program. The last class reads the source of the central files and fails when one
of them names a command again.
"""
import ast
import collections
import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import Command_Declarations as declarations  # noqa: E402
import Command_Engine as ce  # noqa: E402
from Viewer_Command_Portal import ViewerCommandPortal  # noqa: E402
from tests.command_fixtures import scratch_commands  # noqa: E402
from tests.viewer_fixtures import PortalFixture  # noqa: E402

COMMANDS_DIR = SRC_DIR / "commands"
COMMAND_NAMES = sorted(path.stem for path in COMMANDS_DIR.glob("*.py") if not path.stem.startswith("_"))
REFUSAL = "Model-generated agent commands cannot start another model request."
INSTANT, ORDERED = "instant", "ordered"
SOURCES = ("typed", "script", "portal")

# --- The rules as they were hard-coded before the modules declared them -----------------
OLD_HELP_WORDS = frozenset({"help", "-h", "--help", "-?"})
OLD_AGENT_SETTINGS = frozenset({"off", "deactivate"})


def old_queue_kind(command_name, args, source="typed"):
    """Command_Engine.queue_kind, as it was."""
    first = args[0].lower() if args else ""
    if first in OLD_HELP_WORDS or first == "--register-only":
        return INSTANT
    if command_name in ("jobs", "zoom", "run", "console", "help"):  # console and help came later and declare what these did
        return INSTANT
    if command_name == "esmfold" and first == "stop":
        return INSTANT
    if command_name in ("undo", "redo") and source == "typed":
        return INSTANT
    if command_name == "meta" and not args:
        return INSTANT
    if command_name == "agent":
        argument = " ".join(args)
        if not args or first in OLD_AGENT_SETTINGS or (argument.startswith("<") and argument.endswith(">")):
            return INSTANT
    return ORDERED


def old_model_request_refused(cmd_str):
    """The rule execute_command applied to a command run for the portal, as it was."""
    if not (str(cmd_str).split() and str(cmd_str).split()[0].lower() == "agent"):
        return False
    argument = str(cmd_str).partition(" ")[2].strip()
    configuration = (not argument
                     or argument.lower() in {"help", "-h", "--help", "off", "deactivate", "--register-only"}
                     or (argument.startswith("<") and argument.endswith(">")))
    return not configuration


def old_is_stop(command):
    """ViewerCommandPortal._is_stop, as it was, without the Agent review check."""
    words = tuple(command.lower().split())
    return words in {("esmfold", "stop"), ("run", "stop")} or (bool(words) and words[0] in {"jobs"})


ARGUMENT_LISTS = [
    [], ["help"], ["Help"], ["-h"], ["--help"], ["--HELP"], ["-?"], ["--register-only"], ["--register-only", "x"],
    ["stop"], ["STOP"], ["Stop"], ["stop", "large"], ["stop", "now"], ["large"], ["multi"], ["large", "multi"],
    ["off"], ["OFF"], ["deactivate"], ["Deactivate"], ["off", "now"], ["deactivate", "now"], ["help", "me"],
    ["<Model>"], ["<My", "Model>"], ["<x"], ["x>"], ["<>"], ["<", ">"],
    ["hello"], ["hello", "world"], ['"make', "cluster", "1", 'red"'], ["3"], ["0"], ["cancel"], ["cancel", "all"],
    ["cancel", "14"], ["upload", "a.csv"], ["import", "a.csv"], ["red"], ["{Length>10}"],
]
NAMES_TO_TRY = COMMAND_NAMES + ["nosuchcommand", "help"]


class OldRulesTests(unittest.TestCase):
    """Every case the engine and the portal used to decide by name is decided the same way by the modules."""

    def test_every_command_waits_or_runs_at_once_as_it_did(self):
        count = 0
        for name in NAMES_TO_TRY:
            for args in ARGUMENT_LISTS:
                for source in SOURCES:
                    with self.subTest(command=name, args=args, source=source):
                        self.assertEqual(ce.queue_kind(name, list(args), source), old_queue_kind(name, args, source))
                    count += 1
        self.assertGreater(count, 3000)

    def test_the_engine_gives_the_same_answer_for_the_module_it_has_just_loaded(self):
        for name in ("jobs", "zoom", "run", "esmfold", "undo", "redo", "meta", "agent", "select"):
            module = declarations.command_module(name)
            self.assertIsNotNone(module, name)
            for args in ARGUMENT_LISTS:
                for source in SOURCES:
                    with self.subTest(command=name, args=args, source=source):
                        self.assertEqual(ce.queue_kind(name, list(args), source, module),
                                         old_queue_kind(name, args, source))

    def test_a_command_given_to_the_portal_is_refused_when_it_starts_a_model_request_as_it_was(self):
        for name in NAMES_TO_TRY + ["AGENT", "Agent"]:
            for args in ARGUMENT_LISTS:
                line = " ".join([name, *args])
                with self.subTest(command=line):
                    refused = declarations.starts_model_request(name.lower(), list(args))
                    self.assertEqual(refused, old_model_request_refused(line))

    def test_a_request_of_stop_commands_runs_ahead_of_the_head_as_it_did(self):
        lines = []
        for name in NAMES_TO_TRY + ["ESMFOLD", "Run", "JOBS"]:
            for args in ARGUMENT_LISTS:
                lines += [" ".join([name, *args]), "  " + name + "   " + "   ".join(args) + "  "]
        for line in lines:
            with self.subTest(command=line):
                self.assertEqual(ViewerCommandPortal._is_stop({"command": line}), old_is_stop(line))
        self.assertFalse(ViewerCommandPortal._is_stop({"command": ""}))
        self.assertFalse(ViewerCommandPortal._is_stop({"command": "   "}))

    def test_a_stop_command_still_to_be_reviewed_does_not_run_ahead(self):
        for approval, expected in ((None, True), ("approved", True), ("pending", False), ("refused", False),
                                   ("discarded", False)):
            with self.subTest(approval=approval):
                self.assertEqual(ViewerCommandPortal._is_stop({"command": "esmfold stop", "approval": approval}),
                                 expected)

    def test_a_tab_does_not_hide_a_message_to_the_model(self):
        # The old rule read the argument after the first space, so `agent<TAB>hello` had none and was let
        # through, and the dispatch then sent "hello" to the model. The rule now reads the words the dispatch
        # reads.
        self.assertFalse(old_model_request_refused("agent\thello"))
        self.assertTrue(declarations.starts_model_request("agent", "agent\thello".split()[1:]))


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

    def test_every_declaration_of_the_real_commands_is_well_formed(self):
        # A typo in a declaration would silently fall back to the default, so the real modules are read here.
        for name in COMMAND_NAMES:
            module = declarations.command_module(name)
            self.assertIsNotNone(module, name)
            with self.subTest(command=name):
                if hasattr(module, "QUEUE"):
                    self.assertIn(module.QUEUE, (INSTANT, ORDERED))
                for constant in ("STARTS_MODEL_REQUEST", "ACTS_ON_JOBS"):
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

    def test_the_commands_that_declare_something_are_the_ones_that_did_before(self):
        declaring = {name for name in COMMAND_NAMES
                     if any(hasattr(declarations.command_module(name), attribute)
                            for attribute in ("QUEUE", "queue_kind", "STARTS_MODEL_REQUEST", "starts_model_request",
                                              "ACTS_ON_JOBS", "acts_on_jobs"))}
        self.assertEqual(declaring, {"jobs", "zoom", "run", "esmfold", "undo", "redo", "meta", "agent", "console", "help"})


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


# --- No central file names a command -----------------------------------------------------
GENERIC_WORDS = frozenset({"help", "-h", "--help", "-?", "--register-only"})

# Where a central file may hold a word that is also a command's name, and why. Each entry is
# (enclosing function or class, kind, word) with how many times it occurs there: it fails when a
# new one appears (a command named in code) and when one is gone (the entry has to be removed).
SELECTION_GRAMMAR = "the selection grammar's own vocabulary: what a #label# or an amino-acid target is read from"
RESET_TARGETS = "reset's targets, which the viewer's engine carries out (the VR viewer reuses its execute_reset)"
JOBS_KEY = "the 'jobs' entry of a command request's record"
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
        ("_reset_changes_state", "string", "color"): (1, RESET_TARGETS),
        ("_reset_changes_state", "string", "cluster"): (1, RESET_TARGETS),
        ("_reset_changes_state", "string", "group"): (1, RESET_TARGETS),
        ("_reset_changes_state", "string", "hide"): (1, RESET_TARGETS),
        ("_reset_changes_state", "import", "group"): (1, RESET_TARGETS),
        ("execute_reset", "string", "color"): (1, RESET_TARGETS),
        ("execute_reset", "string", "cluster"): (1, RESET_TARGETS),
        ("execute_reset", "string", "group"): (1, RESET_TARGETS),
        ("execute_reset", "string", "hide"): (1, RESET_TARGETS),
        ("execute_reset", "import", "group"): (1, RESET_TARGETS),
        ("execute_reset", "import", "reset"): (1, RESET_TARGETS),
        ("_dispatch_user_command", "string", "run"): (1, "the entry point every command module has: hasattr(module, 'run')"),
    },
    "src/Viewer_Command_Portal.py": {
        ("<module>", "string", "run"): (1, "a value of APPROVAL, the review's verdict, not a command"),
        ("agent_page_connected", "string", "agent"): (1, "the client id of the Agent web page"),
        ("ExecutionContext.add_job", "string", "jobs"): (1, JOBS_KEY),
        ("ExecutionContext.job_event", "string", "jobs"): (1, JOBS_KEY),
        ("ExecutionContext.job_progress", "string", "jobs"): (1, JOBS_KEY),
        ("ViewerCommandPortal._advance", "string", "jobs"): (3, JOBS_KEY),
        ("ViewerCommandPortal._frees_the_line", "string", "jobs"): (1, JOBS_KEY),
        ("ViewerCommandPortal.read_output", "string", "offset"): (1, "a key of an output page"),
    },
    "src/Command_Queue.py": {
        ("CommandQueue._update_places", "string", "jobs"): (1, JOBS_KEY),
    },
    "src/Background_Job_Scheduler.py": {
        ("BackgroundJobScheduler._update_records", "string", "jobs"): (1, JOBS_KEY),
        ("BackgroundJobScheduler._report_succeeded", "string", "jobs"): (1, JOBS_KEY),
    },
    "src/Viewer_Worker_Tracking.py": {
        ("WorkerTracker.poll", "string", "jobs"): (1, JOBS_KEY),
    },
    "src/Command_Declarations.py": {},
}


def command_words_in(source, names):
    """Counter of (enclosing scope, 'string' | 'import', command name) for the code of a module's source.

    A string constant equal to a command's name (docstrings and the generic help words are not counted), and an
    import of a command module (from commands import x, from commands.x import y, import commands.x, commands.x).
    """
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

    found = collections.Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if node.value in names and node.value not in GENERIC_WORDS:
                found[(scope(node), "string", node.value)] += 1
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")
            if module == ["commands"]:
                found.update((scope(node), "import", alias.name) for alias in node.names if alias.name in names)
            elif module[0] == "commands" and len(module) > 1 and module[1] in names:
                found[(scope(node), "import", module[1])] += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "commands" and len(parts) > 1 and parts[1] in names:
                    found[(scope(node), "import", parts[1])] += 1
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "commands"
              and node.attr in names):
            found[(scope(node), "import", node.attr)] += 1
    return found


class CentralCodeNamesNoCommandTests(unittest.TestCase):
    """The engine, the portal, the queue, the scheduler and the worker tracking name no command."""

    def test_the_scanner_finds_a_command_named_in_code(self):
        # A positive control: without it a scanner that finds nothing would pass every file.
        names = {"jobs", "zoom", "run", "undo", "agent"}
        source = ("from commands import zoom\n"
                  "import commands.run\n"
                  "from commands.undo import steps\n"
                  "'''docstring jobs'''\n"
                  "def kind(name, args):\n"
                  "    '''A docstring that says jobs.'''\n"
                  "    if name in ('jobs', 'zoom'):\n"
                  "        return 'instant'\n"
                  "    if name == 'agent' and args == ['help', '--help']:\n"
                  "        return commands.agent\n"
                  "    return 'a jobs sentence'\n")
        self.assertEqual(command_words_in(source, names), collections.Counter({
            ("<module>", "import", "zoom"): 1, ("<module>", "import", "run"): 1, ("<module>", "import", "undo"): 1,
            ("kind", "string", "jobs"): 1, ("kind", "string", "zoom"): 1, ("kind", "string", "agent"): 1,
            ("kind", "import", "agent"): 1}))

    def test_the_command_names_are_the_modules_of_src_commands(self):
        self.assertTrue({"agent", "jobs", "run", "esmfold", "undo", "redo", "meta", "zoom"} <= set(COMMAND_NAMES))

    def test_no_central_file_names_a_command(self):
        for relative, allowed in ALLOWED.items():
            source = (SRC_DIR.parent / relative).read_text(encoding="utf-8")
            found = command_words_in(source, set(COMMAND_NAMES))
            expected = collections.Counter({key: count for key, (count, _reason) in allowed.items()})
            with self.subTest(file=relative):
                named = found - expected
                gone = expected - found
                self.assertFalse(named, f"{relative} names a command in code: {dict(named)}. A command is dispatched "
                                        "by what its module declares (Command_Declarations), not by name here.")
                self.assertFalse(gone, f"{relative} no longer holds {dict(gone)}: remove these entries from ALLOWED.")

    def test_the_central_files_are_the_ones_that_dispatch(self):
        # A file that moved or was renamed would otherwise stop being checked.
        for relative in ALLOWED:
            self.assertTrue((SRC_DIR.parent / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
