"""The reset command (commands/reset.py) and Command_Engine.execute_reset: which
targets they accept, that a refused reset changes nothing and saves no undo
state, and that a reset with nothing to reset saves none either.

The command modules that own the state declare their targets (RESET_TARGETS,
reset, reset_changes_state), and reset.py reads them: the second half of this
file shows what each real module declares, that the engine holds no reset code,
and that a plugin dropped into the commands folder becomes resettable with no
change to the program."""

import ast
import inspect
import io
import itertools
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib.colors as mcolors
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Command_Engine
import EMAPSSN_Config as cfg
from commands import reset as reset_command
from tests.command_fixtures import one_node_viewer, reported_outcomes, scratch_commands


ORIGINAL_COLORS = one_node_viewer().current_colors.copy()


class ResetCommandTests(unittest.TestCase):
    def make_viewer(self):
        viewer = one_node_viewer()
        viewer.tooltip = SimpleNamespace(text="")
        viewer.current_shapes = np.array(["star"], dtype=object)
        return viewer

    def run_reset(self, arguments):
        viewer = self.make_viewer()
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            reset_command.run(viewer, arguments)
        return viewer, succeeded, failed

    def test_unknown_or_missing_targets_fail_and_reset_nothing(self):
        for arguments, expected in (
            (["bogus"], "Unknown reset target(s): bogus."),
            # One unknown target refuses the whole command.
            (["colors", "bogus"], "Unknown reset target(s): bogus."),
            ([], "Specify at least one reset target."),
        ):
            with self.subTest(arguments=arguments):
                viewer, succeeded, failed = self.run_reset(arguments)

                np.testing.assert_array_equal(viewer.current_colors, ORIGINAL_COLORS)
                viewer._save_state.assert_not_called()
                viewer.update_nodes.assert_not_called()
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertIn(expected, failed.call_args.args[1])

    def test_known_targets_reset_as_one_undo_step(self):
        viewer, succeeded, failed = self.run_reset(["Colors", "shape"])

        np.testing.assert_allclose(
            viewer.current_colors, [mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)]
        )
        np.testing.assert_array_equal(viewer.current_shapes, ["disc"])
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Reset successful: colors, shapes.")

    def test_a_target_named_twice_is_reported_once(self):
        for arguments, expected in (
            (["colors", "colors"], "Reset successful: colors."),
            (["Colors", "color", "sizes"], "Reset successful: colors, sizes."),
            (["hide", "hidden"], "Reset successful: hidden."),
            (["order", "layer", "orders"], "Reset successful: node order."),
        ):
            with self.subTest(arguments=arguments):
                viewer, succeeded, failed = self.run_reset(arguments)

                failed.assert_not_called()
                self.assertEqual(succeeded.call_args.args[1], expected)

    def test_reset_saves_undo_state_only_when_something_changes(self):
        def at_defaults():
            viewer = self.make_viewer()
            viewer.current_colors[:] = mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)
            viewer.current_sizes.fill(cfg.NODE_SIZE)
            viewer.current_shapes[:] = "disc"
            viewer.cluster_labels = None
            viewer.last_cluster_params = None
            viewer.n_nodes = 1
            viewer.node_render_order = np.arange(1, dtype=np.int32)
            return viewer

        everything = ["colors", "sizes", "shapes", "clusters", "groups", "hide", "network", "order", "layer"]
        for arguments in (everything, ["hidden"], ["hide", "hidden"]):
            with self.subTest(nothing_to_reset=arguments):
                viewer = at_defaults()
                with reported_outcomes(), redirect_stdout(io.StringIO()):
                    reset_command.run(viewer, arguments)

                viewer._save_state.assert_not_called()

        # Each target alone, and then with a no-op target beside it.
        changes = {
            "colors": lambda v: v.current_colors.__setitem__(0, [1, 0, 0, 1]),
            "sizes": lambda v: v.current_sizes.fill(cfg.NODE_SIZE + 1),
            "shapes": lambda v: v.current_shapes.__setitem__(0, "star"),
            "clusters": lambda v: setattr(v, "cluster_labels", np.array([1])),
            "groups": lambda v: v.group_labels[0].add("kinases"),
            "hide": lambda v: v.visible_mask.__setitem__(0, False),
            "network": lambda v: (setattr(v, "original_pos", np.zeros((1, 2))), setattr(v, "pos", np.ones((1, 2)))),
            # A one-node order is always the identity, so this viewer gets a second node.
            "order": lambda v: (setattr(v, "n_nodes", 2), setattr(v, "node_render_order", np.array([1, 0], dtype=np.int32))),
        }
        for target, change in changes.items():
            for others in ([], ["hidden"]):
                with self.subTest(target=target, with_no_op_targets=others):
                    viewer = at_defaults()
                    change(viewer)
                    with reported_outcomes(), redirect_stdout(io.StringIO()):
                        reset_command.run(viewer, others + [target])

                    viewer._save_state.assert_called_once_with()

    def test_clusters_reset_clears_the_cluster_parameters_too(self):
        # Label, save and MCP inspection read last_cluster_params, which
        # describes the clusters that this reset just cleared.
        viewer = self.make_viewer()
        viewer.last_cluster_params = ("LEIDEN_1.0", 10)
        with reported_outcomes(), redirect_stdout(io.StringIO()):
            reset_command.run(viewer, ["clusters"])

        self.assertIsNone(viewer.cluster_labels)
        self.assertIsNone(viewer.last_cluster_params)
        viewer._save_state.assert_called_once_with()

    def test_clusters_reset_with_only_parameters_left_still_saves_undo_state(self):
        viewer = self.make_viewer()
        viewer.cluster_labels = None
        viewer.last_cluster_params = ("LEIDEN_1.0", 10)
        with reported_outcomes(), redirect_stdout(io.StringIO()):
            reset_command.run(viewer, ["clusters"])

        self.assertIsNone(viewer.last_cluster_params)
        viewer._save_state.assert_called_once_with()

    def test_clusters_reset_removes_the_generated_subcluster_groups(self):
        # They name nodes of the clusters this reset clears (label reset
        # comes here too). Custom groups, lookalikes included, are kept.
        viewer = self.make_viewer()
        viewer.group_labels[0].update({"subcluster_1_1", "subcluster_1_2", "subcluster_0_2", "kinases"})
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            reset_command.run(viewer, ["clusters"])

        self.assertIsNone(viewer.cluster_labels)
        self.assertEqual(viewer.group_labels[0], {"subcluster_0_2", "kinases"})
        viewer._save_state.assert_called_once_with()
        failed.assert_not_called()
        self.assertEqual(
            succeeded.call_args.args[1],
            "Reset successful: clusters. Removed 2 generated subcluster groups of the previous clustering.",
        )

    def test_clusters_reset_with_only_subcluster_groups_left_still_saves_undo_state(self):
        viewer = self.make_viewer()
        viewer.cluster_labels = None
        viewer.last_cluster_params = None
        viewer.group_labels[0].add("subcluster_1_1")
        with reported_outcomes(), redirect_stdout(io.StringIO()):
            reset_command.run(viewer, ["clusters"])

        self.assertEqual(viewer.group_labels[0], set())
        viewer._save_state.assert_called_once_with()

    def test_execute_reset_refuses_unknown_targets_before_saving_state(self):
        # The other commands' "<command> reset" forms call execute_reset
        # directly, so it keeps its own guard.
        viewer = self.make_viewer()
        with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
            Command_Engine.execute_reset(viewer, ["colors", "bogus"])

        np.testing.assert_array_equal(viewer.current_colors, ORIGINAL_COLORS)
        viewer._save_state.assert_not_called()

    def test_every_accepted_target_name_is_handled(self):
        # The names reset accepts (those the command modules declare) have to
        # match the targets it handles; a name only in the first would be
        # accepted and then silently do nothing.
        reported = {
            "color": "colors", "size": "sizes", "shape": "shapes",
            "cluster": "clusters", "group": "groups", "hide": "hidden",
            "hidden": "hidden", "network": "network", "order": "node order",
            "layer": "node order",
        }
        self.assertEqual(set(reset_command.target_names()), set(reported))
        for name, target in reported.items():
            with self.subTest(name=name), redirect_stdout(io.StringIO()):
                message = Command_Engine.execute_reset(self.make_viewer(), [name])
                self.assertEqual(message, f"Reset successful: {target}.")


# --- What each module declares, and that the engine holds no reset code ---------------
N = 5
DEFAULT_RGBA = np.asarray(mcolors.to_rgba(cfg.INITIAL_NODE_COLOR), dtype=np.float32)
ORIGINAL_POS = np.arange(N * 2, dtype=np.float32).reshape(N, 2)
# The groups of full_viewer() that the subcluster command generated; the lookalikes are custom groups.
GENERATED = {"subcluster_1_1", "subcluster_1_2", "subcluster_2_1"}
SUBCLUSTER_NOTE = " Removed 3 generated subcluster groups of the previous clustering."


class Visual:
    def __init__(self):
        self.parent = "canvas"


def full_viewer():
    """A viewer in which everything a reset restores has been changed, with mocks that log what it is asked."""
    calls = []
    return SimpleNamespace(
        n_nodes=N,
        current_colors=np.random.default_rng(1).random((N, 4)).astype(np.float32),
        current_sizes=np.arange(1, N + 1, dtype=np.float32),
        current_shapes=np.array(["star", "disc", "square", "star", "disc"], dtype=object),
        visible_mask=np.array([True, False, True, False, True]),
        cluster_labels=np.array([1, 1, 2, 2, -1]),
        last_cluster_params=("LEIDEN_1.0", 10),
        group_labels=[
            {"subcluster_1_1", "kinases"}, {"subcluster_1_2"}, {"subcluster_0_2", "mutant"},
            set(), {"subcluster_2_1", "subcluster_001_2", "basic"},
        ],
        original_pos=ORIGINAL_POS.copy(),
        pos=ORIGINAL_POS + 3.5,
        node_render_order=np.array([4, 2, 0, 1, 3], dtype=np.int32),
        tooltip=SimpleNamespace(text="hover"),
        label_visuals=[Visual(), Visual()],
        console_text=SimpleNamespace(text=""),
        calls=calls,
        _save_state=mock.Mock(side_effect=lambda: calls.append("save_state")),
        update_nodes=mock.Mock(side_effect=lambda: calls.append("update_nodes")),
        update_edges=mock.Mock(side_effect=lambda: calls.append("update_edges")),
        update_console_background=mock.Mock(),
    )


def state_of(viewer):
    """Everything a reset may touch, as plain values that compare equal when nothing changed."""
    return {
        "current_colors": viewer.current_colors.tolist(),
        "current_sizes": viewer.current_sizes.tolist(),
        "current_shapes": viewer.current_shapes.tolist(),
        "visible_mask": viewer.visible_mask.tolist(),
        "cluster_labels": None if viewer.cluster_labels is None else viewer.cluster_labels.tolist(),
        "last_cluster_params": viewer.last_cluster_params,
        "group_labels": [sorted(names) for names in viewer.group_labels],
        "pos": viewer.pos.tolist(),
        "node_render_order": viewer.node_render_order.tolist(),
        "tooltip": viewer.tooltip.text,
        "label_visuals": len(viewer.label_visuals),
    }


# For each target: the word the reply gives for it, whether it redraws the nodes and the edges, and the state it
# restores (from the state before it). Everything else has to be left as it was.
RESTORED = {
    "color": ("colors", True, False, lambda state: {"current_colors": np.tile(DEFAULT_RGBA, (N, 1)).tolist()}),
    "size": ("sizes", True, False,
             lambda state: {"current_sizes": np.full(N, cfg.NODE_SIZE, dtype=np.float32).tolist()}),
    "shape": ("shapes", True, False, lambda state: {"current_shapes": ["disc"] * N}),
    "cluster": ("clusters", False, False, lambda state: {
        "cluster_labels": None, "last_cluster_params": None, "tooltip": "", "label_visuals": 0,
        "group_labels": [sorted(set(names) - GENERATED) for names in state["group_labels"]]}),
    "group": ("groups", False, False, lambda state: {
        "group_labels": [[] for _ in range(N)], "tooltip": "", "label_visuals": 0}),
    "hide": ("hidden", True, True, lambda state: {"visible_mask": [True] * N}),
    "hidden": ("hidden", True, True, lambda state: {"visible_mask": [True] * N}),
    "network": ("network", True, True, lambda state: {"pos": ORIGINAL_POS.tolist()}),
    "order": ("node order", True, False, lambda state: {"node_render_order": list(range(N))}),
    "layer": ("node order", True, False, lambda state: {"node_render_order": list(range(N))}),
}


def expected_reply(targets):
    """The reply for these targets on full_viewer(): the words in order, once each, and the subcluster note.

    The note is there when a cluster target comes before any group target, as the group reset leaves no
    generated group for the cluster reset to remove.
    """
    words = ", ".join(dict.fromkeys(RESTORED[target][0] for target in targets))
    emptied_by_group = "group" in targets and ("cluster" not in targets or targets.index("group") < targets.index("cluster"))
    return f"Reset successful: {words}." + (SUBCLUSTER_NOTE if "cluster" in targets and not emptied_by_group else "")


class ResetDeclarationTests(unittest.TestCase):
    """Every target restores what its owner declared, and nothing else, as the engine's own code once did.

    These cases were also compared with the engine's execute_reset from before the code moved, over 628
    scenarios: every spelling, every pair of targets, states with nothing to reset or one thing to reset,
    viewers that lack some of the state, unknown targets and help.
    """

    def reset(self, targets, viewer=None):
        viewer = viewer or full_viewer()
        before = state_of(viewer)
        with redirect_stdout(io.StringIO()):
            message = Command_Engine.execute_reset(viewer, list(targets))
        return viewer, before, state_of(viewer), message

    def expected_state(self, before, targets):
        expected = dict(before)
        for target in targets:
            expected.update(RESTORED[target][3](expected))
        return expected

    def test_the_modules_that_own_the_state_declare_the_targets(self):
        owners = {name: [module.__name__.rpartition(".")[2] for module in modules]
                  for name, modules in reset_command.declared_targets().items()}
        self.assertEqual(owners, {
            "color": ["color"], "cluster": ["cluster"], "group": ["group"], "hide": ["hide"], "hidden": ["hide"],
            "size": ["reset"], "shape": ["reset"], "network": ["reset"], "order": ["reset"], "layer": ["reset"],
        })

    def test_a_declaring_module_gives_reset_and_reset_changes_state_the_viewer_and_the_target(self):
        for modules in reset_command.declared_targets().values():
            for module in modules:
                for function in ("reset", "reset_changes_state"):
                    with self.subTest(module=module.__name__, function=function):
                        self.assertEqual(list(inspect.signature(getattr(module, function)).parameters),
                                         ["viewer", "target"])

    def test_each_spelling_of_each_target_restores_its_state_and_nothing_else(self):
        for name, (_, nodes, edges, _) in RESTORED.items():
            for typed in (name, name + "s", name.upper(), name.capitalize() + "s"):
                with self.subTest(target=typed):
                    viewer, before, after, message = self.reset([typed])

                    self.assertEqual(after, self.expected_state(before, [name]))
                    self.assertEqual(message, expected_reply([name]))
                    self.assertEqual(viewer.console_text.text, message)
                    self.assertEqual(viewer.calls, ["save_state"] + ["update_nodes"] * nodes + ["update_edges"] * edges)

    def test_the_label_visuals_the_cluster_and_group_resets_drop_lose_their_parent(self):
        for target in ("clusters", "groups"):
            with self.subTest(target=target):
                viewer = full_viewer()
                visuals = list(viewer.label_visuals)
                with redirect_stdout(io.StringIO()):
                    Command_Engine.execute_reset(viewer, [target])
                self.assertEqual(viewer.label_visuals, [])
                self.assertEqual([visual.parent for visual in visuals], [None, None])

    def test_several_targets_are_one_undo_step_one_redraw_and_one_reply(self):
        every = list(RESTORED)
        viewer, before, after, message = self.reset(every)

        self.assertEqual(after, self.expected_state(before, every))
        self.assertEqual(viewer.calls, ["save_state", "update_nodes", "update_edges"])
        self.assertEqual(message, "Reset successful: colors, sizes, shapes, clusters, groups, hidden, network, "
                                  "node order." + SUBCLUSTER_NOTE)

    def test_every_pair_of_targets_is_the_two_resets_in_one_step(self):
        for pair in itertools.permutations(RESTORED, 2):
            pair = list(pair)
            with self.subTest(targets=pair):
                viewer, before, after, message = self.reset(pair)

                self.assertEqual(after, self.expected_state(before, pair))
                self.assertEqual(message, expected_reply(pair))
                self.assertEqual(viewer.calls.count("save_state"), 1)
                self.assertEqual(viewer.calls.count("update_nodes"), int(any(RESTORED[t][1] for t in pair)))
                self.assertEqual(viewer.calls.count("update_edges"), int(any(RESTORED[t][2] for t in pair)))

    def test_a_cluster_target_named_twice_still_reports_the_groups_removed(self):
        # The first reset removes them. The second has none left to remove, and once replaced the note with nothing.
        for targets in (["cluster", "clusters"], ["clusters", "color", "cluster"]):
            with self.subTest(targets=targets):
                _, _, _, message = self.reset(targets)
                self.assertTrue(message.endswith(SUBCLUSTER_NOTE), message)

    def test_an_unknown_target_resets_nothing_whatever_it_is_given_with(self):
        unknown_words = ("bogus", "nope", "colorss", "s", "", "reset", "colors,")
        for targets in (["bogus"], ["colors", "bogus", "hide"], ["bogus", "colors", "nope"], ["colorss"], ["s"], [""],
                        ["reset"], ["colors,"], []):
            with self.subTest(targets=targets):
                viewer = full_viewer()
                before = state_of(viewer)
                with self.assertRaises(ValueError) as caught, redirect_stdout(io.StringIO()):
                    Command_Engine.execute_reset(viewer, targets)

                self.assertEqual(state_of(viewer), before)
                self.assertEqual(viewer.calls, [])
                text = str(caught.exception)
                unknown = ", ".join(target for target in targets if target in unknown_words)
                self.assertIn(f"Unknown reset target(s): {unknown}. Nothing was reset." if targets
                              else "Specify at least one reset target.", text)
                self.assertTrue(text.endswith(reset_command.USAGE))

    def test_reset_help_prints_the_help_and_resets_nothing(self):
        for word in ("help", "-h", "--help", "HELP"):
            with self.subTest(word=word):
                viewer = full_viewer()
                before = state_of(viewer)
                output = io.StringIO()
                with reported_outcomes() as (succeeded, failed), redirect_stdout(output):
                    reset_command.run(viewer, [word, "colors"])

                self.assertEqual(state_of(viewer), before)
                self.assertEqual(viewer.calls, [])
                self.assertIn("Network Reset Tool", output.getvalue())
                self.assertEqual(viewer.console_text.text, "Help information printed to the Console")
                failed.assert_not_called()
                succeeded.assert_called_once_with(viewer, "Help information printed to the Console.")

    def test_a_viewer_without_some_of_the_state_resets_what_it_has(self):
        for missing, target in (("current_colors", "colors"), ("current_sizes", "sizes"), ("current_shapes", "shapes"),
                                ("original_pos", "network"), ("label_visuals", "clusters")):
            with self.subTest(missing=missing):
                viewer = full_viewer()
                delattr(viewer, missing)
                with redirect_stdout(io.StringIO()):
                    message = Command_Engine.execute_reset(viewer, [target])
                self.assertTrue(message.startswith("Reset successful: "), message)

    def test_the_engines_name_only_passes_the_call_on(self):
        viewer = full_viewer()
        with mock.patch.object(reset_command, "execute_reset", return_value="the reply") as inner:
            self.assertEqual(Command_Engine.execute_reset(viewer, ["colors"]), "the reply")
        inner.assert_called_once_with(viewer, ["colors"])

    def test_the_engine_holds_no_reset_code_and_names_no_reset_target(self):
        tree = ast.parse(Path(SRC_DIR, "Command_Engine.py").read_text(encoding="utf-8"))
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("_reset_changes_state", functions)
        # A docstring, the import of the command, and the call that passes it on.
        self.assertEqual([type(statement).__name__ for statement in functions["execute_reset"].body],
                         ["Expr", "ImportFrom", "Return"])
        words = {"size", "sizes", "shape", "shapes", "network", "order", "layer", "hidden", "node order", "colors"}
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body
                      and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
        named = {node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
                 and node.value in words}
        self.assertEqual(named, set())


# --- A plugin dropped into the commands folder ---------------------------------------
PLUGINS = {
    # Declares a target and nothing else: no word on whether a reset changes anything, so it saves an undo step.
    "zzreset_plain": (
        "from commands.reset import ResetOutcome\n"
        "RESET_TARGETS = ('zzplain',)\n"
        "def reset(viewer, target):\n"
        "    viewer.plugin_log.append(('zzreset_plain', target))\n"
        "    return ResetOutcome('plains')\n"),
    # Says when a reset changes nothing, what to redraw, and what to add to the reply.
    "zzreset_quiet": (
        "from commands.reset import ResetOutcome\n"
        "from utilities.Localization import Message\n"
        "RESET_TARGETS = ('zzquiet',)\n"
        "def reset_changes_state(viewer, target):\n"
        "    return viewer.quiet_dirty\n"
        "def reset(viewer, target):\n"
        "    viewer.plugin_log.append(('zzreset_quiet', target))\n"
        "    return ResetOutcome('quiets', nodes=True, edges=True, note=Message('A note from the plugin.'))\n"),
    # A name that ends in "s", declared as typed, and one declared in capitals.
    "zzreset_names": (
        "from commands.reset import ResetOutcome\n"
        "RESET_TARGETS = ['zzstatus', 'ZZMixed']\n"
        "def reset(viewer, target):\n"
        "    viewer.plugin_log.append(('zzreset_names', target))\n"
        "    return ResetOutcome(target)\n"),
    # Resets its own part of a target the program has: both run, and the word is reported once.
    "zzreset_color": (
        "from commands.reset import ResetOutcome\n"
        "RESET_TARGETS = ('color',)\n"
        "def reset(viewer, target):\n"
        "    viewer.plugin_log.append(('zzreset_color', target))\n"
        "    return ResetOutcome('colors')\n"),
    "zzreset_calm": (
        "from commands.reset import ResetOutcome\n"
        "RESET_TARGETS = ('zzcalm',)\n"
        "def reset_changes_state(viewer, target):\n"
        "    return False\n"
        "def reset(viewer, target):\n"
        "    viewer.plugin_log.append(('zzreset_calm', target))\n"
        "    return ResetOutcome('calms')\n"),
    "zzreset_none": (
        "RESET_TARGETS = ('zznone',)\n"
        "def reset(viewer, target):\n"
        "    viewer.plugin_log.append(('zzreset_none', target))\n"),
    # Modules that declare nothing usable have no targets.
    "zzreset_undeclared": (
        "def reset_changes_state(viewer, target):\n    raise AssertionError('asked')\n"
        "def reset(viewer, target):\n    raise AssertionError('reset')\n"),
    "zzreset_comment": (
        "# RESET_TARGETS is named here, and declared nowhere.\n"
        "def reset(viewer, target):\n    raise AssertionError('reset')\n"),
    "zzreset_no_function": "RESET_TARGETS = ('zznofunction',)\n",
    "zzreset_function_not_callable": "RESET_TARGETS = ('zznotcallable',)\nreset = 'nothing'\n",
    "zzreset_string": "RESET_TARGETS = 'zzstring'\ndef reset(viewer, target):\n    raise AssertionError('reset')\n",
    "zzreset_wrong_names": (
        "RESET_TARGETS = (7, None, '')\n"
        "def reset(viewer, target):\n    raise AssertionError('reset')\n"),
    "zzreset_import_error": "RESET_TARGETS = ('zzbroken',)\nraise RuntimeError('this plugin cannot be imported')\n",
    "_zzreset_private": (
        "RESET_TARGETS = ('zzprivate',)\n"
        "def reset(viewer, target):\n    raise AssertionError('reset')\n"),
}
NO_TARGETS = ("zzundeclared", "zzcomment", "zznofunction", "zznotcallable", "zzstring", "zzbroken", "zzprivate",
              "z", "zz", "7", "none", "")


class ResetPluginTests(unittest.TestCase):
    """A plugin dropped into the commands folder becomes resettable by what it declares, with no change to the program."""

    def setUp(self):
        self.enterContext(scratch_commands(PLUGINS))
        self.viewer = one_node_viewer()
        self.viewer.tooltip = SimpleNamespace(text="")
        self.viewer.current_shapes = np.array(["star"], dtype=object)
        self.viewer.plugin_log = []
        self.viewer.quiet_dirty = False

    def run_reset(self, arguments):
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            reset_command.run(self.viewer, arguments)
        return succeeded, failed

    def test_the_targets_of_the_program_are_still_known_beside_the_plugins(self):
        self.assertEqual(reset_command.target_names() - {"zzplain", "zzquiet", "zzstatus", "zzmixed", "zzcalm", "zznone"},
                         {"color", "cluster", "group", "hide", "hidden", "size", "shape", "network", "order", "layer"})

    def test_a_plugin_that_declares_a_target_is_reset_with_it(self):
        for typed in ("zzplain", "zzplains", "ZZPlain", "ZZPLAINS"):
            with self.subTest(target=typed):
                self.viewer.plugin_log.clear()
                self.viewer._save_state.reset_mock()
                succeeded, failed = self.run_reset([typed])

                failed.assert_not_called()
                self.assertEqual(succeeded.call_args.args[1], "Reset successful: plains.")
                self.assertEqual(self.viewer.plugin_log, [("zzreset_plain", "zzplain")])
                # It does not say that a reset changes nothing, so the undo step is saved.
                self.viewer._save_state.assert_called_once_with()
                self.viewer.update_nodes.assert_not_called()

    def test_a_plugin_says_when_a_reset_changes_nothing_what_to_redraw_and_what_to_add(self):
        self.run_reset(["zzquiet"])
        self.viewer._save_state.assert_not_called()
        self.assertEqual(self.viewer.plugin_log, [("zzreset_quiet", "zzquiet")])
        self.viewer.update_nodes.assert_called_once_with()
        self.viewer.update_edges.assert_called_once_with()

        self.viewer.quiet_dirty = True
        succeeded, _ = self.run_reset(["zzquiet"])
        self.viewer._save_state.assert_called_once_with()
        self.assertEqual(succeeded.call_args.args[1], "Reset successful: quiets. A note from the plugin.")

    def test_no_undo_step_is_saved_when_every_target_says_it_changes_nothing(self):
        # No node is hidden, and the plugin says so for its own target.
        self.run_reset(["hidden", "zzcalm"])
        self.viewer._save_state.assert_not_called()
        self.assertEqual(self.viewer.plugin_log, [("zzreset_calm", "zzcalm")])

        # One that says nothing counts as a change, and so does a target that has one.
        self.run_reset(["zzcalm", "zzplain"])
        self.viewer._save_state.assert_called_once_with()

    def test_a_plugin_can_reset_its_part_of_a_target_the_program_has(self):
        succeeded, failed = self.run_reset(["colors"])

        failed.assert_not_called()
        np.testing.assert_allclose(self.viewer.current_colors, [mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)])
        self.assertEqual(self.viewer.plugin_log, [("zzreset_color", "color")])
        self.assertEqual(succeeded.call_args.args[1], "Reset successful: colors.")
        self.viewer._save_state.assert_called_once_with()

    def test_a_name_that_ends_in_s_is_a_target_as_typed_and_a_name_in_capitals_is_matched_in_lower_case(self):
        for typed, declared in (("zzstatus", "zzstatus"), ("ZZStatus", "zzstatus"),
                                ("zzmixed", "zzmixed"), ("zzmixeds", "zzmixed")):
            with self.subTest(target=typed):
                self.viewer.plugin_log.clear()
                succeeded, failed = self.run_reset([typed])
                failed.assert_not_called()
                self.assertEqual(self.viewer.plugin_log, [("zzreset_names", declared)])
                self.assertEqual(succeeded.call_args.args[1], f"Reset successful: {declared}.")

    def test_a_plugin_that_returns_nothing_adds_nothing_to_the_reply(self):
        succeeded, failed = self.run_reset(["colors", "zznone"])
        failed.assert_not_called()
        self.assertEqual(self.viewer.plugin_log, [("zzreset_color", "color"), ("zzreset_none", "zznone")])
        self.assertEqual(succeeded.call_args.args[1], "Reset successful: colors.")

    def test_a_module_without_a_usable_declaration_has_no_targets(self):
        known = reset_command.target_names()
        for name in NO_TARGETS:
            with self.subTest(target=name):
                self.assertNotIn(name, known)
                self.viewer.plugin_log.clear()
                self.viewer._save_state.reset_mock()
                succeeded, failed = self.run_reset(["colors", name])

                # Refused as unknown, and nothing at all was reset, not even the colours.
                succeeded.assert_not_called()
                self.assertIn(f"Unknown reset target(s): {name}. Nothing was reset.", failed.call_args.args[1])
                self.viewer._save_state.assert_not_called()
                self.assertEqual(self.viewer.plugin_log, [])

    def test_a_module_that_cannot_be_imported_does_not_stop_the_others(self):
        succeeded, failed = self.run_reset(["zzplain", "hide"])
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Reset successful: plains, hidden.")

    def test_an_unknown_target_among_a_plugins_resets_nothing(self):
        succeeded, failed = self.run_reset(["zzplain", "bogus", "zzquiet"])
        succeeded.assert_not_called()
        self.assertIn("Unknown reset target(s): bogus.", failed.call_args.args[1])
        self.assertEqual(self.viewer.plugin_log, [])
        self.viewer._save_state.assert_not_called()


class ResetPluginFolderTests(unittest.TestCase):
    """The targets are those of the commands folder as it is when a reset runs."""

    def test_a_plugin_is_a_target_while_it_is_in_the_folder_and_not_before_or_after(self):
        self.assertNotIn("zzplain", reset_command.target_names())
        with scratch_commands(PLUGINS):
            self.assertIn("zzplain", reset_command.target_names())
        self.assertNotIn("zzplain", reset_command.target_names())

    def test_only_the_modules_that_mention_the_declaration_are_imported_to_find_the_targets(self):
        with scratch_commands({"zzreset_plain": PLUGINS["zzreset_plain"], "zzreset_unnamed": "VALUE = 1\n"}):
            reset_command.target_names()
            self.assertIn("commands.zzreset_plain", sys.modules)
            self.assertNotIn("commands.zzreset_unnamed", sys.modules)


if __name__ == "__main__":
    unittest.main()
