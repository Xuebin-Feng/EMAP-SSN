# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The reset command, and what it can reset.

``reset <target> ...`` restores part of the Viewer's state. The program never names a
target: the command module that owns the state a reset restores declares it, as a module
declares how it is dispatched (Command_Declarations), and a plugin dropped into this
folder becomes resettable the same way. A module may declare:

    RESET_TARGETS = ('cluster',)

        The target names the module resets, in lower case. A user may type a name or its
        plural: "cluster" and "clusters" both stand for 'cluster', and a name that ends in
        "s" may be declared as typed ("status") or without the "s". Several modules may
        declare one name: each resets its own part, in the order of the modules' names.
        Without the declaration (or with one that is not a tuple or list of names, or
        without reset()) a module has no targets.

    def reset(viewer, target):

        Restores the module's state for one of its targets and returns a ResetOutcome
        (this module's), or None when there is nothing to report.

    def reset_changes_state(viewer, target):    # optional

        False only when resetting the target is certain to leave everything an undo step
        saves as it is. The reset saves an undo step only when a target it was given has
        something to reset; a module without this function counts as having.

    RESET_HELP = {'colors': (10, 'Restores all node colors.')}    # optional

        What "reset help" says of the module's targets, and what the usage line of a refusal
        lists. A key is the words a user types for a target, as the help shows them: one word
        ("colors"), or several that do the same, separated by commas ("order, layer"); each is
        read as a typed target is, so "colors" stands for the target 'color'. The value is the
        description, which may run over several lines, or a pair (rank, description): the
        lines are listed in rank order, smaller first, then by the modules' names, and a line
        with no rank comes after those that have one. None says that the word is an
        alternative spelling that the help need not list ("hidden" beside "hide"). A target
        that no key names is listed after the others, by name, without a description; so is
        one whose entry cannot be used (a key that names no target of the module, a value of
        another kind). The usage line lists each word of each key, in the same order.

Which modules declare is found by reading the source of the commands folder for the word
RESET_TARGETS, so only the modules that mention it are imported. The targets with no
command of their own, the node sizes and shapes, the layout and the render order, are
declared below like any other. The catalog entry and the Agent's item below are literals
that are read from the source without importing it, so they name the targets of the
program's own commands; a plugin's target is listed by "reset help" and by the usage line.
"""

import math
import os
import pkgutil
import sys
from typing import NamedTuple, Optional

import Command_Declarations
import Command_Engine
import EMAPSSN_Config as cfg
import numpy as np
from utilities.Localization import JoinedMessage, Message
from desktop.Command_Metadata import argument, choice, entry

# --- What this command declares about itself ---
# Changes only the view and the session's in-memory state.
AGENT_POLICY = 'run'

CATALOG_ENTRY = entry('Restore selected network properties in one undoable action.',
    argument('targets', 'One or more targets: reset colors/sizes to configured defaults, shapes to discs, clear clusters/groups (clearing clusters also removes the generated subcluster_N_M groups), unhide nodes, restore original/last-saved positions, or reset render order. Keywords are case-insensitive; an unknown target fails the command without resetting anything.',
        choice('colors','color'), choice('sizes','size'), choice('shapes','shape'), choice('clusters','cluster'), choice('groups','group'),
        choice('hide','hides','hidden','hiddens'), choice('network','networks'), choice('order','orders','layer','layers')))

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 40
AGENT_PROMPT = """
`reset <TARGET_1> [TARGET_2 ...]`
- Resets any requested combination of `colors`, `sizes`, `shapes`, `clusters`, `groups`, `hide`/`hidden`, `network`, and `order`/`layer`; singular and plural target names are accepted. An unknown target fails the whole command, and nothing is reset.
- Visual targets restore configured defaults, cluster/group targets clear those labels (clearing clusters also removes the generated `subcluster_N_M` groups), hide restores visibility, network restores layout positions to the original or most recently saved baseline, and order/layer restores node rendering to index order.
- The command name must precede all targets.
- Never emit `COMMAND reset` shortcuts such as `label reset`, which clears all topology clusters; use `reset TARGET` instead.
"""

# The targets with no command of their own, named as the user types them. A command's own
# targets are declared by its module; "reset help" and the usage line list them all.
RESET_TARGETS = ("size", "shape", "network", "order", "layer")
RESET_HELP = {
    "sizes": (20, "Restores all node sizes to the configured default size."),
    "shapes": (30, "Restores all node shapes to discs."),
    "network": (70, "Restores node positions to the original or last saved layout."),
    "order, layer": (80, "Restores persistent node rendering to internal-index order."),
}

_DECLARATION = b"RESET_TARGETS"
_OWN_NAME = os.path.splitext(os.path.basename(__file__))[0]


class ResetOutcome(NamedTuple):
    """What a module's reset(viewer, target) reports back to the reset command.

    ``reported`` is the word the reply "Reset successful: ..." gives for the target (a
    word named by several targets is given once; None gives none), ``nodes`` and ``edges``
    say that the nodes, or the edges, have to be drawn again, and ``note`` is a Message
    that the reply adds after the targets.
    """
    reported: Optional[str] = None
    nodes: bool = False
    edges: bool = False
    note: Optional[Message] = None


def print_help():
    # The targets are the modules' to declare, so their list is put together when the help is printed, in the
    # place of the line that says so. The catalog reads this text from the source, and shows that line.
    print("""
    Network Reset Tool
    ==================
    Usage:
      reset <TARGET_1> [TARGET_2] ...
      reset help

    Targets:
      <each target the commands declare, with what it restores>

    Notes:
      Multiple targets may be reset in one command. Singular and plural target
      names are accepted. The reset is stored as one undoable action. An
      unknown target aborts the command, and nothing is reset.

    Examples:
      reset network hide
      reset colors sizes shapes
      reset order
    """.replace("\n      <each target the commands declare, with what it restores>", targets_help()))


def _declaring_modules():
    """The modules of the commands folder that declare RESET_TARGETS, other than this one, by name."""
    import commands

    modules = []
    for info in pkgutil.iter_modules(commands.__path__):
        if info.ispkg or info.name.startswith("_") or info.name == _OWN_NAME:
            continue
        try:
            origin = info.module_finder.find_spec(f"commands.{info.name}").origin
            with open(origin, "rb") as source:
                if _DECLARATION not in source.read():
                    continue
        except (AttributeError, OSError, TypeError):
            continue  # not a source file this can read: it declares nothing
        module = Command_Declarations.command_module(info.name)
        if module is not None:
            modules.append(module)
    return modules


def declared_targets():
    """Every target the modules declare: {name: [the modules that declare it, by name]}."""
    modules = sorted([sys.modules[__name__], *_declaring_modules()], key=lambda module: module.__name__)
    owners = {}
    for module in modules:
        names = getattr(module, "RESET_TARGETS", None)
        if not isinstance(names, (tuple, list)) or not callable(getattr(module, "reset", None)):
            continue
        for name in names:
            if isinstance(name, str) and name:
                owners.setdefault(name.lower(), []).append(module)
    return owners


def target_names():
    """The names of every target that can be reset."""
    return frozenset(declared_targets())


def listed_targets(owners=None):
    """The targets "reset help" lists, in order: [(words as typed, description or None)].

    Each module's RESET_HELP says what to list for its targets (see the module's docstring). What is
    described comes first, by rank, then by the modules' names and the order of each module's entries; the
    targets that nothing describes follow, by name. An entry that cannot be used is left out, and its targets
    are listed with no description.
    """
    if owners is None:
        owners = declared_targets()
    modules = sorted({module for declaring in owners.values() for module in declaring}, key=lambda module: module.__name__)
    described, covered = [], set()
    for module in modules:
        declared = getattr(module, "RESET_HELP", None)
        for position, (key, value) in enumerate(declared.items() if isinstance(declared, dict) else ()):
            words = [word.strip() for word in key.split(",") if word.strip()] if isinstance(key, str) else []
            names = [target_name(word, owners) for word in words]
            if not words or any(module not in owners.get(name, ()) for name in names):
                continue  # it names nothing this module resets
            if value is None:
                covered.update(names)  # an alternative spelling: accepted, not listed
                continue
            rank, text = value if isinstance(value, (tuple, list)) and len(value) == 2 else (math.inf, value)
            if (isinstance(rank, bool) or not isinstance(rank, (int, float)) or math.isnan(rank)
                    or not isinstance(text, str) or not text.strip()):
                continue
            covered.update(names)
            described.append(((rank, module.__name__, position), words, text))
    lines = [(words, text) for _order, words, text in sorted(described, key=lambda item: item[0])]
    return lines + [([name], None) for name in sorted(owners) if name not in covered]


def targets_help(owners=None):
    """The lines of "reset help" that list the targets, each after a new line: the words, then the description."""
    text = ""
    for words, description in listed_targets(owners):
        text += "\n      " + ", ".join(words)
        for line in description.splitlines() if description else ():
            text += "\n          " + line
    return text


def usage(owners=None):
    """The usage line of a refusal: every word of every target that "reset help" lists, as a user types them."""
    return "Usage: reset [" + " | ".join(word for words, _description in listed_targets(owners) for word in words) + "]"


def target_name(target, owners=None):
    """Return the name a typed target stands for ("Colors" -> "color").

    The word itself when a module declares it, otherwise the word without a final "s".
    """
    word = target.lower()
    if owners is None:
        owners = declared_targets()
    if word in owners:
        return word
    return word[:-1] if word.endswith("s") else word


def check_targets(targets, owners=None):
    """Raise ValueError unless at least one target is given and every target is known.

    The error's first line is a Message, which the console line shows
    translated; the usage after it is for the terminal.
    """
    if owners is None:
        owners = declared_targets()
    if not targets:
        raise ValueError(JoinedMessage([Message("Specify at least one reset target."), usage(owners)], separator="\n"))
    unknown = [target for target in targets if target_name(target, owners) not in owners]
    if unknown:
        raise ValueError(JoinedMessage([
            Message("Unknown reset target(s): {targets}. Nothing was reset.", targets=", ".join(unknown)),
            usage(owners),
        ], separator="\n"))


def reset_node_render_order(viewer):
    """Restore the persistent node layer to stable internal-index order."""
    identity_order = np.arange(viewer.n_nodes, dtype=np.int32)
    changed = not np.array_equal(
        getattr(viewer, 'node_render_order', identity_order), identity_order
    )
    viewer.node_render_order = identity_order
    return changed


def reset_changes_state(viewer, target):
    """This module's targets: False only when resetting leaves everything an undo step saves as it is."""
    if target == "size":
        sizes = getattr(viewer, 'current_sizes', None)
        return sizes is not None and bool(np.any(sizes != np.asarray(cfg.NODE_SIZE, dtype=sizes.dtype)))
    if target == "shape":
        shapes = getattr(viewer, 'current_shapes', None)
        return shapes is not None and bool(np.any(shapes != 'disc'))
    if target == "network":
        original = getattr(viewer, 'original_pos', None)
        if original is None:
            return False
        pos = getattr(viewer, 'pos', None)
        return pos is None or not np.array_equal(pos, original)
    identity_order = np.arange(viewer.n_nodes, dtype=np.int32)
    return not np.array_equal(getattr(viewer, 'node_render_order', identity_order), identity_order)


def reset(viewer, target):
    """Reset one of this module's targets: node sizes, node shapes, the layout, or the render order."""
    if target == "size":
        if hasattr(viewer, 'current_sizes'):
            viewer.current_sizes.fill(cfg.NODE_SIZE)
        return ResetOutcome("sizes", nodes=True)
    if target == "shape":
        if hasattr(viewer, 'current_shapes'):
            viewer.current_shapes.fill('disc')
        return ResetOutcome("shapes", nodes=True)
    if target == "network":
        if hasattr(viewer, 'original_pos'):
            viewer.pos = viewer.original_pos.copy()
        return ResetOutcome("network", nodes=True, edges=True)
    reset_node_render_order(viewer)
    return ResetOutcome("node order", nodes=True)


def _changes_state(module, viewer, name):
    """Whether resetting target ``name`` may change what the module's state saves: yes unless it says not."""
    function = getattr(module, "reset_changes_state", None)
    return True if function is None else function(viewer, name)


def execute_reset(viewer, targets):
    """Executes reset on the specified targets.

    Raises ValueError, before any undo state is saved or anything is reset,
    when no target is given or any target is unknown. Undo state is saved
    only when a target has something to reset.
    """
    owners = declared_targets()
    check_targets(targets, owners)
    names = [target_name(target, owners) for target in targets]

    # A reset that changes nothing must not push an undo entry: the history
    # holds 50, and an empty one would push out a real one.
    if any(_changes_state(module, viewer, name) for name in names for module in owners[name]):
        viewer._save_state()

    reported, notes = [], []
    redraw_nodes = redraw_edges = False
    for name in names:
        for module in owners[name]:
            outcome = module.reset(viewer, name)
            if outcome is None:
                continue
            if getattr(outcome, "reported", None):
                reported.append(outcome.reported)
            if getattr(outcome, "note", None) is not None:
                notes.append(outcome.note)
            redraw_nodes = redraw_nodes or bool(getattr(outcome, "nodes", False))
            redraw_edges = redraw_edges or bool(getattr(outcome, "edges", False))

    if redraw_nodes:
        viewer.update_nodes()
    if redraw_edges:
        viewer.update_edges()

    # The targets are the reset command's own words, which stay English. A
    # target named twice is reported once.
    msg = Message("Reset successful: {targets}.", targets=", ".join(dict.fromkeys(reported)))
    if notes:
        msg = JoinedMessage([msg, *notes])

    Command_Engine.show_status(viewer, msg)
    print(f"{msg}")
    if hasattr(viewer, 'update_console_background'):
        viewer.update_console_background()

    return str(msg)


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return

    # Checked here, before the reset, so that a refusal is this command's failure;
    # execute_reset checks too, for the commands that call it ("hide reset").
    try:
        check_targets(args)
    except ValueError as error:
        msg = Message("Error: {error}", error=error)
        Command_Engine.command_failed(viewer, msg)
        # command_failed has reported the message; print_help only shows it.
        Command_Engine.print_help(viewer, msg, report_message=False)
        return

    msg = Command_Engine.execute_reset(viewer, args)
    Command_Engine.command_succeeded(viewer, msg)
