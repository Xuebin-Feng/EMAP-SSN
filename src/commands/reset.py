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

Which modules declare is found by reading the source of the commands folder for the word
RESET_TARGETS, so only the modules that mention it are imported. The targets with no
command of their own, the node sizes and shapes, the layout and the render order, are
declared below like any other.
"""

import os
import pkgutil
import sys
from typing import NamedTuple, Optional

import Command_Declarations
import Command_Engine
import EMAPSSN_Config as cfg
import numpy as np
from utilities.Localization import JoinedMessage, Message

# The targets with no command of their own, named as the user types them. A command's own
# targets are declared by its module, and are listed in print_help and USAGE by hand.
RESET_TARGETS = ("size", "shape", "network", "order", "layer")
USAGE = "Usage: reset [colors | sizes | shapes | clusters | groups | hide | network | order | layer]"

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
    print("""
    Network Reset Tool
    ==================
    Usage:
      reset <TARGET_1> [TARGET_2] ...
      reset help

    Targets:
      colors
          Restores all node colors to the configured default color.
      sizes
          Restores all node sizes to the configured default size.
      shapes
          Restores all node shapes to discs.
      clusters
          Clears all cluster labels, and removes the subcluster groups
          (subcluster_N_M) the subcluster command generated for them. Custom
          groups with lookalike names, such as subcluster_0_2, are kept.
      groups
          Clears all group labels.
      hide
          Makes all hidden nodes visible.
      network
          Restores node positions to the original or last saved layout.
      order, layer
          Restores persistent node rendering to internal-index order.

    Notes:
      Multiple targets may be reset in one command. Singular and plural target
      names are accepted. The reset is stored as one undoable action. An
      unknown target aborts the command, and nothing is reset.

    Examples:
      reset network hide
      reset colors sizes shapes
      reset order
    """)


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
        raise ValueError(JoinedMessage([Message("Specify at least one reset target."), USAGE], separator="\n"))
    unknown = [target for target in targets if target_name(target, owners) not in owners]
    if unknown:
        raise ValueError(JoinedMessage([
            Message("Unknown reset target(s): {targets}. Nothing was reset.", targets=", ".join(unknown)),
            USAGE,
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
