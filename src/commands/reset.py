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

import Command_Engine
import numpy as np

# The target names execute_reset handles, after target_name() normalizes a
# typed target.
TARGET_NAMES = frozenset({
    "color", "size", "shape", "cluster", "group",
    "hide", "hidden", "network", "order", "layer",
})
USAGE = "Usage: reset [colors | sizes | shapes | clusters | groups | hide | network | order | layer]"


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
          Clears all cluster labels.
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


def target_name(target):
    """Return the name execute_reset matches for one typed target ("Colors" -> "color")."""
    target = target.lower()
    return target[:-1] if target.endswith('s') else target


def check_targets(targets):
    """Raise ValueError unless at least one target is given and every target is known."""
    if not targets:
        raise ValueError(f"Specify at least one reset target.\n{USAGE}")
    unknown = [target for target in targets if target_name(target) not in TARGET_NAMES]
    if unknown:
        raise ValueError(f"Unknown reset target(s): {', '.join(unknown)}. Nothing was reset.\n{USAGE}")


def reset_node_render_order(viewer):
    """Restore the persistent node layer to stable internal-index order."""
    identity_order = np.arange(viewer.n_nodes, dtype=np.int32)
    changed = not np.array_equal(
        getattr(viewer, 'node_render_order', identity_order), identity_order
    )
    viewer.node_render_order = identity_order
    return changed


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, "Help information printed to the terminal")
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # Checked here as well as in execute_reset: the VR viewer runs this command
    # with its own execute_reset.
    try:
        check_targets(args)
    except ValueError as error:
        msg = f"Error: {error}"
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    msg = Command_Engine.execute_reset(viewer, args)
    Command_Engine.command_succeeded(viewer, msg)
