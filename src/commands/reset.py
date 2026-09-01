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
      names are accepted. The reset is stored as one undoable action.

    Examples:
      reset network hide
      reset colors sizes shapes
      reset order
    """)


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
            viewer.console_text.text = "Help information printed to the terminal"
        return

    Command_Engine.execute_reset(viewer, args)
