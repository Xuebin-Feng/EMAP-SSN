# Copyright 2026 Xuebin Feng
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


def reset_node_render_order(viewer):
    """Restore the persistent node layer to stable internal-index order."""
    identity_order = np.arange(viewer.n_nodes, dtype=np.int32)
    changed = not np.array_equal(
        getattr(viewer, 'node_render_order', identity_order), identity_order
    )
    viewer.node_render_order = identity_order
    return changed

def run(viewer, args):
    Command_Engine.execute_reset(viewer, args)
