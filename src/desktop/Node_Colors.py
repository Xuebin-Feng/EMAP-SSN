# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Colors the Viewer's commands share, so that one meaning looks the same whichever command paints it.

This module is not shadowed by the VR viewer, which re-exports Command_Engine's names one by one, so
the commands the two viewers share import such constants from here.
"""

# RGBA of a node with no value to color it by or no group to belong to: translucent light gray.
NOISE_COLOR = (0.8, 0.8, 0.8, 0.4)
