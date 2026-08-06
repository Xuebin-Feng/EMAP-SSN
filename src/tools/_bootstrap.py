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

"""Make the sibling ``utilities`` package importable for direct tool runs."""

from __future__ import annotations

import os
import sys


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Reuse support modules that a legacy caller loaded from ``src/utilities`` as
# top-level modules.  This avoids loading the same module twice while callers
# migrate to the package-qualified imports used by the relocated tools.
for _module_name in (
    "Alignment_Score_Kernels",
    "Embedding_HDF5",
    "FASTA_Sanitization",
    "Hardware_Utils",
    "Layout_Hardware",
    "PLM_Plugin_Utils",
):
    _legacy_module = sys.modules.get(_module_name)
    if _legacy_module is not None:
        sys.modules.setdefault(f"utilities.{_module_name}", _legacy_module)
