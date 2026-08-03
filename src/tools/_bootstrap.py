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
