# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Import the Align_Similarity_Matrix tool as a library, so Embedding_MSA can align pairs with its path.

Importing a tool module normally changes the importing process: it sets the thread
variables to 1 and applies the saved Tools settings to the module's values. A library use
wants neither, so the import happens with SSN_ALIGN_AS_LIBRARY set, which the module
reads once as it loads. The variable is put back afterwards; the module's own
align_pairs_in_memory sets it again (and the thread variables) for the pool children it
starts, and puts them back.

This module imports no torch, numba or Qt itself.
"""

import importlib
import os

LIBRARY_VARIABLE = "SSN_ALIGN_AS_LIBRARY"


def load():
    """The Align_Similarity_Matrix module, imported without its tool-run side effects."""
    previous = os.environ.get(LIBRARY_VARIABLE)
    os.environ[LIBRARY_VARIABLE] = "1"
    try:
        try:
            return importlib.import_module("tools.Align_Similarity_Matrix")
        except ModuleNotFoundError as error:
            if error.name not in ("tools", "tools.Align_Similarity_Matrix"):
                raise
            return importlib.import_module("Align_Similarity_Matrix")
    finally:
        if previous is None:
            os.environ.pop(LIBRARY_VARIABLE, None)
        else:
            os.environ[LIBRARY_VARIABLE] = previous
