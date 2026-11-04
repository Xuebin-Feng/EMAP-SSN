# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Where the embedding MSA saves its guide tree, and how its labels are quoted.

The tree of an alignment ``<dir>/<name>_alignment.fasta`` is
``<dir>/Guide_Trees/<name>_guide_tree.nex``. The tool that writes the tree and
anything that looks for it go through guide_tree_path_for_alignment, so both
agree on the place. This module imports no torch, numba or Qt, so a reader can
load it cheaply.
"""

import os
import re

GUIDE_TREE_SUBFOLDER = "Guide_Trees"

_ALIGNMENT_SUFFIXES = ("_sparse.h5", ".h5", ".fasta")
_ALIGNMENT_STEM_END = "_alignment"


def guide_tree_path_for_alignment(msa_path):
    """The guide tree's path for an alignment file, or None when its name does not follow the tool's.

    The name must end in ``_alignment`` followed by ``.fasta``, ``.h5`` or
    ``_sparse.h5`` (the sparse file the converter writes beside a FASTA). The
    folder keeps its spelling, relative or absolute. A renamed or external
    alignment, or an empty path, has no tree.
    """
    if not msa_path or not str(msa_path).strip():
        return None
    folder, filename = os.path.split(os.fspath(msa_path))
    lowered = filename.lower()
    for suffix in _ALIGNMENT_SUFFIXES:
        if lowered.endswith(suffix):
            stem = filename[: len(filename) - len(suffix)]
            break
    else:
        return None
    if not stem.lower().endswith(_ALIGNMENT_STEM_END):
        return None
    name = stem[: len(stem) - len(_ALIGNMENT_STEM_END)]
    if not name:
        return None
    return os.path.join(folder, GUIDE_TREE_SUBFOLDER, f"{name}_guide_tree.nex")


_NEWICK_PLAIN_LABEL = re.compile(r"[A-Za-z0-9.|]+\Z")


def newick_label(text):
    """Quote a label unless it holds only letters, digits, dots and bars.

    NEXUS, like the Newick standard, reads an unquoted underscore as a space
    and splits words at - + * = / < >, so quoting keeps a header as it is.
    """
    if _NEWICK_PLAIN_LABEL.match(text):
        return text
    return nexus_string(text)


def nexus_string(text):
    """Single-quote a NEXUS word, doubling any quote inside it."""
    return "'" + str(text).replace("'", "''") + "'"
