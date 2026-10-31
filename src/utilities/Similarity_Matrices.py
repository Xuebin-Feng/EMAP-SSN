# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The substitution matrices that count similar residue pairs, and the pair table the kernels read.

Pairwise Embedding Alignment and Embedding SSEARCH report percent similarity beside
percent identity. The alignment itself comes from the embeddings; a substitution matrix
only decides which aligned pairs of standard residues count as similar: the pair is
identical, or the matrix scores it above 0 (identity_definition and similarity_definition
in Sequence_Utils word it). X, B, Z, J, U and O never count.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from utilities.Sequence_Utils import IDENTITY_RESIDUES


# The Align_Substitution_Matrix tool's choices, in its order.
SIMILARITY_MATRICES = (
    "BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90",
    "PAM30", "PAM70", "PAM250",
)


def validate_similarity_matrix(matrix_name):
    """Refuse anything but an exact listed name: the name picks a file."""
    if matrix_name not in SIMILARITY_MATRICES:
        raise ValueError(
            f"Unknown SIMILARITY_MATRIX {matrix_name!r}; "
            f"choose {', '.join(SIMILARITY_MATRICES)}."
        )


def load_similarity_matrix(matrix_name):
    """Load a listed substitution matrix (Biopython's, imported on first use)."""
    validate_similarity_matrix(matrix_name)
    from Bio.Align import substitution_matrices

    return substitution_matrices.load(matrix_name)


@lru_cache(maxsize=len(SIMILARITY_MATRICES))
def _pair_table(matrix_name):
    matrix = load_similarity_matrix(matrix_name)
    table = np.zeros((256, 256), dtype=np.uint8)
    for query in IDENTITY_RESIDUES:
        for target in IDENTITY_RESIDUES:
            if query == target or matrix[query, target] > 0:
                table[ord(query), ord(target)] = 1
    table.setflags(write=False)
    return table


def similar_pair_table(matrix_name):
    """``table[query_code, target_code]``: 1 for a similar pair, else 0, as a uint8 (256, 256) array.

    The codes are the ASCII bytes of the residues, as the SSEARCH kernels read them. A
    pair is similar when both residues are among the 20 standard ones and they are
    identical or the matrix scores them above 0, so every identical pair is similar and
    every other code, the unmatched ones included, gives 0. The table is built once per
    matrix and process, and read-only: pass this same array to every kernel call.
    """
    validate_similarity_matrix(matrix_name)
    return _pair_table(matrix_name)


__all__ = [
    "SIMILARITY_MATRICES",
    "validate_similarity_matrix",
    "load_similarity_matrix",
    "similar_pair_table",
]
