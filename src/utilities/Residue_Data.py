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

"""Per-residue data that travels with a folded structure.

`esmfold` computes it in the Viewer, which holds the loaded alignment, and hands
it to the folding worker in the worker's input file. The worker writes it beside
each PDB as ``<pdb stem>.residues.json`` (see esmfold_worker.build_sidecar), and
the Mol* page reads that file.

A property is one number, or null, per residue of the folded protein. Each one
comes from a provider registered in PROVIDERS under the property's id:

    def my_property(viewer_inputs, node_index, sequence):
        ...
        return {"label": ..., "unit": ..., "range": [low, high],
                "description": ..., "values": [...], "source": {...}}

A provider returns the property (everything but its id), or None when it has
nothing to say. When it has nothing because the data does not exist, it raises
ResidueDataUnavailable, whose kind and reason the command reports. To add a
property, write the function and add one line to PROVIDERS; the order of
PROVIDERS is the order of the file, and of the layers the page lists.

The module is Qt-free: the Viewer's command uses it, and so can a script.
"""

import hashlib
import json
import os
import weakref
from dataclasses import dataclass, field

import numpy as np

from utilities.Alignment_Profiles import STANDARD_AAS, logo_column_heights


# The 20 standard amino acids, in the order the logo command counts them.
STANDARD_RESIDUES = "".join(STANDARD_AAS)
_NOT_STANDARD = len(STANDARD_RESIDUES)

# Why a property is missing (ResidueDataUnavailable.kind).
NO_ALIGNMENT = "no_alignment"
NOT_IN_ALIGNMENT = "not_in_alignment"
SEQUENCE_DIFFERS = "sequence_differs"
FAILED = "failed"

# Alignment entries are counted in blocks, so the temporaries stay small however large the MSA is.
_BLOCK_ENTRIES = 1 << 22


class ResidueDataUnavailable(Exception):
    """A provider has no data for this node. ``kind`` is one of the constants above, ``reason`` English."""

    def __init__(self, kind, reason):
        super().__init__(reason)
        self.kind = kind
        self.reason = reason


@dataclass(frozen=True)
class ViewerInputs:
    """What the providers may read from the Viewer.

    ``alignment`` is the Viewer's Alignment_Manager (its ``aln`` rows and
    ``msa_file``), or None when no alignment is loaded. ``viewer_to_aln`` is
    Command_Engine.get_alignment_mapping's first value: for each node, its row
    in ``alignment.aln``, or -1 when the alignment lacks it.
    """

    alignment: object = None
    viewer_to_aln: object = ()
    full_headers: tuple = ()


@dataclass
class ResidueData:
    """One node's providers' results: ``properties`` by id, ``unavailable`` (ResidueDataUnavailable) by id."""

    properties: dict = field(default_factory=dict)
    unavailable: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# What the loaded alignment says about each of its columns
# ---------------------------------------------------------------------

def _csr(matrix):
    """The alignment's matrix as a CSR matrix with sorted column indices (itself when it already is one)."""
    from scipy import sparse

    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(np.asarray(matrix))
    elif matrix.format != "csr":
        matrix = matrix.tocsr()
    if not matrix.has_sorted_indices:
        matrix = matrix.sorted_indices()
    return matrix


def _code_tables(int_to_aa):
    """(standard-residue index by residue code, symbol by code) for an alignment's code table.

    The index is 0-19 for a standard amino acid and 20 for a gap or any other
    symbol, and the table is longer than any code, so a code that is not in
    int_to_aa reads as "not a residue" when clipped to its last entry.
    """
    top = max((int(code) for code in int_to_aa), default=0)
    index = np.full(max(top + 2, 256), _NOT_STANDARD, dtype=np.int8)
    symbols = {}
    for code, residue in int_to_aa.items():
        code = int(code)
        if code <= 0:
            continue  # 0 is a gap.
        text = str(residue).upper()
        symbols[code] = text
        if len(text) == 1 and text in STANDARD_RESIDUES:
            index[code] = STANDARD_RESIDUES.index(text)
    return index, symbols


def _column_counts(matrix, residue_index):
    """How many rows have each standard amino acid in each column: an (n_columns, 20) array.

    Gaps and every other symbol are not counted. One pass over the stored
    entries, in blocks, with no per-row or per-column loop.
    """
    n_columns = matrix.shape[1]
    counts = np.zeros(n_columns * (_NOT_STANDARD + 1), dtype=np.int64)
    clip = len(residue_index) - 1
    low, high = int(matrix.indptr[0]), int(matrix.indptr[-1])
    for start in range(low, high, _BLOCK_ENTRIES):
        end = min(start + _BLOCK_ENTRIES, high)
        residues = residue_index[np.minimum(matrix.data[start:end], clip)]
        keys = matrix.indices[start:end].astype(np.intp) * (_NOT_STANDARD + 1) + residues
        counts += np.bincount(keys, minlength=counts.size)
    return counts.reshape(n_columns, _NOT_STANDARD + 1)[:, :_NOT_STANDARD]


def _content_hash(aln, matrix, int_to_aa):
    """SHA-256 over what the loaded alignment holds: its shape, headers, symbol table and every entry.

    The same loaded rows always give the same digest, whichever file or run
    they came from; a different row selection, MSA or symbol table does not.
    """
    digest = hashlib.sha256(b"emapssn-alignment-1\n")
    digest.update(f"{matrix.shape[0]}x{matrix.shape[1]}\n".encode("ascii"))
    for header in getattr(aln, "headers", ()):
        digest.update(str(header).encode("utf-8", errors="surrogatepass") + b"\n")
    digest.update(json.dumps({str(code): symbol for code, symbol in sorted(int_to_aa.items())}).encode("utf-8"))
    digest.update(np.ascontiguousarray(matrix.indptr, dtype="<i8").tobytes())
    entry_type = "<u1" if matrix.data.dtype.itemsize == 1 else "<i4"
    low, high = int(matrix.indptr[0]), int(matrix.indptr[-1])
    for start in range(low, high, _BLOCK_ENTRIES):
        end = min(start + _BLOCK_ENTRIES, high)
        digest.update(np.ascontiguousarray(matrix.indices[start:end], dtype="<i4").tobytes())
        digest.update(np.ascontiguousarray(matrix.data[start:end], dtype=entry_type).tobytes())
    return "sha256:" + digest.hexdigest()


class _AlignmentStats:
    """Everything the providers need from one loaded alignment, worked out once per alignment.

    It never refers to the alignment object itself, so the cache that holds it
    (see _stats_for) does not keep the alignment alive.
    """

    def __init__(self, aln):
        self.source_matrix = aln.matrix
        self.int_to_aa = dict(getattr(aln, "int_to_aa", {}) or {})
        self.matrix = _csr(aln.matrix)
        self.n_rows, self.n_columns = self.matrix.shape
        self.residue_index, self.symbols = _code_tables(self.int_to_aa)
        self.counts = _column_counts(self.matrix, self.residue_index)
        self.totals = self.counts.sum(axis=1)
        self._information_pct = None
        self._identity = None

    def matches(self, aln):
        return self.source_matrix is aln.matrix and self.int_to_aa == dict(getattr(aln, "int_to_aa", {}) or {})

    def identity(self, aln):
        """The alignment's content digest, taken on first use."""
        if self._identity is None:
            self._identity = _content_hash(aln, self.matrix, self.int_to_aa)
        return self._identity

    @property
    def information_pct(self):
        """Each column's information content as a percentage of log2(20) bits: the logo command's default heights."""
        if self._information_pct is None:
            self._information_pct = logo_column_heights(self.counts, self.n_rows) / np.log2(20) * 100.0
        return self._information_pct


_stats_cache = weakref.WeakKeyDictionary()
# The last alignment that cannot be weak-referenced (a stand-in, never the Viewer's loader), kept strongly.
_unweakable_stats = None


def _stats_for(aln):
    """The _AlignmentStats of ``aln``, kept for as long as the alignment lives and its rows are unchanged."""
    global _unweakable_stats
    try:
        stats = _stats_cache.get(aln)
        weakable = True
    except TypeError:
        weakable = False
        stats = _unweakable_stats[1] if _unweakable_stats and _unweakable_stats[0] is aln else None
    if stats is None or not stats.matches(aln):
        stats = _AlignmentStats(aln)
        if weakable:
            _stats_cache[aln] = stats
        else:
            _unweakable_stats = (aln, stats)
    return stats


# ---------------------------------------------------------------------
# One node's residues in the alignment
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _NodeResidues:
    stats: _AlignmentStats
    aln: object
    alignment: object
    columns: np.ndarray  # The 0-based alignment column of each residue, residue 1 first.
    residues: np.ndarray  # Each residue's index among STANDARD_RESIDUES, or 20.


def _node_residues(inputs, node_index, sequence):
    """Where each residue of ``sequence`` sits in the alignment's row for the node.

    The k-th symbol of the row that is not a gap is residue k. Raises
    ResidueDataUnavailable when there is no alignment, the alignment lacks the
    node, or the row's residues are not ``sequence`` (compared in upper case),
    since then the columns would belong to some other protein.
    """
    alignment = inputs.alignment
    aln = getattr(alignment, "aln", None)
    if alignment is None or aln is None:
        raise ResidueDataUnavailable(NO_ALIGNMENT, "no alignment is loaded")
    mapping = inputs.viewer_to_aln
    row = int(mapping[node_index]) if 0 <= node_index < len(mapping) else -1
    stats = _stats_for(aln)
    if not 0 <= row < stats.n_rows:
        raise ResidueDataUnavailable(NOT_IN_ALIGNMENT, "it is not in the alignment")

    matrix = stats.matrix
    start, end = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    columns = matrix.indices[start:end].astype(np.int64)
    codes = matrix.data[start:end].astype(np.int64)
    # A code that stands for no symbol is a gap, as in logo's row strings.
    present = np.fromiter((int(code) in stats.symbols for code in codes), dtype=bool, count=len(codes))
    columns, codes = columns[present], codes[present]
    row_text = "".join(stats.symbols[int(code)] for code in codes)
    if row_text != sequence.upper():
        raise ResidueDataUnavailable(
            SEQUENCE_DIFFERS,
            f"its sequence ({len(sequence)} residues) differs from the alignment's residues "
            f"for it ({len(row_text)} residues)",
        )
    residues = stats.residue_index[np.minimum(codes, len(stats.residue_index) - 1)]
    return _NodeResidues(stats, aln, alignment, columns, residues)


def _source(node, **extra):
    """The ``source`` of a property computed from the loaded alignment."""
    msa_file = getattr(node.alignment, "msa_file", None)
    return {
        "kind": "msa",
        "file": os.path.basename(str(msa_file)) if msa_file else None,
        "alignment_id": node.stats.identity(node.aln),
        "columns": node.columns.tolist(),
        "sequences": int(node.stats.n_rows),
        **extra,
    }


# ---------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------

MAX_INFORMATION_BITS = float(np.log2(20))


def information_content(inputs, node_index, sequence):
    """How conserved each residue's alignment column is: the logo command's information content, in % of 4.32 bits."""
    node = _node_residues(inputs, node_index, sequence)
    values = [round(float(value), 2) for value in node.stats.information_pct[node.columns]]
    return {
        "label": "Information Content",
        "unit": "%",
        "range": [0, 100],
        "description": (
            "Information content of this residue's alignment column, as a percentage of the "
            "maximum, log2(20) = 4.32 bits. It is the measure the logo command draws by default: "
            "bits, with the height scaled by the share of aligned sequences that have a standard "
            "amino acid in the column, a small-sample correction, and no identity reweighting. "
            "It is a property of the column, the same for every protein with a residue there, "
            "and it is computed from the whole loaded alignment (hidden nodes included)."
        ),
        "values": values,
        "source": _source(
            node,
            measure="logo bits, with_gap, small-sample corrected",
            max_bits=round(MAX_INFORMATION_BITS, 4),
        ),
    }


def same_residue_pct(inputs, node_index, sequence):
    """The share of the aligned sequences with a standard residue in the column that have this protein's residue."""
    node = _node_residues(inputs, node_index, sequence)
    stats = node.stats
    standard = node.residues < _NOT_STANDARD
    same = np.zeros(len(node.columns), dtype=np.float64)
    same[standard] = stats.counts[node.columns[standard], node.residues[standard]]
    totals = stats.totals[node.columns]
    values = [
        round(float(count) / float(total) * 100.0, 2) if is_standard and total else None
        for count, total, is_standard in zip(same, totals, standard)
    ]
    return {
        "label": "Residue Agreement",
        "unit": "%",
        "range": [0, 100],
        "description": (
            "Share of the aligned sequences with a standard amino acid in this column that have "
            "the same amino acid as this protein. Gaps and non-standard symbols are not counted, "
            "this protein's own row is, and the whole loaded alignment is used (hidden nodes "
            "included). It is specific to this protein's residue: another protein at the same "
            "column has its own value. Empty where this protein has a non-standard symbol."
        ),
        "values": values,
        "source": _source(node),
    }


# Property id -> provider, in the order the properties are written and listed.
PROVIDERS = {
    "information_content": information_content,
    "same_residue_pct": same_residue_pct,
}


def compute_residue_data(inputs, node_index, sequence):
    """Run every provider for one node; return a ResidueData.

    A provider that raises ResidueDataUnavailable is recorded as such, and any
    other error from it is recorded as FAILED: a provider's fault must not
    stop a structure from being folded.
    """
    result = ResidueData()
    for property_id, provider in PROVIDERS.items():
        try:
            data = provider(inputs, node_index, sequence)
        except ResidueDataUnavailable as error:
            result.unavailable[property_id] = error
            continue
        except Exception as error:
            result.unavailable[property_id] = ResidueDataUnavailable(FAILED, f"{type(error).__name__}: {error}")
            continue
        if data is not None:
            result.properties[property_id] = data
    return result
