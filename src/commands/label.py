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
import os
import math
import datetime
import tempfile
import numpy as np
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import EMAPSSN_Config as cfg
from utilities.Output_Names import validate_output_basename
from utilities.Sequence_Utils import (
    format_alignment_offset_display,
    sort_alignment_labels,
)
try:
    import commands.cluster as cluster_cmd
    import commands.logo as logo_cmd
except ImportError:
    import cluster as cluster_cmd
    import logo as logo_cmd

CLUSTER_LABEL_DIRECTORY = os.path.join("$analysis_result$", "Cluster_Label")


GLOBAL_CONSERVATION_THRESHOLD = 0.97


class _FrozenSparseAlignment:
    """Detached read-only copy of the sparse alignment data needed by label."""

    def __init__(self, alignment):
        self.matrix = alignment.matrix.copy()
        self.int_to_aa = dict(alignment.int_to_aa)
        self.headers = tuple(getattr(alignment, "headers", ()))
        self.n_seqs, self.n_cols = self.matrix.shape

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, index):
        row = self.matrix[index].toarray()[0]
        sequence = "".join(
            self.int_to_aa.get(int(value), "-") if value != 0 else "-"
            for value in row
        )
        description = (
            self.headers[index] if index < len(self.headers) else f"row_{index}"
        )
        return SeqRecord(
            Seq(sequence),
            id=description.split()[0],
            description=description,
        )

    def __iter__(self):
        for index in range(self.n_seqs):
            yield self[index]

    def get_alignment_length(self):
        return self.n_cols

    def column_major(self):
        """CSC copy of the frozen matrix (rows ascending in every column), built once."""
        if getattr(self, "_column_major", None) is None:
            self._column_major = self.matrix.tocsc()
            self._column_major.sort_indices()
        return self._column_major


class _AlignmentRows:
    """Rows of a sparse alignment, in order, without decoding them to text.

    The row-subset matrix is gathered on each access rather than cached, so
    the task list does not keep every subset's copy alive.
    """

    def __init__(self, alignment, row_indices):
        self.parent = alignment
        self.row_indices = np.asarray(row_indices, dtype=np.intp)
        self.int_to_aa = alignment.int_to_aa

    @property
    def matrix(self):
        return self.parent.matrix[self.row_indices]

    def __len__(self):
        return int(self.row_indices.size)

    def __getitem__(self, index):
        return self.parent[int(self.row_indices[index])]

    def __iter__(self):
        for index in self.row_indices:
            yield self.parent[int(index)]

    def get_alignment_length(self):
        return self.parent.get_alignment_length()


def _sparse_code_limit(matrix):
    """Size of a lookup table covering every stored code, or None."""
    data = matrix.data
    if not np.issubdtype(data.dtype, np.integer):
        return None
    if data.size == 0:
        return 1
    low, high = int(data.min()), int(data.max())
    if low < 0 or high > 65535:
        return None
    return high + 1


def _count_letter_table(int_to_aa, gap_chars, code_limit):
    """Map sparse codes to residue letters the way _get_amino_acid_counts reads them.

    Unmapped codes count as 'X', symbols in gap_chars are skipped and residues
    are upper-cased. Returns (code -> letter index or -1, letters).
    """
    letters = []
    letter_index = {}
    table = np.full(max(1, code_limit), -1, dtype=np.int32)
    for code in range(1, code_limit):
        amino_acid = int_to_aa.get(code, 'X')
        if amino_acid in gap_chars:
            continue
        amino_acid = str(amino_acid).upper()
        if amino_acid not in letter_index:
            letter_index[amino_acid] = len(letters)
            letters.append(amino_acid)
        table[code] = letter_index[amino_acid]
    return table, letters


_COLUMN_BLOCK = 128
_ROW_MAJOR_ENTRY_LIMIT = 4_000_000


def _sparse_column_counts(aln, columns, weights, gap_chars):
    """Return {column: {aa: weighted count}} for a sparse alignment, or None.

    One pass over the stored entries replaces a scan of the whole row-major
    matrix per column. Within every (column, residue) bin np.bincount adds the
    weights in ascending row order, as the per-column loop does, and residues
    keep the order of their first row, so sums and dict order are unchanged.
    Small matrices are read in row-major order directly; large ones in column
    blocks of a column-major copy (cached on the frozen global alignment).
    """
    matrix = aln.matrix
    code_limit = _sparse_code_limit(matrix)
    if code_limit is None or getattr(matrix, "format", None) != "csr":
        return None
    table, letters = _count_letter_table(aln.int_to_aa, gap_chars, code_limit)
    letter_count = max(1, len(letters))
    row_count, column_count = matrix.shape
    bin_total = column_count * letter_count
    sums = np.zeros(bin_total, dtype=float)
    first_entry = np.full(bin_total, np.iinfo(np.int64).max, dtype=np.int64)
    # Skip the gap filter when no stored code is a gap (the loaders' case), and
    # count instead of summing unit weights: the integer totals are exact.
    all_residues = bool(np.all(table[1:] >= 0)) and (
        matrix.nnz == 0 or int(matrix.data.min()) > 0
    )
    unit_weights = bool(np.all(weights == 1.0))

    def bin_sums(keys, entry_rows):
        if unit_weights:
            return np.bincount(keys, minlength=bin_total).astype(float)
        return np.bincount(keys, weights=weights[entry_rows()], minlength=bin_total)

    if matrix.nnz <= _ROW_MAJOR_ENTRY_LIMIT:
        # Row-major entries: a bin's entries arrive in ascending row order.
        letter = table[matrix.data]
        if all_residues:
            kept = np.arange(matrix.nnz, dtype=np.int64)
            keys = matrix.indices.astype(np.int64)
        else:
            kept = np.flatnonzero(letter >= 0)
            keys = matrix.indices[kept].astype(np.int64)
            letter = letter[kept]
        keys *= letter_count
        keys += letter
        sums += bin_sums(keys, lambda: np.repeat(
            np.arange(row_count, dtype=np.int64), np.diff(matrix.indptr)
        )[kept])
        np.minimum.at(first_entry, keys, kept)
    else:
        column_major = getattr(aln, "column_major", None)
        csc = column_major() if column_major is not None else matrix.tocsc()
        csc.sort_indices()
        for block_start in range(0, column_count, _COLUMN_BLOCK):
            block_end = min(column_count, block_start + _COLUMN_BLOCK)
            entry_start = int(csc.indptr[block_start])
            entry_end = int(csc.indptr[block_end])
            letter = table[csc.data[entry_start:entry_end]]
            local_column = np.repeat(
                np.arange(block_end - block_start, dtype=np.int64),
                np.diff(csc.indptr[block_start:block_end + 1]),
            )
            if all_residues:
                kept = np.arange(entry_end - entry_start, dtype=np.int64)
                keys = local_column
            else:
                kept = np.flatnonzero(letter >= 0)
                keys = local_column[kept]
                letter = letter[kept]
            keys *= letter_count
            keys += letter
            bins = slice(block_start * letter_count, block_end * letter_count)
            block_bins = (block_end - block_start) * letter_count
            if unit_weights:
                sums[bins] = np.bincount(keys, minlength=block_bins)
            else:
                sums[bins] = np.bincount(
                    keys,
                    weights=weights[csc.indices[entry_start:entry_end][kept]],
                    minlength=block_bins,
                )
            np.minimum.at(first_entry[bins], keys, kept)

    present = np.flatnonzero(first_entry != np.iinfo(np.int64).max)
    present = present[np.lexsort((first_entry[present], present // letter_count))]
    present_columns = present // letter_count
    starts = np.flatnonzero(np.diff(present_columns, prepend=-1))
    ends = np.append(starts[1:], present.size).tolist()
    names = np.asarray(letters, dtype=object)[present % letter_count].tolist()
    totals = sums[present].tolist()
    by_column = {
        column: dict(zip(names[start:end], totals[start:end]))
        for column, start, end in zip(
            present_columns[starts].tolist(), starts.tolist(), ends
        )
    }
    return {
        column: by_column.get(column, {})
        for column in (int(column) for column in columns)
        if 0 <= column < column_count
    }


def _residue_lengths(aln, gap_chars):
    """Ungapped length of every row, as get_sequence_stats counts decoded rows, or None."""
    matrix = aln.matrix
    if not hasattr(matrix, "indptr") or matrix.format != "csr":
        return None
    code_limit = _sparse_code_limit(matrix)
    if code_limit is None:
        return None
    residues_per_code = np.zeros(code_limit, dtype=np.int64)
    for code in range(1, code_limit):
        decoded = aln.int_to_aa.get(code, "-")
        residues_per_code[code] = sum(1 for char in decoded if char not in gap_chars)
    lengths = np.diff(matrix.indptr).astype(np.int64)
    correction = residues_per_code - 1
    # Every stored entry is one residue unless a code decodes to a gap symbol
    # or to several characters (not the case for the loaders' canonical codes).
    if matrix.nnz and (int(matrix.data.min()) == 0 or np.any(correction[1:] != 0)):
        entries = np.flatnonzero(correction[matrix.data] != 0)
        rows = np.searchsorted(matrix.indptr, entries, side="right") - 1
        lengths += np.bincount(
            rows,
            weights=correction[matrix.data[entries]],
            minlength=lengths.size,
        ).astype(np.int64)
    return lengths


def _standard_residue_rows(aln, block_rows=4096):
    """Rows encoded for logo.calculate_identity_weights (0-19 or -1), or None.

    Equal to encoding [str(record.seq).upper() for record in aln] without
    building the strings.
    """
    matrix = getattr(aln, "matrix", None)
    if matrix is None or getattr(matrix, "format", None) != "csr":
        return None
    code_limit = _sparse_code_limit(matrix)
    if code_limit is None:
        return None
    table = np.full(code_limit, -1, dtype=np.int8)
    for code in range(1, code_limit):
        decoded = aln.int_to_aa.get(code, "-")
        if not isinstance(decoded, str) or len(decoded) != 1:
            return None
        decoded = decoded.upper()
        if decoded in logo_cmd.STANDARD_AAS:
            table[code] = logo_cmd.STANDARD_AAS.index(decoded)
    row_count, column_count = matrix.shape
    encoded = np.empty((row_count, column_count), dtype=np.int8)
    for start in range(0, row_count, block_rows):
        stop = min(row_count, start + block_rows)
        encoded[start:stop] = table[matrix[start:stop].toarray()]
    return encoded


class _FrozenAlignmentManager:
    def __init__(self, alignment, viewer_to_aln=None):
        self.aln = _FrozenSparseAlignment(alignment.aln)
        self.col_to_label = dict(alignment.col_to_label)
        self.label_to_col = dict(alignment.label_to_col)
        self.has_reference = bool(getattr(alignment, "has_reference", False))
        self.resolved_ref_full = getattr(alignment, "resolved_ref_full", None)
        if viewer_to_aln is None:
            viewer_to_aln = getattr(alignment, "viewer_to_aln", ())
        self.viewer_to_aln = np.asarray(viewer_to_aln, dtype=int).copy()
        self.viewer_to_aln.setflags(write=False)

@dataclass(frozen=True)
class _LabelJobEnvelope:
    viewer_snapshot: object
    args: tuple


def _setting(viewer, name, default=None):
    settings = getattr(viewer, "_label_settings", None)
    if settings is not None and name in settings:
        return settings[name]
    return getattr(cfg, name, default)


def print_help():
    print("""
    Differential Labeling & Statistics Tool
    =======================================
    Generates a comprehensive XLSX report comparing the sequence properties and 
    conserved residues of each subset against the global dataset. Output is saved 
    beneath the configured Analysis Results directory
    (default: 'Analysis_Results/Cluster_Label/').

    * PREREQUISITES: 
      1. A Multiple Sequence Alignment (MSA) must be loaded.
      2. A Reference Sequence must be set (use the 'reference' command).

    Usage: label [TARGET] [GLOBAL_MAX] [CLUSTER_MIN] [IDENTITY] [NAME]
       or: label [TARGET] [key value] [<key 2> <value 2> ...] [NAME]

    Targets (Default: all available results):
      cluster / clusters : Analyzes ONLY defined topology clusters.
      group / groups     : Analyzes ONLY custom groups.
      omitted            : Analyzes all available clusters and custom groups.

    Arguments (a fraction '0.4' or a percentage '40' or '40%'; '0.5%' is 0.5%):
      gmax (Outside Max)  : Default 40%. Max frequency a conserved residue can
                            have outside the union of all analyzed subsets where
                            that same residue meets cmin at the same position.
      cmin (Cluster Min)  : Default 98%. Min frequency a residue must have WITHIN 
                            a subset to be reported as conserved.
      id (Identity)       : Optional sequence-redundancy threshold. Equivalent
                            forms: 0.9, 90, or 90%. Reweighting is OFF unless
                            supplied. Without the 'id' keyword, identity must be
                            the third positional number after gmax and cmin.
      NAME                 : Optional final XLSX filename. '.xlsx' is added if
                            omitted. Numeric or reserved names must include the
                            extension, for example '0.4.xlsx' or 'groups.xlsx'.
                            An existing custom filename is replaced. Automatic
                            names use a numeric suffix instead of overwriting.

    Fixed behavior:
      Every amino acid meeting cmin is evaluated. Conserved subsets sharing the
      same amino acid and position use one deduplicated exclusion union; if that
      union leaves no outside sequences, the residue is not subset specific.
      Multiple passing amino acids share one workbook cell (for example,
      "Y120 | F120") in descending subset-frequency order.
      Globally conserved residues are reported when their frequency is greater
      than 97% across all aligned sequences. This threshold is not configurable.
      Label and logo jobs share one sequential background queue. Alignment,
      memberships, reference numbering, and parameters are captured on submission.

    Examples:
      label                       (Uses gmax=40%, cmin=98%, and timestamp naming)
      label 0.4 0.9               (Positional: gmax=40%, cmin=90%)
      label id 90%                 (Uses default gmax/cmin and 90% identity weights)
      label 0.4 0.9 90% report    (Sets gmax, cmin, identity, and filename)
      label cluster cmin 90%      (Analyzes only topology clusters)
      label clusters report       (Cluster-only report; writes report.xlsx)
      label group cmin 90%        (Analyzes only custom groups)
      label groups cmin 90%       (Keyword: Analyzes groups, sets cmin to 90%)
      label groups report         (Writes report.xlsx)
      label 0.4 0.9 report        (Sets thresholds and writes report.xlsx)
      
    Note: Do not mix positional numbers after using keywords. The first two
          positional numbers remain gmax and cmin. A custom filename must be final.
    """)

def parse_percentage(val_str):
    """Read a threshold written as a fraction (0.4) or a percentage (40 or 40%).

    A trailing % always means percent, so "0.5%" is 0.005 and "1%" is 0.01;
    without one, values above 1 are percentages, as in `logo` and `query`.
    Returns None for anything that is not a finite number.
    """
    text = str(val_str).strip()
    is_percent = text.endswith('%')
    try:
        value = float(text[:-1].strip() if is_percent else text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if is_percent or value > 1.0:
        value /= 100.0
    return value


def _is_non_finite_number(val_str):
    """Whether val_str reads as NaN or an infinity, with or without a trailing %.

    parse_percentage returns None for these as for words, but they are numbers,
    so a positional one is rejected rather than taken as the report filename.
    """
    text = str(val_str).strip()
    if text.endswith('%'):
        text = text[:-1].strip()
    try:
        return not math.isfinite(float(text))
    except ValueError:
        return False


def _normalize_output_filename(filename):
    """Return a safe XLSX basename for the configured label output directory."""
    filename = validate_output_basename(filename)
    if not filename.lower().endswith(".xlsx"):
        filename += ".xlsx"
    return filename


def _parse_label_arguments(args):
    """Parse label arguments without reading or mutating viewer state."""
    valid_keys = {
        "gmax", "global_max", "g_max",
        "cmin", "cluster_min", "c_min",
        "id",
    }
    fixed_keys = {"gmin", "global_min", "g_min"}
    valid_targets = {"cluster", "clusters", "group", "groups"}
    forced_target = "all"
    requested_filename = None
    positional_args = []
    keyword_args = {}
    keyword_mode = False

    index = 0
    while index < len(args):
        raw_argument = args[index]
        argument = raw_argument.lower()
        if argument in valid_targets:
            forced_target = (
                "clusters" if argument in {"cluster", "clusters"} else "groups"
            )
            index += 1
            continue
        if argument in fixed_keys:
            raise ValueError(
                "gmin is fixed at 97% and cannot be set by the label command."
            )
        if argument in valid_keys:
            keyword_mode = True
            if index + 1 >= len(args):
                raise ValueError(f"Missing numerical value for '{argument}'.")
            if argument in {"gmax", "global_max", "g_max"}:
                key_name = "gmax"
            elif argument in {"cmin", "cluster_min", "c_min"}:
                key_name = "cmin"
            else:
                key_name = "id"
            value_text = args[index + 1]
            if key_name == "id":
                value = logo_cmd.parse_identity_threshold(value_text)
            else:
                value = parse_percentage(value_text)
                if value is None:
                    raise ValueError(
                        f"Invalid percentage '{value_text}' for '{argument}'."
                    )
            if key_name in keyword_args:
                raise ValueError(f"Duplicate assignment for '{key_name}'.")
            keyword_args[key_name] = value
            index += 2
            continue

        parsed_value = parse_percentage(argument)
        if parsed_value is not None:
            if keyword_mode:
                raise ValueError(
                    f"Ambiguous input. Positional argument '{argument}' found "
                    "after keywords."
                )
            positional_args.append((raw_argument, parsed_value))
            index += 1
            continue
        if _is_non_finite_number(argument):
            raise ValueError(
                f"Invalid percentage '{raw_argument}': thresholds must be finite. "
                "Add .xlsx to use it as the report filename."
            )

        if requested_filename is not None:
            raise ValueError("Provide only one custom output filename.")
        if index != len(args) - 1:
            raise ValueError("A custom output filename must be the final argument.")
        requested_filename = _normalize_output_filename(raw_argument)
        index += 1

    positional_keys = ("gmax", "cmin", "id")
    if len(positional_args) > len(positional_keys):
        raise ValueError("Too many positional numerical arguments.")
    for position, (raw_value, parsed_value) in enumerate(positional_args):
        key_name = positional_keys[position]
        if key_name in keyword_args:
            raise ValueError(
                f"Ambiguous input. '{key_name}' defined both positionally and "
                "via keyword."
            )
        if key_name == "id":
            parsed_value = logo_cmd.parse_identity_threshold(raw_value)
        keyword_args[key_name] = parsed_value

    return {
        "global_max": keyword_args.get("gmax", 0.40),
        "cluster_min": keyword_args.get("cmin", 0.98),
        "identity_threshold": keyword_args.get("id"),
        "forced_target": forced_target,
        "requested_filename": requested_filename,
    }


def _build_label_tasks(viewer, target_mode, viewer_to_aln):
    """Build cluster/group alignment subset tasks for the requested mode."""
    tasks = []

    if target_mode in {"all", "clusters"} and viewer.cluster_labels is not None:
        aln_idx_to_cid = {}
        for node_index, _header in enumerate(viewer.full_headers):
            if node_index >= len(viewer.cluster_labels):
                break
            alignment_index = int(viewer_to_aln[node_index])
            if alignment_index >= 0:
                aln_idx_to_cid[alignment_index] = viewer.cluster_labels[node_index]

        cluster_rows = {}
        for alignment_index in range(len(viewer.alignment.aln)):
            cluster_id = aln_idx_to_cid.get(alignment_index, -1)
            if cluster_id != -1:
                cluster_rows.setdefault(cluster_id, []).append(alignment_index)

        for cluster_id in sorted(cluster_rows):
            row_indices = np.asarray(cluster_rows[cluster_id], dtype=int)
            tasks.append((
                "cluster",
                cluster_id,
                _subset_alignment(viewer.alignment.aln, row_indices),
                viewer.alignment.col_to_label,
                row_indices,
            ))

    if target_mode in {"all", "groups"} and getattr(viewer, 'group_labels', None):
        aln_idx_to_groups = {}
        for node_index, _header in enumerate(viewer.full_headers):
            if node_index >= len(viewer.group_labels):
                break
            groups = viewer.group_labels[node_index]
            alignment_index = int(viewer_to_aln[node_index])
            if groups and alignment_index >= 0:
                aln_idx_to_groups[alignment_index] = groups

        group_rows = {}
        for alignment_index in range(len(viewer.alignment.aln)):
            for group_name in aln_idx_to_groups.get(alignment_index, ()):
                group_rows.setdefault(group_name, []).append(alignment_index)

        for group_name in sorted(group_rows):
            row_indices = np.asarray(group_rows[group_name], dtype=int)
            tasks.append((
                "group",
                group_name,
                _subset_alignment(viewer.alignment.aln, row_indices),
                viewer.alignment.col_to_label,
                row_indices,
            ))

    return tasks


def _subset_alignment(aln, row_indices):
    """A sparse row view for sparse alignments; decoded records otherwise."""
    if hasattr(aln, "matrix") and hasattr(aln, "int_to_aa"):
        return _AlignmentRows(aln, row_indices)
    return MultipleSeqAlignment([aln[int(index)] for index in row_indices])

def get_sequence_stats(aln, gap_chars=None):
    gap_chars = set(cfg.GAP_CHARS if gap_chars is None else gap_chars)
    lengths = _residue_lengths(aln, gap_chars) if hasattr(aln, "matrix") else None
    if lengths is None:
        lengths = []
        for record in aln:
            seq_str = str(record.seq)
            ungapped_len = sum(1 for c in seq_str if c not in gap_chars)
            lengths.append(ungapped_len)
    if not len(lengths): return 0, 0, 0.0, 0.0
    arr = np.array(lengths)
    return int(np.min(arr)), int(np.max(arr)), np.mean(arr), np.std(arr)


def _get_amino_acid_counts(aln, col_idx, weights=None, gap_chars=None):
    """Return non-gap amino-acid counts for one alignment column."""
    gap_chars = frozenset(cfg.GAP_CHARS if gap_chars is None else gap_chars)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(aln):
            raise ValueError("Sequence weights must match the alignment row count.")

        aa_counts = {}
        if hasattr(aln, 'matrix'):
            column = aln.matrix.getcol(col_idx).tocoo()
            for row_idx, aa_int in zip(column.row, column.data):
                aa = aln.int_to_aa.get(int(aa_int), 'X')
                if aa not in gap_chars:
                    aa = str(aa).upper()
                    aa_counts[aa] = aa_counts.get(aa, 0.0) + float(weights[row_idx])
        else:
            for row_idx, record in enumerate(aln):
                aa = str(record.seq[col_idx]).upper()
                if aa not in gap_chars:
                    aa_counts[aa] = aa_counts.get(aa, 0.0) + float(weights[row_idx])
        return aa_counts

    # Unweighted counts always come from the frozen sparse global alignment.
    counts = Counter(aln.matrix[:, col_idx].data)
    aa_counts = {}
    for aa_int, count in counts.items():
        aa = aln.int_to_aa.get(aa_int, 'X')
        if aa not in gap_chars:
            aa_counts[aa] = aa_counts.get(aa, 0) + count

    return {
        str(aa).upper(): int(count)
        for aa, count in aa_counts.items()
    }


def _format_global_amino_acid_profile(frequencies):
    """Format a non-gap column profile using query.py's reporting semantics."""
    profile = []
    for aa, frequency in frequencies.items():
        percentage = frequency * 100.0
        if percentage >= 1.0:
            profile.append((aa, percentage))
    profile.sort(key=lambda item: item[1], reverse=True)

    if not profile:
        return "-"
    return " | ".join(f"{aa} {percentage:>5.1f}%" for aa, percentage in profile)


def _calculate_weighted_frequencies(aln, mapping, weights, gap_chars=None):
    """Return weighted consensus statistics and residue counts by display label."""
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(aln):
        raise ValueError("Sequence weights must match the alignment row count.")

    total_weight = float(weights.sum())
    stats = {}
    counts_by_label = {}
    if total_weight <= 0.0:
        return stats, counts_by_label

    alignment_length = aln.get_alignment_length()
    column_counts = None
    if hasattr(aln, "matrix"):
        column_counts = _sparse_column_counts(
            aln,
            mapping.keys(),
            weights,
            frozenset(cfg.GAP_CHARS if gap_chars is None else gap_chars),
        )

    for col_idx, label in mapping.items():
        if col_idx < 0 or col_idx >= alignment_length:
            continue
        if column_counts is not None:
            counts = column_counts[col_idx]
        else:
            counts = _get_amino_acid_counts(
                aln,
                col_idx,
                weights=weights,
                gap_chars=gap_chars,
            )
        counts_by_label[label] = counts
        non_gap_weight = float(sum(counts.values()))
        occupancy = non_gap_weight / total_weight
        if not counts:
            stats[label] = ('-', 0.0, 0.0)
            continue

        consensus_aa, consensus_count = max(counts.items(), key=lambda item: item[1])
        stats[label] = (
            consensus_aa,
            float(consensus_count) / total_weight,
            occupancy,
        )

    return stats, counts_by_label


def _get_indexed_amino_acid_count(
    aln,
    col_idx,
    amino_acid,
    row_indices,
    weights=None,
):
    """Return one residue's count across a deduplicated set of alignment rows."""
    row_indices = np.unique(np.asarray(row_indices, dtype=int))
    if row_indices.size == 0:
        return 0.0
    if row_indices[0] < 0 or row_indices[-1] >= len(aln):
        raise IndexError("Alignment row index is outside the available alignment.")

    if weights is None:
        selected_weights = np.ones(row_indices.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(aln):
            raise ValueError("Sequence weights must match the alignment row count.")
        selected_weights = weights[row_indices]

    target = str(amino_acid).upper()
    column_major = getattr(aln, "column_major", None)
    if column_major is not None and 0 <= col_idx < aln.matrix.shape[1]:
        csc = column_major()
        start, end = csc.indptr[col_idx], csc.indptr[col_idx + 1]
        column = np.zeros(csc.shape[0], dtype=csc.dtype)
        column[csc.indices[start:end]] = csc.data[start:end]
        encoded = column[row_indices]
    else:
        encoded = aln.matrix[row_indices].getcol(col_idx).toarray().ravel()
    matching_codes = {
        int(code)
        for code, residue in aln.int_to_aa.items()
        if str(residue).upper() == target
    }
    matches = np.isin(
        encoded.astype(np.int64),
        np.fromiter(matching_codes, dtype=np.int64, count=len(matching_codes)),
    )
    return float(selected_weights[matches].sum())


def _calculate_outside_frequency(
    amino_acid,
    global_counts,
    global_size,
    excluded_count,
    excluded_size,
):
    """Return the residue frequency after excluding a union of conserved subsets."""
    outside_size = float(global_size) - float(excluded_size)
    if outside_size <= 1e-12:
        return None

    global_count = float(global_counts.get(str(amino_acid).upper(), 0.0))
    outside_count = global_count - float(excluded_count)
    if outside_count < -1e-12:
        return None
    outside_count = max(0.0, outside_count)
    return outside_count / outside_size


def _format_statistics_summary(
    network_node_count,
    aligned_node_count,
    excluded_node_count,
    effective_sequence_count=None,
):
    """Format the compact workbook statistics metadata value."""
    if effective_sequence_count is None:
        effective_sequence_count = aligned_node_count
    effective_display = f"{float(effective_sequence_count):.2f}".rstrip("0").rstrip(".")
    return (
        f"Aligned {aligned_node_count} of {network_node_count} | "
        f"Excluded {excluded_node_count} | Effective {effective_display}"
    )


def _occupancy_fills(cmap, occupancies, cache):
    """Solid fills colored as cmap(occupancy), one per value, one object per color.

    The colormap is applied to the whole list at once (element-wise identical to
    scalar calls) and each distinct color builds a single PatternFill.
    """
    from openpyxl.styles import PatternFill

    fills = []
    for rgba in cmap(np.asarray(occupancies, dtype=float)).tolist():
        rgba = tuple(rgba)
        fill = cache.get(rgba)
        if fill is None:
            hex_color = mcolors.to_hex(rgba)[1:].upper()
            fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")
            cache[rgba] = fill
        fills.append(fill)
    return fills


def _append_workbook_metadata(
    worksheet,
    out_filename,
    ref_display,
    offset_display,
    global_list,
    network_node_count=None,
    aligned_node_count=None,
    excluded_node_count=None,
    identity_threshold=None,
    effective_sequence_count=None,
):
    worksheet.append([f"Filename: {out_filename}"])
    worksheet.append([f"Reference: {ref_display}"])
    worksheet.append([f"Alignment Offset: {offset_display}"])
    if network_node_count is not None:
        statistics_summary = _format_statistics_summary(
            network_node_count,
            aligned_node_count,
            excluded_node_count,
            effective_sequence_count,
        )
        worksheet.append([f"Statistics: {statistics_summary}"])
    if identity_threshold is not None:
        worksheet.append([f"Identity Threshold: {identity_threshold * 100:g}%"])
    worksheet.append([f"Global Conserved (>{int(GLOBAL_CONSERVATION_THRESHOLD * 100)}%)"])
    worksheet.append(global_list if global_list else ["None"])
    worksheet.append([])


def _run_label_artifact(viewer, args):
    if args and args[0].lower() == 'reset':
        msg = Command_Engine.execute_reset(viewer, ["clusters"])
        return

    try:
        alignment = getattr(viewer, 'alignment', None)
        if alignment is None or alignment.aln is None:
            Command_Engine.show_status(viewer, "Error: Global Alignment not loaded.")
            Command_Engine.command_failed(viewer, viewer.console_text.text)
            print("Error: Global Alignment not loaded.")
            return

        if len(alignment.aln) == 0:
            msg = (
                "Error: The selected MSA contains no aligned rows for the current "
                "network. Label analysis is unavailable."
            )
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.show_status(viewer, msg)
            print(msg)
            return

        if not getattr(alignment, 'has_reference', False):
            Command_Engine.show_status(viewer, "Error: No active alignment reference. Use 'reference <ID>' with "
                "a node present in the current MSA.")
            Command_Engine.command_failed(viewer, viewer.console_text.text)
            print(viewer.console_text.text)
            return

        if args and args[0].lower() in ['help', '-h', '-?']:
            print_help()
            if hasattr(viewer, 'console_text'):
                Command_Engine.show_status(viewer, "Help information printed to the terminal")
            return

        parameters = _parse_label_arguments(args)
        global_max = parameters["global_max"]
        cluster_min = parameters["cluster_min"]
        identity_threshold = parameters["identity_threshold"]
        forced_target = parameters["forced_target"]

        # --- Validations ---
        if forced_target == "clusters" and viewer.cluster_labels is None:
            Command_Engine.show_status(viewer, "Error: Run 'cluster' first.")
            Command_Engine.command_failed(viewer, viewer.console_text.text)
            print("Error: Run 'cluster' first to use cluster mode.")
            return
            
        if forced_target == "groups" and getattr(viewer, 'group_labels', None) is None:
            Command_Engine.show_status(viewer, "Error: No groups defined.")
            Command_Engine.command_failed(viewer, viewer.console_text.text)
            print("Error: No groups defined. Use the 'group' command first.")
            return

        if (
            forced_target == "all"
            and viewer.cluster_labels is None
            and getattr(viewer, 'group_labels', None) is None
        ):
            Command_Engine.show_status(viewer, "Error: No clusters or groups defined.")
            Command_Engine.command_failed(viewer, viewer.console_text.text)
            print("Error: No clusters or groups defined. Use 'cluster' or 'group' first.")
            return

        # --- 1. Global Statistics ---
        print("Calculating Global Stats...")
        gap_chars = frozenset(_setting(viewer, "GAP_CHARS", ("-", ".")))
        if hasattr(viewer, "_label_offset_display"):
            offset_display = viewer._label_offset_display
        else:
            offset_display = format_alignment_offset_display(
                getattr(viewer, "alignment", None),
                getattr(
                    viewer,
                    "alignment_offset",
                    getattr(cfg, "ALIGNMENT_OFFSET", 0),
                ),
            )
        print(f"Alignment Offset: {offset_display}")
        total_global_seqs = len(viewer.alignment.aln)
        total_network_nodes = getattr(viewer, 'n_nodes', len(viewer.full_headers))
        excluded_unaligned_nodes = total_network_nodes - total_global_seqs
        g_min, g_max, g_avg, g_std = get_sequence_stats(
            viewer.alignment.aln,
            gap_chars=gap_chars,
        )

        global_weights = None
        if identity_threshold is None:
            total_global_effective_n = float(total_global_seqs)
            # Unit weights sum to the exact integer counts, in first-row order,
            # so these serve get_global_counts without another column scan.
            g_stats, global_count_cache = _calculate_weighted_frequencies(
                viewer.alignment.aln,
                viewer.alignment.col_to_label,
                np.ones(total_global_seqs, dtype=float),
                gap_chars=gap_chars,
            )
        else:
            print(
                "Calculating global identity-neighbour weights at "
                f"{identity_threshold * 100:g}%..."
            )
            global_sequences = _standard_residue_rows(viewer.alignment.aln)
            if global_sequences is None:
                global_sequences = [
                    str(record.seq).upper() for record in viewer.alignment.aln
                ]
            global_weights = logo_cmd.calculate_identity_weights(
                global_sequences,
                identity_threshold,
                report_backend=True,
            )
            del global_sequences
            total_global_effective_n = float(global_weights.sum())
            g_stats, global_count_cache = _calculate_weighted_frequencies(
                viewer.alignment.aln,
                viewer.alignment.col_to_label,
                global_weights,
                gap_chars=gap_chars,
            )

        global_frequency_cache = {}

        def get_global_counts(label):
            if label not in global_count_cache:
                col_idx = viewer.alignment.label_to_col.get(label)
                global_count_cache[label] = (
                    _get_amino_acid_counts(
                        viewer.alignment.aln,
                        col_idx,
                        gap_chars=gap_chars,
                    )
                    if col_idx is not None
                    else {}
                )
            return global_count_cache[label]

        def get_global_frequencies(label):
            if label not in global_frequency_cache:
                global_frequency_cache[label] = {
                    aa: count / total_global_effective_n
                    for aa, count in get_global_counts(label).items()
                } if total_global_effective_n > 0.0 else {}
            return global_frequency_cache[label]

        # --- 2. Prepare Tasks ---
        print(f"Splitting Global Alignment for {forced_target.upper()}...")
        viewer_to_aln, _ = Command_Engine.get_alignment_mapping(viewer)
        tasks = _build_label_tasks(viewer, forced_target, viewer_to_aln)

        # --- 3. Process Tasks ---
        master_labels = set()
        cluster_results = []
        candidate_pools = {}
        candidate_amino_acids = {}
        
        # Build color map for topology clusters using cluster_cmd
        cluster_ids = [
            entity_id
            for entity_type, entity_id, _, _, _ in tasks
            if entity_type == 'cluster'
        ]
        cluster_color_map = cluster_cmd.get_cluster_color_map(cluster_ids)

        for entity_type, entity_id, c_aln, c_map, aln_indices in tasks:
            try:
                # Calculate sequence stats
                c_size = len(c_aln)
                c_min_len, c_max_len, c_avg_len, c_std_len = get_sequence_stats(
                    c_aln,
                    gap_chars=gap_chars,
                )

                # Calculate complete residue counts and frequencies. Identity-
                # enabled subsets retain their rows' globally calculated weights.
                if global_weights is None:
                    c_weights = np.ones(c_size, dtype=float)
                else:
                    c_weights = global_weights[aln_indices]
                c_effective_n = float(c_weights.sum())
                c_stats, c_counts_by_label = _calculate_weighted_frequencies(
                    c_aln,
                    c_map,
                    c_weights,
                    gap_chars=gap_chars,
                )
                c_occ_dict = {
                    lbl: c_stats[lbl][2]
                    for lbl in sort_alignment_labels(c_stats.keys())
                }
                
                # Format Output Styling
                if entity_type == 'cluster':
                    if entity_id in cluster_color_map:
                        rgb = cluster_color_map[entity_id]
                        hex_code = mcolors.to_hex(rgb)
                    else:
                        hex_code = "-"
                    name_str = f"Cluster {entity_id}"
                    sort_key = (0, entity_id)
                else:
                    hex_code = "-"
                    name_str = f"Group {entity_id}"
                    # Sort groups by count descending (-c_size), then alphabetically
                    sort_key = (1, -c_size, str(entity_id))

                result = {
                    "type": entity_type,
                    "id": entity_id,
                    "name": name_str,
                    "sort_key": sort_key,
                    "count": c_size,
                    "effective_n": c_effective_n,
                    "hex": hex_code,
                    "min": c_min_len,
                    "max": c_max_len,
                    "avg": c_avg_len,
                    "std": c_std_len,
                    "data": {},
                    "occ_data": c_occ_dict
                }
                cluster_results.append(result)

                # First pass: collect every amino acid meeting cmin. The shared
                # outside background is resolved only after every subset is known.
                if c_effective_n > 0.0:
                    for lbl in sort_alignment_labels(c_counts_by_label.keys()):
                        if lbl not in g_stats:
                            continue
                        for amino_acid, count in sorted(
                            c_counts_by_label[lbl].items()
                        ):
                            frequency = float(count) / c_effective_n
                            if frequency < cluster_min:
                                continue
                            amino_acid = str(amino_acid).upper()
                            candidate_pools.setdefault(
                                (lbl, amino_acid), []
                            ).append({
                                "result": result,
                                "frequency": frequency,
                                "aln_indices": aln_indices,
                            })
                            candidate_amino_acids.setdefault(lbl, set()).add(
                                amino_acid
                            )
                    
            except Exception as e: 
                print(f"Skipping {entity_type} {entity_id} due to error: {e}")
                continue

        # Second pass: for each conserved position/residue pair, exclude the
        # deduplicated union of all qualifying cluster/group memberships.
        candidate_labels = sort_alignment_labels(candidate_amino_acids.keys())
        for lbl in candidate_labels:
            amino_acids = sorted(candidate_amino_acids[lbl])
            col_idx = viewer.alignment.label_to_col.get(lbl)
            if col_idx is None:
                continue
            for amino_acid in amino_acids:
                candidates = candidate_pools[(lbl, amino_acid)]
                excluded_indices = np.unique(np.concatenate([
                    candidate["aln_indices"] for candidate in candidates
                ]))
                if global_weights is None:
                    excluded_size = float(excluded_indices.size)
                else:
                    excluded_size = float(global_weights[excluded_indices].sum())
                excluded_count = _get_indexed_amino_acid_count(
                    viewer.alignment.aln,
                    col_idx,
                    amino_acid,
                    excluded_indices,
                    weights=global_weights,
                )
                outside_frequency = _calculate_outside_frequency(
                    amino_acid,
                    get_global_counts(lbl),
                    total_global_effective_n,
                    excluded_count,
                    excluded_size,
                )
                if outside_frequency is None or outside_frequency >= global_max:
                    continue

                master_labels.add(lbl)
                for candidate in candidates:
                    candidate["result"]["data"].setdefault(lbl, []).append((
                        amino_acid,
                        candidate["frequency"],
                    ))

        # One subset-position cell may contain multiple independently passing
        # amino acids. Sort by subset frequency, then residue code for stability.
        for result in cluster_results:
            for lbl, entries in list(result["data"].items()):
                entries.sort(key=lambda item: (-item[1], item[0]))
                result["data"][lbl] = {
                    "text": " | ".join(
                        f"{amino_acid}{lbl}" for amino_acid, _frequency in entries
                    ),
                    "occ": result["occ_data"].get(lbl, 0.0),
                }

        # --- 4. Export XLSX ---
        out_path = os.path.abspath(viewer._label_output_path)
        out_dir = os.path.dirname(out_path)
        out_filename = os.path.basename(out_path)
        allow_overwrite = bool(
            getattr(viewer, "_label_allow_overwrite", False)
        )
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        if not allow_overwrite and os.path.exists(out_path):
            raise FileExistsError(f"Output file already exists: {out_path}")

        global_list = []
        for lbl in sort_alignment_labels(g_stats.keys()):
            aa, freq, occ = g_stats[lbl]
            if freq > GLOBAL_CONSERVATION_THRESHOLD:
                global_list.append(f"{aa}{lbl}")

        sorted_cols = sort_alignment_labels(list(master_labels))
        cluster_results.sort(key=lambda x: x["sort_key"])
        all_occ_labels = sort_alignment_labels(list(g_stats.keys()))
        
        ref_display = (
            getattr(viewer.alignment, 'resolved_ref_full', None)
            or getattr(viewer, 'active_reference', None)
            or "None"
        )
        
        # Prepare Metadata details
        fasta_file = _setting(viewer, 'NODE_FASTA_FILE', None)
        fasta_name = os.path.basename(fasta_file) if fasta_file else _setting(viewer, 'SEQUENCE_SET', 'N/A')
        
        network_file = _setting(viewer, 'INPUT_HDF5', None)
        network_name = os.path.basename(network_file) if network_file else "N/A"
        
        msa_file = _setting(viewer, 'MSA_FILE', None)
        alignment_name = os.path.basename(msa_file) if msa_file else "N/A"
        
        if (
            forced_target in {"all", "clusters"}
            and getattr(viewer, 'last_cluster_params', None)
        ):
            c_mode_param, c_min_param = viewer.last_cluster_params
            parts = c_mode_param.split('_')
            cluster_mode = parts[0] if parts else c_mode_param
            param_val = parts[1] if len(parts) > 1 else ""
            cluster_params = f"Param: {param_val}, Min Size: {c_min_param}" if param_val else f"Min Size: {c_min_param}"
        else:
            cluster_mode = {
                "all": "All",
                "groups": "Groups",
                "clusters": "N/A",
            }[forced_target]
            cluster_params = "N/A"

        label_params = (
            f"gmax_outside={int(global_max*100)}%, cmin={int(cluster_min*100)}%, "
            f"target={forced_target}"
        )
        if identity_threshold is not None:
            label_params += f", identity={identity_threshold * 100:g}%"

        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font
        except ImportError:
            Command_Engine.show_status(viewer, "Error: 'openpyxl' is required for XLSX export. Run: pip install openpyxl")
            Command_Engine.command_failed(viewer, viewer.console_text.text)
            print("Error: openpyxl not installed.")
            return

        try:
            wb = openpyxl.Workbook()
            
            # ==========================================
            # TAB 1: Meta Data
            # ==========================================
            ws_meta = wb.active
            ws_meta.title = "Meta Data"
            
            total_nodes = getattr(viewer, 'n_nodes', total_global_seqs)
            n_clusters = len([r for r in cluster_results if r['type'] == 'cluster'])
            n_groups = len([r for r in cluster_results if r['type'] == 'group'])
            if n_clusters > 0 and n_groups > 0:
                topology_str = f"{total_nodes} nodes with {n_clusters} Clusters, {n_groups} Groups"
            elif n_clusters > 0:
                topology_str = f"{total_nodes} nodes with {n_clusters} Clusters"
            else:
                topology_str = f"{total_nodes} nodes with {n_groups} Groups"

            ws_meta.append(["Metadata Field", "Value"])
            ws_meta.append(["Cluster Mode", cluster_mode])
            ws_meta.append(["Cluster Parameters", cluster_params])
            ws_meta.append(["Network Topology", topology_str])
            ws_meta.append(["Fasta Name", fasta_name])
            ws_meta.append(["Network Name", network_name])
            ws_meta.append(["Alignment Name", alignment_name])
            ws_meta.append(["Label Parameters", label_params])
            ws_meta.append([
                "Statistics",
                _format_statistics_summary(
                    total_network_nodes,
                    total_global_seqs,
                    excluded_unaligned_nodes,
                    total_global_effective_n,
                ),
            ])
            
            # Style header row for Meta Data tab
            header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            for cell in ws_meta[1]:
                cell.fill = header_fill
                cell.font = header_font
            ws_meta.column_dimensions['A'].width = 25
            ws_meta.column_dimensions['B'].width = 50

            effective_percent_column = 2 if identity_threshold is not None else None
            percent_column = 3 if identity_threshold is not None else 2
            effective_n_column = 4 if identity_threshold is not None else None
            hex_color_column = 6 if identity_threshold is not None else 4
            position_start_column = 12 if identity_threshold is not None else 10
            percent_number_format = "0.00%"
            effective_n_number_format = "0.00"

            def node_fraction(count):
                return count / total_network_nodes if total_network_nodes else 0.0

            def effective_fraction(effective_n):
                if total_global_effective_n <= 0.0:
                    return 0.0
                return effective_n / total_global_effective_n

            # ==========================================
            # TAB 2: Subset Specific Matrix
            # ==========================================
            ws1 = wb.create_sheet(title="Subset Stats")
            ws1.freeze_panes = "C1"  # Keep columns A and B visible during horizontal scrolling.
            
            # Write Metadata 
            _append_workbook_metadata(
                ws1,
                out_filename,
                ref_display,
                offset_display,
                global_list,
                total_network_nodes,
                total_global_seqs,
                excluded_unaligned_nodes,
                identity_threshold,
                total_global_effective_n,
            )
            
            # Write Headers 
            ws1.append(["Subset Specific Matrix"])
            if identity_threshold is not None:
                headers1 = [
                    "Subset Name", "Effective Percent", "Percent",
                    "Effective N", "Count N",
                ]
            else:
                headers1 = ["Subset Name", "Percent", "Count"]
            headers1 += [
                "Hex Color", "Min Len", "Max Len", "Avg Len", "Std Dev", ""
            ] + [f"#{c}" for c in sorted_cols]
            ws1.append(headers1)
            
            try:
                occ_cmap1 = cm.get_cmap('Reds_r')
            except AttributeError:
                occ_cmap1 = matplotlib.colormaps['Reds_r']
                
            fill_cache1 = {}

            def get_occ_fills1(occ_values):
                return _occupancy_fills(occ_cmap1, occ_values, fill_cache1)

            # Write Global Row 
            if identity_threshold is not None:
                global_freq_row1 = [
                    "Global Stats",
                    effective_fraction(total_global_effective_n),
                    node_fraction(total_global_seqs),
                    total_global_effective_n,
                    total_global_seqs,
                ]
            else:
                global_freq_row1 = [
                    "Global Stats",
                    node_fraction(total_global_seqs),
                    total_global_seqs,
                ]
            global_freq_row1 += [
                "-", g_min, g_max, round(g_avg, 1), round(g_std, 1), ""
            ]
            
            g_occ_dict1 = {}
            for col in sorted_cols:
                if col in g_stats:
                    _, _, g_occ = g_stats[col]
                    global_freq_row1.append(
                        _format_global_amino_acid_profile(
                            get_global_frequencies(col)
                        )
                    )
                    g_occ_dict1[col] = g_occ
                else: 
                    global_freq_row1.append("-")
                    
            ws1.append(global_freq_row1)
            g_row_idx1 = ws1.max_row
            if effective_percent_column is not None:
                ws1.cell(
                    row=g_row_idx1, column=effective_percent_column
                ).number_format = percent_number_format
            ws1.cell(row=g_row_idx1, column=percent_column).number_format = percent_number_format
            if effective_n_column is not None:
                ws1.cell(
                    row=g_row_idx1, column=effective_n_column
                ).number_format = effective_n_number_format
            
            global_fill_columns1 = [
                (c_idx + position_start_column, g_occ_dict1[col])
                for c_idx, col in enumerate(sorted_cols)
                if col in g_occ_dict1
            ]
            for (col_letter_idx, _), fill in zip(
                global_fill_columns1,
                get_occ_fills1([occ for _, occ in global_fill_columns1]),
            ):
                ws1.cell(row=g_row_idx1, column=col_letter_idx).fill = fill
                    
            ws1.append([]) # Blank row below Global Stats

            # Write Subset Rows
            last_type = None
            for res in cluster_results:
                if last_type == 'cluster' and res['type'] == 'group':
                    ws1.append([]) # Blank row separating Clusters and Groups
                last_type = res['type']
                if identity_threshold is not None:
                    row1 = [
                        res['name'],
                        effective_fraction(res['effective_n']),
                        node_fraction(res['count']),
                        res['effective_n'],
                        res['count'],
                    ]
                else:
                    row1 = [
                        res['name'],
                        node_fraction(res['count']),
                        res['count'],
                    ]
                row1 += [
                    res['hex'], res['min'], res['max'], round(res['avg'], 1),
                    round(res['std'], 1), ""
                ]
                
                row_occs1 = {}
                for c_idx, col in enumerate(sorted_cols): 
                    if col in res['data']:
                        row1.append(res['data'][col]["text"])
                    else:
                        row1.append("")
                        
                    if col in res['occ_data']:
                        row_occs1[c_idx + position_start_column] = res['occ_data'][col]
                    else:
                        row_occs1[c_idx + position_start_column] = 0.0
                        
                ws1.append(row1)
                current_row1 = ws1.max_row
                if effective_percent_column is not None:
                    ws1.cell(
                        row=current_row1, column=effective_percent_column
                    ).number_format = percent_number_format
                ws1.cell(row=current_row1, column=percent_column).number_format = percent_number_format
                if effective_n_column is not None:
                    ws1.cell(
                        row=current_row1, column=effective_n_column
                    ).number_format = effective_n_number_format
                
                if res['hex'] != "-":
                    hex_val = res['hex'].replace("#", "").upper()
                    ws1.cell(row=current_row1, column=hex_color_column).fill = PatternFill(start_color=hex_val, end_color=hex_val, fill_type="solid")
                
                for col_index, fill in zip(
                    row_occs1, get_occ_fills1(list(row_occs1.values()))
                ):
                    ws1.cell(row=current_row1, column=col_index).fill = fill
                    
            # ==========================================
            # TAB 2: Occupancy Stats
            # ==========================================
            ws2 = wb.create_sheet(title="Occupancy Stats")
            ws2.freeze_panes = "C1"  # Keep columns A and B visible during horizontal scrolling.
            
            # Write Metadata 
            _append_workbook_metadata(
                ws2,
                out_filename,
                ref_display,
                offset_display,
                global_list,
                total_network_nodes,
                total_global_seqs,
                excluded_unaligned_nodes,
                identity_threshold,
                total_global_effective_n,
            )
            
            # Write Headers 
            ws2.append(["Occupancy Matrix"])
            if identity_threshold is not None:
                headers2 = [
                    "Subset Name", "Effective Percent", "Percent",
                    "Effective N", "Count N",
                ]
            else:
                headers2 = ["Subset Name", "Percent", "Count"]
            headers2 += [
                "Hex Color", "Min Len", "Max Len", "Avg Len", "Std Dev", ""
            ] + [f"#{c}" for c in all_occ_labels]
            ws2.append(headers2)
            
            try:
                occ_cmap2 = cm.get_cmap('Greens')
            except AttributeError:
                occ_cmap2 = matplotlib.colormaps['Greens']
                
            fill_cache2 = {}

            def get_occ_fills2(occ_values):
                return _occupancy_fills(occ_cmap2, occ_values, fill_cache2)

            # Write Global Row 
            if identity_threshold is not None:
                global_freq_row2 = [
                    "Global Stats",
                    effective_fraction(total_global_effective_n),
                    node_fraction(total_global_seqs),
                    total_global_effective_n,
                    total_global_seqs,
                ]
            else:
                global_freq_row2 = [
                    "Global Stats",
                    node_fraction(total_global_seqs),
                    total_global_seqs,
                ]
            global_freq_row2 += [
                "-", g_min, g_max, round(g_avg, 1), round(g_std, 1), ""
            ]
            
            g_occ_dict2 = {}
            for col in all_occ_labels:
                global_freq_row2.append("") # Keep text blank
                if col in g_stats:
                    g_occ_dict2[col] = g_stats[col][2] # Occupancy is the 3rd item in the tuple
                else: 
                    g_occ_dict2[col] = 0.0
                    
            ws2.append(global_freq_row2)
            g_row_idx2 = ws2.max_row
            if effective_percent_column is not None:
                ws2.cell(
                    row=g_row_idx2, column=effective_percent_column
                ).number_format = percent_number_format
            ws2.cell(row=g_row_idx2, column=percent_column).number_format = percent_number_format
            if effective_n_column is not None:
                ws2.cell(
                    row=g_row_idx2, column=effective_n_column
                ).number_format = effective_n_number_format
            
            for c_idx, fill in enumerate(
                get_occ_fills2([g_occ_dict2[col] for col in all_occ_labels])
            ):
                col_letter_idx = c_idx + position_start_column
                ws2.cell(row=g_row_idx2, column=col_letter_idx).fill = fill
                
            ws2.append([]) # Blank row below Global Stats

            # Write Subset Rows
            last_type = None
            for res in cluster_results:
                if last_type == 'cluster' and res['type'] == 'group':
                    ws2.append([]) # Blank row separating Clusters and Groups
                last_type = res['type']
                if identity_threshold is not None:
                    row2 = [
                        res['name'],
                        effective_fraction(res['effective_n']),
                        node_fraction(res['count']),
                        res['effective_n'],
                        res['count'],
                    ]
                else:
                    row2 = [
                        res['name'],
                        node_fraction(res['count']),
                        res['count'],
                    ]
                row2 += [
                    res['hex'], res['min'], res['max'], round(res['avg'], 1),
                    round(res['std'], 1), ""
                ]
                
                row_occs2 = {}
                for c_idx, col in enumerate(all_occ_labels): 
                    row2.append("") # Keep text blank
                    row_occs2[c_idx + position_start_column] = res['occ_data'].get(col, 0.0)
                        
                ws2.append(row2)
                current_row2 = ws2.max_row
                if effective_percent_column is not None:
                    ws2.cell(
                        row=current_row2, column=effective_percent_column
                    ).number_format = percent_number_format
                ws2.cell(row=current_row2, column=percent_column).number_format = percent_number_format
                if effective_n_column is not None:
                    ws2.cell(
                        row=current_row2, column=effective_n_column
                    ).number_format = effective_n_number_format
                
                if res['hex'] != "-":
                    hex_val = res['hex'].replace("#", "").upper()
                    ws2.cell(row=current_row2, column=hex_color_column).fill = PatternFill(start_color=hex_val, end_color=hex_val, fill_type="solid")
                
                for col_index, fill in zip(
                    row_occs2, get_occ_fills2(list(row_occs2.values()))
                ):
                    ws2.cell(row=current_row2, column=col_index).fill = fill

            # Auto-fit Column A width based on the longest cluster or group name
            name_lengths = [len("Subset Name"), len("Global Stats")] + [len(res['name']) for res in cluster_results]
            max_name_len = max(name_lengths) if name_lengths else 15
            col_a_width = max(max_name_len + 3, 15)

            ws1.column_dimensions['A'].width = col_a_width
            ws2.column_dimensions['A'].width = col_a_width
            ws1.column_dimensions['B'].width = 18 if identity_threshold is not None else 10
            ws2.column_dimensions['B'].width = 18 if identity_threshold is not None else 10
            if identity_threshold is not None:
                ws1.column_dimensions['C'].width = 10
                ws2.column_dimensions['C'].width = 10

            file_descriptor, partial_path = tempfile.mkstemp(
                prefix=f".{os.path.splitext(out_filename)[0]}.",
                suffix=".partial.xlsx",
                dir=out_dir,
            )
            os.close(file_descriptor)
            try:
                wb.save(partial_path)
                if not allow_overwrite and os.path.exists(out_path):
                    raise FileExistsError(
                        f"Output file already exists: {out_path}"
                    )
                os.replace(partial_path, out_path)
                partial_path = None
            finally:
                if partial_path and os.path.exists(partial_path):
                    try:
                        os.remove(partial_path)
                    except OSError:
                        pass
            
            msg = f"Exported to {out_path}"
            Command_Engine.show_status(viewer, msg)
            print(msg)
            return {
                "message": msg,
                "save_path": out_path,
                "reveal_directory": out_dir,
            }
        except Exception as e:
            Command_Engine.show_status(viewer, f"IO Error: {e}")
            raise

    except Exception as e:
        Command_Engine.show_status(viewer, f"Error: {e}")
        Command_Engine.command_failed(viewer, viewer.console_text.text)
        raise


def _execute_label_envelope(envelope):
    result = _run_label_artifact(
        envelope.viewer_snapshot,
        list(envelope.args),
    )
    if not isinstance(result, dict):
        message = getattr(envelope.viewer_snapshot.console_text, "text", "")
        raise RuntimeError(message or "Label generation did not produce an artifact.")
    return result


def _report_label_error(viewer, error):
    message = f"Error: {error}"
    Command_Engine.command_failed(viewer, message)
    if hasattr(viewer, "console_text"):
        Command_Engine.show_status(viewer, message)
    if hasattr(viewer, "update_console_background"):
        viewer.update_console_background()
    print(message)


def run(viewer, args):
    """Validate and snapshot label inputs before enqueuing the heavy work."""
    if args and args[0].lower() == "reset":
        msg = Command_Engine.execute_reset(viewer, ["clusters"])
        Command_Engine.command_succeeded(viewer, msg)
        return

    alignment = getattr(viewer, "alignment", None)
    if alignment is None or alignment.aln is None:
        _report_label_error(viewer, "Global Alignment not loaded.")
        return
    if len(alignment.aln) == 0:
        _report_label_error(
            viewer,
            "The selected MSA contains no aligned rows for the current network. "
            "Label analysis is unavailable.",
        )
        return
    if not getattr(alignment, "has_reference", False):
        _report_label_error(
            viewer,
            "No active alignment reference. Use 'reference <ID>' with a node "
            "present in the current MSA.",
        )
        return
    if args and args[0].lower() in {"help", "-h", "-?"}:
        print_help()
        if hasattr(viewer, "console_text"):
            Command_Engine.show_status(viewer, "Help information printed to the terminal")
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    try:
        parameters = _parse_label_arguments(args)
    except ValueError as error:
        _report_label_error(viewer, error)
        return

    forced_target = parameters["forced_target"]
    if forced_target == "clusters" and getattr(viewer, "cluster_labels", None) is None:
        _report_label_error(viewer, "Run 'cluster' first.")
        return
    if forced_target == "groups" and getattr(viewer, "group_labels", None) is None:
        _report_label_error(viewer, "No groups defined.")
        return
    if (
        forced_target == "all"
        and getattr(viewer, "cluster_labels", None) is None
        and getattr(viewer, "group_labels", None) is None
    ):
        _report_label_error(viewer, "No clusters or groups defined.")
        return

    scheduler = getattr(viewer, "background_job_scheduler", None)
    if scheduler is None:
        _report_label_error(viewer, "The background job scheduler is unavailable.")
        return

    output_directory = os.path.abspath(
        cfg.resolve_directory_path(CLUSTER_LABEL_DIRECTORY)
    )
    requested_filename = parameters["requested_filename"]
    allow_overwrite = requested_filename is not None
    if requested_filename is None:
        generated = (
            "Label_Output_"
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".xlsx"
        )
        output_filename, output_path = logo_cmd._available_automatic_filename(
            scheduler,
            output_directory,
            generated,
        )
    else:
        output_filename = requested_filename
        output_path = os.path.abspath(
            os.path.join(output_directory, output_filename)
        )
        if scheduler.is_output_path_reserved(output_path):
            _report_label_error(
                viewer,
                f"Output file is already reserved by a background job: {output_path}",
            )
            return

    try:
        viewer_to_aln, _ = Command_Engine.get_alignment_mapping(viewer)
        frozen_alignment = _FrozenAlignmentManager(alignment, viewer_to_aln)
    except Exception as error:
        _report_label_error(viewer, f"Could not snapshot label inputs: {error}")
        return

    group_labels = getattr(viewer, "group_labels", None)
    if group_labels is not None:
        group_labels = tuple(
            frozenset(groups) if groups else frozenset()
            for groups in group_labels
        )
    cluster_labels = getattr(viewer, "cluster_labels", None)
    if cluster_labels is not None:
        cluster_labels = tuple(cluster_labels)

    settings = {
        name: getattr(cfg, name, default)
        for name, default in (
            ("NODE_FASTA_FILE", None),
            ("SEQUENCE_SET", "Network"),
            ("INPUT_HDF5", None),
            ("MSA_FILE", None),
            ("GAP_CHARS", ("-", ".")),
        )
    }
    snapshot = SimpleNamespace(
        alignment=frozen_alignment,
        active_reference=getattr(viewer, "active_reference", None),
        alignment_offset=getattr(viewer, "alignment_offset", 0),
        full_headers=tuple(getattr(viewer, "full_headers", ())),
        n_nodes=int(getattr(viewer, "n_nodes", len(getattr(viewer, "full_headers", ())))),
        cluster_labels=cluster_labels,
        group_labels=group_labels,
        last_cluster_params=(
            tuple(viewer.last_cluster_params)
            if getattr(viewer, "last_cluster_params", None)
            else None
        ),
        console_text=SimpleNamespace(text=""),
        _label_settings=settings,
        _label_offset_display=format_alignment_offset_display(
            getattr(viewer, "alignment", None),
            getattr(
                viewer,
                "alignment_offset",
                getattr(cfg, "ALIGNMENT_OFFSET", 0),
            ),
        ),
        _label_output_path=output_path,
        _label_allow_overwrite=allow_overwrite,
    )
    envelope = _LabelJobEnvelope(snapshot, tuple(args))
    try:
        scheduler.enqueue(
            command_name="label",
            description=f"label -> {output_filename}",
            payload=envelope,
            worker=_execute_label_envelope,
            output_path=output_path,
            allow_overwrite=allow_overwrite,
        )
    except (FileExistsError, RuntimeError) as error:
        _report_label_error(viewer, error)
        return
    Command_Engine.command_succeeded(viewer, f"Queued label analysis for {output_path}; waiting for the background job.")
