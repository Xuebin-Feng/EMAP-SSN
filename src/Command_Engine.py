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

import numpy as np
import fnmatch
import re
import os
from dataclasses import dataclass
from enum import Enum
import EMAPSSN_Config as cfg
from utilities.Localization import display_text


class SelectionExpressionError(ValueError):
    """Raised when a Boolean selection expression is invalid."""


class SelectionContextError(SelectionExpressionError):
    """Raised when an expression references data absent from the current SSN."""


class SelectionClassificationKind(Enum):
    """Syntax-only classification for a possible selection-expression argument."""

    VALID_EXPRESSION = "valid_expression"
    NOT_EXPRESSION = "not_expression"
    MALFORMED_EXPRESSION = "malformed_expression"


@dataclass(frozen=True)
class SelectionClassification:
    kind: SelectionClassificationKind
    expression: object = None
    error: SelectionExpressionError = None


@dataclass(frozen=True)
class _SelectionAtom:
    kind: str
    value: object


@dataclass(frozen=True)
class _SelectionUnary:
    operand: object


@dataclass(frozen=True)
class _SelectionBinary:
    operator: str
    left: object
    right: object


@dataclass(frozen=True)
class ResolvedLabelTarget:
    """A context-validated #label# target shared by expressions and export."""

    kind: str
    name: str
    cluster_id: int = None


class _NotSelectionExpression(Exception):
    """Internal signal that a plain argument is not expression-shaped."""


_METADATA_QUERY_PATTERN = re.compile(
    r'^([a-zA-Z0-9_\-]+)\s*(>=|<=|!=|==|>|<|=)\s*(.*)$'
)
# Numeric metadata values. A range is LOW-HIGH; a negative bound must be in
# parentheses, as a negative alignment position must: {GRAVY=(-1)-0}. Any
# single value may be too: {GRAVY>=(-1)} reads as {GRAVY>=-1}.
_UNSIGNED_NUMBER = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_PARENTHESISED_NUMBER = rf'\(\s*([+-]?{_UNSIGNED_NUMBER})\s*\)'
_RANGE_BOUND = rf'(?:{_PARENTHESISED_NUMBER}|({_UNSIGNED_NUMBER}))'
_METADATA_RANGE_PATTERN = re.compile(rf'^{_RANGE_BOUND}\s*-\s*{_RANGE_BOUND}$')
_PARENTHESISED_VALUE_PATTERN = re.compile(rf'^{_PARENTHESISED_NUMBER}$')
_BARE_NEGATIVE_RANGE_PATTERN = re.compile(
    rf'^-?\s*{_UNSIGNED_NUMBER}\s*-\s*-?\s*{_UNSIGNED_NUMBER}$'
)


def parse_metadata_number(value_text):
    """A numeric comparison value, optionally in parentheses; ValueError if none."""
    text = value_text.strip()
    match = _PARENTHESISED_VALUE_PATTERN.match(text)
    return float(match.group(1) if match else text)


def parse_metadata_range(value_text):
    """(low, high) for a range value such as 300-500 or (-1)-0, else None.

    Raises SelectionExpressionError for a range whose negative bound lacks the
    parentheses that keep '-1-0' from reading ambiguously.
    """
    text = value_text.strip()
    match = _METADATA_RANGE_PATTERN.match(text)
    if match:
        low = match.group(1) or match.group(2)
        high = match.group(3) or match.group(4)
        return float(low), float(high)
    if _BARE_NEGATIVE_RANGE_PATTERN.match(text):
        raise SelectionExpressionError(
            f"Negative range bound in '{text}' must be written in parentheses, "
            "for example '(-1)-0' or '(-1.5)-(-0.5)'. Parentheses are required "
            "around negative values in a range."
        )
    return None
_AA_PREDICATE_PATTERN = re.compile(
    r'(?<!\w)([a-zA-Z_])(?:\((-\d+(?:\.\d+)?)\)|([\d.]+))(?![\w.])'
)
_AA_GROUP_PREDICATE_PATTERN = re.compile(
    r'(?<!\w)\(([a-zA-Z]*)\)(?:\((-\d+(?:\.\d+)?)\)|([\d.]+))(?![\w.])'
)
_BARE_NEGATIVE_AA_PATTERN = re.compile(
    r'(?<!\w)([a-zA-Z_])(-\d+(?:\.\d+)?)(?![\w.])'
)
_BARE_NEGATIVE_AA_GROUP_PATTERN = re.compile(
    r'(?<!\w)\(([a-zA-Z]+)\)(-\d+(?:\.\d+)?)(?![\w.])'
)
_SUGGESTION_LIMIT = 10


def _format_available(values, label):
    """Return a stable, bounded list of currently available expression targets."""
    cleaned = []
    seen = set()
    for value in values:
        display = str(value)
        key = display.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(display)

    cleaned.sort(key=lambda value: value.lower())
    if not cleaned:
        return f"No {label} are currently available."

    displayed = cleaned[:_SUGGESTION_LIMIT]
    suffix = ""
    if len(cleaned) > _SUGGESTION_LIMIT:
        suffix = f" ... (+{len(cleaned) - _SUGGESTION_LIMIT} more)"
    return f"Available {label}: {', '.join(displayed)}{suffix}"


def _selection_file_path(target):
    """Resolve an @file@ selection target to its configured header-list path."""
    file_name = target.strip()
    if file_name.lower().startswith('[ncbi]'):
        file_name = file_name[6:]
    elif file_name.lower().startswith('[pdb]'):
        file_name = file_name[5:]

    header_dir = getattr(
        cfg,
        'HEADER_LIST_DIR',
        os.path.join("Input_Files", "Header_Lists"),
    )
    if file_name.lower().endswith(('.fasta', '.txt')):
        return os.path.join(header_dir, file_name), header_dir
    return os.path.join(header_dir, file_name + ".txt"), header_dir


def _validate_file_target(target):
    load_path, header_dir = _selection_file_path(target)
    if not os.path.isfile(load_path):
        raise SelectionContextError(
            f"Selection file '{os.path.basename(load_path)}' does not exist.\n"
            f"Expected location: {header_dir}"
        )


def _available_group_lookup(group_labels):
    lookup = {}
    if group_labels is not None:
        for groups in group_labels:
            if not groups:
                continue
            for group_name in groups:
                display = str(group_name)
                lookup.setdefault(display.lower(), display)
    return lookup


def resolve_label_target(cluster_labels, group_labels, target):
    """Resolve one label using canonical cluster spelling and explicit ambiguity."""
    target_name = str(target).strip()
    target_lower = target_name.lower()
    group_lookup = _available_group_lookup(group_labels)
    group_name = group_lookup.get(target_lower)

    if target_lower == "noise":
        if cluster_labels is None:
            raise SelectionContextError(
                "Noise cannot be selected because clusters have not been defined.\n"
                "Run the cluster command first."
            )
        cluster_values = np.asarray(cluster_labels)
        if not np.any(cluster_values == -1):
            available = [
                f"cluster_{int(cluster_id)}"
                for cluster_id in np.unique(cluster_values)
                if int(cluster_id) != -1
            ]
            raise SelectionContextError(
                "Noise does not exist in the current clustering.\n"
                + _format_available(available, "clusters")
            )
        return ResolvedLabelTarget("noise", "noise", -1)

    cluster_match = re.fullmatch(r'cluster_(0|[1-9]\d*)', target_lower)
    cluster_id = int(cluster_match.group(1)) if cluster_match else None
    cluster_exists = False
    cluster_values = None
    if cluster_id is not None and cluster_labels is not None:
        cluster_values = np.asarray(cluster_labels)
        cluster_exists = bool(np.any(cluster_values == cluster_id))

    if cluster_exists and group_name is not None:
        raise SelectionContextError(
            f"Label '{target_name}' is ambiguous because it names both topology "
            "cluster " + f"cluster_{cluster_id} and a custom group. Rename or remove "
            "the custom group before using this label."
        )
    if cluster_exists:
        return ResolvedLabelTarget("cluster", f"cluster_{cluster_id}", cluster_id)
    if group_name is not None:
        return ResolvedLabelTarget("group", group_name)

    if cluster_id is not None:
        if cluster_labels is None:
            raise SelectionContextError(
                f"Cluster 'cluster_{cluster_id}' cannot be selected because clusters "
                "have not been defined.\nRun the cluster command first."
            )
        available = [
            "noise" if int(value) == -1 else f"cluster_{int(value)}"
            for value in np.unique(cluster_values)
        ]
        raise SelectionContextError(
            f"Cluster 'cluster_{cluster_id}' does not exist in the current SSN.\n"
            + _format_available(available, "clusters")
        )

    raise SelectionContextError(
        f"Group '{target_name}' does not exist in the current SSN.\n"
        + _format_available(group_lookup.values(), "groups")
    )


def _validate_label_target(cluster_labels, group_labels, target):
    return resolve_label_target(cluster_labels, group_labels, target)


def _validate_aa_target(alignment, target_aa, target_pos_label):
    predicate = (
        f"{target_aa}({target_pos_label})"
        if str(target_pos_label).startswith('-')
        else f"{target_aa}{target_pos_label}"
    )
    if not re.fullmatch(r'-?\d+(?:\.\d+)?', target_pos_label):
        raise SelectionExpressionError(
            f"Alignment position '{target_pos_label}' in predicate '{predicate}' is "
            "not a valid integer or insertion-position label."
        )
    if alignment is None or getattr(alignment, 'aln', None) is None:
        raise SelectionContextError(
            f"Amino-acid predicate '{predicate}' cannot be evaluated because no "
            "alignment is loaded."
        )

    label_to_col = getattr(alignment, 'label_to_col', None) or {}
    if target_pos_label not in label_to_col:
        ordered_labels = []
        col_to_label = getattr(alignment, 'col_to_label', None) or {}
        if isinstance(col_to_label, dict):
            ordered_labels = [col_to_label[key] for key in sorted(col_to_label)]
        if not ordered_labels:
            ordered_labels = list(label_to_col.keys())
        raise SelectionContextError(
            f"Alignment position '{target_pos_label}' in predicate '{predicate}' "
            "does not exist in the current displayed numbering.\n"
            + _format_available(ordered_labels, "alignment positions")
        )


def _normalize_aa_group(target_aas):
    """Return a stable, case-insensitive residue set for a grouped predicate."""
    if len(target_aas) < 2:
        display = f"({target_aas})"
        raise SelectionExpressionError(
            f"Grouped amino-acid target '{display}' must contain at least two "
            "one-letter residue symbols."
        )
    return tuple(dict.fromkeys(target_aas.upper()))


def _validate_metadata_target(metadata, target):
    if not metadata:
        raise SelectionContextError(
            "Metadata predicate cannot be evaluated because no metadata is loaded."
        )

    match = _METADATA_QUERY_PATTERN.fullmatch(target.strip())
    if not match:
        raise SelectionExpressionError(
            f"Invalid metadata predicate '{{{target}}}'. Use '{{PropertyOperatorValue}}', "
            "for example '{{Length>500}}'."
        )

    key, operator, value_text = match.groups()
    value_text = value_text.strip()
    if not value_text:
        raise SelectionExpressionError(
            f"Metadata predicate '{{{target}}}' is missing a comparison value."
        )

    metadata_key = next(
        (candidate for candidate in metadata if candidate.lower() == key.lower()),
        None,
    )
    if metadata_key is None:
        raise SelectionContextError(
            f"Metadata property '{key}' does not exist in the current SSN.\n"
            + _format_available(metadata.keys(), "metadata properties")
        )

    property_type = metadata[metadata_key].get("type")
    if property_type == "number":
        try:
            parse_metadata_number(value_text)
            return
        except ValueError:
            if operator in ('=', '==') and parse_metadata_range(value_text) is not None:
                return
            raise SelectionExpressionError(
                f"Value '{value_text}' is not numeric for metadata property "
                f"'{metadata_key}'."
            )

    if property_type == "text":
        if operator not in ('=', '==', '!='):
            raise SelectionExpressionError(
                f"Operator '{operator}' is not supported for text metadata property "
                f"'{metadata_key}'. Use '=', '==', or '!='."
            )
        return

    raise SelectionExpressionError(
        f"Metadata property '{metadata_key}' has unsupported type '{property_type}'."
    )


def report_selection_error(viewer, expression, error, operation="Selection"):
    """Report a concise HUD error and detailed terminal diagnostics."""
    command_failed(viewer, str(error))
    error_lines = str(error).splitlines() or [str(error)]
    message_lines = [f"{operation} error: {error_lines[0]}"]
    message_lines.extend(error_lines[1:])
    if expression:
        message_lines.append(f"Expression: {expression}")
    message_lines.append("Operation aborted; no changes were applied.")
    print_help(viewer, "\n".join(message_lines))


class _LogicMask:
    """Three-state boolean mask used to preserve unknown AA predicates."""

    UNKNOWN = np.int8(-1)
    FALSE = np.int8(0)
    TRUE = np.int8(1)

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int8)

    @classmethod
    def known(cls, mask):
        return cls(np.asarray(mask, dtype=bool).astype(np.int8))

    @classmethod
    def partially_known(cls, mask, known_mask):
        mask = np.asarray(mask, dtype=bool)
        known_mask = np.asarray(known_mask, dtype=bool)
        values = np.full(mask.shape, cls.UNKNOWN, dtype=np.int8)
        values[known_mask & ~mask] = cls.FALSE
        values[known_mask & mask] = cls.TRUE
        return cls(values)

    @classmethod
    def _coerce(cls, other):
        return other if isinstance(other, cls) else cls.known(other)

    def __invert__(self):
        result = self.values.copy()
        result[self.values == self.TRUE] = self.FALSE
        result[self.values == self.FALSE] = self.TRUE
        return _LogicMask(result)

    def __and__(self, other):
        other = self._coerce(other)
        result = np.full(self.values.shape, self.UNKNOWN, dtype=np.int8)
        result[(self.values == self.FALSE) | (other.values == self.FALSE)] = self.FALSE
        result[(self.values == self.TRUE) & (other.values == self.TRUE)] = self.TRUE
        return _LogicMask(result)

    def __or__(self, other):
        other = self._coerce(other)
        result = np.full(self.values.shape, self.UNKNOWN, dtype=np.int8)
        result[(self.values == self.TRUE) | (other.values == self.TRUE)] = self.TRUE
        result[(self.values == self.FALSE) & (other.values == self.FALSE)] = self.FALSE
        return _LogicMask(result)

    def __xor__(self, other):
        other = self._coerce(other)
        result = np.full(self.values.shape, self.UNKNOWN, dtype=np.int8)
        known = (self.values != self.UNKNOWN) & (other.values != self.UNKNOWN)
        result[known] = (self.values[known] != other.values[known]).astype(np.int8)
        return _LogicMask(result)

    def to_bool(self):
        return self.values == self.TRUE


def get_alignment_mapping(viewer):
    """Return the authoritative network-to-alignment map and mapped indices."""
    full_headers = getattr(viewer, 'full_headers', [])
    n_nodes = len(full_headers)
    alignment = getattr(viewer, 'alignment', None)

    if alignment is not None:
        stored_mapping = getattr(alignment, 'viewer_to_aln', None)
        if stored_mapping is not None:
            stored_mapping = np.asarray(stored_mapping, dtype=int)
            if stored_mapping.shape == (n_nodes,):
                return stored_mapping, np.flatnonzero(stored_mapping >= 0)

    viewer_to_aln = np.full(n_nodes, -1, dtype=int)
    if alignment is not None and getattr(alignment, 'aln', None) is not None:
        seq_map = getattr(alignment, 'seq_map', {}) or {}
        for i, header in enumerate(full_headers):
            if header in seq_map:
                viewer_to_aln[i] = seq_map[header]
    return viewer_to_aln, np.flatnonzero(viewer_to_aln >= 0)

def evaluate_string_mask(full_headers, target):
    """Evaluates a raw string, NCBI ID, or wildcard pattern into a boolean mask."""
    mask = np.zeros(len(full_headers), dtype=bool)
    t_lower = target.lower()

    for i, full_header in enumerate(full_headers):
        fh_lower = full_header.lower()
        
        
        # 1. Standard sub-string matching
        if t_lower in fh_lower:
            mask[i] = True
            
        # 2. Comprehensive wildcard evaluation (*, ?, [seq])
        elif fnmatch.fnmatch(fh_lower, t_lower):
            mask[i] = True
            
    return mask

# Identifiers that [NCBI] and [PDB] header lists select by: RefSeq (WP_0123.1)
# or GenBank (ABC12345.1) protein accessions, and PDB IDs with an optional
# chain (1XYZ_A). Network headers are canonical, and sanitize_header turned
# their spaces into "_", so an identifier ends at any character but a letter or
# a digit. \b counts "_" as part of a word and missed WP_0123.1 in
# WP_0123.1_kinase.
_NCBI_ACCESSION_PATTERN = re.compile(
    r'(?<![A-Z0-9])([A-Z]{2}_\d+(?:\.\d+)?|[A-Z]{3}\d{5,7}(?:\.\d+)?)(?![A-Z0-9])',
    re.IGNORECASE,
)
_PDB_ID_PATTERN = re.compile(
    r'(?<![A-Z0-9])([1-9][A-Z0-9]{3})(?:_[A-Z0-9]+)?(?![A-Z0-9])',
    re.IGNORECASE,
)

def _read_header_list(load_path):
    """Entries of a header list: a FASTA file's header lines, or another file's lines.

    A UTF-8 byte-order mark is skipped, as Sequence_Utils.read_fasta skips it.
    Text that is not UTF-8 raises SelectionContextError: decoding it anyway
    would change characters and leave those entries matching nothing.
    """
    is_fasta = load_path.lower().endswith('.fasta')
    entries = []
    try:
        with open(load_path, 'r', encoding='utf-8-sig') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if not is_fasta:
                    entries.append(line)
                elif line.startswith('>'):
                    entries.append(line[1:])
    except UnicodeDecodeError as error:
        raise SelectionContextError(
            f"Selection file '{os.path.basename(load_path)}' is not UTF-8 text.\n"
            "Save it with UTF-8 encoding and try again."
        ) from error
    return entries

def evaluate_file_mask(full_headers, target):
    """Evaluates an external header file, FASTA file, or NCBI/PDB list into a boolean mask.

    Entries are sanitized as network headers were (Sequence_Utils.sanitize_header),
    so a header copied from the source FASTA, description and all, selects its
    node. A plain list matches whole headers. An [NCBI] or [PDB] list matches
    each entry's first accession or PDB ID with a header's first accession, or
    with any PDB ID in it. Letter case is ignored. Entries that select no node
    are printed.
    """
    from utilities.Sequence_Utils import sanitize_header

    mask = np.zeros(len(full_headers), dtype=bool)
    prefix = target.strip().lower()
    if prefix.startswith('[ncbi]'):
        id_pattern = _NCBI_ACCESSION_PATTERN
    elif prefix.startswith('[pdb]'):
        id_pattern = _PDB_ID_PATTERN
    else:
        id_pattern = None

    load_path, header_dir = _selection_file_path(target)
    if not os.path.isfile(load_path):
        print(f"Warning: Could not find file '{os.path.basename(load_path)}' in {header_dir}")
        return mask

    # Distinct entries, by canonical spelling, with the key each selects by:
    # that spelling, or the first identifier in it (None if it holds none).
    entries = {}
    for entry in _read_header_list(load_path):
        canonical = sanitize_header(entry)[0]
        if id_pattern is None:
            key = canonical.lower()
        else:
            match = id_pattern.search(canonical)
            key = match.group(1).lower() if match else None
        entries.setdefault(canonical.lower(), (canonical, key))
    wanted = {key for _, key in entries.values() if key is not None}

    found = set()
    for i, header in enumerate(full_headers):
        if id_pattern is None:
            keys = [header]
        elif id_pattern is _PDB_ID_PATTERN:
            keys = id_pattern.findall(header)
        else:
            match = id_pattern.search(header)
            keys = [match.group(1)] if match else []
        for key in keys:
            key = key.lower()
            if key in wanted:
                mask[i] = True
                found.add(key)

    unmatched = [canonical for canonical, key in entries.values() if key not in found]
    if unmatched:
        shown = unmatched[:_SUGGESTION_LIMIT]
        lines = [
            f"Warning: {len(unmatched)} of {len(entries)} entries in "
            f"'{os.path.basename(load_path)}' matched no node:"
        ]
        lines.extend(f"  {entry}" for entry in shown)
        if len(unmatched) > len(shown):
            lines.append(f"  ... (+{len(unmatched) - len(shown)} more)")
        print("\n".join(lines))
    return mask

def evaluate_label_mask(full_headers, cluster_labels, group_labels, target):
    """Evaluates a #Cluster_N#, #Noise#, or #Custom_Group# into a boolean mask."""
    mask = np.zeros(len(full_headers), dtype=bool)
    resolved = resolve_label_target(cluster_labels, group_labels, target)

    if resolved.kind == "noise":
        return np.asarray(cluster_labels) == -1
    if resolved.kind == "cluster":
        return np.asarray(cluster_labels) == resolved.cluster_id

    target_lower = resolved.name.lower()
    for i in range(len(full_headers)):
        if i >= len(group_labels) or not group_labels[i]:
            continue
        if any(str(group).lower() == target_lower for group in group_labels[i]):
            mask[i] = True

    return mask

def evaluate_aa_mask(full_headers, alignment, target_aa, target_pos_label, viewer_to_aln, valid_indices):
    """Evaluates an Amino Acid position into a boolean mask."""
    mask = np.zeros(len(full_headers), dtype=bool)
    
    if alignment is None or alignment.aln is None:
        print(f"Warning: Cannot evaluate '{target_aa}{target_pos_label}' without an alignment.")
        return mask
        
    target_aa = target_aa.upper()
    if not alignment.label_to_col or target_pos_label not in alignment.label_to_col:
        return mask
        
    col_idx = alignment.label_to_col[target_pos_label]
    is_gap_query = (target_aa == '_')
    aln_rows = viewer_to_aln[valid_indices]

    # The viewer's sparse alignment and frozen snapshot adapters both answer
    # whole-column residue checks.
    if is_gap_query:
        mask_dash = alignment.aln.bulk_residue_check(col_idx, '-')
        mask_dot = alignment.aln.bulk_residue_check(col_idx, '.')
        aln_mask = mask_dash | mask_dot
    else:
        aln_mask = alignment.aln.bulk_residue_check(col_idx, target_aa)
    mask[valid_indices] = aln_mask[aln_rows]

    return mask


def evaluate_aa_group_mask(
    full_headers,
    alignment,
    target_aas,
    target_pos_label,
    viewer_to_aln,
    valid_indices,
):
    """Evaluate membership in a residue set at one displayed alignment position."""
    mask = np.zeros(len(full_headers), dtype=bool)
    for target_aa in target_aas:
        mask |= evaluate_aa_mask(
            full_headers,
            alignment,
            target_aa,
            target_pos_label,
            viewer_to_aln,
            valid_indices,
        )
    return mask


def evaluate_metadata_mask(full_headers, metadata, target):
    """Evaluates a metadata query of the form 'PropertyOperatorValue' into a boolean mask.
    
    Example targets: 'Length>500', 'Organism=Escherichia coli', 'Organism=*coli*'
    """
    mask = np.zeros(len(full_headers), dtype=bool)
    if not metadata:
        print("Warning: No metadata loaded in the viewer to evaluate query.")
        return mask

    # Regex to extract Property, Operator, and Value
    # Supports operators: >=, <=, !=, ==, >, <, =
    match = _METADATA_QUERY_PATTERN.match(target.strip())
    if not match:
        print(f"Warning: Invalid metadata query format '{target}'. Use 'KeyOperatorValue' (e.g. 'Length>500').")
        return mask
        
    key, op, val_str = match.groups()
    val_str = val_str.strip()
    
    # Resolve the metadata key (case-insensitive lookup)
    meta_key = None
    for k in metadata.keys():
        if k.lower() == key.lower():
            meta_key = k
            break
            
    if meta_key is None:
        print(f"Warning: Metadata property '{key}' not found. Available properties: {list(metadata.keys())}")
        return mask
        
    meta_prop = metadata[meta_key]
    prop_type = meta_prop["type"]
    prop_vals = meta_prop["values"]
    
    # --- Numeric Evaluation ---
    if prop_type == "number":
        try:
            val = parse_metadata_number(val_str)
        except ValueError:
            # Range syntax: 100-200, or (-1)-0 with a negative bound.
            if op in ('=', '=='):
                try:
                    bounds = parse_metadata_range(val_str)
                except SelectionExpressionError as error:
                    print(f"Warning: {error}")
                    return mask
                if bounds is not None:
                    low_val, high_val = bounds
                    return (prop_vals >= low_val) & (prop_vals <= high_val)
            print(f"Warning: Cannot convert value '{val_str}' to number for property '{meta_key}'.")
            return mask
            
        if op == '>':
            mask = prop_vals > val
        elif op == '<':
            mask = prop_vals < val
        elif op == '>=':
            mask = prop_vals >= val
        elif op == '<=':
            mask = prop_vals <= val
        elif op in ('==', '='):
            mask = prop_vals == val
        elif op == '!=':
            # Like every other comparison, != skips nodes with no value (NaN);
            # !{Length=100} selects the complement, missing values included.
            mask = (prop_vals != val) & ~np.isnan(prop_vals)

    # --- Text/String Evaluation ---
    else:
        # Optional quotes delimit the value for wildcard patterns too:
        # {Organism="*coli"} is the pattern *coli, not "*coli" with its quotes.
        t_val = val_str.lower().strip('"\'')

        if op in ('==', '='):
            if '*' in t_val or '?' in t_val:
                for i, v in enumerate(prop_vals):
                    if fnmatch.fnmatch(str(v).lower(), t_val):
                        mask[i] = True
            else:
                for i, v in enumerate(prop_vals):
                    if t_val in str(v).lower():
                        mask[i] = True
        elif op == '!=':
            if '*' in t_val or '?' in t_val:
                for i, v in enumerate(prop_vals):
                    if not fnmatch.fnmatch(str(v).lower(), t_val):
                        mask[i] = True
            else:
                for i, v in enumerate(prop_vals):
                    if t_val not in str(v).lower():
                        mask[i] = True
                        
    return mask

def _validate_metadata_syntax(target):
    match = _METADATA_QUERY_PATTERN.fullmatch(target.strip())
    if not match:
        raise SelectionExpressionError(
            f"Invalid metadata predicate '{{{target}}}'. Use '{{PropertyOperatorValue}}', "
            "for example '{{Length>500}}'."
        )
    if not match.group(3).strip():
        raise SelectionExpressionError(
            f"Metadata predicate '{{{target}}}' is missing a comparison value."
        )


class _SelectionSyntaxParser:
    """Recursive-descent parser for the shared Boolean selection language."""

    def __init__(self, text):
        self.text = text
        self.length = len(text)
        self.position = 0

    def _skip_space(self):
        while self.position < self.length and self.text[self.position].isspace():
            self.position += 1

    def _error(self, message=None):
        if message is None:
            message = (
                f"Invalid Boolean expression '{self.text}'. Ensure operators and "
                "parentheses are complete and do not place spaces inside individual "
                "predicates."
            )
        raise SelectionExpressionError(message)

    def parse(self):
        self._skip_space()
        if self.position >= self.length:
            raise SelectionExpressionError("Boolean selection expression is empty.")
        expression = self._parse_or()
        self._skip_space()
        if self.position != self.length:
            self._error()
        return expression

    def _parse_or(self):
        node = self._parse_xor()
        while True:
            self._skip_space()
            if self.position >= self.length or self.text[self.position] != "|":
                return node
            self.position += 1
            node = _SelectionBinary("|", node, self._parse_xor())

    def _parse_xor(self):
        node = self._parse_and()
        while True:
            self._skip_space()
            if self.position >= self.length or self.text[self.position] != "^":
                return node
            self.position += 1
            node = _SelectionBinary("^", node, self._parse_and())

    def _parse_and(self):
        node = self._parse_unary()
        while True:
            self._skip_space()
            if self.position >= self.length or self.text[self.position] != "&":
                return node
            self.position += 1
            node = _SelectionBinary("&", node, self._parse_unary())

    def _parse_unary(self):
        self._skip_space()
        if self.position < self.length and self.text[self.position] == "!":
            self.position += 1
            return _SelectionUnary(self._parse_unary())
        return self._parse_primary()

    def _delimited_atom(self, delimiter, kind):
        start = self.position
        end = self.text.find(delimiter, start + 1)
        if end == -1:
            self._error(f"Unterminated {kind} target in Boolean expression '{self.text}'.")
        value = self.text[start + 1:end]
        if not value:
            self._error(
                f"Boolean expression '{self.text}' contains empty or malformed targets."
            )
        self.position = end + 1
        if kind == "metadata":
            _validate_metadata_syntax(value)
        if kind == "string" and value.lower() == "$sele$":
            return _SelectionAtom("selection", "$sele$")
        return _SelectionAtom(kind, value)

    def _parse_primary(self):
        self._skip_space()
        if self.position >= self.length:
            self._error()

        start = self.position
        lower_remaining = self.text[start:].lower()
        if lower_remaining.startswith("$sele$"):
            self.position += len("$sele$")
            return _SelectionAtom("selection", "$sele$")

        char = self.text[start]
        if char == '"':
            return self._delimited_atom('"', "string")
        if char == "@":
            return self._delimited_atom("@", "file")
        if char == "#":
            return self._delimited_atom("#", "label")
        if char == "{":
            return self._delimited_atom("}", "metadata")

        grouped = _AA_GROUP_PREDICATE_PATTERN.match(self.text, start)
        if grouped:
            self.position = grouped.end()
            target_aas = _normalize_aa_group(grouped.group(1))
            position = grouped.group(2) or grouped.group(3)
            return _SelectionAtom("aa_group", (target_aas, position))

        bare_group = _BARE_NEGATIVE_AA_GROUP_PATTERN.match(self.text, start)
        if bare_group:
            target_aas, position = bare_group.groups()
            raise SelectionExpressionError(
                f"Negative alignment position '({target_aas}){position}' must be written as "
                f"'({target_aas})({position})'. Parentheses are required around negative positions."
            )

        if char == "(":
            self.position += 1
            node = self._parse_or()
            self._skip_space()
            if self.position >= self.length or self.text[self.position] != ")":
                self._error()
            self.position += 1
            return node

        aa_match = _AA_PREDICATE_PATTERN.match(self.text, start)
        if aa_match:
            self.position = aa_match.end()
            aa = aa_match.group(1)
            position = aa_match.group(2) or aa_match.group(3)
            return _SelectionAtom("aa", (aa, position))

        bare_negative = _BARE_NEGATIVE_AA_PATTERN.match(self.text, start)
        if bare_negative:
            aa, position = bare_negative.groups()
            raise SelectionExpressionError(
                f"Negative alignment position '{aa}{position}' must be written as "
                f"'{aa}({position})'. Parentheses are required around negative positions."
            )

        raise _NotSelectionExpression()


def _looks_expression_like(text):
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.lower() == "$sele$" or stripped[0] in '"@#{(!':
        return True
    if any(operator in stripped for operator in "&|^!"):
        return True
    if re.match(r'^[a-zA-Z_]\(?-?[\d.]', stripped):
        return True
    if re.match(r'^\([a-zA-Z]+\)', stripped):
        return True
    return False


def _parse_selection_expression(text):
    if not isinstance(text, str):
        raise _NotSelectionExpression()
    return _SelectionSyntaxParser(text).parse()


def classify_selection_expression(text):
    """Classify text without consulting viewer data or the filesystem."""
    try:
        expression = _parse_selection_expression(text)
        return SelectionClassification(
            SelectionClassificationKind.VALID_EXPRESSION,
            expression=expression,
        )
    except _NotSelectionExpression:
        if _looks_expression_like(str(text)):
            error = SelectionExpressionError(
                f"Invalid Boolean expression '{text}'. Ensure operators and parentheses "
                "are complete and do not place spaces inside individual predicates."
            )
            return SelectionClassification(
                SelectionClassificationKind.MALFORMED_EXPRESSION,
                error=error,
            )
        return SelectionClassification(SelectionClassificationKind.NOT_EXPRESSION)
    except SelectionExpressionError as error:
        return SelectionClassification(
            SelectionClassificationKind.MALFORMED_EXPRESSION,
            error=error,
        )


def parse_selection_expression(text):
    """Return a context-free Boolean AST or raise a syntax error."""
    classification = classify_selection_expression(text)
    if classification.kind == SelectionClassificationKind.VALID_EXPRESSION:
        return classification.expression
    if classification.kind == SelectionClassificationKind.MALFORMED_EXPRESSION:
        raise classification.error
    raise SelectionExpressionError(
        f"'{text}' is not a Boolean selection expression."
    )


def get_selected_mask(viewer):
    """Return the current UI selection as a stable node-length Boolean mask."""
    n_nodes = int(getattr(viewer, "n_nodes", len(getattr(viewer, "full_headers", []))))
    mask = np.zeros(n_nodes, dtype=bool)
    for index in getattr(viewer, "selected_indices", []) or []:
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= index < n_nodes:
            mask[index] = True
    return mask


def _evaluate_selection_node(
    node,
    viewer_to_aln,
    valid_indices,
    full_headers,
    cluster_labels,
    group_labels,
    alignment,
    metadata,
    selection_mask,
):
    if isinstance(node, _SelectionUnary):
        return ~_evaluate_selection_node(
            node.operand,
            viewer_to_aln,
            valid_indices,
            full_headers,
            cluster_labels,
            group_labels,
            alignment,
            metadata,
            selection_mask,
        )
    if isinstance(node, _SelectionBinary):
        left = _evaluate_selection_node(
            node.left,
            viewer_to_aln,
            valid_indices,
            full_headers,
            cluster_labels,
            group_labels,
            alignment,
            metadata,
            selection_mask,
        )
        right = _evaluate_selection_node(
            node.right,
            viewer_to_aln,
            valid_indices,
            full_headers,
            cluster_labels,
            group_labels,
            alignment,
            metadata,
            selection_mask,
        )
        if node.operator == "&":
            return left & right
        if node.operator == "|":
            return left | right
        if node.operator == "^":
            return left ^ right
        raise SelectionExpressionError(f"Unsupported Boolean operator '{node.operator}'.")

    if not isinstance(node, _SelectionAtom):
        raise SelectionExpressionError("Boolean expression contains an unsupported syntax node.")

    if node.kind == "string":
        return _LogicMask.known(evaluate_string_mask(full_headers, node.value))
    if node.kind == "file":
        _validate_file_target(node.value)
        return _LogicMask.known(evaluate_file_mask(full_headers, node.value))
    if node.kind == "metadata":
        _validate_metadata_target(metadata, node.value)
        return _LogicMask.known(
            evaluate_metadata_mask(full_headers, metadata, node.value)
        )
    if node.kind == "label":
        return _LogicMask.known(
            evaluate_label_mask(full_headers, cluster_labels, group_labels, node.value)
        )
    if node.kind == "selection":
        if selection_mask is None:
            raise SelectionContextError(
                "$sele$ cannot be evaluated because the current UI selection was not supplied."
            )
        selected = np.asarray(selection_mask, dtype=bool)
        if selected.shape != (len(full_headers),):
            raise SelectionContextError(
                "$sele$ selection mask does not match the current SSN node count."
            )
        return _LogicMask.known(selected)
    if node.kind == "aa_group":
        target_aas, position = node.value
        display_target = f"({''.join(target_aas)})"
        _validate_aa_target(alignment, display_target, position)
        mask = evaluate_aa_group_mask(
            full_headers,
            alignment,
            target_aas,
            position,
            viewer_to_aln,
            valid_indices,
        )
        return _LogicMask.partially_known(mask, np.asarray(viewer_to_aln) >= 0)
    if node.kind == "aa":
        aa, position = node.value
        _validate_aa_target(alignment, aa, position)
        mask = evaluate_aa_mask(
            full_headers,
            alignment,
            aa,
            position,
            viewer_to_aln,
            valid_indices,
        )
        return _LogicMask.partially_known(mask, np.asarray(viewer_to_aln) >= 0)
    raise SelectionExpressionError(f"Unsupported Boolean target kind '{node.kind}'.")


def evaluate_selection_expression(
    expression,
    viewer_to_aln,
    valid_indices,
    full_headers,
    cluster_labels=None,
    group_labels=None,
    alignment=None,
    metadata=None,
    selection_mask=None,
):
    """Evaluate a parsed Boolean AST against current viewer context."""
    result = _evaluate_selection_node(
        expression,
        viewer_to_aln,
        valid_indices,
        full_headers,
        cluster_labels,
        group_labels,
        alignment,
        metadata,
        selection_mask,
    ).to_bool()
    if result.shape != (len(full_headers),):
        raise SelectionExpressionError(
            "Boolean expression did not resolve to one selection value per SSN node. "
            "Check for empty or malformed targets."
        )
    return result


def parse_advanced_expression(
    expr,
    viewer_to_aln,
    valid_indices,
    full_headers,
    cluster_labels=None,
    group_labels=None,
    alignment=None,
    metadata=None,
    selection_mask=None,
):
    """Compatibility wrapper that parses and evaluates one Boolean expression."""
    expression = parse_selection_expression(expr)
    return evaluate_selection_expression(
        expression,
        viewer_to_aln,
        valid_indices,
        full_headers,
        cluster_labels,
        group_labels,
        alignment,
        metadata,
        selection_mask,
    )

def show_status(viewer, message):
    """Show message, a Message or plain text, on the Viewer's console line.

    Every write to the line goes through here, because it is the one place
    command feedback is translated: the terminal and the command portal (MCP
    clients and the agent page) keep a Message's English text.
    """
    viewer.console_text.text = display_text(message)

def print_help(viewer, msg, *, terminal_msg=None, report_message=True):
    """Prints help/errors to CLI, and a notification or status to the viewer console."""
    from Viewer_Command_Portal import report
    if report_message:
        report(message=msg if terminal_msg is None else terminal_msg, viewer=viewer)
    print(f"\n{msg if terminal_msg is None else terminal_msg}")

    if hasattr(viewer, 'console_text'):
        # Display single-line status, errors, warnings, or help headers directly on the on-screen console
        shown = display_text(msg)
        first_line = shown.split('\n')[0] if '\n' in shown else shown
        show_status(viewer, first_line.strip())
        
        if hasattr(viewer, 'update_console_background'):
            viewer.update_console_background()


def execute_reset(viewer, targets):
    """Executes reset on the specified targets.

    Raises ValueError, before any undo state is saved or anything is reset,
    when no target is given or any target is unknown.
    """
    from commands import reset as reset_command
    reset_command.check_targets(targets)

    targets_found = []
    needs_update = False

    viewer._save_state()

    for p in targets:
        base_p = reset_command.target_name(p)

        if base_p == "color":
            if hasattr(viewer, 'current_colors'):
                import matplotlib.colors as mcolors
                n_rgba = mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)
                viewer.current_colors[:] = n_rgba
            needs_update = True
            targets_found.append("colors")
            
        elif base_p == "size":
            if hasattr(viewer, 'current_sizes'):
                viewer.current_sizes.fill(cfg.NODE_SIZE)
            needs_update = True
            targets_found.append("sizes")
        
        elif base_p == "shape":
            if hasattr(viewer, 'current_shapes'):
                viewer.current_shapes.fill('disc')
            needs_update = True
            targets_found.append("shapes")

        elif base_p == "cluster":
            viewer.cluster_labels = None
            if hasattr(viewer, 'label_visuals'):
                for visual in viewer.label_visuals:
                    visual.parent = None
                viewer.label_visuals = []
            viewer.tooltip.text = "" 
            targets_found.append("clusters")

        elif base_p == "group":
            viewer.group_labels = [set() for _ in range(viewer.n_nodes)]
            if hasattr(viewer, 'label_visuals'):
                for visual in viewer.label_visuals:
                    visual.parent = None
                viewer.label_visuals = []
            viewer.tooltip.text = "" 
            targets_found.append("groups")
                
        elif base_p in ["hide", "hidden"]:
            viewer.visible_mask.fill(True)
            needs_update = True
            targets_found.append("hidden")
            
        elif base_p == "network":
            if hasattr(viewer, 'original_pos'):
                viewer.pos = viewer.original_pos.copy()
            needs_update = True
            targets_found.append("network")

        elif base_p in ["order", "layer"]:
            reset_command.reset_node_render_order(viewer)
            needs_update = True
            if "node order" not in targets_found:
                targets_found.append("node order")

    if needs_update:
        viewer.update_nodes()
        if "hidden" in targets_found or "network" in targets_found:
            viewer.update_edges()

    msg = f"Reset successful: {', '.join(targets_found)}."

    show_status(viewer, msg)
    print(f"{msg}")
    if hasattr(viewer, 'update_console_background'):
        viewer.update_console_background()

    return msg

# Shared command dispatch and explicit outcome reporting.
def command_succeeded(viewer, message=None, artifact=None):
    from Viewer_Command_Portal import report
    report('succeeded', message, artifact, viewer)

def command_failed(viewer, message):
    from Viewer_Command_Portal import report
    report('failed', str(message), viewer=viewer)

def command_cancelled(viewer, message):
    from Viewer_Command_Portal import report
    report('cancelled', str(message), viewer=viewer)

def command_artifact(viewer, path):
    from Viewer_Command_Portal import report
    report(artifact=path, viewer=viewer)

def execute_command(viewer, cmd_str, record_history=True, silent=False):
    from Viewer_Command_Portal import CURRENT, bind
    from PySide6 import QtCore
    if getattr(viewer, '_command_dispatch_active', False):
        # Event processing may reenter manual command dispatch. Defer that input.
        def deferred_manual():
            with bind(None):
                execute_command(viewer, cmd_str, record_history, silent)
        QtCore.QTimer.singleShot(20, deferred_manual)
        return
    context = CURRENT.get()
    if context is not None and str(cmd_str).split() and str(cmd_str).split()[0].lower() == 'agent':
        argument = str(cmd_str).partition(' ')[2].strip()
        configuration = (not argument or argument.lower() in {'help', '-h', '--help', 'off', 'deactivate', '--register-only'} or (argument.startswith('<') and argument.endswith('>')))
        if not configuration:
            command_failed(viewer, 'Model-generated agent commands cannot start another model request.')
            return
    viewer._command_dispatch_active = True
    try:
        _dispatch_user_command(viewer, cmd_str, record_history, silent)
    finally:
        viewer._command_dispatch_active = False

def _dispatch_user_command(viewer, cmd_str, record_history=True, silent=False):
    import os
    import importlib
    from vispy import app
    cmd_str = cmd_str.strip()
    if not cmd_str:
        command_failed(viewer, "Empty command")
        return

    # Normalize reverse commands (e.g., 'color reset' -> 'reset color', 'help color' -> 'color help')


    # --- 0. FILE-BACKED HISTORY ---
    # Only record if it's different from the very last command typed
    if record_history:
        if not viewer.command_history or viewer.command_history[-1] != cmd_str:
            viewer.command_history.append(cmd_str)
            try:
                os.makedirs(os.path.dirname(viewer.history_file), exist_ok=True)
                with open(viewer.history_file, "a", encoding="utf-8") as f:
                    f.write(cmd_str + "\n")

                # Truncate if file exceeds 1 MB (1,048,576 bytes)
                if os.path.getsize(viewer.history_file) > 1048576:
                    # Keep latest ~2000 lines (safely under 1MB limit for string paths)
                    viewer.command_history = viewer.command_history[-2000:]
                    with open(viewer.history_file, "w", encoding="utf-8") as f:
                        for line in viewer.command_history:
                            f.write(line + "\n")
            except Exception as e:
                print(f"Warning: Failed to save history to {viewer.history_file} ({e})")

    # --- 3. PARSE COMMAND ---
    parts = cmd_str.split()
    if not parts: return

    command_name = parts[0].lower()
    args = parts[1:]

    # --- 6. DYNAMIC EXTERNAL COMMANDS ---
    try:
        module = importlib.import_module(f"commands.{command_name}")
        if getattr(module, "__spec__", None) is not None:
            importlib.reload(module)

        if hasattr(module, 'run'):
            if not silent and hasattr(viewer, 'console_text'):
                show_status(viewer, f"Running {command_name}...")
            if not silent and hasattr(viewer, 'update_console_background'):
                viewer.update_console_background()
            if hasattr(app, 'process_events'):
                app.process_events()
            module.run(viewer, args)
            if not silent and hasattr(viewer, 'update_console_background'):
                viewer.update_console_background()
            # Broadcast a complete browser state, including metadata shape.
            viewer.broadcast_metadata_state()
        else:
            command_failed(viewer, f"No run entry point in {command_name}")
            if not silent and hasattr(viewer, 'console_text'):
                show_status(viewer, f"Error: No 'run' in {command_name}")
            if not silent and hasattr(viewer, 'update_console_background'):
                viewer.update_console_background()

    except ModuleNotFoundError as error:
        command_failed(viewer, f"Unknown command: {command_name}" if error.name == f"commands.{command_name}" else f"Command dependency unavailable: {error.name}")
        if not silent and hasattr(viewer, 'console_text'):
            show_status(viewer, f"Unknown command: {command_name}")
        if not silent and hasattr(viewer, 'update_console_background'):
            viewer.update_console_background()
    except Exception as e:
        command_failed(viewer, str(e))
        if not silent and hasattr(viewer, 'console_text'):
            show_status(viewer, f"Error: {e}")
        if not silent and hasattr(viewer, 'update_console_background'):
            viewer.update_console_background()
        print(f"Command Error: {e}")
        import traceback
        traceback.print_exc()
