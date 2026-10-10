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

import os
import re
import numpy as np
from collections import Counter
import EMAPSSN_Config as cfg
import Command_Engine
from utilities.Localization import JoinedMessage, Message
from utilities.Sequence_Utils import (
    DISPLAYED_POSITION_ATOM_PATTERN,
    format_alignment_offset_display,
    normalize_displayed_position_atom,
    reject_bare_negative_positions,
)


_QUERY_POSITION_ENDPOINT_PATTERN = (
    rf"(?:{DISPLAYED_POSITION_ATOM_PATTERN}|E(?:ND)?)"
)
_QUERY_POSITION_RANGE_RE = re.compile(
    rf"^({_QUERY_POSITION_ENDPOINT_PATTERN})\s*-\s*"
    rf"({_QUERY_POSITION_ENDPOINT_PATTERN})$",
    re.IGNORECASE,
)
_FREQUENCY_TARGET_PATTERN = r"(?:\([A-Za-z]*\)|[A-Za-z_]+)"
_FREQUENCY_VALUE_PATTERN = r"\d+(?:\.\d+)?%?"
_PARENTHESIZED_FREQUENCY_CONDITION_RE = re.compile(
    rf"\(\s*({_FREQUENCY_TARGET_PATTERN})\s*(>=|<=|>|<)\s*"
    rf"({_FREQUENCY_VALUE_PATTERN})\s*\)"
)
_SINGLE_FREQUENCY_CONDITION_RE = re.compile(
    rf"\s*({_FREQUENCY_TARGET_PATTERN})\s*(>=|<=|>|<)\s*"
    rf"({_FREQUENCY_VALUE_PATTERN})\s*"
)
_FREQUENCY_PARENTHESES_MESSAGE = (
    "Error: Individual frequency arguments in multi-condition queries must be "
    "enclosed in parentheses '()'.\n"
    "Example: [K>10%], [(K>0.1) & (R>0.05)], "
    "[((RHK)>50%) & ((DE)>20%)]"
)


class _FrequencyParenthesesError(ValueError):
    """Raised when a compound frequency comparison lacks its outer parentheses."""


class _FrequencyLogicParser:
    """Evaluate mask atoms with the shared Boolean-operator precedence."""

    def __init__(self, text, masks):
        self.text = text
        self.masks = masks
        self.position = 0

    def _skip_space(self):
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def _error(self):
        raise ValueError(Message(
            "Invalid frequency Boolean expression. Ensure operators and "
            "parentheses are complete."
        ))

    def parse(self):
        self._skip_space()
        if self.position >= len(self.text):
            self._error()
        result = self._parse_or()
        self._skip_space()
        if self.position != len(self.text):
            self._error()
        return result

    def _parse_or(self):
        result = self._parse_xor()
        while True:
            self._skip_space()
            if self.position >= len(self.text) or self.text[self.position] != "|":
                return result
            self.position += 1
            result = result | self._parse_xor()

    def _parse_xor(self):
        result = self._parse_and()
        while True:
            self._skip_space()
            if self.position >= len(self.text) or self.text[self.position] != "^":
                return result
            self.position += 1
            result = result ^ self._parse_and()

    def _parse_and(self):
        result = self._parse_unary()
        while True:
            self._skip_space()
            if self.position >= len(self.text) or self.text[self.position] != "&":
                return result
            self.position += 1
            result = result & self._parse_unary()

    def _parse_unary(self):
        self._skip_space()
        if self.position < len(self.text) and self.text[self.position] == "!":
            self.position += 1
            return ~self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self):
        self._skip_space()
        if self.position >= len(self.text):
            self._error()
        if self.text[self.position] == "(":
            self.position += 1
            result = self._parse_or()
            self._skip_space()
            if self.position >= len(self.text) or self.text[self.position] != ")":
                self._error()
            self.position += 1
            return result

        match = re.match(r"M_\d+", self.text[self.position:])
        if not match:
            self._error()
        key = match.group(0)
        self.position += len(key)
        try:
            return self.masks[key]
        except KeyError:
            self._error()


def _normalize_frequency_target(target_raw):
    target = target_raw.strip().upper()
    if target.startswith('(') and target.endswith(')'):
        target_aas = target[1:-1]
        if len(target_aas) < 2:
            raise ValueError(Message(
                "Grouped amino-acid target '{target}' must contain at least two "
                "one-letter residue symbols.",
                target=target,
            ))
        return tuple(dict.fromkeys(target_aas))
    if len(target) != 1 and target != 'GAP':
        raise ValueError(Message(
            "Unknown frequency target '{target}'. Use one residue letter, GAP, _, "
            "or a parenthesized group such as (KR).",
            target=target_raw.strip(),
        ))
    return target


def _parse_frequency_threshold(value_text):
    value_clean = value_text.strip()
    if value_clean.endswith('%'):
        return float(value_clean[:-1]) / 100.0
    value = float(value_clean)
    return value if value <= 1.0 else (value / 100.0)


def _evaluate_frequency_condition(
    target_raw,
    operator,
    value_text,
    gap_fractions,
    aa_fractions,
    sequence_count=None,
):
    target = _normalize_frequency_target(target_raw)
    threshold = _parse_frequency_threshold(value_text)

    if isinstance(target, tuple):
        column_frequencies = np.zeros_like(gap_fractions, dtype=float)
        for target_aa in target:
            fractions = aa_fractions.get(
                target_aa,
                np.zeros_like(gap_fractions, dtype=float),
            )
            if sequence_count:
                # Add whole counts, so the group's fraction is one division,
                # exactly as a single residue's is: 0.1 + 0.2 is not 0.3.
                fractions = np.rint(fractions * sequence_count)
            column_frequencies += fractions
        if sequence_count:
            column_frequencies = column_frequencies / sequence_count
    elif target in {'_', 'GAP'}:
        column_frequencies = gap_fractions
    else:
        column_frequencies = aa_fractions.get(
            target,
            np.zeros_like(gap_fractions, dtype=float),
        )

    comparisons = {
        '>': np.greater,
        '<': np.less,
        '>=': np.greater_equal,
        '<=': np.less_equal,
    }
    return comparisons[operator](column_frequencies, threshold)


def evaluate_frequency_logic(inner, gap_fractions, aa_fractions, sequence_count=None):
    """Evaluate query frequency logic against precomputed per-column fractions.

    SEQUENCE_COUNT, the number of sequences the fractions divide by, lets a
    grouped target add whole counts instead of fractions.
    """
    masks = {}
    mask_idx = 0

    def condition_repl(match):
        nonlocal mask_idx
        target_raw, operator, value_text = match.groups()
        mask_key = f"M_{mask_idx}"
        masks[mask_key] = _evaluate_frequency_condition(
            target_raw,
            operator,
            value_text,
            gap_fractions,
            aa_fractions,
            sequence_count,
        )
        mask_idx += 1
        return mask_key

    expression = _PARENTHESIZED_FREQUENCY_CONDITION_RE.sub(
        condition_repl,
        inner,
    )
    if mask_idx == 0:
        single_match = _SINGLE_FREQUENCY_CONDITION_RE.fullmatch(inner)
        if single_match:
            expression = condition_repl(single_match)

    if re.search(r'[><]', expression) or mask_idx == 0:
        raise _FrequencyParenthesesError(_FREQUENCY_PARENTHESES_MESSAGE)

    result = _FrequencyLogicParser(expression, masks).parse()

    result_mask = np.asarray(result, dtype=bool)
    expected_shape = np.asarray(gap_fractions).shape
    if result_mask.shape != expected_shape:
        raise ValueError(Message(
            "Frequency logic did not resolve to one value per alignment position."
        ))
    return result_mask


def parse_query_positions(position_spec, valid_labels):
    """Expand a query position list against mapped alignment labels."""
    text = str(position_spec).strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]

    reject_bare_negative_positions(text)

    parsed_args = [part.strip() for part in text.split(',') if part.strip()]
    expanded_positions = []
    seen_positions = set()
    max_val = valid_labels[-1][0] if valid_labels else (0, 0)

    def parse_to_tuple(value):
        normalized = normalize_displayed_position_atom(value, allow_end=True)
        if normalized in {"E", "END"}:
            return max_val
        major_text, separator, insertion_text = normalized.partition('.')
        return int(major_text), int(insertion_text) if separator else 0

    for part in parsed_args:
        range_match = _QUERY_POSITION_RANGE_RE.fullmatch(part)
        if range_match:
            start_value, end_value = sorted(
                [parse_to_tuple(range_match.group(1)), parse_to_tuple(range_match.group(2))]
            )
            for value, label in valid_labels:
                if start_value <= value <= end_value and label not in seen_positions:
                    seen_positions.add(label)
                    expanded_positions.append(label)
            continue

        normalized = normalize_displayed_position_atom(part, allow_end=True)
        if normalized in {"E", "END"}:
            if valid_labels:
                normalized = valid_labels[-1][1]
        else:
            # Spell the label as the alignment does, as logo does: 01 is 1 and
            # (-0) is 0, and a range already reads its ends as numbers.
            major_text, separator, insertion_text = normalized.partition('.')
            normalized = str(int(major_text))
            if separator:
                normalized += f".{int(insertion_text)}"
        if normalized not in seen_positions:
            seen_positions.add(normalized)
            expanded_positions.append(normalized)

    return expanded_positions


def _subset_column_counts(matrix, target_rows):
    """Return column(col_idx) -> (n_gaps, [(code, count), ...]) for the target rows.

    One O(nnz) pass replaces a sparse matrix[target_rows, col] slice per column
    (each of which is itself O(nnz of the subset)). Residue codes come in the
    order Counter(dense_col[dense_col != 0]) would list them: first occurrence in
    target_rows order, so tie-ordering in the printed tables is unchanged.
    """
    from scipy import sparse

    n_rows = len(target_rows)
    if sparse.issparse(matrix):
        subset = sparse.csc_matrix(matrix[target_rows])
        subset.sort_indices()
        indptr, data = subset.indptr, subset.data
    else:
        dense = np.asarray(matrix)[target_rows]
        indptr = data = None

    def column(col_idx):
        if data is None:
            values = dense[:, col_idx]
        else:
            values = data[indptr[col_idx]:indptr[col_idx + 1]]
        residues = values[values != 0]
        if residues.size == 0:
            return n_rows - residues.size, []
        counts = np.bincount(residues)
        codes, first = np.unique(residues, return_index=True)
        ordered = codes[np.argsort(first, kind="stable")]
        return n_rows - residues.size, [(code, int(counts[code])) for code in ordered.tolist()]

    return column


# Under this percentage a residue has no column of its own in a position's line.
_MIN_LISTED_PERCENT = 1.0


def _print_position_composition(label, gap_percent, residue_percents):
    """Print one position's gap and residue percentages, the largest residue first.

    A residue under 1% has no column of its own; one extra line sums them, so
    the two lines together account for every residue present. X and the other
    symbols are residues here, as everywhere in query.
    """
    listed = sorted(
        ((residue, percent) for residue, percent in residue_percents.items()
         if percent >= _MIN_LISTED_PERCENT),
        key=lambda item: item[1], reverse=True,
    )
    omitted = [percent for percent in residue_percents.values() if 0.0 < percent < _MIN_LISTED_PERCENT]

    line = f"Pos {label:<8}\tGap {gap_percent:>5.1f}%"
    for residue, percent in listed:
        line += f" | {residue} {percent:>5.1f}%"
    print(line)
    if omitted:
        # Indented to the Gap column: the tab after the 12-character "Pos " and label reaches it.
        print(" " * 12 + "\t" + str(Message("other (<1% each): {percent}", percent=f"{sum(omitted):>5.1f}%")))


def print_help():
    print("""
    Subsection Query & Alignment Statistics Tool
    ==========================================
    Usage: query [EXPRESSION] [POSITIONS]
           query [EXPRESSION] [LOGIC_ARGUMENT]
           query help

    Description:
      Queries the loaded alignment for amino acid distribution at specified reference
      positions (Mode 1), OR searches for alignment positions matching specific amino 
      acid frequency criteria (Mode 2).
      
      Can query globally OR on a subset of nodes using logical sequence selection.

      * QUICK USE: If no expression is provided, the command automatically targets 
        the nodes currently selected in the viewer. If no nodes are selected, it 
        defaults to querying ALL nodes in the entire network.

      * Hidden nodes are included: they are used whenever the selection or
        expression covers them.

    Syntax Modes:
      1. Position Breakdown Mode:
         [POSITIONS] - Comma-separated list or ranges enclosed in brackets.
         Accepts decimal positions, and 'E' or 'END' for the last displayed position.
         Negative positions must be enclosed individually in parentheses.
         Example: [(-1), 0, 15.1, 20-30, 250-E, END] or [(-3)-(-1)]

      2. Position Frequency Search Mode:
         [LOGIC_ARGUMENT] - Frequency criteria with operators (>, <, >=, <=) and 
         logical operators (&, |, !, ^). Single arguments do NOT require ().
         Multi-condition queries MUST enclose each individual argument in ().
         Spaces are allowed. Accepts percentages (e.g. 10%) or decimals (e.g. 0.1).
         A bare number up to 1 is a fraction (1 means 100%, 0.5 means 50%); a larger
         bare number is a percentage (5 means 5%). Write 1% for 1 percent.
         Accepts residue codes (A-Z) and 'GAP' or '_' for gaps (case-insensitive).
         A residue code is one letter: write several residues as a group, (KR).
         Parenthesized residue sets sum their frequencies, e.g. [(RHK)>50%].
         Frequencies divide by all mapped sequences in the selected subset, so gaps
         reduce residue percentages.
         In multi-condition logic, an outer pair still encloses each comparison:
         [((RHK)>50%)&((DE)>20%)] or [((RHK)>50%)&(GAP<20%)].

    Output:
      Each position prints its gap percentage and then each residue of at least 1%,
      largest first. X and any other symbol in the alignment count as residues.
      Residues under 1% are summed on one extra line, 'other (<1% each)', so no
      residue present goes unreported.

    Sequence Selection Expression Targets (Do NOT use spaces inside expressions!):
      1. AA Position:  [AA][Pos] (e.g., P106, _100), or ([AA...])[Pos] for
                       alternatives (e.g., (RHK)71); negative positions require
                       parentheses (e.g., K(-1), (RHK)(-1))
      2. Header Text:  "[Text]"  (e.g., "3HMU", "*4A6T*")
      3. File Search:  @[File]@  (e.g., @my_list@, @my_seqs.fasta@)
      4. NCBI List:    @[NCBI][File]@ (Extracts & matches NCBI IDs from file and headers)
      5. Labels:       #[Name]#  (e.g., #cluster_1#, #noise#, #my_group#)
      6. UI Selection: $sele$    (Targets nodes currently selected in viewer)
      7. Metadata:     {Key Op Val} (e.g., {Length>500}, {Organism=*coli*});
                       ranges use = (e.g., {Length=300-500}), and negative
                       range bounds require parentheses (e.g., {GRAVY=(-1)-0})

    Logic Operators:
      & (AND), | (OR), ! (NOT), ^ (XOR)

    Selection Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. Invalid references abort the query. A valid selection
      expression may match zero nodes: the query then reports that no sequences
      matched and succeeds. Nodes that match but are not in the alignment leave
      nothing to query, and the query fails.

    Examples:
      query [10, 15, 20-30]                         (Queries pos 10, 15, and 20 to 30)
      query [(-1),0,10.1]                           (Queries negative, zero, and insertion positions)
      query #cluster_1# [(-3)-2]                    (Queries a range crossing zero in cluster 1)
      query [K>10%]                                 (Finds positions where Lysine > 10%)
      query [(RHK)>50%]                             (Finds positions where R+H+K > 50%)
      query [((RHK)>50%) & ((DE)>20%)]              (Combines grouped comparisons)
      query [(K>0.1) & (R>0.05)]                    (Finds positions where K > 10% and R > 5%)
      query P106 [(K>20%) | (R>20%)]                (Finds positions with K or R > 20% in Pro106 subset)
      query {Length>500} [!(GAP>30%) & (K>5%)]     (Finds positions with <30% gaps and >5% Lys in length>500)
    """)

def run(viewer, args):
    if not args:
        # The console line shows the first line; the usage is for the terminal.
        msg = JoinedMessage([
            Message("Error: Query command requires a POSITIONS or LOGIC_ARGUMENT parameter."),
            "Usage: query [POSITIONS] or query [LOGIC_ARGUMENT]",
        ], separator="\n")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    alignment = getattr(viewer, 'alignment', None)
    if alignment is None or alignment.aln is None:
        msg = Message("Error: No alignment loaded in the viewer.")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
        print(msg)
        return

    if len(alignment.aln) == 0:
        msg = Message(
            "Error: The selected MSA contains no aligned rows for the current network. "
            "Query analysis is unavailable."
        )
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
        print(msg)
        return

    # --- Reconstruct bracketed arguments (in case of spaces within brackets) ---
    reconstructed_args = []
    temp_bracket = []
    in_bracket = False
    
    for a in args:
        if '[' in a and not in_bracket:
            if a.count('[') > a.count(']'):
                in_bracket = True
                temp_bracket.append(a)
            else:
                reconstructed_args.append(a)
        elif in_bracket:
            temp_bracket.append(a)
            if ']' in a:
                joined = " ".join(temp_bracket)
                if joined.count('[') <= joined.count(']'):
                    reconstructed_args.append(joined)
                    temp_bracket = []
                    in_bracket = False
        else:
            reconstructed_args.append(a)
            
    if temp_bracket:
        reconstructed_args.extend(temp_bracket)
    args = reconstructed_args

    # --- 1. Extract Positions/Logic Argument (First argument containing brackets) ---
    bracket_indices = [i for i, a in enumerate(args) if a.strip().startswith('[') and a.strip().endswith(']')]
    
    if not bracket_indices:
        msg = Message("Error: No bracketed argument provided. Use [...] syntax (e.g., [10-20] or [K>10%]).")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return
        
    pos_idx = bracket_indices[0]
    pos_str = args.pop(pos_idx).strip()
    
    # --- 2. Isolate Expression & Apply Smart Fallbacks ---
    expr = "$sele$"
    if len(args) > 0:
        expr = " ".join(args) # Join the remaining tokens as one expression, as hide does

    # Smart Fallback to ALL Nodes
    if expr == "$sele$" and not getattr(viewer, 'selected_indices', []):
        expr = '"*"'  # The wildcard string matches all headers
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("No selection found. Defaulting to ALL nodes."))
        print("No nodes selected. Defaulting to ALL nodes in the network.")

    # --- 3. Compute Subset Rows ---
    # expr is never empty here, so every query runs on an evaluated subset.
    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)

    try:
        mask = Command_Engine.parse_advanced_expression(
            expr,
            viewer_to_aln,
            valid_indices,
            viewer.full_headers,
            getattr(viewer, 'cluster_labels', None),
            getattr(viewer, 'group_labels', None),
            getattr(viewer, 'alignment', None),
            metadata=getattr(viewer, 'metadata', None),
            selection_mask=Command_Engine.get_selected_mask(viewer),
        )
    except Exception as e:
        Command_Engine.report_selection_error(viewer, expr, e, Message("Query"))
        return

    valid_nodes = np.where(mask)[0]
    aln_rows = viewer_to_aln[valid_nodes]
    target_rows = aln_rows[aln_rows != -1]

    n_seqs = len(target_rows)

    if n_seqs == 0:
        # A valid expression may match no node, which is not a failure. Nodes
        # that match but are not in the alignment leave nothing to query, which is.
        nothing_matched = len(valid_nodes) == 0
        if nothing_matched:
            msg = Message("No sequences matched the expression '{expression}'. Aborting query.", expression=expr)
        else:
            msg = Message(
                "The expression '{expression}' matched %n node(s), but none is in the alignment. "
                "Aborting query.",
                n=len(valid_nodes), expression=expr,
            )
        Command_Engine.show_status(viewer, msg)
        print("-" * 50)
        print(msg)
        print("-" * 50)
        if nothing_matched:
            Command_Engine.command_succeeded(viewer, msg)
        else:
            Command_Engine.command_failed(viewer, msg)
        return

    # --- 4. Detect Mode: Position Breakdown (Mode 1) vs Frequency Search (Mode 2) ---
    inner = pos_str[1:-1].strip()
    is_logic_mode = bool(re.search(r'[><]', inner))

    # Retrieve alignment metadata for printing
    msa_file = getattr(viewer.alignment, 'msa_file', None) or getattr(cfg, 'MSA_FILE', 'None')
    if isinstance(msa_file, str) and msa_file:
        msa_file_display = os.path.basename(msa_file)
    else:
        msa_file_display = str(msa_file)

    if getattr(viewer.alignment, 'has_reference', False):
        ref_display = getattr(viewer.alignment, 'resolved_ref_full', None) or getattr(viewer, 'active_reference', 'None')
    else:
        ref_display = "None (Unanchored)"

    offset_display = format_alignment_offset_display(
        getattr(viewer, "alignment", None),
        getattr(viewer, "alignment_offset", getattr(cfg, "ALIGNMENT_OFFSET", 0)),
    )
    # Get mapped position labels in order
    label_to_col = getattr(viewer, 'alignment', None).label_to_col if getattr(viewer, 'alignment', None) else {}
    valid_labels = []
    for lbl in label_to_col.keys():
        try:
            parts = str(lbl).split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            valid_labels.append(((major, minor), lbl))
        except ValueError:
            pass
    valid_labels.sort(key=lambda x: x[0])
    ordered_pos_labels = [lbl for _, lbl in valid_labels]

    if is_logic_mode:
        # =====================================================================
        # MODE 2: POSITION FREQUENCY SEARCH MODE
        # =====================================================================
        print("-" * 50)
        print(f"QUERY POSITION SEARCH SUBSET: '{expr}' ({n_seqs} sequences mapped)")
        print(f"Alignment File:   {msa_file_display}")
        print(f"Active Reference: {ref_display}")
        print(f"Alignment Offset: {offset_display}")
        print(f"Search Criteria:  [{inner}]")
        print("-" * 50)

        n_cols = len(ordered_pos_labels)
        if n_cols == 0:
            message = Message("No valid alignment columns mapped.")
            print(message)
            if hasattr(viewer, 'console_text'):
                Command_Engine.show_status(viewer, message)
            Command_Engine.command_succeeded(viewer, message)
            return

        # Precompute AA and Gap frequencies for all mapped columns
        all_gap_fracs = np.zeros(n_cols, dtype=float)
        all_aa_fracs = {} # aa_char -> 1D numpy array of length n_cols

        column_counts = _subset_column_counts(viewer.alignment.aln.matrix, target_rows)
        for idx, pos_label in enumerate(ordered_pos_labels):
            col_idx = label_to_col[pos_label]
            n_gaps, residue_counts = column_counts(col_idx)

            gap_frac = n_gaps / n_seqs if n_seqs > 0 else 0.0
            all_gap_fracs[idx] = gap_frac

            for aa_int, count in residue_counts:
                aa_char = viewer.alignment.aln.int_to_aa.get(aa_int, 'X').upper()
                if aa_char not in all_aa_fracs:
                    all_aa_fracs[aa_char] = np.zeros(n_cols, dtype=float)
                all_aa_fracs[aa_char][idx] += (count / n_seqs) if n_seqs > 0 else 0.0

        try:
            pos_mask = evaluate_frequency_logic(
                inner,
                all_gap_fracs,
                all_aa_fracs,
                n_seqs,
            )
        except _FrequencyParenthesesError as error:
            # The terminal gets the full explanation with examples, in English.
            details = str(error)
            status = Message("Error: Individual frequency arguments must be enclosed in ()")
            if hasattr(viewer, 'console_text'):
                Command_Engine.show_status(viewer, status)
            Command_Engine.command_failed(viewer, status)
            print("-" * 50)
            print(details)
            print("-" * 50)
            return
        except ValueError as error:
            msg = Message("Error parsing position logic '[{logic}]': {error}", logic=inner, error=error)
            Command_Engine.command_failed(viewer, msg)
            if hasattr(viewer, 'console_text'):
                Command_Engine.show_status(viewer, msg)
            print(msg)
            return

        matching_indices = np.where(pos_mask)[0]
        matching_labels = [ordered_pos_labels[i] for i in matching_indices]

        print(f"Matching Positions ({len(matching_labels)} found):")
        if matching_labels:
            print(", ".join(str(lbl) for lbl in matching_labels))
            print("-" * 50)

            for idx in matching_indices:
                _print_position_composition(
                    ordered_pos_labels[idx],
                    all_gap_fracs[idx] * 100.0,
                    {aa_char: fracs[idx] * 100.0 for aa_char, fracs in all_aa_fracs.items()},
                )
        else:
            print("[No positions matched the search criteria]")

        print("-" * 50)
        message = Message("Found %n matching position(s). Check terminal.", n=len(matching_labels))
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, message)
        Command_Engine.command_succeeded(viewer, message)
        return


    # =========================================================================
    # MODE 1: POSITION BREAKDOWN MODE (Existing Behavior)
    # =========================================================================
    found_count = 0

    print("-" * 50)
    print(f"QUERY SUBSET: '{expr}' ({n_seqs} sequences mapped)")
    print(f"Alignment File:   {msa_file_display}")
    print(f"Active Reference: {ref_display}")
    print(f"Alignment Offset: {offset_display}")
    print("-" * 50)

    try:
        expanded_positions = parse_query_positions(inner, valid_labels)
    except ValueError as exc:
        msg = Message("Error: {error}", error=exc)
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    # Query the Matrix
    column_counts = _subset_column_counts(viewer.alignment.aln.matrix, target_rows)
    for pos in expanded_positions:
        if pos not in (getattr(viewer, 'alignment', None).label_to_col if getattr(viewer, 'alignment', None) else {}):
            print(f"Pos {pos: >5}: [Not found in active alignment mapping]")
            continue
            
        col_idx = viewer.alignment.label_to_col[pos]
        found_count += 1

        n_gaps, residue_counts = column_counts(col_idx)

        aa_counts = {}
        for aa_int, count in residue_counts:
            aa_char = viewer.alignment.aln.int_to_aa.get(aa_int, 'X')
            aa_counts[aa_char] = aa_counts.get(aa_char, 0) + count

        # Gap-Diluted Calculation
        gap_pct = (n_gaps / n_seqs) * 100.0 if n_seqs > 0 else 0.0
        residue_percents = {
            aa: (count / n_seqs) * 100.0 if n_seqs > 0 else 0.0
            for aa, count in aa_counts.items()
        }
        _print_position_composition(pos, gap_pct, residue_percents)
        
    print("-" * 50)
    
    if found_count > 0:
        message = Message("Queried %n position(s). Check terminal.", n=found_count)
        Command_Engine.show_status(viewer, message)
        Command_Engine.command_succeeded(viewer, message)
    else:
        # Nothing was queried, so the command did not do what it was asked.
        message = Message("No valid positions queried.")
        Command_Engine.show_status(viewer, message)
        Command_Engine.command_failed(viewer, message)
