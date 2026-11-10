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
import matplotlib.colors as mcolors
import Command_Engine
from commands.cluster import get_colored_cluster_name
from desktop.Node_Colors import NOISE_COLOR, declare_categories
from commands.query import _QUERY_POSITION_RANGE_RE, parse_query_positions
from utilities.Localization import JoinedMessage, Message
from utilities.Sequence_Utils import rejoin_split_brackets, split_position_list
from desktop.Command_Metadata import argument, entry, EXPRESSION

# --- What this command declares about itself ---
# Changes only the view.
AGENT_POLICY = 'run'

# Other names that run this command (Command_Declarations.canonical_command).
ALIASES = ('dist',)

CATALOG_ENTRY = entry('Color visible nodes by the amino acid each has at one alignment position; requires a nonempty MSA.',
    argument('expression', EXPRESSION + ' Defaults to selected nodes, otherwise all visible nodes. Hidden nodes are skipped and keep their colors, as do nodes that are not in the alignment. An expression that matches no visible node succeeds with a note; nodes that match but are not in the alignment make the command fail.'),
    argument('position', 'Required brackets holding exactly one displayed alignment position, numbered as in query: [120], [(-1)], [15.1], or E or END for the last displayed position. A list, a range or a frequency comparison is refused and nothing changes. Each residue has one fixed color (the same in every run); gaps and every other symbol (X, B, Z, U, O, *) are grey. A table of the residues, their counts and percentages prints to the Console.'),
    requires=('alignment',))

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 145
AGENT_PROMPT = """
`distribution [EXPRESSION] [POSITION] | distribution [POSITION]`
- Requires a loaded MSA and exactly one literal bracketed alignment position (`[120]`, or `[E]`/`[END]` for the last one), numbered as in `query`. A list, a range or a frequency comparison in the bracket fails and nothing changes. EXPRESSION uses the shared expression language and stays outside the brackets.
- Colors the targeted visible nodes by the residue each has at that position, one fixed color per residue; gaps and other symbols are grey. Nodes outside the target, hidden nodes and nodes not in the alignment keep their colors, and `undo` restores the previous colors. A table of the residues with counts and percentages prints to the Console.
- If EXPRESSION is omitted, the selected nodes are used; with no selection, all visible nodes. An expression that matches no visible node succeeds with a note; nodes that match but are not in the alignment make the command fail.
"""

# One color per residue, tab20 hue pairs by chemistry (tab20's grey pair is replaced by tab20b's gold pair, so
# grey means "no residue" only). The first of each pair is the darker one.
RESIDUE_COLORS = {
    'K': '#1f77b4', 'R': '#aec7e8',  # positive
    'D': '#d62728', 'E': '#ff9896',  # negative
    'N': '#e377c2', 'Q': '#f7b6d2',  # amide
    'S': '#ff7f0e', 'T': '#ffbb78',  # hydroxyl
    'I': '#2ca02c', 'V': '#98df8a',  # branched
    'L': '#bcbd22', 'M': '#dbdb8d',  # aliphatic
    'F': '#8c564b', 'Y': '#c49c94',  # aromatic
    'W': '#9467bd', 'H': '#c5b0d5',  # ring
    'A': '#17becf', 'G': '#9edae5',  # small
    'P': '#8c6d31', 'C': '#e7ba52',  # special
}
# The colors a node can get: a residue's opaque color, or cluster's noise color for a gap or any other symbol.
_RESIDUE_RGBA = {residue: mcolors.to_rgba(color) for residue, color in RESIDUE_COLORS.items()}
_NOISE_RGBA = tuple(NOISE_COLOR)


def print_help():
    print("""
    Distribution Coloring Tool
    ==========================
    Usage: distribution [EXPRESSION] [POSITION]
           distribution help

    Description:
      Colors nodes by the amino acid each has at one alignment position, and prints
      a table of the residues there: their counts and percentages of the colored
      nodes, the largest first.

      * QUICK USE: If no expression is provided, the command targets the nodes
        currently selected in the viewer. If no nodes are selected, it targets ALL
        VISIBLE nodes in the network.

      * Hidden nodes are skipped and keep their colors. So do nodes outside the
        target and nodes that are not in the alignment. One undo restores the
        colors the command changed.

    Position:
      [POSITION] - one displayed alignment position in brackets, numbered as in
      query: [120], [(-1)], [0], [15.1], or 'E' / 'END' for the last displayed
      position. The expression and the bracket may come in either order. A list,
      a range or a frequency comparison is refused, and nothing changes.

    Colors (fixed for each residue, upper or lower case):
      K blue         R light blue     (positive)
      D red          E pink-red       (negative)
      N magenta      Q light magenta  (amide)
      S orange       T light orange   (hydroxyl)
      I green        V light green    (branched)
      L olive        M light olive    (aliphatic)
      F brown        Y light brown    (aromatic)
      W purple       H light purple   (ring)
      A cyan         G light cyan     (small)
      P bronze       C gold           (special)
      Gaps and every other symbol (X, B, Z, U, O, *) are grey, as noise is in
      cluster.

    Sequence Selection Expression Targets (Do NOT use spaces inside expressions!):
      The same as in query and color: AA positions such as P106, "header text",
      @file@, #cluster_1#, $sele$ and {Length>500} metadata, joined with
      & (AND), | (OR), ! (NOT) and ^ (XOR).

    Examples:
      distribution [120]                  (Colors the selection, or all visible nodes, by position 120)
      distribution #cluster_1# [120]      (Colors only the nodes of cluster 1)
      distribution [E] #cluster_1#        (The same expression and position in the other order)
    """)


def _position_label(inner, alignment):
    """The one alignment label INNER names, or raise ValueError holding the message to show."""
    try:
        parts = split_position_list(inner)
    except ValueError as exc:
        raise ValueError(Message("Error: {error}", error=exc))
    if len(parts) != 1 or '<' in inner or '>' in inner or _QUERY_POSITION_RANGE_RE.fullmatch(parts[0]):
        raise ValueError(Message(
            "Error: distribution takes only one alignment position, such as [120] or [END]; "
            "a list, a range or a frequency comparison is not allowed."
        ))
    valid_labels = []
    for label in alignment.label_to_col:
        try:
            major, _, minor = str(label).partition('.')
            valid_labels.append(((int(major), int(minor) if minor else 0), label))
        except ValueError:
            pass
    valid_labels.sort(key=lambda item: item[0])
    label = parse_query_positions(parts[0], valid_labels)[0]
    if label not in alignment.label_to_col:
        raise ValueError(Message("Error: Position {position} is not in the active alignment mapping.", position=label))
    return label


def _residue_letters(alignment, codes):
    """Map each alignment code in CODES (0 is a gap) to its upper-case letter, or None for a gap."""
    return {int(code): (alignment.aln.int_to_aa.get(int(code), 'X').upper() if code else None) for code in codes}


def _fail(viewer, message):
    Command_Engine.print_help(viewer, message)
    Command_Engine.command_failed(viewer, message)


def run(viewer, args):
    if not args:
        msg = JoinedMessage([
            Message("Error: Distribution command requires a bracketed POSITION parameter."),
            "Usage: distribution [EXPRESSION] [POSITION]",
        ], separator="\n")
        _fail(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
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
            "Distribution analysis is unavailable."
        )
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
        print(msg)
        return

    # --- 1. The bracketed position, wherever it stands among the arguments ---
    args = rejoin_split_brackets(args)
    bracket_indices = [i for i, a in enumerate(args) if a.strip().startswith('[') and a.strip().endswith(']')]
    if not bracket_indices:
        _fail(viewer, Message("Error: No bracketed argument provided. Use [...] syntax (e.g., [120] or [END])."))
        return
    inner = args.pop(bracket_indices[0]).strip()[1:-1].strip()
    try:
        label = _position_label(inner, alignment)
    except ValueError as exc:
        _fail(viewer, exc.args[0])
        return

    # --- 2. The target nodes: the expression, else the selection, else every visible node ---
    expr = " ".join(args) if args else "$sele$"
    if expr == "$sele$" and not Command_Engine.command_selection(viewer):
        expr = '"*"'
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("No selection found. Defaulting to all visible nodes."))
        print("No nodes selected. Defaulting to all visible nodes in the network.")

    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
    try:
        mask = Command_Engine.parse_advanced_expression(
            expr,
            viewer_to_aln,
            valid_indices,
            viewer.full_headers,
            getattr(viewer, 'cluster_labels', None),
            getattr(viewer, 'group_labels', None),
            alignment,
            metadata=getattr(viewer, 'metadata', None),
            selection_mask=Command_Engine.get_selected_mask(viewer),
            viewer=viewer,
        )
    except Exception as e:
        Command_Engine.report_selection_error(viewer, expr, e, Message("Distribution"))
        return

    # Hidden nodes are outside the command's target domain, as in query and color.
    visible = getattr(viewer, 'visible_mask', None)
    n_hidden_matched = 0
    if visible is not None:
        visible = np.asarray(visible, dtype=bool)
        n_hidden_matched = int(np.sum(mask & ~visible))
        mask = mask & visible

    matched = np.where(mask)[0]
    in_alignment = viewer_to_aln[matched] != -1
    nodes = matched[in_alignment]
    n_unaligned = len(matched) - len(nodes)

    if len(nodes) == 0:
        # A valid expression may match no node, which is not a failure; matched nodes
        # that are all outside the alignment leave nothing to color, which is.
        nothing_matched = len(matched) == 0
        if nothing_matched and n_hidden_matched:
            msg = Message(
                "The expression '{expression}' matched only hidden nodes, which distribution skips. Nothing was colored.",
                expression=expr,
            )
        elif nothing_matched:
            msg = Message("No sequences matched the expression '{expression}'. Nothing was colored.", expression=expr)
        else:
            msg = Message(
                "The expression '{expression}' matched %n node(s), but none is in the alignment. "
                "Nothing was colored.",
                n=len(matched), expression=expr,
            )
        Command_Engine.show_status(viewer, msg)
        print(msg)
        if nothing_matched:
            Command_Engine.command_succeeded(viewer, msg)
        else:
            Command_Engine.command_failed(viewer, msg)
        return

    # --- 3. Each node's residue at the position, and its color ---
    matrix = alignment.aln.matrix
    column = matrix[:, alignment.label_to_col[label]]
    column = np.asarray(column.toarray() if hasattr(column, 'toarray') else column).ravel()
    codes = column[viewer_to_aln[nodes]]
    letters = _residue_letters(alignment, np.unique(codes))
    colors = np.empty((len(nodes), 4))
    counts = {}  # letter (None for a gap) -> nodes
    for code, letter in letters.items():
        selected = codes == code
        colors[selected] = _RESIDUE_RGBA.get(letter, _NOISE_RGBA)
        counts[letter] = counts.get(letter, 0) + int(selected.sum())

    viewer._save_state()
    viewer.current_colors[nodes] = colors
    colored = np.zeros(viewer.n_nodes, dtype=bool)
    colored[nodes] = True
    viewer.promote_nodes(colored)
    viewer.update_nodes()

    # --- 4. The table: named residues by count, then the gap and the other symbols ---
    residues = sorted(((n, letter) for letter, n in counts.items() if letter in _RESIDUE_RGBA),
                      key=lambda item: (-item[0], item[1]))
    n_gap = counts.get(None, 0)
    n_other = sum(n for letter, n in counts.items() if letter is not None and letter not in _RESIDUE_RGBA)
    # What the colors mean: the residues in the table's order, then Gap and Other when present.
    categories = [(letter, _RESIDUE_RGBA[letter]) for _, letter in residues]
    slot_of = {letter: slot for slot, (_, letter) in enumerate(residues)}
    if n_gap:
        slot_of[None] = len(categories)
        categories.append((Message("Gap"), _NOISE_RGBA))
    if n_other:
        other_slot = len(categories)
        categories.append((Message("Other"), _NOISE_RGBA))
    slots = np.empty(len(nodes), dtype=np.int32)
    for code, letter in letters.items():
        slots[codes == code] = slot_of[letter] if letter in slot_of else other_slot
    declare_categories(viewer, Message("Residue at {position}", position=label), categories, slots, nodes=nodes)

    rows = [(letter, f"{letter:<5}", n) for n, letter in residues]
    if n_gap:
        rows.append((-1, f"{'gap':<5}", n_gap))
    if n_other:
        rows.append((-1, f"{'other':<5}", n_other))

    total = len(nodes)
    print("-" * 50)
    print(f"DISTRIBUTION SUBSET: '{expr}' ({total} sequences mapped)  Position {label}")
    print("-" * 50)
    print(f"| {'Residue':<7} | {'Node Count':>10} | {'Percent':>10} |")
    for key, name, n in rows:
        shown = get_colored_cluster_name(key, name, {key: _RESIDUE_RGBA.get(key)})
        print(f"| {shown} | {n:>10} | {n / total * 100:>9.2f}% |")
    print("-" * 50)

    msg = Message("Colored %n node(s) by the residue at position {position}.", n=total, position=label)
    if n_unaligned:
        msg = JoinedMessage([
            msg,
            Message("%n matched node(s) not in the alignment kept their colors.", n=n_unaligned),
        ], separator=" ")
    Command_Engine.show_status(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)
