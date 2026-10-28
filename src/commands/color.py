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

import math
import re

import numpy as np
import matplotlib.colors as mcolors
import EMAPSSN_Config as cfg
import Command_Engine
from utilities.Localization import JoinedMessage, Message

def print_help():
    print("""
    Advanced Coloring & Highlighting Tool
    =====================================
    Usage: color [EXPR] <ATTRIBUTES> [EXPR <ATTRIBUTES> ...]
           color [EXPR] COLOR ALPHA SCALE SHAPE     (loose shorthand, see below)
           color reset
           color help

    Description:
      Colors, makes translucent, scales, and changes the shapes of nodes. You can
      target nodes using complex boolean expressions.

      * QUICK USE: If no expression is provided, the command automatically targets
        the nodes currently selected in the viewer using your mouse.

      * color reset restores every node's configured color, with full alpha.

    Attribute sets:
      The attributes of one target go in a set: <ITEM,ITEM,...> or <ITEM ITEM ...>,
      written literally with the angle brackets. A set holds 1 to 4 items, in any
      order, at most one of each kind. Separate the items all with commas (spaces
      around the commas are ignored) or all with spaces, never with both.
        1. Color: Name (red, blue) or Hex (#ff0000). An 8-digit hex (#ff000080)
           also sets the alpha. A color alone keeps each node's alpha.
        2. Alpha: A number from 0 to 100 suffixed with % (50%, 12.5%, 0%). An alpha
           alone keeps each node's color. Use 100% to make nodes opaque again.
        3. Scale: Suffix with 'x' (e.g., 2x, 0.5x, 0x). Scales must be finite and
           non-negative, and small enough to give a finite node size.
           Old x2 scale syntax is removed; x2 now selects residue X at position 2.
        4. Shape: circle, square, triangle, star, diamond, cross, vbar, hbar, x
           (The shapes > and -> end a word, so in a set written with spaces
           name them arrow and tailed_arrow.)
      A set follows its expression as a word of its own (#cluster_1# <red,50%>)
      or glued to it (#cluster_1#<red 50%>). With no expression before it, a set
      targets the selected nodes. Give a target one set, and start the next target
      with an expression. Inside a set no expression is read, so C53 is a color.

    Loose shorthand:
      Without a set, the items of a target may follow its expression directly
      (#cluster_1# red 50% 2x star). Valid selection expressions take precedence
      over colors and other modifiers there. A target uses either a set or loose
      items, never both.

    Expression Targets (Do NOT use spaces inside expressions!):
      1. AA Position:  [AA][Pos] (e.g., P106, _100 for gap), or ([AA...])[Pos]
                       for alternatives (e.g., (RHK)71); negative positions require
                       parentheses (e.g., K(-1), (RHK)(-1))
      2. Header Text:  "[Text]"  (e.g., "3HMU", "*4A6T*")
      3. File Search:  @[File]@  (e.g., @my_list.txt@)
      4. NCBI/PDB:     @[NCBI][File]@ or @[PDB][File]@ (Regex extraction)
      5. Labels:       #[Name]#  (e.g., #cluster_1#, #noise#, #my_group#)
      6. UI Selection: $sele$     (Explicitly targets selected nodes)
      7. Metadata:     {Key Op Val} (e.g., {Length>500}, {Organism=*coli*});
                       ranges use = (e.g., {Length=300-500}), and negative
                       range bounds require parentheses (e.g., {GRAVY=(-1)-0})

    Logic Operators:
      & (AND), | (OR), ! (NOT), ^ (XOR)

    Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. If any expression or set is invalid, no assignments are
      applied. A valid expression may match zero nodes.

    Examples:
      color <red,2x,triangle>                    (Modifies currently selected nodes)
      color <80%,square>                         (Makes selected nodes 80% opaque squares)
      color P106 <red>                           (Colors nodes with Proline at pos 106 red)
      color (RHK)71 <blue 50%>                   (Colors nodes with R, H, or K at pos 71 blue, half opaque)
      color "ATA"&#cluster_2# <blue,1.5x> #noise# <20%>
                                                 (Colors "ATA" matches inside Cluster 2, then fades the noise)
      color {Organism=*coli*}<green>             (Colors nodes where Organism matches *coli* green)
      color #cluster_1# red 2x #noise# 0x        (Loose shorthand, chaining several targets)
    """)


# The four kinds of item that set a node's look.
_COLOR, _ALPHA, _SCALE, _SHAPE = 'color', 'alpha', 'scale', 'shape'

_SHAPE_SYMBOLS = ['disc', 'arrow', 'ring', 'clobber', 'square', 'x', 'diamond', 'vbar', 'hbar',
                  'cross', 'tailed_arrow', 'triangle_up', 'triangle_down', 'star', 'cross_lines',
                  'o', '+', '++', 's', '-', '|', '->', '>', '^', 'v', '*']

_SHAPE_ALIASES = {
    'circle': 'disc',
    'triangle': 'triangle_up'
}

# A color Matplotlib reads with its own alpha: #RGBA and #RRGGBBAA, and "none".
_COLOR_WITH_ALPHA = re.compile(r'#(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8})\Z')

# Where an expression's own delimiters end: a header text, a file, a label, metadata.
_DELIMITER_ENDS = {'"': '"', '@': '@', '#': '#', '{': '}'}


class _Refusal(Exception):
    """A color command that cannot be read: nothing has been changed, and this says why.

    The message is the one to report. An expression the engine found malformed
    is reported with its own error instead.
    """

    def __init__(self, message=None, *, report_message=True, expression=None, error=None):
        super().__init__(message)
        self.message = message
        self.report_message = report_message
        self.expression = expression
        self.error = error

    def report(self, viewer):
        if self.error is not None:
            Command_Engine.report_selection_error(viewer, self.expression, self.error, Message("Color"))
            return
        Command_Engine.print_help(viewer, self.message, report_message=self.report_message)
        Command_Engine.command_failed(viewer, self.message)


class _Attributes:
    """The look one target's items give: a color, an alpha, a scale and a shape, each optional."""

    def __init__(self):
        self.color = None         # as written, for the report
        self.rgb = None           # the color's red, green and blue
        self.alpha = None         # 0 to 1
        self.alpha_label = None   # the N% as written; None when a color gave the alpha
        self.alpha_source = None  # the item that gave the alpha: an N% or a color that carries one
        self.scale = None
        self.shape = None
        self.loose = None         # the first item written outside a set

    def __bool__(self):
        return (self.color is not None or self.alpha is not None
                or self.scale is not None or self.shape is not None)

    def _give_alpha(self, source, alpha, label):
        if self.alpha_source is not None:
            raise _Refusal(Message(
                "Error: '{first}' and '{second}' both set the alpha of one target. "
                "Give the alpha once, either in an 8-digit hex color or as N%.",
                first=self.alpha_source, second=source,
            ))
        self.alpha, self.alpha_label, self.alpha_source = alpha, label, source

    def add(self, kind, text, value):
        """Take the item `text` of this kind, whose value _read_item gave."""
        if kind == _COLOR:
            self.color, self.rgb = text, tuple(value[:3])
            if _COLOR_WITH_ALPHA.match(text) or text.lower() == 'none':
                self._give_alpha(text, value[3], None)
        elif kind == _ALPHA:
            self._give_alpha(text, value / 100.0, f"{value:g}%")
        elif kind == _SCALE:
            self.scale = value
        else:
            self.shape = value


class _Target:
    """The nodes an expression selects and the attributes they are to get."""

    def __init__(self):
        self.expression = None
        self.attributes = _Attributes()
        self.set_text = None      # the attribute set that gave the attributes, if one did


def _read_scale(text):
    """The scale `text` is, or None when it is not written as one.

    Raises _Refusal for a scale that is written right but cannot be used.
    """
    # Scale uses a trailing lowercase x (e.g., 2.5x).
    if not text.endswith('x'):
        return None
    try:
        scale = float(text[:-1])
    except ValueError:
        return None
    # float() also accepts '-2', 'nan', 'inf' and '1e400' (inf).
    # Nothing has been applied yet, so rejecting here is atomic.
    if not math.isfinite(scale) or scale < 0:
        raise _Refusal(Message(
            "Error: Invalid scale '{scale}'. A scale is a finite, "
            "non-negative number followed by x, e.g. 2x, 0.5x or 0x.",
            scale=text,
        ))
    if scale == 0:
        scale = 0.0  # '-0x' is the scale 0, not -0.0
    with np.errstate(over='ignore'):
        size_is_finite = bool(np.isfinite(np.float32(cfg.NODE_SIZE * scale)))
    if not size_is_finite:
        raise _Refusal(Message(
            "Error: Scale '{scale}' is too large to make a valid node size.",
            scale=text,
        ), report_message=False)
    return scale


def _read_alpha(text):
    """The alpha `text` is, in percent, or None when it is not written as one.

    Raises _Refusal for an alpha that is written right but is not from 0 to 100.
    """
    if not text.endswith('%') or '_' in text:
        return None
    try:
        percent = float(text[:-1])
    except ValueError:
        return None
    if not math.isfinite(percent) or not 0 <= percent <= 100:
        raise _Refusal(Message(
            "Error: Invalid alpha '{alpha}'. An alpha is a number from 0 to 100 "
            "followed by %, e.g. 50%, 12.5% or 0%.",
            alpha=text,
        ))
    if percent == 0:
        percent = 0.0  # '-0%' is the alpha 0, not -0.0
    return percent


def _read_item(text):
    """What attribute `text` is, as (kind, value), or None when it is none of the four.

    The value is the color's RGBA, the alpha's percent, the scale, or the
    shape's VisPy name. Raises _Refusal for a scale or alpha that cannot be used.
    """
    scale = _read_scale(text)
    if scale is not None:
        return _SCALE, scale
    percent = _read_alpha(text)
    if percent is not None:
        return _ALPHA, percent
    lowered = text.lower()
    shape = _SHAPE_ALIASES.get(lowered, lowered)
    if shape in _SHAPE_SYMBOLS:
        return _SHAPE, shape
    try:
        return _COLOR, mcolors.to_rgba(text)
    except (TypeError, ValueError):
        return None


def _split_at_set(arg):
    """arg as (the expression glued before a set, the text from the set's <), either of them None.

    A set opens at the first < outside the expression's own delimiters, so a
    < inside {metadata}, "header text", @file@ or #label# never opens one. An
    expression that is not closed has no set: the expression parser reports it.
    """
    position = 0
    while position < len(arg):
        char = arg[position]
        if char == '<':
            return (arg[:position] or None), arg[position:]
        closer = _DELIMITER_ENDS.get(char)
        if closer is not None:
            end = arg.find(closer, position + 1)
            if end == -1:
                break
            position = end
        position += 1
    return arg, None


def _join_sets(args):
    """The arguments with each attribute set made one text, as typed with single spaces.

    The command line reaches a command split at spaces, so a set written
    <red 50% star> arrives as several words. A set that opens, even glued to an
    expression (#cluster_1#<red), runs through the first word that ends in >.
    A set that never closes takes every word to the end of the command; reading
    it reports that.
    """
    tokens = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        expression, text = _split_at_set(arg)
        if text is None:
            tokens.append(arg)
            continue
        if expression is not None:
            tokens.append(expression)
        while not text.endswith('>') and index < len(args):
            text += ' ' + args[index]
            index += 1
        tokens.append(text)
    return tokens


def _read_set(text):
    """The _Attributes that the attribute set `text` gives: <ITEM,ITEM,...> or <ITEM ITEM ...>.

    Spaces around a comma and next to the brackets are ignored. Raises
    _Refusal, naming the set, for one that cannot be read.
    """
    malformed = _Refusal(Message(
        "Error: Malformed attribute set '{set}'. A set is <ITEM,ITEM,...> or <ITEM ITEM ...>: "
        "opened by <, closed by > and holding no empty item and no other <.",
        set=text,
    ))
    # The text starts at the opening <, and ends at the closing > unless it never closed.
    if not text.endswith('>') or '<' in text[1:]:
        raise malformed
    inner = text[1:-1].strip()
    if not inner:
        raise _Refusal(Message(
            "Error: The attribute set '{set}' is empty. A set holds 1 to 4 attributes, "
            "e.g. <red,50%,star> or <red 50% star>.",
            set=text,
        ))
    if ',' in inner:
        items = [item.strip() for item in inner.split(',')]
        if any(not item for item in items):
            raise malformed
        if any(len(item.split()) > 1 for item in items):
            raise _Refusal(Message(
                "Error: The attribute set '{set}' mixes commas and spaces between its items. "
                "Separate them all with commas (<red,50%,star>) or all with spaces (<red 50% star>).",
                set=text,
            ))
    else:
        items = inner.split()
    attributes = _Attributes()
    seen = {}
    for item in items:
        read = _read_item(item)
        if read is None:
            raise _Refusal(Message(
                "Error: Unknown attribute '{item}' in the set '{set}'. Expected a color, "
                "an alpha such as 50%, a scale such as 2x, or a shape.",
                item=item, set=text,
            ))
        kind, value = read
        if kind in seen:
            raise _Refusal(Message(
                "Error: The attribute set '{set}' has two items of one kind, '{first}' and '{second}'. "
                "A set holds at most one color, one alpha, one scale and one shape.",
                set=text, first=seen[kind], second=item,
            ))
        seen[kind] = item
        attributes.add(kind, item, value)
    return attributes


def _read_arguments(args):
    """The command's assignments, as [(expression, _Attributes)], and the expressions given no attributes.

    Raises _Refusal for an argument that cannot be read, before anything has
    been changed. A target without an expression takes the selected nodes.
    """
    assignments = []
    skipped_expressions = []
    target = _Target()

    def push_assignment():
        nonlocal target
        # A target with attributes but no expression takes the selected nodes.
        if target.attributes and not target.expression:
            target.expression = '$sele$'
        if target.expression and target.attributes:
            assignments.append((target.expression, target.attributes))
        elif target.expression:
            print(f"Warning: Skipping '{target.expression}' (No valid color, alpha, scale, or shape provided)")
            skipped_expressions.append(target.expression)
        target = _Target()

    def refuse_mixing(loose, set_text):
        raise _Refusal(Message(
            "Error: '{loose}' and the attribute set '{set}' are given for one target. "
            "Give a target its attributes either all loose (red 50% star) "
            "or all in one set (<red,50%,star>).",
            loose=loose, set=set_text,
        ))

    for arg in _join_sets(args):
        if arg.startswith('<'):
            attributes = _read_set(arg)
            if target.set_text is not None:
                raise _Refusal(Message(
                    "Error: The attribute sets '{first}' and '{second}' are given for one target. "
                    "Give a target one set, and start the next target with an expression.",
                    first=target.set_text, second=arg,
                ))
            if target.attributes.loose is not None:
                refuse_mixing(target.attributes.loose, arg)
            target.attributes, target.set_text = attributes, arg
            continue

        # Classify a complete Boolean expression.
        classification = Command_Engine.classify_selection_expression(arg)
        if classification.kind == Command_Engine.SelectionClassificationKind.VALID_EXPRESSION:
            # A set belongs to the expression before it, so an expression after one starts a target.
            if target.expression or target.set_text is not None:
                push_assignment()
            target.expression = arg
            continue

        item = _read_item(arg)
        if item is None:
            if classification.kind == Command_Engine.SelectionClassificationKind.MALFORMED_EXPRESSION:
                raise _Refusal(expression=arg, error=classification.error)
            if ',' in arg or arg.endswith('>'):
                raise _Refusal(Message(
                    "Error: Malformed attribute set '{set}'. A set is <ITEM,ITEM,...> or <ITEM ITEM ...>: "
                    "opened by <, closed by > and holding no empty item and no other <.",
                    set=arg,
                ))
            raise _Refusal(Message(
                "Error: Unrecognized color argument '{argument}'. Expected a Boolean "
                "expression, color, alpha with trailing %, scale with trailing x, shape, "
                "or attribute set <...>.",
                argument=arg,
            ))
        if target.set_text is not None:
            refuse_mixing(arg, target.set_text)
        kind, value = item
        # A second color or shape starts a target of its own, for the selected nodes.
        if (kind == _COLOR and target.attributes.color is not None) or (
                kind == _SHAPE and target.attributes.shape is not None):
            push_assignment()
        target.attributes.add(kind, arg, value)
        if target.attributes.loose is None:
            target.attributes.loose = arg

    if target.expression or target.attributes:
        push_assignment()
    return assignments, skipped_expressions


def run(viewer, args):
    if args and args[0].lower() == 'reset':
        msg = Command_Engine.execute_reset(viewer, ["colors"])
        Command_Engine.command_succeeded(viewer, msg)
        return

    if not args:
        # The console line shows the first line; the usage is for the terminal.
        msg = JoinedMessage([
            Message("Error: Color command requires at least one property (color, alpha, scale, or shape) or expression."),
            "Usage: color [EXPR] <ATTRIBUTES> [EXPR <ATTRIBUTES> ...]",
        ], separator="\n")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return

    try:
        assignments, skipped_expressions = _read_arguments(args)
    except _Refusal as refusal:
        refusal.report(viewer)
        return

    if not assignments:
        msg = Message("Error: No valid assignments found.")
        Command_Engine.show_status(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)

    total_modified = 0
    stats = []
    state_saved = False  # <--- NEW FLAG
    modified_nodes = np.zeros(viewer.n_nodes, dtype=bool)

    evaluated_assignments = []
    for expr, attributes in assignments:
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
            Command_Engine.report_selection_error(viewer, expr, e, Message("Color"))
            return

        # Hidden nodes are outside the command's target domain, even when the
        # logical expression itself matches them.
        mask = mask & viewer.visible_mask

        evaluated_assignments.append((expr, attributes, mask, int(np.sum(mask))))

    # Evaluation above is intentionally side-effect free so a bad later
    # expression cannot leave earlier assignments partially applied.
    if not hasattr(viewer, 'current_shapes'):
        viewer.current_shapes = np.full(viewer.n_nodes, 'disc', dtype=object)

    for expr, attributes, mask, count in evaluated_assignments:
        if count <= 0:
            continue

        # Save state only once, and only if a match is actually found.
        if not state_saved:
            viewer._save_state()
            state_saved = True

        # The color and the alpha are independent: a color keeps each node's
        # alpha, and an alpha keeps each node's color.
        if attributes.rgb is not None:
            viewer.current_colors[mask, :3] = attributes.rgb
        if attributes.alpha is not None:
            viewer.current_colors[mask, 3] = attributes.alpha

        if attributes.scale is not None:
            viewer.current_sizes[mask] = cfg.NODE_SIZE * attributes.scale
        if attributes.shape: viewer.current_shapes[mask] = attributes.shape

        modified_nodes |= mask
        total_modified += count

        labels = []
        if attributes.color: labels.append(attributes.color)
        if attributes.alpha_label: labels.append(attributes.alpha_label)
        if attributes.scale is not None:
            labels.append(f"{attributes.scale}x")
        if attributes.shape: labels.append(attributes.shape)
        stats.append(Message("%n node(s) ({properties})", n=count, properties=', '.join(labels)))

    if total_modified > 0:
        viewer.promote_nodes(modified_nodes)
        viewer.update_nodes()
        msg = Message("Applied: {stats}", stats=JoinedMessage(stats, separator="; "))
    else:
        msg = Message("No nodes matched criteria.")
    if skipped_expressions:
        # An expression given no color, alpha, scale or shape is left out; the rest was applied.
        msg = JoinedMessage([
            msg,
            Message(
                "Skipped %n expression(s) with no color, alpha, scale or shape: {skipped}.",
                n=len(skipped_expressions), skipped=', '.join(skipped_expressions),
            ),
        ])
    Command_Engine.show_status(viewer, msg)
    if total_modified > 0:
        print(f"\nSuccess! {msg}")
    else:
        print("\nNo nodes matched your criteria.")
    Command_Engine.command_succeeded(viewer, msg)


# What "reset colors" restores: commands/reset.py asks each command module for its targets.
RESET_TARGETS = ('color',)


def reset_changes_state(viewer, target):
    """False when every node already has the default colour."""
    colors = getattr(viewer, 'current_colors', None)
    if colors is None:
        return False
    initial = np.asarray(mcolors.to_rgba(cfg.INITIAL_NODE_COLOR), dtype=colors.dtype)
    return not np.all(colors == initial)


def reset(viewer, target):
    """Give every node the default colour."""
    from commands.reset import ResetOutcome
    if hasattr(viewer, 'current_colors'):
        viewer.current_colors[:] = mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)
    return ResetOutcome("colors", nodes=True)
