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
import sys
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import Command_Engine
import EMAPSSN_Config as cfg
from desktop.Node_Colors import NOISE_COLOR
from utilities.Localization import JoinedMessage, Message
from utilities.Terminal_Colors import colours_enabled
from desktop.Command_Metadata import argument, entry, EXPRESSION

# --- What this command declares about itself ---
# Changes only the view.
AGENT_POLICY = 'run'

CATALOG_ENTRY = entry('Color visible nodes by a numerical metadata property.',
    argument('property', 'Bare {PROPERTY_NAME} selector for a numerical metadata column; arguments may occur in any order. Omit it with exactly one node selected to color the nodes by their similarity score to that node instead.'),
    argument('expression', EXPRESSION + ' Defaults to the selected nodes, otherwise all visible nodes.'),
    argument('color_scheme', 'Optional installed Matplotlib colormap name, default coolwarm; names are not case-sensitive, and an unknown name is an error. A token that is also a valid selection expression is read as the expression.'))

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 170
AGENT_PROMPT = """
`spectrum [EXPRESSION] {PROPERTY_NAME} [COLOR_SCHEME]`
- Colors visible nodes by values from exactly one loaded numerical metadata property enclosed in braces. A bare target such as `{Length}` selects the spectrum property, while a complete predicate such as `{Length>500}` remains a Boolean selection expression. The expression, property, and optional standalone color scheme may appear in any order.
- If EXPRESSION is omitted, the current selection is targeted; when there is no selection, all visible nodes are. Hidden matches remain unchanged. Text properties are invalid for spectrum coloring. Nodes whose value is missing, non-numeric, or infinite are excluded from the gradient minimum/maximum and get the noise color the cluster command gives unclustered nodes (translucent light gray).
- With exactly one node selected and no property, the visible nodes (or those the expression matches) are colored by their similarity score to that node, read from the network file, and the selected node gets a color outside the scheme; nodes with no score to it get that noise color.
- A bare `spectrum` with exactly one node selected runs the similarity mode with the default scheme over all visible nodes.
- The default Matplotlib color scheme is `coolwarm`. Scheme names are not case-sensitive. An unrecognized scheme is an error that changes no colors, so use a Matplotlib colormap name such as `viridis`. Do not emit the removed `prop:`, `property:`, `scheme:`, or `color:` forms.
"""


if sys.platform == "win32" and not globals().get("_WINDOWS_ANSI_ENABLED", False):
    os.system("")
    _WINDOWS_ANSI_ENABLED = True


ANSI_RESET = "\033[0m"


def _terminal_endpoint_text(label, value, rgba):
    text = f"{label}: {value}"
    if not colours_enabled():
        return text

    red, green, blue = (
        int(round(float(channel) * 255.0)) for channel in rgba[:3]
    )
    return f"\033[38;2;{red};{green};{blue}m{text}{ANSI_RESET}"


def _terminal_range_text(vmin, vmax, cmap):
    if vmax == vmin:
        min_position = max_position = 0.5
    else:
        min_position, max_position = 0.0, 1.0

    min_text = _terminal_endpoint_text("min", vmin, cmap(min_position))
    max_text = _terminal_endpoint_text("max", vmax, cmap(max_position))
    return f"({min_text}, {max_text})"


def _applied_message(count, property_name, range_text, scheme_name, invalid_count, scheme_found):
    """The report of a spectrum coloring; the console line and the terminal differ only in range_text."""
    parts = [Message(
        "Spectrum coloring applied to %n node(s) using property '{property}' {range} with scheme '{scheme}'.",
        n=count,
        property=property_name,
        range=range_text,
        scheme=scheme_name,
    )]
    if invalid_count:
        parts.append(Message("%n node(s) with invalid values colored gray.", n=invalid_count))
    if not scheme_found:
        parts.insert(0, Message("[Warning: '{scheme}' not found, using coolwarm]", scheme=scheme_name))
    return JoinedMessage(parts)

def print_help():
    print("""
    Spectrum Coloring Tool
    ======================
    Usage: spectrum [EXPRESSION] {PROPERTY_NAME} [COLOR_SCHEME]
           spectrum help

    Description:
      Colors nodes along a color gradient (spectrum) based on the values of a numerical property.
      You can optionally target a subset of nodes using a logical expression; without
      one, the selected nodes are colored, or all visible nodes when nothing is selected.
      The sequence of the arguments does not matter.

      * SIMILARITY MODE: with exactly one node selected and no {PROPERTY_NAME}, the nodes
        are colored by their similarity score to that node (the score the network plot
        uses, read from the network file, higher = more similar). The selected node is
        colored a vivid color that is not in the scheme; nodes with no score to it get
        the noise color cluster gives unclustered nodes. EXPRESSION picks the nodes colored (default: all visible nodes, not
        the selection).

      * IMPORTANT: This command only applies to visible nodes.

    Arguments:
      {PROPERTY_NAME}         - The numerical metadata property (e.g., {Length}). Required unless
                                exactly one node is selected (similarity mode).
      COLOR_SCHEME            - (Optional) Matplotlib colormap name. Defaults to 'coolwarm'.
                                Names are not case-sensitive (Viridis is viridis). An unknown
                                name is an error and changes no colors.
                                Supported schemes:
                                * Perceptually Uniform: viridis, plasma, inferno, magma, cividis
                                * Sequential: Blues, BuGn, BuPu, GnBu, Greens, Greys, Oranges, 
                                  OrRd, PuBu, PuBuGn, PuRd, Purples, RdPu, Reds, YlGn, YlGnBu, 
                                  YlOrBr, YlOrRd
                                * Diverging: coolwarm, bwr, seismic, Spectral, BrBG, PiYG, PRGn,
                                  PuOr, RdBu, RdGy, RdYlBu, RdYlGn
                                * Cyclic: twilight, twilight_shifted, hsv
                                * Qualitative: tab10, tab20, tab20b, tab20c, Pastel1, Pastel2, 
                                  Paired, Accent, Dark2, Set1, Set2, Set3
                                * Miscellaneous: jet, rainbow, turbo, ocean, terrain, cubehelix, 
                                  gnuplot, gnuplot2, flag, prism, gist_earth, nipy_spectral
      [EXPRESSION]            - (Optional) Logical expression to select which nodes are colored.
                                If omitted, the selected nodes are colored; when nothing is
                                selected, all visible nodes in the network are.

    Selection Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. Invalid references abort before colors are changed.
      A valid expression may match zero nodes.
      Valid selection expressions take precedence over color scheme names.

    Examples:
      spectrum {Length}
      spectrum {Length} plasma
      spectrum #cluster_1# {Length} coolwarm
      spectrum {Organism=*coli*} {Length}
      spectrum                    (one node selected: similarity to it, coolwarm)
      spectrum viridis            (one node selected: similarity to it)
    """)

def get_colormap(scheme_name):
    # Try using modern matplotlib.colormaps
    try:
        if hasattr(mpl, 'colormaps'):
            return mpl.colormaps[scheme_name], True
    except KeyError:
        pass
    
    # Try using cm.get_cmap
    try:
        return cm.get_cmap(scheme_name), True
    except Exception:
        pass
        
    # Return default 'coolwarm' if the requested scheme is not found or fails
    if scheme_name.lower() != 'coolwarm':
        print(f"Warning: Color scheme '{scheme_name}' not found. Defaulting to 'coolwarm'.")
    try:
        if hasattr(mpl, 'colormaps'):
            return mpl.colormaps['coolwarm'], False
    except Exception:
        pass
    return cm.get_cmap('coolwarm'), False


def registered_colormap_name(scheme_name):
    """Matplotlib's spelling of a colormap name, or None when it is unknown.

    Names are not case-sensitive: an exact match wins, otherwise the name must
    match exactly one registered name ignoring case.
    """
    names = list(mpl.colormaps)
    if scheme_name in names:
        return scheme_name
    matches = [name for name in names if name.lower() == scheme_name.lower()]
    return matches[0] if len(matches) == 1 else None

def _expression_mask(viewer, expr):
    """The nodes a Boolean expression matches, or None after reporting why it could not be read."""
    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
    try:
        return Command_Engine.parse_advanced_expression(
            expr, viewer_to_aln, valid_indices, viewer.full_headers,
            getattr(viewer, 'cluster_labels', None), getattr(viewer, 'group_labels', None),
            getattr(viewer, 'alignment', None), metadata=getattr(viewer, 'metadata', None),
            selection_mask=Command_Engine.get_selected_mask(viewer),
            viewer=viewer,
        )
    except Exception as e:
        Command_Engine.report_selection_error(viewer, expr, e, Message("Spectrum"))
        return None


def plan(viewer, args):
    """Read the arguments: a SpectrumPlan, or None when help ran or they were refused."""
    # No arguments is the similarity mode with its defaults when one node is selected, an error otherwise (start).

    # Parse command-specific roles before Boolean-expression classification.
    expr = None
    prop_name = None
    scheme_name = 'coolwarm'
    scheme_supplied = False
    scheme_unknown = False

    for arg in args:
        arg_lower = arg.lower()
        if arg_lower in ['help', '-h', '--help']:
            print_help()
            if hasattr(viewer, 'console_text'):
                Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
            Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
            return
        if arg_lower.startswith(('prop:', 'property:', 'scheme:', 'color:')):
            # The syntax is a value, so it stays English and its braces are not a placeholder.
            msg = Message(
                "Error: Legacy spectrum prefixes are no longer supported. Use '{syntax}'.",
                syntax="spectrum [EXPRESSION] {PROPERTY_NAME} [COLOR_SCHEME]",
            )
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return

        property_match = re.fullmatch(r'\{([a-zA-Z0-9_\-.]+)\}', arg)
        if property_match:
            if prop_name is not None:
                msg = Message("Error: Spectrum accepts exactly one {syntax}.", syntax="{PROPERTY_NAME}")
                Command_Engine.print_help(viewer, msg)
                Command_Engine.command_failed(viewer, msg)
                return
            prop_name = property_match.group(1)
            continue

        classification = Command_Engine.classify_selection_expression(arg)
        if classification.kind == Command_Engine.SelectionClassificationKind.VALID_EXPRESSION:
            if expr is not None:
                msg = Message("Error: Spectrum accepts at most one Boolean expression.")
                Command_Engine.print_help(viewer, msg)
                Command_Engine.command_failed(viewer, msg)
                return
            expr = arg
            continue
        colormap_name = registered_colormap_name(arg)
        if colormap_name is not None:
            if scheme_supplied:
                msg = Message("Error: Spectrum accepts at most one color scheme.")
                Command_Engine.print_help(viewer, msg)
                Command_Engine.command_failed(viewer, msg)
                return
            scheme_name = colormap_name
            scheme_supplied = True
            continue

        if classification.kind == Command_Engine.SelectionClassificationKind.MALFORMED_EXPRESSION:
            Command_Engine.report_selection_error(
                viewer, arg, classification.error, Message("Spectrum")
            )
            return
        if scheme_supplied:
            msg = Message(
                "Error: Unrecognized extra spectrum argument '{argument}'. Only one "
                "color scheme may be supplied.",
                argument=arg,
            )
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return
        scheme_name = arg
        scheme_supplied = True
        scheme_unknown = True

    return SpectrumPlan(expr, prop_name, scheme_name, scheme_unknown)


def start(viewer, plan):
    """At its turn: color by a property now (and end), or gather what the similarity scan needs."""
    if plan.prop_name is None:
        return start_by_similarity(viewer, plan)
    _color_by_property(viewer, plan)
    return None


def _color_by_property(viewer, plan):
    expr, prop_name = plan.expr, plan.prop_name
    scheme_name, scheme_unknown = plan.scheme_name, plan.scheme_unknown

    # Refused before any change: a colormap name that is not registered.
    if scheme_unknown:
        msg = Message(
            "Error: Unknown color scheme '{scheme}'. Use a matplotlib colormap name such as "
            "viridis; see 'spectrum help' for the list.",
            scheme=scheme_name,
        )
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    if not getattr(viewer, 'metadata', None):
        msg = Message("Error: No metadata loaded in the viewer.")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    # Resolve property case-insensitively
    matched_key = None
    for k in viewer.metadata.keys():
        if k.lower() == prop_name.lower():
            matched_key = k
            break

    if not matched_key:
        msg = Message(
            "Error: Property '{property}' not found. Available properties: {available}",
            property=prop_name,
            available=", ".join(viewer.metadata.keys()),
        )
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    prop_data = viewer.metadata[matched_key]
    if prop_data["type"] != "number":
        msg = Message(
            "Error: Property '{property}' is not numerical (type is '{type}'). "
            "Spectrum coloring requires a numerical property.",
            property=matched_key,
            type=prop_data['type'],
        )
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    # Determine mask
    if expr:
        mask = _expression_mask(viewer, expr)
        if mask is None:
            return
    elif Command_Engine.command_selection(viewer):
        # No expression: the selection, as if `$sele$` had been typed.
        mask = Command_Engine.get_selected_mask(viewer)
    else:
        # Nothing selected: every visible node, with the note query gives.
        mask = np.ones(viewer.n_nodes, dtype=bool)
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("No selection found. Defaulting to all visible nodes."))
        print("No nodes selected. Defaulting to all visible nodes in the network.")

    # Restrict to visible nodes
    mask = mask & viewer.visible_mask

    if np.sum(mask) == 0:
        msg = Message("No nodes matched the selection criteria (only visible nodes are colored).")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    # Extract values and handle coercion to floats safely
    raw_vals = prop_data["values"]
    values = np.full(viewer.n_nodes, np.nan, dtype=np.float64)
    for i in range(viewer.n_nodes):
        try:
            if pd.notna(raw_vals[i]):
                values[i] = float(raw_vals[i])
        except Exception:
            pass

    # Extract target values for coloring
    target_vals = values[mask]
    # Finite values only: an infinite one would stretch the range to inf,
    # flattening every other node to one colour and itself to transparent.
    valid_mask = np.isfinite(target_vals)
    valid_vals = target_vals[valid_mask]

    if len(valid_vals) == 0:
        Command_Engine.print_help(
            viewer,
            Message("Warning: No valid numerical values found in '{property}' for the selected nodes.", property=matched_key),
        )
        Command_Engine.command_succeeded(viewer, f"No valid numerical values found in {matched_key!r} for the selected nodes.")
        return

    # Save viewer state once for undo support
    viewer._save_state()

    # Map values to colormap
    vmin = np.min(valid_vals)
    vmax = np.max(valid_vals)
    
    if vmax == vmin:
        normalized = np.full_like(valid_vals, 0.5)
    else:
        normalized = (valid_vals - vmin) / (vmax - vmin)

    cmap, cmap_ok = get_colormap(scheme_name)
    colors_rgba = cmap(normalized)

    # Color valid nodes
    full_valid_mask = np.zeros(viewer.n_nodes, dtype=bool)
    full_valid_mask[mask] = np.isfinite(values[mask])
    viewer.current_colors[full_valid_mask] = colors_rgba

    # Color NaN and infinite nodes within mask to the noise color
    nan_mask = mask & ~np.isfinite(values)
    if np.any(nan_mask):
        viewer.current_colors[nan_mask] = NOISE_COLOR

    # Update viewer
    viewer.promote_nodes(mask)
    viewer.update_nodes()

    # Automatically invoke "meta display" to show the property used for the spectrum.
    # Only a viewer that has the metadata HUD does: the VR viewer and the other
    # headless viewers keep no hud_displays, and have nothing to show it on.
    if hasattr(viewer, 'hud_displays'):
        try:
            import importlib
            meta_module = importlib.import_module("commands.meta")
            meta_module.run(viewer, ["display", matched_key])
        except Exception as e:
            print(f"Warning: Failed to automatically enable metadata display: {e}")
    
    count = int(np.sum(full_valid_mask))
    invalid_count = int(np.sum(nan_mask))
    msg = _applied_message(
        count, matched_key, Message("(min: {min}, max: {max})", min=vmin, max=vmax),
        scheme_name, invalid_count, cmap_ok,
    )
    terminal_msg = _applied_message(
        count, matched_key, _terminal_range_text(vmin, vmax, cmap), scheme_name, invalid_count, cmap_ok
    )
    
    Command_Engine.print_help(viewer, msg, terminal_msg=terminal_msg, report_message=False)
    Command_Engine.command_succeeded(viewer, msg)


# --- The similarity mode: one node selected, no property -------------------------------
#
# The nodes are colored by their score to the selected node, the score the network plot
# draws its edges with. The Viewer keeps only the edges it draws, so every pair that has
# the selected node at an end is read back from the network file, streamed in slices as
# the layout reads it (Viewer_State._pair_source), on the command queue's worker thread.

MISSING_COLOR = NOISE_COLOR
# What the selected node can be colored, in the order a tie is decided.
REFERENCE_CANDIDATES = (
    ("lime", "#00E676"),
    ("magenta", "#FF00FF"),
    ("gold", "#FFD600"),
    ("cyan", "#00E5FF"),
    ("black", "#000000"),
    ("white", "#FFFFFF"),
)


@dataclass(frozen=True)
class SpectrumPlan:
    """The arguments, read: no property means the similarity mode, which scans the network file."""
    expr: object
    prop_name: object
    scheme_name: str
    scheme_unknown: bool

    @property
    def heavy(self):
        return self.prop_name is None


@dataclass(frozen=True)
class SimilarityInputs:
    """What the scan needs, copied at the command's turn."""
    reference: int
    header: str
    targets: object
    headers: tuple
    network_path: str
    alignment_score: object
    norm_mode: object
    scheme_name: str


@dataclass(frozen=True)
class SimilarityResult:
    """The scan's outcome for apply(): an error to report, or each node's score (NaN where it has none)."""
    inputs: SimilarityInputs
    error: object = None
    scores: object = None


def _srgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    xyz = linear @ matrix.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def reference_color(cmap, background=None):
    """(name, hex, rgba) of the candidate farthest, in CIELAB, from every color of the map, from the noise color and from the canvas ``background`` when given."""
    from matplotlib.colors import to_rgb
    colors = [np.asarray(cmap(np.linspace(0.0, 1.0, 256)))[:, :3], np.asarray(MISSING_COLOR[:3])[None, :]]
    if background is not None:
        colors.append(np.asarray(background, dtype=float)[None, :3])
    palette = np.vstack(colors)
    palette_lab = _srgb_to_lab(palette)
    best = None
    for name, hex_code in REFERENCE_CANDIDATES:
        rgb = to_rgb(hex_code)
        distance = float(np.min(np.linalg.norm(palette_lab - _srgb_to_lab(rgb), axis=1)))
        if best is None or distance > best[0]:
            best = (distance, name, hex_code, rgb)
    _, name, hex_code, rgb = best
    return name, hex_code, (rgb[0], rgb[1], rgb[2], 1.0)


def scores_to_node(source, node, node_count, cancelled=None):
    """The best score of every network node's pair with ``node``, NaN where it has none.

    A pair stored more than once, or in both orientations, counts once at its highest score.
    """
    from utilities import Network_Score_Statistics as scan

    best = np.full(node_count, -np.inf, dtype=np.float64)

    def read_slice(start, stop):
        pairs = source.read(start, stop, nodes=True, check_nodes=True)
        as_source = pairs.sources == node
        hit = as_source | (pairs.targets == node)
        if not hit.any():
            return None
        others = np.where(as_source, pairs.targets, pairs.sources)[hit]
        return others.astype(np.int64), pairs.scores[hit].astype(np.float64)

    for found in scan.run_pass(read_slice, scan.chunk_ranges(source.edge_count), cancelled=cancelled):
        if found is None:
            continue
        others, scores = found
        usable = np.isfinite(scores) & (others != node)
        np.maximum.at(best, others[usable], scores[usable])
    best[np.isneginf(best)] = np.nan
    return best


def _similarity_scores(inputs):
    """The scores of the Viewer's nodes to the reference, from the network file, as a SimilarityResult."""
    import h5py
    import Cache_Manifest as cache_manifest
    from desktop.Viewer_State import _pair_source

    try:
        with h5py.File(inputs.network_path, "r") as data:
            metadata = cache_manifest.validate_network_schema(data)
            network_headers = [
                header.decode("utf-8") if isinstance(header, bytes) else str(header)
                for header in data["headers"][:]
            ]
            position = {}
            for index, header in enumerate(network_headers):
                position.setdefault(header, index)
            viewer_to_network = np.array(
                [position.get(header, -1) for header in inputs.headers], dtype=np.int64
            )
            if (viewer_to_network < 0).any():
                return SimilarityResult(inputs, error=Message(
                    "Error: The network file no longer matches the nodes loaded in the Viewer "
                    "({missing} of them are not in it).",
                    missing=f"{int((viewer_to_network < 0).sum()):,}",
                ))
            kept = np.zeros(len(network_headers), dtype=bool)
            kept[viewer_to_network] = True
            settings = SimpleNamespace(
                INPUT_IS_EVALUE=metadata.network_type == "blast",
                ALIGNMENT_SCORE=inputs.alignment_score,
                NORM_MODE=inputs.norm_mode,
            )
            source = _pair_source(data, settings, kept)
            stored = scores_to_node(source, int(viewer_to_network[inputs.reference]), len(network_headers))
    except Exception as error:
        return SimilarityResult(inputs, error=Message(
            "Error: The network file '{path}' could not be read ({error}).",
            path=inputs.network_path, error=f"{type(error).__name__}: {error}",
        ))
    return SimilarityResult(inputs, scores=stored[viewer_to_network])


def start_by_similarity(viewer, plan):
    """At its turn: check one node is selected and the network file is there; return the scan's inputs, or None."""
    selected = Command_Engine.command_selection(viewer)
    if len(selected) != 1:
        print_help()
        msg = Message(
            "Error: Target property must be specified as {syntax}, or select exactly one "
            "node to color by similarity to it.",
            syntax="{PROPERTY_NAME}",
        )
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return None
    if plan.scheme_unknown:
        msg = Message(
            "Error: Unknown color scheme '{scheme}'. Use a matplotlib colormap name such as "
            "viridis; see 'spectrum help' for the list.",
            scheme=plan.scheme_name,
        )
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return None

    network_path = getattr(cfg, "INPUT_HDF5", "") or ""
    if not network_path or not os.path.isfile(network_path):
        msg = Message(
            "Error: Similarity coloring reads the network file, which was not found: '{path}'.",
            path=network_path,
        )
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return None

    if plan.expr:
        targets = _expression_mask(viewer, plan.expr)
        if targets is None:
            return None
        targets = np.asarray(targets, dtype=bool)
    else:
        # A single node cannot be the range, so no expression means every visible node.
        targets = np.ones(viewer.n_nodes, dtype=bool)
    targets = targets & np.asarray(viewer.visible_mask, dtype=bool)
    if not targets.any():
        msg = Message("No nodes matched the selection criteria (only visible nodes are colored).")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return None

    reference = int(selected[0])
    return SimilarityInputs(
        reference=reference,
        header=str(viewer.full_headers[reference]),
        targets=targets,
        headers=tuple(str(header) for header in viewer.full_headers),
        network_path=network_path,
        alignment_score=getattr(cfg, "ALIGNMENT_SCORE", None),
        norm_mode=getattr(cfg, "NORM_MODE", None),
        scheme_name=plan.scheme_name,
    )


def compute(inputs, progress):
    """The similarity scan, on the command queue's worker thread: no viewer and no Qt."""
    return _similarity_scores(inputs)


def apply(viewer, plan, result):
    """Color the nodes by the scan's scores, as one undo step; an error changes nothing."""
    inputs = result.inputs
    if result.error is not None:
        Command_Engine.print_help(viewer, result.error, report_message=False)
        Command_Engine.command_failed(viewer, result.error)
        return

    reference = inputs.reference
    targets = inputs.targets.copy()
    targets[reference] = False
    scores = np.asarray(result.scores, dtype=np.float64)
    has_score = targets & np.isfinite(scores)
    missing = targets & ~np.isfinite(scores)

    cmap, cmap_ok = get_colormap(inputs.scheme_name)
    canvas = getattr(viewer, 'canvas', None)
    background = getattr(getattr(canvas, 'bgcolor', None), 'rgba', None)
    color_name, color_hex, color_rgba = reference_color(cmap, background)
    viewer._save_state()

    scored = scores[has_score]
    if len(scored):
        vmin, vmax = float(scored.min()), float(scored.max())
        normalized = np.full_like(scored, 0.5) if vmax == vmin else (scored - vmin) / (vmax - vmin)
        viewer.current_colors[has_score] = cmap(normalized)
    else:
        vmin = vmax = None
    viewer.current_colors[missing] = MISSING_COLOR
    viewer.current_colors[reference] = color_rgba

    promoted = targets.copy()
    promoted[reference] = True
    viewer.promote_nodes(promoted)
    viewer.update_nodes()

    count = int(has_score.sum())
    missing_count = int(missing.sum())
    scheme = inputs.scheme_name
    if vmin is None:
        status = Message(
            "No target node has a score to {header}; {missing} node(s) get the noise color.",
            header=inputs.header, missing=f"{missing_count:,}",
        )
        console = status
        terminal = status
    else:
        status = Message(
            "Colored %n node(s) by similarity to {header} (scores {minimum} to {maximum}); "
            "{missing} have no score to it.",
            n=count, header=inputs.header, minimum=f"{vmin:g}", maximum=f"{vmax:g}",
            missing=f"{missing_count:,}",
        )

        def console_line(range_text):
            return JoinedMessage([
                Message(
                    "Similarity spectrum to {header} {range} with scheme '{scheme}': higher scores "
                    "run toward the end of the scheme.",
                    header=inputs.header, range=range_text, scheme=scheme,
                ),
                Message(
                    "The selected node is {color} ({hex}); {missing} node(s) with no score to it "
                    "get the noise color and are left out of the range.",
                    color=color_name, hex=color_hex, missing=f"{missing_count:,}",
                ),
            ])

        console = console_line(Message("(min: {min}, max: {max})", min=f"{vmin:g}", max=f"{vmax:g}"))
        terminal = console_line(_terminal_range_text(vmin, vmax, cmap))
    Command_Engine.print_help(viewer, console, terminal_msg=terminal, report_message=False)
    Command_Engine.command_succeeded(viewer, status)


def _no_progress(text=None):
    """run() reports no progress."""


def run(viewer, args):
    """Run the command at once: its plan, start, compute and apply steps in a row."""
    command_plan = plan(viewer, args)
    if command_plan is None:
        return
    inputs = start(viewer, command_plan)
    if inputs is None:
        return
    apply(viewer, command_plan, compute(inputs, _no_progress))
