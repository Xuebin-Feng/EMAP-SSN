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
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import Command_Engine


if sys.platform == "win32" and not globals().get("_WINDOWS_ANSI_ENABLED", False):
    os.system("")
    _WINDOWS_ANSI_ENABLED = True


ANSI_RESET = "\033[0m"


def _stdout_supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _terminal_endpoint_text(label, value, rgba):
    text = f"{label}: {value}"
    if not _stdout_supports_color():
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

def print_help():
    print("""
    Spectrum Coloring Tool
    ======================
    Usage: spectrum [EXPRESSION] {PROPERTY_NAME} [COLOR_SCHEME]
           spectrum help

    Description:
      Colors nodes along a color gradient (spectrum) based on the values of a numerical property.
      You can optionally target a subset of nodes using a logical expression.
      The sequence of the arguments does not matter.

      * IMPORTANT: This command only applies to visible nodes.

    Arguments:
      {PROPERTY_NAME}         - The required numerical metadata property (e.g., {Length}).
      COLOR_SCHEME            - (Optional) Matplotlib colormap name. Defaults to 'coolwarm'.
                                Supported schemes:
                                * Perceptually Uniform: viridis, plasma, inferno, magma, cividis
                                * Sequential: Blues, BuGn, BuPu, GnBu, Greens, Greys, Oranges, 
                                  OrRd, PuBu, PuBuGn, PuRd, Purples, RdPu, Reds, YlGn, YlGnBu, 
                                  YlOrBr, YlOrRd
                                * Diverging: coolwarm, bwr, seismic, spectral, BrBG, PiYG, PRGn, 
                                  PuOr, RdBu, RdGy, RdYlBu, RdYlGn
                                * Cyclic: twilight, twilight_shifted, hsv
                                * Qualitative: tab10, tab20, tab20b, tab20c, Pastel1, Pastel2, 
                                  Paired, Accent, Dark2, Set1, Set2, Set3
                                * Miscellaneous: jet, rainbow, turbo, ocean, terrain, cubehelix, 
                                  gnuplot, gnuplot2, flag, prism, gist_earth, nipy_spectral
      [EXPRESSION]            - (Optional) Logical expression to select which nodes are colored.
                                If omitted, all visible nodes in the network are colored.

    Selection Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. Invalid references abort before colors are changed.
      A valid expression may match zero nodes.

    Examples:
      spectrum {Length}
      spectrum {Length} plasma
      spectrum #cluster_1# {Length} coolwarm
      spectrum {Organism=*coli*} {Length}
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


def is_registered_colormap(scheme_name):
    try:
        if hasattr(mpl, 'colormaps'):
            mpl.colormaps[scheme_name]
            return True
    except KeyError:
        pass
    try:
        cm.get_cmap(scheme_name)
        return True
    except Exception:
        return False

def run(viewer, args):
    if not args:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: Missing arguments for spectrum coloring."
        return

    # Parse command-specific roles before Boolean-expression classification.
    expr = None
    prop_name = None
    scheme_name = 'coolwarm'
    scheme_supplied = False

    for arg in args:
        arg_lower = arg.lower()
        if arg_lower in ['help', '-h', '--help']:
            print_help()
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = "Help information printed to the terminal"
            return
        if arg_lower.startswith(('prop:', 'property:', 'scheme:', 'color:')):
            Command_Engine.print_help(
                viewer,
                "Error: Legacy spectrum prefixes are no longer supported. Use "
                "'spectrum [EXPRESSION] {PROPERTY_NAME} [COLOR_SCHEME]'.",
            )
            return

        property_match = re.fullmatch(r'\{([a-zA-Z0-9_\-.]+)\}', arg)
        if property_match:
            if prop_name is not None:
                Command_Engine.print_help(
                    viewer, "Error: Spectrum accepts exactly one {PROPERTY_NAME}."
                )
                return
            prop_name = property_match.group(1)
            continue

        if is_registered_colormap(arg):
            if scheme_supplied:
                Command_Engine.print_help(
                    viewer, "Error: Spectrum accepts at most one color scheme."
                )
                return
            scheme_name = arg
            scheme_supplied = True
            continue

        classification = Command_Engine.classify_selection_expression(arg)
        if classification.kind == Command_Engine.SelectionClassificationKind.VALID_EXPRESSION:
            if expr is not None:
                Command_Engine.print_help(
                    viewer, "Error: Spectrum accepts at most one Boolean expression."
                )
                return
            expr = arg
            continue
        if classification.kind == Command_Engine.SelectionClassificationKind.MALFORMED_EXPRESSION:
            Command_Engine.report_selection_error(
                viewer, arg, classification.error, "Spectrum"
            )
            return
        if scheme_supplied:
            Command_Engine.print_help(
                viewer,
                f"Error: Unrecognized extra spectrum argument '{arg}'. Only one "
                "color scheme may be supplied.",
            )
            return
        scheme_name = arg
        scheme_supplied = True

    if not prop_name:
        print_help()
        Command_Engine.print_help(
            viewer, "Error: Target property must be specified as {PROPERTY_NAME}."
        )
        return

    if not getattr(viewer, 'metadata', None):
        Command_Engine.print_help(viewer, "Error: No metadata loaded in the viewer.")
        return

    # Resolve property case-insensitively
    matched_key = None
    for k in viewer.metadata.keys():
        if k.lower() == prop_name.lower():
            matched_key = k
            break

    if not matched_key:
        available = ", ".join(viewer.metadata.keys())
        Command_Engine.print_help(viewer, f"Error: Property '{prop_name}' not found. Available properties: {available}")
        return

    prop_data = viewer.metadata[matched_key]
    if prop_data["type"] != "number":
        Command_Engine.print_help(viewer, f"Error: Property '{matched_key}' is not numerical (type is '{prop_data['type']}'). Spectrum coloring requires a numerical property.")
        return

    # Determine mask
    if expr:
        viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
        
        try:
            mask = Command_Engine.parse_advanced_expression(
                expr, viewer_to_aln, valid_indices, viewer.full_headers,
                getattr(viewer, 'cluster_labels', None), getattr(viewer, 'group_labels', None),
                getattr(viewer, 'alignment', None), metadata=viewer.metadata,
                selection_mask=Command_Engine.get_selected_mask(viewer),
            )
        except Exception as e:
            Command_Engine.report_selection_error(viewer, expr, e, "Spectrum")
            return
    else:
        mask = np.ones(viewer.n_nodes, dtype=bool)

    # Restrict to visible nodes
    mask = mask & viewer.visible_mask

    if np.sum(mask) == 0:
        Command_Engine.print_help(viewer, "No nodes matched the selection criteria (only visible nodes are colored).")
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
    valid_mask = ~np.isnan(target_vals)
    valid_vals = target_vals[valid_mask]

    if len(valid_vals) == 0:
        Command_Engine.print_help(viewer, f"Warning: No valid numerical values found in '{matched_key}' for the selected nodes.")
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
    full_valid_mask[mask] = ~np.isnan(values[mask])
    viewer.current_colors[full_valid_mask] = colors_rgba

    # Color nan nodes within mask to neutral light gray
    nan_mask = mask & np.isnan(values)
    if np.any(nan_mask):
        viewer.current_colors[nan_mask] = (0.7, 0.7, 0.7, 1.0)

    # Update viewer
    viewer.promote_nodes(mask)
    viewer.update_nodes()

    # Automatically invoke "meta display" to show the property used for the spectrum
    try:
        import importlib
        meta_module = importlib.import_module("commands.meta")
        meta_module.run(viewer, ["display", matched_key])
    except Exception as e:
        print(f"Warning: Failed to automatically enable metadata display: {e}")
    
    message_prefix = (
        f"Spectrum coloring applied to {np.sum(full_valid_mask)} nodes using "
        f"property '{matched_key}'"
    )
    message_suffix = f"with scheme '{scheme_name}'."
    msg = f"{message_prefix} (min: {vmin}, max: {vmax}) {message_suffix}"
    terminal_msg = (
        f"{message_prefix} {_terminal_range_text(vmin, vmax, cmap)} "
        f"{message_suffix}"
    )
    if np.any(nan_mask):
        invalid_values_msg = (
            f" {np.sum(nan_mask)} nodes with invalid values colored gray."
        )
        msg += invalid_values_msg
        terminal_msg += invalid_values_msg
    
    if not cmap_ok:
        warning = f"[Warning: '{scheme_name}' not found, using coolwarm] "
        msg = warning + msg
        terminal_msg = warning + terminal_msg
    
    Command_Engine.print_help(viewer, msg, terminal_msg=terminal_msg)
