# Copyright 2026 Xuebin Feng
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
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import SSN_Config as cfg
import SSN_Utils as utils
import Command_Engine

def print_help():
    print("""
    Spectrum Coloring Tool
    ======================
    Usage: spectrum [EXPRESSION] prop:<PROPERTY_NAME> [scheme:<COLOR_SCHEME>]
           spectrum help

    Description:
      Colors nodes along a color gradient (spectrum) based on the values of a numerical property.
      You can optionally target a subset of nodes using a logical expression.
      The sequence of the arguments does not matter.

      * IMPORTANT: This command only applies to visible nodes.

    Arguments:
      prop:<PROPERTY_NAME> or property:<PROPERTY_NAME>
                              - The target numerical property name (e.g., prop:Length, property:Length).
      scheme:<COLOR_SCHEME> or color:<COLOR_SCHEME>
                              - (Optional) Matplotlib colormap name. Defaults to 'coolwarm'.
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
                                If omitted, all nodes in the network are colored.

    Examples:
      spectrum prop:Length
      spectrum property:Length color:plasma
      spectrum #cluster_1# prop:Length scheme:coolwarm
      spectrum {Organism=*coli*} property:Length
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

def run(viewer, args):
    if not args:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: Missing arguments for spectrum coloring."
        return

    # Parse arguments
    expr = None
    prop_name = None
    scheme_name = 'coolwarm'

    for arg in args:
        if arg.startswith('prop:'):
            prop_name = arg[len('prop:'):].strip()
        elif arg.startswith('property:'):
            prop_name = arg[len('property:'):].strip()
        elif arg.startswith('scheme:'):
            scheme_name = arg[len('scheme:'):].strip()
        elif arg.startswith('color:'):
            scheme_name = arg[len('color:'):].strip()
        elif arg.lower() in ['help', '-h', '--help']:
            print_help()
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = "Help information printed to the terminal"
            return
        else:
            expr = arg.strip()

    if not prop_name:
        print_help()
        Command_Engine.print_help(viewer, "Error: Target property must be specified using prop:<property_name> or property:<property_name>.")
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

    # Keep selection cache fresh if selection expression is used
    header_dir = getattr(cfg, 'HEADER_LIST_DIR', os.path.join("Input_Files", "Header_Lists"))
    os.makedirs(header_dir, exist_ok=True)
    sele_path = os.path.join(header_dir, "_sele.txt")
    
    if hasattr(viewer, 'selected_indices') and viewer.selected_indices:
        with open(sele_path, "w", encoding="utf-8") as f:
            for idx in viewer.selected_indices:
                f.write(viewer.full_headers[idx] + "\n")
    else:
        if os.path.exists(sele_path):
            open(sele_path, 'w').close()

    # Preprocess expression (replace $sele$ and remove spaces in {})
    if expr:
        expr = re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr)

    # Determine mask
    if expr:
        viewer_to_aln = np.full(len(viewer.full_headers), -1, dtype=int)
        if (getattr(viewer, 'alignment', None).aln if getattr(viewer, 'alignment', None) else None) is not None:
            for i, h in enumerate(viewer.full_headers):
                if h in viewer.alignment.seq_map:
                    viewer_to_aln[i] = viewer.alignment.seq_map[h]
        valid_indices = np.where(viewer_to_aln != -1)[0]
        
        try:
            mask = Command_Engine.parse_advanced_expression(
                expr, viewer_to_aln, valid_indices, viewer.full_headers,
                getattr(viewer, 'cluster_labels', None), getattr(viewer, 'group_labels', None),
                getattr(viewer, 'alignment', None), metadata=viewer.metadata
            )
        except Exception as e:
            Command_Engine.print_help(viewer, f"Error parsing expression '{expr}': {e}")
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
    viewer.update_nodes()

    # Automatically invoke "meta display" to show the property used for the spectrum
    try:
        import importlib
        meta_module = importlib.import_module("commands.meta")
        meta_module.run(viewer, ["display", matched_key])
    except Exception as e:
        print(f"Warning: Failed to automatically enable metadata display: {e}")
    
    msg = f"Spectrum coloring applied to {np.sum(full_valid_mask)} nodes using property '{matched_key}' (min: {vmin}, max: {vmax}) with scheme '{scheme_name}'."
    if np.any(nan_mask):
        msg += f" {np.sum(nan_mask)} nodes with invalid values colored gray."
    
    if not cmap_ok:
        msg = f"[Warning: '{scheme_name}' not found, using coolwarm] " + msg
    
    Command_Engine.print_help(viewer, msg)
    print(f"\nSuccess! {msg}")
