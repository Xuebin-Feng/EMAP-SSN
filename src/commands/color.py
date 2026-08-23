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
import fnmatch
import numpy as np
import matplotlib.colors as mcolors
import SSN_Config as cfg
import Command_Engine

def print_help():
    print("""
    Advanced Coloring & Highlighting Tool
    =====================================
    Usage: color [EXPR_1] [COLOR_1] [xSCALE_1] [SHAPE_1] [<EXPR_2> ...]
           color help

    Description:
      Colors, scales, and changes shapes of nodes. You can target nodes using complex 
      boolean expressions. 
      
      * QUICK USE: If no expression is provided, the command automatically targets 
        the nodes currently selected in the viewer using your mouse.

    Attributes:
      1. Color: Name (red, blue) or Hex (#ff0000)
      2. Scale: Prefix with 'x' (e.g., x2, x0.5)
      3. Shape: circle, square, triangle, star, diamond, cross, vbar, hbar, x

    Expression Targets (Do NOT use spaces inside expressions!):
      1. AA Position:  [AA][Pos] (e.g., P106, _100 for gap); negative positions
                       require parentheses (e.g., K(-1), K(-1.1))
      2. Header Text:  "[Text]"  (e.g., "3HMU", "*4A6T*")
      3. File Search:  @[File]@  (e.g., @my_list.txt@)
      4. NCBI/PDB:     @[NCBI][File]@ or @[PDB][File]@ (Regex extraction)
      5. Labels:       #[Name]#  (e.g., #cluster_1#, #noise#, #my_group#)
      6. UI Selection: $sele$     (Explicitly targets selected nodes)
      7. Metadata:     {Key Op Val} (e.g., {Length>500}, {Organism=*coli*})

    Logic Operators:
      & (AND), | (OR), ! (NOT), ^ (XOR)

    Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. If any expression is invalid, no assignments are applied.
      A valid expression may match zero nodes.

    Examples:
      color red x2 triangle             (Modifies currently selected nodes)
      color P106 red                    (Colors nodes with Proline at pos 106 red)
      color "ATA"&#cluster_2# blue x1.5 (Colors "ATA" matches inside Cluster 2)
      color {Organism=*coli*} green     (Colors nodes where Organism matches *coli* green)
      color #cluster_1# red #noise# x0  (Chains multiple commands together)
    """)

def run(viewer, args):
    if args and args[0].lower() == 'reset':
        Command_Engine.execute_reset(viewer, ["colors"])
        return

    if not args:
        msg = "Error: Color command requires at least one property (color, scale, or shape) or expression.\nUsage: color [EXPR_1] [COLOR_1] [xSCALE_1] [SHAPE_1]"
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # ALWAYS update the _sele.txt file to keep the cache fresh for implicit selection
    header_dir = getattr(cfg, 'HEADER_LIST_DIR', os.path.join("Input_Files", "Header_Lists"))
    os.makedirs(header_dir, exist_ok=True)
    sele_path = os.path.join(header_dir, "_sele.txt")
    
    if hasattr(viewer, 'selected_indices') and viewer.selected_indices:
        with open(sele_path, "w", encoding="utf-8", newline="\n") as f:
            for idx in viewer.selected_indices:
                f.write(viewer.full_headers[idx] + "\n")
    else:
        # Clear it out so old selections don't apply if nothing is selected
        if os.path.exists(sele_path):
            open(sele_path, "w", encoding="utf-8").close()

    # FIX: Replace with correct file syntax, removing the literal quotes
    args = [re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', arg, flags=re.IGNORECASE) for arg in args]

    vispy_symbols = ['disc', 'arrow', 'ring', 'clobber', 'square', 'x', 'diamond', 'vbar', 'hbar', 
                     'cross', 'tailed_arrow', 'triangle_up', 'triangle_down', 'star', 'cross_lines', 
                     'o', '+', '++', 's', '-', '|', '->', '>', '^', 'v', '*']

    shape_aliases = {
        'circle': 'disc',
        'triangle': 'triangle_up'
    }

    assignments = []
    current_expr = None
    current_color = None
    current_scale = None
    current_shape = None

    def push_assignment():
        nonlocal current_expr, current_color, current_scale, current_shape
        
        # NEW: Default to targeting selected nodes if properties exist but no expression is given
        if not current_expr and (current_color or current_scale or current_shape):
            current_expr = '@_sele.txt@'
            
        if current_expr and (current_color or current_scale or current_shape):
            assignments.append((current_expr, current_color, current_scale, current_shape))
        elif current_expr:
            print(f"Warning: Skipping '{current_expr}' (No valid color, scale, or shape provided)")

    for arg in args:
        # 1. Check if Scale (e.g., x2.5). Force lowercase 'x'.
        if arg.startswith('x'):
            try:
                current_scale = float(arg[1:])
                continue
            except ValueError:
                pass
                
        # 2. Check if Shape
        arg_lower = arg.lower()
        mapped_shape = shape_aliases.get(arg_lower, arg_lower)
        
        if mapped_shape in vispy_symbols:
            if current_shape is not None:
                push_assignment()
                current_color = None; current_scale = None; current_shape = mapped_shape
                current_expr = None
            else:
                current_shape = mapped_shape
            continue

        # 3. Check if explicitly an expression
        if any(c in arg for c in '&|!^"@') or arg.count('#') >= 2 or re.match(r'^[a-zA-Z_][\d\.]+$', arg):
            if current_expr:
                push_assignment()
                current_color = None; current_scale = None; current_shape = None
            current_expr = arg
            continue
            
        # 4. Check if Color
        is_color = False
        try:
            mcolors.to_rgba(arg)
            is_color = True
        except (TypeError, ValueError):
            pass
        
        if is_color:
            if current_color is not None:
                push_assignment()
                current_color = None; current_scale = None; current_shape = None
                current_expr = None
                current_color = arg
            else:
                current_color = arg
            continue
            
        # 5. Fallback for unrecognized tokens
        if current_expr:
            push_assignment()
            current_color = None; current_scale = None; current_shape = None
        current_expr = arg

    if current_expr or current_color or current_scale or current_shape: 
        push_assignment()
        
    if not assignments:
        viewer.console_text.text = "Error: No valid assignments found."
        return

    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
    
    total_modified = 0
    stats = []
    state_saved = False  # <--- NEW FLAG
    modified_nodes = np.zeros(viewer.n_nodes, dtype=bool)

    evaluated_assignments = []
    for expr, color_str, scale_val, shape_val in assignments:
        if expr:
            expr = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr)
        try:
            mask = Command_Engine.parse_advanced_expression(expr, viewer_to_aln, valid_indices, viewer.full_headers, getattr(viewer, 'cluster_labels', None), getattr(viewer, 'group_labels', None), getattr(viewer, 'alignment', None), metadata=getattr(viewer, 'metadata', None))
        except Exception as e:
            Command_Engine.report_selection_error(viewer, expr, e, "Color")
            return

        evaluated_assignments.append(
            (expr, color_str, scale_val, shape_val, mask, int(np.sum(mask)))
        )

    # Evaluation above is intentionally side-effect free so a bad later
    # expression cannot leave earlier assignments partially applied.
    if not hasattr(viewer, 'current_shapes'):
        viewer.current_shapes = np.full(viewer.n_nodes, 'disc', dtype=object)

    for expr, color_str, scale_val, shape_val, mask, count in evaluated_assignments:
        if count <= 0:
            continue

        # Save state only once, and only if a match is actually found.
        if not state_saved:
            viewer._save_state()
            state_saved = True

        if color_str:
            new_rgba = mcolors.to_rgba(color_str)
            viewer.current_colors[mask] = new_rgba

        if scale_val: viewer.current_sizes[mask] = cfg.NODE_SIZE * scale_val
        if shape_val: viewer.current_shapes[mask] = shape_val

        modified_nodes |= mask
        total_modified += count

        labels = []
        if color_str: labels.append(color_str)
        if scale_val: labels.append(f"x{scale_val}")
        if shape_val: labels.append(shape_val)
        stats.append(f"{count} nodes ({', '.join(labels)})")
            
    if total_modified > 0:
        viewer.promote_nodes(modified_nodes)
        viewer.update_nodes()
        msg = f"Applied: {'; '.join(stats)}"
        viewer.console_text.text = msg
        print(f"\nSuccess! {msg}")
    else:
        viewer.console_text.text = "No nodes matched criteria."
        print("\nNo nodes matched your criteria.")
