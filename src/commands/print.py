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

import Command_Engine
import os
import math
import datetime
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from vispy import app
import SSN_Config as cfg
import SSN_Utils as utils

def print_help():
    print("""
    SSN Image Export & Printing Tool
    ================================
    Usage: print [FILENAME] [MODIFIERS]
           print help

    Description:
      Exports a high-resolution snapshot of the current 3D viewer state. 
      Images are automatically saved to your 'Results/Saved_Images/' directory.

    Modifiers (Can be combined, except for SVG):
      transparent : Removes the white background (PNG only).
      full        : Automatically pans the camera and stitches multiple tiles 
                    together to generate a massive, ultra-high-resolution PNG 
                    of the entire network without OpenGL edge-clipping.
      svg         : Reconstructs the visible network as a Scalable Vector Graphic 
                    for infinite zoom without pixelation (Not compatible with 
                    other modifiers).

    Examples:
      print                               (Saves view as a timestamped PNG)
      print my_network                    (Saves view as my_network.png)
      print my_network transparent        (Saves as a transparent PNG)
      print my_network full transparent   (Stitches a massive transparent PNG)
      print my_network svg                (Saves view as a vector SVG file)
    """)

def _export_svg(viewer, filepath):
    """Generates a structured, layered SVG vector file for Adobe Illustrator compatibility."""
    print("Generating layered, editable SVG for Illustrator...")
    
    # 1. Filter visible elements
    vis = viewer.visible_mask
    if not np.any(vis):
        print("Warning: No visible nodes to export.")
        return
        
    pos = viewer.pos[vis]
    colors = viewer.current_colors[vis]
    sizes = viewer.current_sizes[vis]
    shapes = viewer.current_shapes[vis]
    
    # 2. Calculate bounding box
    min_x, min_y = np.min(pos[:, :2], axis=0)
    max_x, max_y = np.max(pos[:, :2], axis=0)
    
    w_bounds = max_x - min_x
    h_bounds = max_y - min_y
    
    # Add 5% padding so outer nodes aren't clipped by the viewport boundaries
    pad_x = max(w_bounds * 0.05, 5.0) 
    pad_y = max(h_bounds * 0.05, 5.0)
    
    target_min_x = min_x - pad_x
    target_max_x = max_x + pad_x
    target_min_y = min_y - pad_y
    target_max_y = max_y + pad_y
    
    width = target_max_x - target_min_x
    height = target_max_y - target_min_y
    if height == 0: height = 1.0
    
    # Coordinate conversion helpers
    def get_svg_coords(x, y):
        # Flip Y axis so Cartesian +Y goes upward, matching viewer coordinates
        return x - target_min_x, target_max_y - y
        
    # Color translation helper for strict SVG 1.1/Illustrator compatibility
    def get_color_attrs(rgba, is_stroke=False):
        r, g, b, a = rgba
        prefix = "stroke" if is_stroke else "fill"
        color_val = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        return f'{prefix}="{color_val}" {prefix}-opacity="{a:.3f}"'

    # 3. Generate SVG XML lines
    svg_lines = []
    svg_lines.append(f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.3f} {height:.3f}" width="{width:.3f}" height="{height:.3f}">')
    
    # 4. Background layer
    bg_color = viewer.canvas.bgcolor.rgba
    bg_color_str = f"rgb({int(bg_color[0]*255)},{int(bg_color[1]*255)},{int(bg_color[2]*255)})"
    svg_lines.append(f'  <!-- Background -->')
    svg_lines.append(f'  <rect width="{width:.3f}" height="{height:.3f}" fill="{bg_color_str}" fill-opacity="{bg_color[3]:.3f}" />')
    
    # 5. Edges layer
    valid_edges_mask = viewer.visible_mask[viewer.edges[:, 0]] & viewer.visible_mask[viewer.edges[:, 1]]
    active_edges = viewer.edges[valid_edges_mask]
    
    edge_alpha = getattr(cfg, 'EDGE_ALPHA', 0.2)
    edge_width = getattr(cfg, 'EDGE_WIDTH', 0.5)
    
    svg_lines.append(f'  <!-- Edges -->')
    svg_lines.append(f'  <g id="edges" name="Edges">')
    for edge in active_edges:
        x1, y1 = get_svg_coords(viewer.pos[edge[0], 0], viewer.pos[edge[0], 1])
        x2, y2 = get_svg_coords(viewer.pos[edge[1], 0], viewer.pos[edge[1], 1])
        svg_lines.append(f'    <line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="rgb(0,0,0)" stroke-opacity="{edge_alpha:.3f}" stroke-width="{edge_width:.3f}" />')
    svg_lines.append(f'  </g>')
    
    # 6. Nodes layer
    svg_lines.append(f'  <!-- Nodes -->')
    svg_lines.append(f'  <g id="nodes" name="Nodes">')
    
    for i in range(len(pos)):
        cx, cy = get_svg_coords(pos[i, 0], pos[i, 1])
        d = sizes[i]
        r = d / 2.0
        shape = shapes[i]
        rgba = colors[i]
        
        # Check if shape is stroke-only
        stroke_only = shape in ['cross', 'x', 'vbar', 'hbar', 'ring']
        
        if stroke_only:
            fill_attrs = 'fill="none"'
            stroke_attrs = get_color_attrs(rgba, is_stroke=True) + f' stroke-width="{r * 0.4:.3f}"'
        else:
            fill_attrs = get_color_attrs(rgba)
            stroke_attrs = 'stroke="rgb(0,0,0)" stroke-opacity="1.0" stroke-width="0.5"'
            
        attrs = f'{fill_attrs} {stroke_attrs}'
        
        # Write shape elements
        if shape in ['circle', 'disc', 'o', 'ring']:
            svg_lines.append(f'    <circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" {attrs} />')
            
        elif shape in ['square', 's']:
            svg_lines.append(f'    <rect x="{cx - r:.3f}" y="{cy - r:.3f}" width="{d:.3f}" height="{d:.3f}" {attrs} />')
            
        elif shape in ['triangle', 'triangle_up', '^']:
            points = f"{cx:.3f},{cy - r:.3f} {cx + 0.866 * r:.3f},{cy + 0.5 * r:.3f} {cx - 0.866 * r:.3f},{cy + 0.5 * r:.3f}"
            svg_lines.append(f'    <polygon points="{points}" {attrs} />')
            
        elif shape in ['triangle_down', 'v']:
            points = f"{cx:.3f},{cy + r:.3f} {cx + 0.866 * r:.3f},{cy - 0.5 * r:.3f} {cx - 0.866 * r:.3f},{cy - 0.5 * r:.3f}"
            svg_lines.append(f'    <polygon points="{points}" {attrs} />')
            
        elif shape in ['diamond', 'D']:
            points = f"{cx:.3f},{cy - r:.3f} {cx + r:.3f},{cy:.3f} {cx:.3f},{cy + r:.3f} {cx - r:.3f},{cy:.3f}"
            svg_lines.append(f'    <polygon points="{points}" {attrs} />')
            
        elif shape in ['star', '*']:
            pts = []
            for j in range(10):
                angle = -math.pi / 2.0 + j * math.pi / 5.0
                rad = r if j % 2 == 0 else r * 0.4
                px = cx + rad * math.cos(angle)
                py = cy + rad * math.sin(angle)
                pts.append(f"{px:.3f},{py:.3f}")
            points = " ".join(pts)
            svg_lines.append(f'    <polygon points="{points}" {attrs} />')
            
          # Write shape elements
        elif shape in ['cross', '+']:
            path_d = f"M {cx - r:.3f} {cy:.3f} L {cx + r:.3f} {cy:.3f} M {cx:.3f} {cy - r:.3f} L {cx:.3f} {cy + r:.3f}"
            svg_lines.append(f'    <path d="{path_d}" {attrs} />')
            
        elif shape == 'x':
            off = 0.707 * r
            path_d = f"M {cx - off:.3f} {cy - off:.3f} L {cx + off:.3f} {cy + off:.3f} M {cx - off:.3f} {cy + off:.3f} L {cx + off:.3f} {cy - off:.3f}"
            svg_lines.append(f'    <path d="{path_d}" {attrs} />')
            
        elif shape in ['vbar', '|']:
            svg_lines.append(f'    <line x1="{cx:.3f}" y1="{cy - r:.3f}" x2="{cx:.3f}" y2="{cy + r:.3f}" {attrs} />')
            
        elif shape in ['hbar', '-', '_']:
            svg_lines.append(f'    <line x1="{cx - r:.3f}" y1="{cy:.3f}" x2="{cx + r:.3f}" y2="{cy:.3f}" {attrs} />')
            
        elif shape in ['arrow', 'tailed_arrow', '->', '>']:
            points = f"{cx + r:.3f},{cy:.3f} {cx - r * 0.5:.3f},{cy - 0.866 * r:.3f} {cx - r * 0.5:.3f},{cy + 0.866 * r:.3f}"
            svg_lines.append(f'    <polygon points="{points}" {attrs} />')
            
        elif shape in ['clobber', 'p']:
            pts = []
            for j in range(5):
                angle = -math.pi / 2.0 + j * 2.0 * math.pi / 5.0
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                pts.append(f"{px:.3f},{py:.3f}")
            points = " ".join(pts)
            svg_lines.append(f'    <polygon points="{points}" {attrs} />')
            
        elif shape in ['cross_lines', 'P', '++']:
            w = r * 0.4
            path_d = f"M {cx-r:.3f} {cy-w:.3f} H {cx-w:.3f} V {cy-r:.3f} H {cx+w:.3f} V {cy-w:.3f} H {cx+r:.3f} V {cy+w:.3f} H {cx+w:.3f} V {cy+r:.3f} H {cx-w:.3f} V {cy+w:.3f} H {cx-r:.3f} Z"
            svg_lines.append(f'    <path d="{path_d}" {attrs} />')
            
        else:
            # Failsafe: circle
            svg_lines.append(f'    <circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" {attrs} />')
            
    svg_lines.append(f'  </g>')
    svg_lines.append(f'</svg>')
    
    # 7. Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_lines))
    print(f"Successfully generated structured SVG at: {filepath}")

def _capture_tile(viewer, is_transparent):
    """Helper function to render the current camera view and extract RGBA."""
    if is_transparent:
        viewer.canvas.bgcolor = 'black'
        viewer.canvas.update()
        app.process_events()
        img_black = viewer.canvas.render()[..., :3].astype(np.float32) / 255.0
        
        viewer.canvas.bgcolor = 'white'
        viewer.canvas.update()
        app.process_events()
        img_white = viewer.canvas.render()[..., :3].astype(np.float32) / 255.0
        
        alpha = 1.0 - img_white + img_black
        alpha_channel = np.clip(np.mean(alpha, axis=2), 0.0, 1.0)
        
        rgb_channels = np.zeros_like(img_black)
        mask = alpha_channel > 1e-6
        for i in range(3):
            rgb_channels[..., i][mask] = np.clip(img_black[..., i][mask] / alpha_channel[mask], 0.0, 1.0)
        
        final_tile = np.zeros((img_black.shape[0], img_black.shape[1], 4), dtype=np.float32)
        final_tile[..., :3] = rgb_channels
        final_tile[..., 3] = alpha_channel
        return final_tile
    else:
        viewer.canvas.update()
        app.process_events()
        img = viewer.canvas.render().astype(np.float32) / 255.0
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgba = np.ones((img.shape[0], img.shape[1], 4), dtype=np.float32)
            rgba[..., :3] = img
            return rgba
        return img

def run(viewer, args):

    # 1. Setup paths
    save_dir = getattr(cfg, 'PRINT_SAVE_DIR', os.path.join("Results", "Saved_Images"))
    os.makedirs(save_dir, exist_ok=True)
    
    # Check for help
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # 2. Parse arguments
    is_transparent = False
    is_full = False
    is_svg = False
    final_args = []
    
    for a in args:
        if a.lower() == "transparent":
            is_transparent = True
        elif a.lower() == "full":
            is_full = True
        elif a.lower() == "svg":
            is_svg = True
        else:
            final_args.append(a)
            
    # SVG Constraints
    if is_svg:
        if is_transparent or is_full:
            msg = "Error: 'SVG' export is not compatible with 'transparent' or 'full'."
            print(f"\n{msg}")
            viewer.console_text.text = msg
            if hasattr(viewer, 'console_bg'): viewer.console_bg.visible = True
            return
            
        if len(args) > 2:
            msg = "Error: Maximum of 2 keywords allowed when using 'SVG' (e.g., 'print [filename] svg')."
            print(f"\n{msg}")
            viewer.console_text.text = msg
            if hasattr(viewer, 'console_bg'): viewer.console_bg.visible = True
            return

    args = final_args
        
    # 3. Determine filename
    ext = ".svg" if is_svg else ".png"
    
    if len(args) > 0:
        filename = "_".join(args)
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{cfg.SEQUENCE_SET}_{timestamp}"
        
    if not filename.lower().endswith(ext):
        filename += ext
        
    filepath = os.path.join(save_dir, filename)
    
    # 4. Store original states
    original_bgcolor = viewer.canvas.bgcolor
    instr_visible = viewer.instr_text.visible
    zoom_visible = viewer.zoom_text.visible if hasattr(viewer, 'zoom_text') else False
    tooltip_visible = viewer.tooltip.visible if hasattr(viewer, 'tooltip') else False
    hidden_visible = viewer.hidden_text.visible if hasattr(viewer, 'hidden_text') else False
    
    orig_rect = viewer.view.camera.rect
    orig_aspect = viewer.view.camera.aspect
    
    try:
        # Hide UI
        viewer.instr_text.visible = False
        if hasattr(viewer, 'zoom_text'): viewer.zoom_text.visible = False
        if hasattr(viewer, 'tooltip'): viewer.tooltip.visible = False
        if hasattr(viewer, 'hidden_text'): viewer.hidden_text.visible = False
        if hasattr(viewer, 'console_bg'):
            viewer.console_bg.visible = False
            viewer.console_text.text = ""
            
        if is_svg:
            _export_svg(viewer, filepath)
            
        elif is_full:
            print("\nCalculating seamless tile grid (bypassing OpenGL edge-clipping)...")
            
            # 1. Keep the aspect ratio locked to preserve rendering proportions
            orig_real_rect = viewer.view.camera._real_rect if hasattr(viewer.view.camera, '_real_rect') else orig_rect
            
            # 2. Get exact physical pixel resolution
            dummy_tile = _capture_tile(viewer, is_transparent)
            tile_px_h, tile_px_w = dummy_tile.shape[:2]
            
            # 3. Define the trash margin (Throw away outer 15% of pixels)
            margin_px = int(min(tile_px_w, tile_px_h) * 0.15)
            
            # Calculate the dimensions of the "safe" middle area we will actually keep
            keep_px_w = tile_px_w - (2 * margin_px)
            keep_px_h = tile_px_h - (2 * margin_px)
            
            # 4. Calculate exact Units-Per-Pixel (UPP) using the actual visible rect bounds
            upp_x = orig_real_rect.width / tile_px_w
            upp_y = orig_real_rect.height / tile_px_h
            
            # Calculate how much world space our "safe" area covers
            step_world_w = keep_px_w * upp_x
            step_world_h = keep_px_h * upp_y
            
            # 5. Get world bounding box with padding
            # Recalculate bounding box using only visible nodes
            vis = viewer.visible_mask
            if not np.any(vis):
                msg = "Error: No visible nodes to export."
                print(f"\n{msg}")
                viewer.console_text.text = msg
                if hasattr(viewer, 'console_bg'): viewer.console_bg.visible = True
                return
                
            visible_pos = viewer.pos[vis, :2]
            min_x, min_y = np.min(visible_pos, axis=0)
            max_x, max_y = np.max(visible_pos, axis=0)
            w_bounds = max_x - min_x
            h_bounds = max_y - min_y
            pad_x = max(w_bounds * 0.05, 5.0) 
            pad_y = max(h_bounds * 0.05, 5.0)
            
            world_left = min_x - pad_x
            world_right = max_x + pad_x
            world_bottom = min_y - pad_y
            world_top = max_y + pad_y
            
            # 6. Calculate required tiles based strictly on the "safe" area
            n_cols = math.ceil((world_right - world_left) / step_world_w)
            n_rows = math.ceil((world_top - world_bottom) / step_world_h)
            total_tiles = n_cols * n_rows
            
            print(f"Network bounding box requires {n_cols}x{n_rows} safe tiles ({total_tiles} total renders).")
            
            # 7. Initialize giant mosaic canvas
            canvas_w = n_cols * keep_px_w
            canvas_h = n_rows * keep_px_h
            final_img = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
            
            # 8. Tiling Loop
            tile_count = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    tile_count += 1
                    print(f"  -> Snapping tile {tile_count}/{total_tiles}...")
                    
                    # Target world coordinates of the piece we KEEP
                    target_keep_left = world_left + (c * step_world_w)
                    target_keep_top = world_top - (r * step_world_h)
                    
                    # The actual camera pushes OUTWARD by the margin size so clipping happens off-screen
                    cam_left = target_keep_left - (margin_px * upp_x)
                    cam_top = target_keep_top + (margin_px * upp_y)
                    cam_bottom = cam_top - orig_real_rect.height
                    
                    # Snap camera
                    viewer.view.camera.rect = (cam_left, cam_bottom, orig_real_rect.width, orig_real_rect.height)
                    app.process_events() 
                    
                    tile_img = _capture_tile(viewer, is_transparent)
                    
                    # The Cookie Cutter: Snip off the unsafe clipped margins
                    cropped_tile = tile_img[margin_px : tile_px_h - margin_px, margin_px : tile_px_w - margin_px]
                    
                    # Paste the perfectly safe center block side-by-side
                    paste_x = c * keep_px_w
                    paste_y = r * keep_px_h
                    final_img[paste_y : paste_y + keep_px_h, paste_x : paste_x + keep_px_w, :] = cropped_tile
                    
            # 9. Crop to exact requested world bounds
            print("Stitching complete. Cropping to exact bounds...")
            target_px_w = int((world_right - world_left) / upp_x)
            target_px_h = int((world_top - world_bottom) / upp_y)
            
            # Clamp to be safe
            target_px_w = min(target_px_w, final_img.shape[1])
            target_px_h = min(target_px_h, final_img.shape[0])
            
            final_img = final_img[:target_px_h, :target_px_w]
            
            mpimg.imsave(filepath, final_img)
            
        else:
            # Standard single-shot render
            final_img = _capture_tile(viewer, is_transparent)
            mpimg.imsave(filepath, final_img)
        
        msg_type = "SVG snapshot" if is_svg else "snapshot"
        if is_transparent and not is_svg: msg_type = "transparent " + msg_type
        if is_full and not is_svg: msg_type = "full stitched " + msg_type
            
        msg = f"Successfully saved {msg_type}: {filepath}"
        print(f"\n{msg}")
        
        viewer.console_text.text = f"Saved {ext.upper()}: {filename}"
        if hasattr(viewer, 'console_bg'): viewer.console_bg.visible = True
        
        # Open the save folder in the system file explorer
        utils.open_in_file_manager(save_dir)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Failed to save {ext.upper()}: {e}"
        print(f"\n{error_msg}")
        viewer.console_text.text = f"Error saving {ext.upper()}. Check console."
        if hasattr(viewer, 'console_bg'): viewer.console_bg.visible = True
             
    finally:
        # --- RESTORE STATE ---
        
        viewer.view.camera.aspect = orig_aspect
        viewer.view.camera.rect = orig_rect
        
        viewer.canvas.bgcolor = original_bgcolor
        viewer.instr_text.visible = instr_visible
        if hasattr(viewer, 'zoom_text'): viewer.zoom_text.visible = zoom_visible
        if hasattr(viewer, 'tooltip'): viewer.tooltip.visible = tooltip_visible
        if hasattr(viewer, 'hidden_text'): viewer.hidden_text.visible = hidden_visible
            
        viewer.canvas.update()
        app.process_events()