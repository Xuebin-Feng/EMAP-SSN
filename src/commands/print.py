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
import time
import datetime
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from PySide6 import QtCore
from vispy import app
from vispy.scene.visuals import VisualNode
from vispy.scene.widgets import Widget
import EMAPSSN_Config as cfg
from desktop.Desktop_App import open_in_file_manager
from utilities.Localization import Message
from utilities.Output_Names import validate_output_basename
from Viewer_Visual_State import edge_stages

PRINT_DIRECTORY = os.path.join("$analysis_result$", "Saved_Images")
PNG_TRIM_PADDING_PX = 20
PNG_ALPHA_TOLERANCE = 1.0 / 255.0
PNG_BACKGROUND_TOLERANCE = 2.0 / 255.0
# Viewer HUD overlays drawn over the network, left out of every capture.
CAPTURE_HIDDEN_OVERLAYS = (
    'instr_text', 'zoom_text', 'tooltip', 'hidden_text', 'console_bg',
    'console_text', 'background_job_status_text', 'selection_box',
)
# Longest stretch of a full capture without letting Qt paint and run timers.
CAPTURE_EVENT_INTERVAL_S = 0.1

def print_help():
    print("""
    SSN Image Export & Printing Tool
    ================================
    Usage: print [FILENAME] [MODIFIERS]
           print help

    Description:
      Exports a high-resolution snapshot of the current viewer state.
      Images are saved beneath the configured Analysis Results directory
      (default: 'Analysis_Results/Saved_Images/'). FILENAME is a plain file
      name, not a path; .png (or .svg) is added when it is missing.
      PNG exports automatically trim empty margins while retaining a 20-pixel
      border around all rendered content. SVG view-box padding is unchanged.

    Modifiers (Can be combined, except for SVG):
      transparent : Removes the configured background color (PNG only).
      full        : Renders the network tile by tile off screen and stitches
                    the tiles into a massive, ultra-high-resolution PNG of
                    the entire network without OpenGL edge-clipping. The view
                    on screen stays put; keyboard and mouse input waits until
                    the capture ends, and resizing the window cancels it.
      svg         : Reconstructs the visible network as a Scalable Vector Graphic 
                    for infinite zoom without pixelation (Not compatible with 
                    other modifiers). Node sizes and line widths follow the
                    current zoom, as on screen.

    Examples:
      print                               (Saves view as a timestamped PNG)
      print my_network                    (Saves view as my_network.png)
      print my_network transparent        (Saves as a transparent PNG)
      print my_network full transparent   (Stitches a massive transparent PNG)
      print my_network svg                (Saves view as a vector SVG file)
    """)

# The marker names a node's shape can hold: vispy's own aliases, which the
# color command stores as typed, and the extra spellings the export draws.
# Each maps to the canonical name the drawing code below is written for.
_SHAPE_ALIASES = {
    'o': 'disc', 'circle': 'disc',
    's': 'square',
    '^': 'triangle_up', 'triangle': 'triangle_up',
    'v': 'triangle_down',
    'D': 'diamond',
    '*': 'star',
    '+': 'cross',
    '|': 'vbar',
    '-': 'hbar', '_': 'hbar',
    '>': 'arrow', '->': 'tailed_arrow',
    'p': 'clobber',
    'P': 'cross_lines', '++': 'cross_lines',
}
# Shapes drawn as coloured strokes rather than as a filled, outlined area.
_STROKE_ONLY_SHAPES = ('cross', 'x', 'vbar', 'hbar', 'ring')


def _canonical_shape(shape):
    """Return a marker's canonical name, so an alias draws like its name."""
    if isinstance(shape, str):
        return _SHAPE_ALIASES.get(shape, shape)
    return shape


def _available_automatic_filename(directory, filename):
    """Add a numeric suffix when a generated timestamped name is taken.

    Two prints in the same second share a timestamp: the second one is saved
    as name_2.png rather than over the first. A name typed by the user is not
    passed here, and is overwritten as before.
    """
    stem, suffix = os.path.splitext(filename)
    candidate = filename
    index = 2
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _scene_units_per_pixel(viewer):
    """Return how many scene units one logical screen pixel spans in the view.

    Marker sizes and line widths are set in logical pixels: VisPy multiplies
    them by the display scale itself, so they read the same on a high-DPI
    screen. The SVG is written in scene coordinates, so every such size is
    multiplied by this scale. The camera maps its real rect (the requested
    one, widened to the view's aspect ratio) onto the view, which is in
    logical pixels too. The Viewer fixes the aspect at 1, so one scale holds
    for both axes. A view with no area keeps the sizes as they are, 1 to 1.
    """
    view = viewer.view
    camera = view.camera
    rect = camera._real_rect if hasattr(camera, '_real_rect') else camera.rect
    try:
        scale = abs(float(rect.width)) / float(view.size[0])
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    return scale if math.isfinite(scale) and scale > 0.0 else 1.0


def _export_svg(viewer, filepath):
    """Generates a structured, layered SVG vector file for Adobe Illustrator compatibility.

    Sizes set in screen pixels (node size and outline, edge width) are written
    in scene units at the current zoom, so the SVG keeps the proportions of
    the view on screen.

    Returns False, writing nothing, when no node is visible.
    """
    # 1. Filter visible elements
    vis = viewer.visible_mask
    if not np.any(vis):
        return False

    print("Generating layered, editable SVG for Illustrator...")

    # Nodes go in the order the screen draws them, the later covering the
    # earlier, so those the color command or a selection raised stay on top.
    render_order = getattr(viewer, 'visible_node_render_order', None)
    visible = render_order() if callable(render_order) else vis

    pos = viewer.pos[visible]
    colors = viewer.current_colors[visible]
    sizes = viewer.current_sizes[visible]
    shapes = [_canonical_shape(shape) for shape in viewer.current_shapes[visible]]

    # 2. Calculate bounding box
    min_x, min_y = np.min(pos[:, :2], axis=0)
    max_x, max_y = np.max(pos[:, :2], axis=0)

    w_bounds = max_x - min_x
    h_bounds = max_y - min_y

    # Node sizes are screen pixels; the SVG is in scene units.
    unit = _scene_units_per_pixel(viewer)

    # Add 5% padding so outer nodes aren't clipped by the viewport boundaries,
    # and at least what the largest node reaches past its centre: its radius,
    # or the ring's wider stroke. Below the 5-unit minimum nothing changes.
    radii = np.asarray(sizes, dtype=np.float64) / 2.0 * unit
    is_ring = np.array([shape == 'ring' for shape in shapes], dtype=bool)
    reach = np.where(is_ring, radii * 1.2, radii)
    reach = reach[np.isfinite(reach)]
    node_pad = float(reach.max()) if reach.size else 0.0
    pad_x = max(w_bounds * 0.05, 5.0, node_pad)
    pad_y = max(h_bounds * 0.05, 5.0, node_pad)

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
    
    # 5. Edges layer: the on-screen filter (similarity threshold, visible
    # endpoints, UMAP selection edges) in the configured edge color and alpha.
    active_edges, _ = edge_stages(viewer, cfg)

    edge_alpha = getattr(cfg, 'EDGE_ALPHA', 0.2)
    edge_width = getattr(cfg, 'EDGE_WIDTH', 0.5) * unit
    edge_rgba = mcolors.to_rgba(getattr(cfg, 'EDGE_COLOR', '#000000'))
    edge_stroke = get_color_attrs((*edge_rgba[:3], edge_alpha), is_stroke=True)
    boundary_rgba = mcolors.to_rgba(getattr(cfg, 'NODE_BOUNDARY_COLOR', '#000000'))
    boundary_width = getattr(cfg, 'NODE_BOUNDARY_WIDTH', 0.5) * unit
    boundary_stroke = get_color_attrs(boundary_rgba, is_stroke=True) + f' stroke-width="{boundary_width:.3f}"'

    svg_lines.append(f'  <!-- Edges -->')
    svg_lines.append(f'  <g id="edges" name="Edges">')
    for edge in active_edges:
        x1, y1 = get_svg_coords(viewer.pos[edge[0], 0], viewer.pos[edge[0], 1])
        x2, y2 = get_svg_coords(viewer.pos[edge[1], 0], viewer.pos[edge[1], 1])
        svg_lines.append(f'    <line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" {edge_stroke} stroke-width="{edge_width:.3f}" />')
    svg_lines.append(f'  </g>')
    
    # 6. Nodes layer
    svg_lines.append(f'  <!-- Nodes -->')
    svg_lines.append(f'  <g id="nodes" name="Nodes">')
    
    for i in range(len(pos)):
        cx, cy = get_svg_coords(pos[i, 0], pos[i, 1])
        d = sizes[i] * unit
        r = d / 2.0
        shape = shapes[i]
        rgba = colors[i]
        
        # Check if shape is stroke-only
        stroke_only = shape in _STROKE_ONLY_SHAPES
        
        if stroke_only:
            fill_attrs = 'fill="none"'
            stroke_attrs = get_color_attrs(rgba, is_stroke=True) + f' stroke-width="{r * 0.4:.3f}"'
        else:
            fill_attrs = get_color_attrs(rgba)
            stroke_attrs = boundary_stroke
            
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
    
    # 7. Write to file, making the save folder only now that there is a file for it
    save_dir = os.path.dirname(filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_lines))
    print(f"Successfully generated structured SVG at: {filepath}")
    return True

def _capture_tile(viewer, is_transparent):
    """Render the current camera view offscreen and return float RGBA.

    canvas.render() draws into a framebuffer object of its own, and the
    transparent pass hands its two backgrounds to render() rather than
    setting canvas.bgcolor: nothing reaches the window and no event runs.
    """
    if is_transparent:
        img_black = viewer.canvas.render(bgcolor='black')[..., :3].astype(np.float32) / 255.0
        img_white = viewer.canvas.render(bgcolor='white')[..., :3].astype(np.float32) / 255.0
        
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
        img = viewer.canvas.render().astype(np.float32) / 255.0
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgba = np.ones((img.shape[0], img.shape[1], 4), dtype=np.float32)
            rgba[..., :3] = img
            return rgba
        return img


def _capture_overlays(viewer):
    """Return the HUD visuals a capture leaves out, each once."""
    overlays = [getattr(viewer, name, None) for name in CAPTURE_HIDDEN_OVERLAYS]
    overlays += [
        getattr(display, 'text_visual', None)
        for display in getattr(viewer, 'hud_displays', {}).values()
    ]
    unique = []
    for overlay in overlays:
        if (hasattr(overlay, 'visible')
                and not any(overlay is seen for seen in unique)):
            unique.append(overlay)
    return unique


def _render_capture(viewer, is_transparent, overlays, rect=None):
    """Capture one tile with the HUD hidden and the camera at rect, if given.

    The live camera and HUD visibility are read here and put back before
    returning, even on error, and no event is processed in between, so a
    paint only ever sees the live view.
    """
    camera = viewer.view.camera
    live_rect = camera.rect
    shown = [overlay.visible for overlay in overlays]
    try:
        for overlay in overlays:
            overlay.visible = False
        if rect is not None:
            camera.rect = rect
        return _capture_tile(viewer, is_transparent)
    finally:
        if rect is not None:
            camera.rect = live_rect
        for overlay, visible in zip(overlays, shown):
            overlay.visible = visible


def _capture_geometry(viewer):
    """Return what tile pixels depend on besides the scene and camera rect."""
    try:
        return (
            tuple(getattr(viewer.canvas, 'size', ())),
            tuple(getattr(viewer.canvas, 'physical_size', ())),
            getattr(viewer.view.camera, 'aspect', None),
        )
    except Exception:
        return None


def _process_events_without_input():
    """Run pending paints, timers and window events; keyboard and mouse
    input stays queued until Qt's event loop runs normally again."""
    application = QtCore.QCoreApplication.instance()
    if application is not None:
        application.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )


class _CaptureInterrupted(RuntimeError):
    """Something a full capture depends on changed between its tiles.

    It is raised with a Message, which the console line shows translated.
    """


class _CaptureEvents:
    """Lets Qt run between full-capture tiles, and cancels spoiled captures.

    Between tiles the live view is on screen, so pending paints, timers and
    window events run while keyboard and mouse input waits. What does run
    can still spoil the mosaic: a resize or display-scale change alters the
    tile geometry, and a web or background action can redraw the network.
    Either cancels the capture. A camera move alone is harmless, because
    every tile sets its own camera and puts the live one back.
    """

    def __init__(self, viewer, overlays):
        self._viewer = viewer
        self._geometry = _capture_geometry(viewer)
        self._changed = False
        self._emitters = []
        self._next_pause = time.perf_counter() + CAPTURE_EVENT_INTERVAL_S
        root = getattr(viewer.canvas, 'scene', None)
        if root is None:
            return
        # Every visual a tile draws, plus any the network gains: the HUD
        # overlays are hidden in tiles, and the view widgets change only
        # with the camera or the window size.
        skipped = {id(overlay) for overlay in overlays}
        self._watch(root.events.children_change)
        nodes = [root]
        while nodes:
            node = nodes.pop()
            nodes.extend(node.children)
            if (isinstance(node, VisualNode) and not isinstance(node, Widget)
                    and id(node) not in skipped):
                self._watch(node.events.update)

    def _watch(self, emitter):
        emitter.connect(self._mark_changed)
        self._emitters.append(emitter)

    def _mark_changed(self, event=None):
        self._changed = True

    def pause_if_due(self):
        """Process events if the last pause was long enough ago."""
        if time.perf_counter() < self._next_pause:
            return
        camera = self._viewer.view.camera
        live_rect = camera.rect
        self._changed = False
        _process_events_without_input()
        if _capture_geometry(self._viewer) != self._geometry:
            raise _CaptureInterrupted(Message(
                "the Viewer window was resized during the capture, so nothing "
                "was saved. Run print again."
            ))
        # A camera move updates every visual too, and only that is harmless.
        if self._changed and camera.rect == live_rect:
            raise _CaptureInterrupted(Message(
                "the network display changed during the capture, so nothing "
                "was saved. Run print again."
            ))
        self._next_pause = time.perf_counter() + CAPTURE_EVENT_INTERVAL_S

    def close(self):
        for emitter in self._emitters:
            emitter.disconnect(self._mark_changed)
        self._emitters = []


def _background_rgb(background_color):
    """Return the canvas background color quantized to rendered 8-bit RGB."""
    if hasattr(background_color, 'rgba'):
        rgba = background_color.rgba
    else:
        rgba = mcolors.to_rgba(background_color)
    rgb = np.asarray(rgba, dtype=np.float32)[:3]
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0) / 255.0


def _trim_png_margins(
    image,
    is_transparent,
    background_color,
    padding_px=PNG_TRIM_PADDING_PX,
):
    """Crop background-only PNG margins while preserving a fixed pixel border."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("PNG image must have height, width, and RGB channels.")

    if is_transparent:
        if image.shape[2] < 4:
            raise ValueError("Transparent PNG trimming requires an alpha channel.")
        content_mask = image[..., 3] > PNG_ALPHA_TOLERANCE
    else:
        background_rgb = _background_rgb(background_color)
        content_mask = np.any(
            np.abs(image[..., :3] - background_rgb) > PNG_BACKGROUND_TOLERANCE,
            axis=2,
        )

    content_rows, content_cols = np.nonzero(content_mask)
    if content_rows.size == 0:
        return image

    padding_px = max(int(padding_px), 0)
    top = max(int(content_rows.min()) - padding_px, 0)
    bottom = min(int(content_rows.max()) + padding_px + 1, image.shape[0])
    left = max(int(content_cols.min()) - padding_px, 0)
    right = min(int(content_cols.max()) + padding_px + 1, image.shape[1])
    return image[top:bottom, left:right]

def run(viewer, args):

    # 1. Setup paths
    save_dir = cfg.resolve_directory_path(PRINT_DIRECTORY)
    
    # Check for help
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
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
        elif a.strip().startswith("-"):
            # Not a modifier, and not a name: it would be saved as "--flag.png".
            msg = Message(
                "Error: Unknown option '{option}'. Modifiers are written without dashes: "
                "transparent, full or svg.",
                option=a,
            )
            Command_Engine.command_failed(viewer, msg)
            print(f"\n{msg}")
            Command_Engine.show_status(viewer, msg)
            return
        else:
            final_args.append(a)
            
    # SVG Constraints
    if is_svg:
        if is_transparent or is_full:
            msg = Message("Error: 'SVG' export is not compatible with 'transparent' or 'full'.")
            Command_Engine.command_failed(viewer, msg)
            print(f"\n{msg}")
            Command_Engine.show_status(viewer, msg)
            return
            
        if len(args) > 2:
            msg = Message("Error: Maximum of 2 keywords allowed when using 'SVG' (e.g., 'print [filename] svg').")
            Command_Engine.command_failed(viewer, msg)
            print(f"\n{msg}")
            Command_Engine.show_status(viewer, msg)
            return

    args = final_args
        
    # 3. Determine filename
    ext = ".svg" if is_svg else ".png"
    image_format = ext[1:].upper()

    automatic_filename = len(args) == 0
    if not automatic_filename:
        try:
            filename = validate_output_basename("_".join(args))
        except ValueError as error:
            msg = Message("Error: {error}", error=error)
            Command_Engine.command_failed(viewer, msg)
            print(f"\n{msg}")
            Command_Engine.show_status(viewer, msg)
            return
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{cfg.SEQUENCE_SET}_{timestamp}"
        
    if not filename.lower().endswith(ext):
        filename += ext
    if automatic_filename:
        filename = _available_automatic_filename(save_dir, filename)
        
    filepath = os.path.join(save_dir, filename)
    
    # 4. A capture leaves the live view alone: each render hides the HUD and
    # moves the camera only between reading and restoring the live state,
    # with no event processed in between (see _render_capture).
    original_bgcolor = viewer.canvas.bgcolor
    overlays = _capture_overlays(viewer)
    events = None

    try:
        if is_svg:
            if not _export_svg(viewer, filepath):
                msg = Message("Error: No visible nodes to export.")
                Command_Engine.command_failed(viewer, msg)
                print(f"\n{msg}")
                Command_Engine.show_status(viewer, msg)
                return

        elif is_full:
            print("\nCalculating seamless tile grid (bypassing OpenGL edge-clipping)...")
            
            vis = viewer.visible_mask
            if not np.any(vis):
                msg = Message("Error: No visible nodes to export.")
                Command_Engine.command_failed(viewer, msg)
                print(f"\n{msg}")
                Command_Engine.show_status(viewer, msg)
                return

            # 1. Keep the aspect ratio locked to preserve rendering proportions
            camera = viewer.view.camera
            orig_real_rect = camera._real_rect if hasattr(camera, '_real_rect') else camera.rect
            events = _CaptureEvents(viewer, overlays)
            
            # 2. Get exact physical pixel resolution
            dummy_tile = _render_capture(viewer, is_transparent, overlays)
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
                    
                    # Render the tile from its own camera; the live one is back on return
                    tile_img = _render_capture(
                        viewer,
                        is_transparent,
                        overlays,
                        (cam_left, cam_bottom, orig_real_rect.width, orig_real_rect.height),
                    )
                    
                    # The Cookie Cutter: Snip off the unsafe clipped margins
                    cropped_tile = tile_img[margin_px : tile_px_h - margin_px, margin_px : tile_px_w - margin_px]
                    
                    # Paste the perfectly safe center block side-by-side
                    paste_x = c * keep_px_w
                    paste_y = r * keep_px_h
                    final_img[paste_y : paste_y + keep_px_h, paste_x : paste_x + keep_px_w, :] = cropped_tile

                    # Let the window paint and timers run; input stays queued
                    events.pause_if_due()
                    
            # 9. Crop to exact requested world bounds
            print("Stitching complete. Cropping to exact bounds...")
            target_px_w = int((world_right - world_left) / upp_x)
            target_px_h = int((world_top - world_bottom) / upp_y)
            
            # Clamp to be safe
            target_px_w = min(target_px_w, final_img.shape[1])
            target_px_h = min(target_px_h, final_img.shape[0])
            
            final_img = final_img[:target_px_h, :target_px_w]

        else:
            # Standard single-shot render
            visible_mask = getattr(viewer, 'visible_mask', None)
            if visible_mask is not None and not np.any(visible_mask):
                msg = Message("Error: No visible nodes to export.")
                Command_Engine.command_failed(viewer, msg)
                print(f"\n{msg}")
                Command_Engine.show_status(viewer, msg)
                return

            final_img = _render_capture(viewer, is_transparent, overlays)

        if not is_svg:
            original_height, original_width = final_img.shape[:2]
            final_img = _trim_png_margins(
                final_img,
                is_transparent,
                original_bgcolor,
            )
            trimmed_height, trimmed_width = final_img.shape[:2]
            if (trimmed_width, trimmed_height) != (original_width, original_height):
                print(
                    "Trimmed empty PNG margins: "
                    f"{original_width}x{original_height} -> "
                    f"{trimmed_width}x{trimmed_height} px "
                    f"({PNG_TRIM_PADDING_PX} px border)."
                )
            os.makedirs(save_dir, exist_ok=True)
            mpimg.imsave(filepath, final_img)
        
        msg_type = "SVG snapshot" if is_svg else "snapshot"
        if is_transparent and not is_svg: msg_type = "transparent " + msg_type
        if is_full and not is_svg: msg_type = "full stitched " + msg_type
            
        # The terminal and MCP clients get the full English report; the console line a short one.
        saved = f"Successfully saved {msg_type}: {filepath}"
        Command_Engine.command_artifact(viewer, filepath)
        print(f"\n{saved}")
        
        Command_Engine.show_status(viewer, Message("Saved {format}: {file}", format=image_format, file=filename))
        
        # Open the save folder in the system file explorer
        open_in_file_manager(save_dir)
        
    except Exception as e:
        interrupted = isinstance(e, _CaptureInterrupted)
        if not interrupted:
            import traceback
            traceback.print_exc()
        error_msg = Message("Failed to save {format}: {error}", format=image_format, error=e)
        Command_Engine.command_failed(viewer, error_msg)
        print(f"\n{error_msg}")
        Command_Engine.show_status(
            viewer,
            error_msg if interrupted else Message("Error saving {format}. Check console.", format=image_format),
        )
        return

    finally:
        if events is not None:
            events.close()
        viewer.canvas.update()
    Command_Engine.command_succeeded(viewer, saved)
