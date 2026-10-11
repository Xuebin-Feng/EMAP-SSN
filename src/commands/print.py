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
import uuid
from dataclasses import dataclass
import numpy as np
import matplotlib.image as mpimg
from matplotlib import colors as mcolors
from PySide6 import QtCore
from vispy import app
from vispy.scene.visuals import VisualNode
from vispy.scene.widgets import Widget
import EMAPSSN_Config as cfg
from commands.zoom import view_for_width
from utilities.Localization import JoinedMessage, Message
from utilities.Output_Names import validate_output_basename
from Viewer_Visual_State import edge_stages

PRINT_DIRECTORY = os.path.join("$analysis_result$", "Saved_Images")
# The words print reads as modifiers rather than as part of the file name.
PRINT_MODIFIERS = ("transparent", "full", "svg", "zoom")
# The modifiers as an error lists them: zoom is written with its N.
PRINT_MODIFIER_SYNTAX = "transparent, full, svg, zoom N"
# A full PNG drawn at a `zoom N` scale whose longest side is shorter than this
# is still saved, with a warning that a smaller N gives a larger image.
ZOOMED_IMAGE_MIN_SIDE_PX = 1000
# print full refuses, before drawing any tile, a PNG of more megapixels than
# this. 1000 is about 31,600 x 31,600 px, seven times an A0 poster at 300 dpi
# (140). Making the image takes about 4 bytes of memory per pixel, 7 with
# transparent (_RawMosaic): 4 GB, or 7 GB, at the limit. Raise it only as far
# as the computer's memory allows.
MAX_FULL_IMAGE_MEGAPIXELS = 1000
# _RawMosaic.finish converts this many pixels at a time, which bounds the
# float arrays it works in.
FINISH_BAND_PIXELS = 1 << 22
PNG_TRIM_PADDING_PX = 20
PNG_ALPHA_TOLERANCE = 1.0 / 255.0
PNG_BACKGROUND_TOLERANCE = 2.0 / 255.0
# Viewer HUD overlays drawn over the network, left out of every capture.
CAPTURE_HIDDEN_OVERLAYS = (
    'zoom_text', 'tooltip', 'hidden_text', 'console_bg',
    'console_text', 'background_job_status_text', 'selection_box',
)
# The SVG is written as it is made, this many edges or nodes at a time, so a
# large network never holds the whole document in memory.
SVG_CHUNK_SIZE = 100_000
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
      name of one word, not a path; .png (or .svg) is added when it is
      missing. A name ending in .svg saves an SVG, as the svg modifier does,
      and cannot be combined with transparent or full; a name ending in .png
      cannot be combined with svg; .png or .svg alone is not a name. A file
      of the same name is replaced, and the report says so; a name left out
      is made from the time and never replaces a file. Modifiers may stand
      anywhere among the words.
      PNG exports automatically trim empty margins while retaining a 20-pixel
      border around all rendered content. SVG view-box padding is unchanged.

    Modifiers (Can be combined, except for SVG):
      transparent : Removes the configured background color (PNG only).
      full        : Renders the network tile by tile off screen and stitches
                    the tiles into a massive, ultra-high-resolution PNG of
                    the entire network without OpenGL edge-clipping. The view
                    on screen stays put; keyboard and mouse input waits while
                    the tiles are drawn, and resizing the window cancels it.
      svg         : Reconstructs the visible network as a Scalable Vector Graphic 
                    for infinite zoom without pixelation (Not compatible with 
                    other modifiers). Node sizes and line widths follow the
                    current zoom, as on screen.

    Scale (with full or svg only, anywhere among the modifiers):
      zoom N      : Draws at the scale 'zoom N' would set, the view N scene
                    units wide on the current canvas, without moving the
                    view. N must be a positive width the view can draw;
                    the zoom command's 0.5 and whole-network limits don't
                    apply. A smaller N zooms in: a larger full PNG, and
                    smaller nodes and lines relative to the network in an
                    SVG. Node sizes stay in screen pixels. A full PNG whose
                    longest side is under 1000 px is saved with a warning.

    Background job: print takes its picture at once, as the network is when
    it runs: it draws the view, or every tile of a full PNG, and an SVG
    copies the nodes and edges shown. The file is then finished and written
    (a PNG trimmed and encoded, an SVG written out) by the background job
    queue that label and logo use, so the Viewer can be used meanwhile and
    later changes don't alter the file. The queue reports when it is saved
    and opens its folder.

    Size limit: a full PNG of more than MAX_FULL_IMAGE_MEGAPIXELS megapixels
    (1000 as shipped) is refused before any tile is drawn, with the smallest
    N that fits. The limit is set in src/commands/print.py.

    PNG captures leave out the hover colour of the node under the mouse and
    the halo of the clicked node; selection borders stay.

    Examples:
      print                               (Saves view as a timestamped PNG)
      print my_network                    (Saves view as my_network.png)
      print my_network transparent        (Saves as a transparent PNG)
      print my_network full transparent   (Stitches a massive transparent PNG)
      print my_network svg                (Saves view as a vector SVG file)
      print my_network.svg                (The same: a .svg name selects SVG)
      print my_network full zoom 500      (Whole network, 500 units per view width)
      print my_network svg zoom 500       (SVG sized as on a view 500 units wide)
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


def _available_automatic_filename(directory, filename, scheduler=None):
    """Add a numeric suffix when a generated timestamped name is taken.

    Two prints in the same second share a timestamp: the second one is saved
    as name_2.png rather than over the first, whether the first is written
    or still waiting in the scheduler's queue. A name typed by the user is
    not passed here, and is overwritten as before.
    """
    stem, suffix = os.path.splitext(filename)
    candidate = filename
    index = 2
    while True:
        path = os.path.join(directory, candidate)
        if not os.path.exists(path) and not (
            scheduler is not None and scheduler.is_output_path_reserved(path)
        ):
            return candidate
        candidate = f"{stem}_{index}{suffix}"
        index += 1


def _scene_units_per_pixel(viewer, view_width=None):
    """Return how many scene units one logical screen pixel spans in the view.

    Marker sizes and line widths are set in logical pixels: VisPy multiplies
    them by the display scale itself, so they read the same on a high-DPI
    screen. The SVG is written in scene coordinates, so every such size is
    multiplied by this scale. The camera maps its real rect (the requested
    one, widened to the view's aspect ratio) onto the view, which is in
    logical pixels too. The Viewer fixes the aspect at 1, so one scale holds
    for both axes. A view with no area keeps the sizes as they are, 1 to 1.

    view_width, the N of `print svg zoom N`, gives the scale of a view N
    scene units wide instead of the current one.
    """
    view = viewer.view
    camera = view.camera
    rect = camera._real_rect if hasattr(camera, '_real_rect') else camera.rect
    width = rect.width if view_width is None else view_width
    try:
        scale = abs(float(width)) / float(view.size[0])
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    return scale if math.isfinite(scale) and scale > 0.0 else 1.0


def _svg_color_attrs(rgba, is_stroke=False):
    """A colour as fill or stroke attributes, for strict SVG 1.1/Illustrator compatibility."""
    r, g, b, a = rgba
    prefix = "stroke" if is_stroke else "fill"
    color_val = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    return f'{prefix}="{color_val}" {prefix}-opacity="{a:.3f}"'


def _svg_points(count):
    return " ".join(["%.3f,%.3f"] * count)


def _svg_star(cx, cy, r, d):
    columns = []
    for j in range(10):
        angle = -math.pi / 2.0 + j * math.pi / 5.0
        rad = r if j % 2 == 0 else r * 0.4
        columns += [cx + rad * math.cos(angle), cy + rad * math.sin(angle)]
    return columns


def _svg_pentagon(cx, cy, r, d):
    columns = []
    for j in range(5):
        angle = -math.pi / 2.0 + j * 2.0 * math.pi / 5.0
        columns += [cx + r * math.cos(angle), cy + r * math.sin(angle)]
    return columns


def _svg_x(cx, cy, r, d):
    off = 0.707 * r
    return [cx - off, cy - off, cx + off, cy + off, cx - off, cy + off, cx + off, cy - off]


def _svg_cross_lines(cx, cy, r, d):
    w = r * 0.4
    return [cx-r, cy-w, cx-w, cy-r, cx+w, cy-w, cx+r, cy+w, cx+w, cy+r, cx-w, cy+w, cx-r]


# The element each canonical marker name is drawn as: a %-template whose last
# field is the node's attributes, and the coordinates that fill the others,
# worked out for many nodes at once from their centres (cx, cy), radii r and
# diameters d in scene units.
_SVG_NODE_ELEMENTS = {
    'circle': (
        '    <circle cx="%.3f" cy="%.3f" r="%.3f" %s />',
        lambda cx, cy, r, d: [cx, cy, r],
    ),
    'square': (
        '    <rect x="%.3f" y="%.3f" width="%.3f" height="%.3f" %s />',
        lambda cx, cy, r, d: [cx - r, cy - r, d, d],
    ),
    'triangle_up': (
        f'    <polygon points="{_svg_points(3)}" %s />',
        lambda cx, cy, r, d: [cx, cy - r, cx + 0.866 * r, cy + 0.5 * r, cx - 0.866 * r, cy + 0.5 * r],
    ),
    'triangle_down': (
        f'    <polygon points="{_svg_points(3)}" %s />',
        lambda cx, cy, r, d: [cx, cy + r, cx + 0.866 * r, cy - 0.5 * r, cx - 0.866 * r, cy - 0.5 * r],
    ),
    'diamond': (
        f'    <polygon points="{_svg_points(4)}" %s />',
        lambda cx, cy, r, d: [cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy],
    ),
    'star': (f'    <polygon points="{_svg_points(10)}" %s />', _svg_star),
    'cross': (
        '    <path d="M %.3f %.3f L %.3f %.3f M %.3f %.3f L %.3f %.3f" %s />',
        lambda cx, cy, r, d: [cx - r, cy, cx + r, cy, cx, cy - r, cx, cy + r],
    ),
    'x': ('    <path d="M %.3f %.3f L %.3f %.3f M %.3f %.3f L %.3f %.3f" %s />', _svg_x),
    'vbar': (
        '    <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" %s />',
        lambda cx, cy, r, d: [cx, cy - r, cx, cy + r],
    ),
    'hbar': (
        '    <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" %s />',
        lambda cx, cy, r, d: [cx - r, cy, cx + r, cy],
    ),
    'arrow': (
        f'    <polygon points="{_svg_points(3)}" %s />',
        lambda cx, cy, r, d: [cx + r, cy, cx - r * 0.5, cy - 0.866 * r, cx - r * 0.5, cy + 0.866 * r],
    ),
    'clobber': (f'    <polygon points="{_svg_points(5)}" %s />', _svg_pentagon),
    'cross_lines': (
        '    <path d="M %.3f %.3f H %.3f V %.3f H %.3f V %.3f H %.3f V %.3f H %.3f V %.3f H %.3f V %.3f H %.3f Z" %s />',
        _svg_cross_lines,
    ),
}
_SVG_NODE_KINDS = tuple(_SVG_NODE_ELEMENTS)


def _svg_node_kind(shape):
    """The element a canonical marker name is drawn as; anything else is a circle."""
    if shape in ['circle', 'disc', 'o', 'ring']:
        return 'circle'
    if shape in ['square', 's']:
        return 'square'
    if shape in ['triangle', 'triangle_up', '^']:
        return 'triangle_up'
    if shape in ['triangle_down', 'v']:
        return 'triangle_down'
    if shape in ['diamond', 'D']:
        return 'diamond'
    if shape in ['star', '*']:
        return 'star'
    if shape in ['cross', '+']:
        return 'cross'
    if shape == 'x':
        return 'x'
    if shape in ['vbar', '|']:
        return 'vbar'
    if shape in ['hbar', '-', '_']:
        return 'hbar'
    if shape in ['arrow', 'tailed_arrow', '->', '>']:
        return 'arrow'
    if shape in ['clobber', 'p']:
        return 'clobber'
    if shape in ['cross_lines', 'P', '++']:
        return 'cross_lines'
    return 'circle'


def _svg_node_codes(shapes):
    """Each node's element (an index into _SVG_NODE_KINDS) and whether it is stroke-only."""
    known = {}
    codes = []
    for shape in shapes:
        try:
            code = known[shape]
        except KeyError:
            code = known[shape] = (
                2 * _SVG_NODE_KINDS.index(_svg_node_kind(shape)) + (shape in _STROKE_ONLY_SHAPES)
            )
        except TypeError:
            code = 2 * _SVG_NODE_KINDS.index(_svg_node_kind(shape)) + (shape in _STROKE_ONLY_SHAPES)
        codes.append(code)
    codes = np.asarray(codes, dtype=np.int64)
    return codes >> 1, (codes & 1).astype(bool)


def _distinct_colors(colors):
    """Return (rows, colour_of): one row index per distinct colour, and each
    node's index into them.

    Colours are told apart bit for bit, so 0.0 and -0.0 stay apart, as the
    text written for them differs.
    """
    colors = np.asarray(colors)
    if colors.dtype.hasobject or colors.ndim != 2:
        everyone = np.arange(len(colors))
        return everyone, everyone
    colors = np.ascontiguousarray(colors)
    keys = colors.view(np.dtype((np.void, colors.dtype.itemsize * colors.shape[1]))).ravel()
    _, rows, colour_of = np.unique(keys, return_index=True, return_inverse=True)
    return rows, colour_of.ravel()


def _svg_edge_chunks(pos, edges, target_min_x, target_max_y, tail):
    """The edge lines, SVG_CHUNK_SIZE edges at a time, each line ending in a newline."""
    edges = np.asarray(edges)
    if edges.size == 0:
        return
    edges = edges.reshape(-1, 2)
    template = '    <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f"' + tail.replace('%', '%%') + '\n'
    for start in range(0, len(edges), SVG_CHUNK_SIZE):
        chunk = edges[start:start + SVG_CHUNK_SIZE]
        first = pos[chunk[:, 0]]
        second = pos[chunk[:, 1]]
        coordinates = np.column_stack((
            first[:, 0] - target_min_x, target_max_y - first[:, 1],
            second[:, 0] - target_min_x, target_max_y - second[:, 1],
        ))
        yield (template * len(chunk)) % tuple(coordinates.ravel().tolist())


def _svg_node_chunks(cx, cy, d, shapes, colors, boundary_stroke):
    """The node lines in drawing order, SVG_CHUNK_SIZE nodes at a time.

    A filled shape takes its colour as the fill and the node boundary as its
    outline; a stroke-only shape is drawn in its colour, with no fill and a
    stroke 0.4 of its radius wide.
    """
    r = d / 2.0
    kinds, stroke_only = _svg_node_codes(shapes)
    rows, colour_of = _distinct_colors(colors)
    fills = [_svg_color_attrs(colors[row]) + ' ' + boundary_stroke for row in rows]
    strokes = [_svg_color_attrs(colors[row], is_stroke=True) for row in rows]
    count = len(cx)
    for start in range(0, count, SVG_CHUNK_SIZE):
        stop = min(start + SVG_CHUNK_SIZE, count)
        lines = [None] * (stop - start)
        chunk_kinds = kinds[start:stop]
        for kind in np.unique(chunk_kinds).tolist():
            slots = np.flatnonzero(chunk_kinds == kind)
            members = slots + start
            template, coordinates = _SVG_NODE_ELEMENTS[_SVG_NODE_KINDS[kind]]
            values = np.column_stack(coordinates(cx[members], cy[members], r[members], d[members])).tolist()
            stroke_widths = (r[members] * 0.4).tolist()
            for slot, row, colour, stroked, stroke_width in zip(
                slots.tolist(), values, colour_of[members].tolist(),
                stroke_only[members].tolist(), stroke_widths,
            ):
                if stroked:
                    attrs = f'fill="none" {strokes[colour]} stroke-width="{stroke_width:.3f}"'
                else:
                    attrs = fills[colour]
                lines[slot] = template % (*row, attrs)
        yield '\n'.join(lines) + '\n'


@dataclass(frozen=True)
class _SvgSnapshot:
    """What an SVG of the view draws, copied from the viewer when print runs.

    It belongs to the background job that writes the SVG: the viewer may
    change while the job waits or writes, and the file shows the network as
    it was. pos, colors, sizes and shapes are the visible nodes in drawing
    order; all_positions holds every node's (x, y), which the edges index.
    """
    pos: np.ndarray
    colors: np.ndarray
    sizes: np.ndarray
    shapes: list
    all_positions: np.ndarray
    edges: np.ndarray
    unit: float
    background: np.ndarray
    edge_color: object
    edge_alpha: float
    edge_width: float
    boundary_color: object
    boundary_width: float


def _svg_snapshot(viewer, view_width=None):
    """Copy what an SVG of the view draws; None when no node is visible.

    Sizes set in screen pixels (node size and outline, edge width) are written
    in scene units at the current zoom, so the SVG keeps the proportions of
    the view on screen; with view_width, the N of `zoom N`, at the zoom of a
    view N scene units wide instead.
    """
    vis = viewer.visible_mask
    if not np.any(vis):
        return None

    # Nodes go in the order the screen draws them, the later covering the
    # earlier, so those the color command or a selection raised stay on top.
    # Indexing copies, so later changes to the viewer leave the snapshot alone.
    render_order = getattr(viewer, 'visible_node_render_order', None)
    visible = render_order() if callable(render_order) else vis

    # The edges on screen: the similarity threshold, visible endpoints and
    # UMAP selection, in a new array.
    active_edges, _ = edge_stages(viewer, cfg)
    return _SvgSnapshot(
        pos=viewer.pos[visible],
        colors=viewer.current_colors[visible],
        sizes=viewer.current_sizes[visible],
        shapes=[_canonical_shape(shape) for shape in viewer.current_shapes[visible]],
        all_positions=np.array(np.asarray(viewer.pos)[:, :2]),
        edges=active_edges,
        # Node sizes are screen pixels; the SVG is in scene units.
        unit=_scene_units_per_pixel(viewer, view_width),
        background=np.array(viewer.canvas.bgcolor.rgba),
        edge_color=getattr(cfg, 'EDGE_COLOR', '#000000'),
        edge_alpha=getattr(cfg, 'EDGE_ALPHA', 0.2),
        edge_width=getattr(cfg, 'EDGE_WIDTH', 0.5),
        boundary_color=getattr(cfg, 'NODE_BOUNDARY_COLOR', '#000000'),
        boundary_width=getattr(cfg, 'NODE_BOUNDARY_WIDTH', 0.5),
    )


def _write_svg(snapshot, filepath):
    """Write a snapshot as a structured, layered SVG for Adobe Illustrator compatibility."""
    print("Generating layered, editable SVG for Illustrator...")

    pos = snapshot.pos
    colors = snapshot.colors
    sizes = snapshot.sizes
    shapes = snapshot.shapes

    # 2. Calculate bounding box
    min_x, min_y = np.min(pos[:, :2], axis=0)
    max_x, max_y = np.max(pos[:, :2], axis=0)

    w_bounds = max_x - min_x
    h_bounds = max_y - min_y

    unit = snapshot.unit

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
    
    # 3. Write the SVG as it is made: the edges and then the nodes go out a
    # chunk at a time into a partial file, which replaces the target only once
    # it is complete, so a failure leaves any earlier file of that name alone.
    bg_color = snapshot.background
    bg_color_str = f"rgb({int(bg_color[0]*255)},{int(bg_color[1]*255)},{int(bg_color[2]*255)})"
    header = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.3f} {height:.3f}" width="{width:.3f}" height="{height:.3f}">',
        # 4. Background layer
        '  <!-- Background -->',
        f'  <rect width="{width:.3f}" height="{height:.3f}" fill="{bg_color_str}" fill-opacity="{bg_color[3]:.3f}" />',
    ]

    # 5. Edges layer: the edges on screen, in the configured edge color and alpha.
    edge_alpha = snapshot.edge_alpha
    edge_width = snapshot.edge_width * unit
    edge_rgba = mcolors.to_rgba(snapshot.edge_color)
    edge_stroke = _svg_color_attrs((*edge_rgba[:3], edge_alpha), is_stroke=True)
    boundary_rgba = mcolors.to_rgba(snapshot.boundary_color)
    boundary_width = snapshot.boundary_width * unit
    boundary_stroke = _svg_color_attrs(boundary_rgba, is_stroke=True) + f' stroke-width="{boundary_width:.3f}"'
    edge_tail = f' {edge_stroke} stroke-width="{edge_width:.3f}" />'

    # 6. Nodes layer. Y is flipped so Cartesian +Y goes upward, matching the
    # viewer's coordinates, and sizes in screen pixels become scene units.
    node_x = pos[:, 0] - target_min_x
    node_y = target_max_y - pos[:, 1]
    diameters = sizes * unit

    # 7. Write the file, making the save folder only now that there is a file for it
    save_dir = os.path.dirname(filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(filepath))[0]
    partial_path = os.path.join(save_dir, f".{stem}.{uuid.uuid4().hex}.partial.svg")
    try:
        with open(partial_path, 'x', encoding='utf-8') as handle:
            handle.write('\n'.join(header) + '\n')
            handle.write('  <!-- Edges -->\n  <g id="edges" name="Edges">\n')
            for text in _svg_edge_chunks(
                snapshot.all_positions, snapshot.edges, target_min_x, target_max_y, edge_tail,
            ):
                handle.write(text)
            handle.write('  </g>\n  <!-- Nodes -->\n  <g id="nodes" name="Nodes">\n')
            for text in _svg_node_chunks(node_x, node_y, diameters, shapes, colors, boundary_stroke):
                handle.write(text)
            handle.write('  </g>\n</svg>')
        os.replace(partial_path, filepath)
    except BaseException:
        try:
            os.remove(partial_path)
        except OSError:
            pass
        raise
    print(f"Successfully generated structured SVG at: {filepath}")


def _render_tile(viewer, is_transparent):
    """Render the current camera view offscreen, as the 8-bit pixels render() returns.

    Returns (rgba,), or for a transparent capture (on_black, on_white): the
    view rendered on black and on white. canvas.render() draws into a
    framebuffer object of its own, and the transparent pass hands its two
    backgrounds to render() rather than setting canvas.bgcolor: nothing
    reaches the window and no event runs.
    """
    canvas = viewer.canvas
    if is_transparent:
        return canvas.render(bgcolor='black'), canvas.render(bgcolor='white')
    return (canvas.render(),)


def _transparent_rgba(on_black, on_white):
    """The float RGBA of a picture rendered on black and on white (8-bit, RGB used).

    Where white shows through, alpha falls: it is one minus the difference
    of the two, averaged over the channels, and the colour is the render on
    black divided by it. Each pixel is worked out on its own.
    """
    img_black = on_black[..., :3].astype(np.float32) / 255.0
    img_white = on_white[..., :3].astype(np.float32) / 255.0

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


def _tile_rgba(renders):
    """The float RGBA of a tile's renders, as _render_tile gave them."""
    if len(renders) == 2:
        return _transparent_rgba(*renders)
    img = renders[0].astype(np.float32) / 255.0
    if len(img.shape) == 3 and img.shape[2] == 3:
        rgba = np.ones((img.shape[0], img.shape[1], 4), dtype=np.float32)
        rgba[..., :3] = img
        return rgba
    return img


def _capture_tile(viewer, is_transparent):
    """Render the current camera view offscreen and return float RGBA."""
    return _tile_rgba(_render_tile(viewer, is_transparent))


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


def _shows_transient_marks(viewer):
    """Whether the network shows a hover colour or a left-click halo."""
    if getattr(viewer, 'hovered_node_idx', None) is not None:
        return True
    left_clicked = getattr(viewer, '_left_click_node_indices', None)
    return callable(left_clicked) and len(left_clicked()) > 0


def _hide_transient_marks(viewer):
    """Take the hover colour and the left-click halo off the network for a PNG.

    They mark where the mouse is and what was last clicked, not the network,
    and the SVG never had them. Selection borders, set on purpose, stay. The
    Viewer only stops drawing them (see transient_marks_hidden in its
    update_nodes): its hover and click state is left as it is, so they come
    back exactly as they were, the hover colour too while the mouse is still
    over its node.

    Returns the function that puts them back, or None when none is shown.
    """
    update_nodes = getattr(viewer, 'update_nodes', None)
    if not callable(update_nodes) or not _shows_transient_marks(viewer):
        return None

    def restore():
        viewer.transient_marks_hidden = False
        update_nodes()

    viewer.transient_marks_hidden = True
    try:
        update_nodes()
    except Exception:
        restore()
        raise
    return restore


def _render_capture(viewer, is_transparent, overlays, rect=None, raw=False):
    """Capture one tile with the HUD hidden and the camera at rect, if given.

    It is the float RGBA of _capture_tile, or with raw the renders of
    _render_tile. The live camera and HUD visibility are read here and put
    back before returning, even on error, and no event is processed in
    between, so a paint only ever sees the live view.
    """
    camera = viewer.view.camera
    live_rect = camera.rect
    shown = [overlay.visible for overlay in overlays]
    try:
        for overlay in overlays:
            overlay.visible = False
        if rect is not None:
            camera.rect = rect
        return (_render_tile if raw else _capture_tile)(viewer, is_transparent)
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


def _write_png(filepath, image):
    """Write image as the PNG imsave writes, into a partial file that replaces filepath when complete.

    Encoding a full capture takes seconds to minutes; a failure on the way,
    or the Viewer closing, leaves any earlier file of that name as it was and
    no partial file behind.
    """
    directory = os.path.dirname(filepath)
    stem = os.path.splitext(os.path.basename(filepath))[0]
    partial_path = os.path.join(directory, f".{stem}.{uuid.uuid4().hex}.partial.png")
    try:
        mpimg.imsave(partial_path, image)
        os.replace(partial_path, filepath)
    except BaseException:
        try:
            os.remove(partial_path)
        except OSError:
            pass
        raise


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
    if is_transparent and image.shape[2] < 4:
        raise ValueError("Transparent PNG trimming requires an alpha channel.")

    background_rgb = None if is_transparent else _background_rgb(background_color)
    content_mask = _content_mask(image, is_transparent, background_rgb)
    box = _trim_box(content_mask.any(axis=1), content_mask.any(axis=0), padding_px)
    if box is None:
        return image
    top, bottom, left, right = box
    return image[top:bottom, left:right]


def _content_mask(image, is_transparent, background_rgb):
    """The pixels of a float RGBA image that are not background."""
    if is_transparent:
        return image[..., 3] > PNG_ALPHA_TOLERANCE
    return np.any(
        np.abs(image[..., :3] - background_rgb) > PNG_BACKGROUND_TOLERANCE,
        axis=2,
    )


def _trim_box(content_rows, content_cols, padding_px=PNG_TRIM_PADDING_PX):
    """Return (top, bottom, left, right), the content plus padding_px around it.

    content_rows and content_cols say which rows and columns hold content.
    Returns None when nothing does.
    """
    rows = np.flatnonzero(content_rows)
    cols = np.flatnonzero(content_cols)
    if rows.size == 0 or cols.size == 0:
        return None
    padding_px = max(int(padding_px), 0)
    top = max(int(rows[0]) - padding_px, 0)
    bottom = min(int(rows[-1]) + padding_px + 1, len(content_rows))
    left = max(int(cols[0]) - padding_px, 0)
    right = min(int(cols[-1]) + padding_px + 1, len(content_cols))
    return top, bottom, left, right


def _png_bytes(image):
    """A float RGBA image in 0..1 as the 8-bit RGBA that imsave writes for it.

    It is matplotlib's own conversion: a pixel with any NaN channel becomes
    transparent black, a value outside 0..1 is an error, and the rest are
    scaled by 255 and truncated.
    """
    nans = np.isnan(image)
    if np.any(nans):
        image = image.copy()
        image[np.any(nans, axis=2), :] = 0
    if image.size and (image.max() > 1 or image.min() < 0):
        raise ValueError("Floating point image RGB values must be in the [0,1] range")
    return (image * 255).astype(np.uint8)


def _compact(image, box):
    """image[top:bottom, left:right] moved to the start of image's own memory, C-contiguous.

    imsave needs a C-contiguous array and would copy a cut-out, which at a
    gigapixel is gigabytes. Each row moves up or left within the array,
    never onto a row still to be moved; image is the trimmed picture's
    memory afterwards.
    """
    top, bottom, left, right = box
    height, width, channels = image.shape
    if left == 0 and right == width:
        return image[top:bottom]
    flat = image.reshape(-1)
    row = (right - left) * channels
    stride = width * channels
    for index in range(bottom - top):
        start = (top + index) * stride + left * channels
        flat[index * row:(index + 1) * row] = flat[start:start + row]
    return flat[:(bottom - top) * row].reshape(bottom - top, right - left, channels)


class _RawMosaic:
    """A full capture's tiles, as rendered, finished into the PNG later.

    Drawing needs the Viewer's OpenGL context, so print renders the tiles on
    the GUI thread and pastes their 8-bit renders here, cut to the picture's
    size; finish() makes the PNG of them in the background job, in place. A
    plain capture keeps each render's RGBA, which is already the PNG's; a
    transparent one keeps its render on black, in the RGBA array the PNG's
    pixels then replace, and its render on white (RGB). That makes 4 bytes a
    pixel, or 7 with transparent.
    """

    def __init__(self, height, width, is_transparent):
        self.image = np.zeros((height, width, 4), dtype=np.uint8)
        self.on_white = np.zeros((height, width, 3), dtype=np.uint8) if is_transparent else None
        self.is_transparent = is_transparent

    @property
    def shape(self):
        """The picture's (height, width) before the trim."""
        return self.image.shape[:2]

    def paste(self, renders, top, left):
        """Put a tile's renders, as _render_tile gave them, with its top left corner at (top, left).

        What reaches past the picture's right or bottom edge is left out.
        """
        first = renders[0]
        height = min(first.shape[0], self.image.shape[0] - top)
        width = min(first.shape[1], self.image.shape[1] - left)
        if height <= 0 or width <= 0:
            return
        target = self.image[top:top + height, left:left + width]
        channels = min(first.shape[2], 4)
        target[..., :channels] = first[:height, :width, :channels]
        if channels < 4:
            target[..., 3] = 255
        if self.on_white is not None:
            self.on_white[top:top + height, left:left + width] = renders[1][:height, :width, :3]

    def finish(self, background_color, padding_px=PNG_TRIM_PADDING_PX):
        """Turn the renders into the PNG's 8-bit RGBA and trim its margins, in place.

        The pixels and the margins are those _tile_rgba, _png_bytes and
        _trim_png_margins give for the whole float picture: each step works
        pixel by pixel, or row and column by row and column, so taking a band
        of rows at a time changes nothing. Returns the trimmed picture, a
        C-contiguous array in this mosaic's memory, which imsave writes as it
        is.
        """
        height, width = self.image.shape[:2]
        background_rgb = None if self.is_transparent else _background_rgb(background_color)
        content_rows = np.zeros(height, dtype=bool)
        content_cols = np.zeros(width, dtype=bool)
        band = max(1, FINISH_BAND_PIXELS // max(width, 1))
        for top in range(0, height, band):
            rows = slice(top, min(top + band, height))
            if self.on_white is None:
                # x / 255 * 255 in float32 truncates back to x for every byte,
                # so the PNG's pixels are the render's own, left as they are.
                picture = self.image[rows, :, :3].astype(np.float32) / 255.0
            else:
                picture = _transparent_rgba(self.image[rows], self.on_white[rows])
                self.image[rows] = _png_bytes(picture)
            content = _content_mask(picture, self.is_transparent, background_rgb)
            content_rows[rows] = content.any(axis=1)
            content_cols |= content.any(axis=0)
        self.on_white = None
        box = _trim_box(content_rows, content_cols, padding_px)
        if box is None:
            return self.image
        return _compact(self.image, box)


def _round_up(value):
    """value rounded up to three significant digits, as text."""
    exponent = math.floor(math.log10(value)) - 2
    step = 10.0 ** exponent
    return f"{math.ceil(value / step) * step:.{max(0, -exponent)}f}"


def full_image_size_refusal(width, height, view_width):
    """The Message refusing a full PNG of width x height px, or None within the limit.

    view_width is the scene width the image is drawn at, the N of `zoom N` or
    the current view's width. The pixel count falls with its square, which
    gives the smallest N that fits; 1% more allows for rounding.
    """
    megapixels = width * height / 1e6
    if megapixels <= MAX_FULL_IMAGE_MEGAPIXELS:
        return None
    smallest = view_width * math.sqrt(megapixels / MAX_FULL_IMAGE_MEGAPIXELS) * 1.01
    return Message(
        "Error: The full image would be {width}×{height} px ({megapixels} megapixels), "
        "more than the limit of {limit}. Use {syntax} with N of at least {n}, or zoom out. "
        "The limit is {setting} in {file}.",
        width=width, height=height, megapixels=math.ceil(megapixels),
        limit=MAX_FULL_IMAGE_MEGAPIXELS, syntax="print full zoom N", n=_round_up(smallest),
        setting="MAX_FULL_IMAGE_MEGAPIXELS", file="src/commands/print.py",
    )


def _refuse(viewer, msg):
    """Report msg, an error found before anything is drawn or written."""
    Command_Engine.command_failed(viewer, msg)
    print(f"\n{msg}")
    Command_Engine.show_status(viewer, msg)


@dataclass(frozen=True)
class _PrintJob:
    """One print's picture, as run() took it, and where its file goes.

    picture is an _SvgSnapshot ('svg'), a full capture's _RawMosaic ('full')
    or a view's float RGBA ('png'). Nothing in it refers to the live viewer,
    so the background job can finish it while the viewer changes.
    """
    kind: str
    picture: object
    filepath: str
    filename: str
    save_dir: str
    allow_overwrite: bool
    is_transparent: bool = False
    background: object = None
    zoom_width: float = None


def _ready_to_write(job):
    """Make the save folder, and say whether a file of the name is there to replace.

    A name the user gave replaces such a file; a timestamped one never does.
    """
    os.makedirs(job.save_dir, exist_ok=True)
    replacing = os.path.exists(job.filepath)
    if replacing and not job.allow_overwrite:
        raise FileExistsError(Message("Output file already exists: {path}", path=job.filepath))
    return replacing


def _print_job(job):
    """Finish one print's picture and write its file: the background job's worker.

    It runs on the scheduler's thread: it trims and encodes a PNG, or writes
    an SVG, from what run() took, and returns the report the scheduler shows.
    """
    size_warning = None
    if job.kind == "svg":
        image_format, description = "SVG", "SVG snapshot"
        replaced = _ready_to_write(job)
        _write_svg(job.picture, job.filepath)
    else:
        image_format, description = "PNG", "snapshot"
        if job.is_transparent:
            description = "transparent " + description
        if job.kind == "full":
            description = "full stitched " + description
            original_height, original_width = job.picture.shape
            image = job.picture.finish(job.background)
        else:
            original_height, original_width = job.picture.shape[:2]
            image = _trim_png_margins(job.picture, job.is_transparent, job.background)
        trimmed_height, trimmed_width = image.shape[:2]
        if (trimmed_width, trimmed_height) != (original_width, original_height):
            print(
                "Trimmed empty PNG margins: "
                f"{original_width}x{original_height} -> "
                f"{trimmed_width}x{trimmed_height} px "
                f"({PNG_TRIM_PADDING_PX} px border)."
            )
        replaced = _ready_to_write(job)
        _write_png(job.filepath, image)
        # A scale the user chose can make a small picture; it is saved all the same.
        if job.zoom_width is not None and max(trimmed_width, trimmed_height) < ZOOMED_IMAGE_MIN_SIDE_PX:
            size_warning = Message(
                "Warning: the image is only {width}×{height} px. N in {syntax} is the view "
                "width, so a smaller N zooms in and gives a larger image.",
                width=trimmed_width, height=trimmed_height, syntax="zoom N",
            )

    # The terminal gets the full English report; the scheduler shows the short one.
    print(f"\nSuccessfully saved {description}: {job.filepath}")
    status = Message("Saved {format}: {file}", format=image_format, file=job.filename)
    notes = []
    if replaced:
        notes.append(Message("It replaced an existing file of that name."))
    if size_warning is not None:
        notes.append(size_warning)
    for note in notes:
        print(note)
    return {
        "message": JoinedMessage([status, *notes]) if notes else status,
        # The scheduler opens the save folder in the system file explorer.
        "reveal_directory": job.save_dir,
    }


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
    zoom_text = None
    name = None

    words = iter(args)
    for a in words:
        if a.lower() == "transparent":
            is_transparent = True
        elif a.lower() == "full":
            is_full = True
        elif a.lower() == "svg":
            is_svg = True
        elif a.lower() == "zoom":
            # The next word is N, whatever it holds, so it never joins the
            # file name; a name such as zoom.png is not this keyword.
            width = next(words, None)
            if width is None or width.lower() in PRINT_MODIFIERS:
                _refuse(viewer, Message(
                    "Error: {syntax} needs a view width, as in {example}.",
                    syntax="zoom", example="print full zoom 500",
                ))
                return
            if zoom_text is not None:
                _refuse(viewer, Message("Error: {syntax} can be given only once.", syntax="zoom N"))
                return
            zoom_text = width
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
        elif name is not None:
            # The file name is one word, so a second one is a mistyped modifier
            # (print fig1 ful) rather than the rest of the name.
            _refuse(viewer, Message(
                "Error: Unrecognized print argument '{argument}'. A file name is one word; "
                "the modifiers are {modifiers}.",
                argument=a, modifiers=PRINT_MODIFIER_SYNTAX,
            ))
            return
        else:
            name = a

    # A name ending in .svg selects SVG as the svg modifier does; one ending
    # in .png cannot be an SVG's name; one that is only .png or .svg is no name.
    if name is not None:
        stem, extension = name.strip()[:-4], name.strip()[-4:]
        if not stem and extension.lower() in (".png", ".svg"):
            # A name that is only an extension, as logo refuses one too.
            _refuse(viewer, Message(
                "Error: {error}",
                error=Message("Filename '{file}' needs a name before its extension.", file=name.strip()),
            ))
            return
        if extension.lower() == ".svg":
            is_svg = True
        elif is_svg and extension.lower() == ".png":
            _refuse(viewer, Message(
                "Error: '{file}' ends in {extension}, but {modifier} saves an SVG file. "
                "Use '{suggestion}' or leave the extension off.",
                file=name, extension=extension, modifier="svg", suggestion=stem + ".svg",
            ))
            return

    # zoom N sets the scale of the whole network, which a capture of the
    # view on screen does not draw.
    if zoom_text is not None and not (is_full or is_svg):
        _refuse(viewer, Message(
            "Error: {syntax} works only with {full} or {svg}.",
            syntax="zoom N", full="full", svg="svg",
        ))
        return

    # SVG Constraints
    if is_svg:
        if is_transparent or is_full:
            msg = Message("Error: 'SVG' export is not compatible with 'transparent' or 'full'.")
            Command_Engine.command_failed(viewer, msg)
            print(f"\n{msg}")
            Command_Engine.show_status(viewer, msg)
            return

    # N is checked as `zoom N` checks it, against the current canvas and view.
    zoom_width = None
    if zoom_text is not None:
        try:
            zoom_width = view_for_width(viewer, zoom_text)[2]
        except ValueError as error:
            _refuse(viewer, error.args[0])
            return

    # 3. Determine filename
    ext = ".svg" if is_svg else ".png"
    image_format = ext[1:].upper()

    automatic_filename = name is None
    if not automatic_filename:
        try:
            filename = validate_output_basename(name)
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

    # The file is finished and written by the queue label and logo use.
    scheduler = getattr(viewer, "background_job_scheduler", None)
    if scheduler is None:
        _refuse(viewer, Message(
            "Error: {error}", error=Message("The background job scheduler is unavailable."),
        ))
        return
    if automatic_filename:
        filename = _available_automatic_filename(save_dir, filename, scheduler)
    filepath = os.path.join(save_dir, filename)
    # A name the user typed replaces a file of that name, but not one a
    # queued job is still to write.
    if not automatic_filename and scheduler.is_output_path_reserved(filepath):
        _refuse(viewer, Message("Error: {error}", error=Message(
            "Output file is already reserved by a background job: {path}", path=filepath,
        )))
        return

    # 4. The picture is taken now, as the network is: drawing needs the
    # Viewer's OpenGL context, which the GUI thread holds. A capture leaves
    # the live view alone: each render hides the HUD and moves the camera
    # only between reading and restoring the live state, with no event
    # processed in between (see _render_capture). A PNG also leaves out the
    # hover colour and click halo, put back once it is drawn.
    background = viewer.canvas.bgcolor
    if hasattr(background, 'rgba'):
        background = tuple(float(value) for value in background.rgba)
    overlays = _capture_overlays(viewer)
    events = None
    restore_marks = None

    try:
        if is_svg:
            kind = "svg"
            picture = _svg_snapshot(viewer, view_width=zoom_width)
            if picture is None:
                msg = Message("Error: No visible nodes to export.")
                Command_Engine.command_failed(viewer, msg)
                print(f"\n{msg}")
                Command_Engine.show_status(viewer, msg)
                return

        elif is_full:
            kind = "full"
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
            # Each tile shows the scene area the view shows now, or with zoom N
            # the view `zoom N` sets: N units wide, in the view's own shape.
            tile_world_w = orig_real_rect.width
            tile_world_h = orig_real_rect.height
            if zoom_width is not None:
                tile_world_h = tile_world_h * zoom_width / tile_world_w
                tile_world_w = zoom_width
            restore_marks = _hide_transient_marks(viewer)
            events = _CaptureEvents(viewer, overlays)
            
            # 2. Get exact physical pixel resolution
            dummy_tile = _render_capture(viewer, is_transparent, overlays, raw=True)
            tile_px_h, tile_px_w = dummy_tile[0].shape[:2]
            
            # 3. Define the trash margin (Throw away outer 15% of pixels)
            margin_px = int(min(tile_px_w, tile_px_h) * 0.15)
            
            # Calculate the dimensions of the "safe" middle area we will actually keep
            keep_px_w = tile_px_w - (2 * margin_px)
            keep_px_h = tile_px_h - (2 * margin_px)
            
            # 4. Calculate exact Units-Per-Pixel (UPP) using the actual visible rect bounds
            upp_x = tile_world_w / tile_px_w
            upp_y = tile_world_h / tile_px_h
            
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

            # 7. Initialize giant mosaic canvas, cut to the exact requested
            # world bounds (clamped to the tiles): the tiles' parts beyond
            # them are never kept
            canvas_w = n_cols * keep_px_w
            canvas_h = n_rows * keep_px_h
            target_px_w = int((world_right - world_left) / upp_x)
            target_px_h = int((world_top - world_bottom) / upp_y)
            refusal = full_image_size_refusal(target_px_w, target_px_h, tile_world_w)
            if refusal is not None:
                _refuse(viewer, refusal)
                return
            picture = _RawMosaic(
                min(target_px_h, canvas_h), min(target_px_w, canvas_w), is_transparent,
            )

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
                    cam_bottom = cam_top - tile_world_h
                    
                    # Render the tile from its own camera; the live one is back on return
                    renders = _render_capture(
                        viewer,
                        is_transparent,
                        overlays,
                        (cam_left, cam_bottom, tile_world_w, tile_world_h),
                        raw=True,
                    )

                    # The Cookie Cutter: Snip off the unsafe clipped margins
                    cropped = [
                        render[margin_px : tile_px_h - margin_px, margin_px : tile_px_w - margin_px]
                        for render in renders
                    ]

                    # Paste the perfectly safe center block side-by-side
                    paste_x = c * keep_px_w
                    paste_y = r * keep_px_h
                    picture.paste(cropped, paste_y, paste_x)

                    # Let the window paint and timers run; input stays queued
                    events.pause_if_due()

            print("Stitching complete. The PNG is finished in the background job.")

        else:
            # Standard single-shot render
            kind = "png"
            visible_mask = getattr(viewer, 'visible_mask', None)
            if visible_mask is not None and not np.any(visible_mask):
                msg = Message("Error: No visible nodes to export.")
                Command_Engine.command_failed(viewer, msg)
                print(f"\n{msg}")
                Command_Engine.show_status(viewer, msg)
                return

            restore_marks = _hide_transient_marks(viewer)
            picture = _render_capture(viewer, is_transparent, overlays)

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
        # The picture is taken, or not: the live view gets its hover colour
        # and click halo back, before any job finishes the file.
        if events is not None:
            events.close()
        if restore_marks is not None:
            restore_marks()
        viewer.canvas.update()

    # 5. The background job trims and encodes the PNG, or writes the SVG; the
    # scheduler reports it, with the file, and opens the save folder.
    job = _PrintJob(
        kind=kind, picture=picture, filepath=filepath, filename=filename, save_dir=save_dir,
        allow_overwrite=not automatic_filename, is_transparent=is_transparent,
        background=background, zoom_width=zoom_width,
    )
    try:
        scheduler.enqueue(
            command_name="print",
            description=f"print -> {filename}",
            payload=job,
            worker=_print_job,
            output_path=filepath,
            allow_overwrite=not automatic_filename,
        )
    except (FileExistsError, RuntimeError) as error:
        _refuse(viewer, Message("Failed to save {format}: {error}", format=image_format, error=error))
        return
    Command_Engine.command_succeeded(viewer, f"Queued print for {filepath}; waiting for the background job.")
