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
import sys
import json


def _configure_headless_platform(argv=None, environment=None):
    """Route VisPy to Qt's offscreen platform before Qt/OpenGL is imported."""
    argv = sys.argv if argv is None else argv
    environment = os.environ if environment is None else environment
    if "--headless" in argv or environment.get("SSN_VIEWER_HEADLESS", "").lower() in {"1", "true", "yes"}:
        environment["QT_QPA_PLATFORM"] = "offscreen"
        return True
    return False


_HEADLESS_LAUNCH = _configure_headless_platform()


def _configure_hidden_platform(environment=None, platform_name=None):
    """A hidden Viewer (SSN_VIEWER_WINDOW=hidden) keeps Qt's native platform, or falls back to offscreen.

    Before Qt/OpenGL is imported: see utilities.Viewer_Display.
    """
    from utilities.Viewer_Display import apply_hidden_environment
    return apply_hidden_environment(environment, platform_name)


_configure_hidden_platform()


def _parse_viewer_arguments(argv=None):
    """Parse command-line arguments for EMAPSSN_Viewer.py."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="EMAPSSN_Viewer.py",
        description="Interactive and headless 2D sequence similarity network viewer.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run viewer in headless mode without a visible desktop window.",
    )
    parser.add_argument(
        "--settings",
        dest="settings_file",
        help="Path to a JSON file containing viewer settings.",
    )
    parser.add_argument(
        "--settings-json",
        dest="settings_json",
        help="Direct JSON string containing viewer settings.",
    )
    parser.add_argument("--delete-settings", action="store_true", help="Consume an internally generated settings snapshot.")
    return parser.parse_args(argv)


# Resolve explicit launch settings before importing Qt or configuration consumers.
_startup_settings = None
_startup_args = None
if __name__ == "__main__":
    # Before any output: an MCP client gives the Viewer a pipe (Viewer_Terminal.py) or a log file, and a
    # Config window gives it a hidden console. A headless Viewer, or --console, only gets UTF-8 streams
    # (utilities.Console_Capture); every other keeps what it prints for the Console window, and still
    # passes it on to a pipe or file that is read.
    from utilities.Console_Capture import start as start_console_capture
    start_console_capture("viewer", headless=_HEADLESS_LAUNCH)
    from utilities import Startup_Status
    Startup_Status.report(Startup_Status.CHECKING_SETTINGS, 0.1)
    from desktop.Viewer_State import read_viewer_settings, resolve_viewer_document
    _startup_args = _parse_viewer_arguments()
    print("Validating Viewer settings and inputs...", flush=True)
    _settings_path = _startup_args.settings_file or os.environ.get("SSN_VIEWER_SETTINGS_PATH")
    _document = json.loads(_startup_args.settings_json) if _startup_args.settings_json else None
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        _startup_settings = resolve_viewer_document(
            read_viewer_settings(settings_document=_document, settings_path=_settings_path,
                                 project_root=_project_root), _project_root
        )
    finally:
        # Consumed even when rejected, so a launch that fails here leaves no snapshot behind.
        if _startup_args.delete_settings and _settings_path:
            try:
                os.unlink(_settings_path)
            except FileNotFoundError:
                pass
    os.environ["SSN_VIEWER_EXPLICIT_SETTINGS"] = "1"
    os.environ.pop("SSN_VIEWER_SETTINGS_PATH", None)


def _configure_linux_vispy_platform(
    environment=None,
    platform_name=sys.platform,
):
    """Route VisPy through XWayland before Qt/OpenGL is imported.

    On affected Linux systems Qt's native Wayland plugin creates an OpenGL ES
    context while VisPy 0.16 compiles desktop GLSL 1.20 shaders.  XCB creates
    the compatible desktop OpenGL context.  Preserve non-window platforms
    such as ``offscreen`` for tests and headless use.
    """
    environment = os.environ if environment is None else environment
    if not str(platform_name).startswith('linux'):
        return False

    selected_platform = environment.get("QT_QPA_PLATFORM", "").lower()
    wayland_session = environment.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    native_wayland_selected = selected_platform.startswith("wayland")
    if native_wayland_selected or (wayland_session and not selected_platform):
        environment["QT_QPA_PLATFORM"] = "xcb"
        return True
    return False


# This must run before torch, VisPy, or PySide6 can initialize Qt/OpenGL.
_configure_linux_vispy_platform()

import unicodedata  # Pre-load to prevent Windows DLL search path conflicts with Qt/OpenGL
from utilities import Startup_Status
Startup_Status.report(Startup_Status.LOADING_PYTORCH, 0.25)
try:
    import torch  # Pre-load to prevent DLL initialization conflicts between PyTorch and PySide6/OpenGL
except Exception:
    pass
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"

# Ensure src/ (the directory containing all project modules) is on sys.path.
# This is needed when the script is launched as a subprocess (e.g. from EMAPSSN_Config.py).
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import h5py
import numpy as np
import importlib
from collections import deque
import ctypes
import gc
import itertools
import queue
from vispy import scene, app
from vispy.color import Color
from PySide6 import QtWidgets, QtCore, QtGui
import shiboken6

import EMAPSSN_Config as cfg
if _startup_settings is not None:
    cfg.__dict__.update(_startup_settings)
import Command_Engine
import Cache_Manifest as cache_manifest
from Background_Job_Scheduler import BackgroundJobScheduler
from Command_Queue import CommandQueue
from utilities.Sequence_Utils import (
    load_sanitized_fasta,
)
from utilities.Localization import JoinedMessage, Message, display_text
from desktop.Desktop_App import (
    APP_USER_MODEL_IDS,
    APPLICATION_VERSION,
    CONFIG_THEME_SETTING,
    UI_QSS_FONT_STACK,
    VIEWER_DESKTOP_FILE_NAME,
    VIEWER_DISPLAY_NAME,
    VISPY_FALLBACK_FACE,
    configure_linux_qt_desktop_identity,
    configure_qt_application_fonts,
    configured_theme_color,
    apply_studio_theme,
    fit_buttons_to_text,
    install_translations,
    installed_language,
    register_vispy_application_fonts,
    set_app_user_model_id,
    signal_launcher_ready,
    startup_language,
    translate,
    vispy_language_face,
    vispy_points_at_reference_dpi,
    vispy_points_for_logical_pixels,
)
from desktop.Studio_Theme import TOKENS, set_role, studio_icon
from desktop.Viewer_Zoom import SmoothWheelZoom
from desktop.Viewer_Widgets import (
    ConsoleButton, ControlsButton, SIDEBAR_INSET, SidebarButton, SidebarPanel, ThresholdSpinBox, ZoomBox,
)
from desktop.Viewer_State import (
    fewer_edges_while_moving,
    knn_topology_layout,
    low_resource_animations,
    low_resource_drawing,
    low_resource_webserver,
    prepare_network,
    resolve_selected_cache,
)
from desktop.Viewer_Inspection import ViewerInspectionService
from desktop.Viewer_Window import HEADLESS, HIDDEN, NORMAL, ViewerWindow
from desktop.Viewer_Edges import ViewerEdges
from desktop.Viewer_Motion import MotionDetail
from desktop.Viewer_Fps import FPS_GAP, FPS_HEIGHT, FpsOverlay
from desktop.Viewer_Jobs import BADGE_GAP, JobsBadge, ViewerJobs, job_lines
from desktop.Viewer_Markers import HoverMarkers, MarkerPicker, symbol_codes_of, uniform_symbol
from web_ui.Browser_Page import open_browser_page


def _load_selected_fasta_records(fasta_path):
    """Load the same canonical FASTA records used by embedding generation."""
    headers, sequences, _ = load_sanitized_fasta(fasta_path)
    return list(zip(headers, sequences))


def _build_sequence_lookup(records):
    """Index canonical FASTA records by both full header and first token."""
    sequence_lookup = {}
    for header, sequence in records:
        sequence_lookup[header] = sequence
        header_parts = header.split()
        if header_parts:
            sequence_lookup[header_parts[0]] = sequence
    return sequence_lookup


def _wrap_console_text_for_display(text, max_width, measure_width):
    """Wrap one logical console line without removing or changing its text."""
    if not text or max_width <= 0 or measure_width(text) <= max_width:
        return text

    lines = []
    remaining = text
    while remaining:
        if measure_width(remaining) <= max_width:
            lines.append(remaining)
            break

        # Find the longest prefix that fits. Glyph advances are monotonic, so a
        # binary search avoids repeatedly measuring every possible substring.
        low, high = 1, len(remaining)
        while low < high:
            midpoint = (low + high + 1) // 2
            if measure_width(remaining[:midpoint]) <= max_width:
                low = midpoint
            else:
                high = midpoint - 1

        split_at = max(1, low)

        # Prefer a natural whitespace boundary, while retaining that whitespace
        # so removing the visual newlines reconstructs the exact logical text.
        whitespace_break = -1
        for index in range(split_at - 1, 0, -1):
            if remaining[index].isspace():
                whitespace_break = index + 1
                break
        if whitespace_break > 0:
            split_at = whitespace_break

        lines.append(remaining[:split_at])
        remaining = remaining[split_at:]

    return "\n".join(lines)


# How long the sidebar takes to slide fully open or closed, in ms; 0 switches it at once.
SIDEBAR_SLIDE_MS = 220
# The command line's box: a light tint of the theme colour, a touch see-through over the network.
COMMAND_BOX_OPACITY = 0.95
# Between a command being typed and what a click did, on the command line.
CLICK_STATUS_SEPARATOR = "  ·  "
# Between the last button of the slider row (the Console button, after the
# controls' info button) and the threshold's value, in px: wide enough that
# the button doesn't read as part of the number.
SLIDER_BUTTONS_GAP = 16
# The slider row along the canvas's bottom, in px: the info and Console
# buttons and the threshold slider from the left, the Zoom box at the right,
# level with it.
SLIDER_ROW_HEIGHT = 45
SLIDER_ROW_BOTTOM = 15
SLIDER_ROW_LEFT = 20
SLIDER_BOX_GAP = 20  # between the slider's end and the Zoom box
# The threshold slider gives up room to the buttons and the boxes, down to
# this width, in px; the window is kept wide enough that it never gets less.
SLIDER_MIN_WIDTH = 60
# The info card over the canvas stays this far (px) from its left, right and
# bottom edges, and starts this far down: clear of the command line and the
# background-job status text under it.
SESSION_CARD_MARGIN = 8
SESSION_CARD_TOP = 96


def _slider_row_centre(canvas_height):
    """The y, in canvas px, of the middle of the slider row."""
    return canvas_height - SLIDER_ROW_BOTTOM - SLIDER_ROW_HEIGHT / 2.0


def _sidebar_room(viewer):
    """The canvas width the Viewer's sidebar covers: its width when open, none
    when closed, and the width shown so far while it slides
    (MainViewer.set_sidebar_visible)."""
    sliding = getattr(viewer, '_sidebar_slide_width', None)
    if sliding is not None:
        return sliding
    panel = getattr(viewer, 'right_panel', None)
    if panel is not None and panel.isVisible():
        return getattr(viewer, '_panel_w', 120)
    return 0


def _vispy_text_width_pixels(text_visual, text):
    """Measure one rendered row in logical pixels using VisPy's own glyph metrics."""
    if not text or not hasattr(text_visual, '_font'):
        return 0.0

    font = text_visual._font
    n_pix = (text_visual.font_size / 72.0) * text_visual.transforms.dpi
    ratio = 1.0 / getattr(font, 'ratio', 4.0)
    width_val = 0.0
    previous = None
    for char in text:
        glyph = font[char]
        kerning = glyph['kerning'].get(previous, 0.0) * ratio
        width_val += glyph['advance'] * ratio + kerning
        previous = char
    return (width_val / 64.0) * n_pix


def _vispy_text_line_height_pixels(text_visual):
    """Return VisPy's rendered multiline advance in logical pixels."""
    n_pix = (text_visual.font_size / 72.0) * text_visual.transforms.dpi
    line_height = getattr(text_visual, '_line_height', 1.2)
    font = getattr(text_visual, '_font', None)
    if font is None:
        return n_pix * line_height

    try:
        ratio = 1.0 / font.ratio
        slop = font.slop
        ascender = 0.0
        descender = 0.0
        for char in 'ÅÉÑŐjgpqy':
            glyph = font[char]
            y0 = glyph['offset'][1] * ratio + slop
            y1 = y0 - glyph['size'][1]
            ascender = max(ascender, y0 - slop)
            descender = min(descender, y1 + slop)
        glyph_height = ascender - descender
        lowres_size = float(font._lowres_size)
        if glyph_height > 0.0 and lowres_size > 0.0:
            return (glyph_height / lowres_size) * n_pix * line_height
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        pass

    return n_pix * line_height


def _apply_safe_rectangle_geometry(rectangle, center, width, height, radius):
    """Apply rounded-rectangle geometry without exposing invalid interim state."""
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    radius = min(
        max(float(radius), 0.0),
        width / 2.0,
        height / 2.0,
    )

    # VisPy regenerates and validates vertices after every property assignment.
    # Resetting the radius first also repairs rectangles left in an invalid
    # partially-mutated state by an earlier failed width/height assignment.
    rectangle.radius = 0.0
    rectangle.width = width
    rectangle.height = height
    rectangle.center = (float(center[0]), float(center[1]))
    rectangle.radius = radius

    return width, height, radius


def _build_adjacency_index(edges, node_count):
    """Index an undirected edge list as CSR offsets and neighbour ids.

    Neighbours of node ``i`` are ``indices[indptr[i]:indptr[i + 1]]``, so a
    neighbour query costs the summed degree of the queried nodes instead of a
    scan of every edge. Duplicate and out-of-range pairs are dropped, and each
    node's neighbours come back ascending and unique.
    """
    from scipy import sparse

    node_count = max(int(node_count), 0)
    pairs = np.asarray(
        edges if edges is not None else (), dtype=np.int64
    ).reshape(-1, 2)
    if node_count == 0 or len(pairs) == 0:
        return (
            np.zeros(node_count + 1, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )

    in_range = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 0] < node_count)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 1] < node_count)
    )
    pairs = pairs[in_range]
    if len(pairs) == 0:
        return (
            np.zeros(node_count + 1, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )

    # Both directions, so an undirected edge is reachable from either endpoint.
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
    matrix = sparse.csr_array(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(node_count, node_count),
    )
    return (
        matrix.indptr.astype(np.int64, copy=False),
        matrix.indices.astype(np.int32, copy=False),
    )


class _SelectionNeighbours:
    """The nodes adjacent to a selection, kept up to date as it changes.

    Each node counts how many selected nodes it touches, so a change costs
    the summed degree of the nodes added and removed, not of the whole
    selection: a box selection growing over dense clusters pays per move
    for what that move adds. The counts are exact integers and never drift.
    """

    def __init__(self, indptr, indices, node_count):
        self.indptr = indptr
        self.indices = indices
        self.degree = np.diff(indptr)
        self.selected = np.zeros(node_count, dtype=bool)
        self.touches = np.zeros(node_count, dtype=np.int64)
        self.neighbours = np.empty(0, dtype=np.int32)

    def _count(self, nodes, step):
        """Add ``step`` to the count of every neighbour of ``nodes``."""
        starts = self.indptr[nodes]
        counts = self.indptr[nodes + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        # Expand every node's CSR slice into one flat gather: entry j of
        # group g reads indices[starts[g] + (j - group_start[g])].
        group_starts = np.cumsum(counts) - counts
        gather = np.arange(total, dtype=np.int64) + np.repeat(
            starts - group_starts, counts
        )
        reached = np.bincount(self.indices[gather], minlength=len(self.touches))
        if step > 0:
            self.touches += reached
        else:
            self.touches -= reached

    def neighbours_of(self, nodes):
        """Return the neighbours of ``nodes`` (unique, in range), ascending,
        leaving out ``nodes`` themselves."""
        selected = np.zeros(len(self.selected), dtype=bool)
        selected[nodes] = True
        added = np.flatnonzero(selected & ~self.selected)
        removed = np.flatnonzero(self.selected & ~selected)
        if len(added) or len(removed):
            changed = int(self.degree[added].sum()) + int(self.degree[removed].sum())
            if changed >= int(self.degree[nodes].sum()):
                # A new selection: counting it afresh costs no more.
                self.touches[:] = 0
                self._count(nodes, 1)
            else:
                self._count(added, 1)
                self._count(removed, -1)
            self.selected = selected
            self.neighbours = np.flatnonzero(
                (self.touches > 0) & ~selected
            ).astype(np.int32, copy=False)
        return self.neighbours.copy()


def _web_table_cell(value):
    """One metadata value as the web table receives it."""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    if isinstance(value, (float, np.floating)) and np.isinf(value):
        # json.dumps writes Infinity, which the browser's JSON.parse rejects;
        # a number cell takes "inf" back.
        return "inf" if value > 0 else "-inf"
    return value.item() if hasattr(value, "item") else value


def _web_table_column(values, row_count):
    """The first ``row_count`` values of a metadata column as web table cells.

    The result equals ``[_web_table_cell(values[i]) for i in
    range(row_count)]``. A numpy column is converted in one pass instead of
    one Python call per cell, which is what makes a 500,000-node table cost
    less than a second.
    """
    if (
        type(values) is np.ndarray
        and values.ndim == 1
        and len(values) >= row_count
    ):
        kind = values.dtype.kind
        column = values[:row_count]
        if kind == "O":
            # Python objects are converted one by one, as a single value is.
            return [
                value if type(value) is str else _web_table_cell(value)
                for value in column.tolist()
            ]
        # tolist() applies item() to every element, as _web_table_cell does.
        cells = column.tolist()
        if kind == "f":
            for row in np.flatnonzero(np.isnan(column)).tolist():
                cells[row] = ""
            for row in np.flatnonzero(np.isinf(column)).tolist():
                cells[row] = "inf" if column[row] > 0 else "-inf"
        return cells
    # A list, a masked array or a column too short for the nodes keeps the
    # per-cell path, and with it the error a missing cell raises.
    return [_web_table_cell(values[row]) for row in range(row_count)]


def _web_clients_connected(viewer):
    """Whether any browser page is listening for the Viewer's events."""
    server = getattr(viewer, "web_server", None)
    if not server:
        return False
    with server.queues_lock:
        return bool(server.event_queues)


# =========================================================================
# MANUAL CUSTOM ATTRIBUTES INITIALIZATION SECTION
# Users can add custom attributes to be initialized on the viewer at startup
# and registered for layout cache saving/loading.
# Format: "attribute_name": default_value (e.g. "my_scores": None)
# =========================================================================
CUSTOM_ATTRIBUTES_INIT = {
    # Add your custom attributes here:
    "sidebar_buttons_to_persist": []
}

# Cacheable attributes that undo and redo leave alone. The sidebar's list of
# buttons to register at startup follows what the user has opened, not the
# network's history: restoring an older list would drop a button that a later
# `save` or `meta` call added. `save` still writes it into the cache.
UNDO_EXCLUDED_ATTRIBUTES = frozenset({"sidebar_buttons_to_persist"})


# --- Undo history snapshots -------------------------------------------------
# A full-state history entry holds read-only copies of the viewer's arrays.
# Nothing ever changes an array held in an entry, so an entry may hold the very
# array an earlier entry holds when its contents are unchanged: a command that
# recolours nodes then costs one new colour array, not a copy of every array.

def _object_array_pointers(array):
    """The element pointers of a C-contiguous object array, as unsigned ints."""
    pointers = (ctypes.c_void_p * array.size).from_address(array.ctypes.data)
    return np.frombuffer(pointers, dtype=np.uintp)


def _holds_same_bits(array, snapshot):
    """Whether ``snapshot``, an array made by ``_history_copy``, holds exactly
    what ``array`` holds: the same type, shape, dtype and bytes.

    Bytes are compared, so a float array whose NaNs sit where the snapshot's
    do is unchanged, and 0.0 is not -0.0. An object array is unchanged when it
    points at the very objects the snapshot points at; equal but distinct
    objects count as a change. The snapshot keeps its objects alive, so an
    address it holds cannot belong to another object meanwhile.
    """
    if (
        type(snapshot) is not np.ndarray
        # A history copy is read-only and owns its memory: no other array
        # can change it.
        or snapshot.flags.writeable
        or snapshot.base is not None
        or snapshot.shape != array.shape
        or snapshot.dtype != array.dtype
        or not array.flags.c_contiguous
        or not snapshot.flags.c_contiguous
    ):
        return False
    if array.size == 0:
        return True
    if array.dtype.hasobject:
        if array.dtype != object:
            return False
        return np.array_equal(
            _object_array_pointers(array), _object_array_pointers(snapshot)
        )
    unit = np.uint64 if array.nbytes % 8 == 0 else np.uint8
    return np.array_equal(
        array.reshape(-1).view(np.uint8).view(unit),
        snapshot.reshape(-1).view(np.uint8).view(unit),
    )


def _history_copy(value, previous=None):
    """A history entry's copy of ``value``.

    A numpy array becomes a read-only copy, or ``previous`` itself when that is
    the earlier entry's copy and ``value`` has not changed since. Anything else
    is copied with its own ``copy()``, as before.
    """
    if type(value) is not np.ndarray:
        return value.copy()
    if previous is not None and _holds_same_bits(value, previous):
        return previous
    snapshot = value.copy()
    snapshot.flags.writeable = False
    return snapshot


class _GroupLabelsSnapshot:
    """The undo history's form of ``MainViewer.group_labels``, the list of one
    set of group names per node: for each distinct name, the sorted indices of
    the nodes whose set holds it.

    ``keys`` pairs each name with its type, so names that compare equal across
    types, such as 1 and 1.0, stay apart. Like the history's arrays, a
    snapshot is never changed after it is made, and later entries share it or
    its arrays while the groups are unchanged.
    """

    __slots__ = ("length", "keys", "members")

    def __init__(self, length, keys, members):
        self.length = length
        self.keys = keys
        self.members = members

    @classmethod
    def capture(cls, group_labels, previous=None):
        """Snapshot ``group_labels``; reuse ``previous`` where nothing changed.

        Only a list of plain sets is encoded. Anything else is copied as the
        history always copied it, one ``copy()`` per element.
        """
        if type(group_labels) is not list or set(map(type, group_labels)) - {set}:
            return [g.copy() for g in group_labels]
        buckets = {}
        for index in itertools.compress(range(len(group_labels)), group_labels):
            for name in group_labels[index]:
                key = (type(name), name)
                try:
                    buckets[key].append(index)
                except KeyError:
                    buckets[key] = [index]

        earlier = {}
        if isinstance(previous, cls):
            earlier = dict(zip(previous.keys, previous.members))
        index_type = np.int32 if len(group_labels) <= np.iinfo(np.int32).max else np.int64
        members = []
        for key, indices in buckets.items():
            array = np.fromiter(indices, dtype=index_type, count=len(indices))
            kept = earlier.get(key)
            if kept is not None and _holds_same_bits(array, kept):
                array = kept
            else:
                array.flags.writeable = False
            members.append(array)
        keys, members = tuple(buckets), tuple(members)

        if (
            isinstance(previous, cls)
            and previous.length == len(group_labels)
            and previous.keys == keys
            and all(a is b for a, b in zip(members, previous.members))
        ):
            return previous
        return cls(len(group_labels), keys, members)

    def to_list(self):
        """A new list of new sets equal to the captured ``group_labels``."""
        # One new set per node: on a large network the cyclic collector,
        # which these allocations keep triggering, would take most of the
        # time. Sets of names hold no cycles, so it waits until they exist.
        collecting = gc.isenabled()
        gc.disable()
        try:
            labels = [set() for _ in range(self.length)]
            for (_, name), indices in zip(self.keys, self.members):
                for index in indices.tolist():
                    labels[index].add(name)
        finally:
            if collecting:
                gc.enable()
        return labels


# Fix High-DPI scaling
class HUDDisplay:
    def __init__(self, viewer, name, pos_fn, anchor_x='right', anchor_y='bottom'):
        self.viewer = viewer
        self.name = name
        self.pos_fn = pos_fn  # lambda size: (x, y)
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.text_visual = None
        self.visible = False

    def show(self, text):
        w, h = self.viewer.canvas.size
        pos = self.pos_fn((w - _sidebar_room(self.viewer), h))

        if self.text_visual is None:
            self.text_visual = scene.visuals.Text(
                text=text,
                bold=True,
                face=self.viewer.vispy_ui_face,
                font_size=self.viewer._hud_font_size_points(),
                color=cfg.TEXT_COLOR,
                pos=pos,
                anchor_x=self.anchor_x,
                anchor_y=self.anchor_y,
                parent=self.viewer.canvas.scene
            )
        else:
            self.text_visual.text = text
            self.text_visual.pos = pos
            self.text_visual.visible = True
        self.visible = True

    def hide(self):
        if self.text_visual is not None:
            self.text_visual.visible = False
            self.text_visual.text = ""
        self.visible = False

    def update_position(self):
        if self.text_visual is not None and self.visible:
            w, h = self.viewer.canvas.size
            self.text_visual.pos = self.pos_fn((w - _sidebar_room(self.viewer), h))

    def on_node_clicked(self, node_idx):
        """Override in subclasses to handle left-click updates."""
        pass

    def on_right_click(self):
        """Handle right-click event. Hides by default, override if custom behavior is needed."""
        self.hide()


class MainViewer:
    def __init__(self, *, headless=None, settings=None):
        if settings is not None:
            from desktop.Viewer_State import resolve_viewer_document
            cfg.__dict__.update(resolve_viewer_document(settings, cfg.PROJECT_ROOT))
        self.headless = bool(
            headless or (_startup_args and _startup_args.headless)
            or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            or os.environ.get("SSN_VIEWER_HEADLESS", "").lower() in {"1", "true", "yes"}
        )
        # normal, hidden (native platform, window kept off screen) or headless (offscreen platform);
        # see desktop.Viewer_Window. A hidden request that found no display runs headless.
        self.requested_window_mode = HIDDEN if os.environ.get("SSN_VIEWER_WINDOW") == "hidden" else (
            HEADLESS if self.headless else NORMAL)
        self.window_mode = HEADLESS if self.headless else self.requested_window_mode
        self.window_fallback_reason = (
            os.environ.get("SSN_VIEWER_WINDOW_FALLBACK") or None) if self.window_mode != self.requested_window_mode else None

        if self.window_mode == HIDDEN:
            # Before the network is loaded: a host whose native platform has no OpenGL stops here.
            self._check_native_opengl()

        # --- 1. Viewer State ---
        self.console_mode = False
        from web_ui.Plugin_Manager import WebPluginRegistry
        self.web_plugin_registry = WebPluginRegistry(self)
        self.web_action_handlers = self.web_plugin_registry.actions
        self.vispy_ui_face = VISPY_FALLBACK_FACE
        self.vispy_monospace_face = VISPY_FALLBACK_FACE
        self._display_window_handle = None
        self._active_screen = None
        self._active_screen_signal_bindings = []
        self._display_refresh_pending = False
        self._last_display_signature = None
        
        # =========================================================================
        # HUD & CONSOLE LAYOUT CONFIGURATION SECTION
        # Users can adjust the positions, sizes, and padding of the text and 
        # background elements below. Adjust these coordinates if elements do not
        # align correctly on your screen or High-DPI display.
        # Note: All coordinates are defined in logical pixels and are automatically
        # scaled by the canvas pixel scale (DPI factor) at runtime.
        # =========================================================================
        self.hud_layout = {
            # 1. Command Line Text (" Cmd: <input> "). The mouse and keyboard
            #    controls open from the info button by the threshold slider.
            "console_text_x": 30.0,      # Horizontal coordinate from left edge
            "console_text_y": 20.0,      # Top of the text; its box's top then lines up with the sidebar toggle's
            "console_text_anchor_x": "left",
            "console_text_anchor_y": "bottom",
            "font_size_px": 16.0,        # Logical-pixel size shared by all HUD text

            # 2. Command Line Background Box
            "console_bg_height": 40.0,   # Vertical height of the background box
            "console_bg_min_width": 250.0, # Minimum width of the background box when empty/short
            "console_bg_left_offset": 20.0, # Fixed horizontal left position of the box
            "console_bg_y_offset": 20.0, # Positive values move only the box downward
            "console_bg_radius": 10.0,    # Radius for the rounded corners (0.0 for sharp corners)
            "console_bg_padding_x": 20.0, # Fixed logical padding added to the end of the command box
            # 3. Background-job status text below the command box
            "background_job_status_gap": 8.0,
            "background_job_status_right_padding": 20.0,

            # 4. Bottom-right status stack, above the Zoom box (level with
            #    the threshold slider), from bottom to top: Hidden Nodes, then
            #    the selected metadata property. The box's row and the two lines
            #    are evenly spaced, middle to middle, so each position is the
            #    middle of its line. ("center" because VisPy's "top" and
            #    "bottom" are flipped on this canvas, whose y grows downward.)
            "status_x_offset": 10.0,       # Distance from the right edge of the window, the box's too
            "status_line_spacing": 34.0,   # Middle to middle; leaves ~15 px between Hidden Nodes and the box
            "status_anchor_x": "right",
            "status_anchor_y": "center"
        }
        
        self.input_buffer = ""
        self.cursor_pos = 0       # Tracks cursor position
        
        # ---> Persistent Command History (Per Layout) <---
        self.command_history = []
        try:
            cache_path = resolve_selected_cache(cfg)
            cache_dir = os.path.dirname(cache_path)
            self.history_file = os.path.join(cache_dir, "cli_history.txt")
            
            # Migration check: if old history exists in Cache_Files/History/folder_name.txt, copy to new location
            folder_name = os.path.basename(cache_dir)
            old_history_file = os.path.join("Cache_Files", "History", f"{folder_name}.txt")
            if not os.path.exists(self.history_file) and os.path.exists(old_history_file):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(old_history_file, self.history_file)
                except Exception as e:
                    print(f"Warning: Could not migrate old CLI history: {e}")
            
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.command_history = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Warning: Could not bind specific history file ({e}). Defaulting format.")
            saved_layout_dir = getattr(cfg, 'SAVED_LAYOUT_DIR', os.path.join("Cache_Files", "Saved_Layouts"))
            self.history_file = os.path.join(saved_layout_dir, "cli_history.txt")

        self.history_index = len(self.command_history)
        
        self.cluster_labels = None
        self.label_visuals = []
        

        self.full_headers = [] # Original headers (for caching/integrity)

        
        self.last_cluster_params = None
        
        # Alignment Data 
        self.active_reference = cfg.ALIGNMENT_REFERENCE
        try:
            self.alignment_offset = int(getattr(cfg, 'ALIGNMENT_OFFSET', 0))
        except (TypeError, ValueError):
            self.alignment_offset = 0
        self.alignment = None
        self.col_to_label = None  
        self.label_to_col = None  
        # Embedding Data (optional): the manifest only; rows are read on demand.
        self.embedding = None
        
        # ---> NEW: Selection & Drag State <---
        self.selected_indices = []
        # Counts the selections made by hand (clicks, boxes): a command waiting
        # in the command queue uses the selection it was given, and its own
        # selection replaces only one older than it (Command_Queue.Job).
        self.selection_epoch = 0
        self.is_box_selecting = False
        self._box_selection_unsent = False
        self.is_multi_dragging = False
        self.drag_start_mouse = None
        self.drag_start_screen = None
        self.drag_start_nodes_pos = None
        self.position_history = []  # Tracks states for Undo
        self.hud_displays = {}

        # --- 2. Data Loading & Simulation ---
        self.load_and_simulate()
        self.original_pos = self.pos.copy()  # <--- NEW: Backup original layout
        self.load_global_alignment()
        self.load_embedding()
        
        # --- 3. Setup Window & Canvas ---
        print(f"Building network display: {self.n_nodes} nodes, {len(self.edges)} edges.")
        Startup_Status.report(Startup_Status.BUILDING_NETWORK, 0.6,
                              nodes=f"{self.n_nodes:,}", edges=f"{len(self.edges):,}")
        self.canvas = scene.SceneCanvas(
            keys=None,
            show=False,
            title=f"{VIEWER_DISPLAY_NAME} (Live)",
            bgcolor='white',
            # The edges' overdraw limit counts in the stencil buffer
            # (desktop.Viewer_Edges.EdgeStencilLimit, which checks what it got).
            config=dict(stencil_size=8),
        )
        # Keep the complete traceback available if a future VisPy draw
        # callback fails; the default logarithmic reminders hide the cause
        # after the first occurrence.
        self.canvas.events.draw.print_callback_errors = 'first'
        qapp = QtWidgets.QApplication.instance()
        if qapp:
            try:
                qt_font_status = configure_qt_application_fonts(qapp)
                # The canvas was made before the bundled font became the
                # application's; give it that font now, so the sidebar and
                # slider built on it inherit it rather than the system font.
                self.canvas.native.setFont(qapp.font())
                vispy_font_status = register_vispy_application_fonts(
                    qt_font_status
                )
                self.vispy_ui_face = vispy_font_status.ui_face
                self.vispy_monospace_face = vispy_font_status.monospace_face
            except Exception as e:
                print(f"Warning: Could not configure bundled application fonts: {e}")
            try:
                install_translations(qapp, startup_language())
            except Exception as e:
                print(f"Warning: Could not load translations, so text stays English: {e}")
            # VisPy draws a text in one face, so a language the core faces
            # lack, such as Chinese, draws all its text in a face of its own.
            language_face = vispy_language_face(installed_language())
            if language_face is not None:
                self.vispy_ui_face = self.vispy_monospace_face = language_face
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.resize.connect(self.on_resize)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        
        # --- NEW: Hook mouse wheel and move for dynamic tooltips and HUD ---
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        
        self.selected_node_idx = None
        # Rename the timer so it handles all dynamic HUD elements
        self._hud_timer = app.Timer(0.001, connect=self._update_hud_elements, iterations=1)
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom' 
        self.view.camera.aspect = 1
        
        # --- Disable default Vispy Backspace reset ---
        # Save the original bound method
        self._original_camera_key_event = self.view.camera.viewbox_key_event
        
        # Create a custom wrapper that filters out Backspace
        def safe_camera_key_event(event):
            if event.key == 'Backspace':
                return  # Block the event from reaching the camera
            self._original_camera_key_event(event)
            
        # Replace the camera's handler with our custom wrapper
        self.view.camera.viewbox_key_event = safe_camera_key_event

        # The camera's own mouse handling, less the right button and with a
        # smooth wheel (_camera_mouse_event); zooms keep to the width limits.
        self._original_camera_mouse_event = self.view.camera.viewbox_mouse_event
        self.view.camera.viewbox_mouse_event = self._camera_mouse_event
        self._limit_camera_zoom()
        self._wheel_zoom = SmoothWheelZoom(self.view.camera, self.view.camera.zoom, self._after_wheel_zoom_step)

        # --- 4. Draw Initial State ---
        self.draw_network()
        self.create_hud()
        # One count numbers the commands that wait their turn and the files being written.
        self._job_numbers = itertools.count(1)
        self.background_job_scheduler = BackgroundJobScheduler(self, job_numbers=self._job_numbers)
        # Commands that read or change the network run here, one at a time, in the order given.
        self.command_queue = CommandQueue(self, job_numbers=self._job_numbers)
        self.command_queue.changed.connect(self._refresh_jobs_badge)
        # The badge and the Session card's Background jobs list follow the scheduler's lanes.
        self.background_job_scheduler.changed.connect(self._on_background_jobs_changed)
        if hasattr(self.canvas.events, "close"):
            self.canvas.events.close.connect(
                lambda _event: self.command_queue.shutdown()
            )
            self.canvas.events.close.connect(
                lambda _event: self.background_job_scheduler.shutdown()
            )
            self.canvas.events.close.connect(
                lambda _event: self.shutdown_web_server()
            )
        if qapp:
            qapp.aboutToQuit.connect(self.command_queue.shutdown)
            qapp.aboutToQuit.connect(self.background_job_scheduler.shutdown)
            qapp.aboutToQuit.connect(self.shutdown_web_server)
        
        # Open on every node with a margin, so none touches the window's edge
        # (commands.zoom.opening_view); the widest zoom always leaves this view.
        from commands.zoom import opening_view
        self.view.camera.rect = opening_view(self.pos)
        
        # --- 5. Setup Similarity / E-value Slider Bar ---
        self.is_evalue = getattr(cfg, 'INPUT_IS_EVALUE', False)
        
        min_val = getattr(cfg, 'SIMILARITY_THRESHOLD', None)
        if min_val is None or min_val == "None":
            if hasattr(self, 'edge_scores') and len(self.edge_scores) > 0:
                min_val = float(np.min(self.edge_scores))
            else:
                min_val = 0.0
        self.min_threshold = float(min_val)
        
        if hasattr(self, 'edge_scores') and len(self.edge_scores) > 0:
            self.max_threshold = float(np.max(self.edge_scores))
        else:
            self.max_threshold = self.min_threshold + 1.0
            
        if self.min_threshold >= self.max_threshold:
            self.max_threshold = self.min_threshold + 1.0

        self.current_slider_threshold = self.min_threshold
        # The initial line visual is created from the complete topology. Apply
        # the restored visibility mask and active threshold before the window
        # is shown so cached hidden nodes cannot leave stale edges on screen.
        self.update_edges()

        # The shared window theme, on the QApplication managed by Vispy, in
        # Config's theme colour as it is now; the Agent and Meta Data pages take it too.
        if qapp:
            try:
                apply_studio_theme(qapp, configured_theme_color(CONFIG_THEME_SETTING))
                # The network and the HUD were drawn before the theme colour was known.
                self._color_command_box()
                self._color_selection_box()
                self._color_hidden_count()
            except Exception as e:
                print(f"Warning: Could not apply the window theme: {e}")
            
            # Set Application-wide Icon (covers Vispy and all spawned windows/dialogs)
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.ico")
            if not os.path.exists(icon_path):
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.png")
            if os.path.exists(icon_path):
                qapp.setWindowIcon(QtGui.QIcon(icon_path))

        self._build_slider_overlay()
        
        # --- 6. Set up MainWindow & WebServer ---
        self._build_main_window()
        self._build_fps_overlay()
        self._build_jobs_badge()
        
        # Initialize thread-safe QtCommunicator for server commands
        from web_ui import Web_Server
        self.communicator = Web_Server.QtCommunicator(self)
        self.viewer_inspection = ViewerInspectionService(self, cfg)
        
        # Discover bundled web plugins before the server begins accepting requests.
        self.web_server = None
        self.web_server_url = None
        self.web_plugin_manager = None
        try:
            from web_ui.Plugin_Manager import WebPluginManager
            self.web_plugin_manager = WebPluginManager(self)
            self.web_plugin_manager.discover_and_register()
        except Exception as error:
            print(f"Web plugin discovery failed: {error}")

        # A hidden Viewer's window is opened (off screen) before the session is published, so
        # a platform that cannot give it an OpenGL context fails before a client can connect.
        if self.window_mode == HIDDEN:
            self._open_hidden_window()

        # Initialize background WebServer
        self.start_web_server()
        
        # Run persistent sidebar button registration commands
        if getattr(self, 'sidebar_buttons_to_persist', None):
            for cmd_name in self.sidebar_buttons_to_persist:
                self.process_command(f"{cmd_name} --register-only", record_history=False, silent=True)
        
        # Ensure the side panel is hidden at startup
        self.set_sidebar_visible(False)
        
        if self.window_mode == NORMAL:
            Startup_Status.report(Startup_Status.OPENING_WINDOW, 0.9)
            self.window_control.open()
            # Once it is on screen, tell the launcher that started this Viewer (it closes its splash).
            QtCore.QTimer.singleShot(150, lambda: signal_launcher_ready(self.main_window))
        if self.window_mode != HEADLESS:
            QtCore.QTimer.singleShot(0, self._initialize_display_tracking)
        
        self._hud_timer.start()
        print("\n" + self._ready_line(self.window_mode))

    @staticmethod
    def _ready_line(window_mode):
        """The line a Viewer prints when it is up, for a terminal, the Console and the log.

        [ENTER] opens the command line that is drawn over the network, in the Viewer's window: no Viewer
        reads its terminal, so the key does nothing there. A Viewer with no window on screen (headless,
        or hidden until shown) has no command line to open, so it only says that it is ready.
        """
        if window_mode == NORMAL:
            return "Viewer ready. Press [ENTER] in the Viewer window to type commands."
        return "Viewer ready."

    def shows_windows(self):
        """Whether the user is at this Viewer, so dialogs, message boxes, browser pages and terminals may open.

        True for a normal Viewer, and for a hidden one while its window is on screen
        (show_session); false for a headless one and for a hidden one not shown.
        """
        if getattr(self, 'headless', False):
            return False
        control = getattr(self, 'window_control', None)
        return control.shows_windows() if control is not None else True

    @staticmethod
    def _hidden_window_unavailable(error):
        """End a hidden launch whose platform cannot give the canvas an OpenGL context.

        The MCP client launches the Viewer headless again when it sees the marker line and
        the exit code. Nothing is published or written yet, so the process just stops.
        """
        from utilities.Console_Capture import hard_exit
        from utilities.Viewer_Display import HIDDEN_UNAVAILABLE_EXIT_CODE, HIDDEN_UNAVAILABLE_MARKER
        print(f"{HIDDEN_UNAVAILABLE_MARKER}: {error}", flush=True)
        # Not os._exit: with the capture on, the marker is still in a pipe that a thread of this
        # process has to pass on, and the MCP client needs it to start the headless fallback.
        hard_exit(HIDDEN_UNAVAILABLE_EXIT_CODE)

    def _check_native_opengl(self):
        """Make an OpenGL context on the native platform, as a hidden Viewer's canvas will need."""
        try:
            # VisPy's backend sets Qt's context-sharing attribute, then makes the QApplication.
            app.use_app().native
            context = QtGui.QOpenGLContext()
            surface = QtGui.QOffscreenSurface()
            surface.create()
            if not (context.create() and surface.isValid() and context.makeCurrent(surface)):
                raise RuntimeError("the native platform cannot create an OpenGL context")
            context.doneCurrent()
        except Exception as error:
            self._hidden_window_unavailable(error)

    def _open_hidden_window(self):
        """Open the hidden-mode window off screen, and check its canvas has an OpenGL context."""
        try:
            self.window_control.open()
            if not self.canvas.native.isValid():
                raise RuntimeError("the native platform could not create an OpenGL context for the canvas")
        except Exception as error:
            self._hidden_window_unavailable(error)

    def _limit_camera_zoom(self):
        """Keep zooming by the wheel or a pinch within the zooms the zoom
        command keeps to (commands.zoom.zoom_limits): it stops at the highest zoom
        (MAX_ZOOM_PERCENT) and at the zoom, rounded down to two figures, that
        shows every node, wherever they are now."""
        original_zoom = self.view.camera.zoom

        def limited_zoom(factor, center=None):
            """Zoom, cut at the limits, and return the factor applied."""
            from commands.zoom import limited_zoom_factor
            limited = limited_zoom_factor(self, factor)
            original_zoom(limited, center)
            return limited

        self.view.camera.zoom = limited_zoom

    def _note_view_motion(self):
        """A pan, zoom or node-drag input: the edges draw every k-th edge until
        the view settles (desktop.Viewer_Motion). Only input handlers call this,
        so a camera moved by print, a command or a capture keeps full detail."""
        layer = getattr(self, 'edge_layer', None)
        if layer is not None:
            layer.note_motion()

    def _after_wheel_zoom_step(self):
        """Each frame of a smooth wheel zoom: the view moves, and the HUD follows."""
        self._note_view_motion()
        self._hud_timer.start()

    def _camera_mouse_event(self, event):
        """The camera's mouse handling, without the right button, which selects
        and drags nodes, and with the wheel zooming smoothly."""
        if getattr(event, 'button', None) == 2 or 2 in getattr(event, 'buttons', []):
            return
        # A left-button drag pans, the wheel and a pinch zoom.
        event_type = getattr(event, 'type', None)
        if (event_type == 'mouse_move' and 1 in getattr(event, 'buttons', [])) \
                or event_type == 'mouse_wheel' or str(event_type).startswith('gesture'):
            self._note_view_motion()
        camera = self.view.camera
        # VisPy's zoom a notch, about the point under the cursor, played out
        # over a few frames (desktop.Viewer_Zoom) rather than in one; Low
        # Resource Mode's No Zoom or Sidebar Animation zooms at once, sparing the frames between.
        if (event.type == 'mouse_wheel' and not event.handled and camera.interactive
                and not low_resource_animations(cfg)):
            center = camera._scene_transform.imap(event.pos)
            self._wheel_zoom.add((1 + camera.zoom_factor) ** (-event.delta[1] * 30), center)
            event.handled = True
            return
        self._original_camera_mouse_event(event)

    def _build_slider_overlay(self):
        """Set up the slider row over the canvas's bottom: the threshold slider from
        the left, the controls' info button and the Console button at its start, and
        the Zoom box at the right."""
        # The overlay is a child of the native canvas.
        self.slider_overlay = QtWidgets.QWidget(self.canvas.native)
        self.slider_overlay.setObjectName("sliderOverlay")
        
        overlay_layout = QtWidgets.QHBoxLayout(self.slider_overlay)
        overlay_layout.setContentsMargins(5, 5, 5, 5)
        overlay_layout.setSpacing(10)

        # The info button at the slider's start opens the session (the nodes,
        # edges and files) and the mouse and keyboard controls on a card above
        # it. The card is a child of the canvas, so it stays open while the
        # network is used; _refresh_session_card keeps it current (after a
        # command too: Command_Engine calls it).
        self.controls_button = ControlsButton(
            translate("Viewer", "Session and controls"),
            translate("Viewer", "[ENTER] Command | [LeftClick] Highlight | [RightClick] Select/Clear | [Scroll] Zoom | [LeftClick + Shift/Ctrl] Copy Node Header/Sequence | [LeftClick + Drag] Pan | [RightClick + Drag] GroupSelect/MoveNodes"),
            host=self.canvas.native,
            session=self._session_overview,
            controls_title=translate("Viewer", "Mouse and keyboard controls"),
            area=self._session_card_area,
            # The card's pick buttons type "alignment" and "embedding": the same command path.
            command=self.process_command,
            # Ctrl+C copies a selected path of the card, unless the command line is open: it copies its own text.
            command_line_open=lambda: getattr(self, 'console_mode', False),
            # The background jobs the card lists, and its buttons that arrange them.
            jobs=ViewerJobs(self),
        )
        overlay_layout.addWidget(self.controls_button)

        # The Console button, after the info button: the terminal icon and the word,
        # as tall as the boxes after it. It opens the Console, as the `console` command does.
        self.console_button = ConsoleButton(
            translate("Viewer", "Console"),
            translate("Viewer", "Show everything this Viewer has printed: progress, warnings and errors"),
            opened=self._console_button_clicked,
        )
        overlay_layout.addWidget(self.console_button)
        overlay_layout.addSpacing(SLIDER_BUTTONS_GAP - overlay_layout.spacing())

        # The threshold, to read and to set, then the slider: each number of
        # the box is one of the slider's positions, so each follows the other.
        self.threshold_box = ThresholdSpinBox()
        self.threshold_box.setObjectName("thresholdBox")
        self.threshold_box.set_scale(self.min_threshold, self.max_threshold)
        positions = self.threshold_box.positions

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, positions)
        # The wheel over the slider and a click on its track still move it
        # about a thousandth and a hundredth of the way, however many positions it has.
        self.slider.setSingleStep(max(1, positions // 1000))
        self.slider.setPageStep(max(1, positions // 100))
        start = self.threshold_box.position_of(self.current_slider_threshold)
        self.slider.setValue(start)
        self.threshold_box.show_value(self.threshold_box.value_at(start))
        self.slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        # The slider is what gives up room to the buttons and the boxes.
        self.slider.setMinimumWidth(SLIDER_MIN_WIDTH)
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.threshold_box.valueChanged.connect(self._set_threshold_from_box)
        self.threshold_box.editingFinished.connect(self._return_keyboard_to_canvas)
        self.threshold_box.dismissed.connect(self._return_keyboard_to_canvas)

        overlay_layout.addWidget(self.threshold_box)
        overlay_layout.addWidget(self.slider)

        # The slider and the box take the window theme, the box in the Zoom box's font.
        self.slider_overlay.setStyleSheet("""
            QWidget#sliderOverlay {
                background: transparent;
            }
            QDoubleSpinBox#thresholdBox { font-family: %(font)s; font-size: 12pt; }
        """ % {"font": UI_QSS_FONT_STACK})

        # The view's zoom, to read and to set as `zoom` does, at the row's
        # right; its text keeps the HUD's text colour, as the slider's value does.
        from commands.zoom import ZOOM_100_PERCENT_WIDTH
        self.zoom_box = ZoomBox(translate("Viewer", "Zoom: {zoom}%"), self.canvas.native)
        self.zoom_box.setToolTip(translate(
            "Viewer",
            "A bigger zoom is closer: 100% is a view {width} layout units wide. "
            "Type a percentage, or use the arrows or the mouse wheel over the box.",
        ).format(width=f"{ZOOM_100_PERCENT_WIDTH:g}"))
        self.zoom_box.setStyleSheet("""
            QWidget#zoomBox { background: transparent; }
            QWidget#zoomBox QLabel {
                font-family: %(font)s;
                font-size: 12pt;
                color: %(text_color)s;
                background: transparent;
            }
            QWidget#zoomBox QDoubleSpinBox { font-family: %(font)s; font-size: 12pt; }
        """ % {"font": UI_QSS_FONT_STACK, "text_color": cfg.TEXT_COLOR})
        spin_box = self.zoom_box.spin_box
        spin_box.valueChanged.connect(self._zoom_to_box)
        spin_box.editingFinished.connect(self._return_keyboard_to_canvas)
        spin_box.dismissed.connect(self._return_keyboard_to_canvas)

        self.position_slider_overlay()
        self.slider_overlay.show()
        self.zoom_box.show()

    def _session_overview(self):
        """The counts and files the info card lists (ViewerInspectionService.session_overview)."""
        service = getattr(self, 'viewer_inspection', None)
        if service is None:  # The card is built before the service is.
            service = ViewerInspectionService(self, cfg)
        return service.session_overview()

    def _session_card_area(self):
        """Where the info card may be, in canvas px: the canvas less the sidebar's
        room, a margin all round, and clear of the command line at the top."""
        width, height = self.canvas.size
        right = width - _sidebar_room(self) - SESSION_CARD_MARGIN
        return QtCore.QRect(
            SESSION_CARD_MARGIN, SESSION_CARD_TOP,
            max(0, right - SESSION_CARD_MARGIN), max(0, height - SESSION_CARD_TOP - SESSION_CARD_MARGIN),
        )

    def _refresh_session_card(self):
        """Bring the info card up to date, if it is open: at no cost when it is not.

        update_edges (the threshold, hide and undo), _update_hud_elements
        (every change to the nodes) and Command_Engine, after every command
        (one may change a file the card shows, as alignment does), call it. It reads counts the Viewer
        already holds, so it adds no work per edge.
        """
        button = getattr(self, 'controls_button', None)
        if button is not None:
            button.refresh_card()

    def _zoom_to_box(self, percent):
        """Zoom to the percentage set in the Zoom box, as `zoom` does (commands.zoom.set_zoom)."""
        from commands.zoom import set_zoom
        self._note_view_motion()
        try:
            set_zoom(self, percent)
        except ValueError as error:
            print(error)
            self._show_click_status(error.args[0])
        self._return_keyboard_to_canvas()
        # The box shows the zoom the view took, or kept.
        self._hud_timer.start()

    def _return_keyboard_to_canvas(self):
        """Give the keyboard back to the network once a box in the slider row is done with it, so [ENTER] opens the command line."""
        if self.zoom_box.spin_box.hasFocus() or self.threshold_box.hasFocus():
            self.canvas.native.setFocus()

    def _build_main_window(self):
        """Set up the main window around the canvas: its title, the sidebar and the sidebar's toggle."""
        # The sidebar's width until its buttons size it (_fit_sidebar_to_buttons).
        self._panel_w = 180
        self.main_window = QtWidgets.QMainWindow()
        from utilities.Viewer_Sessions import ensure_viewer_identity
        ensure_viewer_identity(self)
        alias = self.inspection_session_alias
        # MCP reports the title in English (Viewer_Inspection); the window shows it translated.
        self.inspection_window_title = f"{VIEWER_DISPLAY_NAME} [{alias}]"
        self.main_window.setWindowTitle(f"{translate('Viewer', VIEWER_DISPLAY_NAME)} [{alias}]")
        
        # Set Window Icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.png")
        if os.path.exists(icon_path):
            self.main_window.setWindowIcon(QtGui.QIcon(icon_path))
            
        self.main_window.resize(1200, 800)
        self.main_window.setMinimumWidth(self._panel_w)
        
        # Set the Vispy canvas directly as the central widget
        self.main_window.setCentralWidget(self.canvas.native)
        
        # Collapsible right panel over the canvas.native: a card with round
        # corners, clear of the canvas's top, right and bottom edges, so its
        # right margin holds that inset too.
        self.right_panel = SidebarPanel(self.canvas.native)
        self.right_panel.setObjectName("rightPanel")
        right_panel_layout = QtWidgets.QVBoxLayout(self.right_panel)
        right_panel_layout.setContentsMargins(10, 20, 10 + SIDEBAR_INSET, 20)
        right_panel_layout.setSpacing(15)
        right_panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        # Stretch at the bottom to keep dynamic buttons at the top
        self.right_panel_layout = right_panel_layout
        self.right_panel_layout.addStretch()
        
        # Single floating toggle button on the canvas.native to collapse/expand sidebar
        self.toggle_sidebar_btn = QtWidgets.QPushButton(self.canvas.native)
        self.toggle_sidebar_btn.setObjectName("toggleSidebarBtn")
        set_role(self.toggle_sidebar_btn, "icon")
        self.toggle_sidebar_btn.setIcon(studio_icon("panel-right-close"))
        self.toggle_sidebar_btn.setIconSize(QtCore.QSize(16, 16))
        self.toggle_sidebar_btn.setToolTip(translate("Viewer", "Toggle sidebar panel"))
        self.toggle_sidebar_btn.setFixedWidth(30)
        self.toggle_sidebar_btn.setFixedHeight(30)
        self.toggle_sidebar_btn.clicked.connect(self.toggle_sidebar)
        self.toggle_sidebar_btn.hide()

        # The sidebar's buttons (SidebarButton) start 10 px in, where their
        # icons go. The card paints itself; the rest is the window theme.
        self.main_window.setStyleSheet("""
            QWidget#rightPanel QPushButton { padding-left: 10px; min-height: 24px; }
            QWidget#rightPanel QPushButton:focus { padding-left: 9px; }
        """)

        # The window's mode and, for a hidden Viewer, showing and hiding it (desktop.Viewer_Window).
        mode = getattr(self, 'window_mode', HEADLESS if getattr(self, 'headless', False) else NORMAL)
        self.window_control = ViewerWindow(
            self.main_window, mode, requested=getattr(self, 'requested_window_mode', mode),
            fallback_reason=getattr(self, 'window_fallback_reason', None))
        # The slider row is built first: the window keeps the room it needs.
        self._update_window_minimum_width()

    def _command_line_message(self):
        """The command being typed, with an underscore at the cursor."""
        buf = self.input_buffer
        c = self.cursor_pos
        return Message("Cmd: {command}", command=f"{buf[:c]}_{buf[c:]}")

    def _update_console_text(self):
        """Helper to render the command line with a visible cursor."""
        Command_Engine.show_status(self, self._command_line_message())
        self.update_console_background()
        self.canvas.update()

    def _show_click_status(self, message):
        """Show what a click did on the console line.

        While a command is being typed, message follows the command instead
        of replacing it, so the command stays in view; the next key shows
        the command alone again.
        """
        if not getattr(self, 'console_mode', False):
            Command_Engine.show_status(self, message)
            return
        Command_Engine.show_status(
            self,
            JoinedMessage((self._command_line_message(), message), separator=CLICK_STATUS_SEPARATOR),
        )
        self.update_console_background()
        self.canvas.update()

    def _canvas_dpi(self):
        """Return the DPI value VisPy uses to convert text points to pixels."""
        return float(getattr(self.canvas, 'dpi', 96.0))

    def _hud_font_size_points(self):
        """Return a cross-platform VisPy point size for the logical HUD size."""
        logical_pixels = self.hud_layout.get("font_size_px", 16.0)
        return vispy_points_for_logical_pixels(logical_pixels, self._canvas_dpi())

    def _tooltip_font_size_points(self):
        """Normalize the configurable tooltip size to its 96-DPI appearance."""
        return vispy_points_at_reference_dpi(cfg.TEXT_SIZE, self._canvas_dpi())

    def _status_hud_position(self, line_index, size=None):
        """Return one position in the bottom-right HUD stack above the Zoom box.

        Line 0 is Hidden Nodes, line 1 the selected metadata property. Each is
        the right end and middle of its line, a status_line_spacing step above
        the line, or the Zoom box's row, below it.
        """
        if size is None:
            size = self.canvas.size
        cfg_hud = self.hud_layout
        return (
            size[0] - cfg_hud["status_x_offset"],
            _slider_row_centre(size[1]) - (line_index + 1) * cfg_hud["status_line_spacing"],
        )

    def _apply_vispy_text_scaling(self):
        """Apply logical-pixel sizing to static and dynamically-created text."""
        hud_font_size = self._hud_font_size_points()
        for attribute_name in (
            'console_text',
            'background_job_status_text',
            'hidden_text',
        ):
            visual = getattr(self, attribute_name, None)
            if visual is not None:
                visual.font_size = hud_font_size

        tooltip = getattr(self, 'tooltip', None)
        if tooltip is not None:
            tooltip.font_size = self._tooltip_font_size_points()

        for display in getattr(self, 'hud_displays', {}).values():
            visual = getattr(display, 'text_visual', None)
            if visual is not None:
                visual.font_size = hud_font_size

    def _color_command_box(self):
        """Fill the command line's box with a light tint of the theme colour, edged in the colour itself."""
        box = getattr(self, 'console_bg', None)
        if box is not None:
            box.color = Color(TOKENS["tint"], alpha=COMMAND_BOX_OPACITY)
            box.border_color = TOKENS["accent"]

    def _color_selection_box(self):
        """Draw the right-drag selection box in the theme colour, as the command line's box is edged."""
        box = getattr(self, 'selection_box', None)
        if box is not None:
            box.set_data(color=TOKENS["accent"])

    def update_console_background(self):
        """Wrap the console visually and size its background to the rendered rows."""
        if not hasattr(self, 'console_bg') or not hasattr(self, 'console_text'):
            return
        
        cfg_hud = self.hud_layout
        text_visual = self.console_text

        def measure_text_width(text):
            return _vispy_text_width_pixels(text_visual, text)

        left_edge = cfg_hud["console_bg_left_offset"]
        left_gap = cfg_hud["console_text_x"] - cfg_hud["console_bg_left_offset"]
        padding_x = cfg_hud.get("console_bg_padding_x", 20.0)

        # The box runs from left_edge to at most _command_line_right().
        max_width = max(1.0, self._command_line_right() - left_edge)
        available_text_width = max(
            1.0,
            max_width - left_gap - padding_x,
        )

        current_text = text_visual.text or ""
        previous_rendered_text = getattr(self, '_console_rendered_text', None)
        if previous_rendered_text is None or current_text != previous_rendered_text:
            # Overlay output remains one logical line. Newlines supplied by a
            # command are normalized; only the rendered copy receives wrapping.
            logical_text = current_text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
            self._console_logical_text = logical_text
        else:
            logical_text = getattr(self, '_console_logical_text', current_text)

        rendered_text = _wrap_console_text_for_display(
            logical_text,
            available_text_width,
            measure_text_width,
        )
        self._console_rendered_text = rendered_text
        if current_text != rendered_text:
            text_visual.text = rendered_text

        rendered_lines = rendered_text.split('\n') if rendered_text else ['']
        text_width = max(
            (measure_text_width(line) for line in rendered_lines),
            default=0.0,
        )

        min_width = min(
            cfg_hud["console_bg_min_width"],
            max_width,
        )
        desired_width = left_gap + text_width + padding_x
        width = min(max_width, max(min_width, desired_width))

        base_height = cfg_hud["console_bg_height"]
        font_line_height = _vispy_text_line_height_pixels(text_visual)
        extra_height = max(0, len(rendered_lines) - 1) * font_line_height
        height = base_height + extra_height
        radius = cfg_hud["console_bg_radius"]
        center_x = left_edge + width / 2.0
        text_y = cfg_hud["console_text_y"]
        text_visual.pos = (cfg_hud["console_text_x"], text_y)
        text_anchor_y = cfg_hud.get("console_text_anchor_y", "bottom")
        if text_anchor_y == "top":
            single_line_center_y = text_y + font_line_height / 2.0
        elif text_anchor_y in ("center", "middle"):
            single_line_center_y = text_y
        else:
            single_line_center_y = text_y - font_line_height / 2.0
        # Keep the single-line top edge fixed so wrapped command lines and the
        # background both extend downward together.
        center_y = (
            single_line_center_y
            + extra_height / 2.0
            + cfg_hud.get("console_bg_y_offset", 0.0)
        )

        _apply_safe_rectangle_geometry(
            self.console_bg,
            center=(center_x, center_y),
            width=width,
            height=height,
            radius=radius,
        )

        self.update_background_job_status_layout()

    def update_background_job_status_layout(self):
        """Wrap and position scheduler status beneath the current command box."""
        if (
            not hasattr(self, 'background_job_status_text')
            or not hasattr(self, 'console_bg')
        ):
            return

        cfg_hud = self.hud_layout
        text_visual = self.background_job_status_text

        def measure_text_width(text):
            return _vispy_text_width_pixels(text_visual, text)

        effective_canvas_width = self.canvas.size[0] - _sidebar_room(self)
        right_padding = cfg_hud.get("background_job_status_right_padding", 20.0)
        available_width = max(
            1.0,
            effective_canvas_width - cfg_hud["console_text_x"] - right_padding,
        )

        current_text = text_visual.text or ""
        previous_rendered_text = getattr(
            self,
            '_background_job_status_rendered_text',
            None,
        )
        if previous_rendered_text is None or current_text != previous_rendered_text:
            logical_text = current_text.replace(
                '\r\n', ' '
            ).replace('\r', ' ').replace('\n', ' ')
            self._background_job_status_logical_text = logical_text
        else:
            logical_text = getattr(
                self,
                '_background_job_status_logical_text',
                current_text,
            )

        rendered_text = _wrap_console_text_for_display(
            logical_text,
            available_width,
            measure_text_width,
        )
        self._background_job_status_rendered_text = rendered_text
        if current_text != rendered_text:
            text_visual.text = rendered_text

        box_bottom = self.console_bg.center[1] + self.console_bg.height / 2.0
        text_visual.pos = (
            cfg_hud["console_text_x"],
            box_bottom + cfg_hud.get("background_job_status_gap", 8.0),
        )
        text_visual.visible = bool(logical_text)

    def set_background_job_status(self, message):
        """Show a scheduler lifecycle message without altering command input."""
        if not hasattr(self, 'background_job_status_text'):
            return
        self.background_job_status_text.text = display_text(message)
        self.update_background_job_status_layout()
        self.canvas.update()

    def clear_background_job_status(self):
        """Hide the previous scheduler status when a new command is opened."""
        if not hasattr(self, 'background_job_status_text'):
            return
        self._background_job_status_logical_text = ""
        self._background_job_status_rendered_text = ""
        self.background_job_status_text.text = ""
        self.background_job_status_text.visible = False
        self.canvas.update()

    def load_and_simulate(self):
            """
            Loads layout. Saves original headers to cache, but uses simplified headers in memory.
            """
            if not os.path.exists(cfg.SAVED_LAYOUT_DIR):
                os.makedirs(cfg.SAVED_LAYOUT_DIR)
                
            # --- Resolve Path ---
            cache_path = resolve_selected_cache(cfg)
            print(f"Target Cache File: {cache_path}")
            if not os.path.exists(cache_path):
                raise RuntimeError(
                    f"Selected cache file does not exist: {cache_path}. "
                    "Please generate the layout cache first using EMAPSSN_Config or Layout_Cache_Generator.py."
                )
            cache_mode = getattr(cfg, 'TARGET_CACHE_MODE', 'existing')

            selected_fasta_path = cfg.NODE_FASTA_FILE
            self._selected_fasta_records = _load_selected_fasta_records(selected_fasta_path)
            self.sequences_map = _build_sequence_lookup(self._selected_fasta_records)
            selected_fasta_headers = [header for header, _ in self._selected_fasta_records]
            current_manifest = cache_manifest.build_manifest_for_files(
                selected_fasta_path, cfg.INPUT_HDF5,
                alignment_score=cfg.ALIGNMENT_SCORE, normalization=cfg.NORM_MODE,
                umap_mode=cfg.UMAP_MODE, umap_neighbors=cfg.UMAP_NEIGHBORS,
                layout_method=getattr(cfg, 'LAYOUT_METHOD', None) or 'umap',
                pacmap_neighbors=getattr(cfg, 'PACMAP_NEIGHBORS', None) or 10,
                top_edge_percent=cfg.TOP_EDGE_PERCENT,
                similarity_threshold=cfg.SIMILARITY_THRESHOLD,
            )
            stored_manifest = cache_manifest.read_manifest(
                os.path.dirname(cache_path), current_manifest["compatibility"]
            )
            self.cache_manifest = stored_manifest
            self.cache_manifest_id = stored_manifest["manifest_id"]
            cfg.CACHE_MANIFEST_ID = self.cache_manifest_id
            with h5py.File(cfg.INPUT_HDF5, "r") as raw_data:
                expected_headers, expected_edges, expected_edge_scores = prepare_network(
                    raw_data, settings=cfg, selected_fasta_headers=selected_fasta_headers,
                )

            raw_loaded = False

            # --- Try Loading Cache ---
            if cache_mode == 'existing':
                if not os.path.exists(cache_path):
                    raise RuntimeError(f"Selected cache file does not exist: {cache_path}")
                print(f"--- Found Cached Layout! ---")
                try:
                    import json
                    with h5py.File(cache_path, "r") as hf:
                        raw_headers = hf["headers"][:]
                        self.full_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                        self.pos = hf["positions"][:].astype(np.float32)

                        cache_manifest.validate_cache_hdf5(
                            hf, expected_headers, self.cache_manifest_id
                        )
                        from utilities.Cache_Metadata import validate_cache_provenance
                        self._cache_provenance = validate_cache_provenance(
                            hf.attrs, self.cache_manifest_id
                        )

                        self.n_nodes = len(self.full_headers)

                        if "colors" in hf: self.current_colors = hf["colors"][:]
                        if "sizes" in hf:
                            self.current_sizes = hf["sizes"][:].astype(np.float32)
                            # Sync/Rescale cached sizes with current cfg.NODE_SIZE
                            old_base = hf.attrs.get("base_node_size", None)
                            if old_base is not None:
                                try:
                                    old_base = float(old_base)
                                    new_base = float(cfg.NODE_SIZE)
                                    if old_base > 0 and old_base != new_base:
                                        self.current_sizes = self.current_sizes * (new_base / old_base)
                                except Exception as e:
                                    print(f"Warning: Failed to rescale cached node sizes: {e}")
                            else:
                                # Backward compatibility: if all sizes are uniform, overwrite with current cfg.NODE_SIZE
                                if len(self.current_sizes) > 0:
                                    first_val = self.current_sizes[0]
                                    if np.allclose(self.current_sizes, first_val):
                                        self.current_sizes.fill(cfg.NODE_SIZE)
                                    else:
                                        # Rescale assuming old base was 10.0
                                        new_base = float(cfg.NODE_SIZE)
                                        self.current_sizes = self.current_sizes * (new_base / 10.0)
                        if "shapes" in hf:
                            raw_shapes = hf["shapes"][:]
                            self.current_shapes = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in raw_shapes], dtype=object)
                        if "visible_mask" in hf: self.visible_mask = hf["visible_mask"][:]
                        if "node_render_order" in hf:
                            self.node_render_order = cache_manifest.validate_node_render_order(
                                hf["node_render_order"][:], self.n_nodes
                            ).copy()
                        if "cluster_labels" in hf: self.cluster_labels = hf["cluster_labels"][:]

                        # --- Load Metadata from Cache ---
                        self.metadata = {}
                        if "metadata" in hf:
                            meta_group = hf["metadata"]
                            for prop_name in meta_group.keys():
                                ds = meta_group[prop_name]
                                prop_type = ds.attrs.get("type", "text")
                                raw_vals = ds[:]
                                if prop_type == "number":
                                    values = raw_vals.astype(np.float64)
                                else:
                                    values = np.array([v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in raw_vals], dtype=object)
                                self.metadata[prop_name] = {
                                    "type": prop_type,
                                    "values": values
                                }

                        # --- Load Custom Dynamic Attributes from Cache (Root Level) ---
                        if not hasattr(self, '_cacheable_attrs'):
                            self._cacheable_attrs = set()

                        # Register manually configured attributes
                        for attr_name in CUSTOM_ATTRIBUTES_INIT.keys():
                            self._cacheable_attrs.add(attr_name)

                        # Scan root-level keys for any non-core custom datasets
                        CORE_DATASETS = {
                            "headers", "positions", "colors", "sizes", "shapes",
                            "visible_mask", "cluster_labels", "group_labels", "metadata",
                            "connectivity", "edge_scores", "node_render_order"
                        }
                        for key in hf.keys():
                            if key not in CORE_DATASETS:
                                ds = hf[key]
                                if "is_json" in ds.attrs and ds.attrs["is_json"]:
                                    import json
                                    raw_val = ds[()]
                                    if isinstance(raw_val, bytes):
                                        raw_val = raw_val.decode('utf-8')
                                    setattr(self, key, json.loads(raw_val))
                                else:
                                    setattr(self, key, ds[:])
                                self._cacheable_attrs.add(key)

                        # --- Safely decode strings/bytes ---
                        if "group_labels" in hf:
                            gl_data = hf["group_labels"][()]
                            if isinstance(gl_data, bytes):
                                gl_data = gl_data.decode('utf-8')
                            self.group_labels = [set(g) for g in json.loads(gl_data)]

                        if "last_cluster_params" in hf.attrs:
                            val = hf.attrs["last_cluster_params"]
                            if isinstance(val, bytes): val = val.decode('utf-8')
                            if isinstance(val, str) and val.startswith('['):
                                self.last_cluster_params = tuple(json.loads(val))
                            else:
                                self.last_cluster_params = tuple(val)

                    self.edges = expected_edges.astype(np.int32, copy=False)
                    self.edge_scores = expected_edge_scores.astype(np.float32, copy=False)

                    base_box = np.sqrt(self.n_nodes) * 2.5 + 5.0
                    self.box_limit = base_box * cfg.BOX_SCALE

                    raw_loaded = True

                except Exception as e:
                    raise RuntimeError(
                        f"Selected cache is incompatible or invalid: {e}. "
                        "Choose '(New Layout Cache)' in EMAP-SSN Configuration."
                    ) from e

            if not raw_loaded:
                raise RuntimeError(
                    f"Selected cache could not be loaded: {cache_path}. "
                    "Please regenerate the layout cache using EMAPSSN_Config or Layout_Cache_Generator.py."
                )

            self._init_colors()
            # Index the topology now that the layout has arrived, so the first
            # selection does not pay for the build.
            self._adjacency_index()

    def _init_colors(self):
        import matplotlib.colors as mcolors
        
        # Only initialize if they weren't loaded from the cache
        if not hasattr(self, 'current_colors'):
            n_rgba = mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)
            self.current_colors = np.tile(n_rgba, (self.n_nodes, 1)).astype(np.float32)
        if not hasattr(self, 'current_sizes'): self.current_sizes = np.full(self.n_nodes, cfg.NODE_SIZE, dtype=np.float32)
        if not hasattr(self, 'current_shapes'): self.current_shapes = np.full(self.n_nodes, 'disc', dtype=object)
        if not hasattr(self, 'visible_mask'): self.visible_mask = np.ones(self.n_nodes, dtype=bool)
        if not hasattr(self, 'node_render_order'):
            self.node_render_order = np.arange(self.n_nodes, dtype=np.int32)
        if not hasattr(self, 'redo_stack'): self.redo_stack = []
        if not hasattr(self, 'selected_indices'): self.selected_indices = []
        if not hasattr(self, 'cluster_labels'): self.cluster_labels = None
        if not hasattr(self, 'group_labels'): self.group_labels = [set() for _ in range(self.n_nodes)]
        if not hasattr(self, 'metadata'): self.metadata = {}
        
        # Layout generation writes the initial metadata group, so the viewer only
        # orders what the cache provides.  "Length" is the first property.
        if self.metadata and "Length" in self.metadata:
            ordered_metadata = {"Length": self.metadata["Length"]}
            for k, v in self.metadata.items():
                if k != "Length":
                    ordered_metadata[k] = v
            self.metadata = ordered_metadata
            
        # Initialize Dynamic Registry
        if not hasattr(self, '_cacheable_attrs'):
            self._cacheable_attrs = set()
            
        # Initialize manual custom attributes from the top-level section
        for attr_name, default_val in CUSTOM_ATTRIBUTES_INIT.items():
            if not hasattr(self, attr_name):
                setattr(self, attr_name, default_val)
            self._cacheable_attrs.add(attr_name)

    def _get_current_state(self, share_with=None):
        """Helper to package the entire visual and spatial state.

        The state is a history entry, and nothing may change what it holds
        afterwards: its arrays are read-only copies, and its groups a
        _GroupLabelsSnapshot. ``share_with``, an earlier full-state entry,
        lends the new entry each array (and the groups) that has not changed
        since that entry was taken; see _history_copy.
        """
        previous = share_with if isinstance(share_with, dict) else {}
        previous_metadata = previous.get('metadata')
        if not isinstance(previous_metadata, dict):
            previous_metadata = {}

        def previous_values(name):
            entry = previous_metadata.get(name)
            return entry.get('values') if isinstance(entry, dict) else None

        return {
            'pos': _history_copy(self.pos, previous.get('pos')) if hasattr(self, 'pos') else None,
            'visible_mask': _history_copy(self.visible_mask, previous.get('visible_mask')) if hasattr(self, 'visible_mask') else None,
            'colors': _history_copy(self.current_colors, previous.get('colors')) if hasattr(self, 'current_colors') else None,
            'sizes': _history_copy(self.current_sizes, previous.get('sizes')) if hasattr(self, 'current_sizes') else None,
            'shapes': _history_copy(self.current_shapes, previous.get('shapes')) if hasattr(self, 'current_shapes') else None,
            'node_render_order': _history_copy(self.node_render_order, previous.get('node_render_order')) if hasattr(self, 'node_render_order') else None,
            'clusters': _history_copy(self.cluster_labels, previous.get('clusters')) if getattr(self, 'cluster_labels', None) is not None else None,
            'groups': _GroupLabelsSnapshot.capture(self.group_labels, previous.get('groups')) if getattr(self, 'group_labels', None) is not None else None,
            'last_cluster_params': self.last_cluster_params if getattr(self, 'last_cluster_params', None) is not None else None,
            'metadata': {k: {'type': v['type'], 'values': _history_copy(v['values'], previous_values(k))} for k, v in self.metadata.items()} if getattr(self, 'metadata', None) else {},
            # Just the name of the property the metadata HUD displays, which `spectrum` and `meta show` set.
            'meta_display_prop': getattr(self, 'meta_display_prop', None),
            '_custom_data': self._get_custom_attributes_snapshot(previous.get('_custom_data'))
        }

    def _apply_state(self, state):
        """Helper to unpack a state dictionary and apply it to the viewer."""
        if state['pos'] is not None: self.pos = state['pos'].copy()
        if state['visible_mask'] is not None: self.visible_mask = state['visible_mask'].copy()
        if state['colors'] is not None: self.current_colors = state['colors'].copy()
        if state['sizes'] is not None: self.current_sizes = state['sizes'].copy()
        if state['shapes'] is not None: self.current_shapes = state['shapes'].copy()
        if state.get('node_render_order') is not None:
            self.node_render_order = cache_manifest.validate_node_render_order(
                state['node_render_order'], self.n_nodes
            ).copy()
        else:
            self.node_render_order = np.arange(self.n_nodes, dtype=np.int32)
        
        if state['clusters'] is not None: 
            self.cluster_labels = state['clusters'].copy()
            self.last_cluster_params = state.get('last_cluster_params')
        else:
            self.cluster_labels = None
            self.last_cluster_params = None

        groups = state.get('groups')
        if isinstance(groups, _GroupLabelsSnapshot):
            self.group_labels = groups.to_list()
        elif groups is not None:
            # A list of sets, as states saved before _GroupLabelsSnapshot held.
            self.group_labels = [g.copy() for g in groups]
        else:
            self.group_labels = [set() for _ in range(self.n_nodes)]
            
        if state.get('metadata') is not None:
            self.metadata = {k: {'type': v['type'], 'values': v['values'].copy()} for k, v in state['metadata'].items()}
        else:
            self.metadata = {}
            
        if '_custom_data' in state and state['_custom_data'] is not None:
            self._apply_custom_attributes_snapshot(state['_custom_data'])

        # The HUD follows the property the restored state displayed; the redraw
        # of its value comes with the undo or redo that called this.
        if 'meta_display_prop' in state and state['meta_display_prop'] != getattr(self, 'meta_display_prop', None):
            self._restore_metadata_hud_property(state['meta_display_prop'])

        # Clean up any active selections if those nodes are now hidden in this restored state
        if hasattr(self, 'selected_indices'):
            self.selected_indices = [i for i in self.selected_indices if self.visible_mask[i]]
            
        self.update_selection_visual()
        self.update_edges()

    def _get_custom_attributes_snapshot(self, share_with=None):
        """Copy the cacheable attributes undo restores. An array may be the one
        ``share_with``, an earlier entry's custom data, holds (see
        _history_copy); anything else is deep-copied."""
        if not getattr(self, '_cacheable_attrs', None):
            return {}
        previous = share_with if isinstance(share_with, dict) else {}
        snapshot = {}
        for attr_name in self._cacheable_attrs - UNDO_EXCLUDED_ATTRIBUTES:
            val = getattr(self, attr_name, None)
            if isinstance(val, np.ndarray):
                snapshot[attr_name] = _history_copy(val, previous.get(attr_name))
            else:
                import copy
                snapshot[attr_name] = copy.deepcopy(val)
        return snapshot

    def _apply_custom_attributes_snapshot(self, snapshot):
        if not hasattr(self, '_cacheable_attrs'):
            self._cacheable_attrs = set()
        for attr_name, val in snapshot.items():
            if attr_name in UNDO_EXCLUDED_ATTRIBUTES:
                continue
            if isinstance(val, np.ndarray):
                setattr(self, attr_name, val.copy())
            else:
                import copy
                setattr(self, attr_name, copy.deepcopy(val))
            self._cacheable_attrs.add(attr_name)

    def _append_history_entry(self, entry):
        """Append a full-state or compact history entry and clear redo state."""
        self.position_history.append(entry)
        if len(self.position_history) > 50:
            self.position_history.pop(0)
        self.redo_stack.clear()

    def _save_metadata_cell_history(self, column, row, before, after):
        """Record one metadata cell edit without copying the full network state."""
        self._append_history_entry({
            "_history_kind": "metadata_cell",
            "column": column,
            "row": int(row),
            "before": before,
            "after": after,
        })

    def _save_metadata_column_history(self, removed_columns, hud_property=None):
        """Record deleted metadata columns and their original ordering."""
        self._append_history_entry({
            "_history_kind": "metadata_columns",
            "columns": removed_columns,
            "hud_property": hud_property,
        })

    def _set_metadata_hud_property(self, property_name):
        """Synchronize the optional clicked-node metadata HUD with history."""
        self.meta_display_prop = property_name
        display = getattr(self, "hud_displays", {}).get("meta_display")
        if display is None:
            return
        if property_name is None:
            display.hide()
            return

        node_idx = getattr(self, "selected_node_idx", None)
        entry = getattr(self, "metadata", {}).get(property_name)
        if entry is None or node_idx is None:
            display.show(f"{property_name}: -")
            return

        value = entry["values"][node_idx]
        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            value_text = "N/A"
        else:
            value_text = str(value).strip() or "N/A"
        display.show(f"{property_name}: {value_text}")

    def _restore_metadata_hud_property(self, property_name):
        """Set the HUD's property as undo or redo restores it, unless the user closed the HUD.

        A HUD closed with `meta show off` stays closed: undo and redo may close
        it or switch the property it shows, but never reopen it. The next
        `meta show` or `spectrum` opens it again.
        """
        if (property_name is not None and getattr(self, "meta_display_prop", None) is None
                and getattr(self, "meta_display_dismissed", False)):
            return
        self._set_metadata_hud_property(property_name)

    def _apply_metadata_history_entry(self, entry, undo):
        """Apply one compact metadata history entry in either direction."""
        kind = entry.get("_history_kind")
        if kind == "metadata_cell":
            column = entry["column"]
            metadata_entry = self.metadata.get(column)
            if metadata_entry is not None:
                value = entry["before"] if undo else entry["after"]
                metadata_entry["values"][entry["row"]] = value
        elif kind == "metadata_columns":
            removed_columns = entry["columns"]
            if undo:
                restored_items = list(self.metadata.items())
                for removed in sorted(
                    removed_columns, key=lambda item: item["index"]
                ):
                    restored_entry = {
                        "type": removed["type"],
                        "values": removed["values"].copy(),
                    }
                    restored_items.insert(
                        min(removed["index"], len(restored_items)),
                        (removed["name"], restored_entry),
                    )
                self.metadata = dict(restored_items)
                if entry.get("hud_property"):
                    self._restore_metadata_hud_property(entry["hud_property"])
            else:
                for removed in removed_columns:
                    self.metadata.pop(removed["name"], None)
                if entry.get("hud_property"):
                    self._set_metadata_hud_property(None)
        else:
            raise ValueError(f"Unsupported history entry kind: {kind}")

    def _latest_full_history_state(self):
        """The newest full-state entry on the undo history, or None."""
        for entry in reversed(self.position_history):
            if not entry.get("_history_kind"):
                return entry
        return None

    def _save_state(self):
        """Saves current state to history and clears redo stack."""
        # The newest full state is the likeliest to match: since it was taken,
        # usually just the last command's field has changed.
        self._append_history_entry(
            self._get_current_state(share_with=self._latest_full_history_state())
        )

    def _do_undo(self, announce=True):
        """Undo one step; return whether there was one.

        With announce False, as for each step of `undo N`, it neither reports
        the step nor refreshes the metadata HUD and the web pages: the command
        calls _refresh_after_history once the steps are done.
        """
        if len(self.position_history) > 0:
            state = self.position_history.pop()
            if state.get("_history_kind"):
                self.redo_stack.append(state)
                self._apply_metadata_history_entry(state, undo=True)
            else:
                # The current state differs from the one undo restores in
                # what the undone command changed; it shares the rest.
                self.redo_stack.append(self._get_current_state(share_with=state))
                self._apply_state(state)
            msg = Message("Undo successful.")
            changed = True
        else:
            msg = Message("Nothing to undo.")
            changed = False
        if announce:
            Command_Engine.show_status(self, msg)
            print(msg)
            if changed:
                self._refresh_after_history()
        return changed

    def _refresh_after_history(self):
        """What follows undo or redo steps: the metadata HUD redrawn, the web pages told."""
        self._redraw_metadata_hud()
        self.broadcast_metadata_state()

    def _redraw_metadata_hud(self):
        """Redraw the visible `meta show` HUD for the clicked node, as undo or
        redo may have changed the value it shows."""
        display = getattr(self, "hud_displays", {}).get("meta_display")
        redraw = getattr(display, "on_node_clicked", None)
        prop = getattr(self, "meta_display_prop", None)
        node_idx = getattr(self, "selected_node_idx", None)
        if (callable(redraw) and getattr(display, "visible", False) and prop
                and node_idx is not None and prop in getattr(self, "metadata", {})):
            redraw(node_idx)

    def _do_redo(self, announce=True):
        """Redo one step; return whether there was one. announce is as for _do_undo."""
        if len(self.redo_stack) > 0:
            state = self.redo_stack.pop()
            if state.get("_history_kind"):
                self.position_history.append(state)
                self._apply_metadata_history_entry(state, undo=False)
            else:
                self.position_history.append(self._get_current_state(share_with=state))
                self._apply_state(state)
            msg = Message("Redo successful.")
            changed = True
        else:
            msg = Message("Nothing to redo.")
            changed = False
        if announce:
            Command_Engine.show_status(self, msg)
            print(msg)
            if changed:
                self._refresh_after_history()
        return changed

    def load_global_alignment(self, reuse_loaded=False):
        """
        Loads alignment using the new standalone Alignment_Manager.

        With REUSE_LOADED, the rows already loaded are kept when the MSA file
        is unchanged on disk, and only the numbering is rebuilt; `reference`
        asks for this, since a new reference changes nothing else.
        """
        import Alignment_Manager
        self.alignment = Alignment_Manager.Alignment_Manager(
            cfg.MSA_FILE,
            full_headers=self.full_headers,
            active_reference=self.active_reference,
            alignment_offset=self.alignment_offset,
            reuse=getattr(self, 'alignment', None) if reuse_loaded else None,
        )

    def load_embedding(self, reuse_loaded=False):
        """Match the optional embedding file's manifest to the nodes (Embedding_Manager).

        Only the manifest is read here; a command reads the per-residue rows or
        pooled vectors it needs when it runs. With REUSE_LOADED, the manifest
        already read is kept when the file is unchanged on disk.
        """
        import Embedding_Manager
        self.embedding = Embedding_Manager.Embedding_Manager(
            getattr(cfg, 'EMBED_FILE', ''),
            full_headers=self.full_headers,
            sequences=getattr(self, 'sequences_map', None),
            reuse=getattr(self, 'embedding', None) if reuse_loaded else None,
        )


    def draw_network(self):
        if len(self.edges) > 0:
            import matplotlib.colors as mcolors
            # Fetch custom edge color, fallback to black
            edge_rgba = list(mcolors.to_rgba(getattr(cfg, 'EDGE_COLOR', '#000000')))
            edge_rgba[3] = cfg.EDGE_ALPHA # Apply transparency

            # Sorts the edges by score once; update_edges draws them.
            self.edge_layer = ViewerEdges(
                self.view.scene, self.edges, getattr(self, 'edge_scores', None),
                self.pos, color=tuple(edge_rgba), width=cfg.EDGE_WIDTH,
                motion=MotionDetail(setting=fewer_edges_while_moving(cfg)),
            )
        else:
            self.edge_layer = None

        # Left-click rings are interleaved into this marker visual immediately
        # before their nodes so their layer relationship remains exact.
        self.left_click_highlight = None
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.update_nodes()
        
        # --- MODIFIED: Parent changed to canvas.scene ---
        self.tooltip = scene.visuals.Text(
            text="",
            color=cfg.TEXT_COLOR,
            pos=(0, 0),
            anchor_x='left',
            face=self.vispy_ui_face,
            font_size=self._tooltip_font_size_points(),
            parent=self.canvas.scene,
        )

        # ---> NEW: Visuals for Selection Feedback <---
        self.selection_box = scene.visuals.Line(
            method='gl', parent=self.view.scene, name='selection_box',
        )
        self._color_selection_box()
        self.selection_box.visible = False
        
        self.selection_highlight = scene.visuals.Markers(parent=self.view.scene)
        # Initialize with a single dummy point so Vispy builds the internal vertex buffers
        self.selection_highlight.set_data(pos=np.array([[0.0, 0.0]], dtype=np.float32))
        self.selection_highlight.set_gl_state('translucent', depth_test=False)
        self.selection_highlight.visible = False

    def _move_dragged_nodes(self):
        """A drag move under Low Resource Mode's Update Colors After Mouse Release: the
        markers and the edges go where the nodes now are, with the colours
        update_nodes last gave them. The selection and neighbour colouring,
        and the hover pick, are worked out again on release
        (update_selection_visual)."""
        data = getattr(self, '_submitted_marker_data', None)
        order = getattr(self, '_submitted_marker_node_order', None)
        if data is None or order is None or not len(order):
            self.update_nodes()
        else:
            self.markers.set_data(pos=self.pos[order], **data)
            # Both were made for the markers' old places; release remakes them.
            self._hover_markers = None
            self._marker_picker = None
            self.canvas.update()
            self._update_hud_elements()
        self.update_edges()

    def update_selection_visual(self, broadcast=True):
        """Triggers a node update to draw selection edges and ensures the old highlight is hidden.

        broadcast=False leaves the web pages, such as the Meta Data sheet, to
        hear of the selection later, as a node drag under Low Resource Mode's
        Update Web Pages After Mouse Release does on release.
        """
        if hasattr(self, 'selection_highlight'):
            self.selection_highlight.visible = False

        self.update_nodes()
        self.update_edges()

        # Broadcast the selection change to SSE clients
        if broadcast:
            self.broadcast_event({"type": "selection_changed", "indices": self.selected_indices})
    
    def position_slider_overlay(self):
        """Lay out the slider row: the Zoom box at the right, centred on
        the row, and the threshold slider in the room left of it."""
        if hasattr(self, 'slider_overlay') and hasattr(self, 'canvas'):
            canvas_w, canvas_h = self.canvas.size
            right = canvas_w - _sidebar_room(self) - self.hud_layout["status_x_offset"]
            box = getattr(self, 'zoom_box', None)
            box_room = 0
            if box is not None:
                box.ensurePolished()  # Measured in the stylesheet's font.
                size = box.sizeHint()
                top = _slider_row_centre(canvas_h) - size.height() / 2.0
                box.setGeometry(round(right - size.width()), round(top), size.width(), size.height())
                box_room = size.width() + SLIDER_BOX_GAP
            overlay_w = max(100, round(right - box_room - SLIDER_ROW_LEFT))
            overlay_y = canvas_h - SLIDER_ROW_HEIGHT - SLIDER_ROW_BOTTOM
            self.slider_overlay.setGeometry(SLIDER_ROW_LEFT, overlay_y, overlay_w, SLIDER_ROW_HEIGHT)
            # The info card, if open, stays above its button inside the canvas.
            button = getattr(self, 'controls_button', None)
            if button is not None and button.card_is_open():
                self.slider_overlay.layout().activate()
                button.place_card()

    def on_slider_value_changed(self, value):
        self.current_slider_threshold = self.threshold_box.value_at(value)
        self.threshold_box.show_value(self.current_slider_threshold)
        # Under Low Resource Mode's Update Colors After Mouse Release a drag shows the
        # threshold as it goes and redraws the edges once, when let go
        # (_on_slider_released).
        slider = getattr(self, 'slider', None)
        if low_resource_drawing(cfg) and slider is not None and slider.isSliderDown():
            return
        self.update_edges()

    def _on_slider_released(self):
        """Redraw the edges for the threshold a slider drag under Update Colors After Mouse Release ended on."""
        if low_resource_drawing(cfg):
            self.update_edges()

    def _set_threshold_from_box(self, threshold):
        """Draw the edges for a threshold set in the box, and move the slider to it."""
        self.current_slider_threshold = threshold
        blocked = self.slider.blockSignals(True)
        try:
            self.slider.setValue(self.threshold_box.position_of(threshold))
        finally:
            self.slider.blockSignals(blocked)
        self.threshold_box.show_value(threshold)  # What Esc puts back.
        self.update_edges()

    def update_edges(self):
        from Viewer_Visual_State import edge_stages, edges_follow_selection
        edge_layer = getattr(self, 'edge_layer', None)
        if edge_layer is None:
            self._refresh_session_card()
            return
        # The edge layer re-sends only what changed: the node positions (a
        # small texture), the visible edges, or the selection's edge list. A
        # threshold change only changes how many score-sorted edges it draws.
        if edges_follow_selection(self, cfg):
            active_edges, _ = edge_stages(self, cfg, cache=True)
            drawn = edge_layer.show_edges(self.pos, active_edges)
        else:
            threshold = getattr(
                self, 'current_slider_threshold',
                getattr(cfg, 'SIMILARITY_THRESHOLD', 0.0),
            )
            drawn = edge_layer.show_threshold(
                self.pos, getattr(self, 'visible_mask', None), threshold,
            )
        edge_layer.visible = drawn > 0
        self._refresh_session_card()

    def promote_nodes(self, indices):
        """Move one node group to the top of the persistent render order."""
        values = np.asarray(indices)
        if values.dtype == np.bool_:
            if values.ndim != 1 or len(values) != self.n_nodes:
                raise ValueError("Node promotion mask must match the node count.")
            promoted = np.flatnonzero(values)
        else:
            promoted = values.astype(np.int64, copy=False).reshape(-1)

        promoted = np.unique(promoted)
        if promoted.size == 0:
            return False
        if np.any(promoted < 0) or np.any(promoted >= self.n_nodes):
            raise ValueError("Node promotion indices are outside the network.")

        current_order = getattr(self, 'node_render_order', None)
        if current_order is None:
            current_order = np.arange(self.n_nodes, dtype=np.int32)
        else:
            current_order = cache_manifest.validate_node_render_order(
                current_order, self.n_nodes
            )

        keep_mask = ~np.isin(current_order, promoted)
        new_order = np.concatenate(
            (current_order[keep_mask], np.sort(promoted))
        ).astype(np.int32, copy=False)
        changed = not np.array_equal(new_order, current_order)
        self.node_render_order = new_order
        return changed

    def _valid_node_indices(self, values):
        """Return unique in-range node indices as an ascending array."""
        if values is None:
            return np.empty(0, dtype=np.int32)
        array = np.asarray(list(values) if isinstance(values, set) else values)
        if array.size == 0:
            return np.empty(0, dtype=np.int32)
        indices = array.astype(np.int64, copy=False).reshape(-1)
        indices = indices[(indices >= 0) & (indices < self.n_nodes)]
        # A presence mask sorts and deduplicates in one pass over the nodes;
        # np.unique hashes or sorts, which costs more for a large selection.
        present = np.zeros(self.n_nodes, dtype=bool)
        present[indices] = True
        return np.flatnonzero(present).astype(np.int32, copy=False)

    def _selected_node_indices(self):
        """Return command and box-selected node indices."""
        return self._valid_node_indices(getattr(self, 'selected_indices', None))

    def _left_click_node_indices(self):
        """Return ordinary and metadata-driven left-click highlights."""
        highlight_parts = []
        highlighted = self._valid_node_indices(
            getattr(self, 'left_click_highlight_indices', None)
        )
        if len(highlighted):
            highlight_parts.append(highlighted)

        clicked_index = getattr(self, 'selected_node_idx', None)
        if clicked_index is not None:
            highlight_parts.append(np.array([clicked_index], dtype=np.int64))

        if not highlight_parts:
            return np.empty(0, dtype=np.int32)
        return self._valid_node_indices(np.concatenate(highlight_parts))

    def _connected_node_identification_enabled(self, boundary_rgba=None, connected_rgba=None):
        """Return whether connected-node border and ordering work is enabled."""
        import matplotlib.colors as mcolors

        if boundary_rgba is None:
            boundary_rgba = mcolors.to_rgba(
                getattr(cfg, 'NODE_BOUNDARY_COLOR', '#000000')
            )
        if connected_rgba is None:
            connected_rgba = mcolors.to_rgba(
                getattr(cfg, 'CONNECTED_NODE_COLOR', '#ff0000')
            )
        return tuple(boundary_rgba) != tuple(connected_rgba)

    def _adjacency_index(self):
        """Return the CSR adjacency index over the complete topology.

        The index is built once per edge list. Holding the indexed array keeps
        the identity test exact, so a replaced topology rebuilds and an
        unchanged one never does.
        """
        edges = getattr(self, 'edges', None)
        cached = getattr(self, '_adjacency_index_cache', None)
        if cached is not None and cached[0] is edges:
            return cached[1], cached[2]
        indptr, indices = _build_adjacency_index(edges, self.n_nodes)
        self._adjacency_index_cache = (edges, indptr, indices)
        return indptr, indices

    def _connected_to_selected_indices(self, selected):
        """Return nodes adjacent to selected nodes in the complete topology.

        The neighbour counts follow the selection from call to call
        (_SelectionNeighbours), so each mouse move of a box selection costs
        what it adds or removes. A new topology starts the counts afresh.
        """
        nodes = self._valid_node_indices(selected).astype(np.int64, copy=False)
        indptr, indices = self._adjacency_index()
        tracker = getattr(self, '_selection_neighbours', None)
        if tracker is None or tracker.indices is not indices:
            tracker = _SelectionNeighbours(indptr, indices, self.n_nodes)
            self._selection_neighbours = tracker
        return tracker.neighbours_of(nodes)

    def visible_node_render_order(self, identify_connected=None):
        """Return the effective low-to-high order for currently visible nodes."""
        base_order = getattr(self, 'node_render_order', None)
        if base_order is None:
            base_order = np.arange(self.n_nodes, dtype=np.int32)
        else:
            base_order = np.asarray(base_order, dtype=np.int32)
            if base_order.ndim != 1 or len(base_order) != self.n_nodes:
                raise ValueError("Node render order does not match the node count.")

        visible = np.asarray(self.visible_mask, dtype=bool)
        selected = self._selected_node_indices()
        left_clicked = self._left_click_node_indices()

        selected = selected[visible[selected]]
        left_clicked = left_clicked[visible[left_clicked]]
        selected_for_connections = selected
        if len(left_clicked):
            selected = selected[~np.isin(selected, left_clicked)]

        if identify_connected is None:
            identify_connected = self._connected_node_identification_enabled()
        if identify_connected and len(selected_for_connections):
            connected = self._connected_to_selected_indices(
                selected_for_connections
            )
            connected = connected[visible[connected]]
            if len(left_clicked):
                connected = connected[~np.isin(connected, left_clicked)]
        else:
            connected = np.empty(0, dtype=np.int32)

        elevated = np.zeros(self.n_nodes, dtype=bool)
        elevated[connected] = True
        elevated[selected] = True
        elevated[left_clicked] = True
        remaining = base_order[visible[base_order] & ~elevated[base_order]]
        return np.concatenate(
            (
                remaining,
                np.sort(connected),
                np.sort(selected),
                np.sort(left_clicked),
            )
        ).astype(np.int32, copy=False)

    def node_outlines(self, identify_connected=None):
        """Return each node's outline colour (RGBA) and width in pixels, by node index.

        Every node has the boundary colour and width set in Config. Selected
        nodes have the hover colour and, while connected-node identification
        is on, their neighbours the connected colour, both 2 pixels wide.
        update_nodes draws these, and print's SVG copies them.
        """
        import matplotlib.colors as mcolors
        bound_rgba = mcolors.to_rgba(getattr(cfg, 'NODE_BOUNDARY_COLOR', '#000000'))
        conn_rgba = mcolors.to_rgba(getattr(cfg, 'CONNECTED_NODE_COLOR', '#ff0000'))
        if identify_connected is None:
            identify_connected = self._connected_node_identification_enabled(
                bound_rgba, conn_rgba
            )

        edge_colors = np.zeros((self.n_nodes, 4), dtype=np.float32)
        edge_colors[:] = bound_rgba

        # ---> NEW: Fetch custom boundary width, fallback to 0.5
        b_width = getattr(cfg, 'NODE_BOUNDARY_WIDTH', 0.5)
        edge_widths = np.full(self.n_nodes, b_width, dtype=np.float32)

        if getattr(self, 'selected_indices', None) is not None and len(self.selected_indices) > 0:
            hover_rgba = mcolors.to_rgba(cfg.HOVER_COLOR)
            # Converted once: indexing with the list itself converts it again.
            selected = self._selected_node_indices()

            if identify_connected:
                neighbor_indices = self._connected_to_selected_indices(selected)
                if len(neighbor_indices) > 0:
                    edge_colors[neighbor_indices] = conn_rgba
                    edge_widths[neighbor_indices] = 2.0

            edge_colors[selected] = hover_rgba
            edge_widths[selected] = 2.0
        return edge_colors, edge_widths

    def update_nodes(self):
        colors = self.current_colors.copy()
        import matplotlib.colors as mcolors
        # A PNG capture (commands/print.py) sets transient_marks_hidden to leave
        # out the hover colour and the left-click rings. Only their drawing
        # stops: the hover and click state stays, so they return when it is cleared.
        transient_marks = not getattr(self, 'transient_marks_hidden', False)
        if transient_marks and getattr(self, 'hovered_node_idx', None) is not None:
            colors[self.hovered_node_idx] = mcolors.to_rgba(cfg.HOVER_COLOR)
            
        sizes = getattr(self, 'current_sizes', cfg.NODE_SIZE)
        shapes = getattr(self, 'current_shapes', np.full(self.n_nodes, 'disc', dtype=object))

        identify_connected = self._connected_node_identification_enabled()
        edge_colors, edge_widths = self.node_outlines(identify_connected)

        # Submit every node attribute in the same effective low-to-high order.
        draw_order = self.visible_node_render_order(
            identify_connected=identify_connected
        )
        self._submitted_visible_node_order = draw_order.copy()
        self._submitted_marker_data = None
        self._hover_markers = None
        previous_picker, self._marker_picker = getattr(self, '_marker_picker', None), None
        if len(draw_order) == 0:
            self._submitted_marker_node_order = np.empty(0, dtype=np.int32)
            self._submitted_marker_ring_mask = np.empty(0, dtype=bool)
            self.markers.visible = False
        else:
            self.markers.visible = True

            # A clicked node's enlarged translucent ring and the node itself
            # must share one draw call. Interleave each ring directly before
            # its node; separate VisPy visuals can only be layered as wholes.
            left_clicked = self._left_click_node_indices() if transient_marks else ()
            if len(left_clicked):
                ring_before = np.isin(draw_order, left_clicked)
            else:
                ring_before = np.zeros(len(draw_order), dtype=bool)
            marker_count = len(draw_order) + int(np.count_nonzero(ring_before))
            marker_node_order = np.empty(marker_count, dtype=np.int32)
            marker_ring_mask = np.zeros(marker_count, dtype=bool)

            node_slots = np.arange(len(draw_order)) + np.cumsum(ring_before)
            marker_node_order[node_slots] = draw_order
            ring_slots = node_slots[ring_before] - 1
            marker_node_order[ring_slots] = draw_order[ring_before]
            marker_ring_mask[ring_slots] = True

            self._submitted_marker_node_order = marker_node_order.copy()
            self._submitted_marker_ring_mask = marker_ring_mask.copy()

            marker_face_colors = colors[marker_node_order].copy()
            marker_edge_colors = edge_colors[marker_node_order].copy()
            marker_edge_widths = edge_widths[marker_node_order].copy()
            if isinstance(sizes, np.ndarray):
                marker_sizes = sizes[marker_node_order].copy()
            else:
                marker_sizes = np.full(marker_count, sizes, dtype=np.float32)

            if len(ring_slots):
                marker_face_colors[ring_slots, 3] *= 0.5
                marker_edge_colors[ring_slots] = 0.0
                marker_edge_widths[ring_slots] = 0.0
                marker_sizes[ring_slots] *= 2.0
            
            # One name when every node has the same shape: VisPy looks a list
            # up name by name in Python (3 ms at 60,000 nodes).
            marker_shapes = shapes[marker_node_order]
            symbol = uniform_symbol(marker_shapes)
            marker_data = dict(
                face_color=marker_face_colors,
                edge_color=marker_edge_colors,
                size=marker_sizes,
                edge_width=marker_edge_widths,
                symbol=marker_shapes.tolist() if symbol is None else symbol,
            )
            self.markers.set_data(pos=self.pos[marker_node_order], **marker_data)
            # What a node drag under Update Colors After Mouse Release sends again, moved (_move_dragged_nodes).
            self._submitted_marker_data = marker_data
            # What a hover then changes is two face colours (_show_hover_change).
            self._hover_markers = HoverMarkers(
                self, self.current_colors, sizes, shapes, (edge_colors, edge_widths),
                identify_connected, self._left_click_node_indices(),
                marker_node_order, marker_ring_mask,
            )
            # Which node the cursor is on, as these markers are drawn (pick_hovered_node).
            if symbol is not None:
                codes = symbol_codes_of([symbol] * marker_count)
            elif self._hover_markers.problem is None:
                codes = self.markers._data['a_symbol'].astype(np.int64)
            else:
                codes = symbol_codes_of(marker_shapes.tolist())
            self._marker_picker = MarkerPicker(
                marker_node_order, marker_ring_mask, self.pos,
                marker_sizes, marker_edge_widths, codes, getattr(self.markers, 'antialias', 1.0),
                previous=previous_picker,
            )
        self.markers.set_gl_state('translucent', depth_test=False)

        self.canvas.update()
        
        # ---> NEW: Force HUD to instantly sync whenever visual state changes
        self._update_hud_elements()

    def create_hud(self):
        cfg_hud = self.hud_layout
        hud_font_size = self._hud_font_size_points()

        self.console_bg = scene.visuals.Rectangle(
            center=(
                cfg_hud["console_bg_left_offset"]
                + cfg_hud["console_bg_min_width"] / 2.0,
                cfg_hud["console_text_y"]
                + cfg_hud.get("console_bg_y_offset", 0.0),
            ),
            width=cfg_hud["console_bg_min_width"],
            height=cfg_hud["console_bg_height"],
            radius=min(
                max(cfg_hud["console_bg_radius"], 0.0),
                cfg_hud["console_bg_min_width"] / 2.0,
                cfg_hud["console_bg_height"] / 2.0,
            ),
            parent=self.canvas.scene
        )
        self._color_command_box()
        self.console_bg.visible = False

        self._console_logical_text = ""
        self._console_rendered_text = ""
        self.console_text = scene.visuals.Text(
            text="", 
            bold=True, 
            face=self.vispy_monospace_face,
            font_size=hud_font_size,
            color=cfg.TEXT_COLOR, 
            pos=(cfg_hud["console_text_x"], cfg_hud["console_text_y"]), 
            anchor_y=cfg_hud["console_text_anchor_y"], 
            anchor_x=cfg_hud["console_text_anchor_x"], 
            parent=self.canvas.scene
        )

        self._background_job_status_logical_text = ""
        self._background_job_status_rendered_text = ""
        self.background_job_status_text = scene.visuals.Text(
            text="",
            bold=True,
            face=self.vispy_monospace_face,
            font_size=hud_font_size,
            color=cfg.TEXT_COLOR,
            pos=(cfg_hud["console_text_x"], cfg_hud["console_text_y"]),
            anchor_y=cfg_hud["console_text_anchor_y"],
            anchor_x=cfg_hud["console_text_anchor_x"],
            parent=self.canvas.scene,
        )
        self.background_job_status_text.visible = False

        # The view's zoom shows in the Zoom box, by the threshold slider
        # (_build_slider_overlay); Hidden Nodes sits above it.
        self.hidden_text = scene.visuals.Text(
            text="", 
            bold=False, 
            face=self.vispy_ui_face,
            font_size=hud_font_size,
            color=cfg.TEXT_COLOR,
            pos=self._status_hud_position(0),
            anchor_y=cfg_hud["status_anchor_y"],
            anchor_x=cfg_hud["status_anchor_x"],
            parent=self.canvas.scene
        )

    def process_command(self, cmd_str, record_history=True, silent=False):
        from Viewer_Command_Portal import bind
        with bind(None):
            return Command_Engine.execute_command(self, cmd_str, record_history, silent)

    def on_key_press(self, event):
        # --- SAFE KEY DETECTION ---
        # Vispy sometimes fails to map hardware toggles (like CapsLock, NumLock) 
        # and passes None. We must ignore these to prevent attribute errors.
        if getattr(event, 'key', None) is None:
            return

        # --- NEW: Global Escape Interceptor ---
        if event.key == 'Escape':
            event.handled = True  # Block Vispy from closing the window
            if self.console_mode:
                self.console_mode = False
                self.console_bg.visible = False
                Command_Engine.show_status(self, "")
                self.canvas.update()
            return

        # --- Console Typing Mode ---
        if self.console_mode:
            event.handled = True 
            
            # Safely extract the key name as a string
            key_name = getattr(event.key, 'name', '') or ''

            # Detect modifiers for Copy/Paste safely
            is_modifier_active = 'Control' in event.modifiers or 'Meta' in event.modifiers
            is_paste = (key_name.lower() == 'v') and is_modifier_active
            is_copy = (key_name.lower() == 'c') and is_modifier_active

            if event.key in ['Enter', 'Return']:
                self.process_command(self.input_buffer)
                self.console_mode = False; self.console_bg.visible = False; self.canvas.update()
                
            elif event.key == 'Backspace':
                if self.cursor_pos > 0:
                    self.input_buffer = self.input_buffer[:self.cursor_pos-1] + self.input_buffer[self.cursor_pos:]
                    self.cursor_pos -= 1
                    self._update_console_text()
                    
            elif event.key == 'Delete':
                if self.cursor_pos < len(self.input_buffer):
                    self.input_buffer = self.input_buffer[:self.cursor_pos] + self.input_buffer[self.cursor_pos+1:]
                    self._update_console_text()
                    
            elif event.key == 'Left':
                self.cursor_pos = max(0, self.cursor_pos - 1)
                self._update_console_text()
                
            elif event.key == 'Right':
                self.cursor_pos = min(len(self.input_buffer), self.cursor_pos + 1)
                self._update_console_text()
                
            elif event.key == 'Up':
                if hasattr(self, 'command_history') and self.command_history:
                    self.history_index = max(0, self.history_index - 1)
                    self.input_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.input_buffer)
                    self._update_console_text()
                    
            elif event.key == 'Down':
                if hasattr(self, 'command_history') and self.command_history:
                    self.history_index = min(len(self.command_history), self.history_index + 1)
                    if self.history_index == len(self.command_history):
                        self.input_buffer = ""
                    else:
                        self.input_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.input_buffer)
                    self._update_console_text()
                    
            elif is_paste:
                try:
                    from vispy import app as vispy_app
                    native_app = vispy_app.use_app().native
                    
                    cb_text = native_app.clipboard().text()
                    if cb_text:
                        cb_text = cb_text.replace('\n', ' ').replace('\r', '') 
                        self.input_buffer = self.input_buffer[:self.cursor_pos] + cb_text + self.input_buffer[self.cursor_pos:]
                        self.cursor_pos += len(cb_text)
                        self._update_console_text()
                except Exception as e:
                    print(f"Paste failed: {e}")
                    
            elif is_copy:
                try:
                    from vispy import app as vispy_app
                    native_app = vispy_app.use_app().native
                    
                    native_app.clipboard().setText(self.input_buffer)
                    print("Copied command to clipboard.")
                except Exception as e:
                    print(f"Copy failed: {e}")
                    
            elif len(event.text) > 0 and not is_modifier_active and event.key not in ['Shift', 'Alt']:
                self.input_buffer = self.input_buffer[:self.cursor_pos] + event.text + self.input_buffer[self.cursor_pos:]
                self.cursor_pos += len(event.text)
                self._update_console_text()
                
        # --- Opening the Console (and hotkeys) ---
        else:
            key_name = getattr(event.key, 'name', '') or ''
            is_modifier_active = 'Control' in event.modifiers or 'Meta' in event.modifiers
            
            # Handle Undo / Redo
            if (key_name.lower() == 'z') and is_modifier_active:
                self.undo_by_hand()
                event.handled = True
                return
            if (key_name.lower() == 'y') and is_modifier_active:
                self.redo_by_hand()
                event.handled = True
                return

            if event.key in ['Enter', 'Return']:
                self.clear_background_job_status()
                self.console_mode = True
                self.input_buffer = ""
                self.cursor_pos = 0
                
                if hasattr(self, 'command_history'):
                    self.history_index = len(self.command_history)
                else:
                    self.history_index = 0
                    
                self.console_bg.visible = True
                self._update_console_text()
                event.handled = True

    def on_mouse_press(self, event):
        # Clicks work while a command is typed: the command waits in
        # input_buffer, and a click's message follows it (_show_click_status).
        # ---> 1. RIGHT-CLICK LOGIC (Drag & Select) <---
        if event.button == 2:
            self.tooltip.text = "" 
            self.selected_node_idx = None
            
            # Clear or hide registered HUD displays
            for display in self.hud_displays.values():
                if getattr(display, 'on_right_click', None):
                    display.on_right_click()
            
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]

            # The node the hover highlights here (pick_hovered_node); over
            # no node the press starts a box selection.
            nearest_idx = self.pick_hovered_node(event.pos)
            is_node_clicked = nearest_idx is not None
            self.drag_start_mouse = mouse_world
            self.drag_start_screen = event.pos
            
            if is_node_clicked:
                self._save_state()

                # Modifier logic for individual node clicking
                before = list(self.selected_indices)
                if 'Shift' in event.modifiers:
                    if nearest_idx not in self.selected_indices:
                        self.selected_indices.append(nearest_idx)
                elif 'Control' in event.modifiers or 'Meta' in event.modifiers:
                    if nearest_idx in self.selected_indices:
                        self.selected_indices.remove(nearest_idx)
                else:
                    if nearest_idx not in self.selected_indices:
                        self.selected_indices = [nearest_idx]
                if self.selected_indices != before:
                    self._note_hand_selection()

                starts_drag = nearest_idx in self.selected_indices and not knn_topology_layout(cfg)
                # The web pages (the Meta Data sheet) hear of the selection once
                # a drag: now, or under Low Resource Mode's Update Web Pages
                # After Mouse Release on release (on_mouse_release).
                unsent = starts_drag and low_resource_webserver(cfg)
                self.update_selection_visual(broadcast=not unsent)

                if starts_drag:
                    self.is_multi_dragging = True
                    self._drag_selection_unsent = unsent
                    self.drag_start_nodes_pos = self.pos[self.selected_indices, :2].copy()
            else:
                # Clicked empty space: Store the current state, DO NOT clear yet
                self._pre_drag_selection = set(self.selected_indices)
                self._box_selection_unsent = False
                self.is_box_selecting = True
                self.selection_box.set_data(
                    pos=np.zeros((5, 2), dtype=np.float32),
                )
                self.selection_box.visible = True
                
            event.handled = True 
            return
            
        # ---> 2. SHIFT + LEFT-CLICK LOGIC (Copy to Clipboard) <---
        # Vispy Button 1 = Left Click
        if event.button == 1 and 'Shift' in event.modifiers:
            nearest_idx = self.pick_hovered_node(event.pos)
            if nearest_idx is not None:
                full_header = self.full_headers[nearest_idx]
                try:
                    # Use Vispy's native app instance, just like in on_key_press
                    from vispy import app as vispy_app
                    native_app = vispy_app.use_app().native
                    native_app.clipboard().setText(full_header)
                    
                    self._show_click_status(Message("Copied: {node}", node=full_header))
                    print(f"Copied to clipboard: {full_header}")
                except Exception as e:
                    self._show_click_status(Message("Copy Failed: {node}", node=full_header))
                    print(f"Clipboard Error: {e}")

                self.update_console_background()
                
            event.handled = True
            return
        
        # ---> 2.5 CONTROL + LEFT-CLICK LOGIC (Copy Sequence to Clipboard) <---
        # Vispy Button 1 = Left Click
        if event.button == 1 and ('Control' in event.modifiers or 'Meta' in event.modifiers):
            nearest_idx = self.pick_hovered_node(event.pos)
            if nearest_idx is not None:
                full_header = self.full_headers[nearest_idx]
                rec_id = full_header.split()[0]
                
                # Lazy-load sequence map if not already loaded
                if not hasattr(self, 'sequences_map'):
                    fasta_path = getattr(cfg, 'NODE_FASTA_FILE', None) or getattr(cfg, 'SEQUENCES_FILE', '')
                    if fasta_path and os.path.exists(fasta_path):
                        try:
                            records = _load_selected_fasta_records(fasta_path)
                            self.sequences_map = _build_sequence_lookup(records)
                        except Exception as e:
                            self.sequences_map = {}
                            print(f"Warning: Failed to parse FASTA for sequences: {e}")
                    else:
                        self.sequences_map = {}
                
                # Look up sequence
                sequence = None
                if full_header in self.sequences_map:
                    sequence = self.sequences_map[full_header]
                elif rec_id in self.sequences_map:
                    sequence = self.sequences_map[rec_id]
                
                if sequence:
                    try:
                        from vispy import app as vispy_app
                        native_app = vispy_app.use_app().native
                        native_app.clipboard().setText(sequence)
                        self._show_click_status(Message("Copied sequence of: {node}", node=rec_id))
                        print(f"Copied sequence to clipboard: {rec_id} ({len(sequence)} aa)")
                    except Exception as e:
                        self._show_click_status(Message("Copy Failed: {node}", node=rec_id))
                        print(f"Clipboard Error: {e}")
                else:
                    self._show_click_status(Message("Sequence not found for: {node}", node=rec_id))
                    print(f"Sequence not found in FASTA for: {rec_id}")

                self.update_console_background()
                    
            event.handled = True
            return
        
        # ---> 3. PLAIN LEFT-CLICK LOGIC (Show Highlight) <---
        if event.button == 1 and 'Shift' not in event.modifiers:
            # On the node the hover highlights, apply the shared left-click focus.
            nearest_idx = self.pick_hovered_node(event.pos)
            if nearest_idx is not None:
                self.apply_left_click_focus(nearest_idx)
            else:
                # Clicking empty space clears the left-click highlight
                self.clear_left_click_focus()
                
            event.handled = True
            return

    def apply_left_click_focus(self, node_idx):
        """Apply the shared temporary focus used by SSN and metadata clicks."""
        node_idx = int(node_idx)
        self.selected_node_idx = node_idx
        self.left_click_highlight_indices = None
        if getattr(self, 'tooltip', None) is not None:
            self.tooltip.text = ""

        label = self.full_headers[node_idx]
        if getattr(self, 'cluster_labels', None) is not None:
            cluster_id = self.cluster_labels[node_idx]
            if cluster_id == -1:
                label = Message("[Noise] {node}", node=label)
            else:
                label = Message("[Cluster {cluster}] {node}", cluster=cluster_id, node=label)

        if getattr(self, 'group_labels', None) and self.group_labels[node_idx]:
            group_text = ", ".join(sorted(self.group_labels[node_idx]))
            label = Message("{node} [Groups: {groups}]", node=label, groups=group_text)

        print(f"Node Selected: {self.full_headers[node_idx]}")
        if getattr(self, 'console_text', None) is not None:
            self._show_click_status(Message("Selected: {node}", node=label))

        for display in getattr(self, 'hud_displays', {}).values():
            if getattr(display, 'on_node_clicked', None):
                display.on_node_clicked(node_idx)

        self.update_nodes()
        self.broadcast_event({"type": "highlight_row", "index": node_idx})

    def clear_left_click_focus(self):
        """Clear temporary click focus without changing command selection."""
        self.selected_node_idx = None
        self.left_click_highlight_indices = None
        if getattr(self, 'tooltip', None) is not None:
            self.tooltip.text = ""
        self.update_nodes()
        self.broadcast_event({"type": "highlight_row", "index": None})
    
    def _color_hidden_count(self):
        """Colour the hidden-node count, and return it: the HUD's text colour
        at 0, otherwise a dark shade of the theme colour, readable on white."""
        hidden_count = int(np.sum(~self.visible_mask))
        self.hidden_text.color = TOKENS["accent_text"] if hidden_count else cfg.TEXT_COLOR
        return hidden_count

    def _update_hud_elements(self, event=None):
        """Updates the zoom indicator, hidden nodes count, and maintains the tooltip pixel gap."""
        effective_canvas_w = self.canvas.size[0] - _sidebar_room(self)
        effective_size = (effective_canvas_w, self.canvas.size[1])
        
        # 1. The Zoom box: the view's zoom, and the limits zooms keep to
        box = getattr(self, 'zoom_box', None)
        if box is not None:
            from commands.zoom import current_zoom, zoom_limits
            zoom = current_zoom(self)
            if zoom is not None:
                box.spin_box.show_zoom(zoom, *zoom_limits(self))

        # 2. Update Hidden Nodes Indicator
        if hasattr(self, 'hidden_text') and hasattr(self, 'visible_mask'):
            hidden_count = self._color_hidden_count()
            self.hidden_text.text = translate("Viewer", "Hidden Nodes: {count}").format(count=hidden_count)
            self.hidden_text.pos = self._status_hud_position(0, effective_size)

        # 3. Update Tooltip Distance
        if getattr(self, 'selected_node_idx', None) is not None and getattr(self, 'tooltip', None) and self.tooltip.text != "":
            tr = self.canvas.scene.node_transform(self.view.scene)
            screen_pos = tr.inverse.map(self.pos[self.selected_node_idx])

            self.tooltip.pos = screen_pos[:2] + [15, -15]

        # Keep the one-line console value visually wrapped as the canvas or
        # sidebar width changes.
        self.update_console_background()

        # 4. Update any registered HUD displays
        for display in self.hud_displays.values():
            if getattr(display, 'update_position', None):
                display.update_position()

        # 5. The info card's counts, if it is open
        self._refresh_session_card()

        self.canvas.update()

    def _initialize_display_tracking(self):
        """Bind DPI notifications after the canvas has its final top-level window."""
        main_window = getattr(self, 'main_window', None)
        window_handle = main_window.windowHandle() if main_window is not None else None

        if window_handle is not getattr(self, '_display_window_handle', None):
            previous_handle = getattr(self, '_display_window_handle', None)
            if previous_handle is not None:
                try:
                    previous_handle.screenChanged.disconnect(self._on_display_screen_changed)
                except (RuntimeError, TypeError):
                    pass

            self._display_window_handle = window_handle
            if window_handle is not None:
                window_handle.screenChanged.connect(self._on_display_screen_changed)

        screen = window_handle.screen() if window_handle is not None else None
        self._bind_active_screen(screen)
        self._schedule_display_refresh()

    def _bind_active_screen(self, screen):
        """Follow DPI and geometry changes from only the window's active screen."""
        signal_bindings = getattr(self, '_active_screen_signal_bindings', [])
        if screen is getattr(self, '_active_screen', None) and signal_bindings:
            return

        for signal, callback in signal_bindings:
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass

        self._active_screen_signal_bindings = []
        self._active_screen = screen
        if screen is None:
            return

        callback = self._on_active_screen_metrics_changed
        for signal_name in (
            'logicalDotsPerInchChanged',
            'physicalDotsPerInchChanged',
            'geometryChanged',
        ):
            signal = getattr(screen, signal_name, None)
            if signal is None:
                continue
            signal.connect(callback)
            self._active_screen_signal_bindings.append((signal, callback))

    def _on_display_screen_changed(self, screen):
        self._bind_active_screen(screen)
        self._schedule_display_refresh()

    def _on_active_screen_metrics_changed(self, *args):
        self._schedule_display_refresh()

    def _schedule_display_refresh(self):
        """Debounce display notifications until Qt's GL transition settles."""
        if getattr(self, '_display_refresh_pending', False):
            return
        self._display_refresh_pending = True
        QtCore.QTimer.singleShot(100, self._refresh_display_layout)

    def _refresh_display_layout(self):
        """Refresh logical overlays after Qt/VisPy update the framebuffer."""
        self._display_refresh_pending = False
        canvas = getattr(self, 'canvas', None)

        # QOpenGLWidget owns resizeGL and invokes it with a current context.
        # VisPy's Qt backend is already connected to the native window's
        # screenChanged signal. Calling either method manually here happens
        # outside paintGL and can race context migration between monitors.

        for method_name in (
            '_apply_vispy_text_scaling',
            'position_slider_overlay',
            'reposition_expand_btn',
            '_update_hud_elements',
        ):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception as e:
                print(f"Warning: Could not refresh {method_name}: {e}")

        if canvas is not None:
            try:
                canvas.update()
            except Exception as e:
                print(f"Warning: Could not redraw canvas after display refresh: {e}")

        self._print_display_diagnostics()

    def _print_display_diagnostics(self):
        """Print current logical/physical display state only when it changes."""
        canvas = getattr(self, 'canvas', None)
        if canvas is None:
            return

        screen = getattr(self, '_active_screen', None)
        native_canvas = getattr(canvas, 'native', None)
        try:
            screen_name = screen.name() if screen is not None else "Unknown display"
            geometry = screen.geometry() if screen is not None else None
            screen_geometry = (
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height(),
            ) if geometry is not None else (0, 0, 0, 0)
            dpr = (
                float(native_canvas.devicePixelRatioF())
                if native_canvas is not None and hasattr(native_canvas, 'devicePixelRatioF')
                else float(getattr(canvas, 'pixel_scale', 1.0))
            )
            logical_size = tuple(canvas.size)
            physical_size = tuple(canvas.physical_size)
            qapp = QtWidgets.QApplication.instance()
            qt_platform = qapp.platformName() if qapp is not None else "unknown"
        except Exception as e:
            print(f"Warning: Could not read display information: {e}")
            return

        signature = (
            screen_name,
            screen_geometry,
            round(dpr, 6),
            logical_size,
            physical_size,
            qt_platform,
        )
        if signature == getattr(self, '_last_display_signature', None):
            return
        self._last_display_signature = signature

        print(
            "Display: "
            f"{screen_name} | geometry={screen_geometry[2]}x{screen_geometry[3]} "
            f"at ({screen_geometry[0]}, {screen_geometry[1]}) | DPR={dpr:g} | "
            f"Qt platform={qt_platform} | "
            f"canvas logical={logical_size[0]}x{logical_size[1]} | "
            f"physical={physical_size[0]}x{physical_size[1]}"
        )

    def on_resize(self, event): 
        # Resize events already run after VisPy has updated the canvas size and
        # scene transforms. Refresh the HUD synchronously so pixel-anchored
        # text follows the window instead of waiting for a Qt-backed timer,
        # which can be starved during interactive window resizing. The Qt
        # overlays move first.
        if hasattr(self, 'slider_overlay'):
            self.position_slider_overlay()
        if hasattr(self, 'reposition_expand_btn'):
            self.reposition_expand_btn()
        self._update_hud_elements()

    def reposition_expand_btn(self):
        """Put the sidebar and its toggle where the room it covers now leaves them.

        While it slides, the panel's left part shows and the canvas clips the
        rest; the toggle keeps 10 px clear of its left edge, and the frame
        rate, if shown, FPS_GAP left of the toggle.
        """
        if hasattr(self, 'canvas'):
            w, h = self.canvas.size
            left = round(w - _sidebar_room(self))
            panel_w = getattr(self, '_panel_w', 120)
            if hasattr(self, 'right_panel'):
                self.right_panel.setGeometry(left, 0, panel_w, h)
            if hasattr(self, 'toggle_sidebar_btn'):
                self.toggle_sidebar_btn.setGeometry(left - 40, 10, 30, 30)
            overlay = getattr(self, 'fps_overlay', None)
            if overlay is not None:
                overlay.place(self._fps_overlay_right(left), 25)
            badge = getattr(self, 'jobs_badge', None)
            if badge is not None:
                # Under the frame rate's row (the toggle's), right-aligned with it.
                badge.place(self._fps_overlay_right(left), 25 + FPS_HEIGHT / 2 + BADGE_GAP)

    def _fps_overlay_right(self, left):
        """The frame rate's right edge, in canvas px, with the sidebar's room starting at left:
        FPS_GAP left of the toggle, or where the toggle would be when it is hidden."""
        button = getattr(self, 'toggle_sidebar_btn', None)
        if button is not None and not button.isHidden():
            return left - 40 - FPS_GAP
        return left - 10

    def _build_jobs_badge(self):
        """The jobs badge (desktop.Viewer_Jobs): shown while commands wait or run, or files are written."""
        self.jobs_badge = JobsBadge(self.canvas)
        self.reposition_expand_btn()

    def _on_background_jobs_changed(self):
        """A lane or a running job changed: the badge counts again and an open card lists them again."""
        self._refresh_jobs_badge()
        button = getattr(self, 'controls_button', None)
        if button is not None:
            button.refresh_card_jobs()

    def _refresh_jobs_badge(self):
        badge = getattr(self, 'jobs_badge', None)
        if badge is not None:
            badge.show_jobs(job_lines(getattr(self, 'command_queue', None),
                                      getattr(self, 'background_job_scheduler', None)))

    def _note_hand_selection(self):
        """A selection was made by hand: it is newer than every command given so far."""
        self.selection_epoch = getattr(self, 'selection_epoch', 0) + 1

    def undo_by_hand(self):
        """Ctrl+Z and the Meta Data sheet's undo: take back the newest command that waits or runs, else undo.

        A command that has not finished has not happened yet: undoing it is not
        doing it. Returns whether anything was taken back.
        """
        queue = getattr(self, 'command_queue', None)
        if queue is not None and queue.pending():
            _cancelled, message = queue.take_back(1)
            Command_Engine.show_status(self, message)
            return True
        return self._do_undo()

    def redo_by_hand(self):
        """Ctrl+Y and the sheet's redo: nothing to redo while commands wait (they come after the undone steps)."""
        queue = getattr(self, 'command_queue', None)
        if queue is not None and queue.pending():
            message = Message("Nothing to redo: the undone steps were followed by %n waiting command(s).",
                              n=len(queue.pending()))
            Command_Engine.show_status(self, message)
            print(message)
            return False
        return self._do_redo()

    def _build_fps_overlay(self):
        """The frame rate over the canvas's top right, if Config's Show Frame Rate (FPS) is on (desktop.Viewer_Fps)."""
        self.fps_overlay = None
        if not getattr(cfg, 'SHOW_FPS', False):
            return

        def edge_detail():
            layer = getattr(self, 'edge_layer', None)
            visual = getattr(layer, 'visual', None)
            if visual is None or not layer.visible:
                return 1
            return getattr(visual, 'detail', 1)
        self.fps_overlay = FpsOverlay(self.canvas, edge_detail)
        self.fps_overlay.attach()
        self.fps_overlay.show()
        self.reposition_expand_btn()

    def _command_line_right(self):
        """Where the command line's box may reach, in canvas px: as wide as the
        canvas less the sidebar's room and 40 px, from its left offset, and
        short of the frame rate's room when that is shown."""
        left = getattr(self, 'hud_layout', {}).get("console_bg_left_offset", 20.0)
        right = left + self.canvas.size[0] - _sidebar_room(self) - 40.0
        overlay = getattr(self, 'fps_overlay', None)
        if overlay is not None:
            right = min(right, self._fps_overlay_right(self.canvas.size[0] - _sidebar_room(self))
                        - overlay.widest_width() - FPS_GAP)
        return right

    def toggle_sidebar(self):
        if hasattr(self, 'right_panel'):
            # Mid-slide the panel is still shown: the toggle turns it around.
            opening = getattr(self, '_sidebar_open', None)
            if opening is None:
                opening = self.right_panel.isVisible()
            self.set_sidebar_visible(not opening)

    def set_sidebar_visible(self, visible, *, animate=None):
        """Open or close the sidebar.

        In a shown window it slides over SIDEBAR_SLIDE_MS, and its toggle,
        the slider and the HUD follow it at each frame. animate=False, a
        hidden or headless window, or SIDEBAR_SLIDE_MS 0 switch it at once.
        A call while it slides turns it around from where it is.
        """
        has_buttons = bool(getattr(self, 'sidebar_buttons', {}))
        if hasattr(self, 'right_panel'):
            start = _sidebar_room(self)
            self._stop_sidebar_slide()
            if not has_buttons:
                self._sidebar_open = False
                self.right_panel.hide()
                if hasattr(self, 'toggle_sidebar_btn'):
                    self.toggle_sidebar_btn.hide()
                self.reposition_expand_btn()
                return

            self._sidebar_open = visible
            if hasattr(self, 'toggle_sidebar_btn'):
                self.toggle_sidebar_btn.setVisible(True)
                self.toggle_sidebar_btn.setIcon(
                    studio_icon("panel-right-close" if visible else "panel-right-open")
                )
            panel_w = getattr(self, '_panel_w', 120)
            end = panel_w if visible else 0
            duration = SIDEBAR_SLIDE_MS * abs(end - start) / max(panel_w, 1)
            if animate is None:
                animate = self._sidebar_can_slide()
            if not animate or duration < 1:
                self.right_panel.setVisible(visible)
                self._follow_sidebar()
                return

            # Shown both ways while it slides; hidden at the end of a close.
            self.right_panel.show()
            animation = QtCore.QVariantAnimation(self.main_window)
            animation.setStartValue(float(start))
            animation.setEndValue(float(end))
            animation.setDuration(round(duration))
            animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

            def step(value, animation=animation):
                if getattr(self, '_sidebar_animation', None) is animation:
                    self._sidebar_slide_width = float(value)
                    self._follow_sidebar()

            def finish(animation=animation):
                if getattr(self, '_sidebar_animation', None) is animation:
                    self._stop_sidebar_slide()
                    self.right_panel.setVisible(self._sidebar_open)
                    self._follow_sidebar()

            animation.valueChanged.connect(step)
            animation.finished.connect(finish)
            self._sidebar_animation = animation
            self._sidebar_slide_width = float(start)
            animation.start()

    def _sidebar_can_slide(self):
        """Whether the sidebar slides: only in a shown window, with a slide time,
        and not under Low Resource Mode's No Zoom or Sidebar Animation, which switches it at once."""
        window = getattr(self, 'main_window', None)
        return (
            SIDEBAR_SLIDE_MS > 0
            and not low_resource_animations(cfg)
            and self.shows_windows()
            and window is not None
            and window.isVisible()
        )

    def _stop_sidebar_slide(self):
        """Stop a slide where it is; the sidebar's room is its open or closed width again."""
        animation = getattr(self, '_sidebar_animation', None)
        self._sidebar_animation = None
        self._sidebar_slide_width = None
        if animation is not None and shiboken6.isValid(animation):
            animation.stop()
            animation.deleteLater()

    def _follow_sidebar(self):
        """Move the panel, its toggle, the slider and the HUD to the sidebar's room now."""
        self.reposition_expand_btn()
        if hasattr(self, 'position_slider_overlay'):
            self.position_slider_overlay()
        if hasattr(self, '_update_hud_elements'):
            self._update_hud_elements()

    # show_existing_dialog is keyword-only so that a Qt signal connected to
    # these (the sidebar button's clicked(bool)) never passes it a value.
    def open_metadata_ui(self, *, show_existing_dialog=True):
        return self._open_web_ui(
            "/meta.html",
            Message("Metadata UI"),
            "meta",
            show_existing_dialog=show_existing_dialog,
        )

    def open_agent_ui(self, *, show_existing_dialog=True):
        return self._open_web_ui(
            "/agent.html",
            Message("Agent UI"),
            "agent",
            show_existing_dialog=show_existing_dialog,
        )

    def open_console_window(self):
        """Open the Console window, which shows what this Viewer's process has printed.

        Returns "opened", "no_window" when the Viewer shows no windows (headless, or hidden and not
        shown), or "not_kept" when it was started with --console, so nothing is kept to show.
        The console command reports each; the Console button explains the last.
        """
        if not self.shows_windows():
            return "no_window"
        from desktop.Console_View import console_available, show_console
        if not console_available():
            return "not_kept"
        return "opened" if show_console(self.main_window) is not None else "not_kept"

    def _console_button_clicked(self):
        """The Console button's click: open the Console, or say why nothing is kept to show."""
        if self.open_console_window() == "not_kept":
            from desktop.Console_View import not_available_text
            QtWidgets.QMessageBox.information(self.main_window, translate("Viewer", "Console"), not_available_text())

    def _open_web_ui(
        self,
        path,
        label,
        client_id,
        *,
        show_existing_dialog=True,
    ):
        return open_browser_page(
            self,
            path,
            label,
            client_id,
            show_existing_dialog=show_existing_dialog,
        )

    def get_web_url(self, path="/"):
        """Return a URL served by this Viewer instance."""
        self.ensure_web_server()
        if not self.web_server_url:
            raise RuntimeError(Message("This Viewer instance's web server is unavailable."))
        return f"{self.web_server_url}/{str(path).lstrip('/')}"

    def add_sidebar_button(self, name, label, callback, tooltip=None, icon=None):
        if not hasattr(self, 'sidebar_buttons'):
            self.sidebar_buttons = {}
        
        # A button that already exists is shown again, and the sidebar is left
        # open or closed as the user left it: only a new button opens it.
        if name in self.sidebar_buttons:
            self.sidebar_buttons[name].show()
            return self.sidebar_buttons[name]
        
        btn = SidebarButton(label, self.right_panel)
        btn.setObjectName(name)
        if icon:  # a bundled Lucide icon's name, shown before the label
            btn.setIcon(studio_icon(icon))
            btn.setIconSize(QtCore.QSize(16, 16))
        if tooltip:
            btn.setToolTip(tooltip)
        btn.clicked.connect(callback)
        
        # Insert button in layout right before the bottom stretch spacer
        layout = self.right_panel_layout
        layout.insertWidget(layout.count() - 1, btn)
        
        self.sidebar_buttons[name] = btn
        self._fit_sidebar_to_buttons()
        self.set_sidebar_visible(True)
        return btn

    def _fit_sidebar_to_buttons(self):
        """Give every sidebar button the widest label's width, and the panel room for them."""
        margins = self.right_panel_layout.contentsMargins()
        button_width = fit_buttons_to_text(*self.sidebar_buttons.values())
        self._panel_w = button_width + margins.left() + margins.right()
        self._update_window_minimum_width()

    def _slider_row_width(self):
        """The canvas width the slider row needs for nothing in it to overlap, the sidebar aside.

        From the left: the row's margin, the buttons, the threshold box and the slider at
        its least (SLIDER_MIN_WIDTH), the gap, and the Zoom box with its margin at the right
        (position_slider_overlay lays them out so). 0 while the row is not built.
        """
        overlay, box = getattr(self, 'slider_overlay', None), getattr(self, 'zoom_box', None)
        hud_layout = getattr(self, 'hud_layout', None)
        if overlay is None or box is None or hud_layout is None:
            return 0
        # Measured in the stylesheets' fonts, as the row is laid out.
        for widget in (overlay, box, *overlay.findChildren(QtWidgets.QWidget)):
            widget.ensurePolished()
        overlay.layout().invalidate()
        return (SLIDER_ROW_LEFT + overlay.layout().minimumSize().width() + SLIDER_BOX_GAP
                + box.sizeHint().width() + round(hud_layout["status_x_offset"]))

    def _update_window_minimum_width(self):
        """Keep the window wide enough for the sidebar's buttons, and for the slider row beside the sidebar.

        The row needs _slider_row_width and, while the sidebar has buttons, the room it takes
        when open, so that opening it never makes the row overlap. A Viewer without the row
        keeps the sidebar's width alone.
        """
        room = self._panel_w if getattr(self, 'sidebar_buttons', None) else 0
        self.main_window.setMinimumWidth(max(self._panel_w, self._slider_row_width() + room))


    def start_web_server(self):
        try:
            from web_ui import Web_Server
            self.web_server = Web_Server.start_server(self)
            port = int(self.web_server.server_address[1])
            self.web_server_url = f"http://{Web_Server.LOOPBACK_HOST}:{port}"
            print(f"WebServer started at {self.web_server_url}")
            return True
        except Exception as e:
            self.web_server = None
            self.web_server_url = None
            print(f"Error starting WebServer: {e}")
            return False

    def ensure_web_server(self):
        """Keep this Viewer attached to one live local web server."""
        from web_ui import Web_Server

        current_server = getattr(self, "web_server", None)
        if Web_Server.is_running(current_server):
            return current_server
        try:
            self.web_server = Web_Server.ensure_server(self, current_server)
        except Exception as error:
            self.web_server = None
            self.web_server_url = None
            raise RuntimeError(
                Message("This Viewer instance's web server is unavailable: {error}", error=error)
            ) from error
        port = int(self.web_server.server_address[1])
        self.web_server_url = f"http://{Web_Server.LOOPBACK_HOST}:{port}"
        print(f"WebServer restarted at {self.web_server_url}")
        return self.web_server

    def shutdown_web_server(self):
        """Release this Viewer's local web server and port exactly once."""
        from web_ui import Web_Server

        server = getattr(self, "web_server", None)
        try:
            Web_Server.stop_server(server)
        except Exception as error:
            print(f"Error stopping WebServer: {error}")
        finally:
            if getattr(self, "web_server", None) is server:
                self.web_server = None
                self.web_server_url = None

    def broadcast_event(self, event):
        if hasattr(self, 'web_server') and self.web_server:
            with self.web_server.queues_lock:
                queues = list(self.web_server.event_queues)
            for q in queues:
                q.put(event)

    def broadcast_metadata_state(self):
        """Broadcast metadata rows and schema as one authoritative state."""
        if not _web_clients_connected(self):
            # Building the rows costs seconds on a large network, and no page
            # is listening. A page that connects later is sent the full
            # current state by get_initial_web_state.
            return
        self.broadcast_event({
            "type": "state_updated",
            "visible_mask": self.visible_mask.tolist(),
            "selected_indices": self.selected_indices,
            "metadata": self.get_serializable_metadata(),
            "columns": ["Node ID"] + list(self.metadata.keys()),
            "types": {
                key: entry["type"] for key, entry in self.metadata.items()
            },
        })

    def handle_web_action(self, data):
        action = data.get("action")
        if action in self.web_action_handlers:
            self.web_action_handlers[action](data)
        else:
            print(f"Warning: No handler registered for web action '{action}'")

    def get_serializable_metadata(self):
        row_count = self.n_nodes
        node_ids = [str(header) for header in self.full_headers[:row_count]]
        if len(node_ids) < row_count:
            raise IndexError("full_headers is shorter than n_nodes")
        keys = ["id", "Node ID"]
        columns = [range(row_count), node_ids]
        for key, entry in self.metadata.items():
            if key in ("id", "Node ID"):
                # The web table finds a row by these two keys, so no
                # metadata column may overwrite them.
                continue
            keys.append(key)
            columns.append(_web_table_column(entry["values"], row_count))
        # Each column is converted once, then the columns are zipped into rows.
        return [dict(zip(keys, row)) for row in zip(*columns)]

    def get_initial_web_state(self):
        state = {
            "rows": self.get_serializable_metadata(),
            "columns": ["Node ID"] + [k for k in self.metadata.keys() if k.lower() != "length"],
            "selected_indices": self.selected_indices,
            "visible_mask": self.visible_mask.tolist(),
            "llm_loaded": getattr(self, 'llm_loaded', False),
            "llm_backend": getattr(self, 'llm_backend', None),
            "llm_model_name": getattr(self, 'llm_model_name', "Unknown")
        }
        registry = getattr(self, "web_plugin_registry", None)
        if registry is not None:
            return registry.apply_state_providers(state)
        return state

    def on_mouse_wheel(self, event):
        self._hud_timer.start()

        # ---> Rotate selected nodes if right-click dragging <---
        if getattr(self, 'is_multi_dragging', False) and 2 in event.buttons:
            self._note_view_motion()
            # event.delta[1] is > 0 for scroll up, < 0 for scroll down
            angle = event.delta[1] * (np.pi / 36) # 5 degrees per tick
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            center = self.drag_start_mouse
            
            pts = self.drag_start_nodes_pos - center
            rotated_x = pts[:, 0] * cos_a - pts[:, 1] * sin_a
            rotated_y = pts[:, 0] * sin_a + pts[:, 1] * cos_a
            self.drag_start_nodes_pos[:, 0] = rotated_x + center[0]
            self.drag_start_nodes_pos[:, 1] = rotated_y + center[1]
            
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            delta = mouse_world - self.drag_start_mouse
            self.pos[self.selected_indices, :2] = self.drag_start_nodes_pos + delta
            
            if low_resource_drawing(cfg):
                self._move_dragged_nodes()
            else:
                # A drag moves the selection without changing it: the web pages heard of it on press.
                self.update_selection_visual(broadcast=False)
            
            event.handled = True
            return

    def on_mouse_move(self, event):
        tr = self.canvas.scene.node_transform(self.view.scene)
        mouse_world = tr.map(event.pos)[:2]

        # ---> 0. FAILSAFE: CATCH MISSED MOUSE RELEASES <---
        # If we are dragging or boxing, but the right button is NO LONGER held down:
        if getattr(self, 'is_multi_dragging', False) or getattr(self, 'is_box_selecting', False):
            if 2 not in event.buttons:
                self.on_mouse_release(event)
                return

        # ---> 1. MULTI-NODE DRAGGING <---
        if getattr(self, 'is_multi_dragging', False):
            self._note_view_motion()
            delta = mouse_world - self.drag_start_mouse
            self.pos[self.selected_indices, :2] = self.drag_start_nodes_pos + delta
            
            if low_resource_drawing(cfg):
                self._move_dragged_nodes()
            else:
                # A drag moves the selection without changing it: the web pages heard of it on press.
                self.update_selection_visual(broadcast=False)
            
            event.handled = True
            return
        # ---> 2. BOX SELECTION DRAWING <---
        if getattr(self, 'is_box_selecting', False):
            x0, y0 = self.drag_start_mouse
            x1, y1 = mouse_world
            
            rect_pts = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]
            ], dtype=np.float32)
            self.selection_box.set_data(
                pos=rect_pts,
            )
            
            # --- Dynamic Highlighting Math ---
            if not low_resource_drawing(cfg):
                min_x, max_x = min(x0, x1), max(x0, x1)
                min_y, max_y = min(y0, y1), max(y0, y1)
                xs = self.pos[:, 0]
                ys = self.pos[:, 1]
                mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y) & self.visible_mask
                box_indices = set(np.where(mask)[0].tolist())
                
                pre_drag = getattr(self, '_pre_drag_selection', set())
                
                if 'Shift' in event.modifiers:
                    current_selection = pre_drag.union(box_indices)     # Add
                elif 'Control' in event.modifiers or 'Meta' in event.modifiers:
                    current_selection = pre_drag.difference(box_indices)# Remove
                else:
                    current_selection = box_indices                     # Replace
                    
                self.selected_indices = list(current_selection)
                self._note_hand_selection()
                # Under Low Resource Mode's Update Web Pages After Mouse Release the web
                # pages (the Meta Data sheet) hear of it once, on release.
                unsent = low_resource_webserver(cfg)
                self.update_selection_visual(broadcast=not unsent)
                self._box_selection_unsent = getattr(self, '_box_selection_unsent', False) or unsent

            event.handled = True
            return
        # ---> 3. PANNING HUD OVERLAY <---
        # Update if we are panning (Left Click Drag only)
        if 1 in event.buttons:
            self._hud_timer.start()

        # ---> 4. HOVER EFFECT <---
        if not event.buttons:
            previous = getattr(self, 'hovered_node_idx', None)
            hovered = self.pick_hovered_node(event.pos)
            if hovered != previous:
                self.hovered_node_idx = hovered
                self._show_hover_change(previous)

    def _resume_command_queue(self):
        """A drag or box selection ended: a command result that waited for it applies now."""
        queue = getattr(self, 'command_queue', None)
        if queue is not None and queue.pending():
            queue.resume()

    def pick_hovered_node(self, cursor):
        """The node a cursor at canvas position cursor hovers: the topmost one
        whose drawn shape covers the pixel under it, or, over no node, the
        one drawn within a pixel of it (desktop.Viewer_Markers.MarkerPicker).
        Clicks act on this node too, so a click acts on what is highlighted."""
        picker = getattr(self, '_marker_picker', None)
        markers = getattr(self, 'markers', None)
        if picker is None or markers is None or not markers.visible:
            return None
        return picker.pick(markers.transforms, cursor, self.canvas.pixel_scale)

    def _show_hover_change(self, previous):
        """Show hovered_node_idx, the hover having left node previous.

        When nothing else has changed since update_nodes sent the markers,
        only the two nodes' face colours are sent (desktop.Viewer_Markers);
        otherwise every marker is, as update_nodes does for any other change.
        """
        hover = getattr(self, '_hover_markers', None)
        if hover is not None and hover.apply(self, previous, self.hovered_node_idx):
            # The markers now differ from what update_nodes sent.
            self._submitted_marker_data = None
            self.canvas.update()
            self._update_hud_elements()
            return
        self.update_nodes()
    
    def on_mouse_release(self, event):
        # ---> 1. MULTI-DRAG RELEASE <---
        if getattr(self, 'is_multi_dragging', False):
            self.is_multi_dragging = False
            self.update_selection_visual(broadcast=getattr(self, '_drag_selection_unsent', True))
            self._drag_selection_unsent = False
            self._resume_command_queue()
            event.handled = True
            return
            
        # ---> 2. FINALIZE BOX SELECTION <---
        if getattr(self, 'is_box_selecting', False):
            self.is_box_selecting = False
            self.selection_box.visible = False
            
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            x0, y0 = self.drag_start_mouse
            x1, y1 = mouse_world
            
            # Calculate drag distance in screen coordinates (pixels)
            drag_start_screen = getattr(self, 'drag_start_screen', None)
            if drag_start_screen is None:
                drag_start_screen = event.pos
            dx = event.pos[0] - drag_start_screen[0]
            dy = event.pos[1] - drag_start_screen[1]
            drag_dist_screen = np.hypot(dx, dy)
            
            if low_resource_drawing(cfg) and drag_dist_screen >= 5.0:
                min_x, max_x = min(x0, x1), max(x0, x1)
                min_y, max_y = min(y0, y1), max(y0, y1)
                xs = self.pos[:, 0]
                ys = self.pos[:, 1]
                mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y) & self.visible_mask
                box_indices = set(np.where(mask)[0].tolist())
                
                pre_drag = getattr(self, '_pre_drag_selection', set())
                
                if 'Shift' in event.modifiers:
                    current_selection = pre_drag.union(box_indices)     # Add
                elif 'Control' in event.modifiers or 'Meta' in event.modifiers:
                    current_selection = pre_drag.difference(box_indices)# Remove
                else:
                    current_selection = box_indices                     # Replace
                    
                self.selected_indices = list(current_selection)
                self._note_hand_selection()
                self.update_selection_visual()
                self._box_selection_unsent = False

            # Single click detection (no drag distance in screen pixels)
            if drag_dist_screen < 5.0:
                if 'Shift' not in event.modifiers and 'Control' not in event.modifiers and 'Meta' not in event.modifiers:
                    self.selected_indices = []
                    self._note_hand_selection()
                    self.update_selection_visual()
                    self._box_selection_unsent = False
            # A live box under Update Web Pages After Mouse Release sent nothing as it moved.
            if getattr(self, '_box_selection_unsent', False):
                self._box_selection_unsent = False
                self.broadcast_event({"type": "selection_changed", "indices": self.selected_indices})

            # Output final count to console
            if len(self.selected_indices) > 0:
                self._show_click_status(Message("Selected %n node(s).", n=len(self.selected_indices)))
            else:
                self._show_click_status(Message("Selection cleared."))
            self._resume_command_queue()
                
            event.handled = True
            return
        
if __name__ == '__main__':
    from utilities.Console_Capture import install_qt_message_handler
    install_qt_message_handler()  # Qt's warnings reach the Console tagged; a no-op without a capture
    set_app_user_model_id(APP_USER_MODEL_IDS["viewer"])  # the taskbar says EMAP-SSN, grouped apart from Config
    # Set the identity before VisPy creates QApplication or any native window.
    QtWidgets.QApplication.setApplicationVersion(APPLICATION_VERSION)
    configure_linux_qt_desktop_identity(
        QtWidgets.QApplication, VIEWER_DESKTOP_FILE_NAME
    )
    viewer = MainViewer()
    app.run()
