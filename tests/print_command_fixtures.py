"""Shared fixtures for the print command tests (src/commands/print.py).

Loads src/commands/print.py once with stub Command_Engine and EMAPSSN_Config
modules, installs module-level zoom-width patches via setUpModule/tearDownModule,
and provides fake viewers, schedulers, markers, SVG helpers, and tile renderers.
"""
import datetime
import importlib.util
import io
import os
import re
import sys
import tempfile
import time
import types
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def load_print_command():
    # Import print.py's real dependencies before the sys.modules window below.
    # A module first imported inside the window is dropped from sys.modules when
    # the window closes, while its parent package and print.py keep that copy:
    # later imports then load a second copy that patches of the first miss.
    import matplotlib.pyplot  # noqa: F401
    import vispy.app  # noqa: F401
    import vispy.scene  # noqa: F401
    import desktop.Desktop_App  # noqa: F401
    import desktop.Viewer_Edges  # noqa: F401
    import desktop.Viewer_State  # noqa: F401
    import desktop.Viewer_Zoom  # noqa: F401
    import utilities.Job_Cancellation  # noqa: F401  (its thread-local token must be the scheduler's)
    import utilities.Output_Names  # noqa: F401
    import Viewer_Visual_State  # noqa: F401
    command_engine = types.ModuleType("Command_Engine")
    for name in ("command_artifact", "command_succeeded", "command_failed"):
        setattr(command_engine, name, mock.Mock())

    def show_status(viewer, message):
        viewer.console_text.text = str(message)

    command_engine.show_status = show_status
    config = types.ModuleType("EMAPSSN_Config")
    config.ANALYSIS_RESULT_DIR = "Analysis_Results"
    config.SEQUENCE_SET = "test_sequences"
    config.resolve_directory_path = lambda value: value

    spec = importlib.util.spec_from_file_location(
        "print_command_under_test", os.path.join(SRC_DIR, "commands", "print.py")
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "Command_Engine": command_engine,
            "EMAPSSN_Config": config,
        },
    ):
        spec.loader.exec_module(module)
    return module


print_command = load_print_command()

_zoom_patches = []


def setUpModule():
    # `zoom N` is a percentage of a view ZOOM_100_PERCENT_WIDTH units wide. The numbers in
    # these tests (zoom 200 is a view 500 units wide, zoom 2000 is 50) assume that is 1000, so
    # they hold whatever the shipped number is; test_zoom_command.ShippedZoomTests checks that.
    import desktop.Viewer_Zoom

    for target in (desktop.Viewer_Zoom, print_command):
        patcher = mock.patch.object(target, "ZOOM_100_PERCENT_WIDTH", 1000.0)
        patcher.start()
        _zoom_patches.append(patcher)


def tearDownModule():
    while _zoom_patches:
        _zoom_patches.pop().stop()


class InlineScheduler:
    """The viewer's background job queue, run at once: each job's worker runs
    inside enqueue, on the calling thread, and its report is kept."""

    def __init__(self):
        self.jobs = []
        self.reports = []
        self.errors = []

    def is_output_path_reserved(self, path):
        return False

    def enqueue(self, command_name, description, payload, worker, output_path, allow_overwrite=False):
        if not allow_overwrite and os.path.exists(output_path):
            raise FileExistsError(f"Output file already exists: {output_path}")
        self.jobs.append(SimpleNamespace(
            command_name=command_name, description=description, payload=payload,
            output_path=output_path, allow_overwrite=allow_overwrite,
        ))
        try:
            self.reports.append(dict(worker(payload)))
        except Exception as error:
            self.errors.append(error)
        return len(self.jobs)

    @property
    def report(self):
        """The text of the last job's report."""
        return str(self.reports[-1]["message"])


def raw_renders(tile, is_transparent):
    """What _render_tile returns for a float RGBA tile: its 8-bit render, or
    for a transparent capture its renders on black and on white."""
    if not is_transparent:
        return (np.round(tile * 255).astype(np.uint8),)
    rgb, alpha = tile[..., :3], tile[..., 3:]
    on_black = np.zeros(tile.shape[:2] + (4,), dtype=np.uint8)
    on_black[..., :3] = np.round(rgb * alpha * 255)
    on_black[..., 3] = 255
    on_white = np.round((rgb * alpha + 1.0 - alpha) * 255).astype(np.uint8)
    return on_black, on_white


def white_renders(viewer, is_transparent, shape=(100, 100)):
    """A blank tile's renders, as _render_tile returns them."""
    return raw_renders(np.ones(shape + (4,), dtype=np.float32), is_transparent)


def make_viewer(background="white"):
    return SimpleNamespace(
        background_job_scheduler=InlineScheduler(),
        canvas=SimpleNamespace(
            bgcolor=background,
            update=mock.Mock(),
        ),
        console_bg=SimpleNamespace(visible=True),
        console_text=SimpleNamespace(text="previous message"),
        hidden_text=SimpleNamespace(visible=True),
        hud_displays={},
        tooltip=SimpleNamespace(visible=True),
        view=SimpleNamespace(
            camera=SimpleNamespace(rect="original rect", aspect=1.0)
        ),
        zoom_text=SimpleNamespace(visible=True),
    )


class FixedClock:
    @staticmethod
    def now():
        return datetime.datetime(2026, 1, 2, 3, 4, 5)


def view_at(units_per_pixel, size=(800, 600)):
    """A view showing `units_per_pixel` scene units per logical screen pixel.

    The camera's real rect is the scene area the view of `size` pixels shows.
    """
    real_rect = SimpleNamespace(
        width=size[0] * units_per_pixel, height=size[1] * units_per_pixel
    )
    return SimpleNamespace(
        size=size,
        camera=SimpleNamespace(rect=real_rect, _real_rect=real_rect, aspect=1.0),
    )


def square_network(**overrides):
    """Four visible nodes on a square; two of the four edges pass a 0.5 threshold.

    The view shows one scene unit per pixel, so sizes in pixels read as the
    scene units the SVG writes; the scale tests set a view of their own.
    """
    viewer = SimpleNamespace(
        visible_mask=np.ones(4, dtype=bool),
        pos=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
        current_colors=np.tile([0.25, 0.5, 1.0, 1.0], (4, 1)),
        current_sizes=np.full(4, 10.0),
        current_shapes=np.array(["disc"] * 4, dtype=object),
        canvas=SimpleNamespace(bgcolor=SimpleNamespace(rgba=(1.0, 1.0, 1.0, 1.0))),
        view=view_at(1.0),
        edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32),
        edge_scores=np.array([0.90, 0.20, 0.95, 0.10]),
        current_slider_threshold=0.5,
        selected_indices=[],
    )
    for name, value in overrides.items():
        setattr(viewer, name, value)
    return viewer


def export_svg(viewer, **configuration):
    settings = {
        "EDGE_COLOR": "#000000",
        "NODE_BOUNDARY_COLOR": "#000000",
        "EDGE_ALPHA": 0.2,
        "EDGE_WIDTH": 0.5,
        "UMAP_MODE": False,
    }
    settings.update(configuration)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "network.svg")
        with mock.patch.multiple(print_command.cfg, create=True, **settings), \
                redirect_stdout(io.StringIO()):
            print_command._write_svg(print_command._svg_snapshot(viewer), path)
        with open(path, encoding="utf-8") as handle:
            return handle.read()


def elements(svg, tag):
    return [line.strip() for line in svg.splitlines() if line.strip().startswith(f"<{tag} ")]


class ElementTreeLines:
    """The line elements of an SVG as their coordinates, in file order."""

    @staticmethod
    def parse(svg):
        import xml.etree.ElementTree as ElementTree
        namespace = "{http://www.w3.org/2000/svg}"
        return [tuple(line.get(name) for name in ("x1", "y1", "x2", "y2"))
                for line in ElementTree.fromstring(svg).iter(namespace + "line")]


def node_fills(svg, tag="circle"):
    """The red channel each node's element is filled with, in file order."""
    return [
        int(match.group(1))
        for match in re.finditer(rf'<{tag} [^>]*fill="rgb\((\d+),', svg)
    ]


def numbered_colors(count=4):
    """One colour per node whose red channel (5, 15, 25, ...) names the node."""
    colors = np.zeros((count, 4))
    colors[:, 0] = (10.0 * np.arange(count) + 5.5) / 255.0
    colors[:, 3] = 1.0
    return colors


def number(element, name):
    """The numeric attribute `name` of one SVG element line."""
    return float(re.search(rf'(?<![\w-]){name}="(-?[\d.]+)"', element).group(1))


def vispy_outline_coverage(r, width, antialias, filled=True):
    """How much of a marker outline's colour VisPy's fragment shader puts at
    signed distance r from the shape's edge (vispy/visuals/markers.py), with
    an opaque fill inside; the shader's arithmetic, one distance at a time."""
    opacity = min(width, 1.0)
    solid = max(0.5 * width - antialias, 0.0)
    if r > 0.5 * width + antialias:
        return 0.0
    d = abs(r) - solid
    if d < 0.0:
        return opacity
    fade = np.exp(-(d / antialias) ** 2)
    if r > 0.0 or not filled:
        return opacity * fade
    # mix(fill, outline, fade), whose alpha is mix(1, opacity, fade).
    return (1.0 - fade + fade * opacity) * fade


# A rendered tile, rows by columns: the 800 x 600 canvas of zoom_viewer().
TILE_SHAPE = (600, 800)


def zoom_viewer():
    """A viewer whose 800 x 600 view shows 100 x 75 scene units, 0.125 per pixel.

    Its four visible nodes span 100 x 50 units; with the 5 units the full
    capture pads each side with, the network is 110 x 60 units.
    """
    viewer = make_viewer()
    live_rect = SimpleNamespace(pos=(0.0, -12.5), width=100.0, height=75.0)
    viewer.view = SimpleNamespace(
        size=(800, 600),
        camera=SimpleNamespace(rect=live_rect, _real_rect=live_rect, aspect=1.0),
    )
    viewer.canvas.size = (800, 600)
    viewer.visible_mask = np.ones(4, dtype=bool)
    viewer.pos = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 50.0, 0.0], [0.0, 50.0, 0.0]]
    )
    return viewer


class StatusStub:
    """The viewer a real scheduler reports to: keeps its status lines."""

    def __init__(self):
        self.statuses = []

    def set_background_job_status(self, message):
        self.statuses.append(str(message))


def wait_for(predicate, timeout=10.0):
    """Let Qt deliver the scheduler's queued signals until predicate holds."""
    from PySide6 import QtCore

    deadline = time.monotonic() + timeout
    application = QtCore.QCoreApplication.instance()
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    application.processEvents()
    return predicate()


class FakeMarkers:
    """The node Markers visual: keeps the data the Viewer last submitted."""

    def __init__(self):
        self.visible = True
        self.data = None

    def set_data(self, **kwargs):
        self.data = kwargs

    def set_gl_state(self, *args, **kwargs):
        pass


def marked_viewer(hovered=1, clicked=2, selected=(0,)):
    """A real Viewer's node drawing on zoom_viewer()'s network.

    Node `hovered` is under the mouse, node `clicked` has the left-click
    halo, and the nodes in `selected` have selection borders.
    """
    from EMAPSSN_Viewer import MainViewer

    viewer = MainViewer.__new__(MainViewer)
    for name, value in vars(zoom_viewer()).items():
        setattr(viewer, name, value)
    viewer.n_nodes = 4
    viewer.node_render_order = np.arange(4, dtype=np.int32)
    viewer.current_colors = np.tile([0.25, 0.5, 1.0, 1.0], (4, 1)).astype(np.float32)
    viewer.current_sizes = np.full(4, 10.0, dtype=np.float32)
    viewer.current_shapes = np.full(4, "disc", dtype=object)
    viewer.selected_indices = list(selected)
    viewer.selected_node_idx = clicked
    viewer.left_click_highlight_indices = None
    viewer.hovered_node_idx = hovered
    viewer.edges = np.empty((0, 2), dtype=np.int32)
    viewer.markers = FakeMarkers()
    viewer._update_hud_elements = mock.Mock()
    viewer.update_nodes()
    return viewer


def drawn(viewer):
    """What the Viewer submitted to its markers, copied."""
    data = {
        name: (np.array(value, copy=True) if isinstance(value, np.ndarray) else list(value))
        for name, value in viewer.markers.data.items()
    }
    data["node_order"] = viewer._submitted_marker_node_order.copy()
    data["rings"] = viewer._submitted_marker_ring_mask.copy()
    return data


def node_slot(data, node):
    """The marker slot of node itself, not of its ring."""
    return int(np.flatnonzero((data["node_order"] == node) & ~data["rings"])[0])
