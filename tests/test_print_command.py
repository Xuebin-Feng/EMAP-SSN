"""Print command (src/commands/print.py), loaded once with stub Command_Engine
and EMAPSSN_Config modules: HUD hiding during capture, PNG margin trimming,
output-name confinement, and the layered SVG export (edges filtered like the
screen, configured colours, a failure with nothing written when every node is
hidden), plus what the command checks before it writes (options, visible nodes),
when it makes the save folder, and its automatic names."""

import datetime
import importlib.util
import io
import os
import re
import sys
import tempfile
import types
import unittest
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
    import desktop.Desktop_App  # noqa: F401
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


class PrintMetadataHUDTests(unittest.TestCase):
    def test_metadata_hud_is_hidden_during_capture_then_restored(self):
        metadata_visual = SimpleNamespace(visible=True)
        viewer = SimpleNamespace(
            canvas=SimpleNamespace(
                bgcolor="white",
                update=mock.Mock(),
            ),
            console_bg=SimpleNamespace(visible=True),
            console_text=SimpleNamespace(text="previous message"),
            hidden_text=SimpleNamespace(visible=True),
            hud_displays={
                "meta_display": SimpleNamespace(text_visual=metadata_visual)
            },
            instr_text=SimpleNamespace(visible=True),
            tooltip=SimpleNamespace(visible=True),
            view=SimpleNamespace(
                camera=SimpleNamespace(rect="original rect", aspect=1.0)
            ),
            zoom_text=SimpleNamespace(visible=True),
        )

        def capture_while_hud_is_hidden(_viewer, _is_transparent):
            self.assertFalse(metadata_visual.visible)
            return np.zeros((2, 2, 4), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            print_command, "PRINT_DIRECTORY", temp_dir
        ), mock.patch.object(
            print_command, "_capture_tile", side_effect=capture_while_hud_is_hidden
        ), mock.patch.object(
            print_command.mpimg, "imsave"
        ), mock.patch.object(
            print_command, "open_in_file_manager"
        ), mock.patch.object(
            print_command.app, "process_events"
        ):
            print_command.run(viewer, ["metadata_hud"])

        self.assertTrue(metadata_visual.visible)


class PrintMarginTrimTests(unittest.TestCase):
    @staticmethod
    def make_viewer(background="white"):
        return SimpleNamespace(
            canvas=SimpleNamespace(
                bgcolor=background,
                update=mock.Mock(),
            ),
            console_bg=SimpleNamespace(visible=True),
            console_text=SimpleNamespace(text="previous message"),
            hidden_text=SimpleNamespace(visible=True),
            hud_displays={},
            instr_text=SimpleNamespace(visible=True),
            tooltip=SimpleNamespace(visible=True),
            view=SimpleNamespace(
                camera=SimpleNamespace(rect="original rect", aspect=1.0)
            ),
            zoom_text=SimpleNamespace(visible=True),
        )

    def test_transparent_content_is_cropped_with_twenty_pixel_border(self):
        image = np.zeros((100, 120, 4), dtype=np.float32)
        image[40:50, 60:70, :3] = 0.5
        image[40:50, 60:70, 3] = 1.0

        cropped = print_command._trim_png_margins(image, True, "white")

        self.assertEqual(cropped.shape, (50, 50, 4))
        np.testing.assert_array_equal(cropped, image[20:70, 40:90])

    def test_normal_content_uses_custom_background_color(self):
        background = (0.2, 0.4, 0.6, 1.0)
        background_color = SimpleNamespace(rgba=background)
        image = np.empty((100, 120, 4), dtype=np.float32)
        image[...] = background
        image[35:45, 50:65, :3] = (0.9, 0.1, 0.2)

        cropped = print_command._trim_png_margins(
            image,
            False,
            background_color,
        )

        self.assertEqual(cropped.shape, (50, 55, 4))
        np.testing.assert_array_equal(cropped, image[15:65, 30:85])

    def test_normal_background_tolerance_uses_rendered_eight_bit_color(self):
        rendered_background = round(0.1 * 255.0) / 255.0
        image = np.ones((12, 14, 4), dtype=np.float32)
        image[..., :3] = rendered_background
        image[5, 5, 0] += 1.0 / 255.0
        image[8, 9, 0] += 3.0 / 255.0

        cropped = print_command._trim_png_margins(
            image,
            False,
            (0.1, 0.1, 0.1, 1.0),
            padding_px=0,
        )

        self.assertEqual(cropped.shape, (1, 1, 4))
        np.testing.assert_array_equal(cropped[0, 0], image[8, 9])

    def test_padding_clamps_to_image_bounds(self):
        image = np.zeros((60, 70, 4), dtype=np.float32)
        image[5:10, 3:8, 3] = 1.0

        cropped = print_command._trim_png_margins(image, True, "white")

        self.assertEqual(cropped.shape, (30, 28, 4))
        np.testing.assert_array_equal(cropped, image[:30, :28])

    def test_blank_and_full_frame_images_remain_unchanged(self):
        blank = np.zeros((20, 30, 4), dtype=np.float32)
        full = np.ones((20, 30, 4), dtype=np.float32)

        self.assertIs(
            print_command._trim_png_margins(blank, True, "white"),
            blank,
        )
        full_result = print_command._trim_png_margins(full, True, "white")
        self.assertEqual(full_result.shape, full.shape)
        np.testing.assert_array_equal(full_result, full)

    def test_run_saves_trimmed_normal_and_transparent_png_arrays(self):
        for is_transparent in (False, True):
            with self.subTest(is_transparent=is_transparent):
                if is_transparent:
                    captured = np.zeros((100, 120, 4), dtype=np.float32)
                    captured[40:50, 60:70, :3] = 0.5
                    captured[40:50, 60:70, 3] = 1.0
                else:
                    captured = np.ones((100, 120, 4), dtype=np.float32)
                    captured[40:50, 60:70, :3] = 0.0

                viewer = self.make_viewer()
                arguments = ["trimmed"]
                if is_transparent:
                    arguments.append("transparent")

                output = io.StringIO()
                with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
                    print_command, "PRINT_DIRECTORY", temp_dir
                ), mock.patch.object(
                    print_command, "_capture_tile", return_value=captured
                ) as capture, mock.patch.object(
                    print_command.mpimg, "imsave"
                ) as save, mock.patch.object(
                    print_command, "open_in_file_manager"
                ), mock.patch.object(
                    print_command.app, "process_events"
                ), redirect_stdout(output):
                    print_command.run(viewer, arguments)

                capture.assert_called_once_with(viewer, is_transparent)
                saved_image = save.call_args.args[1]
                self.assertEqual(saved_image.shape, (50, 50, 4))
                self.assertIn("120x100 -> 50x50 px", output.getvalue())

    def test_full_stitched_png_uses_shared_final_trim_before_save(self):
        viewer = self.make_viewer()
        camera_rect = SimpleNamespace(width=100.0, height=100.0)
        viewer.view.camera.rect = camera_rect
        viewer.view.camera._real_rect = camera_rect
        viewer.visible_mask = np.array([True, True])
        viewer.pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        captured_tile = np.ones((100, 100, 4), dtype=np.float32)
        trimmed_image = np.ones((2, 3, 4), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            print_command, "PRINT_DIRECTORY", temp_dir
        ), mock.patch.object(
            print_command, "_capture_tile", return_value=captured_tile
        ), mock.patch.object(
            print_command, "_trim_png_margins", return_value=trimmed_image
        ) as trim, mock.patch.object(
            print_command.mpimg, "imsave"
        ) as save, mock.patch.object(
            print_command, "open_in_file_manager"
        ), mock.patch.object(
            print_command.app, "process_events"
        ):
            print_command.run(viewer, ["stitched", "full"])

        trim.assert_called_once()
        self.assertIs(save.call_args.args[1], trimmed_image)

    def test_svg_export_does_not_use_png_trimming(self):
        viewer = self.make_viewer()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            print_command, "PRINT_DIRECTORY", temp_dir
        ), mock.patch.object(
            print_command, "_export_svg"
        ) as export_svg, mock.patch.object(
            print_command, "_trim_png_margins"
        ) as trim, mock.patch.object(
            print_command, "open_in_file_manager"
        ), mock.patch.object(
            print_command.app, "process_events"
        ):
            print_command.run(viewer, ["vector", "svg"])

        export_svg.assert_called_once()
        trim.assert_not_called()


class PrintOutputNameTests(unittest.TestCase):
    def run_print(self, save_dir, arguments):
        viewer = PrintMarginTrimTests.make_viewer()
        failed = print_command.Command_Engine.command_failed
        failed.reset_mock()
        with mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)
        ) as capture, mock.patch.object(
            print_command, "_export_svg"
        ) as export_svg, mock.patch.object(
            print_command.mpimg, "imsave"
        ) as save, mock.patch.object(
            print_command, "open_in_file_manager"
        ), mock.patch.object(
            print_command.app, "process_events"
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, arguments)
        return viewer, failed, capture, export_svg, save

    def test_a_path_is_refused_before_anything_is_rendered(self):
        # os.path.join(save_dir, name) used to drop save_dir for an absolute
        # name, and "..\" climbed out of it.
        with tempfile.TemporaryDirectory() as root:
            save_dir = os.path.join(root, "Saved_Images")
            outside = os.path.join(root, "outside", "escaped")
            for arguments in ([r"..\escaped"], ["../escaped", "transparent"],
                              [outside], [outside, "svg"]):
                with self.subTest(arguments=arguments):
                    viewer, failed, capture, export_svg, save = self.run_print(
                        save_dir, arguments
                    )
                    capture.assert_not_called()
                    export_svg.assert_not_called()
                    save.assert_not_called()
                    self.assertIn("path separators", str(failed.call_args.args[1]))
                    self.assertTrue(viewer.instr_text.visible)
            # Nothing was written, so the save folder was not even made.
            self.assertEqual(os.listdir(root), [])

    def test_plain_names_are_joined_into_the_save_directory(self):
        with tempfile.TemporaryDirectory() as save_dir:
            _, failed, _, _, save = self.run_print(save_dir, ["my", "network"])
        failed.assert_not_called()
        self.assertEqual(save.call_args.args[0], os.path.join(save_dir, "my_network.png"))


def square_network(**overrides):
    """Four visible nodes on a square; two of the four edges pass a 0.5 threshold."""
    viewer = SimpleNamespace(
        visible_mask=np.ones(4, dtype=bool),
        pos=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
        current_colors=np.tile([0.25, 0.5, 1.0, 1.0], (4, 1)),
        current_sizes=np.full(4, 10.0),
        current_shapes=np.array(["disc"] * 4, dtype=object),
        canvas=SimpleNamespace(bgcolor=SimpleNamespace(rgba=(1.0, 1.0, 1.0, 1.0))),
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
            print_command._export_svg(viewer, path)
        with open(path, encoding="utf-8") as handle:
            return handle.read()


def elements(svg, tag):
    return [line.strip() for line in svg.splitlines() if line.strip().startswith(f"<{tag} ")]


class PrintSvgExportTests(unittest.TestCase):
    def test_svg_edges_follow_the_on_screen_similarity_threshold(self):
        viewer = square_network()

        lines = elements(export_svg(viewer), "line")

        # Scores 0.9 and 0.95 pass the 0.5 slider; 0.2 and 0.1 are hidden on screen.
        self.assertEqual(len(lines), 2)

    def test_svg_edges_match_shared_render_filter(self):
        viewer = square_network(current_slider_threshold=0.15)
        expected, _ = print_command.edge_stages(viewer, SimpleNamespace(UMAP_MODE=False))

        lines = elements(export_svg(viewer), "line")

        self.assertEqual(len(lines), len(expected))
        self.assertEqual(len(lines), 3)

    def test_svg_in_umap_mode_draws_only_edges_of_selected_nodes(self):
        viewer = square_network(current_slider_threshold=0.0, selected_indices=[0])

        lines = elements(export_svg(viewer, UMAP_MODE=True), "line")

        # Node 0 touches edges (0, 1) and (3, 0), matching the UMAP-mode display.
        self.assertEqual(len(lines), 2)

    def test_svg_uses_configured_edge_and_node_boundary_colors(self):
        svg = export_svg(
            square_network(),
            EDGE_COLOR="#ff0000",
            NODE_BOUNDARY_COLOR="#00ff00",
            EDGE_ALPHA=0.4,
        )

        lines = elements(svg, "line")
        circles = elements(svg, "circle")
        self.assertTrue(lines)
        self.assertTrue(all('stroke="rgb(255,0,0)" stroke-opacity="0.400"' in line for line in lines))
        self.assertEqual(len(circles), 4)
        self.assertTrue(all('stroke="rgb(0,255,0)"' in circle for circle in circles))


class PrintSvgOutcomeTests(unittest.TestCase):
    """What `print <name> svg` writes, reports and opens."""

    def run_svg_print(self, visible_mask):
        viewer = square_network(visible_mask=np.array(visible_mask))
        for name, value in vars(PrintMarginTrimTests.make_viewer()).items():
            if name != "canvas":
                setattr(viewer, name, value)
        viewer.canvas.update = mock.Mock()
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        with tempfile.TemporaryDirectory() as save_dir, mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command, "open_in_file_manager"
        ) as open_folder, mock.patch.object(
            print_command.app, "process_events"
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, ["network", "svg"])
            written = os.listdir(save_dir)
        return viewer, engine, open_folder, written, save_dir

    def test_svg_with_every_node_hidden_fails_and_writes_nothing(self):
        viewer, engine, open_folder, written, _ = self.run_svg_print([False] * 4)

        self.assertEqual(written, [])
        engine.command_failed.assert_called_once()
        self.assertEqual(str(engine.command_failed.call_args.args[1]), "Error: No visible nodes to export.")
        engine.command_succeeded.assert_not_called()
        engine.command_artifact.assert_not_called()
        open_folder.assert_not_called()
        self.assertEqual(viewer.console_text.text, "Error: No visible nodes to export.")
        self.assertTrue(viewer.instr_text.visible)

    def test_svg_with_visible_nodes_reports_the_written_file(self):
        viewer, engine, open_folder, written, save_dir = self.run_svg_print([True] * 4)

        self.assertEqual(written, ["network.svg"])
        engine.command_artifact.assert_called_once_with(
            viewer, os.path.join(save_dir, "network.svg")
        )
        engine.command_failed.assert_not_called()
        engine.command_succeeded.assert_called_once()
        open_folder.assert_called_once_with(save_dir)
        # The format is named without the extension's dot ("Saved .SVG" before).
        self.assertEqual(viewer.console_text.text, "Saved SVG: network.svg")


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


class PrintSvgNodeOrderTests(unittest.TestCase):
    """The SVG stacks nodes as the screen does: the later covers the earlier."""

    def test_nodes_are_written_in_the_order_the_viewer_draws_them(self):
        viewer = square_network(
            current_colors=numbered_colors(),
            visible_node_render_order=lambda: np.array([2, 0, 3, 1], dtype=np.int32),
        )

        self.assertEqual(node_fills(export_svg(viewer)), [25, 5, 35, 15])

    def test_hidden_nodes_stay_out_in_the_viewers_order(self):
        # The viewer's order lists visible nodes only.
        viewer = square_network(
            current_colors=numbered_colors(),
            visible_mask=np.array([False, True, False, True]),
            visible_node_render_order=lambda: np.array([3, 1], dtype=np.int32),
        )

        self.assertEqual(node_fills(export_svg(viewer)), [35, 15])

    def test_a_viewer_without_a_render_order_keeps_index_order(self):
        viewer = square_network(
            current_colors=numbered_colors(),
            visible_mask=np.array([True, False, True, True]),
        )

        self.assertEqual(node_fills(export_svg(viewer)), [5, 25, 35])

    def test_promoted_and_selected_nodes_are_written_last_like_on_screen(self):
        import EMAPSSN_Config
        from EMAPSSN_Viewer import MainViewer

        def real_viewer(**overrides):
            viewer = MainViewer.__new__(MainViewer)
            for name, value in vars(
                square_network(current_colors=numbered_colors(), **overrides)
            ).items():
                setattr(viewer, name, value)
            viewer.n_nodes = 4
            viewer.node_render_order = np.arange(4, dtype=np.int32)
            viewer.selected_node_idx = None
            viewer.left_click_highlight_indices = None
            return viewer

        # What the color command does to the nodes it recolours.
        promoted = real_viewer()
        promoted.promote_nodes([1])
        self.assertEqual(node_fills(export_svg(promoted)), [5, 25, 35, 15])

        # Node 0 is selected, so its neighbours 1 and 3 are raised below it.
        with mock.patch.multiple(
            EMAPSSN_Config, create=True,
            CONNECTED_NODE_COLOR="red", NODE_BOUNDARY_COLOR="black",
        ):
            selected = real_viewer(selected_indices=[0])
            self.assertEqual(node_fills(export_svg(selected)), [25, 15, 35, 5])


class PrintSvgShapeTests(unittest.TestCase):
    ALIASES = {
        "o": "disc", "circle": "disc", "s": "square", "^": "triangle_up",
        "triangle": "triangle_up", "v": "triangle_down", "D": "diamond",
        "*": "star", "+": "cross", "|": "vbar", "-": "hbar", "_": "hbar",
        ">": "arrow", "->": "tailed_arrow", "p": "clobber",
        "P": "cross_lines", "++": "cross_lines",
    }

    @staticmethod
    def with_shape(shape):
        return export_svg(square_network(current_shapes=np.array([shape] * 4, dtype=object)))

    def test_the_table_covers_every_vispy_marker_alias(self):
        from vispy.visuals.markers import symbol_aliases

        for alias, canonical in symbol_aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(print_command._canonical_shape(alias), canonical)

    def test_an_alias_is_drawn_exactly_like_its_canonical_name(self):
        for alias, canonical in self.ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(self.with_shape(alias), self.with_shape(canonical))

    def test_stroke_aliases_keep_the_node_colour_and_have_no_fill(self):
        # '+', '|' and '-' were drawn as a filled path with a black outline.
        for alias, tag in (("+", "path"), ("|", "line"), ("-", "line"), ("_", "line")):
            with self.subTest(alias=alias):
                nodes = self.with_shape(alias).split('<g id="nodes"')[1]
                drawn = elements(nodes, tag)
                self.assertEqual(len(drawn), 4)
                for element in drawn:
                    self.assertIn('fill="none"', element)
                    self.assertIn('stroke="rgb(63,127,255)"', element)

    def test_an_unknown_shape_is_still_drawn_as_a_circle(self):
        self.assertEqual(self.with_shape("blob"), self.with_shape("disc"))
        self.assertIsNone(print_command._canonical_shape(None))


class PrintSvgPaddingTests(unittest.TestCase):
    @staticmethod
    def view_box(svg):
        match = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        return float(match.group(1)), float(match.group(2))

    def test_large_nodes_at_the_edge_fit_inside_the_view_box(self):
        svg = export_svg(square_network(current_sizes=np.full(4, 40.0)))

        width, height = self.view_box(svg)
        for circle in elements(svg, "circle"):
            cx, cy, r = (
                float(re.search(rf'{name}="([\d.]+)"', circle).group(1))
                for name in ("cx", "cy", "r")
            )
            self.assertGreaterEqual(cx - r, 0.0)
            self.assertLessEqual(cx + r, width)
            self.assertGreaterEqual(cy - r, 0.0)
            self.assertLessEqual(cy + r, height)

    def test_a_ring_is_padded_for_its_wide_stroke(self):
        svg = export_svg(
            square_network(
                current_sizes=np.full(4, 40.0),
                current_shapes=np.array(["ring"] * 4, dtype=object),
            )
        )

        # Radius 20 plus half of its 8-unit stroke, on each side of the 10-unit square.
        self.assertEqual(self.view_box(svg), (58.0, 58.0))

    def test_small_nodes_keep_the_five_unit_padding(self):
        svg = export_svg(square_network(current_sizes=np.full(4, 2.0)))

        self.assertEqual(self.view_box(svg), (20.0, 20.0))

    def test_a_wide_network_keeps_its_five_percent_padding(self):
        viewer = square_network(
            pos=np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 200.0], [0.0, 200.0]])
        )

        self.assertEqual(self.view_box(export_svg(viewer)), (440.0, 220.0))


class PrintCommandChecksTests(unittest.TestCase):
    """What `print` refuses, and what it leaves untouched, before it writes."""

    def run_print(self, arguments, visible=True):
        viewer = PrintMarginTrimTests.make_viewer()
        viewer.visible_mask = np.array([visible] * 4)
        viewer.pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        camera_rect = SimpleNamespace(width=100.0, height=100.0)
        viewer.view.camera.rect = camera_rect
        viewer.view.camera._real_rect = camera_rect
        viewer.canvas.update = mock.Mock()
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        save_dir = os.path.join(root.name, "Saved_Images")
        with mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command, "_capture_tile", return_value=np.ones((100, 100, 4), dtype=np.float32)
        ) as capture, mock.patch.object(
            print_command.mpimg, "imsave"
        ) as save, mock.patch.object(
            print_command, "open_in_file_manager"
        ), mock.patch.object(
            print_command.app, "process_events"
        ), redirect_stdout(io.StringIO()) as output:
            print_command.run(viewer, arguments)
        return viewer, engine, capture, save, save_dir, output.getvalue()

    def test_an_unknown_option_is_refused_instead_of_becoming_a_file_name(self):
        cases = (
            (["--transparent"], "--transparent"),
            (["-x"], "-x"),
            (["name", "--full"], "--full"),
            (["name", "-", "svg"], "-"),
        )
        for arguments, option in cases:
            with self.subTest(arguments=arguments):
                viewer, engine, capture, save, save_dir, _ = self.run_print(arguments)

                capture.assert_not_called()
                save.assert_not_called()
                engine.command_failed.assert_called_once()
                refusal = str(engine.command_failed.call_args.args[1])
                self.assertTrue(refusal.startswith(f"Error: Unknown option '{option}'."))
                self.assertEqual(viewer.console_text.text, refusal)
                self.assertFalse(os.path.exists(save_dir))

    def test_a_dash_inside_a_name_and_the_help_flags_still_work(self):
        _, engine, _, save, save_dir, _ = self.run_print(["my-network"])
        engine.command_failed.assert_not_called()
        self.assertEqual(save.call_args.args[0], os.path.join(save_dir, "my-network.png"))

        for flag in ("help", "-h", "--help"):
            with self.subTest(flag=flag):
                viewer, engine, capture, save, _, output = self.run_print([flag])
                capture.assert_not_called()
                self.assertIn("SSN Image Export", output)
                engine.command_failed.assert_not_called()

    def test_help_and_refusals_do_not_create_the_save_folder(self):
        for arguments in (["help"], ["-h"], ["--help"], ["../x"], ["--bad"], ["svg", "full"]):
            with self.subTest(arguments=arguments):
                save_dir = self.run_print(arguments)[4]
                self.assertFalse(os.path.exists(save_dir))

    def test_a_png_makes_the_save_folder_when_it_writes(self):
        for arguments in (["picture"], ["picture", "transparent"], ["picture", "full"]):
            with self.subTest(arguments=arguments):
                _, engine, _, save, save_dir, _ = self.run_print(arguments)
                engine.command_failed.assert_not_called()
                save.assert_called_once()
                self.assertTrue(os.path.isdir(save_dir))

    def test_an_svg_makes_the_save_folder_when_it_writes(self):
        viewer = square_network()
        for name, value in vars(PrintMarginTrimTests.make_viewer()).items():
            if name != "canvas":
                setattr(viewer, name, value)
        viewer.canvas.update = mock.Mock()
        with tempfile.TemporaryDirectory() as root:
            save_dir = os.path.join(root, "Saved_Images")
            with mock.patch.object(
                print_command, "PRINT_DIRECTORY", save_dir
            ), mock.patch.object(
                print_command, "open_in_file_manager"
            ), mock.patch.multiple(
                print_command.cfg, create=True, EDGE_COLOR="#000000",
                NODE_BOUNDARY_COLOR="#000000", EDGE_ALPHA=0.2, EDGE_WIDTH=0.5,
                UMAP_MODE=False,
            ), redirect_stdout(io.StringIO()):
                print_command.run(viewer, ["network", "svg"])
            self.assertEqual(os.listdir(save_dir), ["network.svg"])

    def test_png_with_every_node_hidden_is_refused_like_svg_and_full(self):
        for arguments in (["picture"], ["picture", "transparent"], []):
            with self.subTest(arguments=arguments):
                viewer, engine, capture, save, save_dir, _ = self.run_print(
                    arguments, visible=False
                )

                capture.assert_not_called()
                save.assert_not_called()
                engine.command_failed.assert_called_once()
                self.assertEqual(
                    str(engine.command_failed.call_args.args[1]),
                    "Error: No visible nodes to export.",
                )
                engine.command_succeeded.assert_not_called()
                engine.command_artifact.assert_not_called()
                self.assertEqual(viewer.console_text.text, "Error: No visible nodes to export.")
                self.assertTrue(viewer.instr_text.visible)
                self.assertFalse(os.path.exists(save_dir))

    def test_png_with_visible_nodes_is_still_saved(self):
        viewer, engine, capture, save, save_dir, _ = self.run_print(["picture"])

        engine.command_failed.assert_not_called()
        capture.assert_called_once()
        self.assertEqual(save.call_args.args[0], os.path.join(save_dir, "picture.png"))


class PrintAutomaticNameTests(unittest.TestCase):
    """Two prints in one second must not overwrite each other."""

    class FixedClock:
        @staticmethod
        def now():
            return datetime.datetime(2026, 1, 2, 3, 4, 5)

    def print_in_this_second(self, save_dir, arguments, svg_viewer=False):
        viewer = square_network() if svg_viewer else PrintMarginTrimTests.make_viewer()
        if svg_viewer:
            for name, value in vars(PrintMarginTrimTests.make_viewer()).items():
                if name != "canvas":
                    setattr(viewer, name, value)
            viewer.canvas.update = mock.Mock()
        else:
            viewer.visible_mask = np.ones(4, dtype=bool)
        with mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command, "datetime", SimpleNamespace(datetime=self.FixedClock)
        ), mock.patch.object(
            print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)
        ), mock.patch.object(
            print_command, "open_in_file_manager"
        ), mock.patch.object(
            print_command.app, "process_events"
        ), mock.patch.multiple(
            print_command.cfg, create=True, EDGE_COLOR="#000000",
            NODE_BOUNDARY_COLOR="#000000", EDGE_ALPHA=0.2, EDGE_WIDTH=0.5,
            UMAP_MODE=False,
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, arguments)
        return viewer

    def test_the_available_name_gets_the_first_free_numeric_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                print_command._available_automatic_filename(directory, "set_1.png"),
                "set_1.png",
            )
            for taken in ("set_1.png", "set_1_2.png"):
                open(os.path.join(directory, taken), "w").close()
            self.assertEqual(
                print_command._available_automatic_filename(directory, "set_1.png"),
                "set_1_3.png",
            )
            self.assertEqual(
                print_command._available_automatic_filename(directory, "set_2.png"),
                "set_2.png",
            )

    def test_prints_in_the_same_second_are_saved_side_by_side(self):
        with tempfile.TemporaryDirectory() as save_dir:
            for _ in range(3):
                viewer = self.print_in_this_second(save_dir, [])
            self.assertEqual(
                sorted(os.listdir(save_dir)),
                [
                    "test_sequences_20260102_030405.png",
                    "test_sequences_20260102_030405_2.png",
                    "test_sequences_20260102_030405_3.png",
                ],
            )
            # The console line names the file that was written.
            self.assertEqual(
                viewer.console_text.text, "Saved PNG: test_sequences_20260102_030405_3.png"
            )

    def test_automatic_svg_names_are_numbered_too(self):
        with tempfile.TemporaryDirectory() as save_dir:
            for _ in range(2):
                self.print_in_this_second(save_dir, ["svg"], svg_viewer=True)
            self.assertEqual(
                sorted(os.listdir(save_dir)),
                ["test_sequences_20260102_030405.svg", "test_sequences_20260102_030405_2.svg"],
            )

    def test_an_explicit_name_is_still_overwritten(self):
        with tempfile.TemporaryDirectory() as save_dir:
            for _ in range(2):
                self.print_in_this_second(save_dir, ["mine"])
            self.assertEqual(os.listdir(save_dir), ["mine.png"])


if __name__ == "__main__":
    unittest.main()
