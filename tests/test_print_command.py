"""Print command (src/commands/print.py), loaded once with stub Command_Engine
and EMAPSSN_Config modules: HUD hiding during capture, PNG margin trimming,
output-name confinement, and the layered SVG export (edges filtered like the
screen, configured colours, a failure with nothing written when every node is
hidden), plus what the command checks before it writes (options, visible nodes),
when it makes the save folder, and its automatic names; the `zoom N` scale of
full and svg, and the hover colour and click halo a PNG leaves out; the file
name as one word, an .svg name selecting SVG, and the report of a file replaced."""

import contextlib
import datetime
import importlib.util
import io
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
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
    import commands.zoom  # noqa: F401
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

    def test_full_stitched_png_saves_the_float_picture_trimmed_as_eight_bit(self):
        # Pixels at one scene unit each; the network spans 165 units with its
        # padding, so 9 tiles keep their 70-pixel middles (15-pixel margins)
        # and the 210-pixel mosaic is cropped to 165.
        for is_transparent in (False, True):
            with self.subTest(is_transparent=is_transparent):
                viewer = self.make_viewer()
                camera_rect = SimpleNamespace(width=100.0, height=100.0)
                viewer.view.camera.rect = camera_rect
                viewer.view.camera._real_rect = camera_rect
                viewer.visible_mask = np.array([True, True])
                viewer.pos = np.array([[0.0, 0.0, 0.0], [150.0, 150.0, 0.0]])
                rng = np.random.default_rng(4)
                tiles = []
                for index in range(10):
                    tile = np.zeros((100, 100, 4), dtype=np.float32)
                    if not is_transparent:
                        tile[...] = 1.0
                    # Marks in the middle tile only (call 6: row 1, column 1),
                    # so the trim has margins to take off.
                    if index == 5:
                        spots = np.zeros((100, 100), dtype=bool)
                        spots[40:60, 40:60] = rng.random((20, 20)) < 0.2
                        tile[spots] = rng.random((int(spots.sum()), 4)).astype(np.float32) * 0.5
                    tiles.append(tile)
                # The float picture the stitching used to hold, cropped and trimmed.
                mosaic = np.zeros((210, 210, 4), dtype=np.float32)
                for index, tile in enumerate(tiles[1:]):
                    row, col = divmod(index, 3)
                    mosaic[row * 70:(row + 1) * 70, col * 70:(col + 1) * 70] = tile[15:85, 15:85]
                expected = print_command._trim_png_margins(mosaic[:165, :165], is_transparent, "white")

                output = io.StringIO()
                with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
                    print_command, "PRINT_DIRECTORY", temp_dir
                ), mock.patch.object(
                    print_command, "_capture_tile", side_effect=tiles
                ), mock.patch.object(
                    print_command.mpimg, "imsave"
                ) as save, mock.patch.object(
                    print_command, "open_in_file_manager"
                ), mock.patch.object(
                    print_command.app, "process_events"
                ), redirect_stdout(output):
                    print_command.run(viewer, ["stitched", "full"] + (["transparent"] if is_transparent else []))

                saved = save.call_args.args[1]
                self.assertEqual(saved.dtype, np.uint8)
                np.testing.assert_array_equal(saved, (expected * 255).astype(np.uint8))
                self.assertLess(saved.shape[:2], (165, 165))
                self.assertIn(f"165x165 -> {saved.shape[1]}x{saved.shape[0]} px", output.getvalue())


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


class PrintPngMosaicTests(unittest.TestCase):
    """The full capture holds its picture as the 8-bit RGBA imsave writes."""

    def test_eight_bit_pixels_are_those_matplotlib_writes_for_the_float_image(self):
        from matplotlib.colorizer import Colorizer

        rng = np.random.default_rng(9)
        image = rng.random((40, 50, 4)).astype(np.float32)
        image[rng.random((40, 50)) < 0.05] = 1.0
        image[rng.random((40, 50)) < 0.05] = 0.0
        image[3, 4, 2] = np.nan

        np.testing.assert_array_equal(
            print_command._png_bytes(image), Colorizer().to_rgba(image, bytes=True)
        )
        # The float tile is left as it is.
        self.assertTrue(np.isnan(image[3, 4, 2]))
        with self.assertRaisesRegex(ValueError, r"must be in the \[0,1\] range"):
            print_command._png_bytes(np.full((2, 2, 4), 1.5, dtype=np.float32))

    def test_the_trim_finds_the_margins_of_the_float_picture(self):
        rng = np.random.default_rng(12)
        for case in range(60):
            is_transparent = bool(case % 2)
            background = "white" if case % 3 else (0.2, 0.4, 0.6, 1.0)
            tile_h, tile_w = int(rng.integers(5, 30)), int(rng.integers(5, 30))
            rows, cols = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            crop_h = int(rng.integers(1, rows * tile_h + 5))
            crop_w = int(rng.integers(1, cols * tile_w + 5))
            with self.subTest(case=case):
                mosaic = print_command._PngMosaic(
                    rows * tile_h, cols * tile_w, crop_h, crop_w, is_transparent, background
                )
                picture = np.zeros((rows * tile_h, cols * tile_w, 4), dtype=np.float32)
                for row in range(rows):
                    for col in range(cols):
                        tile = np.zeros((tile_h, tile_w, 4), dtype=np.float32)
                        if not is_transparent:
                            tile[..., :3] = print_command._background_rgb(
                                SimpleNamespace(rgba=background) if isinstance(background, tuple) else background
                            )
                            tile[..., 3] = 1.0
                        spots = rng.random((tile_h, tile_w)) < 0.01
                        tile[spots] = rng.random((int(spots.sum()), 4)).astype(np.float32)
                        mosaic.paste(tile, row * tile_h, col * tile_w)
                        picture[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = tile
                padding = int(rng.integers(0, 25))
                expected = print_command._trim_png_margins(
                    picture[:crop_h, :crop_w], is_transparent,
                    SimpleNamespace(rgba=background) if isinstance(background, tuple) else background,
                    padding_px=padding,
                )

                self.assertEqual(mosaic.cropped_size(), picture[:crop_h, :crop_w].shape[:2])
                np.testing.assert_array_equal(
                    mosaic.trimmed(padding), (expected * 255).astype(np.uint8)
                )

    def test_content_cut_off_by_the_crop_does_not_widen_the_trim(self):
        # The only mark in rows 10-11 lies right of the crop; the trim ignores it.
        tile = np.zeros((20, 30, 4), dtype=np.float32)
        tile[2, 3, 3] = 1.0
        tile[10:12, 25, 3] = 1.0
        mosaic = print_command._PngMosaic(20, 30, 20, 20, True, "white")
        mosaic.paste(tile, 0, 0)

        self.assertEqual(mosaic.trimmed(padding_px=0).shape, (1, 1, 4))

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
            _, failed, _, _, save = self.run_print(save_dir, ["my_network"])
        failed.assert_not_called()
        self.assertEqual(save.call_args.args[0], os.path.join(save_dir, "my_network.png"))


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


class PrintSvgStreamingTests(unittest.TestCase):
    """The SVG is written a chunk at a time into a file that replaces the target when done."""

    @staticmethod
    def mixed_network(count=23):
        rng = np.random.default_rng(5)
        shapes = ["disc", "ring", "square", "^", "v", "D", "*", "+", "x", "|", "-", ">", "p", "P", "blob"]
        return square_network(
            visible_mask=rng.random(count) < 0.9,
            pos=(rng.random((count, 2)) * 50).astype(np.float32),
            current_colors=rng.random((count, 4)).astype(np.float32),
            current_sizes=rng.choice([4.0, 10.0, 17.5], count).astype(np.float32),
            current_shapes=np.array([shapes[i % len(shapes)] for i in range(count)], dtype=object),
            edges=rng.integers(0, count, (60, 2)).astype(np.int32),
            edge_scores=rng.random(60),
            current_slider_threshold=0.3,
        )

    def test_the_file_is_the_same_whatever_the_chunk_size(self):
        viewer = self.mixed_network()
        whole = export_svg(viewer)
        self.assertGreater(len(elements(whole, "line")), 10)
        for chunk in (1, 2, 7):
            with self.subTest(chunk=chunk), mock.patch.object(print_command, "SVG_CHUNK_SIZE", chunk):
                self.assertEqual(export_svg(viewer), whole)

    def test_a_failure_while_writing_keeps_the_earlier_file_and_leaves_nothing_behind(self):
        viewer = self.mixed_network()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.svg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("earlier picture")

            def broken_nodes(*arguments):
                yield "    <circle />\n"
                raise RuntimeError("disk full")

            with mock.patch.multiple(print_command.cfg, create=True, UMAP_MODE=False), \
                    mock.patch.object(print_command, "_svg_node_chunks", broken_nodes), \
                    redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "disk full"):
                print_command._export_svg(viewer, path)

            self.assertEqual(os.listdir(directory), ["network.svg"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "earlier picture")


class PrintSvgOutcomeTests(unittest.TestCase):
    """What `print <name> svg` writes, reports and opens."""

    def run_svg_print(self, visible_mask):
        viewer = square_network(visible_mask=np.array(visible_mask))
        for name, value in vars(PrintMarginTrimTests.make_viewer()).items():
            if name not in ("canvas", "view"):
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

        # At one unit per pixel: radius 20 plus half of its 8-unit stroke, on
        # each side of the 10-unit square.
        self.assertEqual(self.view_box(svg), (58.0, 58.0))

    def test_padding_follows_the_node_radius_in_scene_units(self):
        # 40-pixel nodes at half a scene unit per pixel have a radius of 10 units.
        sizes = np.full(4, 40.0)
        discs = export_svg(square_network(view=view_at(0.5), current_sizes=sizes))
        rings = export_svg(
            square_network(
                view=view_at(0.5),
                current_sizes=sizes,
                current_shapes=np.array(["ring"] * 4, dtype=object),
            )
        )

        self.assertEqual(self.view_box(discs), (30.0, 30.0))
        # The ring's stroke reaches 1.2 times as far: 12 units each side.
        self.assertEqual(self.view_box(rings), (34.0, 34.0))

    def test_small_nodes_keep_the_five_unit_padding(self):
        svg = export_svg(square_network(current_sizes=np.full(4, 2.0)))

        self.assertEqual(self.view_box(svg), (20.0, 20.0))

    def test_a_wide_network_keeps_its_five_percent_padding(self):
        viewer = square_network(
            pos=np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 200.0], [0.0, 200.0]])
        )

        self.assertEqual(self.view_box(export_svg(viewer)), (440.0, 220.0))


def number(element, name):
    """The numeric attribute `name` of one SVG element line."""
    return float(re.search(rf'(?<![\w-]){name}="(-?[\d.]+)"', element).group(1))


class PrintSvgScaleTests(unittest.TestCase):
    """Sizes set in screen pixels are written in scene units at the zoom shown.

    The view below shows 160 scene units across its 800 pixels, so one pixel
    is 0.2 units and a 12-pixel node has a radius of 6 pixels, or 1.2 units.
    """

    UNITS_PER_PIXEL = 0.2

    def network(self, **overrides):
        overrides.setdefault("view", view_at(self.UNITS_PER_PIXEL))
        overrides.setdefault("current_sizes", np.full(4, 12.0))
        return square_network(**overrides)

    def test_node_radius_is_half_its_size_in_pixels_in_scene_units(self):
        circles = elements(export_svg(self.network()), "circle")

        self.assertEqual(len(circles), 4)
        for circle in circles:
            self.assertAlmostEqual(number(circle, "r"), 12 / 2 * self.UNITS_PER_PIXEL)

    def test_every_shape_is_scaled_by_the_same_factor(self):
        squares = export_svg(
            self.network(current_shapes=np.array(["square"] * 4, dtype=object))
        )
        crosses = export_svg(
            self.network(current_shapes=np.array(["cross"] * 4, dtype=object))
        )

        for square in elements(squares, "rect")[1:]:  # the first is the background
            self.assertAlmostEqual(number(square, "width"), 12 * self.UNITS_PER_PIXEL)
        # A stroke-only shape is drawn 0.4 of its radius wide.
        for cross in elements(crosses, "path"):
            self.assertAlmostEqual(
                number(cross, "stroke-width"), 0.4 * 6 * self.UNITS_PER_PIXEL
            )

    def test_edge_and_node_outline_widths_are_converted_too(self):
        svg = export_svg(self.network(), EDGE_WIDTH=1.5)

        lines = elements(svg, "line")
        self.assertTrue(lines)
        for line in lines:
            self.assertAlmostEqual(number(line, "stroke-width"), 1.5 * self.UNITS_PER_PIXEL)
        # The 0.5-pixel outline of every filled node.
        for circle in elements(svg, "circle"):
            self.assertAlmostEqual(number(circle, "stroke-width"), 0.5 * self.UNITS_PER_PIXEL)

    def test_node_outline_follows_the_boundary_width_setting(self):
        svg = export_svg(self.network(), NODE_BOUNDARY_WIDTH=2.0)

        for circle in elements(svg, "circle"):
            self.assertAlmostEqual(number(circle, "stroke-width"), 2.0 * self.UNITS_PER_PIXEL)

    def test_doubling_the_zoom_halves_the_sizes_relative_to_the_scene(self):
        def drawn(units_per_pixel):
            svg = export_svg(self.network(view=view_at(units_per_pixel)), EDGE_WIDTH=1.5)
            return (
                elements(svg, "circle")[0],
                elements(svg, "line")[0],
            )

        circle, line = drawn(0.2)
        zoomed_circle, zoomed_line = drawn(0.1)

        self.assertAlmostEqual(number(zoomed_circle, "r"), number(circle, "r") / 2)
        self.assertAlmostEqual(
            number(zoomed_circle, "stroke-width"), number(circle, "stroke-width") / 2
        )
        self.assertAlmostEqual(
            number(zoomed_line, "stroke-width"), number(line, "stroke-width") / 2
        )
        # The network itself keeps its scene coordinates.
        for name in ("cx", "cy"):
            self.assertEqual(number(zoomed_circle, name), number(circle, name))
        for name in ("x1", "y1", "x2", "y2"):
            self.assertEqual(number(zoomed_line, name), number(line, name))

    def test_a_node_is_drawn_as_wide_next_to_its_neighbour_as_on_screen(self):
        # Nodes 0 and 1 are 10 scene units apart: 50 pixels on screen at 0.2 per pixel.
        circles = elements(export_svg(self.network()), "circle")
        centres = [number(circle, "cx") for circle in circles[:2]]

        svg_ratio = 2 * number(circles[0], "r") / abs(centres[1] - centres[0])

        self.assertAlmostEqual(svg_ratio, 12 / 50)

    def test_the_view_scale_comes_from_the_camera_real_rect(self):
        # The camera widens the rect it was asked for to the view's aspect
        # ratio; the widened one is what lies across the view's pixels.
        view = view_at(self.UNITS_PER_PIXEL)
        view.camera.rect = SimpleNamespace(width=100.0, height=100.0)

        circles = elements(export_svg(self.network(view=view)), "circle")

        self.assertAlmostEqual(number(circles[0], "r"), 12 / 2 * self.UNITS_PER_PIXEL)

    def test_a_camera_without_a_real_rect_uses_its_rect(self):
        view = view_at(self.UNITS_PER_PIXEL)
        del view.camera._real_rect

        circles = elements(export_svg(self.network(view=view)), "circle")

        self.assertAlmostEqual(number(circles[0], "r"), 12 / 2 * self.UNITS_PER_PIXEL)

    def test_a_high_dpi_display_changes_nothing(self):
        # Sizes are logical pixels, which VisPy scales to the display itself,
        # and the view is measured in them: only the physical size doubles.
        sharp = self.network()
        sharp.canvas.size = (800, 600)
        sharp.canvas.physical_size = (1600, 1200)
        sharp.canvas.pixel_scale = 2.0

        self.assertEqual(export_svg(sharp), export_svg(self.network()))

    def test_a_view_without_area_leaves_sizes_as_they_are(self):
        svg = export_svg(self.network(view=view_at(self.UNITS_PER_PIXEL, size=(0, 0))))

        for circle in elements(svg, "circle"):
            self.assertAlmostEqual(number(circle, "r"), 6.0)

    def test_the_scale_matches_the_real_camera_mapping_in_any_view_shape(self):
        # What VisPy itself maps from the scene to the view's pixels, in a view
        # wider than the requested range, taller than it, and after a zoom.
        from vispy import scene

        view = scene.ViewBox()
        view.camera = scene.PanZoomCamera(aspect=1)
        view.camera.set_range(x=(0, 100), y=(0, 100))
        for size, zoom in (((800, 600), 1.0), ((400, 900), 1.0), ((800, 600), 0.5)):
            with self.subTest(size=size, zoom=zoom):
                view.size = size
                view.camera.zoom(zoom)
                transform = view.camera.transform
                pixels_per_unit_x, pixels_per_unit_y = abs(transform.scale[0]), abs(transform.scale[1])

                scale = print_command._scene_units_per_pixel(SimpleNamespace(view=view))

                self.assertAlmostEqual(scale * pixels_per_unit_x, 1.0, places=5)
                self.assertAlmostEqual(scale * pixels_per_unit_y, 1.0, places=5)


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
            if name not in ("canvas", "view"):
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
                if name not in ("canvas", "view"):
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

    def test_an_overwritten_file_is_reported_after_a_real_write(self):
        # The file is looked for before the write: the first print of a name
        # does not report itself, the second one reports the file it replaced.
        replaced = "It replaced an existing file of that name."
        for arguments, svg_viewer, written in (
            (["mine"], False, "Saved PNG: mine.png"),
            (["mine", "svg"], True, "Saved SVG: mine.svg"),
        ):
            with self.subTest(arguments=arguments):
                with tempfile.TemporaryDirectory() as save_dir:
                    first = self.print_in_this_second(save_dir, arguments, svg_viewer=svg_viewer)
                    second = self.print_in_this_second(save_dir, arguments, svg_viewer=svg_viewer)
                self.assertEqual(first.console_text.text, written)
                self.assertEqual(second.console_text.text, f"{written} {replaced}")


# A rendered tile, rows by columns: the 800 x 600 canvas of zoom_viewer().
TILE_SHAPE = (600, 800)


def zoom_viewer():
    """A viewer whose 800 x 600 view shows 100 x 75 scene units, 0.125 per pixel.

    Its four visible nodes span 100 x 50 units; with the 5 units the full
    capture pads each side with, the network is 110 x 60 units.
    """
    viewer = PrintMarginTrimTests.make_viewer()
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


class PrintZoomTests(unittest.TestCase):
    """`zoom N` with full and svg: the scale of the view `zoom N` would set."""

    def run_print(self, arguments, viewer=None):
        viewer = viewer if viewer is not None else zoom_viewer()
        live_rect = viewer.view.camera.rect
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        tile_rects = []

        def capture(captured_viewer, _is_transparent):
            tile_rects.append(captured_viewer.view.camera.rect)
            return np.ones(TILE_SHAPE + (4,), dtype=np.float32)

        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        save_dir = os.path.join(root.name, "Saved_Images")
        with mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command, "_capture_tile", side_effect=capture
        ) as capture_tile, mock.patch.object(
            print_command, "_export_svg", return_value=True
        ) as export_svg, mock.patch.object(
            print_command.mpimg, "imsave"
        ) as save, mock.patch.object(
            print_command, "open_in_file_manager"
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, arguments)
        # The live camera is the one it was before, untouched.
        self.assertIs(viewer.view.camera.rect, live_rect)
        self.assertEqual(
            (live_rect.pos, live_rect.width, live_rect.height), ((0.0, -12.5), 100.0, 75.0)
        )
        return SimpleNamespace(
            viewer=viewer, engine=engine, capture=capture_tile, export_svg=export_svg,
            save=save, save_dir=save_dir, tile_rects=tile_rects,
        )

    def failure(self, result):
        result.engine.command_failed.assert_called_once()
        return str(result.engine.command_failed.call_args.args[1])

    def assert_refused(self, arguments, message):
        result = self.run_print(arguments)
        result.capture.assert_not_called()
        result.export_svg.assert_not_called()
        result.save.assert_not_called()
        result.engine.command_succeeded.assert_not_called()
        self.assertEqual(self.failure(result), message)
        self.assertEqual(result.viewer.console_text.text, message)
        self.assertTrue(result.viewer.instr_text.visible)
        self.assertFalse(os.path.exists(result.save_dir))

    # --- parsing ---------------------------------------------------------

    def test_zoom_n_may_stand_anywhere_among_the_modifiers(self):
        cases = (
            (["full", "zoom", "500"], None),
            (["fig1", "full", "zoom", "500"], "fig1.png"),
            (["zoom", "500", "fig1", "full"], "fig1.png"),
            (["fig1", "zoom", "500", "full", "transparent"], "fig1.png"),
            (["my_net", "zoom", "500", "full"], "my_net.png"),
            (["ZOOM", "500", "Full"], None),
        )
        for arguments, name in cases:
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                result.save.assert_called_once()
                saved = os.path.basename(result.save.call_args.args[0])
                if name is None:
                    self.assertTrue(saved.startswith("test_sequences_"), saved)
                else:
                    self.assertEqual(saved, name)
                # Every tile after the first, which measures the canvas, is 500 units wide.
                self.assertTrue(all(rect[2] == 500.0 for rect in result.tile_rects[1:]))

    def test_svg_takes_zoom_n_and_the_name_alongside_it(self):
        for arguments, name in (
            (["svg", "zoom", "500"], None),
            (["fig1", "svg", "zoom", "500"], "fig1.svg"),
            (["zoom", "500", "svg", "fig1"], "fig1.svg"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                result.export_svg.assert_called_once()
                self.assertEqual(result.export_svg.call_args.kwargs["view_width"], 500.0)
                saved = os.path.basename(result.export_svg.call_args.args[1])
                if name is None:
                    self.assertTrue(saved.startswith("test_sequences_"), saved)
                else:
                    self.assertEqual(saved, name)

    def test_without_zoom_the_svg_keeps_the_view_scale(self):
        result = self.run_print(["fig1", "svg"])
        self.assertIsNone(result.export_svg.call_args.kwargs["view_width"])

    def test_a_name_such_as_zoom_png_is_still_a_name(self):
        for arguments, name in (
            (["zoom.png"], "zoom.png"),
            (["zoom.png", "full", "zoom", "500"], "zoom.png"),
            (["my_zoom.png"], "my_zoom.png"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertEqual(os.path.basename(result.save.call_args.args[0]), name)

    def test_zoom_without_a_width_is_refused(self):
        message = "Error: zoom needs a view width, as in print full zoom 500."
        for arguments in (["full", "zoom"], ["zoom", "full"], ["svg", "zoom", "svg"],
                          ["fig1", "zoom", "transparent", "full"], ["zoom"]):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, message)

    def test_an_invalid_width_is_refused_as_the_zoom_command_refuses_it(self):
        cases = (
            ("wide", "Error: Zoom width must be a valid number."),
            ("500.png", "Error: Zoom width must be a valid number."),
            ("0", "Error: Zoom width must be a positive, finite number."),
            ("-5", "Error: Zoom width must be a positive, finite number."),
            ("nan", "Error: Zoom width must be a positive, finite number."),
            ("inf", "Error: Zoom width must be a positive, finite number."),
            ("1e400", "Error: Zoom width must be a positive, finite number."),
            ("1e50", "Error: Zoom width is too large for the current view."),
            ("1e-14", "Error: Zoom width is too small to draw accurately at the current view centre."),
        )
        for width, message in cases:
            for modifier in ("full", "svg"):
                with self.subTest(width=width, modifier=modifier):
                    self.assert_refused([modifier, "zoom", width], message)

    def test_a_canvas_without_area_is_refused(self):
        viewer = zoom_viewer()
        viewer.canvas.size = (0, 600)
        result = self.run_print(["full", "zoom", "500"], viewer)
        result.capture.assert_not_called()
        self.assertEqual(
            self.failure(result),
            "Error: The canvas has no visible area, so the zoom cannot be applied.",
        )

    def test_zoom_given_twice_is_refused(self):
        self.assert_refused(
            ["full", "zoom", "500", "zoom", "300"], "Error: zoom N can be given only once."
        )

    def test_zoom_with_a_plain_png_is_refused(self):
        message = "Error: zoom N works only with full or svg."
        for arguments in (["zoom", "500"], ["fig1", "zoom", "500"],
                          ["fig1", "transparent", "zoom", "500"]):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, message)

    def test_the_svg_rules_still_hold_with_zoom(self):
        self.assert_refused(
            ["svg", "full", "zoom", "500"],
            "Error: 'SVG' export is not compatible with 'transparent' or 'full'.",
        )
        # A second word is refused as a name before the svg rules apply.
        self.assert_refused(
            ["my", "net", "svg", "zoom", "500"],
            "Error: Unrecognized print argument 'net'. A file name is one word; "
            "the modifiers are transparent, full, svg, zoom N.",
        )

    # --- full ------------------------------------------------------------

    def test_full_with_zoom_renders_tiles_n_units_wide_and_leaves_the_camera(self):
        # 50 units across 800 pixels: 0.0625 units per pixel, so the 110 x 60
        # network is 1760 x 960 pixels. The tiles keep the view's 4:3 shape.
        result = self.run_print(["fig1", "full", "zoom", "50"])

        result.engine.command_failed.assert_not_called()
        # The first render measures the canvas from the live camera.
        self.assertEqual(result.tile_rects[0].width, 100.0)
        tiles = result.tile_rects[1:]
        self.assertEqual(len(tiles), 9)
        for rect in tiles:
            self.assertEqual(rect[2:], (50.0, 37.5))
        saved_image = result.save.call_args.args[1]
        self.assertEqual(saved_image.shape, (960, 1760, 4))
        self.assertIs(result.viewer.view.camera._real_rect, result.viewer.view.camera.rect)

    def test_full_without_zoom_renders_at_the_view_scale(self):
        # 0.125 units per pixel: 880 x 480 pixels, in 2 x 2 tiles of the live view's size.
        result = self.run_print(["fig1", "full"])

        tiles = result.tile_rects[1:]
        self.assertEqual(len(tiles), 4)
        for rect in tiles:
            self.assertEqual(rect[2:], (100.0, 75.0))
        self.assertEqual(result.save.call_args.args[1].shape, (480, 880, 4))

    def test_zoom_at_the_view_width_gives_the_same_picture_as_no_zoom(self):
        plain = self.run_print(["fig1", "full"])
        zoomed = self.run_print(["fig1", "full", "zoom", "100"])

        self.assertEqual(plain.tile_rects[1:], zoomed.tile_rects[1:])
        np.testing.assert_array_equal(
            plain.save.call_args.args[1], zoomed.save.call_args.args[1]
        )

    def test_a_small_zoomed_image_is_saved_with_a_warning(self):
        # 500 units across 800 pixels: 0.625 per pixel, so 176 x 96 pixels.
        result = self.run_print(["fig1", "full", "zoom", "500"])

        result.save.assert_called_once()
        self.assertEqual(result.save.call_args.args[1].shape, (96, 176, 4))
        result.engine.command_failed.assert_not_called()
        report = str(result.engine.command_succeeded.call_args.args[1])
        warning = (
            "Warning: the image is only 176×96 px. N in zoom N is the view width, "
            "so a smaller N zooms in and gives a larger image."
        )
        self.assertTrue(report.endswith(warning), report)
        self.assertTrue(report.startswith("Successfully saved full stitched snapshot: "))
        self.assertEqual(result.viewer.console_text.text, f"Saved PNG: fig1.png {warning}")

    def test_the_warning_needs_zoom_and_an_image_under_1000_pixels(self):
        # No zoom: 880 x 480 pixels, and no warning. zoom 50: 1760 x 960, none either.
        for arguments in (["fig1", "full"], ["fig1", "full", "zoom", "50"],
                          ["fig1", "full", "transparent"]):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                report = str(result.engine.command_succeeded.call_args.args[1])
                self.assertNotIn("Warning", report)
                self.assertEqual(result.viewer.console_text.text, "Saved PNG: fig1.png")
        # An SVG has no pixel size, so it never warns.
        result = self.run_print(["fig1", "svg", "zoom", "50000"])
        self.assertNotIn("Warning", str(result.engine.command_succeeded.call_args.args[1]))


class PrintSvgZoomTests(unittest.TestCase):
    """`print svg zoom N` writes the sizes of a view N units wide: N / view width per pixel."""

    def svg(self, arguments, units_per_pixel=0.2):
        viewer = square_network(
            view=view_at(units_per_pixel), current_sizes=np.full(4, 12.0)
        )
        for name, value in vars(PrintMarginTrimTests.make_viewer()).items():
            if name not in ("canvas", "view"):
                setattr(viewer, name, value)
        viewer.view.camera.rect.pos = (0.0, 0.0)
        viewer.canvas.size = viewer.view.size
        viewer.canvas.update = mock.Mock()
        with tempfile.TemporaryDirectory() as save_dir, mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command, "open_in_file_manager"
        ), mock.patch.multiple(
            print_command.cfg, create=True, EDGE_COLOR="#000000",
            NODE_BOUNDARY_COLOR="#000000", EDGE_ALPHA=0.2, EDGE_WIDTH=1.5,
            UMAP_MODE=False,
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, arguments)
            with open(os.path.join(save_dir, "net.svg"), encoding="utf-8") as handle:
                return handle.read()

    def test_node_radius_is_half_its_size_times_n_over_the_view_width(self):
        svg = self.svg(["net", "svg", "zoom", "500"])

        circles = elements(svg, "circle")
        self.assertEqual(len(circles), 4)
        # The SVG is written to three decimals.
        for circle in circles:
            self.assertAlmostEqual(number(circle, "r"), 12 / 2 * 500 / 800, delta=0.001)
            self.assertAlmostEqual(number(circle, "stroke-width"), 0.5 * 500 / 800, delta=0.001)
        for line in elements(svg, "line"):
            self.assertAlmostEqual(number(line, "stroke-width"), 1.5 * 500 / 800, delta=0.001)

    def test_the_scale_ignores_the_current_zoom(self):
        self.assertEqual(
            self.svg(["net", "svg", "zoom", "500"], units_per_pixel=0.2),
            self.svg(["net", "svg", "zoom", "500"], units_per_pixel=0.05),
        )

    def test_zoom_at_the_view_width_writes_the_same_svg_as_no_zoom(self):
        # The view at 0.2 units per pixel is 160 units wide.
        self.assertEqual(self.svg(["net", "svg"]), self.svg(["net", "svg", "zoom", "160"]))

    def test_the_scale_function_takes_the_view_width(self):
        viewer = SimpleNamespace(view=view_at(0.2))
        self.assertAlmostEqual(print_command._scene_units_per_pixel(viewer), 0.2)
        self.assertAlmostEqual(print_command._scene_units_per_pixel(viewer, 400.0), 0.5)


class PrintFileNameTests(unittest.TestCase):
    """The file name is one word, an .svg name selects SVG, a replaced file is reported."""

    REPLACED = "It replaced an existing file of that name."

    def run_print(self, arguments, existing=()):
        """Run print with rendering and saving mocked; `existing` are files already saved."""
        viewer = zoom_viewer()
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        save_dir = os.path.join(root.name, "Saved_Images")
        if existing:
            os.makedirs(save_dir)
            for name in existing:
                open(os.path.join(save_dir, name), "w").close()
        with mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command, "_capture_tile",
            return_value=np.ones(TILE_SHAPE + (4,), dtype=np.float32),
        ) as capture, mock.patch.object(
            print_command, "_export_svg", return_value=True
        ) as export_svg, mock.patch.object(
            print_command.mpimg, "imsave"
        ) as save, mock.patch.object(
            print_command, "open_in_file_manager"
        ), redirect_stdout(io.StringIO()) as output:
            print_command.run(viewer, arguments)
        return SimpleNamespace(
            viewer=viewer, engine=engine, capture=capture, export_svg=export_svg,
            save=save, save_dir=save_dir, output=output.getvalue(),
        )

    def written(self, result):
        """The base name of the one file the run wrote, as a PNG or an SVG."""
        if result.export_svg.called:
            result.save.assert_not_called()
            result.export_svg.assert_called_once()
            return os.path.basename(result.export_svg.call_args.args[1])
        result.export_svg.assert_not_called()
        result.save.assert_called_once()
        return os.path.basename(result.save.call_args.args[0])

    def assert_refused(self, arguments, message):
        result = self.run_print(arguments)
        result.capture.assert_not_called()
        result.export_svg.assert_not_called()
        result.save.assert_not_called()
        result.engine.command_succeeded.assert_not_called()
        result.engine.command_failed.assert_called_once()
        self.assertEqual(str(result.engine.command_failed.call_args.args[1]), message)
        self.assertEqual(result.viewer.console_text.text, message)
        self.assertTrue(result.viewer.instr_text.visible)
        self.assertFalse(os.path.exists(result.save_dir))

    # --- one word --------------------------------------------------------

    def test_a_second_word_is_refused_and_named_instead_of_joining_the_name(self):
        # "fig1 ful" used to save fig1_ful.png, "my full network" my_network.png
        # and "a b" a_b.png; "a b svg" was refused for another reason.
        template = (
            "Error: Unrecognized print argument '{}'. A file name is one word; "
            "the modifiers are transparent, full, svg, zoom N."
        )
        for arguments, second in (
            (["fig1", "ful"], "ful"),
            (["my", "full", "network"], "network"),
            (["a", "b"], "b"),
            (["a", "b", "svg"], "b"),
            (["my", "zoom.png"], "zoom.png"),
            (["fig1", "transparent", "fig2", "full"], "fig2"),
            (["one", "two", "three"], "two"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, template.format(second))

    def test_modifiers_may_still_stand_anywhere_around_the_name(self):
        for arguments, name in (
            (["fig1"], "fig1.png"),
            (["transparent", "fig1"], "fig1.png"),
            (["fig1", "transparent"], "fig1.png"),
            (["full", "transparent", "fig1"], "fig1.png"),
            (["fig1", "FULL", "Transparent"], "fig1.png"),
            (["full", "zoom", "500", "fig1"], "fig1.png"),
            (["svg", "fig1"], "fig1.svg"),
            (["fig1", "svg"], "fig1.svg"),
            (["zoom", "500", "svg", "fig1"], "fig1.svg"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertEqual(self.written(result), name)

    def test_a_repeated_modifier_is_accepted(self):
        # The SVG "maximum of 2 keywords" rule counted words; with one name word
        # and the modifiers refused beside svg, nothing is left for it to catch.
        for arguments, name in ((["fig1", "svg", "svg", "svg"], "fig1.svg"),
                                (["fig1", "full", "full"], "fig1.png"),
                                (["fig1", "transparent", "transparent"], "fig1.png")):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertEqual(self.written(result), name)

    # --- the extension picks the format ----------------------------------

    def test_a_name_ending_in_svg_saves_an_svg_as_the_modifier_does(self):
        # `print foo.svg` in PNG mode used to write foo.svg.png.
        for arguments, name in (
            (["foo.svg"], "foo.svg"),
            (["FOO.SVG"], "FOO.SVG"),
            (["Foo.Svg", "svg"], "Foo.Svg"),
            (["svg", "foo.svg"], "foo.svg"),
            (["a.b.svg"], "a.b.svg"),
            (["foo.png.svg"], "foo.png.svg"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                result.capture.assert_not_called()
                self.assertEqual(self.written(result), name)
                self.assertIsNone(result.export_svg.call_args.kwargs["view_width"])
                self.assertEqual(result.viewer.console_text.text, f"Saved SVG: {name}")
                self.assertTrue(
                    str(result.engine.command_succeeded.call_args.args[1]).startswith(
                        "Successfully saved SVG snapshot: "
                    )
                )

    def test_an_svg_name_takes_zoom_n_like_the_modifier(self):
        for arguments in (["foo.svg", "zoom", "500"], ["zoom", "500", "foo.svg"]):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertEqual(self.written(result), "foo.svg")
                self.assertEqual(result.export_svg.call_args.kwargs["view_width"], 500.0)

    def test_the_svg_rules_apply_to_an_svg_name(self):
        message = "Error: 'SVG' export is not compatible with 'transparent' or 'full'."
        for arguments in (["foo.svg", "full"], ["foo.svg", "transparent"],
                          ["transparent", "full", "foo.svg"], ["foo.svg", "full", "zoom", "500"]):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, message)

    def test_a_png_name_with_svg_is_refused_as_a_mismatch(self):
        for arguments, message in (
            (["foo.png", "svg"],
             "Error: 'foo.png' ends in .png, but svg saves an SVG file. "
             "Use 'foo.svg' or leave the extension off."),
            (["svg", "FOO.PNG"],
             "Error: 'FOO.PNG' ends in .PNG, but svg saves an SVG file. "
             "Use 'FOO.svg' or leave the extension off."),
            (["foo.png", "svg", "zoom", "500"],
             "Error: 'foo.png' ends in .png, but svg saves an SVG file. "
             "Use 'foo.svg' or leave the extension off."),
        ):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, message)

    def test_a_png_name_in_png_mode_is_unchanged(self):
        for arguments, name in (
            (["foo.png"], "foo.png"),
            (["FOO.PNG"], "FOO.PNG"),
            (["foo.png", "full"], "foo.png"),
            (["foo.png", "transparent"], "foo.png"),
            (["foo.jpg"], "foo.jpg.png"),
            (["foo"], "foo.png"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                result.export_svg.assert_not_called()
                self.assertEqual(self.written(result), name)

    def test_a_name_that_is_only_an_extension_is_refused(self):
        # `print .svg` would write a file named .svg; logo refuses such a name too.
        for arguments, shown in (
            ([".svg"], ".svg"), ([".png"], ".png"), ([".SVG"], ".SVG"), ([".Png"], ".Png"),
            (["svg", ".svg"], ".svg"), ([".png", "full"], ".png"),
            ([".png", "transparent"], ".png"), ([".svg", "zoom", "500"], ".svg"),
            # A mismatch would otherwise suggest the nonsense name ".svg".
            ([".png", "svg"], ".png"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_refused(
                    arguments, f"Error: Filename '{shown}' needs a name before its extension."
                )

    def test_a_name_with_something_before_its_extension_is_still_accepted(self):
        for arguments, name in ((["a.svg"], "a.svg"), (["a.png"], "a.png"),
                                ([".hidden"], ".hidden.png"), (["x.jpg"], "x.jpg.png")):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertEqual(self.written(result), name)

    def test_an_svg_name_that_is_a_path_is_still_refused(self):
        result = self.run_print(["../foo.svg"])
        result.export_svg.assert_not_called()
        result.engine.command_failed.assert_called_once()
        self.assertIn("path separators", str(result.engine.command_failed.call_args.args[1]))
        self.assertFalse(os.path.exists(result.save_dir))

    # --- an explicit name replaces a file, and the report says so ---------

    def test_a_replaced_file_is_reported(self):
        for arguments, existing, shown in (
            (["mine"], "mine.png", "Saved PNG: mine.png"),
            (["mine.png", "transparent"], "mine.png", "Saved PNG: mine.png"),
            (["mine", "full"], "mine.png", "Saved PNG: mine.png"),
            (["mine", "svg"], "mine.svg", "Saved SVG: mine.svg"),
            (["mine.svg"], "mine.svg", "Saved SVG: mine.svg"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments, existing=(existing,))
                result.engine.command_failed.assert_not_called()
                self.assertEqual(result.viewer.console_text.text, f"{shown} {self.REPLACED}")
                report = str(result.engine.command_succeeded.call_args.args[1])
                self.assertTrue(report.startswith("Successfully saved "), report)
                self.assertTrue(report.endswith(f" {self.REPLACED}"), report)
                self.assertIn(self.REPLACED, result.output)

    def test_a_new_name_is_not_reported_as_replacing_anything(self):
        # Another file in the folder, or a file of the other format, is no replacement.
        for arguments, existing in (
            (["mine"], ()),
            (["mine"], ("other.png", "mine.svg")),
            (["mine", "svg"], ("mine.png",)),
            ([], ("mine.png",)),
        ):
            with self.subTest(arguments=arguments, existing=existing):
                result = self.run_print(arguments, existing=existing)
                result.engine.command_failed.assert_not_called()
                self.assertNotIn(self.REPLACED, result.viewer.console_text.text)
                self.assertNotIn(self.REPLACED, str(result.engine.command_succeeded.call_args.args[1]))
                self.assertNotIn(self.REPLACED, result.output)

    def test_the_replacement_note_and_the_size_warning_both_ride_along(self):
        # 500 units across 800 pixels gives a 176 x 96 picture: the warning applies.
        result = self.run_print(["mine", "full", "zoom", "500"], existing=("mine.png",))
        warning = (
            "Warning: the image is only 176×96 px. N in zoom N is the view width, "
            "so a smaller N zooms in and gives a larger image."
        )
        self.assertEqual(
            result.viewer.console_text.text, f"Saved PNG: mine.png {self.REPLACED} {warning}"
        )
        report = str(result.engine.command_succeeded.call_args.args[1])
        self.assertTrue(report.endswith(f"{self.REPLACED} {warning}"), report)

    def test_a_refusal_never_reports_a_replacement(self):
        result = self.run_print(["mine", "ful"], existing=("mine.png",))
        result.engine.command_succeeded.assert_not_called()
        self.assertNotIn(self.REPLACED, result.viewer.console_text.text)

    def test_the_help_describes_the_one_word_name_the_extension_and_the_replacement(self):
        with mock.patch("builtins.print") as printed:
            print_command.print_help()
        text = " ".join(printed.call_args.args[0].split())
        self.assertIn("FILENAME is a plain file name of one word", text)
        self.assertIn("A name ending in .svg saves an SVG, as the svg modifier does", text)
        self.assertIn("a name ending in .png cannot be combined with svg", text)
        self.assertIn("A file of the same name is replaced, and the report says so", text)
        self.assertIn("print my_network.svg", text)


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


class PrintTransientMarksTests(unittest.TestCase):
    """A PNG leaves out the hover colour and the left-click halo, and keeps
    the selection borders; the live view gets both marks back afterwards."""

    def setUp(self):
        import matplotlib.colors as mcolors
        import EMAPSSN_Config

        self.hover_rgba = np.array(mcolors.to_rgba(EMAPSSN_Config.HOVER_COLOR), dtype=np.float32)
        patcher = mock.patch.multiple(
            EMAPSSN_Config, create=True, CONNECTED_NODE_COLOR="red", NODE_BOUNDARY_COLOR="black",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_print(self, viewer, arguments, capture_error=None, interrupt=False):
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        during = []

        def capture(captured_viewer, _is_transparent):
            hidden = getattr(captured_viewer, "transient_marks_hidden", False)
            during.append((drawn(captured_viewer), hidden))
            if capture_error is not None:
                raise capture_error
            return np.ones(TILE_SHAPE + (4,), dtype=np.float32)

        save_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, save_dir, True)
        patches = [
            mock.patch.object(print_command, "PRINT_DIRECTORY", save_dir),
            mock.patch.object(print_command, "_capture_tile", side_effect=capture),
            mock.patch.object(print_command.mpimg, "imsave"),
            mock.patch.object(print_command, "open_in_file_manager"),
        ]
        if interrupt:
            patches.append(mock.patch.object(
                print_command._CaptureEvents, "pause_if_due",
                side_effect=print_command._CaptureInterrupted(print_command.Message("window resized")),
            ))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            print_command.run(viewer, arguments)
        return engine, during

    def assert_drawn_without_transient_marks(self, viewer, before, during):
        self.assertTrue(during)
        for data, hidden in during:
            self.assertTrue(hidden)
            # No halo: one marker per visible node, in the same order as before.
            self.assertFalse(data["rings"].any())
            np.testing.assert_array_equal(
                data["node_order"], before["node_order"][~before["rings"]]
            )
            # The hovered node has its own colour, not the hover colour.
            np.testing.assert_array_equal(
                data["face_color"][node_slot(data, 1)], viewer.current_colors[1]
            )
            # The clicked node is drawn as before, without its ring.
            np.testing.assert_array_equal(
                data["face_color"][node_slot(data, 2)],
                before["face_color"][node_slot(before, 2)],
            )
            # Selection borders stay: colour and width of the selected node.
            for name in ("edge_color", "edge_width"):
                np.testing.assert_array_equal(
                    data[name][node_slot(data, 0)], before[name][node_slot(before, 0)]
                )
            np.testing.assert_array_equal(data["edge_color"][node_slot(data, 0)], self.hover_rgba)
            self.assertEqual(data["edge_width"][node_slot(data, 0)], 2.0)

    def assert_restored(self, viewer, before):
        after = drawn(viewer)
        self.assertFalse(viewer.transient_marks_hidden)
        self.assertEqual(viewer.hovered_node_idx, 1)
        self.assertEqual(viewer.selected_node_idx, 2)
        self.assertEqual(viewer.selected_indices, [0])
        self.assertEqual(set(after), set(before))
        for name in before:
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)
        # The mouse is still over node 1, so it has the hover colour again,
        # and node 2 its halo.
        np.testing.assert_array_equal(after["face_color"][node_slot(after, 1)], self.hover_rgba)
        self.assertEqual(after["node_order"][after["rings"]].tolist(), [2])

    def test_the_viewer_draws_the_marks_this_test_relies_on(self):
        before = drawn(marked_viewer())
        np.testing.assert_array_equal(before["face_color"][node_slot(before, 1)], self.hover_rgba)
        self.assertEqual(before["node_order"][before["rings"]].tolist(), [2])

    def test_a_png_leaves_out_hover_and_halo_and_gets_them_back(self):
        for arguments in (["marks"], ["marks", "transparent"], ["marks", "full"],
                          ["marks", "full", "zoom", "50"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer()
                before = drawn(viewer)

                engine, during = self.run_print(viewer, arguments)

                engine.command_failed.assert_not_called()
                engine.command_succeeded.assert_called_once()
                self.assert_drawn_without_transient_marks(viewer, before, during)
                self.assert_restored(viewer, before)

    def test_the_marks_come_back_when_the_capture_fails(self):
        for arguments in (["marks"], ["marks", "full"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer()
                before = drawn(viewer)

                engine, during = self.run_print(
                    viewer, arguments, capture_error=RuntimeError("render failed")
                )

                engine.command_failed.assert_called_once()
                engine.command_succeeded.assert_not_called()
                self.assert_drawn_without_transient_marks(viewer, before, during)
                self.assert_restored(viewer, before)

    def test_the_marks_come_back_when_a_full_capture_is_cancelled(self):
        viewer = marked_viewer()
        before = drawn(viewer)

        engine, during = self.run_print(viewer, ["marks", "full"], interrupt=True)

        engine.command_failed.assert_called_once()
        self.assertIn("window resized", str(engine.command_failed.call_args.args[1]))
        self.assert_drawn_without_transient_marks(viewer, before, during)
        self.assert_restored(viewer, before)

    def test_an_svg_and_a_refusal_leave_the_drawing_alone(self):
        viewer = marked_viewer()
        with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update, \
                mock.patch.object(print_command, "_export_svg", return_value=True):
            self.run_print(viewer, ["marks", "svg"])
            self.run_print(viewer, ["marks", "zoom", "500"])
        update.assert_not_called()

    def test_a_network_without_marks_is_not_redrawn(self):
        viewer = marked_viewer(hovered=None, clicked=None)
        with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update:
            engine, during = self.run_print(viewer, ["marks"])
        engine.command_failed.assert_not_called()
        update.assert_not_called()
        self.assertFalse(during[0][1])


if __name__ == "__main__":
    unittest.main()
