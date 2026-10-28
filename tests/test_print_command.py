"""Print command (src/commands/print.py), loaded once with stub Command_Engine
and EMAPSSN_Config modules: HUD hiding during capture, PNG margin trimming,
output-name confinement, and the layered SVG export (edges filtered like the
screen, configured colours, a failure with nothing written when every node is
hidden), plus what the command checks before it writes (options, visible nodes),
when it makes the save folder, and its automatic names; the SVG's opacity bands under Edge
Opacity's Auto switch; the `zoom N` scale of
full and svg, and the hover colour and click halo a PNG leaves out; the file
name as one word, an .svg name selecting SVG, and the report of a file replaced;
the background job that finishes and writes the file, in the viewer's queue; and
the tiles of a full capture drawn in slices by the command queue, the event loop,
input included, running between them."""

import contextlib
import datetime
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
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

from tests.command_fixtures import end_scheduler  # noqa: E402
from tests.held_file_fixtures import held_file_error, hold_without_delete_sharing, transient_hold  # noqa: E402
from utilities import Job_Cancellation  # noqa: E402


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


class PrintMetadataHUDTests(unittest.TestCase):
    def test_metadata_hud_is_hidden_during_capture_then_restored(self):
        metadata_visual = SimpleNamespace(visible=True)
        viewer = SimpleNamespace(
            background_job_scheduler=InlineScheduler(),
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
            print_command, "_write_png"
        ) as write_png, mock.patch.object(
            print_command.app, "process_events"
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, ["metadata_hud"])

        write_png.assert_called_once()
        self.assertTrue(metadata_visual.visible)


class PrintMarginTrimTests(unittest.TestCase):
    @staticmethod
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
                    print_command, "_write_png"
                ) as save, mock.patch.object(
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
                renders = []
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
                    renders.append(raw_renders(tile, is_transparent))
                # The float picture the stitching used to hold, from the float
                # tiles the renders made, cropped and trimmed.
                mosaic = np.zeros((210, 210, 4), dtype=np.float32)
                for index, tile_renders in enumerate(renders[1:]):
                    row, col = divmod(index, 3)
                    tile = print_command._tile_rgba(tile_renders)
                    mosaic[row * 70:(row + 1) * 70, col * 70:(col + 1) * 70] = tile[15:85, 15:85]
                expected = print_command._trim_png_margins(mosaic[:165, :165], is_transparent, "white")

                output = io.StringIO()
                with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
                    print_command, "PRINT_DIRECTORY", temp_dir
                ), mock.patch.object(
                    print_command, "_render_tile", side_effect=renders
                ), mock.patch.object(
                    print_command, "_write_png"
                ) as save, mock.patch.object(
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
            print_command, "_svg_snapshot"
        ), mock.patch.object(
            print_command, "_write_svg"
        ) as write_svg, mock.patch.object(
            print_command, "_trim_png_margins"
        ) as trim, mock.patch.object(
            print_command.app, "process_events"
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, ["vector", "svg"])

        write_svg.assert_called_once()
        trim.assert_not_called()


class PrintPngMosaicTests(unittest.TestCase):
    """The full capture keeps its tiles' renders and finishes them in place
    into the 8-bit RGBA imsave writes for the float picture."""

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

    def test_a_renders_png_bytes_are_the_render_itself(self):
        # So a plain capture's renders are kept as the PNG's pixels unchanged.
        values = np.arange(256, dtype=np.uint8)
        render = np.stack([values, values[::-1], values, values[::-1]], axis=-1).reshape(16, 16, 4)
        np.testing.assert_array_equal(
            print_command._png_bytes(print_command._tile_rgba((render,))), render
        )

    @staticmethod
    def random_renders(rng, shape, is_transparent, background_rgb):
        """A tile's renders: background with a few random marks."""
        tile = np.zeros(shape + (4,), dtype=np.float32)
        if not is_transparent:
            tile[..., :3] = background_rgb
            tile[..., 3] = 1.0
        spots = rng.random(shape) < 0.02
        tile[spots] = rng.random((int(spots.sum()), 4)).astype(np.float32)
        renders = raw_renders(tile, is_transparent)
        # Renders on black and white that disagree, as anti-aliasing can make them.
        if is_transparent:
            noise = rng.random(shape) < 0.01
            renders[1][noise] = rng.integers(0, 256, (int(noise.sum()), 3))
        return renders

    def test_the_picture_and_its_trim_are_those_of_the_float_picture(self):
        rng = np.random.default_rng(12)
        for case in range(60):
            is_transparent = bool(case % 2)
            background = "white" if case % 3 else SimpleNamespace(rgba=(0.2, 0.4, 0.6, 1.0))
            background_rgb = print_command._background_rgb(background)
            tile_h, tile_w = int(rng.integers(5, 30)), int(rng.integers(5, 30))
            rows, cols = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            crop_h = int(rng.integers(1, rows * tile_h + 1))
            crop_w = int(rng.integers(1, cols * tile_w + 1))
            # Bands of a few rows, one row, or the whole picture at once.
            band_pixels = int(rng.choice([1, 37, 1 << 22]))
            with self.subTest(case=case), mock.patch.object(
                print_command, "FINISH_BAND_PIXELS", band_pixels
            ):
                mosaic = print_command._RawMosaic(crop_h, crop_w, is_transparent)
                picture = np.zeros((rows * tile_h, cols * tile_w, 4), dtype=np.float32)
                for row in range(rows):
                    for col in range(cols):
                        renders = self.random_renders(rng, (tile_h, tile_w), is_transparent, background_rgb)
                        mosaic.paste(renders, row * tile_h, col * tile_w)
                        picture[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = (
                            print_command._tile_rgba(renders)
                        )
                padding = int(rng.integers(0, 25))
                expected = print_command._trim_png_margins(
                    picture[:crop_h, :crop_w], is_transparent, background, padding_px=padding,
                )

                self.assertEqual(mosaic.shape, (crop_h, crop_w))
                np.testing.assert_array_equal(
                    mosaic.finish(background, padding), print_command._png_bytes(expected)
                )

    def test_content_cut_off_by_the_crop_does_not_widen_the_trim(self):
        # The only mark in rows 10-11 lies right of the crop; the trim ignores it.
        tile = np.zeros((20, 30, 4), dtype=np.float32)
        tile[2, 3, 3] = 1.0
        tile[10:12, 25, 3] = 1.0
        mosaic = print_command._RawMosaic(20, 20, True)
        mosaic.paste(raw_renders(tile, True), 0, 0)

        self.assertEqual(mosaic.finish("white", padding_px=0).shape, (1, 1, 4))

    def test_a_tile_past_the_edge_is_cut_and_one_outside_is_left_out(self):
        mosaic = print_command._RawMosaic(5, 7, False)
        tile = np.full((4, 4, 4), 9, dtype=np.uint8)
        mosaic.paste((tile,), 3, 5)
        mosaic.paste((tile,), 5, 0)
        mosaic.paste((tile,), 0, 7)
        expected = np.zeros((5, 7, 4), dtype=np.uint8)
        expected[3:, 5:] = 9
        np.testing.assert_array_equal(mosaic.image, expected)

    def test_a_render_without_alpha_is_opaque(self):
        mosaic = print_command._RawMosaic(2, 2, False)
        mosaic.paste((np.full((2, 2, 3), 7, dtype=np.uint8),), 0, 0)
        np.testing.assert_array_equal(mosaic.image[..., :3], 7)
        np.testing.assert_array_equal(mosaic.image[..., 3], 255)

    def test_the_finished_picture_is_the_mosaics_own_memory_in_c_order(self):
        # imsave writes a C-ordered array as it is; a cut-out would be copied.
        for is_transparent in (False, True):
            with self.subTest(is_transparent=is_transparent):
                tile = np.zeros((40, 50, 4), dtype=np.float32)
                if not is_transparent:
                    tile[...] = 1.0
                tile[25:28, 30:33] = (0.5, 0.25, 0.75, 1.0)
                mosaic = print_command._RawMosaic(40, 50, is_transparent)
                mosaic.paste(raw_renders(tile, is_transparent), 0, 0)
                memory = mosaic.image

                picture = mosaic.finish("white", padding_px=2)

                self.assertEqual(picture.shape, (7, 7, 4))
                self.assertTrue(picture.flags.c_contiguous)
                self.assertTrue(np.shares_memory(picture, memory))
                self.assertIsNone(mosaic.on_white)

    def test_compacting_keeps_every_box_of_the_picture(self):
        rng = np.random.default_rng(8)
        for case in range(40):
            height, width = int(rng.integers(1, 12)), int(rng.integers(1, 12))
            top, bottom = sorted(int(value) for value in rng.integers(0, height + 1, 2))
            left, right = sorted(int(value) for value in rng.integers(0, width + 1, 2))
            bottom, right = max(bottom, top + 1), max(right, left + 1)
            if bottom > height or right > width:
                continue
            with self.subTest(case=case):
                image = rng.integers(0, 256, (height, width, 4)).astype(np.uint8)
                expected = image[top:bottom, left:right].copy()
                compact = print_command._compact(image, (top, bottom, left, right))
                self.assertTrue(compact.flags.c_contiguous)
                self.assertTrue(np.shares_memory(compact, image))
                np.testing.assert_array_equal(compact, expected)


class PrintPngEncodingTests(unittest.TestCase):
    """The PNG goes into a partial file that replaces the target when it is complete."""

    def test_the_file_is_the_one_imsave_writes(self):
        rng = np.random.default_rng(3)
        images = (
            rng.random((37, 53, 4)).astype(np.float32),
            (rng.random((20, 31, 4)) * 255).astype(np.uint8),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, image in enumerate(images):
                with self.subTest(dtype=str(image.dtype)):
                    written = os.path.join(directory, f"written{index}.png")
                    direct = os.path.join(directory, f"direct{index}.png")
                    print_command._write_png(written, image)
                    print_command.mpimg.imsave(direct, image)
                    with open(written, "rb") as a, open(direct, "rb") as b:
                        self.assertEqual(a.read(), b.read())
            self.assertEqual(
                sorted(os.listdir(directory)),
                ["direct0.png", "direct1.png", "written0.png", "written1.png"],
            )

    def test_a_failure_keeps_the_earlier_file_and_leaves_nothing_behind(self):
        error = OSError("disk full")

        def broken_encode(target, image, **kwargs):
            # imsave writes into the partial file through the cancellable file the PNG goes out by.
            target.write(b"half a picture")
            raise error

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "picture.png")
            with open(path, "wb") as handle:
                handle.write(b"earlier picture")
            with mock.patch.object(print_command.mpimg, "imsave", side_effect=broken_encode), \
                    self.assertRaises(OSError) as raised:
                print_command._write_png(path, np.zeros((2, 2, 4), dtype=np.uint8))

            self.assertIs(raised.exception, error)
            self.assertEqual(os.listdir(directory), ["picture.png"])
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"earlier picture")

    def test_a_virus_scan_holding_the_finished_file_for_a_moment_does_not_fail_the_picture(self):
        # Windows refuses the rename (WinError 32) while a scan has the file just written open.
        image = (np.random.default_rng(3).random((20, 31, 4)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "picture.png")
            direct = os.path.join(directory, "direct.png")
            print_command.mpimg.imsave(direct, image)
            with transient_hold(3) as (attempts, pauses):
                print_command._write_png(path, image)
            self.assertEqual(len(attempts), 4)
            self.assertEqual(len(pauses), 3)
            with open(path, "rb") as written, open(direct, "rb") as expected:
                self.assertEqual(written.read(), expected.read())
            self.assertEqual(sorted(os.listdir(directory)), ["direct.png", "picture.png"])

    @unittest.skipUnless(sys.platform == "win32", "holds a file open as a Windows virus scanner does")
    def test_a_program_really_holding_the_picture_it_replaces_for_a_moment_does_not_fail_the_picture(self):
        image = (np.random.default_rng(3).random((20, 31, 4)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "picture.png")
            direct = os.path.join(directory, "direct.png")
            print_command.mpimg.imsave(direct, image)
            with open(path, "wb") as handle:
                handle.write(b"earlier picture")
            # Windows refuses to replace a file that is open without delete sharing (WinError 5).
            with hold_without_delete_sharing(path, seconds=0.3):
                print_command._write_png(path, image)
            with open(path, "rb") as written, open(direct, "rb") as expected:
                self.assertEqual(written.read(), expected.read())
            self.assertEqual(sorted(os.listdir(directory)), ["direct.png", "picture.png"])

    def test_a_hold_that_never_ends_fails_the_picture_saying_so_and_leaves_nothing_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "picture.png")
            with open(path, "wb") as handle:
                handle.write(b"earlier picture")
            with transient_hold(10 ** 6), self.assertRaises(PermissionError) as raised:
                print_command._write_png(path, np.zeros((2, 2, 4), dtype=np.uint8))

            self.assertIn("another program kept the file in use", str(raised.exception))
            self.assertIn("being used by another process", str(raised.exception))
            self.assertEqual(os.listdir(directory), ["picture.png"])
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"earlier picture")

    def test_a_partial_file_that_stays_held_is_named_for_a_later_deletion(self):
        leftovers = []
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "picture.png")
            real_remove = os.remove

            def remove(name):
                if name.endswith(".partial.png"):
                    raise held_file_error(32, name)
                real_remove(name)

            with transient_hold(10 ** 6), \
                    mock.patch.object(Job_Cancellation.os, "remove", side_effect=remove), \
                    mock.patch.object(Job_Cancellation, "leave_partial", side_effect=leftovers.append), \
                    self.assertRaises(PermissionError):
                print_command._write_png(path, np.zeros((2, 2, 4), dtype=np.uint8))

            self.assertEqual(len(leftovers), 1)
            self.assertTrue(leftovers[0].endswith(".partial.png"))
            self.assertEqual(os.listdir(directory), [os.path.basename(leftovers[0])])


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
            print_command, "_svg_snapshot"
        ) as export_svg, mock.patch.object(
            print_command, "_write_svg"
        ), mock.patch.object(
            print_command, "_write_png"
        ) as save, mock.patch.object(
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
                    self.assertTrue(viewer.zoom_text.visible)
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
            print_command._write_svg(print_command._svg_snapshot(viewer), path)
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


class PrintSvgOpacityBandsTests(unittest.TestCase):
    """With Edge Opacity's Auto switch on, the SVG's edges go out in opacity bands.

    The screen draws each edge at an alpha that follows its score (15 % at the
    lowest score drawn, 85 % at the highest); the SVG groups the edges in
    SVG_OPACITY_BANDS evenly spaced opacities, most opaque first, each in a
    <g> with its stroke-opacity (and on every line too, so a reader that
    ignores the group's still has it). With Auto off, or in a layout that
    ignores it, the file is the one it always was.
    """

    SVG = "{http://www.w3.org/2000/svg}"
    SCORES = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5]

    @classmethod
    def network(cls, scores=None, threshold=0.3, by_score=True, **overrides):
        from desktop.Viewer_Edges import ViewerEdges
        scores = np.array(cls.SCORES if scores is None else scores)
        viewer = square_network(
            edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=np.int32)[:len(scores)],
            edge_scores=scores, current_slider_threshold=threshold, **overrides)
        layer = ViewerEdges(None, viewer.edges, scores, viewer.pos, (0, 0, 0, 0.2), 1.0,
                            use_position_texture=True)
        layer.set_opacity_by_score(by_score)
        viewer.edge_layer = layer
        return viewer

    @staticmethod
    def band_opacity(alpha, bands=128):
        """The opacity of the band an edge of this alpha belongs in, worked out here on its own."""
        step = round((alpha - 0.15) / 0.7 * (bands - 1))
        return 0.15 + 0.7 * step / (bands - 1)

    def groups(self, svg):
        import xml.etree.ElementTree as ElementTree
        root = ElementTree.fromstring(svg)
        edges = next(group for group in root.iter(self.SVG + "g") if group.get("id") == "edges")
        return edges, edges.findall(self.SVG + "g")

    @staticmethod
    def coordinates(line):
        return tuple(line.get(name) for name in ("x1", "y1", "x2", "y2"))

    def test_the_edges_are_in_bands_most_opaque_first_with_each_edges_own_opacity(self):
        viewer = self.network()
        svg = export_svg(viewer, AUTO_EDGE_ALPHA=True)
        plain = ElementTreeLines.parse(export_svg(viewer, AUTO_EDGE_ALPHA=False))
        edges, groups = self.groups(svg)
        # The scores kept by the 0.3 threshold, in loaded order: 0.4, 0.6, 0.8, 1.0, 0.5 (0.2 is below it).
        kept = [score for score in self.SCORES if score >= 0.3]
        alphas = [0.15 + 0.7 * (score - 0.4) / 0.6 for score in kept]
        wanted = [f"{self.band_opacity(alpha):.3f}" for alpha in alphas]
        self.assertEqual(len(plain), len(kept))
        by_edge = {}
        opacities = []
        for number, group in enumerate(groups, start=1):
            self.assertEqual(group.get("id"), f"edges-band-{number}")
            opacities.append(group.get("stroke-opacity"))
            lines = group.findall(self.SVG + "line")
            self.assertTrue(lines)
            for line in lines:
                # Every line states its own opacity, the group's.
                self.assertEqual(line.get("stroke-opacity"), group.get("stroke-opacity"))
                self.assertEqual(line.get("stroke"), "rgb(0,0,0)")
                by_edge[self.coordinates(line)] = group.get("stroke-opacity")
        self.assertEqual(len(by_edge), len(kept))
        for coordinates, expected in zip(plain, wanted):
            self.assertEqual(by_edge[coordinates], expected)
        # Most opaque first: the edges go out in descending score, as the screen draws them.
        self.assertEqual(opacities, sorted(set(wanted), reverse=True))
        self.assertEqual(opacities[0], "0.850")
        self.assertEqual(opacities[-1], "0.150")
        # The groups hold the edges and nothing else; the nodes layer is as it was.
        self.assertEqual(len(edges), len(groups))
        self.assertEqual(len(elements(svg, "circle")), 4)

    def test_the_file_stays_plain_svg(self):
        import xml.etree.ElementTree as ElementTree
        svg = export_svg(self.network(), AUTO_EDGE_ALPHA=True)
        root = ElementTree.fromstring(svg)  # well formed
        self.assertEqual(root.tag, self.SVG + "svg")
        _, groups = self.groups(svg)
        for group in groups:
            self.assertEqual(sorted(group.attrib), ["id", "name", "stroke-opacity"])
        # Only the elements and attributes the file always had, and the groups.
        tags = {element.tag.removeprefix(self.SVG) for element in root.iter()}
        self.assertEqual(tags, {"svg", "rect", "g", "line", "circle"})
        lines = elements(svg, "line")
        for line in lines:
            self.assertEqual(re.findall(r'(\w[\w-]*)="', line),
                             ["x1", "y1", "x2", "y2", "stroke", "stroke-opacity", "stroke-width"])

    def test_equal_scores_are_one_band_at_85_percent(self):
        for scores in ([0.7, 0.7, 0.7], [0.9, 0.2]):
            with self.subTest(scores=scores):
                threshold = 0.5 if len(scores) == 2 else 0.3
                svg = export_svg(self.network(scores, threshold=threshold), AUTO_EDGE_ALPHA=True)
                _, groups = self.groups(svg)
                self.assertEqual([group.get("stroke-opacity") for group in groups], ["0.850"])
                self.assertEqual(len(groups[0].findall(self.SVG + "line")), len(
                    [score for score in scores if score >= threshold]))

    def test_the_threshold_decides_which_edge_is_the_faintest(self):
        low = self.groups(export_svg(self.network(threshold=0.3), AUTO_EDGE_ALPHA=True))[1]
        high = self.groups(export_svg(self.network(threshold=0.7), AUTO_EDGE_ALPHA=True))[1]
        for groups in (low, high):
            self.assertEqual(groups[0].get("stroke-opacity"), "0.850")
            self.assertEqual(groups[-1].get("stroke-opacity"), "0.150")
        self.assertEqual(sum(len(group) for group in low), 5)
        self.assertEqual(sum(len(group) for group in high), 2)

    def test_the_file_is_the_same_whatever_the_chunk_size(self):
        viewer = self.network()
        whole = export_svg(viewer, AUTO_EDGE_ALPHA=True)
        for chunk in (1, 2, 4):
            with self.subTest(chunk=chunk), mock.patch.object(print_command, "SVG_CHUNK_SIZE", chunk):
                self.assertEqual(export_svg(viewer, AUTO_EDGE_ALPHA=True), whole)

    def test_auto_off_is_the_file_it_always_was(self):
        # Byte for byte what the code before the switch wrote for this network.
        expected_edges = (
            '  <g id="edges" name="Edges">\n'
            '    <line x1="26.250" y1="36.250" x2="36.250" y2="36.250" stroke="rgb(0,0,0)" '
            'stroke-opacity="0.200" stroke-width="1.000" />\n'
            '    <line x1="36.250" y1="26.250" x2="26.250" y2="26.250" stroke="rgb(0,0,0)" '
            'stroke-opacity="0.200" stroke-width="1.000" />\n'
            '  </g>\n')
        bare = export_svg(square_network())
        self.assertIn(expected_edges, bare)
        for configuration in ({}, {"AUTO_EDGE_ALPHA": False}):
            with self.subTest(configuration=configuration):
                viewer = self.network(scores=[0.90, 0.20, 0.95, 0.10], threshold=0.5)
                self.assertEqual(export_svg(viewer, **configuration), bare)
        self.assertNotIn("edges-band", bare)

    def test_a_layout_that_ignores_auto_writes_the_plain_file(self):
        # UMAP and PaCMAP draw every edge at the slider's alpha.
        viewer = self.network(threshold=0.0, selected_indices=[0])
        plain = export_svg(viewer, AUTO_EDGE_ALPHA=False, UMAP_MODE=True)
        self.assertEqual(export_svg(viewer, AUTO_EDGE_ALPHA=True, UMAP_MODE=True), plain)
        self.assertNotIn("edges-band", plain)
        # So does an edge layer that does not draw by score (the coordinate fallback, no scores).
        viewer = self.network(by_score=False)
        self.assertEqual(export_svg(viewer, AUTO_EDGE_ALPHA=True), export_svg(viewer, AUTO_EDGE_ALPHA=False))
        viewer = square_network()
        self.assertEqual(export_svg(viewer, AUTO_EDGE_ALPHA=True), export_svg(viewer, AUTO_EDGE_ALPHA=False))

    def test_the_scores_line_up_with_the_edges_the_screen_keeps(self):
        rng = np.random.default_rng(8)
        visible = rng.random(20) > 0.2
        viewer = SimpleNamespace(
            edges=rng.integers(0, 20, (200, 2)).astype(np.int32), edge_scores=rng.random(200).astype(np.float32),
            visible_mask=visible, n_nodes=20, current_slider_threshold=0.4, selected_indices=[])
        configuration = SimpleNamespace(UMAP_MODE=False)
        active, _ = print_command.edge_stages(viewer, configuration)
        scores = print_command.active_edge_scores(viewer, configuration)
        keep = (viewer.edge_scores >= 0.4) & visible[viewer.edges[:, 0]] & visible[viewer.edges[:, 1]]
        np.testing.assert_array_equal(active, viewer.edges[keep])
        np.testing.assert_array_equal(scores, viewer.edge_scores[keep])

    def test_at_least_32_bands_and_each_edge_is_within_half_a_step_of_its_alpha(self):
        self.assertEqual(print_command.SVG_OPACITY_BANDS, 128)
        self.assertGreaterEqual(print_command.SVG_OPACITY_BANDS, 32)
        rng = np.random.default_rng(3)
        alphas = np.concatenate(([0.15, 0.85], rng.uniform(0.15, 0.85, 5000))).astype(np.float32)
        order, bands = print_command._svg_edge_bands(alphas)
        self.assertEqual(sorted(order.tolist()), list(range(len(alphas))))
        self.assertEqual(bands[0][1], 0)
        self.assertEqual(bands[-1][2], len(alphas))
        opacities = [band[0] for band in bands]
        self.assertEqual(opacities, sorted(opacities, reverse=True))
        step = 0.7 / 127
        worst = 0.0
        for opacity, start, stop in bands:
            members = order[start:stop]
            self.assertTrue((np.diff(members) > 0).all())  # a band keeps the edges' own order
            worst = max(worst, float(np.abs(alphas[members] - opacity).max()))
        self.assertLessEqual(worst, step / 2 + 1e-6)
        # Half a step is under one 8-bit level for a contrast of 255: 0.0028 * 255 < 1.
        self.assertLess(step / 2 * 255, 1.0)
        self.assertEqual(len(bands), len({round(opacity, 9) for opacity in opacities}))
        # No edges, no bands.
        empty_order, empty = print_command._svg_edge_bands(np.empty(0, dtype=np.float32))
        self.assertEqual((len(empty_order), empty), (0, []))


class ElementTreeLines:
    """The line elements of an SVG as their coordinates, in file order."""

    @staticmethod
    def parse(svg):
        import xml.etree.ElementTree as ElementTree
        namespace = "{http://www.w3.org/2000/svg}"
        return [tuple(line.get(name) for name in ("x1", "y1", "x2", "y2"))
                for line in ElementTree.fromstring(svg).iter(namespace + "line")]


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
                print_command._write_svg(print_command._svg_snapshot(viewer), path)

            self.assertEqual(os.listdir(directory), ["network.svg"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "earlier picture")

    def test_a_virus_scan_holding_the_finished_file_for_a_moment_does_not_fail_the_picture(self):
        viewer = self.mixed_network()
        whole = export_svg(viewer)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.svg")
            with mock.patch.multiple(print_command.cfg, create=True, UMAP_MODE=False), \
                    transient_hold(2, winerror=5) as (attempts, pauses), \
                    redirect_stdout(io.StringIO()):
                print_command._write_svg(print_command._svg_snapshot(viewer), path)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(os.listdir(directory), ["network.svg"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), whole)

    def test_a_hold_that_never_ends_fails_the_picture_saying_so_and_leaves_nothing_behind(self):
        viewer = self.mixed_network()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.svg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("earlier picture")
            with mock.patch.multiple(print_command.cfg, create=True, UMAP_MODE=False), \
                    transient_hold(10 ** 6), redirect_stdout(io.StringIO()), \
                    self.assertRaises(PermissionError) as raised:
                print_command._write_svg(print_command._svg_snapshot(viewer), path)

            self.assertIn("another program kept the file in use", str(raised.exception))
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
            print_command.app, "process_events"
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, ["network", "svg"])
            written = os.listdir(save_dir)
        return viewer, engine, viewer.background_job_scheduler, written, save_dir

    def test_svg_with_every_node_hidden_fails_and_writes_nothing(self):
        viewer, engine, scheduler, written, _ = self.run_svg_print([False] * 4)

        self.assertEqual(written, [])
        engine.command_failed.assert_called_once()
        self.assertEqual(str(engine.command_failed.call_args.args[1]), "Error: No visible nodes to export.")
        engine.command_succeeded.assert_not_called()
        self.assertEqual(scheduler.jobs, [])
        self.assertEqual(viewer.console_text.text, "Error: No visible nodes to export.")
        self.assertTrue(viewer.zoom_text.visible)

    def test_svg_with_visible_nodes_reports_the_written_file(self):
        viewer, engine, scheduler, written, save_dir = self.run_svg_print([True] * 4)

        self.assertEqual(written, ["network.svg"])
        engine.command_failed.assert_not_called()
        engine.command_succeeded.assert_called_once()
        # The job is print's, for the file, and its report opens the save folder.
        [job] = scheduler.jobs
        self.assertEqual(job.command_name, "print")
        self.assertEqual(job.output_path, os.path.join(save_dir, "network.svg"))
        self.assertEqual(scheduler.reports[0]["reveal_directory"], save_dir)
        # The format is named without the extension's dot ("Saved .SVG" before).
        self.assertEqual(scheduler.report, "Saved SVG: network.svg")


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
            EMAPSSN_Config,
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

    def test_bars_crosses_and_rings_are_filled_and_outlined_like_every_shape(self):
        # VisPy fills them in the node's colour and outlines them; the SVG drew
        # them as bare strokes in the node's colour.
        for alias, tag in (("+", "polygon"), ("x", "polygon"), ("|", "polygon"), ("-", "polygon"),
                           ("ring", "path")):
            with self.subTest(alias=alias):
                nodes = self.with_shape(alias).split('<g id="nodes"')[1]
                drawn = elements(nodes, tag)
                self.assertEqual(len(drawn), 4)
                for element in drawn:
                    self.assertIn('fill="rgb(63,127,255)"', element)
                    self.assertIn('stroke="rgb(0,0,0)"', element)

    def test_a_ring_is_the_band_from_half_the_radius_to_the_radius(self):
        nodes = self.with_shape("ring").split('<g id="nodes"')[1]
        ring = elements(nodes, "path")[0]
        self.assertIn('fill-rule="evenodd"', ring)
        # 10-pixel nodes at one unit per pixel: circles of radius 5 and 2.5.
        radii = {float(value) for value in re.findall(r"A ([\d.]+) ", ring)}
        self.assertEqual(radii, {5.0, 2.5})

    def test_cross_lines_is_two_lines_in_the_outline_colour_alone(self):
        # VisPy's cross_lines has no area: only its outline shows.
        nodes = self.with_shape("P").split('<g id="nodes"')[1]
        drawn = elements(nodes, "path")
        self.assertEqual(len(drawn), 4)
        for element in drawn:
            self.assertIn('fill="none"', element)
            self.assertIn('stroke="rgb(0,0,0)"', element)
            self.assertIn('stroke-linecap="round"', element)

    def test_shapes_have_vispys_proportions(self):
        polygons = dict(print_command._SVG_POLYGONS)
        # An equilateral triangle with sides of the node's size, centred on its box.
        xs, ys = zip(*polygons["triangle_up"])
        self.assertEqual((min(xs), max(xs)), (-1.0, 1.0))
        self.assertAlmostEqual(min(ys), -np.sqrt(3) / 2)
        self.assertAlmostEqual(max(ys), np.sqrt(3) / 2)
        # A regular star whose top and bottom points are as far from the centre.
        star = np.array(polygons["star"])
        self.assertAlmostEqual(-star[:, 1].min(), star[:, 1].max(), delta=0.01)
        inner = np.hypot(star[1::2, 0], star[1::2, 1] - 0.1)
        np.testing.assert_allclose(inner, np.sin(np.pi / 10) / np.sin(3 * np.pi / 10))
        # Bars two thirds of the radius wide.
        self.assertEqual(sorted({x for x, _ in polygons["vbar"]}), [-1 / 3, 1 / 3])
        # Clobber's outline turns where its circles cross, as far out as each other.
        radius, crossings = print_command._unit_clobber()
        self.assertEqual(radius, 0.64)
        distances = [np.hypot(x, y - 0.25 * 0.72 / np.sqrt(3)) for x, y in crossings]
        np.testing.assert_allclose(distances, distances[0])

    def test_an_unknown_shape_is_still_drawn_as_a_circle(self):
        self.assertEqual(self.with_shape("blob"), self.with_shape("disc"))
        self.assertIsNone(print_command._canonical_shape(None))


class PrintSvgFrameTests(unittest.TestCase):
    """The view box frames the network as print's PNG does: what is drawn, and
    a border of 20 pixels of the PNG, which are device pixels.

    square_network's four nodes lie on a 10-unit square at one unit per
    logical pixel. A node of radius r with a 0.5-pixel outline is drawn to
    r + 0.25 + 1 from its centre: half its outline, then VisPy's 1-pixel
    antialias band.
    """

    @staticmethod
    def view_box(svg):
        match = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        return float(match.group(1)), float(match.group(2))

    def test_the_view_box_is_the_drawing_and_a_twenty_pixel_border(self):
        # 10 between the centres, 5 + 1.25 drawn past them, 20 of border.
        svg = export_svg(square_network())
        self.assertEqual(self.view_box(svg), (62.5, 62.5))
        # The first node's centre is 6.25 + 20 in from the left, and as far
        # down from the top (y points down in the SVG).
        circle = elements(svg, "circle")[0]
        self.assertEqual((number(circle, "cx"), number(circle, "cy")), (26.25, 36.25))

    def test_the_border_is_twenty_pixels_of_the_png_at_the_display_scale(self):
        # At twice the display scale, 20 device pixels are 10 logical ones,
        # and the outline (1 device pixel) and its band reach 0.5 + 0.5.
        sharp = square_network()
        sharp.canvas.pixel_scale = 2.0
        self.assertEqual(self.view_box(export_svg(sharp)), (41.5, 41.5))

    def test_a_ring_reaches_as_far_as_a_disc(self):
        rings = square_network(current_shapes=np.array(["ring"] * 4, dtype=object))
        self.assertEqual(self.view_box(export_svg(rings)), self.view_box(export_svg(square_network())))

    def test_each_shape_is_framed_by_its_own_reach(self):
        # A bar of 30-pixel nodes reaches 15 across but only 5 up and down;
        # its square corners move the outline straight out.
        bars = square_network(
            current_sizes=np.full(4, 30.0), current_shapes=np.array(["hbar"] * 4, dtype=object),
        )
        self.assertEqual(self.view_box(export_svg(bars)), (82.5, 62.5))

    def test_a_sharp_corner_reaches_a_mitre_further(self):
        # A diamond's corners are right angles: the outline and its band go
        # sqrt(2) times as far along their bisector as a side moves.
        diamonds = square_network(current_shapes=np.array(["diamond"] * 4, dtype=object))
        reach = 5.0 + 1.25 * np.sqrt(2)
        width, height = self.view_box(export_svg(diamonds))
        self.assertAlmostEqual(width, 10 + 2 * reach + 40, places=3)
        self.assertAlmostEqual(height, width)

    def test_nodes_without_size_are_framed_by_their_edges(self):
        # GL widens an edge across its main direction only: the two
        # horizontal edges shown (scores 0.9 and 0.95) by 1.5 up and down.
        bare = square_network(current_sizes=np.zeros(4))
        self.assertEqual(self.view_box(export_svg(bare, EDGE_WIDTH=3.0)), (50.0, 53.0))
        # The edges are measured a chunk at a time, every chunk counting.
        with mock.patch.object(print_command, "SVG_CHUNK_SIZE", 1):
            self.assertEqual(self.view_box(export_svg(bare, EDGE_WIDTH=3.0)), (50.0, 53.0))
        bare.current_slider_threshold = 0.0
        with mock.patch.object(print_command, "SVG_CHUNK_SIZE", 1):
            self.assertEqual(self.view_box(export_svg(bare, EDGE_WIDTH=3.0)), (53.0, 53.0))

    def test_a_wide_network_gets_the_same_border_on_every_side(self):
        viewer = square_network(
            pos=np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 200.0], [0.0, 200.0]])
        )

        self.assertEqual(self.view_box(export_svg(viewer)), (452.5, 252.5))

    def test_large_nodes_at_the_edge_fit_inside_the_view_box(self):
        svg = export_svg(square_network(current_sizes=np.full(4, 40.0)))

        width, height = self.view_box(svg)
        circles = elements(svg, "circle")
        self.assertEqual(len(circles), 4)
        for circle in circles:
            cx, cy, r = (
                float(re.search(rf'{name}="([\d.]+)"', circle).group(1))
                for name in ("cx", "cy", "r")
            )
            self.assertGreaterEqual(cx - r, 0.0)
            self.assertLessEqual(cx + r, width)
            self.assertGreaterEqual(cy - r, 0.0)
            self.assertLessEqual(cy + r, height)

    def test_the_frame_follows_the_zoom_in_scene_units(self):
        # At half a scene unit per pixel, 40-pixel nodes reach 20 + 1.25
        # pixels past their centres, 10.625 units, and the border is 10 units.
        svg = export_svg(square_network(view=view_at(0.5), current_sizes=np.full(4, 40.0)))

        self.assertEqual(self.view_box(svg), (51.25, 51.25))


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

        drawn_squares = elements(squares, "rect")[1:]  # the first is the background
        self.assertEqual(len(drawn_squares), 4)
        for square in drawn_squares:
            self.assertAlmostEqual(number(square, "width"), 12 * self.UNITS_PER_PIXEL)
        # A cross reaches the radius, with bars a third of it each side of its centre.
        drawn_crosses = elements(crosses, "polygon")
        self.assertEqual(len(drawn_crosses), 4)
        for cross in drawn_crosses:
            points = re.search(r'points="([^"]+)"', cross).group(1).split()
            xs = [float(point.split(",")[0]) for point in points]
            offsets = sorted({round(x - (max(xs) + min(xs)) / 2, 3) for x in xs})
            self.assertEqual(offsets, [-1.2, -0.4, 0.4, 1.2])

    def test_edge_and_node_outline_widths_are_converted_too(self):
        svg = export_svg(self.network(), EDGE_WIDTH=1.5)

        lines = elements(svg, "line")
        self.assertEqual(len(lines), 2)
        # GL draws a 1.5-pixel edge 2 pixels wide.
        for line in lines:
            self.assertAlmostEqual(number(line, "stroke-width"), 2 * self.UNITS_PER_PIXEL)
        # The 0.5-pixel outline of every filled node, as wide as VisPy draws it.
        circles = elements(svg, "circle")
        self.assertEqual(len(circles), 4)
        for circle in circles:
            self.assertAlmostEqual(
                number(circle, "stroke-width"), 0.9818 * self.UNITS_PER_PIXEL, delta=0.001,
            )

    def test_node_outline_follows_the_boundary_width_setting(self):
        # Outlines of 2 pixels and more look 0.23 pixels narrower than set.
        for setting in (2.0, 4.0):
            with self.subTest(setting=setting):
                svg = export_svg(self.network(), NODE_BOUNDARY_WIDTH=setting)
                circles = elements(svg, "circle")
                self.assertEqual(len(circles), 4)
                for circle in circles:
                    self.assertAlmostEqual(
                        number(circle, "stroke-width"), (setting - 0.2317) * self.UNITS_PER_PIXEL,
                        delta=0.001,
                    )

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
        # The network itself keeps its scene coordinates: only the frame,
        # which follows what is drawn, moves it as a whole.
        shift = number(zoomed_circle, "cx") - number(circle, "cx")
        self.assertAlmostEqual(number(zoomed_circle, "cy") - number(circle, "cy"), shift, places=3)
        for name in ("x1", "y1", "x2", "y2"):
            self.assertAlmostEqual(number(zoomed_line, name) - number(line, name), shift, places=3)

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

    def test_a_high_dpi_display_draws_what_its_png_shows(self):
        # Sizes are logical pixels, which VisPy scales to the display itself,
        # and the view is measured in them, so node sizes stay. Outlines and
        # edges are drawn in device pixels: a 0.5-pixel outline is 1 device
        # pixel there, which looks 1.74 device pixels wide, and a 1.3-pixel
        # edge is 2.6, which GL draws 3 device pixels wide.
        def drawn(pixel_scale):
            viewer = self.network()
            viewer.canvas.pixel_scale = pixel_scale
            svg = export_svg(viewer, EDGE_WIDTH=1.3)
            return elements(svg, "circle")[0], elements(svg, "line")[0]

        circle, line = drawn(1.0)
        sharp_circle, sharp_line = drawn(2.0)

        self.assertEqual(number(sharp_circle, "r"), number(circle, "r"))
        self.assertAlmostEqual(number(circle, "stroke-width"), 0.9818 * 0.2, delta=0.001)
        self.assertAlmostEqual(number(sharp_circle, "stroke-width"), 1.7423 / 2 * 0.2, delta=0.001)
        self.assertAlmostEqual(number(line, "stroke-width"), 1.0 * 0.2)
        self.assertAlmostEqual(number(sharp_line, "stroke-width"), 1.5 * 0.2)

    def test_a_view_without_area_leaves_sizes_as_they_are(self):
        svg = export_svg(self.network(view=view_at(self.UNITS_PER_PIXEL, size=(0, 0))))

        circles = elements(svg, "circle")
        self.assertEqual(len(circles), 4)
        for circle in circles:
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


class PrintSvgWidthModelTests(unittest.TestCase):
    """Outline and edge widths as the screen draws them, in device pixels."""

    def test_an_outline_is_as_wide_as_the_colour_vispy_spreads_across_it(self):
        step = 0.0005
        distances = np.arange(-12.0, 12.0, step) + step / 2
        for antialias in (1.0, 0.5, 2.0):
            for width in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0):
                for filled in (True, False):
                    with self.subTest(antialias=antialias, width=width, filled=filled):
                        if filled:
                            spread = sum(vispy_outline_coverage(r, width, antialias) for r in distances) * step
                        else:
                            # cross_lines: a line with background on both sides.
                            spread = 2 * step * sum(
                                vispy_outline_coverage(r, width, antialias, filled=False)
                                for r in distances[distances > 0]
                            )
                        self.assertAlmostEqual(
                            print_command._outline_ink_px(width, antialias, open_lines=not filled),
                            spread, delta=2e-3,
                        )

    def test_a_thin_outline_looks_about_one_and_three_quarter_pixels_wide(self):
        self.assertAlmostEqual(print_command._outline_ink_px(0.5), 0.9818, places=4)
        self.assertAlmostEqual(print_command._outline_ink_px(1.0), 1.7424, places=4)
        self.assertAlmostEqual(print_command._outline_ink_px(5.0), 5.0 - 0.2317, places=4)

    def test_no_outline_is_drawn_without_width_and_a_hard_one_without_antialiasing(self):
        self.assertEqual(print_command._outline_ink_px(0.0), 0.0)
        self.assertEqual(print_command._outline_ink_px(float("nan")), 0.0)
        # Without the fading band, an outline under a pixel is that much fainter.
        self.assertEqual(print_command._outline_ink_px(0.5, antialias=0.0), 0.25)
        self.assertEqual(print_command._outline_ink_px(3.0, antialias=0.0), 3.0)

    def test_gl_draws_edges_a_whole_number_of_device_pixels_wide(self):
        for width, scale, drawn in (
            (0.5, 1.0, 1.0), (1.3, 1.0, 1.0), (1.5, 1.0, 2.0), (2.5, 1.0, 2.0),
            (0.25, 2.0, 1.0), (1.3, 2.0, 3.0), (1.0, 1.5, 2.0), (3.0, 1.5, 4.0),
            (0.0, 2.0, 0.0),
        ):
            with self.subTest(width=width, scale=scale):
                self.assertEqual(print_command._gl_line_px(width, scale), drawn)

    def test_the_display_scale_and_band_come_from_the_viewer(self):
        viewer = SimpleNamespace(
            canvas=SimpleNamespace(pixel_scale=1.5), markers=SimpleNamespace(antialias=0.5),
        )
        self.assertEqual(print_command._display_scale(viewer), 1.5)
        self.assertEqual(print_command._marker_antialias(viewer), 0.5)
        # A viewer that cannot say draws as VisPy does by default.
        for unknown in (SimpleNamespace(), SimpleNamespace(canvas=SimpleNamespace(pixel_scale=0))):
            self.assertEqual(print_command._display_scale(unknown), 1.0)
            self.assertEqual(print_command._marker_antialias(unknown), 1.0)


class PrintSvgOutlineTests(unittest.TestCase):
    """Each node's outline is the one the screen draws it with."""

    @staticmethod
    def outlined_network(**overrides):
        colors = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
        colors[2] = (1.0, 0.0, 0.0, 1.0)
        widths = np.array([0.5, 0.5, 2.0, 0.5])
        return square_network(
            current_colors=numbered_colors(),
            node_outlines=lambda: (colors, widths),
            **overrides,
        )

    def test_a_node_drawn_with_its_own_outline_keeps_it(self):
        circles = elements(export_svg(self.outlined_network()), "circle")

        self.assertEqual(len(circles), 4)
        for index, circle in enumerate(circles):
            with self.subTest(node=index):
                if index == 2:
                    self.assertIn('stroke="rgb(255,0,0)"', circle)
                    self.assertAlmostEqual(number(circle, "stroke-width"), 1.768, delta=0.001)
                else:
                    self.assertIn('stroke="rgb(0,0,0)"', circle)
                    self.assertAlmostEqual(number(circle, "stroke-width"), 0.982, delta=0.001)

    def test_outlines_follow_the_drawing_order(self):
        viewer = self.outlined_network(
            visible_node_render_order=lambda: np.array([2, 0, 3, 1], dtype=np.int32),
        )
        svg = export_svg(viewer)

        self.assertEqual(node_fills(svg), [25, 5, 35, 15])
        self.assertIn('stroke="rgb(255,0,0)"', elements(svg, "circle")[0])

    def test_selected_nodes_and_their_neighbours_keep_their_borders(self):
        import EMAPSSN_Config
        from EMAPSSN_Viewer import MainViewer

        viewer = MainViewer.__new__(MainViewer)
        for name, value in vars(square_network(current_colors=numbered_colors())).items():
            setattr(viewer, name, value)
        viewer.n_nodes = 4
        viewer.node_render_order = np.arange(4, dtype=np.int32)
        viewer.selected_node_idx = None
        viewer.left_click_highlight_indices = None
        viewer.selected_indices = [0]
        with mock.patch.multiple(
            EMAPSSN_Config,
            CONNECTED_NODE_COLOR="red", NODE_BOUNDARY_COLOR="black", HOVER_COLOR="blue",
        ):
            svg = export_svg(viewer)

        # Node 0 is selected and drawn last, after its neighbours 1 and 3.
        self.assertEqual(node_fills(svg), [25, 15, 35, 5])
        strokes = [re.search(r'stroke="(rgb\([\d,]+\))"', circle).group(1) for circle in elements(svg, "circle")]
        self.assertEqual(strokes, ["rgb(0,0,0)", "rgb(255,0,0)", "rgb(255,0,0)", "rgb(0,0,255)"])
        widths = [number(circle, "stroke-width") for circle in elements(svg, "circle")]
        self.assertEqual(widths, [0.982, 1.768, 1.768, 1.768])


# Draws scenes with the Viewer's own visuals on a real canvas, prints each as a
# PNG and as an SVG with print itself, draws the SVG with QtSvg at the PNG's
# pixels per scene unit, and reports how the two compare.
_PNG_SVG_SCRIPT = textwrap.dedent(
    r"""
    import importlib.util, json, os, shutil, sys, tempfile, types
    from types import SimpleNamespace
    from unittest import mock
    import numpy as np
    SRC = sys.argv[1]
    sys.path.insert(0, SRC)
    from vispy import app, scene
    app.use_app("pyside6")
    import matplotlib.image as mpimg
    from PySide6 import QtCore, QtGui, QtSvg
    from desktop.Viewer_Edges import ViewerEdges
    import desktop.Viewer_Zoom, utilities.Output_Names, Viewer_Visual_State

    SAVE_DIR = tempfile.mkdtemp(prefix="print_png_svg_")
    engine = types.ModuleType("Command_Engine")
    for name in ("command_artifact", "command_succeeded", "command_failed", "show_status"):
        setattr(engine, name, mock.Mock())
    config = types.ModuleType("EMAPSSN_Config")
    config.SEQUENCE_SET = "test"
    config.resolve_directory_path = lambda value: SAVE_DIR
    spec = importlib.util.spec_from_file_location("print_under_test", os.path.join(SRC, "commands", "print.py"))
    print_command = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"Command_Engine": engine, "EMAPSSN_Config": config}):
        spec.loader.exec_module(print_command)

    canvas = scene.SceneCanvas(size=(400, 300), bgcolor="white", show=False)
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(0, 200), y=(0, 150), margin=0)
    markers = scene.visuals.Markers(parent=view.scene)
    markers.set_gl_state("translucent", depth_test=False)
    if (canvas.render()[..., 0] < 250).sum() != 0:
        sys.exit(3)
    print("CONTEXT OK", flush=True)
    DPR = float(canvas.pixel_scale)
    # Device pixels per scene unit, from the view rather than from print.
    PPU = canvas.physical_size[0] / view.camera.rect.width

    class Scheduler:
        def is_output_path_reserved(self, path):
            return False

        def enqueue(self, command_name, description, payload, worker, output_path, allow_overwrite=False):
            worker(payload)

    layer = None

    def show(pos, sizes, shapes, outline_colors, outline_widths, edges=(), edge_width=1.0,
             color=(1.0, 1.0, 1.0, 1.0), scores=None, auto=False):
        global layer
        pos = np.asarray(pos, dtype=np.float32)
        count = len(pos)
        colors = np.tile(np.asarray(color, dtype=np.float32), (count, 1))
        outline_colors = np.asarray(outline_colors, dtype=np.float32)
        outline_widths = np.asarray(outline_widths, dtype=np.float32)
        markers.set_data(pos=pos, face_color=colors, edge_color=outline_colors,
                         size=np.asarray(sizes, np.float32), edge_width=outline_widths, symbol=list(shapes))
        if layer is not None:
            layer.visual.parent = None
            layer = None
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        if len(edges):
            layer = ViewerEdges(view.scene, edges, scores, pos, (0.0, 0.0, 0.0, 1.0), edge_width)
            layer.set_opacity_by_score(auto)
            layer.visible = layer.show_threshold(pos, None, 0.0) > 0
            markers.order = 1
        config.EDGE_COLOR, config.EDGE_ALPHA, config.EDGE_WIDTH = "#000000", 1.0, edge_width
        config.UMAP_MODE = False
        config.AUTO_EDGE_ALPHA = auto
        return SimpleNamespace(
            canvas=canvas, view=view, markers=markers, n_nodes=count,
            visible_mask=np.ones(count, dtype=bool), pos=pos, current_colors=colors,
            current_sizes=np.asarray(sizes, dtype=np.float32), current_shapes=np.array(shapes, dtype=object),
            edges=edges, edge_scores=np.ones(len(edges)) if scores is None else np.asarray(scores),
            current_slider_threshold=0.0, selected_indices=[], edge_layer=layer,
            node_outlines=lambda: (outline_colors, outline_widths), background_job_scheduler=Scheduler(),
        )

    def raster(path):
        renderer = QtSvg.QSvgRenderer(path)
        box = renderer.viewBoxF()
        width, height = int(round(box.width() * PPU)), int(round(box.height() * PPU))
        image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGBA8888)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QtCore.QRectF(0, 0, width, height))
        painter.end()
        pixels = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
        return pixels[:, :width, :3].astype(np.float64) / 255.0

    def prints(viewer, name):
        # The PNG and the SVG print makes, the SVG drawn at the PNG's scale: coverage 0..1.
        print_command.run(viewer, [name])
        print_command.run(viewer, [name + "_svg", "svg"])
        png = mpimg.imread(os.path.join(SAVE_DIR, name + ".png"))[..., :3].astype(np.float64)
        svg = raster(os.path.join(SAVE_DIR, name + "_svg.svg"))
        return 1.0 - png.mean(axis=2), 1.0 - svg.mean(axis=2)

    def margins(cover):
        mask = cover > print_command.PNG_BACKGROUND_TOLERANCE
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        return [int(cols[0]), int(rows[0]), cover.shape[1] - 1 - int(cols[-1]), cover.shape[0] - 1 - int(rows[-1])]

    def frame(png, svg):
        return {"png_shape": list(png.shape), "svg_shape": list(svg.shape),
                "png_margins": margins(png), "svg_margins": margins(svg)}

    results = {"dpr": DPR}

    # White discs with outlines of several widths, one a selection's (red,
    # 2 pixels): all the colour on the page is outline.
    widths = [0.5, 0.25, 1.0, 2.0, 0.5]
    outline_colors = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))
    outline_colors[3] = (1.0, 0.0, 0.0, 1.0)
    pos = [[20.0 + 40.0 * k, 75.0 + (8.0 if k % 2 else 0.0)] for k in range(5)]
    png, svg = prints(show(pos, [24.0] * 5, ["disc"] * 5, outline_colors, widths), "outlines")
    inks = []
    for cover in (png, svg):
        left, top = margins(cover)[:2]
        half, per_node = int(18.0 * DPR), []
        for x, y in pos:
            cx = left + (x - pos[0][0]) * PPU + 12.0 * DPR
            cy = top + (max(p[1] for p in pos) - y) * PPU + 12.0 * DPR
            window = cover[int(cy) - half:int(cy) + half + 1, int(cx) - half:int(cx) + half + 1]
            per_node.append(float(window.sum() / (2.0 * np.pi * 12.0 * DPR) / DPR))
        inks.append(per_node)
    results["outlines"] = dict(frame(png, svg), png_ink=inks[0], svg_ink=inks[1])

    # Long straight edges at odd places between nodes of no size.
    pos = [[10.0, 20.0], [190.0, 20.0], [10.3, 40.0], [190.3, 40.0],
           [30.0, 60.0], [30.0, 140.0], [60.2, 60.0], [60.2, 140.0]]
    png, svg = prints(show(pos, [0.0] * 8, ["disc"] * 8, np.tile([0, 0, 0, 1.0], (8, 1)), [0.5] * 8,
                           [[0, 1], [2, 3], [4, 5], [6, 7]], 1.3), "edges")
    length = (2 * 180.0 + 2 * 80.0) * PPU * DPR
    results["edges"] = dict(frame(png, svg), png_ink=float(png.sum() / length), svg_ink=float(svg.sum() / length))

    # Edges whose alpha follows their score (Auto): the PNG has them drawn by the shader, the SVG
    # writes them in opacity bands; each edge is as dark on both. Scores 0.1 to 1: 15 % to 85 %.
    pos = [[10.0, 20.0], [190.0, 20.0], [10.0, 50.0], [190.0, 50.0],
           [10.0, 80.0], [190.0, 80.0], [10.0, 110.0], [190.0, 110.0]]
    scores = [0.1, 0.4, 0.7, 1.0]
    png, svg = prints(show(pos, [0.0] * 8, ["disc"] * 8, np.tile([0, 0, 0, 1.0], (8, 1)), [0.5] * 8,
                           [[0, 1], [2, 3], [4, 5], [6, 7]], 1.3, scores=scores, auto=True), "auto_edges")

    def edge_inks(cover):
        # The ink of each edge: the coverage of the rows it lies in, one run of rows to an edge.
        rows = np.flatnonzero(cover.max(axis=1) > 0.02)
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        return [float(cover[run[0]:run[-1] + 1].sum()) for run in runs]

    results["auto_edges"] = dict(frame(png, svg), png_inks=edge_inks(png), svg_inks=edge_inks(svg))

    # One node of each marker, filled and outlined.
    shapes = {}
    for shape in ("disc", "square", "triangle_up", "triangle_down", "diamond", "star", "cross", "x",
                  "vbar", "hbar", "arrow", "tailed_arrow", "clobber", "cross_lines", "ring"):
        png, svg = prints(show([[100.0, 75.0]], [60.0], [shape], [[0.0, 0.0, 0.0, 1.0]], [1.0],
                               color=(0.2, 0.4, 0.9, 1.0)), "shape_" + shape)
        height, width = min(png.shape[0], svg.shape[0]), min(png.shape[1], svg.shape[1])
        a, b = png[:height, :width] > 0.25, svg[:height, :width] > 0.25
        # Overlap at the best shift of up to 3 pixels: the frames may differ by that.
        overlap = max(
            (a & np.roll(b, (dy, dx), axis=(0, 1))).sum() / max((a | np.roll(b, (dy, dx), axis=(0, 1))).sum(), 1)
            for dy in range(-3, 4) for dx in range(-3, 4)
        )
        shapes[shape] = dict(png_shape=list(png.shape), svg_shape=list(svg.shape),
                             overlap=float(overlap), ink_ratio=float(svg.sum() / max(png.sum(), 1e-9)))
    results["shapes"] = shapes
    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    print("RESULTS " + json.dumps(results), flush=True)
    """
)


class PrintSvgMatchesPngTests(unittest.TestCase):
    """An SVG of the view looks like the PNG of it, on a 100 % and a 200 % screen.

    A real OpenGL canvas draws each scene as the Viewer does; print makes a
    PNG and an SVG of it; the SVG is drawn at the PNG's pixels per scene
    unit. Qt's high-DPI scaling is switched off and QT_SCALE_FACTOR sets the
    display scale, so the test runs at both on any screen.
    """

    @classmethod
    def setUpClass(cls):
        cls.results = {}

    def compare(self, scale):
        if scale in self.results:
            return self.results[scale]
        environment = dict(os.environ)
        # tests/__init__.py defaults Qt to offscreen, which has no OpenGL on
        # Windows; a hidden canvas on the native platform renders.
        environment.pop("QT_QPA_PLATFORM", None)
        environment.update(
            PYTHONDONTWRITEBYTECODE="1", QT_ENABLE_HIGHDPI_SCALING="0", QT_SCALE_FACTOR=str(scale),
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", _PNG_SVG_SCRIPT, SRC_DIR],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=environment, timeout=300,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("the GL render probe timed out")
        if "CONTEXT OK" not in result.stdout:
            self.skipTest(f"no OpenGL context to render with (exit {result.returncode})")
        lines = [line for line in result.stdout.splitlines() if line.startswith("RESULTS ")]
        self.assertTrue(lines, result.stdout[-2000:] + result.stderr[-3000:])
        results = json.loads(lines[-1][len("RESULTS "):])
        if results["dpr"] != scale:
            self.skipTest(f"Qt drew at a display scale of {results['dpr']}, not {scale}")
        self.results[scale] = results
        return results

    def assert_frames_match(self, scene):
        # The PNG's trim is whole pixels and its sharp points' faintest tips
        # may miss a pixel centre; anything else is a different frame.
        for png, svg in zip(scene["png_shape"], scene["svg_shape"]):
            self.assertLessEqual(abs(png - svg), 4, scene)

    def test_the_svg_is_framed_like_the_png(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                results = self.compare(scale)
                for name in ("outlines", "edges"):
                    scene = results[name]
                    self.assertEqual(scene["png_margins"], [20, 20, 20, 20], name)
                    self.assert_frames_match(scene)
                    for margin in scene["svg_margins"]:
                        self.assertLessEqual(abs(margin - 20), 2, (name, scene))
                for shape, scene in results["shapes"].items():
                    with self.subTest(shape=shape):
                        self.assert_frames_match(scene)

    def test_outlines_are_as_wide_as_on_the_png(self):
        # Outlines of 0.5, 0.25, 1 and 2 pixels; the 2-pixel one is a
        # selection's, which the SVG copies from the viewer.
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                scene = self.compare(scale)["outlines"]
                for node, (png, svg) in enumerate(zip(scene["png_ink"], scene["svg_ink"])):
                    self.assertAlmostEqual(svg / png, 1.0, delta=0.08, msg=(node, scene))

    def test_edges_are_as_wide_as_on_the_png(self):
        # 1.3-pixel edges: 1 device pixel at 100 %, 3 at 200 %.
        for scale, drawn in ((1, 1.0), (2, 1.5)):
            with self.subTest(display_scale=scale):
                scene = self.compare(scale)["edges"]
                self.assertAlmostEqual(scene["png_ink"], drawn, delta=0.05, msg=scene)
                self.assertAlmostEqual(scene["svg_ink"] / scene["png_ink"], 1.0, delta=0.05, msg=scene)

    def test_edges_whose_alpha_follows_the_score_are_as_dark_as_on_the_png(self):
        # Four edges of scores 0.1, 0.4, 0.7 and 1 (listed from the top row down): opacities 85, 62, 38 and 15 %.
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                scene = self.compare(scale)["auto_edges"]
                png, svg = scene["png_inks"], scene["svg_inks"]
                self.assertEqual((len(png), len(svg)), (4, 4), scene)
                # The PNG really draws them by score, the top row (score 1) darkest: 85 % to 15 %.
                self.assertEqual(png, sorted(png, reverse=True), scene)
                self.assertAlmostEqual(png[0] / png[3], 0.85 / 0.15, delta=0.6, msg=scene)
                for edge, (on_png, on_svg) in enumerate(zip(png, svg)):
                    self.assertAlmostEqual(on_svg / on_png, 1.0, delta=0.06, msg=(edge, scene))

    def test_every_shape_is_drawn_as_on_the_png(self):
        for scale in (1, 2):
            for shape, scene in self.compare(scale)["shapes"].items():
                with self.subTest(display_scale=scale, shape=shape):
                    self.assertAlmostEqual(scene["ink_ratio"], 1.0, delta=0.1, msg=scene)
                    # cross_lines is a line a pixel or two wide, whose overlap
                    # says little; its colour above says how wide it is.
                    if shape != "cross_lines":
                        self.assertGreaterEqual(scene["overlap"], 0.9, scene)


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
            print_command, "_render_tile", side_effect=white_renders
        ) as capture, mock.patch.object(
            print_command, "_write_png"
        ) as save, mock.patch.object(
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
                self.assertTrue(viewer.zoom_text.visible)
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

    def test_a_name_a_queued_job_holds_is_taken_too(self):
        # A print still in the queue has no file yet, but its name is reserved.
        with tempfile.TemporaryDirectory() as directory:
            reserved = {os.path.join(directory, name) for name in ("set_1.png", "set_1_2.png")}
            scheduler = SimpleNamespace(is_output_path_reserved=lambda path: path in reserved)
            open(os.path.join(directory, "set_1_3.png"), "w").close()
            self.assertEqual(
                print_command._available_automatic_filename(directory, "set_1.png", scheduler),
                "set_1_4.png",
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
            # The job's report names the file that was written.
            self.assertEqual(
                viewer.background_job_scheduler.report,
                "Saved PNG: test_sequences_20260102_030405_3.png",
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
                self.assertEqual(first.background_job_scheduler.report, written)
                self.assertEqual(second.background_job_scheduler.report, f"{written}. {replaced}")


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

        def capture(captured_viewer, is_transparent):
            tile_rects.append(captured_viewer.view.camera.rect)
            return white_renders(captured_viewer, is_transparent, TILE_SHAPE)

        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        save_dir = os.path.join(root.name, "Saved_Images")
        with mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command, "_render_tile", side_effect=capture
        ) as capture_tile, mock.patch.object(
            print_command, "_svg_snapshot"
        ) as export_svg, mock.patch.object(
            print_command, "_write_svg"
        ), mock.patch.object(
            print_command, "_write_png"
        ) as save, redirect_stdout(io.StringIO()) as output:
            print_command.run(viewer, arguments)
        # The live camera is the one it was before, untouched.
        self.assertIs(viewer.view.camera.rect, live_rect)
        self.assertEqual(
            (live_rect.pos, live_rect.width, live_rect.height), ((0.0, -12.5), 100.0, 75.0)
        )
        return SimpleNamespace(
            viewer=viewer, engine=engine, capture=capture_tile, export_svg=export_svg,
            save=save, save_dir=save_dir, tile_rects=tile_rects,
            scheduler=viewer.background_job_scheduler, output=output.getvalue(),
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
        self.assertTrue(result.viewer.zoom_text.visible)
        self.assertFalse(os.path.exists(result.save_dir))

    # --- parsing ---------------------------------------------------------

    def test_zoom_n_may_stand_anywhere_among_the_modifiers(self):
        cases = (
            (["full", "zoom", "200"], None),
            (["fig1", "full", "zoom", "200"], "fig1.png"),
            (["zoom", "200", "fig1", "full"], "fig1.png"),
            (["fig1", "zoom", "200", "full", "transparent"], "fig1.png"),
            (["my_net", "zoom", "200", "full"], "my_net.png"),
            (["ZOOM", "200", "Full"], None),
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
                # Every tile after the first, which measures the canvas, is 500 units wide: 200%.
                self.assertTrue(all(rect[2] == 500.0 for rect in result.tile_rects[1:]))

    def test_the_percent_sign_is_optional_and_may_stand_apart_from_n(self):
        cases = (
            (["fig1", "full", "zoom", "200%"], "fig1.png"),
            (["fig1", "full", "zoom", "200", "%"], "fig1.png"),
            (["zoom", "200", "%", "fig1", "full"], "fig1.png"),
            (["full", "zoom", "200", "%", "transparent"], None),
            (["ZOOM", "200%", "Full"], None),
            (["fig1", "full", "zoom", "2e2"], "fig1.png"),
        )
        for arguments, name in cases:
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                result.save.assert_called_once()
                if name is not None:
                    self.assertEqual(os.path.basename(result.save.call_args.args[0]), name)
                self.assertTrue(all(rect[2] == 500.0 for rect in result.tile_rects[1:]))
        for arguments in (["svg", "zoom", "200%"], ["svg", "zoom", "200", "%"]):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertEqual(result.export_svg.call_args.kwargs["view_width"], 500.0)

    def test_a_percent_sign_after_something_else_is_no_part_of_n(self):
        # Only the word right after N may be the sign; a % after a file name is a second word.
        self.assert_refused(
            ["full", "zoom", "200", "fig1", "%"],
            "Error: Unrecognized print argument '%'. A file name is one word; "
            "the modifiers are transparent, full, svg, zoom N.",
        )

    def test_svg_takes_zoom_n_and_the_name_alongside_it(self):
        for arguments, name in (
            (["svg", "zoom", "200"], None),
            (["fig1", "svg", "zoom", "200"], "fig1.svg"),
            (["zoom", "200", "svg", "fig1"], "fig1.svg"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                result.export_svg.assert_called_once()
                self.assertEqual(result.export_svg.call_args.kwargs["view_width"], 500.0)
                saved = os.path.basename(result.scheduler.jobs[0].output_path)
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
            (["zoom.png", "full", "zoom", "200"], "zoom.png"),
            (["my_zoom.png"], "my_zoom.png"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertEqual(os.path.basename(result.save.call_args.args[0]), name)

    def test_zoom_equals_n_is_zoom_n_in_one_word(self):
        cases = (
            (["full", "zoom=200"], ["full", "zoom", "200"]),
            (["fig1", "full", "zoom=200"], ["fig1", "full", "zoom", "200"]),
            (["zoom=200", "fig1", "full"], ["zoom", "200", "fig1", "full"]),
            (["fig1", "zoom=200", "full", "transparent"], ["fig1", "zoom", "200", "full", "transparent"]),
            (["ZOOM=200", "Full"], ["ZOOM", "200", "Full"]),
            (["fig1", "full", "zoom=200%"], ["fig1", "full", "zoom", "200%"]),
            (["fig1", "full", "zoom=200", "%"], ["fig1", "full", "zoom", "200", "%"]),
            (["fig1", "full", "zoom=2e2"], ["fig1", "full", "zoom", "2e2"]),
            (["full", "zoom=2000"], ["full", "zoom", "2000"]),
        )
        for equals, spaced in cases:
            with self.subTest(arguments=equals):
                one = self.run_print(equals)
                two = self.run_print(spaced)
                one.engine.command_failed.assert_not_called()
                one.save.assert_called_once()
                self.assertEqual(one.tile_rects, two.tile_rects)
                self.assertEqual(
                    os.path.basename(one.save.call_args.args[0]).split("_")[0],
                    os.path.basename(two.save.call_args.args[0]).split("_")[0],
                )
                self.assertTrue(one.tile_rects[1:])
        for equals, spaced in (
            (["svg", "zoom=200"], ["svg", "zoom", "200"]),
            (["fig1", "svg", "zoom=200%"], ["fig1", "svg", "zoom", "200%"]),
        ):
            with self.subTest(arguments=equals):
                one, two = self.run_print(equals), self.run_print(spaced)
                one.engine.command_failed.assert_not_called()
                self.assertEqual(one.export_svg.call_args.kwargs["view_width"], 500.0)
                self.assertEqual(
                    os.path.basename(one.scheduler.jobs[0].output_path).split("_")[0],
                    os.path.basename(two.scheduler.jobs[0].output_path).split("_")[0],
                )

    def test_zoom_equals_n_is_not_a_file_name(self):
        result = self.run_print(["full", "zoom=200"])
        self.assertTrue(os.path.basename(result.save.call_args.args[0]).startswith("test_sequences_"))
        self.assertTrue(all(rect[2] == 500.0 for rect in result.tile_rects[1:]))

    def test_a_name_with_an_equals_sign_that_does_not_start_with_zoom_equals_is_still_a_name(self):
        for arguments, name in (
            (["my=name"], "my=name.png"),
            (["a=zoom"], "a=zoom.png"),
            (["zoomed=3"], "zoomed=3.png"),
            (["zoom-3=x"], "zoom-3=x.png"),
            (["zoomx=1", "full"], "zoomx=1.png"),
            (["fig=1", "full", "zoom=200"], "fig=1.png"),
            (["zoom=200", "my=name", "full"], "my=name.png"),
            (["my=name.svg"], "my=name.svg"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                saved = (
                    result.scheduler.jobs[0].output_path if name.endswith(".svg") else result.save.call_args.args[0]
                )
                self.assertEqual(os.path.basename(saved), name)

    def test_zoom_equals_is_refused_as_zoom_is(self):
        needs = "Error: zoom needs a zoom percentage, as in print full zoom 200."
        for arguments in (["full", "zoom="], ["full", "zoom=full"], ["svg", "zoom=svg"], ["zoom=", "200", "full"],
                          ["fig1", "zoom=transparent", "full"], ["ZOOM=", "full"]):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, needs)
        for arguments in (["zoom=200"], ["fig1", "zoom=200"], ["fig1", "transparent", "zoom=200%"]):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, "Error: zoom N works only with full or svg.")
        for arguments in (["full", "zoom=200", "zoom", "300"], ["full", "zoom", "200", "zoom=300"],
                          ["full", "zoom=200", "zoom=300"], ["svg", "zoom=200%", "zoom=200"]):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, "Error: zoom N can be given only once.")
        self.assert_refused(
            ["my", "zoom=200", "net", "full"],
            "Error: Unrecognized print argument 'net'. A file name is one word; "
            "the modifiers are transparent, full, svg, zoom N.",
        )
        for value, message in (
            ("wide", "Error: Zoom must be a number, with an optional % sign, as in zoom 200 or zoom 200%."),
            ("0", "Error: Zoom must be a positive, finite percentage."),
            ("-5", "Error: Zoom must be a positive, finite percentage."),
            ("200%%", "Error: Zoom must be a number, with an optional % sign, as in zoom 200 or zoom 200%."),
        ):
            for modifier in ("full", "svg"):
                with self.subTest(value=value, modifier=modifier):
                    self.assert_refused([modifier, f"zoom={value}"], message)

    def test_zoom_without_a_percentage_is_refused(self):
        message = "Error: zoom needs a zoom percentage, as in print full zoom 200."
        for arguments in (["full", "zoom"], ["zoom", "full"], ["svg", "zoom", "svg"],
                          ["fig1", "zoom", "transparent", "full"], ["zoom"]):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, message)

    def test_an_invalid_zoom_is_refused_as_the_zoom_command_refuses_it(self):
        not_a_number = "Error: Zoom must be a number, with an optional % sign, as in zoom 200 or zoom 200%."
        not_positive = "Error: Zoom must be a positive, finite percentage."
        cases = (
            ("wide", not_a_number),
            ("500.png", not_a_number),
            ("%", not_a_number),
            ("200%%", not_a_number),
            ("0", not_positive),
            ("-5", not_positive),
            ("0%", not_positive),
            ("nan", not_positive),
            ("inf", not_positive),
            ("1e400", not_positive),
            ("1e-47", "Error: Zoom is too low for the current view."),
            ("1e19", "Error: Zoom is too high to draw accurately at the current view centre."),
        )
        for zoom, message in cases:
            for modifier in ("full", "svg"):
                with self.subTest(zoom=zoom, modifier=modifier):
                    self.assert_refused([modifier, "zoom", zoom], message)

    def test_a_canvas_without_area_is_refused(self):
        viewer = zoom_viewer()
        viewer.canvas.size = (0, 600)
        result = self.run_print(["full", "zoom", "200"], viewer)
        result.capture.assert_not_called()
        self.assertEqual(
            self.failure(result),
            "Error: The canvas has no visible area, so the zoom cannot be applied.",
        )

    def test_zoom_given_twice_is_refused(self):
        self.assert_refused(
            ["full", "zoom", "200", "zoom", "300"], "Error: zoom N can be given only once."
        )

    def test_zoom_with_a_plain_png_is_refused(self):
        message = "Error: zoom N works only with full or svg."
        for arguments in (["zoom", "200"], ["fig1", "zoom", "200"],
                          ["fig1", "transparent", "zoom", "200"]):
            with self.subTest(arguments=arguments):
                self.assert_refused(arguments, message)

    def test_the_svg_rules_still_hold_with_zoom(self):
        self.assert_refused(
            ["svg", "full", "zoom", "200"],
            "Error: 'SVG' export is not compatible with 'transparent' or 'full'.",
        )
        # A second word is refused as a name before the svg rules apply.
        self.assert_refused(
            ["my", "net", "svg", "zoom", "200"],
            "Error: Unrecognized print argument 'net'. A file name is one word; "
            "the modifiers are transparent, full, svg, zoom N.",
        )

    # --- full ------------------------------------------------------------

    def test_full_with_zoom_renders_tiles_n_units_wide_and_leaves_the_camera(self):
        # 2000% is 50 units across 800 pixels: 0.0625 units per pixel, so the 110 x 60
        # network is 1760 x 960 pixels. The tiles keep the view's 4:3 shape.
        result = self.run_print(["fig1", "full", "zoom", "2000"])

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

    def test_zoom_at_the_views_own_zoom_gives_the_same_picture_as_no_zoom(self):
        plain = self.run_print(["fig1", "full"])
        zoomed = self.run_print(["fig1", "full", "zoom", "1000"])

        self.assertEqual(plain.tile_rects[1:], zoomed.tile_rects[1:])
        np.testing.assert_array_equal(
            plain.save.call_args.args[1], zoomed.save.call_args.args[1]
        )

    def test_a_small_zoomed_image_is_saved_with_a_warning(self):
        # 200% is 500 units across 800 pixels: 0.625 per pixel, so 176 x 96 pixels.
        result = self.run_print(["fig1", "full", "zoom", "200"])

        result.save.assert_called_once()
        self.assertEqual(result.save.call_args.args[1].shape, (96, 176, 4))
        result.engine.command_failed.assert_not_called()
        warning = (
            "Warning: the image is only 176×96 px. N in zoom N is the zoom percentage, "
            "so a bigger N zooms in and gives a larger image."
        )
        self.assertIn("Successfully saved full stitched snapshot: ", result.output)
        self.assertTrue(result.output.rstrip().endswith(warning), result.output)
        self.assertEqual(result.scheduler.report, f"Saved PNG: fig1.png. {warning}")

    # --- size limit ------------------------------------------------------

    def limited(self, megapixels):
        patcher = mock.patch.object(print_command, "MAX_FULL_IMAGE_MEGAPIXELS", megapixels)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_full_image_over_the_limit_is_refused_before_any_tile(self):
        # zoom 2000 makes 1760 x 960 px, 1.69 megapixels.
        self.limited(1)
        result = self.run_print(["fig1", "full", "zoom", "2000"])

        message = (
            "Error: The full image would be 1760×960 px (2 megapixels), more than the "
            "limit of 1. Use print full zoom N with N of at most 1520, or zoom out. "
            "The limit is MAX_FULL_IMAGE_MEGAPIXELS in src/commands/print.py."
        )
        self.assertEqual(self.failure(result), message)
        self.assertEqual(result.viewer.console_text.text, message)
        # Only the render that measures the canvas: no tile was drawn.
        self.assertEqual(len(result.tile_rects), 1)
        result.save.assert_not_called()
        result.engine.command_succeeded.assert_not_called()
        self.assertTrue(result.viewer.zoom_text.visible)
        self.assertFalse(os.path.exists(result.save_dir))

    def test_the_n_it_suggests_fits_the_limit(self):
        for limit, arguments, highest in (
            # 2000% is 50 units wide; 65.64 units, 1523.5%, is the narrowest that fits.
            (1, ["fig1", "full", "zoom", "2000"], "1520"),
            # Without zoom N, the scale is the view's: 100 units wide, 880 x 480 px.
            (0.2, ["fig1", "full"], "681"),
        ):
            with self.subTest(arguments=arguments):
                self.limited(limit)
                refused = self.run_print(arguments)
                self.assertIn(f"N of at most {highest}, or zoom out", self.failure(refused))

                result = self.run_print(["fig1", "full", "zoom", highest])
                result.engine.command_failed.assert_not_called()
                height, width = result.save.call_args.args[1].shape[:2]
                self.assertLessEqual(width * height, limit * 1e6)

    def test_an_image_at_the_limit_is_made(self):
        self.limited(1)
        self.assertIsNone(print_command.full_image_size_refusal(1000, 1000, 50.0))
        self.assertIsNotNone(print_command.full_image_size_refusal(1001, 1000, 50.0))

    def test_the_suggested_n_is_rounded_down_to_three_digits(self):
        # Down, so the zoom it names is never past the limit.
        for value, text in ((65.64, "65.6"), (1234.5, "1230"), (100.0, "100"),
                            (999.9, "999"), (0.012345, "0.0123")):
            with self.subTest(value=value):
                self.assertEqual(print_command._round_down(value), text)

    def test_the_warning_needs_zoom_and_an_image_under_1000_pixels(self):
        # No zoom: 880 x 480 pixels, and no warning. zoom 2000: 1760 x 960, none either.
        for arguments in (["fig1", "full"], ["fig1", "full", "zoom", "2000"],
                          ["fig1", "full", "transparent"]):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertNotIn("Warning", result.output)
                self.assertEqual(result.scheduler.report, "Saved PNG: fig1.png")
        # An SVG has no pixel size, so it never warns.
        result = self.run_print(["fig1", "svg", "zoom", "2"])
        self.assertNotIn("Warning", result.output)
        self.assertEqual(result.scheduler.report, "Saved SVG: fig1.svg")

    def test_the_limit_is_a_thousand_megapixels(self):
        self.assertEqual(print_command.MAX_FULL_IMAGE_MEGAPIXELS, 1000)
        # 31,622 px square is just under a gigapixel; 31,623 just over.
        self.assertIsNone(print_command.full_image_size_refusal(31622, 31622, 50.0))
        self.assertIsNotNone(print_command.full_image_size_refusal(31623, 31623, 50.0))


class PrintSvgZoomTests(unittest.TestCase):
    """`print svg zoom N` writes the sizes of the view N% sets: its width / the view's width per pixel."""

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
        ), mock.patch.multiple(
            print_command.cfg, create=True, EDGE_COLOR="#000000",
            NODE_BOUNDARY_COLOR="#000000", EDGE_ALPHA=0.2, EDGE_WIDTH=1.5,
            UMAP_MODE=False,
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, arguments)
            with open(os.path.join(save_dir, "net.svg"), encoding="utf-8") as handle:
                return handle.read()

    def test_node_radius_is_half_its_size_times_the_zooms_width_over_the_view_width(self):
        svg = self.svg(["net", "svg", "zoom", "200"])

        circles = elements(svg, "circle")
        self.assertEqual(len(circles), 4)
        # The SVG is written to three decimals.
        for circle in circles:
            self.assertAlmostEqual(number(circle, "r"), 12 / 2 * 500 / 800, delta=0.001)
            # The 0.5-pixel outline looks 0.98 pixels wide on screen.
            self.assertAlmostEqual(number(circle, "stroke-width"), 0.9818 * 500 / 800, delta=0.001)
        # GL draws the 1.5-pixel edges 2 pixels wide.
        lines = elements(svg, "line")
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertAlmostEqual(number(line, "stroke-width"), 2 * 500 / 800, delta=0.001)

    def test_the_scale_ignores_the_current_zoom(self):
        self.assertEqual(
            self.svg(["net", "svg", "zoom", "200"], units_per_pixel=0.2),
            self.svg(["net", "svg", "zoom", "200"], units_per_pixel=0.05),
        )

    def test_zoom_at_the_views_own_zoom_writes_the_same_svg_as_no_zoom(self):
        # The view at 0.2 units per pixel is 160 units wide: 625%.
        self.assertEqual(self.svg(["net", "svg"]), self.svg(["net", "svg", "zoom", "625"]))

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
            print_command, "_render_tile",
            side_effect=lambda viewer, is_transparent: white_renders(viewer, is_transparent, TILE_SHAPE),
        ) as capture, mock.patch.object(
            print_command, "_svg_snapshot"
        ) as export_svg, mock.patch.object(
            print_command, "_write_svg"
        ) as write_svg, mock.patch.object(
            print_command, "_write_png"
        ) as save, redirect_stdout(io.StringIO()) as output:
            print_command.run(viewer, arguments)
        return SimpleNamespace(
            viewer=viewer, engine=engine, capture=capture, export_svg=export_svg,
            write_svg=write_svg, save=save, save_dir=save_dir, output=output.getvalue(),
            scheduler=viewer.background_job_scheduler,
        )

    def written(self, result):
        """The base name of the one file the run wrote, as a PNG or an SVG."""
        if result.export_svg.called:
            result.save.assert_not_called()
            result.write_svg.assert_called_once()
            return os.path.basename(result.write_svg.call_args.args[1])
        result.write_svg.assert_not_called()
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
        self.assertTrue(result.viewer.zoom_text.visible)
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
            (["full", "zoom", "200", "fig1"], "fig1.png"),
            (["svg", "fig1"], "fig1.svg"),
            (["fig1", "svg"], "fig1.svg"),
            (["zoom", "200", "svg", "fig1"], "fig1.svg"),
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
                self.assertEqual(result.scheduler.report, f"Saved SVG: {name}")
                self.assertIn("Successfully saved SVG snapshot: ", result.output)

    def test_an_svg_name_takes_zoom_n_like_the_modifier(self):
        for arguments in (["foo.svg", "zoom", "200"], ["zoom", "200", "foo.svg"]):
            with self.subTest(arguments=arguments):
                result = self.run_print(arguments)
                result.engine.command_failed.assert_not_called()
                self.assertEqual(self.written(result), "foo.svg")
                self.assertEqual(result.export_svg.call_args.kwargs["view_width"], 500.0)

    def test_the_svg_rules_apply_to_an_svg_name(self):
        message = "Error: 'SVG' export is not compatible with 'transparent' or 'full'."
        for arguments in (["foo.svg", "full"], ["foo.svg", "transparent"],
                          ["transparent", "full", "foo.svg"], ["foo.svg", "full", "zoom", "200"]):
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
            (["foo.png", "svg", "zoom", "200"],
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
            ([".png", "transparent"], ".png"), ([".svg", "zoom", "200"], ".svg"),
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
                self.assertEqual(result.scheduler.report, f"{shown}. {self.REPLACED}")  # the report ends with a full stop
                self.assertIn("Successfully saved ", result.output)
                self.assertTrue(result.output.rstrip().endswith(self.REPLACED), result.output)

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
                self.assertNotIn(self.REPLACED, result.scheduler.report)
                self.assertNotIn(self.REPLACED, result.output)

    def test_the_replacement_note_and_the_size_warning_both_ride_along(self):
        # 200% is 500 units across 800 pixels: a 176 x 96 picture, so the warning applies.
        result = self.run_print(["mine", "full", "zoom", "200"], existing=("mine.png",))
        warning = (
            "Warning: the image is only 176×96 px. N in zoom N is the zoom percentage, "
            "so a bigger N zooms in and gives a larger image."
        )
        self.assertEqual(
            result.scheduler.report, f"Saved PNG: mine.png. {self.REPLACED} {warning}"
        )
        self.assertTrue(result.output.rstrip().endswith(f"{self.REPLACED}\n{warning}"), result.output)

    def test_the_report_with_a_note_joins_in_chinese_punctuation_and_shows_every_part_translated(self):
        from PySide6.QtWidgets import QApplication
        from tests.translation_fixtures import outside_the_catalog, pseudo_language
        from utilities import Localization
        from utilities.Language_Packs import read_pack

        message = self.run_print(["mine"], existing=("mine.png",)).scheduler.reports[-1]["message"]
        # The report's own full stop is the language's: the sentence after it follows with no space.
        translated = {
            "{report}.": "{report}。",
            "Saved {format}: {file}": "已保存 {format}：{file}",
            self.REPLACED: "已替换同名的现有文件。",
        }
        previous = Localization.set_translator(lambda template, n: translated.get(template, template))
        self.addCleanup(Localization.set_translator, previous)
        previous_punctuation = Localization.set_punctuation(read_pack("zh_CN").punctuation)
        self.addCleanup(Localization.set_punctuation, previous_punctuation)
        self.assertEqual(message.display(), "已保存 PNG：mine.png。已替换同名的现有文件。")
        # Under the pseudo-language no letter of it comes from outside the catalog.
        Localization.set_translator(previous)
        Localization.set_punctuation(previous_punctuation)
        pseudo_language(self, QApplication.instance() or QApplication([]))
        self.assertEqual(outside_the_catalog(message.display()), "")

    def test_a_refusal_never_reports_a_replacement(self):
        result = self.run_print(["mine", "ful"], existing=("mine.png",))
        result.engine.command_succeeded.assert_not_called()
        self.assertEqual(result.scheduler.jobs, [])
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


class PrintBackgroundJobTests(unittest.TestCase):
    """print in the viewer's real background job queue: the command returns
    once the picture is taken, and the queue's thread finishes and writes
    the file from what print copied, then reports it."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtCore

        # Reuse an application another test module created and leave it alive;
        # only an application created here is shut down in tearDownClass.
        cls._owns_application = QtCore.QCoreApplication.instance() is None
        cls.application = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    @classmethod
    def tearDownClass(cls):
        from PySide6 import QtCore

        application = cls.application
        cls.application = None
        if cls._owns_application:
            application.quit()
            application.deleteLater()
            application.processEvents()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        del application
        import gc
        gc.collect()

    def setUp(self):
        from Background_Job_Scheduler import BackgroundJobScheduler

        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.save_dir = os.path.join(root.name, "Saved_Images")
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        self.engine = engine
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(print_command, "PRINT_DIRECTORY", self.save_dir))
        self.open_folder = stack.enter_context(mock.patch("Background_Job_Scheduler.open_in_file_manager"))
        stack.enter_context(mock.patch.multiple(
            print_command.cfg, create=True, EDGE_COLOR="#000000",
            NODE_BOUNDARY_COLOR="#000000", EDGE_ALPHA=0.2, EDGE_WIDTH=0.5, UMAP_MODE=False,
        ))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.status = StatusStub()
        self.scheduler = BackgroundJobScheduler(self.status)
        # First of the cleanups: no job runs once the patches and folder go.
        self.addCleanup(self.stop_scheduler)

    def stop_scheduler(self, release=None):
        """Discard the queued jobs, let a blocked one end, wait for the queue to empty, and delete the scheduler."""
        self.scheduler.shutdown()
        if release is not None:
            release.set()
        self.assertTrue(wait_for(lambda: self.scheduler.queue_depth == 0))
        end_scheduler(self.scheduler)

    def viewer(self, svg=False):
        if svg:
            viewer = square_network()
            for name, value in vars(PrintMarginTrimTests.make_viewer()).items():
                if name not in ("canvas", "view"):
                    setattr(viewer, name, value)
            viewer.canvas.update = mock.Mock()
        else:
            viewer = zoom_viewer()
        viewer.background_job_scheduler = self.scheduler
        return viewer

    def block_the_queue(self):
        """Put a job in front that runs until the returned event is set."""
        started, release = threading.Event(), threading.Event()

        def blocker(_payload):
            started.set()
            release.wait(10)
            return {"message": "blocker done"}

        self.scheduler.enqueue(
            command_name="blocker", description="blocker", payload=None, worker=blocker,
            output_path=os.path.join(self.save_dir, "blocker.txt"),
        )
        self.assertTrue(started.wait(10))
        self.addCleanup(self.stop_scheduler, release)
        return release

    def finished(self, count=1):
        """Wait until count jobs have reported, and return the status lines."""
        done = lambda: sum(
            "completed" in line or "failed" in line for line in self.status.statuses
        ) >= count
        self.assertTrue(wait_for(done), self.status.statuses)
        return self.status.statuses

    def test_print_returns_while_the_queue_thread_encodes_the_png(self):
        viewer = self.viewer()
        threads, release = [], threading.Event()

        def encode(target, image, **kwargs):
            threads.append(threading.current_thread().name)
            self.assertTrue(release.wait(10))
            original_imsave(target, image, **kwargs)

        original_imsave = print_command.mpimg.imsave
        path = os.path.join(self.save_dir, "queued.png")
        with mock.patch.object(print_command, "_render_tile", side_effect=lambda v, t: white_renders(v, t, TILE_SHAPE)), \
                mock.patch.object(print_command.mpimg, "imsave", side_effect=encode):
            print_command.run(viewer, ["queued", "full"])
            # The command is done; the job waits in the encoder.
            self.engine.command_failed.assert_not_called()
            self.assertEqual(
                str(self.engine.command_succeeded.call_args.args[1]),
                f"Queued print for {path}; waiting for the background job.",
            )
            self.assertTrue(wait_for(lambda: threads))
            self.assertFalse(os.path.exists(path))
            self.assertTrue(self.scheduler.is_output_path_reserved(path))
            release.set()
            statuses = self.finished()

        self.assertEqual(threads, ["SSN-Background-Jobs"])
        self.assertEqual(os.listdir(self.save_dir), ["queued.png"])
        self.assertRegex(statuses[-1], r"^Background job #1 completed in [\d.]+s: Saved PNG: queued\.png$")
        self.open_folder.assert_called_once_with(self.save_dir)
        self.assertFalse(self.scheduler.is_output_path_reserved(path))

    def test_the_file_shows_the_network_as_it_was_when_print_ran(self):
        expected = export_svg(square_network())
        release = self.block_the_queue()
        viewer = self.viewer(svg=True)

        print_command.run(viewer, ["net", "svg"])
        # While the job waits behind another, the network changes in place
        # and in whole: nodes move, recolour, hide; edges and threshold change.
        viewer.pos[:] += 1000.0
        viewer.current_colors[:] = 0.0
        viewer.current_sizes[:] = 3.0
        viewer.visible_mask[1] = False
        viewer.edges = np.array([[0, 2]], dtype=np.int32)
        viewer.current_slider_threshold = 0.0
        viewer.canvas.bgcolor = SimpleNamespace(rgba=(0.0, 0.0, 0.0, 1.0))
        release.set()
        self.finished(count=2)

        with open(os.path.join(self.save_dir, "net.svg"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), expected)

    def test_a_name_a_queued_print_holds_is_refused(self):
        self.block_the_queue()
        viewer = self.viewer()
        path = os.path.join(self.save_dir, "same.png")
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)):
            print_command.run(viewer, ["same"])
            print_command.run(viewer, ["same"])

        self.engine.command_succeeded.assert_called_once()
        self.assertEqual(
            str(self.engine.command_failed.call_args.args[1]),
            f"Error: Output file is already reserved by a background job: {path}",
        )
        self.assertEqual(self.scheduler.queue_depth, 2)

    def test_timestamped_names_skip_the_ones_queued_prints_hold(self):
        release = self.block_the_queue()
        viewer = self.viewer()
        clock = SimpleNamespace(datetime=PrintAutomaticNameTests.FixedClock)
        with mock.patch.object(print_command, "datetime", clock), \
                mock.patch.object(print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)):
            print_command.run(viewer, [])
            print_command.run(viewer, [])
            release.set()
            self.finished(count=3)

        self.assertEqual(
            sorted(os.listdir(self.save_dir)),
            ["test_sequences_20260102_030405.png", "test_sequences_20260102_030405_2.png"],
        )

    def test_a_timestamped_name_taken_while_queued_is_not_written_over(self):
        # Another program saves a file of that name while the print waits.
        release = self.block_the_queue()
        viewer = self.viewer()
        clock = SimpleNamespace(datetime=PrintAutomaticNameTests.FixedClock)
        path = os.path.join(self.save_dir, "test_sequences_20260102_030405.png")
        with mock.patch.object(print_command, "datetime", clock), \
                mock.patch.object(print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)), \
                redirect_stderr(io.StringIO()):
            print_command.run(viewer, [])
            os.makedirs(self.save_dir)
            with open(path, "w") as handle:
                handle.write("someone else's")
            release.set()
            statuses = self.finished(count=2)

        self.assertIn("Output file already exists", statuses[-1])
        with open(path) as handle:
            self.assertEqual(handle.read(), "someone else's")
        self.assertEqual(os.listdir(self.save_dir), ["test_sequences_20260102_030405.png"])

    def test_a_failed_job_is_reported_and_leaves_no_file(self):
        viewer = self.viewer()
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)), \
                mock.patch.object(print_command.mpimg, "imsave", side_effect=OSError("disk full")), \
                redirect_stderr(io.StringIO()):
            print_command.run(viewer, ["broken"])
            statuses = self.finished()

        self.engine.command_succeeded.assert_called_once()
        self.assertRegex(statuses[-1], r"^Background job #1 failed after [\d.]+s \(print\): disk full$")
        self.assertEqual(os.listdir(self.save_dir), [])
        self.open_folder.assert_not_called()

    # --- a job stopped while it runs ---------------------------------------------------------

    def stop_job(self):
        """Tell the running print job to stop, as the Session card's remove does."""
        (running,) = [entry for entry in self.scheduler.snapshot() if entry.running]
        self.assertTrue(self.scheduler.cancel(running.job.job_id))

    def stopped(self):
        """Wait until the job has ended as stopped, and the line is empty."""
        reported = lambda: any("stopped before it finished" in line for line in self.status.statuses)
        self.assertTrue(wait_for(reported), self.status.statuses)
        self.assertTrue(wait_for(lambda: self.scheduler.queue_depth == 0))

    def stop_when_called(self, name, **patch_options):
        """Patch print_command.<name> to ask for the stop on its first call, and then do what it did."""
        real = getattr(print_command, name)
        calls = []

        def stopping(*args, **kwargs):
            if not calls:
                self.stop_job()
            calls.append(name)
            return real(*args, **kwargs)

        return mock.patch.object(print_command, name, side_effect=stopping, **patch_options), calls

    def test_a_png_stopped_while_it_is_encoded_deletes_its_partial_file(self):
        for arguments in (["half"], ["half", "full"]):
            with self.subTest(arguments=arguments):
                viewer = self.viewer()
                seen, writes, test = [], [], self

                class StopsOnWrite(print_command.CancellableFile):
                    def write(inner, data):
                        writes.append(1)
                        # The partial file is opened by the first write, so it is there for the second.
                        if len(writes) == 2:
                            seen.extend(os.listdir(test.save_dir))
                            test.stop_job()
                        return super().write(data)

                writes.clear()
                seen.clear()
                self.status.statuses.clear()
                with mock.patch.object(print_command, "_render_tile", side_effect=lambda v, t: white_renders(v, t, TILE_SHAPE)), \
                        mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)), \
                        mock.patch.object(print_command, "CancellableFile", StopsOnWrite):
                    print_command.run(viewer, arguments)
                    self.stopped()
                self.assertEqual(len(seen), 1)
                self.assertIn(".partial.png", seen[0], "the partial file existed when the stop came")
                self.assertEqual(os.listdir(self.save_dir), [])
                self.assertFalse(self.scheduler.is_output_path_reserved(os.path.join(self.save_dir, "half.png")))
                self.open_folder.assert_not_called()

    def test_a_full_png_stopped_while_it_is_finished_writes_nothing(self):
        viewer = self.viewer()
        patch, calls = self.stop_when_called("_content_mask")
        with mock.patch.object(print_command, "_render_tile", side_effect=lambda v, t: white_renders(v, t, TILE_SHAPE)), patch:
            print_command.run(viewer, ["finishing", "full"])
            self.stopped()
        self.assertTrue(calls, "the picture was being finished")
        self.assertEqual(os.listdir(self.save_dir) if os.path.exists(self.save_dir) else [], [])

    def test_an_svg_stopped_while_it_is_written_deletes_its_partial_file(self):
        viewer = self.viewer(svg=True)
        seen = []

        def stops_at_the_nodes(*args, **kwargs):
            # The header and the edges are in the partial file by now.
            seen.extend(name for name in os.listdir(self.save_dir))
            self.stop_job()
            yield from real_chunks(*args, **kwargs)

        real_chunks = print_command._svg_node_chunks
        with mock.patch.object(print_command, "_svg_node_chunks", side_effect=stops_at_the_nodes):
            print_command.run(viewer, ["drawing", "svg"])
            self.stopped()
        self.assertEqual(len(seen), 1)
        self.assertIn(".partial.svg", seen[0])
        self.assertEqual(os.listdir(self.save_dir), [])

    def close_the_viewer_during_the_save(self, arguments, *, job_obeys, wait=2.5, at_write=1):
        """Run print and close the scheduler, as closing the Viewer does, at a write of its PNG.

        With ``job_obeys`` the job waits at that write until it is told to stop; without, it
        holds there until the test lets it go, as a job stuck in one long encode does. The
        file is opened by the first write, so it exists from the second on. Returns (how long
        closing took, what the folder held when closing returned, which is before the job is
        let go).
        """
        from Background_Job_Scheduler import BackgroundJobScheduler
        from utilities.Job_Cancellation import cancel_requested

        self.scheduler = BackgroundJobScheduler(self.status)
        self.addCleanup(self.stop_scheduler)
        viewer = self.viewer()
        entered, release = threading.Event(), threading.Event()

        class HoldsAtFirstWrite(print_command.CancellableFile):
            writes = 0

            def write(inner, data):
                if data:
                    HoldsAtFirstWrite.writes += 1
                    if HoldsAtFirstWrite.writes == at_write:
                        entered.set()
                        if job_obeys:
                            while not cancel_requested():
                                time.sleep(0.005)
                        else:
                            release.wait(30)
                return super().write(data)

        with mock.patch.object(print_command, "_render_tile", side_effect=lambda v, t: white_renders(v, t, TILE_SHAPE)),                 mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)),                 mock.patch.object(print_command, "CancellableFile", HoldsAtFirstWrite):
            print_command.run(viewer, arguments)
            self.assertTrue(entered.wait(30))
            started = time.monotonic()
            self.scheduler.shutdown(wait=wait)
            took = time.monotonic() - started
            left_when_closed = sorted(os.listdir(self.save_dir)) if os.path.isdir(self.save_dir) else []
            release.set()
            deadline = time.monotonic() + 30
            while self.scheduler._thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.005)
        return took, left_when_closed

    def test_closing_the_viewer_during_the_save_leaves_neither_a_partial_file_nor_the_picture(self):
        for arguments in (["closing"], ["closing", "full"]):
            for job_obeys in (True, False):
                with self.subTest(arguments=arguments, job_obeys=job_obeys):
                    if os.path.isdir(self.save_dir):
                        shutil.rmtree(self.save_dir)
                    took, left_when_closed = self.close_the_viewer_during_the_save(arguments, job_obeys=job_obeys, wait=0.4)
                    self.assertLess(took, 3.0, "closing returns within the bound")
                    self.assertEqual(left_when_closed, [])
                    self.assertEqual(os.listdir(self.save_dir), [])

    def test_a_picture_the_job_is_stuck_in_the_middle_of_is_deleted_by_the_scheduler_where_the_system_allows(self):
        # Held after the first write, the PNG's partial file is open in the job's thread. Closing
        # deletes it where a file can be deleted while open; Windows cannot, and there it goes
        # when the job gets on (the stop is then waiting for it), which is all the process needs.
        took, left_when_closed = self.close_the_viewer_during_the_save(["held"], job_obeys=False, wait=0.4, at_write=2)
        self.assertLess(took, 3.0)
        if os.name != "nt":
            self.assertEqual(left_when_closed, [])
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_closing_the_viewer_during_a_print_leaves_an_earlier_finished_picture_as_it_was(self):
        viewer = self.viewer()
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)):
            print_command.run(viewer, ["first"])
            self.finished()
        first = os.path.join(self.save_dir, "first.png")
        with open(first, "rb") as handle:
            before = handle.read()
        self.close_the_viewer_during_the_save(["second"], job_obeys=False, wait=0.3)
        self.assertEqual(os.listdir(self.save_dir), ["first.png"])
        with open(first, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_a_stop_leaves_an_existing_picture_of_that_name_as_it_was(self):
        os.makedirs(self.save_dir)
        existing = os.path.join(self.save_dir, "keep.png")
        with open(existing, "wb") as handle:
            handle.write(b"previous picture")
        viewer = self.viewer()
        patch, calls = self.stop_when_called("_trim_png_margins")
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)), patch:
            print_command.run(viewer, ["keep"])
            self.stopped()
        self.assertTrue(calls)
        self.assertEqual(os.listdir(self.save_dir), ["keep.png"])
        with open(existing, "rb") as handle:
            self.assertEqual(handle.read(), b"previous picture")

    def test_the_name_is_free_again_and_the_next_print_runs_after_a_stop(self):
        viewer = self.viewer()
        patch, _calls = self.stop_when_called("_trim_png_margins")
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)):
            with patch:
                print_command.run(viewer, ["again"])
                self.stopped()
            path = os.path.join(self.save_dir, "again.png")
            self.assertFalse(self.scheduler.is_output_path_reserved(path))
            self.status.statuses.clear()
            print_command.run(viewer, ["again"])
            self.finished()
        self.assertEqual(os.listdir(self.save_dir), ["again.png"])
        self.assertEqual(self.engine.command_failed.call_count, 0)

    def test_without_a_scheduler_nothing_is_drawn_or_written(self):
        viewer = self.viewer()
        del viewer.background_job_scheduler
        with mock.patch.object(print_command, "_render_tile") as render:
            for arguments in (["picture"], ["picture", "full"], ["picture", "svg"]):
                with self.subTest(arguments=arguments):
                    self.engine.command_failed.reset_mock()
                    print_command.run(viewer, arguments)
                    self.assertEqual(
                        str(self.engine.command_failed.call_args.args[1]),
                        "Error: The background job scheduler is unavailable.",
                    )
        render.assert_not_called()
        self.engine.command_succeeded.assert_not_called()
        self.assertFalse(os.path.exists(self.save_dir))


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
            EMAPSSN_Config, CONNECTED_NODE_COLOR="red", NODE_BOUNDARY_COLOR="black",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_print(self, viewer, arguments, capture_error=None, interrupt=False, write_png=None):
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        during = []

        def capture(captured_viewer, is_transparent):
            hidden = getattr(captured_viewer, "transient_marks_hidden", False)
            during.append((drawn(captured_viewer), hidden))
            if capture_error is not None:
                raise capture_error
            return white_renders(captured_viewer, is_transparent, TILE_SHAPE)

        save_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, save_dir, True)
        patches = [
            mock.patch.object(print_command, "PRINT_DIRECTORY", save_dir),
            mock.patch.object(print_command, "_render_tile", side_effect=capture),
            mock.patch.object(print_command, "_write_png", side_effect=write_png),
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
                          ["marks", "full", "zoom", "2000"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer()
                before = drawn(viewer)

                engine, during = self.run_print(viewer, arguments)

                engine.command_failed.assert_not_called()
                engine.command_succeeded.assert_called_once()
                self.assert_drawn_without_transient_marks(viewer, before, during)
                self.assert_restored(viewer, before)

    def test_the_marks_are_back_while_the_png_is_encoded(self):
        for arguments in (["marks"], ["marks", "full"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer()
                before = drawn(viewer)
                while_encoding = []

                def encode(_path, _image):
                    while_encoding.append(viewer.transient_marks_hidden)
                    self.assert_restored(viewer, before)

                engine, _during = self.run_print(viewer, arguments, write_png=encode)

                engine.command_succeeded.assert_called_once()
                self.assertEqual(viewer.background_job_scheduler.errors, [])
                self.assertEqual(while_encoding, [False])

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
                mock.patch.object(print_command, "_svg_snapshot"), \
                mock.patch.object(print_command, "_write_svg"):
            self.run_print(viewer, ["marks", "svg"])
            self.run_print(viewer, ["marks", "zoom", "200"])
        update.assert_not_called()

    def test_a_network_without_marks_is_not_redrawn_but_keeps_new_ones_out_of_the_tiles(self):
        # The mouse moves and clicks while a full capture is drawn, and neither may reach a tile drawn after it.
        for arguments in (["marks"], ["marks", "full"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer(hovered=None, clicked=None)
                with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update:
                    engine, during = self.run_print(viewer, arguments)
                engine.command_failed.assert_not_called()
                update.assert_not_called()
                self.assertTrue(all(hidden for _data, hidden in during))
                self.assertFalse(viewer.transient_marks_hidden)

    def test_a_hover_or_click_that_comes_during_the_capture_is_drawn_when_it_ends(self):
        viewer = marked_viewer(hovered=None, clicked=None)
        with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update:
            restore = print_command._hide_transient_marks(viewer)
            self.assertIsNotNone(restore)
            self.assertTrue(viewer.transient_marks_hidden)
            update.assert_not_called()
            viewer.hovered_node_idx = 1
            viewer.selected_node_idx = 2
            restore()
        update.assert_called_once()
        after = drawn(viewer)
        self.assertFalse(viewer.transient_marks_hidden)
        np.testing.assert_array_equal(after["face_color"][node_slot(after, 1)], self.hover_rgba)
        self.assertEqual(after["node_order"][after["rings"]].tolist(), [2])

    def test_a_hover_or_click_that_is_over_by_then_is_not_redrawn(self):
        viewer = marked_viewer()
        with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update:
            restore = print_command._hide_transient_marks(viewer)
            update.reset_mock()  # the marks that were shown were redrawn away
            viewer.hovered_node_idx = None
            viewer.selected_node_idx = None
            restore()
        update.assert_not_called()
        self.assertFalse(viewer.transient_marks_hidden)

    def test_a_viewer_that_draws_no_marks_has_nothing_to_hide(self):
        viewer = zoom_viewer()
        self.assertIsNone(print_command._hide_transient_marks(viewer))
        self.assertFalse(getattr(viewer, "transient_marks_hidden", False))


class PrintFirstUseTests(unittest.TestCase):
    def test_the_module_does_not_load_the_image_writer_on_the_gui_thread(self):
        # The first print of a session loads the module on the GUI thread: matplotlib's image writer waits for the job.
        script = (
            "import sys\n"
            f"sys.path[:0] = [{PROJECT_ROOT!r}, {SRC_DIR!r}]\n"
            "import tests  # noqa: F401\n"
            "from commands import print as print_command\n"
            "assert 'matplotlib.image' not in sys.modules\n"
            "writer = print_command.mpimg\n"
            "assert writer is sys.modules['matplotlib.image']\n"
            "print('LOADED WHEN ASKED FOR')\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=PROJECT_ROOT,
                                timeout=300, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout[-1500:] + result.stderr[-2500:])
        self.assertIn("LOADED WHEN ASKED FOR", result.stdout)


class PrintCooperativeTests(unittest.TestCase):
    """print as the command queue runs it: start() gives a generator that yields after each tile of a full capture."""

    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.save_dir = os.path.join(root.name, "Saved_Images")
        self.engine = print_command.Command_Engine
        for reporter in (self.engine.command_artifact, self.engine.command_succeeded, self.engine.command_failed):
            reporter.reset_mock()
        self.rects = []
        self.render_mock = mock.Mock(side_effect=self.render)
        patches = (
            mock.patch.object(print_command, "PRINT_DIRECTORY", self.save_dir),
            mock.patch.object(print_command, "_render_tile", self.render_mock),
            mock.patch.object(print_command, "_write_png"),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.save = print_command._write_png
        hidden = redirect_stdout(io.StringIO())
        hidden.__enter__()
        self.addCleanup(hidden.__exit__, None, None, None)

    def render(self, viewer, is_transparent):
        self.rects.append(viewer.view.camera.rect)
        return white_renders(viewer, is_transparent, TILE_SHAPE)

    def steps(self, viewer, arguments):
        """The generator the command queue steps for `print arguments`."""
        return print_command.start(viewer, print_command.plan(viewer, arguments))

    def test_the_plan_and_start_are_what_the_queue_reads_and_nothing_is_drawn_until_it_steps(self):
        import inspect
        viewer = zoom_viewer()

        plan = print_command.plan(viewer, ["fig1", "full"])
        steps = print_command.start(viewer, plan)

        self.assertTrue(plan.cooperative)
        self.assertFalse(getattr(plan, "heavy", False))
        self.assertTrue(inspect.isgenerator(steps))
        self.render_mock.assert_not_called()
        self.engine.command_succeeded.assert_not_called()

    def saved_images(self):
        """The pictures _write_png was given since the last call."""
        images = [call.args[1] for call in self.save.call_args_list]
        self.save.reset_mock()
        return images

    def test_stepping_it_to_the_end_saves_what_run_saves(self):
        for arguments in (["fig1", "full"], ["fig1", "full", "zoom", "2000"], ["fig1", "full", "transparent"],
                          ["fig1"], ["fig1", "transparent"]):
            with self.subTest(arguments=arguments):
                print_command.run(zoom_viewer(), arguments)
                expected, expected_rects = self.saved_images(), list(self.rects)
                self.rects.clear()

                stepped = zoom_viewer()
                for _ in self.steps(stepped, arguments):
                    pass

                images = self.saved_images()
                self.assertEqual(len(images), 1)
                self.assertEqual(len(images), len(expected))
                np.testing.assert_array_equal(images[0], expected[0])
                self.assertEqual(self.rects, expected_rects)
                self.assertEqual(len(stepped.background_job_scheduler.jobs), 1)
                self.rects.clear()

    def test_it_yields_the_progress_after_each_tile(self):
        # 100 x 75 units at 0.125 per pixel: a 2 x 2 grid of tiles.
        progress = list(self.steps(zoom_viewer(), ["fig1", "full"]))

        self.assertEqual([str(message) for message in progress], [f"Capturing tile {n} of 4" for n in range(1, 5)])
        self.assertTrue(all(isinstance(message, print_command.Message) for message in progress))

    def test_a_plain_view_and_an_svg_are_taken_in_one_step(self):
        with mock.patch.object(print_command, "_svg_snapshot"), mock.patch.object(print_command, "_write_svg"):
            for arguments in (["fig1"], ["fig1", "svg"], ["help"]):
                with self.subTest(arguments=arguments):
                    self.assertEqual(list(self.steps(zoom_viewer(), arguments)), [])

    def test_a_resize_between_tiles_cancels_it_and_nothing_is_saved(self):
        viewer = zoom_viewer()
        steps = self.steps(viewer, ["fig1", "full"])
        next(steps)
        next(steps)

        viewer.canvas.size = (900, 600)
        self.assertEqual(list(steps), [])  # it reported the failure itself and went no further

        self.engine.command_failed.assert_called_once()
        self.assertIn("window was resized", str(self.engine.command_failed.call_args.args[1]))
        self.engine.command_succeeded.assert_not_called()
        self.save.assert_not_called()
        self.assertEqual(viewer.background_job_scheduler.jobs, [])
        self.assertEqual(self.render_mock.call_count, 3)  # the measuring render and two tiles

    def capture_events(self):
        """Keep the _CaptureEvents a print makes, to tell it what the event loop did."""
        made = []
        real = print_command._CaptureEvents

        def make(*arguments, **keywords):
            made.append(real(*arguments, **keywords))
            return made[-1]

        patch = mock.patch.object(print_command, "_CaptureEvents", side_effect=make)
        patch.start()
        self.addCleanup(patch.stop)
        return made

    def test_a_change_of_the_network_between_tiles_cancels_it(self):
        made = self.capture_events()
        viewer = zoom_viewer()
        steps = self.steps(viewer, ["fig1", "full"])
        next(steps)

        made[0]._mark_changed()  # a visual of the network was updated while the event loop ran
        self.assertEqual(list(steps), [])

        self.engine.command_failed.assert_called_once()
        self.assertIn("network display changed", str(self.engine.command_failed.call_args.args[1]))
        self.save.assert_not_called()
        self.assertEqual(viewer.background_job_scheduler.jobs, [])

    def test_a_camera_move_between_tiles_does_not(self):
        made = self.capture_events()
        viewer = zoom_viewer()
        steps = self.steps(viewer, ["fig1", "full"])
        next(steps)

        made[0]._mark_changed()  # every visual is updated by a camera move too
        viewer.view.camera.rect = SimpleNamespace(pos=(10.0, -12.5), width=100.0, height=75.0)
        remaining = list(steps)

        self.assertEqual(len(remaining), 3)
        self.engine.command_failed.assert_not_called()
        self.engine.command_succeeded.assert_called_once()
        self.save.assert_called_once()

    def test_the_input_that_changes_nothing_cancels_nothing(self):
        made = self.capture_events()
        steps = self.steps(zoom_viewer(), ["fig1", "full"])
        for _ in steps:
            # The event loop ran between every two tiles, and nothing was updated.
            self.assertFalse(made[0]._changed)
        self.engine.command_failed.assert_not_called()
        self.save.assert_called_once()

    def test_events_that_changed_the_network_are_noted_from_the_moment_the_event_loop_is_let_run(self):
        viewer = zoom_viewer()
        events = print_command._CaptureEvents(viewer, [])

        events._mark_changed()  # something before the pause is no part of it
        events.release()
        events.reclaim()

        events._mark_changed()
        events.release()
        events._mark_changed()
        with self.assertRaises(print_command._CaptureInterrupted):
            events.reclaim()

    def composing_viewer(self):
        """A zoom_viewer() whose canvas widget has the signals the QOpenGLWidget sends as its window composes and swaps."""
        viewer = zoom_viewer()
        viewer.canvas.native = SimpleNamespace(
            aboutToCompose=mock.Mock(), frameSwapped=mock.Mock(), window=mock.Mock(return_value=mock.Mock()),
        )
        return viewer

    def compose_for(self, events, clock, seconds):
        """The window composes, its swap taking `seconds`, and the event loop's pass ends."""
        events.release()
        events._compose_began()
        clock[0] += seconds
        events._compose_ended()
        events.reclaim()

    def test_a_window_whose_swap_waits_for_the_desktop_is_not_painted_again_until_the_capture_ends(self):
        # Covered, under XWayland, the swap of every paint the event loop runs takes a second: 144 tiles took 71 s.
        viewer = self.composing_viewer()
        window = viewer.canvas.native.window.return_value
        clock = [100.0]
        with mock.patch.object(print_command, "time", SimpleNamespace(perf_counter=lambda: clock[0])):
            events = print_command._CaptureEvents(viewer, [])
            viewer.canvas.native.aboutToCompose.connect.assert_called_once()
            viewer.canvas.native.frameSwapped.connect.assert_called_once()

            self.compose_for(events, clock, 0.02)  # a window that shows its frames is left alone
            window.setUpdatesEnabled.assert_not_called()

            self.compose_for(events, clock, 1.04)
            window.setUpdatesEnabled.assert_called_once_with(False)

            clock[0] += print_command.CAPTURE_PRESENT_PROBE_S / 2
            events.release()
            events.reclaim()
            window.setUpdatesEnabled.assert_called_once_with(False)  # still held

            clock[0] += print_command.CAPTURE_PRESENT_PROBE_S
            events.release()
            events.reclaim()
            self.assertEqual(window.setUpdatesEnabled.call_args_list[-1], mock.call(True))  # let paint, to see

            self.compose_for(events, clock, 1.04)  # and it still does not show a frame
            self.assertEqual(window.setUpdatesEnabled.call_args_list[-1], mock.call(False))

            events.close()
            self.assertEqual(window.setUpdatesEnabled.call_args_list[-1], mock.call(True))
            events.close()  # closing twice does no more
            self.assertEqual(window.setUpdatesEnabled.call_count, 4)
            viewer.canvas.native.aboutToCompose.disconnect.assert_called_once()
            viewer.canvas.native.frameSwapped.disconnect.assert_called_once()

    def test_a_window_that_was_never_held_is_not_touched_when_the_capture_ends(self):
        viewer = self.composing_viewer()
        window = viewer.canvas.native.window.return_value
        clock = [100.0]
        with mock.patch.object(print_command, "time", SimpleNamespace(perf_counter=lambda: clock[0])):
            events = print_command._CaptureEvents(viewer, [])
            for _ in range(20):
                self.compose_for(events, clock, 0.03)
            events.close()
        window.setUpdatesEnabled.assert_not_called()

    def test_a_canvas_without_composing_signals_is_left_to_the_event_loop_alone(self):
        events = print_command._CaptureEvents(zoom_viewer(), [])
        events.release()
        events.reclaim()
        events.close()

    def scene_viewer(self):
        """A zoom_viewer() whose canvas holds a scene: the edge layer and the node markers the capture watches."""
        from vispy import scene
        from desktop.Viewer_Edges import ViewerEdges

        root = scene.Node()
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        layer = ViewerEdges(root, np.array([[0, 1], [1, 2]], dtype=np.int32), np.array([0.5, 0.9]), positions,
                            (0, 0, 0, 0.5), 1.0, use_position_texture=False)
        markers = scene.visuals.Markers(parent=root)
        viewer = zoom_viewer()
        viewer.canvas.scene = root
        return viewer, layer, markers

    def test_an_edge_layer_that_only_repaints_is_no_change_and_one_that_draws_other_edges_is(self):
        # After a pan with fewer edges drawn the layer draws every edge again, and updates its visual to say so:
        # the picture it draws is the same, which its generation, bumped by every real change, tells.
        viewer, layer, _markers = self.scene_viewer()
        events = print_command._CaptureEvents(viewer, [])
        self.addCleanup(events.close)

        events.release()
        layer.visual.update()
        events.reclaim()  # a repaint

        events.release()
        layer.visible = not layer.visible  # what draws changed
        with self.assertRaises(print_command._CaptureInterrupted):
            events.reclaim()

    def test_that_holds_with_the_camera_moved_too(self):
        viewer, layer, _markers = self.scene_viewer()
        events = print_command._CaptureEvents(viewer, [])
        self.addCleanup(events.close)

        events.release()
        viewer.view.camera.rect = SimpleNamespace(pos=(10.0, -12.5), width=100.0, height=75.0)
        layer.visual.update()
        events.reclaim()
        events.release()
        viewer.view.camera.rect = SimpleNamespace(pos=(20.0, -12.5), width=100.0, height=75.0)
        layer.visible = not layer.visible
        with self.assertRaises(print_command._CaptureInterrupted):
            events.reclaim()

    def test_a_node_marker_update_is_still_a_change_unless_the_camera_moved(self):
        viewer, _layer, markers = self.scene_viewer()
        events = print_command._CaptureEvents(viewer, [])
        self.addCleanup(events.close)

        events.release()
        markers.update()
        with self.assertRaises(print_command._CaptureInterrupted):
            events.reclaim()

        events.release()
        viewer.view.camera.rect = SimpleNamespace(pos=(10.0, -12.5), width=100.0, height=75.0)
        markers.update()
        events.reclaim()

    def test_it_closed_between_tiles_leaves_no_job_and_no_report(self):
        viewer = zoom_viewer()
        steps = self.steps(viewer, ["fig1", "full"])
        next(steps)
        next(steps)

        steps.close()

        viewer.canvas.update.assert_called()
        self.save.assert_not_called()
        self.assertEqual(viewer.background_job_scheduler.jobs, [])
        self.engine.command_failed.assert_not_called()
        self.engine.command_succeeded.assert_not_called()
        # The live view is as it was.
        self.assertEqual(viewer.view.camera.rect.width, 100.0)
        self.assertTrue(viewer.zoom_text.visible)


class PrintInTheCommandQueueTests(unittest.TestCase):
    """The real command queue steps a print's tiles in slices: the event loop, and with it the Viewer's input, runs between them."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        import Command_Engine as real_engine
        from Command_Queue import CommandQueue
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.log = []
        self.engine = print_command.Command_Engine
        for reporter in (self.engine.command_artifact, self.engine.command_succeeded, self.engine.command_failed):
            reporter.reset_mock()
        # The queue records the outcome the real engine is told of.
        self.engine.command_succeeded.side_effect = lambda viewer, message=None: real_engine._record_outcome("succeeded")
        self.addCleanup(setattr, self.engine.command_succeeded, "side_effect", None)

        def render(viewer, is_transparent):
            self.log.append("tile")
            time.sleep(0.01)
            return white_renders(viewer, is_transparent, TILE_SHAPE)

        for patch in (mock.patch.object(print_command, "PRINT_DIRECTORY", os.path.join(root.name, "Saved_Images")),
                      mock.patch.object(print_command, "_render_tile", side_effect=render),
                      mock.patch.object(print_command, "_write_png"),
                      redirect_stdout(io.StringIO())):
            patch.__enter__()
            self.addCleanup(patch.__exit__, None, None, None)
        class QueuedViewer(SimpleNamespace):
            """The queue keeps a weak reference to its viewer, which a SimpleNamespace itself cannot be given."""

        self.viewer = QueuedViewer(**vars(zoom_viewer()))
        self.queue = CommandQueue(self.viewer)
        self.viewer.command_queue = self.queue
        self.addCleanup(self.queue.shutdown)

    def pump_until(self, condition, timeout=20):
        deadline = time.monotonic() + timeout
        while not condition():
            self.assertLess(time.monotonic(), deadline, "the command queue did not get there")
            self.app.processEvents()
            time.sleep(0.001)

    def test_the_event_loop_runs_between_the_tiles_and_the_picture_is_the_same(self):
        from PySide6 import QtCore
        timer = QtCore.QTimer()
        timer.setInterval(1)
        timer.timeout.connect(lambda: self.log.append("tick"))
        timer.start()
        self.addCleanup(timer.stop)
        arguments = ["fig1", "full", "zoom", "2000"]  # 3 x 3 tiles of 10 ms each, in slices of 30 ms

        job = self.queue.dispatch(print_command, "print", arguments, "print " + " ".join(arguments))
        self.assertIsNotNone(job)
        self.pump_until(lambda: not self.queue.pending())
        timer.stop()

        self.assertEqual(self.log.count("tile"), 10)  # the render that measures the canvas, and nine tiles
        first, last = self.log.index("tile"), len(self.log) - 1 - self.log[::-1].index("tile")
        self.assertGreaterEqual(self.log[first:last].count("tick"), 2)
        self.assertEqual(self.queue.finished[-1].status, "succeeded")
        self.assertEqual(len(self.viewer.background_job_scheduler.jobs), 1)
        saved = print_command._write_png.call_args.args[1]
        self.assertEqual(saved.shape, (960, 1760, 4))

    def test_a_command_given_meanwhile_waits_for_the_picture(self):
        import Command_Engine as real_engine
        order = []

        def later_start(viewer, plan):
            order.append(self.log.count("tile"))
            real_engine._record_outcome("succeeded")

        arguments = ["fig1", "full", "zoom", "2000"]
        self.queue.dispatch(print_command, "print", arguments, "print fig1 full zoom 2000")
        later = SimpleNamespace(plan=lambda viewer, args: SimpleNamespace(cooperative=False, heavy=False),
                                start=later_start)
        self.queue.dispatch(later, "later", [], "later")
        self.assertEqual([job.name for job in self.queue.pending()], ["print", "later"])
        self.pump_until(lambda: not self.queue.pending())
        # `later` took its turn when every tile had been drawn.
        self.assertEqual(order, [10])

    def test_undo_stops_a_print_between_tiles_and_nothing_is_saved(self):
        arguments = ["fig1", "full", "zoom", "2000"]
        job = self.queue.dispatch(print_command, "print", arguments, "print fig1 full zoom 2000")
        self.pump_until(lambda: self.log.count("tile") >= 3)
        self.assertTrue(job.cooperation is not None)

        self.queue.take_back(1)

        self.assertEqual(self.queue.pending(), [])
        self.assertEqual(job.status, "cancelled")
        drawn = self.log.count("tile")
        for _ in range(20):
            self.app.processEvents()
        self.assertEqual(self.log.count("tile"), drawn)
        print_command._write_png.assert_not_called()
        self.assertEqual(self.viewer.background_job_scheduler.jobs, [])
        self.assertEqual(self.viewer.view.camera.rect.width, 100.0)


if __name__ == "__main__":
    unittest.main()
