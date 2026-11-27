"""Print command PNG capture tests (src/commands/print.py): HUD hiding during
capture, PNG margin trimming, multi-tile PNG mosaic assembly, PNG encoding and
partial-file cleanup, and output-name confinement."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.held_file_fixtures import held_file_error, hold_without_delete_sharing, transient_hold
from tests.print_command_fixtures import (
    InlineScheduler,
    export_svg,
    make_viewer,
    print_command,
    raw_renders,
    setUpModule,
    tearDownModule,
)
from utilities import Job_Cancellation


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
    make_viewer = staticmethod(make_viewer)

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
        viewer = make_viewer()
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


if __name__ == "__main__":
    unittest.main()
