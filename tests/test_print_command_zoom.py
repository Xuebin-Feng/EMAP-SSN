"""Print command option validation, naming, and zoom tests (src/commands/print.py):
pre-write checks, automatic timestamped names without overwrite, `zoom N` scaling
for full PNG and SVG captures, and single-word filename / extension handling."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.print_command_fixtures import (
    FixedClock,
    TILE_SHAPE,
    elements,
    export_svg,
    make_viewer,
    number,
    print_command,
    setUpModule,
    square_network,
    tearDownModule,
    view_at,
    white_renders,
    zoom_viewer,
)


class PrintCommandChecksTests(unittest.TestCase):
    """What `print` refuses, and what it leaves untouched, before it writes."""

    def run_print(self, arguments, visible=True):
        viewer = make_viewer()
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
        for name, value in vars(make_viewer()).items():
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

    FixedClock = FixedClock

    def print_in_this_second(self, save_dir, arguments, svg_viewer=False):
        viewer = square_network() if svg_viewer else make_viewer()
        if svg_viewer:
            for name, value in vars(make_viewer()).items():
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
        for name, value in vars(make_viewer()).items():
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


if __name__ == "__main__":
    unittest.main()
