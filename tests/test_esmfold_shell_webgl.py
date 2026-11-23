# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page over a real canvas (tests/esmfold_gl_scenarios.py).

The page's states that depend on a canvas, the colour the canvas takes from
each theme, the camera buttons and the Advanced door's round trip need WebGL,
which the suite's own Qt WebEngine does not have (tests/test_esmfold_shell.py
checks the rest of the page without it). They run once in a child process that
asks for the GPU, and are skipped where the machine gives it none.
"""

import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.esmfold_gl_scenarios import RESULT_PREFIX


def run_scenario():
    environment = {key: value for key, value in os.environ.items() if key != "QTWEBENGINE_CHROMIUM_FLAGS"}
    done = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "esmfold_gl_scenarios.py")],
        capture_output=True, text=True, timeout=300, env=environment, cwd=str(ROOT),
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        raise AssertionError(f"the scenario printed no answer (exit {done.returncode}): {done.stderr[-800:]}")
    found = json.loads(lines[-1][len(RESULT_PREFIX):])
    if "error" in found:
        raise AssertionError(f"the scenario failed: {found['error']}")
    return found


def distance(a, b):
    return math.dist(a, b)


def hex_channels(colour):
    return [int(colour[index:index + 2], 16) for index in (1, 3, 5)]


# A pixel read back from the canvas is the colour of its token as the screen shows it: exactly on Windows and
# Linux, within a step of 1 per channel on macOS, whose colour management shifts #fafafa to #f9f9f9.
PIXEL_TOLERANCE = 2 if sys.platform == "darwin" else 0


class ShellWebglTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = run_scenario()
        if not cls.found["webgl"]:
            raise unittest.SkipTest("this machine gave the child process no WebGL")

    def test_the_page_says_empty_without_structures_and_ready_with_them(self):
        self.assertEqual((self.found["viewEmpty"], self.found["viewReady"]), ("empty", "ready"))
        self.assertEqual(self.found["errors"], [])
        self.assertEqual(self.found["errorsReady"], [])

    def assertColourIs(self, pixel, token):
        for read, wanted in zip(hex_channels(pixel), hex_channels(token)):
            self.assertLessEqual(abs(read - wanted), PIXEL_TOLERANCE, (pixel, token))

    def test_the_canvas_takes_the_colour_of_each_theme_and_draws_the_structures_on_it(self):
        found = self.found
        self.assertEqual(found["lightToken"], "#fafafa")
        self.assertColourIs(found["light"]["corner"], found["lightToken"])
        self.assertNotEqual(found["darkToken"], found["lightToken"])
        self.assertColourIs(found["dark"]["corner"], found["darkToken"])
        self.assertEqual(found["darkRenderer"], found["darkToken"])
        for theme in ("light", "dark"):
            self.assertGreater(found[theme]["drawn"], 0.002, theme)

    def test_the_advanced_door_keeps_the_canvas_and_the_picture_through_a_round_trip(self):
        found = self.found
        self.assertTrue(found["advancedSameCanvas"])
        self.assertTrue(found["closedSameCanvas"])
        self.assertGreater(found["advancedDrawn"], 0.002)
        self.assertGreater(found["closedDrawn"], 0.002)

    def test_zoom_comes_to_a_structure_and_reset_view_frames_the_scene_again(self):
        found = self.found
        self.assertGreater(distance(found["cameraMoved"]["target"], found["center"]), 30)
        # Within a quarter of the structure's width (the camera frames its bounding sphere, not its centre).
        self.assertLess(distance(found["cameraZoomed"]["target"], found["center"]), found["cameraZoomed"]["radius"] / 4)
        self.assertLess(distance(found["cameraReset"]["target"], found["cameraFramed"]["target"]), 1)
        self.assertLess(distance(found["cameraReset"]["position"], found["cameraFramed"]["position"]), 3)

    def test_the_axes_gizmo_is_in_the_right_corner_and_the_status_line_stays_clear_of_it(self):
        found = self.found
        self.assertEqual(found["gizmo"], "bottom-right")
        self.assertGreater(found["gizmoRight"], 0.02)
        self.assertLess(found["gizmoLeft"], 0.002)
        toast = found["toast"]
        # Over the middle of the view's bottom edge, neither corner.
        self.assertGreater(toast["left"], 130)
        self.assertLess(toast["right"], toast["stageWidth"] - 130)
        self.assertGreater(toast["top"], toast["stageHeight"] / 2)

    def test_a_narrow_window_opens_the_panel_as_a_strip_and_puts_the_status_line_at_the_top(self):
        self.assertEqual(self.found["narrowPanel"], "1")
        toast = self.found["narrowToast"]
        self.assertLess(toast["bottom"], 80)
        self.assertLess(toast["right"], toast["stageWidth"] - 150)

    def test_at_600_px_the_status_line_and_the_legend_on_the_canvas_do_not_meet(self):
        toast, legend = self.found["narrowToast"], self.found["narrowLegend"]
        self.assertEqual(self.found["narrowShown"], "1")
        apart_in_height = toast["bottom"] <= legend["top"] or legend["bottom"] <= toast["top"]
        apart_in_width = toast["right"] <= legend["left"] or legend["right"] <= toast["left"]
        self.assertTrue(apart_in_height or apart_in_width, (toast, legend))
        # The legend is in the canvas's bottom left, inside it.
        self.assertGreater(legend["top"], legend["stageHeight"] / 2)
        self.assertGreaterEqual(legend["left"], 0)
        self.assertLessEqual(legend["right"], legend["stageWidth"] / 2 + 100)

    def test_save_an_image_draws_the_legend_into_the_picture_at_the_size_asked(self):
        image = self.found["image"]
        self.assertEqual(image["doubleSize"], [2 * image["size"][0], 2 * image["size"][1]])
        # The legend (its box, title, bar and ends) is in the bottom left corner and nowhere else; the rest of
        # the picture is the same as without it, up to the renderer's own noise between two pictures.
        self.assertGreater(image["legendPixels"], 300)
        self.assertLess(image["elsewherePixels"], 4 * self.found["noise"][1] + 150)
        # An opaque picture, and with the background off one that is see-through.
        self.assertEqual(image["alpha"], [255, 255, 0])

    def test_a_run_with_no_structure_yet_shows_the_progress_tile_over_a_real_canvas(self):
        self.assertEqual(self.found["foldingView"], "folding")
        self.assertEqual(self.found["failureShown"], 1)

    def test_the_legend_on_the_canvas_names_the_layer_with_its_range(self):
        self.assertEqual(self.found["layerLegend"], {"shown": "1", "title": "Information Content (%)", "ends": ["0 %", "100 %"]})

    def test_letting_go_of_a_pin_leaves_the_scene_as_it_was_and_so_does_a_session_that_had_one(self):
        found = self.found
        before, after, restored = found["sceneBefore"], found["sceneAfterPin"], found["sceneRestored"]
        # Mol*'s click put the residue in focus with five nodes of its own; Escape took them, and what they drew.
        self.assertEqual(found["pinnedAgain"]["nodes"], 5)
        self.assertEqual((before["nodes"], before["focus"], before["marked"]), (0, False, 0))
        noise = found["noise"][1]
        for name, scene, (drawn, differing) in (("after the pin", after, found["afterPin"]), ("after a restore", restored, found["restored"])):
            self.assertEqual((scene["nodes"], scene["focus"], scene["marked"]), (0, False, 0), name)
            self.assertEqual(scene["cells"], before["cells"], name)
            # The same picture from the same camera: as much drawn, and no more of it changed than between two
            # pictures of one scene (a focus left behind draws sticks and contacts around the residue).
            self.assertLess(abs(drawn - found["drawnBefore"]), 0.08 * found["drawnBefore"], name)
            self.assertLess(differing, 4 * noise + 150, (name, differing, noise))
        self.assertIsNone(found["restoredPinned"])

    def test_the_spot_the_real_pointer_is_moved_to_lies_inside_the_canvas_in_page_pixels_at_any_pixel_ratio(self):
        # The spot is worked out from the camera, which answers in the canvas's own pixels (twice the page's at a ratio of 2); a spot
        # that adds those to the canvas's page position lands outside the canvas, on whatever is below it.
        pointer = self.found["pointer"]
        left, top, right, bottom = pointer["rect"]
        message = (f"the pointer spot {pointer['spot']} is not inside the canvas {pointer['rect']} (page pixels) at a pixel ratio of "
                   f"{pointer['ratio']}, canvas buffer {pointer['buffer']}, page size {pointer['client']}: device pixels were taken for page pixels "
                   f"(the element under it is {pointer['under']})")
        self.assertTrue(left <= pointer["spot"][0] <= right and top <= pointer["spot"][1] <= bottom, message)
        self.assertTrue(pointer["under"].startswith("CANVAS"), message)
        # The canvas's buffer is its page size times the pixel ratio, to a pixel.
        for buffer, client in zip(pointer["buffer"], pointer["client"]):
            self.assertLessEqual(abs(buffer - client * pointer["ratio"]), 1, pointer)

    def test_a_real_pointer_over_the_canvas_shows_the_tip_pins_a_residue_and_escape_lets_go(self):
        found = self.found
        self.assertIsNotNone(found["hover"])
        self.assertEqual(found["hover"][1], "A")
        self.assertTrue(found["tip"]["shown"])
        self.assertIn("pLDDT", found["tip"]["text"])
        self.assertIn("Information Content", found["tip"]["text"])
        box = found["tip"]["box"]
        self.assertGreaterEqual(box["left"], 0)
        self.assertGreaterEqual(box["top"], 0)
        self.assertLessEqual(box["right"], box["stageWidth"])
        self.assertLessEqual(box["bottom"], box["stageHeight"])
        # The click pins the residue under the pointer, Mol* marks it, and the Residue card shows it.
        self.assertEqual(found["pinned"], found["hover"][0])
        self.assertGreater(found["marked"], 0)
        self.assertTrue(found["residueCard"]["head"].endswith(str(found["pinned"])), found["residueCard"])
        self.assertGreaterEqual(found["residueCard"]["rows"], 3)
        # Escape hides the tip first (the pointer is still over the residue) and the pin stays; the next one lets go of it.
        self.assertEqual(found["afterFirstEscape"], {"tip": False, "pinned": found["pinned"]})
        self.assertIsNone(found["pinnedAfterEscape"])
        self.assertEqual(found["markedAfterEscape"], 0)

    def test_a_session_saved_by_an_earlier_mol_star_puts_the_camera_back_with_the_structures_in_view(self):
        # tests/esmfold_saved_sessions.py: sessions of a page with a canvas, which carry the camera, by both routes.
        saved = self.found["savedCamera"]
        self.assertEqual(sorted(saved), ["marked/file", "marked/viewer", "pair/file", "pair/viewer"])
        for name, found in saved.items():
            with self.subTest(scene=name):
                self.assertEqual(found["structures"], 2)
                self.assertLess(distance(found["restored"]["target"], found["saved"]["target"]), 1e-3)
                self.assertLess(distance(found["restored"]["position"], found["saved"]["position"]), 1e-3)
                self.assertAlmostEqual(found["restored"]["radius"], found["saved"]["radius"], places=3)
                self.assertGreater(found["drawn"], 0.002)

    def test_the_canvas_follows_the_panels_handle_is_drawn_again_at_its_new_size_and_is_not_stretched(self):
        steps = {step["label"]: step for step in self.found["panel"]}
        self.assertEqual(list(steps), ["default", "dragging", "let go", "widest", "window of 1100 px", "restored"])
        # A real pointer pulled the edge 200 px to the left, and the keys and the window took it further and back.
        self.assertEqual([round(steps[label]["panel"]) for label in steps], [340, 540, 540, 640, 550, 340])
        first = steps["default"]
        for label, step in steps.items():
            with self.subTest(step=label):
                # The canvas is as big as the stage (inside its 1 px border), and as big in its own pixels as in the page's (what
                # is drawn again, not scaled).
                self.assertLessEqual(abs(step["client"][0] - step["stage"][0]), 2.5)
                self.assertLessEqual(abs(step["client"][1] - step["stage"][1]), 2.5)
                ratio = step["ratio"]
                self.assertLessEqual(abs(step["buffer"][0] - step["client"][0] * ratio), 1)
                self.assertLessEqual(abs(step["buffer"][1] - step["client"][1] * ratio), 1)
                # The camera draws into the same shape: no stretch in the picture the canvas shows.
                self.assertAlmostEqual(step["viewport"][0] / step["viewport"][1], step["client"][0] / step["client"][1], delta=0.01)
                self.assertAlmostEqual(step["shot"][0] / step["shot"][1], step["client"][0] / step["client"][1], delta=0.01)
                self.assertGreater(step["drawn"], 0.002)
                # Two residues are as far apart on the screen as they were: the structure is drawn at its size, not scaled with the canvas.
                self.assertLess(abs(step["apart"][0] - first["apart"][0]), 2.5)
                self.assertLess(abs(step["apart"][1] - first["apart"][1]), 2.5)
        # The stage gave up as much as the panel took.
        self.assertAlmostEqual(first["stage"][0] - steps["let go"]["stage"][0], 200, delta=1)
        self.assertAlmostEqual(first["stage"][0] - steps["widest"]["stage"][0], 300, delta=1)
        self.assertAlmostEqual(steps["restored"]["stage"][0], first["stage"][0], delta=1)
        # Mol* was told too (it also watches its container), at the end of the drag and for each key.
        self.assertGreater(steps["restored"]["calls"], steps["default"]["calls"])

    def test_the_page_draws_its_canvas_in_simplified_chinese_too(self):
        self.assertEqual((self.found["chineseHeader"], self.found["chineseView"]), ("EMAP-SSN 结构视图", "ready"))


if __name__ == "__main__":
    unittest.main()
