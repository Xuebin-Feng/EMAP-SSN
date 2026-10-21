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

    def test_the_canvas_takes_the_colour_of_each_theme_and_draws_the_structures_on_it(self):
        found = self.found
        self.assertEqual(found["lightToken"], "#fafafa")
        self.assertEqual(found["light"]["corner"], found["lightToken"])
        self.assertNotEqual(found["darkToken"], found["lightToken"])
        self.assertEqual(found["dark"]["corner"], found["darkToken"])
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

    def test_a_narrow_window_still_opens_the_panel_as_a_strip(self):
        self.assertEqual(self.found["narrowPanel"], "1")

    def test_the_page_draws_its_canvas_in_simplified_chinese_too(self):
        self.assertEqual((self.found["chineseHeader"], self.found["chineseView"]), ("EMAP-SSN 结构视图", "ready"))


if __name__ == "__main__":
    unittest.main()
