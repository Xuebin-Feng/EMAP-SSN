# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's tools over the canvas (esmfold_tools.js) and its Save an image dialog (esmfold_image.js).

Zoom to the pinned residue, Measure (two clicks and the distance between the
C-alpha atoms is drawn and said) and Labels (a click names a residue): while one
is on it takes the canvas's clicks, and Escape leaves it before it lets go of a
pin. Like the other page tests these open esmfold.html in Qt WebEngine over the
real server and Mol*, with Mol*'s own click behavior fed the loci a click on a
residue gives; the picture itself is checked over a real canvas in
tests/test_esmfold_shell_webgl.py.
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.test_esmfold_colors import A, B
from tests.test_esmfold_residue import ResidueTestCase


def helix_distance(first, second):
    """The C-alpha distance between residues `first` and `second` (numbers) of the fixture's first structure."""
    def point(number):
        index = number - 1
        angle = math.radians(100 * index)
        return (2.3 * math.cos(angle), 2.3 * math.sin(angle), 1.5 * index)
    return math.dist(point(first), point(second))


class ToolTestCase(ResidueTestCase):
    def two(self):
        self.write_with(A, "information_content")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        self.settle(200)

    def toast_text(self):
        return self.script("return document.getElementById('toast').querySelector('span').textContent")

    def tools(self):
        return self.script("""
            const state = plugin.managers.structure.measurement.state;
            return {measure: document.getElementById('t-measure').getAttribute('aria-pressed'),
                    label: document.getElementById('t-label').getAttribute('aria-pressed'), mode: toolMode,
                    distances: state.distances.length, labels: state.labels.length,
                    clear: document.getElementById('t-clear').hidden, zoom: document.getElementById('t-zoom').disabled,
                    pinned: pinnedResidue ? pinnedResidue.number : null, nodes: focusNodes().length,
                    marked: plugin.managers.structure.selection.stats.elementCount, cells: plugin.state.data.cells.size};
            """)

    def escape(self):
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.settle(400)


class MeasureAndLabelTests(ToolTestCase):
    def test_two_clicks_measure_the_distance_between_two_residues_and_clear_takes_it_away(self):
        self.two()
        base = self.tools()
        self.assertEqual((base["measure"], base["clear"], base["zoom"]), ("false", True, True))
        self.click("#t-measure", 200)
        self.assertEqual(self.tools()["measure"], "true")
        self.assertEqual(self.toast_text(), "Click two residues to measure the distance between them.")
        self.script("await clickOn(0, 3); return 1")
        self.settle(300)
        first = self.tools()
        self.assertEqual(self.toast_text(), "Ala 4: now click the second residue.")
        self.assertGreater(first["marked"], 0)
        self.assertIsNone(first["pinned"])
        self.script("await clickOn(0, 20); return 1")
        self.settle(800)
        done = self.tools()
        self.assertEqual(done["distances"], 1)
        self.assertEqual(done["marked"], 0)
        # Neither click pinned anything, and Mol*'s own focus on the clicks is gone.
        self.assertEqual((done["pinned"], done["nodes"]), (None, 0))
        text = self.toast_text()
        self.assertRegex(text, r"^Ala 4 to Arg 21: \d+\.\d Å$")
        self.assertAlmostEqual(float(text.split(": ")[1].split(" ")[0]), helix_distance(4, 21), delta=0.06)
        self.assertFalse(done["clear"])
        self.click("#t-clear", 600)
        cleared = self.tools()
        self.assertEqual((cleared["distances"], cleared["labels"], cleared["clear"]), (0, 0, True))
        self.assertEqual(cleared["cells"], base["cells"])
        self.assertEqual(self.toast_text(), "Measurements and labels cleared.")
        self.assertEqual(self.console_errors(), [])

    def test_the_same_residue_twice_measures_nothing(self):
        self.two()
        self.click("#t-measure", 200)
        self.script("await clickOn(0, 3); return 1")
        self.settle(300)
        self.script("await clickOn(0, 3); return 1")
        self.settle(500)
        state = self.tools()
        self.assertEqual((state["distances"], state["marked"]), (0, 0))
        self.assertEqual(self.toast_text(), "Click two residues to measure the distance between them.")

    def test_a_click_while_labels_are_on_names_the_residue_and_pins_nothing(self):
        self.two()
        self.click("#t-label", 200)
        state = self.tools()
        self.assertEqual((state["label"], state["measure"], state["mode"]), ("true", "false", "label"))
        self.script("await clickOn(0, 5); return 1")
        self.settle(800)
        labeled = self.tools()
        self.assertEqual((labeled["labels"], labeled["pinned"], labeled["nodes"]), (1, None, 0))
        self.assertEqual(self.toast_text(), "Labeled Ile 6.")
        self.assertFalse(labeled["clear"])
        self.click("#t-clear", 600)
        self.assertEqual(self.tools()["labels"], 0)
        self.assertEqual(self.console_errors(), [])

    def test_one_tool_at_a_time_and_the_same_button_again_leaves_it(self):
        self.two()
        self.click("#t-measure", 200)
        self.click("#t-label", 200)
        self.assertEqual((self.tools()["mode"], self.tools()["measure"]), ("label", "false"))
        self.click("#t-label", 200)
        self.assertEqual((self.tools()["mode"], self.tools()["label"]), (None, "false"))
        # With no tool on, a click pins, as it did.
        self.script("await clickOn(0, 5); return 1")
        self.settle(300)
        self.assertEqual(self.tools()["pinned"], 6)

    def test_escape_leaves_the_tool_first_and_then_lets_go_of_the_pin(self):
        self.two()
        self.script("await clickOn(0, 5); return 1")
        self.settle(300)
        self.click("#t-measure", 200)
        self.assertEqual(self.tools()["measure"], "true")
        self.escape()
        after = self.tools()
        self.assertEqual((after["measure"], after["mode"], after["pinned"]), ("false", None, 6))
        self.escape()
        self.assertIsNone(self.tools()["pinned"])


class ZoomTests(ToolTestCase):
    def test_zoom_comes_to_the_pinned_residue_and_is_off_without_one(self):
        self.two()
        self.script("""
            window.__focused = 0;
            const camera = plugin.managers.camera;
            const focus = camera.focusLoci.bind(camera);
            camera.focusLoci = (...args) => { window.__focused++; return focus(...args); };
            return 1;
            """)
        self.assertTrue(self.tools()["zoom"])
        self.script("await clickOn(0, 5); return 1")
        self.settle(300)
        before = self.script("return window.__focused")
        self.assertFalse(self.tools()["zoom"])
        self.click("#t-zoom", 200)
        self.assertEqual(self.script("return window.__focused"), before + 1)
        self.escape()
        self.assertTrue(self.tools()["zoom"])


class ImageDialogTests(ToolTestCase):
    def test_the_dialog_opens_on_its_first_option_closes_with_escape_and_cancel_and_gives_the_focus_back(self):
        self.two()
        # (Without WebGL the page hides the tools; the toolbar is shown here to give the button the focus.)
        self.script("document.body.dataset.view = 'ready'; document.getElementById('t-image').focus(); return 1")
        self.click("#t-image", 200)
        opened = self.script("""
            return {open: document.getElementById('image-modal').classList.contains('open'), focus: document.activeElement.id,
                    legend: document.getElementById('image-legend').checked, transparent: document.getElementById('image-transparent').checked,
                    size: document.getElementById('image-size').value,
                    sizes: [...document.getElementById('image-size').options].map(o => [o.value, o.textContent]),
                    title: document.getElementById('image-title').textContent};
            """)
        self.assertEqual(opened["open"], True)
        self.assertEqual(opened["focus"], "image-legend")
        self.assertEqual((opened["legend"], opened["transparent"], opened["size"]), (True, False, "1"))
        self.assertEqual(opened["sizes"], [["1", "Window size"], ["2", "Twice the window size"], ["4", "Four times the window size"]])
        self.assertEqual(opened["title"], "Save an image")
        self.escape()
        closed = self.script("return {open: document.getElementById('image-modal').classList.contains('open'), focus: document.activeElement.id}")
        self.assertEqual(closed, {"open": False, "focus": "t-image"})
        self.click("#t-image", 200)
        self.click("#image-cancel", 200)
        self.assertFalse(self.script("return document.getElementById('image-modal').classList.contains('open')"))
        self.click("#t-image", 200)
        self.script("document.getElementById('image-modal').click(); return 1")
        self.assertFalse(self.script("return document.getElementById('image-modal').classList.contains('open')"))

    def test_with_no_canvas_saving_says_it_could_not(self):
        self.two()
        self.click("#t-image", 200)
        self.click("#image-save", 800)
        self.assertTrue(self.toast_text().startswith("Could not do that: "), self.toast_text())
        self.assertFalse(self.script("return document.getElementById('image-save').disabled"))


if __name__ == "__main__":
    unittest.main()
