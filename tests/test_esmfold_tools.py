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

import json
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
                    clear: document.getElementById('t-clear').hidden, zoom: document.getElementById('t-zoom').getAttribute('aria-disabled') === 'true',
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


class ToolbarTests(ToolTestCase):
    """The toolbar over the canvas is one Tab stop, and the arrow keys move along it."""

    STOPS = "return [...document.querySelectorAll('.view-toolbar button')].filter(node => node.tabIndex === 0).map(node => node.id)"

    def two(self):
        super().two()
        # The offscreen engine has no WebGL, so the page would show its "webgl" card and none of the view's tools.
        self.js("document.body.dataset.view = 'ready'")
        self.settle(200)

    def press(self, key):
        self.js("document.activeElement.dispatchEvent(new KeyboardEvent('keydown', {key: %s, bubbles: true, cancelable: true}))" % json.dumps(key))
        self.settle(50)
        return self.script("return document.activeElement.id")

    def test_it_is_a_toolbar_with_one_tab_stop_that_the_arrow_keys_home_and_end_move(self):
        self.two()
        self.assertEqual(self.script("const bar = document.querySelector('.view-toolbar'); return [bar.getAttribute('role'), bar.getAttribute('aria-label')]"),
                         ["toolbar", "View tools"])
        self.assertEqual(self.script(self.STOPS), ["t-reset"])
        self.script("document.getElementById('t-measure').focus(); return 1")
        self.assertEqual(self.script(self.STOPS), ["t-measure"])
        self.assertEqual(self.press("ArrowRight"), "t-label")
        self.assertEqual(self.press("End"), "t-adv")
        # Round the ends, and Home and End; the stop is where the focus last was.
        self.assertEqual(self.press("ArrowRight"), "t-reset")
        self.assertEqual(self.press("ArrowLeft"), "t-adv")
        self.assertEqual(self.press("Home"), "t-reset")
        self.assertEqual(self.press("ArrowRight"), "t-zoom")
        self.assertEqual(self.script(self.STOPS), ["t-zoom"])
        self.script("document.getElementById('t-image').focus(); document.getElementById('t-image').blur(); return 1")
        self.assertEqual(self.script(self.STOPS), ["t-image"])
        # A button that is not shown is no stop and no step (Clear, which is there once something is measured).
        self.assertTrue(self.script("return document.getElementById('t-clear').hidden"))
        self.script("document.getElementById('t-measure').focus(); return 1")
        self.assertEqual(self.press("ArrowRight"), "t-label")
        self.assertEqual(self.press("ArrowRight"), "t-image")

    def test_the_stop_is_one_that_is_shown_when_the_one_it_was_on_goes(self):
        self.two()
        self.script("document.getElementById('t-measure').focus(); document.getElementById('t-measure').blur(); return 1")
        self.assertEqual(self.script(self.STOPS), ["t-measure"])
        self.js("document.body.dataset.view = 'empty'")
        self.settle(200)
        self.assertEqual(self.script(self.STOPS), ["t-adv"])
        # And it is the one it was when it is shown again.
        self.js("document.body.dataset.view = 'ready'")
        self.settle(200)
        self.assertEqual(self.script(self.STOPS), ["t-measure"])

    def test_zoom_is_aria_disabled_not_disabled_so_it_keeps_the_focus_and_ignores_a_press(self):
        self.two()
        found = self.script("""
            const zoom = document.getElementById('t-zoom');
            window.__zoomed = 0;
            const original = zoomToResidue;
            zoomToResidue = (...args) => { window.__zoomed++; return original(...args); };
            zoom.focus();
            zoom.click();
            return {disabled: zoom.disabled, aria: zoom.getAttribute('aria-disabled'), focus: document.activeElement.id, zoomed: window.__zoomed};
            """)
        self.assertEqual(found, {"disabled": False, "aria": "true", "focus": "t-zoom", "zoomed": 0})
        # Pinned, it is on; it goes off again with the focus still on it.
        self.script("await clickOn(0, 5); return 1")
        self.settle(300)
        self.assertEqual(self.script("return document.getElementById('t-zoom').getAttribute('aria-disabled')"), "false")
        self.js("document.body.dataset.view = 'ready'")
        self.settle(100)
        self.script("document.getElementById('t-zoom').focus(); pinnedResidue = null; refreshTools(); return 1")
        self.assertEqual(self.script("return [document.getElementById('t-zoom').getAttribute('aria-disabled'), document.activeElement.id]"), ["true", "t-zoom"])


class ImageDialogTests(ToolTestCase):
    def test_the_dialog_opens_on_its_first_option_closes_with_escape_and_cancel_and_gives_the_focus_back(self):
        self.two()
        # (Without WebGL the page hides the tools; the toolbar is shown here to give the button the focus.)
        self.script("document.body.dataset.view = 'ready'; document.getElementById('t-image').focus(); return 1")
        self.click("#t-image", 200)
        opened = self.script("""
            return {open: document.getElementById('image-modal').open, focus: document.activeElement.id,
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
        closed = self.script("return {open: document.getElementById('image-modal').open, focus: document.activeElement.id}")
        self.assertEqual(closed, {"open": False, "focus": "t-image"})
        self.click("#t-image", 200)
        self.click("#image-cancel", 200)
        self.assertFalse(self.script("return document.getElementById('image-modal').open"))
        self.click("#t-image", 200)
        self.script("document.getElementById('image-modal').click(); return 1")
        self.assertFalse(self.script("return document.getElementById('image-modal').open"))

    def test_with_no_canvas_saving_says_it_could_not_in_the_dialog_which_stays_open(self):
        self.two()
        self.script("document.body.dataset.view = 'ready'; document.getElementById('t-image').focus(); return 1")
        self.click("#t-image", 200)
        self.script("document.getElementById('image-save').focus(); return 1")
        self.click("#image-save", 800)
        # A message over the canvas would be under the dialog's backdrop: the failure is said in the dialog, by the button it describes.
        found = self.script("""
            const error = document.getElementById('image-error'), save = document.getElementById('image-save');
            return {open: document.getElementById('image-modal').open, text: error.textContent, role: error.getAttribute('role'),
                    shown: getComputedStyle(error).display !== 'none', describes: save.getAttribute('aria-describedby'),
                    disabled: save.disabled, focus: document.activeElement.id, toast: document.getElementById('toast').classList.contains('show')
                        && document.getElementById('toast').textContent.includes('Could not do that')};""")
        self.assertTrue(found["text"].startswith("Could not do that: "), found["text"])
        self.assertEqual({key: found[key] for key in ("open", "role", "shown", "describes", "disabled", "focus", "toast")},
                         {"open": True, "role": "alert", "shown": True, "describes": "image-error", "disabled": False,
                          "focus": "image-save", "toast": False})
        # Closed and opened again, the dialog starts with no message.
        self.escape()
        self.assertEqual(self.script("return document.activeElement.id"), "t-image")
        self.click("#t-image", 200)
        self.assertEqual(self.script("""
            const error = document.getElementById('image-error');
            return [error.textContent, getComputedStyle(error).display];"""), ["", "none"])


class ImageLegendTests(ToolTestCase):
    """The legend drawn into a saved picture is the legend over the canvas: the same band limits and the same "No data" key."""

    DRAW = """
        const canvas = document.createElement('canvas');
        canvas.width = 600; canvas.height = 400;
        const context = canvas.getContext('2d');
        const texts = [];
        const original = context.fillText.bind(context);
        context.fillText = (text, ...rest) => { texts.push(text); return original(text, ...rest); };
        const box = drawLegend(context, 600, 400, 1, await colorModel());
        const compact = document.getElementById('canvas-legend');
        return {box, texts, canvas: [...compact.querySelectorAll('.t, .ticks span, .band span')].map(n => n.firstChild ? n.firstChild.textContent : n.textContent)};"""

    def test_the_picture_has_plddts_band_limits_and_the_partial_no_data_key_as_the_canvas_has_them(self):
        self.write_with(A, "same_residue_pct")
        self.open_scene(A)
        plddt = self.script(self.DRAW)
        self.assertEqual(plddt["texts"], ["pLDDT", "50", "70", "90", "0", "100"])
        self.assertEqual(plddt["canvas"], ["pLDDT", "0", "50", "70", "90", "100"])
        # The bar is below the limits: the legend is taller by a line than it is for a layer's bar.
        self.choose("same_residue_pct")
        layer = self.script(self.DRAW)
        self.assertEqual(layer["texts"], ["Residue Agreement (%)", "0 %", "100 %", "No data"])
        self.assertEqual(layer["canvas"], ["Residue Agreement", "0 %", "100 %", "No data"])
        self.assertEqual(sorted(layer["texts"][1:3]), sorted(layer["canvas"][1:3]))
        self.assertGreater(plddt["box"]["height"], layer["box"]["height"] - 20)
        # Whole structures with no values, and a layer covering all of a structure, have the same key or none, as the canvas does.
        self.choose("builtin:structure")
        self.assertIsNone(self.script("return drawLegend(document.createElement('canvas').getContext('2d'), 600, 400, 1, await colorModel())"))


if __name__ == "__main__":
    unittest.main()
