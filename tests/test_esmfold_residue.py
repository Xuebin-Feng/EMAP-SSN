# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's hover tooltip, pinned residue and Residue card (esmfold_residue.js).

What Mol* says the pointer is over is read for the residue under it, and a
click pins one. The offscreen engine has no WebGL, so there is no pointer
over a canvas here: the tests give Mol*'s own hover and click behaviors the
loci the representation would give for a residue, which is what the canvas
does, and read the page's DOM; tests/test_esmfold_shell_webgl.py moves a real
pointer over a real canvas.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests import esmfold_layer_fixtures as fixtures
from tests.test_esmfold_colors import A, B, C, ColorTestCase

# Mol*'s pointer events: the loci of the residue `group` (from 0) of a structure, as the representation gives them.
POINTER = """
window.lociOf = function (index, group) {
    const repr = loadedStructures()[index].components[0].representations[0].cell.obj.data.repr;
    return {loci: repr.getLoci({objectId: repr.renderObjects[0].id, instanceId: 0, groupId: group}), repr};
};
window.hoverOn = async function (index, group) {
    plugin.behaviors.interaction.hover.next({current: lociOf(index, group), modifiers: {}, buttons: 0, button: 0});
    await new Promise(resolve => setTimeout(resolve, 250));
};
window.hoverOff = async function () {
    plugin.behaviors.interaction.hover.next({current: {loci: {kind: 'empty-loci'}}, modifiers: {}, buttons: 0, button: 0});
    await new Promise(resolve => setTimeout(resolve, 250));
};
window.clickOn = async function (index, group) {
    plugin.behaviors.interaction.click.next({current: index === null ? {loci: {kind: 'empty-loci'}} : lociOf(index, group), modifiers: {}, buttons: 1, button: 1});
    await new Promise(resolve => setTimeout(resolve, 350));
};
window.tipNow = function () {
    const tip = document.getElementById('tip');
    return {shown: tip.classList.contains('show'), head: (tip.querySelector('.h b') || {}).textContent,
            where: (tip.querySelector('.h span') || {}).textContent,
            rows: [...tip.querySelectorAll('tr')].map(row => [...row.children].map(c => c.textContent)),
            pinned: (tip.querySelector('.pinned-note') || {}).textContent || null};
};
window.residueNow = function () {
    const body = document.getElementById('residue-body');
    return {head: (body.querySelector('.res-head div') || {}).firstChild?.textContent ?? null,
            where: (body.querySelector('.res-head div span') || {}).textContent ?? null,
            secondary: (body.querySelector('.res-head > span') || {}).textContent ?? null,
            header: [...body.querySelectorAll('th')].map(c => c.textContent),
            rows: [...body.querySelectorAll('tr')].slice(1).map(row => [...row.children].map(c => c.textContent)),
            buttons: [...body.querySelectorAll('.res-foot button')].map(b => b.textContent),
            hint: (body.querySelector('.hint') || {}).textContent ?? null};
};
"""


class ResidueTestCase(ColorTestCase):
    def open_scene(self, *names):
        super().open_scene(*names)
        self.js(POINTER)

    def plddt(self, group, offset=0):
        return f"{fixtures.plddt_values(33, offset)[group]:.1f}"


class TooltipTests(ResidueTestCase):
    def test_hovering_a_residue_shows_its_name_its_plddt_and_the_layer_it_is_colored_by(self):
        self.write_with(A, "information_content", "same_residue_pct")
        self.write_with(B, "information_content", "same_residue_pct", sequence=fixtures.SECOND_SEQUENCE, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        self.script("await hoverOn(0, 5); return 1")
        tip = self.script("return tipNow()")
        self.assertTrue(tip["shown"])
        self.assertEqual((tip["head"], tip["where"]), ("Ile 6", "chain A · 11"))
        value = float(self.plddt(5))
        band = "Very high" if value > 90 else "Confident" if value > 70 else "Low" if value > 50 else "Very low"
        self.assertEqual(tip["rows"], [["pLDDT", f"{value:.1f} · {band}"]])
        self.assertIsNone(tip["pinned"])
        # Colored by a layer, the hovered residue's value of it is in the tip, as the Viewer words it.
        self.choose("information_content")
        self.script("await hoverOn(0, 5); return 1")
        rows = self.script("return tipNow().rows")
        self.assertEqual(rows[1], ["Information Content", f"{fixtures.information_content(33)[5]:g} %"])
        # A second structure's residue is named with its own residue and structure.
        self.script("await hoverOn(1, 5); return 1")
        second = self.script("return tipNow()")
        self.assertEqual(second["head"], "Val 6")
        self.assertEqual(second["where"], "chain A · 22")
        # Over nothing the tip goes.
        self.script("await hoverOff(); return 1")
        self.assertFalse(self.script("return tipNow().shown"))
        self.assertEqual(self.console_errors(), [])

    def test_a_residue_with_no_value_for_the_layer_says_no_data(self):
        self.write_with(A, "same_residue_pct")
        self.open_scene(A)
        self.choose("same_residue_pct")
        # Residue 4 (group 3) is the one the fixture gives no agreement for.
        self.script("await hoverOn(0, 3); return 1")
        self.assertEqual(self.script("return tipNow().rows[1]"), ["Residue Agreement", "No data"])

    def test_the_tip_follows_the_pointer_and_stays_inside_the_stage(self):
        self.write_with(A, "information_content")
        self.open_scene(A)
        self.script("""
            const viewer = document.getElementById('viewer');
            const stage = document.getElementById('stage').getBoundingClientRect();
            viewer.dispatchEvent(new PointerEvent('pointermove', {clientX: stage.left + 200, clientY: stage.top + 150, bubbles: true}));
            await hoverOn(0, 5);
            const tip = document.getElementById('tip');
            window.__first = [parseFloat(tip.style.left), parseFloat(tip.style.top)];
            viewer.dispatchEvent(new PointerEvent('pointermove', {clientX: stage.right - 4, clientY: stage.bottom - 4, bubbles: true}));
            window.__corner = [parseFloat(tip.style.left), parseFloat(tip.style.top), tip.offsetWidth, tip.offsetHeight, stage.width, stage.height];
            return 1;
            """)
        first = self.script("return window.__first")
        self.assertEqual(first, [216, 166])
        left, top, width, height, stage_width, stage_height = self.script("return window.__corner")
        self.assertLessEqual(left + width, stage_width)
        self.assertLessEqual(top + height, stage_height)
        # Leaving the view hides it.
        self.script("document.getElementById('viewer').dispatchEvent(new PointerEvent('pointerleave', {bubbles: false})); return 1")
        self.assertFalse(self.script("return tipNow().shown"))


class PinTests(ResidueTestCase):
    def test_a_click_pins_a_residue_and_the_card_shows_its_equivalent_in_every_structure(self):
        self.write_with(A, "information_content", "same_residue_pct")
        self.write_with(B, "information_content", "same_residue_pct", sequence=fixtures.SECOND_SEQUENCE, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.write_with(C, shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.open_scene(A, B, C)
        empty = self.script("return residueNow()")
        self.assertIn("Hover a residue", empty["hint"])
        self.script("await clickOn(0, 5); return 1")
        card = self.script("return residueNow()")
        self.assertEqual((card["head"], card["where"]), ("Ile 6", "chain A · 11"))
        self.assertEqual(card["header"], ["Structure", "pLDDT", "Cα–Cα (Å)"])
        names = [row[0] for row in card["rows"]]
        self.assertEqual(names, ["11", "22", "33"])
        # The residue each structure has at the same place of the alignment: the pLDDT of that one.
        self.assertEqual(card["rows"][0][1:], [self.plddt(5), "–"])
        for row in card["rows"][1:]:
            self.assertEqual(row[1], self.plddt(5))
            self.assertLess(float(row[2]), 3.0)
        self.assertEqual(card["buttons"], ["Zoom to", "Clear"])
        # Mol*'s selection marks the pinned residue.
        self.assertGreater(self.script("return plugin.managers.structure.selection.stats.elementCount"), 0)
        # Colored by a layer, a column of its values joins, and a structure with none says so.
        self.choose("information_content")
        self.settle(400)
        card = self.script("return residueNow()")
        self.assertEqual(card["header"], ["Structure", "pLDDT", "Information Content", "Cα–Cα (Å)"])
        self.assertEqual([row[2] for row in card["rows"]], [f"{fixtures.information_content(33)[5]:g} %"] * 2 + ["No data"])
        # Hovering another residue while one is pinned leaves the card on the pinned one and shows the pin in the tip.
        self.script("await hoverOn(0, 5); return 1")
        self.assertEqual(self.script("return tipNow().pinned"), "Pinned. Escape clears it.")
        self.script("await hoverOn(0, 9); return 1")
        self.assertEqual(self.script("return residueNow().head"), "Ile 6")
        self.assertIsNone(self.script("return tipNow().pinned"))
        # Clear lets go, and the mark with it.
        self.script("document.querySelector('#residue-body .res-foot button:last-child').click(); await new Promise(r => setTimeout(r, 300)); return 1")
        self.assertIsNone(self.script("return pinnedResidue"))
        self.assertEqual(self.script("return plugin.managers.structure.selection.stats.elementCount"), 0)
        self.assertEqual(self.console_errors(), [])

    def test_the_card_follows_the_hover_until_something_is_pinned(self):
        self.write_with(A, "information_content")
        self.open_scene(A)
        self.script("await hoverOn(0, 7); return 1")
        self.assertEqual(self.script("return residueNow().head"), "Lys 8")
        self.script("await hoverOff(); return 1")
        self.assertIn("Hover a residue", self.script("return residueNow().hint"))

    def test_a_click_on_nothing_and_escape_and_removing_the_structure_let_go_of_the_pin(self):
        self.write_with(A, "information_content")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        pinned = "return pinnedResidue ? pinnedResidue.number : null"
        self.script("await clickOn(1, 5); return 1")
        self.assertEqual(self.script(pinned), 6)
        self.script("await clickOn(null); return 1")
        self.assertIsNone(self.script(pinned))
        # Escape: a dialog first, then the pin, then Advanced.
        self.script("await clickOn(1, 5); return 1")
        self.js("document.getElementById('help-btn').click()")
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.assertEqual(self.script(pinned), 6)
        self.script("setAdvanced(true); return 1")
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.assertIsNone(self.script(pinned))
        self.assertEqual(self.script("return document.body.dataset.adv"), "1")
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.assertEqual(self.script("return document.body.dataset.adv"), "0")
        # Removing the structure it is on takes the pin with it.
        self.script("await clickOn(1, 5); return 1")
        self.click(self.row(2) + " .srow-main button:last-child", 800)
        self.assertIsNone(self.script(pinned))
        self.assertIn("Hover a residue", self.script("return residueNow().hint"))
        self.assertEqual(self.console_errors(), [])


if __name__ == "__main__":
    unittest.main()
