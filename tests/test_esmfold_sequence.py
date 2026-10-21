# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's sequence strip and its keyboard (esmfold_sequence.js).

The strip draws the residues of one chain in blocks of ten, each with a bar in
the color it has in 3D; pointing at one lights it and fills the Residue card, a
click or Enter pins it, and the arrow keys move through the residues on the
strip and, with the canvas focused, pin the next or previous one. The tests open
the page in Qt WebEngine over the real server and Mol*, and read its DOM and
Mol*'s state.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests import esmfold_layer_fixtures as fixtures
from tests.test_esmfold_colors import A, B
from tests.test_esmfold_residue import ResidueTestCase

FIRST = fixtures.FIRST_SEQUENCE
SECOND = fixtures.SECOND_SEQUENCE


class SequenceTestCase(ResidueTestCase):
    def two(self):
        self.write_with(A, "information_content", "same_residue_pct")
        self.write_with(B, "information_content", "same_residue_pct", sequence=SECOND, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        self.settle(300)

    def strip(self):
        return self.script("""
            const body = document.getElementById('seq-body');
            const buttons = [...body.querySelectorAll('.seq-res')];
            return {letters: buttons.map(b => b.textContent).join(''),
                    numbers: [...body.querySelectorAll('.seq-num')].map(n => n.textContent),
                    colors: buttons.map(b => b.style.getPropertyValue('--rc')),
                    stops: buttons.filter(b => b.tabIndex === 0).map(b => b.getAttribute('aria-label')),
                    pinned: buttons.filter(b => b.classList.contains('pinned')).map(b => b.getAttribute('aria-label')),
                    hover: buttons.filter(b => b.classList.contains('hover')).map(b => b.getAttribute('aria-label')),
                    focus: document.activeElement.classList.contains('seq-res') ? document.activeElement.getAttribute('aria-label') : null,
                    label: body.getAttribute('aria-label'), empty: document.getElementById('seqbar').dataset.empty,
                    select: {hidden: document.getElementById('seq-structure').hidden, value: document.getElementById('seq-structure').value,
                             options: [...document.getElementById('seq-structure').options].map(o => o.textContent)},
                    live: document.getElementById('live').textContent};
            """)

    def key(self, selector, key):
        self.script("""
            const target = %s;
            target.dispatchEvent(new KeyboardEvent('keydown', {key: %s, bubbles: true, cancelable: true}));
            return 1;
            """ % (selector, json.dumps(key)))
        self.settle(250)


class StripTests(SequenceTestCase):
    def test_the_strip_has_the_residues_in_blocks_of_ten_each_with_the_color_it_has_in_3d(self):
        self.two()
        strip = self.strip()
        self.assertEqual(strip["empty"], "0")
        self.assertEqual(strip["letters"], FIRST)
        self.assertEqual(strip["numbers"], ["1", "11", "21", "31"])
        self.assertEqual(strip["label"], "Sequence of 11, chain A")
        # What Mol* draws each residue in: pLDDT here.
        plddt = self.script("return colors(0, 'plddt-confidence', {})")
        self.assertEqual(strip["colors"], [plddt[str(number)] for number in range(1, 34)])
        # And as the scene's coloring changes, so do the bars.
        self.choose("information_content")
        self.settle(500)
        layered = self.strip()
        rows = self.script("return await (await fetch('/api/esmfold/layers/11.pdb/information_content')).json()")
        by_number = {str(row["auth_seq_id"]): row["color"] for row in rows}
        self.assertEqual(layered["colors"], [by_number[str(number)] for number in range(1, 34)])
        self.choose("builtin:structure")
        self.settle(500)
        own = set(self.strip()["colors"])
        self.assertEqual(len(own), 1)
        self.assertEqual(own, {self.script("return structureColor(loadedStructures()[0].cell.transform.ref)")})
        self.assertEqual(self.console_errors(), [])

    def test_with_two_structures_a_select_chooses_which_one_is_shown(self):
        self.two()
        strip = self.strip()
        self.assertEqual(strip["select"], {"hidden": False, "value": strip["select"]["value"], "options": ["11", "22"]})
        self.script("""
            const select = document.getElementById('seq-structure');
            select.value = select.options[1].value; select.dispatchEvent(new Event('change', {bubbles: true}));
            return 1;
            """)
        self.settle(300)
        second = self.strip()
        self.assertEqual(second["letters"], SECOND)
        self.assertEqual(second["label"], "Sequence of 22, chain A")
        # With one structure there is nothing to choose.
        self.click(self.row(2) + " .srow-main button:last-child", 800)
        one = self.strip()
        self.assertTrue(one["select"]["hidden"])
        self.assertEqual(one["letters"], FIRST)

    def test_it_collapses_and_hides_with_no_structure_and_on_a_phone(self):
        self.two()
        visible = "return getComputedStyle(document.getElementById('seq-body')).display !== 'none'"
        self.assertTrue(self.script(visible))
        self.click("#seq-toggle", 400)
        self.assertFalse(self.script(visible))
        self.assertEqual(self.script("return document.getElementById('seq-toggle').getAttribute('aria-expanded')"), "false")
        self.click("#seq-toggle", 400)
        self.assertTrue(self.script(visible))
        # No structure, no strip.
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='clear']", 100)
        self.click("#confirm-ok", 800)
        self.assertEqual(self.script("return getComputedStyle(document.getElementById('seqbar')).display"), "none")
        self.view.resize(600, 800)
        try:
            self.write_with(A, "information_content")
            self.open_scene(A)
            self.assertEqual(self.script("return getComputedStyle(document.getElementById('seqbar')).display"), "none")
        finally:
            self.view.resize(1300, 800)


class StripResidueTests(SequenceTestCase):
    def test_pointing_at_a_residue_fills_the_card_and_a_click_pins_it(self):
        self.two()
        self.script("""
            const button = document.querySelector('#seq-body .seq-res[aria-label="Ile 6"]');
            button.dispatchEvent(new PointerEvent('pointerover', {bubbles: true}));
            return 1;
            """)
        self.settle(300)
        hovered = self.strip()
        self.assertEqual(hovered["hover"], ["Ile 6"])
        self.assertEqual(self.script("return residueNow().head"), "Ile 6")
        self.script("document.getElementById('seq-body').dispatchEvent(new PointerEvent('pointerleave', {bubbles: false})); return 1")
        self.settle(300)
        self.assertEqual(self.strip()["hover"], [])
        self.assertIn("Hover a residue", self.script("return residueNow().hint"))
        # A click pins it: marked on the strip, marked in Mol*, shown in the card, let go with Escape.
        self.click("#seq-body .seq-res[aria-label=\"Ile 6\"]", 500)
        pinned = self.strip()
        self.assertEqual(pinned["pinned"], ["Ile 6"])
        self.assertEqual(self.script("return pinnedResidue.number"), 6)
        self.assertGreater(self.script("return plugin.managers.structure.selection.stats.elementCount"), 0)
        self.assertEqual(self.script("return residueNow().head"), "Ile 6")
        # No Mol* focus: a pin made on the strip draws nothing around the residue.
        self.assertEqual(self.script("return focusNodes().length"), 0)
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.settle(500)
        self.assertEqual(self.strip()["pinned"], [])
        self.assertEqual(self.console_errors(), [])

    def test_a_residue_pinned_on_the_canvas_of_the_other_structure_moves_the_strip_to_it(self):
        self.two()
        self.assertEqual(self.strip()["letters"], FIRST)
        self.script("await clickOn(1, 5); return 1")
        self.settle(500)
        after = self.strip()
        self.assertEqual(after["letters"], SECOND)
        self.assertEqual(after["pinned"], ["Val 6"])

    def test_the_arrow_keys_move_through_the_strip_with_one_tab_stop_and_enter_pins(self):
        self.two()
        strip = self.strip()
        self.assertEqual(len(strip["stops"]), 1)
        self.script("document.querySelector('#seq-body .seq-res[tabindex=\"0\"]').focus(); return 1")
        self.settle(200)
        self.assertEqual(self.strip()["focus"], "Met 1")
        for key, expected in (("ArrowRight", "Lys 2"), ("ArrowRight", "Thr 3"), ("ArrowDown", "Ser 13"), ("ArrowLeft", "Ile 12"),
                              ("ArrowUp", "Lys 2"), ("End", "Gln 33"), ("ArrowRight", "Gln 33"), ("Home", "Met 1"), ("ArrowUp", "Met 1")):
            self.key("document.activeElement", key)
            state = self.strip()
            self.assertEqual(state["focus"], expected, key)
            self.assertEqual(state["stops"], [expected], key)
        # The residue the focus is on is said, and lit; Enter pins it.
        self.key("document.activeElement", "ArrowRight")
        self.assertEqual(self.strip()["live"], "Residue Lys 2, chain A, pLDDT %s." % f"{fixtures.plddt_values(33)[1]:.1f}")
        self.assertEqual(self.strip()["hover"], ["Lys 2"])
        self.key("document.activeElement", "Enter")
        self.assertEqual(self.strip()["pinned"], ["Lys 2"])
        self.assertEqual(self.script("return pinnedResidue.number"), 2)
        self.assertEqual(self.console_errors(), [])


class CanvasKeyboardTests(SequenceTestCase):
    def test_the_canvas_is_one_tab_stop_whose_arrow_keys_pin_the_next_and_previous_residue(self):
        self.two()
        canvas = self.script("""
            const view = document.getElementById('viewer');
            return {tab: view.tabIndex, role: view.getAttribute('role'), label: view.getAttribute('aria-label')};
            """)
        self.assertEqual((canvas["tab"], canvas["role"]), (0, "application"))
        self.assertIn("Arrow keys step through the residues", canvas["label"])
        pinned = "return pinnedResidue ? pinnedResidue.number : null"
        self.script("document.getElementById('viewer').focus(); return 1")
        for key, expected in (("ArrowRight", 1), ("ArrowRight", 2), ("ArrowDown", 3), ("ArrowLeft", 2), ("End", 33), ("ArrowRight", 33),
                              ("Home", 1), ("ArrowUp", 1)):
            self.key("document.getElementById('viewer')", key)
            self.assertEqual(self.script(pinned), expected, key)
        self.key("document.getElementById('viewer')", "ArrowRight")
        self.assertEqual(self.strip()["live"], "Residue Lys 2, chain A, pLDDT %s." % f"{fixtures.plddt_values(33)[1]:.1f}")
        self.assertEqual(self.strip()["pinned"], ["Lys 2"])
        # A pin made with the keys draws nothing of Mol*'s own around it; Escape lets go.
        self.assertEqual(self.script("return focusNodes().length"), 0)
        self.js("document.getElementById('viewer').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.settle(400)
        self.assertIsNone(self.script(pinned))

    def test_r_resets_the_view_and_f_comes_to_the_pinned_residue(self):
        self.two()
        self.script("""
            window.__calls = [];
            const camera = plugin.managers.camera;
            const reset = camera.reset.bind(camera), focus = camera.focusLoci.bind(camera);
            camera.reset = (...args) => { window.__calls.push('reset'); return reset(...args); };
            camera.focusLoci = (...args) => { window.__calls.push('focus'); return focus(...args); };
            document.getElementById('viewer').focus();
            return 1;
            """)
        self.key("document.getElementById('viewer')", "f")
        self.assertEqual(self.script("return window.__calls"), [])
        self.key("document.getElementById('viewer')", "ArrowRight")
        self.key("document.getElementById('viewer')", "F")
        self.key("document.getElementById('viewer')", "r")
        self.assertEqual(self.script("return window.__calls"), ["focus", "reset"])


class OpenedStripTests(SequenceTestCase):
    def test_a_structure_opened_from_a_file_has_its_b_factor_said_not_a_confidence(self):
        self.write_with(A, "information_content")
        self.open_scene(A)
        text = json.dumps(fixtures.helix_pdb(FIRST, fixtures.plddt_values(33), shift=(1.0, 2.0, 0.0), wobble=0.3))
        self.script("openStructureFiles([new File([%s], 'mine.pdb')]); await structureQueue; return 1" % text)
        self.settle(600)
        self.script("""
            const select = document.getElementById('seq-structure');
            select.value = select.options[1].value; select.dispatchEvent(new Event('change', {bubbles: true}));
            return 1;
            """)
        self.settle(300)
        self.script("document.querySelector('#seq-body .seq-res[tabindex=\"0\"]').focus(); return 1")
        self.key("document.activeElement", "ArrowRight")
        self.assertEqual(self.strip()["live"], "Residue Lys 2, chain A, B-factor %s." % f"{fixtures.plddt_values(33)[1]:.1f}")


if __name__ == "__main__":
    unittest.main()
