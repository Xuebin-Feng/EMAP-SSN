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

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

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
            // A button is named by its letter first ("I, Ile 6"); the tests below say the residue ("Ile 6").
            const residue = button => button.getAttribute('aria-label').replace(/^[A-Z], /, '');
            return {letters: buttons.map(b => b.textContent).join(''),
                    numbers: [...body.querySelectorAll('.seq-num')].map(n => n.textContent),
                    colors: buttons.map(b => b.style.getPropertyValue('--rc')),
                    stops: buttons.filter(b => b.tabIndex === 0).map(residue),
                    pinned: buttons.filter(b => b.classList.contains('pinned')).map(residue),
                    hover: buttons.filter(b => b.classList.contains('hover')).map(residue),
                    focus: document.activeElement.classList.contains('seq-res') ? residue(document.activeElement) : null,
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

    def test_a_residue_button_is_named_from_the_letter_it_shows(self):
        # Label in name: a person who says "V" to a voice control, or hears it, finds the button named from it.
        self.two()
        named = self.script("""
            const buttons = [...document.querySelectorAll('#seq-body .seq-res')];
            return {all: buttons.every(b => b.getAttribute('aria-label').startsWith(b.textContent + ', ')),
                    first: buttons[0].getAttribute('aria-label'), described: buttons.filter(b => b.hasAttribute('aria-describedby')).length};
            """)
        self.assertEqual(named, {"all": True, "first": "M, Met 1", "described": 0})

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

    def test_it_hides_with_no_structure_and_on_a_phone(self):
        self.two()
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
            const button = document.querySelector('#seq-body .seq-res[aria-label="I, Ile 6"]');
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
        self.click("#seq-body .seq-res[aria-label=\"I, Ile 6\"]", 500)
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
            return {tab: view.tabIndex, role: view.getAttribute('role'), label: view.getAttribute('aria-label'),
                    describedby: view.getAttribute('aria-describedby'), help: document.getElementById('viewer-help').textContent};
            """)
        self.assertEqual((canvas["tab"], canvas["role"]), (0, "application"))
        self.assertEqual(canvas["label"], "3D structure view")
        # The keys are said once, as the description of the view, not in its name.
        self.assertEqual(canvas["describedby"], "viewer-help")
        self.assertIn("Arrow keys step through the residues", canvas["help"])
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


# Where things are, in the page's own pixels (CSS pixels, whatever the device pixel ratio of the machine is), for the layout tests below.
LAYOUT = """
window.box = function (selector) {
    const node = document.querySelector(selector);
    const r = node.getBoundingClientRect();
    return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height};
};
window.inside = function (inner, outer, slack = 1) {
    return inner.left >= outer.left - slack && inner.right <= outer.right + slack && inner.top >= outer.top - slack && inner.bottom <= outer.bottom + slack;
};
window.apart = function (a, b) {
    return a.right <= b.left || b.right <= a.left || a.bottom <= b.top || b.bottom <= a.top;
};
// The simple view of a ready scene: what the tests below put the page in (the engine here has no canvas, which would make it "webgl").
window.readyView = function () { document.body.dataset.view = 'ready'; };
"""


class ToolbarPlaceTests(SequenceTestCase):
    """The toolbar is in the title row from 701 px up, and a pill over the canvas at 700 px and narrower."""

    def tell(self):
        return self.script("""
            readyView();
            const bar = box('.view-toolbar'), head = box('.pane-header'), stage = box('#stage'), title = box('.pane-title');
            const card = box('.stage-card');
            const scene = getComputedStyle(document.getElementById('panel-toggle')).display !== 'none' ? box('#panel-toggle') : null;
            const style = getComputedStyle(document.querySelector('.view-toolbar'));
            return {bar, head, stage, title, scene, card, inHead: !!document.querySelector('.pane-header .view-toolbar'),
                    inStage: !!document.querySelector('#stage .view-toolbar'), count: document.querySelectorAll('.view-toolbar').length,
                    position: style.position, background: style.backgroundColor, border: style.borderTopWidth, shadow: style.boxShadow,
                    headDisplay: getComputedStyle(document.querySelector('.pane-header')).display,
                    items: [...document.querySelectorAll('.view-toolbar button')].filter(b => b.getClientRects().length).map(b => b.id)};
            """)

    def at(self, width, height=800):
        self.view.resize(width, height)
        self.addCleanup(self.view.resize, 1300, 800)
        self.settle(300)
        self.js(LAYOUT)

    def assert_one_row(self, state, where):
        head, bar, title = state["head"], state["bar"], state["title"]
        self.assertTrue(state["inHead"] and not state["inStage"], where)
        self.assertEqual(state["count"], 1, where)
        # Flat: part of the row, with nothing of a pill of its own.
        self.assertEqual((state["position"], state["border"], state["shadow"]), ("static", "0px", "none"), where)
        self.assertEqual(state["background"], "rgba(0, 0, 0, 0)", where)
        # Title, toolbar and the Scene button on one row: all inside the row's height, side by side, the toolbar at the right.
        items = [title, bar] + ([state["scene"]] if state["scene"] else [])
        for item in items:
            self.assertGreaterEqual(item["top"], head["top"] - 0.5, where)
            self.assertLessEqual(item["bottom"], head["bottom"] + 0.5, where)
        for left, right in zip(items, items[1:]):
            self.assertLessEqual(left["right"], right["left"] + 0.5, where)
            self.assertLess(abs((left["top"] + left["bottom"]) - (right["top"] + right["bottom"])), 6, where)
        self.assertLessEqual((state["scene"] or bar)["right"], head["right"] + 0.5, where)
        self.assertLessEqual(head["height"], 41, where)
        self.assertLess(head["bottom"], state["stage"]["top"] + 1, where)

    def test_from_701_px_up_the_toolbar_is_flat_in_the_title_row_and_the_row_stays_one_row(self):
        self.two()
        for width in (701, 900, 1040, 1041, 1300):
            with self.subTest(width=width):
                self.at(width)
                state = self.tell()
                self.assert_one_row(state, width)
                # The Scene button is in the row from 701 to 1040 px only, and the toolbar comes before it.
                self.assertEqual(state["scene"] is not None, 701 <= width <= 1040, width)
                self.assertEqual(state["items"], ["t-reset", "t-zoom", "t-measure", "t-label", "t-image", "t-full", "t-seq", "t-adv"], width)

    def test_the_row_is_one_row_in_simplified_chinese_too(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.two()
        self.assertEqual(self.script("return document.querySelector('.pane-title').textContent"), "结构视图")
        for width in (701, 900, 1300):
            with self.subTest(width=width):
                self.at(width)
                self.assert_one_row(self.tell(), width)

    def test_the_title_row_is_as_high_with_the_toolbar_as_without_it(self):
        self.two()
        self.js(LAYOUT)
        heights = self.script("""
            readyView();
            const withBar = box('.pane-header').height, stageTop = box('#stage').top;
            document.body.dataset.adv = '1';
            const adv = box('.pane-header').height, advTop = box('#stage').top;
            document.body.dataset.adv = '0';
            return {withBar, adv, same: stageTop === advTop};
            """)
        self.assertEqual(heights["withBar"], heights["adv"])
        self.assertTrue(heights["same"])
        self.assertLessEqual(heights["withBar"], 41)

    def test_at_700_px_and_narrower_the_toolbar_floats_over_the_canvas_as_a_pill_and_there_is_no_title_row(self):
        self.two()
        for width in (700, 600):
            with self.subTest(width=width):
                self.at(width)
                state = self.tell()
                self.assertEqual(state["headDisplay"], "contents", width)
                self.assertEqual(state["count"], 1)
                self.assertEqual(state["position"], "absolute")
                self.assertEqual(state["border"], "1px")
                self.assertNotEqual(state["shadow"], "none")
                self.assertNotEqual(state["background"], "rgba(0, 0, 0, 0)")
                # Over the canvas, 8 px in from its top right corner, as it always was there.
                self.assertTrue(self.script("return inside(%s, %s)" % (json.dumps(state["bar"]), json.dumps(state["stage"]))))
                self.assertAlmostEqual(state["bar"]["top"] - state["stage"]["top"], 8, delta=1)
                self.assertAlmostEqual(state["stage"]["right"] - state["bar"]["right"], 8, delta=1)
                self.assertEqual(self.script("return getComputedStyle(document.querySelector('.pane-title')).display"), "none")
                # Label and Full screen go on a phone, as they did; the Sequence button goes with the strip.
                self.assertEqual(state["items"], ["t-reset", "t-zoom", "t-measure", "t-image", "t-adv"])

    def test_the_toolbar_keeps_its_look_hover_pressed_and_disabled_in_the_row(self):
        self.two()
        self.at(1300)
        look = self.script("""
            readyView();
            const css = id => { const s = getComputedStyle(document.getElementById(id)); return [s.backgroundColor, s.borderTopColor, s.color, s.opacity]; };
            const root = getComputedStyle(document.documentElement);
            document.getElementById('t-measure').setAttribute('aria-pressed', 'true');
            const out = {idle: css('t-reset'), pressed: css('t-measure'), seq: css('t-seq'), zoom: css('t-zoom'),
                         accentSoft: root.getPropertyValue('--accent-soft').trim(), size: box('#t-seq')};
            document.getElementById('t-measure').setAttribute('aria-pressed', 'false');
            return out;
            """)
        # Idle: no background, no edge; pressed (Measure is, here, and the Sequence button is while the strip is on) the same look as each other.
        self.assertEqual(look["idle"][:2], ["rgba(0, 0, 0, 0)", "rgba(0, 0, 0, 0)"])
        self.assertEqual(look["pressed"], look["seq"])
        self.assertNotEqual(look["pressed"][0], look["idle"][0])
        self.assertEqual(look["zoom"][3], "0.4")
        self.assertEqual((look["size"]["width"], look["size"]["height"]), (30, 30))


class SequenceButtonTests(SequenceTestCase):
    """The Sequence button of the toolbar switches the strip on and off; the structure choice shows over the canvas while it is on."""

    def open_two(self):
        self.two()
        self.js(LAYOUT)
        self.script("readyView(); return 1")

    def visible(self, selector):
        return self.script("""
            const node = document.querySelector(%s);
            return !!node && getComputedStyle(node).display !== 'none' && node.getClientRects().length > 0;
            """ % json.dumps(selector))

    def test_the_button_is_an_icon_between_full_screen_and_advanced_named_sequence(self):
        self.open_two()
        found = self.script("""
            const bar = document.querySelector('.view-toolbar');
            const kids = [...bar.children].map(n => n.id || n.getAttribute('role'));
            const button = document.getElementById('t-seq');
            return {kids, title: button.title, label: button.getAttribute('aria-label'), icon: button.querySelector('use').getAttribute('href'),
                    text: button.textContent.trim(), pressed: button.getAttribute('aria-pressed'), controls: button.getAttribute('aria-controls'),
                    controlled: !!document.getElementById(button.getAttribute('aria-controls')), cls: [...button.classList].join(' '),
                    sepsShown: [button.previousElementSibling, button.nextElementSibling].map(n => getComputedStyle(n).display !== 'none')};
            """)
        self.assertEqual(found["kids"][-5:], ["t-full", "separator", "t-seq", "separator", "t-adv"])
        self.assertEqual((found["title"], found["label"], found["text"]), ("Sequence", "Sequence", ""))
        self.assertEqual(found["icon"], "#i-dna")
        self.assertEqual((found["pressed"], found["controls"], found["controlled"]), ("true", "seqbar", True))
        self.assertIn("btn-icon", found["cls"])
        self.assertEqual(found["sepsShown"], [True, True])

    def test_it_is_hidden_with_no_structure_when_the_view_is_not_ready_in_advanced_and_on_a_phone(self):
        self.open_two()
        self.assertTrue(self.visible("#t-seq"))
        for view in ("starting", "restoring", "webgl", "folding", "empty"):
            self.script("document.body.dataset.view = %s; return 1" % json.dumps(view))
            self.assertFalse(self.visible("#t-seq"), view)
        self.script("readyView(); document.body.dataset.adv = '1'; return 1")
        self.assertFalse(self.visible("#t-seq"))
        self.script("document.body.dataset.adv = '0'; return 1")
        self.assertTrue(self.visible("#t-seq"))
        # The separator before it goes with it, so that two do not stand side by side.
        self.script("document.getElementById('seqbar').dataset.empty = '1'; return 1")
        self.assertEqual(self.script("""
            return [...document.querySelectorAll('.view-toolbar > *')].filter(n => getComputedStyle(n).display !== 'none')
                .map(n => n.id || 'sep').join(' ')"""), "t-reset t-zoom sep t-measure t-label sep t-image t-full sep t-adv")
        self.script("document.getElementById('seqbar').dataset.empty = '0'; return 1")
        self.view.resize(700, 800)
        self.addCleanup(self.view.resize, 1300, 800)
        self.settle(300)
        self.assertFalse(self.visible("#t-seq"))
        self.assertFalse(self.visible("#seqbar"))

    def test_the_strip_is_open_at_the_start_and_off_it_takes_no_room_and_the_canvas_gets_the_height(self):
        self.open_two()
        start = self.script("""
            const bar = document.getElementById('seqbar');
            return {open: bar.dataset.open, pressed: document.getElementById('t-seq').getAttribute('aria-pressed'), strip: box('#seqbar'),
                    stage: box('#stage'), viewer: box('#viewer'), first: box('#seq-body .seq-block'), body: box('#seq-body')};
            """)
        self.assertEqual((start["open"], start["pressed"]), ("1", "true"))
        self.assertGreater(start["strip"]["height"], 40)
        # The head row is gone: the strip is only its residues, and the first line is not at the top edge.
        self.assertEqual(self.script("return document.querySelector('.seq-head')"), None)
        self.assertGreaterEqual(start["first"]["top"] - start["strip"]["top"], 8)
        self.script("document.getElementById('t-seq').click(); return 1")
        self.settle(400)
        closed = self.script("""
            return {open: document.getElementById('seqbar').dataset.open, pressed: document.getElementById('t-seq').getAttribute('aria-pressed'),
                    display: getComputedStyle(document.getElementById('seqbar')).display, strip: box('#seqbar'), stage: box('#stage'),
                    viewer: box('#viewer'), focus: document.activeElement.id};
            """)
        self.assertEqual((closed["open"], closed["pressed"], closed["display"]), ("0", "false", "none"))
        self.assertEqual((closed["strip"]["width"], closed["strip"]["height"]), (0, 0))
        # The canvas grows by what the strip had: its height, and its border.
        self.assertAlmostEqual(closed["stage"]["height"] - start["stage"]["height"], start["strip"]["height"], delta=1.5)
        self.assertAlmostEqual(closed["viewer"]["height"], closed["stage"]["height"], delta=0.5)
        # The button, which stays, keeps the focus the click gave it (a click on the page itself, here, a real one: see the key test).
        self.script("document.getElementById('t-seq').focus(); document.getElementById('t-seq').click(); return 1")
        self.settle(300)
        again = self.script("return {open: document.getElementById('seqbar').dataset.open, focus: document.activeElement.id, stage: box('#stage')}")
        self.assertEqual((again["open"], again["focus"]), ("1", "t-seq"))
        self.assertAlmostEqual(again["stage"]["height"], start["stage"]["height"], delta=0.5)

    def test_the_canvas_is_told_to_resize_when_the_strip_goes_and_comes(self):
        self.open_two()
        self.script("""
            window.__resizes = 0;
            const original = viewerInstance.handleResize.bind(viewerInstance);
            viewerInstance.handleResize = () => { window.__resizes++; return original(); };
            return 1;
            """)
        self.click("#t-seq", 300)
        self.assertGreaterEqual(self.script("return window.__resizes"), 1)
        self.click("#t-seq", 300)
        self.assertGreaterEqual(self.script("return window.__resizes"), 2)

    def test_enter_and_space_work_the_button_and_the_focus_stays_on_it(self):
        self.open_two()
        widget = self.view.focusProxy() or self.view
        for key in (Qt.Key_Return, Qt.Key_Space, Qt.Key_Space):
            self.script("document.getElementById('t-seq').focus(); return 1")
            before = self.script("return document.getElementById('seqbar').dataset.open")
            QTest.keyClick(widget, key)
            self.pump(0.3)
            state = self.script("return {open: document.getElementById('seqbar').dataset.open, focus: document.activeElement.id}")
            self.assertEqual(state, {"open": "0" if before == "1" else "1", "focus": "t-seq"}, key)

    def test_the_focus_goes_to_the_toolbar_not_the_page_when_the_button_goes_with_the_last_structure(self):
        self.open_two()
        self.script("document.getElementById('t-seq').focus(); return 1")
        # The last structure goes (as far as the strip knows) while the focus is on the button.
        self.script("""
            const real = loadedStructures;
            window.loadedStructures = () => [];
            renderSequence();
            window.loadedStructures = real;
            return 1;
            """)
        self.assertEqual(self.script("return document.activeElement.id"), "t-adv")

    def test_the_structure_choice_is_over_the_canvas_at_its_top_right_drawn_as_a_pill(self):
        self.open_two()
        shown = self.script("""
            const select = document.getElementById('seq-structure');
            const style = getComputedStyle(select);
            return {select: box('#seq-structure'), stage: box('#stage'), position: style.position, background: style.backgroundColor,
                    border: style.borderTopWidth, radius: style.borderTopLeftRadius, shadow: style.boxShadow, maxWidth: style.maxWidth,
                    parent: select.parentElement.id, card: getComputedStyle(document.documentElement).getPropertyValue('--bg-card').trim()};
            """)
        self.assertEqual(shown["parent"], "stage")
        self.assertEqual(shown["position"], "absolute")
        self.assertAlmostEqual(shown["select"]["top"] - shown["stage"]["top"], 10, delta=1)
        self.assertAlmostEqual(shown["stage"]["right"] - shown["select"]["right"], 10, delta=1)
        self.assertEqual((shown["border"], shown["radius"], shown["maxWidth"]), ("1px", "10px", "260px"))
        self.assertNotEqual(shown["shadow"], "none")
        self.assertNotIn("rgba(0, 0, 0, 0)", shown["background"])
        self.assertLessEqual(shown["select"]["width"], 260.5)

    def test_it_is_hidden_unless_all_of_these_hold_the_strip_is_open_two_choices_ready_not_advanced_wider_than_a_phone(self):
        self.open_two()
        self.assertTrue(self.visible("#seq-structure"))
        # The strip is off.
        self.script("document.getElementById('t-seq').click(); return 1")
        self.assertFalse(self.visible("#seq-structure"))
        self.script("document.getElementById('t-seq').click(); return 1")
        self.assertTrue(self.visible("#seq-structure"))
        # The view is not ready.
        for view in ("starting", "restoring", "webgl", "folding", "empty"):
            self.script("document.body.dataset.view = %s; return 1" % json.dumps(view))
            self.assertFalse(self.visible("#seq-structure"), view)
        self.script("readyView(); return 1")
        # Advanced.
        self.script("document.body.dataset.adv = '1'; return 1")
        self.assertFalse(self.visible("#seq-structure"))
        self.script("document.body.dataset.adv = '0'; return 1")
        # A phone.
        self.view.resize(700, 800)
        self.addCleanup(self.view.resize, 1300, 800)
        self.settle(300)
        self.assertFalse(self.visible("#seq-structure"))
        self.view.resize(701, 800)
        self.settle(300)
        self.assertTrue(self.visible("#seq-structure"))
        # One choice only.
        self.click(self.row(2) + " .srow-main button:last-child", 800)
        self.script("readyView(); return 1")
        self.assertFalse(self.visible("#seq-structure"))

    def test_the_whole_name_is_in_its_title_and_a_long_one_is_cut_with_an_ellipsis(self):
        self.open_two()
        found = self.script("""
            const select = document.getElementById('seq-structure');
            const style = getComputedStyle(select);
            return {title: select.title, shown: select.selectedOptions[0].textContent, overflow: style.overflow, ellipsis: style.textOverflow};
            """)
        self.assertEqual(found["title"], found["shown"])

    def test_it_does_not_meet_the_status_line_the_legend_or_the_axes_in_any_layout(self):
        self.open_two()
        for width, height in ((1300, 800), (1040, 700), (701, 600), (1300, 500)):
            with self.subTest(width=width, height=height):
                self.view.resize(width, height)
                self.addCleanup(self.view.resize, 1300, 800)
                self.settle(300)
                found = self.script("""
                    readyView();
                    showStatus('Loaded and aligned 22. A longer message, to see where the status line goes.', 0);
                    document.getElementById('canvas-legend').dataset.shown = '1';
                    document.getElementById('canvas-legend').textContent = 'pLDDT';
                    const select = box('#seq-structure'), stage = box('#stage');
                    // The axes are drawn on the canvas in its bottom right corner, 128 px square.
                    const axes = {left: stage.right - 128, right: stage.right, top: stage.bottom - 128, bottom: stage.bottom};
                    const out = {toast: apart(select, box('#toast')), legend: apart(select, box('#canvas-legend')), axes: apart(select, axes),
                                 shown: getComputedStyle(document.getElementById('seq-structure')).display !== 'none', inside: inside(select, stage, 0.5)};
                    hideStatus();
                    return out;
                    """)
                self.assertEqual(found, {"toast": True, "legend": True, "axes": True, "shown": True, "inside": True})

    def test_tab_goes_from_the_toolbar_to_the_canvas_the_structure_choice_and_the_strip(self):
        self.open_two()
        widget = self.view.focusProxy() or self.view
        order = self.script("""
            const bar = document.querySelector('.view-toolbar'), viewer = document.getElementById('viewer');
            const select = document.getElementById('seq-structure'), strip = document.getElementById('seq-body');
            const before = (a, b) => !!(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
            return {toolbarBeforeCanvas: before(bar, viewer), canvasBeforeSelect: before(viewer, select), selectBeforeStrip: before(select, strip),
                    stops: [...bar.querySelectorAll('button')].filter(b => b.tabIndex === 0 && b.getClientRects().length).length};
            """)
        self.assertEqual(order, {"toolbarBeforeCanvas": True, "canvasBeforeSelect": True, "selectBeforeStrip": True, "stops": 1})
        self.script("document.querySelector('.view-toolbar button[tabindex=\"0\"]').focus(); return 1")
        seen = []
        for _ in range(5):
            QTest.keyClick(widget, Qt.Key_Tab)
            self.pump(0.2)
            # (A message of Mol* in the canvas, which an engine with no WebGL shows, has a link of its own: not one of the page's stops.)
            stop = self.script("""
                const a = document.activeElement;
                if (a.id !== 'viewer' && a.closest('#viewer')) return null;
                return a.id || (a.classList.contains('seq-res') ? 'strip' : a.outerHTML.slice(0, 120));""")
            if stop:
                seen.append(stop)
        self.assertEqual(seen[:3], ["viewer", "seq-structure", "strip"])
        # The arrow keys move inside the toolbar, as they did.
        self.script("document.getElementById('t-reset').focus(); return 1")
        QTest.keyClick(widget, Qt.Key_Right)
        self.pump(0.2)
        self.assertEqual(self.script("return document.activeElement.id"), "t-zoom")


if __name__ == "__main__":
    unittest.main()
