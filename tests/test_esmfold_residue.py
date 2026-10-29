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

from PySide6.QtCore import QUrl  # noqa: E402  (after the harness has set the engine's environment)

# Mol*'s pointer events: the loci of the residue `group` (from 0) of a structure, as the representation gives them.
POINTER = """
// The keys Mol*'s pointer events say are held: none, as an object with every key, which is what its bindings match.
const NO_KEYS = {shift: false, alt: false, control: false, meta: false};
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
    plugin.behaviors.interaction.click.next({current: index === null ? {loci: {kind: 'empty-loci'}} : lociOf(index, group), modifiers: NO_KEYS, buttons: 1, button: 1});
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


class FocusTests(ResidueTestCase):
    """Mol*'s own click puts the residue in focus and draws its surroundings with nodes of its own: a pin lets go of them all."""

    STATE = """return {cells: plugin.state.data.cells.size, nodes: focusNodes().length, focus: !!plugin.managers.structure.focus.current,
                    marked: plugin.managers.structure.selection.stats.elementCount, pinned: pinnedResidue ? pinnedResidue.number : null,
                    saved: takeSnapshot().data.tree.transforms.length, raw: plugin.state.getSnapshot({camera: false}).data.tree.transforms.length}"""

    def open_pinnable(self):
        self.write_with(A, "information_content")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        base = self.script(self.STATE)
        self.assertEqual((base["nodes"], base["focus"], base["marked"]), (0, False, 0))
        return base

    def pin(self):
        self.script("await clickOn(0, 5); return 1")
        self.settle(300)
        pinned = self.script(self.STATE)
        # Mol* did put the residue in focus, with five nodes (two selections, the sticks, the surroundings and their contacts).
        self.assertEqual((pinned["nodes"], pinned["focus"], pinned["pinned"]), (5, True, 6))
        self.assertEqual(pinned["cells"], pinned["raw"])
        return pinned

    def test_letting_go_of_a_pin_leaves_nothing_in_the_scene_whichever_way_it_goes(self):
        base = self.open_pinnable()
        for way in ("escape", "clear", "nothing", "remove the structure"):
            self.pin()
            if way == "escape":
                self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
            elif way == "clear":
                self.script("document.querySelector('#residue-body .res-foot button:last-child').click(); return 1")
            elif way == "nothing":
                self.script("await clickOn(null); return 1")
            else:
                self.click(self.row(1) + " .srow-main button:last-child", 400)
            self.settle(700)
            after = self.script(self.STATE)
            self.assertEqual((after["nodes"], after["focus"], after["marked"], after["pinned"]), (0, False, 0, None), way)
            if way != "remove the structure":
                self.assertEqual(after["cells"], base["cells"], way)
            else:
                self.assertEqual(self.script("return focusNodes().length"), 0)
        self.assertEqual(self.console_errors(), [])

    def test_a_session_saved_while_a_residue_is_pinned_has_none_of_the_focus(self):
        self.open_pinnable()
        self.settle(1800)
        del self.saved[:]
        self.pin()
        self.settle(1800)
        # The autosave that follows holds the scene, not the pin.
        self.script("document.querySelector('#color-body [data-coloring=\"builtin:structure\"]').click(); await structureQueue; return 1")
        self.settle(1800)
        self.assertTrue(self.saved)
        for session in self.saved:
            tags = [tag for node in session["data"]["tree"]["transforms"] for tag in node.get("tags", [])]
            self.assertEqual([tag for tag in tags if tag.startswith("structure-focus-")], [])
            self.assertEqual(session["structureFocus"], {})

    def test_a_session_that_comes_back_with_a_focus_in_it_shows_none_of_it(self):
        base = self.open_pinnable()
        self.pin()
        # What an earlier page saved: the tree with the focus' nodes and the focus itself.
        raw = self.script("return plugin.state.getSnapshot({camera: false})")
        tags = [tag for node in raw["data"]["tree"]["transforms"] for tag in node.get("tags", [])]
        self.assertEqual(len([tag for tag in tags if tag.startswith("structure-focus-")]), 5)
        self.assertIn("current", raw["structureFocus"])
        self.session = raw
        self.settle(1800)
        self.open_page()
        self.settle(900)
        restored = self.script(self.STATE)
        self.assertEqual((restored["nodes"], restored["focus"], restored["marked"], restored["pinned"]), (0, False, 0, None))
        self.assertEqual(restored["cells"], base["cells"])
        self.assertEqual(self.script("return loadedStructures().length"), 2)
        # (The stream of the page that was left reports its end.)
        self.assertEqual([error for error in self.console_errors() if "SSE Connection Error" not in error[1]], [])

    def test_undo_brings_back_the_scene_without_the_focus_that_was_in_it(self):
        base = self.open_pinnable()
        self.pin()
        self.click(self.row(2) + " .srow-main button:last-child", 600)
        self.click("#toast .toast-action", 1500)
        restored = self.script(self.STATE)
        self.assertEqual((restored["nodes"], restored["focus"], restored["marked"], restored["pinned"]), (0, False, 0, None))
        self.assertEqual(restored["cells"], base["cells"])


# How the Residue card's table is laid out, measured in the page from where each text is drawn.
CARD_LAYOUT = """
window.cardLayout = function () {
    const body = document.getElementById('residue-body');
    const table = body.querySelector('.res-table');
    // The line boxes of a cell's own text (the swatch is not text): one top for a text on one line.
    const rangeOf = cell => {
        const text = [...cell.childNodes].find(node => node.nodeType === Node.TEXT_NODE);
        const range = document.createRange();
        range.selectNodeContents(text || cell);
        return range;
    };
    const lines = cell => new Set([...rangeOf(cell).getClientRects()].filter(box => box.width > 0).map(box => Math.round(box.top))).size;
    const measure = row => [...row.cells].map(cell => {
        const box = cell.getBoundingClientRect();
        const text = rangeOf(cell).getBoundingClientRect();
        const swatch = cell.querySelector('.res-swatch');
        const dot = swatch ? swatch.getBoundingClientRect() : null;
        return {text: cell.textContent, lines: lines(cell), clipped: cell.scrollWidth > cell.clientWidth,
                overflow: getComputedStyle(cell).textOverflow, title: cell.title, gap: Math.round(box.right - text.right),
                swatchBeside: dot ? dot.top >= text.top - 2 && dot.bottom <= text.bottom + 2 && dot.right <= text.left + 8 : null,
                swatchWidth: dot ? Math.round(dot.width) : null};
    });
    const rows = [...table.rows];
    return {header: measure(rows[0]), rows: rows.slice(1).map(measure),
            heights: rows.slice(1).map(row => Math.round(row.getBoundingClientRect().height)),
            overflowing: body.scrollWidth > body.clientWidth};
};
"""


class ResidueTableLayoutTests(ResidueTestCase):
    """A structure's name stays on one line beside its swatch, cut with an ellipsis when it is too long, and no value wraps."""

    # Three names of about thirty characters, and one short one.
    NAMES = ["serine_protease_inhibitor_kazal", "beta_lactamase_TEM1_variant_07", "putative_uncharacterized_prot_3", "alpha"]

    def scene(self, layer, names=NAMES):
        for index, name in enumerate(names):
            self.write_with(name + ".pdb", "information_content", "same_residue_pct", "hydro",
                            sequence=fixtures.FIRST_SEQUENCE if index % 2 == 0 else fixtures.SECOND_SEQUENCE,
                            shift=(3.0 * index, 1.0 * index, 0.0), wobble=0.4 * index)
        self.open_scene(*[name + ".pdb" for name in names])
        self.js(CARD_LAYOUT)
        if layer:
            self.choose(layer)
        self.script("await clickOn(0, 20); return 1")

    def at(self, width):
        """The card's layout with the window `width` pixels wide: the panel beside the canvas, a drawer over it, or a strip."""
        self.view.resize(width, 800)
        self.addCleanup(self.view.resize, 1300, 800)
        self.script("setPanel(true); setTab('residue'); await new Promise(resolve => setTimeout(resolve, 450)); return 1")
        return self.script("return cardLayout()")

    def assert_laid_out(self, layout, context, names, cut, layer_lines=None):
        """Every name and value is on one line. `cut`: whether the names of more than twelve letters are cut (None: not said).

        `layer_lines`: how many lines the layer's name takes in the header (None: no layer, or it may take one or two).
        """
        self.assertEqual(len({*layout["heights"]}), 1, (context, layout["heights"]))
        self.assertFalse(layout["overflowing"], context)
        shown = [row[0] for row in layout["rows"]]
        self.assertEqual([name["text"] for name in shown], names, context)
        for name in shown:
            self.assertEqual(name["lines"], 1, (context, name))
            self.assertEqual((name["swatchBeside"], name["swatchWidth"]), (True, 8), (context, name))
            # What is cut is cut with an ellipsis, and the whole name is on hover.
            self.assertEqual((name["overflow"], name["title"]), ("ellipsis", name["text"]), (context, name))
        long_names = [name for name in shown if len(name["text"]) > 12]
        if cut is not None:
            self.assertEqual([name["clipped"] for name in long_names], [cut] * len(long_names), context)
        self.assertFalse(shown[-1]["clipped"], (context, shown[-1]))
        for row in layout["rows"]:
            for value in row[1:]:
                self.assertEqual(value["lines"], 1, (context, value))
                self.assertFalse(value["clipped"], (context, value))
        # The headers are whole, each on one line but a layer's name, which breaks at its spaces unless the strip has the room, and
        # the text of every column keeps to its right edge.
        for head in layout["header"]:
            self.assertFalse(head["clipped"], (context, head))
        layer = layout["header"][2] if len(layout["header"]) == 4 else None
        for head in layout["header"]:
            if head is layer and layer_lines is not None:
                self.assertEqual(head["lines"], layer_lines, (context, head))
            elif head is layer:
                self.assertLessEqual(head["lines"], 2, (context, head))
            else:
                self.assertEqual(head["lines"], 1, (context, head))
        for cell in [*layout["header"][1:], *[value for row in layout["rows"] for value in row[1:]]]:
            self.assertLessEqual(cell["gap"], 1, (context, cell))

    def test_long_names_and_values_stay_on_one_line_with_each_layer_in_each_layout(self):
        for layer in (None, "information_content", "same_residue_pct", "hydro"):
            self.scene(layer)
            # The panel is 340 px wide beside the canvas and in the drawer, and cuts the names of a layer's table; the strip is as
            # wide as the window and cuts none. (Without a layer the names have more room and may fit.)
            for width, cut in ((1300, True), (1040, True), (600, False)):
                with self.subTest(layer=layer, width=width):
                    layout = self.at(width)
                    label = layout["header"][2]["text"] if layer else ""
                    # A name with a space breaks there beside the canvas and in the drawer (a word, such as Hydrophobicity, does not).
                    self.assert_laid_out(layout, f"{layer or 'pLDDT'} at {width} px", self.NAMES, cut if layer else None,
                                         (2 if " " in label and width > 700 else 1) if layer else None)
            self.view.setUrl(QUrl("about:blank"))

    def test_a_value_with_a_unit_does_not_break_under_a_short_layer_name(self):
        # "39.5 %" has a space, and under a header as short as "IC" its column has nothing else to keep it from breaking there.
        for index, name in enumerate(self.NAMES):
            sequence = fixtures.FIRST_SEQUENCE if index % 2 == 0 else fixtures.SECOND_SEQUENCE
            fixtures.write_structure(self.directory, name + ".pdb", sequence, sidecar=False,
                                     shift=(3.0 * index, 1.0 * index, 0.0), wobble=0.4 * index)
            layer = fixtures.residue_property("IC", "%", [0, 100], fixtures.information_content(len(sequence)), None, "Short.")
            fixtures.write_sidecar(self.directory, name + ".pdb",
                                   fixtures.residue_data(name + ".pdb", sequence, properties={"ic": layer}))
        self.open_scene(*[name + ".pdb" for name in self.NAMES])
        self.js(CARD_LAYOUT)
        self.choose("ic")
        self.script("await clickOn(0, 20); return 1")
        for width in (1300, 1040, 600):
            with self.subTest(width=width):
                layout = self.at(width)
                self.assertEqual(layout["header"][2]["text"], "IC")
                self.assertTrue(all(row[2]["text"].endswith(" %") for row in layout["rows"]), layout["rows"])
                self.assert_laid_out(layout, f"a short layer at {width} px", self.NAMES, width > 600, 1)

    def test_the_names_of_the_reference_picture_fit_beside_their_swatches(self):
        self.scene("information_content", ["protein_a", "protein_b"])
        layout = self.at(1300)
        self.assert_laid_out(layout, "protein_a and protein_b", ["protein_a", "protein_b"], False, 2)
        self.assertEqual([head["text"] for head in layout["header"]], ["Structure", "pLDDT", "Information Content", "Cα–Cα (Å)"])
        # Only the layer's long header takes two lines, and the values stay on one.
        self.assertEqual([head["lines"] for head in layout["header"]], [1, 1, 2, 1])

    def test_the_card_comes_in_simplified_chinese_in_the_same_shape(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.scene("same_residue_pct")
        layout = self.at(1300)
        # The layer's name is translated, and a name with no spaces stays on one line, whole, as a Latin word does.
        self.assertEqual(layout["header"][2]["text"], "残基一致率")
        self.assert_laid_out(layout, "zh_CN", self.NAMES, True, 1)


if __name__ == "__main__":
    unittest.main()
