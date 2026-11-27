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

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests import esmfold_layer_fixtures as fixtures
from tests.esmfold_page_helpers import A, B, C, POINTER, ResidueTestCase  # noqa: F401 (POINTER kept for esmfold_session_maker)

from PySide6.QtCore import Qt, QUrl  # noqa: E402  (after the harness has set the engine's environment)
from PySide6.QtTest import QTest  # noqa: E402


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

    COLORS = """
        const probe = token => { const n = document.createElement('i'); n.style.color = 'var(' + token + ')'; document.body.append(n);
                                 const c = getComputedStyle(n).color; n.remove(); return c; };
        const probeBg = token => { const n = document.createElement('i'); n.style.backgroundColor = 'var(' + token + ')'; document.body.append(n);
                                   const c = getComputedStyle(n).backgroundColor; n.remove(); return c; };
        const channels = c => c.match(/[\\d.]+/g).slice(0, 3).map(Number);
        const luminance = c => { const [r, g, b] = channels(c).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
                                 return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
        const contrast = (a, b) => { const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
        const tip = document.getElementById('tip'), toast = document.getElementById('toast');
        const style = getComputedStyle(tip), tipBg = style.backgroundColor;
        const muted = [tip.querySelector('.h span'), tip.querySelector('td:first-child')].map(n => getComputedStyle(n).color);
        const dot = tip.querySelector('.swatch-dot');
        dot.style.background = '#ffffff';
        const dotStyle = getComputedStyle(dot);
        return {
            bg: tipBg, text: style.color, border: style.borderTopColor, shadow: style.boxShadow,
            card: probeBg('--bg-card'), primary: probe('--text-primary'), secondary: probe('--text-secondary'), edge: probe('--border-color'),
            dialogShadow: getComputedStyle(document.documentElement).getPropertyValue('--shadow-dialog').trim(),
            muted, contrastText: contrast(style.color, tipBg), contrastMuted: muted.map(c => contrast(c, tipBg)),
            toastBg: getComputedStyle(toast).backgroundColor, toastText: getComputedStyle(toast).color, toastBorder: getComputedStyle(toast).borderTopColor,
            dot: [dotStyle.borderTopWidth, dotStyle.borderTopColor, dotStyle.backgroundColor] };
        """

    def tip_colors(self, theme):
        self.script("applyTheme(%s, false); await hoverOn(0, 5); return 1" % json.dumps(theme))
        return self.script(self.COLORS)

    def test_the_hover_box_follows_the_colour_mode_and_the_status_line_does_not(self):
        self.write_with(A, "information_content")
        self.open_scene(A)
        found = {theme: self.tip_colors(theme) for theme in ("light", "dark")}
        for theme, now in found.items():
            with self.subTest(theme=theme):
                # The page's card, its text, its edge and its dialog shadow, as the Tree View's tooltip has them.
                self.assertEqual((now["bg"], now["text"], now["border"]), (now["card"], now["primary"], now["edge"]))
                self.assertEqual(now["muted"], [now["secondary"]] * 2)
                self.assertNotEqual(now["shadow"], "none")
                # Readable: ordinary text and the muted texts at 4.5 to 1 against the box.
                self.assertGreaterEqual(now["contrastText"], 4.5)
                self.assertGreaterEqual(min(now["contrastMuted"]), 4.5)
                # A white swatch has an edge that shows on the box (a light one) and still reads on a dark one.
                self.assertEqual(now["dot"][0], "1px")
                self.assertNotEqual(now["dot"][1], now["dot"][2])
        self.assertNotEqual(found["light"]["bg"], found["dark"]["bg"])
        self.assertNotEqual(found["light"]["text"], found["dark"]["text"])
        self.assertEqual(found["light"]["bg"], "rgb(255, 255, 255)")
        # The status line is as it was: dark in both themes (the --tip-* colors).
        self.assertEqual((found["light"]["toastBg"], found["light"]["toastText"], found["light"]["toastBorder"]),
                         ("rgb(24, 24, 27)", "rgb(250, 250, 250)", "rgb(39, 39, 42)"))
        self.assertEqual((found["dark"]["toastBg"], found["dark"]["toastText"], found["dark"]["toastBorder"]),
                         ("rgb(39, 39, 42)", "rgb(250, 250, 250)", "rgb(63, 63, 70)"))
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

    def test_escape_hides_the_tip_before_anything_else_until_the_pointer_moves_again(self):
        self.write_with(A, "information_content")
        self.open_scene(A)
        escape = "document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))"
        move = """
            const stage = document.getElementById('stage').getBoundingClientRect();
            document.getElementById('viewer').dispatchEvent(new PointerEvent('pointermove', {clientX: stage.left + %d, clientY: stage.top + 150, bubbles: true}));
            await new Promise(resolve => setTimeout(resolve, 250));
            return 1"""
        # A pinned residue, and the tip of the hovered one: Escape takes the tip only, the pin stays.
        self.script("await clickOn(0, 2); await hoverOn(0, 5); return 1")
        self.assertTrue(self.script("return tipNow().shown"))
        self.js(escape)
        self.assertFalse(self.script("return tipNow().shown"))
        self.assertEqual(self.script("return pinnedResidue ? pinnedResidue.number : null"), 3)
        # It stays hidden while the pointer is still over the residue, even if the residue is drawn again (the pin changes)...
        self.script("await clickOn(0, 4); return 1")
        self.assertFalse(self.script("return tipNow().shown"))
        # ... and comes back with the pointer's next move.
        self.script(move % 200)
        self.assertTrue(self.script("return tipNow().shown"))
        # The next Escape is the pin's.
        self.js(escape)
        self.assertFalse(self.script("return tipNow().shown"))
        self.js(escape)
        self.assertIsNone(self.script("return pinnedResidue"))


class PinTests(ResidueTestCase):
    def test_a_click_pins_a_residue_and_the_card_shows_its_equivalent_in_every_structure(self):
        self.write_with(A, "information_content", "same_residue_pct")
        self.write_with(B, "information_content", "same_residue_pct", sequence=fixtures.SECOND_SEQUENCE, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.write_with(C, shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.open_scene(A, B, C)
        empty = self.script("return residueNow()")
        self.assertIn("Hover a residue", empty["hint"])
        self.assertEqual([button[2] for button in empty["buttons"]], ["true", "true"])
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
        self.assertEqual(card["buttons"], [["residue-zoom", "Zoom to the pinned residue", "false"], ["residue-clear", "Clear", "false"]])
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
        self.script("document.getElementById('residue-clear').click(); await new Promise(r => setTimeout(r, 300)); return 1")
        self.assertIsNone(self.script("return pinnedResidue"))
        self.assertEqual(self.script("return plugin.managers.structure.selection.stats.elementCount"), 0)
        self.assertEqual(self.console_errors(), [])

    def test_the_cards_buttons_are_not_drawn_again_with_the_card_so_the_focus_stays_on_them_clear_included(self):
        self.write_with(A, "information_content")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        self.script("await clickOn(0, 5); return 1")
        focus = "return document.activeElement.id"
        # The card is drawn again whenever the structures are (here, as any change does it): Zoom keeps the focus.
        self.script("""
            document.getElementById('residue-zoom').focus();
            renderStructures();
            await new Promise(resolve => setTimeout(resolve, 400));
            return 1;""")
        self.assertEqual(self.script(focus), "residue-zoom")
        # Clear lets go of the pin and stays, greyed out: the focus is still on it, not on the page.
        self.script("""
            const clear = document.getElementById('residue-clear');
            clear.focus();
            clear.click();
            await new Promise(resolve => setTimeout(resolve, 400));
            return 1;""")
        self.assertIsNone(self.script("return pinnedResidue"))
        self.assertEqual(self.script(focus), "residue-clear")
        self.assertEqual(self.script("return residueNow().buttons.map(b => b[2])"), ["true", "true"])
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
                self.script("document.getElementById('residue-clear').click(); return 1")
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
        self.choose("builtin:structure")
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


class ResidueButtonTests(ResidueTestCase):
    """Zoom to and Clear are icon buttons in the Residue card's title row, on only while a residue is pinned."""

    LONG = "A_very_long_structure_name_that_has_no_spaces_to_break_at_and_goes_on_and_on_for_a_good_while"

    def open_two(self):
        self.write_with(A, "information_content")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)

    def buttons(self):
        return self.script("""
            const head = document.querySelector('.sec-head:has(#h-residue)').getBoundingClientRect();
            const rect = id => { const r = document.getElementById(id).getBoundingClientRect(); return [r.left, r.right, r.top, r.bottom]; };
            const title = document.getElementById('h-residue').getBoundingClientRect();
            return {state: [...document.querySelectorAll('#residue-zoom, #residue-clear')].map(b => b.getAttribute('aria-disabled')),
                    inRow: ['residue-zoom', 'residue-clear'].every(id => document.getElementById(id).closest('.sec-head') === document.querySelector('.sec-head:has(#h-residue)')),
                    names: ['residue-zoom', 'residue-clear'].map(id => [document.getElementById(id).title, document.getElementById(id).getAttribute('aria-label'),
                        document.getElementById(id).querySelector('use').getAttribute('href'), document.getElementById(id).textContent.trim()]),
                    head: [head.left, head.right, head.top, head.bottom], title: [title.left, title.right, title.top, title.bottom],
                    zoom: rect('residue-zoom'), clear: rect('residue-clear'),
                    foot: document.querySelectorAll('.res-foot').length,
                    look: ['residue-zoom', 'residue-clear'].map(id => getComputedStyle(document.getElementById(id)).opacity),
                    reachable: ['residue-zoom', 'residue-clear'].map(id => { const b = document.getElementById(id); b.focus(); return document.activeElement === b; })};
            """)

    def test_they_are_in_the_title_row_at_its_right_greyed_out_with_nothing_shown_or_only_a_hovered_residue(self):
        self.open_two()
        found = self.buttons()
        self.assertEqual(found["state"], ["true", "true"])
        self.assertEqual(found["names"], [["Zoom to the pinned residue", "Zoom to the pinned residue", "#i-crosshair", ""], ["Clear", "Clear", "#i-x", ""]])
        self.assertTrue(found["inRow"])
        self.assertEqual(found["foot"], 0)
        self.assertEqual(found["look"], ["0.4", "0.4"])
        self.assertEqual(found["reachable"], [True, True])
        # In the row: beside the heading (not under it), at the right of the card's head.
        self.assertGreaterEqual(found["zoom"][0], found["title"][1] - 0.5)
        self.assertLessEqual(found["clear"][1], found["head"][1] + 0.5)
        self.assertLess(found["zoom"][1], found["clear"][0] + 0.5)
        self.assertLess(abs((found["zoom"][2] + found["zoom"][3]) - (found["title"][2] + found["title"][3])), 6)
        # A hovered residue fills the card, and is not a choice yet: both stay off, and a press does nothing.
        self.script("await hoverOn(0, 5); window.__calls = []; const real = plugin.managers.camera.focusLoci; plugin.managers.camera.focusLoci = (...a) => window.__calls.push(1); return 1")
        self.assertIn("Ile 6", self.script("return residueNow().head"))
        self.assertEqual(self.buttons()["state"], ["true", "true"])
        self.script("document.getElementById('residue-zoom').click(); document.getElementById('residue-clear').click(); return 1")
        self.assertEqual(self.script("return window.__calls"), [])
        self.assertIn("Ile 6", self.script("return residueNow().head"))

    def test_with_a_pinned_residue_both_are_on_and_each_does_its_job(self):
        self.open_two()
        self.script("await clickOn(0, 5); window.__calls = []; plugin.managers.camera.focusLoci = (...a) => window.__calls.push(1); return 1")
        found = self.buttons()
        self.assertEqual(found["state"], ["false", "false"])
        self.assertEqual(found["look"], ["1", "1"])
        self.script("document.getElementById('residue-zoom').click(); return 1")
        self.assertEqual(self.script("return window.__calls"), [1])
        self.assertEqual(self.script("return pinnedResidue.number"), 6)
        self.script("document.getElementById('residue-clear').click(); await new Promise(r => setTimeout(r, 300)); return 1")
        self.assertIsNone(self.script("return pinnedResidue"))
        self.assertEqual(self.buttons()["state"], ["true", "true"])
        self.assertEqual(self.console_errors(), [])

    def test_they_work_from_the_keyboard_and_the_focus_is_never_lost_to_the_page(self):
        self.open_two()
        widget = self.view.focusProxy() or self.view
        self.script("await clickOn(0, 5); document.getElementById('residue-clear').focus(); return 1")
        QTest.keyClick(widget, Qt.Key_Return)
        self.pump(0.4)
        self.assertIsNone(self.script("return pinnedResidue"))
        self.assertEqual(self.script("return document.activeElement.id"), "residue-clear")

    def test_they_are_in_the_title_row_in_the_drawer_layout_and_in_the_strip_of_a_phone(self):
        self.open_two()
        self.script("await clickOn(0, 5); return 1")
        for width, tab in ((900, None), (600, "residue")):
            with self.subTest(width=width):
                self.view.resize(width, 800)
                self.addCleanup(self.view.resize, 1300, 800)
                self.settle(300)
                self.script("setPanel(true); %s return 1" % ("setTab('%s');" % tab if tab else ""))
                self.settle(300)
                found = self.buttons()
                self.assertTrue(found["inRow"])
                for key in ("zoom", "clear"):
                    box = found[key]
                    self.assertGreater(box[1] - box[0], 20)
                    self.assertGreaterEqual(box[0], found["head"][0] - 0.5)
                    self.assertLessEqual(box[1], found["head"][1] + 0.5)
                    self.assertLessEqual(box[3], found["head"][3] + 0.5)
                    # One row with the heading.
                    self.assertLess(abs((box[2] + box[3]) - (found["title"][2] + found["title"][3])), 6)

    THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU",
             "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}

    def write_with_secondary(self, name):
        """The fixture structure with a HELIX record over residues 1 to 11 and a SHEET record over 15 to 20 (it has none of its own)."""
        self.write_with(name, "information_content")
        path = os.path.join(self.directory, name)
        with open(path, encoding="utf-8") as stream:
            text = stream.read()
        word = lambda number: self.THREE[fixtures.FIRST_SEQUENCE[number - 1]]
        helix = "HELIX  %3d %3s %3s %1s %4d  %3s %1s %4d  %2d%30s %5d" % (1, "H1", word(1), "A", 1, word(11), "A", 11, 1, "", 11)
        sheet = "SHEET  %3d %3s%2d %3s %1s%4d  %3s %1s%4d  %2d" % (1, "S1", 1, word(15), "A", 15, word(20), "A", 20, 0)
        with open(path, "w", encoding="utf-8", newline="") as stream:
            stream.write(helix + chr(10) + sheet + chr(10) + text)

    def pin_with_name(self, name, number=6):
        self.write_with_secondary(name)
        self.open_scene(name)
        self.script("await clickOn(0, %d); return 1" % (number - 1))

    HEAD_GEOMETRY = """
        const tile = document.querySelector('#residue-body .res-head .ss-tile');
        if (!tile) return null;
        const box = node => { const r = node.getBoundingClientRect(); return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height}; };
        const picture = tile.querySelector('svg'), word = tile.querySelector('.ss-word'), title = document.querySelector('#residue-body .res-title');
        const range = document.createRange(); range.selectNodeContents(word);
        return {tile: box(tile), picture: box(picture), word: box(word), title: box(title), body: box(document.getElementById('residue-body')),
                text: tile.textContent, use: picture.querySelector('use').getAttribute('href'), hidden: picture.getAttribute('aria-hidden'),
                lines: new Set([...range.getClientRects()].map(r => Math.round(r.top))).size, panel: document.getElementById('side').getBoundingClientRect().width,
                titleLines: Math.round(box(title).height / 17), shrink: getComputedStyle(tile).flexShrink, wrap: getComputedStyle(word).whiteSpace};
        """

    def test_the_card_head_has_a_tile_with_a_picture_of_the_kind_and_its_word(self):
        # Residue 6 is in the fixture's helix and residue 17 in its strand; it has no turn, so that picture is made with a made-up residue.
        self.write_with_secondary(A)
        self.open_scene(A)
        for number, kind, word in ((6, "helix", "Helix"), (17, "strand", "Strand")):
            with self.subTest(kind=kind):
                self.script("await clickOn(0, %d); return 1" % (number - 1))
                found = self.script(self.HEAD_GEOMETRY)
                self.assertIsNotNone(found, "the fixture has no secondary structure for this residue")
                self.assertEqual((found["use"], found["text"], found["hidden"]), ("#i-ss-%s-art" % kind, word, "true"), found)
        made_up = self.script("""
            const out = {};
            for (const [kind, word] of Object.entries({helix: 'Helix', strand: 'Strand', turn: 'Turn'})) {
                const tile = secondaryTile({secondary: word, secondaryKind: kind});
                out[kind] = [tile.className, tile.querySelector('svg use').getAttribute('href'), tile.querySelector('.ss-word').textContent,
                             tile.querySelector('svg').getAttribute('aria-hidden'), tile.firstChild.tagName.toLowerCase()];
            }
            return out;
            """)
        for kind, word in (("helix", "Helix"), ("strand", "Strand"), ("turn", "Turn")):
            self.assertEqual(made_up[kind], ["ss-tile", "#i-ss-%s-art" % kind, word, "true", "svg"])

    def test_the_tile_has_the_picture_on_top_and_the_word_centred_under_it_and_the_title_gives_way(self):
        for language in ("en", "zh_CN"):
            with self.subTest(language=language):
                if language != "en":
                    from desktop.Desktop_App import install_translations

                    install_translations(self.app, language)
                    self.addCleanup(install_translations, self.app, None)
                widths, lines = {}, {}
                for name in ("A.pdb", self.LONG + ".pdb"):
                    for number in (6, 17):
                        self.pin_with_name(name, number)
                        found = self.script(self.HEAD_GEOMETRY)
                        self.assertIsNotNone(found, "the fixture has no secondary structure for this residue")
                        picture, word, tile, title = found["picture"], found["word"], found["tile"], found["title"]
                        self.assertGreaterEqual(picture["width"], 44, found)
                        self.assertGreaterEqual(picture["height"], 44, found)
                        # The word starts at or below the picture's bottom, on one line, centred under it.
                        self.assertGreaterEqual(word["top"], picture["bottom"] - 0.5, found)
                        self.assertEqual((found["lines"], found["shrink"], found["wrap"]), (1, "0", "nowrap"), found)
                        self.assertLess(abs((word["left"] + word["right"]) / 2 - (picture["left"] + picture["right"]) / 2), 1, found)
                        # The tile is inside the card, and the title, which gives way, never runs under it.
                        self.assertGreaterEqual(tile["left"], found["body"]["left"] - 0.5, found)
                        self.assertLessEqual(tile["right"], found["body"]["right"] + 0.5, found)
                        self.assertLessEqual(title["right"], tile["left"] + 0.5, found)
                        self.assertGreaterEqual(tile["width"], 64, found)
                        self.assertEqual(found["panel"], 340)
                        widths[(name, number)], lines[(name, number)] = tile["width"], found["titleLines"]
                # The tile is as wide with a long name as with a short one, and the name is what took more lines.
                for number in (6, 17):
                    self.assertEqual(widths[("A.pdb", number)], widths[(self.LONG + ".pdb", number)])
                    self.assertGreater(lines[(self.LONG + ".pdb", number)], lines[("A.pdb", number)])

    def test_a_residue_with_no_secondary_structure_has_no_tile(self):
        self.write_with_secondary(A)
        self.open_scene(A)
        # Residue 13 is between the helix (1 to 11) and the strand (15 to 20) and is in neither.
        self.script("await clickOn(0, 12); return 1")
        found = self.script("return {head: !!document.querySelector('#residue-body .res-head'), tile: !!document.querySelector('#residue-body .ss-tile'), label: !!document.querySelector('#residue-body .ss-label')}")
        self.assertEqual(found, {"head": True, "tile": False, "label": False})

    def test_the_hover_box_keeps_its_small_icon_before_its_word_on_one_line(self):
        self.write_with_secondary(A)
        self.open_scene(A)
        found = self.script("""
            const kinds = {helix: 'Helix', strand: 'Strand', turn: 'Turn'};
            const out = {};
            for (const [kind, word] of Object.entries(kinds)) {
                const label = secondaryLabel({secondary: word, secondaryKind: kind});
                out[kind] = [label.querySelector('use').getAttribute('href'), label.textContent, label.firstChild.getAttribute('aria-hidden')];
            }
            const flags = molstar.lib.structure.Types.SecondaryStructureType.Flag;
            out.kinds = [secondaryKind(flags.Helix), secondaryKind(flags.Beta), secondaryKind(flags.Turn), secondaryKind(0)];
            return out;
            """)
        self.assertEqual(found["helix"], ["#i-ss-helix", "Helix", "true"])
        self.assertEqual(found["strand"], ["#i-ss-strand", "Strand", "true"])
        self.assertEqual(found["turn"], ["#i-ss-turn", "Turn", "true"])
        self.assertEqual(found["kinds"], ["helix", "strand", "turn", None])
        # The hover box says it on one line, in its Secondary structure row (residue 6 is in the helix, residue 17 in the strand).
        row_script = """
            const cell = [...document.querySelectorAll('#tip tr')].find(r => r.children[0].textContent === 'Secondary structure');
            if (!cell) return null;
            const label = cell.children[1].querySelector('.ss-label'), range = document.createRange(); range.selectNodeContents(label.lastChild);
            return [label.querySelector('svg use').getAttribute('href'), cell.children[1].textContent, new Set([...range.getClientRects()].map(r => Math.round(r.top))).size,
                    label.querySelector('svg').getBoundingClientRect().width, !!document.querySelector('#tip .ss-tile')];
            """
        self.script("await hoverOn(0, 5); return 1")
        self.assertEqual(self.script(row_script), ["#i-ss-helix", "Helix", 1, 14, False])
        self.script("await hoverOn(0, 16); return 1")
        self.assertEqual(self.script(row_script), ["#i-ss-strand", "Strand", 1, 14, False])

    def test_the_pictures_are_in_the_sprite_in_the_order_they_are_drawn_and_the_small_glyphs_are_the_new_ones(self):
        self.write_with_secondary(A)
        self.open_scene(A)
        found = self.script("""
            const ids = ['i-ss-helix-art', 'i-ss-strand-art', 'i-ss-turn-art', 'i-ss-helix', 'i-ss-strand', 'i-ss-turn'];
            const symbol = id => document.getElementById(id);
            const paths = id => [...symbol(id).querySelectorAll('path')].map(p => ({classes: p.getAttribute('class').split(' ').sort(), d: p.getAttribute('d')}));
            const small = id => symbol(id).querySelector('path').getAttribute('d');
            const art = ['i-ss-helix-art', 'i-ss-strand-art', 'i-ss-turn-art'];
            return {ids: ids.map(id => [id, !!symbol(id), symbol(id)?.getAttribute('viewBox')]),
                    helix: paths('i-ss-helix-art'), strand: paths('i-ss-strand-art'), turn: paths('i-ss-turn-art'),
                    small: ['i-ss-strand', 'i-ss-turn'].map(small),
                    fills: art.map(id => getComputedStyle(symbol(id).querySelector('path')).strokeWidth)};
            """)
        self.assertEqual(found["ids"], [["i-ss-helix-art", True, "0 0 48 48"], ["i-ss-strand-art", True, "0 0 48 48"], ["i-ss-turn-art", True, "0 0 48 48"],
                                        ["i-ss-helix", True, "0 0 24 24"], ["i-ss-strand", True, "0 0 24 24"], ["i-ss-turn", True, "0 0 24 24"]])
        # The helix is one thin chain line, then pieces of a tube. Every piece is two paths with the same line, the wide edge and the narrower core over it.
        helix = found["helix"]
        self.assertEqual([p["classes"] for p in helix].count(["ss-chain"]), 1)
        self.assertEqual(helix[0]["classes"], ["ss-chain"])
        tube = helix[1:]
        for edge, core in zip(tube[0::2], tube[1::2]):
            self.assertIn("ss-edge", edge["classes"])
            self.assertIn("ss-core", core["classes"])
            self.assertEqual(edge["d"], core["d"])
            self.assertEqual([c for c in edge["classes"] if c != "ss-edge"], [c for c in core["classes"] if c != "ss-core"])
        self.assertEqual(len(tube), 14)
        # The drawing order is the whole point: every far half of a turn is drawn before every near one (an edge that is neither far nor an end cap),
        # so that nothing shows that lies behind another part. A far path after a near one would draw over it.
        far = [i for i, p in enumerate(helix) if "ss-far" in p["classes"]]
        near = [i for i, p in enumerate(helix) if "ss-edge" in p["classes"] and "ss-far" not in p["classes"] and "ss-cap" not in p["classes"]]
        self.assertEqual((len(far), len(near)), (4, 3))
        self.assertLess(max(far), min(near), helix)
        # The strand is a hairpin: one chain path of three pieces (the line over the top and a line at each lower end), then two arrows, one up and one down.
        strand = found["strand"]
        self.assertEqual([p["classes"] for p in strand], [["ss-chain"], ["ss-front"], ["ss-front"]])
        self.assertEqual(strand[0]["d"].count("M"), 3)
        self.assertTrue(strand[1]["d"].startswith("M13 10 "), strand[1])
        self.assertTrue(strand[2]["d"].startswith("M35 41 "), strand[2])
        # The turn is one tube, round at its two ends, with a short straight piece at each end.
        turn = found["turn"]
        self.assertEqual([p["classes"] for p in turn], [["ss-edge", "ss-round"], ["ss-core", "ss-round"]])
        self.assertEqual(turn[0]["d"], turn[1]["d"])
        self.assertTrue(turn[0]["d"].startswith("M5 39H12") and turn[0]["d"].endswith("H43"), turn[0])
        # The small glyphs of the hover box: the strand as two arrows in opposite directions, the turn with a short line at each end.
        self.assertEqual(found["small"], ["M7 20V4M3 8l4-4 4 4M17 4v16M13 16l4 4 4-4", "M2 20h3a2 2 0 0 0 2-2v-7a5 5 0 0 1 10 0v7a2 2 0 0 0 2 2h3"])
        # The rules reach the paths (the widths of the tube are the rules', not the symbol's 2).
        self.assertEqual(found["fills"][0], "2px")

    def test_the_rules_that_style_the_pictures_name_the_paths_class_alone(self):
        """The copy of a symbol that a <use> draws is matched inside the symbol only by a current Chromium (Edge, Chrome: the browser the Fold View
        page may be opened in), while this QtWebEngine also matches a rule that reaches the path through the sprite or the place of use. A rule
        written with an ancestor or another class would style the pictures here and leave them unfilled and thin there."""
        self.write_with_secondary(A)
        self.open_scene(A)
        rules = self.script("""
            const classes = ['ss-front', 'ss-chain', 'ss-edge', 'ss-core', 'ss-far', 'ss-cap', 'ss-round'];
            const found = [];
            for (const sheet of document.styleSheets) {
                let list; try { list = sheet.cssRules; } catch (error) { continue; }
                const walk = rules => { for (const rule of rules) {
                    if (rule.selectorText && classes.some(name => rule.selectorText.includes('.' + name))) found.push(rule.selectorText);
                    if (rule.cssRules) walk(rule.cssRules);
                } };
                walk(list);
            }
            return found;
            """)
        self.assertGreaterEqual(len(rules), 6, rules)
        for text in rules:
            for selector in text.split(","):
                self.assertRegex(selector.strip(), r"^(\.ss-(front|chain|edge|core|far|cap|round))+$", text)


if __name__ == "__main__":
    unittest.main()
