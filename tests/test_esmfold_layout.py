# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""How the Fold View page lays out text that has no spaces to break at: names and file names of a person's own.

The page's texts were made with short names in mind (protein_a, 11.pdb). A protein from a database comes with
a header of 150 characters, and the alignment it was folded with is a file name of the same kind, none of it with a
space. A name that does not fit must not push a value out of its box, break a label that must stay whole, or
run past the edge of a card; where the text matters it breaks anywhere to fit, and where space is tight it is cut with an
ellipsis and the whole of it is on hover. These tests give the page such names (tests/esmfold_long_names.py)
and measure where each text is laid out. Like the other page tests they open esmfold.html in Qt WebEngine over
the real server and Mol*.
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
from tests.esmfold_long_names import NAME_120, OPENED_NAME, SECOND_NAME, UNDERSCORE_NAME, USER_MSA, USER_NAME, write_long
from tests.test_esmfold_residue import ResidueTestCase

from PySide6.QtCore import QUrl  # noqa: E402  (after the harness has set the engine's environment)

# How the tooltip is laid out, measured in the page from where each text is drawn.
TIP_LAYOUT = """
window.tipLayout = function () {
    const tip = document.getElementById('tip');
    const stage = document.getElementById('stage').getBoundingClientRect();
    const box = tip.getBoundingClientRect();
    const label = tip.querySelector('.h b'), where = tip.querySelector('.h span');
    // The boxes of a node's own text (not of a swatch beside it): one top for a text on one line.
    const textBoxes = node => {
        const boxes = [];
        const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
        for (let text = walker.nextNode(); text; text = walker.nextNode()) {
            const range = document.createRange();
            range.selectNodeContents(text);
            boxes.push(...[...range.getClientRects()].filter(r => r.width > 0));
        }
        return boxes;
    };
    const linesOf = node => new Set(textBoxes(node).map(r => Math.round(r.top))).size;
    const lineHeight = parseFloat(getComputedStyle(where).lineHeight);
    const labelBox = label.getBoundingClientRect(), whereBox = where.getBoundingClientRect();
    const edgeOf = cell => Math.max(...textBoxes(cell).map(r => r.right));
    return {
        shown: tip.classList.contains('show'), width: box.width, inset: box.right - 11, stageWidth: stage.width,
        insideStage: box.left >= stage.left && box.right <= stage.right && box.top >= stage.top && box.bottom <= stage.bottom,
        label: label.textContent, labelLines: linesOf(label), labelHeight: labelBox.height, lineHeight,
        where: where.textContent, whereRight: whereBox.right, whereLeft: whereBox.left, labelRight: labelBox.right,
        whereLines: Math.round(whereBox.height / lineHeight), whereClipped: where.scrollHeight > where.clientHeight + 1,
        valueEdges: [...tip.querySelectorAll('tr td:last-child')].map(edgeOf),
        valueRights: [...tip.querySelectorAll('tr td:last-child')].map(cell => cell.getBoundingClientRect().right),
        rows: [...tip.querySelectorAll('tr')].map(row => [...row.children].map(cell => cell.textContent)),
        rowLines: [...tip.querySelectorAll('tr')].map(row => Math.max(...[...row.children].map(linesOf))),
        valueLines: [...tip.querySelectorAll('tr td:last-child')].map(linesOf),
        labelRights: [...tip.querySelectorAll('tr td:first-child')].map(cell => Math.max(...textBoxes(cell).map(r => r.right)))
    };
};
"""


# Where a text or a box of a part of the page is laid out beyond what holds it. `overflowing(selector)` lists, for the part
# the selector names (a card, the tooltip, the status line), every text and every element that lies outside it, or outside an
# element between it and the part that clips (overflow other than visible), or is cut with an ellipsis without the whole text on hover.
# An empty list is a part with nothing out of its place.
OVERFLOW = """
window.overflowing = function (selector) {
    const root = document.querySelector(selector);
    if (!root) return ['missing ' + selector];
    const limit = root.getBoundingClientRect();
    const found = [];
    const label = node => node.tagName.toLowerCase() + (node.id ? '#' + node.id : '') + [...node.classList].map(name => '.' + name).join('');
    const drawn = node => {
        const style = getComputedStyle(node), box = node.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden' && (box.width > 0 || box.height > 0);
    };
    const clips = node => getComputedStyle(node).overflowX !== 'visible';
    const cuts = node => clips(node) && getComputedStyle(node).textOverflow === 'ellipsis';
    const inner = node => { const box = node.getBoundingClientRect(); return {left: box.left + node.clientLeft, right: box.left + node.clientLeft + node.clientWidth}; };
    const within = (box, area) => box.left >= area.left - 0.5 && box.right <= area.right + 0.5;
    const above = node => { const list = []; for (let a = node.parentElement; a && a !== root; a = a.parentElement) list.push(a); return list; };
    const report = (node, what, box, area) => found.push(label(node) + ': ' + what + ' ' + Math.round(box.left) + '..' + Math.round(box.right)
        + ' not in ' + Math.round(area.left) + '..' + Math.round(area.right));
    const check = (node, boxes, ancestors) => {
        for (const box of boxes) {
            for (const holder of ancestors) if (clips(holder) && !within(box, inner(holder))) return report(node, 'beyond ' + label(holder), box, inner(holder));
            if (!within(box, limit)) return report(node, 'beyond the card', box, limit);
        }
    };
    for (const node of root.querySelectorAll('*')) {
        if (!drawn(node) || node.closest('.sr-only, svg')) continue;
        const ancestors = above(node);
        if (ancestors.some(cuts)) continue;
        check(node, [node.getBoundingClientRect()], ancestors);
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let text = walker.nextNode(); text; text = walker.nextNode()) {
        const element = text.parentElement;
        if (!text.textContent.trim() || !drawn(element) || element.closest('.sr-only, svg')) continue;
        const range = document.createRange();
        range.selectNodeContents(text);
        const boxes = [...range.getClientRects()].filter(box => box.width > 0 && box.height > 0);
        const ancestors = [element, ...above(element)];
        const cutter = ancestors.find(cuts);
        if (cutter) {
            // Cut with an ellipsis on purpose: the whole text is on hover.
            const titled = cutter.closest('[title]');
            if (!titled || !titled.title.includes(text.textContent.trim())) found.push(label(cutter) + ': cut with no whole text on hover: ' + text.textContent.trim().slice(0, 40));
            continue;
        }
        check(element, boxes, ancestors);
    }
    for (const holder of [root, ...root.querySelectorAll('*')]) {
        if (clips(holder) && holder.scrollWidth > holder.clientWidth + 1 && drawn(holder) && !cuts(holder)) {
            found.push(label(holder) + ': scrolls sideways ' + holder.scrollWidth + ' > ' + holder.clientWidth);
        }
    }
    return found;
};
"""


class LongNameTestCase(ResidueTestCase):
    """A scene of the structures `names`, which have the two usual layers, whose source is a long file name."""

    def scene(self, *names, layer=None, first=1, msa=USER_MSA, **labels):
        """The structures `names` (each of the two fixture sequences in turn), numbered from `first`, colored by `layer`.

        `labels` (label_a, label_b) name the two layers.
        """
        for index, name in enumerate(names):
            write_long(self.directory, name, fixtures.FIRST_SEQUENCE if index % 2 == 0 else fixtures.SECOND_SEQUENCE, msa,
                       shift=(3.0 * index, 1.0 * index, 0.0), wobble=0.4 * index, first=first, **labels)
        self.open_scene(*[name + ".pdb" for name in names])
        self.js(OVERFLOW)
        self.js(TIP_LAYOUT)
        if layer:
            self.choose(layer)

    def at(self, width, tab="color"):
        """The page with the window `width` pixels wide: the panel beside the canvas, a drawer over it, or a strip."""
        self.view.resize(width, 800)
        self.addCleanup(self.view.resize, 1300, 800)
        self.script("setPanel(true); setTab(%s); await new Promise(resolve => setTimeout(resolve, 450)); return 1" % json.dumps(tab))

    def overflowing(self, selector):
        return self.script("return overflowing(%s)" % json.dumps(selector))

    def reset_page(self):
        self.view.setUrl(QUrl("about:blank"))
        self.pump(0.3)


class TooltipLayoutTests(LongNameTestCase):
    """The residue's label stays on one line; a long name breaks anywhere to fit the box, in two lines at most, with an ellipsis."""

    def scene(self, *names, layer="information_content", **options):
        # Colored by a layer, so that the tip has a row of its value beside the pLDDT's.
        super().scene(*names, layer=layer, **options)

    def tip_at(self, width=1300, group=5, structure=0, x=200, y=120):
        self.view.resize(width, 800)
        self.addCleanup(self.view.resize, 1300, 800)
        self.settle(300)
        self.script("""
            const viewer = document.getElementById('viewer');
            const stage = document.getElementById('stage').getBoundingClientRect();
            viewer.dispatchEvent(new PointerEvent('pointermove', {clientX: stage.left + %d, clientY: stage.top + %d, bubbles: true}));
            await hoverOn(%d, %d);
            return 1;""" % (x, y, structure, group))
        return self.script("return tipLayout()")

    def assert_laid_out(self, layout, context, clipped):
        """`clipped`: whether the name is cut (it is too long for two lines)."""
        self.assertTrue(layout["shown"], context)
        self.assertTrue(layout["insideStage"], context)
        self.assertLessEqual(layout["width"], 360 + 0.01, context)
        self.assertLessEqual(layout["width"], layout["stageWidth"] - 16 + 0.01, context)
        # The residue ("Val 185") is on one line, whatever the name does.
        self.assertEqual(layout["labelLines"], 1, context)
        self.assertLess(layout["labelHeight"], 1.5 * layout["lineHeight"], context)
        # The name stays inside the box, beside the label, in two lines at most.
        self.assertLessEqual(layout["whereRight"], layout["inset"] + 0.5, context)
        self.assertGreaterEqual(layout["whereLeft"], layout["labelRight"], context)
        self.assertLessEqual(layout["whereLines"], 2, context)
        self.assertEqual(layout["whereClipped"], clipped, context)
        # No row of the table breaks, and its values end on one edge.
        self.assertEqual(set(layout["rowLines"]), {1}, (context, layout["rows"]))
        self.assertLessEqual(max(layout["valueEdges"]) - min(layout["valueEdges"]), 0.6, (context, layout["valueEdges"]))
        self.assertLessEqual(max(layout["valueRights"]), layout["inset"] + 0.5, context)

    def test_a_name_of_148_characters_wraps_inside_the_box_in_two_lines_and_the_residue_stays_whole(self):
        # Numbered from 180, so that the residues have three digits, as the report's "Val 185" has.
        self.scene(USER_NAME, SECOND_NAME, first=180)
        # (At 380 px, a phone's, the stage is narrower than the box's widest.)
        for width in (1300, 1040, 600, 380):
            with self.subTest(width=width):
                layout = self.tip_at(width)
                self.assert_laid_out(layout, f"the user's name at {width} px", clipped=True)
                self.assertEqual(layout["label"], "Ile 185")
                self.assertEqual(layout["whereLines"], 2)
                self.assertEqual(layout["where"], f"chain A · {USER_NAME}")
                # The layer's row and the pLDDT row line up whatever the name does.
                self.assertEqual([row[0] for row in layout["rows"]], ["pLDDT", "Information Content"])
        self.script("await hoverOff(); return 1")
        layout = self.tip_at(1300, structure=1)
        self.assert_laid_out(layout, "the second structure", clipped=True)
        self.assertEqual(layout["label"], "Val 185")

    def test_a_name_with_nothing_but_underscores_to_break_at_still_fills_two_lines_and_ends_in_an_ellipsis(self):
        self.assertNotRegex(UNDERSCORE_NAME, r"[-. ]")
        self.scene(UNDERSCORE_NAME)
        for width in (1300, 600):
            with self.subTest(width=width):
                layout = self.tip_at(width)
                self.assert_laid_out(layout, f"an underscore name at {width} px", clipped=True)
                self.assertEqual(layout["whereLines"], 2)
                self.assertEqual(layout["where"], f"chain A · {UNDERSCORE_NAME}")

    def test_a_name_that_fits_is_not_cut(self):
        self.scene("11", "22")
        layout = self.tip_at()
        self.assert_laid_out(layout, "a short name", clipped=False)
        self.assertEqual((layout["whereLines"], layout["where"], layout["label"]), (1, "chain A · 11", "Ile 6"))
        # The box is no wider than its content asks, within its least width.
        self.assertLess(layout["width"], 260)

    def test_the_values_keep_their_column_with_a_secondary_structure_row_and_a_pin(self):
        self.scene(USER_NAME, layer="information_content")
        self.script("await clickOn(0, 5); return 1")
        layout = self.tip_at(group=5)
        self.assert_laid_out(layout, "a pinned residue", clipped=True)
        self.assertEqual([row[0] for row in layout["rows"]][:2], ["pLDDT", "Information Content"])
        # The pin's note is the tip's last line and is inside it too.
        self.assertEqual(self.script("return tipNow().pinned"), "Pinned. Escape clears it.")

    def test_a_long_layer_name_breaks_in_its_own_cell_and_its_value_stays_whole_beside_it(self):
        # A layer's name is the data's: with spaces it breaks at them, without any it breaks anywhere; the value never breaks.
        for label in ("Information Content measured over the alignment columns after the small-sample correction was applied",
                      "Information_Content_measured_over_the_alignment_columns_after_the_small-sample_correction_was_applied"):
            with self.subTest(label=label[:30] + "...", spaces=" " in label):
                self.scene(USER_NAME, label_a=label)
                layout = self.tip_at()
                self.assertTrue(layout["insideStage"])
                self.assertEqual(layout["rows"][1], [label, "35.5 %"])
                self.assertEqual(layout["valueLines"], [1, 1])
                self.assertEqual(layout["labelLines"], 1)
                self.assertLessEqual(max(layout["labelRights"]), layout["inset"] + 0.5)
                self.assertLessEqual(max(layout["valueRights"]), layout["inset"] + 0.5)
                self.assertLessEqual(max(layout["valueEdges"]) - min(layout["valueEdges"]), 0.6)
                self.assertLessEqual(layout["width"], 360 + 0.01)
                self.script("await hoverOff(); return 1")
                self.reset_page()

    def test_the_whole_name_is_where_the_tip_sends_a_person_for_it(self):
        self.scene(USER_NAME)
        self.script("await clickOn(0, 5); return 1")
        # The Residue card has it whole (breaking anywhere), and so does the row of the Structures card.
        self.assertEqual(self.script("return residueNow().where"), f"chain A · {USER_NAME}")
        self.assertEqual(self.script("return document.querySelector('#struct-list .srow-name').textContent"), USER_NAME)

    def test_the_tip_comes_in_simplified_chinese_in_the_same_shape(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.scene(USER_NAME, SECOND_NAME, first=180)
        layout = self.tip_at()
        self.assert_laid_out(layout, "zh_CN", clipped=True)
        self.assertEqual(layout["label"], "Ile 185")


class LegendNoteTests(LongNameTestCase):
    """The notes and legends of the Color by card say a file's name and a structure's: they stay inside the card."""

    COLORINGS = ("plddt", "builtin:structure", "information_content", "same_residue_pct", "builtin:secondary-structure",
                 "builtin:residue-position")

    def test_the_source_note_breaks_anywhere_inside_its_card_and_keeps_the_whole_file_name(self):
        self.scene(USER_NAME, SECOND_NAME)
        for layer in ("information_content", "same_residue_pct"):
            self.choose(layer)
            for width in (1300, 1040, 600):
                with self.subTest(layer=layer, width=width):
                    self.at(width)
                    note = self.script("""
                        const details = document.querySelector('#color-legend details.src');
                        const span = details.querySelector('summary span');
                        const range = document.createRange();
                        range.selectNodeContents(span);
                        return {text: span.textContent, lines: new Set([...range.getClientRects()].map(r => Math.round(r.top))).size,
                                open: details.open, ellipsis: getComputedStyle(span).textOverflow};""")
                    # The whole name, as the Viewer words it, on as many lines as it takes, and nothing cut.
                    self.assertEqual(note["text"], f"From {USER_MSA}, computed when folded, using 3925 sequences")
                    self.assertGreater(note["lines"], 1)
                    self.assertEqual(note["ellipsis"], "clip")
                    self.assertEqual(self.overflowing("#color-legend"), [])
                    # Open, the layer's description is inside the card as well.
                    self.script("document.querySelector('#color-legend details.src').open = true; return 1")
                    self.assertEqual(self.overflowing("#color-legend"), [])
                    self.script("document.querySelector('#color-legend details.src').open = false; return 1")

    def test_every_coloring_of_the_card_with_long_names_has_nothing_outside_the_card(self):
        # Three structures that are folded (one has no residue data) and one opened from a file, whose names have no spaces to break at.
        self.write_with(NAME_120 + ".pdb", shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.scene(USER_NAME, SECOND_NAME)
        self.push(NAME_120 + ".pdb")
        self.script("openStructureFiles([new File([%s], %s)]); await structureQueue; return 1" % (
            json.dumps(fixtures.helix_pdb(fixtures.FIRST_SEQUENCE, fixtures.plddt_values(33), shift=(1.0, 2.0, 0.0))),
            json.dumps(OPENED_NAME + ".pdb")))
        self.settle(600)
        # The opened file shows pLDDT too, which says its colors are its B-factors (the note that holds its name).
        self.script("await setStructureColoring(loadedStructures()[3].cell.transform.ref, 'plddt'); return 1")
        self.settle(500)
        # (Secondary structure and Residue position are under More colorings.)
        self.script("document.querySelector('.more-toggle').click(); return 1")
        self.settle(150)
        for coloring in self.COLORINGS:
            self.choose(coloring)
            self.settle(150)
            for width in (1300, 1040, 600):
                with self.subTest(coloring=coloring, width=width):
                    self.at(width)
                    self.assertEqual(self.overflowing("section[data-tab=color]"), [])
                    # The legend drawn over the canvas (which a canvas-less engine hides, so the test says it is ready).
                    self.assertEqual(self.script("""
                        document.body.dataset.view = 'ready';
                        const found = overflowing('#canvas-legend');
                        updateView();
                        return found;"""), [])
        # The notes of the card do hold the names.
        self.choose("information_content")
        card = self.script("return card()")
        self.assertIn(NAME_120, card["noData"])
        self.assertIn(OPENED_NAME, card["noData"])
        self.assertTrue(any(NAME_120 in note for note in card["layerNotes"]), card["layerNotes"])
        self.choose("plddt")
        self.assertIn(OPENED_NAME, self.script("return card().caution"))
        self.assertEqual(self.console_errors(), [])


if __name__ == "__main__":
    unittest.main()
