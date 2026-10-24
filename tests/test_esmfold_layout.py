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

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests import esmfold_layer_fixtures as fixtures
from tests.esmfold_long_names import SECOND_NAME, UNDERSCORE_NAME, USER_MSA, USER_NAME, write_long
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


class TooltipLayoutTests(ResidueTestCase):
    """The residue's label stays on one line; a long name breaks anywhere to fit the box, in two lines at most, with an ellipsis."""

    def scene(self, *names, layer="information_content", first=1, **labels):
        """The structures `names` (each of the two fixture sequences in turn), numbered from `first`, colored by `layer`.

        `labels` (label_a, label_b) name the two layers.
        """
        for index, name in enumerate(names):
            write_long(self.directory, name, fixtures.FIRST_SEQUENCE if index % 2 == 0 else fixtures.SECOND_SEQUENCE, USER_MSA,
                       shift=(3.0 * index, 1.0 * index, 0.0), wobble=0.4 * index, first=first, **labels)
        self.open_scene(*[name + ".pdb" for name in names])
        self.js(TIP_LAYOUT)
        if layer:
            self.choose(layer)

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

    def reset_page(self):
        self.view.setUrl(QUrl("about:blank"))
        self.pump(0.3)

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


if __name__ == "__main__":
    unittest.main()
