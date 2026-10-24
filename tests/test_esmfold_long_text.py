# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A sweep of the Fold View page for text that has no spaces to break at: every place a name, a file name, a log path or a reason is shown.

tests/test_esmfold_layout.py lays out the places a long name made trouble in (the tooltip, the legends, the Color by chips); this
one gives the whole page names of 120 characters and more, a long MSA file name, a file opened from disk, a run in progress and
failed folds with long reasons, and checks every part of the page that shows them (the Structures rows and their details, the
Residue card, the sequence strip, the status line, the folding card, the dialogs and the legends, also the one drawn into a saved
image) at 1300, 1040 and 600 px, in English and Simplified Chinese, light and dark. Nothing may lie outside its card; text that
is cut with an ellipsis has the whole of it on hover.
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
from tests.esmfold_long_names import (LONG_LOG, MSA_120, NAME_120, OPENED_NAME, OPENED_PLAIN, SECOND_NAME, UNDERSCORE_NAME, USER_NAME)
from tests.test_esmfold_layout import OVERFLOW, LongNameTestCase


LONG_LABEL = "Information_Content_after_the_small-sample_correction_of_logo"
LONG_REASON = ("The folding worker stopped: CUDA out of memory while folding " + NAME_120 + " (sequence of 2204 residues); log at " + LONG_LOG)


class LongTextSweepTests(LongNameTestCase):
    """Every place a structure's name, a file name, a log path or a reason is shown, with names that have no spaces."""

    WIDTHS = (1300, 1040, 600)

    def language(self, code):
        if code == "zh_CN":
            from desktop.Desktop_App import install_translations

            install_translations(self.app, "zh_CN")
            self.addCleanup(install_translations, self.app, None)

    def full_scene(self):
        """Folded structures (one without layers), a file opened from disk, a run in progress with two failures, a pinned residue."""
        self.write_with(NAME_120 + ".pdb", shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.scene(USER_NAME, SECOND_NAME, msa=MSA_120, label_a=LONG_LABEL)
        self.push(NAME_120 + ".pdb")
        for name in (OPENED_NAME, OPENED_PLAIN):
            self.script("openStructureFiles([new File([%s], %s)]); await structureQueue; return 1" % (
                json.dumps(fixtures.helix_pdb(fixtures.FIRST_SEQUENCE, fixtures.plddt_values(33), shift=(1.0, 2.0, 0.0))),
                json.dumps(name + ".pdb")))
            self.settle(400)
        self.script("""
            handleJobEvent({event: 'started', total: 3});
            handleJobEvent({event: 'progress', total: 3, index: 2, name: %s});
            handleJobEvent({event: 'structure_failed', name: %s, reason: %s, log: %s});
            handleJobEvent({event: 'structure_failed', name: %s, reason: %s, log: %s});
            return 1;""" % (json.dumps(UNDERSCORE_NAME), json.dumps(USER_NAME), json.dumps(LONG_REASON), json.dumps(LONG_LOG),
                            json.dumps(UNDERSCORE_NAME), json.dumps(LONG_LOG), json.dumps(LONG_LOG)))
        # Every row open, the opened file shows pLDDT too (which says its colors are B-factors), and a residue is pinned.
        self.script("""
            await clickOn(0, 5);
            for (const n of [1, 2, 3, 4, 5]) document.querySelector('#struct-list .srow:nth-child(' + n + ') .exp-btn').click();
            for (const index of [3, 4]) await setStructureColoring(loadedStructures()[index].cell.transform.ref, 'plddt');
            return 1;""")
        self.settle(500)

    def check_side_panel(self, context):
        for coloring in ("plddt", "information_content"):
            self.choose(coloring)
            for width in self.WIDTHS:
                for tab in ("structures", "color", "residue"):
                    for theme in (("light", "dark") if width != 1040 else ("light",)):
                        self.script("applyTheme(%s, false); return 1" % json.dumps(theme))
                        self.at(width, tab)
                        with self.subTest(context=context, coloring=coloring, width=width, tab=tab, theme=theme):
                            self.assertEqual(self.overflowing("#side"), [])
            # The legend over the canvas (which a canvas-less engine hides, so the test says it is ready), and the legend's unit.
            for width in (1300, 600):
                self.at(width)
                with self.subTest(context=context, coloring=coloring, width=width, legend="on the canvas"):
                    self.assertEqual(self.script("""
                        document.body.dataset.view = 'ready';
                        const found = overflowing('#canvas-legend');
                        updateView();
                        return found;"""), [])

    def test_the_side_panel_has_nothing_outside_its_cards_in_english(self):
        self.full_scene()
        self.check_side_panel("en")

    def test_the_side_panel_has_nothing_outside_its_cards_in_simplified_chinese(self):
        self.language("zh_CN")
        self.full_scene()
        self.check_side_panel("zh_CN")

    def test_a_legends_unit_stays_whole_beside_a_heading_that_breaks_anywhere(self):
        # The heading of a layer whose name is as long as can be, and "(%)" beside it: not "(" "%" ")" on three lines.
        label = "Information_Content_after_the_small-sample_correction_of_the_logo_bits_over_all_columns_of_the_alignment"
        self.scene(USER_NAME, label_a=label)
        for width in self.WIDTHS:
            with self.subTest(width=width):
                self.at(width)
                self.choose("information_content")
                unit = self.script("""
                    const small = document.querySelector('#color-legend .t small');
                    const range = document.createRange(); range.selectNodeContents(small);
                    return {text: small.textContent, lines: new Set([...range.getClientRects()].map(r => Math.round(r.top))).size,
                            width: small.getBoundingClientRect().width, heading: document.querySelector('#color-legend .t').firstChild.textContent};""")
                self.assertEqual((unit["text"], unit["lines"], unit["heading"]), ("(%)", 1, label))
                self.assertGreater(unit["width"], 15)

    def test_a_structures_details_break_a_long_name_anywhere_and_keep_each_value_whole(self):
        # "Aligned to <name>" is a term of the details; its value is "RMSD 0.3 Å" beside it.
        self.scene(USER_NAME, SECOND_NAME)
        self.script("document.querySelector('#struct-list .srow:nth-child(2) .exp-btn').click(); return 1")
        self.settle(300)
        for width in self.WIDTHS:
            with self.subTest(width=width):
                self.at(width, "structures")
                facts = self.script("""
                    const kv = document.querySelector('#struct-list .srow:nth-child(2) .kv');
                    const lines = node => { const range = document.createRange(); range.selectNodeContents(node);
                        return new Set([...range.getClientRects()].filter(r => r.width > 0).map(r => Math.round(r.top))).size; };
                    const terms = [...kv.querySelectorAll('dt')], values = [...kv.querySelectorAll('dd')];
                    return {terms: terms.map(t => [t.textContent, lines(t)]), values: values.map(v => [v.textContent, lines(v)]),
                            rights: values.map(v => Math.round(v.getBoundingClientRect().right * 10) / 10), box: kv.getBoundingClientRect().right};""")
                self.assertEqual([value[1] for value in facts["values"]], [1] * len(facts["values"]), facts)
                self.assertEqual(len(set(facts["rights"])), 1, facts)
                self.assertLessEqual(facts["rights"][0], facts["box"] + 0.5)
                aligned = [term for term in facts["terms"] if term[0].startswith("Aligned to")]
                self.assertEqual(aligned[0][0], f"Aligned to {USER_NAME}")
                self.assertGreater(aligned[0][1], 1)
                self.assertEqual(self.overflowing("#struct-list"), [])

    def test_a_name_too_long_for_its_row_is_cut_with_an_ellipsis_whole_on_hover_and_whole_in_the_open_row(self):
        self.scene(USER_NAME, SECOND_NAME)
        self.at(1300, "structures")
        first = self.script("""
            const name = document.querySelector('#struct-list .srow:nth-child(1) .srow-name');
            return {title: name.title, text: name.textContent, cut: name.scrollWidth > name.clientWidth, ellipsis: getComputedStyle(name).textOverflow,
                    lines: Math.round(name.getBoundingClientRect().height / parseFloat(getComputedStyle(name).lineHeight))};""")
        self.assertEqual((first["title"], first["text"], first["ellipsis"], first["lines"]), (USER_NAME, USER_NAME, "ellipsis", 1))
        self.assertTrue(first["cut"])
        # Open, the row has the whole name, breaking anywhere.
        self.script("document.querySelector('#struct-list .srow:nth-child(1) .exp-btn').click(); return 1")
        self.settle(200)
        opened = self.script("""
            const name = document.querySelector('#struct-list .srow:nth-child(1) .srow-name');
            return {text: name.textContent, cut: name.scrollWidth > name.clientWidth, lines: Math.round(name.getBoundingClientRect().height / parseFloat(getComputedStyle(name).lineHeight))};""")
        self.assertEqual((opened["text"], opened["cut"]), (USER_NAME, False))
        self.assertGreater(opened["lines"], 1)

    def test_a_failure_row_keeps_its_chip_beside_a_name_that_breaks_anywhere_and_shows_the_whole_reason(self):
        self.open_page()
        self.js(OVERFLOW)
        self.script("""
            handleJobEvent({event: 'structure_failed', name: %s, reason: %s, log: %s});
            return 1;""" % (json.dumps(UNDERSCORE_NAME), json.dumps(LONG_REASON), json.dumps(LONG_LOG)))
        self.settle(300)
        for width in self.WIDTHS:
            with self.subTest(width=width):
                self.at(width, "structures")
                row = self.script("""
                    const row = document.querySelector('#struct-list .srow.failed');
                    const box = row.getBoundingClientRect(), chip = row.querySelector('.chip').getBoundingClientRect();
                    const name = row.querySelector('.srow-name span');
                    const buttons = [...row.querySelectorAll('.row-actions button')].map(button => button.textContent);
                    return {name: name.textContent, reason: row.querySelector('.srow-sub').textContent, chipInside: chip.right <= box.right && chip.left >= box.left,
                            chipText: row.querySelector('.chip').textContent, buttons};""")
                self.assertEqual((row["name"], row["reason"], row["chipInside"], row["chipText"]), (UNDERSCORE_NAME, LONG_REASON, True, "Failed"))
                self.assertEqual(row["buttons"], ["Copy log path", "Dismiss"])
                self.assertEqual(self.overflowing("#struct-list"), [])

    def test_the_status_line_breaks_any_name_path_or_reason_inside_itself_with_or_without_an_action(self):
        self.open_page()
        self.js(OVERFLOW)
        texts = [f"Folding {UNDERSCORE_NAME} failed: {LONG_REASON}", f"Loaded and aligned {UNDERSCORE_NAME}.", f"Opened {OPENED_NAME}.",
                 f"Could not copy {LONG_LOG}", f"{USER_NAME}: now click the second residue."]
        for width in self.WIDTHS:
            self.at(width)
            for text in texts:
                for options in ("{error: true}", "{label: 'Undo', run: () => {}}"):
                    with self.subTest(width=width, text=text[:20], options=options):
                        self.script("showStatus(%s, 0, %s); return 1" % (json.dumps(text), options))
                        self.assertEqual(self.overflowing("#toast"), [])
                        self.assertEqual(self.script("return document.querySelector('#toast span').textContent"), text)
                        action = self.script("""
                            const button = document.querySelector('#toast .toast-action');
                            if (!button) return null;
                            const range = document.createRange(); range.selectNodeContents(button);
                            return {text: button.textContent, lines: new Set([...range.getClientRects()].map(r => Math.round(r.top))).size,
                                    width: button.getBoundingClientRect().width, scrolls: button.scrollWidth > button.clientWidth};""")
                        if "label" in options:
                            self.assertEqual((action["text"], action["lines"], action["scrolls"]), ("Undo", 1, False))
                            self.assertGreater(action["width"], 30)

    def test_the_folding_card_names_the_sequence_whole_and_a_start_failure_its_reason(self):
        self.open_page()
        self.js(OVERFLOW)
        self.script("handleJobEvent({event: 'started', total: 3}); handleJobEvent({event: 'progress', total: 3, index: 1, name: %s}); return 1" % json.dumps(UNDERSCORE_NAME))
        self.settle(300)
        for width in self.WIDTHS:
            with self.subTest(width=width):
                self.at(width, "structures")
                # (A canvas-less engine is never in the state "folding", so the test says it is.)
                self.assertEqual(self.script("""
                    document.body.dataset.view = 'folding';
                    const found = overflowing('.empty[data-for=folding] .empty-card');
                    updateView();
                    return found;"""), [])
                self.assertEqual(self.script("return document.getElementById('folding-name').textContent"), UNDERSCORE_NAME)
                # The row of the run in the Structures card cuts the name with an ellipsis and has the whole of it on hover.
                self.assertEqual(self.overflowing("#struct-list"), [])
                self.assertEqual(self.script("return document.querySelector('#struct-list .srow.pending .srow-name').title"), UNDERSCORE_NAME)
                label = self.script("""
                    const label = document.querySelector('#struct-list .srow.pending .row-label');
                    const name = label.parentElement.getBoundingClientRect(), box = label.getBoundingClientRect();
                    return {ellipsis: getComputedStyle(label).textOverflow, cut: label.scrollWidth > label.clientWidth, inside: box.right <= name.right + 0.5,
                            text: label.textContent};""")
                self.assertEqual((label["ellipsis"], label["cut"], label["inside"], label["text"]), ("ellipsis", True, True, UNDERSCORE_NAME))
        self.script("failStart(new Error(%s)); return 1" % json.dumps("Could not load " + LONG_LOG))
        for width in (1300, 600):
            with self.subTest(width=width, state="Mol* did not start"):
                self.at(width)
                self.assertEqual(self.overflowing(".empty[data-for=failed] .empty-card"), [])

    def test_the_sequence_strips_select_has_the_whole_name_on_hover(self):
        self.scene(USER_NAME, SECOND_NAME)
        self.at(1300)
        select = self.script("""
            const select = document.getElementById('seq-structure');
            return {title: select.title, shown: select.selectedOptions[0].textContent, hidden: select.hidden, width: select.getBoundingClientRect().width};""")
        self.assertEqual(select["title"], select["shown"])
        self.assertEqual(select["title"], f"{USER_NAME}")
        self.assertFalse(select["hidden"])
        self.assertLessEqual(select["width"], 260.5)
        self.assertEqual(self.overflowing("#seqbar"), [])

    def test_the_dialogs_and_the_session_menu_hold_no_name_and_stay_in_their_boxes(self):
        self.scene(USER_NAME, SECOND_NAME)
        for name, selector, opener in (
                ("help", "#help-modal .modal", "document.getElementById('help-modal').classList.add('open')"),
                ("image", "#image-modal .modal", "document.getElementById('image-modal').classList.add('open')"),
                ("confirm", "#confirm-modal .modal", "askToConfirm('Clear the scene?', 'This removes every structure.', 'Clear scene', () => {})"),
                ("menu", "#session-menu", "document.getElementById('session-menu').classList.add('open')")):
            for width in self.WIDTHS:
                with self.subTest(dialog=name, width=width):
                    self.at(width)
                    self.script(opener + "; return 1")
                    self.assertEqual(self.overflowing(selector), [])
                    self.script("for (const node of document.querySelectorAll('.modal-overlay, .menu')) node.classList.remove('open'); return 1")

    def test_the_legend_drawn_into_a_saved_image_is_cut_to_fit_not_run_out_of_the_picture(self):
        # drawLegend(context, width, height, k, model): at 1 and at 2 times the page's size, with a layer whose name is far too long.
        self.scene(USER_NAME, SECOND_NAME, label_a=LONG_LABEL + "_and_then_some_more_of_it_to_be_sure")
        for coloring in ("plddt", "information_content"):
            self.choose(coloring)
            for k, (width, height) in ((1, (600, 400)), (2, (1200, 800)), (1, (260, 300))):
                with self.subTest(coloring=coloring, k=k, picture=(width, height)):
                    drawn = self.script("""
                        const canvas = document.createElement('canvas');
                        canvas.width = %d; canvas.height = %d;
                        const context = canvas.getContext('2d');
                        const texts = [];
                        const original = context.fillText.bind(context);
                        context.fillText = (text, ...rest) => { texts.push(text); return original(text, ...rest); };
                        const box = drawLegend(context, %d, %d, %d, await colorModel());
                        return {box, texts};""" % (width, height, width, height, k))
                    box = drawn["box"]
                    self.assertGreaterEqual(box["x"], 0)
                    self.assertLessEqual(box["x"] + box["width"], width)
                    self.assertLessEqual(box["y"] + box["height"], height)
                    self.assertGreaterEqual(box["y"], 0)
                    self.assertLessEqual(box["width"], 220 * k + 1)
                    if coloring == "information_content":
                        # The heading is the one that is too long: it ends in an ellipsis, and starts as the layer's name does.
                        self.assertTrue(drawn["texts"][0].endswith("…"), drawn["texts"])
                        self.assertTrue(drawn["texts"][0].startswith("Information_Content"), drawn["texts"])
                    else:
                        self.assertEqual(drawn["texts"][0], "pLDDT")


if __name__ == "__main__":
    unittest.main()
