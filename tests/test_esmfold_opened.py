# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A structure opened from a file in the Fold View page: its coloring (esmfold_color.js).

A structure the Viewer folded has its pLDDT in the B-factor column; one opened
from a file (the Open button, the Session menu, a drop) may have B-factors,
which are not a confidence. So it starts in its own color when the scene's
choice is pLDDT, with pLDDT one click away in its row; when the scene is
colored by Structure color or a layer it follows that, a layer it has no values
for greying it. The scene's choice is never changed by it, and a restored
session, an opened session file or Undo work the choice out without it.
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
from tests.test_esmfold_colors import A, B, GREY
from tests.test_esmfold_residue import ResidueTestCase

OWN = "builtin:structure"


class OpenedStructureTestCase(ResidueTestCase):
    def pdb(self):
        return json.dumps(fixtures.helix_pdb(fixtures.FIRST_SEQUENCE, fixtures.plddt_values(33), shift=(1.0, 2.0, 0.0), wobble=0.3))

    def bring_in(self, how, name="mine"):
        """Open a file named `name`.pdb the way `how` says ("open": the Open button's function, "drop": a file dropped on the view)."""
        if how == "open":
            self.script("openStructureFiles([new File([%s], '%s.pdb')]); await structureQueue; return 1" % (self.pdb(), name))
        else:
            self.script("""
                const transfer = new DataTransfer();
                transfer.items.add(new File([%s], '%s.pdb'));
                const target = document.getElementById('viewer');
                for (const type of ['dragenter', 'dragover', 'drop']) {
                    target.dispatchEvent(new DragEvent(type, {dataTransfer: transfer, bubbles: true, cancelable: true}));
                }
                await structureQueue;
                return 1;
                """ % (self.pdb(), name))
        self.settle(600)

    def scene(self, mode=None):
        """A folded structure with layers, colored as `mode` says (a coloring id; None leaves pLDDT)."""
        self.write_with(A, "information_content", "same_residue_pct")
        self.open_scene(A)
        if mode:
            self.choose(mode)


class OpenedColoringTests(OpenedStructureTestCase):
    def test_with_the_scene_on_plddt_an_opened_file_starts_in_its_own_color_and_plddt_is_one_click_away(self):
        for how in ("open", "drop"):
            with self.subTest(how=how):
                self.scene()
                # (The page of the round before makes noise as it goes.)
                self.console.messages.clear()
                self.bring_in(how)
                shown = self.shown()
                self.assertEqual([item["coloring"] for item in shown], ["plddt", OWN])
                self.assertEqual(shown[1]["name"], "mine")
                self.assertEqual("#%06x" % shown[1]["value"], shown[1]["swatch"])
                # The scene's choice is what it was, and the opened structure's row shows its own.
                self.assertEqual(self.script("return card().selected"), "plddt")
                self.open_details(2)
                row = self.script("""
                    const select = document.querySelector('#struct-list .srow:nth-child(2) select[id^="col-"]');
                    return {value: select.value, options: [...select.options].map(o => o.value)};
                    """)
                self.assertEqual(row["value"], OWN)
                self.assertIn("plddt", row["options"])
                # One click in the row, and it is pLDDT, with a line saying what the colors are.
                self.script("""
                    const select = document.querySelector('#struct-list .srow:nth-child(2) select[id^="col-"]');
                    select.value = 'plddt'; select.dispatchEvent(new Event('change', {bubbles: true}));
                    await structureQueue;
                    return 1;
                    """)
                self.settle(500)
                self.assertEqual([item["coloring"] for item in self.shown()], ["plddt", "plddt"])
                self.assertEqual(self.script("return card().caution"), "mine: the colors are the file's B-factor column. In an experimental structure that is not confidence.")
                self.assertEqual(self.console_errors(), [])

    def test_the_scenes_next_choice_is_for_an_opened_file_too_and_a_folded_structure_follows_the_scene(self):
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.scene()
        self.bring_in("open")
        self.choose("information_content")
        self.settle(300)
        self.assertEqual([item["coloring"] for item in self.shown()], ["information_content", "nodata"])
        self.choose("plddt")
        self.settle(300)
        # The scene's pLDDT reaches it: it is in pLDDT now, as the scene is.
        self.assertEqual([item["coloring"] for item in self.shown()], ["plddt", "plddt"])
        self.assertEqual(self.script("return card().caution").startswith("mine:"), True)
        # A structure folded now takes the scene's pLDDT: the rule for folded structures is the old one.
        self.push(B)
        self.assertEqual([item["coloring"] for item in self.shown()], ["plddt", "plddt", "plddt"])

    def test_with_the_scene_on_a_layer_an_opened_file_has_no_data_and_on_structure_color_it_is_in_its_own(self):
        for how in ("open", "drop"):
            with self.subTest(how=how, scene="layer"):
                self.scene("information_content")
                self.bring_in(how)
                shown = self.shown()
                self.assertEqual([item["coloring"] for item in shown], ["information_content", "nodata"])
                self.assertEqual((shown[1]["theme"], shown[1]["value"]), ("uniform", GREY))
                self.assertIn("No data", shown[1]["chips"])
            with self.subTest(how=how, scene="structure color"):
                self.scene(OWN)
                self.bring_in(how)
                shown = self.shown()
                self.assertEqual([item["coloring"] for item in shown], [OWN, OWN])
                self.assertNotEqual(shown[0]["value"], shown[1]["value"])
                self.assertNotIn("No data", shown[1]["chips"])

    def test_an_opened_file_shows_a_b_factor_where_a_folded_structure_shows_a_confidence(self):
        self.scene()
        self.bring_in("open")
        # In its own color the legend has nothing to say of it; the tip says what its number is.
        self.assertIsNone(self.script("return card().caution"))
        self.script("await hoverOn(1, 5); return 1")
        opened = self.script("return tipNow()")
        self.assertEqual(opened["rows"], [["B-factor", f"{fixtures.plddt_values(33)[5]:.1f}"]])
        self.script("await hoverOn(0, 5); return 1")
        folded = self.script("return tipNow().rows[0][0]")
        self.assertEqual(folded, "pLDDT")
        # The Residue card says it too, once a residue is pinned.
        self.script("await clickOn(1, 5); return 1")
        self.assertIn("mine: the number is the file's B-factor, which is not confidence in an experimental structure.",
                      self.script("return [...document.querySelectorAll('#residue-body .res-note')].map(n => n.textContent)"))
        self.assertEqual(self.console_errors(), [])


class OpenedSessionTests(OpenedStructureTestCase):
    def after_restart(self, snapshot):
        self.session = json.loads(json.dumps(snapshot))
        self.open_page()
        self.settle(900)

    def test_a_restored_session_keeps_the_opened_files_in_their_color_and_the_scene_in_its_choice(self):
        self.scene()
        self.bring_in("open")
        self.bring_in("open", "other")
        before = self.shown()
        own = before[1]["value"]
        self.settle(1800)
        self.assertTrue(self.saved)
        self.after_restart(self.saved[-1])
        shown = self.shown()
        self.assertEqual([item["coloring"] for item in shown], ["plddt", OWN, OWN])
        self.assertEqual(shown[1]["value"], own)
        self.assertEqual(["#%06x" % item["value"] for item in shown[1:]], [item["swatch"] for item in shown[1:]])
        # The scene's choice is pLDDT: the two structures in a color of their own did not outvote the one in pLDDT.
        self.assertEqual(self.script("return [card().selected, colorMode]"), ["plddt", "plddt"])
        self.assertEqual(self.script("return loadedStructures().map(isFolded)"), [True, False, False])
        self.assertEqual(self.script("return loadedStructures().map(s => structureFile(s) || null)"), [A, None, None])

    def test_a_session_file_with_opened_files_comes_back_the_same_way(self):
        self.scene()
        self.bring_in("open")
        self.bring_in("open", "other")
        before = self.shown()
        self.script("""
            const make = URL.createObjectURL;
            URL.createObjectURL = blob => { window.__blob = blob; return make.call(URL, blob); };
            HTMLAnchorElement.prototype.click = function () {};
            return 1;
            """)
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='download']", 800)
        text = self.script("return await window.__blob.text()")
        # Clearing the scene, then opening the file.
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='clear']", 100)
        self.click("#confirm-ok", 800)
        self.assertEqual(self.script("return loadedStructures().length"), 0)
        self.script("await openSessionFile(new File([%s], 'back.molj')); await structureQueue; return 1" % json.dumps(text))
        self.settle(900)
        shown = self.shown()
        self.assertEqual([item["coloring"] for item in shown], ["plddt", OWN, OWN])
        self.assertEqual([item["value"] for item in shown], [item["value"] for item in before])
        self.assertEqual(self.script("return [card().selected, loadedStructures().map(isFolded)]"), ["plddt", [True, False, False]])

    def test_the_scenes_choice_is_worked_out_without_the_opened_files(self):
        self.write_with(B, "information_content", "same_residue_pct", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.scene()
        self.push(B)
        self.choose("information_content")
        self.bring_in("open")
        self.settle(300)
        derive = """
            const out = {};
            colorMode = 'plddt'; deriveColorMode(); out.layers = colorMode;
            return out;
            """
        # Folded: both in the layer. Opened: grey, which is not counted either way.
        self.assertEqual(self.script(derive), {"layers": "information_content"})
        # The opened structure in a color of its own does not outvote two folded ones in pLDDT.
        self.choose("plddt")
        self.script("await setStructureColoring(loadedStructures()[2].cell.transform.ref, 'builtin:structure'); return 1")
        self.settle(300)
        self.assertEqual(self.script("colorMode = 'builtin:structure'; deriveColorMode(); return colorMode"), "plddt")
        # And a scene of nothing but an opened file is the default one, pLDDT.
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='clear']", 100)
        self.click("#confirm-ok", 800)
        self.bring_in("open")
        self.assertEqual(self.script("colorMode = 'builtin:structure'; deriveColorMode(); return colorMode"), "plddt")


if __name__ == "__main__":
    unittest.main()
