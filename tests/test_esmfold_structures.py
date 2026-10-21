# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's Structures card and Session menu (esmfold_structures.js, esmfold_session.js).

A row of the Structures card opens to a structure's details (representation,
opacity, mean pLDDT, what the alignment found) and its actions (Make pivot,
Re-align, Download PDB); a structure file can be opened or dropped on the
view; the Session menu saves, downloads and opens a session file and clears
the scene. Like tests/test_esmfold_shell.py, these open the page in Qt
WebEngine over the real web server and the real Mol*, and read its DOM and
Mol*'s state.
"""

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests import esmfold_layer_fixtures as fixtures
from tests.esmfold_page_harness import FoldViewPageMixin, FoldViewServerMixin, http_get
from tests.esmfold_session_fixture import session_before_the_redesign, write_session_structures

NAMES = ("11.pdb", "22.pdb", "33.pdb")


class StructuresTestCase(FoldViewPageMixin, FoldViewServerMixin, unittest.TestCase):
    """The page over a folder of fixture structures, with the Viewer's side of its conversation."""

    def write(self, name, sequence=fixtures.FIRST_SEQUENCE, **options):
        return fixtures.write_structure(self.directory, name, sequence, **options)

    def write_all(self):
        self.write(NAMES[0])
        self.write(NAMES[1], fixtures.SECOND_SEQUENCE, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.write(NAMES[2], fixtures.FIRST_SEQUENCE, shift=(-2.0, 4.0, 1.0), wobble=0.7)

    def open_with(self, count=2):
        self.write_all()
        self.open_page()
        for name in NAMES[:count]:
            self.push(name)
        self.settle(200)

    def settle(self, milliseconds):
        self.script(f"await new Promise(resolve => setTimeout(resolve, {milliseconds})); return 1")

    def click(self, selector, wait=300):
        self.script(f"document.querySelector({json.dumps(selector)}).click(); return 1")
        self.settle(wait)

    def row(self, number):
        return f"#struct-list .srow:nth-child({number})"

    def open_details(self, number):
        if self.script(f"return !document.querySelector({json.dumps(self.row(number) + ' .srow-body')})"):
            self.click(self.row(number) + " .exp-btn", 100)

    def details(self, number):
        return self.script("""
            const row = document.querySelector(%s);
            const body = row.querySelector('.srow-body');
            if (!body) return null;
            const facts = {};
            for (const term of body.querySelectorAll('dt')) facts[term.textContent] = term.nextElementSibling.textContent;
            return {facts, hint: (body.querySelector('.hint') || {}).textContent,
                    buttons: [...body.querySelectorAll('.row-actions button')].map(button => button.textContent),
                    representation: body.querySelector('select').value,
                    opacity: body.querySelector('input[type="range"]').value,
                    opacityLabel: body.querySelector('.field-label .field-value').textContent,
                    chips: [...row.querySelectorAll('.srow-main .chip')].map(chip => chip.textContent),
                    expanded: row.querySelector('.exp-btn').getAttribute('aria-expanded'),
                    controls: row.querySelector('.exp-btn').getAttribute('aria-controls'), id: body.id};
            """ % json.dumps(self.row(number)))

    def toast(self):
        return self.script("""
            const toast = document.getElementById('toast');
            return {shown: toast.classList.contains('show'), text: toast.querySelector('span').textContent,
                    undo: !!toast.querySelector('.toast-action')};
            """)

    def structure_kinds(self):
        return self.script("""
            const kinds = {};
            for (const cell of plugin.state.data.cells.values()) {
                const id = cell.transform.transformer.id.replace('ms-plugin.', '');
                kinds[id] = (kinds[id] || 0) + 1;
            }
            return kinds;
            """)


class StructureDetailsTests(StructuresTestCase):
    def test_a_row_opens_to_its_details_and_closes_again(self):
        self.open_with(2)
        self.open_details(2)
        second = self.details(2)
        self.assertEqual(second["expanded"], "true")
        self.assertEqual(second["controls"], second["id"])
        self.assertEqual(second["representation"], "cartoon")
        self.assertEqual((second["opacity"], second["opacityLabel"]), ("100", "100%"))
        self.assertEqual(second["buttons"], ["Make pivot", "Re-align", "Download PDB"])
        facts = second["facts"]
        mean = sum(fixtures.plddt_values(33)) / 33
        self.assertAlmostEqual(float(facts["Mean pLDDT"]), mean, delta=0.06)
        self.assertEqual(facts["Aligned to 11"][:5], "RMSD ")
        self.assertRegex(facts["Cα pairs"], r"^\d+ of 33$")
        self.assertRegex(facts["Sequence identity"], r"^\d+%$")
        # The pivot has no alignment of its own to show.
        self.open_details(1)
        pivot = self.details(1)
        self.assertEqual(pivot["hint"], "This is the pivot: the others are aligned to it.")
        self.assertEqual(pivot["buttons"], ["Download PDB"])
        self.assertNotIn("Cα pairs", pivot["facts"])
        # Closing takes the details away, and the button keeps the focus.
        self.script("document.querySelector(%s).focus(); return 1" % json.dumps(self.row(2) + " .exp-btn"))
        self.click(self.row(2) + " .exp-btn", 100)
        self.assertIsNone(self.details(2))
        self.assertEqual(self.script("return document.activeElement.dataset.focusKey.split(':')[1]"), "details")
        self.assertEqual(self.console_errors(), [])

    def test_the_representation_changes_and_the_colors_and_opacity_stay(self):
        self.open_with(1)
        self.open_details(1)
        plddt = fixtures.plddt_values(33)
        before = self.script("return colors(0, 'plddt-confidence', {})")
        self.script("""
            const slider = document.querySelector('#struct-list input[type="range"]');
            slider.value = '40'; slider.dispatchEvent(new Event('input', {bubbles: true}));
            await structureQueue;
            return 1;
            """)
        self.settle(300)
        self.assertEqual(self.script("return structureOpacity(loadedStructures()[0])"), 40)
        self.assertEqual(self.details(1)["opacityLabel"], "40%")
        self.script("""
            const select = document.querySelector('#struct-list select');
            select.value = 'ball-and-stick'; select.dispatchEvent(new Event('change', {bubbles: true}));
            await structureQueue;
            return 1;
            """)
        self.settle(500)
        state = self.script("""
            const reprs = representationRefs(loadedStructures()[0]).map(r => r.cell.transform.params);
            return {types: reprs.map(r => r.type.name), alphas: reprs.map(r => r.type.params.alpha),
                    sizes: reprs.map(r => r.sizeTheme.name), themes: reprs.map(r => r.colorTheme.name),
                    colors: colors(0, 'plddt-confidence', {}), shown: document.querySelector('#struct-list select').value};
            """)
        self.assertEqual(state["types"], ["ball-and-stick"])
        self.assertEqual(state["alphas"], [0.4])
        self.assertEqual(state["sizes"], ["physical"])
        self.assertEqual(state["themes"], ["plddt-confidence"])
        self.assertEqual(state["colors"], before)
        self.assertEqual(state["shown"], "ball-and-stick")
        # A layer's colors come along too: the annotation theme with the layer it had.
        layered = self.script("""
            await runExclusive(() => applyLayer(loadedStructures()[0], 'information_content'));
            const first = reprThemes()[0];
            const select = document.querySelector('#struct-list select');
            select.value = 'spacefill'; select.dispatchEvent(new Event('change', {bubbles: true}));
            await structureQueue;
            await new Promise(resolve => setTimeout(resolve, 500));
            const reprs = representationRefs(loadedStructures()[0]).map(r => r.cell.transform.params);
            return {first, after: reprs.map(r => [r.type.name, r.colorTheme]), active: activeLayerId(loadedStructures()[0])};
            """)
        self.assertEqual(layered["first"]["name"], "mvs-annotation")
        self.assertEqual(layered["after"], [["spacefill", layered["first"]]])
        self.assertEqual(layered["active"], "information_content")
        self.assertEqual(self.console_errors(), [])
        self.assertEqual(len(plddt), 33)

    def test_the_download_button_gives_the_file_as_it_was_folded(self):
        self.open_with(2)
        self.open_details(2)
        self.script("""
            window.__clicked = [];
            HTMLAnchorElement.prototype.click = function () { window.__clicked.push([this.getAttribute('href'), this.download]); };
            return 1;
            """)
        self.click(self.row(2) + " .row-actions button:last-child", 100)
        clicked = self.script("return window.__clicked")
        self.assertEqual(clicked, [["/structures/22.pdb", "22.pdb"]])
        status, _, body = http_get(self.server, "/structures/22.pdb")
        self.assertEqual(status, 200)
        self.assertEqual(body, (Path(self.directory) / "22.pdb").read_bytes())

    def test_removing_a_structure_takes_everything_it_came_from(self):
        self.open_with(2)
        self.click(self.row(2) + " .srow-main button:last-child", 600)
        kinds = self.structure_kinds()
        for kind in ("download", "trajectory-from-pdb", "model-from-trajectory", "structure-from-model"):
            self.assertEqual(kinds[kind], 1, kind)
        self.assertNotIn("transform-structure-conformation", kinds)


class AlignmentTests(StructuresTestCase):
    def test_a_restored_session_gets_its_alignment_measured(self):
        write_session_structures(self.directory)
        self.session = session_before_the_redesign()
        self.open_page()
        self.settle(300)
        # Nothing in the session says what the alignment found; the page measures the pose it has.
        found = self.script("return {stats: fitStats.size, pivot: structureName(pivotStructure())}")
        self.assertEqual(found["pivot"], "protein_a")
        self.open_details(2)
        details = self.details(2)
        self.assertRegex(details["chips"][0], r"^RMSD 0\.\d Å$")
        self.assertEqual(details["facts"]["Cα pairs"], "33 of 33")
        self.assertEqual(self.console_errors(), [])

    def test_a_pose_changed_elsewhere_is_measured_again_and_re_align_fixes_it(self):
        self.open_with(2)
        self.open_details(2)
        good = self.details(2)["chips"][0]
        self.assertRegex(good, r"^RMSD 0\.\d Å$")
        self.script("""
            const matrix = molstar.lib.math.LinearAlgebra.Mat4.fromTranslation(molstar.lib.math.LinearAlgebra.Mat4(), [0, 0, 6]);
            await runExclusive(() => applyStructureTransform(loadedStructures()[1], matrix));
            return 1;
            """)
        self.settle(400)
        moved = self.details(2)["chips"][0]
        self.assertRegex(moved, r"^RMSD [4-9]\.\d Å$")
        self.click(self.row(2) + " .row-actions button:nth-child(2)", 800)
        self.assertRegex(self.details(2)["chips"][0], r"^RMSD 0\.\d Å$")
        self.assertEqual(self.toast()["text"], "Aligned 22 again: RMSD %s." % self.details(2)["chips"][0][5:])
        self.assertEqual(self.console_errors(), [])

    def test_make_pivot_moves_the_pivot_and_the_next_structure_follows_it(self):
        self.open_with(2)
        self.open_details(2)
        self.click(self.row(2) + " .row-actions button:first-child", 800)
        state = self.script("""
            const pivot = pivotStructure();
            const [first, second] = loadedStructures();
            return {pivot: structureName(pivot), names: loadedStructures().map(structureName),
                    identity: [first, second].map(s => { const data = structureMatrix(s); return data ? isIdentityMatrix(data) : null; }),
                    rmsd: measurePose(pivot, first).rmsd, chips: [...document.querySelectorAll('#struct-list .srow-main .chip')].map(c => c.className + ':' + c.textContent),
                    toast: document.getElementById('toast').textContent};
            """)
        self.assertEqual(state["pivot"], "22")
        self.assertEqual(state["names"], ["11", "22"])
        self.assertEqual(state["identity"], [False, True])
        self.assertLess(state["rmsd"], 0.6)
        self.assertEqual(state["chips"][0].split(":")[0], "chip ok")
        self.assertTrue(state["chips"][1].startswith("chip pivot:Pivot"), state["chips"])
        self.assertIn("22 is the pivot now. The other structures are aligned to it.", state["toast"])
        # A structure folded now is aligned to the new pivot, not to the first one loaded.
        self.push(NAMES[2])
        self.settle(300)
        third = self.script("""
            const structures = loadedStructures();
            const pivot = pivotStructure();
            return {pivot: structureName(pivot), measured: measurePose(pivot, structures[2]).rmsd,
                    onFirst: measurePose(structures[0], structures[2]).rmsd,
                    chips: [...document.querySelectorAll('#struct-list .srow-main .chip')].map(c => c.className.split(' ')[1])};
            """)
        self.assertEqual(third["pivot"], "22")
        self.assertLess(third["measured"], 1.5)
        self.assertEqual(third["chips"], ["ok", "pivot", "ok"])
        self.assertEqual(self.console_errors(), [])

    def test_a_structure_that_is_removed_leaves_the_others_aligned_to_the_one_that_remains_first(self):
        self.open_with(3)
        self.click(self.row(1) + " .srow-main button:last-child", 600)
        found = self.script("""
            const structures = loadedStructures();
            return {names: structures.map(structureName), pivot: structureName(pivotStructure()),
                    chips: [...document.querySelectorAll('#struct-list .srow-main .chip')].map(c => c.className.split(' ')[1])};
            """)
        self.assertEqual(found["names"], ["22", "33"])
        # Both keep the pose they had; the first of them stands in as the pivot.
        self.assertEqual(found["pivot"], "22")
        self.assertEqual(found["chips"][0], "pivot")
        # A structure folded now is aligned to that one, not to itself.
        self.write("44.pdb", fixtures.SECOND_SEQUENCE, shift=(1.0, -3.0, 2.0), wobble=0.5)
        self.push("44.pdb")
        self.settle(300)
        fourth = self.script("""
            const structures = loadedStructures();
            return {names: structures.map(structureName), pivot: structureName(pivotStructure()),
                    rmsd: measurePose(pivotStructure(), structures[2]).rmsd,
                    chips: [...document.querySelectorAll('#struct-list .srow-main .chip')].map(c => c.className.split(' ')[1])};
            """)
        self.assertEqual(fourth["names"], ["22", "33", "44"])
        self.assertEqual(fourth["pivot"], "22")
        self.assertLess(fourth["rmsd"], 1.5)
        self.assertEqual(fourth["chips"], ["pivot", "ok", "ok"])


class SessionMenuTests(StructuresTestCase):
    MENU = "JSON.stringify({open: document.getElementById('session-menu').classList.contains('open'), expanded: document.getElementById('session-btn').getAttribute('aria-expanded'), focus: document.activeElement.id || document.activeElement.dataset.action || ''})"

    def menu(self):
        return json.loads(self.js(self.MENU))

    def test_the_menu_opens_and_closes_with_the_keyboard_and_loses_nothing_when_offline(self):
        self.open_with(2)
        self.js("document.getElementById('session-btn').focus(); document.getElementById('session-btn').click()")
        self.assertEqual(self.menu(), {"open": True, "expanded": "true", "focus": "mi-save"})
        for key, focus in (("ArrowDown", "download"), ("ArrowDown", "open"), ("End", "mi-clear"), ("ArrowDown", "mi-save"),
                           ("ArrowUp", "mi-clear"), ("Home", "mi-save")):
            self.js("document.activeElement.dispatchEvent(new KeyboardEvent('keydown', {key: %s, bubbles: true}))" % json.dumps(key))
            self.assertEqual(self.menu()["focus"], focus, key)
        self.js("document.activeElement.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.assertEqual(self.menu(), {"open": False, "expanded": "false", "focus": "session-btn"})
        # A click anywhere else closes it without taking the focus.
        self.js("document.getElementById('session-btn').click()")
        self.js("document.getElementById('app').click()")
        self.assertFalse(self.menu()["open"])
        # With no connection to the Viewer there is nothing to save to, and the menu says so.
        self.script("setConnection('reconnecting'); return 1")
        self.js("document.getElementById('session-btn').click()")
        offline = self.script("""
            return {save: document.getElementById('mi-save').disabled, note: document.getElementById('menu-saved').textContent,
                    others: [...document.querySelectorAll('#session-menu .menu-item')].filter(i => i.disabled).length};
            """)
        self.assertEqual(offline, {"save": True, "note": "Not saved: no connection to the Viewer.", "others": 1})
        self.script("setConnection('connected'); return 1")
        self.assertFalse(self.script("return document.getElementById('mi-save').disabled"))

    def test_save_now_posts_the_session_and_the_header_says_when(self):
        self.open_with(2)
        self.settle(1800)
        del self.saved[:]
        self.js("document.getElementById('session-btn').click()")
        self.click("#mi-save", 600)
        self.assertGreaterEqual(len(self.saved), 1)
        files = [node["transformer"] for node in self.saved[-1]["data"]["tree"]["transforms"]]
        self.assertEqual(files.count("ms-plugin.download"), 2)
        self.assertRegex(self.script("return document.getElementById('saved-note').textContent"), r"^Saved \d{1,2}:\d{2}")
        self.assertEqual(self.toast()["text"], "Session saved.")

    def test_a_save_the_viewer_refuses_is_reported(self):
        self.open_with(1)
        self.script("""
            window.fetch = () => Promise.resolve({ok: false, status: 500});
            return 1;
            """)
        self.js("document.getElementById('session-btn').click()")
        self.click("#mi-save", 400)
        self.assertEqual(self.toast()["text"], "The session could not be saved.")
        self.assertEqual(self.script("return document.getElementById('saved-note').textContent"), "Not saved")

    def test_the_session_downloads_as_a_molj_file_and_a_cleared_scene_comes_back_from_it(self):
        self.open_with(2)
        before = self.script("return {fit: fit(1), files: loadedStructures().map(structureFile)}")
        self.script("""
            window.__download = null;
            const make = URL.createObjectURL;
            URL.createObjectURL = blob => { window.__blob = blob; return make.call(URL, blob); };
            HTMLAnchorElement.prototype.click = function () { window.__download = this.download; };
            return 1;
            """)
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='download']", 600)
        saved = self.script("""
            const text = await window.__blob.text();
            const file = JSON.parse(text);
            return {name: window.__download, current: !!file.current, entries: file.entries.length,
                    structures: file.entries[0].snapshot.data.tree.transforms.filter(t => t.transformer === 'ms-plugin.download').length,
                    text};
            """)
        self.assertEqual(saved["name"], "fold-view-session.molj")
        self.assertEqual((saved["entries"], saved["structures"]), (1, 2))
        # Clearing needs a yes; the answer No leaves everything where it was.
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='clear']", 100)
        asked = self.script("""
            return {open: document.getElementById('confirm-modal').classList.contains('open'), focus: document.activeElement.id,
                    title: document.getElementById('confirm-title').textContent, ok: document.getElementById('confirm-ok').textContent,
                    n: loadedStructures().length};
            """)
        self.assertEqual(asked, {"open": True, "focus": "confirm-cancel", "title": "Clear the scene?", "ok": "Clear scene", "n": 2})
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.assertEqual(self.script("return {open: document.getElementById('confirm-modal').classList.contains('open'), n: loadedStructures().length, focus: document.activeElement.id}"),
                         {"open": False, "n": 2, "focus": "session-btn"})
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='clear']", 100)
        self.click("#confirm-ok", 800)
        cleared = self.script("""
            return {n: loadedStructures().length, cells: plugin.state.data.cells.size, rows: document.querySelectorAll('#struct-list .srow').length,
                    hint: document.querySelector('#struct-list .hint').textContent, toast: document.getElementById('toast').textContent,
                    count: document.getElementById('struct-count').textContent};
            """)
        self.assertEqual(cleared["n"], 0)
        self.assertEqual(cleared["cells"], 1)
        self.assertEqual((cleared["rows"], cleared["count"]), (0, ""))
        self.assertEqual(cleared["hint"], "Structures appear here as they are folded.")
        self.assertEqual(cleared["toast"], "The scene was cleared.Undo")
        # The file opens again: the same structures in the same poses.
        self.script("window.__file = new File([%s], 'mine.molj'); await openSessionFile(window.__file); await structureQueue; return 1" % json.dumps(saved["text"]))
        self.settle(800)
        opened = self.script("return {files: loadedStructures().map(structureFile), fit: fit(1), toast: document.getElementById('toast').textContent}")
        self.assertEqual(opened["files"], before["files"])
        self.assertEqual(opened["fit"]["matrix"], before["fit"]["matrix"])
        self.assertEqual(opened["toast"], "Opened mine.molj.Undo")
        # A file that is not a session changes nothing and says so.
        self.script("await openSessionFile(new File(['{\"a\": 1}'], 'other.json')); return 1")
        self.assertEqual(self.toast()["text"], "other.json is not a Mol* session file.")
        self.assertEqual(self.script("return loadedStructures().length"), 2)
        self.assertEqual(self.console_errors(), [])

    def test_clear_scene_can_be_undone_and_is_not_offered_with_nothing_to_clear(self):
        self.open_with(2)
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='clear']", 100)
        self.click("#confirm-ok", 800)
        self.assertEqual(self.script("return loadedStructures().length"), 0)
        self.click("#toast .toast-action", 1200)
        self.assertEqual(self.script("return loadedStructures().map(structureFile)"), ["11.pdb", "22.pdb"])
        self.assertEqual(self.script("return document.querySelectorAll('#struct-list .srow').length"), 2)
        # With no structure the item is not offered.
        self.js("document.getElementById('session-btn').click()")
        self.click("#session-menu [data-action='clear']", 100)
        self.click("#confirm-ok", 800)
        self.js("document.getElementById('session-btn').click()")
        self.assertTrue(self.script("return document.getElementById('mi-clear').disabled"))
        self.assertEqual(self.console_errors(), [])


class OpenFileTests(StructuresTestCase):
    def pdb_text(self):
        return json.dumps(fixtures.helix_pdb(fixtures.FIRST_SEQUENCE, fixtures.plddt_values(33), shift=(1.0, 2.0, 0.0), wobble=0.3))

    def test_a_structure_file_is_opened_and_aligned_like_a_folded_one(self):
        self.open_with(1)
        self.script("openStructureFiles([new File([%s], 'mine.pdb')]); await structureQueue; return 1" % self.pdb_text())
        self.settle(500)
        found = self.script("""
            const structures = loadedStructures();
            const last = structures[structures.length - 1];
            return {names: structures.map(structureName), file: structureFile(last) || null,
                    themes: reprThemes().map(t => t.name), pivot: structureName(pivotStructure()),
                    rmsd: measurePose(pivotStructure(), last).rmsd,
                    chips: [...document.querySelectorAll('#struct-list .srow-main .chip')].map(c => c.textContent)};
            """)
        self.assertEqual(found["names"], ["11", "mine"])
        self.assertIsNone(found["file"])
        self.assertEqual(found["themes"], ["plddt-confidence", "plddt-confidence"])
        self.assertEqual(found["pivot"], "11")
        self.assertLess(found["rmsd"], 0.6)
        self.assertEqual(found["chips"][0], "Pivot")
        self.assertRegex(found["chips"][1], r"^RMSD 0\.\d Å$")
        # Without a file in the Viewer's folder there is no PDB to download and no pLDDT to average.
        self.open_details(2)
        details = self.details(2)
        self.assertEqual(details["buttons"], ["Make pivot", "Re-align"])
        self.assertNotIn("Mean pLDDT", details["facts"])
        self.assertEqual(self.console_errors(), [])

    def test_a_file_that_is_not_a_structure_is_refused_and_the_button_opens_the_chooser(self):
        self.open_page()
        self.script("openStructureFiles([new File(['x'], 'notes.txt')]); return 1")
        toast = self.toast()
        self.assertEqual(toast["text"], "Only PDB and mmCIF files can be opened here.")
        self.script("""
            window.__chooser = [];
            HTMLInputElement.prototype.click = function () { window.__chooser.push(this.id); };
            return 1;
            """)
        self.script("document.getElementById('open-file').click(); return 1")
        self.assertEqual(self.script("return window.__chooser"), ["open-structure-input"])
        self.assertEqual(self.script("return document.body.dataset.adv"), "0")
        self.assertEqual(self.script("return document.getElementById('open-structure-input').accept"), ".pdb,.ent,.cif,.mmcif")

    def test_a_file_dropped_on_the_view_is_opened_once_by_the_page(self):
        self.open_page()
        self.script("""
            const transfer = new DataTransfer();
            transfer.items.add(new File([%s], 'dropped.pdb'));
            const target = document.getElementById('viewer');
            target.dispatchEvent(new DragEvent('dragenter', {dataTransfer: transfer, bubbles: true, cancelable: true}));
            window.__hinted = document.getElementById('stage').classList.contains('dropping');
            const over = new DragEvent('dragover', {dataTransfer: transfer, bubbles: true, cancelable: true});
            target.dispatchEvent(over);
            window.__accepted = over.defaultPrevented;
            target.dispatchEvent(new DragEvent('drop', {dataTransfer: transfer, bubbles: true, cancelable: true}));
            await structureQueue;
            return 1;
            """ % self.pdb_text())
        self.settle(800)
        found = self.script("""
            return {hinted: window.__hinted, accepted: window.__accepted, dropping: document.getElementById('stage').classList.contains('dropping'),
                    names: loadedStructures().map(structureName)};
            """)
        self.assertEqual(found, {"hinted": True, "accepted": True, "dropping": False, "names": ["dropped"]})


if __name__ == "__main__":
    unittest.main()
