# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's own interface around the Mol* canvas (src/web_ui/esmfold.html and esmfold_*.js).

The page hides Mol*'s interface and draws a header, a structure list, a colour
switch, a tooltip, a sequence strip and the empty and error states of its own,
in the look of the Agent and Meta Data pages. These tests open it in Qt
WebEngine over the real web server and the real bundled Mol* (the harness is
tests/esmfold_page_harness.py), and check the page through its DOM and Mol*'s
state: the offscreen engine has no WebGL, and tests/test_esmfold_shell_webgl.py
holds the checks that need a canvas.
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
from tests.translation_fixtures import pseudo_language


class PageTestCase(FoldViewPageMixin, FoldViewServerMixin, unittest.TestCase):
    """The page over a folder of fixture structures, with the Viewer's side of its conversation."""

    BANDS = ((90, "#0053d6"), (70, "#65cbf3"), (50, "#ffdb13"), (-1, "#ff7d45"))

    def band(self, value):
        return next(color for floor, color in self.BANDS if value > floor)


class SavedSessionTests(PageTestCase):
    """A session the page saved before it was redesigned keeps loading."""

    def test_a_session_saved_before_the_redesign_loads_as_it_was(self):
        saved = session_before_the_redesign()
        pose = next(node["params"]["transform"]["params"]["data"] for node in saved["data"]["tree"]["transforms"]
                    if node["transformer"] == "ms-plugin.transform-structure-conformation")
        write_session_structures(self.directory)
        self.session = saved
        self.open_page()
        restored = self.script("""
            return {n: loadedStructures().length, themes: reprThemes().map(theme => theme.name),
                    active: loadedStructures().map(activeLayerId), files: loadedStructures().map(structureFile),
                    fit: fit(1), colors: [colors(0, 'plddt-confidence', {}), colors(1, 'plddt-confidence', {})]};
            """)
        self.assertEqual(restored["n"], 2)
        self.assertEqual(restored["files"], ["protein_a.pdb", "protein_b.pdb"])
        self.assertEqual(restored["themes"], ["plddt-confidence", "plddt-confidence"])
        self.assertEqual(restored["active"], ["plddt", "plddt"])
        # The second structure keeps the pose that was saved, and is on the first.
        self.assertTrue(restored["fit"]["transformed"])
        self.assertEqual(restored["fit"]["matrix"], pose)
        self.assertLess(restored["fit"]["rmsd"], 0.6)
        for index, offset in ((0, 0), (1, 17)):
            plddt = fixtures.plddt_values(33, offset)
            for number in range(1, 34):
                self.assertEqual(restored["colors"][index][str(number)], self.band(plddt[number - 1]), (index, number))


class ShellTests(PageTestCase):
    """The page's header, canvas states, structure list, theme, Advanced door and help dialog."""

    FIRST, SECOND = "11.pdb", "22.pdb"

    def write(self, name, sequence=fixtures.FIRST_SEQUENCE, **options):
        return fixtures.write_structure(self.directory, name, sequence, **options)

    def open_with_two(self):
        self.write(self.FIRST)
        self.write(self.SECOND, fixtures.SECOND_SEQUENCE, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_page()
        self.push(self.FIRST)
        self.push(self.SECOND)
        self.script("await new Promise(resolve => setTimeout(resolve, 150)); return 1")

    def rows(self):
        return self.script("""
            return [...document.querySelectorAll('#struct-list .srow')].map(row => ({
                name: row.querySelector('.srow-name').textContent,
                chips: [...row.querySelectorAll('.chip')].map(chip => [chip.textContent, chip.className, chip.title]),
                size: row.querySelector('.srow-sub span:last-child').textContent,
                buttons: [...row.querySelectorAll('.srow-main button')].map(button => [button.title, button.getAttribute('aria-label'),
                    button.getAttribute('aria-pressed'), button.querySelector('use').getAttribute('href')]),
                hidden: row.classList.contains('hidden-structure')}));
            """)

    def test_mol_stars_own_interface_is_hidden_and_the_page_draws_its_own(self):
        self.open_page()
        found = self.script("""
            const shown = element => element.getClientRects().length > 0 && getComputedStyle(element).visibility !== 'hidden';
            return {
                regions: document.querySelectorAll('.msp-layout-left, .msp-layout-right, .msp-layout-top, .msp-layout-bottom').length,
                buttons: [...document.querySelectorAll('#viewer button')].filter(shown).length,
                header: document.querySelector('header h1').textContent,
                adv: document.body.dataset.adv, panel: document.body.dataset.panel,
                structures: document.getElementById('struct-list').textContent,
                count: document.getElementById('struct-count').textContent,
                connection: document.getElementById('connection-status').dataset.state,
                status: document.getElementById('connection-status').textContent,
                lang: document.documentElement.lang,
                controls: plugin.layout.state.showControls};
            """)
        self.assertEqual(found["regions"], 0)
        self.assertEqual(found["buttons"], 0)
        self.assertEqual(found["header"], "EMAP-SSN Fold View")
        self.assertEqual((found["adv"], found["panel"]), ("0", "1"))
        self.assertEqual(found["structures"], "Structures appear here as they are folded.")
        self.assertEqual(found["count"], "")
        self.assertEqual((found["connection"], found["status"]), ("connected", "Connected"))
        self.assertEqual(found["lang"], "en")
        self.assertIs(found["controls"], False)
        self.assertEqual(self.console_errors(), [])

    def test_the_canvas_area_says_what_it_is_waiting_for(self):
        self.open_page()
        # Whether the engine this runs in gives Mol* a canvas depends on the machine and on what ran
        # before in the process, so each state is made by standing a canvas in or out while the page's
        # rule is read, then the real one is put back.
        found = self.script("""
            const own = Object.getOwnPropertyDescriptor(plugin, 'canvas3d');
            const stand = value => Object.defineProperty(plugin, 'canvas3d', {value, configurable: true, writable: true});
            const wasRestoring = isRestoring;
            const state = () => {
                updateView();
                const cards = [...document.querySelectorAll('.empty[data-for]')]
                    .filter(card => getComputedStyle(card).display !== 'none').map(card => card.dataset.for);
                return {view: document.body.dataset.view, count: document.body.dataset.count, cards};
            };
            const out = {};
            try {
                stand(undefined); isRestoring = false; out.nocanvas = state();
                stand({}); isRestoring = true; out.restoring = state();
                isRestoring = false; out.empty = state();
            } finally {
                if (own) Object.defineProperty(plugin, 'canvas3d', own); else delete plugin.canvas3d;
                isRestoring = wasRestoring;
                updateView();
            }
            return out;
            """)
        self.assertEqual(found["nocanvas"], {"view": "webgl", "count": "0", "cards": ["webgl"]})
        self.assertEqual(found["restoring"], {"view": "restoring", "count": "0", "cards": ["restoring"]})
        self.assertEqual(found["empty"], {"view": "empty", "count": "0", "cards": ["empty"]})

    def test_each_structure_gets_a_row_with_what_the_alignment_found(self):
        self.open_with_two()
        rows = self.rows()
        self.assertEqual([row["name"] for row in rows], ["11", "22"])
        pivot, second = rows
        self.assertEqual(pivot["chips"][0][:2], ["Pivot", "chip pivot"])
        self.assertEqual(pivot["chips"][0][2], "The other structures are aligned to this one.")
        self.assertEqual(second["chips"][0][1], "chip ok")
        self.assertRegex(second["chips"][0][0], r"^RMSD \d\.\d Å$")
        self.assertRegex(second["chips"][0][2], r"^Aligned to the pivot: \d+ of \d+ Cα pairs, \d+% sequence identity\.$")
        self.assertEqual(pivot["size"], "33 residues")
        self.assertEqual(
            [[title, href] for title, _, _, href in second["buttons"]],
            [["Hide 22", "#i-eye"], ["Zoom to 22", "#i-crosshair"], ["Details of 22", "#i-chevron-down"],
             ["Remove 22", "#i-trash-2"]])
        self.assertTrue(all(title == label for title, label, _, _ in second["buttons"]))
        self.assertEqual(self.script("return document.getElementById('struct-count').textContent"), "(2)")
        self.assertEqual(self.script("return document.body.dataset.count"), "2")
        self.assertEqual(self.console_errors(), [])

    def test_the_eye_hides_a_structure_and_the_row_follows_the_scene(self):
        self.open_with_two()
        self.script("document.querySelector('#struct-list .srow:nth-child(2) button').click(); return 1")
        self.script("await new Promise(resolve => setTimeout(resolve, 300)); return 1")
        hidden = self.rows()[1]
        self.assertTrue(hidden["hidden"])
        self.assertEqual(hidden["buttons"][0][:3], ["Show 22", "Show 22", "true"])
        self.assertEqual(hidden["buttons"][0][3], "#i-eye-off")
        self.assertFalse(self.rows()[0]["hidden"])
        # A change made in Mol* itself shows in the list too.
        self.script("plugin.managers.structure.component.toggleVisibility(loadedStructures()[1].components); return 1")
        self.script("await new Promise(resolve => setTimeout(resolve, 300)); return 1")
        self.assertFalse(self.rows()[1]["hidden"])

    def test_a_removed_structure_comes_back_with_undo_as_it_was(self):
        self.open_with_two()
        before = self.script("return {fit: fit(1), files: loadedStructures().map(structureFile)}")
        self.script("document.querySelector('#struct-list .srow:nth-child(2) button:last-child').click(); return 1")
        self.script("await new Promise(resolve => setTimeout(resolve, 500)); return 1")
        gone = self.script("""
            return {files: loadedStructures().map(structureFile), rows: document.querySelectorAll('#struct-list .srow').length,
                    toast: document.getElementById('toast').textContent, shown: document.getElementById('toast').classList.contains('show'),
                    undo: !!document.querySelector('#toast .toast-action')};
            """)
        self.assertEqual(gone["files"], ["11.pdb"])
        self.assertEqual(gone["rows"], 1)
        self.assertEqual(gone["toast"], "Removed 22.Undo")
        self.assertTrue(gone["shown"] and gone["undo"])
        self.script("document.querySelector('#toast .toast-action').click(); return 1")
        self.script("await new Promise(resolve => setTimeout(resolve, 800)); await structureQueue; return 1")
        after = self.script("return {fit: fit(1), files: loadedStructures().map(structureFile), rows: document.querySelectorAll('#struct-list .srow').length}")
        self.assertEqual(after["files"], before["files"])
        self.assertEqual(after["rows"], 2)
        self.assertEqual(after["fit"]["matrix"], before["fit"]["matrix"])
        self.assertEqual(self.console_errors(), [])

    def test_the_theme_button_changes_the_page_and_mol_stars_skin_and_is_remembered(self):
        self.open_page()
        self.addCleanup(self.js, "localStorage.removeItem('ssn_theme')")
        self.js("localStorage.removeItem('ssn_theme')")
        state = """JSON.stringify({theme: document.documentElement.getAttribute('data-theme'),
            light: document.getElementById('skin-light').disabled, dark: document.getElementById('skin-dark').disabled,
            label: document.getElementById('theme-toggle-label').textContent,
            icon: document.querySelector('#theme-toggle-icon use').getAttribute('href'),
            canvas: canvasBackground(), saved: localStorage.getItem('ssn_theme')})"""
        # The test machine's own setting decides where it starts: make it light.
        self.script("applyTheme('light', false); return 1")
        light = json.loads(self.js(state))
        self.assertEqual((light["theme"], light["light"], light["dark"]), ("light", False, True))
        self.assertEqual((light["label"], light["icon"]), ("Dark Mode", "#i-moon"))
        self.assertEqual(light["canvas"].lower(), "#fafafa")
        self.js("document.getElementById('theme-toggle-btn').click()")
        dark = json.loads(self.js(state))
        self.assertEqual((dark["theme"], dark["light"], dark["dark"]), ("dark", True, False))
        self.assertEqual((dark["label"], dark["icon"], dark["saved"]), ("Light Mode", "#i-sun", "dark"))
        self.assertNotEqual(dark["canvas"], light["canvas"])
        # Another tab of the Viewer's pages choosing a theme moves this one too, without saving it again.
        self.js("window.dispatchEvent(new StorageEvent('storage', {key: 'ssn_theme', newValue: 'light'}))")
        self.assertEqual(json.loads(self.js(state))["theme"], "light")
        self.assertEqual(self.console_errors(), [])

    def test_the_advanced_door_shows_mol_stars_panels_and_comes_back_with_escape(self):
        self.open_with_two()
        door = """JSON.stringify({adv: document.body.dataset.adv, pressed: document.getElementById('t-adv').getAttribute('aria-pressed'),
            controls: plugin.layout.state.showControls, regions: document.querySelectorAll('#viewer .msp-layout-left').length,
            focus: document.activeElement.id, rows: document.querySelectorAll('#struct-list .srow').length})"""
        self.js("document.getElementById('t-adv').click()")
        self.pump(1.0)
        opened = json.loads(self.js(door))
        self.assertEqual((opened["adv"], opened["pressed"], opened["controls"]), ("1", "true", True))
        self.assertEqual((opened["regions"], opened["focus"], opened["rows"]), (1, "adv-back", 2))
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.pump(0.6)
        closed = json.loads(self.js(door))
        self.assertEqual((closed["adv"], closed["pressed"], closed["controls"]), ("0", "false", False))
        self.assertEqual((closed["regions"], closed["focus"], closed["rows"]), (0, "t-adv", 2))
        self.assertEqual(self.console_errors(), [])

    def test_the_help_dialog_takes_focus_and_gives_it_back(self):
        self.open_page()
        self.js("document.getElementById('help-btn').focus(); document.getElementById('help-btn').click()")
        opened = self.script("return {open: document.getElementById('help-modal').classList.contains('open'), focus: document.activeElement.id}")
        self.assertEqual(opened, {"open": True, "focus": "help-ok"})
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        closed = self.script("return {open: document.getElementById('help-modal').classList.contains('open'), focus: document.activeElement.id, adv: document.body.dataset.adv}")
        self.assertEqual(closed, {"open": False, "focus": "help-btn", "adv": "0"})

    def test_the_help_dialog_lists_every_way_to_use_the_canvas_the_page_has(self):
        # The strip's and the canvas's keys (esmfold_sequence.js), the tools' clicks (esmfold_tools.js) and Escape's order
        # (esmfold_shell.js) are each tested where they live; the dialog has to say them, in the page's words.
        self.open_page()
        mouse, keyboard = self.script("""
            return [...document.querySelectorAll('#help-modal table')].map(table =>
                [...table.rows].map(row => [...row.cells].map(cell => cell.textContent)));
            """)
        self.assertEqual(mouse, [
            ["Rotate", "Drag with the left button"],
            ["Zoom", "Scroll wheel, or pinch"],
            ["Move", "Drag with the right button, or Shift + drag"],
            ["Pin a residue", "Click it, in 3D or in the sequence; click empty space to let go"],
            ["Measure or label", "Choose the ruler or the tag in the toolbar, then click two residues (one for a label)"],
        ])
        self.assertEqual(keyboard, [
            ["Tab", "Moves through the controls; the 3D view and the sequence are one stop each"],
            ["Arrow keys", "Pin the next or previous residue; in the sequence, Up and Down move ten"],
            ["F", "Comes to the pinned residue"],
            ["R", "Resets the view"],
            ["Esc", "Closes a dialog, leaves Measure or Labels, lets go of the pin, or leaves the Advanced view, in that order"],
        ])

    # The layers' own names are the data of the residue file, not the page's words, and the residues of the
    # sequence strip (their letters and three-letter names) are the structure's data too.
    SHOWN = """(() => {
        const DATA = '#seq-body button, [data-coloring]:not([data-coloring^="builtin:"]):not([data-coloring="plddt"])';
        const shown = [document.title];
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            if (!node.parentElement.closest('script, style, code, #viewer, ' + DATA)) shown.push(node.textContent);
        }
        for (const element of document.querySelectorAll('[title], [aria-label]')) {
            if (element.matches(DATA) || element.parentElement.closest('#viewer, ' + DATA)) continue;
            for (const name of ['title', 'aria-label']) if (element.hasAttribute(name)) shown.push(element.getAttribute(name));
        }
        return shown.filter(text => text.trim());
    })()"""

    def test_every_text_of_the_page_comes_in_the_viewers_language(self):
        # Pseudo-language: a text that came from a catalog has no plain letter in it. The names of
        # the structures here are numbers, the <code> commands stay as typed, the layers' names are
        # the residue file's data, and pLDDT is a name; the pseudo-language leaves an 's. The strip's label
        # names its chain, which is data too (the chain's name ends the pseudo-translated text).
        pseudo_language(self, self.app)
        self.open_with_two()
        self.script("document.getElementById('help-btn').click(); return 1")
        shown = self.script("return " + self.SHOWN)
        self.assertGreater(len(shown), 40)
        plain = [text for text in shown
                 if re.search("[A-Za-z]", re.sub(r"'s\b|\.pdb\b|pLDDT|\bA\]$", "", text))]
        self.assertEqual(plain, [])
        self.assertEqual(self.js("document.documentElement.lang"), "en")
        self.assertEqual(self.console_errors(), [])

    def test_the_page_comes_in_simplified_chinese(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.open_with_two()
        found = self.script("""
            return {lang: document.documentElement.lang, header: document.querySelector('header h1').textContent,
                    status: document.getElementById('connection-status').textContent,
                    pivot: document.querySelector('#struct-list .chip.pivot').textContent,
                    size: document.querySelector('#struct-list .srow-sub span:last-child').textContent,
                    theme: document.getElementById('theme-toggle-label').textContent,
                    eye: document.querySelector('#struct-list .srow button').title};
            """)
        self.assertEqual(found, {"lang": "zh-CN", "header": "EMAP-SSN 结构视图", "status": "已连接", "pivot": "基准",
                                 "size": "33 个残基", "theme": "深色模式", "eye": "隐藏 11"})

    def test_losing_the_viewer_shows_a_banner_and_keeps_the_structures(self):
        self.open_with_two()
        banner = "getComputedStyle(document.querySelector('.banner')).display !== 'none'"
        self.assertFalse(self.js(banner))
        self.script("setConnection('reconnecting'); return 1")
        self.assertTrue(self.js(banner))
        self.assertEqual(self.js("document.getElementById('connection-status').textContent"), "Reconnecting...")
        self.assertEqual(self.script("return loadedStructures().length"), 2)
        self.script("setConnection('connected'); return 1")
        self.assertFalse(self.js(banner))


class ScriptPolicyTests(PageTestCase):
    """The page's script policy (Web_Server.PAGE_POLICY_ADDITIONS) keeps 'unsafe-eval' because Mol*'s bundle needs it to load."""

    def test_the_bundle_cannot_load_without_unsafe_eval_because_of_embinds_new_function(self):
        from unittest import mock
        from web_ui import Web_Server

        additions = {"esmfold.html": {"connect-src": "https: data:"}}
        with mock.patch.dict(Web_Server.PAGE_POLICY_ADDITIONS, additions):
            self.view.page().scripts().insert(self.record_script())
            self.load_page()
            self.pump(3)
            _, headers, _ = http_get(self.server, "/esmfold.html")
            self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
            self.assertNotIn("unsafe-eval", headers["Content-Security-Policy"])
            # The bundle stops where its MP4 encoder's embind code makes a named function with new Function, as it did in
            # 5.10.1 (that was the version this check was first made on), so there is no `molstar` for the page to use.
            self.assertEqual(self.js("typeof molstar"), "undefined")
            errors = json.loads(self.js("JSON.stringify(window.__errors)"))
            self.assertTrue(errors, "the bundle did not fail: 'unsafe-eval' may be droppable (see Web_Server.py)")
            self.assertIn("unsafe-eval", errors[0][0])
            self.assertIn("at new Function", errors[0][1])
            self.assertIn("/esmfold/molstar.js", errors[0][1])
            self.assertEqual(self.js("document.body.dataset.view"), "starting")

    def record_script(self):
        """A script that runs before the page's own, keeping what the page's scripts fail with."""
        from PySide6.QtWebEngineCore import QWebEngineScript

        script = QWebEngineScript()
        script.setName("errors")
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setSourceCode(
            "window.__errors = []; window.addEventListener('error', event => window.__errors.push("
            "[String(event.message), event.error && event.error.stack ? String(event.error.stack) : '']));")
        self.addCleanup(self.view.page().scripts().remove, script)
        return script


if __name__ == "__main__":
    unittest.main()
