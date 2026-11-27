# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's Colors and Representation card: its two tabs, the Layers group that starts with pLDDT, and the Representation tab.

The card has two tabs where its title was. "Colors" (open at every load) holds the colorings, with pLDDT the first chip of "Layers";
"Representation" holds one chip for each kind a structure can be drawn in, checked from what Mol* has (the kind every structure shares),
and gives every structure the kind that is chosen. Like the other page tests, they open esmfold.html in Qt WebEngine over the real
server and Mol*, and read its DOM and Mol*'s state (the engine has no WebGL, so what a kind looks like is not checked here).
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
from tests.esmfold_page_helpers import A, B, C, SEQUENCE, ColorTestCase

from PySide6.QtCore import Qt  # noqa: E402  (after the harness has set the engine's environment)
from PySide6.QtTest import QTest  # noqa: E402

KINDS = [["cartoon", "Cartoon"], ["ball-and-stick", "Ball and stick"], ["spacefill", "Spacefill"],
         ["gaussian-surface", "Surface"], ["backbone", "Backbone"]]
NOTES = {
    "cartoon": "The chain as a ribbon: helices as coils, strands as arrows.",
    "ball-and-stick": "Every atom as a ball and every bond as a stick.",
    "spacefill": "Every atom as a sphere of its own size.",
    "gaussian-surface": "A smooth surface around all the atoms.",
    "backbone": "The backbone alone, as a thin tube from residue to residue.",
}
DIFFERENT = "The structures are drawn in different ways. Choose one here for all of them, or set each in its row."


class CardTabsTestCase(ColorTestCase):
    def tabs(self):
        return self.script("""
            const tab = name => document.getElementById('card-tab-' + name);
            const list = document.querySelector('[data-tab="color"] [role="tablist"]');
            const panel = id => document.getElementById(id);
            return {
                list: [list.getAttribute('role'), list.getAttribute('aria-labelledby')],
                tabs: ['colors', 'representation'].map(name => [tab(name).getAttribute('role'), tab(name).textContent,
                    tab(name).getAttribute('aria-selected'), tab(name).getAttribute('aria-controls'), tab(name).tabIndex]),
                panels: ['color-body', 'representation-body'].map(id => [panel(id).getAttribute('role'),
                    panel(id).getAttribute('aria-labelledby'), panel(id).hidden]),
                heading: document.getElementById('h-color').textContent, focus: document.activeElement.id};
            """)

    def key(self, key):
        QTest.keyClick(self.view.focusProxy() or self.view, key)
        self.pump(0.2)

    def choose_kind(self, kind):
        """Click the chip for `kind`, let the structures follow, and return the Representation tab."""
        return self.script("""
            document.querySelector('#representation-body [data-representation="%s"]').click();
            await structureQueue;
            await cardNow();
            return repCard();
            """ % kind)

    def open_representation(self, *names):
        # A structure the test has not written is written, with no residue data, a little apart from the others.
        for index, name in enumerate(names):
            if not (Path(self.directory) / name).exists():
                self.write_with(name, shift=(3.0 * index, 1.0, 0.0), wobble=0.4 * index)
        self.open_scene(*names)
        self.click("#card-tab-representation", 100)


class TabTests(CardTabsTestCase):
    def test_the_tabs_have_their_roles_and_labels_and_colors_is_open_with_the_representation_panel_hidden(self):
        self.open_page()
        state = self.tabs()
        self.assertEqual(state["list"], ["tablist", "h-color"])
        self.assertEqual(state["heading"], "Colors and representation")
        self.assertEqual(state["tabs"], [["tab", "Colors", "true", "color-body", 0],
                                         ["tab", "Representation", "false", "representation-body", -1]])
        self.assertEqual(state["panels"], [["tabpanel", "card-tab-colors", False], ["tabpanel", "card-tab-representation", True]])
        self.assertEqual(self.console_errors(), [])

    def test_a_click_opens_the_other_tab_and_hides_the_panel_of_the_first(self):
        self.open_page()
        self.click("#card-tab-representation", 100)
        state = self.tabs()
        self.assertEqual([tab[2:] for tab in state["tabs"]], [["false", "color-body", -1], ["true", "representation-body", 0]])
        self.assertEqual([panel[2] for panel in state["panels"]], [True, False])
        self.click("#card-tab-colors", 100)
        state = self.tabs()
        self.assertEqual([tab[2] for tab in state["tabs"]], ["true", "false"])
        self.assertEqual([panel[2] for panel in state["panels"]], [False, True])

    def test_the_arrow_keys_home_and_end_move_between_the_tabs_and_open_them_with_one_tab_stop(self):
        self.open_page()
        self.script("document.getElementById('card-tab-colors').focus(); return 1")
        for key, opened in ((Qt.Key_Right, "representation"), (Qt.Key_Right, "colors"), (Qt.Key_Left, "representation"),
                            (Qt.Key_Home, "colors"), (Qt.Key_End, "representation"), (Qt.Key_Left, "colors")):
            with self.subTest(key=key):
                self.key(key)
                state = self.tabs()
                self.assertEqual(state["focus"], "card-tab-" + opened)
                self.assertEqual([tab[2] for tab in state["tabs"]], ["true" if name == opened else "false" for name in ("colors", "representation")])
                self.assertEqual([panel[2] for panel in state["panels"]], [opened != "colors", opened == "colors"])
                # One Tab stop: only the open tab can be reached by Tab.
                self.assertEqual([tab[4] for tab in state["tabs"]], [0 if name == opened else -1 for name in ("colors", "representation")])

    def test_the_card_opens_on_colors_at_every_load_whichever_tab_was_open(self):
        self.open_page()
        self.click("#card-tab-representation", 100)
        self.assertEqual(self.tabs()["tabs"][1][2], "true")
        self.open_page()
        state = self.tabs()
        self.assertEqual([tab[2] for tab in state["tabs"]], ["true", "false"])
        self.assertEqual([panel[2] for panel in state["panels"]], [False, True])

    def test_opening_either_tab_leaves_both_tabs_where_they_were(self):
        self.open_page()
        edges = """return ['colors', 'representation'].map(name => {
            const box = document.getElementById('card-tab-' + name).getBoundingClientRect();
            return [box.left, box.width];});"""
        before = self.script(edges)
        for name in ("representation", "colors", "representation"):
            self.click("#card-tab-" + name, 100)
            after = self.script(edges)
            for old, new in zip(before, after):
                self.assertAlmostEqual(old[0], new[0], delta=0.5, msg=(name, before, after))
                self.assertAlmostEqual(old[1], new[1], delta=0.5, msg=(name, before, after))
        # Both are weight 600: the open one is told by its colour and line, not by being bolder.
        self.assertEqual(self.script("return ['colors', 'representation'].map(n => getComputedStyle(document.getElementById('card-tab-' + n)).fontWeight)"),
                         ["600", "600"])

    def test_in_simplified_chinese_the_tabs_and_the_heading_are_translated(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.open_page()
        state = self.tabs()
        self.assertEqual([tab[1] for tab in state["tabs"]], ["颜色", "表示方式"])
        self.assertNotEqual(state["heading"], "Colors and representation")
        self.assertTrue(state["heading"])


class ColorsTabTests(CardTabsTestCase):
    def write_layers(self, count):
        fixtures.write_structure(self.directory, A, SEQUENCE, sidecar=False)
        length = len(SEQUENCE)
        properties = {f"prop_{index}": fixtures.residue_property(
            f"Property {index}", "", [0, 1], [round(((index + step) % 10) / 10, 1) for step in range(length)], None, "")
            for index in range(count)}
        fixtures.write_sidecar(self.directory, A, fixtures.residue_data(A, SEQUENCE, properties=properties))

    def test_the_groups_are_layers_and_properties_and_plddt_is_the_first_chip_of_layers_in_one_radio_group_with_them(self):
        self.write_with(A, "information_content", "same_residue_pct")
        self.open_scene(A)
        state = self.script("""
            const body = document.getElementById('color-body');
            const group = body.querySelector('.seg-group');
            const first = group.querySelector('.seg-btn');
            const radios = [...group.querySelectorAll('input[type="radio"]')];
            return {labels: [...body.querySelectorAll('.group-label')].map(n => n.textContent),
                    first: first.dataset.coloring, firstLabel: group.querySelector('.group-label').textContent,
                    role: group.getAttribute('role'), name: group.getAttribute('aria-label'),
                    ids: radios.map(r => r.value), names: [...new Set(radios.map(r => r.name))],
                    before: [...group.children].map(n => n.className)};
            """)
        self.assertEqual(state["labels"], ["Layers", "Properties"])
        self.assertEqual(state["first"], "plddt")
        self.assertEqual((state["role"], state["name"]), ("radiogroup", "Colors"))
        self.assertEqual(state["ids"], ["plddt", "information_content", "same_residue_pct"])
        self.assertEqual(state["names"], ["coloring"])
        # The arrow keys go from pLDDT to the layers.
        self.script("document.querySelector('#color-body [data-coloring=\"plddt\"] input').focus(); return 1")
        self.key(Qt.Key_Right)
        self.script("await structureQueue; return 1")
        self.assertEqual(self.script("return document.activeElement.closest('.seg-btn').dataset.coloring"), "information_content")

    def test_in_simplified_chinese_the_group_titles_are_layers_and_properties(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.write_with(A, "information_content")
        self.open_scene(A)
        self.assertEqual(self.script("return [...document.querySelectorAll('#color-body .group-label')].map(n => n.textContent)"),
                         ["图层", "性质"])
        self.assertEqual(self.script("return document.querySelector('#color-body .seg-group').getAttribute('aria-label')"), "颜色")

    def test_with_many_layers_plddt_is_outside_the_scroller_and_the_filter_and_stays_in_view(self):
        self.write_layers(10)
        self.open_scene(A)
        state = self.script("""
            const body = document.getElementById('color-body');
            const plddt = body.querySelector('[data-coloring="plddt"]');
            const scroller = body.querySelector('.seg.scroll');
            return {outside: !scroller.contains(plddt), inScroller: scroller.querySelectorAll('.seg-btn').length,
                    order: [...body.querySelector('.seg-group').children].map(n => n.matches('.filter') ? 'filter' : n.matches('.seg.scroll') ? 'scroll'
                        : n.matches('.group-label') ? 'label' : n.querySelector('[data-coloring="plddt"]') ? 'plddt' : '?')};
            """)
        self.assertEqual(state, {"outside": True, "inScroller": 10, "order": ["label", "plddt", "filter", "scroll"]})
        self.script("""
            const filter = document.querySelector('#color-body .filter');
            filter.focus(); filter.value = 'nothing like it'; filter.dispatchEvent(new Event('input', {bubbles: true}));
            await new Promise(resolve => setTimeout(resolve, 300));
            return 1;
            """)
        shown = self.script("""
            const body = document.getElementById('color-body');
            const chip = body.querySelector('[data-coloring="plddt"]');
            const box = chip.getBoundingClientRect();
            return {hint: body.querySelector('.hint').textContent, visible: box.width > 0 && box.height > 0 && !chip.closest('[hidden]'),
                    selected: body.querySelector('[data-coloring="plddt"] input').checked};
            """)
        self.assertEqual(shown, {"hint": "No layer matches.", "visible": True, "selected": True})

    def test_with_no_layer_layers_shows_plddt_alone_and_the_advice_stays_under_it(self):
        self.write_with(A)
        self.open_scene(A)
        state = self.script("""
            const body = document.getElementById('color-body');
            const group = body.querySelector('.seg-group');
            const empty = body.querySelector('.empty-layers');
            return {chips: [...group.querySelectorAll('.seg-btn')].map(c => c.dataset.coloring),
                    tag: empty.tagName, open: !!empty.open,
                    advice: empty.querySelector('summary > span, :scope > span').textContent,
                    reasons: [...empty.querySelectorAll('.empty-layer-list > div')].map(n => n.textContent),
                    after: !!(group.compareDocumentPosition(empty) & Node.DOCUMENT_POSITION_FOLLOWING),
                    beforeProperties: !!(empty.compareDocumentPosition(body.querySelector('.properties')) & Node.DOCUMENT_POSITION_FOLLOWING)};
            """)
        self.assertEqual(state["chips"], ["plddt"])
        self.assertEqual((state["tag"], state["open"], state["after"], state["beforeProperties"]), ("DETAILS", False, True, True))
        self.assertEqual(state["advice"], "No extra layers. Fold with an alignment loaded.")
        self.assertEqual(len(state["reasons"]), 1)
        self.assertTrue(state["reasons"][0].startswith("11: "), state["reasons"])


class RepresentationTabTests(CardTabsTestCase):
    def test_the_five_chips_are_in_order_and_with_one_kind_shared_that_chip_is_checked_with_its_note(self):
        self.open_representation(A, B)
        state = self.script("return repCard()")
        self.assertEqual(state["chips"], KINDS)
        self.assertEqual((state["checked"], state["note"], state["group"], state["hidden"]),
                         ("cartoon", NOTES["cartoon"], "Representation", False))
        self.assertEqual(self.script("return [...document.querySelectorAll('#representation-body input')].map(i => i.name)"), ["representation"] * 5)
        self.assertEqual(self.console_errors(), [])

    def test_choosing_a_chip_gives_every_structure_the_kind_and_keeps_each_ones_color_and_opacity(self):
        self.write_with(A, "information_content")
        self.write_with(B, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.write_with(C, shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.open_representation(A, B, C)
        # Each structure has a color and an opacity of its own to keep.
        self.script("""
            await runExclusive(() => applyLayer(loadedStructures()[0], 'information_content'));
            await setOpacity(loadedStructures()[1].cell.transform.ref, 40);
            await setStructureColoring(loadedStructures()[2].cell.transform.ref, STRUCTURE_COLORING);
            await structureQueue;
            return 1;
            """)
        before = self.script("return drawn()")
        self.assertEqual([item["opacity"] for item in before], [100, 40, 100])
        self.assertEqual(len({item["theme"] for item in before}), 3)
        for kind, label in KINDS[1:] + KINDS[:1]:
            with self.subTest(kind=kind):
                state = self.choose_kind(kind)
                self.assertEqual((state["checked"], state["note"]), (kind, NOTES[kind]))
                after = self.script("return drawn()")
                self.assertEqual([item["type"] for item in after], [kind] * 3)
                self.assertEqual([item["opacity"] for item in after], [100, 40, 100])
                self.assertEqual([item["theme"] for item in after], [item["theme"] for item in before])
        self.assertEqual(self.console_errors(), [])

    def test_the_rows_selects_follow_the_chip_and_a_rows_own_change_leaves_no_chip_checked_and_says_so(self):
        self.open_representation(A, B)
        self.open_details(1)
        self.open_details(2)
        self.choose_kind("spacefill")
        self.settle(300)
        self.assertEqual([self.details(1)["representation"], self.details(2)["representation"]], ["spacefill", "spacefill"])
        # A row's own change affects that structure only, and the tab then says what is true.
        self.script("""
            const select = document.querySelector('#struct-list .srow:nth-child(1) select');
            select.value = 'ball-and-stick'; select.dispatchEvent(new Event('change', {bubbles: true}));
            await structureQueue;
            await new Promise(resolve => setTimeout(resolve, 500));
            await cardNow();
            return 1;
            """)
        self.assertEqual([item["type"] for item in self.script("return drawn()")], ["ball-and-stick", "spacefill"])
        state = self.script("return repCard()")
        self.assertEqual((state["checked"], state["note"]), (None, DIFFERENT))
        # And a chip makes them one again.
        state = self.choose_kind("backbone")
        self.assertEqual((state["checked"], state["note"]), ("backbone", NOTES["backbone"]))
        self.assertEqual([item["type"] for item in self.script("return drawn()")], ["backbone"] * 2)

    def test_a_kind_outside_the_five_leaves_no_chip_checked(self):
        self.open_representation(A)
        self.script("""
            const ref = loadedStructures()[0].cell.transform.ref;
            await runExclusive(() => changeRepresentation(ref, 'putty'));
            await cardNow();
            return 1;
            """)
        state = self.script("return repCard()")
        self.assertEqual((state["checked"], state["note"]), (None, DIFFERENT))

    def test_the_change_is_one_task_that_says_so_and_saves_the_session_once(self):
        self.open_representation(A, B)
        self.settle(1800)
        del self.saved[:]
        self.script("""
            window.__statuses = [];
            const show = showStatus;
            showStatus = (text, ...rest) => { window.__statuses.push(text); return show(text, ...rest); };
            document.querySelector('#representation-body [data-representation="spacefill"]').click();
            window.__started = loadedStructures().map(representationType);
            await structureQueue;
            return 1;
            """)
        self.settle(2200)
        self.assertEqual(self.script("return window.__statuses"), ["Changing the representation..."])
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(self.script("return document.getElementById('toast').classList.contains('show')"), False)

    def test_a_failure_is_reported_as_the_rows_select_reports_it(self):
        self.open_representation(A)
        self.script("""
            window.__reported = [];
            const original = changeRepresentation;
            changeRepresentation = async () => { throw new Error('no way'); };
            const report = reportFailure;
            reportFailure = (message, err) => { window.__reported.push([message, err.message]); };
            document.querySelector('#representation-body [data-representation="spacefill"]').click();
            await structureQueue;
            await new Promise(resolve => setTimeout(resolve, 200));
            changeRepresentation = original; reportFailure = report;
            return 1;
            """)
        self.assertEqual(self.script("return window.__reported"), [["Could not change the representation:", "no way"]])

    def test_a_structure_that_arrives_takes_the_kind_the_others_share_and_keeps_its_color(self):
        self.write_with(A)
        self.write_with(B, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.write_with(C, shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.open_representation(A, B)
        self.choose_kind("spacefill")
        self.push(C)
        self.settle(300)
        after = self.script("return drawn()")
        self.assertEqual([item["type"] for item in after], ["spacefill"] * 3)
        # It was given the scene's coloring (pLDDT), as the others have.
        self.assertEqual(len({item["theme"] for item in after}), 1)
        self.assertEqual(self.script("return coloringOf(loadedStructures()[2])"), "plddt")
        state = self.script("await cardNow(); return repCard()")
        self.assertEqual(state["checked"], "spacefill")
        self.assertEqual(self.console_errors(), [])

    def test_a_structure_that_arrives_into_structures_that_differ_keeps_the_default_and_the_first_into_none_too(self):
        self.write_all()
        self.open_page()
        self.push("11.pdb")
        self.assertEqual(self.script("return drawn().map(item => item.type)"), ["cartoon"])
        self.script("""
            await runExclusive(() => setRepresentation(loadedStructures()[0].cell.transform.ref, 'ball-and-stick'));
            return 1;
            """)
        # (One structure shares its kind with itself, so the second takes it.)
        self.push("22.pdb")
        self.assertEqual(self.script("return drawn().map(item => item.type)"), ["ball-and-stick", "ball-and-stick"])
        self.script("""
            await runExclusive(() => setRepresentation(loadedStructures()[1].cell.transform.ref, 'cartoon'));
            return 1;
            """)
        self.push("33.pdb")
        self.assertEqual(self.script("return drawn().map(item => item.type)"), ["ball-and-stick", "cartoon", "cartoon"])
        self.assertEqual(self.script("await cardNow(); return repCard().checked"), None)

    EMPTY = "There is no structure in the scene yet."
    OFF = """
        const radios = [...document.querySelectorAll('#representation-body input')];
        return {disabled: radios.map(r => r.disabled), checked: radios.filter(r => r.checked).length, note: repCard().note,
                cursor: radios.map(r => getComputedStyle(r.closest('.seg-btn')).cursor)};"""

    def test_with_no_structure_the_five_radios_are_off_none_is_checked_and_the_note_says_so_until_one_arrives_and_again_after_the_last_goes(self):
        self.write(A)
        self.open_page()
        self.click("#card-tab-representation", 100)
        state = self.script(self.OFF)
        self.assertEqual(state, {"disabled": [True] * 5, "checked": 0, "note": self.EMPTY, "cursor": ["not-allowed"] * 5})
        self.assertEqual(self.script("return repCard().chips"), KINDS)
        # A click on an off chip changes nothing and the focus cannot come to it.
        self.script("""
            document.querySelector('#representation-body [data-representation="spacefill"]').click();
            document.querySelector('#representation-body input').focus();
            await structureQueue;
            return 1;""")
        self.assertEqual(self.script("return document.activeElement.closest('#representation-body') === null"), True)
        self.assertEqual(self.script(self.OFF)["checked"], 0)
        # A structure arrives: the chips are live and show its kind.
        self.push(A)
        self.settle(200)
        state = self.script(self.OFF)
        self.assertEqual((state["disabled"], state["checked"], state["note"]), ([False] * 5, 1, NOTES["cartoon"]))
        self.assertEqual(self.script("return repCard().checked"), "cartoon")
        # The last one is removed: off again.
        self.script("removeStructure(loadedStructures()[0]); await structureQueue; return 1")
        self.settle(300)
        state = self.script("await cardNow(); " + self.OFF)
        self.assertEqual((state["disabled"], state["checked"], state["note"]), ([True] * 5, 0, self.EMPTY))
        self.assertEqual(self.console_errors(), [])

    def test_in_simplified_chinese_the_empty_scene_note_is_translated(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.open_page()
        note = self.script("return repCard().note")
        self.assertNotEqual(note, self.EMPTY)
        self.assertTrue(note)
        self.assertEqual(self.script("return [...document.querySelectorAll('#representation-body input')].every(r => r.disabled)"), True)

    def test_a_session_saved_and_restored_shows_the_restored_kind(self):
        self.open_representation(A, B)
        self.settle(1800)
        del self.saved[:]
        self.choose_kind("gaussian-surface")
        self.settle(2000)
        self.assertTrue(self.saved)
        self.session = json.loads(json.dumps(self.saved[-1]))
        self.open_page()
        self.settle(300)
        self.assertEqual(self.script("return drawn().map(item => item.type)"), ["gaussian-surface"] * 2)
        self.click("#card-tab-representation", 100)
        state = self.script("await cardNow(); return repCard()")
        self.assertEqual((state["checked"], state["note"]), ("gaussian-surface", NOTES["gaussian-surface"]))


class RepresentationTextTests(CardTabsTestCase):
    def test_in_simplified_chinese_the_chips_and_every_note_are_translated(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.open_representation(A, B)
        seen = []
        for kind, _ in KINDS:
            state = self.choose_kind(kind)
            seen.append(state["note"])
            self.assertNotEqual(state["note"], NOTES[kind])
        self.script("""
            const ref = loadedStructures()[0].cell.transform.ref;
            await runExclusive(() => setRepresentation(ref, 'cartoon'));
            await cardNow();
            return 1;
            """)
        state = self.script("return repCard()")
        seen.append(state["note"])
        self.assertNotEqual(state["note"], DIFFERENT)
        self.assertEqual(len(set(seen)), 6)
        self.assertNotEqual([chip[1] for chip in state["chips"]], [chip[1] for chip in KINDS])


if __name__ == "__main__":
    unittest.main()
