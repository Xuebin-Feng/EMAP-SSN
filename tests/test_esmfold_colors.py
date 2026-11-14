# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's Color by card (esmfold_color.js).

One choice of coloring for the whole scene (pLDDT, a structure's own color, a
layer of the Viewer's residue data, or one of Mol*'s secondary-structure and
residue-position colorings), the legend of the chosen one, the compact legend
on the canvas, a coloring of its own for each structure, and the grey that a
structure with no values for the chosen layer shows. The layers' own logic
(attaching, switching, restoring) is tested in tests/test_esmfold_layers.py;
here the card, and the way the scene follows it. Like the other page tests,
they open esmfold.html in Qt WebEngine over the real server and Mol*, and read
its DOM and Mol*'s state.
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
from tests.test_esmfold_structures import StructuresTestCase

from PySide6.QtCore import Qt  # noqa: E402  (after the harness has set the engine's environment)
from PySide6.QtTest import QTest  # noqa: E402

GREY = 0xDCDCDC
A, B, C = "11.pdb", "22.pdb", "33.pdb"
SEQUENCE = fixtures.FIRST_SEQUENCE


def hydrophobicity(length):
    return [round(((index * 3) % 11) - 5.0, 1) for index in range(length)]


class ColorTestCase(StructuresTestCase):
    def write_with(self, name, *properties, sequence=SEQUENCE, **options):
        """A structure with residue data that holds only the named fixture properties."""
        fixtures.write_structure(self.directory, name, sequence, sidecar=False, **options)
        length = len(sequence)
        available = {
            "information_content": fixtures.residue_property(
                "Information Content", "%", [0, 100], fixtures.information_content(length), None, "How conserved."),
            "same_residue_pct": fixtures.residue_property(
                "Residue Agreement", "%", [0, 100], fixtures.same_residue_pct(length), None, "How alike."),
            "hydro": fixtures.residue_property("Hydrophobicity", "", [-5, 5], hydrophobicity(length), None, "Greasy."),
        }
        if properties:
            data = fixtures.residue_data(name, sequence, properties={key: available[key] for key in properties})
            fixtures.write_sidecar(self.directory, name, data)

    def open_scene(self, *names):
        self.open_page()
        for name in names:
            self.push(name)
        self.settle(200)

    def choose(self, coloring):
        """Click the card's button for `coloring`, let the scene follow, and return the card."""
        return self.script("""
            document.querySelector('#color-body [data-coloring="%s"]').click();
            await structureQueue;
            return await cardNow();
            """ % coloring)

    def shown(self):
        """What each structure shows: its coloring id, its first theme's name and colour parameter, its row's chips."""
        return self.script("""
            await new Promise(resolve => setTimeout(resolve, 150));
            const rows = [...document.querySelectorAll('#struct-list .srow')];
            return loadedStructures().map((structure, index) => ({
                name: structureName(structure), coloring: coloringOf(structure), theme: reprThemes()[index].name,
                value: reprThemes()[index].params.value ?? null, layer: layerState(index).active,
                chips: [...rows[index].querySelectorAll('.srow-main .chip')].map(chip => chip.textContent),
                swatch: rows[index].querySelector('.srow-swatch').style.getPropertyValue('--sw')}));
            """)


class SceneColoringTests(ColorTestCase):
    def test_the_card_offers_the_union_of_the_layers_and_says_how_many_structures_have_each(self):
        self.write_with(A, "information_content", "hydro")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.write_with(C, shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.open_scene(A, B, C)
        card = self.script("return await cardNow()")
        self.assertEqual(card["chips"], [
            ["builtin:structure", "Structure color", ""], ["plddt", "pLDDT", ""],
            ["information_content", "Information Content", "2 of 3"], ["hydro", "Hydrophobicity", "1 of 3"]])
        self.assertEqual(card["selected"], "plddt")
        # The structure with no residue data says why, in the Viewer's words.
        self.assertEqual(len(card["layerNotes"]), 1)
        self.assertTrue(card["layerNotes"][0].startswith("33: "), card["layerNotes"])
        self.assertEqual(card["legendTitle"], "pLDDT")
        self.assertEqual(len(card["keys"]), 4)

    def test_a_structure_with_no_values_for_the_chosen_layer_is_grey_and_says_so_and_comes_back(self):
        self.write_with(A, "information_content", "hydro")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.write_with(C, shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.open_scene(A, B, C)
        card = self.choose("hydro")
        shown = self.shown()
        self.assertEqual([item["coloring"] for item in shown], ["hydro", "nodata", "nodata"])
        for item in shown[1:]:
            self.assertEqual((item["theme"], item["value"], item["layer"]), ("uniform", GREY, "plddt"), item)
            self.assertIn("No data", item["chips"])
        self.assertNotIn("No data", shown[0]["chips"])
        self.assertEqual(card["selected"], "hydro")
        self.assertEqual(card["legendTitle"], "Hydrophobicity")
        self.assertEqual(card["noData"], "No data: 22, 33. This layer has no values for them.")
        self.assertEqual(card["compact"]["title"], "Hydrophobicity")
        # Back to pLDDT: the grey structures have their confidence colors again, and no chip.
        self.choose("plddt")
        shown = self.shown()
        self.assertEqual([item["coloring"] for item in shown], ["plddt"] * 3)
        self.assertEqual([item["theme"] for item in shown], ["plddt-confidence"] * 3)
        self.assertTrue(all("No data" not in item["chips"] for item in shown))
        before = self.script("return colors(1, 'plddt-confidence', {})")
        plddt = fixtures.plddt_values(33)
        self.assertEqual(len(before), 33)
        self.assertEqual(len(plddt), 33)
        self.assertEqual(self.console_errors(), [])

    def test_a_structure_loaded_after_the_choice_takes_it_and_the_others_keep_theirs(self):
        self.write_with(A, "information_content", "hydro")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.write_with(C, shift=(-2.0, 4.0, 1.0), wobble=0.7)
        self.open_scene(A)
        self.choose("information_content")
        # Mol* is told directly that the first structure shows another layer: a load must leave it alone.
        self.script("await runExclusive(() => applyLayer(loadedStructures()[0], 'hydro')); return 1")
        self.push(B)
        self.push(C)
        shown = self.shown()
        self.assertEqual([item["coloring"] for item in shown], ["hydro", "information_content", "nodata"])
        self.assertEqual(shown[2]["value"], GREY)
        self.assertIn("No data", shown[2]["chips"])
        # And the default, with nothing chosen, is pLDDT for each new structure.
        self.open_page()
        self.push(A)
        self.push(B)
        self.assertEqual([item["coloring"] for item in self.shown()], ["plddt", "plddt"])

    def test_a_row_can_color_its_structure_on_its_own_and_the_scene_takes_it_back(self):
        self.write_with(A, "information_content", "same_residue_pct")
        self.write_with(B, "information_content", "same_residue_pct", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        self.open_details(2)
        options = self.script("""
            const select = document.querySelector('#struct-list .srow:nth-child(2) select[id^="col-"]');
            return {options: [...select.options].map(o => [o.value, o.textContent]), value: select.value};
            """)
        self.assertEqual(options["options"], [
            ["", "Same as the scene"], ["plddt", "pLDDT"], ["builtin:structure", "Structure color"],
            ["information_content", "Information Content"], ["same_residue_pct", "Residue Agreement"]])
        self.assertEqual(options["value"], "")
        self.script("""
            const select = document.querySelector('#struct-list .srow:nth-child(2) select[id^="col-"]');
            select.value = 'same_residue_pct'; select.dispatchEvent(new Event('change', {bubbles: true}));
            await structureQueue;
            return 1;
            """)
        self.settle(500)
        shown = self.shown()
        self.assertEqual([item["coloring"] for item in shown], ["plddt", "same_residue_pct"])
        self.assertEqual(self.script("return document.querySelector('#struct-list .srow:nth-child(2) select[id^=\"col-\"]').value"),
                         "same_residue_pct")
        self.assertEqual(self.script("return card().selected"), "plddt")
        # The scene's choice is for the whole scene: both follow it, and the row is "Same as the scene" again.
        self.choose("information_content")
        self.settle(300)
        self.assertEqual([item["coloring"] for item in self.shown()], ["information_content"] * 2)
        self.assertEqual(self.script("return document.querySelector('#struct-list .srow:nth-child(2) select[id^=\"col-\"]').value"), "")
        # "Same as the scene" in the row gives a structure back to it.
        self.script("""
            const select = document.querySelector('#struct-list .srow:nth-child(2) select[id^="col-"]');
            select.value = 'plddt'; select.dispatchEvent(new Event('change', {bubbles: true}));
            await structureQueue;
            select.value = ''; select.dispatchEvent(new Event('change', {bubbles: true}));
            await structureQueue;
            return 1;
            """)
        self.settle(500)
        self.assertEqual([item["coloring"] for item in self.shown()], ["information_content"] * 2)
        self.assertEqual(self.console_errors(), [])

    def test_the_built_in_colorings_and_each_structures_own_color(self):
        self.write_with(A, "information_content")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        self.choose("information_content")
        card = self.choose("builtin:structure")
        shown = self.shown()
        self.assertEqual([item["coloring"] for item in shown], ["builtin:structure"] * 2)
        # Each structure has the color of its swatch, and the layer let go.
        for item in shown:
            self.assertEqual(item["theme"], "uniform")
            self.assertEqual("#%06x" % item["value"], item["swatch"])
            self.assertEqual(item["layer"], "plddt")
        self.assertNotEqual(shown[0]["value"], shown[1]["value"])
        self.assertEqual(card["selected"], "builtin:structure")
        self.assertEqual(card["compact"]["shown"], "0")
        # The others are behind "More colorings".
        self.assertEqual([chip[0] for chip in card["chips"] if chip[0] in ("builtin:secondary-structure", "builtin:residue-position")], [])
        self.script("document.querySelector('#color-body .more-toggle').click(); return 1")
        opened = self.script("return await cardNow()")
        self.assertEqual([chip[0] for chip in opened["chips"]][-2:], ["builtin:secondary-structure", "builtin:residue-position"])
        secondary = self.choose("builtin:secondary-structure")
        self.assertEqual([item["theme"] for item in self.shown()], ["secondary-structure"] * 2)
        self.assertEqual(secondary["legendTitle"], "Secondary structure")
        self.assertEqual(secondary["keys"], ["Helix", "Strand", "Coil"])
        self.assertEqual(secondary["compact"]["shown"], "1")
        position = self.choose("builtin:residue-position")
        self.assertEqual([item["theme"] for item in self.shown()], ["sequence-id"] * 2)
        self.assertEqual(position["ends"], ["N-terminus", "C-terminus"])
        self.assertTrue(position["bar"])
        # And from there to a layer: it attaches again.
        self.choose("information_content")
        self.assertEqual([item["layer"] for item in self.shown()], ["information_content"] * 2)
        self.assertEqual(self.console_errors(), [])

    def test_the_scene_choice_is_worked_out_from_the_structures_when_a_session_comes_back(self):
        self.write_with(A, "information_content")
        self.write_with(B, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        self.settle(1800)
        del self.saved[:]
        self.choose("information_content")
        self.settle(1800)
        self.assertTrue(self.saved)
        snapshot = json.loads(json.dumps(self.saved[-1]))
        # One structure has the layer, the other none: grey, and not counted against the choice.
        self.assertEqual([item["coloring"] for item in self.shown()], ["information_content", "nodata"])
        self.session = snapshot
        self.open_page()
        self.settle(300)
        card = self.script("return await cardNow()")
        self.assertEqual(card["selected"], "information_content")
        shown = self.shown()
        self.assertEqual([item["coloring"] for item in shown], ["information_content", "nodata"])
        self.assertIn("No data", shown[1]["chips"])
        # A scene colored by each structure's own color keeps those colors after a restore.
        self.choose("builtin:structure")
        before = [item["value"] for item in self.shown()]
        self.settle(1800)
        self.session = json.loads(json.dumps(self.saved[-1]))
        self.open_page()
        self.settle(300)
        after = self.shown()
        self.assertEqual([item["value"] for item in after], before)
        self.assertEqual(self.script("return card().selected"), "builtin:structure")
        self.assertEqual(["#%06x" % item["value"] for item in after], [item["swatch"] for item in after])


class LayerNameTests(ColorTestCase):
    """The names of the two layers the Viewer writes are shown in the page's language, not saved that way."""

    def open_with_three_layers(self):
        self.write_with(A, "information_content", "same_residue_pct", "hydro")
        self.open_scene(A)

    def layer_chips(self, card):
        return {chip[0]: chip[1] for chip in card["chips"] if chip[0] not in ("plddt", "builtin:structure")}

    def test_in_simplified_chinese_the_card_and_its_legend_name_them_and_another_writers_layer_keeps_its_name(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.open_with_three_layers()
        card = self.script("return await cardNow()")
        self.assertEqual(self.layer_chips(card), {
            "information_content": "信息含量", "same_residue_pct": "残基一致率", "hydro": "Hydrophobicity"})
        card = self.choose("information_content")
        self.assertEqual(card["legendTitle"], "信息含量")
        self.assertTrue(card["compact"]["title"].startswith("信息含量"), card["compact"])
        self.assertEqual(self.console_errors(), [])

    def test_in_english_the_names_are_the_files_and_the_file_keeps_its_english_in_any_language(self):
        from desktop.Desktop_App import install_translations

        self.open_with_three_layers()
        self.assertEqual(self.layer_chips(self.script("return await cardNow()")), {
            "information_content": "Information Content", "same_residue_pct": "Residue Agreement", "hydro": "Hydrophobicity"})
        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.open_page()
        self.push(A)
        self.assertEqual(self.layer_chips(self.script("return await cardNow()"))["information_content"], "信息含量")
        # What the page shows is a choice of the page: the saved file keeps the label it was written with.
        with open(Path(self.directory) / "11.residues.json", encoding="utf-8") as handle:
            labels = [entry["label"] for entry in json.load(handle)["properties"].values()]
        self.assertEqual(labels, ["Information Content", "Residue Agreement", "Hydrophobicity"])


class CardTests(ColorTestCase):
    def test_more_than_eight_layers_get_a_filter_and_a_scroll_area_and_fewer_do_not(self):
        fixtures.write_structure(self.directory, A, SEQUENCE, sidecar=False)
        length = len(SEQUENCE)
        properties = {f"prop_{index}": fixtures.residue_property(
            f"Property {index}", "", [0, 1], [round(((index + step) % 10) / 10, 1) for step in range(length)], None, "")
            for index in range(10)}
        fixtures.write_sidecar(self.directory, A, fixtures.residue_data(A, SEQUENCE, properties=properties))
        self.open_scene(A)
        many = self.script("""
            const body = document.getElementById('color-body');
            return {filter: !!body.querySelector('.filter'), scroll: !!body.querySelector('.seg.scroll'),
                    chips: body.querySelectorAll('.seg-btn').length};
            """)
        self.assertEqual(many, {"filter": True, "scroll": True, "chips": 12})
        # Typing narrows the buttons, and the filter keeps the focus.
        self.script("""
            const filter = document.querySelector('#color-body .filter');
            filter.focus(); filter.value = 'property 7'; filter.dispatchEvent(new Event('input', {bubbles: true}));
            await new Promise(resolve => setTimeout(resolve, 300));
            return 1;
            """)
        narrowed = self.script("""
            const body = document.getElementById('color-body');
            return {labels: [...body.querySelectorAll('.seg.scroll .seg-btn')].map(c => c.querySelector('.seg-label').textContent),
                    focus: document.activeElement.classList.contains('filter')};
            """)
        self.assertEqual(narrowed, {"labels": ["Property 7"], "focus": True})
        self.script("""
            const filter = document.querySelector('#color-body .filter');
            filter.value = 'nothing'; filter.dispatchEvent(new Event('input', {bubbles: true}));
            await new Promise(resolve => setTimeout(resolve, 300));
            return 1;
            """)
        self.assertEqual(self.script("return document.querySelector('#color-body .hint').textContent"), "No layer matches.")
        # With few layers there is no filter, and with none the card says what to do.
        self.write_with(B)
        self.open_page()
        self.push(B)
        self.settle(200)
        none = self.script("""
            const body = document.getElementById('color-body');
            return {filter: !!body.querySelector('.filter'), chips: [...body.querySelectorAll('.seg-btn')].map(c => c.dataset.coloring),
                    notes: [...body.querySelectorAll('.empty-layers div > div')].map(n => n.textContent)};
            """)
        self.assertFalse(none["filter"])
        self.assertEqual(none["chips"], ["builtin:structure", "plddt"])
        self.assertEqual(none["notes"][0], "Layers such as conservation need an alignment: fold with one loaded in the Viewer.")

    def test_the_layer_filter_stays_the_same_field_and_waits_for_an_ime_to_finish_its_word(self):
        fixtures.write_structure(self.directory, A, SEQUENCE, sidecar=False)
        length = len(SEQUENCE)
        properties = {f"prop_{index}": fixtures.residue_property(
            f"Property {index}", "", [0, 1], [round(((index + step) % 10) / 10, 1) for step in range(length)], None, "")
            for index in range(10)}
        fixtures.write_sidecar(self.directory, A, fixtures.residue_data(A, SEQUENCE, properties=properties))
        self.open_scene(A)
        field = """
            const body = document.getElementById('color-body'), filter = body.querySelector('.filter');
            return {same: filter === window.__filter, focus: document.activeElement === filter, value: filter.value, caret: filter.selectionStart,
                    labels: [...body.querySelectorAll('.seg.scroll .seg-btn')].map(chip => chip.querySelector('.seg-label').textContent).join(', ')};"""
        everything = ", ".join(f"Property {index}" for index in range(10))
        self.script("window.__filter = document.querySelector('#color-body .filter'); window.__filter.focus(); return 1")
        # An IME composing (the pinyin of a word not yet chosen) changes the text, and the filter waits.
        self.script("""
            const filter = window.__filter;
            filter.dispatchEvent(new CompositionEvent('compositionstart', {bubbles: true, data: ''}));
            filter.value = 'property 7';
            filter.dispatchEvent(new InputEvent('input', {bubbles: true, isComposing: true, inputType: 'insertCompositionText', data: '7'}));
            await new Promise(resolve => setTimeout(resolve, 300));
            return 1;""")
        self.assertEqual(self.script(field), {"same": True, "focus": True, "value": "property 7", "caret": 10, "labels": everything})
        # The composition ends: the filter follows, in the same field.
        self.script("""
            window.__filter.dispatchEvent(new CompositionEvent('compositionend', {bubbles: true, data: '7'}));
            await new Promise(resolve => setTimeout(resolve, 300));
            return 1;""")
        self.assertEqual(self.script(field), {"same": True, "focus": True, "value": "property 7", "caret": 10, "labels": "Property 7"})
        # Typing as such (no IME) filters at once, in the same field.
        self.script("""
            const filter = window.__filter;
            filter.value = 'property 1';
            filter.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: '1'}));
            return 1;""")
        self.assertEqual(self.script(field)["labels"], "Property 1")
        # The card drawn again (as any change of the structures draws it) leaves the field as it is: the focus, the text and the caret.
        self.script("window.__filter.setSelectionRange(3, 3); await renderColorCard(true); return 1")
        self.assertEqual(self.script(field), {"same": True, "focus": True, "value": "property 1", "caret": 3, "labels": "Property 1"})
        # And so does another coloring chosen: the card is drawn again with the filter where it was.
        self.choose("prop_1")
        self.assertEqual(self.script(field)["same"], True)
        self.assertEqual(self.script("return card().selected"), "prop_1")
        self.assertEqual(self.console_errors(), [])

    def test_the_chips_are_native_radios_each_group_one_tab_stop_with_the_arrow_keys_of_a_radio_group(self):
        self.write_with(A, "information_content", "same_residue_pct")
        self.open_scene(A)
        self.script("document.querySelector('#color-body .more-toggle').click(); return 1")
        self.settle(200)
        group = self.script("""
            const body = document.getElementById('color-body');
            const radios = [...body.querySelectorAll('input[type="radio"]')];
            const names = {};
            for (const radio of radios) (names[radio.name] ||= []).push(radio.closest('.seg-btn').dataset.coloring);
            return {groups: [...body.querySelectorAll('[role="radiogroup"]')].map(g => g.getAttribute('aria-label')), names,
                    checked: radios.filter(r => r.checked).map(r => r.value),
                    roles: body.querySelectorAll('[role="radio"], .seg-btn[tabindex]').length,
                    labelled: radios.every(r => r.labels.length === 1 && r.labels[0].textContent.trim().length > 0)};
            """)
        self.assertEqual(group, {
            "groups": ["Color by", "More colorings"],
            "names": {"coloring": ["builtin:structure", "plddt", "information_content", "same_residue_pct"],
                      "coloring-more": ["builtin:secondary-structure", "builtin:residue-position"]},
            "checked": ["plddt"], "roles": 0, "labelled": True})
        # The browser's own keys: Right and Left move along the group (round its ends) and choose, as radio buttons do.
        widget = self.view.focusProxy() or self.view
        self.script("document.querySelector('#color-body [data-coloring=\"plddt\"] input').focus(); return 1")
        for key, expected in ((Qt.Key_Right, "information_content"), (Qt.Key_Right, "same_residue_pct"), (Qt.Key_Right, "builtin:structure"),
                              (Qt.Key_Left, "same_residue_pct")):
            QTest.keyClick(widget, key)
            self.pump(0.3)
            self.script("await structureQueue; await cardNow(); return 1")
            self.settle(150)
            state = self.script("""return {selected: card().selected, focus: document.activeElement.closest('.seg-btn')?.dataset.coloring,
                                           coloring: coloringOf(loadedStructures()[0])}""")
            self.assertEqual(state, {"selected": expected, "focus": expected, "coloring": expected}, key)

    def test_the_compact_legend_on_the_canvas_follows_the_choice(self):
        self.write_with(A, "information_content")
        self.open_scene(A)
        self.assertEqual(self.script("return card().compact"), {"shown": "1", "title": "pLDDT", "ends": ["0", "100"]})
        layer = self.choose("information_content")
        self.assertEqual(layer["compact"], {"shown": "1", "title": "Information Content (%)", "ends": ["0 %", "100 %"]})
        self.assertEqual(self.choose("builtin:structure")["compact"]["shown"], "0")


class PanelTabTests(ColorTestCase):
    def test_on_a_phone_the_cards_take_turns_in_the_strip_and_elsewhere_all_are_in_view(self):
        self.write_with(A, "information_content")
        self.open_scene(A)
        visible = """return ['structures', 'color', 'residue'].filter(name => getComputedStyle(document.querySelector('.sec[data-tab="' + name + '"]')).display !== 'none')"""
        self.assertEqual(self.script(visible), ["structures", "color", "residue"])
        self.assertEqual(self.script("return getComputedStyle(document.querySelector('.side-tabs')).display"), "none")
        self.view.resize(600, 800)
        try:
            self.open_scene(A)
            self.assertEqual(self.script("return getComputedStyle(document.querySelector('.side-tabs')).display"), "flex")
            self.assertEqual(self.script(visible), ["structures"])
            self.script("document.getElementById('tab-color').click(); return 1")
            self.assertEqual(self.script(visible), ["color"])
            tabs = self.script("""
                return [...document.querySelectorAll('.side-tabs [role="tab"]')].map(t => [t.dataset.tab, t.getAttribute('aria-selected'), t.tabIndex]);
                """)
            self.assertEqual(tabs, [["structures", "false", -1], ["color", "true", 0], ["residue", "false", -1]])
            self.script("""
                document.getElementById('tab-color').dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true, cancelable: true}));
                return 1;
                """)
            self.assertEqual(self.script(visible), ["residue"])
            self.assertEqual(self.script("return document.activeElement.id"), "tab-residue")
            # Past the last one it comes round to the first.
            self.script("""
                document.getElementById('tab-residue').dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true, cancelable: true}));
                return 1;
                """)
            self.assertEqual(self.script(visible), ["structures"])
            # Home and End go to the first and the last.
            for key, expected in (("End", "tab-residue"), ("Home", "tab-structures")):
                self.script("""
                    document.activeElement.dispatchEvent(new KeyboardEvent('keydown', {key: %s, bubbles: true, cancelable: true}));
                    return 1;
                    """ % json.dumps(key))
                self.assertEqual(self.script("return document.activeElement.id"), expected, key)
        finally:
            self.view.resize(1300, 800)

    def test_the_strip_is_a_focusgroup_and_its_cards_are_tab_panels_named_by_their_tabs(self):
        self.write_with(A, "information_content")
        self.open_scene(A)
        markup = self.script("""
            const list = document.querySelector('.side-tabs');
            return {group: list.getAttribute('focusgroup'), start: [...list.querySelectorAll('[focusgroupstart]')].map(t => t.id),
                    panels: [...document.querySelectorAll('.sec[data-tab]')].map(p => [p.getAttribute('role'), p.getAttribute('aria-labelledby'),
                        document.getElementById(p.getAttribute('aria-labelledby')).dataset.tab === p.dataset.tab])};
            """)
        self.assertEqual(markup, {"group": "tablist nomemory", "start": ["tab-structures"],
                                  "panels": [["tabpanel", "tab-structures", True], ["tabpanel", "tab-color", True], ["tabpanel", "tab-residue", True]]})
        self.script("document.getElementById('tab-color').click(); return 1")
        self.assertEqual(self.script("return [...document.querySelectorAll('.side-tabs [focusgroupstart]')].map(t => t.id)"), ["tab-color"])
        # The script's roving tabindex is the fallback only: where the browser has focusgroup it leaves tabindex alone.
        native = self.script("return 'focusGroup' in HTMLElement.prototype")
        stops = self.script("return [...document.querySelectorAll('.side-tabs [role=\"tab\"], .view-toolbar button')].filter(b => b.hasAttribute('tabindex')).length")
        self.assertEqual(stops == 0, native)


if __name__ == "__main__":
    unittest.main()
