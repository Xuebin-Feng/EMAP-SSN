# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What Fold View's page tests share (not collected by unittest).

The Fold View page tests are split across sections (fold_color, fold_struct,
fold_canvas, fold_session, fold_layout, fold_text, fold_panel), each opening
esmfold.html in Qt WebEngine over tests/esmfold_page_harness.py. This module
holds the shared fixture constants, JavaScript probes and base TestCase classes
(none with ``test_*`` methods) so no test module imports another test module.

Import with ``from tests.esmfold_page_helpers import ...``.
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

from tests import esmfold_layer_fixtures as fixtures  # noqa: E402
from tests.esmfold_long_names import USER_MSA, write_long  # noqa: E402
from tests.esmfold_page_harness import FoldViewPageMixin, FoldViewServerMixin  # noqa: E402

from PySide6.QtCore import QUrl  # noqa: E402  (after the harness has set the engine's environment)

GREY = 0xDCDCDC
A, B, C = "11.pdb", "22.pdb", "33.pdb"
NAMES = (A, B, C)
# The "Properties" group of the Colors tab, as a card lists it: the chips' ids and texts, in order.
PROPERTY_CHIPS = [["builtin:secondary-structure", "Secondary structure", ""], ["builtin:accessible-surface-area", "Solvent accessibility", ""],
                  ["builtin:hydrophobicity", "Hydrophobicity", ""], ["builtin:residue-charge", "Charge", ""]]
PROPERTY_IDS = [chip[0] for chip in PROPERTY_CHIPS]
# "More colorings", which is open at the start: its chips come last in a card.
MORE_CHIPS = [["builtin:structure", "Structure color", ""], ["builtin:residue-position", "Residue position", ""],
              ["builtin:element-symbol", "Atoms", ""]]
MORE_IDS = [chip[0] for chip in MORE_CHIPS]
SEQUENCE = fixtures.FIRST_SEQUENCE


def hydrophobicity(length):
    return [round(((index * 3) % 11) - 5.0, 1) for index in range(length)]


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
            header: [...body.querySelectorAll('th')].map(c => c.textContent),
            rows: [...body.querySelectorAll('tr')].slice(1).map(row => [...row.children].map(c => c.textContent)),
            buttons: [...document.querySelectorAll('#residue-zoom, #residue-clear')].map(b => [b.id, b.getAttribute('aria-label'), b.getAttribute('aria-disabled')]),
            hint: (body.querySelector('.hint') || {}).textContent ?? null};
};
"""


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
    // A part that is not drawn (a state the engine is not in, a closed dialog) has nothing out of its place: no answer to give.
    if (getComputedStyle(root).display === 'none' || (limit.width === 0 && limit.height === 0)) return ['not drawn: ' + selector];
    const label = node => node.tagName.toLowerCase() + (node.id ? '#' + node.id : '') + [...node.classList].map(name => '.' + name).join('')
        + (node.children.length ? '' : ' "' + node.textContent.trim().slice(0, 24) + '"');
    const drawn = node => {
        const style = getComputedStyle(node), box = node.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden' && (box.width > 0 || box.height > 0);
    };
    const clips = node => getComputedStyle(node).overflowX !== 'visible';
    // Cut with an ellipsis: an element that clips, says so, and has more text than room.
    const cuts = node => clips(node) && getComputedStyle(node).textOverflow === 'ellipsis' && node.scrollWidth > node.clientWidth + 1;
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
        """Click the card's button for `coloring`, let the scene follow, and return the card.

        The colorings under "More colorings" (Structure color, Residue position) are in a group that is open at the start
        and that a test may have closed: it is opened first when their chip is not in the card.
        """
        return self.script("""
            if (!document.querySelector('#color-body [data-coloring="%s"]')) {
                document.querySelector('#color-body .more-toggle').click();
                await cardNow();
            }
            document.querySelector('#color-body [data-coloring="%s"]').click();
            await structureQueue;
            return await cardNow();
            """ % (coloring, coloring))

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


class ResidueTestCase(ColorTestCase):
    def open_scene(self, *names):
        super().open_scene(*names)
        self.js(POINTER)

    def plddt(self, group, offset=0):
        return f"{fixtures.plddt_values(33, offset)[group]:.1f}"


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
        self.script("setPanel(true); setTab(%s); await new Promise(resolve => setTimeout(resolve, 320)); return 1" % json.dumps(tab))

    def overflowing(self, selector):
        return self.script("return overflowing(%s)" % json.dumps(selector))

    def reset_page(self):
        self.view.setUrl(QUrl("about:blank"))
        self.pump(0.3)
