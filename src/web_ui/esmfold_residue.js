/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's hover tooltip, the pinned residue and the Residue card.
 */

// What Mol* says a pointer is over (plugin.behaviors.interaction.hover) is
// read for the residue under it: its name and number, its pLDDT (the
// B-factor), its secondary structure, and the value the structure's layer
// gives it (the annotation rows the Viewer serves, the same ones Mol*
// colors from). A click pins a residue (Mol*'s selection marks it), and the
// Residue card shows it for every structure at once, each structure's
// equivalent residue found by the same sequence alignment the superposition
// uses. Nothing here keeps a copy of Mol*'s state: the residue and the
// structures are asked again whenever they are drawn. Loaded after
// esmfold_color.js; setupResidueEvents() runs once Mol* is up.

let hoveredResidue = null;
let pinnedResidue = null;
let residueRenderId = 0;
let tipRenderId = 0;
let pointerAt = { x: 0, y: 0 };
// The Viewer's annotation rows by address, as promises of maps from "chain|number" to row.
const annotationRows = new Map();
const nativeTraces = new Map();
const alignedPairs = new Map();

// ---- The residue under a pointer -----------------------------------------------------
function residueOf(loci) {
    const { StructureElement, StructureProperties } = molstar.lib.structure;
    if (!loci || !StructureElement.Loci.is(loci) || StructureElement.Loci.isEmpty(loci)) return null;
    const structure = loadedStructures().find(candidate => candidate.model?.cell.obj.data === loci.structure.model);
    if (!structure) return null;
    const location = StructureElement.Loci.getFirstLocation(loci);
    return {
        ref: structure.cell.transform.ref,
        chain: StructureProperties.chain.auth_asym_id(location),
        number: StructureProperties.residue.auth_seq_id(location),
        name: StructureProperties.atom.label_comp_id(location),
        plddt: StructureProperties.atom.B_iso_or_equiv(location),
        secondary: secondaryName(StructureProperties.residue.secondary_structure_type(location))
    };
}

function secondaryName(flags) {
    const type = molstar.lib.structure.Types.SecondaryStructureType;
    if (!flags) return null;
    if (type.is(flags, type.Flag.Helix)) return t("Helix");
    if (type.is(flags, type.Flag.Beta)) return t("Strand");
    if (type.is(flags, type.Flag.Turn)) return t("Turn");
    return null;
}

function residueTitle(residue) {
    return residue.name.charAt(0) + residue.name.slice(1).toLowerCase() + ' ' + residue.number;
}

function sameResidue(a, b) {
    return !!a && !!b && a.ref === b.ref && a.chain === b.chain && a.number === b.number;
}

// ---- What the Viewer knows of a residue ---------------------------------------------------
// The annotation rows of a structure's layer, by residue.
function annotationFor(structure, layerId) {
    const entry = resolvedListings.get(structureFile(structure))?.layers.find(layer => layer.id === layerId && layer.annotation);
    if (!entry) return Promise.resolve(null);
    if (!annotationRows.has(entry.annotation)) {
        annotationRows.set(entry.annotation, fetch(entry.annotation)
            .then(response => (response.ok ? response.json() : []))
            .then(rows => new Map(rows.map(row => [row.auth_asym_id + '|' + row.auth_seq_id, row])))
            .catch(() => new Map()));
    }
    return annotationRows.get(entry.annotation);
}

// The layer a structure is colored by, when it is one of the Viewer's (not pLDDT, a built-in, grey or Advanced's).
function layerOfStructure(structure) {
    const id = coloringOf(structure);
    return id === PLDDT_LAYER || id === NO_DATA || id === OTHER_COLORING || id.startsWith(BUILTIN_PREFIX) ? null : id;
}

function layerEntry(structure, layerId) {
    return resolvedListings.get(structureFile(structure))?.layers.find(layer => layer.id === layerId);
}

function roundedValue(value) {
    return String(Number(value.toFixed(2)));
}

// pLDDT's band as a short name ("Very high"), from the Viewer's own words for it.
function plddtBandName(value, structure) {
    const bands = plddtBands(resolvedListings.get(structureFile(structure))?.layers.find(layer => layer.id === PLDDT_LAYER));
    const band = bands[value > 90 ? 0 : value > 70 ? 1 : value > 50 ? 2 : 3];
    return { color: band.color, name: band.label.replace(/\s*\(.*\)\s*$/, '') };
}

// The CA atoms of a structure as they were folded: codes, numbers and pLDDTs.
function nativeTrace(structure) {
    const ref = structure.cell.transform.ref;
    if (!nativeTraces.has(ref)) nativeTraces.set(ref, readCaTrace(structure.cell.obj.data));
    return nativeTraces.get(ref);
}

// Which residue of `other` is the residue `index` of `structure`, by the sequence alignment (null where it has none).
function equivalentIndex(structure, other, index) {
    const key = structure.cell.transform.ref + '|' + other.cell.transform.ref;
    if (!alignedPairs.has(key)) {
        alignedPairs.set(key, new Map(alignSequences(nativeTrace(structure).codes, nativeTrace(other).codes)));
    }
    return alignedPairs.get(key).get(index) ?? null;
}

function displayedCoordinates(structure) {
    return readCaTrace((structure.transform || structure).cell.obj.data).coords;
}

// ---- The tooltip ----------------------------------------------------------------------------
function placeTip() {
    const tip = byId('tip');
    const stage = byId('stage').getBoundingClientRect();
    const width = tip.offsetWidth, height = tip.offsetHeight;
    let x = pointerAt.x + 16, y = pointerAt.y + 16;
    if (x + width > stage.width - 8) x = pointerAt.x - width - 12;
    if (y + height > stage.height - 8) y = pointerAt.y - height - 12;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
}

function hideTip() {
    tipRenderId++;
    byId('tip').classList.remove('show');
}

function tipRow(label, value, color) {
    const row = make('tr');
    const name = make('td', undefined, label);
    const cell = make('td');
    if (color) {
        const dot = make('span', 'swatch-dot');
        dot.style.background = color;
        cell.append(dot);
    }
    cell.append(value);
    row.append(name, cell);
    return row;
}

async function showTip(residue) {
    const renderId = ++tipRenderId;
    const structure = structureByRef(residue.ref);
    if (!structure) return hideTip();
    const layerId = layerOfStructure(structure);
    const rows = layerId ? await annotationFor(structure, layerId) : null;
    if (renderId !== tipRenderId || !hoveredResidue || !sameResidue(hoveredResidue, residue)) return;
    const tip = byId('tip');
    const head = make('div', 'h');
    const title = make('b', undefined, residueTitle(residue));
    const where = make('span', undefined, t("chain {chain} · {name}", {chain: residue.chain, name: structureName(structure)}));
    head.append(title, where);
    const table = make('table');
    // A structure opened from a file has B-factors, which are not known to be a confidence: no band for it.
    const band = plddtBandName(residue.plddt, structure);
    const confidence = isFolded(structure)
        ? tipRow(coloringLabel(PLDDT_LAYER), residue.plddt.toFixed(1) + ' · ' + band.name, band.color)
        : tipRow(t("B-factor"), residue.plddt.toFixed(1));
    table.append(confidence);
    if (layerId) {
        const row = rows?.get(residue.chain + '|' + residue.number);
        const entry = layerEntry(structure, layerId);
        table.append(tipRow(entry?.label ?? layerId, row ? roundedValue(row.value) + (entry?.unit ? ' ' + entry.unit : '') : t("No data")));
    }
    if (residue.secondary) table.append(tipRow(t("Secondary structure"), residue.secondary));
    const parts = [head, table];
    if (sameResidue(pinnedResidue, residue)) {
        const note = make('div', 'pinned-note', t("Pinned. Escape clears it."));
        parts.push(note);
    }
    tip.replaceChildren(...parts);
    tip.classList.add('show');
    placeTip();
}

// ---- Pinning a residue -------------------------------------------------------------------------
function residueLoci(residue) {
    const structure = structureByRef(residue.ref);
    const { Queries, QueryContext, StructureProperties, StructureSelection } = molstar.lib.structure;
    const data = (structure.transform || structure).cell.obj.data;
    const query = Queries.generators.atoms({
        residueTest: context => StructureProperties.residue.auth_seq_id(context.element) === residue.number
            && StructureProperties.chain.auth_asym_id(context.element) === residue.chain
    });
    return StructureSelection.toLociWithSourceUnits(query(new QueryContext(data)));
}

function pinResidue(residue, loci) {
    pinnedResidue = residue;
    // The strip shows the chain it is on.
    sequenceTarget = { ref: residue.ref, chain: residue.chain };
    scheduleSequence();
    try {
        plugin.managers.structure.selection.fromLoci('set', molstar.lib.structure.StructureElement.Loci.extendToWholeResidues(loci));
    } catch (err) {
        console.warn("Could not mark the pinned residue:", err);
    }
    renderResidueCard();
    if (hoveredResidue) showTip(hoveredResidue);
}

// Mol*'s own click puts the clicked residue in focus, and draws its surroundings (sticks, contacts) with
// nodes it adds to the state tree. Clearing the focus empties them but leaves them there, to be saved with
// the session and drawn again from it, so a pin lets go of them for good: the focus, then its nodes.
const FOCUS_TAG = 'structure-focus-';

function focusNodes() {
    return [...plugin.state.data.cells.values()].filter(cell => cell.transform.tags?.some(tag => tag.startsWith(FOCUS_TAG)));
}

async function removeFocusNodes() {
    // The selections own the representations: deleting the roots takes the whole of it.
    const roots = focusNodes().filter(cell => cell.transform.tags.some(tag => tag.endsWith('-sel')));
    if (!roots.length) return;
    const state = plugin.state.data;
    const update = state.build();
    for (const cell of roots) update.delete(cell.transform.ref);
    await plugin.runTask(state.updateTree(update));
}

// Lets go of the focus Mol*'s click put in, and of the focus' own nodes (not of the mark on a residue).
function releaseFocus() {
    plugin.managers.structure.focus.clear();
    return runExclusive(async () => {
        // Mol* empties the nodes in its own time; deleting them after that leaves nothing for it to redraw.
        await new Promise(resolve => setTimeout(resolve, 150));
        await removeFocusNodes();
    }).catch(err => console.warn("Could not remove the focus:", err));
}

// The session as Mol* gives it, without the focus (a residue pinned when it is taken is not pinned when it comes back).
function withoutFocus(snapshot) {
    const tree = snapshot.data?.tree;
    if (!tree) return snapshot;
    const transforms = tree.transforms.filter(node => !(node.tags || []).some(tag => String(tag).startsWith(FOCUS_TAG)));
    return { ...snapshot, structureFocus: {}, data: { ...snapshot.data, tree: { ...tree, transforms } } };
}

// Lets go of the pinned residue; says whether there was one.
function clearPin() {
    if (!pinnedResidue) return false;
    pinnedResidue = null;
    plugin.managers.structure.selection.clear();
    releaseFocus();
    renderResidueCard();
    if (hoveredResidue) showTip(hoveredResidue);
    return true;
}

// After a session came back, or Undo: nothing is hovered or pinned, whatever the session had in focus.
function resetPointerState() {
    hoveredResidue = null;
    pinnedResidue = null;
    hideTip();
    renderResidueCard();
    plugin.managers.structure.selection.clear();
    return releaseFocus();
}

function zoomToResidue(residue) {
    plugin.managers.camera.focusLoci(residueLoci(residue));
}

// ---- The Residue card -------------------------------------------------------------------------------
// One row for each structure: the residue at the same place of the alignment, its pLDDT and layer value.
async function residueRows(residue) {
    const source = structureByRef(residue.ref);
    const a = nativeTrace(source);
    const index = a.ids.findIndex(([chain, number]) => chain === residue.chain && number === residue.number);
    const layerId = colorMode === PLDDT_LAYER || colorMode.startsWith(BUILTIN_PREFIX) ? null : colorMode;
    const sourceCoordinates = index < 0 ? null : displayedCoordinates(source)[index];
    const rows = [];
    for (const structure of loadedStructures()) {
        const same = structure.cell.transform.ref === residue.ref;
        const b = nativeTrace(structure);
        const at = same ? index : (index < 0 ? null : equivalentIndex(source, structure, index));
        const item = { ref: structure.cell.transform.ref, name: structureName(structure), layer: null, plddt: null, distance: null, found: same || (at !== null && at >= 0) };
        if (item.found) {
            const [chain, number] = same ? [residue.chain, residue.number] : b.ids[at];
            item.plddt = same ? residue.plddt : b.plddt[at];
            if (layerId) {
                const found = (await annotationFor(structure, layerId))?.get(chain + '|' + number);
                const unit = layerEntry(structure, layerId)?.unit;
                item.layer = found ? roundedValue(found.value) + (unit ? ' ' + unit : '') : null;
            }
            if (!same && sourceCoordinates) {
                const other = displayedCoordinates(structure)[at];
                item.distance = Math.hypot(...other.map((value, d) => value - sourceCoordinates[d]));
            }
        }
        rows.push(item);
    }
    const layer = layerId ? loadedStructures().map(structure => layerEntry(structure, layerId)).find(Boolean) : null;
    return { rows, layerLabel: layer?.label ?? null };
}

function cell(tag, text, className) {
    return make(tag, className, text);
}

async function renderResidueCard() {
    if (!plugin) return;
    const renderId = ++residueRenderId;
    const body = byId('residue-body');
    const residue = pinnedResidue || hoveredResidue;
    if (!residue || !structureByRef(residue.ref)) {
        if (pinnedResidue && !structureByRef(pinnedResidue.ref)) pinnedResidue = null;
        const hint = make('div', 'hint', t("Hover a residue to see its values here. Click it to pin it; Escape clears the pin."));
        body.replaceChildren(hint);
        refreshSequenceMarks();
        refreshTools();
        return;
    }
    const { rows, layerLabel } = await residueRows(residue);
    if (renderId !== residueRenderId) return;
    const head = make('div', 'res-head');
    const title = make('div', undefined, residueTitle(residue));
    const where = make('span', undefined, t("chain {chain} · {name}", {chain: residue.chain, name: structureName(structureByRef(residue.ref))}));
    title.append(' ', where);
    head.append(title);
    if (residue.secondary) {
        const secondary = make('span', undefined, residue.secondary);
        head.append(secondary);
    }
    const table = make('table', 'res-table');
    const header = make('tr');
    // A layer's name may take two lines in its header (it is the data's, and as long as the data says); every value is one line.
    const columns = [[t("Structure"), undefined], ['pLDDT', undefined]];
    if (layerLabel) columns.push([layerLabel, 'res-layer']);
    if (rows.length > 1) columns.push([t("Cα–Cα (Å)"), 'res-dist']);
    header.append(...columns.map(([label, className]) => {
        const node = cell('th', label, className);
        // In the strip a layer's name is on one line, cut with an ellipsis if it is long: the whole of it is on hover.
        if (className === 'res-layer') node.title = label;
        return node;
    }));
    table.append(header);
    for (const item of rows) {
        const row = make('tr');
        // The name stays on one line beside its swatch; one too long for the room is cut with an ellipsis (esmfold.html),
        // and the whole name is on hover (the text itself is whole in the page, for screen readers).
        const name = make('td', 'res-name');
        const swatch = make('span', 'res-swatch');
        swatch.style.background = structureColor(item.ref);
        name.title = item.name;
        name.append(swatch, item.name);
        const values = [item.found ? item.plddt.toFixed(1) : '–'];
        if (layerLabel) values.push(!item.found ? '–' : item.layer ?? t("No data"));
        if (rows.length > 1) values.push(item.distance === null ? '–' : item.distance.toFixed(1));
        row.append(name, ...values.map(value => cell('td', value)));
        table.append(row);
    }
    const parts = [head, table];
    const note = make('div', 'res-note', t("Each structure shows the residue at the same place of the alignment."));
    parts.push(note);
    const opened = loadedStructures().filter(structure => !isFolded(structure)).map(structure => structureName(structure));
    if (opened.length) {
        const caution = make('div', 'res-note', t("{names}: the number is the file's B-factor, which is not confidence in an experimental structure.", {names: opened.join(', ')}));
        parts.push(caution);
    }
    const foot = make('div', 'res-foot');
    const zoom = make('button', 'btn btn-sm', t("Zoom to"));
    zoom.type = 'button';
    zoom.addEventListener('click', () => zoomToResidue(residue));
    foot.append(zoom);
    if (pinnedResidue) {
        const clear = make('button', 'btn btn-sm', t("Clear"));
        clear.type = 'button';
        clear.addEventListener('click', clearPin);
        foot.append(clear);
    }
    parts.push(foot);
    body.replaceChildren(...parts);
    refreshSequenceMarks();
    refreshTools();
}

// What a change of the structures does to a residue the page holds: it goes if its structure went.
function refreshResidue() {
    if (hoveredResidue && !structureByRef(hoveredResidue.ref)) {
        hoveredResidue = null;
        hideTip();
    }
    if (pinnedResidue && !structureByRef(pinnedResidue.ref)) pinnedResidue = null;
    for (const ref of [...nativeTraces.keys()]) {
        if (!structureByRef(ref)) nativeTraces.delete(ref);
    }
    for (const key of [...alignedPairs.keys()]) {
        if (!key.split('|').every(ref => structureByRef(ref))) alignedPairs.delete(key);
    }
    renderResidueCard();
}

// ---- Mol*'s events --------------------------------------------------------------------------------
function setupResidueEvents() {
    plugin.behaviors.interaction.hover.subscribe(event => {
        hoveredResidue = residueOf(event.current.loci);
        if (hoveredResidue) showTip(hoveredResidue);
        else hideTip();
        if (!pinnedResidue) renderResidueCard();
    });
    plugin.behaviors.interaction.click.subscribe(event => {
        if (event.button !== 1) return;
        const residue = residueOf(event.current.loci);
        if (toolMode) {
            if (residue) toolClick(residue, event.current.loci);
        } else if (residue) {
            pinResidue(residue, event.current.loci);
        } else {
            clearPin();
        }
    });
    byId('viewer').addEventListener('pointermove', event => {
        const stage = byId('stage').getBoundingClientRect();
        pointerAt = { x: event.clientX - stage.left, y: event.clientY - stage.top };
        if (byId('tip').classList.contains('show')) placeTip();
    });
    byId('viewer').addEventListener('pointerleave', () => {
        hoveredResidue = null;
        hideTip();
        if (!pinnedResidue) renderResidueCard();
    });
    renderResidueCard();
}
