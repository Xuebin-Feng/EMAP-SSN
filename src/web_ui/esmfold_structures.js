/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's Structures card: one row for each structure in Mol*'s hierarchy, and what a row can do.
 */

// The rows are drawn from Mol*'s state and the page's alignment numbers
// (esmfold_superposition.js), and drawn again whenever either changes, so
// what a person does in Mol*'s own panels (Advanced) or Undo shows here as
// it is. A row's buttons act through Mol*'s managers and state, never on a
// copy. Loaded after esmfold_shell.js, whose helpers (byId, make, icon,
// showStatus) and the Viewer's connection it builds on.

const STRUCTURE_COLORS = ['#5b5bd6', '#0f9d8f', '#c2750c', '#c2185b', '#2563eb', '#65a30d'];
// The refs of the rows whose details are open.
const expandedRows = new Set();
const residueCounts = new Map();
const meanPlddts = new Map();
let structuresTimer = null;
// A slider is being dragged: the rows stay as they are until it is let go, so that it keeps its hold.
let rowControlsBusy = false;
let renderRowsAgain = false;

function scheduleStructures() {
    clearTimeout(structuresTimer);
    structuresTimer = setTimeout(renderStructures, 30);
}

// A folded structure is named by its file; an opened one by the label its data was given (the file's name).
function structureName(structure) {
    const file = structureFile(structure);
    if (file) return file.replace(/\.pdb$/i, '');
    const root = plugin.state.data.cells.get(structureDataRoot(structure));
    const label = root?.obj?.label || structure.cell.obj?.label;
    return label ? label.replace(/\.(pdb|ent|cif|mmcif)$/i, '') : t("Structure");
}

function structureHidden(structure) {
    return structure.components.length > 0 && structure.components.every(component => component.cell.state.isHidden);
}

function residueCount(structure) {
    const ref = structure.cell.transform.ref;
    if (!residueCounts.has(ref)) residueCounts.set(ref, readCaTrace(structure.cell.obj.data).codes.length);
    return residueCounts.get(ref);
}

// The mean of the CA atoms' B-factors: a model's pLDDT, which ESMFold writes there.
function meanPlddt(structure) {
    const ref = structure.cell.transform.ref;
    if (!meanPlddts.has(ref)) {
        const { StructureElement, StructureProperties, Unit } = molstar.lib.structure;
        const location = StructureElement.Location.create(structure.cell.obj.data);
        let sum = 0, count = 0;
        for (const unit of structure.cell.obj.data.units) {
            if (!Unit.isAtomic(unit)) continue;
            location.unit = unit;
            for (const element of unit.elements) {
                location.element = element;
                if (StructureProperties.atom.label_atom_id(location) !== 'CA') continue;
                sum += StructureProperties.atom.B_iso_or_equiv(location);
                count++;
            }
        }
        meanPlddts.set(ref, count ? sum / count : NaN);
    }
    return meanPlddts.get(ref);
}

function chip(text, className, title) {
    const node = make('span', 'chip ' + className, text);
    if (title) node.title = title;
    return node;
}

// ---- What a row can change -----------------------------------------------------
function representationOptions() {
    return [
        ['cartoon', t("Cartoon")],
        ['ball-and-stick', t("Ball and stick")],
        ['spacefill', t("Spacefill")],
        ['gaussian-surface', t("Surface")],
        ['backbone', t("Backbone")]
    ];
}

function representationRefs(structure) {
    return structure.components.flatMap(component => component.representations);
}

function representationType(structure) {
    return representationRefs(structure)[0]?.cell.transform.params.type.name;
}

function structureOpacity(structure) {
    const alpha = representationRefs(structure)[0]?.cell.transform.params.type.params.alpha;
    return Math.round(100 * (alpha ?? 1));
}

// Another type of representation: the new one is made with Mol*'s own defaults for the type
// and the colors the old one had (a layer stays), then the old one goes.
async function setRepresentation(ref, type) {
    const structure = structureByRef(ref);
    if (!structure) return;
    const state = plugin.state.data;
    for (const component of structure.components) {
        const old = component.representations.filter(representation => representation.cell.transform.params.type.name !== type);
        if (!old.length) continue;
        const { type: oldType, colorTheme } = old[0].cell.transform.params;
        await plugin.builders.structure.representation.addRepresentation(component.cell, {
            type, typeParams: { alpha: oldType.params.alpha }, color: colorTheme.name, colorParams: colorTheme.params
        });
        const removal = state.build();
        for (const representation of old) removal.delete(representation.cell.transform.ref);
        await plugin.runTask(state.updateTree(removal));
    }
    scheduleStructures();
    setTimeout(saveSession, 1500);
}

async function setOpacity(ref, percent) {
    const structure = structureByRef(ref);
    if (!structure) return;
    const state = plugin.state.data;
    const update = state.build();
    for (const representation of representationRefs(structure)) {
        update.to(representation.cell).update(old => ({
            ...old, type: { ...old.type, params: { ...old.type.params, alpha: percent / 100 } }
        }));
    }
    await plugin.runTask(state.updateTree(update));
    setTimeout(saveSession, 1500);
}

// A slider sends many values: one update runs at a time, and the last value wins.
const opacityRequests = new Map();
function applyOpacity(ref, percent) {
    const running = opacityRequests.has(ref);
    opacityRequests.set(ref, percent);
    if (running) return;
    runExclusive(async () => {
        while (opacityRequests.has(ref)) {
            const wanted = opacityRequests.get(ref);
            await setOpacity(ref, wanted);
            if (opacityRequests.get(ref) === wanted) opacityRequests.delete(ref);
        }
    }).catch(err => {
        opacityRequests.delete(ref);
        console.warn("Could not change the opacity:", err);
    });
}

// The pivot's pose goes back to where it was folded and every other structure is fitted to it.
async function makePivot(ref) {
    const target = structureByRef(ref);
    if (!target || pivotStructure()?.cell.transform.ref === ref) return;
    const name = structureName(target);
    if (target.transform) await applyStructureTransform(target, molstar.lib.math.LinearAlgebra.Mat4.identity());
    chosenPivotRef = ref;
    const failed = [];
    for (const other of loadedStructures()) {
        const otherRef = other.cell.transform.ref;
        if (otherRef === ref) continue;
        try {
            recordFit(otherRef, ref, await superposeOnto(structureByRef(ref), structureByRef(otherRef)));
        } catch (err) {
            console.warn("Structural alignment failed:", err);
            fitStats.delete(otherRef);
            failed.push(structureName(other));
        }
    }
    fitStats.delete(ref);
    scheduleStructures();
    setTimeout(saveSession, 1500);
    if (failed.length) showStatus(t("{name} is the pivot now, but {others} could not be aligned to it.", {name, others: failed.join(', ')}), 6000, {error: true});
    else showStatus(t("{name} is the pivot now. The other structures are aligned to it.", {name}));
}

async function realign(ref) {
    const pivot = pivotStructure();
    const target = structureByRef(ref);
    if (!pivot || !target || pivot.cell.transform.ref === ref) return;
    const fit = await superposeOnto(pivot, target);
    recordFit(ref, pivot.cell.transform.ref, fit);
    scheduleStructures();
    setTimeout(saveSession, 1500);
    showStatus(t("Aligned {name} again: RMSD {rmsd} Å.", {name: structureName(target), rmsd: fit.rmsd.toFixed(1)}));
}

function reportFailure(message, err) {
    console.warn(message, err);
    showStatus(t("Could not do that: {reason}", {reason: err.message}), 6000, {error: true});
}

// A structure the Viewer folded is loaded from its own folder; any other (a file opened or dropped here, or one
// Advanced loaded) is not, and its B-factors need not be a confidence. The tree says which, so a session does too.
function isFolded(structure) {
    return !!structureSourceUrl(structure)?.startsWith('/structures/');
}

// The URL a structure was loaded from, so that its file can be had again as the Viewer wrote it.
function structureSourceUrl(structure) {
    const cells = plugin.state.data.cells;
    for (let cell = structure.cell; cell; ) {
        const params = cell.transform.params;
        if (cell.transform.transformer.id === 'ms-plugin.download' && params?.url?.url) return params.url.url;
        const parent = cell.transform.parent;
        cell = parent && parent !== cell.transform.ref ? cells.get(parent) : undefined;
    }
    return undefined;
}

function downloadStructureFile(url, file) {
    const link = make('a');
    link.href = url;
    link.download = file;
    document.body.append(link);
    link.click();
    link.remove();
}

// ---- Rows ------------------------------------------------------------------------
function safeId(ref) {
    return ref.replace(/[^A-Za-z0-9_-]/g, '');
}

function focusable(node, key) {
    node.dataset.focusKey = key;
    return node;
}

function detailLine(label, value) {
    const term = make('dt', undefined, label);
    const description = make('dd', undefined, value);
    return [term, description];
}

function structureDetails(structure, pivot) {
    const ref = structure.cell.transform.ref;
    const id = safeId(ref);
    const isPivot = ref === pivot.cell.transform.ref;
    const body = make('div', 'srow-body');
    body.id = 'sb-' + id;

    // Representation
    const kind = make('div');
    const kindLabel = make('label', 'field-label', t("Representation"));
    kindLabel.htmlFor = 'rep-' + id;
    const current = representationType(structure);
    const kindSelect = focusable(make('select', 'select'), ref + ':representation');
    kindSelect.id = 'rep-' + id;
    const options = representationOptions().map(([type, label]) => {
        const option = make('option', undefined, label);
        option.value = type;
        return option;
    });
    if (!options.some(option => option.value === current)) {
        const other = make('option', undefined, t("Other (set in Advanced)"));
        other.value = '';
        other.disabled = true;
        options.unshift(other);
    }
    kindSelect.append(...options);
    kindSelect.value = options.some(option => option.value === current) ? current : '';
    kindSelect.addEventListener('change', () => {
        if (!kindSelect.value) return;
        runExclusive(() => setRepresentation(ref, kindSelect.value)).catch(err => reportFailure("Could not change the representation:", err));
    });
    kind.append(kindLabel, kindSelect);

    // Color: the scene's, or one of its own
    const coloring = make('div');
    const coloringLabelNode = make('label', 'field-label', t("Color"));
    coloringLabelNode.htmlFor = 'col-' + id;
    const coloringSelect = focusable(make('select', 'select'), ref + ':coloring');
    coloringSelect.id = 'col-' + id;
    const scene = make('option', undefined, t("Same as the scene"));
    scene.value = '';
    const choices = structureColoringOptions(structure).map(([value, label]) => {
        const option = make('option', undefined, label);
        option.value = value;
        return option;
    });
    const shown = coloringOf(structure);
    const own = shown === colorMode || shown === NO_DATA ? '' : shown;
    if (own && !choices.some(option => option.value === own)) {
        const other = make('option', undefined, coloringLabel(own));
        other.value = own;
        other.disabled = true;
        choices.unshift(other);
    }
    coloringSelect.append(scene, ...choices);
    coloringSelect.value = own;
    coloringSelect.addEventListener('change', () => setStructureColoring(ref, coloringSelect.value));
    coloring.append(coloringLabelNode, coloringSelect);

    // Opacity
    const opacity = make('div');
    const opacityLabel = make('label', 'field-label', t("Opacity"));
    opacityLabel.htmlFor = 'op-' + id;
    const percent = structureOpacity(structure);
    const opacityValue = make('span', 'field-value', percent + '%');
    opacityLabel.append(' ', opacityValue);
    const slider = focusable(make('input'), ref + ':opacity');
    slider.type = 'range';
    slider.id = 'op-' + id;
    slider.min = '10';
    slider.max = '100';
    slider.step = '5';
    slider.value = String(percent);
    slider.addEventListener('input', () => {
        opacityValue.textContent = slider.value + '%';
        applyOpacity(ref, Number(slider.value));
    });
    opacity.append(opacityLabel, slider);
    body.append(kind, coloring, opacity);

    // What is known about the structure and its alignment
    const facts = make('dl', 'kv');
    if (isFolded(structure)) {
        const plddt = meanPlddt(structure);
        if (Number.isFinite(plddt)) facts.append(...detailLine(t("Mean pLDDT"), plddt.toFixed(1)));
    }
    const fit = fitStats.get(ref);
    if (!isPivot && fit) {
        facts.append(
            ...detailLine(t("Aligned to {name}", {name: structureName(pivot)}), t("RMSD {rmsd} Å", {rmsd: fit.rmsd.toFixed(1)})),
            ...detailLine(t("Cα pairs"), t("{used} of {aligned}", {used: fit.used, aligned: fit.aligned})),
            ...detailLine(t("Sequence identity"), Math.round(100 * fit.identity) + '%')
        );
    }
    if (facts.children.length) body.append(facts);
    if (isPivot) {
        const note = make('div', 'hint', t("This is the pivot: the others are aligned to it."));
        body.append(note);
    } else if (!fit && !structure.transform) {
        const note = make('div', 'hint', t("Not aligned to the pivot."));
        body.append(note);
    }

    // Actions
    const actions = make('div', 'row-actions');
    if (!isPivot) {
        const pivotButton = focusable(make('button', 'btn btn-sm', t("Make pivot")), ref + ':pivot');
        pivotButton.type = 'button';
        pivotButton.title = t("Move this structure back to where it was folded and align the others to it.");
        pivotButton.addEventListener('click', () => runExclusive(() => makePivot(ref)).catch(err => reportFailure("Could not make the pivot:", err)));
        const again = focusable(make('button', 'btn btn-sm'), ref + ':realign');
        again.type = 'button';
        again.title = t("Align this structure to the pivot again.");
        const againLabel = make('span', undefined, t("Re-align"));
        const againIcon = icon("rotate-ccw");
        again.append(againIcon, againLabel);
        again.addEventListener('click', () => runExclusive(() => realign(ref)).catch(err => reportFailure("Could not align:", err)));
        actions.append(pivotButton, again);
    }
    const url = structureSourceUrl(structure);
    if (url) {
        const download = focusable(make('button', 'btn btn-sm'), ref + ':download');
        download.type = 'button';
        download.title = t("The PDB file as it was folded, not moved by the alignment.");
        const downloadLabel = make('span', undefined, t("Download PDB"));
        const downloadIcon = icon("download");
        download.append(downloadIcon, downloadLabel);
        download.addEventListener('click', () => downloadStructureFile(url, structureFile(structure)));
        actions.append(download);
    }
    body.append(actions);
    return body;
}

function structureRow(structure, index, pivot) {
    const ref = structure.cell.transform.ref;
    const name = structureName(structure);
    const hidden = structureHidden(structure);
    const open = expandedRows.has(ref);
    const row = make('div', 'srow' + (hidden ? ' hidden-structure' : '') + (open ? ' expanded' : ''));
    row.dataset.ref = ref;
    const main = make('div', 'srow-main');
    const swatch = make('span', 'srow-swatch');
    swatch.style.setProperty('--sw', structureColor(ref));
    const text = make('div', 'srow-text');
    text.addEventListener('click', () => toggleRow(ref));
    const nameLine = make('div', 'srow-name', name);
    text.append(nameLine);
    const sub = make('div', 'srow-sub');
    const fit = fitStats.get(ref);
    let badge = null;
    if (ref === pivot.cell.transform.ref) {
        badge = chip(t("Pivot"), 'pivot', t("The other structures are aligned to this one."));
    } else if (fit) {
        badge = chip(t("RMSD {rmsd} Å", {rmsd: fit.rmsd.toFixed(1)}), 'ok',
            t("Aligned to the pivot: {used} of {aligned} Cα pairs, {identity}% sequence identity.",
                {used: fit.used, aligned: fit.aligned, identity: Math.round(100 * fit.identity)}));
    } else if (structure.transform) {
        badge = chip(t("Aligned"), 'ok', t("Aligned to the pivot."));
    }
    const noData = coloringOf(structure) === NO_DATA ? chip(t("No data"), 'warn', t("This structure has no values for the layer the scene is colored by.")) : null;
    const count = make('span', undefined, t("{n} residues", {n: residueCount(structure)}));
    sub.append(...(badge ? [badge] : []), ...(noData ? [noData] : []), count);
    text.append(sub);

    const eyeLabel = hidden ? t("Show {name}", {name}) : t("Hide {name}", {name});
    const eyeIcon = icon("eye");
    setIcon(eyeIcon, hidden ? "eye-off" : "eye");
    const eye = focusable(iconButton(eyeLabel, eyeIcon, () => {
        plugin.managers.structure.component.toggleVisibility(structure.components);
    }), ref + ':eye');
    eye.setAttribute('aria-pressed', hidden ? 'true' : 'false');
    const zoom = focusable(iconButton(t("Zoom to {name}", {name}), icon("crosshair"), () => zoomToStructure(structure)), ref + ':zoom');
    const details = focusable(iconButton(t("Details of {name}", {name}), icon("chevron-down"), () => toggleRow(ref), 'exp-btn'), ref + ':details');
    details.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) details.setAttribute('aria-controls', 'sb-' + safeId(ref));
    const remove = focusable(iconButton(t("Remove {name}", {name}), icon("trash-2"), () => removeStructure(structure), 'btn-danger-ghost'), ref + ':remove');
    main.append(swatch, text, eye, zoom, details, remove);
    row.append(main);
    if (open) row.append(structureDetails(structure, pivot));
    return row;
}

function toggleRow(ref) {
    if (expandedRows.has(ref)) expandedRows.delete(ref);
    else expandedRows.add(ref);
    renderStructures();
}

function renderStructures() {
    if (!plugin) return;
    if (rowControlsBusy) {
        renderRowsAgain = true;
        return;
    }
    const structures = loadedStructures();
    const list = byId('struct-list');
    const focused = list.contains(document.activeElement) ? document.activeElement.dataset.focusKey : null;
    for (const ref of [...expandedRows]) {
        if (!structures.some(structure => structure.cell.transform.ref === ref)) expandedRows.delete(ref);
    }
    if (structures.length) {
        try {
            refreshFitStats();
        } catch (err) {
            console.warn("Could not measure the alignments:", err);
        }
        const pivot = pivotStructure();
        list.replaceChildren(...structures.map((structure, index) => structureRow(structure, index, pivot)), ...jobRows());
    } else {
        fitStats.clear();
        const hint = make('div', 'hint', t("Structures appear here as they are folded."));
        const rows = jobRows();
        list.replaceChildren(...(rows.length ? rows : [hint]));
    }
    byId('struct-count').textContent = structures.length ? '(' + structures.length + ')' : '';
    updateView();
    renderColorCard();
    refreshResidue();
    scheduleSequence();
    if (focused) list.querySelector('[data-focus-key="' + focused.replace(/["\\]/g, '\\$&') + '"]')?.focus();
}

byId('struct-list').addEventListener('pointerdown', event => {
    if (event.target.matches?.('input[type="range"]')) rowControlsBusy = true;
});
function releaseRows() {
    if (!rowControlsBusy) return;
    rowControlsBusy = false;
    if (renderRowsAgain) {
        renderRowsAgain = false;
        scheduleStructures();
    }
}
document.addEventListener('pointerup', releaseRows);
document.addEventListener('pointercancel', releaseRows);

function zoomToStructure(structure) {
    const loci = molstar.lib.structure.Structure.toStructureElementLoci(structure.cell.obj.data);
    plugin.managers.camera.focusLoci(loci);
}

// ---- Removing -----------------------------------------------------------------------
// The top of the chain of nodes a structure came from (download, parse, model,
// structure): Mol*'s own remove takes the structure and leaves the rest behind.
function structureDataRoot(structure) {
    const state = plugin.state.data;
    let cell = structure.cell;
    while (cell.transform.parent && cell.transform.parent !== state.tree.root.ref) cell = state.cells.get(cell.transform.parent);
    return cell.transform.ref;
}

async function deleteStructureData(structure) {
    const state = plugin.state.data;
    await plugin.runTask(state.updateTree(state.build().delete(structureDataRoot(structure))));
}

// Removing a structure can be undone for a few seconds: the session as it was comes back.
function removeStructure(structure) {
    const ref = structure.cell.transform.ref;
    const name = structureName(structure);
    const before = snapshotForUndo();
    runExclusive(async () => {
        const target = structureByRef(ref);
        if (target) await deleteStructureData(target);
        fitStats.delete(ref);
        renderColorCard();
        scheduleStructures();
        setTimeout(saveSession, 1500);
    }).catch(err => reportFailure("Could not remove the structure:", err));
    const undo = before ? {label: t("Undo"), run: () => undoSceneChange(before)} : {};
    showStatus(t("Removed {name}.", {name}), 6000, undo);
}

// The session as it is, to come back to; null when Mol* cannot say.
function snapshotForUndo() {
    try {
        return takeSnapshot();
    } catch (error) {
        console.warn("Could not keep the scene for Undo:", error);
        return null;
    }
}

function undoSceneChange(snapshot) {
    runExclusive(async () => {
        await plugin.state.setSnapshot(snapshot);
        await reconcileLayers();
        syncCanvasProps();
        deriveColorMode();
        resetPointerState();
        scheduleStructures();
        setTimeout(saveSession, 1500);
    });
}

// ---- Opening files ---------------------------------------------------------------------
const STRUCTURE_FILE = /\.(pdb|ent|cif|mmcif)$/i;

// Each file is loaded as a folded structure is: it is colored like the others and aligned to the pivot.
function openStructureFiles(files) {
    // A file dropped while Mol* is not running goes nowhere (the drop is still kept from the browser, which would open it).
    if (!plugin || startFailure) return;
    const chosen = [...files].filter(file => STRUCTURE_FILE.test(file.name));
    if (!chosen.length) {
        showStatus(t("Only PDB and mmCIF files can be opened here."), 6000, {error: true});
        return;
    }
    for (const file of chosen) {
        const format = /\.(cif|mmcif)$/i.test(file.name) ? 'mmcif' : 'pdb';
        enqueueLoad(async () => {
            const data = await file.text();
            await viewerInstance.loadStructureFromData(data, format, {dataLabel: file.name});
        }, file.name);
    }
}

byId('open-file').addEventListener('click', () => byId('open-structure-input').click());
byId('open-structure-input').addEventListener('change', event => {
    openStructureFiles(event.target.files);
    event.target.value = '';
});

// A file dropped on the view is opened by the page, so that it is colored and aligned like the others, and
// Mol*'s own drop overlay (English only) does not take it too; the page shows its own hint while one is over.
const stage = byId('stage');
function droppedFiles(event) {
    return event.dataTransfer && [...event.dataTransfer.types].includes('Files');
}
document.addEventListener('dragenter', event => {
    if (!droppedFiles(event) || !stage.contains(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    stage.classList.add('dropping');
}, true);
document.addEventListener('dragover', event => {
    if (!droppedFiles(event) || !stage.contains(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
}, true);
document.addEventListener('dragleave', event => {
    if (!stage.contains(event.relatedTarget)) stage.classList.remove('dropping');
}, true);
document.addEventListener('drop', event => {
    stage.classList.remove('dropping');
    if (!droppedFiles(event) || !stage.contains(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    openStructureFiles(event.dataTransfer.files);
}, true);
