/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's per-residue layers: a structure's colors from the Viewer's residue data, as MolViewSpec annotations.
 */

// ---- Per-residue layers ------------------------------------------
// esmfold writes numbers for each residue (such as how conserved its
// alignment column is) beside a structure's PDB, and the Viewer offers
// each number as a layer: /api/esmfold/layers/<structure>.pdb lists
// them, and each layer's address serves a MolViewSpec annotation (rows
// of residue selector, colour and tooltip; src/web_ui/residue_layers.py).
// The structure is not reloaded. Mol* gives every structure properties
// nodes in its state tree that hold MolViewSpec annotations, so a layer
// is attached by updating those nodes and giving the structure's
// representations Mol*'s mvs-annotation colour theme. The PDB, its
// pLDDT, the superposition and the saved session stay as they are, and
// pLDDT stays the default: no layer is attached until one is chosen.
//
// What the page's controls build on:
//   layerListing(file, refresh)   the Viewer's listing for a PDB file name
//   activeLayerId(structure)      'plddt', or the id of the layer attached
//   applyLayer(structure, id)     attach layer `id` and colour by it
//   restorePlddt(structure)       detach the layer and colour by pLDDT
//   reconcileLayers()             after a session restore: re-attach or drop
//   renderLayerPanel()            draw the colour switch and legend
// and a 'esmfold-layers' event on window, after a layer changes, with
// detail.structures: [{ref, file, active, layers: [ids]}].
const LAYER_ROUTE = '/api/esmfold/layers/';
const PLDDT_LAYER = 'plddt';
const layerListings = new Map();
const layerPanel = document.getElementById('layer-panel');

// Runs `task` after the structure loads and layer changes queued before it.
function runExclusive(task) {
    const result = structureQueue.then(task);
    structureQueue = result.catch(err => console.error("Error changing a layer:", err));
    return result;
}

function loadedStructures() {
    return plugin.managers.structure.hierarchy.current.structures;
}

// The PDB file a structure was loaded from, from its download node.
function structureFile(structure) {
    const cells = plugin.state.data.cells;
    for (let cell = structure.cell; cell; ) {
        const params = cell.transform.params;
        if (cell.transform.transformer.id === 'ms-plugin.download' && params?.url?.url) {
            return decodeURIComponent(params.url.url.split('?')[0].split('/').pop());
        }
        const parent = cell.transform.parent;
        cell = parent && parent !== cell.transform.ref ? cells.get(parent) : undefined;
    }
    return undefined;
}

function layerListing(file, refresh = false) {
    if (refresh || !layerListings.has(file)) {
        layerListings.set(file, fetch(LAYER_ROUTE + encodeURIComponent(file))
            .then(response => (response.ok ? response.json() : null))
            .catch(() => null));
    }
    return layerListings.get(file);
}

function activeLayerId(structure) {
    const annotations = structure.model?.properties?.cell.transform.params
        ?.properties?.['mvs-annotations']?.annotations;
    return annotations?.length ? annotations[0].id : PLDDT_LAYER;
}

// Attaches `layer` (a listing entry), or with none detaches whatever is
// attached, through the annotation properties of the structure's model
// and the tooltip properties of its structure. Listing them in
// autoAttach is what makes Mol* fetch the annotation. With keepOthers
// the annotations already attached stay, so that the structure is
// never coloured by one that is not there while it changes layers.
async function setLayerProperties(structure, layer, keepOthers = false) {
    const state = plugin.state.data;
    const modelCell = structure.model?.properties?.cell;
    const structureCell = structure.properties?.cell;
    if (!modelCell || !structureCell) throw new Error(t("the structure has no properties nodes"));
    const model = modelCell.transform.params;
    const attached = model.properties['mvs-annotations']?.annotations ?? [];
    const annotation = layer ? [{
        source: { name: 'url', params: { url: layer.annotation, format: 'json' } },
        schema: 'auth_residue',
        cifBlock: { name: 'index', params: { index: 0 } },
        cifCategory: null,
        fieldRemapping: [],
        id: layer.id
    }, ...(keepOthers ? attached.filter(other => other.id !== layer.id) : [])] : [];
    await plugin.runTask(state.updateTree(state.build().to(modelCell.transform.ref).update({
        ...model,
        autoAttach: layer ? ['mvs-is-mvs-model', 'mvs-annotations'] : [],
        properties: {
            ...model.properties,
            'mvs-is-mvs-model': { isMvs: !!layer },
            'mvs-annotations': { annotations: annotation }
        }
    })));
    const properties = structureCell.transform.params;
    await plugin.runTask(state.updateTree(state.build().to(structureCell.transform.ref).update({
        ...properties,
        autoAttach: layer ? ['mvs-annotation-tooltips'] : [],
        properties: {
            ...properties.properties,
            'mvs-annotation-tooltips': {
                tooltips: layer ? [{ annotationId: layer.id, fieldName: 'tooltip', textFormat: '{}' }] : []
            }
        }
    })));
}

function themeStructure(structure, theme) {
    return plugin.managers.structure.component.updateRepresentationsTheme(structure.components, theme);
}

async function restorePlddt(structure) {
    if (activeLayerId(structure) === PLDDT_LAYER) return;
    // Colour first, while the annotation theme is still applicable.
    await themeStructure(structure, { color: 'plddt-confidence' });
    await setLayerProperties(structure, null);
    layersChanged();
}

async function applyLayer(structure, layerId) {
    if (layerId === PLDDT_LAYER) return restorePlddt(structure);
    const file = structureFile(structure);
    const listing = file && await layerListing(file, true);
    const layer = listing?.layers.find(entry => entry.id === layerId && entry.annotation);
    if (!layer) throw new Error(t("no layer {layer} for {file}", {layer: layerId, file}));
    const switching = activeLayerId(structure) !== PLDDT_LAYER;
    await setLayerProperties(structure, layer, switching);
    await themeStructure(structure, {
        color: 'mvs-annotation',
        colorParams: { annotationId: layer.id, fieldName: 'color', palette: { name: 'direct', params: {} } }
    });
    // The colours have moved to the new layer, so the old one can go.
    if (switching) await setLayerProperties(structure, layer);
    layersChanged();
}

// A restored session names the layers it had attached, and the Viewer's
// residue data may have changed since: attach each again from a fresh
// listing, or go back to pLDDT when the layer is no longer offered.
async function reconcileLayers() {
    for (const structure of loadedStructures()) {
        const active = activeLayerId(structure);
        if (active === PLDDT_LAYER) continue;
        try {
            await applyLayer(structure, active);
        } catch (err) {
            console.warn("A restored layer is no longer available:", err);
            await restorePlddt(structure);
        }
    }
    layersChanged();
}

function layersChanged() {
    renderLayerPanel();
    window.dispatchEvent(new CustomEvent('esmfold-layers', { detail: { structures: layerSummary() } }));
    setTimeout(saveSession, 1500);
}

function layerSummary() {
    return loadedStructures().map(structure => ({
        ref: structure.cell.transform.ref,
        file: structureFile(structure),
        active: activeLayerId(structure)
    }));
}

// The colour switch and legend: plain controls over the functions above.
let panelRenderId = 0;
const HEX_COLOR = /^#[0-9a-f]{6}$/i;

// The card is placed by the page's style (top left of the stage), over the canvas.

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function legendKey(color, text) {
    const key = element('div', 'layer-key');
    const swatch = element('span', 'layer-swatch');
    if (HEX_COLOR.test(color)) swatch.style.background = color;
    const label = element('span', undefined, text);
    key.append(swatch, label);
    return key;
}

function layerLegend(layer, listing) {
    const legend = element('div', 'layer-legend');
    const shape = layer.legend;
    if (shape.kind === 'gradient') {
        const span = shape.max - shape.min;
        const stops = shape.stops.filter(([, color]) => HEX_COLOR.test(color))
            .map(([value, color]) => `${color} ${(100 * (value - shape.min) / span).toFixed(1)}%`);
        const bar = element('div', 'layer-bar');
        bar.style.background = `linear-gradient(to right, ${stops.join(', ')})`;
        const unit = layer.unit ? ` ${layer.unit}` : '';
        const ends = element('div', 'layer-ends');
        const low = element('span', undefined, `${shape.min}${unit}`);
        const high = element('span', undefined, `${shape.max}${unit}`);
        ends.append(low, high);
        legend.append(bar, ends);
        if (layer.covered < layer.residues) legend.append(legendKey(listing.background, listing.texts.no_data));
    } else {
        for (const band of shape.bands) legend.append(legendKey(band.color, band.label));
    }
    return legend;
}

async function renderLayerPanel() {
    if (!plugin) return;
    const renderId = ++panelRenderId;
    const rows = [];
    let title = '';
    for (const structure of loadedStructures()) {
        const file = structureFile(structure);
        const listing = file && await layerListing(file);
        if (!listing) continue;
        const active = listing.layers.find(entry => entry.id === activeLayerId(structure)) || listing.layers[0];
        const row = element('div', 'layer-row');
        const name = element('div', 'layer-name', file.replace(/\.pdb$/i, ''));
        row.append(name);
        if (listing.layers.length > 1) {
            const select = element('select');
            select.setAttribute('aria-label', `${listing.texts.color_by} ${file}`);
            for (const entry of listing.layers) {
                const option = element('option', undefined, entry.label);
                option.value = entry.id;
                option.selected = entry === active;
                select.append(option);
            }
            const ref = structure.cell.transform.ref;
            select.addEventListener('change', () => runExclusive(async () => {
                const target = loadedStructures().find(s => s.cell.transform.ref === ref);
                try {
                    if (target) await applyLayer(target, select.value);
                } catch (err) {
                    console.warn("Could not apply the layer:", err);
                    showStatus(t("Could not color {file} by that layer.", {file}));
                    renderLayerPanel();
                }
            }));
            row.append(select);
        }
        row.title = active.description || '';
        row.append(layerLegend(active, listing));
        for (const words of [active.source, listing.layers.length === 1 && listing.unavailable]) {
            if (!words) continue;
            const note = element('div', 'layer-note', words);
            row.append(note);
        }
        title = title || listing.texts.color_by;
        rows.push(row);
    }
    // A newer render replaces this one, which an await made stale.
    if (renderId !== panelRenderId) return;
    const heading = rows.length ? [element('div', 'layer-title', title)] : [];
    layerPanel.replaceChildren(...heading, ...rows);
    layerPanel.style.display = rows.length ? 'block' : 'none';
}
