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
// and a 'esmfold-layers' event on window, after a layer changes, with
// detail.structures: [{ref, file, active, layers: [ids]}]. The card that
// offers the layers and draws their legends is esmfold_color.js.
const LAYER_ROUTE = '/api/esmfold/layers/';
const PLDDT_LAYER = 'plddt';
const layerListings = new Map();
// The Viewer writes the names of its two layers into the residue data file in English (utilities/Residue_Data.py), and a
// saved file keeps its language whichever the page is shown in. So the page shows these two names in its own language,
// and any other layer's name as its file has it.
const LAYER_NAMES = new Map([
    ['Information Content', t("Information Content")],
    ['Residue Agreement', t("Residue Agreement")],
]);

// Runs `task` after the structure loads and layer changes queued before it.
function runExclusive(task) {
    const result = structureQueue.then(task);
    structureQueue = result.catch(err => console.error("Error changing a layer:", err));
    return result;
}

// ---- The scene, while Mol* changes it --------------------------------------------------
// Mol* rebuilds its structure hierarchy when an update has ended, not while it runs, but an update that
// replaces the scene (a session opened, Undo, Clear, a structure removed) takes the old nodes out of the
// state at its start and then waits for the new ones to load. For that time the hierarchy still lists
// structures whose nodes are gone, and nothing says so: a timer or a pointer event of the page can land
// there, and a name, a file or a layer read through such a structure finds nothing above it (the page once
// threw there, from the list's redraw). So the page reads the hierarchy through the functions below:
// nodesAbove() says what a structure stands on, or that it is gone, loadedStructures() lists the structures
// that are not, and sceneBeingReplaced() says when the page waits before it draws.

// The structures Mol*'s hierarchy lists, as it last was built.
function listedStructures() {
    return plugin.managers.structure.hierarchy.current.structures;
}

// The nodes a structure stands on, its own first: the structure, its model, the parse and the download or
// raw data it came from. null when any of them is not in Mol*'s state (a node that came back under the same
// ref is another node): the structure is gone, and has nothing above it to read and no top of a chain to
// delete; a guess at either would be another structure's.
function nodesAbove(structure) {
    const state = plugin.state.data;
    if (state.cells.get(structure.cell.transform.ref) !== structure.cell) return null;
    const nodes = [structure.cell];
    for (let cell = structure.cell; ; ) {
        const parent = cell.transform.parent;
        if (!parent || parent === cell.transform.ref || parent === state.tree.root.ref) return nodes;
        cell = state.cells.get(parent);
        if (!cell) return null;
        nodes.push(cell);
    }
}

// The structures of the scene as it is: the ones the hierarchy lists that are not gone.
function loadedStructures() {
    return listedStructures().filter(structure => nodesAbove(structure));
}

// Mol* is replacing the scene: an update is running and the hierarchy still lists a structure it has removed.
// What the page draws or forgets from the hierarchy now is drawn or forgotten from a scene that is about to
// change, so it waits for the update to end (renderStructures, and the subscription in esmfold_shell.js).
function sceneBeingReplaced() {
    return plugin.state.data.behaviors.isUpdating.value && listedStructures().some(structure => !nodesAbove(structure));
}

// The file an address names, such as the PDB file of /structures/<file>.
function fileOfUrl(url) {
    return decodeURIComponent(url.split('?')[0].split('/').pop());
}

// The PDB file a structure was loaded from, from its download node.
function structureFile(structure) {
    const url = structureSourceUrl(structure);
    return url ? fileOfUrl(url) : undefined;
}

function layerListing(file, refresh = false) {
    if (refresh || !layerListings.has(file)) {
        layerListings.set(file, fetch(LAYER_ROUTE + encodeURIComponent(file))
            .then(response => (response.ok ? response.json() : null))
            .then(listing => {
                for (const layer of listing?.layers ?? []) layer.label = LAYER_NAMES.get(layer.label) ?? layer.label;
                return listing;
            })
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
    renderColorCard();
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
