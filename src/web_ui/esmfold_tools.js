/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's tools over the canvas: zoom to the pinned residue, measure a distance, label residues.
 */

// Two tools take the clicks on the canvas for themselves while they are on:
// Measure (two residues, and the distance between their C-alpha atoms is
// drawn and said) and Labels (a click puts the residue's name on it). Both
// use Mol*'s measurement manager, so what they draw is Mol*'s own and Clear
// removes it all; neither changes the pinned residue. Escape leaves the tool
// first. Loaded after esmfold_sequence.js.

let toolMode = null;
let measureFirst = null;

// ---- Which tool, and how it looks --------------------------------------------------------------
function setTool(mode) {
    if (toolMode === mode) mode = null;
    toolMode = mode;
    measureFirst = null;
    plugin.managers.structure.selection.clear();
    byId('t-measure').setAttribute('aria-pressed', mode === 'measure' ? 'true' : 'false');
    byId('t-label').setAttribute('aria-pressed', mode === 'label' ? 'true' : 'false');
    document.body.dataset.tool = mode || '';
    hideStatus();
    if (mode === 'measure') showStatus(t("Click two residues to measure the distance between them."), 0);
    else if (mode === 'label') showStatus(t("Click a residue to label it."), 0);
}

// Leaves the tool that is on; says whether one was.
function leaveTool() {
    if (!toolMode) return false;
    setTool(toolMode);
    return true;
}

byId('t-measure').addEventListener('click', () => setTool('measure'));
byId('t-label').addEventListener('click', () => setTool('label'));
byId('t-zoom').addEventListener('click', () => {
    if (pinnedResidue) zoomToResidue(pinnedResidue);
});

// ---- Measuring and labeling -----------------------------------------------------------------------
// The CA atom of a residue, as Mol* loci: what the distance is drawn between.
function caLoci(residue) {
    const structure = structureByRef(residue.ref);
    const { Queries, QueryContext, StructureProperties, StructureSelection } = molstar.lib.structure;
    const data = (structure.transform || structure).cell.obj.data;
    const query = Queries.generators.atoms({
        residueTest: context => StructureProperties.residue.auth_seq_id(context.element) === residue.number
            && StructureProperties.chain.auth_asym_id(context.element) === residue.chain,
        atomTest: context => StructureProperties.atom.label_atom_id(context.element) === 'CA'
    });
    return StructureSelection.toLociWithSourceUnits(query(new QueryContext(data)));
}

// Where a residue's CA atom is as drawn (after the alignment), or null.
function caPosition(residue) {
    const structure = structureByRef(residue.ref);
    const trace = nativeTrace(structure);
    const k = trace.ids.findIndex(([chain, number]) => chain === residue.chain && number === residue.number);
    return k < 0 ? null : displayedCoordinates(structure)[k];
}

async function measureClick(residue) {
    if (!measureFirst) {
        measureFirst = residue;
        plugin.managers.structure.selection.fromLoci('set', residueLoci(residue));
        showStatus(t("{name}: now click the second residue.", {name: residueTitle(residue)}), 0);
        return;
    }
    const first = measureFirst;
    measureFirst = null;
    plugin.managers.structure.selection.clear();
    if (sameResidue(first, residue)) {
        showStatus(t("Click two residues to measure the distance between them."), 0);
        return;
    }
    const a = caPosition(first), b = caPosition(residue);
    if (!a || !b) {
        showStatus(t("Those residues have no C-alpha atom to measure from."), 6000, {error: true});
        return;
    }
    await plugin.managers.structure.measurement.addDistance(caLoci(first), caLoci(residue));
    const distance = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    showStatus(t("{from} to {to}: {distance} Å", {from: residueTitle(first), to: residueTitle(residue), distance: distance.toFixed(1)}), 0);
}

async function labelClick(residue) {
    await plugin.managers.structure.measurement.addLabel(residueLoci(residue));
    showStatus(t("Labeled {name}.", {name: residueTitle(residue)}), 3000);
}

// A click on the canvas while a tool is on; Mol*'s own click has put the residue in focus, which is let go of.
function toolClick(residue, loci) {
    const work = toolMode === 'measure' ? measureClick(residue) : labelClick(residue);
    work.catch(err => reportFailure("Could not do that:", err));
    releaseFocus();
}

// ---- Clearing --------------------------------------------------------------------------------------
function annotationCount() {
    const state = plugin.managers.structure.measurement.state;
    return state.labels.length + state.distances.length + state.angles.length + state.dihedrals.length
        + state.orientations.length + state.planes.length;
}

function clearAnnotations() {
    const group = plugin.managers.structure.measurement.getGroup();
    if (!group) return;
    runExclusive(async () => {
        const state = plugin.state.data;
        await plugin.runTask(state.updateTree(state.build().delete(group.ref)));
        showStatus(t("Measurements and labels cleared."));
    }).catch(err => reportFailure("Could not clear them:", err));
}

byId('t-clear').addEventListener('click', clearAnnotations);

// The tools' buttons follow what there is to use them on: Zoom needs a pinned residue, Clear something to clear.
function refreshTools() {
    if (!plugin) return;
    byId('t-zoom').disabled = !pinnedResidue;
    byId('t-clear').hidden = annotationCount() === 0;
    document.body.dataset.annotations = annotationCount() ? '1' : '0';
}

function setupToolEvents() {
    plugin.managers.structure.measurement.behaviors.state.subscribe(refreshTools);
    refreshTools();
}
