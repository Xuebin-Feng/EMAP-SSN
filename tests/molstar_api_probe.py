# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What the bundled Mol* offers the Fold View page, probed in a browser of its own.

The page builds its controls on the bundled Mol*'s public surface: the
Viewer's options that hide its interface, the layout call that brings it back,
the hover labels, the pointer events and the click's focus nodes, the managers
(measurement, snapshots, components), the picture helper, the colour themes and
legends, the layers' MolViewSpec properties. This script
runs that surface in Qt WebEngine and prints what it found as one JSON line,
so that tests/test_molstar_api_surface.py can pin it, and so that a newer Mol*
(the vendored files in src/resources/esmfold/) can be accepted only when it
still offers all of it.

It runs as a child process, because Chromium reads its flags once per process,
and a child can ask for the GPU that the test suite's own offscreen WebEngine
does not use: with ``--ignore-gpu-blocklist --disable-gpu-compositing`` Mol*
gets WebGL where the machine has a GPU (the answer says whether it did).

    python tests/molstar_api_probe.py [--no-webgl]

Not collected by unittest.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GL_FLAGS = "--ignore-gpu-blocklist --disable-gpu-compositing"
NO_GL_FLAGS = "--disable-gpu"
RESULT_PREFIX = "PROBE "

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{css}"><script src="{js}"></script>
<style>html, body {{ margin: 0; height: 100%; }} #viewer, #wrap {{ position: absolute; inset: 0; }}</style></head>
<body><div id="viewer"></div><div id="wrap" style="display:none"><canvas id="cv"></canvas></div></body></html>"""

# Everything below runs in the page. `PDB` is a small helix, as text.
PROBE = r"""
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const out = {};
const count = selector => document.querySelectorAll(selector).length;
const regions = () => ({left: count('.msp-layout-left'), right: count('.msp-layout-right'),
                        top: count('.msp-layout-top'), bottom: count('.msp-layout-bottom')});
out.version = molstar.version;
out.webgl = !!document.createElement('canvas').getContext('webgl2');
out.extensions = ['mvs', 'ma-quality-assessment', 'backgrounds'].filter(name => name in molstar.ExtensionMap);
out.libPlugin = ['PluginContext', 'DefaultPluginSpec', 'PluginConfig', 'StateTransforms'].filter(name => name in molstar.lib.plugin);
out.libStructure = ['StructureElement', 'StructureProperties', 'Unit', 'Queries', 'QueryContext', 'StructureSelection']
    .filter(name => name in molstar.lib.structure);
out.libMath = ['LinearAlgebra'].filter(name => name in molstar.lib.math);
// Mol*'s superposition algorithms are not on the global, which is why esmfold_superposition.js has its own.
out.superposition = Object.entries(molstar.lib).flatMap(([group, namespace]) => Object.keys(namespace || {}).map(name => group + '.' + name))
    .filter(name => /superpos|tmalign|alignandsuperpose/i.test(name));

// The Viewer as the page makes it: no Mol* interface, only the canvas.
const viewer = await molstar.Viewer.create('viewer', {
    layoutIsExpanded: false, layoutShowControls: false, layoutShowRemoteState: false,
    layoutShowSequence: true, layoutShowLog: true, layoutShowLeftPanel: true,
    viewportShowControls: false, viewportShowExpand: false, viewportShowToggleFullscreen: false,
    viewportShowSettings: false, viewportShowSelectionMode: false, viewportShowAnimation: false,
    viewportShowReset: false, viewportShowScreenshotControls: false
});
const plugin = viewer.plugin;
out.hiddenRegions = regions();
out.hiddenButtons = count('button');
out.leftoverClasses = ['msp-viewport-controls', 'msp-highlight-toast-wrapper', 'msp-logo', 'msp-drag-drop-overlay']
    .filter(name => count('.' + name) > 0);
out.viewerMethods = ['loadStructureFromData', 'loadStructureFromUrl', 'handleResize'].filter(name => typeof viewer[name] === 'function');
await viewer.loadStructureFromData(PDB, 'pdb', {dataLabel: 'probe'});
await wait(800);
const structures = plugin.managers.structure.hierarchy.current.structures;
out.structures = structures.length;

// The door: Mol*'s own panels, at run time. setProps changes the state without drawing; updateProps draws.
plugin.layout.setProps({showControls: true});
await wait(400);
out.setPropsDraws = regions().left > 0;
plugin.layout.setProps({showControls: false});
plugin.layout.updateProps({showControls: true});
await wait(800);
out.doorRegions = regions();
out.doorButtons = count('button');
plugin.layout.updateProps({showControls: false});
await wait(500);
out.closedAgain = regions();

// The hover label the page's tooltip is built from.
const L = molstar.lib.structure;
const structure = structures[0].cell.obj.data;
const loci = L.StructureSelection.toLociWithSourceUnits(L.Queries.generators.atoms({
    residueTest: c => L.StructureProperties.residue.label_seq_id(c.element) === 5})(new L.QueryContext(structure)));
const seen = [];
const subscription = plugin.behaviors.labels.highlight.subscribe(event => seen.push(event.labels.map(String)));
plugin.managers.interactivity.lociHighlights.highlightOnly({loci});
await wait(400);
subscription.unsubscribe();
out.labels = seen.filter(labels => labels.length).slice(-1)[0] || [];

// The managers and calls the shell drives.
const m = plugin.managers;
const has = (object, names) => names.filter(name => typeof object[name] === 'function');
out.managers = {
    hierarchy: Object.keys(m.structure).includes('hierarchy'),
    component: has(m.structure.component, ['addRepresentation', 'updateRepresentations', 'updateRepresentationsTheme', 'updateReprParams', 'applyPreset', 'toggleVisibility', 'removeRepresentations']),
    measurement: has(m.structure.measurement, ['addDistance', 'addAngle', 'addDihedral', 'addLabel']),
    selection: has(m.structure.selection, ['fromLoci', 'add', 'remove', 'set', 'clear']),
    focus: has(m.structure.focus, ['set', 'setFromLoci', 'addFromLoci', 'clear']),
    camera: has(m.camera, ['focusLoci', 'reset']),
    snapshot: has(m.snapshot, ['add', 'serialize', 'open']),
    interactivity: typeof m.interactivity.lociHighlights.highlightOnly === 'function',
    lociLabels: has(m.lociLabels, ['addProvider', 'getLabels']),
    // Without WebGL there is no screenshot helper.
    screenshot: plugin.helpers.viewportScreenshot ? has(plugin.helpers.viewportScreenshot, ['getImageDataUri', 'download', 'copyToClipboard']) : null,
    layout: has(plugin.layout, ['updateProps', 'setProps']),
    state: has(plugin.state, ['getSnapshot', 'setSnapshot']),
};
out.removeStructure = has(m.structure.hierarchy, ['remove']);
const reprs = plugin.representation.structure.registry.list.map(entry => entry.name);
out.representations = ['cartoon', 'backbone', 'ball-and-stick', 'spacefill', 'gaussian-surface', 'molecular-surface']
    .filter(name => reprs.includes(name));
const themes = plugin.representation.structure.themes.colorThemeRegistry.list.map(entry => entry.name);
out.themes = ['plddt-confidence', 'mvs-annotation', 'uniform', 'secondary-structure', 'sequence-id', 'chain-id', 'hydrophobicity']
    .filter(name => themes.includes(name));
out.transforms = ['TransformStructureConformation', 'CustomModelProperties', 'CustomStructureProperties']
    .filter(name => name in molstar.lib.plugin.StateTransforms.Model);
const snapshot = plugin.state.getSnapshot({camera: !!plugin.canvas3d});
out.snapshotKeys = Object.keys(snapshot);

// ---- What the page's pointer, pin, tools, representations, legends, session and picture rely on, beyond the names above.
const subject = value => !!value && typeof value.subscribe === 'function' && typeof value.next === 'function' && 'value' in value;
const interaction = plugin.behaviors.interaction;
out.subjects = {
    hover: subject(interaction.hover), click: subject(interaction.click),
    measurement: subject(m.structure.measurement.behaviors.state),
    hierarchySelection: typeof m.structure.hierarchy.behaviors.selection.subscribe === 'function',
    cellStateUpdated: typeof plugin.state.data.events.cell.stateUpdated.subscribe === 'function',
    log: has(plugin.log, ['info', 'warn', 'error'])
};
const hierarchy = () => m.structure.hierarchy.current.structures;
const stateCells = () => plugin.state.data.cells;
// The residue under a pointer, as esmfold_residue.js reads it from a loci.
const residueOf = candidate => {
    const entry = hierarchy().find(item => item.model?.cell.obj.data === candidate.structure.model);
    const location = L.StructureElement.Loci.getFirstLocation(candidate);
    const flags = L.StructureProperties.residue.secondary_structure_type(location);
    const types = L.Types.SecondaryStructureType;
    return {structureFound: !!entry, chain: L.StructureProperties.chain.auth_asym_id(location),
            number: L.StructureProperties.residue.auth_seq_id(location), name: L.StructureProperties.atom.label_comp_id(location),
            bfactor: L.StructureProperties.atom.B_iso_or_equiv(location),
            secondary: types.is(flags, types.Flag.Helix) ? 'Helix' : types.is(flags, types.Flag.Beta) ? 'Strand' : types.is(flags, types.Flag.Turn) ? 'Turn' : null};
};
out.residue = residueOf(loci);
const secondaryTypes = L.Types.SecondaryStructureType;
out.secondaryFlags = {flags: ['Helix', 'Beta', 'Turn'].filter(name => typeof secondaryTypes.Flag[name] === 'number'), is: typeof secondaryTypes.is};
// A pointer event is whatever is pushed into the subject, and reaches its subscribers as it is.
const NO_KEYS = {shift: false, alt: false, control: false, meta: false};
{
    const heard = [];
    const hearing = interaction.hover.subscribe(event => heard.push(event));
    const event = {current: {loci}, modifiers: {}, buttons: 0, button: 0};
    interaction.hover.next(event);
    hearing.unsubscribe();
    out.hoverDelivered = heard.length >= 2 && heard[heard.length - 1] === event;
}
// The representation's picking: a group of a render object is a residue (the tests, and a real pointer, rely on it).
const reprOf = () => hierarchy()[0].components[0].representations[0].cell.obj.data.repr;
const pick = group => ({loci: reprOf().getLoci({objectId: reprOf().renderObjects[0].id, instanceId: 0, groupId: group}), repr: reprOf()});
out.pick = {renderObjects: reprOf().renderObjects.length > 0, residue: residueOf(pick(5).loci).number, structureLoci: L.StructureElement.Loci.is(pick(5).loci)};
// Mol*'s own click puts the residue in focus with nodes of its own, tagged structure-focus-*: the page lets go of them.
const focusCells = () => [...stateCells().values()].filter(cell => cell.transform.tags?.some(tag => tag.startsWith('structure-focus-')));
const cellsBefore = stateCells().size;
interaction.click.next({current: pick(5), modifiers: NO_KEYS, buttons: 1, button: 1});
await wait(700);
out.focusClick = {nodes: focusCells().length, focus: !!m.structure.focus.current,
                  tags: [...new Set(focusCells().flatMap(cell => cell.transform.tags.filter(tag => tag.startsWith('structure-focus-'))))].sort(),
                  roots: focusCells().filter(cell => cell.transform.tags.some(tag => tag.endsWith('-sel'))).length,
                  grown: stateCells().size - cellsBefore};
{
    const raw = plugin.state.getSnapshot({camera: false});
    out.focusInSnapshot = {tags: raw.data.tree.transforms.filter(node => (node.tags || []).some(tag => tag.startsWith('structure-focus-'))).length,
                           focusKeys: Object.keys(raw.structureFocus || {}).sort()};
}
m.structure.focus.clear();
await wait(400);
{
    const state = plugin.state.data, update = state.build();
    for (const cell of focusCells().filter(item => item.transform.tags.some(tag => tag.endsWith('-sel')))) update.delete(cell.transform.ref);
    await plugin.runTask(state.updateTree(update));
    out.focusReleased = {nodes: focusCells().length, focus: !!m.structure.focus.current, back: stateCells().size === cellsBefore};
}
// The tools: a distance between two CA atoms and a label, in the measurement manager's own group, which Clear deletes.
const residueLoci = number => L.StructureSelection.toLociWithSourceUnits(L.Queries.generators.atoms({
    residueTest: c => L.StructureProperties.residue.auth_seq_id(c.element) === number})(new L.QueryContext(structure)));
const caLoci = number => L.StructureSelection.toLociWithSourceUnits(L.Queries.generators.atoms({
    residueTest: c => L.StructureProperties.residue.auth_seq_id(c.element) === number,
    atomTest: c => L.StructureProperties.atom.label_atom_id(c.element) === 'CA'})(new L.QueryContext(structure)));
{
    const measurement = m.structure.measurement;
    let heard = 0;
    const hearing = measurement.behaviors.state.subscribe(() => { heard++; });
    out.measurementBefore = {group: !!measurement.getGroup(), stateKeys: Object.keys(measurement.state).sort()};
    await measurement.addDistance(caLoci(2), caLoci(8));
    await measurement.addLabel(residueLoci(4));
    const group = measurement.getGroup();
    out.measurement = {distances: measurement.state.distances.length, labels: measurement.state.labels.length,
                       groupKeys: group ? Object.keys(group).sort() : null, groupRef: group ? typeof group.ref : null,
                       groupState: group ? typeof group.state : null, heard: heard > 1};
    const state = plugin.state.data;
    await plugin.runTask(state.updateTree(state.build().delete(group.ref)));
    hearing.unsubscribe();
    out.measurementCleared = {distances: measurement.state.distances.length, labels: measurement.state.labels.length, group: !!measurement.getGroup()};
}
// Colour legends of the built-in themes, as the Color by card draws them.
{
    const registry = plugin.representation.structure.themes.colorThemeRegistry;
    const ss = registry.create('secondary-structure', {structure}, {}).legend;
    const position = registry.create('sequence-id', {structure}, {}).legend;
    const integers = list => Array.isArray(list) && list.length > 1 && list.every(Number.isInteger);
    out.legends = {
        secondary: {kind: ss?.kind, names: (ss?.table || []).map(row => row[0]).sort(), colors: integers((ss?.table || []).map(row => row[1]))},
        position: {kind: position?.kind, colors: integers(position?.colors)}
    };
    const annotation = registry.get('mvs-annotation');
    out.annotationTheme = ['annotationId', 'fieldName', 'palette'].filter(name => name in annotation.getParams({structure}));
}
// The undocumented MolViewSpec properties the page's layers attach (esmfold_layers.js).
out.mvsProperties = {
    model: ['mvs-is-mvs-model', 'mvs-annotations'].filter(name => !!plugin.customModelProperties.get(name)),
    structure: ['mvs-annotation-tooltips'].filter(name => !!plugin.customStructureProperties.get(name))
};
// Another representation with Mol*'s defaults and its own colours, the colours of every representation changed, a representation removed.
{
    const component = hierarchy()[0].components[0];
    const before = component.representations.length;
    await plugin.builders.structure.representation.addRepresentation(component.cell, {
        type: 'ball-and-stick', typeParams: {alpha: 0.5}, color: 'uniform', colorParams: {value: 0xff0000}});
    await wait(400);
    const made = hierarchy()[0].components[0].representations;
    const params = made[made.length - 1].cell.transform.params;
    out.addRepresentation = {added: made.length - before, type: params.type.name, alpha: params.type.params.alpha,
                             theme: params.colorTheme.name, value: params.colorTheme.params.value};
    await m.structure.component.updateRepresentationsTheme(hierarchy()[0].components, {color: 'uniform', colorParams: {value: 0x00ff00}});
    await wait(300);
    out.themeUpdate = hierarchy()[0].components[0].representations.map(item => [item.cell.transform.params.colorTheme.name, item.cell.transform.params.colorTheme.params.value]);
    const state = plugin.state.data;
    await plugin.runTask(state.updateTree(state.build().delete(made[made.length - 1].cell.transform.ref)));
    out.representationRemoved = hierarchy()[0].components[0].representations.length === before;
}
// The session: Mol*'s own file (what the Session menu downloads and opens) and the plain snapshot the page posts to the Viewer.
{
    const camera = !!plugin.canvas3d;
    const taken = plugin.state.getSnapshot({camera});
    const blob = await m.snapshot.serialize({params: {camera}});
    const file = JSON.parse(await blob.text());
    out.serialize = {blob: blob instanceof Blob, keys: Object.keys(file).sort(), entries: file.entries.length,
                     entryKeys: Object.keys(file.entries[0]).sort(), version: file.version, snapshotKeys: Object.keys(file.entries[0].snapshot)};
    const clear = async () => {
        const state = plugin.state.data, update = state.build();
        for (const ref of state.tree.children.get(state.tree.root.ref).toArray()) update.delete(ref);
        await plugin.runTask(state.updateTree(update));
    };
    await clear();
    const cleared = hierarchy().length;
    await m.snapshot.open(new File([JSON.stringify(file)], 'probe.molj'));
    await wait(500);
    const opened = hierarchy().length;
    await clear();
    await plugin.state.setSnapshot(taken);
    await wait(500);
    out.sessionRoundTrip = {cleared, opened, setSnapshot: hierarchy().length};
}

// A plugin without the Viewer's interface: pLDDT colours need the quality-assessment extension.
const headless = async extensions => {
    const spec = molstar.lib.plugin.DefaultPluginSpec();
    for (const name of extensions) spec.behaviors.push(molstar.ExtensionMap[name]);
    const bare = new molstar.lib.plugin.PluginContext(spec);
    await bare.init();
    return bare.representation.structure.themes.colorThemeRegistry.list.some(entry => entry.name === 'plddt-confidence');
};
out.plddtWithoutExtension = await headless([]);
out.plddtWithExtension = await headless(['ma-quality-assessment']);

// The canvas, where there is WebGL.
if (plugin.canvas3d) {
    const uri = await plugin.helpers.viewportScreenshot.getImageDataUri();
    out.screenshot = uri.startsWith('data:image/png;base64,') && uri.length > 2000;
    out.project = typeof plugin.canvas3d.camera.project === 'function';
    out.transparentBackground = typeof plugin.canvas3d.setProps === 'function';
    // The picture the page saves: transparent or not, without the axes, at the canvas's size or a size asked for.
    const helper = plugin.helpers.viewportScreenshot;
    const kept = helper.behaviors.values.value;
    const canvas = document.querySelector('#viewer canvas');
    const decode = async options => {
        helper.behaviors.values.next({...kept, axes: {name: 'off', params: {}}, ...options});
        const image = new Image();
        image.src = await helper.getImageDataUri();
        await image.decode();
        const picture = document.createElement('canvas');
        picture.width = image.width;
        picture.height = image.height;
        const context = picture.getContext('2d', {willReadFrequently: true});
        context.drawImage(image, 0, 0);
        return {size: [image.width, image.height], alpha: context.getImageData(2, 2, 1, 1).data[3]};
    };
    try {
        const viewport = await decode({transparent: false, resolution: {name: 'viewport', params: {}}});
        const custom = await decode({transparent: false, resolution: {name: 'custom', params: {width: 2 * canvas.width, height: 2 * canvas.height}}});
        const clear = await decode({transparent: true, resolution: {name: 'viewport', params: {}}});
        out.pictures = {values: Object.keys(kept).sort(), transparentIsBoolean: typeof kept.transparent === 'boolean',
                        axesName: kept.axes && kept.axes.name, resolutionName: kept.resolution && kept.resolution.name,
                        viewportIsCanvas: viewport.size[0] === canvas.width && viewport.size[1] === canvas.height,
                        customSize: custom.size[0] === 2 * canvas.width && custom.size[1] === 2 * canvas.height,
                        opaqueAlpha: viewport.alpha, transparentAlpha: clear.alpha};
    } finally {
        helper.behaviors.values.next(kept);
    }
    // What the page sets on the canvas whatever a session says: the background, and the axes gizmo in a corner.
    const axes = plugin.canvas3d.props.camera.helper.axes;
    plugin.canvas3d.setProps({renderer: {backgroundColor: 0x123456},
                              camera: {helper: {axes: axes.name === 'on' ? {name: 'on', params: {...axes.params, location: 'bottom-right'}} : axes}}});
    out.canvasProps = {background: plugin.canvas3d.props.renderer.backgroundColor === 0x123456, axes: plugin.canvas3d.props.camera.helper.axes.name,
                       location: (plugin.canvas3d.props.camera.helper.axes.params || {}).location};
    out.cameraState = ['position', 'target', 'up', 'radius'].filter(name => name in plugin.canvas3d.camera.state);
} else {
    out.screenshot = null;
}

// Mol*'s own remove of a structure takes the structure and leaves the nodes it came from (the download or raw data,
// its parse and its model), which esmfold_structures.js deletes itself from the top of the chain (structureDataRoot).
{
    const kinds = () => {
        const counts = {};
        for (const cell of stateCells().values()) {
            const id = cell.transform.transformer.id.replace('ms-plugin.', '');
            counts[id] = (counts[id] || 0) + 1;
        }
        return counts;
    };
    const before = kinds();
    await m.structure.hierarchy.remove(hierarchy());
    await wait(500);
    const left = kinds();
    const state = plugin.state.data, update = state.build();
    for (const ref of state.tree.children.get(state.tree.root.ref).toArray()) update.delete(ref);
    await plugin.runTask(state.updateTree(update));
    out.hierarchyRemove = {before, left, structures: hierarchy().length, afterRootDelete: stateCells().size};
}
return out;
"""


def main(argv):
    from tests import esmfold_layer_fixtures as fixtures

    webgl = "--no-webgl" not in argv
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = GL_FLAGS if webgl else NO_GL_FLAGS
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer, QUrl
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    esmfold = SRC / "resources" / "esmfold"
    folder = tempfile.TemporaryDirectory()
    page = Path(folder.name, "probe.html")
    page.write_text(PAGE.format(css=(esmfold / "molstar.css").as_uri(), js=(esmfold / "molstar.js").as_uri()),
                    encoding="utf-8")
    pdb = fixtures.helix_pdb(fixtures.FIRST_SEQUENCE, fixtures.plddt_values(33))
    source = ("const PDB = " + json.dumps(pdb) + ";\nwindow.__result = undefined;\n(async () => { try {"
              " window.__result = JSON.stringify(await (async () => {" + PROBE + "})()); }"
              " catch (error) { window.__result = JSON.stringify({error: String(error && error.stack || error)}); } })(); 0")

    app = QApplication(sys.argv[:1])
    view = QWebEngineView()
    view.resize(1200, 800)
    view.show()
    outcome = {"code": 3}
    started = time.monotonic()

    def poll():
        def got(value):
            if value:
                print(RESULT_PREFIX + value, flush=True)
                outcome["code"] = 0
                app.quit()
            elif time.monotonic() - started > 150:
                print(RESULT_PREFIX + json.dumps({"error": "timed out"}), flush=True)
                app.quit()
            else:
                QTimer.singleShot(250, poll)
        view.page().runJavaScript("window.__result", 0, got)

    def loaded(ok):
        if not ok:
            print(RESULT_PREFIX + json.dumps({"error": "the page did not load"}), flush=True)
            app.quit()
            return
        view.page().runJavaScript(source, 0, lambda _: poll())

    view.loadFinished.connect(loaded)
    view.load(QUrl.fromLocalFile(str(page)))
    app.exec()
    view.close()
    folder.cleanup()
    return outcome["code"]


if __name__ == "__main__":
    for folder in (SRC, ROOT):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    sys.exit(main(sys.argv[1:]))
