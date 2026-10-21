# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What the bundled Mol* offers the Fold View page, probed in a browser of its own.

The page builds its controls on the Mol* 5.10.1 bundle's public surface: the
Viewer's options that hide its interface, the layout call that brings it back,
the hover labels, the managers, the colour themes the layers need. This script
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
} else {
    out.screenshot = null;
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
