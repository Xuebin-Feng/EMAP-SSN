# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What the Fold View page does over a real canvas, found in a browser of its own.

The test suite's Qt WebEngine runs with ``--disable-gpu``, so Mol* has no
canvas there and the page says so. This script opens the page in a child
process that asks for the GPU (``--ignore-gpu-blocklist
--disable-gpu-compositing``, as tests/molstar_api_probe.py does), over the
same server, structures and harness as tests/test_esmfold_shell.py, and
prints what it measured as one JSON line for tests/test_esmfold_shell_webgl.py
to check: the answer says whether the machine gave it WebGL at all.

    python tests/esmfold_gl_scenarios.py [--pictures FOLDER]

With --pictures, it also saves the page as pictures of the window (light,
dark, a narrow window, Simplified Chinese) in FOLDER, for a person to look at.

Not collected by unittest.
"""

import base64
import json
import os
import sys
import unittest
from pathlib import Path

import numpy

GL_FLAGS = "--ignore-gpu-blocklist --disable-gpu-compositing"
# Only the child process asks for the GPU: tests/test_esmfold_shell_webgl.py imports this file in the test
# process, and the pages of every other test there must stay without WebGL (as in a clean checkout).
if __name__ == "__main__":
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = GL_FLAGS
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

RESULT_PREFIX = "SCENARIO "
FACTS = {}
PICTURES = []

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from tests import esmfold_layer_fixtures as fixtures  # noqa: E402
from tests.esmfold_page_harness import FoldViewPageMixin, FoldViewServerMixin  # noqa: E402
from tests.esmfold_saved_sessions import saved_scene, write_saved_structures  # noqa: E402

# What the page's canvas draws at a pixel, as the colour the screenshot helper gives it, and how much of the canvas is not background.
PIXELS = """
window.canvasPixels = async function (spots) {
    const uri = await plugin.helpers.viewportScreenshot.getImageDataUri();
    const image = new Image();
    image.src = uri;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext('2d', {willReadFrequently: true});
    context.drawImage(image, 0, 0);
    const hex = data => '#' + [data[0], data[1], data[2]].map(v => v.toString(16).padStart(2, '0')).join('');
    const corner = hex(context.getImageData(2, 2, 1, 1).data);
    const all = context.getImageData(0, 0, image.width, image.height).data;
    let other = 0;
    for (let i = 0; i < all.length; i += 4) if (hex(all.slice(i, i + 3)) !== corner) other++;
    return {corner, drawn: other / (all.length / 4), width: image.width, height: image.height};
};
// The scene as it is drawn (from the camera's reset position, as a picture the size of the canvas on the page, whatever its
// device pixel ratio) and as it is kept: what Mol*'s focus has put in the tree.
window.sceneNow = async function () {
    plugin.managers.camera.reset();
    await new Promise(resolve => setTimeout(resolve, 1500));
    const helper = plugin.helpers.viewportScreenshot, kept = helper.behaviors.values.value, canvas = document.querySelector('#viewer canvas');
    helper.behaviors.values.next({...kept, resolution: {name: 'custom', params: {width: canvas.clientWidth, height: canvas.clientHeight}}});
    let picture;
    try { picture = await helper.getImageDataUri(); } finally { helper.behaviors.values.next(kept); }
    return {picture, nodes: focusNodes().length,
            focus: !!plugin.managers.structure.focus.current, cells: plugin.state.data.cells.size,
            marked: plugin.managers.structure.selection.stats.elementCount};
};
// How much of a part of the picture (fractions of its width and height) is not its background.
window.regionDrawn = async function (left, top, width, height) {
    const uri = await plugin.helpers.viewportScreenshot.getImageDataUri();
    const image = new Image();
    image.src = uri;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext('2d', {willReadFrequently: true});
    context.drawImage(image, 0, 0);
    const [x, y, w, h] = [left * image.width, top * image.height, width * image.width, height * image.height].map(Math.round);
    const data = context.getImageData(x, y, w, h).data;
    const corner = context.getImageData(2, 2, 1, 1).data;
    let other = 0;
    for (let i = 0; i < data.length; i += 4) {
        if (Math.abs(data[i] - corner[0]) + Math.abs(data[i + 1] - corner[1]) + Math.abs(data[i + 2] - corner[2]) > 12) other++;
    }
    return other / (data.length / 4);
};
// Where the status line is, in the stage's own pixels.
window.toastRect = function () {
    const stage = document.getElementById('stage').getBoundingClientRect();
    const toast = document.getElementById('toast').getBoundingClientRect();
    return {left: toast.left - stage.left, right: toast.right - stage.left, top: toast.top - stage.top,
            bottom: toast.bottom - stage.top, stageWidth: stage.width, stageHeight: stage.height};
};
// Where a residue of a structure is on the page, in page pixels, and the boxes of the canvas's overlays in the stage's.
window.spotOf = function (index, number) {
    const structure = loadedStructures()[index];
    const trace = readCaTrace((structure.transform || structure).cell.obj.data);
    const k = trace.ids.findIndex(([chain, n]) => n === number);
    const rect = document.querySelector('#viewer canvas').getBoundingClientRect();
    const projected = molstar.lib.math.LinearAlgebra.Vec4();
    plugin.canvas3d.camera.project(projected, trace.coords[k]);
    return [rect.left + projected[0], rect.top + projected[1]];
};
window.boxIn = function (selector) {
    const stage = document.getElementById('stage').getBoundingClientRect();
    const box = document.querySelector(selector).getBoundingClientRect();
    return {left: box.left - stage.left, right: box.right - stage.left, top: box.top - stage.top, bottom: box.bottom - stage.top,
            stageWidth: stage.width, stageHeight: stage.height};
};
window.cameraNow = function () {
    const state = plugin.canvas3d.camera.state;
    return {target: Array.from(state.target), position: Array.from(state.position), radius: state.radius};
};
"""


class Scenario(FoldViewPageMixin, FoldViewServerMixin, unittest.TestCase):
    def wait(self, ms):
        self.script(f"await new Promise(resolve => setTimeout(resolve, {ms})); return 1")

    def picture(self, name):
        self.pump(0.4)
        PICTURES.append(name)
        if self.folder:
            self.view.grab().save(str(Path(self.folder) / f"{name}.png"))

    folder = None

    def scene(self):
        """The scene's state and its picture, as an array of pixels, from the camera's reset position."""
        state = self.script("return await sceneNow()")
        data = base64.b64decode(state.pop("picture").split(",", 1)[1])
        image = QImage.fromData(data).convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = numpy.frombuffer(image.constBits(), dtype=numpy.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
        return state, pixels[:, :image.width(), :3].astype(int)

    def saved_image(self, **options):
        """The picture "Save an image" makes with `options` (legend, transparent, scale), as an array of RGBA pixels."""
        uri = self.script("""
            const blob = await buildImage(%s);
            return await new Promise(resolve => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.readAsDataURL(blob); });
            """ % json.dumps({"legend": False, "transparent": False, "scale": 1, **options}))
        image = QImage.fromData(base64.b64decode(uri.split(",", 1)[1])).convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = numpy.frombuffer(image.constBits(), dtype=numpy.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
        return pixels[:, :image.width(), :].astype(int)

    @staticmethod
    def difference(a, b):
        """(pixels that are not the background in one picture, pixels that differ much between the two pictures)."""
        drawn = int((numpy.abs(b - b[2, 2]).sum(axis=2) > 12).sum())
        return drawn, int((numpy.abs(a - b).sum(axis=2) > 90).sum())

    def drawn(self, stage, x, y, width, height):
        """The share of a part of the stage (pixels from its corner) that is not the stage's background."""
        image = self.view.grab().toImage()
        left, top = round(stage["x"] + x), round(stage["y"] + y)
        background = image.pixelColor(round(stage["x"]) + 4, round(stage["y"]) + 4)
        other = total = 0
        for row in range(top, top + height):
            for column in range(left, left + width):
                color = image.pixelColor(column, row)
                total += 1
                if abs(color.red() - background.red()) + abs(color.green() - background.green()) + abs(color.blue() - background.blue()) > 12:
                    other += 1
        return other / total

    def test_scenario(self):
        fixtures.write_structure(self.directory, "11.pdb", fixtures.FIRST_SEQUENCE)
        fixtures.write_structure(self.directory, "22.pdb", fixtures.SECOND_SEQUENCE, shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_page()
        self.js(PIXELS)
        self.script("applyTheme('light', false); return 1")
        FACTS["webgl"] = bool(self.script("return !!plugin.canvas3d"))
        if not FACTS["webgl"]:
            FACTS["view"] = self.script("return document.body.dataset.view")
            return
        FACTS["errors"] = self.console_errors()
        FACTS["viewEmpty"] = self.script("return document.body.dataset.view")
        self.picture("empty-light")
        # The states of a Viewer with no window and of a run in progress, as the Viewer's events make them.
        self.broadcast({"type": "viewer_info", "hidden": True})
        self.wait(300)
        self.picture("no-window-light")
        self.broadcast({"type": "viewer_info", "hidden": False})
        self.broadcast({"type": "esmfold_job", "event": "started", "total": 3, "names": ["node_a", "node_b", "node_c"]})
        self.broadcast({"type": "esmfold_job", "event": "progress", "index": 2, "total": 3, "name": "UniRef50_Q31000"})
        self.wait(500)
        FACTS["foldingView"] = self.script("return document.body.dataset.view")
        self.picture("folding-light")
        self.broadcast({"type": "esmfold_job", "event": "structure_failed", "name": "UniRef50_X31274", "reason": "CUDA ran out of memory."})
        self.broadcast({"type": "esmfold_job", "event": "finished", "succeeded": 1, "total": 3, "failed": True})
        self.wait(500)
        self.picture("failed-light")
        FACTS["failureShown"] = self.script("return document.querySelectorAll('#struct-list .srow.failed').length")
        self.script("document.querySelector('#struct-list .srow.failed button').click(); return 1")
        self.wait(300)
        self.push("11.pdb")
        self.push("22.pdb")
        self.wait(800)
        FACTS["viewReady"] = self.script("return document.body.dataset.view")
        FACTS["lightToken"] = self.script("return canvasBackground()")
        FACTS["light"] = self.script("return await canvasPixels()")
        FACTS["camera0"] = self.script("return cameraNow()")
        # Two pictures of the same scene differ a little (the renderer's own anti-aliasing and occlusion move
        # from frame to frame); what a pin leaves behind has to stand out from that.
        FACTS["sceneBefore"], before = self.scene()
        _, again = self.scene()
        FACTS["noise"] = self.difference(before, again)
        FACTS["drawnBefore"] = self.difference(before, before)[0]
        # The axes gizmo is in the bottom right corner, and the status line is not on top of it.
        FACTS["gizmo"] = self.script("return plugin.canvas3d.props.camera.helper.axes.params.location")
        # The gizmo is part of the live canvas, not of the screenshot helper's picture: look at the window.
        corner = self.script("const r = document.getElementById('stage').getBoundingClientRect(); return {x: r.left, y: r.top, w: r.width, h: r.height}")
        # (Inside the stage's own border, which is not background either.)
        # (The legend on the canvas is in the left corner, by design: look at the gizmo without it.)
        self.script("document.getElementById('canvas-legend').style.visibility = 'hidden'; return 1")
        self.pump(0.3)
        FACTS["gizmoRight"] = self.drawn(corner, corner["w"] - 128, corner["h"] - 128, 120, 120)
        FACTS["gizmoLeft"] = self.drawn(corner, 8, corner["h"] - 128, 120, 120)
        self.script("document.getElementById('canvas-legend').style.visibility = ''; return 1")
        self.script("showStatus('Loaded and aligned 22. A longer message, to see where the status line goes.', 0); return 1")
        self.wait(200)
        FACTS["toast"] = self.script("return toastRect()")
        self.script("hideStatus(); return 1")
        self.picture("structures-light")

        # Colored by a layer: the card, its legend and the compact one on the canvas.
        self.script("document.querySelector('#color-body [data-coloring=\"information_content\"]').click(); await structureQueue; await cardNow(); return 1")
        self.wait(400)
        FACTS["layerLegend"] = self.script("return card().compact")
        self.picture("color-layer-light")
        # Save an image: the picture as the window shows it, with the legend drawn in or left out, bigger, and see-through.
        plain = self.saved_image()
        with_legend = self.saved_image(legend=True)
        double = self.saved_image(legend=True, scale=2)
        see_through = self.saved_image(transparent=True)
        height, width = plain.shape[:2]
        corner = numpy.abs(plain[:, :, :3] - with_legend[:, :, :3]).sum(axis=2) > 90
        where = numpy.zeros(corner.shape, dtype=bool)
        where[int(0.7 * height):, :int(0.4 * width)] = True
        FACTS["image"] = {
            "size": [width, height], "doubleSize": [double.shape[1], double.shape[0]],
            "legendPixels": int((corner & where).sum()), "elsewherePixels": int((corner & ~where).sum()),
            "alpha": [int(plain[2, 2, 3]), int(with_legend[2, 2, 3]), int(see_through[2, 2, 3])],
        }
        if self.folder:
            QImage(with_legend.astype(numpy.uint8).tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888).save(
                str(Path(self.folder) / "saved-image.png"))
        # A real pointer over the real canvas: hover shows the tip, a click pins, Escape lets go.
        widget = self.view.focusProxy() or self.view
        spot = QPoint(*[round(value) for value in self.script("return spotOf(0, 6)")])
        QTest.mouseMove(widget, spot)
        self.pump(1.0)
        FACTS["hover"] = self.script("return hoveredResidue ? [hoveredResidue.number, hoveredResidue.chain] : null")
        FACTS["tip"] = self.script("const tip = document.getElementById('tip'); return {shown: tip.classList.contains('show'), text: tip.textContent, box: boxIn('#tip')}")
        QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, spot)
        self.pump(1.0)
        FACTS["pinned"] = self.script("return pinnedResidue ? pinnedResidue.number : null")
        FACTS["marked"] = self.script("return plugin.managers.structure.selection.stats.elementCount")
        FACTS["residueCard"] = self.script(
            "const body = document.getElementById('residue-body'); return {head: body.querySelector('.res-head div').firstChild.textContent, rows: body.querySelectorAll('tr').length}")
        self.picture("pinned-light")
        QTest.keyClick(widget, Qt.Key_Escape)
        self.pump(0.6)
        FACTS["pinnedAfterEscape"] = self.script("return pinnedResidue ? pinnedResidue.number : null")
        FACTS["markedAfterEscape"] = self.script("return plugin.managers.structure.selection.stats.elementCount")
        QTest.mouseMove(widget, QPoint(100, 600))
        self.pump(1.0)
        QTest.mouseMove(widget, QPoint(40, 130))
        self.pump(1.0)
        self.script("document.querySelector('#color-body [data-coloring=\"plddt\"]').click(); await structureQueue; await cardNow(); return 1")
        self.wait(300)
        FACTS["sceneAfterPin"], pixels = self.scene()
        FACTS["afterPin"] = self.difference(before, pixels)
        FACTS["errorsReady"] = self.console_errors()
        # A session saved with a pin in it, as an earlier page saved them, comes back without it.
        spot = QPoint(*[round(value) for value in self.script("return spotOf(0, 6)")])
        QTest.mouseMove(widget, spot)
        self.pump(1.0)
        QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, spot)
        self.pump(1.5)
        FACTS["pinnedAgain"] = self.script("return {nodes: focusNodes().length, pinned: pinnedResidue ? pinnedResidue.number : null}")
        pinned_session = self.script("return plugin.state.getSnapshot({camera: true})")
        QTest.keyClick(widget, Qt.Key_Escape)
        self.pump(0.6)
        self.wait(1800)
        self.session = pinned_session
        self.open_page()
        self.js(PIXELS)
        self.script("applyTheme('light', false); return 1")
        self.wait(1500)
        FACTS["sceneRestored"], pixels = self.scene()
        FACTS["restored"] = self.difference(before, pixels)
        FACTS["restoredPinned"] = self.script("return pinnedResidue ? pinnedResidue.number : null")
        self.session = None
        widget = self.view.focusProxy() or self.view

        # The row's details, the Session menu and the question it asks, for a person to look at.
        self.script("document.querySelector('#struct-list .srow:nth-child(2) .exp-btn').click(); return 1")
        self.picture("details-light")
        self.script("document.querySelector('#struct-list .srow:nth-child(2) .exp-btn').click(); return 1")
        self.js("document.getElementById('session-btn').click()")
        self.picture("session-menu-light")
        self.script("document.querySelector('#session-menu [data-action=\"clear\"]').click(); return 1")
        self.picture("clear-dialog-light")
        self.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
        self.pump(0.3)

        # Reset view returns the camera to where Mol* frames the scene; Zoom to a structure comes to it.
        self.script("plugin.managers.camera.reset(); return 1")
        self.wait(1500)
        FACTS["cameraFramed"] = self.script("return cameraNow()")
        move = "plugin.canvas3d.camera.setState({target: [40, 40, 40], position: [40, 40, 200]}); return cameraNow()"
        FACTS["cameraMoved"] = self.script(move)
        self.script("document.querySelector('#struct-list .srow:nth-child(2) button[title^=\"Zoom to\"]').click(); return 1")
        self.wait(1500)
        FACTS["cameraZoomed"] = self.script("return cameraNow()")
        FACTS["center"] = self.script("return Array.from(loadedStructures()[1].cell.obj.data.boundary.sphere.center)")
        self.script(move)
        self.script("document.getElementById('t-reset').click(); return 1")
        self.wait(1500)
        FACTS["cameraReset"] = self.script("return cameraNow()")

        # The dark theme: the canvas takes the page's dark canvas colour.
        self.js("document.getElementById('theme-toggle-btn').click()")
        self.wait(500)
        FACTS["darkToken"] = self.script("return canvasBackground()")
        FACTS["dark"] = self.script("return await canvasPixels()")
        FACTS["darkRenderer"] = self.script("return '#' + plugin.canvas3d.props.renderer.backgroundColor.toString(16).padStart(6, '0')")
        self.picture("structures-dark")

        # The Advanced door keeps the same canvas and gives it back.
        FACTS["canvasBefore"] = self.script("window.__canvas = plugin.canvas3d; return 1")
        self.js("document.getElementById('t-adv').click()")
        self.pump(1.5)
        FACTS["advancedSameCanvas"] = self.script("return plugin.canvas3d === window.__canvas")
        FACTS["advancedDrawn"] = self.script("return (await canvasPixels()).drawn")
        self.picture("advanced-dark")
        self.js("document.getElementById('adv-back').click()")
        self.pump(1.0)
        FACTS["closedSameCanvas"] = self.script("return plugin.canvas3d === window.__canvas")
        FACTS["closedDrawn"] = self.script("return (await canvasPixels()).drawn")

        # The panel's width: a real pointer drags the handle on the panel's edge, and the canvas, a real one, takes the room that is
        # left. It is as wide as the stage, drawn again at that size (its buffer, and the camera's viewport) and not stretched: two
        # residues are as many pixels apart as before, which a picture that was only scaled up or down would not leave them.
        def canvas_now(label):
            found = self.script("""
                const canvas = document.querySelector('#viewer canvas');
                const stage = document.getElementById('stage').getBoundingClientRect();
                const a = spotOf(0, 4), b = spotOf(0, 28), viewport = plugin.canvas3d.camera.viewport;
                const picture = await canvasPixels();
                return {panel: document.getElementById('side').getBoundingClientRect().width, client: [canvas.clientWidth, canvas.clientHeight],
                        buffer: [canvas.width, canvas.height], ratio: window.devicePixelRatio, stage: [stage.width, stage.height],
                        viewport: [viewport.width, viewport.height], apart: [b[0] - a[0], b[1] - a[1]], drawn: picture.drawn,
                        shot: [picture.width, picture.height], calls: window.__resizeCalls || 0};""")
            found["label"] = label
            FACTS["panel"].append(found)

        FACTS["panel"] = []
        self.script("window.__resizeCalls = 0; const resize = viewerInstance.handleResize.bind(viewerInstance); viewerInstance.handleResize = () => { window.__resizeCalls++; return resize(); }; return 1")
        self.wait(400)
        canvas_now("default")
        edge = self.script("const r = document.getElementById('side-resizer').getBoundingClientRect(); return [document.getElementById('side').getBoundingClientRect().left, r.top + r.height / 2]")
        start = QPoint(round(edge[0]) + 2, round(edge[1]))
        QTest.mouseMove(widget, start)
        QTest.mousePress(widget, Qt.LeftButton, Qt.NoModifier, start)
        for step in range(1, 6):
            QTest.mouseMove(widget, QPoint(start.x() - 40 * step, start.y()))
            self.pump(0.05)
        self.pump(0.6)
        canvas_now("dragging")
        QTest.mouseRelease(widget, Qt.LeftButton, Qt.NoModifier, QPoint(start.x() - 200, start.y()))
        self.pump(0.5)
        canvas_now("let go")
        key = "document.getElementById('side-resizer').focus(); document.getElementById('side-resizer').dispatchEvent(new KeyboardEvent('keydown', {key: %s, bubbles: true, cancelable: true})); return 1"
        self.script(key % "'End'")
        self.wait(600)
        canvas_now("widest")
        self.view.resize(1100, 800)
        self.wait(800)
        canvas_now("window of 1100 px")
        self.view.resize(1300, 800)
        self.wait(500)
        self.script(key % "'Home'")
        self.wait(600)
        canvas_now("restored")
        self.picture("panel-restored-dark")

        # Sessions saved by an earlier Mol* (tests/esmfold_saved_sessions.py) carry the camera of a page with a canvas:
        # it comes back, by the Viewer's route and from the downloaded file, and the structures are in view.
        FACTS["savedCamera"] = {}
        for name in ("pair", "marked"):
            scene = saved_scene(name)
            write_saved_structures(self.directory)
            for route in ("viewer", "file"):
                if route == "viewer":
                    self.session = json.loads(json.dumps(scene["session"]))
                    self.open_page()
                    self.js(PIXELS)
                else:
                    self.session = None
                    self.open_page()
                    self.js(PIXELS)
                    self.script("await openSessionFile(new File([%s], 'saved.molj')); await structureQueue; return 1" % json.dumps(scene["molj"]))
                self.wait(2000)
                current = scene["session"]["camera"]["current"]
                FACTS["savedCamera"][f"{name}/{route}"] = {
                    "saved": {key: current[key] for key in ("target", "position", "radius")},
                    "restored": self.script("return cameraNow()"), "drawn": self.script("return (await canvasPixels()).drawn"),
                    "structures": self.script("return loadedStructures().length")}
        self.session = None

        # A narrow window, in light, opened narrow as a person's window would be.
        self.view.resize(600, 800)
        self.open_page()
        self.js(PIXELS)
        self.push("11.pdb")
        self.push("22.pdb")
        self.wait(800)
        self.script("applyTheme('light', false); return 1")
        self.picture("narrow-light")
        FACTS["narrowPanel"] = self.script("return document.body.dataset.panel")
        self.script("showStatus('Loaded and aligned 22.', 0); return 1")
        self.wait(200)
        FACTS["narrowToast"] = self.script("return toastRect()")
        FACTS["narrowLegend"] = self.script("return boxIn('#canvas-legend')")
        FACTS["narrowShown"] = self.script("return document.getElementById('canvas-legend').dataset.shown")
        self.script("hideStatus(); return 1")
        # The Color card is a tab in the strip, and nothing of it floats over the canvas.
        self.script("document.getElementById('tab-color').click(); return 1")
        self.picture("narrow-color-tab")
        self.script("document.getElementById('tab-residue').click(); return 1")
        self.script("document.getElementById('tab-structures').click(); return 1")
        self.view.resize(1300, 800)
        self.pump(0.5)

        # Simplified Chinese: the same page again.
        from desktop.Desktop_App import install_translations
        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.open_page()
        self.js(PIXELS)
        self.push("11.pdb")
        self.push("22.pdb")
        self.wait(800)
        self.script("applyTheme('light', false); return 1")
        self.picture("structures-zh")
        FACTS["chineseHeader"] = self.script("return document.querySelector('header h1').textContent")
        FACTS["chineseView"] = self.script("return document.body.dataset.view")


def main(argv):
    Scenario.folder = argv[argv.index("--pictures") + 1] if "--pictures" in argv else None
    if Scenario.folder:
        os.makedirs(Scenario.folder, exist_ok=True)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Scenario))
    if not result.wasSuccessful():
        problems = [text for _, text in result.failures + result.errors]
        print(RESULT_PREFIX + json.dumps({"error": problems[0][-1500:] if problems else "failed"}), flush=True)
        return 1
    print(RESULT_PREFIX + json.dumps({**FACTS, "pictures": PICTURES}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
