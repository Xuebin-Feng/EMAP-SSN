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

import json
import os
import sys
import unittest
from pathlib import Path

GL_FLAGS = "--ignore-gpu-blocklist --disable-gpu-compositing"
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = GL_FLAGS
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

RESULT_PREFIX = "SCENARIO "
FACTS = {}
PICTURES = []

from tests import esmfold_layer_fixtures as fixtures  # noqa: E402
from tests.esmfold_page_harness import FoldViewPageMixin, FoldViewServerMixin  # noqa: E402

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
        self.push("11.pdb")
        self.push("22.pdb")
        self.wait(800)
        FACTS["viewReady"] = self.script("return document.body.dataset.view")
        FACTS["lightToken"] = self.script("return canvasBackground()")
        FACTS["light"] = self.script("return await canvasPixels()")
        FACTS["camera0"] = self.script("return cameraNow()")
        self.picture("structures-light")

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
        FACTS["errorsReady"] = self.console_errors()

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
