# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Makes tests/esmfold_saved_session_data.py: the scenes of tests/esmfold_saved_sessions.py, saved by the page.

It opens the Fold View page in Qt WebEngine over the real web server and the
Mol* the checkout vendors (the harness of tests/esmfold_page_harness.py), over
the two fixture structures of tests/esmfold_session_fixture.py, does what each
scene says through the page's own functions (a folded structure arriving, the
Color by card, a row's coloring, representation and opacity, the Measure and
Labels tools, a click that pins a residue), and keeps what the page then gives
away: the session it posts to the Viewer, and the file its Session menu
downloads. It then restores each scene in a fresh page, by both routes, and
keeps what the page shows as the scene's ``expected`` summary.

    python tests/esmfold_session_maker.py --write tests/esmfold_saved_session_data.py [--no-webgl]

The sessions are frozen once made (they carry random node references and a
time stamp, so a second run gives other bytes), and the test that keeps them
(tests/test_esmfold_saved_sessions.py) is the reason to keep the old ones when
Mol* moves on: run this only to add a scene, or to start a new generation.
With WebGL (the default, where the machine gives the child a GPU), the
sessions carry the camera, as a person's would.

Not collected by unittest.
"""

import json
import os
import sys
import unittest
from pathlib import Path

GL_FLAGS = "--ignore-gpu-blocklist --disable-gpu-compositing"
NO_GL_FLAGS = "--disable-gpu"
# Chromium reads its flags once, so they are set before the harness starts Qt WebEngine.
if __name__ == "__main__":
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = NO_GL_FLAGS if "--no-webgl" in sys.argv else GL_FLAGS
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests import esmfold_saved_sessions as saved  # noqa: E402
from tests.esmfold_page_harness import FoldViewPageMixin, FoldViewServerMixin  # noqa: E402
from tests.test_esmfold_residue import POINTER  # noqa: E402

RESULT_PREFIX = "MAKER "
SCENES = {}
FACTS = {}

HEADER = '''# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The scenes of tests/esmfold_saved_sessions.py, as the Fold View page saved them with Mol* {version}.

Each scene is the JSON {{"session", "molj", "expected"}} of tests/esmfold_saved_sessions.py,
deflated and written in base64. Made by tests/esmfold_session_maker.py; frozen.
"""

MOLSTAR = "{version}"
PACKED = {{
'''


class Maker(FoldViewPageMixin, FoldViewServerMixin, unittest.TestCase):
    def settle(self, milliseconds):
        self.script(f"await new Promise(resolve => setTimeout(resolve, {milliseconds})); return 1")

    def start(self, *push):
        """The page open over the fixture structures, with the structures in `push` folded into it."""
        saved.write_saved_structures(self.directory)
        self.open_page()
        self.js(POINTER)
        self.js(saved.SUMMARY_JS)
        FACTS["molstar"] = self.script("return molstar.version")
        FACTS["webgl"] = bool(self.script("return !!plugin.canvas3d"))
        for name in push:
            self.push(name)
        self.settle(300)

    def capture(self, name):
        """Keep the session the page posts and the file its Session menu downloads, as they are now."""
        del self.saved[:]
        self.script("await saveSession(); return 1")
        deadline = 100
        while not self.saved and deadline:
            self.pump(0.1)
            deadline -= 1
        self.assertTrue(self.saved, "the page posted no session")
        session = self.saved[-1]
        self.script("""
            const make = URL.createObjectURL;
            URL.createObjectURL = blob => { window.__blob = blob; return make.call(URL, blob); };
            HTMLAnchorElement.prototype.click = function () { window.__download = this.download; };
            await downloadSession();
            return 1;
            """)
        self.settle(300)
        molj = self.script("return await window.__blob.text()")
        self.assertEqual(self.script("return window.__download"), "fold-view-session.molj")
        SCENES[name] = {"session": session, "molj": molj}

    def test_01_pair(self):
        self.start(saved.FIRST, saved.SECOND)
        self.script("""
            document.querySelector('#color-body [data-coloring="information_content"]').click();
            await structureQueue;
            await setStructureColoring(loadedStructures()[1].cell.transform.ref, 'same_residue_pct');
            return 1;
            """)
        self.settle(500)
        live = self.script("return (await sessionSummary()).structures.map(s => [s.coloring, s.themes])")
        self.assertEqual(live, [[ "information_content", ["mvs-annotation"]], ["same_residue_pct", ["mvs-annotation"]]])
        self.capture("pair")

    def test_02_styled(self):
        self.start(saved.FIRST, saved.SECOND)
        self.script("""
            const [first, second] = loadedStructures().map(structure => structure.cell.transform.ref);
            await runExclusive(() => setRepresentation(first, 'ball-and-stick'));
            await runExclusive(() => setRepresentation(second, 'gaussian-surface'));
            await runExclusive(() => setOpacity(first, 60));
            await runExclusive(() => setOpacity(second, 45));
            return 1;
            """)
        self.settle(500)
        live = self.script("return (await sessionSummary()).structures.map(s => [s.representation, s.opacity])")
        self.assertEqual(live, [["ball-and-stick", 60], ["gaussian-surface", 45]])
        self.capture("styled")

    def test_03_opened(self):
        self.start(saved.FIRST)
        self.script("openStructureFiles([new File([%s], %s)]); await structureQueue; return 1" % (
            json.dumps(saved.opened_structure_pdb()), json.dumps(saved.OPENED_NAME)))
        self.settle(800)
        live = self.script("return (await sessionSummary()).structures.map(s => [s.file, s.folded, s.coloring])")
        self.assertEqual(live, [[saved.FIRST, True, "plddt"], [None, False, "builtin:structure"]])
        self.capture("opened")

    def test_04_marked(self):
        self.start(saved.FIRST, saved.SECOND)
        self.script("""
            setTool('measure');
            await clickOn(0, 5);
            await clickOn(1, 12);
            setTool('label');
            await clickOn(0, 20);
            leaveTool();
            await new Promise(resolve => setTimeout(resolve, 600));
            return 1;
            """)
        measured = self.script("return (await sessionSummary()).measurements")
        self.assertEqual(measured, {"distances": 1, "labels": 1})
        self.script("await clickOn(0, 9); return 1")
        self.settle(500)
        pinned = self.script("return (await sessionSummary()).focus")
        self.assertEqual((pinned["nodes"], pinned["current"], pinned["pinned"]), (5, True, 10))
        self.capture("marked")
        session = SCENES["marked"]["session"]
        tags = [tag for node in session["data"]["tree"]["transforms"] for tag in node.get("tags", [])]
        self.assertEqual([tag for tag in tags if tag.startswith("structure-focus-")], [])
        self.assertEqual(session["structureFocus"], {})
        # The file is Mol*'s own: it has what the pin put in the tree.
        molj = json.loads(SCENES["marked"]["molj"])
        raw = [tag for node in molj["entries"][0]["snapshot"]["data"]["tree"]["transforms"] for tag in node.get("tags", [])]
        self.assertEqual(len([tag for tag in raw if tag.startswith("structure-focus-")]), 5)

    def test_05_restore_each_scene_by_both_routes(self):
        """What the page shows of each scene, restored by the Viewer's route and from the file, is its `expected`."""
        for name in saved.SCENES:
            scene = SCENES[name]
            with self.subTest(scene=name, route="session"):
                saved.write_saved_structures(self.directory)
                self.session = json.loads(json.dumps(scene["session"]))
                self.open_page()
                self.js(POINTER)
                self.js(saved.SUMMARY_JS)
                self.settle(1200)
                scene["expected"] = self.script("return await sessionSummary()")
                self.assertEqual(self.script("return isRestoring"), False)
            with self.subTest(scene=name, route="file"):
                self.session = None
                self.open_page()
                self.js(POINTER)
                self.js(saved.SUMMARY_JS)
                self.script("await openSessionFile(new File([%s], 'saved.molj')); await structureQueue; return 1" % json.dumps(scene["molj"]))
                self.settle(1500)
                from_file = self.script("return await sessionSummary()")
                self.assertEqual(from_file, scene["expected"], name)
        # (The stream of each page that was left reports its end.)
        self.assertEqual([error for error in self.console_errors() if "SSE Connection Error" not in error[1]], [],
                         "the page wrote errors to the console")


def write_module(path):
    chunks = [HEADER.format(version=FACTS["molstar"])]
    for name in saved.SCENES:
        chunks.append(f'    "{name}": (\n')
        chunks.extend(f'        "{line}"\n' for line in saved.pack(SCENES[name]))
        chunks.append("    ),\n")
    chunks.append("}\n")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(chunks))


def main(argv):
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Maker))
    if not result.wasSuccessful():
        problems = [text for _, text in result.failures + result.errors]
        print(RESULT_PREFIX + json.dumps({"error": problems[0][-1500:] if problems else "failed"}), flush=True)
        return 1
    if "--write" in argv:
        write_module(argv[argv.index("--write") + 1])
    print(RESULT_PREFIX + json.dumps({"facts": FACTS, "scenes": {name: {key: (len(value) if isinstance(value, str) else None) for key, value in scene.items() if key == "molj"} for name, scene in SCENES.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
