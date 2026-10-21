# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's own interface around the Mol* canvas (src/web_ui/esmfold.html and esmfold_*.js).

The page hides Mol*'s interface and draws a header, a structure list, a colour
switch, a tooltip, a sequence strip and the empty and error states of its own,
in the look of the Agent and Meta Data pages. These tests open it in Qt
WebEngine over the real web server and the real bundled Mol* (the harness is
tests/esmfold_page_harness.py), and check the page through its DOM and Mol*'s
state: the offscreen engine has no WebGL, and tests/test_esmfold_shell_webgl.py
holds the checks that need a canvas.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests import esmfold_layer_fixtures as fixtures
from tests.esmfold_page_harness import FoldViewPageMixin, FoldViewServerMixin
from tests.esmfold_session_fixture import session_before_the_redesign, write_session_structures


class PageTestCase(FoldViewPageMixin, FoldViewServerMixin, unittest.TestCase):
    """The page over a folder of fixture structures, with the Viewer's side of its conversation."""

    BANDS = ((90, "#0053d6"), (70, "#65cbf3"), (50, "#ffdb13"), (-1, "#ff7d45"))

    def band(self, value):
        return next(color for floor, color in self.BANDS if value > floor)


class SavedSessionTests(PageTestCase):
    """A session the page saved before it was redesigned keeps loading."""

    def test_a_session_saved_before_the_redesign_loads_as_it_was(self):
        saved = session_before_the_redesign()
        pose = next(node["params"]["transform"]["params"]["data"] for node in saved["data"]["tree"]["transforms"]
                    if node["transformer"] == "ms-plugin.transform-structure-conformation")
        write_session_structures(self.directory)
        self.session = saved
        self.open_page()
        restored = self.script("""
            return {n: loadedStructures().length, themes: reprThemes().map(theme => theme.name),
                    active: loadedStructures().map(activeLayerId), files: loadedStructures().map(structureFile),
                    fit: fit(1), colors: [colors(0, 'plddt-confidence', {}), colors(1, 'plddt-confidence', {})]};
            """)
        self.assertEqual(restored["n"], 2)
        self.assertEqual(restored["files"], ["protein_a.pdb", "protein_b.pdb"])
        self.assertEqual(restored["themes"], ["plddt-confidence", "plddt-confidence"])
        self.assertEqual(restored["active"], ["plddt", "plddt"])
        # The second structure keeps the pose that was saved, and is on the first.
        self.assertTrue(restored["fit"]["transformed"])
        self.assertEqual(restored["fit"]["matrix"], pose)
        self.assertLess(restored["fit"]["rmsd"], 0.6)
        for index, offset in ((0, 0), (1, 17)):
            plddt = fixtures.plddt_values(33, offset)
            for number in range(1, 34):
                self.assertEqual(restored["colors"][index][str(number)], self.band(plddt[number - 1]), (index, number))


if __name__ == "__main__":
    unittest.main()
