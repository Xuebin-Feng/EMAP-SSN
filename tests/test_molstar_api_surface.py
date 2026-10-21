# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Mol* surface the Fold View page is built on (tests/molstar_api_probe.py).

The page hides Mol*'s own interface and draws its own controls from Mol*'s
managers, so a Mol* that moves one of those calls breaks the page quietly.
These tests pin what the vendored bundle (src/resources/esmfold/molstar.js)
offers, in a browser run by a child process; they are also the gate for taking
a newer Mol* in: the bundle is accepted when they pass, together with
tests/test_esmfold_layers.py, which holds the layers' dependence on Mol*'s
MolViewSpec property and colour theme names, and the saved-session test of
tests/test_esmfold_shell.py. The canvas checks need WebGL and are skipped
where the machine gives the child none.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.molstar_api_probe import RESULT_PREFIX


def run_probe(extra=()):
    environment = {key: value for key, value in os.environ.items() if key != "QTWEBENGINE_CHROMIUM_FLAGS"}
    done = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "molstar_api_probe.py"), *extra],
        capture_output=True, text=True, timeout=240, env=environment, cwd=str(ROOT),
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        raise AssertionError(f"the probe printed no answer (exit {done.returncode}): {done.stderr[-800:]}")
    result = json.loads(lines[-1][len(RESULT_PREFIX):])
    if "error" in result:
        raise AssertionError(f"the probe failed: {result['error']}")
    return result


class MolstarApiSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = run_probe()

    def test_the_bundle_is_the_pinned_version_with_the_extensions_the_page_uses(self):
        self.assertEqual(self.found["version"], "5.10.1")
        # mvs colours the layers, ma-quality-assessment is where the pLDDT theme comes from.
        self.assertEqual(self.found["extensions"], ["mvs", "ma-quality-assessment", "backgrounds"])
        self.assertEqual(self.found["libPlugin"], ["PluginContext", "DefaultPluginSpec", "PluginConfig", "StateTransforms"])
        self.assertEqual(self.found["libStructure"], [
            "StructureElement", "StructureProperties", "Unit", "Queries", "QueryContext", "StructureSelection"])
        self.assertEqual(self.found["libMath"], ["LinearAlgebra"])
        self.assertEqual(self.found["viewerMethods"], ["loadStructureFromData", "loadStructureFromUrl", "handleResize"])

    def test_the_viewer_draws_no_interface_of_its_own_but_for_two_buttons_the_page_hides(self):
        self.assertEqual(self.found["hiddenRegions"], {"left": 0, "right": 0, "top": 0, "bottom": 0})
        # The illumination and VR buttons stay (the page's style hides them); the hover label and the
        # drop target stay in the DOM, so the page can read the first and files can still be dropped.
        self.assertEqual(self.found["hiddenButtons"], 2)
        self.assertEqual(self.found["leftoverClasses"], [
            "msp-viewport-controls", "msp-highlight-toast-wrapper", "msp-logo", "msp-drag-drop-overlay"])
        self.assertEqual(self.found["structures"], 1)

    def test_mol_stars_own_panels_come_back_at_run_time_through_updateProps_only(self):
        self.assertFalse(self.found["setPropsDraws"], "setProps began to draw: updateProps may be redundant now")
        self.assertEqual(self.found["doorRegions"], {"left": 1, "right": 1, "top": 1, "bottom": 1})
        self.assertGreater(self.found["doorButtons"], 40)
        self.assertEqual(self.found["closedAgain"], {"left": 0, "right": 0, "top": 0, "bottom": 0})

    def test_a_hovered_residue_reaches_a_subscriber_as_the_html_the_tooltip_reads(self):
        labels = self.found["labels"]
        self.assertEqual(len(labels), 1)
        self.assertIn("<b>A</b>", labels[0])
        self.assertIn("<b>TYR 5</b>", labels[0])
        self.assertIn("<small>Model 1</small>", labels[0])

    def test_the_managers_and_calls_the_shell_drives_are_there(self):
        managers = self.found["managers"]
        self.assertTrue(managers["hierarchy"])
        self.assertEqual(managers["component"], [
            "addRepresentation", "updateRepresentations", "updateRepresentationsTheme", "updateReprParams",
            "applyPreset", "toggleVisibility", "removeRepresentations"])
        self.assertEqual(managers["measurement"], ["addDistance", "addAngle", "addDihedral", "addLabel"])
        self.assertEqual(managers["selection"], ["fromLoci", "add", "remove", "set", "clear"])
        self.assertEqual(managers["focus"], ["set", "setFromLoci", "addFromLoci", "clear"])
        self.assertEqual(managers["camera"], ["focusLoci", "reset"])
        self.assertEqual(managers["snapshot"], ["add", "serialize", "open"])
        self.assertTrue(managers["interactivity"])
        self.assertEqual(managers["lociLabels"], ["addProvider", "getLabels"])
        self.assertEqual(managers["screenshot"], ["getImageDataUri", "download", "copyToClipboard"])
        self.assertEqual(managers["layout"], ["updateProps", "setProps"])
        self.assertEqual(managers["state"], ["getSnapshot", "setSnapshot"])
        self.assertEqual(self.found["removeStructure"], ["remove"])

    def test_the_representations_themes_and_transforms_the_page_names_exist(self):
        self.assertEqual(self.found["representations"], [
            "cartoon", "backbone", "ball-and-stick", "spacefill", "gaussian-surface", "molecular-surface"])
        self.assertEqual(self.found["themes"], [
            "plddt-confidence", "mvs-annotation", "uniform", "secondary-structure", "sequence-id", "chain-id",
            "hydrophobicity"])
        self.assertEqual(self.found["transforms"], [
            "TransformStructureConformation", "CustomModelProperties", "CustomStructureProperties"])

    def test_a_session_has_the_keys_the_page_saves_and_restores(self):
        self.assertEqual(self.found["snapshotKeys"], [
            "id", "data", "behaviour", "animation", "startAnimation", "camera", "canvas3dContext", "canvas3d",
            "interactivity", "structureFocus", "structureSelection", "structureComponentManager", "durationInMs"])

    def test_the_plddt_theme_comes_from_an_extension_a_plugin_must_ask_for(self):
        # Restoring a session into a plugin without it silently falls back to chain colours.
        self.assertFalse(self.found["plddtWithoutExtension"])
        self.assertTrue(self.found["plddtWithExtension"])

    def test_the_canvas_makes_a_picture_and_projects_points(self):
        if not self.found["webgl"]:
            self.skipTest("this machine gave the child process no WebGL")
        self.assertTrue(self.found["screenshot"])
        self.assertTrue(self.found["project"])
        self.assertTrue(self.found["transparentBackground"])

    def test_the_probe_still_answers_with_no_gpu_flags_where_the_suite_has_none(self):
        found = run_probe(["--no-webgl"])
        self.assertFalse(found["webgl"])
        self.assertEqual(found["hiddenRegions"], self.found["hiddenRegions"])
        self.assertEqual(found["doorRegions"], self.found["doorRegions"])
        self.assertEqual(found["labels"], self.found["labels"])
        self.assertIsNone(found["screenshot"])


if __name__ == "__main__":
    unittest.main()
