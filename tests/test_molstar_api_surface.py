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
    # The vendored bundle's version, which THIRD_PARTY_LICENSES.md and src/resources/esmfold/ name too.
    VERSION = "5.13.1"

    @classmethod
    def setUpClass(cls):
        cls.found = run_probe()

    def test_the_bundle_is_the_pinned_version_with_the_extensions_the_page_uses(self):
        self.assertEqual(self.found["version"], self.VERSION)
        # mvs colours the layers, ma-quality-assessment is where the pLDDT theme comes from.
        self.assertEqual(self.found["extensions"], ["mvs", "ma-quality-assessment", "backgrounds"])
        self.assertEqual(self.found["libPlugin"], ["PluginContext", "DefaultPluginSpec", "PluginConfig", "StateTransforms"])
        self.assertEqual(self.found["libStructure"], [
            "StructureElement", "StructureProperties", "Unit", "Queries", "QueryContext", "StructureSelection"])
        self.assertEqual(self.found["libMath"], ["LinearAlgebra"])
        self.assertEqual(self.found["viewerMethods"], ["loadStructureFromData", "loadStructureFromUrl", "handleResize"])

    def test_the_global_still_has_no_superposition_of_mol_stars_for_the_page_to_use(self):
        # esmfold_superposition.js fits structures itself because Mol*'s alignAndSuperpose and tmAlign are not on the global
        # If one appears here, the page's own fit could be compared with it, and its comment in esmfold_superposition.js is stale.
        self.assertEqual(self.found["superposition"], [])

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

    def test_the_pointer_events_are_subjects_the_page_both_hears_and_feeds_and_a_loci_names_its_residue(self):
        found = self.found
        self.assertEqual(found["subjects"], {
            "hover": True, "click": True, "measurement": True, "hierarchySelection": True, "cellStateUpdated": True,
            "log": ["info", "warn", "error"]})
        # What is pushed into the hover subject reaches a subscriber as it is (the page's tests, and its own tools, push events).
        self.assertTrue(found["hoverDelivered"])
        residue = found["residue"]
        self.assertTrue(residue["structureFound"])
        self.assertEqual((residue["chain"], residue["number"], residue["name"], residue["bfactor"]), ("A", 5, "TYR", 87))
        self.assertIn(residue["secondary"], (None, "Helix", "Strand", "Turn"))
        self.assertEqual(found["secondaryFlags"], {"flags": ["Helix", "Beta", "Turn"], "is": "function"})

    def test_a_group_of_the_representations_render_object_is_a_residue_and_a_click_focuses_it_with_tagged_nodes(self):
        found = self.found
        # The tests, and a real pointer's pick, turn a group of render object 0 into the loci of residue group + 1.
        self.assertEqual(found["pick"], {"renderObjects": True, "residue": 6, "structureLoci": True})
        # A click puts the residue in focus with five nodes of its own (two selections, the target, the surroundings and their
        # contacts), tagged structure-focus-*: esmfold_residue.js finds them by that tag, and a pinned residue lets go of them.
        click = found["focusClick"]
        self.assertEqual(click["nodes"], 5)
        self.assertTrue(click["focus"])
        self.assertEqual(click["roots"], 2)
        self.assertEqual(click["grown"], 5)
        self.assertEqual(click["tags"], [
            "structure-focus-surr-nci-repr", "structure-focus-surr-repr", "structure-focus-surr-sel",
            "structure-focus-target-repr", "structure-focus-target-sel"])
        # The session keeps them and the focus (the page leaves both out of what it saves), and deleting the two roots takes all five.
        self.assertEqual(found["focusInSnapshot"], {"tags": 5, "focusKeys": ["current"]})
        self.assertEqual(found["focusReleased"], {"nodes": 0, "focus": False, "back": True})

    def test_the_measurement_manager_draws_a_distance_and_a_label_in_a_group_that_a_delete_clears(self):
        found = self.found
        self.assertEqual(found["measurementBefore"]["stateKeys"], [
            "angles", "dihedrals", "distances", "labels", "options", "orientations", "planes"])
        self.assertEqual(found["measurement"], {
            "distances": 1, "labels": 1, "groupKeys": ["ref", "root", "state"], "groupRef": "string", "groupState": "object",
            "heard": True})
        self.assertEqual(found["measurementCleared"]["distances"], 0)
        self.assertEqual(found["measurementCleared"]["labels"], 0)

    def test_the_colour_legends_the_card_draws_and_the_layers_properties_are_there_under_their_names(self):
        found = self.found
        self.assertEqual(found["legends"]["secondary"]["kind"], "table-legend")
        self.assertTrue(set(["alphaHelix", "betaStrand", "coil"]) <= set(found["legends"]["secondary"]["names"]))
        self.assertTrue(found["legends"]["secondary"]["colors"])
        self.assertEqual(found["legends"]["position"], {"kind": "scale-legend", "colors": True})
        # The layers' MolViewSpec property names are not documented: these are what tests/test_esmfold_layers.py colours through.
        self.assertEqual(found["annotationTheme"], ["annotationId", "fieldName", "palette"])
        self.assertEqual(found["mvsProperties"], {
            "model": ["mvs-is-mvs-model", "mvs-annotations"], "structure": ["mvs-annotation-tooltips"]})

    def test_a_representation_is_added_with_its_defaults_its_colours_changed_and_removed(self):
        found = self.found
        self.assertEqual(found["addRepresentation"], {
            "added": 1, "type": "ball-and-stick", "alpha": 0.5, "theme": "uniform", "value": 0xff0000})
        self.assertEqual(found["themeUpdate"], [["uniform", 0x00ff00]] * 2)
        self.assertTrue(found["representationRemoved"])

    def test_a_session_file_and_a_snapshot_each_bring_the_scene_back(self):
        found = self.found
        file = found["serialize"]
        self.assertTrue(file["blob"])
        self.assertEqual((file["keys"], file["entries"], file["entryKeys"]), (["current", "entries", "playback", "timestamp", "version"], 1, ["snapshot", "timestamp"]))
        self.assertEqual(file["version"], self.VERSION)
        for key in ("id", "data", "animation", "camera", "canvas3d", "interactivity", "structureFocus", "structureComponentManager"):
            self.assertIn(key, file["snapshotKeys"])
        self.assertEqual(found["sessionRoundTrip"], {"cleared": 0, "opened": 1, "setSnapshot": 1})

    def test_mol_stars_own_remove_takes_the_structure_and_leaves_the_nodes_it_came_from(self):
        removed = self.found["hierarchyRemove"]
        # esmfold_structures.js deletes the top of the chain itself (structureDataRoot); if Mol* ever took the whole chain
        # this would change, and the page's own deletion could be simplified, not before.
        self.assertEqual(removed["structures"], 0)
        self.assertEqual(removed["left"], {
            "build-in.root": 1, "raw-data": 1, "trajectory-from-pdb": 1, "model-from-trajectory": 1, "custom-model-properties": 1})
        self.assertNotIn("structure-from-model", removed["left"])
        self.assertEqual(removed["afterRootDelete"], 1)

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
        self.assertEqual(self.found["cameraState"], ["position", "target", "up", "radius"])

    def test_the_picture_the_page_saves_is_transparent_or_not_without_axes_at_the_size_asked(self):
        if not self.found["webgl"]:
            self.skipTest("this machine gave the child process no WebGL")
        pictures = self.found["pictures"]
        # esmfold_image.js reads and puts back these values of the screenshot helper, and gives it a resolution of its own.
        self.assertEqual(pictures["values"], ["axes", "format", "illumination", "resolution", "transparent"])
        self.assertTrue(pictures["transparentIsBoolean"])
        self.assertEqual((pictures["axesName"], pictures["resolutionName"]), ("off", "viewport"))
        self.assertTrue(pictures["viewportIsCanvas"])
        self.assertTrue(pictures["customSize"])
        self.assertEqual((pictures["opaqueAlpha"], pictures["transparentAlpha"]), (255, 0))

    def test_the_page_sets_the_canvas_background_and_puts_the_axes_gizmo_in_a_corner(self):
        if not self.found["webgl"]:
            self.skipTest("this machine gave the child process no WebGL")
        self.assertEqual(self.found["canvasProps"], {"background": True, "axes": "on", "location": "bottom-right"})

    def test_the_probe_still_answers_with_no_gpu_flags_where_the_suite_has_none(self):
        found = run_probe(["--no-webgl"])
        self.assertFalse(found["webgl"])
        self.assertEqual(found["hiddenRegions"], self.found["hiddenRegions"])
        self.assertEqual(found["doorRegions"], self.found["doorRegions"])
        self.assertEqual(found["labels"], self.found["labels"])
        self.assertIsNone(found["screenshot"])


if __name__ == "__main__":
    unittest.main()
