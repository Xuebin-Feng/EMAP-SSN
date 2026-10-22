# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Sessions the Fold View page saved before (tests/esmfold_saved_sessions.py) restore under the vendored Mol*.

The page saves its session to the Viewer after every change and takes it back
when it opens; the Session menu hands the same session over as a .molj file and
reads one back. Moving the page to a newer Mol* must not strand either kind of
file, so the scenes saved with Mol* 5.10.1 (two folded structures with their
superposition, a layer and a row's own layer, a changed representation and
opacity, a structure opened from a file, a measured distance, a label and a
pinned residue) are restored here by both routes: the Viewer's, as the page
asks for it when it opens, and the file's, through the Session menu's function.
Each is read through the page's own functions and Mol*'s state and compared
with what the page that saved it showed, and with what the files themselves
say (the pose that was stored, the rows the Viewer serves for the layers, the
file the session carries). The page of tests/test_esmfold_shell.py's
SavedSessionTests is the older session of the same promise.
"""

import json
import os
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
from tests.esmfold_saved_session_data import MOLSTAR
from tests.esmfold_saved_sessions import (
    FIRST, OPENED_NAME, SCENES, SECOND, SUMMARY_JS, opened_structure_pdb, saved_scene, write_saved_structures)
from tests.test_esmfold_residue import POINTER
from web_ui.residue_layers import layer_annotation, read_residue_data

ROUTES = ("viewer", "file")
BANDS = ((90, "#0053d6"), (70, "#65cbf3"), (50, "#ffdb13"), (-1, "#ff7d45"))


def band(value):
    return next(color for floor, color in BANDS if value > floor)


def pose_in(session):
    """The matrix the session stored for the structure that was fitted onto the first, from its own tree."""
    poses = [node["params"]["transform"]["params"]["data"] for node in session["data"]["tree"]["transforms"]
             if node["transformer"] == "ms-plugin.transform-structure-conformation"]
    assert len(poses) == 1, poses
    return poses[0]


def tags_in(session):
    return [tag for node in session["data"]["tree"]["transforms"] for tag in node.get("tags", [])]


# What the page does when it is asked, from Mol*'s own event, at the point where an update that replaces the scene has
# taken the old nodes out of the state and has not ended (window.midReplace, set once): the questions a timer or a pointer
# event of the page asks there, and what it left of the list.
MID_REPLACE_JS = """
window.midReplace = null;
{
    const state = plugin.state.data;
    const old = listedStructures().map(structure => structure.cell);
    const watch = state.events.cell.stateUpdated.subscribe(() => {
        if (window.midReplace || !state.behaviors.isUpdating.value) return;
        if (!old.length || old.some(cell => state.cells.get(cell.transform.ref) === cell)) return;
        const errors = [];
        const attempt = (what, run) => { try { return run(); } catch (error) { errors.push(what + ': ' + error.message); return null; } };
        const list = byId('struct-list');
        const html = list.innerHTML;
        const listed = listedStructures();
        const out = {
            listed: listed.length, loaded: loadedStructures().length, replacing: sceneBeingReplaced(),
            names: listed.map(structure => attempt('structureName', () => structureName(structure))),
            roots: listed.map(structure => attempt('structureDataRoot', () => structureDataRoot(structure) ?? null)),
            byRef: old.map(cell => attempt('structureByRef', () => !!structureByRef(cell.transform.ref)))
        };
        attempt('renderStructures', () => renderStructures());
        out.untouched = list.innerHTML === html;
        out.pending = renderRowsAfterUpdate;
        // The old rows are on the page for a moment still: their buttons do nothing, and say nothing.
        const said = byId('live').textContent;
        const camera = plugin.managers.camera;
        const focusLoci = camera.focusLoci;
        let zoomed = 0;
        camera.focusLoci = () => { zoomed++; };
        const buttons = [':eye', ':zoom', ':remove'].map(suffix => list.querySelector('[data-focus-key$="' + suffix + '"]'));
        out.buttons = buttons.filter(Boolean).length;
        for (const button of buttons) if (button) attempt('press ' + button.dataset.focusKey, () => button.click());
        camera.focusLoci = focusLoci;
        out.silent = byId('live').textContent === said && zoomed === 0;
        out.errors = errors;
        window.midReplace = out;
        watch.unsubscribe();
    });
}
return 1;
"""


class SavedSceneTestCase(FoldViewPageMixin, FoldViewServerMixin, unittest.TestCase):
    """The page over the two fixture structures the scenes name, with the Viewer's side of its conversation."""

    def settle(self, milliseconds):
        self.script(f"await new Promise(resolve => setTimeout(resolve, {milliseconds})); return 1")

    def prepare(self):
        write_saved_structures(self.directory)
        self.open_page()
        self.js(POINTER)
        self.js(SUMMARY_JS)

    def restore(self, name, route):
        """The scene `name` restored through `route` ("viewer": the page's own load; "file": the Session menu's Open), and what it shows."""
        scene = saved_scene(name)
        write_saved_structures(self.directory)
        if route == "viewer":
            self.session = json.loads(json.dumps(scene["session"]))
            self.open_page()
            self.js(POINTER)
            self.js(SUMMARY_JS)
            self.assertFalse(self.script("return isRestoring"))
        else:
            self.session = None
            self.prepare()
            self.script("await openSessionFile(new File([%s], 'saved.molj')); await structureQueue; return 1" % json.dumps(scene["molj"]))
        self.settle(1200)
        return self.script("return await sessionSummary()")

    def rows(self, file, layer):
        """What the Viewer says each residue's color is for a layer: the rows of its annotation, by residue number."""
        name = file[:-len(".pdb")]
        data = read_residue_data(file, os.path.join(self.directory, file), os.path.join(self.directory, name + ".residues.json"))
        return {str(row["auth_seq_id"]): row["color"] for row in layer_annotation(data, layer)}

    def errors(self):
        # (The stream of each page that was left reports its end.)
        return [error for error in self.console_errors() if "SSE Connection Error" not in error[1]]


class PageReadsTheHierarchyThroughOneDoorTests(unittest.TestCase):
    """The race that ReplacedSceneTests puts the page in was a read of Mol*'s hierarchy, and a walk up a node's parents, taking the hierarchy for the scene.

    esmfold_layers.js answers for both (loadedStructures and nodesAbove); a script that reaches past them
    brings the race back for whatever structure it asks about, so none does.
    """

    def test_only_the_layers_script_reads_the_hierarchy_and_walks_a_nodes_parents(self):
        readers = {}
        for script in sorted((SRC / "web_ui").glob("esmfold_*.js")):
            text = script.read_text(encoding="utf-8")
            for door in ("hierarchy.current.structures", "transform.parent"):
                if door in text:
                    readers.setdefault(door, []).append(script.name)
        self.assertEqual(readers, {"hierarchy.current.structures": ["esmfold_layers.js"], "transform.parent": ["esmfold_layers.js"]})


class SceneFileTests(unittest.TestCase):
    """The files themselves: what each scene's session and .molj hold, as the page that made them wrote them."""

    def test_the_scenes_were_saved_by_the_mol_star_of_the_redesign_with_a_camera(self):
        self.assertEqual(MOLSTAR, "5.10.1")
        for name in SCENES:
            with self.subTest(scene=name):
                scene = saved_scene(name)
                self.assertEqual(sorted(scene), ["expected", "molj", "session"])
                session, molj = scene["session"], json.loads(scene["molj"])
                self.assertEqual(molj["version"], "5.10.1")
                self.assertEqual(len(molj["entries"]), 1)
                self.assertEqual(sorted(molj["entries"][0]["snapshot"]), sorted(session))
                self.assertEqual(sorted(session), [
                    "animation", "camera", "canvas3d", "canvas3dContext", "data", "durationInMs", "id", "interactivity",
                    "structureComponentManager", "structureFocus"])
                # The camera is what a page with a canvas saves (the earlier fixture's page had none).
                self.assertEqual(session["camera"]["current"]["mode"], "perspective")
                self.assertEqual(len(session["camera"]["current"]["position"]), 3)

    def test_the_session_the_page_posts_has_no_focus_and_the_file_it_downloads_keeps_what_a_pin_put_there(self):
        for name in SCENES:
            with self.subTest(scene=name):
                scene = saved_scene(name)
                self.assertEqual(scene["session"]["structureFocus"], {})
                self.assertEqual([tag for tag in tags_in(scene["session"]) if tag.startswith("structure-focus-")], [])
        marked = json.loads(saved_scene("marked")["molj"])["entries"][0]["snapshot"]
        # Mol*'s own click puts five nodes in the tree and the residue in focus; the Session menu hands the session over as Mol* has it.
        self.assertEqual(len([tag for tag in tags_in(marked) if tag.startswith("structure-focus-")]), 5)
        self.assertIn("current", marked["structureFocus"])

    def test_each_scene_holds_what_it_is_named_for(self):
        pair = saved_scene("pair")["session"]
        kinds = sorted(node["transformer"].replace("ms-plugin.", "") for node in pair["data"]["tree"]["transforms"])
        self.assertEqual(kinds.count("transform-structure-conformation"), 1)
        self.assertEqual(kinds.count("download"), 2)
        annotations = [node["params"]["properties"]["mvs-annotations"]["annotations"][0]["id"]
                       for node in pair["data"]["tree"]["transforms"]
                       if node["transformer"] == "ms-plugin.custom-model-properties"]
        self.assertEqual(annotations, ["information_content", "same_residue_pct"])
        styled = saved_scene("styled")["session"]
        types = [(node["params"]["type"]["name"], node["params"]["type"]["params"]["alpha"])
                 for node in styled["data"]["tree"]["transforms"] if node["transformer"] == "ms-plugin.structure-representation-3d"]
        self.assertEqual(types, [("ball-and-stick", 0.6), ("gaussian-surface", 0.45)])
        opened = saved_scene("opened")["session"]
        raw = [node["params"] for node in opened["data"]["tree"]["transforms"] if node["transformer"] == "ms-plugin.raw-data"]
        # The coordinates travel inside the session: the file is not on the Viewer's side.
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["data"], opened_structure_pdb())
        marked = saved_scene("marked")["session"]
        shapes = sorted(node["transformer"].replace("ms-plugin.", "") for node in marked["data"]["tree"]["transforms"]
                        if "selections" in node["transformer"])
        self.assertEqual(shapes, ["structure-selections-distance-3d", "structure-selections-label-3d"])


class SavedSceneTests(SavedSceneTestCase):
    def test_two_folded_structures_come_back_with_their_pose_and_each_in_the_layer_it_was_on(self):
        scene = saved_scene("pair")
        expected = scene["expected"]
        stored = pose_in(scene["session"])
        for route in ROUTES:
            with self.subTest(route=route):
                restored = self.restore("pair", route)
                self.assertEqual(restored, expected)
                first, second = restored["structures"]
                self.assertEqual([first["file"], second["file"]], [FIRST, SECOND])
                self.assertEqual([first["coloring"], second["coloring"]], ["information_content", "same_residue_pct"])
                self.assertEqual([first["themes"], second["themes"]], [["mvs-annotation"]] * 2)
                self.assertEqual(restored["card"]["selected"], "information_content")
                # The second one is where it was fitted: the stored matrix, bit for bit, and on the first.
                self.assertIsNone(first["matrix"])
                self.assertEqual(second["matrix"], stored)
                self.assertLess(second["rmsd"], 0.6)
                # Every residue has the color the Viewer's rows for its layer give it.
                self.assertEqual(first["colors"], self.rows(FIRST, "information_content"))
                for number, color in self.rows(SECOND, "same_residue_pct").items():
                    self.assertEqual(second["colors"][number], color, number)
                self.assertEqual(self.errors(), [])

    def test_a_changed_representation_and_opacity_come_back(self):
        scene = saved_scene("styled")
        for route in ROUTES:
            with self.subTest(route=route):
                restored = self.restore("styled", route)
                self.assertEqual(restored, scene["expected"])
                first, second = restored["structures"]
                self.assertEqual([(s["representation"], s["opacity"]) for s in (first, second)],
                                 [("ball-and-stick", 60), ("gaussian-surface", 45)])
                self.assertEqual([s["representations"] for s in (first, second)], [1, 1])
                self.assertEqual(self.script("""
                    return loadedStructures().map(s => s.components[0].representations[0].cell.transform.params.type.params.alpha);
                    """), [0.6, 0.45])
                # Still pLDDT, as it was.
                for index, structure in enumerate((first, second)):
                    plddt = fixtures.plddt_values(33, 17 * index)
                    for number in range(1, 34):
                        self.assertEqual(structure["colors"][str(number)], band(plddt[number - 1]), (index, number))
                self.assertEqual(self.errors(), [])

    def test_a_structure_opened_from_a_file_comes_back_with_its_coordinates_and_its_own_color(self):
        scene = saved_scene("opened")
        for route in ROUTES:
            with self.subTest(route=route):
                restored = self.restore("opened", route)
                self.assertEqual(restored, scene["expected"])
                folded, opened = restored["structures"]
                self.assertEqual((folded["file"], folded["folded"], folded["coloring"]), (FIRST, True, "plddt"))
                self.assertEqual((opened["file"], opened["folded"], opened["coloring"]), (None, False, "builtin:structure"))
                self.assertEqual(opened["name"], OPENED_NAME[:-len(".pdb")])
                self.assertEqual(opened["themes"], ["uniform"])
                # One color for the whole structure, and the scene's choice is still pLDDT.
                self.assertEqual(len(set(opened["colors"].values())), 1)
                self.assertEqual(restored["card"]["selected"], "plddt")
                self.assertEqual(restored["kinds"]["raw-data"], 1)
                # Its residues are where the file put them, moved by the alignment that was stored.
                self.assertEqual(self.script("""
                    const trace = readCaTrace(loadedStructures()[1].cell.obj.data);
                    return [trace.codes.length, trace.ids[0], trace.plddt[5]];
                    """), [33, ["A", 1], fixtures.plddt_values(33)[5]])
                self.assertTrue(opened["matrix"])
                self.assertLess(opened["rmsd"], 0.6)
                self.assertEqual(self.errors(), [])

    def test_a_measured_distance_a_label_and_a_pinned_residue_come_back_without_the_pin(self):
        scene = saved_scene("marked")
        for route in ROUTES:
            with self.subTest(route=route):
                restored = self.restore("marked", route)
                self.assertEqual(restored, scene["expected"])
                self.assertEqual(restored["measurements"], {"distances": 1, "labels": 1})
                # The residue pinned when the session was taken is not pinned when it comes back, and nothing of Mol*'s focus is.
                self.assertEqual(restored["focus"], {"nodes": 0, "current": False, "marked": 0, "pinned": None})
                self.assertEqual(self.script("return [...plugin.state.data.cells.values()].filter(c => c.transform.tags?.some(t => t.startsWith('structure-focus-'))).length"), 0)
                self.assertEqual(restored["kinds"]["structure-selections-distance-3d"], 1)
                self.assertEqual(restored["kinds"]["structure-selections-label-3d"], 1)
                # The page's own Clear finds what came back and takes it away.
                self.script("clearAnnotations(); await structureQueue; return 1")
                self.settle(500)
                cleared = self.script("return await sessionSummary()")
                self.assertEqual(cleared["measurements"], {"distances": 0, "labels": 0})
                self.assertNotIn("structure-selections-distance-3d", cleared["kinds"])
                self.assertEqual(self.errors(), [])

    def test_every_scene_saves_again_and_what_it_saves_comes_back_the_same(self):
        for name in SCENES:
            with self.subTest(scene=name):
                expected = saved_scene(name)["expected"]
                self.assertEqual(self.restore(name, "viewer"), expected)
                del self.saved[:]
                self.assertTrue(self.script("return await saveSession()"))
                self.pump(0.5)
                self.assertTrue(self.saved)
                again = json.loads(json.dumps(self.saved[-1]))
                self.assertEqual(again["structureFocus"], {})
                self.assertEqual([tag for tag in tags_in(again) if tag.startswith("structure-focus-")], [])
                self.session = again
                self.open_page()
                self.js(POINTER)
                self.js(SUMMARY_JS)
                self.settle(1200)
                self.assertEqual(self.script("return await sessionSummary()"), expected)
        self.assertEqual(self.errors(), [])

    def test_a_scene_opened_over_another_replaces_it_and_undo_brings_the_other_back(self):
        self.restore("pair", "viewer")
        before = saved_scene("pair")["expected"]
        scene = saved_scene("opened")
        self.script("await openSessionFile(new File([%s], 'other.molj')); await structureQueue; return 1" % json.dumps(scene["molj"]))
        self.settle(1200)
        self.assertEqual(self.script("return await sessionSummary()"), scene["expected"])
        self.script("document.querySelector('#toast .toast-action').click(); await structureQueue; return 1")
        self.settle(1500)
        self.assertEqual(self.script("return await sessionSummary()"), before)
        self.assertEqual(self.errors(), [])


class ReplacedSceneTests(SavedSceneTestCase):
    """The page while Mol* changes the scene under it.

    Mol* rebuilds its structure hierarchy when an update ends. An update that replaces the scene (a session
    opened, Undo, Clear, a structure removed) takes the old nodes out of the state at its start and then waits
    for the new ones to load, so for that time the hierarchy still lists structures whose nodes are gone.
    A timer of the page (the list's redraw, 30 ms after a cell changed) or a pointer event can land there:
    on a busy machine one did, and threw "Cannot read properties of undefined (reading 'transform')" from
    the walk up a structure's nodes. These tests put the page at that point without the machine's help.
    """

    def test_the_page_stays_whole_in_the_middle_of_a_scene_being_replaced(self):
        self.restore("pair", "viewer")
        scene = saved_scene("opened")
        self.script(MID_REPLACE_JS)
        self.script("await openSessionFile(new File([%s], 'other.molj')); await structureQueue; return 1" % json.dumps(scene["molj"]))
        self.settle(1200)
        middle = self.script("return window.midReplace")
        # (If a newer Mol* rebuilds its hierarchy earlier, this is where that shows, and the page's waiting can go.)
        self.assertIsNotNone(middle, "the update never reached the point where its old nodes were gone and the hierarchy still listed them")
        # Asked there, the page does not fail and does not act on the structures that are going.
        self.assertEqual(middle["errors"], [])
        self.assertEqual((middle["listed"], middle["loaded"], middle["replacing"]), (2, 0, True))
        self.assertEqual(middle["byRef"], [False, False])
        self.assertEqual(middle["roots"], [None, None])
        self.assertTrue(all(isinstance(name, str) and name for name in middle["names"]), middle["names"])
        # The redraw due then is put off, not drawn from them (a list of none would show the hint and forget the rows' state).
        self.assertTrue(middle["untouched"])
        self.assertTrue(middle["pending"])
        # The rows of the structures that are going do not act on them when pressed (Remove would have said it removed one).
        self.assertEqual(middle["buttons"], 3)
        self.assertTrue(middle["silent"])
        # When the update ends the list is the new scene's, and nothing is left waiting.
        after = self.script("""return {pending: renderRowsAfterUpdate, replacing: sceneBeingReplaced(),
            names: [...document.querySelectorAll('#struct-list .srow-name')].map(node => node.textContent)}""")
        self.assertEqual(after, {"pending": False, "replacing": False, "names": [item["name"] for item in scene["expected"]["structures"]]})
        self.assertEqual(self.script("return await sessionSummary()"), scene["expected"])
        self.assertEqual(self.errors(), [])

    def test_a_structure_with_a_node_missing_has_no_name_file_or_root_to_delete(self):
        self.restore("pair", "viewer")
        found = self.script("""
            const state = plugin.state.data;
            const structure = loadedStructures()[1];
            const chain = nodesAbove(structure);
            const read = () => ({nodes: nodesAbove(structure) && nodesAbove(structure).length, root: structureDataRoot(structure) ?? null,
                url: structureSourceUrl(structure) ?? null, file: structureFile(structure) ?? null, loaded: loadedStructures().length});
            const whole = {...read(), name: structureName(structure), rows: document.querySelectorAll('#struct-list .srow').length};
            let updates = 0;
            const watch = state.behaviors.isUpdating.subscribe(updating => { if (updating) updates++; });
            const missing = [];
            for (const node of chain) {
                // The node is not in the state, as when Mol* has removed it and its hierarchy has not caught up.
                const size = state.tree.transforms.size;
                const started = updates;
                state.cells.delete(node.transform.ref);
                try {
                    const out = {...read(), name: structureName(structure)};
                    renderStructures();
                    out.rows = document.querySelectorAll('#struct-list .srow').length;
                    await deleteStructureData(structure);
                    out.kept = state.tree.transforms.size === size;
                    out.updates = updates - started;
                    missing.push(out);
                } catch (error) {
                    missing.push('ERROR ' + error.message);
                } finally {
                    state.cells.set(node.transform.ref, node);
                }
            }
            watch.unsubscribe();
            return {chain: chain.length, whole, missing};
            """)
        self.assertEqual(found["chain"], 5)
        self.assertEqual(found["whole"]["nodes"], 5)
        self.assertEqual((found["whole"]["file"], found["whole"]["name"]), (SECOND, SECOND[:-len(".pdb")]))
        self.assertTrue(found["whole"]["root"] and found["whole"]["url"].startswith("/structures/"))
        # Whichever node is gone, the structure is gone with it: no name read from what is above, no guess at a top to delete,
        # nothing deleted, and the list shows the structure that is whole.
        # (Its name is whatever its own node is labeled; nothing shows it.)
        names = [item.pop("name") if isinstance(item, dict) else item for item in found["missing"]]
        self.assertTrue(all(isinstance(name, str) and name for name in names), names)
        self.assertEqual(found["missing"], [
            {"nodes": None, "root": None, "url": None, "file": None, "loaded": 1, "rows": 1, "kept": True, "updates": 0}] * found["chain"])
        self.assertEqual(self.errors(), [])

    def test_a_list_put_off_while_a_scene_is_replaced_is_drawn_when_the_update_ends(self):
        self.restore("pair", "viewer")
        found = self.script("""
            const state = plugin.state.data;
            const list = byId('struct-list');
            const node = nodesAbove(loadedStructures()[1])[1];
            const mark = make('i');
            list.append(mark);
            // The state has lost a node the hierarchy still lists a structure of, and an update is running.
            state.cells.delete(node.transform.ref);
            state.behaviors.isUpdating.next(true);
            renderStructures();
            const waiting = {replacing: sceneBeingReplaced(), drawn: !mark.isConnected, pending: renderRowsAfterUpdate};
            state.cells.set(node.transform.ref, node);
            await new Promise(resolve => setTimeout(resolve, 200));
            const stillWaiting = {drawn: !mark.isConnected, pending: renderRowsAfterUpdate};
            state.behaviors.isUpdating.next(false);
            await new Promise(resolve => setTimeout(resolve, 300));
            return {waiting, stillWaiting, ended: {drawn: !mark.isConnected, pending: renderRowsAfterUpdate, rows: list.querySelectorAll('.srow').length}};
            """)
        self.assertEqual(found["waiting"], {"replacing": True, "drawn": False, "pending": True})
        # Nothing draws it before the update has ended.
        self.assertEqual(found["stillWaiting"], {"drawn": False, "pending": True})
        self.assertEqual(found["ended"], {"drawn": True, "pending": False, "rows": 2})
        self.assertEqual(self.errors(), [])


if __name__ == "__main__":
    unittest.main()
