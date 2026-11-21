# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's page: pass-through points of hidden nodes, the overview map, level of detail, and a tree of 44,000 leaves.

Opened offscreen over the real web server like the rest (tests/tree_page_harness.py); the first half of the page's
tests is test_tree_page.py, the colour half test_tree_color_page.py. The helpers are in tests/tree_page_helpers.py."""

import json
import os
import pathlib
import sys
import time
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for folder in (SRC_DIR, PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.tree_page_harness import TreePageMixin  # noqa: E402
from tests.tree_fixtures import (  # noqa: E402
    headers_of,
    shaped_linkage,
)
from tests.tree_page_helpers import (  # noqa: E402
    LayoutCase,
    clade_of,
    nj_options,
)
from utilities.Guide_Tree_File import read_guide_tree  # noqa: E402


class PassThroughTests(LayoutCase):
    """While the Viewer hides nodes, a node left with one child that is shown is no clade: it joins nothing, and its branch
    runs straight on into its child's. The stretch through a chain of such nodes belongs to the first clade or leaf below."""

    COUNT = 100
    KINDS = ("rectangular", "circular", "unrooted")

    def clade(self, low=12, high=30):
        return self.state(clade_of(low, high))

    def blank_names(self):
        """Give every leaf a one-character name, so that where a stretch is pointed at no other leaf's name is in the way."""
        self.script("TreeView.state.model.labels = TreeView.state.model.labels.map(() => '.'); return 1")

    def hide_all_but(self, clade, extra=0, zoom=False):
        """Hide every leaf of ``clade`` but its first (and the next ``extra``): the clade's chain of nodes down to it is left
        with one child each. Returns the kept leaf, the node of it, and the nodes that are pass-through points. With ``zoom``
        the view is on the clade, so that its stretch is long and no other leaf is near it."""
        return self.script(f"""const s = TreeView.state, m = s.model;
            const keep = m.firstLeaf[{clade}];
            const removed = new Uint8Array(m.leafCount);
            for (let k = m.firstLeaf[{clade}]; k <= m.lastLeaf[{clade}]; k++) if (k > keep + {extra}) removed[k] = 1;
            TreeView.applyRemoved(removed);
            if ({'true' if zoom else 'false'}) TreeView.zoomToNode({clade});
            TreeView.drawNow();
            const through = [];
            for (let i = 0; i < m.nodeCount; i++) if (s.layout.through[i]) through.push(i);
            return {{keep, node: m.leafNode[keep], through}};""")

    def stretch_points(self, nodes):
        """The middle of each node's stem in the canvas's pixels, for the stems that are long enough to be pointed at."""
        return self.state(f"""(() => {{ const s = TreeView.state, v = s.view, o = new Float64Array(8), points = [];
            for (const i of {json.dumps(nodes)}) {{
                s.layout.stem(i, o);
                const length = Math.hypot((o[2] - o[0]) * v.sx, (o[3] - o[1]) * v.sy);
                if (length > 14) points.push([i, v.ox + (o[0] + o[2]) / 2 * v.sx, v.oy + (o[1] + o[3]) / 2 * v.sy]);
            }}
            return points; }})()""")

    def test_a_layout_of_the_whole_tree_has_no_pass_through_points(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.assertEqual(self.state("[TreeView.state.layout.through, TreeView.state.layout.shown, TreeView.state.layout.live]"),
                                 [None, None, None])

    def test_the_nodes_left_with_one_child_are_the_pass_through_points(self):
        clade = self.clade()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                info = self.hide_all_but(clade)
                expected = self.state(f"""(() => {{ const m = TreeView.state.model, out = [];
                    for (let i = 0; i < m.nodeCount; i++) {{
                        if (m.childCount[i] > 0 && i !== m.root && TreeView.isInside(m.leafNode[{info['keep']}], i) && m.size[i] <= m.size[{clade}]) out.push(i);
                    }}
                    return out; }})()""")
                self.assertTrue(expected)
                self.assertEqual(info["through"], expected)
                self.script("TreeView.applyRemoved(null); return 1")
                self.assertEqual(self.state("TreeView.state.layout.through"), None)

    def test_a_pass_through_point_is_no_hit_target_and_its_stretch_belongs_to_the_node_below(self):
        clade = self.clade()
        self.blank_names()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                info = self.hide_all_but(clade, zoom=True)
                points = self.stretch_points(info["through"])
                self.assertTrue(points, kind)
                for node, x, y in points:
                    hit = self.state(f"TreeView.hitTest({x}, {y})")
                    self.assertEqual(hit, {"node": info["node"], "part": "leaf"}, (kind, node))
                # Nowhere on the canvas is a pass-through point found, whatever the pointer is over.
                found = self.script("""const s = TreeView.state, found = new Set();
                    for (let x = 4; x < TreeView.canvas.clientWidth; x += 9) for (let y = 4; y < TreeView.canvas.clientHeight; y += 9) {
                        const hit = TreeView.hitTest(x, y);
                        if (hit) found.add(hit.node);
                    }
                    return Array.from(found).filter(i => s.layout.through[i]);""")
                self.assertEqual(found, [])

    def test_the_keys_walk_over_pass_through_points_and_never_onto_a_hidden_node(self):
        clade = self.clade()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                info = self.hide_all_but(clade)
                # Up from the leaf that is left: the first clade above it that joins two shown branches.
                self.script(f"TreeView.state.pinned = {info['node']}; fire.key('ArrowLeft', {{ctrlKey: true}}); return 1")
                above = self.state("TreeView.state.pinned")
                chain = self.state(f"""(() => {{ const m = TreeView.state.model, out = [];
                    for (let a = m.parent[{info['node']}]; a !== {above}; a = m.parent[a]) out.push(a);
                    return out; }})()""")
                self.assertNotIn(above, info["through"])
                self.assertTrue(chain)
                self.assertEqual(sorted(chain), sorted(set(chain) & set(info["through"])))
                # Its children and their siblings, as the keys find them: the leaf that is left is one, and none is
                # a pass-through point or a node that is hidden.
                visited = self.script("""const s = TreeView.state, seen = [];
                    fire.key('ArrowRight', {ctrlKey: true});
                    for (let n = 0; n < 50; n++) {
                        seen.push(s.pinned);
                        fire.key('ArrowDown', {ctrlKey: true});
                        if (s.pinned === seen[seen.length - 1]) break;
                    }
                    return seen.map(i => [i, s.layout.live[i], s.layout.through[i]]);""")
                self.assertIn(info["node"], [node for node, _, _ in visited])
                for node, live, through in visited:
                    self.assertEqual((live, through), (1, 0), node)
                self.script("TreeView.applyRemoved(null); TreeView.state.pinned = -1; return 1")

    def test_the_stretch_of_a_clade_with_several_leaves_left_belongs_to_that_clade(self):
        clade = self.clade(12, 30)
        self.blank_names()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                # Three leaves are left in it: the first two form a clade of the shown tree, the chain above that is pass-through.
                info = self.hide_all_but(clade, extra=2, zoom=True)
                real = self.state(f"""(() => {{ const s = TreeView.state, m = s.model; let i = m.leafNode[{info['keep']}];
                    while (s.layout.through[i] || m.childCount[i] === 0) i = m.parent[i];
                    return i; }})()""")
                points = [p for p in self.stretch_points(info["through"]) if p[0] != real]
                for node, x, y in points:
                    below = self.state(f"TreeView.cladeBelow({node})")
                    hit = self.state(f"TreeView.hitTest({x}, {y})")
                    self.assertFalse(self.state(f"TreeView.state.layout.through[{below}]"))
                    self.assertEqual(hit["node"], below, (kind, node, hit))
                    self.assertIn(hit["part"], ("branch", "leaf"))

    def test_hovering_the_stretch_describes_the_node_below_and_highlights_the_whole_stretch(self):
        clade = self.clade()
        self.blank_names()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                info = self.hide_all_but(clade, zoom=True)
                points = self.stretch_points(info["through"])
                self.assertTrue(points)
                node, x, y = points[0]
                self.script(f"TreeView.drawNow(); return 1")
                quiet = self.state(f"fire.pixel({x}, {y})")
                self.script(f"fire.move({x}, {y}); return 1")
                self.wait_for(f"TreeView.state.hover === {info['node']}")
                self.assertIn("A node of this network", self.state("document.getElementById('tooltip').textContent"))
                self.assertNotIn("A clade", self.state("document.getElementById('tooltip').textContent"))
                self.script("TreeView.drawNow(); return 1")
                # The line under the pointer is drawn in the highlight's colour now, though it is a pass-through point's.
                self.assertNotEqual(self.state(f"fire.pixel({x}, {y})"), quiet)

    def test_a_pass_through_point_has_no_square_folds_nothing_and_is_not_zoomed_to_by_a_gesture(self):
        clade = self.clade()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                info = self.hide_all_but(clade)
                top = max(info["through"])
                self.assertEqual(self.script(f"return TreeView.toggleFold({top})"), False)
                self.assertEqual(self.state("TreeView.state.collapsedCount"), 0)
                self.script(f"TreeView.state.hover = {top}; TreeView.state.pinned = {top}; return 1")
                self.assertEqual(self.drawn()["handles"], 0)
                self.assertEqual(self.state(f"TreeView.state.model.parent[{top}] >= 0"), True)
                # A double-click on the stretch zooms to the node below, which is a leaf.
                x, y = self.stretch_points([top])[0][1:]
                self.script(f"TreeView.state.hover = -1; TreeView.state.pinned = -1; fire.dblclick({x}, {y}); return 1")
                self.assertEqual(self.state("TreeView.state.pinned"), -1)

    def test_hiding_lets_go_of_a_pass_through_point_that_was_hovered_or_pinned(self):
        clade = self.clade()
        self.choose("rectangular")
        self.script(f"TreeView.state.hover = {clade}; TreeView.state.pinned = {clade}; return 1")
        self.hide_all_but(clade)
        self.assertEqual(self.state("[TreeView.state.hover, TreeView.state.pinned]"), [-1, -1])

    def test_a_pass_through_point_has_no_stability_dot_and_a_clade_keeps_its_value(self):
        clade = self.clade(12, 30)
        self.script("""const m = TreeView.state.model;
            m.support = new Int16Array(m.nodeCount).fill(70); m.hasSupport = true;
            TreeView.state.showSupport = true; return 1""")
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                whole = self.drawn()["supports"]
                self.assertGreater(whole, 0)
                info = self.hide_all_but(clade, extra=1)
                # Dots only where there is a clade: counted from the shown tree's nodes with two or more children.
                expected = self.script("""const s = TreeView.state, m = s.model, b = s.layout.boxes, v = s.view;
                    let count = 0;
                    const stack = [m.root];
                    while (stack.length) {
                        const i = stack.pop();
                        if (m.childCount[i] === 0) continue;
                        if (s.layout.tiny(i, v.sx, v.sy, TreeView.COLLAPSE_PX) || s.collapsed[i]) continue;
                        const w = (b.x1[i] - b.x0[i]) * v.sx, h = (b.y1[i] - b.y0[i]) * v.sy;
                        if (!s.layout.through[i] && i !== m.root && Math.min(w, h) >= TreeView.SUPPORT_CLADE_PX) count++;
                        for (let k = m.childStart[i]; k < m.childStart[i + 1]; k++) if (s.layout.live[m.childList[k]]) stack.push(m.childList[k]);
                    }
                    return count;""")
                self.assertEqual(self.drawn()["supports"], expected)
                self.assertLess(expected, whole)
                real = self.state(f"""(() => {{ const s = TreeView.state, m = s.model; let i = m.leafNode[{info['keep']}];
                    while (s.layout.through[i] || m.childCount[i] === 0) i = m.parent[i];
                    return i; }})()""")
                lines = self.state(f"TreeView.describeNode({real}).lines")
                self.assertIn("Stable in 70% of the replicate trees", lines)
                self.script("TreeView.applyRemoved(null); return 1")

    def test_a_clade_counts_the_leaves_that_are_shown_and_says_how_many_are_hidden(self):
        clade = self.clade(12, 30)
        size = self.state(f"TreeView.leavesUnder(TreeView.state.model, {clade})")
        whole = self.state(f"TreeView.describeNode({clade}).lines")
        self.assertIn(f"Leaves in this clade: {size}", whole)
        self.assertIn(f"Of them in this network: {size}", whole)
        self.assertFalse([line for line in whole if line.startswith("Hidden")])
        self.script(f"""const m = TreeView.state.model, removed = new Uint8Array(m.leafCount);
            removed[m.firstLeaf[{clade}]] = 1; removed[m.firstLeaf[{clade}] + 1] = 1; removed[m.lastLeaf[{clade}]] = 1;
            TreeView.applyRemoved(removed); return 1""")
        lines = self.state(f"TreeView.describeNode({clade}).lines")
        self.assertIn(f"Leaves in this clade: {size - 3}", lines)
        self.assertIn(f"Of them in this network: {size - 3}", lines)
        self.assertIn("Hidden in the network: 3", lines)
        # The root counts the same way.
        root = self.state("TreeView.describeNode(TreeView.state.model.root).lines")
        self.assertIn("Hidden in the network: 3", root)
        self.script("TreeView.applyRemoved(null); return 1")
        self.assertEqual(self.state(f"TreeView.describeNode({clade}).lines"), whole)

    def test_hidden_leaves_that_are_not_nodes_of_the_network_are_not_counted_among_its_nodes(self):
        # Of the leaves shown, those that are nodes of the network are counted as before: only hidden ones leave the count.
        clade = self.clade(12, 30)
        before = self.state(f"TreeView.shownUnder(TreeView.state.model, {clade})")
        self.script(f"""const m = TreeView.state.model, removed = new Uint8Array(m.leafCount);
            removed[m.firstLeaf[{clade}]] = 1; TreeView.applyRemoved(removed); return 1""")
        self.assertIn(f"Of them in this network: {before - 1}", self.state(f"TreeView.describeNode({clade}).lines"))

    def select_and_hide(self, kind, extra_selected=False):
        """Select two leaves in one half of a clade G and one in the other half, then hide the other half of G."""
        self.choose(kind)
        return self.script(f"""const s = TreeView.state, m = s.model;
            let g = -1;
            for (let i = 0; i < m.nodeCount && g < 0; i++) {{
                if (m.childCount[i] !== 2 || i === m.root) continue;
                const a = m.childList[m.childStart[i]], b = m.childList[m.childStart[i] + 1];
                if (m.lastLeaf[a] - m.firstLeaf[a] >= 3 && m.lastLeaf[b] - m.firstLeaf[b] >= 3) g = i;
            }}
            const left = m.childList[m.childStart[g]], right = m.childList[m.childStart[g] + 1];
            const a = m.firstLeaf[left], b = m.lastLeaf[left], c = m.firstLeaf[right];
            TreeView.applySelection([m.viewerNode[a], m.viewerNode[b], m.viewerNode[c]]);
            const before = s.selection.clade;
            const removed = new Uint8Array(m.leafCount);
            for (let k = m.firstLeaf[right]; k <= m.lastLeaf[right]; k++) removed[k] = 1;
            TreeView.applyRemoved(removed);
            TreeView.drawNow();
            let node = m.leafNode[a];
            while (m.lastLeaf[node] < b) node = m.parent[node];
            return {{g, left, right, a, b, c, before, pair: node,
                     outline: TreeView.outlineClade(m, s.layout, s.selection),
                     throughG: s.layout.through[g] === 1, throughOutline: s.layout.through[TreeView.outlineClade(m, s.layout, s.selection)] === 1}};""")

    def test_the_outline_of_the_selection_never_lands_on_a_pass_through_point(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                info = self.select_and_hide(kind)
                # The network still named a hidden node (as a hide and a selection arriving in turn can): the clade of all
                # three is G, which the hiding has left with one child.
                self.assertEqual(info["before"], info["g"])
                self.assertTrue(info["throughG"])
                self.assertEqual(info["outline"], info["pair"])
                self.assertFalse(info["throughOutline"])
                self.assertEqual(self.drawn()["selected"], 2)

    def test_a_selection_with_one_leaf_shown_has_no_outline(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                info = self.script("""const s = TreeView.state, m = s.model;
                    TreeView.applySelection([m.viewerNode[0], m.viewerNode[1]]);
                    const removed = new Uint8Array(m.leafCount); removed[1] = 1;
                    TreeView.applyRemoved(removed); TreeView.drawNow();
                    return {outline: TreeView.outlineClade(m, s.layout, s.selection), clade: s.selection.clade};""")
                self.assertEqual(info["outline"], -1)

    def test_without_hidden_nodes_the_outline_is_the_selections_clade(self):
        self.choose("rectangular")
        info = self.script("""const s = TreeView.state, m = s.model;
            TreeView.applySelection([m.viewerNode[3], m.viewerNode[9]]);
            return {outline: TreeView.outlineClade(m, s.layout, s.selection), clade: s.selection.clade};""")
        self.assertGreaterEqual(info["clade"], 0)
        self.assertEqual(info["outline"], info["clade"])

    def test_the_viewer_never_keeps_a_hidden_node_in_its_selection(self):
        # What the page is told of the selection is the Viewer's, and the Viewer drops what it hides (commands/hide.py).
        source = (PROJECT_ROOT / "src" / "commands" / "hide.py").read_text(encoding="utf-8")
        self.assertIn("The Viewer never keeps a hidden node selected", source)

    def folded_with_one_shown(self, kind):
        self.choose(kind)
        clade = self.clade(12, 30)
        self.script(f"TreeView.toggleFold({clade}); return 1")
        return clade

    def test_a_folded_clade_with_one_leaf_shown_is_drawn_as_the_leaf_and_is_folded_again_when_more_are_shown(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                clade = self.folded_with_one_shown(kind)
                size = self.state(f"TreeView.leavesUnder(TreeView.state.model, {clade})")
                self.assertEqual(self.drawn()["collapsed"], 1)
                info = self.hide_all_but(clade)
                self.assertIn(clade, info["through"])
                frame = self.drawn()
                self.assertEqual(frame["collapsed"], 0)
                # The fold is kept, but it hides nothing: its leaf has a tip, and can be found and named.
                self.assertEqual(self.state(f"[TreeView.state.collapsed[{clade}], TreeView.state.collapsedCount]"), [1, 1])
                self.assertEqual(self.state("TreeView.state.leafHidden.reduce((a, b) => a + b, 0)"), 0)
                self.assertEqual(self.state(f"TreeView.leafIsDrawn({info['keep']})"), True)
                tip = self.state(f"(() => {{ const s = TreeView.state, b = TreeView.nodePixels(s, {info['node']}); return [b.x, b.y]; }})()")
                self.assertEqual(self.state(f"TreeView.hitTest({tip[0]}, {tip[1]})"), {"node": info["node"], "part": "leaf"})
                # Focusing the leaf leaves the fold alone.
                self.script(f"TreeView.focusLeaf({info['keep']}); return 1")
                self.assertEqual(self.state(f"TreeView.state.collapsed[{clade}]"), 1)
                # More of its leaves shown: two are enough for the fold to be the shape of the clade again.
                self.hide_all_but(clade, extra=1)
                self.assertEqual(self.drawn()["collapsed"], 1)
                self.assertEqual(self.state("TreeView.state.leafHidden.reduce((a, b) => a + b, 0)"), size)
                self.script("TreeView.applyRemoved(null); return 1")
                self.assertEqual(self.drawn()["collapsed"], 1)
                self.assertEqual(self.state("TreeView.state.leafHidden.reduce((a, b) => a + b, 0)"), size)
                self.script(f"TreeView.toggleFold({clade}); return 1")

    def test_a_fold_that_is_a_clade_of_the_shown_tree_has_its_stretch_and_its_square(self):
        # A folded clade left with two leaves is still a clade, so the pass-through points above it belong to it.
        clade = self.folded_with_one_shown("rectangular")
        self.hide_all_but(clade, extra=1)
        self.assertEqual(self.state(f"TreeView.state.layout.through[{clade}] === 1 || TreeView.state.layout.shown[{clade}] >= 2"), True)
        self.assertEqual(self.state(f"TreeView.isThrough(TreeView.state.layout, TreeView.state.collapsed, {clade})"), False)
        parent = self.state(f"TreeView.state.model.parent[{clade}]")
        if self.state(f"TreeView.state.layout.through[{parent}] === 1"):
            self.assertEqual(self.state(f"TreeView.cladeBelow({parent})"), clade)

    def test_a_chain_of_pass_through_points_is_followed_in_one_step_for_each_node(self):
        clade = self.clade()
        self.choose("rectangular")
        info = self.hide_all_but(clade)
        below = self.state(f"""(() => {{ const out = []; for (const i of {json.dumps(info['through'])}) out.push(TreeView.cladeBelow(i)); return out; }})()""")
        self.assertEqual(below, [info["node"]] * len(info["through"]))
        # A node that is a clade is its own.
        self.assertEqual(self.state(f"TreeView.cladeBelow({info['node']})"), info["node"])
        self.assertEqual(self.state("TreeView.cladeBelow(TreeView.state.model.root)"), self.state("TreeView.state.model.root"))

    def test_nothing_changes_for_the_tree_when_nothing_is_hidden_again(self):
        clade = self.clade()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.script(f"TreeView.toggleFold({self.clade(5, 10)}); TreeView.state.showSupport = true; return 1")
                before = self.script("""const frame = TreeView.drawNow();
                    return {frame, image: TreeView.canvas.toDataURL(), view: TreeView.state.view,
                            hit: [TreeView.hitTest(300, 200), TreeView.hitTest(600, 400), TreeView.hitTest(900, 300)]};""")
                self.hide_all_but(clade)
                after = self.script("""TreeView.applyRemoved(null);
                    const frame = TreeView.drawNow();
                    return {frame, image: TreeView.canvas.toDataURL(), view: TreeView.state.view,
                            hit: [TreeView.hitTest(300, 200), TreeView.hitTest(600, 400), TreeView.hitTest(900, 300)]};""")
                for key in ("visited", "bars", "collapsed", "labels", "dots", "handles", "supports"):
                    self.assertEqual(after["frame"][key], before["frame"][key], (kind, key))
                self.assertEqual(after["view"], before["view"])
                self.assertEqual(after["hit"], before["hit"])
                self.assertEqual(after["image"], before["image"])
                self.script(f"TreeView.state.collapsed.fill(0); TreeView.state.collapsedCount = 0; TreeView.state.leafHidden = null; return 1")

    def test_node_numbers_stay_those_of_the_whole_tree_through_a_hide_and_an_unhide(self):
        clade = self.clade()
        self.choose("circular")
        self.script(f"TreeView.toggleFold({clade}); TreeView.state.pinned = {clade}; return 1")
        self.script("""const m = TreeView.state.model;
            TreeView.applySelection([m.viewerNode[m.firstLeaf[%d]], m.viewerNode[m.lastLeaf[%d]]]); return 1""" % (clade, clade))
        before = self.state("[TreeView.state.selection.clade, Array.from(TreeView.state.collapsed).reduce((a, b) => a + b, 0), TreeView.state.pinned]")
        self.hide_all_but(self.clade(40, 100))
        self.script("TreeView.applyRemoved(null); return 1")
        self.assertEqual(self.state("[TreeView.state.selection.clade, Array.from(TreeView.state.collapsed).reduce((a, b) => a + b, 0), TreeView.state.pinned]"),
                         before)
        self.assertEqual(self.state("TreeView.outlineClade(TreeView.state.model, TreeView.state.layout, TreeView.state.selection)"), before[0])

    def test_the_overview_map_and_the_page_draw_a_hidden_tree_without_error(self):
        clade = self.clade()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.hide_all_but(clade)
                self.script("TreeView.minimapChanged(); TreeView.drawMinimap(); return 1")
        self.assertEqual(self.console_errors(), [])


class OverviewMapTests(LayoutCase):
    COUNT = 100

    def map_view(self):
        return self.state("({...TreeView.minimapView})")

    def test_the_map_shows_the_whole_tree_and_the_rectangle_of_the_view_on_it(self):
        info = self.script("""TreeView.drawNow();
            const mini = document.getElementById('minimap'), ctx = mini.getContext('2d');
            const data = ctx.getImageData(0, 0, mini.width, mini.height).data;
            let ink = 0;
            for (let i = 3; i < data.length; i += 4) if (data[i] > 200) ink++;
            const box = mini.getBoundingClientRect();
            return {hidden: mini.hidden, ink, pixels: data.length / 4, width: Math.round(box.width), height: Math.round(box.height),
                    inside: [mini.clientWidth, mini.clientHeight], view: {...TreeView.minimapView}};""")
        self.assertFalse(info["hidden"])
        self.assertEqual((info["width"], info["height"]), (180, 140))
        self.assertEqual(info["inside"], [178, 138])
        # Opaque ink, which the tree is drawn in and the rectangle of the view is not.
        self.assertGreater(info["ink"], 100)
        self.assertLess(info["ink"], info["pixels"] * 0.5)
        # The whole tree is in view: the rectangle takes in the map's picture of it.
        view = info["view"]
        self.assertLessEqual(view["left"], 12)
        self.assertGreaterEqual(view["right"], info["width"] - 12)

    def test_the_rectangle_follows_the_view_in_every_layout(self):
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                whole = self.map_view()
                self.script("TreeView.zoomAbout(650, 330, 4, 4); TreeView.drawNow(); return 1")
                zoomed = self.map_view()
                self.assertLess((zoomed["right"] - zoomed["left"]) * (zoomed["bottom"] - zoomed["top"]),
                                (whole["right"] - whole["left"]) * (whole["bottom"] - whole["top"]) * 0.8)
                before = self.map_view()
                self.script("TreeView.pan(-120, -90); TreeView.drawNow(); return 1")
                after = self.map_view()
                self.assertGreater(after["left"] + after["top"], before["left"] + before["top"])

    def test_pressing_and_dragging_on_the_map_moves_the_view_there(self):
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                self.script("TreeView.zoomAbout(650, 330, 3, 3); TreeView.drawNow(); return 1")
                info = self.script("""
                    const mini = document.getElementById('minimap'), rect = mini.getBoundingClientRect(), s = TreeView.state;
                    const press = (type, x, y) => mini.dispatchEvent(new PointerEvent(type, {pointerId: 3, button: 0, buttons: type === 'pointerup' ? 0 : 1,
                        clientX: rect.left + mini.clientLeft + x, clientY: rect.top + mini.clientTop + y, bubbles: true}));
                    const centre = () => [(document.getElementById('tree-canvas').clientWidth / 2 - s.view.ox) / s.view.sx,
                                          (document.getElementById('tree-canvas').clientHeight / 2 - s.view.oy) / s.view.sy];
                    const out = {};
                    for (const [name, x, y] of [['first', 60, 50], ['second', 110, 90]]) {
                        press(name === 'first' ? 'pointerdown' : 'pointermove', x, y);
                        out[name] = {at: TreeView.minimapPoint(x, y), centre: centre()};
                    }
                    press('pointerup', 110, 90);
                    out.fitted = s.fitted;
                    return out;""")
                for name in ("first", "second"):
                    self.assertAlmostEqual(info[name]["centre"][0], info[name]["at"][0], delta=abs(info[name]["at"][0]) * 1e-6 + 1e-6)
                    self.assertAlmostEqual(info[name]["centre"][1], info[name]["at"][1], delta=abs(info[name]["at"][1]) * 1e-6 + 1e-6)
                self.assertFalse(info["fitted"])

    def test_the_whole_layout_is_inside_the_map_in_every_layout(self):
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                info = self.script("""
                    const mini = document.getElementById('minimap'), b = TreeView.state.layout.bounds;
                    const a = TreeView.minimapPoint(0, 0), z = TreeView.minimapPoint(mini.clientWidth, mini.clientHeight);
                    return {left: a[0] - b.minX, top: a[1] - b.minY, right: z[0] - b.maxX, bottom: z[1] - b.maxY};""")
                # The map's corners lie beyond the tree's, and one side of it at least is filled by the tree.
                self.assertLessEqual(info["left"], 1e-9)
                self.assertLessEqual(info["top"], 1e-9)
                self.assertGreaterEqual(info["right"], -1e-9)
                self.assertGreaterEqual(info["bottom"], -1e-9)

    def test_pressing_the_middle_of_the_map_puts_the_middle_of_the_tree_in_the_middle_of_the_view(self):
        # Not read back through the map's own arithmetic: the middle of the map is the middle of the tree in every layout.
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                self.script("TreeView.zoomAbout(650, 330, 3, 3); TreeView.drawNow(); return 1")
                info = self.script("""
                    const mini = document.getElementById('minimap'), rect = mini.getBoundingClientRect(), s = TreeView.state, b = s.layout.bounds;
                    const down = (x, y) => mini.dispatchEvent(new PointerEvent('pointerdown', {pointerId: 5, button: 0, buttons: 1,
                        clientX: rect.left + mini.clientLeft + x, clientY: rect.top + mini.clientTop + y, bubbles: true}));
                    const up = () => mini.dispatchEvent(new PointerEvent('pointerup', {pointerId: 5, button: 0, buttons: 0, bubbles: true}));
                    const view = document.getElementById('tree-canvas');
                    const centre = () => [(view.clientWidth / 2 - s.view.ox) / s.view.sx, (view.clientHeight / 2 - s.view.oy) / s.view.sy];
                    down(mini.clientWidth / 2, mini.clientHeight / 2);
                    up();
                    const middle = centre();
                    // The map keeps six pixels clear of its edge: its top left corner of the tree is there.
                    down(6, 6);
                    up();
                    return {middle, wanted: [(b.minX + b.maxX) / 2, (b.minY + b.maxY) / 2], corner: centre(), minimum: [b.minX, b.minY], uniform: s.layout.uniform};""")
                for got, wanted in zip(info["middle"], info["wanted"]):
                    self.assertAlmostEqual(got, wanted, delta=abs(wanted) * 1e-6 + 1e-6)
                if not info["uniform"]:
                    for got, wanted in zip(info["corner"], info["minimum"]):
                        self.assertAlmostEqual(got, wanted, delta=abs(wanted) * 1e-6 + 1e-6)

    def test_each_match_is_a_point_on_the_map(self):
        def matching_pixels():
            return self.state("""(() => { TreeView.drawNow();
                const mini = document.getElementById('minimap'), data = mini.getContext('2d').getImageData(0, 0, mini.width, mini.height).data;
                const ctx = document.createElement('canvas').getContext('2d');
                ctx.fillStyle = TreeView.state.palette.match; ctx.fillRect(0, 0, 1, 1);
                const want = ctx.getImageData(0, 0, 1, 1).data;
                let n = 0;
                for (let i = 0; i < data.length; i += 4) {
                    if (Math.abs(data[i] - want[0]) < 60 && Math.abs(data[i + 1] - want[1]) < 60 && Math.abs(data[i + 2] - want[2]) < 60 && data[i + 3] > 200) n++;
                }
                return n; })()""")
        self.choose("unrooted")
        # (The map's frame of the view is in the theme colour too, and the network's plain colour is a blue near it.)
        before = matching_pixels()
        self.script("TreeView.runSearch('PROT_1'); return 1")
        self.assertGreater(matching_pixels(), before + 5)

    def test_the_overview_button_hides_and_shows_the_map(self):
        self.js("document.getElementById('overview-btn').click()")
        self.assertEqual(self.state("[document.getElementById('minimap').hidden, document.getElementById('overview-btn').getAttribute('aria-pressed')]"),
                         [True, "false"])
        self.js("document.getElementById('overview-btn').click()")
        self.assertEqual(self.state("[document.getElementById('minimap').hidden, document.getElementById('overview-btn').getAttribute('aria-pressed')]"),
                         [False, "true"])
        # The choice holds through a layout change.
        self.js("document.getElementById('overview-btn').click()")
        self.choose("circular")
        self.assertTrue(self.state("document.getElementById('minimap').hidden"))

    def test_the_map_is_drawn_again_when_what_it_shows_changes(self):
        def picture():
            return self.state("""(() => { TreeView.drawNow(); const mini = document.getElementById('minimap');
                const data = mini.getContext('2d').getImageData(0, 0, mini.width, mini.height).data;
                let sum = 0; for (let i = 0; i < data.length; i++) sum = (sum * 31 + data[i]) % 1000003; return sum; })()""")
        first = picture()
        self.choose("circular")
        circular = picture()
        self.assertNotEqual(first, circular)
        clade = self.state(clade_of(10, 30))
        self.script(f"TreeView.toggleFold({clade}); return 1")
        folded = picture()
        self.assertNotEqual(circular, folded)
        self.script("TreeView.runSearch('PROT_1'); return 1")
        after_search = picture()
        self.assertNotEqual(folded, after_search)
        # The theme's colours: the map keeps its picture in them.
        background = self.state("TreeView.state.palette.background")
        self.js("document.getElementById('theme-toggle-btn').click()")
        self.wait_for("TreeView.state.palette.background !== " + json.dumps(background))
        self.assertNotEqual(self.state("TreeView.state.palette.background"), background)
        self.assertNotEqual(picture(), after_search)
        self.js("(() => { try { localStorage.removeItem('ssn_theme'); } catch (error) {} return 1; })()")


class LevelOfDetailEverywhereTests(TreePageMixin, unittest.TestCase):
    """A frame costs what can be seen, in every layout: clades under three pixels are one line."""

    COUNT = 3000

    def serve_shape(self, shape):
        self.headers = headers_of(self.COUNT)
        self.serve_headers(self.headers, linkage=shaped_linkage(shape, self.COUNT))
        self.open_page()

    def serve_neighbor_joining(self):
        self.headers = headers_of(self.COUNT)
        self.serve_headers(self.headers, **nj_options(self.COUNT))
        self.open_page()

    def tearDown(self):
        self.view.resize(1300, 800)
        super().tearDown()

    def test_each_layout_draws_far_less_than_the_tree_at_fit_and_names_leaves_when_zoomed(self):
        # A rectangular and a circular tree are as wide as their leaves are many, which a balanced tree shows well. An
        # unrooted one is as long as its branches, and the tree of neighbor-joining (which its files hold) has short
        # ones at its tips, by thousands.
        for kinds, serve in ((("rectangular", "circular"), lambda: self.serve_shape("balanced")),
                             (("unrooted",), self.serve_neighbor_joining)):
            serve()
            total = self.state("TreeView.state.model.nodeCount")
            for kind in kinds:
                with self.subTest(kind=kind):
                    self.script(f"TreeView.setLayout({json.dumps(kind)}, 'lengths'); return 1")
                    fit = self.drawn()
                    self.assertGreater(fit["bars"], 20)
                    self.assertLess(fit["visited"], total * 0.3)
                    self.assertEqual((fit["labels"], fit["dots"]), (0, 0))
                    # Brought up to be read, a leaf has others round it that can be told from it, in every layout.
                    self.script("TreeView.focusLeaf(100); return 1")
                    zoomed = self.drawn()
                    self.assertGreater(zoomed["labels"], 3)
                    self.assertGreater(zoomed["dots"], 3)
                    self.assertLess(self.state("TreeView.leavesInView()"), 61)
                    self.assertLess(zoomed["visited"], total * 0.3)

    def test_a_caterpillar_and_a_chain_of_clades_are_drawn_in_a_frame_in_every_layout(self):
        for shape in ("caterpillar", "chains"):
            self.serve_shape(shape)
            for kind in ("rectangular", "circular", "unrooted"):
                with self.subTest(shape=shape, kind=kind):
                    self.script(f"TreeView.setLayout({json.dumps(kind)}, 'lengths'); return 1")
                    stats = self.drawn()
                    self.assertGreaterEqual(stats["visited"], 1)
                    self.assertEqual(self.console_errors(), [])
                    # A pointer over it finds something or nothing, never fails.
                    self.script("for (let k = 0; k < 40; k++) TreeView.hitTest(100 + k * 29, 80 + k * 13); return 1")
                    self.script("TreeView.zoomToNode(TreeView.state.model.root); TreeView.drawNow(); return 1")

    def test_no_line_wholly_outside_the_canvas_is_added_to_the_drawing(self):
        # A clade is visited for its box, which can be far bigger than its own line; the line is not added to the path
        # when nobody could see it, which is where the cost of a zoomed view of a spiky tree went.
        self.serve_neighbor_joining()
        info = self.script("""
            TreeView.setLayout('unrooted', 'lengths');
            TreeView.focusLeaf(100);
            const w = TreeView.canvas.clientWidth, h = TreeView.canvas.clientHeight, margin = 8;
            const calls = [], move = Path2D.prototype.moveTo, line = Path2D.prototype.lineTo;
            let from = null;
            Path2D.prototype.moveTo = function (x, y) { from = [x, y]; return move.call(this, x, y); };
            Path2D.prototype.lineTo = function (x, y) { if (from) calls.push([from[0], from[1], x, y]); from = [x, y]; return line.call(this, x, y); };
            try { TreeView.drawNow(); } finally { Path2D.prototype.moveTo = move; Path2D.prototype.lineTo = line; }
            const outside = calls.filter(([a, b, c, d]) => (a < -margin && c < -margin) || (a > w + margin && c > w + margin)
                || (b < -margin && d < -margin) || (b > h + margin && d > h + margin));
            return {drawn: calls.length, outside: outside.slice(0, 3)};""")
        self.assertGreater(info["drawn"], 10)
        self.assertEqual(info["outside"], [])

    def test_tiny_clades_are_one_opaque_line_each_in_a_few_thicknesses(self):
        # A see-through shape costs several times as much to draw where there is no GPU: nothing about a tiny clade is.
        self.serve_shape("balanced")
        info = self.script("""
            const s = TreeView.state;
            const palette = s.palette;
            const rgb = TreeView.rgbOf;
            const mark = rgb(palette.mark), wedge = rgb(palette.wedge);
            return {mark, wedge, branch: rgb(palette.branch), background: rgb(palette.background),
                    opaque: [palette.mark, palette.wedge].every(c => /^rgb\\(\\d+, \\d+, \\d+\\)$/.test(c))};""")
        self.assertTrue(info["opaque"])
        # Between the branch colour and the background, nearer to the background for the fill.
        for channel in range(3):
            low, high = sorted((info["branch"][channel], info["background"][channel]))
            self.assertTrue(low <= info["mark"][channel] <= high)
            self.assertTrue(low <= info["wedge"][channel] <= high)

    def test_a_rectangular_tree_is_drawn_on_the_middle_of_its_pixels(self):
        # Lines that fall on the middle of a pixel draw several times faster than those spread over two.
        # Measured on the page as it draws at rest: at a pixel ratio of 1 a stem is one row of full colour with the
        # background beside it; at 2 it is centred on a device pixel (full colour there) and is 2 device pixels wide
        # (the page makes it of one-pixel lines moved half a pixel up, down, left and right), so the row on each side
        # is half of it and the next is the background. Other ratios have not been measured.
        self.serve_shape("balanced")
        ratio = self.state("window.devicePixelRatio")
        if ratio not in (1, 2):
            self.skipTest(f"the stems were measured at pixel ratios 1 and 2, not {ratio}")
        info = self.script("""
            const s = TreeView.state, v = s.view, L = s.layout, ratio = window.devicePixelRatio;
            TreeView.zoomToNode(TreeView.state.model.root);
            TreeView.zoomAbout(600, 300, 1, 60);
            TreeView.drawNow();
            // A leaf's stem is one row of full colour: the row beside it is the background.
            const ctx = document.createElement('canvas').getContext('2d');
            const colour = (css) => { ctx.fillStyle = css; ctx.fillRect(0, 0, 1, 1); return Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)); };
            const branch = colour(s.palette.branch), background = colour(s.palette.background);
            let crisp = 0, blurred = 0;
            const m = s.model, canvas = document.getElementById('tree-canvas');
            for (let k = 0; k < m.leafCount; k++) {
                const box = TreeView.nodePixels(s, m.leafNode[k]);
                if (box.y < 20 || box.y > 640 || box.x0 < 40 || box.x - box.x0 < 40) continue;
                // Device pixels: the stem's own row is the one the page puts its middle on.
                const x = Math.round((box.x0 + 20) * ratio), row = Math.floor(box.y * ratio);
                const at = (r) => Array.from(canvas.getContext('2d').getImageData(x, r, 1, 1).data.slice(0, 3));
                // (A step in a channel is antialiasing: at a ratio of 2 the blue of the stem reads 90 for 91.)
                const on = at(row), same = (a, b) => a.every((value, i) => Math.abs(value - b[i]) <= 3);
                const between = (a) => a.every((value, i) => Math.abs(value - background[i]) > 0 && Math.abs(value - branch[i]) > 0
                    && Math.min(branch[i], background[i]) <= value && value <= Math.max(branch[i], background[i]));
                const beside = ratio === 1 ? [[-1, background], [1, background]] : [[-2, background], [2, background]];
                const half = ratio === 1 ? [] : [at(row - 1), at(row + 1)];
                if (same(on, branch) && beside.every(([d, c]) => same(at(row + d), c)) && half.every(between)) crisp++; else blurred++;
            }
            return {crisp, blurred};""")
        self.assertGreater(info["crisp"], 5)
        self.assertLessEqual(info["blurred"], info["crisp"] * 0.2)


class PerformanceTests(TreePageMixin, unittest.TestCase):
    """The page on a tree of 44,000 leaves in three shapes: measured here, with loose budgets.

    The timings include a flush of the canvas, so they are the cost of the pixels and not only of the drawing
    commands. This engine has no GPU, so they are what a machine without one would take; the budgets are far
    above what a sane machine needs, and the measurements are what is recorded (set TREE_PERFORMANCE_FILE to a path
    to have each run's figures appended to it as JSON lines).
    """

    BUDGET_MS = {"layout": 400, "decode": 400, "frame": 1500, "pan": 1500, "hit": 150, "fold": 150, "recolor": 1500}

    MEASURE = """
        const flush = () => { document.getElementById('tree-canvas').getContext('2d').getImageData(0, 0, 1, 1); };
        const time = (f) => { const a = performance.now(); f(); flush(); return performance.now() - a; };
        const median = (values) => values.slice().sort((x, y) => x - y)[values.length >> 1];
        const payload = await (await fetch('/api/tree/data')).json();
        const out = {nodes: payload.counts.nodes, decode: time(() => TreeView.decodeTree(payload))};
        const s = TreeView.state, m = s.model;
        // The network's colours: the payload, its decoding into classes and clade summaries, and the whole of taking new
        // colours (the request, the decoding, the legend and a frame).
        const colorText = await (await fetch('/api/tree/colors')).text();
        out.colors_bytes = colorText.length;
        const colorPayload = JSON.parse(colorText);
        out.colors_decode = time(() => TreeView.decodeColors(colorPayload, m));
        const painted = TreeView.decodeColors(colorPayload, m);
        out.colors_classes = painted.classCount;
        s.colors = null;
        out.recolor = await (async () => { const a = performance.now(); await TreeView.loadColors(); flush(); return performance.now() - a; })();
        const find = (low, high) => { for (let i = 0; i < m.nodeCount; i++) { const n = m.lastLeaf[i] - m.firstLeaf[i] + 1; if (m.childCount[i] > 0 && n >= low && n <= high) return i; } return -1; };
        for (const kind of ['rectangular', 'circular', 'unrooted']) {
            const r = {};
            r.layout = time(() => TreeView.layoutTree(m, kind, 'lengths'));
            TreeView.setLayout(kind, 'lengths');
            r.first_frame = time(() => TreeView.drawNow());
            r.fit_frame = median(Array.from({length: 5}, () => time(() => TreeView.drawNow())));
            r.visited = TreeView.lastStats.visited;
            // What the colours cost a frame: the same one, in the tree's own colours.
            const coloured = s.colors;
            s.colors = null;
            r.fit_frame_plain = median(Array.from({length: 5}, () => time(() => TreeView.drawNow())));
            s.colors = coloured;
            r.hit = median(Array.from({length: 15}, (_, k) => time(() => TreeView.hitTest(300 + k * 37, 200 + k * 21))));
            TreeView.zoomAbout(650, 330, 1, 20);
            TreeView.zoomAbout(650, 330, 20, 1);
            r.pan_frame = median(Array.from({length: 15}, () => { TreeView.pan(11, 5); return time(() => TreeView.drawNow()); }));
            const clade = find(30, 60);
            if (clade >= 0) { TreeView.zoomToNode(clade); r.zoomed_frame = median(Array.from({length: 5}, () => time(() => TreeView.drawNow()))); }
            TreeView.fitView();
            const big = find(Math.max(100, m.leafCount / 50), m.leafCount / 5);
            if (big >= 0) { r.fold = time(() => TreeView.toggleFold(big)); r.fold_frame = time(() => TreeView.drawNow()); TreeView.toggleFold(big); }
            r.legend_mark = time(() => TreeView.state.matches = TreeView.leavesOfClass(s.colors, s.colors.unlabelled[0].cls));
            r.marked_frame = median(Array.from({length: 3}, () => time(() => TreeView.drawNow())));
            TreeView.state.matches = new Int32Array(0);
            r.overview = time(() => { TreeView.minimapChanged(); TreeView.drawMinimap(); });
            out[kind] = r;
        }
        return out;
    """

    def paint_by_tree_order(self, headers, block=150):
        """Colour the Viewer's nodes by their place in the tree, in clusters of ``block`` leaves, as clusters mostly are
        clades; hide one in a hundred, and give each its cluster number."""
        from matplotlib import colors as mcolors

        order = {header: k for k, header in enumerate(read_guide_tree(self.tree_file).labels)}
        ranks = np.array([order[header] for header in headers])
        cluster = ranks // block
        hues = (cluster * 0.61803398875) % 1.0
        rgba = np.ones((len(headers), 4), dtype=np.float32)
        rgba[:, :3] = mcolors.hsv_to_rgb(np.column_stack([hues, np.full(len(headers), 0.65), np.full(len(headers), 0.9)]))
        self.viewer.current_colors = rgba
        self.viewer.visible_mask = ranks % 100 != 7
        self.viewer.cluster_labels = cluster.astype(np.int32)

    def measure(self, shape, count):
        headers = headers_of(count)
        self.serve_headers(headers, linkage=shaped_linkage(shape, count))
        self.paint_by_tree_order(headers)
        started = time.perf_counter()
        self.open_page()
        ready = time.perf_counter() - started
        result = self.script(self.MEASURE, timeout=300)
        result.update(shape=shape, leaves=count, page_ready_seconds=round(ready, 3))
        path = os.environ.get("TREE_PERFORMANCE_FILE")
        if path:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
        return result

    def check(self, result):
        self.assertLess(result["decode"], self.BUDGET_MS["decode"])
        self.assertLess(result["colors_decode"], self.BUDGET_MS["decode"])
        self.assertLess(result["recolor"], self.BUDGET_MS["recolor"])
        # Four bytes and about a third of a byte a node, in base64, and the clusters: well under 400 KB at 44,000 nodes.
        self.assertLess(result["colors_bytes"], result["nodes"] * 5)
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(shape=result["shape"], leaves=result["leaves"], kind=kind):
                found = result[kind]
                self.assertLess(found["layout"], self.BUDGET_MS["layout"])
                self.assertLess(found["fit_frame"], self.BUDGET_MS["frame"])
                self.assertLess(found["first_frame"], self.BUDGET_MS["frame"] * 3)
                self.assertLess(found["marked_frame"], self.BUDGET_MS["frame"])
                self.assertLess(found["pan_frame"], self.BUDGET_MS["pan"])
                self.assertLess(found["hit"], self.BUDGET_MS["hit"])
                self.assertLess(found.get("fold", 0), self.BUDGET_MS["fold"])
        # (Reading a pixel to flush the canvas, as the timings do, makes the engine advise willReadFrequently: its own
        # advice about the measuring, not about the page.)
        self.assertEqual([entry for entry in self.console_errors() if "willReadFrequently" not in entry[1]], [])


    def test_forty_four_thousand_leaves_in_every_shape(self):
        for shape in ("bushy", "caterpillar", "chains"):
            with self.subTest(shape=shape):
                self.check(self.measure(shape, 44000))


if __name__ == "__main__":
    unittest.main()
