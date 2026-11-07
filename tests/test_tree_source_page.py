# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's chooser of trees (tree.html, tree_session.js): the guide tree, and the tree of the network's own edges.

The threshold tree is single linkage over the edges the Viewer draws (utilities/Threshold_Tree.py). The page lists
the trees in a chooser, keeps the colours, the selection and the hidden nodes across a change of tree (they go through the
nodes the leaves stand for), says in the Details panel what the tree is built from, and draws the Viewer's threshold
as a cut line that follows the generic threshold_changed event. The page is opened offscreen over the real web server;
its drawing is checked through spies on the canvas calls, the frame's statistics and its pixels.
"""

import json
import pathlib
import sys
import time
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for folder in (SRC_DIR, PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.test_tree_page import SPY  # noqa: E402
from tests.tree_fixtures import give_edges, headers_of, node_colors  # noqa: E402
from tests.tree_page_harness import TreePageMixin  # noqa: E402
from web_ui import tree_backend  # noqa: E402

COUNT = 30
CUT = 10.5


class SourceCase(TreePageMixin, unittest.TestCase):
    """A page over a network of ``COUNT`` nodes in a chain (scores 29 down to 1), with the Viewer's threshold at ``CUT``."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(tree_backend._THRESHOLD_CACHE, {"order": None, "count": 0, "built": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self, *, guide=True, extra_nodes=0, colors=False, hidden=(), select=(), expect="tree", source=None):
        headers = headers_of(COUNT)
        network = headers + [f"isolated node {k}" for k in range(extra_nodes)]
        options = {} if guide else {"tree": None}
        self.serve_headers(headers, network=network, colors=node_colors(len(network), {2: "#ff0000", 5: "#0000ff"}) if colors else None,
                           hidden=hidden, **options)
        give_edges(self.viewer, [(n, n + 1) for n in range(COUNT - 1)], np.arange(COUNT - 1, 0, -1), threshold=CUT)
        if select or colors:
            self.enable_selection()
        if select:
            self.viewer.select(select)
        if source:
            tree_backend.tree_state(self.viewer, source)
        self.open_page(expect)
        if colors:
            self.wait_for("TreeView.state.colors !== null")
        self.wait_listening()

    def wait_listening(self):
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)

    def choose(self, source):
        """Pick a tree in the chooser, as a user does, and wait until the page shows it."""
        self.script(f"""const select = document.getElementById('source-select');
            select.value = {json.dumps(source)}; select.dispatchEvent(new Event('change')); return 1""")
        self.wait_for(f"TreeView.state.source === {json.dumps(source)} && document.body.dataset.view !== 'loading'")

    def layout(self, kind, mode="lengths"):
        self.script(f"TreeView.setLayout({json.dumps(kind)}, {json.dumps(mode)}); TreeView.drawNow(); return 1")

    def cut_strokes(self):
        """The strokes of the accent colour that are a cut line: a vertical line the canvas's height, or a full circle."""
        return self.script(SPY + """
            let result;
            try {
                const stats = TreeView.drawNow();
                const s = TreeView.state, accent = TreeView.rgbOf(s.palette.accent);
                const height = document.getElementById('tree-canvas').clientHeight;
                const found = drawn.filter((e) => e.op === 'stroke' && e.main && e.style.every((v, k) => v === accent[k]));
                const lines = [], circles = [];
                for (const e of found) {
                    const p = e.parts;
                    if (p.length === 2 && p[0][0] === 'm' && p[1][0] === 'l' && p[0][1] === p[1][1] && p[0][2] === 0 && p[1][2] === height) lines.push(p[0][1]);
                    if (p.length === 1 && p[0][0] === 'a') circles.push({x: p[0][1], y: p[0][2], r: p[0][3]});
                }
                result = {cut: stats.cut, lines, circles, ox: s.view.ox, oy: s.view.oy, sx: s.view.sx};
            } finally { restore(); }
            return result;""")

    def expected_distance(self):
        return self.state("TreeView.state.cut - TreeView.state.model.threshold.root_score")


class ChooserTests(SourceCase):
    def test_the_chooser_lists_both_trees_and_the_guide_tree_is_the_default_when_there_is_one(self):
        self.start()
        found = self.state("""(() => { const s = document.getElementById('source-select');
            return {value: s.value, disabled: s.disabled, options: Array.from(s.options).map(o => [o.value, o.textContent.trim(), o.disabled]),
                    source: TreeView.state.source, score: TreeView.state.model.score !== null, leaves: TreeView.state.model.leafCount}; })()""")
        self.assertEqual(found["value"], "guide")
        self.assertFalse(found["disabled"])
        # The last entry opens a tree file: it is no tree, and is always there.
        self.assertEqual(found["options"], [["threshold", "Network threshold tree", False], ["guide", "MSA Guide Tree", False],
                                            ["open", "Open a tree file...", False]])
        self.assertEqual((found["source"], found["score"], found["leaves"]), ("guide", False, COUNT))
        self.assertEqual(self.console_errors(), [])

    def test_without_a_guide_tree_the_threshold_tree_is_the_default_and_the_guide_option_says_why(self):
        self.start(guide=False)
        found = self.state("""(() => { const s = document.getElementById('source-select');
            const guide = Array.from(s.options).find(o => o.value === 'guide');
            return {value: s.value, guideDisabled: guide.disabled, why: guide.title, score: TreeView.state.model.score !== null,
                    view: document.body.dataset.view}; })()""")
        self.assertEqual((found["value"], found["guideDisabled"], found["score"], found["view"]), ("threshold", True, True, "tree"))
        self.assertEqual(found["why"], "No guide tree was found for this alignment")
        self.assertEqual(self.console_errors(), [])

    def test_choosing_a_tree_shows_it_and_the_viewer_keeps_the_choice(self):
        self.start()
        self.choose("threshold")
        self.assertEqual(self.state("[TreeView.state.model.leafCount, TreeView.state.model.nodeCount, TreeView.state.model.rooted]"),
                         [COUNT, 2 * COUNT - 1, True])
        self.assertEqual(self.viewer._tree_source, "threshold")
        # The page opened again shows the tree the Viewer kept.
        self.open_page()
        self.assertEqual(self.state("TreeView.state.source"), "threshold")
        self.assertEqual(self.state("document.getElementById('source-select').value"), "threshold")
        self.choose("guide")
        self.assertIsNone(self.state("TreeView.state.model.score"))
        self.assertEqual(self.viewer._tree_source, "guide")
        self.assertEqual(self.console_errors(), [])

    def test_when_neither_tree_can_be_shown_the_card_says_why_and_the_chooser_stays_on(self):
        self.start(guide=False, expect="tree")
        self.viewer.edge_layer = None
        self.script("return await TreeView.loadTree()")
        self.wait_for("document.body.dataset.view === 'empty'")
        found = self.state("""({title: document.getElementById('empty-title').textContent, disabled: document.getElementById('source-select').disabled,
            options: Array.from(document.getElementById('source-select').options).map(o => o.disabled)})""")
        self.assertEqual(found["title"], "No guide tree was found for this alignment")
        self.assertFalse(found["disabled"])
        self.assertEqual(found["options"], [True, True, False])

    def test_a_network_with_no_edges_says_so_when_its_tree_is_asked_for(self):
        self.start()
        self.viewer.edge_layer = None
        self.choose("threshold")
        self.wait_for("document.body.dataset.view === 'empty'")
        self.assertEqual(self.state("document.getElementById('empty-title').textContent"), "This network has no edges")
        self.assertEqual(self.state("document.getElementById('source-select').value"), "threshold")


class DetailsTests(SourceCase):
    def test_the_details_say_what_the_tree_is_built_from(self):
        self.start(guide=False)
        text = self.state("document.getElementById('details').textContent")
        for words in ("This network's edges (single linkage)", "Similarity", "Edges scoring at least 1", "29", "One: every node is joined",
                      "Rooted", "Each branch point is where two groups of nodes first share an edge"):
            self.assertIn(words, text)
        self.assertNotIn("Alignment", text)
        self.assertNotIn("Unrooted", text)

    def test_a_tree_that_is_rooted_and_has_no_support_shows_neither_note_nor_legend(self):
        self.start(guide=False)
        found = self.state("""({legend: document.getElementById('support-legend').hidden, rooting: document.querySelector('.rooting-note') === null,
            detail: document.getElementById('details-rooting') === null})""")
        self.assertEqual(found, {"legend": True, "rooting": True, "detail": True})
        # In the unrooted layout too.
        self.layout("unrooted")
        self.assertIsNone(self.state("document.querySelector('.rooting-note')"))

    def test_a_manifold_layout_says_the_tree_is_of_the_kept_edges(self):
        import EMAPSSN_Config as cfg

        with mock.patch.object(cfg, "UMAP_MODE", True), mock.patch.object(cfg, "UMAP_NEIGHBORS", 9):
            self.start(guide=False)
            text = self.state("document.getElementById('details').textContent")
        self.assertIn("The best 9 edges of each node, from the kept edges", text)

    def test_the_tooltip_of_a_branch_point_says_the_score_it_joins_at(self):
        self.start(guide=False)
        lines = self.state("""(() => { const m = TreeView.state.model;
            let node = -1; for (let i = 0; i < m.nodeCount - 1; i++) if (m.childCount[i] > 0) { node = i; break; }
            return {inner: TreeView.describeNode(node).lines, leaf: TreeView.describeNode(0).lines, score: m.score[node]}; })()""")
        self.assertTrue(any(line.startswith("Groups join at score") for line in lines["inner"]), lines)
        self.assertFalse(any(line.startswith("Branch length") for line in lines["inner"]))
        self.assertEqual(lines["leaf"], ["A node of this network"])


class CutLineTests(SourceCase):
    def test_a_rectangular_tree_has_a_vertical_line_at_the_sliders_score(self):
        self.start(guide=False)
        found = self.cut_strokes()
        distance = self.expected_distance()
        self.assertAlmostEqual(distance, CUT - 1.0, places=4)  # the root joins at the lowest score, 1
        expected = round((found["ox"] + distance * found["sx"]) * 2) / 2
        # One path, stroked in a few hairline passes (strokeLines).
        self.assertEqual(set(found["lines"]), {expected})
        self.assertEqual(found["cut"], {"x": expected})
        self.assertEqual(found["circles"], [])

    def test_the_line_sits_between_the_branch_points_the_score_separates(self):
        self.start(guide=False)
        found = self.script("""
            const s = TreeView.state, m = s.model, L = s.layout, v = s.view, cut = s.cut;
            // A branch point scoring above the cut is left of the line, one scoring below it is right of it... in the tree's own
            // way: a node at score h is at x = ox + (h - root) * sx, so the line is right of those below the cut score.
            let left = true, right = true, checked = 0;
            for (let i = 0; i < m.nodeCount; i++) {
                if (m.childCount[i] === 0) continue;
                const x = v.ox + L.x[i] * v.sx, line = v.ox + (cut - m.threshold.root_score) * v.sx;
                if (m.score[i] < cut) { left = left && x < line; checked++; } else { right = right && x >= line - 1e-6; checked++; }
            }
            return {left, right, checked};""")
        self.assertTrue(found["left"] and found["right"] and found["checked"] > 10, found)

    def test_a_circular_tree_has_a_circle_and_an_unrooted_one_none(self):
        self.start(guide=False)
        self.layout("circular")
        found = self.cut_strokes()
        self.assertEqual(len({(c["x"], c["y"], c["r"]) for c in found["circles"]}), 1)
        circle = found["circles"][0]
        self.assertAlmostEqual(circle["r"], self.expected_distance() * found["sx"], places=3)
        self.assertAlmostEqual(circle["x"], found["ox"], places=3)
        self.assertAlmostEqual(circle["y"], found["oy"], places=3)
        self.assertEqual(found["lines"], [])
        self.layout("unrooted")
        found = self.cut_strokes()
        self.assertEqual((found["lines"], found["circles"], found["cut"]), ([], [], None))

    def test_a_cladogram_has_no_line_because_its_distances_are_steps(self):
        self.start(guide=False)
        self.layout("rectangular", "cladogram")
        found = self.cut_strokes()
        self.assertEqual((found["lines"], found["cut"]), ([], None))

    def test_the_guide_tree_has_no_line(self):
        self.start()
        found = self.cut_strokes()
        self.assertEqual((found["lines"], found["circles"], found["cut"]), ([], [], None))

    def test_the_line_is_drawn_in_the_accent_colour_and_is_there_in_the_pixels(self):
        self.start(guide=False)
        info = self.script("""
            const s = TreeView.state, canvas = document.getElementById('tree-canvas');
            const accent = TreeView.rgbOf(s.palette.accent);
            const withLine = TreeView.drawNow().cut.x;
            const column = (x) => { const ctx = canvas.getContext('2d'); const out = [];
                for (let y = 3; y < 9; y++) out.push(Array.from(ctx.getImageData(Math.floor(x), y, 1, 1).data)); return out; };
            const on = column(withLine);
            const saved = s.cut; s.cut = null; TreeView.drawNow();
            const off = column(withLine); s.cut = saved; TreeView.drawNow();
            const close = (p) => Math.abs(p[0] - accent[0]) + Math.abs(p[1] - accent[1]) + Math.abs(p[2] - accent[2]);
            return {on: on.map(close), off: off.map(close), accent};""")
        # Along the line, near the top, it is the accent colour (or within the antialiasing of it); without it, it is not.
        self.assertLess(min(info["on"]), 60)
        self.assertGreater(min(info["off"]), 200)

    def test_the_line_follows_the_threshold_event_and_nothing_else_is_asked_or_drawn(self):
        self.start(guide=False)
        before = self.cut_strokes()["lines"][0]
        self.js("""window.__draws = 0; window.__fetches = 0;
            const drawNow = TreeView.drawNow; TreeView.drawNow = function () { window.__draws++; return drawNow.apply(this, arguments); };
            const fetchOriginal = window.fetch; window.fetch = function () { window.__fetches++; return fetchOriginal.apply(this, arguments); }; 1""")
        self.viewer.broadcast_event({"type": "threshold_changed", "value": 20.0})
        self.wait_for("TreeView.state.cut === 20")
        self.pump(0.4)
        self.assertEqual(self.state("[window.__draws, window.__fetches]"), [1, 0])
        after = self.cut_strokes()["lines"][0]
        self.assertGreater(after, before)
        # Other events and unusable values move nothing and draw nothing.
        self.js("window.__draws = 0; 1")
        for event in ({"type": "state_updated"}, {"type": "threshold_changed", "value": "high"}, {"type": "threshold_changed"},
                      {"type": "threshold_changed", "value": None}):
            self.viewer.broadcast_event(event)
        self.pump(0.5)
        self.assertEqual(self.state("[TreeView.state.cut, window.__draws]"), [20, 0])
        self.assertEqual(self.console_errors(), [])

    def test_the_threshold_event_with_the_guide_tree_draws_nothing(self):
        self.start()
        self.js("window.__draws = 0; const d = TreeView.drawNow; TreeView.drawNow = function () { window.__draws++; return d.apply(this, arguments); }; 1")
        self.viewer.broadcast_event({"type": "threshold_changed", "value": 3.0})
        self.wait_for("TreeView.state.cut === 3")
        self.pump(0.3)
        self.assertEqual(self.state("window.__draws"), 0)

    def test_the_line_is_where_the_viewers_slider_is_when_the_page_opens(self):
        self.start(guide=False)
        self.assertEqual(self.state("TreeView.state.cut"), CUT)
        self.assertEqual(self.state("TreeView.state.model.threshold.current"), CUT)

    def test_a_score_below_the_root_or_above_the_leaves_has_no_line_in_view(self):
        self.start(guide=False)
        self.script("TreeView.state.cut = 0.5; return 1")
        self.assertIsNone(self.cut_strokes()["cut"])
        self.script("TreeView.state.cut = 1e6; return 1")
        self.assertIsNone(self.cut_strokes()["cut"])

    def test_a_frame_draws_the_line_in_a_few_calls(self):
        self.start(guide=False)
        calls = self.script(SPY + """
            let result;
            try {
                const before = drawn.length; TreeView.drawNow(); const withLine = drawn.length - before;
                const saved = TreeView.state.cut; TreeView.state.cut = null;
                const middle = drawn.length; TreeView.drawNow(); const without = drawn.length - middle;
                TreeView.state.cut = saved;
                result = {withLine, without};
            } finally { restore(); }
            return result;""")
        # The line is one path stroked in a handful of hairline passes (strokeLines): a few more calls, not one for each node.
        self.assertLessEqual(calls["withLine"] - calls["without"], 5)
        self.assertGreater(calls["withLine"], calls["without"])


class ForestTests(SourceCase):
    def test_groups_that_share_no_edge_hang_from_one_root_and_the_page_says_how_many(self):
        self.start(guide=False, extra_nodes=10)
        found = self.state("""({forest: TreeView.state.model.threshold.forest, groups: TreeView.state.model.threshold.groups,
            rootChildren: TreeView.state.model.childCount[TreeView.state.model.root],
            lines: TreeView.describeNode(TreeView.state.model.root).lines,
            details: document.getElementById('details').textContent})""")
        self.assertEqual((found["forest"], found["groups"], found["rootChildren"]), (True, 11, 11))
        self.assertIn("11 groups share no kept edge", found["lines"])
        self.assertIn("11 groups share no kept edge", found["details"])
        self.assertIn("the broken branches", found["details"])

    def test_the_branches_of_the_root_are_drawn_broken_and_a_connected_network_has_none(self):
        self.start(guide=False, extra_nodes=10)
        found = self.script("""
            const s = TreeView.state, m = s.model, canvas = document.getElementById('tree-canvas');
            const stats = TreeView.drawNow();
            const background = TreeView.rgbOf(s.palette.background);
            let child = -1;
            for (let k = m.childStart[m.root]; k < m.childStart[m.root + 1]; k++) { child = m.childList[k]; if (m.childCount[child] === 0) break; }
            const b = TreeView.nodePixels(s, child);
            const mx = Math.round((b.x0 + b.x) / 2), my = Math.floor(b.y);
            const ratio = canvas.width / canvas.clientWidth;
            const pixel = () => Array.from(canvas.getContext('2d').getImageData(Math.round(mx * ratio), Math.round(my * ratio), 1, 1).data).slice(0, 3);
            const broken = pixel();
            m.threshold.forest = false; TreeView.drawNow(); const whole = pixel(); m.threshold.forest = true;
            return {breaks: stats.breaks, background, broken, whole, length: b.x - b.x0};""")
        self.assertGreaterEqual(found["breaks"], 1)
        self.assertEqual(found["broken"], found["background"])
        self.assertNotEqual(found["whole"], found["background"])
        self.start(guide=False)
        self.assertEqual(self.state("TreeView.drawNow().breaks"), 0)

    def test_a_connected_network_says_one_group(self):
        self.start(guide=False)
        text = self.state("document.getElementById('details').textContent")
        self.assertIn("One: every node is joined", text)
        self.assertNotIn("share no kept edge", text)


class CarryOverTests(SourceCase):
    def test_colours_selection_and_hidden_nodes_carry_over_a_change_of_tree_and_folds_and_view_do_not(self):
        self.start(colors=True, hidden=[3, 4], select=[1, 2])
        self.wait_for("TreeView.state.selection.count === 2")
        before = self.state("""({legend: TreeView.state.colors.legend.map(e => e.css), removed: TreeView.state.removed.reduce((a, v) => a + v, 0)})""")
        self.assertEqual(before["removed"], 2)
        # Fold a clade and move the view, then change the tree.
        self.script("""const m = TreeView.state.model; for (let i = 0; i < m.nodeCount - 1; i++) if (m.childCount[i] > 0) { TreeView.toggleFold(i); break; }
            TreeView.state.view.sx *= 1.5; TreeView.state.fitted = false; return TreeView.state.collapsedCount""")
        self.assertGreaterEqual(self.state("TreeView.state.collapsedCount"), 1)
        self.choose("threshold")
        self.wait_for("TreeView.state.colors !== null && TreeView.state.selection.count === 2 && TreeView.state.removed !== null")
        after = self.state("""(() => { const s = TreeView.state, m = s.model;
            const nodesOf = (leaves) => Array.from(leaves, (k) => m.viewerNode[k]).sort((a, b) => a - b);
            const removed = []; for (let k = 0; k < m.leafCount; k++) if (s.removed[k]) removed.push(m.viewerNode[k]);
            return {legend: s.colors.legend.map(e => e.css), selected: nodesOf(s.selection.leaves), removed: removed.sort((a, b) => a - b),
                    folds: s.collapsedCount, fitted: s.fitted, source: s.source,
                    red: m.viewerNode.findIndex(n => n === 2), class: s.colors.leafClass[m.viewerNode.findIndex(n => n === 2)]}; })()""")
        self.assertEqual(sorted(after["legend"]), sorted(before["legend"]))
        self.assertEqual(after["selected"], [1, 2])
        self.assertEqual(after["removed"], [3, 4])
        self.assertEqual((after["folds"], after["fitted"], after["source"]), (0, True, "threshold"))
        self.assertGreater(after["class"], 0)  # the red node's leaf has a colour class of its own
        # And back to the guide tree: the same again.
        self.choose("guide")
        self.wait_for("TreeView.state.colors !== null && TreeView.state.selection.count === 2")
        self.assertEqual(sorted(self.state("TreeView.state.colors.legend.map(e => e.css)")), sorted(before["legend"]))
        self.assertEqual(self.console_errors(), [])

    def test_the_viewer_hiding_a_node_leaves_it_out_of_the_threshold_tree_too(self):
        self.start(guide=False, colors=True)
        self.assertEqual(self.state("TreeView.state.model.leafCount"), COUNT)
        self.viewer.visible_mask[[0, 1]] = False
        self.viewer.broadcast_event({"type": "tree_colors_changed", "version": 12345})
        self.wait_for("TreeView.state.removed !== null")
        self.assertEqual(self.state("TreeView.drawNow().visited > 0"), True)
        self.assertEqual(self.state("TreeView.state.layout.live[TreeView.state.model.leafNode[TreeView.state.model.viewerNode.indexOf(0)]]"), 0)


if __name__ == "__main__":
    unittest.main()
