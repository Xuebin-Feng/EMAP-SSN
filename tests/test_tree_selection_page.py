# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's links to the network's selection (src/web_ui/tree_select.js), and the stability of the splits.

The Viewer tells every page which nodes are selected (selection_changed) and which has the click focus
(highlight_row); the page shows them as a layer of its own on the leaves, outlines the smallest clade that holds
the selection, and, asked to, tells the Viewer to select the nodes of a clade or to focus on a leaf (the tree_select
and tree_focus actions, which the Viewer's own selecting follows). The page is opened offscreen over the real web
server and a Viewer that has the little a selection needs (tests/tree_page_harness.py: enable_selection), and
checked through its state and its pixels.
"""

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

from tests.test_tree_page import EIGHT_LEAVES, SPY, write_tree_file  # noqa: E402
from tests.tree_fixtures import headers_of, node_colors, random_linkage, shaped_linkage, writer  # noqa: E402
from tests.tree_page_harness import TreePageMixin  # noqa: E402

KINDS = ("rectangular", "circular", "unrooted")

# Spies on the context calls the selection is drawn with: each stroke, stroke of a rectangle and arc of the main
# canvas, with the stroke colour, whether the line was dashed, and the arguments.
DRAWING_SPY = """
    const proto = CanvasRenderingContext2D.prototype;
    const original = {stroke: proto.stroke, strokeRect: proto.strokeRect, arc: proto.arc};
    const log = [];
    const entry = (op, context, extra) => {
        if (context.canvas.id !== 'tree-canvas') return;
        log.push({op, style: TreeView.rgbOf(context.strokeStyle), dashed: context.getLineDash().length > 0, ...extra});
    };
    proto.stroke = function () { entry('stroke', this, {}); return original.stroke.apply(this, arguments); };
    proto.strokeRect = function (x, y, w, h) { entry('strokeRect', this, {x, y, w, h}); return original.strokeRect.apply(this, arguments); };
    proto.arc = function (x, y, r) { entry('arc', this, {x, y, r}); return original.arc.apply(this, arguments); };
    const restore = () => Object.assign(proto, original);
    const selectedRgb = () => TreeView.rgbOf(TreeView.state.palette.selected);
    const same = (a, b) => a.length === b.length && a.every((value, k) => value === b[k]);
"""


class SelectionCase(TreePageMixin, unittest.TestCase):
    """A page over a tree of ``COUNT`` leaves whose network the Viewer has, with a selection a command can make."""

    COUNT = 40

    def start(self, *, preselect=(), headers=None, network=None, hidden=(), tree_options=None):
        headers = headers or headers_of(self.COUNT)
        count = len(network or headers)
        self.serve_headers(headers, network=network, colors=node_colors(count), hidden=hidden, **(tree_options or {}))
        self.enable_selection()
        if preselect:
            self.viewer.select(preselect)
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)
        self.listen()

    def listen(self):
        """Count the times the page takes a selection or a focus from the Viewer, in window.__heard."""
        self.js("""window.__heard = 0; const previous = TreeView.onSelectionChanged;
            TreeView.onSelectionChanged = () => { window.__heard++; previous(); }; 1""")

    def heard(self):
        return self.state("window.__heard")

    def after(self, act):
        """Run ``act`` (which makes the Viewer tell the page something) and wait until the page has taken it."""
        before = self.heard()
        act()
        self.wait_for(f"window.__heard > {before}")

    def select(self, indices):
        """What a command's selecting does, then wait until the page has taken it."""
        self.after(lambda: self.viewer.select(indices))

    def focus(self, index):
        self.after(lambda: self.viewer.apply_left_click_focus(index))

    def clear_focus(self):
        self.after(self.viewer.clear_left_click_focus)

    def nodes_of(self, leaves):
        """The Viewer's nodes the leaves (numbers in tree order) stand for."""
        return self.state(f"{json.dumps(list(leaves))}.map(k => TreeView.state.model.viewerNode[k])")

    def choose(self, kind, mode="lengths"):
        self.script(f"TreeView.setLayout({json.dumps(kind)}, {json.dumps(mode)}); TreeView.drawNow(); return 1")

    def selection(self):
        return self.state("""(() => { const s = TreeView.state.selection;
            return {leaves: Array.from(s.leaves), clade: s.clade, focus: s.focus, count: s.count}; })()""")

    def spied(self, body):
        return self.script(DRAWING_SPY + f"""
            let result;
            try {{ TreeView.drawNow(); result = (() => {{ {body} }})(); }} finally {{ restore(); }}
            return result;""")


class FollowTheNetworkTests(SelectionCase):
    """What the network selects is what the tree shows selected."""

    def test_the_selected_nodes_are_found_among_the_leaves_in_tree_order(self):
        self.start()
        picked = self.nodes_of([30, 4, 17, 5, 22])
        self.select(picked)
        found = self.selection()
        self.assertEqual(found["leaves"], [4, 5, 17, 22, 30])
        self.assertEqual(found["count"], 5)

    def test_the_smallest_clade_holding_the_selection_is_the_one_found_by_looking_at_every_clade(self):
        self.start()
        for leaves in ([4, 5], [4, 30], [0, 39], [10, 11, 12], [7, 20, 33]):
            with self.subTest(leaves=leaves):
                self.select(self.nodes_of(leaves))
                found = self.selection()
                brute = self.state("""(() => { const m = TreeView.state.model, L = TreeView.state.selection.leaves;
                    let best = -1;
                    for (let i = 0; i < m.nodeCount; i++) {
                        if (m.firstLeaf[i] <= L[0] && m.lastLeaf[i] >= L[L.length - 1] && (best < 0 || m.size[i] < m.size[best])) best = i;
                    }
                    return best; })()""")
                self.assertEqual(found["clade"], brute)
                self.assertGreaterEqual(brute, 0)

    def test_no_leaf_and_one_leaf_have_no_clade(self):
        self.start()
        self.select(self.nodes_of([9]))
        self.assertEqual(self.selection()["clade"], -1)
        self.select([])
        self.assertEqual((self.selection()["clade"], self.selection()["leaves"]), (-1, []))

    def test_the_whole_network_selected_marks_the_root(self):
        self.start()
        self.select(list(range(self.COUNT)))
        self.assertEqual(self.selection()["clade"], self.state("TreeView.state.model.root"))

    def test_clearing_the_selection_clears_the_layer(self):
        self.start()
        self.select(self.nodes_of([3, 4]))
        self.select([])
        self.assertEqual(self.selection(), {"leaves": [], "clade": -1, "focus": -1, "count": 0})

    def test_a_page_opened_after_the_selection_was_made_finds_it(self):
        self.start()
        picked = self.nodes_of([6, 7, 12])
        self.viewer.selected_indices = picked
        self.viewer.selected_node_idx = picked[0]
        self.open_page()
        self.wait_for("TreeView.state.selection.count === 3")
        self.listen()
        self.assertEqual(self.selection()["leaves"], [6, 7, 12])
        self.assertEqual(self.selection()["focus"], 6)

    def test_nodes_that_are_not_leaves_of_the_tree_are_counted_but_not_drawn(self):
        headers = headers_of(30)
        extra = [f"extra_{k}" for k in range(5)]
        self.start(headers=headers, network=headers + extra)
        self.select([0, 1, 30, 31, 34])
        found = self.selection()
        self.assertEqual((found["count"], len(found["leaves"])), (5, 2))
        self.assertIn("5 selected in the network, 2 of them in this tree", self.state("document.getElementById('summary').textContent"))

    def test_a_repeated_node_is_one_leaf(self):
        self.start()
        node = self.nodes_of([3])[0]
        self.select([node, node, node])
        self.assertEqual(len(self.selection()["leaves"]), 1)

    def test_the_status_line_says_how_many_nodes_are_selected_and_the_pinned_node_has_it(self):
        self.start()
        self.assertNotIn("selected in the network", self.state("document.getElementById('summary').textContent"))
        self.select(self.nodes_of([3, 4, 5]))
        self.assertIn("3 selected in the network", self.state("document.getElementById('summary').textContent"))
        self.select([])
        self.assertNotIn("selected in the network", self.state("document.getElementById('summary').textContent"))
        # A node that is pinned is what the line is about; the clade says how many of its leaves are selected.
        self.select(self.nodes_of([3, 4, 5]))
        said = self.state("""(() => { const m = TreeView.state.model; TreeView.state.pinned = TreeView.state.selection.clade;
            return TreeView.describeNode(TreeView.state.selection.clade).lines; })()""")
        self.assertIn("Selected in the network: 3", said)
        leaf = self.state("TreeView.describeNode(TreeView.state.model.leafNode[4]).lines")
        self.assertIn("Selected in the network", leaf)
        other = self.state("TreeView.describeNode(TreeView.state.model.leafNode[20]).lines")
        self.assertNotIn("Selected in the network", other)

    def test_the_selection_is_a_layer_of_its_own_beside_the_search_and_the_legend(self):
        self.start()
        found = self.state("TreeView.runSearch('prot_1')")
        self.assertGreater(found, 1)
        matches = self.state("Array.from(TreeView.state.matches)")
        self.select(self.nodes_of([3, 4]))
        self.assertEqual(self.state("Array.from(TreeView.state.matches)"), matches)
        self.assertEqual(self.selection()["leaves"], [3, 4])
        # A search after the selection leaves the selection.
        self.state("TreeView.runSearch('prot_2')")
        self.assertEqual(self.selection()["leaves"], [3, 4])
        self.select([])
        self.assertGreater(len(self.state("Array.from(TreeView.state.matches)")), 0)

    def test_the_selection_stays_when_the_layout_changes_and_a_clade_is_folded(self):
        self.start()
        self.select(self.nodes_of([3, 4, 20]))
        before = self.selection()
        for kind in KINDS:
            self.choose(kind)
            self.assertEqual(self.selection(), before)
        self.choose("rectangular")
        self.script("TreeView.toggleFold(TreeView.state.selection.clade); TreeView.drawNow(); return 1")
        self.assertEqual(self.selection(), before)

    def test_an_answer_older_than_an_event_does_not_take_the_events_place(self):
        self.start()
        # The page asks for the selection; the answer is held back, and a selection event comes first.
        self.js("""window.__real = window.fetch; window.__release = null;
            window.fetch = (url, options) => String(url).startsWith('/api/tree/selection')
                ? new Promise((resolve) => { window.__release = () => resolve(new Response(JSON.stringify(
                    {status: 'ok', count: 0, indices: '', focus: null}), {status: 200})); })
                : window.__real(url, options);
            window.__loaded = null; TreeView.loadSelection().then((value) => { window.__loaded = value; }); 1""")
        self.wait_for("window.__release !== null")
        self.select(self.nodes_of([3, 4]))
        self.js("window.__release(); 1")
        self.wait_for("window.__loaded !== null")
        self.assertEqual(self.state("window.__loaded"), False)
        self.assertEqual(self.selection()["leaves"], [3, 4])
        self.js("window.fetch = window.__real; 1")

    def test_a_loaded_selection_with_no_event_is_taken(self):
        self.start()
        picked = self.nodes_of([8, 9])
        self.viewer.selected_indices = picked
        self.assertEqual(self.state("TreeView.loadSelection()"), True)
        self.assertEqual(self.selection()["leaves"], [8, 9])

    def test_an_event_that_is_not_a_list_changes_nothing_and_breaks_nothing(self):
        self.start()
        self.select(self.nodes_of([3, 4]))
        self.viewer.broadcast_event({"type": "selection_changed", "indices": "nope"})
        self.viewer.broadcast_event({"type": "highlight_row", "index": "nope"})
        self.pump(0.4)
        self.assertEqual(self.selection()["leaves"], [3, 4])
        self.assertEqual(self.console_errors(), [])

    def test_the_first_big_message_stays_unparsed(self):
        # The page reads four kinds of event by their first words, and a message of any other kind is not parsed.
        self.start()
        self.script("""
            window.__parsed = 0; window.__realParse = JSON.parse;
            JSON.parse = function () { window.__parsed++; return window.__realParse.apply(this, arguments); };
            return 1""")
        self.viewer.broadcast_event({"type": "init", "rows": [[0] * 50 for _ in range(50)]})
        self.viewer.broadcast_event({"type": "something_else", "indices": [1, 2, 3]})
        self.pump(0.4)
        parsed = self.state("window.__parsed")
        self.js("JSON.parse = window.__realParse; 1")
        self.assertEqual(parsed, 0)


class ShownOnTheTreeTests(SelectionCase):
    """The selection is drawn in every layout."""

    def test_selected_leaves_are_dots_in_the_selection_colour_in_every_layout(self):
        self.start()
        self.select(self.nodes_of([3, 4, 5, 6]))
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                result = self.script("""
                    const s = TreeView.state, v = s.view, L = s.layout, m = s.model;
                    const stats = TreeView.drawNow();
                    const at = (k) => { const n = m.leafNode[k]; return fire.pixel(v.ox + L.x[n] * v.sx, v.oy + L.y[n] * v.sy); };
                    return {marked: stats.selected, wanted: TreeView.rgbOf(s.palette.selected), picked: [3, 4, 5, 6].map(at),
                            others: [0, 12, 25, 39].map(at)};""")
                self.assertEqual(result["marked"], 4)
                for pixel in result["picked"]:
                    self.assertEqual(pixel[:3], result["wanted"])
                for pixel in result["others"]:
                    self.assertNotEqual(pixel[:3], result["wanted"])

    def test_the_selection_colour_is_the_themes_and_differs_from_the_searchs(self):
        self.start()
        palette = self.state("TreeView.state.palette")
        self.assertNotEqual(palette["selected"], palette["match"])
        self.js("document.documentElement.setAttribute('data-theme', 'dark'); 1")
        self.wait_for("TreeView.state.palette.selected !== " + json.dumps(palette["selected"]))
        self.js("document.documentElement.setAttribute('data-theme', 'light'); 1")

    def test_the_smallest_clade_is_outlined_dashed_in_every_layout_and_a_single_leaf_has_no_outline(self):
        self.start()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.select(self.nodes_of([3, 4, 20]))
                dashed = self.spied("return log.filter(e => e.dashed && same(e.style, selectedRgb())).length")
                self.assertGreaterEqual(dashed, 1)
                self.select(self.nodes_of([3]))
                self.assertEqual(self.spied("return log.filter(e => e.dashed).length"), 0)
                self.select([])

    def test_a_rectangular_clade_is_outlined_round_its_rows(self):
        self.start()
        self.choose("rectangular")
        self.select(self.nodes_of([3, 4, 5]))
        result = self.spied("""
            const rects = log.filter(e => e.op === 'strokeRect' && e.dashed);
            const box = TreeView.nodePixels(TreeView.state, TreeView.state.selection.clade);
            return {count: rects.length, top: rects[0].y, bottom: rects[0].y + rects[0].h, left: rects[0].x, boxTop: box.top, boxBottom: box.bottom, boxLeft: box.x0};""")
        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["top"], result["boxTop"] - 2, delta=0.5)
        self.assertAlmostEqual(result["bottom"], result["boxBottom"] + 2, delta=0.5)
        self.assertAlmostEqual(result["left"], result["boxLeft"] - 3, delta=0.5)

    def test_a_clade_many_screens_across_is_outlined_unbroken_and_a_small_one_dashed(self):
        self.start()
        self.select(self.nodes_of([3, 4, 20]))
        for kind in ("circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                self.js("TreeView.DASH_LIMIT_PX = 1; 1")
                self.assertEqual(self.spied("return [log.filter(e => e.dashed).length, log.filter(e => e.op === 'stroke' && same(e.style, selectedRgb())).length]")[0], 0)
                self.assertGreaterEqual(self.spied("return log.filter(e => e.op === 'stroke' && same(e.style, selectedRgb())).length"), 1)
                self.js("TreeView.DASH_LIMIT_PX = 6000; 1")
                self.assertGreaterEqual(self.spied("return log.filter(e => e.dashed).length"), 1)

    def test_the_dashes_are_taken_off_after_the_outline(self):
        self.start()
        self.select(self.nodes_of([3, 4, 20]))
        self.assertEqual(self.state("document.getElementById('tree-canvas').getContext('2d').getLineDash().length"), 0)

    def test_the_click_focus_is_a_ring_round_its_leaf(self):
        self.start()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                node = self.nodes_of([12])[0]
                self.focus(node)
                found = self.spied("""
                    const n = TreeView.state.model.leafNode[12], v = TreeView.state.view, L = TreeView.state.layout;
                    const ring = log.filter(e => e.op === 'arc' && e.r === 9);
                    return {rings: ring.length, x: ring[0] && ring[0].x - (v.ox + L.x[n] * v.sx), y: ring[0] && ring[0].y - (v.oy + L.y[n] * v.sy)};""")
                self.assertEqual(found["rings"], 1)
                self.assertAlmostEqual(found["x"], 0, delta=0.01)
                self.assertAlmostEqual(found["y"], 0, delta=0.01)
                self.clear_focus()
                self.assertEqual(self.spied("return log.filter(e => e.op === 'arc' && e.r === 9).length"), 0)

    def test_the_focus_follows_highlight_row_and_the_selection_does_not_move_it(self):
        self.start()
        node = self.nodes_of([12])[0]
        self.focus(node)
        self.select(self.nodes_of([3, 4]))
        self.assertEqual(self.selection()["focus"], 12)
        self.assertEqual(self.selection()["leaves"], [3, 4])
        self.clear_focus()
        self.assertEqual(self.selection()["leaves"], [3, 4])

    def test_a_folded_clade_that_holds_selected_leaves_is_outlined_in_the_selection_colour(self):
        self.start()
        self.choose("rectangular")
        self.select(self.nodes_of([3, 4]))
        clade = self.selection()["clade"]
        self.script(f"TreeView.openAll(); TreeView.toggleFold({clade}); return 1")
        result = self.spied("""
            const m = TreeView.state.model;
            const strokes = log.filter(e => e.op === 'stroke' && same(e.style, selectedRgb()));
            return {strokes: strokes.length, folded: TreeView.state.collapsedCount, dots: TreeView.lastStats.selected};""")
        self.assertEqual(result["folded"], 1)
        self.assertGreaterEqual(result["strokes"], 1)
        # The leaves inside it have no tips to mark.
        self.assertEqual(result["dots"], 0)
        # Open again, and the tips are marked and the outline is not a stroke of the wedge.
        self.script(f"TreeView.toggleFold({clade}); return 1")
        self.assertEqual(self.state("TreeView.drawNow().selected"), 2)

    def test_a_folded_clade_with_no_selected_leaf_is_not_outlined(self):
        self.start()
        self.choose("rectangular")
        self.select(self.nodes_of([3, 4]))
        other = self.state("""(() => { const m = TreeView.state.model;
            for (let i = 0; i < m.nodeCount; i++) if (m.childCount[i] > 1 && m.firstLeaf[i] >= 20 && m.lastLeaf[i] - m.firstLeaf[i] >= 2) return i; return -1; })()""")
        self.script(f"TreeView.openAll(); TreeView.toggleFold({other}); return 1")
        self.assertEqual(self.spied("return log.filter(e => e.op === 'stroke' && same(e.style, selectedRgb()) && !e.dashed).length"), 0)

    def test_the_overview_and_the_ruler_show_the_selection(self):
        self.start()
        self.choose("rectangular")
        counts = "const found = (canvas, x0, x1) => { const w = canvas.width, h = canvas.height; const d = canvas.getContext('2d').getImageData(0, 0, w, h).data; let n = 0; " \
                 "const r = TreeView.rgbOf(TreeView.state.palette.selected); for (let y = 0; y < h; y++) for (let x = x0(w); x < x1(w); x++) { const i = (y * w + x) * 4; " \
                 "if (Math.abs(d[i] - r[0]) < 12 && Math.abs(d[i + 1] - r[1]) < 12 && Math.abs(d[i + 2] - r[2]) < 12) n++; } return n; };"
        probe = counts + "TreeView.drawNow(); return {map: found(document.getElementById('minimap'), w => 0, w => w), ruler: found(document.getElementById('tree-canvas'), w => w - 14, w => w)};"
        self.assertEqual(self.script(probe), {"map": 0, "ruler": 0})
        self.select(self.nodes_of([3, 4, 20, 30]))
        found = self.script(probe)
        self.assertGreater(found["map"], 0)
        self.assertGreater(found["ruler"], 0)
        self.select([])
        self.assertEqual(self.script(probe), {"map": 0, "ruler": 0})

    def test_an_unselected_page_draws_nothing_of_the_selection(self):
        self.start()
        self.assertEqual(self.spied("return [log.filter(e => e.dashed).length, log.filter(e => e.op === 'arc' && e.r === 9).length, TreeView.lastStats.selected]"), [0, 0, 0])
        self.assertEqual(self.console_errors(), [])

    def test_a_selection_in_a_thin_layout_is_drawn_once_a_pixel(self):
        self.start(headers=headers_of(3000), tree_options={"linkage": random_linkage(3000, seed=5)})
        self.choose("rectangular")
        self.select(list(range(3000)))
        marked = self.state("TreeView.drawNow().selected")
        height = self.state("document.getElementById('tree-canvas').clientHeight")
        self.assertLessEqual(marked, height + 8)
        self.assertGreater(marked, 100)
        # The clades too small to be more than a line are outlined, in the selection's colour, as one that holds a match is.
        self.assertGreaterEqual(self.spied("return log.filter(e => e.op === 'stroke' && !e.dashed && same(e.style, selectedRgb())).length"), 1)
        self.assertGreater(self.state("TreeView.lastStats.bars"), 0)

    def test_a_new_tree_takes_the_old_ones_selection_away_at_once(self):
        self.start()
        self.select(self.nodes_of([3, 4]))
        # The answer to the page's question about the selection is held, to see what the page shows before it comes.
        shown = self.script("""
            TreeView.loadSelection = async () => false;
            const payload = await (await fetch('/api/tree/data')).json();
            TreeView.applyPayload(payload);
            return {leaves: TreeView.state.selection.leaves.length, clade: TreeView.state.selection.clade, count: TreeView.state.selection.count};""")
        self.assertEqual(shown, {"leaves": 0, "clade": -1, "count": 0})


class AskTheNetworkTests(SelectionCase):
    """A clade's nodes are selected in the network, and a pinned leaf is focused there, through the two actions."""

    def ask(self, node, mode="replace"):
        """Ask the Viewer, through the page, to select a clade's nodes; wait until its action has run."""
        before = len(self.viewer.action_results)
        sent = self.script(f"return await TreeView.selectClade({node}, {json.dumps(mode)})")
        if sent["sent"]:
            deadline = time.monotonic() + 10
            while len(self.viewer.action_results) <= before and time.monotonic() < deadline:
                self.pump(0.02)
        return sent

    def leaf_node(self, leaf):
        return self.state(f"TreeView.state.model.leafNode[{leaf}]")

    def clade_over(self, first, last):
        return self.state(f"""(() => {{ const m = TreeView.state.model; let best = -1;
            for (let i = 0; i < m.nodeCount; i++) if (m.firstLeaf[i] === {first} && m.lastLeaf[i] === {last}) best = i;
            return best; }})()""")

    def test_a_clade_replaces_the_networks_selection_and_the_page_hears_of_it(self):
        self.start()
        self.viewer.select([0, 1])
        clade = self.state("TreeView.state.model.parent[TreeView.state.model.leafNode[10]]")
        first, last = self.state(f"[TreeView.state.model.firstLeaf[{clade}], TreeView.state.model.lastLeaf[{clade}]]")
        sent = self.ask(clade)
        self.assertEqual((sent["sent"], sent["ok"], sent["count"]), (True, True, last - first + 1))
        self.assertEqual(self.viewer.action_results[-1][0], "tree_select")
        self.assertEqual(self.viewer.action_results[-1][1]["changed"], True)
        self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of(range(first, last + 1))))
        # A hand selection, as a box selection is; and the page's layer follows the event.
        self.assertEqual(self.viewer.hand_selections, 1)
        self.wait_for(f"TreeView.state.selection.leaves.length === {last - first + 1}")
        self.assertEqual(self.selection()["clade"], clade)

    def test_a_leaf_is_a_clade_of_one(self):
        self.start()
        sent = self.ask(self.leaf_node(7))
        self.assertEqual((sent["sent"], sent["count"]), (True, 1))
        self.assertEqual(self.viewer.selected_indices, self.nodes_of([7]))

    def test_add_and_toggle_combine_with_the_selection(self):
        self.start()
        self.ask(self.leaf_node(3))
        self.ask(self.leaf_node(5), "add")
        self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of([3, 5])))
        self.ask(self.leaf_node(5), "toggle")
        self.assertEqual(self.viewer.selected_indices, self.nodes_of([3]))
        self.ask(self.leaf_node(6), "toggle")
        self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of([3, 6])))

    def hide(self, nodes):
        """The Viewer hides these nodes, and the page takes the colours (and with them which nodes are hidden) again."""
        self.viewer.visible_mask[nodes] = False
        self.script("TreeView.state.colors = null; await TreeView.loadColors(); return 1")

    def test_a_clade_of_hidden_nodes_sends_nothing(self):
        self.start()
        both = self.state("""(() => { const m = TreeView.state.model;
            for (let i = 0; i < m.nodeCount; i++) if (m.childCount[i] === 2 && m.size[i] === 3) return i; return -1; })()""")
        self.assertGreaterEqual(both, 0)
        first, last = self.state(f"[TreeView.state.model.firstLeaf[{both}], TreeView.state.model.lastLeaf[{both}]]")
        self.hide(self.nodes_of([first, last]))
        self.assertEqual(self.state(f"TreeView.nodesOfClade({both}).length"), 0)
        sent = self.script(f"return await TreeView.selectClade({self.leaf_node(first)}, 'replace')")
        self.assertEqual(sent, {"sent": False, "count": 0})
        self.assertEqual(self.script(f"return await TreeView.selectClade({both}, 'replace')"), {"sent": False, "count": 0})
        self.pump(0.3)
        self.assertEqual(self.viewer.action_results, [])

    def test_a_clade_with_some_hidden_nodes_sends_the_others(self):
        self.start()
        clade = self.state("TreeView.state.model.parent[TreeView.state.model.leafNode[10]]")
        first, last = self.state(f"[TreeView.state.model.firstLeaf[{clade}], TreeView.state.model.lastLeaf[{clade}]]")
        nodes = self.nodes_of(range(first, last + 1))
        self.hide([nodes[0]])
        sent = self.ask(clade)
        self.assertEqual((sent["sent"], sent["count"]), (True, len(nodes) - 1))
        self.assertEqual(sorted(self.viewer.selected_indices), sorted(nodes[1:]))

    def test_a_node_that_is_not_in_the_network_is_not_sent(self):
        headers = headers_of(30)
        self.start(headers=headers, network=headers[:20])
        absent = self.state("(() => { const m = TreeView.state.model; for (let k = 0; k < m.leafCount; k++) if (m.viewerNode[k] < 0) return k; return -1; })()")
        self.assertGreaterEqual(absent, 0)
        sent = self.script(f"return await TreeView.selectClade(TreeView.state.model.leafNode[{absent}], 'replace')")
        self.assertEqual(sent["sent"], False)

    def test_a_bad_node_number_sends_nothing(self):
        self.start()
        for node in (-1, 99999, "null"):
            with self.subTest(node=node):
                self.assertEqual(self.script(f"return await TreeView.selectClade({node})"), {"sent": False, "count": 0})
        self.assertEqual(self.viewer.action_results, [])

    def test_pinning_a_leaf_focuses_its_node_in_the_network_and_unpinning_takes_it_off(self):
        self.start()
        self.choose("rectangular")
        x, y = self.state("""(() => { const s = TreeView.state, n = s.model.leafNode[12]; return [s.view.ox + s.layout.x[n] * s.view.sx, s.view.oy + s.layout.y[n] * s.view.sy]; })()""")
        self.script(f"fire.click({x}, {y}); return 1")
        deadline = time.monotonic() + 10
        while self.viewer.selected_node_idx is None and time.monotonic() < deadline:
            self.pump(0.02)
        self.assertEqual(self.viewer.selected_node_idx, self.nodes_of([12])[0])
        self.assertEqual(self.viewer.action_results[-1][0], "tree_focus")
        # The ring comes from the Viewer's own highlight_row.
        self.wait_for("TreeView.state.selection.focus === 12")
        # The selection is not touched by a focus.
        self.assertEqual(self.viewer.selected_indices, [])
        self.script("fire.key('Escape'); return 1")
        deadline = time.monotonic() + 10
        while self.viewer.selected_node_idx is not None and time.monotonic() < deadline:
            self.pump(0.02)
        self.assertIsNone(self.viewer.selected_node_idx)
        self.wait_for("TreeView.state.selection.focus === -1")

    def test_pinning_a_clade_asks_nothing_and_pinning_another_leaf_moves_the_focus(self):
        self.start()
        self.choose("rectangular")
        clade = self.state("TreeView.state.model.parent[TreeView.state.model.leafNode[10]]")
        x, y = self.state(f"""(() => {{ const b = TreeView.nodePixels(TreeView.state, {clade}); return [(b.x0 + b.x) / 2, b.y]; }})()""")
        self.script(f"fire.click({x}, {y}); return 1")
        self.assertEqual(self.state("TreeView.state.pinned >= 0 && TreeView.state.model.childCount[TreeView.state.pinned] > 0"), True)
        self.pump(0.3)
        self.assertEqual(self.viewer.action_results, [])
        # A leaf, then another: the focus is asked for each, and is the second's.
        for leaf in (12, 20):
            px, py = self.state(f"""(() => {{ const s = TreeView.state, n = s.model.leafNode[{leaf}]; return [s.view.ox + s.layout.x[n] * s.view.sx, s.view.oy + s.layout.y[n] * s.view.sy]; }})()""")
            before = len(self.viewer.action_results)
            self.script(f"fire.click({px}, {py}); return 1")
            deadline = time.monotonic() + 10
            while len(self.viewer.action_results) == before and time.monotonic() < deadline:
                self.pump(0.02)
        self.assertEqual(self.viewer.selected_node_idx, self.nodes_of([20])[0])
        self.assertEqual([name for name, _ in self.viewer.action_results], ["tree_focus", "tree_focus"])

    def test_a_pinned_leaf_that_is_not_in_the_network_asks_nothing_new(self):
        headers = headers_of(30)
        self.start(headers=headers, network=headers[:20])
        self.choose("rectangular")
        absent = self.state("(() => { const m = TreeView.state.model; for (let k = 0; k < m.leafCount; k++) if (m.viewerNode[k] < 0) return k; return -1; })()")
        x, y = self.state(f"""(() => {{ const s = TreeView.state, n = s.model.leafNode[{absent}]; return [s.view.ox + s.layout.x[n] * s.view.sx, s.view.oy + s.layout.y[n] * s.view.sy]; }})()""")
        self.script(f"fire.click({x}, {y}); return 1")
        self.pump(0.3)
        self.assertEqual(self.viewer.action_results, [])
        self.assertIsNone(self.viewer.selected_node_idx)


class CladesOfTheEightLeavesTests(SelectionCase):
    """The smallest clade, on a tree small enough to know: ((a,b),(c,d)),((e,f),(g,h)), nodes in post-order."""

    def setUp(self):
        super().setUp()
        headers = list("abcdefgh")
        folder = self.make_folder()
        write_tree_file(folder, headers, EIGHT_LEAVES)
        self.serve_headers(headers, folder=folder, tree=None, colors=node_colors(8))
        self.enable_selection()
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)
        self.listen()

    def test_each_selection_marks_the_clade_that_holds_it(self):
        # a0 b1 (ab)2 c3 d4 (cd)5 (abcd)6 e7 f8 (ef)9 g10 h11 (gh)12 (efgh)13 root14
        cases = [([0, 1], 2), ([0, 2], 6), ([0, 1, 2, 3], 6), ([2, 3], 5), ([4, 5], 9), ([6, 7], 12), ([4, 7], 13), ([0, 7], 14),
                 ([3, 4], 14), ([0], -1), ([], -1)]
        for leaves, clade in cases:
            with self.subTest(leaves=leaves):
                self.select(leaves)
                self.assertEqual(self.state("TreeView.state.selection.clade"), clade)

    def test_a_leaf_that_is_a_clades_own_clade_stays_out_of_the_outline(self):
        self.select([5])
        self.assertEqual(self.state("TreeView.state.selection.leaves.length"), 1)


class StabilityOfTheSplitsTests(TreePageMixin, unittest.TestCase):
    """Dots on the clades, shaded by how many replicate trees hold the split: where there is room, when asked for."""

    COUNT = 40

    def start(self, replicates=True):
        tool = writer()
        headers = headers_of(self.COUNT)
        link = random_linkage(self.COUNT, seed=3)
        options = {}
        if replicates:
            keys = tool.guide_tree_leaf_keys(self.COUNT, 42)
            # A few replicate trees that hold half the splits each, so that the percentages differ from clade to clade.
            all_keys = tool.replicate_split_keys(link, self.COUNT, keys)
            replicate_keys = np.sort(np.concatenate([all_keys, all_keys[::2], all_keys[::4]]))
            options = {"leaf_keys": keys, "replicate_keys": replicate_keys, "num_trees": 3}
        self.serve_headers(headers, linkage=link, **options)
        self.open_page()

    def dots(self):
        """The dots a frame draws (arcs of radius SUPPORT_RADIUS in a path that is filled), with the colour each is filled in."""
        return self.script(SPY + """
            let result;
            try {
                TreeView.drawNow();
                const radius = TreeView.SUPPORT_RADIUS;
                result = drawn.filter(e => e.op === 'fill' && e.main && e.parts.some(p => p[0] === 'a' && p[3] === radius))
                    .map(e => ({style: e.style, at: e.parts.filter(p => p[0] === 'a').map(p => [p[1], p[2]])}));
            } finally { restore(); }
            return result;""")

    def test_the_dots_are_off_until_asked_for_and_the_button_is_a_switch(self):
        self.start()
        self.assertEqual(self.state("TreeView.drawNow().supports"), 0)
        self.assertEqual(self.dots(), [])
        self.assertEqual(self.state("document.getElementById('support-btn').getAttribute('aria-pressed')"), "false")
        self.assertEqual(self.state("document.getElementById('support-body').hidden"), True)
        self.js("document.getElementById('support-btn').click(); 1")
        self.assertEqual(self.state("TreeView.state.showSupport"), True)
        self.assertEqual(self.state("document.getElementById('support-btn').getAttribute('aria-pressed')"), "true")
        self.assertEqual(self.state("document.getElementById('support-body').hidden"), False)
        self.assertGreater(self.state("TreeView.drawNow().supports"), 0)
        self.js("document.getElementById('support-btn').click(); 1")
        self.assertEqual(self.state("TreeView.drawNow().supports"), 0)

    def test_a_dot_is_drawn_on_each_clade_that_has_room_and_a_value_and_nowhere_else(self):
        self.start()
        self.js("TreeView.state.showSupport = true; 1")
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.script(f"TreeView.setLayout({json.dumps(kind)}, 'lengths'); TreeView.drawNow(); return 1")
                expected = self.state("""(() => { const s = TreeView.state, m = s.model, b = s.layout.boxes, v = s.view, out = [];
                    const stack = [m.root];
                    const stem = new Float64Array(8);
                    while (stack.length) {
                        const i = stack.pop();
                        const w = (b.x1[i] - b.x0[i]) * v.sx, h = (b.y1[i] - b.y0[i]) * v.sy;
                        if (s.layout.tiny(i, v.sx, v.sy, TreeView.COLLAPSE_PX) || s.collapsed[i] || m.childCount[i] === 0) continue;
                        if (i !== m.root && m.support[i] >= 0 && Math.min(w, h) >= TreeView.SUPPORT_CLADE_PX) {
                            s.layout.stem(i, stem);
                            out.push([m.support[i], Math.round((v.ox + stem[2] * v.sx) * 100) / 100, Math.round((v.oy + stem[3] * v.sy) * 100) / 100]);
                        }
                        for (let k = m.childStart[i]; k < m.childStart[i + 1]; k++) stack.push(m.childList[k]);
                    }
                    return out; })()""")
                self.assertGreater(len(expected), 3)
                drawn = self.dots()
                points = sorted((x, y) for entry in drawn for x, y in entry["at"])
                wanted = sorted((x, y) for _, x, y in expected)
                self.assertEqual(len(points), len(wanted))
                for found, want in zip(points, wanted):
                    self.assertAlmostEqual(found[0], want[0], delta=0.02)
                    self.assertAlmostEqual(found[1], want[1], delta=0.02)

    def test_each_dot_is_shaded_by_its_value_from_pale_to_the_accent_colour(self):
        self.start()
        self.js("TreeView.state.showSupport = true; 1")
        palette = self.state("TreeView.state.palette.support.map(c => TreeView.rgbOf(c))")
        self.assertEqual(len(set(map(tuple, palette))), 5)
        # Every fill is one of the five steps, and a dot's step is its value's fifth (100 is the last step).
        by_step = self.script("""
            const s = TreeView.state, m = s.model, v = s.view, stem = new Float64Array(8);
            const wanted = new Map();
            for (let i = 0; i < m.nodeCount; i++) {
                if (m.childCount[i] === 0 || i === m.root || m.support[i] < 0) continue;
                s.layout.stem(i, stem);
                wanted.set(Math.round((v.ox + stem[2] * v.sx) * 100) + ',' + Math.round((v.oy + stem[3] * v.sy) * 100), Math.min(4, Math.floor(m.support[i] / 20)));
            }
            return Array.from(wanted);""")
        wanted = dict(by_step)
        seen = 0
        for entry in self.dots():
            step = palette.index(entry["style"])
            for x, y in entry["at"]:
                key = f"{round(x * 100)},{round(y * 100)}"
                if key in wanted:
                    seen += 1
                    self.assertEqual(wanted[key], step, key)
        self.assertGreater(seen, 3)
        self.assertGreater(len({palette.index(entry["style"]) for entry in self.dots()}), 1)

    def test_a_clade_drawn_small_or_folded_has_no_dot_and_zooming_back_brings_them(self):
        self.start()
        self.js("TreeView.state.showSupport = true; 1")
        fit = self.state("TreeView.drawNow().supports")
        self.assertGreater(fit, 3)
        # Zoomed out, the clades have less room and fewer have a dot.
        self.script("TreeView.zoomAbout(600, 300, 0.3, 0.3); return 1")
        self.assertLess(self.state("TreeView.drawNow().supports"), fit)
        self.script("TreeView.fitView(); return 1")
        # A clade folded up is one shape: its dot, and those of the clades inside it, are not drawn.
        folded = self.state("""(() => { const m = TreeView.state.model; let big = -1;
            for (let i = 0; i < m.root; i++) if (m.childCount[i] > 1 && (big < 0 || m.size[i] > m.size[big])) big = i;
            TreeView.toggleFold(big); return TreeView.drawNow().supports; })()""")
        self.assertLess(folded, fit)
        self.script("TreeView.openAll(); TreeView.fitView(); return 1")
        self.assertEqual(self.state("TreeView.drawNow().supports"), fit)

    def test_the_root_has_no_dot_whatever_its_label_says(self):
        headers = list("abcd")
        folder = self.make_folder()
        write_tree_file(folder, headers, "#NEXUS\nBEGIN TREES;\n  TREE t = [&R] ((a:1,b:1)90:1,(c:1,d:1)80:1)100;\nEND;\n")
        self.serve_headers(headers, folder=folder, tree=None)
        self.open_page()
        self.assertEqual(self.state("TreeView.state.model.support[TreeView.state.model.root]"), 100)
        self.js("TreeView.state.showSupport = true; 1")
        self.assertEqual(self.state("TreeView.drawNow().supports"), 2)

    def test_a_frame_draws_at_most_the_cap(self):
        self.start()
        self.js("TreeView.state.showSupport = true; TreeView.SUPPORT_MAX = 3; 1")
        self.assertEqual(self.state("TreeView.drawNow().supports"), 3)

    def test_the_legend_has_the_ramp_the_ends_and_the_sentence_and_never_the_word_as_a_label(self):
        self.start()
        legend = self.state("document.getElementById('support-legend').hidden")
        self.assertEqual(legend, False)
        self.js("document.getElementById('support-btn').click(); 1")
        text = self.state("document.getElementById('support-legend').innerText")
        self.assertIn("Stability under distance noise, not bootstrap support.", text)
        self.assertEqual(text.lower().count("bootstrap"), 1)
        self.assertIn("Split stability", text)
        self.assertEqual(self.state("document.getElementById('support-ramp').children.length"), 5)
        colours = self.state("Array.from(document.getElementById('support-ramp').children, e => e.style.backgroundColor)")
        self.assertEqual(len(set(colours)), 5)
        # Nothing else on the page, tooltips included, is labelled with it.
        others = self.state("""(() => { const s = TreeView.state, m = s.model, out = [];
            for (let i = 0; i < m.nodeCount; i++) if (TreeView.describeNode(i).lines.join(' ').toLowerCase().includes('bootstrap')) out.push(i);
            return out; })()""")
        self.assertEqual(others, [])
        self.assertNotIn("bootstrap", self.state("document.getElementById('support-btn').textContent + document.getElementById('support-btn').title").lower())

    def test_a_tree_that_says_nothing_of_stability_has_no_switch_and_no_dots(self):
        self.start(replicates=False)
        self.assertEqual(self.state("TreeView.hasSupport(TreeView.state.model)"), False)
        self.assertEqual(self.state("document.getElementById('support-legend').hidden"), True)
        self.js("TreeView.state.showSupport = true; 1")
        self.assertEqual(self.state("TreeView.drawNow().supports"), 0)


class SelectionPerformanceTests(TreePageMixin, unittest.TestCase):
    """A network of 44,000 nodes all selected: what the page costs to take it and to draw it (software raster, no GPU)."""

    MEASURE = """
        const flush = () => { document.getElementById('tree-canvas').getContext('2d').getImageData(0, 0, 1, 1); };
        const time = (f) => { const a = performance.now(); f(); flush(); return performance.now() - a; };
        const median = (values) => values.slice().sort((x, y) => x - y)[values.length >> 1];
        const s = TreeView.state, m = s.model;
        const find = (low, high) => { for (let i = 0; i < m.nodeCount; i++) { const n = m.lastLeaf[i] - m.firstLeaf[i] + 1; if (m.childCount[i] > 0 && n >= low && n <= high) return i; } return -1; };
        const n = m.leafCount;
        // The Viewer's node numbers in an order of its own, as the Viewer's selection comes.
        let seeded = 99;
        const order = Array.from({length: n}, (_, k) => k);
        for (let k = n - 1; k > 0; k--) { seeded = (seeded * 1103515245 + 12345) % 2147483648; const j = seeded % (k + 1); [order[k], order[j]] = [order[j], order[k]]; }
        const text = JSON.stringify({type: 'selection_changed', indices: order});
        const out = {message_bytes: text.length};
        let parsed;
        out.parse = time(() => { parsed = JSON.parse(text); });
        out.apply = time(() => TreeView.applySelection(parsed.indices));
        out.selected = s.selection.leaves.length;
        out.clade = s.selection.clade;
        // The route a page opened later asks: the request, the decoding and taking it.
        out.route = await (async () => { const a = performance.now(); await TreeView.loadSelection(); flush(); return performance.now() - a; })();
        for (const kind of ['rectangular', 'circular', 'unrooted']) {
            TreeView.setLayout(kind, 'lengths');
            const r = {};
            r.first_frame = time(() => TreeView.drawNow());
            r.fit_frame = median(Array.from({length: 5}, () => time(() => TreeView.drawNow())));
            const none = s.selection;
            s.selection = {leaves: new Int32Array(0), clade: -1, focus: -1, count: 0};
            r.fit_frame_unselected = median(Array.from({length: 5}, () => time(() => TreeView.drawNow())));
            s.selection = none;
            TreeView.drawNow();
            r.marked = TreeView.lastStats.selected;
            const clade = find(30, 60);
            if (clade >= 0) TreeView.zoomToNode(clade);
            r.zoomed_frame = median(Array.from({length: 5}, () => time(() => TreeView.drawNow())));
            s.selection = {leaves: new Int32Array(0), clade: -1, focus: -1, count: 0};
            r.zoomed_frame_unselected = median(Array.from({length: 5}, () => time(() => TreeView.drawNow())));
            s.selection = none;
            TreeView.fitView();
            r.overview = time(() => { TreeView.minimapChanged(); TreeView.drawMinimap(); });
            out[kind] = r;
        }
        return out;
    """

    def measure(self, shape, count):
        headers = headers_of(count)
        self.serve_headers(headers, linkage=shaped_linkage(shape, count), colors=node_colors(count))
        self.enable_selection()
        self.viewer.selected_indices = list(range(count))
        self.open_page()
        self.wait_for("TreeView.state.selection.count === " + str(count), timeout=60)
        self.js("TreeView.clearSelection(); 1")
        result = self.script(self.MEASURE, timeout=300)
        result.update(shape=shape, leaves=count, test="selection")
        path = os.environ.get("TREE_PERFORMANCE_FILE")
        if path:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
        return result

    def test_forty_four_thousand_nodes_all_selected(self):
        for shape in ("bushy", "caterpillar"):
            with self.subTest(shape=shape):
                result = self.measure(shape, 44000)
                self.assertEqual(result["selected"], 44000)
                self.assertLess(result["parse"], 400)
                self.assertLess(result["apply"], 400)
                self.assertLess(result["route"], 1500)
                for kind in KINDS:
                    with self.subTest(kind=kind):
                        self.assertLess(result[kind]["fit_frame"], 1500)
                        self.assertLess(result[kind]["zoomed_frame"], 1500)
                        self.assertLess(result[kind]["overview"], 1500)
                self.assertEqual([entry for entry in self.console_errors() if "willReadFrequently" not in entry[1]], [])


if __name__ == "__main__":
    unittest.main()
