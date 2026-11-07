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
        log.push({op, style: TreeView.rgbOf(context.strokeStyle), width: context.lineWidth, dashed: context.getLineDash().length > 0, ...extra});
    };
    proto.stroke = function () { entry('stroke', this, {}); return original.stroke.apply(this, arguments); };
    proto.strokeRect = function (x, y, w, h) { entry('strokeRect', this, {x, y, w, h}); return original.strokeRect.apply(this, arguments); };
    proto.arc = function (x, y, r) { entry('arc', this, {x, y, r}); return original.arc.apply(this, arguments); };
    const restore = () => Object.assign(proto, original);
    const selectedRgb = () => TreeView.rgbOf(TreeView.state.marks.selected);
    const same = (a, b) => a.length === b.length && a.every((value, k) => value === b[k]);
    // The box drawn with the right button: a solid rectangle in the accent colour (a clade's outline is dashed).
    const isBox = (e) => e.op === 'strokeRect' && !e.dashed && same(e.style, TreeView.rgbOf(TreeView.state.palette.accent));
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

    def test_selected_leaves_are_hollow_rings_in_the_selection_colour_in_every_layout(self):
        self.start()
        self.select(self.nodes_of([3, 4, 5, 6]))
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                result = self.script("""
                    const s = TreeView.state, v = s.view, L = s.layout, m = s.model;
                    const stats = TreeView.drawNow();
                    const wanted = TreeView.rgbOf(s.marks.selected);
                    // The pixels round a tip, 10 pixels out each way: a ring has none in colour at its centre.
                    const line = (k) => { const n = m.leafNode[k], x = Math.round(v.ox + L.x[n] * v.sx), y = Math.round(v.oy + L.y[n] * v.sy);
                        const out = []; for (let dy = -10; dy <= 10; dy++) for (let dx = -10; dx <= 10; dx++) out.push([dx, dy, fire.pixel(x + dx, y + dy).slice(0, 3)]);
                        return out; };
                    const coloured = (k) => line(k).filter(([dx, dy, p]) => p.every((c, i) => Math.abs(c - wanted[i]) < 40)).map(([dx, dy]) => [dx, dy]);
                    return {marked: stats.selected, thin: L.kind === 'rectangular' && v.sy < 6, picked: [3, 4, 5, 6].map(coloured), others: [0, 12, 25, 39].map(coloured)};""")
                self.assertEqual(result["marked"], 4)
                for hits in result["picked"]:
                    self.assertTrue(hits, "a mark in the selection colour at the tip")
                    if not result["thin"] and kind != "unrooted":  # unrooted rings of neighbouring tips can cross
                        self.assertNotIn([0, 0], hits, "hollow: the tip itself is not filled")
                for hits in result["others"]:
                    self.assertEqual(hits, [])

    def test_the_selection_colour_is_the_viewers_and_differs_from_the_searchs(self):
        self.start()
        found = self.state("""({marks: TreeView.state.marks, palette: Object.keys(TreeView.state.palette),
            selected: TreeView.rgbOf(TreeView.state.marks.selected), match: TreeView.rgbOf(TreeView.state.palette.match)})""")
        # The Viewer outlines what is selected in its highlight colour, and the page has no colour of its own for it.
        self.assertEqual(found["marks"]["selected"], "rgb(255, 170, 0)")
        self.assertNotIn("selected", found["palette"])
        self.assertNotEqual(found["selected"], found["match"])
        light = self.state("TreeView.state.palette.background")
        self.js("document.documentElement.setAttribute('data-theme', 'dark'); 1")
        self.wait_for("TreeView.state.palette.background !== " + json.dumps(light))
        self.assertEqual(self.state("TreeView.state.marks.selected"), "rgb(255, 170, 0)")
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
                    const ring = log.filter(e => e.op === 'arc' && e.r === 11);
                    return {rings: ring.length, x: ring[0] && ring[0].x - (v.ox + L.x[n] * v.sx), y: ring[0] && ring[0].y - (v.oy + L.y[n] * v.sy)};""")
                self.assertEqual(found["rings"], 1)
                self.assertAlmostEqual(found["x"], 0, delta=0.01)
                self.assertAlmostEqual(found["y"], 0, delta=0.01)
                self.clear_focus()
                self.assertEqual(self.spied("return log.filter(e => e.op === 'arc' && e.r === 11).length"), 0)

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
        self.assertEqual(self.spied("return log.filter(e => e.op === 'stroke' && same(e.style, selectedRgb()) && !e.dashed && e.width > 1).length"), 0)

    def test_the_overview_and_the_ruler_show_the_selection(self):
        self.start()
        self.choose("rectangular")
        counts = "const found = (canvas, x0, x1) => { const w = canvas.width, h = canvas.height; const d = canvas.getContext('2d').getImageData(0, 0, w, h).data; let n = 0; " \
                 "const r = TreeView.rgbOf(TreeView.state.marks.selected); for (let y = 0; y < h; y++) for (let x = x0(w); x < x1(w); x++) { const i = (y * w + x) * 4; " \
                 "if (Math.abs(d[i] - r[0]) < 30 && Math.abs(d[i + 1] - r[1]) < 30 && Math.abs(d[i + 2] - r[2]) < 30) n++; } return n; };"
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
        self.assertEqual(self.spied("return [log.filter(e => e.dashed).length, log.filter(e => e.op === 'arc' && e.r === 11).length, TreeView.lastStats.selected]"), [0, 0, 0])
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


class RightClickTests(SelectionCase):
    """A right-click on the tree selects in the network: replace, with Shift toggle, on empty canvas clear."""

    RIGHT = """
        window.__bodies = [];
        if (!window.__fetch) {
            window.__fetch = window.fetch;
            window.fetch = (url, options) => { if (String(url) === '/api/action') window.__bodies.push(JSON.parse(options.body)); return window.__fetch(url, options); };
        }
        window.rightClick = (x, y, shift = false) => {
            const c = document.getElementById('tree-canvas'), r = c.getBoundingClientRect();
            const at = {clientX: r.left + x, clientY: r.top + y, bubbles: true, cancelable: true, shiftKey: shift};
            c.dispatchEvent(new PointerEvent('pointerdown', {pointerId: 7, button: 2, buttons: 2, ...at}));
            const menu = new MouseEvent('contextmenu', {button: 2, ...at});
            c.dispatchEvent(menu);
            c.dispatchEvent(new PointerEvent('pointerup', {pointerId: 7, button: 2, buttons: 0, ...at}));
            return menu.defaultPrevented;
        };
        window.tip = (leaf) => { const s = TreeView.state, n = s.model.leafNode[leaf]; return [s.view.ox + s.layout.x[n] * s.view.sx, s.view.oy + s.layout.y[n] * s.view.sy]; };
        window.stem = (node) => { const b = TreeView.nodePixels(TreeView.state, node); return [(b.x0 + b.x) / 2, (b.y0 + b.y) / 2]; };
        1"""

    def start_right(self):
        self.start()
        self.js(self.RIGHT)

    def right(self, point, shift=False):
        """Right-click at the JavaScript expression ``point`` (an [x, y]); wait until the Viewer has run the action."""
        before = len(self.viewer.action_results)
        prevented = self.script(f"const [x, y] = {point}; return window.rightClick(x, y, {json.dumps(shift)})")
        deadline = time.monotonic() + 10
        while len(self.viewer.action_results) == before and time.monotonic() < deadline:
            self.pump(0.02)
        self.assertEqual(len(self.viewer.action_results), before + 1)
        self.pump(0.1)
        return prevented

    def under(self, node):
        first, last = self.state(f"[TreeView.state.model.firstLeaf[{node}], TreeView.state.model.lastLeaf[{node}]]")
        return sorted(self.nodes_of(range(first, last + 1)))

    def internal(self):
        return self.state("TreeView.state.model.parent[TreeView.state.model.leafNode[10]]")

    def test_a_leaf_and_a_clade_replace_the_selection_in_every_layout(self):
        self.start_right()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.viewer.select([0, 1, 2])
                self.assertTrue(self.right("window.tip(12)"))
                self.assertEqual(self.viewer.selected_indices, self.nodes_of([12]))
                clade = self.internal()
                hit = self.state(f"(() => {{ const [x, y] = window.stem({clade}); const h = TreeView.hitTest(x, y); return h && h.node; }})()")
                self.assertGreater(self.state(f"TreeView.state.model.childCount[{hit}]"), 0)
                self.right(f"window.stem({clade})")
                self.assertEqual(sorted(self.viewer.selected_indices), self.under(hit))

    def test_shift_toggles_the_nodes_in_every_layout(self):
        self.start_right()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.viewer.select([])
                self.right("window.tip(3)", shift=True)
                self.assertEqual(self.viewer.selected_indices, self.nodes_of([3]))
                self.right("window.tip(30)", shift=True)
                self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of([3, 30])))
                self.right("window.tip(3)", shift=True)
                self.assertEqual(self.viewer.selected_indices, self.nodes_of([30]))

    def test_empty_canvas_clears_the_selection_in_every_layout(self):
        self.start_right()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.viewer.select([0, 1, 2])
                empty = self.state("""(() => { const c = document.getElementById('tree-canvas');
                    for (let y = 4; y < c.clientHeight; y += 9) for (let x = 4; x < c.clientWidth - 20; x += 9) if (!TreeView.hitTest(x, y)) return [x, y];
                    return null; })()""")
                self.assertIsNotNone(empty)
                self.right(json.dumps(empty))
                self.assertEqual(self.viewer.selected_indices, [])
                self.assertEqual(self.viewer.action_results[-1][1]["changed"], True)

    def test_a_folded_clade_is_selected_whole(self):
        self.start_right()
        self.choose("rectangular")
        clade = self.internal()
        self.script(f"TreeView.openAll(); TreeView.toggleFold({clade}); TreeView.drawNow(); return 1")
        self.right(f"window.stem({clade})")
        self.assertEqual(sorted(self.viewer.selected_indices), self.under(clade))
        self.assertEqual(self.state("TreeView.state.collapsedCount"), 1)

    def test_the_browsers_menu_stays_shut_and_nothing_else_happens(self):
        self.start_right()
        self.choose("rectangular")
        clade = self.internal()
        snapshot = "(() => { const s = TreeView.state; return {pinned: s.pinned, view: s.view, folds: s.collapsedCount, fitted: s.fitted}; })()"
        before = self.state(snapshot)
        for point in ("window.tip(12)", f"window.stem({clade})", "[5, 5]"):
            self.assertTrue(self.right(point))
        self.assertEqual(self.state(snapshot), before)
        # Only selections were asked for: no focus, and the canvas is not left in a drag.
        self.assertEqual({name for name, _ in self.viewer.action_results}, {"tree_select"})
        self.assertEqual(self.state("document.getElementById('tree-canvas').classList.contains('dragging')"), False)
        self.assertIsNone(self.viewer.selected_node_idx)

    def test_the_menu_is_shut_on_the_canvas_only(self):
        self.start_right()
        self.assertEqual(self.state("(() => { const e = new MouseEvent('contextmenu', {bubbles: true, cancelable: true}); document.getElementById('legend').dispatchEvent(e); return e.defaultPrevented; })()"), False)

    def test_the_request_the_page_posts(self):
        self.start_right()
        self.choose("rectangular")
        self.right("window.tip(12)")
        self.right("window.tip(5)", shift=True)
        self.right("[5, 5]")
        self.assertEqual(self.state("window.__bodies"), [
            {"action": "tree_select", "indices": self.nodes_of([12]), "mode": "replace"},
            {"action": "tree_select", "indices": self.nodes_of([5]), "mode": "toggle"},
            {"action": "tree_select", "indices": [], "mode": "replace"},
        ])

    def pin_leaf(self, leaf):
        """Left-click a leaf: it is pinned, and the Viewer's click focus (and the page's ring) is on its node."""
        x, y = self.state(f"window.tip({leaf})")
        self.script(f"fire.click({x}, {y}); return 1")
        deadline = time.monotonic() + 10
        while self.viewer.selected_node_idx is None and time.monotonic() < deadline:
            self.pump(0.02)
        self.assertEqual(self.viewer.selected_node_idx, self.nodes_of([leaf])[0])
        self.wait_for(f"TreeView.state.selection.focus === {leaf}")
        self.assertGreaterEqual(self.state("TreeView.state.pinned"), 0)

    def right_after_pin(self, point):
        """Right-click at ``point`` and wait for its two actions: the focus taken off, and the selection."""
        before = len(self.viewer.action_results)
        self.script(f"const [x, y] = {point}; return window.rightClick(x, y)")
        deadline = time.monotonic() + 10
        while len(self.viewer.action_results) < before + 2 and time.monotonic() < deadline:
            self.pump(0.02)
        self.pump(0.1)
        return [(name, result) for name, result in self.viewer.action_results[before:]]

    def test_a_right_click_on_a_leaf_unpins_and_takes_the_focus_off(self):
        self.start_right()
        self.choose("rectangular")
        self.pin_leaf(12)
        self.script("window.__bodies.length = 0; return 1")
        self.right_after_pin("window.tip(12)")
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.assertEqual(self.state("window.__bodies"), [
            {"action": "tree_focus", "index": None},
            {"action": "tree_select", "indices": self.nodes_of([12]), "mode": "replace"},
        ])
        self.assertIsNone(self.viewer.selected_node_idx)
        self.assertEqual(self.viewer.selected_indices, self.nodes_of([12]))        # it still selects
        self.wait_for("TreeView.state.selection.focus === -1")

    def test_a_right_click_on_empty_canvas_unpins_and_takes_the_focus_off(self):
        self.start_right()
        self.choose("rectangular")
        self.viewer.select([0, 1])
        self.pin_leaf(12)
        self.script("window.__bodies.length = 0; return 1")
        empty = self.state("""(() => { const c = document.getElementById('tree-canvas');
            for (let y = 4; y < c.clientHeight; y += 9) for (let x = 4; x < c.clientWidth - 20; x += 9) if (!TreeView.hitTest(x, y)) return [x, y];
            return null; })()""")
        self.assertIsNotNone(empty)
        self.right_after_pin(json.dumps(empty))
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.assertEqual(self.state("window.__bodies"), [
            {"action": "tree_focus", "index": None},
            {"action": "tree_select", "indices": [], "mode": "replace"},
        ])
        self.assertIsNone(self.viewer.selected_node_idx)
        self.assertEqual(self.viewer.selected_indices, [])

    def test_a_right_click_takes_off_a_focus_the_network_put_on_though_nothing_is_pinned(self):
        self.start_right()
        self.choose("rectangular")
        self.focus(self.nodes_of([12])[0])                             # a left click in the network
        self.wait_for("TreeView.state.selection.focus === 12")
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.right_after_pin("window.tip(20)")
        self.assertIsNone(self.viewer.selected_node_idx)
        self.wait_for("TreeView.state.selection.focus === -1")

    def network_ring(self):
        """A left click in the network put a ring on leaf 12; nothing is pinned here; the page's bodies are cleared."""
        self.start_right()
        self.choose("rectangular")
        self.focus(self.nodes_of([12])[0])
        self.wait_for("TreeView.state.selection.focus === 12")
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.script("window.__bodies.length = 0; return 1")

    def test_folding_a_clade_leaves_a_ring_the_network_put_on(self):
        self.network_ring()
        clade = self.internal()
        self.script(f"TreeView.openAll(); TreeView.toggleFold({clade}); TreeView.drawNow(); return 1")
        x, y = self.state(f"""(() => {{ const s = TreeView.state, v = s.view, o = new Float64Array(8); s.layout.mark({clade}, o);
            return [v.ox + (o[0] + o[2]) / 2 * v.sx, v.oy + (o[1] + o[3]) / 2 * v.sy]; }})()""")
        self.script(f"fire.click({x}, {y}); return 1")                 # a folded clade opens when it is clicked
        self.wait_for("TreeView.state.collapsedCount === 0")
        self.pump(0.3)
        self.assertEqual(self.state("window.__bodies"), [])
        self.assertEqual(self.viewer.selected_node_idx, self.nodes_of([12])[0])

    def test_escape_leaves_a_ring_the_network_put_on(self):
        self.network_ring()
        self.script("fire.key('Escape'); return 1")
        self.pump(0.3)
        self.assertEqual(self.state("window.__bodies"), [])
        self.assertEqual(self.viewer.selected_node_idx, self.nodes_of([12])[0])

    def test_a_left_click_on_empty_canvas_takes_a_ring_the_network_put_on_off(self):
        self.network_ring()
        empty = self.state("""(() => { const c = document.getElementById('tree-canvas');
            for (let y = 4; y < c.clientHeight; y += 9) for (let x = 4; x < c.clientWidth - 20; x += 9) if (!TreeView.hitTest(x, y)) return [x, y];
            return null; })()""")
        self.script(f"fire.click({empty[0]}, {empty[1]}); return 1")
        deadline = time.monotonic() + 10
        while self.viewer.selected_node_idx is not None and time.monotonic() < deadline:
            self.pump(0.02)
        self.assertIsNone(self.viewer.selected_node_idx)
        self.assertEqual(self.state("window.__bodies"), [{"action": "tree_focus", "index": None}])

    def test_a_right_click_on_a_clade_unpins_a_pinned_leaf(self):
        self.start_right()
        self.choose("rectangular")
        self.pin_leaf(12)
        clade = self.internal()
        self.right_after_pin(f"window.stem({clade})")
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.assertIsNone(self.viewer.selected_node_idx)

    def test_the_ruler_is_not_the_tree(self):
        self.start_right()
        self.choose("rectangular")
        self.js("window.rightClick(document.getElementById('tree-canvas').clientWidth - 5, 100); 1")
        self.pump(0.3)
        self.assertEqual(self.viewer.action_results, [])

    def test_the_hints_say_it(self):
        self.start_right()
        for kind in ("rectangular", "circular"):
            self.choose(kind)
            self.assertIn("Right-click or right-drag: select in the network.", self.state("TreeView.hintText()"))


class BoxSelectionTests(SelectionCase):
    """Dragging with the right button draws a box that selects the leaves whose tips are in it, as the Viewer's does."""

    RIGHT = RightClickTests.RIGHT
    start_right = RightClickTests.start_right
    pin_leaf = RightClickTests.pin_leaf
    internal = RightClickTests.internal

    HELPERS = """
        window.mouse = (type, x, y, shift = false, extra = {}) => {
            const c = document.getElementById('tree-canvas'), r = c.getBoundingClientRect();
            c.dispatchEvent(new PointerEvent(type, {pointerId: 7, button: type === 'pointermove' ? -1 : 2,
                buttons: type === 'pointerup' ? 0 : 2, clientX: r.left + x, clientY: r.top + y, bubbles: true, cancelable: true,
                shiftKey: shift, ...extra}));
        };
        window.tips = () => { const s = TreeView.state, m = s.model, out = [];
            for (let k = 0; k < m.leafCount; k++) { const n = m.leafNode[k]; out.push([s.view.ox + s.layout.x[n] * s.view.sx, s.view.oy + s.layout.y[n] * s.view.sy]); }
            return out; };
        1"""

    def start_box(self, **options):
        self.start(**options)
        self.js(self.RIGHT)
        self.js(self.HELPERS)

    def mouse(self, kind, point, shift=False, **extra):
        self.script(f"window.mouse({json.dumps(kind)}, {point[0]}, {point[1]}, {json.dumps(shift)}, {json.dumps(extra)}); return 1")

    def drag(self, start, end, shift=False, release=True):
        """Press the right button at ``start`` (shift on the press), move to ``end`` in steps, and let go there."""
        self.mouse("pointerdown", start, shift)
        steps = 4
        for step in range(1, steps + 1):
            self.mouse("pointermove", (start[0] + (end[0] - start[0]) * step / steps, start[1] + (end[1] - start[1]) * step / steps), shift)
        if release:
            self.release(end, shift)

    def release(self, point, shift=False, expect=None):
        """Let go at ``point``; wait for the Viewer to run what the page asked (``expect`` actions, or none at all)."""
        before = len(self.viewer.action_results)
        self.mouse("pointerup", point, shift)
        deadline = time.monotonic() + (10 if expect else 0.4)
        while len(self.viewer.action_results) < before + (expect or 0) and time.monotonic() < deadline:
            self.pump(0.02)
        self.pump(0.1)
        return self.viewer.action_results[before:]

    def bodies(self):
        return self.state("window.__bodies")

    def tips(self):
        return self.state("window.tips()")

    def around(self, leaves, margin=3):
        """Two corners, whole pixels, of a box that holds the tips of ``leaves`` with ``margin`` to spare."""
        tips = self.tips()
        xs = [tips[k][0] for k in leaves]
        ys = [tips[k][1] for k in leaves]
        return [int(min(xs)) - margin, int(min(ys)) - margin], [int(max(xs)) + margin + 1, int(max(ys)) + margin + 1]

    def inside(self, a, b, eligible=None):
        """The leaves whose tips are in the box with corners ``a`` and ``b`` as the page reads them."""
        (ax, ay), (bx, by) = [self.script(f"return fire.local({p[0]}, {p[1]})") for p in (a, b)]
        left, right, top, bottom = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
        return [k for k, (x, y) in enumerate(self.tips())
                if left <= x <= right and top <= y <= bottom and (eligible is None or k in eligible)]

    def settle(self):
        self.script("TreeView.drawNow(); return 1")

    def test_a_box_replaces_the_selection_with_the_leaves_whose_tips_are_in_it_in_every_layout(self):
        self.start_box()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.viewer.select([0, 1, 2])
                a, b = self.around([10, 11, 12, 13])
                expected = self.inside(a, b)
                self.assertTrue({10, 11, 12, 13} <= set(expected))
                self.assertLess(len(expected), 40)
                self.script("window.__bodies.length = 0; return 1")
                self.drag(a, b, release=False)
                self.assertEqual(self.bodies(), [])
                self.release(b, expect=1)
                self.assertEqual(self.bodies(), [{"action": "tree_select", "indices": self.nodes_of(expected), "mode": "replace"}])
                self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of(expected)))

    def test_the_corners_can_be_dragged_in_any_direction(self):
        self.start_box()
        self.choose("rectangular")
        a, b = self.around([10, 11, 12, 13])
        expected = self.inside(a, b)
        for start, end in (([b[0], a[1]], [a[0], b[1]]), (b, a), ([a[0], b[1]], [b[0], a[1]])):
            self.viewer.select([])
            self.drag(start, end)
            self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of(expected)))

    def test_shift_toggles_the_leaves_in_the_box_against_the_selection_in_every_layout(self):
        self.start_box()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                a, b = self.around([10, 11, 12, 13])
                expected = set(self.inside(a, b))
                before = {11, 30, 31}
                self.select(self.nodes_of(sorted(before)))
                self.drag(a, b, shift=True, release=False)
                self.release(b, shift=True, expect=1)
                self.assertEqual(self.bodies()[-1], {"action": "tree_select", "indices": self.nodes_of(sorted(expected)), "mode": "toggle"})
                self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of(sorted(expected ^ before))))

    def test_shift_is_read_when_the_button_is_let_go(self):
        self.start_box()
        self.choose("rectangular")
        a, b = self.around([10, 11, 12, 13])
        expected = self.inside(a, b)
        self.select(self.nodes_of([11, 30]))
        self.drag(a, b, shift=False, release=False)
        self.release(b, shift=True, expect=1)
        self.assertEqual(self.bodies()[-1]["mode"], "toggle")
        self.select(self.nodes_of([11, 30]))
        self.drag(a, b, shift=True, release=False)
        self.release(b, shift=False, expect=1)
        self.assertEqual(self.bodies()[-1], {"action": "tree_select", "indices": self.nodes_of(expected), "mode": "replace"})

    def test_an_empty_box_clears_and_with_shift_changes_nothing(self):
        self.start_box()
        self.choose("rectangular")
        self.viewer.select([0, 1, 2])
        tip = self.tips()[39]
        a = [int(tip[0]) + 30, int(tip[1]) - 6]
        b = [a[0] + 12, a[1] + 12]
        self.assertEqual(self.inside(a, b), [])
        self.script("window.__bodies.length = 0; return 1")
        self.drag(a, b, shift=True)
        self.assertEqual(self.bodies(), [])
        self.assertEqual(self.viewer.selected_indices, [0, 1, 2])
        self.drag(a, b)
        self.pump(0.3)
        self.assertEqual(self.bodies(), [{"action": "tree_select", "indices": [], "mode": "replace"}])
        self.assertEqual(self.viewer.selected_indices, [])

    def test_a_right_click_still_selects_a_clade_or_a_leaf_and_shift_on_empty_canvas_does_nothing(self):
        self.start_box()
        self.choose("rectangular")
        self.viewer.select([0, 1, 2])
        clade = self.internal()
        x, y = self.state(f"window.stem({clade})")
        self.drag([x, y], [x, y])
        self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of(range(*self.state(
            f"[TreeView.state.model.firstLeaf[{clade}], TreeView.state.model.lastLeaf[{clade}] + 1]")))))
        self.viewer.select([5])
        empty = self.state("""(() => { const c = document.getElementById('tree-canvas');
            for (let y = 4; y < c.clientHeight; y += 9) for (let x = 4; x < c.clientWidth - 20; x += 9) if (!TreeView.hitTest(x, y)) return [x, y];
            return null; })()""")
        self.script("window.__bodies.length = 0; return 1")
        self.drag(empty, empty, shift=True)
        self.assertEqual(self.bodies(), [])
        self.assertEqual(self.viewer.selected_indices, [5])
        self.drag(empty, empty)
        self.pump(0.3)
        self.assertEqual(self.viewer.selected_indices, [])

    def test_a_move_of_less_than_five_pixels_is_a_click_and_five_make_a_box(self):
        self.start_box()
        self.choose("rectangular")
        clade = self.internal()
        x, y = [int(v) for v in self.state(f"window.stem({clade})")]
        self.assertEqual(self.state(f"(() => {{ const h = TreeView.hitTest({x}, {y}); return h && h.node; }})()"), clade)
        self.drag([x, y], [x + 3, y + 3])                              # 4.2 pixels: a click on the clade
        first, last = self.state(f"[TreeView.state.model.firstLeaf[{clade}], TreeView.state.model.lastLeaf[{clade}] + 1]")
        self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of(range(first, last))))
        self.viewer.select([])
        self.drag([x, y], [x + 4, y + 3], release=False)               # 5 pixels: a box, which holds no tip it can reach
        self.assertEqual(self.spied("return log.filter(isBox).length"), 1)
        self.release([x + 4, y + 3], expect=1)
        self.assertEqual(self.bodies()[-1], {"action": "tree_select", "indices": self.nodes_of(self.inside([x, y], [x + 4, y + 3])), "mode": "replace"})

    def test_the_box_takes_the_tips_edges_included(self):
        self.start_box()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.state("""(() => { const t = window.tips(), s = TreeView.state; let ok = true;
                    for (const k of [0, 7, 21, 39]) { const [x, y] = t[k];
                        if (!Array.from(TreeView.leavesInBox(s, x, y, x, y)).includes(k)) ok = false;
                        if (Array.from(TreeView.leavesInBox(s, x + 1e-6, y, x + 5, y + 5)).includes(k)) ok = false; }
                    return ok; })()""")
                self.assertTrue(found)

    def test_a_leaf_whose_name_but_not_tip_is_in_the_box_is_not_selected(self):
        self.start_box()
        self.choose("rectangular", "cladogram")
        self.settle()
        self.assertGreater(self.state("TreeView.lastStats.labels"), 0)
        tip = self.tips()[20]
        a = [int(tip[0]) + 8, int(tip[1]) - 3]
        b = [a[0] + 60, a[1] + 6]
        self.assertEqual(self.inside(a, b), [])
        self.viewer.select([1])
        self.drag(a, b)
        self.pump(0.3)
        self.assertEqual(self.bodies()[-1], {"action": "tree_select", "indices": [], "mode": "replace"})

    def test_a_folded_clade_has_no_tips_in_the_box(self):
        self.start_box()
        self.choose("rectangular")
        clade = self.internal()
        first, last = self.state(f"[TreeView.state.model.firstLeaf[{clade}], TreeView.state.model.lastLeaf[{clade}]]")
        a, b = self.around(list(range(0, 40)), margin=20)
        everything = self.inside(a, b)
        self.assertEqual(len(everything), 40)
        self.script(f"TreeView.openAll(); TreeView.toggleFold({clade}); TreeView.drawNow(); return 1")
        self.assertEqual(self.state("TreeView.state.collapsedCount"), 1)
        self.drag(a, b)
        got = sorted(self.viewer.selected_indices)
        self.assertEqual(got, sorted(self.nodes_of([k for k in range(40) if not first <= k <= last])))
        # A box over the folded triangle only selects nothing.
        x, y = self.state(f"window.stem({clade})")
        tiny = [[x - 2, y - 2], [x + 2, y + 2]]
        self.viewer.select([3])
        self.drag(tiny[0], tiny[1])
        self.pump(0.3)
        self.assertEqual(self.viewer.selected_indices, [])

    def test_hidden_leaves_are_not_selected(self):
        self.start_box(hidden=[12, 13])
        self.choose("rectangular")
        self.wait_for("TreeView.state.colors !== null")
        leaf = lambda node: self.state(f"TreeView.leafOfNode(TreeView.state.model)[{node}]")
        gone = [leaf(12), leaf(13)]
        self.assertEqual(self.state(f"{gone}.map(k => TreeView.state.colors.leafClass[k] === TreeView.NO_CLASS)"), [True, True])
        a, b = self.around(gone + [leaf(10), leaf(11)])
        expected = self.inside(a, b)
        self.assertTrue(set(gone) <= set(expected))
        self.drag(a, b)
        wanted = sorted(self.nodes_of([k for k in expected if k not in gone]))
        self.assertEqual(sorted(self.bodies()[-1]["indices"]), wanted)
        self.assertEqual(sorted(self.viewer.selected_indices), wanted)

    def test_a_leaf_that_the_layout_has_left_out_is_not_in_the_box(self):
        self.start_box()
        self.choose("rectangular")
        found = self.state("""(() => { const s = TreeView.state, m = s.model, [x, y] = window.tips()[7];
            const live = new Uint8Array(m.nodeCount).fill(1); live[m.leafNode[7]] = 0;
            const scene = Object.assign({}, s, {layout: Object.assign(Object.create(s.layout), {live})});
            return [Array.from(TreeView.leavesInBox(s, x - 1, y - 1, x + 1, y + 1)).includes(7),
                    Array.from(TreeView.leavesInBox(scene, x - 1, y - 1, x + 1, y + 1)).includes(7)]; })()""")
        self.assertEqual(found, [True, False])

    def test_leaves_that_are_not_in_the_network_are_not_selected(self):
        headers = headers_of(40)
        self.start_box(headers=headers, network=headers[:25])
        self.choose("rectangular")
        absent = self.state("(() => { const m = TreeView.state.model, out = []; for (let k = 0; k < m.leafCount; k++) if (m.viewerNode[k] < 0) out.push(k); return out; })()")
        self.assertEqual(len(absent), 15)
        a, b = self.around(list(range(40)), margin=20)
        self.drag(a, b)
        self.assertEqual(sorted(self.bodies()[-1]["indices"]), list(range(25)))
        self.assertEqual(sorted(self.viewer.selected_indices), list(range(25)))

    def test_the_box_is_drawn_in_the_accent_colour_and_the_selection_it_will_make_is_previewed(self):
        self.start_box()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.select([0, 1, 2])
                network = self.selection()["leaves"]
                a, b = self.around([10, 11, 12, 13])
                expected = self.inside(a, b)
                self.script("window.__bodies.length = 0; return 1")
                self.drag(a, b, release=False)
                drawn = self.spied("""const rects = log.filter(isBox);
                    return {rects, accent: TreeView.rgbOf(TreeView.state.palette.accent), marked: TreeView.lastStats.selected};""")
                self.assertEqual(len(drawn["rects"]), 1)
                rect = drawn["rects"][0]
                self.assertEqual(rect["style"], drawn["accent"])
                self.assertEqual(rect["width"], 1 / self.state("window.devicePixelRatio") * round(self.state("window.devicePixelRatio")))
                self.assertAlmostEqual(rect["w"], b[0] - a[0], delta=1.5)
                self.assertAlmostEqual(rect["h"], b[1] - a[1], delta=1.5)
                self.assertEqual(drawn["marked"], len(expected))
                self.assertEqual(self.bodies(), [])                    # nothing is sent while the box is open
                self.assertEqual(self.selection()["leaves"], network)
                self.assertEqual(self.state("TreeView.state.selection.count"), 3)
                self.release(b, expect=1)
                self.settle()
                self.assertEqual(self.spied("return log.filter(isBox).length"), 0)

    def test_the_preview_with_shift_is_the_networks_selection_toggled(self):
        self.start_box()
        self.choose("rectangular")
        a, b = self.around([10, 11, 12, 13])
        inside = set(self.inside(a, b))
        before = {11, 30, 31}
        self.select(self.nodes_of(sorted(before)))
        self.drag(a, b, shift=True, release=False)
        marked = self.spied("return TreeView.lastStats.selected")
        self.assertEqual(marked, len(inside ^ before))
        self.mouse("pointermove", b, shift=False)
        self.assertEqual(self.spied("return TreeView.lastStats.selected"), len(inside))     # Shift is read on every move
        self.assertEqual(self.bodies(), [])

    def test_the_nodes_next_to_the_selection_are_not_marked_while_the_box_is_open(self):
        self.start_box()
        self.choose("rectangular")
        a, b = self.around([10, 11, 12, 13])
        self.script("TreeView.applyConnected(TreeView.state.model.viewerNode.slice(20, 25)); return 1")
        self.assertEqual(self.state("TreeView.state.selection.connected.length"), 5)
        self.drag(a, b, release=False)
        calls = self.script("""const original = TreeView.drawLeafMarks, lengths = [];
            TreeView.drawLeafMarks = function (ctx, scene, leaves) { lengths.push(leaves.length); return original.apply(this, arguments); };
            try { TreeView.drawNow(); } finally { TreeView.drawLeafMarks = original; }
            return lengths;""")
        self.assertEqual(calls, [0, len(self.inside(a, b))])           # no neighbours, then the box's leaves
        self.assertEqual(self.state("TreeView.state.selection.connected.length"), 5)
        self.mouse("pointerdown", a)            # a new press while open ends the old box; the network's marks are untouched
        self.assertEqual(self.state("TreeView.state.selection.connected.length"), 5)

    def test_a_wheel_zoom_during_the_drag_keeps_the_first_corner_on_the_tree(self):
        self.start_box()
        self.choose("rectangular")
        a = [int(v) for v in self.tips()[10]]
        b = [a[0] + 60, a[1] + 40]
        before = self.state("TreeView.state.view.sy")
        self.drag(a, b, release=False)
        anchor = self.state("(() => { const v = TreeView.state.view; return [(%d - v.ox) / v.sx, (%d - v.oy) / v.sy]; })()" % tuple(a))
        self.script("fire.wheel(-300, 20, 20); fire.wheel(-200, 20, 20, {shiftKey: true}); return 1")
        moved = self.state("(() => { const v = TreeView.state.view; return [v.ox, v.oy, v.sx, v.sy]; })()")
        self.assertNotEqual(moved[3], before)
        corner = [moved[0] + anchor[0] * moved[2], moved[1] + anchor[1] * moved[3]]
        rect = self.spied("return log.filter(isBox)")[0]
        # The first corner is on the same place of the tree; the other is still at the pointer.
        left, top, right, bottom = rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"]
        near = lambda value, *options: min(abs(value - o) for o in options) <= 1.01
        self.assertTrue(near(corner[0], left, right) and near(corner[1], top, bottom), (corner, rect))
        self.assertTrue(near(b[0], left, right) and near(b[1], top, bottom))
        self.release(b, expect=1)
        got = self.bodies()[-1]
        self.assertEqual(got["mode"], "replace")
        self.assertEqual(sorted(got["indices"]), sorted(self.nodes_of(self.inside(
            [min(corner[0], b[0]), min(corner[1], b[1])], [max(corner[0], b[0]), max(corner[1], b[1])]))))

    def test_escape_closes_the_box_and_sends_nothing_and_the_unpin_stays(self):
        self.start_box()
        self.choose("rectangular")
        self.viewer.select([0, 1])
        self.wait_for("TreeView.state.selection.count === 2")
        self.pin_leaf(12)
        self.script("window.__bodies.length = 0; return 1")
        a, b = self.around([10, 11, 12, 13])
        self.drag(a, b, release=False)
        self.assertTrue(self.script("return fire.key('Escape')"))
        self.assertEqual(self.spied("return log.filter(isBox).length"), 0)
        self.assertEqual(self.release(b), [])
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.assertEqual(self.bodies(), [{"action": "tree_focus", "index": None}])
        self.assertEqual(self.viewer.selected_indices, [0, 1])
        self.assertEqual(self.spied("return TreeView.lastStats.selected"), 2)             # the network's own selection is back
        # With no box, Escape is what it was.
        self.pin_leaf(12)
        self.script("fire.key('Escape'); return 1")
        self.assertEqual(self.state("TreeView.state.pinned"), -1)

    def test_a_cancelled_pointer_or_a_lost_capture_closes_the_box_and_sends_nothing(self):
        self.start_box()
        self.choose("rectangular")
        self.viewer.select([0, 1])
        a, b = self.around([10, 11, 12, 13])
        for kind in ("pointercancel", "lostpointercapture"):
            with self.subTest(kind=kind):
                self.script("window.__bodies.length = 0; return 1")
                self.drag(a, b, release=False)
                self.mouse(kind, b)
                self.assertEqual(self.spied("return log.filter(isBox).length"), 0)
                self.assertEqual(self.release(b), [])
                self.assertEqual(self.bodies(), [])
                self.assertEqual(self.viewer.selected_indices, [0, 1])

    def test_a_contextmenu_event_before_the_release_does_nothing_by_itself(self):
        self.start_box()
        self.choose("rectangular")
        self.viewer.select([0, 1])
        a, b = self.around([10, 11, 12, 13])
        expected = self.inside(a, b)
        self.drag(a, b, release=False)
        prevented = self.state("(() => { const c = document.getElementById('tree-canvas'), e = new MouseEvent('contextmenu', {button: 2, bubbles: true, cancelable: true}); c.dispatchEvent(e); return e.defaultPrevented; })()")
        self.assertTrue(prevented)
        self.assertEqual(self.bodies(), [])
        self.assertEqual(self.state("TreeView.state.selection.count"), 2)
        self.assertEqual(self.spied("return log.filter(isBox).length"), 1)       # the box is still open
        self.release(b, expect=1)
        self.assertEqual(sorted(self.viewer.selected_indices), sorted(self.nodes_of(expected)))

    def test_a_right_press_unpins_at_once_and_takes_the_focus_off(self):
        self.start_box()
        self.choose("rectangular")
        self.pin_leaf(12)
        self.script("window.__bodies.length = 0; return 1")
        self.mouse("pointerdown", self.tips()[3])
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        deadline = time.monotonic() + 10
        while self.viewer.selected_node_idx is not None and time.monotonic() < deadline:
            self.pump(0.02)
        self.assertIsNone(self.viewer.selected_node_idx)
        self.assertEqual(self.bodies(), [{"action": "tree_focus", "index": None}])
        self.mouse("pointerup", self.tips()[3])

    def test_a_press_in_the_ruler_starts_no_box(self):
        self.start_box()
        self.choose("rectangular")
        self.viewer.select([0, 1])
        width = self.state("document.getElementById('tree-canvas').clientWidth")
        self.drag([width - 5, 20], [width - 5, 200])
        self.assertEqual(self.bodies(), [])
        self.assertEqual(self.viewer.selected_indices, [0, 1])
        self.assertEqual(self.spied("return log.filter(isBox).length"), 0)

    def test_the_left_button_is_not_changed_and_is_ignored_while_a_box_is_open(self):
        self.start_box()
        self.choose("rectangular")
        a, b = self.around([10, 11, 12, 13])
        view = self.state("TreeView.state.view")
        self.drag(a, b, release=False)
        self.script(f"fire.pointer('pointerdown', {b[0]}, {b[1]}); return 1")
        self.assertEqual(self.state("TreeView.state.view"), view)
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.assertEqual(self.spied("return log.filter(isBox).length"), 1)
        self.release(b, expect=1)
        # A left drag still pans, and a left click still pins.
        self.script("fire.pointer('pointerdown', 100, 100); fire.pointer('pointermove', 140, 130); fire.pointer('pointerup', 140, 130); return 1")
        self.assertNotEqual(self.state("TreeView.state.view"), view)
        x, y = self.tips()[12]
        self.script(f"fire.click({x}, {y}); return 1")
        self.assertGreaterEqual(self.state("TreeView.state.pinned"), 0)

    def test_one_request_for_the_whole_gesture(self):
        self.start_box()
        self.choose("rectangular")
        a, b = self.around([10, 11, 12, 13])
        self.script("window.__bodies.length = 0; return 1")
        self.drag(a, b)
        self.pump(0.3)
        self.assertEqual(len(self.bodies()), 1)
        self.assertEqual(len(self.viewer.action_results), 1)

    def test_the_work_for_a_frame_is_quick(self):
        self.start_box()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                a, b = self.around([5, 30])
                self.drag(a, b, release=False)
                timing = self.script("""const s = TreeView.state, a = performance.now();
                    for (let i = 0; i < 50; i++) { const inside = TreeView.leavesInBox(s, 0, 0, 400, 300);
                        TreeView.smallestClade(s.model, TreeView.toggleLeaves(s.selection.leaves, inside)); }
                    return (performance.now() - a) / 50;""")
                self.assertLess(timing, 20)
                self.mouse("pointercancel", b)


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
