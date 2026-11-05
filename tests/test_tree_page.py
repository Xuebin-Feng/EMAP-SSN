# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's page (src/web_ui/tree.html and tree_*.js) in Qt WebEngine over the real web server.

The page asks the Viewer for the tree of the loaded alignment and draws it, or says why there is none; says how
the tree fits the alignment and the network; zooms, pans, searches and tells what is under the pointer. The page
is opened offscreen. Its canvas is checked through its own state (the model, the layout, the view, what the last
frame drew) and its pixels, and the layout is checked against a computation from the tree file made here in
Python. A header is the file's text and is only ever shown as text.
"""

import json
import os
import pathlib
import re
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

from PySide6.QtCore import QUrl  # noqa: E402

from tests.tree_page_harness import TreePageMixin, http_get, page_headers, server_viewer  # noqa: E402
from tests.tree_fixtures import (  # noqa: E402
    ALIGNMENT_NAME,
    TREE_NAME,
    headers_of,
    make_alignment_folder,
    make_viewer,
    neighbor_joining_tree,
    random_linkage,
    save_guide_tree,
    shaped_linkage,
    writer,
)
from utilities.Guide_Tree_File import read_guide_tree  # noqa: E402
from web_ui import tree_backend  # noqa: E402

WEB_DIR = SRC_DIR / "web_ui"
PAGE_FILES = ["tree.html", "tree_model.js", "tree_layout.js", "tree_render.js", "tree_nav.js", "tree_minimap.js", "tree_session.js"]


def clade_of(size_low, size_high):
    """JavaScript that finds the first internal node with between size_low and size_high leaves."""
    return f"""(() => {{
        const m = TreeView.state.model;
        for (let i = 0; i < m.nodeCount; i++) {{
            const n = m.lastLeaf[i] - m.firstLeaf[i] + 1;
            if (m.childCount[i] > 0 && n >= {size_low} && n <= {size_high}) return i;
        }}
        return -1;
    }})()"""


# The pixel position of the tip of leaf number k, and a point on its name, as JavaScript.
def tip_of(k):
    return f"(() => {{ const s = TreeView.state; return TreeView.nodePixels(s, s.model.leafNode[{k}]); }})()"


def write_tree_file(folder, headers, text):
    """Make an alignment folder for ``headers`` whose tree file is the hand-written NEXUS ``text``."""
    make_alignment_folder(folder, headers, tree=None)
    guide = pathlib.Path(folder) / "Guide_Trees"
    guide.mkdir(exist_ok=True)
    (guide / TREE_NAME).write_text(text, encoding="utf-8")


class EmptyStateTests(TreePageMixin, unittest.TestCase):
    """The page says why there is no tree to draw, in a card with what to do about it."""

    def serve_alignment(self, name=ALIGNMENT_NAME, *, headers=None, **options):
        folder = self.make_folder()
        headers = headers or headers_of(20)
        msa, tree = make_alignment_folder(folder, headers, alignment=name, **options)
        viewer = server_viewer()
        fixture = make_viewer(msa, headers)
        viewer.alignment, viewer.full_headers, viewer.n_nodes = fixture.alignment, fixture.full_headers, len(headers)
        self.viewer, self.msa, self.tree_file = viewer, msa, tree
        return self.start_page_server(viewer)

    def card(self):
        return self.script("""
            const text = id => document.getElementById(id).textContent;
            return {
                view: document.body.dataset.view,
                shown: document.getElementById('empty').classList.contains('shown'),
                title: text('empty-title'), body: text('empty-body'), more: text('empty-more'),
                path: text('empty-path'), pathHidden: document.getElementById('empty-path').hidden,
                moreHidden: document.getElementById('empty-more').hidden,
                copyHidden: document.getElementById('empty-copy').hidden,
                controls: ['search-input', 'fit-btn', 'details-btn', 'search-prev', 'search-next']
                    .map(id => document.getElementById(id).disabled),
                banner: document.getElementById('banner').classList.contains('shown'),
                summary: text('summary'),
            };""")

    def test_no_alignment_loaded(self):
        viewer = server_viewer()
        viewer.alignment, viewer.full_headers = None, []
        self.start_page_server(viewer)
        self.open_page("empty")
        card = self.card()
        self.assertTrue(card["shown"])
        self.assertEqual(card["title"], "No alignment is loaded")
        self.assertIn("Load an alignment", card["body"])
        self.assertTrue(card["pathHidden"] and card["copyHidden"])
        self.assertEqual(card["controls"], [True] * 5)
        self.assertEqual(self.console_errors(), [])

    def test_an_alignment_the_tool_did_not_name(self):
        self.serve_alignment("mafft_result.fasta", tree=None)
        self.open_page("empty")
        card = self.card()
        self.assertEqual(card["title"], "This alignment has no guide tree")
        self.assertIn("mafft_result.fasta was not named by the Embedding MSA tool", card["body"])
        self.assertTrue(card["pathHidden"])

    def test_a_missing_tree_names_the_file_it_looked_for_and_offers_to_copy_it(self):
        self.serve_alignment(tree=None)
        self.open_page("empty")
        card = self.card()
        self.assertEqual(card["title"], "No guide tree was found for this alignment")
        expected = os.path.abspath(pathlib.Path(self.msa).parent / "Guide_Trees" / TREE_NAME)
        self.assertEqual(card["path"], expected)
        self.assertFalse(card["pathHidden"] or card["copyHidden"] or card["moreHidden"])
        self.assertIn(ALIGNMENT_NAME, card["body"])
        self.assertIn("press Check again", card["more"])
        self.assertEqual(card["controls"], [True] * 5)

    def test_a_file_that_is_not_a_tree_says_why(self):
        self.serve_alignment()
        self.tree_file.write_text("this is not a tree", encoding="utf-8")
        self.open_page("empty")
        card = self.card()
        self.assertEqual(card["title"], "The guide tree could not be read")
        self.assertIn("it does not begin with #NEXUS", card["body"])
        self.assertEqual(card["path"], os.path.abspath(self.tree_file))

    def test_a_tree_that_matches_no_node_of_the_network_is_not_drawn(self):
        self.serve_alignment(headers=headers_of(20))
        self.viewer.full_headers = headers_of(20, prefix="tr|Z")
        self.open_page("empty")
        card = self.card()
        self.assertEqual(card["title"], "The tree does not match this network")
        self.assertIn("None of the 20 leaves", card["body"])

    def test_a_viewer_that_answers_with_an_error_is_reported_not_hidden(self):
        self.serve_alignment()
        with mock.patch.object(tree_backend, "tree_state", side_effect=RuntimeError("boom")):
            self.open_page("empty")
            card = self.card()
        self.assertEqual(card["title"], "Tree View could not reach the Viewer")
        self.assertIn("500", card["body"])

    def test_check_again_reads_the_file_that_was_put_there_meanwhile(self):
        self.serve_alignment(tree=None)
        self.open_page("empty")
        self.assertEqual(self.card()["title"], "No guide tree was found for this alignment")
        save_guide_tree(pathlib.Path(self.msa).parent / "Guide_Trees" / TREE_NAME, headers_of(20),
                        notes=[("ALIGNMENT", ALIGNMENT_NAME)])
        self.js("document.getElementById('empty-retry').click()")
        self.wait_for("document.body.dataset.view === 'tree'")
        card = self.card()
        self.assertFalse(card["shown"])
        self.assertEqual(card["controls"][:3], [False, False, False])
        self.assertEqual(card["summary"], "20 leaves, 20 of them nodes of this network")

    def test_a_name_with_markup_in_it_is_shown_as_it_is(self):
        self.serve_alignment("x&lt;b&gt;_alignment.fasta", tree=None)
        self.open_page("empty")
        card = self.script("""
            const body = document.getElementById('empty-body');
            return {text: body.textContent, children: body.children.length,
                    path: document.getElementById('empty-path').textContent};""")
        self.assertIn("x&lt;b&gt;_alignment.fasta", card["text"])
        self.assertEqual(card["children"], 0)
        self.assertIn("x&lt;b&gt;_guide_tree.nex", card["path"])


class TreeCase(TreePageMixin, unittest.TestCase):
    """A page with a tree: a view of 1300 x 800, the real tree of ``COUNT`` leaves and its file's own reading."""

    COUNT = 300

    def setUp(self):
        super().setUp()
        self.headers = headers_of(self.COUNT)
        self.serve_headers(self.headers)
        self.open_page()

    def tearDown(self):
        self.view.resize(1300, 800)
        super().tearDown()

    def view_state(self):
        return self.state("""({...TreeView.state.view, fitted: TreeView.state.fitted,
            w: document.getElementById('tree-canvas').clientWidth, h: document.getElementById('tree-canvas').clientHeight})""")


class DrawingTests(TreeCase):
    def test_the_whole_tree_is_in_the_picture_at_fit(self):
        info = self.script("""
            const s = TreeView.state, m = s.model, L = s.layout, v = s.view;
            const canvas = document.getElementById('tree-canvas');
            let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
            for (let i = 0; i < m.nodeCount; i++) {
                const x = v.ox + L.x[i] * v.sx, y = v.oy + L.y[i] * v.sy;
                minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            }
            return {leaves: m.leafCount, nodes: m.nodeCount, kind: L.kind, mode: L.mode, fitted: s.fitted, rooted: m.rooted,
                    hasLengths: m.hasLengths, w: canvas.clientWidth, h: canvas.clientHeight, minX, maxX, minY, maxY};""")
        self.assertEqual((info["leaves"], info["nodes"], info["kind"], info["mode"]), (300, 599, "rectangular", "lengths"))
        self.assertTrue(info["fitted"] and info["rooted"] and info["hasLengths"])
        self.assertGreaterEqual(info["minX"], 0)
        self.assertLessEqual(info["maxX"], info["w"] - 14)  # the ruler's strip is the page's right edge
        self.assertGreaterEqual(info["minY"], 0)
        self.assertLessEqual(info["maxY"], info["h"])
        self.assertEqual(self.console_errors(), [])

    def test_the_canvas_holds_a_drawing_in_the_pages_colours(self):
        info = self.script("""
            const canvas = document.getElementById('tree-canvas');
            const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
            const ctx = document.createElement('canvas').getContext('2d');
            ctx.fillStyle = TreeView.state.palette.background; ctx.fillRect(0, 0, 1, 1);
            const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
            let ink = 0;
            for (let i = 0; i < data.length; i += 4) if (data[i] !== r || data[i + 1] !== g || data[i + 2] !== b) ink++;
            return {ink, pixels: data.length / 4};""")
        self.assertGreater(info["ink"], 2000)
        self.assertLess(info["ink"], info["pixels"] * 0.5)

    def test_the_layout_equals_an_independent_computation_from_the_file(self):
        tree = read_guide_tree(self.tree_file)
        parent, length = tree.parent, tree.length.astype(np.float64)
        count = tree.node_count
        children = [[] for _ in range(count)]
        for node in range(count - 1):
            children[parent[node]].append(node)
        x = np.zeros(count)
        for node in range(count - 2, -1, -1):
            x[node] = x[parent[node]] + length[node]
        y = np.zeros(count)
        ordinal = 0
        for node in range(count):
            if children[node]:
                y[node] = (y[children[node][0]] + y[children[node][-1]]) / 2
            else:
                y[node] = ordinal + 0.5
                ordinal += 1
        reach = x.copy()
        for node in range(count - 1):
            reach[parent[node]] = max(reach[parent[node]], reach[node])
        layout = self.state("""({x: Array.from(TreeView.state.layout.x), y: Array.from(TreeView.state.layout.y),
                                 reach: Array.from(TreeView.state.layout.reach)})""")
        np.testing.assert_allclose(layout["x"], x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(layout["y"], y, rtol=1e-12)
        np.testing.assert_allclose(layout["reach"], reach, rtol=1e-12, atol=1e-12)

    def test_the_arrays_the_page_derives_agree_with_walking_the_tree(self):
        bad = self.script("""
            const m = TreeView.state.model;
            const below = i => {
                const leaves = [], nodes = [];
                const stack = [i];
                while (stack.length) {
                    const x = stack.pop();
                    nodes.push(x);
                    if (m.childCount[x] === 0) leaves.push(m.leafOrdinal[x]);
                    else for (let k = m.childStart[x]; k < m.childStart[x + 1]; k++) stack.push(m.childList[k]);
                }
                leaves.sort((a, b) => a - b);
                return {leaves, nodes};
            };
            const wrong = [];
            for (let i = 0; i < m.nodeCount; i++) {
                const {leaves, nodes} = below(i);
                const contiguous = leaves.every((value, index) => value === leaves[0] + index);
                if (!contiguous || leaves[0] !== m.firstLeaf[i] || leaves[leaves.length - 1] !== m.lastLeaf[i]
                    || nodes.length !== m.size[i] || TreeView.leavesUnder(m, i) !== leaves.length) wrong.push(i);
                if (m.childCount[i] > 0) {
                    for (const leaf of [m.firstLeaf[i], m.lastLeaf[i], Math.floor((m.firstLeaf[i] + m.lastLeaf[i]) / 2)]) {
                        const child = TreeView.childHolding(m, i, leaf);
                        if (m.parent[child] !== i || leaf < m.firstLeaf[child] || leaf > m.lastLeaf[child]) wrong.push(-i - 1);
                    }
                }
            }
            for (let k = 0; k < m.leafCount; k++) if (m.leafOrdinal[m.leafNode[k]] !== k) wrong.push(-100000 - k);
            return wrong;""")
        self.assertEqual(bad, [])

    def test_leaves_the_network_lacks_are_drawn_faded(self):
        # Half the network: the tree's other leaves are faded dots, the rest are drawn as the lines are.
        headers = headers_of(40)
        self.serve_headers(headers, network=headers[:20])
        self.open_page()
        colours = self.script("""
            const s = TreeView.state, m = s.model;
            TreeView.zoomToNode(m.root);
            const ctx = document.createElement('canvas').getContext('2d');
            const rgb = css => { ctx.fillStyle = css; ctx.fillRect(0, 0, 1, 1); return Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)); };
            // A line one pixel wide on the middle of a pixel row, and branches wide enough to sample.
            // (Every tip of this tree is at the same distance from the root: put them at x = 900.)
            s.view.sx *= 50; s.view.ox = 900 - s.layout.width * s.view.sx; s.view.sy = 30; s.view.oy = 0.5; TreeView.drawNow();
            const found = {shown: {}, faded: {}};
            for (let k = 0; k < 14; k++) {
                const box = TreeView.nodePixels(s, m.leafNode[k]);
                const key = m.viewerNode[k] >= 0 ? 'shown' : 'faded';
                if (!found[key].dot) found[key].dot = fire.pixel(box.x, box.y);
                if (!found[key].branch && box.x - box.x0 >= 14) found[key].branch = fire.pixel(box.x - 9, Math.floor(box.y));
            }
            return {found, shown: rgb(s.palette.branch), faded: rgb(s.palette.faded)};""")
        for part in ("dot", "branch"):
            with self.subTest(part=part):
                self.assertEqual(colours["found"]["shown"][part][:3], colours["shown"])
                self.assertEqual(colours["found"]["faded"][part][:3], colours["faded"])
        self.assertNotEqual(colours["shown"], colours["faded"])

    def test_a_tree_without_branch_lengths_is_drawn_as_a_cladogram(self):
        headers = list("abcde")
        folder = self.make_folder()
        write_tree_file(folder, headers, "#NEXUS\nBEGIN EMAPSSN;\n  ALIGNMENT = 'Set_[model]_alignment.fasta';\nEND;\n"
                                         "BEGIN TREES;\n  TREE t = [&R] ((a,b),(c,(d,e)));\nEND;\n")
        self.serve_headers(headers, folder=folder, tree=None)
        self.open_page()
        info = self.script("""
            const L = TreeView.state.layout, m = TreeView.state.model;
            return {mode: L.mode, hasLengths: m.hasLengths, x: Array.from(L.x), width: L.width, parent: Array.from(m.parent)};""")
        self.assertEqual((info["mode"], info["hasLengths"]), ("cladogram", False))
        # In post-order: a, b, (a,b), c, d, e, (d,e), (c,(d,e)), root. The leaves stand at the right, and a clade
        # is one step left of its deepest child.
        self.assertEqual(info["x"], [3, 3, 2, 3, 3, 3, 2, 1, 0])
        self.assertEqual(info["width"], 3)
        for node, parent in enumerate(info["parent"][:-1]):
            self.assertGreater(info["x"][node], info["x"][parent])

    def test_lengths_that_sum_to_nothing_fall_back_to_a_cladogram(self):
        headers = list("abc")
        folder = self.make_folder()
        write_tree_file(folder, headers, "#NEXUS\nBEGIN TREES;\n  TREE t = [&R] ((a:0,b:0):0,c:0);\nEND;\n")
        self.serve_headers(headers, folder=folder, tree=None)
        self.open_page()
        info = self.state("({mode: TreeView.state.layout.mode, width: TreeView.state.layout.width, fit: TreeView.state.view.sx})")
        self.assertEqual(info["mode"], "cladogram")
        self.assertGreater(info["width"], 0)
        self.assertTrue(np.isfinite(info["fit"]))

    def test_a_single_leaf_is_drawn(self):
        headers = ["only"]
        folder = self.make_folder()
        write_tree_file(folder, headers, "#NEXUS\nBEGIN TREES;\n  TREE t = [&R] only;\nEND;\n")
        self.serve_headers(headers, folder=folder, tree=None)
        self.open_page()
        stats = self.drawn()
        self.assertEqual(self.state("TreeView.state.model.leafCount"), 1)
        self.assertEqual(stats["visited"], 1)
        self.assertEqual(self.console_errors(), [])

    def test_a_payload_that_is_not_a_tree_is_refused_by_the_page(self):
        results = self.script("""
            const payload = await (await fetch('/api/tree/data')).json();
            const results = {};
            const bytes = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
            const text = array => btoa(String.fromCharCode(...array));
            const attempt = (name, change) => {
                const copy = JSON.parse(JSON.stringify(payload));
                change(copy);
                try { TreeView.decodeTree(copy); results[name] = 'decoded'; } catch (error) { results[name] = error.message; }
            };
            attempt('intact', p => {});
            attempt('too few names', p => p.labels.pop());
            attempt('counts', p => { p.counts.nodes = 3; });
            attempt('no leaves', p => { p.counts.leaves = 0; });
            attempt('size', p => { p.parent = text(bytes(p.parent).slice(4)); });
            attempt('missing', p => { delete p.length; });
            attempt('not post-order', p => { const b = bytes(p.parent); new Int32Array(b.buffer)[0] = 0; p.parent = text(b); });
            attempt('parent out of range', p => { const b = bytes(p.parent); new Int32Array(b.buffer)[0] = 99999; p.parent = text(b); });
            attempt('root', p => { const b = bytes(p.parent); const a = new Int32Array(b.buffer); a[a.length - 1] = 0; p.parent = text(b); });
            attempt('leaves said wrong', p => { p.counts.leaves = p.counts.leaves - 1; p.labels.pop(); p.viewer_node = text(bytes(p.viewer_node).slice(4)); });
            return results;""")
        self.assertEqual(results.pop("intact"), "decoded")
        # Each is refused for its own reason.
        reasons = {
            "too few names": "leaf names", "counts": "counts do not describe", "no leaves": "counts do not describe",
            "size": "wrong size", "missing": "length is missing", "not post-order": "post-order",
            "parent out of range": "post-order", "root": "not the root", "leaves said wrong": "more leaves than it says",
        }
        self.assertEqual(set(results), set(reasons))
        for name, reason in reasons.items():
            with self.subTest(name):
                self.assertIn(reason, results[name])


class LevelOfDetailTests(TreeCase):
    COUNT = 4000

    def test_what_a_frame_costs_follows_what_can_be_seen_not_the_size_of_the_tree(self):
        total = self.state("TreeView.state.model.nodeCount")
        fit = self.drawn()
        # 4,000 leaves in some 650 pixels: clades of a few leaves are one bar each, and no leaf is named.
        self.assertGreater(fit["bars"], 50)
        self.assertLess(fit["visited"], total * 0.25)
        self.assertEqual((fit["labels"], fit["dots"]), (0, 0))
        self.script(f"TreeView.zoomToNode({clade_of(10, 14)}); return 1")
        zoomed = self.drawn()
        self.assertGreater(zoomed["labels"], 5)
        self.assertGreater(zoomed["dots"], 5)
        self.assertEqual(zoomed["bars"], 0)
        # Out of view is not visited.
        self.assertLess(zoomed["visited"], 200)

    def test_a_frame_is_quick_at_fit_and_zoomed_in(self):
        timings = self.script(f"""
            const time = () => {{ const t = performance.now(); TreeView.drawNow(); return performance.now() - t; }};
            const fit = Math.min(time(), time(), time());
            TreeView.zoomToNode({clade_of(30, 60)});
            const zoomed = Math.min(time(), time(), time());
            return {{fit, zoomed}};""")
        # A loose bound that no sane machine misses; the measurements are the performance run's.
        self.assertLess(timings["fit"], 250)
        self.assertLess(timings["zoomed"], 250)

    def test_a_clade_that_is_a_bar_is_what_the_pointer_finds_there(self):
        hit = self.script("""
            const s = TreeView.state, m = s.model;
            const collapsed = i => {
                const box = TreeView.nodePixels(s, i);
                return m.lastLeaf[i] > m.firstLeaf[i] && box.bottom - box.top < TreeView.COLLAPSE_PX;
            };
            // The topmost bar: the one the pointer reaches first, going down from the root.
            let bar = -1;
            for (let i = m.nodeCount - 2; i >= 0 && bar < 0; i--) {
                const box = TreeView.nodePixels(s, i);
                if (collapsed(i) && !collapsed(m.parent[i]) && m.size[i] > 8 && box.bottom - box.top > 1) bar = i;
            }
            const box = TreeView.nodePixels(s, bar);
            const found = TreeView.hitTest((box.x + box.reach) / 2, (box.top + box.bottom) / 2);
            return {bar, found};""")
        self.assertGreaterEqual(hit["bar"], 0)
        self.assertEqual(hit["found"], {"node": hit["bar"], "part": "clade"})


class NavigationTests(TreeCase):
    def test_the_wheel_zooms_the_rows_about_the_pointer_and_leaves_the_width(self):
        before = self.view_state()
        prevented = self.script("return fire.wheel(-120, 600, 300)")
        after = self.view_state()
        self.assertTrue(prevented)
        self.assertAlmostEqual(after["sy"] / before["sy"], np.exp(0.18), places=6)
        self.assertEqual(after["sx"], before["sx"])
        # The row under the pointer stays under it.
        self.assertAlmostEqual((300 - after["oy"]) / after["sy"], (300 - before["oy"]) / before["sy"], places=6)
        self.assertFalse(after["fitted"])

    def test_shift_and_the_wheel_zoom_the_width_and_ctrl_both(self):
        before = self.view_state()
        self.script("return fire.wheel(-120, 600, 300, {shiftKey: true})")
        wide = self.view_state()
        self.assertAlmostEqual(wide["sx"] / before["sx"], np.exp(0.18), places=6)
        self.assertEqual(wide["sy"], before["sy"])
        self.assertAlmostEqual((600 - wide["ox"]) / wide["sx"], (600 - before["ox"]) / before["sx"], places=6)
        self.script("return fire.wheel(-120, 600, 300, {ctrlKey: true})")
        both = self.view_state()
        self.assertAlmostEqual(both["sx"] / wide["sx"], np.exp(0.18), places=6)
        self.assertAlmostEqual(both["sy"] / wide["sy"], np.exp(0.18), places=6)

    def test_zoom_is_held_between_limits(self):
        limits = self.state("TreeView.limits()")
        for _ in range(60):
            self.js("fire.wheel(-400, 600, 300); fire.wheel(-400, 600, 300, {shiftKey: true}); 1")
        zoomed = self.view_state()
        self.assertEqual(zoomed["sy"], limits["syMax"])
        self.assertEqual(zoomed["sx"], limits["sxMax"])
        for _ in range(120):
            self.js("fire.wheel(400, 600, 300); fire.wheel(400, 600, 300, {shiftKey: true}); 1")
        zoomed_out = self.view_state()
        self.assertEqual(zoomed_out["sy"], limits["syMin"])
        self.assertEqual(zoomed_out["sx"], limits["sxMin"])

    def test_dragging_moves_the_tree_and_is_no_click(self):
        self.script("fire.wheel(-200, 600, 300); return 1")  # room to move in
        before = self.view_state()
        self.script("""
            fire.pointer('pointerdown', 400, 300);
            fire.pointer('pointermove', 450, 340);
            fire.pointer('pointerup', 450, 340);
            return 1""")
        after = self.view_state()
        self.assertEqual((after["ox"] - before["ox"], after["oy"] - before["oy"]), (50, 40))
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.assertFalse(self.state("document.getElementById('tree-canvas').classList.contains('dragging')"))

    def test_the_tree_cannot_be_dragged_out_of_sight(self):
        self.script("TreeView.pan(1e6, 1e6); return 1")
        view = self.view_state()
        self.assertLessEqual(view["ox"], view["w"] - 60)
        self.assertLessEqual(view["oy"], view["h"] - 60)
        self.script("TreeView.pan(-1e7, -1e7); return 1")
        view = self.view_state()
        self.assertGreaterEqual(view["ox"] + self.state("TreeView.state.layout.width") * view["sx"], 60 - 300 - 1)
        self.assertGreaterEqual(view["oy"] + 300 * view["sy"], 60 - 1)

    def test_keys_zoom_pan_and_fit(self):
        start = self.view_state()
        self.assertTrue(self.script("return fire.key('+')"))
        rows = self.view_state()
        self.assertAlmostEqual(rows["sy"] / start["sy"], 1.25)
        self.assertEqual(rows["sx"], start["sx"])
        self.script("fire.key(']'); return 1")
        self.assertAlmostEqual(self.view_state()["sx"] / start["sx"], 1.25)
        before = self.view_state()
        self.script("fire.key('ArrowRight'); fire.key('ArrowDown'); return 1")
        moved = self.view_state()
        self.assertEqual((moved["ox"] - before["ox"], moved["oy"] - before["oy"]), (-80, -80))
        self.assertTrue(self.script("return fire.key('Home')"))
        fitted = self.view_state()
        self.assertTrue(fitted["fitted"])
        self.assertEqual((fitted["sx"], fitted["sy"]), (start["sx"], start["sy"]))
        self.assertFalse(self.script("return fire.key('a')"))
        self.assertFalse(self.script("return fire.key('+', {ctrlKey: true})"))

    def test_the_fit_button_shows_the_whole_tree_again(self):
        start = self.view_state()
        self.script("fire.wheel(-300, 600, 300); fire.wheel(-300, 600, 300, {shiftKey: true}); return 1")
        self.assertFalse(self.view_state()["fitted"])
        self.js("document.getElementById('fit-btn').click()")
        again = self.view_state()
        self.assertTrue(again["fitted"])
        self.assertEqual((again["sx"], again["sy"], again["ox"], again["oy"]), (start["sx"], start["sy"], start["ox"], start["oy"]))

    def test_a_fitted_view_follows_the_window_and_a_zoomed_one_keeps_its_place(self):
        fit = self.view_state()
        self.view.resize(900, 600)
        self.wait_for(f"document.getElementById('tree-canvas').clientWidth !== {fit['w']} && TreeView.state.view.sx !== {fit['sx']}")
        smaller = self.view_state()
        self.assertTrue(smaller["fitted"])
        self.assertLess(smaller["sx"], fit["sx"])
        self.assertLessEqual(smaller["oy"] + 300 * smaller["sy"], smaller["h"])
        self.script("fire.wheel(-200, 400, 300); return 1")
        zoomed = self.view_state()
        self.view.resize(1100, 700)
        self.wait_for(f"document.getElementById('tree-canvas').clientWidth !== {zoomed['w']}")
        self.pump(0.3)
        after = self.view_state()
        self.assertEqual((after["sx"], after["sy"], after["ox"], after["oy"]), (zoomed["sx"], zoomed["sy"], zoomed["ox"], zoomed["oy"]))

    def test_double_clicking_a_clade_brings_it_to_fill_the_view(self):
        view = self.script("""
            const s = TreeView.state, m = s.model;
            const box = TreeView.nodePixels(s, m.leafNode[120]);
            const hit = TreeView.hitTest(box.x - 2, box.y);
            fire.dblclick(box.x - 2, box.y);
            const after = TreeView.nodePixels(s, hit.node);
            return {node: hit.node, top: after.top, bottom: after.bottom, h: document.getElementById('tree-canvas').clientHeight,
                    rows: m.lastLeaf[hit.node] - m.firstLeaf[hit.node] + 1, fitted: s.fitted, limit: TreeView.limits().syMax};""")
        self.assertGreaterEqual(view["top"], 0)
        self.assertLessEqual(view["bottom"], view["h"])
        self.assertFalse(view["fitted"])
        # A clade fills the view's height less its margins, or stands at the tallest row the zoom allows; a leaf
        # is put in the middle at a readable row height.
        if view["rows"] > 1:
            self.assertAlmostEqual(view["bottom"] - view["top"], min(view["rows"] * view["limit"], view["h"] - 48), delta=1.5)

    def test_double_clicking_empty_space_shows_the_whole_tree(self):
        self.script("fire.wheel(-300, 600, 300); return 1")
        self.script("fire.dblclick(5, 5); return 1")
        self.assertTrue(self.view_state()["fitted"])

    def test_a_leaf_is_brought_to_the_middle_at_a_readable_row_height(self):
        self.script("TreeView.focusLeaf(150); return 1")
        info = self.script("""
            const s = TreeView.state;
            const box = TreeView.nodePixels(s, s.model.leafNode[150]);
            const stats = TreeView.drawNow();
            return {y: box.y, h: document.getElementById('tree-canvas').clientHeight, sy: s.view.sy, labels: stats.labels, tip: box.x};""")
        self.assertAlmostEqual(info["y"], info["h"] / 2, delta=1)
        self.assertGreaterEqual(info["sy"], 16)
        self.assertGreater(info["labels"], 0)
        self.assertGreaterEqual(info["tip"], 40)

    def test_the_ruler_jumps_to_where_it_is_pressed_and_follows_the_pointer(self):
        self.script("fire.wheel(-400, 600, 300); return 1")
        info = self.script("""
            const canvas = document.getElementById('tree-canvas'), s = TreeView.state;
            const w = canvas.clientWidth, h = canvas.clientHeight;
            const middleRow = () => (h / 2 - s.view.oy) / s.view.sy;
            fire.pointer('pointerdown', w - 5, h * 0.5);
            const half = middleRow();
            fire.pointer('pointermove', w - 5, h * 0.25);
            const quarter = middleRow();
            fire.pointer('pointerup', w - 5, h * 0.25);
            return {half, quarter, rows: s.model.leafCount, pinned: s.pinned};""")
        self.assertAlmostEqual(info["half"], info["rows"] * 0.5, delta=1)
        self.assertAlmostEqual(info["quarter"], info["rows"] * 0.25, delta=1)
        self.assertEqual(info["pinned"], -1)


class HitTestTests(TreeCase):
    """What is under the pointer, found by going down the row: compared with looking at every node."""

    SCRIPT = """
        const s = TreeView.state, m = s.model, L = s.layout, tol = 5;
        const brute = (px, py) => {
            // The view is replaced by a zoom to a clade, so it is read when it is used.
            const v = s.view;
            const found = [];
            for (let i = 0; i < m.nodeCount; i++) {
                const b = TreeView.nodePixels(s, i);
                if (m.childCount[i] === 0) {
                    if (Math.abs(py - b.y) <= Math.max(tol, v.sy / 2) && px >= b.x0 - tol && px <= b.x + tol) found.push([i, 'leaf']);
                } else {
                    if (Math.abs(py - b.y) <= tol && px >= b.x0 - tol && px <= b.x + tol) found.push([i, 'branch']);
                    const f = m.childList[m.childStart[i]], l = m.childList[m.childStart[i + 1] - 1];
                    if (Math.abs(px - b.x) <= tol && py >= v.oy + L.y[f] * v.sy - tol && py <= v.oy + L.y[l] * v.sy + tol) found.push([i, 'connector']);
                    // The square that folds a clade up, at its node, for the clade hovered or selected, if big enough for one.
                    if ((i === s.hover || i === s.pinned) && i !== m.root && Math.max(b.right - b.left, b.bottom - b.top) >= TreeView.HANDLE_CLADE_PX
                        && Math.abs(px - b.x) <= TreeView.HANDLE_PX / 2 + 1 && Math.abs(py - b.y) <= TreeView.HANDLE_PX / 2 + 1) found.push([i, 'handle']);
                }
            }
            return found;
        };
    """

    def test_a_point_on_a_branch_finds_its_node_and_every_hit_is_on_a_line(self):
        result = self.script(f"""
            {self.SCRIPT}
            TreeView.zoomToNode({clade_of(40, 70)});
            TreeView.drawNow();
            const unseen = [], unsound = [];
            let tried = 0;
            for (let i = 0; i < m.nodeCount; i++) {{
                const b = TreeView.nodePixels(s, i);
                if (b.top > 0 && b.bottom < 760 && b.x0 > 5 && b.x < 1250 && b.x - b.x0 > 14) {{
                    tried++;
                    const hit = TreeView.hitTest((b.x0 + b.x) / 2, b.y);
                    if (!hit || hit.node !== i) unseen.push([i, hit]);
                }}
            }}
            // The line that joins a clade's children is found as that clade, away from where its own branch ends.
            const unjoined = [];
            let joins = 0;
            for (let i = 0; i < m.nodeCount; i++) {{
                if (m.childCount[i] === 0) continue;
                const b = TreeView.nodePixels(s, i);
                const f = m.childList[m.childStart[i]], l = m.childList[m.childStart[i + 1] - 1];
                const top = s.view.oy + L.y[f] * s.view.sy, bottom = s.view.oy + L.y[l] * s.view.sy;
                if (bottom - top > 60 && top > 5 && bottom < 760 && b.x - b.x0 > 14 && b.x < 1250) {{
                    joins++;
                    const hit = TreeView.hitTest(b.x, top + 0.25 * (bottom - top));
                    if (!hit || hit.node !== i || hit.part !== 'connector') unjoined.push([i, hit]);
                }}
            }}
            let seeded = 12345;
            const random = () => (seeded = (seeded * 1103515245 + 12345) % 2147483648) / 2147483648;
            let hits = 0;
            for (let n = 0; n < 4000; n++) {{
                const px = random() * 1286, py = random() * 800;
                const hit = TreeView.hitTest(px, py);
                if (!hit) continue;
                // A leaf's name, when the rows are tall enough to show it, is on no line.
                if (hit.part === 'leaf' && px > TreeView.nodePixels(s, hit.node).x + tol) continue;
                hits++;
                if (!brute(px, py).some(([i, part]) => i === hit.node && part === hit.part)) unsound.push([px, py, hit]);
            }}
            return {{tried, unseen: unseen.slice(0, 5), unjoined: unjoined.slice(0, 5), joins, unsound: unsound.slice(0, 5), hits}};""")
        self.assertGreater(result["tried"], 20)
        self.assertEqual(result["unseen"], [])
        self.assertGreater(result["joins"], 3)
        self.assertEqual(result["unjoined"], [])
        self.assertGreater(result["hits"], 50)
        self.assertEqual(result["unsound"], [])

    def test_nothing_is_found_in_empty_space_or_off_the_tree(self):
        result = self.script(f"""
            const clade = {clade_of(10, 14)};
            TreeView.zoomToNode(clade);
            TreeView.drawNow();
            const box = TreeView.nodePixels(TreeView.state, clade);
            return {{
                above: TreeView.hitTest(300, box.top - 200),
                below: TreeView.hitTest(300, box.bottom + 200),
                farLeft: TreeView.hitTest(box.x0 - 50, (box.top + box.bottom) / 2),
            }};""")
        self.assertEqual(result, {"above": None, "below": None, "farLeft": None})

    def test_a_leaf_is_found_by_its_name_as_well_as_its_branch(self):
        result = self.script("""
            TreeView.focusLeaf(100);
            TreeView.drawNow();
            const s = TreeView.state, m = s.model;
            const box = TreeView.nodePixels(s, m.leafNode[100]);
            const onName = TreeView.hitTest(box.x + 40, box.y);
            const pastName = TreeView.hitTest(box.x + 900, box.y);
            return {onName, pastName, leaf: m.leafNode[100]};""")
        self.assertEqual(result["onName"], {"node": result["leaf"], "part": "leaf"})
        self.assertIsNone(result["pastName"])


class TooltipAndPinTests(TreeCase):
    def hover(self, x, y):
        self.script(f"fire.move({x}, {y}); return 1")
        self.wait_for("document.getElementById('tooltip').style.display === 'block'")

    def name_point(self, leaf):
        """The pixel of a point on the name of leaf number ``leaf``, brought into view."""
        self.script(f"TreeView.focusLeaf({leaf}); TreeView.drawNow(); return 1")
        box = self.state(tip_of(leaf))
        return box["x"] + 40, box["y"]

    def test_hovering_a_leaf_names_it_and_says_whether_it_is_in_the_network(self):
        self.hover(*self.name_point(100))
        tooltip = self.script("""
            const tip = document.getElementById('tooltip');
            return {name: tip.querySelector('.name').textContent, lines: [...tip.querySelectorAll('.line')].map(l => l.textContent),
                    hover: TreeView.state.hover, label: TreeView.state.model.labels[100]};""")
        self.assertEqual(tooltip["name"], tooltip["label"])
        self.assertEqual(tooltip["lines"][0], "A node of this network")
        self.assertRegex(tooltip["lines"][1], r"^Branch length: \d")

    @staticmethod
    def below(tree, node):
        """A boolean array over nodes: which are in the subtree of ``node`` (post-order: its block of nodes)."""
        found = np.zeros(tree.node_count, dtype=bool)
        found[node] = True
        for other in range(node - 1, -1, -1):
            if found[tree.parent[other]]:
                found[other] = True
        return found

    def test_hovering_a_clade_gives_its_leaf_counts_and_the_root_is_the_root(self):
        node = self.state(clade_of(8, 14))
        tree = read_guide_tree(self.tree_file)
        is_leaf = np.isin(np.arange(tree.node_count), tree.leaf_nodes())
        count = int(np.count_nonzero(is_leaf & self.below(tree, node)))
        described = self.state(f"TreeView.describeNode({node})")
        self.assertEqual(described["title"], "A clade")
        self.assertEqual(described["lines"][0], f"Leaves in this clade: {count}")
        self.assertEqual(described["lines"][1], f"Of them in this network: {count}")
        root = self.state("TreeView.describeNode(TreeView.state.model.root)")
        self.assertEqual(root["title"], "The root of the tree")
        self.assertEqual(root["lines"][0], "Leaves in this clade: 300")
        self.assertFalse(any(line.startswith("Branch length") for line in root["lines"]))

    def test_the_tooltip_goes_when_the_pointer_moves_off_or_leaves(self):
        point = self.name_point(100)
        self.hover(*point)
        self.script("fire.move(5, 5); return 1")
        self.wait_for("document.getElementById('tooltip').style.display === 'none'")
        self.hover(*point)
        self.script("fire.canvas().dispatchEvent(new PointerEvent('pointerleave', {bubbles: false})); return 1")
        self.assertEqual(self.state("document.getElementById('tooltip').style.display"), "none")
        self.assertEqual(self.state("TreeView.state.hover"), -1)

    def test_a_click_selects_a_node_for_the_status_line_and_another_click_lets_go(self):
        x, y = self.name_point(100)
        counts = "300 leaves, 300 of them nodes of this network"
        self.assertEqual(self.state("document.getElementById('summary').textContent"), counts)
        self.script(f"fire.click({x}, {y}); return 1")
        said = self.state("document.getElementById('summary').textContent")
        label = self.state("TreeView.state.model.labels[100]")
        self.assertTrue(said.startswith(label), said)
        self.assertIn("A node of this network", said)
        self.script(f"fire.click({x}, {y}); return 1")
        self.assertEqual(self.state("document.getElementById('summary').textContent"), counts)
        self.script(f"fire.click({x}, {y}); fire.key('Escape'); return 1")
        self.assertEqual(self.state("TreeView.state.pinned"), -1)
        self.assertEqual(self.state("document.getElementById('summary').textContent"), counts)


class SupportTests(TreePageMixin, unittest.TestCase):
    """Split support is the share of replicate trees that hold a split: said so, and never as bootstrap."""

    def test_a_clade_with_support_says_how_many_replicate_trees_hold_it(self):
        tool = writer()
        headers = headers_of(30)
        link = random_linkage(30, seed=3)
        keys = tool.guide_tree_leaf_keys(30, 42)
        replicate_keys = np.sort(np.concatenate([tool.replicate_split_keys(link, 30, keys)] * 2))
        self.serve_headers(headers, linkage=link, leaf_keys=keys, replicate_keys=replicate_keys, num_trees=2)
        self.open_page()
        lines = self.script("""
            const m = TreeView.state.model;
            const out = [];
            for (let i = 0; i < m.nodeCount; i++) if (m.support[i] >= 0) out.push([m.support[i], TreeView.describeNode(i).lines]);
            return out;""")
        self.assertTrue(lines)
        for support, described in lines:
            self.assertIn(f"Stable in {support}% of the replicate trees", described)
        text = self.state("document.body.innerText + TreeView.describeNode(TreeView.state.model.root).lines.join(' ')")
        self.assertNotIn("bootstrap", text.lower())
        self.js("document.getElementById('details-btn').click()")
        self.assertIn("not bootstrap", self.state("document.getElementById('details').textContent"))


class SearchTests(TreeCase):
    def test_a_substring_matches_in_any_case_in_tree_order(self):
        count = self.state("TreeView.runSearch('prot_1')")
        expected = [header for header in self.headers if "prot_1" in header.lower()]
        self.assertEqual(count, len(expected))
        matched = self.state("Array.from(TreeView.state.matches, k => TreeView.state.model.labels[k])")
        self.assertEqual(sorted(matched), sorted(expected))
        ordinals = self.state("Array.from(TreeView.state.matches)")
        self.assertEqual(ordinals, sorted(ordinals))

    def test_a_header_in_quotes_matches_only_itself(self):
        exact = self.headers[7]
        self.assertEqual(self.state(f"TreeView.runSearch({json.dumps(chr(34) + exact + chr(34))})"), 1)
        self.assertEqual(self.state(f"TreeView.runSearch({json.dumps(chr(34) + exact[:-3] + chr(34))})"), 0)
        # Two quotes and nothing between them name a header that is empty, which none is.
        self.assertEqual(self.state(f"TreeView.runSearch({json.dumps(chr(34) * 2)})"), 0)

    def test_nothing_in_a_query_is_a_pattern(self):
        # None of these is in any header, and none is read as a pattern (a regular expression would match all or fail).
        for query in ("PROT_1[", "(", ".*", "\\", "*", "[a-z]", "^sp"):
            with self.subTest(query=query):
                self.assertEqual(self.state(f"TreeView.runSearch({json.dumps(query)})"), 0)
        self.assertEqual(self.state("TreeView.runSearch('|')"), 300)

    def test_an_empty_search_finds_nothing_and_clears_the_last(self):
        self.state("TreeView.runSearch('PROT')")
        self.assertEqual(self.state("TreeView.runSearch('   ')"), 0)
        self.assertEqual(self.state("TreeView.state.matches.length"), 0)

    def type(self, text):
        self.script(f"""
            const input = document.getElementById('search-input');
            input.value = {json.dumps(text)};
            input.dispatchEvent(new Event('input', {{bubbles: true}}));
            return 1;""")

    def ui(self):
        return self.script("""
            return {count: document.getElementById('search-count').textContent,
                    prev: document.getElementById('search-prev').disabled, next: document.getElementById('search-next').disabled,
                    hasText: document.getElementById('search-box').classList.contains('has-text'),
                    current: TreeView.state.current, matches: TreeView.state.matches.length};""")

    def test_typing_finds_the_first_match_and_goes_to_it(self):
        self.type("PROT_29")
        self.wait_for("TreeView.state.current === 0")
        ui = self.ui()
        # PROT_29 and PROT_290 to PROT_299.
        self.assertEqual(ui["matches"], 11)
        self.assertEqual(ui["count"], "1 of 11")
        self.assertFalse(ui["prev"] or ui["next"])
        self.assertTrue(ui["hasText"])
        leaf_y = self.state("(() => { const s = TreeView.state; return TreeView.nodePixels(s, s.model.leafNode[s.matches[0]]).y; })()")
        height = self.state("document.getElementById('tree-canvas').clientHeight")
        self.assertAlmostEqual(leaf_y, height / 2, delta=1)

    def test_enter_and_shift_enter_step_through_the_matches_and_round_the_ends(self):
        self.type("PROT_29")
        self.wait_for("TreeView.state.current === 0")
        self.script("""
            document.getElementById('search-input').dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
            return 1;""")
        self.assertEqual(self.ui()["count"], "2 of 11")
        self.script("""
            const input = document.getElementById('search-input');
            for (let i = 0; i < 2; i++) input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', shiftKey: true, bubbles: true}));
            return 1;""")
        self.assertEqual(self.ui()["count"], "11 of 11")  # before the first is the last
        self.script("document.getElementById('search-next').click(); return 1")
        self.assertEqual(self.ui()["count"], "1 of 11")
        self.script("document.getElementById('search-prev').click(); return 1")
        self.assertEqual(self.ui()["count"], "11 of 11")

    def test_a_query_that_finds_nothing_says_so_and_clearing_it_goes_quiet(self):
        self.type("no such header")
        self.wait_for("document.getElementById('search-count').textContent === 'No match'")
        ui = self.ui()
        self.assertTrue(ui["prev"] and ui["next"])
        self.script("document.getElementById('search-clear').click(); return 1")
        ui = self.ui()
        self.assertEqual((ui["count"], ui["hasText"], ui["matches"]), ("", False, 0))
        self.assertEqual(self.state("document.activeElement.id"), "search-input")

    def test_escape_empties_the_search_and_returns_to_the_tree(self):
        self.type("PROT_1")
        self.wait_for("TreeView.state.current === 0")
        self.script("document.getElementById('search-input').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true})); return 1")
        self.assertEqual(self.state("document.getElementById('search-input').value"), "")
        self.assertEqual(self.state("TreeView.state.matches.length"), 0)
        self.assertEqual(self.state("document.activeElement.id"), "tree-canvas")

    def test_the_slash_key_goes_to_the_search_box(self):
        self.script("document.getElementById('tree-canvas').focus(); return 1")
        self.script("document.dispatchEvent(new KeyboardEvent('keydown', {key: '/', bubbles: true, cancelable: true})); return 1")
        self.assertEqual(self.state("document.activeElement.id"), "search-input")

    def test_the_matches_are_marked_on_the_ruler_in_the_highlight_colour(self):
        self.type("PROT_29")
        self.wait_for("TreeView.state.current === 0")
        info = self.script("""
            TreeView.fitView(); TreeView.drawNow();
            const s = TreeView.state, canvas = document.getElementById('tree-canvas');
            const h = canvas.clientHeight, w = canvas.clientWidth;
            const ctx = document.createElement('canvas').getContext('2d');
            ctx.fillStyle = s.palette.match; ctx.fillRect(0, 0, 1, 1);
            const match = Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3));
            const ticks = Array.from(s.matches, k => Math.round((k + 0.5) / s.model.leafCount * h));
            let quiet = 3;
            while (ticks.some(y => Math.abs(y - quiet) < 4)) quiet++;
            return {match, seen: ticks.map(y => fire.pixel(w - 8, y).slice(0, 3)), none: fire.pixel(w - 8, quiet).slice(0, 3)};""")
        for pixel in info["seen"]:
            self.assertEqual(pixel, info["match"])
        self.assertNotEqual(info["none"], info["match"])


# ---------------------------------------------------------------------------------------------------------------------
# Phase 2: the three layouts, folding clades, the overview map, level of detail
# ---------------------------------------------------------------------------------------------------------------------

def nj_options(count, seed=4):
    """The writer's options for the neighbor-joining tree of ``count`` random points: a tree its file flags [&U]."""
    linkage, lengths = neighbor_joining_tree(count, seed)
    return {"linkage": linkage, "nj_branch_lengths": lengths}


class LayoutCase(TreePageMixin, unittest.TestCase):
    """A page over a neighbor-joining tree of 60 leaves (a file that says it is unrooted), in a view of 1300 x 800."""

    COUNT = 60

    def setUp(self):
        super().setUp()
        self.headers = headers_of(self.COUNT)
        self.serve_headers(self.headers, **nj_options(self.COUNT))
        self.open_page()

    def tearDown(self):
        self.view.resize(1300, 800)
        super().tearDown()

    def view_state(self):
        return self.state("""({...TreeView.state.view, fitted: TreeView.state.fitted,
            w: document.getElementById('tree-canvas').clientWidth, h: document.getElementById('tree-canvas').clientHeight})""")

    def choose(self, kind=None, mode=None):
        """Choose a layout and form as a person does, with the page's own controls."""
        if kind:
            self.script(f"""const select = document.getElementById('layout-select');
                select.value = {json.dumps(kind)}; select.dispatchEvent(new Event('change')); return 1""")
        if mode:
            self.js(f"document.getElementById('{'phylogram' if mode == 'lengths' else 'cladogram'}-btn').click()")
        self.script("TreeView.drawNow(); return 1")


class LayoutChoiceTests(LayoutCase):
    def test_an_unrooted_file_opens_unrooted_and_a_rooted_one_rectangular(self):
        info = self.script("""return {kind: TreeView.state.kind, shown: TreeView.state.layout.kind, mode: TreeView.state.layout.mode,
            select: document.getElementById('layout-select').value, rooted: TreeView.state.model.rooted,
            hint: document.getElementById('hints').textContent};""")
        self.assertEqual((info["kind"], info["shown"], info["select"], info["rooted"]), ("unrooted",) * 2 + ("unrooted", False))
        self.assertEqual(info["mode"], "lengths")
        self.assertEqual(info["hint"], "Wheel: zoom. Drag: move. Double-click a clade: zoom to it. Click its square to fold it up.")
        # UPGMA's file says it is rooted; one that says neither is taken as rooted.
        self.serve_headers(headers_of(20))
        self.open_page()
        self.assertEqual(self.state("[TreeView.state.kind, document.getElementById('layout-select').value, TreeView.state.model.rooted]"),
                         ["rectangular", "rectangular", True])
        folder = self.make_folder()
        write_tree_file(folder, list("abc"), "#NEXUS\nBEGIN TREES;\n  TREE t = ((a:1,b:2):1,c:3);\nEND;\n")
        self.serve_headers(list("abc"), folder=folder, tree=None)
        self.open_page()
        self.assertEqual(self.state("[TreeView.state.kind, TreeView.state.model.rooted]"), ["rectangular", None])

    def test_the_selector_changes_the_layout_and_shows_the_whole_tree_again(self):
        self.script("TreeView.zoomAbout(600, 300, 3, 3); return 1")
        self.assertFalse(self.view_state()["fitted"])
        for kind in ("circular", "rectangular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                info = self.script("""const s = TreeView.state, v = s.view, L = s.layout, b = L.bounds;
                    return {kind: s.kind, layoutKind: L.kind, fitted: s.fitted, uniform: L.uniform, sx: v.sx, sy: v.sy,
                            left: v.ox + b.minX * v.sx, right: v.ox + b.maxX * v.sx, top: v.oy + b.minY * v.sy, bottom: v.oy + b.maxY * v.sy,
                            w: document.getElementById('tree-canvas').clientWidth, h: document.getElementById('tree-canvas').clientHeight};""")
                self.assertEqual((info["kind"], info["layoutKind"], info["fitted"]), (kind, kind, True))
                self.assertEqual(info["uniform"], kind != "rectangular")
                self.assertGreaterEqual(info["left"], 0)
                self.assertLessEqual(info["right"], info["w"])
                self.assertGreaterEqual(info["top"], 0)
                self.assertLessEqual(info["bottom"], info["h"])
                if info["uniform"]:
                    self.assertEqual(info["sx"], info["sy"])
        self.assertEqual(self.console_errors(), [])

    def test_the_branch_form_follows_the_buttons_and_keeps_the_layout(self):
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind, "cladogram")
                info = self.script("""return {mode: TreeView.state.layout.mode, kind: TreeView.state.kind,
                    pressed: ['phylogram-btn', 'cladogram-btn'].map(id => document.getElementById(id).getAttribute('aria-pressed'))};""")
                self.assertEqual((info["mode"], info["kind"], info["pressed"]), ("cladogram", kind, ["false", "true"]))
                self.choose(None, "lengths")
                info = self.script("""return {mode: TreeView.state.layout.mode,
                    pressed: ['phylogram-btn', 'cladogram-btn'].map(id => document.getElementById(id).getAttribute('aria-pressed'))};""")
                self.assertEqual((info["mode"], info["pressed"]), ("lengths", ["true", "false"]))

    def test_changing_the_layout_keeps_the_branch_form(self):
        for mode in ("cladogram", "lengths"):
            with self.subTest(mode=mode):
                self.choose(None, mode)
                for kind in ("circular", "unrooted", "rectangular"):
                    self.choose(kind)
                    self.assertEqual(self.state("TreeView.state.layout.mode"), mode, kind)

    def test_a_tree_with_no_branch_lengths_has_no_phylogram_to_choose(self):
        folder = self.make_folder()
        write_tree_file(folder, list("abcd"), "#NEXUS\nBEGIN TREES;\n  TREE t = [&R] ((a,b),(c,d));\nEND;\n")
        self.serve_headers(list("abcd"), folder=folder, tree=None)
        self.open_page()
        info = self.script("""return {mode: TreeView.state.layout.mode, disabled: ['phylogram-btn', 'cladogram-btn'].map(id => document.getElementById(id).disabled),
            pressed: ['phylogram-btn', 'cladogram-btn'].map(id => document.getElementById(id).getAttribute('aria-pressed'))};""")
        self.assertEqual((info["mode"], info["disabled"], info["pressed"]), ("cladogram", [True, True], ["false", "true"]))
        for kind in ("circular", "unrooted"):
            self.choose(kind)
            self.assertEqual(self.state("TreeView.state.layout.mode"), "cladogram")

    def test_the_controls_wait_for_a_tree(self):
        viewer = server_viewer()
        viewer.alignment, viewer.full_headers = None, []
        self.start_page_server(viewer)
        self.open_page("empty")
        disabled = self.state("""['layout-select', 'phylogram-btn', 'cladogram-btn', 'unfold-btn', 'overview-btn', 'fit-btn']
            .map(id => document.getElementById(id).disabled)""")
        self.assertEqual(disabled, [True] * 6)
        self.assertTrue(self.state("document.getElementById('minimap').hidden"))


class LayoutGeometryTests(LayoutCase):
    """The layouts, as the arrays they come to: every branch its length, every leaf in its place."""

    COUNT = 80

    def test_every_branch_is_as_long_as_its_length_in_a_phylogram_and_a_step_in_a_cladogram(self):
        bad = self.script("""
            const m = TreeView.state.model, wrong = [];
            const near = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
            for (const kind of ['rectangular', 'circular', 'unrooted']) {
                const lengths = TreeView.layoutTree(m, kind, 'lengths'), steps = TreeView.layoutTree(m, kind, 'cladogram');
                if (lengths.mode !== 'lengths' || steps.mode !== 'cladogram') wrong.push([kind, 'mode']);
                for (let i = 0; i < m.nodeCount - 1; i++) {
                    const p = m.parent[i];
                    // The distance between a node and its parent along the branch: x for a rectangular tree, the radius for a
                    // circular one, the straight line for an unrooted one.
                    const along = (L) => kind === 'rectangular' ? L.x[i] - L.x[p] : kind === 'circular' ? L.depth[i] - L.depth[p]
                        : Math.hypot(L.x[i] - L.x[p], L.y[i] - L.y[p]);
                    if (!near(along(lengths), Math.max(0, m.length[i]))) wrong.push([kind, 'length', i, along(lengths), m.length[i]]);
                    if (kind === 'unrooted' ? !near(along(steps), 1) : !(along(steps) >= 1 - 1e-9)) wrong.push([kind, 'step', i, along(steps)]);
                }
                // In a cladogram every leaf stands at the end.
                if (kind !== 'unrooted') {
                    const ends = new Set(Array.from(m.leafNode, (node) => (kind === 'rectangular' ? steps.x[node] : steps.depth[node])));
                    if (ends.size !== 1) wrong.push([kind, 'cladogram leaves', [...ends]]);
                }
            }
            return wrong;""")
        self.assertEqual(bad, [])

    def test_the_leaves_keep_their_order_and_share_the_circle_in_proportion(self):
        bad = self.script("""
            const m = TreeView.state.model, wrong = [];
            const near = (a, b) => Math.abs(a - b) <= 1e-9;
            const L = m.leafCount;
            const rect = TreeView.layoutTree(m, 'rectangular', 'lengths'), circ = TreeView.layoutTree(m, 'circular', 'lengths');
            const unr = TreeView.layoutTree(m, 'unrooted', 'lengths');
            for (let k = 0; k < L; k++) {
                const node = m.leafNode[k];
                if (!near(rect.y[node], k + 0.5)) wrong.push(['rect row', k]);
                if (!near(circ.angle[node], -Math.PI / 2 + 2 * Math.PI * (k + 0.5) / L)) wrong.push(['circular angle', k]);
            }
            // Equal angle: the root has the whole circle; a clade has the share its leaves are of the leaves, its children
            // divide it end to end in order, and its branch runs along the middle of its share.
            if (!near(unr.low[m.root], -Math.PI) || !near(unr.high[m.root], Math.PI)) wrong.push(['root share']);
            for (let i = 0; i < m.nodeCount; i++) {
                const share = unr.high[i] - unr.low[i];
                if (!near(share, 2 * Math.PI * TreeView.leavesUnder(m, i) / L)) wrong.push(['share', i]);
                if (m.childCount[i] > 0) {
                    let at = unr.low[i];
                    for (let k = m.childStart[i]; k < m.childStart[i + 1]; k++) {
                        const child = m.childList[k];
                        if (!near(unr.low[child], at)) wrong.push(['child start', i, child]);
                        at = unr.high[child];
                    }
                    if (!near(at, unr.high[i])) wrong.push(['children end', i]);
                }
                if (i !== m.root && m.length[i] > 1e-6) {
                    const p = m.parent[i], direction = Math.atan2(unr.y[i] - unr.y[p], unr.x[i] - unr.x[p]);
                    if (!near(Math.cos(direction), Math.cos((unr.low[i] + unr.high[i]) / 2))) wrong.push(['direction', i]);
                }
            }
            return wrong;""")
        self.assertEqual(bad, [])

    def test_circular_and_unrooted_layouts_equal_an_independent_computation_from_the_file(self):
        tree = read_guide_tree(self.tree_file)
        count, parent, length = tree.node_count, tree.parent, tree.length.astype(np.float64)
        children = [[] for _ in range(count)]
        for node in range(count - 1):
            children[parent[node]].append(node)
        under = np.zeros(count)
        for node in range(count):
            under[node] = sum(under[child] for child in children[node]) if children[node] else 1
        # The equal-angle layout.
        low, high, x, y = np.zeros(count), np.zeros(count), np.zeros(count), np.zeros(count)
        low[-1], high[-1] = -np.pi, np.pi
        for node in range(count - 1, -1, -1):
            start = low[node]
            for child in children[node]:
                share = (high[node] - low[node]) * under[child] / under[node]
                low[child], high[child] = start, start + share
                start += share
                direction = (low[child] + high[child]) / 2
                x[child] = x[node] + max(0.0, length[child]) * np.cos(direction)
                y[child] = y[node] + max(0.0, length[child]) * np.sin(direction)
        # The circular layout: the leaves round the circle from the top, a clade at the middle of its first and last child.
        angle, depth = np.zeros(count), np.zeros(count)
        leaf = 0
        for node in range(count):
            if children[node]:
                angle[node] = (angle[children[node][0]] + angle[children[node][-1]]) / 2
            else:
                angle[node] = -np.pi / 2 + 2 * np.pi * (leaf + 0.5) / tree.leaf_count
                leaf += 1
        for node in range(count - 2, -1, -1):
            depth[node] = depth[parent[node]] + max(0.0, length[node])
        seen = self.script("""const m = TreeView.state.model, u = TreeView.layoutTree(m, 'unrooted', 'lengths'), c = TreeView.layoutTree(m, 'circular', 'lengths');
            return {ux: Array.from(u.x), uy: Array.from(u.y), cx: Array.from(c.x), cy: Array.from(c.y), angle: Array.from(c.angle)};""")
        np.testing.assert_allclose(seen["ux"], x, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(seen["uy"], y, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(seen["angle"], angle, rtol=1e-12)
        np.testing.assert_allclose(seen["cx"], depth * np.cos(angle), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(seen["cy"], depth * np.sin(angle), rtol=1e-9, atol=1e-9)

    def test_every_nodes_box_holds_its_lines_its_shape_and_its_childrens_boxes(self):
        bad = self.script("""
            const m = TreeView.state.model, wrong = [], o = new Float64Array(8);
            for (const kind of ['rectangular', 'circular', 'unrooted']) for (const mode of ['lengths', 'cladogram']) {
                const L = TreeView.layoutTree(m, kind, mode), b = L.boxes;
                const inside = (i, x, y) => x >= b.x0[i] - 1e-9 && x <= b.x1[i] + 1e-9 && y >= b.y0[i] - 1e-9 && y <= b.y1[i] + 1e-9;
                for (let i = 0; i < m.nodeCount; i++) {
                    L.stem(i, o);
                    if (!inside(i, o[0], o[1]) || !inside(i, o[2], o[3])) wrong.push([kind, mode, i, 'stem']);
                    L.join(i, o);
                    if (o[0] === 1 && (!inside(i, o[1], o[2]) || !inside(i, o[3], o[4]))) wrong.push([kind, mode, i, 'line']);
                    if (o[0] === 2) for (let s = 0; s <= 16; s++) {
                        const a = o[2] + (o[3] - o[2]) * s / 16;
                        if (!inside(i, o[1] * Math.cos(a), o[1] * Math.sin(a))) wrong.push([kind, mode, i, 'arc', s]);
                    }
                    if (m.childCount[i] > 0) {
                        L.wedge(i, o);
                        if (o[0] === 0) {
                            for (const [x, y] of [[o[1], o[2]], [o[3], o[4]], [o[5], o[6]]]) if (!inside(i, x, y)) wrong.push([kind, mode, i, 'triangle']);
                        } else {
                            if (!inside(i, o[1], o[2])) wrong.push([kind, mode, i, 'point']);
                            for (let s = 0; s <= 16; s++) {
                                const a = o[6] + (o[7] - o[6]) * s / 16;
                                if (!inside(i, o[3] + o[5] * Math.cos(a), o[4] + o[5] * Math.sin(a))) wrong.push([kind, mode, i, 'slice', s]);
                            }
                        }
                    }
                    const p = m.parent[i];
                    if (p >= 0 && !(b.x0[i] >= b.x0[p] - 1e-9 && b.x1[i] <= b.x1[p] + 1e-9 && b.y0[i] >= b.y0[p] - 1e-9 && b.y1[i] <= b.y1[p] + 1e-9)) wrong.push([kind, mode, i, 'child box']);
                }
            }
            return wrong.slice(0, 10);""")
        self.assertEqual(bad, [])

    def test_the_tiny_clades_line_goes_from_the_node_to_where_the_clade_reaches(self):
        bad = self.script("""
            const m = TreeView.state.model, wrong = [], o = new Float64Array(8), w = new Float64Array(8);
            for (const kind of ['rectangular', 'circular', 'unrooted']) {
                const L = TreeView.layoutTree(m, kind, 'lengths'), b = L.boxes;
                for (let i = 0; i < m.nodeCount; i++) {
                    if (m.childCount[i] === 0) continue;
                    L.mark(i, o);
                    L.stem(i, w);
                    if (Math.hypot(o[0] - w[2], o[1] - w[3]) > 1e-9) wrong.push([kind, i, 'starts at the node']);
                    for (const [x, y] of [[o[0], o[1]], [o[2], o[3]]]) {
                        if (x < b.x0[i] - 1e-9 || x > b.x1[i] + 1e-9 || y < b.y0[i] - 1e-9 || y > b.y1[i] + 1e-9) wrong.push([kind, i, 'inside the box']);
                    }
                    if (!(L.thickness(i, 1, 1) >= 0)) wrong.push([kind, i, 'thickness']);
                }
            }
            return wrong.slice(0, 10);""")
        self.assertEqual(bad, [])

    def test_everything_of_an_unrooted_clade_lies_in_the_slice_of_the_circle_that_is_its_share(self):
        # A clade that is long but a few pixels wide at its far end is drawn as one line along the middle of its slice.
        # That stands for the clade only if nothing of it lies outside the slice, from its node as far as it reaches.
        bad = self.script("""
            const m = TreeView.state.model, wrong = [], tau = 2 * Math.PI;
            for (const mode of ['lengths', 'cladogram']) {
                const L = TreeView.layoutTree(m, 'unrooted', mode);
                for (let k = 0; k < m.nodeCount; k++) {
                    for (let a = m.parent[k]; a >= 0; a = m.parent[a]) {
                        const dx = L.x[k] - L.x[a], dy = L.y[k] - L.y[a], r = Math.hypot(dx, dy);
                        if (r > L.far[a] + 1e-9) wrong.push([mode, k, a, 'beyond its reach']);
                        if (r > 1e-9) {
                            const turn = (((Math.atan2(dy, dx) - L.low[a]) % tau) + tau) % tau;
                            if (turn > L.high[a] - L.low[a] + 1e-9) wrong.push([mode, k, a, 'outside its slice']);
                        }
                    }
                }
            }
            return wrong.slice(0, 10);""")
        self.assertEqual(bad, [])

    def test_an_unrooted_clades_line_runs_along_the_middle_of_its_slice_and_is_no_wider_than_it(self):
        bad = self.script("""
            const m = TreeView.state.model, wrong = [], o = new Float64Array(8);
            for (const mode of ['lengths', 'cladogram']) {
                const L = TreeView.layoutTree(m, 'unrooted', mode);
                for (let i = 0; i < m.nodeCount; i++) {
                    if (m.childCount[i] === 0) continue;
                    L.mark(i, o);
                    const middle = (L.low[i] + L.high[i]) / 2;
                    if (Math.hypot(o[2] - (L.x[i] + L.far[i] * Math.cos(middle)), o[3] - (L.y[i] + L.far[i] * Math.sin(middle))) > 1e-9) wrong.push([mode, i, 'direction']);
                    // Half the width of the slice at its far end, at most: the mean width of the slice.
                    if (L.thickness(i, 1, 1) > L.far[i] * (L.high[i] - L.low[i]) * 0.5 + 1e-12) wrong.push([mode, i, 'thickness']);
                }
            }
            return wrong.slice(0, 10);""")
        self.assertEqual(bad, [])

    def test_a_single_leaf_and_a_pair_have_every_layout(self):
        for text, headers, leaves in (("only;", ["only"], 1), ("(a:1,b:2);", ["a", "b"], 2)):
            with self.subTest(leaves=leaves):
                folder = self.make_folder()
                write_tree_file(folder, headers, f"#NEXUS\nBEGIN TREES;\n  TREE t = [&U] {text}\nEND;\n")
                self.serve_headers(headers, folder=folder, tree=None)
                self.open_page()
                for kind in ("rectangular", "circular", "unrooted"):
                    for mode in ("lengths", "cladogram"):
                        self.choose(kind, mode)
                        info = self.script("""const L = TreeView.state.layout, v = TreeView.state.view;
                            return {finite: [v.sx, v.sy, v.ox, v.oy, L.bounds.minX, L.bounds.maxX, L.bounds.minY, L.bounds.maxY].every(Number.isFinite),
                                    stats: TreeView.lastStats};""")
                        self.assertTrue(info["finite"], (kind, mode))
                        self.assertGreaterEqual(info["stats"]["visited"], 1)
                    self.script("TreeView.focusLeaf(0); TreeView.drawNow(); return 1")
                self.assertEqual(self.console_errors(), [])


class UniformZoomTests(LayoutCase):
    """The circular and unrooted layouts have one scale: no zoom of the rows or the width alone."""

    def test_every_wheel_gesture_zooms_both_ways_alike_about_the_pointer(self):
        for kind in ("circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                for options, factor in (({}, np.exp(0.18)), ({"shiftKey": True}, np.exp(0.18)), ({"ctrlKey": True}, np.exp(0.18))):
                    before = self.view_state()
                    self.script(f"return fire.wheel(-120, 700, 300, {json.dumps(options)})")
                    after = self.view_state()
                    self.assertAlmostEqual(after["sx"] / before["sx"], factor, places=6, msg=options)
                    self.assertEqual(after["sx"], after["sy"])
                    # The point under the pointer stays under it.
                    self.assertAlmostEqual((700 - after["ox"]) / after["sx"], (700 - before["ox"]) / before["sx"], places=6)
                    self.assertAlmostEqual((300 - after["oy"]) / after["sy"], (300 - before["oy"]) / before["sy"], places=6)

    def test_keys_zoom_and_fit_in_one_scale_and_the_zoom_is_held_between_limits(self):
        self.choose("circular")
        limits = self.state("TreeView.limits()")
        self.assertEqual((limits["sxMin"], limits["sxMax"]), (limits["syMin"], limits["syMax"]))
        for key in ("]", "+", "="):
            before = self.view_state()
            self.script(f"fire.key({json.dumps(key)}); return 1")
            after = self.view_state()
            self.assertAlmostEqual(after["sx"] / before["sx"], 1.25)
            self.assertEqual(after["sx"], after["sy"])
        for _ in range(80):
            self.js("fire.wheel(-400, 600, 300); 1")
        self.assertEqual(self.view_state()["sx"], limits["sxMax"])
        for _ in range(160):
            self.js("fire.wheel(400, 600, 300); 1")
        self.assertEqual(self.view_state()["sx"], limits["sxMin"])
        self.script("fire.key('Home'); return 1")
        self.assertTrue(self.view_state()["fitted"])

    def test_the_overview_and_the_ruler_belong_to_the_layouts_that_have_them(self):
        # A rectangular tree has the ruler, with its rows; the others have none, and the pointer is not taken by a strip.
        self.choose("rectangular")
        self.assertTrue(self.state("TreeView.drawNow() && true"))
        ruler = self.state("fire.pixel(document.getElementById('tree-canvas').clientWidth - 6, 300).slice(0, 3)")
        self.choose("circular")
        none = self.state("fire.pixel(document.getElementById('tree-canvas').clientWidth - 6, 300).slice(0, 3)")
        self.assertNotEqual(ruler, none)
        before = self.view_state()
        self.script("""const w = document.getElementById('tree-canvas').clientWidth;
            fire.pointer('pointerdown', w - 5, 100); fire.pointer('pointerup', w - 5, 100); return 1""")
        self.assertEqual(self.view_state()["oy"], before["oy"])

    def test_a_leaf_is_brought_into_view_among_the_leaves_near_it(self):
        for kind in ("circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind, "cladogram")
                info = self.script("""TreeView.focusLeaf(20); TreeView.drawNow();
                    const s = TreeView.state, box = TreeView.nodePixels(s, s.model.leafNode[20]);
                    return {x: box.x, y: box.y, w: document.getElementById('tree-canvas').clientWidth, h: document.getElementById('tree-canvas').clientHeight,
                            fitted: s.fitted, labels: TreeView.lastStats.labels};""")
                self.assertTrue(0 < info["x"] < info["w"] and 0 < info["y"] < info["h"], info)
                self.assertFalse(info["fitted"])
                self.assertGreater(info["labels"], 0)


    def test_names_need_room_in_an_unrooted_tree_and_dots_need_less(self):
        self.choose("unrooted", "lengths")

        def with_leaves_apart_by(pixels):
            # The view in which the leaves at the farthest rim are this far from each other, round the tree's root.
            return self.script(f"""const s = TreeView.state, L = s.layout, root = s.model.root;
                const scale = {pixels} / (L.far[L.far.length - 1] * L.slot);
                s.view = {{sx: scale, sy: scale, ox: 650 - L.x[root] * scale, oy: 330 - L.y[root] * scale}};
                s.fitted = false;
                TreeView.drawNow(); return {{...TreeView.lastStats}};""")
        crowded, roomy, wide = with_leaves_apart_by(2), with_leaves_apart_by(8), with_leaves_apart_by(24)
        self.assertEqual((crowded["dots"], crowded["labels"]), (0, 0))
        self.assertGreater(roomy["dots"], 0)
        self.assertEqual(roomy["labels"], 0)
        self.assertGreater(wide["dots"], 0)
        self.assertGreater(wide["labels"], 0)

    def test_every_arc_that_is_in_view_is_drawn(self):
        self.choose("circular", "lengths")
        info = self.script("""
            TreeView.zoomAbout(650, 330, 5, 5);
            const s = TreeView.state, m = s.model, L = s.layout, v = s.view, o = new Float64Array(8);
            const w = TreeView.canvas.clientWidth, h = TreeView.canvas.clientHeight;
            const drawn = new Set(), arc = Path2D.prototype.arc;
            Path2D.prototype.arc = function (cx, cy, r, a0, a1) { drawn.add(r.toFixed(3) + '|' + a0.toFixed(6)); return arc.apply(this, arguments); };
            try { TreeView.drawNow(); } finally { Path2D.prototype.arc = arc; }
            let inView = 0;
            const missing = [];
            for (let i = 0; i < m.nodeCount; i++) {
                L.join(i, o);
                if (o[0] !== 2) continue;
                let seen = false;
                for (let k = 0; k <= 24 && !seen; k++) {
                    const a = o[2] + (o[3] - o[2]) * k / 24;
                    const px = v.ox + o[1] * Math.cos(a) * v.sx, py = v.oy + o[1] * Math.sin(a) * v.sy;
                    seen = px >= 0 && px <= w && py >= 0 && py <= h;
                }
                if (!seen) continue;
                inView++;
                if (!drawn.has((o[1] * v.sx).toFixed(3) + '|' + o[2].toFixed(6))) missing.push(i);
            }
            return {inView, missing: missing.slice(0, 5), arcs: drawn.size};""")
        self.assertGreater(info["inView"], 5)
        self.assertEqual(info["missing"], [])

    def test_zooming_to_a_node_that_is_not_there_changes_nothing(self):
        before = self.view_state()
        self.script("for (const node of [-1, 1e6, NaN, undefined]) TreeView.zoomToNode(node); return 1")
        self.assertEqual(self.view_state(), before)


class HitTestEverywhereTests(LayoutCase):
    """What is under the pointer in the circular and unrooted layouts, from the clades near it, against looking at every node.

    (The rectangular layout's is checked above, line by line, by HitTestTests; a row of it is as tall as its row.)
    """

    COUNT = 80

    SCRIPT = """
        const s = TreeView.state, m = s.model, tol = 5, o = new Float64Array(8);
        const segment = (px, py, x0, y0, x1, y1) => {
            const dx = x1 - x0, dy = y1 - y0, q = dx * dx + dy * dy;
            const t = q > 0 ? Math.min(1, Math.max(0, ((px - x0) * dx + (py - y0) * dy) / q)) : 0;
            return Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
        };
        const arc = (px, py, cx, cy, r, a0, a1) => {
            const dx = px - cx, dy = py - cy, tau = 2 * Math.PI;
            const turn = (((Math.atan2(dy, dx) - a0) % tau) + tau) % tau;
            if (turn <= a1 - a0) return Math.abs(Math.hypot(dx, dy) - r);
            return Math.min(Math.hypot(px - (cx + r * Math.cos(a0)), py - (cy + r * Math.sin(a0))), Math.hypot(px - (cx + r * Math.cos(a1)), py - (cy + r * Math.sin(a1))));
        };
        // The distance from a pixel to each line of node i, as {part: distance}.
        const distances = (i, px, py) => {
            const v = s.view, L = s.layout, found = {};
            L.stem(i, o);
            const x0 = v.ox + o[0] * v.sx, y0 = v.oy + o[1] * v.sy, x1 = v.ox + o[2] * v.sx, y1 = v.oy + o[3] * v.sy;
            found[m.childCount[i] === 0 ? 'leaf' : 'branch'] = segment(px, py, x0, y0, x1, y1);
            L.join(i, o);
            if (o[0] === 1) found.connector = segment(px, py, v.ox + o[1] * v.sx, v.oy + o[2] * v.sy, v.ox + o[3] * v.sx, v.oy + o[4] * v.sy);
            if (o[0] === 2) found.connector = arc(px, py, v.ox, v.oy, o[1] * v.sx, o[2], o[3]);
            return found;
        };
        // The nearest line of any node, and the pixel's distance to it.
        const nearest = (px, py) => {
            let best = {distance: Infinity};
            for (let i = 0; i < m.nodeCount; i++) {
                for (const [part, distance] of Object.entries(distances(i, px, py))) if (distance < best.distance) best = {node: i, part, distance};
            }
            return best;
        };
        // No names and no dots, which are the page's own business: the lines are what is compared.
        TreeView.leafDetail = () => ({dots: false, labels: false, size: 14});
    """

    def test_the_nearest_line_is_found_in_every_layout_as_by_looking_at_every_node(self):
        result = self.script(f"""
            {self.SCRIPT}
            const out = {{}};
            for (const kind of ['circular', 'unrooted']) {{
                TreeView.setLayout(kind, 'cladogram');
                TreeView.drawNow();
                let seeded = 4321;
                const random = () => (seeded = (seeded * 1103515245 + 12345) % 2147483648) / 2147483648;
                const missed = [], wrong = [];
                let hits = 0, empty = 0;
                for (let n = 0; n < 700; n++) {{
                    const px = random() * 1286, py = random() * 660 + 4;
                    const best = nearest(px, py), hit = TreeView.hitTest(px, py);
                    if (best.distance > tol) {{
                        empty++;
                        if (hit) wrong.push(['hit where nothing is', px, py, hit]);
                        continue;
                    }}
                    if (!hit) {{ missed.push([px, py, best]); continue; }}
                    hits++;
                    const found = distances(hit.node, px, py)[hit.part];
                    if (found === undefined || Math.abs(found - best.distance) > 1e-6) wrong.push([px, py, hit, best]);
                }}
                out[kind] = {{hits, empty, missed: missed.slice(0, 3), wrong: wrong.slice(0, 3)}};
            }}
            return out;""")
        for kind, found in result.items():
            with self.subTest(kind=kind):
                self.assertGreater(found["hits"], 15)
                self.assertGreater(found["empty"], 100)
                self.assertEqual((found["missed"], found["wrong"]), ([], []))

    def test_a_point_on_a_branch_or_a_join_finds_its_node_in_every_layout(self):
        result = self.script(f"""
            {self.SCRIPT}
            const out = {{}};
            for (const kind of ['circular', 'unrooted']) {{
                TreeView.setLayout(kind, 'cladogram');
                TreeView.drawNow();
                const v = s.view, L = s.layout, unseen = [];
                let tried = 0;
                for (let i = 0; i < m.nodeCount; i++) {{
                    L.stem(i, o);
                    const length = Math.hypot((o[2] - o[0]) * v.sx, (o[3] - o[1]) * v.sy);
                    if (i === m.root || length < 16) continue;
                    const px = v.ox + (o[0] + o[2]) / 2 * v.sx, py = v.oy + (o[1] + o[3]) / 2 * v.sy;
                    if (px < 4 || px > 1280 || py < 4 || py > 660) continue;
                    tried++;
                    const hit = TreeView.hitTest(px, py);
                    // The point is on node i's line, so what is found is a line that passes through it.
                    const away = hit ? distances(hit.node, px, py)[hit.part] : undefined;
                    if (away === undefined || away > 1e-6) unseen.push([i, hit]);
                }}
                // The arc that joins a clade's children, at its middle.
                let joined = 0;
                for (let i = 0; i < m.nodeCount && kind === 'circular'; i++) {{
                    L.join(i, o);
                    if (o[0] !== 2 || (o[3] - o[2]) * o[1] * v.sx < 30) continue;
                    const a = (o[2] + o[3]) / 2, px = v.ox + o[1] * Math.cos(a) * v.sx, py = v.oy + o[1] * Math.sin(a) * v.sy;
                    if (px < 4 || px > 1280 || py < 4 || py > 660) continue;
                    joined++;
                    const hit = TreeView.hitTest(px, py);
                    const away = hit ? distances(hit.node, px, py)[hit.part] : undefined;
                    if (away === undefined || away > 1e-6) unseen.push([i, 'arc', hit]);
                }}
                out[kind] = {{tried, joined, unseen: unseen.slice(0, 4)}};
            }}
            return out;""")
        self.assertGreater(result["circular"]["tried"], 15)
        self.assertGreater(result["circular"]["joined"], 3)
        self.assertGreater(result["unrooted"]["tried"], 15)
        for kind, found in result.items():
            with self.subTest(kind=kind):
                self.assertEqual(found["unseen"], [])

    def test_a_leaf_is_found_by_its_name_in_every_layout(self):
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind, "cladogram")
                result = self.script("""
                    TreeView.focusLeaf(25);
                    TreeView.drawNow();
                    const s = TreeView.state, m = s.model, box = TreeView.nodePixels(s, m.leafNode[25]);
                    const stats = TreeView.lastStats;
                    // Where the name is: along the leaf's own direction in a circular tree (its name runs outwards), to its
                    // side in an unrooted one, to its right in a rectangular one.
                    const toward = s.layout.kind === 'circular' ? [Math.cos(s.layout.angle[m.leafNode[25]]), Math.sin(s.layout.angle[m.leafNode[25]])]
                        : s.layout.kind === 'unrooted' ? [Math.cos(Math.atan2(box.y - box.y0, box.x - box.x0)) < 0 ? -1 : 1, 0] : [1, 0];
                    const onName = TreeView.hitTest(box.x + 40 * toward[0], box.y + 40 * toward[1]);
                    const far = TreeView.hitTest(box.x + 1500 * toward[0], box.y + 1500 * toward[1]);
                    return {labels: stats.labels, onName, far, leaf: m.leafNode[25], isLeaf: onName ? m.childCount[onName.node] === 0 : null};""")
                self.assertGreater(result["labels"], 0)
                self.assertEqual(result["onName"]["part"], "leaf")
                self.assertTrue(result["isLeaf"])
                if kind != "unrooted":
                    # Names of unrooted leaves may lie over each other; the others are one to a row or a slot.
                    self.assertEqual(result["onName"]["node"], result["leaf"])
                self.assertIsNone(result["far"])


class FoldingTests(LayoutCase):
    """A clade folds into one shape (its point, and the arc of what it held), and opens again."""

    COUNT = 100

    def clade(self, low=12, high=30):
        return self.state(clade_of(low, high))

    def test_a_folded_clade_is_one_shape_in_every_layout_and_holds_nothing_else(self):
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                clade = self.clade()
                before = self.drawn()
                self.assertEqual(self.script(f"return TreeView.toggleFold({clade})"), True)
                after = self.drawn()
                self.assertEqual((before["collapsed"], after["collapsed"]), (0, 1))
                self.assertLess(after["visited"], before["visited"])
                info = self.script(f"""const s = TreeView.state, m = s.model, o = new Float64Array(8);
                    s.layout.wedge({clade}, o);
                    const wedge = Array.from(o);
                    return {{count: s.collapsedCount, hidden: s.leafHidden.reduce((a, b) => a + b, 0), held: TreeView.leavesUnder(m, {clade}),
                            wedge, shape: s.collapsed[{clade}]}};""")
                self.assertEqual(info["count"], 1)
                self.assertEqual(info["hidden"], info["held"])
                # A wedge says where its point is and what it spans.
                self.assertEqual(info["wedge"][0], 0 if kind == "rectangular" else 1)
                self.assertTrue(np.all(np.isfinite(info["wedge"])))
                self.script(f"TreeView.toggleFold({clade}); return 1")
                self.assertEqual(self.drawn()["collapsed"], 0)
                self.assertEqual(self.state("[TreeView.state.collapsedCount, TreeView.state.leafHidden]"), [0, None])

    def test_the_shape_of_a_folded_clade_is_what_it_reaches_in_each_layout(self):
        for kind, expected in (("rectangular", 0), ("circular", 1), ("unrooted", 1)):
            with self.subTest(kind=kind):
                self.choose(kind)
                clade = self.clade()
                wedge = self.state(f"""(() => {{ const o = new Float64Array(8); TreeView.state.layout.wedge({clade}, o); return Array.from(o); }})()""")
                reach = self.state(f"""(() => {{ const s = TreeView.state, b = s.layout.boxes; return {{x1: b.x1[{clade}], y0: b.y0[{clade}], y1: b.y1[{clade}], first: s.model.firstLeaf[{clade}], last: s.model.lastLeaf[{clade}]}}; }})()""")
                self.assertEqual(wedge[0], expected)
                if kind == "rectangular":
                    # A triangle: the node, and the two ends of the rows at the clade's reach.
                    self.assertEqual((wedge[3], wedge[4], wedge[5], wedge[6]), (reach["x1"], reach["first"], reach["x1"], reach["last"] + 1))

    def test_nothing_can_be_folded_but_a_clade_that_is_not_the_root(self):
        clade = self.clade()
        info = self.script(f"""const m = TreeView.state.model;
            return [TreeView.toggleFold(m.root), TreeView.toggleFold(m.leafNode[3]), TreeView.toggleFold(-1), TreeView.toggleFold({clade}),
                    TreeView.state.collapsedCount];""")
        self.assertEqual(info, [False, False, False, True, 1])

    def test_the_square_shows_for_a_clade_that_is_hovered_selected_or_folded_and_nowhere_else(self):
        clade = self.clade(15, 40)
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                self.script(f"TreeView.zoomToNode({clade}); TreeView.drawNow(); return 1")
                self.assertEqual(self.drawn()["handles"], 0)
                self.script(f"TreeView.state.hover = {clade}; return 1")
                self.assertEqual(self.drawn()["handles"], 1)
                self.script(f"TreeView.state.hover = -1; TreeView.state.pinned = {clade}; return 1")
                self.assertEqual(self.drawn()["handles"], 1)
                self.script(f"TreeView.state.pinned = -1; TreeView.toggleFold({clade}); return 1")
                self.assertEqual(self.drawn()["handles"], 1)
                self.script(f"TreeView.toggleFold({clade}); return 1")

    def test_pointing_at_a_clades_node_shows_its_square_and_clicking_it_folds_and_opens(self):
        clade = self.clade(15, 40)
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                self.script(f"TreeView.zoomToNode({clade}); TreeView.drawNow(); return 1")
                point = self.state(f"(() => {{ const b = TreeView.nodePixels(TreeView.state, {clade}); return [b.x, b.y]; }})()")
                self.script(f"fire.move({point[0]}, {point[1]}); return 1")
                self.wait_for(f"TreeView.state.hover === {clade}")
                self.script("TreeView.drawNow(); return 1")
                self.assertEqual(self.state(f"TreeView.hitTest({point[0]}, {point[1]})"), {"node": clade, "part": "handle"})
                # Pointing at the square says what it does.
                self.script(f"fire.move({point[0]}, {point[1]}); return 1")
                self.wait_for("document.getElementById('tooltip').textContent.includes('Click to fold this clade up')")
                self.script(f"fire.click({point[0]}, {point[1]}); return 1")
                self.assertEqual(self.state("TreeView.state.collapsed[" + str(clade) + "]"), 1)
                self.assertEqual(self.state("document.getElementById('unfold-btn').disabled"), False)
                self.script("TreeView.drawNow(); return 1")
                # The folded clade, clicked on its body (the middle of the line from its node to where it reaches), opens.
                body = self.state(f"""(() => {{ const s = TreeView.state, v = s.view, o = new Float64Array(8); s.layout.mark({clade}, o);
                    return [v.ox + (o[0] + o[2]) / 2 * v.sx, v.oy + (o[1] + o[3]) / 2 * v.sy]; }})()""")
                self.assertEqual(self.state(f"TreeView.hitTest({body[0]}, {body[1]})"), {"node": clade, "part": "clade"})
                self.script(f"fire.click({body[0]}, {body[1]}); return 1")
                self.assertEqual(self.state("TreeView.state.collapsed[" + str(clade) + "]"), 0)
                self.assertEqual(self.state("document.getElementById('unfold-btn').disabled"), True)
                self.assertEqual(self.state("document.getElementById('tooltip').style.display"), "none")

    def test_the_c_key_folds_the_selected_clade_and_open_all_opens_every_fold(self):
        self.choose("circular")
        first, second = self.clade(5, 10), self.clade(20, 40)
        self.script(f"TreeView.state.pinned = {first}; fire.key('c'); TreeView.state.pinned = {second}; fire.key('c'); return 1")
        self.assertEqual(self.state("TreeView.state.collapsedCount"), 2)
        self.assertEqual(self.state("document.getElementById('unfold-btn').disabled"), False)
        self.js("document.getElementById('unfold-btn').click()")
        self.assertEqual(self.state("[TreeView.state.collapsedCount, document.getElementById('unfold-btn').disabled, TreeView.state.leafHidden]"),
                         [0, True, None])
        # With nothing selected the key does nothing, and says so by not taking the key.
        self.script("TreeView.state.pinned = -1; return 1")
        self.assertFalse(self.script("return fire.key('c')"))

    def test_folds_and_the_selection_go_with_a_change_of_layout(self):
        clade = self.clade(12, 30)
        pinned = self.state("TreeView.state.model.leafNode[2]")
        self.script(f"TreeView.toggleFold({clade}); TreeView.state.pinned = {pinned}; return 1")
        for kind in ("circular", "rectangular", "unrooted"):
            self.choose(kind)
            self.assertEqual(self.state(f"[TreeView.state.collapsed[{clade}], TreeView.state.pinned]"), [1, pinned])
            self.assertEqual(self.drawn()["collapsed"], 1)

    def test_a_selection_inside_a_clade_that_is_folded_moves_to_the_clade(self):
        clade = self.clade(12, 30)
        inside = self.state(f"TreeView.state.model.leafNode[TreeView.state.model.firstLeaf[{clade}]]")
        self.script(f"TreeView.state.pinned = {inside}; TreeView.state.hover = {inside}; TreeView.toggleFold({clade}); return 1")
        self.assertEqual(self.state("[TreeView.state.pinned, TreeView.state.hover]"), [clade, clade])

    def test_a_search_opens_the_clade_that_holds_what_it_found(self):
        clade = self.clade(12, 30)
        leaf = self.state(f"TreeView.state.model.firstLeaf[{clade}]")
        header = self.state(f"TreeView.state.model.labels[{leaf}]")
        self.script(f"TreeView.toggleFold({clade}); return 1")
        self.assertEqual(self.state(f"TreeView.state.leafHidden[{leaf}]"), 1)
        self.assertEqual(self.state(f"TreeView.runSearch({json.dumps(chr(34) + header + chr(34))})"), 1)
        self.script("TreeView.stepMatch(1); return 1")
        info = self.state(f"[TreeView.state.collapsed[{clade}], TreeView.state.collapsedCount, TreeView.state.leafHidden]")
        self.assertEqual(info, [0, 0, None])
        self.assertEqual(self.state("document.getElementById('unfold-btn').disabled"), True)

    def test_a_match_inside_a_folded_clade_outlines_the_clade_and_has_no_ring_of_its_own(self):
        clade = self.clade(12, 30)
        leaf = self.state(f"TreeView.state.model.firstLeaf[{clade}]")
        header = self.state(f"TreeView.state.model.labels[{leaf}]")
        self.choose("rectangular")
        self.script(f"""TreeView.zoomToNode({clade}); TreeView.toggleFold({clade});
            TreeView.runSearch({json.dumps(chr(34) + header + chr(34))}); TreeView.drawNow(); return 1""")
        info = self.script(f"""const s = TreeView.state, o = new Float64Array(8), v = s.view;
            s.layout.wedge({clade}, o);
            // A point on the wedge's edge, half way along it from the node to the first end.
            const px = v.ox + (o[1] + o[3]) / 2 * v.sx, py = v.oy + (o[2] + o[4]) / 2 * v.sy;
            const ctx = document.createElement('canvas').getContext('2d');
            ctx.fillStyle = s.palette.match; ctx.fillRect(0, 0, 1, 1);
            const near = (x, y) => {{
                let best = Infinity;
                for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {{
                    const pixel = fire.pixel(Math.round(x) + dx, Math.round(y) + dy);
                    best = Math.min(best, Math.max(...[0, 1, 2].map((c) => Math.abs(pixel[c] - match[c]))));
                }}
                return best;
            }};
            const match = Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3));
            return {{match, onTheLine: near(px, py), away: near(px + 400, py)}};""")
        # Within a pixel of the line the outline is drawn in the match colour; away from it nothing is.
        self.assertLessEqual(info["onTheLine"], 24)
        self.assertGreater(info["away"], 60)


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
        self.assertEqual(matching_pixels(), 0)
        self.script("TreeView.runSearch('PROT_1'); return 1")
        self.assertGreater(matching_pixels(), 5)

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
        self.serve_shape("balanced")
        info = self.script("""
            const s = TreeView.state, v = s.view, L = s.layout;
            TreeView.zoomToNode(TreeView.state.model.root);
            TreeView.zoomAbout(600, 300, 1, 60);
            TreeView.drawNow();
            // A leaf's stem is one row of full colour: the row beside it is the background.
            const ctx = document.createElement('canvas').getContext('2d');
            const colour = (css) => { ctx.fillStyle = css; ctx.fillRect(0, 0, 1, 1); return Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)); };
            const branch = colour(s.palette.branch), background = colour(s.palette.background);
            let crisp = 0, blurred = 0;
            const m = s.model;
            for (let k = 0; k < m.leafCount; k++) {
                const box = TreeView.nodePixels(s, m.leafNode[k]);
                if (box.y < 20 || box.y > 640 || box.x0 < 40 || box.x - box.x0 < 40) continue;
                const x = Math.round(box.x0 + 20), row = Math.floor(box.y);
                const on = fire.pixel(x, row).slice(0, 3), above = fire.pixel(x, row - 1).slice(0, 3), below = fire.pixel(x, row + 1).slice(0, 3);
                const same = (a, b) => a.every((value, i) => value === b[i]);
                if (same(on, branch) && same(above, background) && same(below, background)) crisp++; else blurred++;
            }
            return {crisp, blurred};""")
        self.assertGreater(info["crisp"], 5)
        self.assertLessEqual(info["blurred"], info["crisp"] * 0.2)


class PerformanceTests(TreePageMixin, unittest.TestCase):
    """The page on trees of 13,000 and 44,000 leaves in four shapes: measured here, with loose budgets.

    The timings include a flush of the canvas, so they are the cost of the pixels and not only of the drawing
    commands. This engine has no GPU, so they are what a machine without one would take; the budgets are far
    above what a sane machine needs, and the measurements are what is recorded (set TREE_PERFORMANCE_FILE to a path
    to have each run's figures appended to it as JSON lines).
    """

    BUDGET_MS = {"layout": 400, "decode": 400, "frame": 1500, "pan": 1500, "hit": 150, "fold": 150}

    MEASURE = """
        const flush = () => { document.getElementById('tree-canvas').getContext('2d').getImageData(0, 0, 1, 1); };
        const time = (f) => { const a = performance.now(); f(); flush(); return performance.now() - a; };
        const median = (values) => values.slice().sort((x, y) => x - y)[values.length >> 1];
        const payload = await (await fetch('/api/tree/data')).json();
        const out = {nodes: payload.counts.nodes, decode: time(() => TreeView.decodeTree(payload))};
        const s = TreeView.state, m = s.model;
        const find = (low, high) => { for (let i = 0; i < m.nodeCount; i++) { const n = m.lastLeaf[i] - m.firstLeaf[i] + 1; if (m.childCount[i] > 0 && n >= low && n <= high) return i; } return -1; };
        for (const kind of ['rectangular', 'circular', 'unrooted']) {
            const r = {};
            r.layout = time(() => TreeView.layoutTree(m, kind, 'lengths'));
            TreeView.setLayout(kind, 'lengths');
            r.first_frame = time(() => TreeView.drawNow());
            r.fit_frame = median(Array.from({length: 5}, () => time(() => TreeView.drawNow())));
            r.visited = TreeView.lastStats.visited;
            r.hit = median(Array.from({length: 15}, (_, k) => time(() => TreeView.hitTest(300 + k * 37, 200 + k * 21))));
            TreeView.zoomAbout(650, 330, 1, 20);
            TreeView.zoomAbout(650, 330, 20, 1);
            r.pan_frame = median(Array.from({length: 15}, () => { TreeView.pan(11, 5); return time(() => TreeView.drawNow()); }));
            const clade = find(30, 60);
            if (clade >= 0) { TreeView.zoomToNode(clade); r.zoomed_frame = median(Array.from({length: 5}, () => time(() => TreeView.drawNow()))); }
            TreeView.fitView();
            const big = find(Math.max(100, m.leafCount / 50), m.leafCount / 5);
            if (big >= 0) { r.fold = time(() => TreeView.toggleFold(big)); r.fold_frame = time(() => TreeView.drawNow()); TreeView.toggleFold(big); }
            r.overview = time(() => { TreeView.minimapChanged(); TreeView.drawMinimap(); });
            out[kind] = r;
        }
        return out;
    """

    def measure(self, shape, count):
        headers = headers_of(count)
        self.serve_headers(headers, linkage=shaped_linkage(shape, count))
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
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(shape=result["shape"], leaves=result["leaves"], kind=kind):
                found = result[kind]
                self.assertLess(found["layout"], self.BUDGET_MS["layout"])
                self.assertLess(found["fit_frame"], self.BUDGET_MS["frame"])
                self.assertLess(found["first_frame"], self.BUDGET_MS["frame"] * 3)
                self.assertLess(found["pan_frame"], self.BUDGET_MS["pan"])
                self.assertLess(found["hit"], self.BUDGET_MS["hit"])
                self.assertLess(found.get("fold", 0), self.BUDGET_MS["fold"])
        # (Reading a pixel to flush the canvas, as the timings do, makes the engine advise willReadFrequently: its own
        # advice about the measuring, not about the page.)
        self.assertEqual([entry for entry in self.console_errors() if "willReadFrequently" not in entry[1]], [])

    def test_thirteen_thousand_leaves_in_every_shape(self):
        for shape in ("bushy", "caterpillar", "chains"):
            with self.subTest(shape=shape):
                self.check(self.measure(shape, 13000))

    def test_forty_four_thousand_leaves_in_every_shape(self):
        for shape in ("bushy", "caterpillar", "chains"):
            with self.subTest(shape=shape):
                self.check(self.measure(shape, 44000))


class FitReportTests(TreePageMixin, unittest.TestCase):
    """How the tree fits is shown: warnings in a banner, the counts in the status line, the rest in Details."""

    def banner(self):
        return self.script("""
            return {shown: document.getElementById('banner').classList.contains('shown'),
                    lines: [...document.querySelectorAll('#warning-list li')].map(li => li.textContent),
                    summary: document.getElementById('summary').textContent};""")

    def test_a_tree_that_fits_has_no_banner(self):
        self.serve_headers(headers_of(30))
        self.open_page()
        banner = self.banner()
        self.assertFalse(banner["shown"])
        self.assertEqual(banner["lines"], [])
        self.assertEqual(banner["summary"], "30 leaves, 30 of them nodes of this network")

    def test_leaves_the_network_lacks_are_counted_in_the_status_line_and_warned_about(self):
        headers = headers_of(30)
        self.serve_headers(headers, network=headers[:20] + headers_of(5, prefix="tr|Y"))
        self.open_page()
        banner = self.banner()
        self.assertTrue(banner["shown"])
        self.assertEqual(banner["summary"], "30 leaves, 20 of them nodes of this network")
        self.assertEqual(banner["lines"], [
            "Leaves that are not nodes of this network: 10. They are drawn faded.",
            "Nodes of this network that are not leaves of the tree: 5.",
        ])

    def test_a_tree_made_for_another_alignment_is_drawn_with_a_warning_naming_both(self):
        self.serve_headers(headers_of(30), notes=[("ALIGNMENT", "Other_[model]_alignment.fasta")])
        self.open_page()
        self.assertEqual(self.banner()["lines"], [
            f"This tree was made for the alignment Other_[model]_alignment.fasta, but the loaded alignment is {ALIGNMENT_NAME}."])
        self.assertEqual(self.state("document.body.dataset.view"), "tree")

    def test_an_alignment_older_than_its_tree_is_warned_about(self):
        now = time.time()
        self.serve_headers(headers_of(30), notes=[("ALIGNMENT", ALIGNMENT_NAME),
                                                  ("WRITTEN", time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(now)))])
        os.utime(self.msa, (now - 3600, now - 3600))
        self.open_page()
        self.assertEqual(self.banner()["lines"], [
            "The loaded alignment is older than this tree, so the alignment step of the run that made the tree may not have finished."])

    def test_leaves_the_alignment_lacks_are_warned_about(self):
        self.serve_headers(headers_of(30))
        self.viewer.alignment.aligned_node_mask[:5] = False
        self.open_page()
        self.assertEqual(self.banner()["lines"], ["Leaves whose sequence is not in the loaded alignment: 5."])

    def test_details_lists_the_file_the_run_and_the_counts(self):
        self.serve_headers(headers_of(30), notes=[
            ("WRITTEN", "2026-11-05 14:03:22 -0500"), ("PROGRAM", "EMAP-SSN 0.5.0"), ("ALIGNMENT", ALIGNMENT_NAME),
            ("SEQUENCE_SET", "Set"), ("MODEL", "model"), ("TREE_METHOD", "UPGMA")])
        self.open_page()
        self.assertFalse(self.state("document.getElementById('details').classList.contains('shown')"))
        self.js("document.getElementById('details-btn').click()")
        details = self.script("""
            const panel = document.getElementById('details');
            const rows = {};
            for (const dt of panel.querySelectorAll('dt')) rows[dt.textContent] = dt.nextElementSibling.textContent;
            return {shown: panel.classList.contains('shown'), rows, notes: [...panel.querySelectorAll('p.note')].map(p => p.textContent)};""")
        self.assertTrue(details["shown"])
        rows = details["rows"]
        self.assertEqual(rows["File"], os.path.abspath(self.tree_file))
        self.assertEqual((rows["Alignment"], rows["Saved"], rows["Program"]), (ALIGNMENT_NAME, "2026-11-05 14:03:22 -0500", "EMAP-SSN 0.5.0"))
        self.assertEqual((rows["Sequence set"], rows["Model"], rows["Tree method"], rows["Kind"]), ("Set", "model", "UPGMA", "Rooted"))
        self.assertEqual((rows["Leaves"], rows["Leaves in this network"], rows["Leaves not in this network"]), ("30", "30", "0"))
        self.assertEqual(rows["Network nodes not in the tree"], "0")
        self.assertTrue(any("UPGMA root" in note for note in details["notes"]))
        self.js("document.getElementById('details-btn').click()")
        self.assertFalse(self.state("document.getElementById('details').classList.contains('shown')"))


class TextAndSafetyTests(TreePageMixin, unittest.TestCase):
    """A header is the file's text, shown as text; the page runs no code it did not ship, and asks for no file."""

    MARKUP = ["<img src=x onerror=window.__pwned=1>", "<b>bold</b> & <i>it</i>", "&amp; &lt;script&gt;", "plain header"]

    def setUp(self):
        super().setUp()
        self.serve_headers(self.MARKUP)
        self.open_page()

    def test_headers_with_markup_in_them_are_shown_as_text(self):
        result = self.script("""
            const s = TreeView.state, m = s.model;
            const texts = [];
            const summary = document.getElementById('summary'), tip = document.getElementById('tooltip');
            for (let k = 0; k < m.leafCount; k++) {
                TreeView.focusLeaf(k);
                TreeView.drawNow();
                const box = TreeView.nodePixels(s, m.leafNode[k]);
                TreeView.showTooltip(tip, 50, 50, m.leafNode[k]);
                const entry = {label: m.labels[k], shown: tip.querySelector('.name').textContent,
                               elements: tip.querySelectorAll('img, b, i, script').length};
                fire.click(box.x + 40, box.y);
                entry.status = summary.textContent;
                entry.statusElements = summary.children.length;
                fire.click(box.x + 40, box.y);
                texts.push(entry);
            }
            return {texts, pwned: typeof window.__pwned};""")
        for entry in result["texts"]:
            self.assertEqual(entry["shown"], entry["label"])
            self.assertEqual(entry["elements"], 0)
            self.assertTrue(entry["status"].startswith(entry["label"]), entry["status"])
            self.assertEqual(entry["statusElements"], 0)
        self.assertEqual(sorted(entry["label"] for entry in result["texts"]), sorted(self.MARKUP))
        self.assertEqual(result["pwned"], "undefined")

    def test_a_search_over_markup_finds_it_as_text(self):
        self.assertEqual(self.state("TreeView.runSearch('<b>')"), 1)
        self.assertEqual(self.state("TreeView.runSearch('&amp;')"), 1)
        self.assertEqual(self.state("TreeView.runSearch('onerror')"), 1)

    def test_the_pages_files_use_no_markup_or_code_from_text(self):
        for name in PAGE_FILES:
            source = (WEB_DIR / name).read_text(encoding="utf-8")
            if name.endswith(".js"):
                for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
                                  "setTimeout(\"", "setTimeout('", "setInterval(\"", "setInterval('", "Function("):
                    with self.subTest(file=name, forbidden=forbidden):
                        self.assertNotIn(forbidden, source)
            with self.subTest(file=name, word="bootstrap"):
                self.assertNotIn("bootstrap", source.lower())

    def test_the_page_asks_the_viewer_for_the_tree_and_never_names_a_file(self):
        requested = self.state("performance.getEntriesByType('resource').map(entry => new URL(entry.name).pathname)")
        self.assertIn("/api/tree/data", requested)
        self.assertFalse([path for path in requested if ".nex" in path.lower() or "Guide_Trees" in path])
        origins = self.state("[...new Set(performance.getEntriesByType('resource').map(entry => new URL(entry.name).origin))]")
        self.assertEqual(origins, [f"http://127.0.0.1:{self.server.server_address[1]}"])

    def test_the_page_is_served_with_the_base_policy_and_nothing_more(self):
        from web_ui.Web_Server import content_security_policy

        status, headers, body = http_get(self.server, "/tree.html", page_headers(self.server))
        self.assertEqual(status, 200)
        policy = headers["Content-Security-Policy"]
        self.assertEqual(policy, content_security_policy(str(WEB_DIR / "tree.html"), body))
        self.assertNotIn("unsafe-eval", policy)
        self.assertIn("connect-src 'self';", policy)
        self.assertNotIn("https:", policy)
        # One inline script, the theme's, runs by its hash.
        self.assertEqual(len(re.findall(r"'sha256-[A-Za-z0-9+/=]+'", policy)), 1)
        self.assertEqual(self.console_errors(), [])

    def test_the_theme_changes_what_the_canvas_draws_in(self):
        light = self.script("""
            TreeView.drawNow();
            return {theme: document.documentElement.getAttribute('data-theme'), background: TreeView.state.palette.background,
                    label: document.getElementById('theme-toggle-label').textContent, pixel: fire.pixel(5, 5)};""")
        # The test browser keeps the choice as the page does; leave it as it was found.
        self.addCleanup(lambda: self.js("(() => { try { localStorage.removeItem('ssn_theme'); } catch (error) {} return 1; })()"))
        self.js("document.getElementById('theme-toggle-btn').click()")
        self.wait_for("TreeView.state.palette.background !== " + json.dumps(light["background"]))
        dark = self.script("""
            TreeView.drawNow();
            return {theme: document.documentElement.getAttribute('data-theme'), background: TreeView.state.palette.background,
                    label: document.getElementById('theme-toggle-label').textContent, pixel: fire.pixel(5, 5),
                    stored: localStorage.getItem('ssn_theme')};""")
        self.assertNotEqual(light["theme"], dark["theme"])
        self.assertNotEqual(light["pixel"], dark["pixel"])
        self.assertNotEqual(light["label"], dark["label"])
        self.assertEqual(dark["stored"], dark["theme"])
        self.js("(() => { try { localStorage.removeItem('ssn_theme'); } catch (error) {} return 1; })()")

    def test_the_page_is_known_to_the_viewer_while_it_is_open(self):
        # Connecting the event stream is how the Viewer knows not to open a second page.
        self.wait_for("document.getElementById('connection-status').dataset.state === 'connected'")
        self.assertEqual(self.state("document.getElementById('connection-status').textContent"), "Connected")
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)
        self.assertTrue(self.server.has_event_client("tree"))
        self.view.setUrl(QUrl("about:blank"))
        deadline = time.monotonic() + 10
        while self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)
        self.assertFalse(self.server.has_event_client("tree"))


if __name__ == "__main__":
    unittest.main()
