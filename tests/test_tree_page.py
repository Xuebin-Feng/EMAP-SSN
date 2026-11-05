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
    random_linkage,
    save_guide_tree,
    writer,
)
from utilities.Guide_Tree_File import read_guide_tree  # noqa: E402
from web_ui import tree_backend  # noqa: E402

WEB_DIR = SRC_DIR / "web_ui"
PAGE_FILES = ["tree.html", "tree_model.js", "tree_layout.js", "tree_render.js", "tree_nav.js", "tree_session.js"]


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
