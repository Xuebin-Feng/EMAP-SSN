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

from PySide6.QtCore import QUrl, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from tests.tree_page_harness import TreePageMixin, http_get, page_headers, server_viewer  # noqa: E402
from tests.tree_fixtures import (  # noqa: E402
    ALIGNMENT_NAME,
    TREE_NAME,
    headers_of,
    make_alignment_folder,
    make_viewer,
    neighbor_joining_tree,
    node_colors,
    random_linkage,
    save_guide_tree,
    shaped_linkage,
    writer,
)
from utilities.Guide_Tree_File import read_guide_tree  # noqa: E402
from web_ui import tree_backend  # noqa: E402

WEB_DIR = SRC_DIR / "web_ui"
PAGE_FILES = ["tree.html", "tree_model.js", "tree_color.js", "tree_layout.js", "tree_render.js", "tree_nav.js", "tree_select.js",
              "tree_minimap.js", "tree_session.js"]

# The legend's sentence about the split stability, the one place the page uses the word.
LEGEND_NOT_BOOTSTRAP = "Stability under distance noise, not bootstrap support."


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
        # It points at the tool and nothing else: a tree put there by hand is not one the page reads.
        self.assertEqual(card["more"], "Run the Embedding MSA tool again to make it, and press Check again.")
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
    def test_a_tree_opens_by_its_rooting(self):
        info = self.script("""return {kind: TreeView.state.kind, shown: TreeView.state.layout.kind, mode: TreeView.state.layout.mode,
            select: document.getElementById('layout-select').value, rooted: TreeView.state.model.rooted,
            hint: document.getElementById('hints').textContent};""")
        self.assertEqual((info["kind"], info["shown"], info["select"], info["rooted"]), ("unrooted",) * 2 + ("unrooted", False))
        self.assertEqual(info["mode"], "lengths")
        self.assertEqual(info["hint"], "Wheel: zoom. Drag: move. Double-click a clade: zoom to it. Click its square to fold it up. Right-click: select in the network.")
        # What decides it is what the file says of its rooting, [&R], [&U] or nothing, and not how the tree was made: a
        # rooted tree opens rectangular, an unrooted one opens unrooted, and one that does not say is taken as unrooted.
        for flag, rooted, expected in (("[&R] ", True, "rectangular"), ("[&U] ", False, "unrooted"), ("", None, "unrooted")):
            with self.subTest(rooted=rooted):
                folder = self.make_folder()
                write_tree_file(folder, list("abc"), f"#NEXUS\nBEGIN TREES;\n  TREE t = {flag}((a:1,b:2):1,c:3);\nEND;\n")
                self.serve_headers(list("abc"), folder=folder, tree=None)
                self.open_page()
                self.assertEqual(self.state("[TreeView.state.model.rooted, TreeView.state.kind, TreeView.state.layout.kind, "
                                            "document.getElementById('layout-select').value]"),
                                 [rooted, expected, expected, expected])

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
                # Where the page finds the pointer: the canvas can lie on a fraction of a pixel, and an event cannot.
                pointer_x, pointer_y = self.script("return fire.local(700, 300)")
                for options, factor in (({}, np.exp(0.18)), ({"shiftKey": True}, np.exp(0.18)), ({"ctrlKey": True}, np.exp(0.18))):
                    before = self.view_state()
                    self.script(f"return fire.wheel(-120, 700, 300, {json.dumps(options)})")
                    after = self.view_state()
                    self.assertAlmostEqual(after["sx"] / before["sx"], factor, places=6, msg=options)
                    self.assertEqual(after["sx"], after["sy"])
                    # The point under the pointer stays under it.
                    self.assertAlmostEqual((pointer_x - after["ox"]) / after["sx"], (pointer_x - before["ox"]) / before["sx"], places=6)
                    self.assertAlmostEqual((pointer_y - after["oy"]) / after["sy"], (pointer_y - before["oy"]) / before["sy"], places=6)

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
            r.legend_mark = time(() => TreeView.state.matches = TreeView.leavesOfClass(s.colors, s.colors.legend[0].cls));
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

    def test_thirteen_thousand_leaves_in_every_shape(self):
        for shape in ("bushy", "caterpillar", "chains"):
            with self.subTest(shape=shape):
                self.check(self.measure(shape, 13000))

    def test_forty_four_thousand_leaves_in_every_shape(self):
        for shape in ("bushy", "caterpillar", "chains"):
            with self.subTest(shape=shape):
                self.check(self.measure(shape, 44000))


EIGHT_LEAVES = ("#NEXUS\nBEGIN TREES;\n  TREE t = [&R] (((a:1,b:1):1,(c:1,d:1):1):1,((e:1,f:1):1,(g:1,h:1):1):1);\nEND;\n")
# The tree above, node by node (post-order): a0 b1 (ab)2 c3 d4 (cd)5 (abcd)6 e7 f8 (ef)9 g10 h11 (gh)12 (efgh)13 root14.
RED, BLUE, GREEN = "#ff0000", "#0000ff", "#00ff00"
BASE_PAINT = {0: RED, 1: RED, 2: RED, 3: RED, 4: BLUE, 5: BLUE, 6: GREEN}  # h (7) is left plain
NO_CLASS = 65535

# Spies on what the page draws: every stroke and fill of a Path2D, with the lines and arcs the path holds.
SPY = """
    const original = {stroke: CanvasRenderingContext2D.prototype.stroke, fill: CanvasRenderingContext2D.prototype.fill,
        moveTo: Path2D.prototype.moveTo, lineTo: Path2D.prototype.lineTo, arc: Path2D.prototype.arc};
    const parts = new WeakMap();
    const note = (path, item) => { let list = parts.get(path); if (!list) parts.set(path, list = []); list.push(item); };
    Path2D.prototype.moveTo = function (x, y) { note(this, ['m', x, y]); return original.moveTo.call(this, x, y); };
    Path2D.prototype.lineTo = function (x, y) { note(this, ['l', x, y]); return original.lineTo.call(this, x, y); };
    Path2D.prototype.arc = function (x, y, r, a, b) { note(this, ['a', x, y, r, a, b]); return original.arc.apply(this, arguments); };
    const drawn = [];
    const main = (context) => context.canvas.id === 'tree-canvas';
    CanvasRenderingContext2D.prototype.stroke = function (path) {
        if (path instanceof Path2D) drawn.push({op: 'stroke', main: main(this), style: TreeView.rgbOf(this.strokeStyle), width: this.lineWidth, parts: parts.get(path) || []});
        return original.stroke.apply(this, arguments);
    };
    CanvasRenderingContext2D.prototype.fill = function (path) {
        if (path instanceof Path2D) drawn.push({op: 'fill', main: main(this), style: TreeView.rgbOf(this.fillStyle), parts: parts.get(path) || []});
        return original.fill.apply(this, arguments);
    };
    const restore = () => {
        CanvasRenderingContext2D.prototype.stroke = original.stroke; CanvasRenderingContext2D.prototype.fill = original.fill;
        Path2D.prototype.moveTo = original.moveTo; Path2D.prototype.lineTo = original.lineTo; Path2D.prototype.arc = original.arc;
    };
    // The colours drawn lines of the segment (x0, y0)-(x1, y1) of the main canvas: those of the strokes that hold it (to within tol pixels).
    const stylesOf = (x0, y0, x1, y1, tol = 0.75) => {
        const found = [];
        const close = (a, b, c, d) => Math.abs(a - c) <= tol && Math.abs(b - d) <= tol;
        for (const entry of drawn) {
            if (entry.op !== 'stroke' || !entry.main) continue;
            for (let k = 0; k + 1 < entry.parts.length; k++) {
                const [kind, ax, ay] = entry.parts[k], [next, bx, by] = entry.parts[k + 1];
                if (kind !== 'm' || next !== 'l') continue;
                if (close(ax, ay, x0, y0) && close(bx, by, x1, y1) || close(ax, ay, x1, y1) && close(bx, by, x0, y0)) found.push(entry.style);
            }
        }
        return found;
    };
    const stemOf = (node) => { const b = TreeView.nodePixels(TreeView.state, node); return [b.x0, b.y0, b.x, b.y]; };
"""


class ColorCase(TreePageMixin, unittest.TestCase):
    """A page over the eight-leaf tree above, whose network the Viewer has coloured."""

    def open_colored(self, painted=None, *, hidden=(), clusters=None, network=None, text=EIGHT_LEAVES):
        headers = list("abcdefgh")
        folder = self.make_folder()
        write_tree_file(folder, headers, text)
        count = len(network or headers)
        self.serve_headers(headers, folder=folder, tree=None, network=network,
                           colors=node_colors(count, BASE_PAINT if painted is None else painted), hidden=hidden, clusters=clusters)
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        # The page is listening: what the Viewer says of its colours reaches it.
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)

    def info(self):
        return self.state("""(() => { const c = TreeView.state.colors;
            return {version: c.version, leafClass: Array.from(c.leafClass), nodeClass: Array.from(c.nodeClass), dominant: Array.from(c.dominant),
                    purity: Array.from(c.purity).map(v => Math.round(v * 100) / 100), present: Array.from(c.present),
                    legend: c.legend.map(e => ({cls: e.cls, count: e.count, css: e.css, hex: e.hex})), more: c.more, css: c.css}; })()""")

    def choose(self, kind, mode="lengths"):
        self.script(f"TreeView.setLayout({json.dumps(kind)}, {json.dumps(mode)}); TreeView.drawNow(); return 1")

    def spied(self, body):
        """Draw a frame with spies on the drawing, and run ``body`` (which may use drawn, stylesOf and stemOf) on it."""
        return self.script(SPY + f"""
            let result;
            try {{ TreeView.drawNow(); result = (() => {{ {body} }})(); }} finally {{ restore(); }}
            return result;""")


class ColorModelTests(ColorCase):
    """What a clade's colour is: the colour that most of its leaves have, when nine in ten do."""

    def test_leaves_take_their_nodes_colours_and_clades_that_are_nine_tenths_one_colour_take_it(self):
        self.open_colored()
        found = self.info()
        # Red is class 1, blue 2, green 3 (in the order they are met), plain 0.
        self.assertEqual(found["leafClass"], [1, 1, 1, 1, 2, 2, 3, 0])
        # The plain leaf (h) is a class like the others, and (gh), half green and half plain, is neither.
        self.assertEqual(found["nodeClass"], [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 0, NO_CLASS, NO_CLASS, NO_CLASS])
        self.assertEqual(found["present"], [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8])
        # What most of a mixed clade has, and how much of it: (gh) is half green and half plain, the root is half red.
        self.assertEqual((found["dominant"][12], found["purity"][12]), (3, 0.5))
        self.assertEqual((found["dominant"][13], found["purity"][13]), (2, 0.5))
        self.assertEqual((found["dominant"][14], found["purity"][14]), (1, 0.5))
        self.assertEqual((found["dominant"][6], found["purity"][6]), (1, 1.0))

    def test_plain_is_drawn_but_is_no_choice_so_the_legend_leaves_it_out(self):
        self.open_colored()
        found = self.info()
        self.assertEqual([entry["cls"] for entry in found["legend"]], [1, 2, 3])
        self.assertEqual([entry["count"] for entry in found["legend"]], [4, 2, 1])
        self.assertEqual(found["more"], 0)
        # Nothing coloured: a tree in the plain colour, and no legend.
        self.open_colored({})
        self.assertEqual(self.info()["legend"], [])
        self.assertEqual(self.state("document.getElementById('legend').classList.contains('shown')"), False)

    def test_hidden_nodes_and_nodes_outside_the_network_have_no_colour_and_do_not_count(self):
        self.open_colored(hidden=[2])
        found = self.info()
        self.assertEqual(found["leafClass"][2], NO_CLASS)
        # (cd) holds one leaf that is shown, which is red: the clade is red.
        self.assertEqual((found["present"][5], found["nodeClass"][5]), (1, 1))
        self.assertEqual(found["legend"][0]["count"], 3)
        # A leaf that is not a node of the network has no colour either; (gh) is then green all through.
        self.open_colored(network=list("abcdefg"))
        found = self.info()
        self.assertEqual(found["leafClass"][7], NO_CLASS)
        self.assertEqual((found["present"][12], found["nodeClass"][12]), (1, 3))

    def test_colours_are_compared_in_classes_of_sixteen_levels_so_that_a_gradient_forms_clades(self):
        self.open_colored({0: "#fa0000", 1: "#ff0000", 2: "#e60000", 3: "#f10000"})
        found = self.info()
        self.assertEqual(found["leafClass"][:4], [1, 1, 2, 1])
        # A class is drawn in the first colour met in it, exactly.
        self.assertEqual(found["css"][1], "rgb(250, 0, 0)")
        self.assertEqual(self.state("TreeView.state.colors.legend[0].hex"), "#fa0000")

    def test_a_clade_that_is_not_quite_pure_takes_the_colour_by_the_votes_not_the_first_child(self):
        # Ten leaves under one parent: nine red among others and a blue one first.
        text = "#NEXUS\nBEGIN TREES;\n  TREE t = [&R] (j:1,a:1,b:1,c:1,d:1,e:1,f:1,g:1,h:1,i:1);\nEND;\n"
        headers = list("abcdefghij")
        folder = self.make_folder()
        write_tree_file(folder, headers, text)
        painted = {n: RED for n in range(10)}
        painted[9] = BLUE  # j, the first child
        self.serve_headers(headers, folder=folder, tree=None, colors=node_colors(10, painted))
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        found = self.info()
        root = found["nodeClass"][-1]
        # Blue is met first, so it is class 1 and red class 2.
        self.assertEqual((found["dominant"][-1], found["purity"][-1], root), (2, 0.9, 2))

    def test_cluster_numbers_name_the_colours_that_are_one_cluster_and_the_others_by_their_hex(self):
        self.open_colored(clusters=[0, 0, 0, 0, 1, 1, 2, -1])
        names = self.state("TreeView.state.colors.legend.map(e => TreeView.classLabel(e.cls))")
        self.assertEqual(names, ["Cluster 0", "Cluster 1", "Cluster 2"])
        # Two clusters in a colour, or none given: its #rrggbb.
        self.open_colored(clusters=[0, 0, 5, 5, 1, 1, 2, -1])
        self.assertEqual(self.state("TreeView.classLabel(1)"), "#ff0000")
        self.open_colored()
        self.assertEqual(self.state("TreeView.classLabel(2)"), "#0000ff")

    def test_the_legend_lists_twelve_colours_and_says_how_many_more_there_are(self):
        headers = headers_of(30)
        painted = {n: "#%02x%02x%02x" % (8 + 8 * n, 255 - 8 * n, (n * 37) % 256) for n in range(30)}
        self.serve_headers(headers, colors=node_colors(30, painted))
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        info = self.state("""({rows: document.querySelectorAll('#legend-list .legend-row').length, more: document.getElementById('legend-more').textContent,
                               counts: TreeView.state.colors.legend.map(e => e.count)})""")
        self.assertEqual(info["rows"], 12)
        self.assertEqual(info["more"], "18 more colours")
        self.assertEqual(info["counts"], sorted(info["counts"], reverse=True))

    def test_a_viewer_with_no_colours_leaves_the_tree_in_its_own(self):
        self.serve_headers(headers_of(20))
        self.open_page()
        self.pump(0.5)
        self.assertEqual(self.state("TreeView.state.colors"), None)
        self.assertEqual(self.state("document.getElementById('legend').classList.contains('shown')"), False)
        self.assertEqual(self.console_errors(), [])


class ColorDrawingTests(ColorCase):
    """The colours on the canvas, in each layout, from what is stroked and filled."""

    def test_a_leaf_is_drawn_in_its_colour_a_plain_one_in_the_plain_colour_and_a_hidden_one_not_at_all(self):
        self.open_colored(hidden=[2])
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied("""
                    const m = TreeView.state.model, p = TreeView.state.palette, at = (k) => stylesOf(...stemOf(m.leafNode[k]));
                    return {a: at(0), e: at(4), g: at(6), h: at(7), c: at(2)};""")
                self.assertEqual(found["a"], [[255, 0, 0]])
                self.assertEqual(found["e"], [[0, 0, 255]])
                self.assertEqual(found["g"], [[0, 255, 0]])
                self.assertEqual(found["h"], [[0x44, 0x88, 0xFF]])
                # A node the Viewer hides has no branch on the tree (test_tree_appearance_page.py).
                self.assertEqual(found["c"], [])

    def test_the_branches_of_a_clade_that_is_one_colour_are_drawn_in_it_and_a_mixed_clade_in_the_branch_colour(self):
        self.open_colored()
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied("""
                    const m = TreeView.state.model, p = TreeView.state.palette, at = (i) => stylesOf(...stemOf(i));
                    return {ab: at(2), abcd: at(6), ef: at(9), efgh: at(13), gh: at(12), branch: TreeView.rgbOf(p.branch)};""")
                self.assertEqual(found["ab"], [[255, 0, 0]])
                self.assertEqual(found["abcd"], [[255, 0, 0]])
                self.assertEqual(found["ef"], [[0, 0, 255]])
                self.assertEqual(found["efgh"], [found["branch"]])
                self.assertEqual(found["gh"], [found["branch"]])

    def test_a_clade_too_small_to_draw_is_one_line_in_its_colour_or_in_the_neutral_one(self):
        self.open_colored()
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                # The tree at a size that makes every clade under three pixels.
                found = self.spied("""
                    const s = TreeView.state, v = s.view, L = s.layout, start = {...v};
                    // Smaller by tenths until (abcd) is under three pixels: (efgh) is its twin, the root is twice as big.
                    let k = 1;
                    while (k > 1e-4 && !L.tiny(6, v.sx, v.sy, 3)) {
                        k *= 0.9;
                        v.sx = start.sx * k; v.sy = start.sy * k; v.ox = 650 + (start.ox - 650) * k; v.oy = 330 + (start.oy - 330) * k;
                    }
                    TreeView.drawNow();
                    const o = new Float64Array(8), line = (i) => { L.mark(i, o); return stylesOf(
                        v.ox + o[0] * v.sx, v.oy + o[1] * v.sy, v.ox + o[2] * v.sx, v.oy + o[3] * v.sy); };
                    return {bars: TreeView.lastStats.bars, rootTiny: L.tiny(14, v.sx, v.sy, 3), abcd: line(6), efgh: line(13),
                            mark: TreeView.rgbOf(s.palette.mark), branch: TreeView.rgbOf(s.palette.branch)};""")
                self.assertGreater(found["bars"], 0)
                self.assertFalse(found["rootTiny"])
                # (The clade's stem and its line are a pixel apart at this size, so each is found; both are in its colour.)
                self.assertEqual({tuple(style) for style in found["abcd"]}, {(255, 0, 0)})
                self.assertIn(found["mark"], found["efgh"])
                self.assertEqual({tuple(style) for style in found["efgh"]} - {tuple(found["mark"]), tuple(found["branch"])}, set())

    def test_a_folded_clade_is_filled_in_its_colour_and_has_a_bar_of_the_colours_it_holds(self):
        self.open_colored()
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                self.script("TreeView.openAll(); TreeView.toggleFold(6); TreeView.toggleFold(13); return 1")
                found = self.spied("""
                    const s = TreeView.state, p = s.palette;
                    const fills = drawn.filter(e => e.op === 'fill' && e.main).map(e => e.style);
                    const bars = drawn.filter(e => e.op === 'stroke' && e.main && e.width === 4).map(e => ({style: e.style, parts: e.parts.length}));
                    return {fills, bars, wedge: TreeView.rgbOf(p.wedge), tint: TreeView.mixColors('rgb(255, 0, 0)', p.background, 0.45)};""")
                # (abcd) is red all through: a red-tinted fill; (efgh) is mixed: the plain wedge.
                tint = TreeView_rgb(found["tint"])
                self.assertIn(tint, found["fills"])
                self.assertIn(found["wedge"], found["fills"])
                # (abcd) a bar in red alone, (efgh) one in blue, plain and green, as long as their shares (a half, two quarters).
                self.assertEqual(sorted(bar["style"] for bar in found["bars"]),
                                 sorted([[255, 0, 0], [0, 0, 255], [0x44, 0x88, 0xFF], [0, 255, 0]]))

    def test_the_bar_of_a_folded_rectangular_clade_is_in_pieces_as_long_as_the_shares(self):
        self.open_colored()
        self.choose("rectangular")
        self.script("TreeView.toggleFold(13); return 1")
        found = self.spied("""
            const s = TreeView.state, v = s.view, o = new Float64Array(8);
            s.layout.wedge(13, o);
            const x = v.ox + o[3] * v.sx, top = v.oy + o[4] * v.sy, bottom = v.oy + o[6] * v.sy;
            const pieces = drawn.filter(e => e.op === 'stroke' && e.main && e.width === 4).map(e => ({
                style: e.style, from: (e.parts[0][2] - top) / (bottom - top), to: (e.parts[1][2] - top) / (bottom - top), x: e.parts[0][1] - x}));
            return pieces.sort((a, b) => a.from - b.from);""")
        self.assertEqual([piece["style"] for piece in found], [[0, 0, 255], [0x44, 0x88, 0xFF], [0, 255, 0]])
        self.assertEqual([(round(piece["from"], 3), round(piece["to"], 3)) for piece in found], [(0.0, 0.5), (0.5, 0.75), (0.75, 1.0)])
        self.assertTrue(all(abs(piece["x"]) < 1e-6 for piece in found))

    def test_a_leaf_with_room_for_a_dot_has_it_in_its_colour(self):
        self.open_colored()
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied("""
                    const m = TreeView.state.model, p = TreeView.state.palette;
                    const dots = drawn.filter(e => e.op === 'fill' && e.main && e.parts.some(x => x[0] === 'a'));
                    const at = (k) => { const b = TreeView.nodePixels(TreeView.state, m.leafNode[k]);
                        return dots.filter(d => d.parts.some(x => x[0] === 'a' && Math.abs(x[1] - b.x) < 0.01 && Math.abs(x[2] - b.y) < 0.01)).map(d => d.style); };
                    return {a: at(0), e: at(4), h: at(7)};""")
                self.assertEqual(found["a"], [[255, 0, 0]])
                self.assertEqual(found["e"], [[0, 0, 255]])
                self.assertEqual(found["h"], [[0x44, 0x88, 0xFF]])

    def test_the_overview_map_is_drawn_in_the_colours_too(self):
        self.open_colored()
        for kind in ("rectangular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                reddish = self.state("""(() => { TreeView.drawNow();
                    const mini = document.getElementById('minimap'), data = mini.getContext('2d').getImageData(0, 0, mini.width, mini.height).data;
                    let n = 0;
                    for (let i = 0; i < data.length; i += 4) if (data[i] > 200 && data[i + 1] < 150 && data[i + 2] < 150 && data[i + 3] > 200) n++;
                    return n; })()""")
                self.assertGreater(reddish, 3)

    def test_a_tree_nobody_has_coloured_is_drawn_in_the_plain_colour(self):
        self.open_colored({})
        found = self.spied("""
            const m = TreeView.state.model, p = TreeView.state.palette;
            const reds = drawn.filter(e => e.op === 'stroke' && e.main && JSON.stringify(e.style) === JSON.stringify(TreeView.rgbOf(p.branch)));
            return {strokeStyles: new Set(drawn.filter(e => e.op === 'stroke' && e.main).map(e => JSON.stringify(e.style))).size,
                    leaf: stylesOf(...stemOf(m.leafNode[0])), branch: TreeView.rgbOf(p.branch)};""")
        self.assertEqual(found["leaf"], [[0x44, 0x88, 0xFF]])

    def test_drawing_with_colours_writes_nothing_to_the_console(self):
        self.open_colored(hidden=[1], clusters=[0, 0, 0, 0, 1, 1, 2, -1])
        for kind in ("rectangular", "circular", "unrooted"):
            self.choose(kind)
            self.script("TreeView.toggleFold(13); TreeView.drawNow(); TreeView.toggleFold(13); TreeView.drawNow(); return 1")
        self.assertEqual(self.console_errors(), [])


def TreeView_rgb(css):
    """The red, green and blue of the CSS colour ``rgb(r, g, b)`` the page's mixColors makes."""
    return [int(part) for part in css[css.index("(") + 1:css.index(")")].split(",")]


class ColorFollowTests(ColorCase):
    """The page follows the Viewer: its colours, which nodes are hidden, and the alignment."""

    def fetches(self):
        return self.state("performance.getEntriesByType('resource').filter(e => e.name.includes('/api/tree/colors')).length")

    def repaint(self):
        """What the Viewer does when it draws its nodes again: tell its listeners."""
        for listener in list(self.viewer.node_update_listeners):
            listener()

    def test_a_change_of_colour_is_followed_with_one_request_however_many_times_it_is_drawn(self):
        self.open_colored()
        before = self.info()
        fetched = self.fetches()
        for node in (0, 1, 2):
            self.viewer.current_colors[node] = (1.0, 0.5, 0.0, 1.0)
            self.repaint()
        self.wait_for(f"TreeView.state.colors.version !== {before['version']}")
        self.pump(0.4)
        self.assertEqual(self.fetches(), fetched + 1)
        after = self.info()
        # Orange is met first, so it is class 1, then red 2, blue 3 and green 4.
        self.assertEqual(after["leafClass"][:4], [1, 1, 1, 2])
        self.assertEqual([entry["count"] for entry in after["legend"]], [3, 2, 1, 1])
        self.assertEqual(self.console_errors(), [])

    def test_drawing_the_nodes_again_in_the_same_colours_asks_for_nothing(self):
        self.open_colored()
        fetched = self.fetches()
        for _ in range(4):
            self.repaint()
        self.pump(0.6)
        self.assertEqual(self.fetches(), fetched)

    def test_hiding_a_node_is_followed(self):
        self.open_colored()
        version = self.info()["version"]
        self.viewer.visible_mask[3] = False
        self.repaint()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        self.assertEqual(self.info()["leafClass"][3], NO_CLASS)

    def test_a_change_of_clusters_is_followed(self):
        self.open_colored(clusters=[0, 0, 0, 0, 1, 1, 2, -1])
        self.assertEqual(self.state("TreeView.classLabel(1)"), "Cluster 0")
        version = self.info()["version"]
        self.viewer.cluster_labels = np.array([4, 4, 4, 4, 1, 1, 2, -1], dtype=np.int32)
        self.repaint()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        self.assertEqual(self.state("TreeView.classLabel(1)"), "Cluster 4")

    def test_a_viewer_that_loses_its_colours_leaves_the_tree_in_its_own(self):
        self.open_colored()
        del self.viewer.current_colors
        self.viewer._tree_color_watch.latest = None
        self.repaint()
        self.wait_for("TreeView.state.colors === null")
        self.assertEqual(self.state("document.getElementById('legend').classList.contains('shown')"), False)

    def test_a_new_alignment_brings_its_tree_and_its_colours(self):
        self.open_colored()
        self.assertEqual(self.state("TreeView.state.model.leafCount"), 8)
        headers = headers_of(20)
        folder = self.make_folder()
        msa, _tree = make_alignment_folder(folder, headers)
        other = make_viewer(msa, headers, colors=node_colors(20, {0: RED, 1: RED}))
        self.viewer.alignment, self.viewer.full_headers, self.viewer.n_nodes = other.alignment, other.full_headers, other.n_nodes
        self.viewer.current_colors, self.viewer.visible_mask = other.current_colors, other.visible_mask
        self.viewer.broadcast_event({"type": "alignment_changed"})
        self.wait_for("TreeView.state.model.leafCount === 20 && TreeView.state.colors !== null && TreeView.state.colors.leafClass.length === 20")
        self.assertEqual(self.state("TreeView.state.colors.legend[0].count"), 2)
        self.assertEqual(self.console_errors(), [])

    def test_a_new_alignment_without_a_tree_says_so(self):
        self.open_colored()
        folder = self.make_folder()
        msa, _tree = make_alignment_folder(folder, headers_of(5), tree=None)
        other = make_viewer(msa, headers_of(5))
        self.viewer.alignment = other.alignment
        self.viewer.broadcast_event({"type": "alignment_changed"})
        self.wait_for("document.body.dataset.view === 'empty'")
        self.assertEqual(self.state("document.getElementById('empty-title').textContent"), "No guide tree was found for this alignment")

    def test_events_for_other_pages_are_ignored_and_not_parsed(self):
        self.open_colored()
        fetched = self.fetches()
        for event in ({"type": "selection_changed", "indices": [1, 2]}, {"type": "state_updated", "metadata": {}}, {"type": "unknown"}):
            self.viewer.broadcast_event(event)
        self.pump(0.5)
        self.assertEqual(self.fetches(), fetched)
        self.assertEqual(self.console_errors(), [])


class LegendTests(ColorCase):
    def rows(self):
        return self.state("""[...document.querySelectorAll('#legend-list .legend-row')].map(b => ({name: b.querySelector('.name').textContent,
            count: b.querySelector('.count').textContent, pressed: b.getAttribute('aria-pressed'), swatch: b.querySelector('.swatch').style.backgroundColor}))""")

    def test_the_legend_lists_each_colour_with_its_name_and_number_of_leaves(self):
        self.open_colored(clusters=[0, 0, 0, 0, 1, 1, 2, -1])
        self.assertEqual(self.rows(), [
            {"name": "Cluster 0", "count": "4", "pressed": "false", "swatch": "rgb(255, 0, 0)"},
            {"name": "Cluster 1", "count": "2", "pressed": "false", "swatch": "rgb(0, 0, 255)"},
            {"name": "Cluster 2", "count": "1", "pressed": "false", "swatch": "rgb(0, 255, 0)"},
        ])
        self.assertTrue(self.state("document.getElementById('legend').classList.contains('shown')"))

    def test_a_colour_marks_its_leaves_and_marks_them_off_again(self):
        self.open_colored()
        self.js("document.querySelectorAll('#legend-list .legend-row')[1].click()")
        self.assertEqual(self.state("Array.from(TreeView.state.matches)"), [4, 5])
        self.assertEqual([row["pressed"] for row in self.rows()], ["false", "true", "false"])
        self.js("document.querySelectorAll('#legend-list .legend-row')[1].click()")
        self.assertEqual(self.state("Array.from(TreeView.state.matches)"), [])
        self.assertEqual([row["pressed"] for row in self.rows()], ["false"] * 3)
        # Another colour in place of the first.
        self.js("document.querySelectorAll('#legend-list .legend-row')[0].click()")
        self.js("document.querySelectorAll('#legend-list .legend-row')[2].click()")
        self.assertEqual(self.state("Array.from(TreeView.state.matches)"), [6])
        self.assertEqual([row["pressed"] for row in self.rows()], ["false", "false", "true"])

    def test_a_search_replaces_what_a_colour_marked_and_a_colour_replaces_a_search(self):
        self.open_colored()
        self.js("document.querySelectorAll('#legend-list .legend-row')[0].click()")
        self.assertEqual(self.state("Array.from(TreeView.state.matches)"), [0, 1, 2, 3])
        self.script("""const input = document.getElementById('search-input'); input.value = 'g'; input.dispatchEvent(new Event('input')); return 1""")
        self.wait_for("TreeView.state.matches.length === 1")
        self.assertEqual([row["pressed"] for row in self.rows()], ["false"] * 3)
        self.js("document.querySelectorAll('#legend-list .legend-row')[1].click()")
        self.assertEqual(self.state("Array.from(TreeView.state.matches)"), [4, 5])
        self.assertEqual(self.state("document.getElementById('search-input').value"), "")

    def test_marked_leaves_are_drawn_as_a_search_marks_its_matches(self):
        self.open_colored()
        self.js("document.querySelectorAll('#legend-list .legend-row')[1].click()")
        info = self.script("""TreeView.drawNow();
            const m = TreeView.state.model, b = TreeView.nodePixels(TreeView.state, m.leafNode[4]), p = TreeView.state.palette;
            const probe = document.createElement('canvas').getContext('2d'); probe.fillStyle = p.match; probe.fillRect(0, 0, 1, 1);
            const want = probe.getImageData(0, 0, 1, 1).data;
            let best = 255;
            for (let dx = -12; dx <= 12; dx++) for (let dy = -12; dy <= 12; dy++) {
                const px = fire.pixel(Math.round(b.x) + dx, Math.round(b.y) + dy);
                best = Math.min(best, Math.max(Math.abs(px[0] - want[0]), Math.abs(px[1] - want[1]), Math.abs(px[2] - want[2])));
            }
            return best;""")
        self.assertLess(info, 30)

    def test_the_button_at_the_top_hides_and_shows_the_list(self):
        self.open_colored()
        self.assertEqual(self.state("document.getElementById('legend-list').hidden"), False)
        self.js("document.getElementById('legend-toggle').click()")
        self.assertEqual(self.state("[document.getElementById('legend-list').hidden, document.getElementById('legend-toggle').getAttribute('aria-expanded')]"),
                         [True, "false"])
        self.js("document.getElementById('legend-toggle').click()")
        self.assertEqual(self.state("document.getElementById('legend-list').hidden"), False)

    def test_new_colours_take_the_marks_off(self):
        self.open_colored()
        self.js("document.querySelectorAll('#legend-list .legend-row')[0].click()")
        version = self.info()["version"]
        self.viewer.current_colors[7] = (1.0, 0.0, 1.0, 1.0)
        for listener in list(self.viewer.node_update_listeners):
            listener()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        self.assertEqual(self.state("[Array.from(TreeView.state.matches), TreeView.state.legendClass]"), [[], -1])

    def test_the_legend_is_inside_the_theme_and_the_page(self):
        self.open_colored()
        info = self.state("""(() => { const box = document.getElementById('legend').getBoundingClientRect(), stage = document.getElementById('stage').getBoundingClientRect();
            return {inside: box.left >= stage.left && box.right <= stage.right && box.top >= stage.top && box.bottom <= stage.bottom,
                    minimap: document.getElementById('minimap').getBoundingClientRect().right <= box.left}; })()""")
        self.assertEqual(info, {"inside": True, "minimap": True})


class ColorHoverTests(ColorCase):
    def hover_text(self, node):
        point = self.state(f"""(() => {{ const b = TreeView.nodePixels(TreeView.state, {node});
            return [(b.x0 + b.x) / 2, (b.y0 + b.y) / 2]; }})()""")
        self.script(f"fire.move({point[0]}, {point[1]}); return 1")
        self.wait_for("document.getElementById('tooltip').style.display === 'block'")
        return self.state("[...document.getElementById('tooltip').children].map(e => e.textContent)")

    def test_a_leaf_says_its_cluster_and_a_clade_says_the_colour_most_of_its_leaves_have(self):
        self.open_colored(clusters=[0, 0, 0, 0, 1, 1, 2, -1])
        self.choose("rectangular")
        leaf = self.hover_text(0)
        self.assertIn("Cluster: 0", leaf)
        clade = self.hover_text(6)
        self.assertIn("Colour of 100% of its leaves: Cluster 0", clade)

    def test_a_leaf_with_no_cluster_and_a_clade_with_no_dominant_colour_say_nothing_of_them(self):
        self.open_colored(clusters=[0, 0, 0, 0, 1, 1, 2, -1])
        self.choose("rectangular")
        self.assertFalse(any(line.startswith("Cluster:") for line in self.hover_text(11)))
        # (efgh) is half blue: no colour has most of its leaves.
        text = self.hover_text(13)
        self.assertFalse(any("Colour of" in line for line in text), text)

    def test_without_clusters_a_clade_names_its_colour_by_its_hex(self):
        self.open_colored()
        self.choose("rectangular")
        self.assertIn("Colour of 100% of its leaves: #ff0000", self.hover_text(6))



class RootingNoteTests(TreePageMixin, unittest.TestCase):
    """An unrooted tree drawn from a root says so, in the banner and in Details; no other case says anything."""

    NOTE = "Drawn from the midpoint root; the tree itself is unrooted."

    def open_tree(self, flag, headers=None, **options):
        headers = headers or list("abcd")
        folder = self.make_folder()
        write_tree_file(folder, headers, f"#NEXUS\nBEGIN TREES;\n  TREE t = {flag}(((a:1,b:2):1,c:3):1,d:2);\nEND;\n")
        self.serve_headers(headers, folder=folder, tree=None, **options)
        self.open_page()

    def choose(self, kind):
        self.script(f"""const select = document.getElementById('layout-select');
            select.value = {json.dumps(kind)}; select.dispatchEvent(new Event('change')); TreeView.drawNow(); return 1""")

    def shown(self):
        return self.script("""
            const detail = document.getElementById('details-rooting');
            return {banner: document.getElementById('banner').classList.contains('shown'), kind: TreeView.state.kind,
                    items: [...document.querySelectorAll('#warning-list li')].map(li => li.textContent),
                    noted: [...document.querySelectorAll('#warning-list li.rooting-note > svg use')].map(use => use.getAttribute('href')),
                    detail: detail.hidden ? null : detail.textContent};""")

    def test_the_note_is_shown_for_exactly_an_unrooted_tree_in_a_layout_that_has_a_root(self):
        # [&R] is a rooted tree; [&U] an unrooted one; a file that says nothing is taken as unrooted.
        for flag, rooted in (("[&R] ", True), ("[&U] ", False), ("", None)):
            self.open_tree(flag)
            self.assertEqual(self.state("TreeView.state.model.rooted"), rooted)
            # The layouts in turn, so that the note is seen to follow the layout on the same page.
            for kind in ("unrooted", "rectangular", "circular", "unrooted", "circular", "rectangular"):
                with self.subTest(rooted=rooted, layout=kind):
                    self.choose(kind)
                    found = self.shown()
                    wanted = rooted is not True and kind != "unrooted"
                    self.assertEqual(found["kind"], kind)
                    self.assertEqual(found["items"], [self.NOTE] if wanted else [])
                    self.assertEqual(found["banner"], wanted)
                    self.assertEqual(found["detail"], self.NOTE if wanted else None)
        self.assertEqual(self.console_errors(), [])

    def test_a_rooted_tree_opens_without_a_note_and_an_unrooted_one_too(self):
        # The default layouts are the ones that need none.
        for flag in ("[&R] ", "[&U] ", ""):
            with self.subTest(flag=flag):
                self.open_tree(flag)
                found = self.shown()
                self.assertEqual((found["banner"], found["items"], found["detail"]), (False, [], None))

    def test_the_note_comes_after_the_warnings_and_has_its_own_icon(self):
        headers = list("abcd")
        self.open_tree("[&U] ", headers=headers, network=["a", "b", "c", "zz"])
        self.choose("rectangular")
        found = self.shown()
        self.assertEqual(len(found["items"]), 3)
        self.assertEqual(found["items"][-1], self.NOTE)
        self.assertEqual(found["noted"], ["#i-info"])
        self.choose("unrooted")
        found = self.shown()
        self.assertEqual(len(found["items"]), 2)
        self.assertTrue(found["banner"])
        self.assertNotIn(self.NOTE, found["items"])

    # ---- The x on the note: it hides the note for the life of this page and is remembered nowhere ----------------

    def close_button(self):
        return self.script("""const buttons = [...document.querySelectorAll('#warning-list .note-close')];
            return buttons.map(b => ({tag: b.tagName, type: b.type, parent: b.parentElement.className, label: b.getAttribute('aria-label'),
                title: b.title, tabIndex: b.tabIndex, disabled: b.disabled, icon: b.querySelector('use').getAttribute('href'),
                last: b.parentElement.lastElementChild === b}));""")

    def press(self, key):
        """A real key press on the page (not a script's event): a button answers it only if it is a real button."""
        QTest.keyClick(self.view.focusProxy() or self.view, key)
        self.pump(0.1)

    def click_close(self):
        self.script("document.querySelector('#warning-list .note-close').click(); return 1")

    def test_the_x_is_there_exactly_when_the_note_is_and_is_a_labelled_icon_button(self):
        self.open_tree("[&U] ", headers=list("abcd"), network=["a", "b", "c", "zz"])
        for kind, wanted in (("rectangular", 1), ("unrooted", 0), ("circular", 1)):
            with self.subTest(layout=kind):
                self.choose(kind)
                buttons = self.close_button()
                self.assertEqual(len(buttons), wanted)
                if wanted:
                    self.assertEqual(buttons, [{"tag": "BUTTON", "type": "button", "parent": "rooting-note", "label": "Hide this note",
                                                "title": "Hide this note", "tabIndex": 0, "disabled": False, "icon": "#i-x", "last": True}])
        # Warnings have none: only the note has an x.
        self.assertEqual(self.state("document.querySelectorAll('#warning-list li').length"), 3)
        self.assertEqual(self.state("document.querySelectorAll('#warning-list button').length"), 1)
        # A rooted tree has no note, so no x.
        self.open_tree("[&R] ")
        self.choose("rectangular")
        self.assertEqual(self.close_button(), [])
        self.assertEqual(self.console_errors(), [])

    def test_a_click_removes_the_note_and_keeps_the_warnings(self):
        self.open_tree("[&U] ", headers=list("abcd"), network=["a", "b", "c", "zz"])
        self.choose("rectangular")
        self.assertEqual(len(self.shown()["items"]), 3)
        self.click_close()
        found = self.shown()
        self.assertEqual(len(found["items"]), 2)
        self.assertNotIn(self.NOTE, found["items"])
        self.assertTrue(found["banner"])
        self.assertEqual(self.close_button(), [])
        # The Details panel keeps its copy.
        self.assertEqual(found["detail"], self.NOTE)

    def test_a_click_on_a_banner_of_the_note_alone_hides_the_banner_and_refits_the_drawing(self):
        self.open_tree("[&U] ")
        self.choose("rectangular")
        self.wait_for("TreeView.state.fitted")
        self.wait_for("document.getElementById('banner').classList.contains('shown')")
        before = self.state("document.getElementById('tree-canvas').clientHeight")
        spread = "(() => { const s = TreeView.state, v = s.view; let m = -1e9; for (let i = 0; i < s.model.nodeCount; i++) m = Math.max(m, v.oy + s.layout.y[i] * v.sy); return m; })()"
        low = self.state(spread)
        self.click_close()
        found = self.shown()
        self.assertEqual((found["banner"], found["items"], found["detail"]), (False, [], self.NOTE))
        self.wait_for(f"document.getElementById('tree-canvas').clientHeight > {before}")
        self.wait_for("TreeView.state.fitted")
        # Refitted at once, to the room there is now (not left for the resize observer): the whole tree is inside it.
        info = self.script("""const s = TreeView.state, v = s.view, L = s.layout, c = document.getElementById('tree-canvas');
            let maxY = -1e9, minY = 1e9;
            for (let i = 0; i < s.model.nodeCount; i++) { const y = v.oy + L.y[i] * v.sy; maxY = Math.max(maxY, y); minY = Math.min(minY, y); }
            return {minY, maxY, height: c.clientHeight};""")
        self.assertGreaterEqual(info["minY"], 0)
        self.assertLessEqual(info["maxY"], info["height"])
        self.assertGreater(info["maxY"], low)

    def test_the_refit_is_made_by_the_click_itself(self):
        # The resize observer would refit too, a frame later: fitView is watched to see that the click asks for it.
        self.open_tree("[&U] ")
        self.choose("rectangular")
        self.wait_for("TreeView.state.fitted")
        self.script("window.__fits = 0; const fit = TreeView.fitView; TreeView.fitView = function () { window.__fits++; return fit.apply(this, arguments); }; return 1")
        # Click and count in one script run, so that nothing the observer does a frame later is counted.
        fits = self.script("document.querySelector('#warning-list .note-close').click(); return window.__fits")
        self.assertGreaterEqual(fits, 1)

    def test_the_note_stays_gone_across_layouts_and_a_new_alignment_payload(self):
        self.open_tree("[&U] ", headers=list("abcd"), network=["a", "b", "c", "zz"])
        self.choose("rectangular")
        self.click_close()
        for kind in ("circular", "unrooted", "rectangular", "circular"):
            with self.subTest(layout=kind):
                self.choose(kind)
                found = self.shown()
                self.assertNotIn(self.NOTE, found["items"])
                self.assertEqual(self.close_button(), [])
                # Details still repeats it where the layout calls for it.
                self.assertEqual(found["detail"], None if kind == "unrooted" else self.NOTE)
        self.assertTrue(self.script("return await TreeView.loadTree()"))
        self.viewer.broadcast_event({"type": "alignment_changed"})
        self.pump(0.5)
        self.wait_for("document.body.dataset.view === 'tree'")
        # A new payload opens its tree by its rooting again; the layouts that have a root must still show no note.
        for kind in ("rectangular", "circular"):
            self.choose(kind)
            found = self.shown()
            self.assertNotIn(self.NOTE, found["items"])
            self.assertEqual(self.close_button(), [])
            self.assertEqual(found["detail"], self.NOTE)
        self.assertEqual(self.console_errors(), [])

    def test_a_reload_brings_the_note_back_and_nothing_is_stored(self):
        self.open_tree("[&U] ")
        self.js("(() => { try { localStorage.clear(); sessionStorage.clear(); } catch (error) {} return 1; })()")
        self.choose("rectangular")
        self.click_close()
        self.assertEqual(self.shown()["items"], [])
        stored = self.script("""const keys = {local: [], session: []};
            try { for (let i = 0; i < localStorage.length; i++) keys.local.push(localStorage.key(i)); } catch (error) {}
            try { for (let i = 0; i < sessionStorage.length; i++) keys.session.push(sessionStorage.key(i)); } catch (error) {}
            return {keys, cookie: document.cookie};""")
        self.assertEqual(stored, {"keys": {"local": [], "session": []}, "cookie": ""})
        self.open_page()
        self.choose("rectangular")
        self.assertEqual(self.shown()["items"], [self.NOTE])
        self.assertEqual(len(self.close_button()), 1)

    def test_the_x_is_reached_by_tab_and_answered_by_enter_and_space(self):
        for key in (Qt.Key_Return, Qt.Key_Space):
            with self.subTest(key=int(key)):
                self.open_tree("[&U] ")
                self.choose("rectangular")
                self.script("document.querySelector('#warning-list .note-close').focus(); return 1")
                self.assertEqual(self.state("document.activeElement.className"), "note-close")
                self.press(key)
                self.wait_for("document.querySelectorAll('#warning-list .note-close').length === 0")
                self.assertEqual(self.shown()["items"], [])
        # Tab reaches it: from the search box on, through the controls, in the page's order.
        self.open_tree("[&U] ")
        self.choose("rectangular")
        order = self.script("""const x = document.querySelector('#warning-list .note-close');
            const focusable = [...document.querySelectorAll('button, input, select, canvas[tabindex], [tabindex]')]
                .filter(e => !e.disabled && e.tabIndex >= 0 && e.offsetParent !== null);
            return {reachable: focusable.includes(x), banner: document.getElementById('banner').compareDocumentPosition(x)};""")
        self.assertTrue(order["reachable"])

    def test_the_drawing_fits_the_room_that_is_left_under_the_note(self):
        self.open_tree("[&U] ")
        self.choose("unrooted")
        before = self.state("document.getElementById('tree-canvas').clientHeight")
        self.choose("rectangular")
        self.wait_for(f"document.getElementById('tree-canvas').clientHeight < {before}")
        self.wait_for("TreeView.state.fitted")
        info = self.script("""const s = TreeView.state, v = s.view, L = s.layout, c = document.getElementById('tree-canvas');
            let maxY = -1e9, minY = 1e9;
            for (let i = 0; i < s.model.nodeCount; i++) { const y = v.oy + L.y[i] * v.sy; maxY = Math.max(maxY, y); minY = Math.min(minY, y); }
            return {minY, maxY, height: c.clientHeight, fitted: s.fitted};""")
        self.assertTrue(info["fitted"])
        self.assertGreaterEqual(info["minY"], 0)
        self.assertLessEqual(info["maxY"], info["height"])
        # And back: the room comes back with the note gone.
        self.choose("unrooted")
        self.wait_for(f"document.getElementById('tree-canvas').clientHeight >= {before}")


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


class StrokeAndGestureTests(LayoutCase):
    """The lines of the tree are hairline passes where the pixel ratio would send them to Skia's slow path, and the
    names wait for the view to rest after a gesture."""

    # What each stroke() call of a frame was: the path it drew, the line width and where the context was moved (device
    # pixels), in the order of the calls.
    STROKES = """(ratio) => {
        Object.defineProperty(window, 'devicePixelRatio', {get: () => ratio, configurable: true});
        const proto = CanvasRenderingContext2D.prototype, original = proto.stroke, paths = [], calls = [];
        proto.stroke = function (path) {
            if (path instanceof Path2D) {
                let id = paths.indexOf(path);
                if (id < 0) id = paths.push(path) - 1;
                const m = this.getTransform();
                calls.push({path: id, width: this.lineWidth, dx: Math.round((m.e) * 1000) / 1000, dy: Math.round(m.f * 1000) / 1000, ratio: m.a});
            }
            return original.apply(this, arguments);
        };
        try { TreeView.drawNow(); } finally { proto.stroke = original; delete window.devicePixelRatio; }
        return calls;
    }"""

    def strokes(self, ratio):
        """The stroke calls of tree lines in a frame drawn at the device pixel ratio ``ratio``, as a list of passes per path."""
        calls = self.script(f"return ({self.STROKES})({ratio})")
        groups = {}
        for call in calls:
            groups.setdefault(call["path"], []).append(call)
        return list(groups.values())

    def test_a_one_pixel_line_at_ratio_one_is_one_stroke(self):
        groups = self.strokes(1)
        # The branches, 1 pixel wide, are drawn in one stroke at their own width and place.
        singles = [passes for passes in groups if len(passes) == 1 and passes[0]["width"] == 1]
        self.assertTrue(singles)
        for passes in singles:
            self.assertEqual((passes[0]["dx"], passes[0]["dy"]), (0, 0))
        # Marks of 2 and 3 pixels are hairline passes of 1 pixel; no line of 1 pixel is drawn as several.
        for passes in groups:
            if len(passes) > 1:
                self.assertEqual({p["width"] for p in passes}, {1})
                self.assertIn(len(passes), (4, 5))

    def test_a_two_device_pixel_line_is_four_hairlines_moved_by_half_a_pixel(self):
        groups = self.strokes(2)
        fours = [passes for passes in groups if len(passes) == 4]
        self.assertTrue(fours)
        for passes in fours:
            self.assertEqual({p["width"] for p in passes}, {0.5})
            # Device pixels: the context is moved left, right, up and down by half a pixel, on a base of 0.
            self.assertEqual(sorted((p["dx"], p["dy"]) for p in passes), [(-0.5, 0), (0, -0.5), (0, 0.5), (0.5, 0)])

    def test_a_three_device_pixel_line_is_five_hairlines(self):
        groups = self.strokes(3)
        fives = [passes for passes in groups if len(passes) == 5]
        self.assertTrue(fives)
        for passes in fives:
            self.assertEqual({round(p["width"] * 3, 3) for p in passes}, {1})
            self.assertEqual(sorted((p["dx"], p["dy"]) for p in passes), [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)])

    def test_lines_over_three_and_a_half_device_pixels_are_not_touched(self):
        groups = self.strokes(4)
        self.assertTrue(groups)
        for passes in groups:
            self.assertEqual(len(passes), 1)
            self.assertGreaterEqual(passes[0]["width"] * 4, 4)
            self.assertEqual((passes[0]["dx"], passes[0]["dy"]), (0, 0))

    def names(self):
        return self.state("TreeView.lastStats.labels")

    def show_names(self):
        self.script("TreeView.focusLeaf(5); TreeView.drawNow(); return 1")
        self.assertGreater(self.names(), 0)

    FILLTEXT = """(() => {
        const proto = CanvasRenderingContext2D.prototype, original = proto.fillText;
        let count = 0;
        proto.fillText = function () { count++; return original.apply(this, arguments); };
        try { TreeView.drawNow(); } finally { proto.fillText = original; }
        return count;
    })()"""

    def test_no_names_are_drawn_while_a_gesture_moves_the_view_and_they_return_when_it_rests(self):
        gestures = {
            "wheel": "fire.wheel(-20, 600, 300)",
            "drag": "fire.pointer('pointerdown', 600, 300); fire.pointer('pointermove', 640, 330)",
            "key": "fire.key('ArrowLeft')",
        }
        for name, gesture in gestures.items():
            with self.subTest(gesture=name):
                self.script("TreeView.focusLeaf(5); TreeView.drawNow(); return 1")
                self.assertGreater(self.names(), 0)
                self.js(gesture + "; 1")
                self.assertEqual(self.state(self.FILLTEXT), 0)
                self.assertEqual(self.names(), 0)
                if name == "drag":
                    self.js("fire.pointer('pointerup', 640, 330); 1")
                # After the idle delay a frame with the names follows by itself.
                self.wait_for("TreeView.lastStats.labels > 0")
                self.assertGreater(self.state(self.FILLTEXT), 0)

    def test_jumping_to_a_leaf_or_changing_the_selection_still_draws_names(self):
        self.script("TreeView.focusLeaf(7); return 1")
        self.assertGreater(self.state(self.FILLTEXT), 0)
        self.script("TreeView.applySelection([5, 6, 7]); return 1")
        self.assertGreater(self.state(self.FILLTEXT), 0)
        self.assertFalse(self.state("TreeView.state.moving"))
        # The page's own zoom and pan are no gestures either.
        self.script("TreeView.zoomAbout(600, 300, 1.2, 1.2); TreeView.pan(10, 5); return 1")
        self.assertFalse(self.state("TreeView.state.moving"))
        self.assertGreater(self.state(self.FILLTEXT), 0)

    def test_the_marks_of_matches_and_the_selection_are_hairline_passes_too(self):
        # At a ratio of 1.5 their 2 pixels are 3 device pixels: no stroke of the frame is left at the line width of 2.
        self.script("""const m = TreeView.state.model;
            let node = m.childList[m.childStart[m.root]];
            TreeView.toggleFold(node);
            TreeView.state.matches = Int32Array.of(m.firstLeaf[node]);
            TreeView.applySelection([m.viewerNode[m.firstLeaf[node]]]);
            return 1""")
        groups = self.strokes(1.5)
        self.assertTrue(groups)
        self.assertNotIn(2, {p["width"] for passes in groups for p in passes})


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
                # The legend says, once, what the stability is not; nothing is ever labelled with the word.
                self.assertNotIn("bootstrap", source.replace(LEGEND_NOT_BOOTSTRAP, "").lower())

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



class FitNamesTests(LayoutCase):
    """At the fitted view the leaf names that are drawn are all inside the canvas: the fit leaves room for them."""

    COUNT = 40

    def setUp(self):
        # Names of a length a leaf can carry: 18 characters, in a view where the rows are tall enough for them.
        TreePageMixin.setUp(self)
        self.headers = [f"leaf{i:03d} protein {i % 10}" for i in range(self.COUNT)]
        self.serve_headers(self.headers, **nj_options(self.COUNT))
        self.open_page()

    # Every name a frame draws: its box in canvas pixels (the corners of its text rectangle under the transform in use).
    NAME_BOXES = """(() => {
        const proto = CanvasRenderingContext2D.prototype, original = proto.fillText, boxes = [];
        proto.fillText = function (text, x, y) {
            const m = this.getTransform(), width = this.measureText(text).width;
            const size = parseFloat(this.font), left = this.textAlign === 'right' ? x - width : x;
            const ratio = TreeView.canvas.width / TreeView.canvas.clientWidth;
            const corners = [[left, y - size * 0.6], [left + width, y - size * 0.6], [left, y + size * 0.6], [left + width, y + size * 0.6]]
                .map(([px, py]) => [(m.a * px + m.c * py + m.e) / ratio, (m.b * px + m.d * py + m.f) / ratio]);
            boxes.push({text, x0: Math.min(...corners.map(c => c[0])), x1: Math.max(...corners.map(c => c[0])),
                        y0: Math.min(...corners.map(c => c[1])), y1: Math.max(...corners.map(c => c[1]))});
            return original.apply(this, arguments);
        };
        try { TreeView.drawNow(); } finally { proto.fillText = original; }
        const canvas = TreeView.canvas;
        return {boxes, width: canvas.clientWidth, height: canvas.clientHeight, ruler: TreeView.RULER_WIDTH,
                fitted: TreeView.state.fitted, sy: TreeView.state.view.sy, sx: TreeView.state.view.sx};
    })()"""

    def fit_names(self, kind=None):
        if kind:
            self.script(f"TreeView.setLayout({json.dumps(kind)}, 'lengths'); return 1")
        self.script("TreeView.fitView(); return 1")
        return self.state(self.NAME_BOXES)

    def assert_inside(self, frame, kind):
        self.assertTrue(frame["fitted"])
        self.assertGreater(len(frame["boxes"]), 0, kind)
        right = frame["width"] - (frame["ruler"] if kind == "rectangular" else 0)
        for box in frame["boxes"]:
            self.assertGreaterEqual(box["x0"], 0, (kind, box))
            self.assertLessEqual(box["x1"], right, (kind, box))
            self.assertGreaterEqual(box["y0"], 0, (kind, box))
            self.assertLessEqual(box["y1"], frame["height"], (kind, box))

    def test_every_name_drawn_at_fit_is_inside_the_canvas_and_left_of_the_ruler(self):
        frame = self.fit_names("rectangular")
        self.assert_inside(frame, "rectangular")
        # Every leaf is named: the room was made, not the names dropped.
        self.assertEqual(len(frame["boxes"]), self.COUNT)

    def test_the_circular_and_unrooted_fit_is_the_one_without_room_for_names(self):
        for kind in ("circular", "unrooted"):
            with self.subTest(kind=kind):
                info = self.script(f"""
                    TreeView.setLayout({json.dumps(kind)}, 'lengths');
                    const layout = TreeView.state.layout, b = layout.bounds, c = TreeView.canvas, v = TreeView.state.view;
                    let bare = Math.min((c.clientWidth - 48) / (b.maxX - b.minX), (c.clientHeight - 32) / (b.maxY - b.minY));
                    if (kind === 'circular') bare = Math.min(bare, 40 / (layout.reach[layout.reach.length - 1] * layout.slot));
                    return {{sx: v.sx, sy: v.sy, bare, ox: v.ox, oy: v.oy, cx: c.clientWidth / 2 - (b.minX + b.maxX) / 2 * bare,
                             cy: c.clientHeight / 2 - (b.minY + b.maxY) / 2 * bare, labels: TreeView.drawNow().labels}};""".replace("kind ===", json.dumps(kind) + " ==="))
                self.assertGreater(info["labels"], 0)
                for got, want in (("sx", "bare"), ("sy", "bare"), ("ox", "cx"), ("oy", "cy")):
                    self.assertAlmostEqual(info[got], info[want], delta=1e-9)

    def test_names_of_a_dark_theme_or_a_long_name_keep_the_same_promise(self):
        # The longest name decides the room; a longer one than the rest takes more of it.
        long_name = "x" * 90
        self.script(f"TreeView.state.model.labels = TreeView.state.model.labels.map((t, k) => k === 7 ? {json.dumps(long_name)} : t); return 1")
        frame = self.fit_names("rectangular")
        self.assert_inside(frame, "rectangular")
        self.assertIn(long_name, [box["text"] for box in frame["boxes"]])

    def test_a_name_longer_than_the_share_of_the_canvas_is_given_that_share_only(self):
        self.fit_names("rectangular")
        self.script("TreeView.state.model.labels = TreeView.state.model.labels.map((t, k) => k === 7 ? 'W'.repeat(150) : t); return 1")
        frame = self.fit_names("rectangular")
        room = frame["width"] - 24 - 24 - frame["ruler"] - frame["sx"] * self.state("TreeView.state.layout.width")
        self.assertAlmostEqual(room, 0.4 * frame["width"], delta=1)

    def test_rows_too_thin_for_names_keep_the_full_width_fit(self):
        self.script("TreeView.setLayout('rectangular', 'lengths'); return 1")
        self.view.resize(1300, 500)
        self.wait_for("document.getElementById('tree-canvas').clientHeight < 600")
        self.script("TreeView.fitView(); return 1")
        frame = self.state(self.NAME_BOXES)
        self.assertLess(frame["sy"], 12)
        self.assertEqual(frame["boxes"], [])
        full = (frame["width"] - 24 - 24 - frame["ruler"]) / self.state("TreeView.state.layout.width")
        self.assertAlmostEqual(frame["sx"], full, delta=1e-6)

    def test_a_resize_refits_with_room_for_the_names(self):
        self.script("TreeView.setLayout('rectangular', 'lengths'); return 1")
        for width, height in ((900, 800), (700, 820), (1500, 900)):
            with self.subTest(size=(width, height)):
                self.view.resize(width, height)
                self.wait_for(f"document.getElementById('tree-canvas').clientWidth > {width - 400} && "
                              f"document.getElementById('tree-canvas').clientWidth < {width + 1}")
                self.pump(0.4)
                frame = self.state(self.NAME_BOXES)
                self.assertTrue(frame["fitted"])
                self.assertEqual(len(frame["boxes"]), self.COUNT)
                self.assert_inside(frame, "rectangular")

    def test_hiding_nodes_refits_with_room_for_the_names_that_are_left(self):
        # The longest name is hidden: the rows grow and the room follows the names that remain.
        self.script("TreeView.state.model.labels = TreeView.state.model.labels.map((t, k) => k === 3 ? 'L'.repeat(100) : t); return 1")
        before = self.fit_names("rectangular")
        self.assertIn("L" * 100, [box["text"] for box in before["boxes"]])
        self.script("""const hidden = new Uint8Array(TreeView.state.model.leafCount); hidden[3] = 1;
            TreeView.applyRemoved(hidden); return 1""")
        after = self.state(self.NAME_BOXES)
        self.assertTrue(after["fitted"])
        self.assertNotIn("L" * 100, [box["text"] for box in after["boxes"]])
        self.assert_inside(after, "rectangular")
        self.assertGreater(after["sx"], before["sx"])

    def test_zoom_pan_and_the_ruler_are_measured_from_the_fit_that_made_room(self):
        self.script("TreeView.setLayout('rectangular', 'lengths'); return 1")
        fit = self.state("TreeView.state.fit")
        self.assertEqual(fit, self.state("TreeView.state.view"))
        self.script("fire.wheel(-200, 400, 300); return 1")
        self.script("fire.key('0'); return 1")
        self.assertEqual(self.state("TreeView.state.view"), fit)
        # The ruler strip is not covered by the tips: the tree's far edge is left of the names' room.
        frame = self.state(self.NAME_BOXES)
        tips = fit["ox"] + fit["sx"] * self.state("TreeView.state.layout.width")
        self.assertLess(tips, frame["width"] - frame["ruler"])

    def test_names_are_measured_once_for_a_tree_and_not_at_all_where_none_are_drawn(self):
        # A frame of 3000 leaves at fit has no names, so the fit measures none of them.
        counts = self.script("""
            const proto = CanvasRenderingContext2D.prototype, original = proto.measureText;
            let calls = 0;
            proto.measureText = function () { calls++; return original.apply(this, arguments); };
            try {
                TreeView.state.model.labels = TreeView.state.model.labels.slice();
                TreeView.setLayout('rectangular', 'lengths');
                const first = calls; TreeView.fitView();
                const second = calls; TreeView.fitView();
                return {first, second: second - first, third: calls - second};
            } finally { proto.measureText = original; }""")
        self.assertGreater(counts["first"], 0)
        self.assertEqual(counts["third"], 0)


class FitNamesThinTreeTests(TreePageMixin, unittest.TestCase):
    COUNT = 300

    def test_a_tree_of_thin_rows_is_fitted_to_the_full_width_and_measures_no_name(self):
        headers = headers_of(self.COUNT)
        self.serve_headers(headers, **nj_options(self.COUNT))
        self.open_page()
        info = self.script("""
            const proto = CanvasRenderingContext2D.prototype, original = proto.measureText;
            let calls = 0;
            proto.measureText = function () { calls++; return original.apply(this, arguments); };
            try {
                TreeView.setLayout('rectangular', 'lengths');
                const view = TreeView.state.view, layout = TreeView.state.layout;
                return {calls, sy: view.sy, sx: view.sx, full: (TreeView.canvas.clientWidth - 48 - TreeView.RULER_WIDTH) / layout.width};
            } finally { proto.measureText = original; }""")
        self.assertLess(info["sy"], 12)
        self.assertEqual(info["calls"], 0)
        self.assertAlmostEqual(info["sx"], info["full"], delta=1e-9)
        # With more leaves than are ever named, the others are fitted as they were, and name nothing.
        for kind in ("circular", "unrooted"):
            with self.subTest(kind=kind):
                frame = self.script(f"""
                    TreeView.setLayout({json.dumps(kind)}, 'lengths');
                    const layout = TreeView.state.layout, b = layout.bounds, c = TreeView.canvas;
                    const bare = Math.min((c.clientWidth - 48) / (b.maxX - b.minX), (c.clientHeight - 32) / (b.maxY - b.minY));
                    return {{scale: TreeView.state.view.sx, bare, labels: TreeView.drawNow().labels}};""")
                self.assertEqual(frame["labels"], 0)
                self.assertAlmostEqual(frame["scale"], frame["bare"], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
