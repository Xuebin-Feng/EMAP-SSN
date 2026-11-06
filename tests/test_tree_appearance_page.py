# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's look as the network's: the colours it draws in, and what it leaves out.

The tree is drawn in the colours of the network: a node nobody coloured is the Viewer's initial node colour and the
tree has it too; the nodes the network has selected are outlined in the colour the Viewer outlines them in, and those
next to them in the Viewer's connected node colour, while it identifies connected nodes; what the search and the
legend mark is in the theme colour; and a node the Viewer has hidden is not on the tree at all, which is the tree of
the nodes that are left. The page is opened offscreen over the real web server (tests/tree_page_harness.py), and which
colour each part is drawn in is read from spies on the drawing, as the colour tests of test_tree_page.py do.
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

from tests.test_tree_page import BASE_PAINT, EIGHT_LEAVES, NO_CLASS, SPY, write_tree_file  # noqa: E402
from tests.tree_fixtures import headers_of, node_colors  # noqa: E402
from tests.tree_page_harness import TreePageMixin  # noqa: E402
from web_ui import tree_backend  # noqa: E402

KINDS = ("rectangular", "circular", "unrooted")
PLAIN = [0x44, 0x88, 0xFF]
SELECTED = [0xFF, 0xAA, 0x00]
CONNECTED = [0xFF, 0x00, 0x00]

# SPY (the strokes and fills of paths, with their lines and arcs) and the strokes, fills and rectangles that are
# not of a path: the rings and ticks of the search, the ticks of the ruler.
SPY_ALL = SPY + """
    const protoAll = CanvasRenderingContext2D.prototype;
    const pathStroke = protoAll.stroke, pathFill = protoAll.fill;
    const rectOriginal = {fillRect: protoAll.fillRect, strokeRect: protoAll.strokeRect};
    protoAll.stroke = function (path) {
        if (main(this) && !(path instanceof Path2D)) drawn.push({op: 'stroke', main: true, style: TreeView.rgbOf(this.strokeStyle), width: this.lineWidth, parts: [], bare: true});
        return pathStroke.apply(this, arguments);
    };
    protoAll.fill = function (path) {
        if (main(this) && !(path instanceof Path2D)) drawn.push({op: 'fill', main: true, style: TreeView.rgbOf(this.fillStyle), parts: [], bare: true});
        return pathFill.apply(this, arguments);
    };
    protoAll.fillRect = function (x, y, w, h) {
        if (main(this)) drawn.push({op: 'fillRect', main: true, style: TreeView.rgbOf(this.fillStyle), rect: [x, y, w, h], parts: []});
        return rectOriginal.fillRect.apply(this, arguments);
    };
    protoAll.strokeRect = function (x, y, w, h) {
        if (main(this)) drawn.push({op: 'strokeRect', main: true, style: TreeView.rgbOf(this.strokeStyle), rect: [x, y, w, h], parts: []});
        return rectOriginal.strokeRect.apply(this, arguments);
    };
    const restoreAll = () => { restore(); Object.assign(protoAll, rectOriginal); };
    const same = (a, b) => a.length === b.length && a.every((value, k) => value === b[k]);
    // The fills of the marks of a leaf (an arc as big as a mark, not a dot), by colour.
    const marksOf = (k) => {
        const s = TreeView.state, b = TreeView.nodePixels(s, s.model.leafNode[k]);
        return drawn.filter(e => e.op === 'fill' && e.main && e.parts.some(p => p[0] === 'a' && p[3] >= 3.4 && Math.abs(p[1] - b.x) < 0.01 && Math.abs(p[2] - b.y) < 0.01))
            .map(e => e.style);
    };
"""


class AppearanceCase(TreePageMixin, unittest.TestCase):
    """A page over the eight-leaf tree of test_tree_page.py (a to h, node 0 to 7), whose network the Viewer has coloured."""

    def open_tree(self, painted=None, *, hidden=(), network=None, edges=None, text=EIGHT_LEAVES, headers=None):
        headers = headers or list("abcdefgh")
        folder = self.make_folder()
        write_tree_file(folder, headers, text)
        count = len(network or headers)
        self.serve_headers(headers, folder=folder, tree=None, network=network,
                           colors=node_colors(count, BASE_PAINT if painted is None else painted), hidden=hidden)
        self.enable_selection()
        if edges is not None:
            self.give_network(edges)
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)
        self.js("""window.__heard = 0; const previous = TreeView.onSelectionChanged;
            TreeView.onSelectionChanged = () => { window.__heard++; previous(); }; 1""")

    def give_network(self, edges):
        """Give the served Viewer the edges, and the two methods of the real Viewer that name a selection's neighbours."""
        from EMAPSSN_Viewer import MainViewer

        viewer = self.viewer
        viewer.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        for name in ("_adjacency_index", "_connected_node_identification_enabled"):
            setattr(viewer, name, getattr(MainViewer, name).__get__(viewer))

    def select(self, indices, *, neighbours=None):
        """What a command's selecting does; waits until the page has the selection, and the neighbours when it should."""
        before = self.state("window.__heard")
        self.viewer.select(indices)
        self.wait_for(f"window.__heard > {before}")
        if neighbours is not None:
            self.wait_for(f"TreeView.state.selection.connected.length === {neighbours}")

    def hide(self, nodes):
        """The Viewer hides these nodes (and shows the rest): it draws the nodes, and the page is told its colours changed."""
        self.viewer.visible_mask[:] = True
        self.viewer.visible_mask[list(nodes)] = False
        for listener in self.viewer.node_update_listeners:
            listener()
        wanted = tree_backend.colors_snapshot(self.viewer)["version"]
        self.wait_for(f"TreeView.state.colors.version === {wanted}", timeout=20)

    def choose(self, kind, mode="lengths"):
        self.script(f"TreeView.setLayout({json.dumps(kind)}, {json.dumps(mode)}); TreeView.drawNow(); return 1")

    def spied(self, body):
        return self.script(SPY_ALL + f"""
            let result;
            try {{ TreeView.drawNow(); result = (() => {{ {body} }})(); }} finally {{ restoreAll(); }}
            return result;""")


class PlainColourTests(AppearanceCase):
    """A node nobody coloured is the Viewer's initial node colour, and the tree draws it in that colour."""

    def test_leaves_and_clades_that_are_plain_are_drawn_in_the_plain_colour_in_every_layout(self):
        # Every node plain: the whole tree is in it.
        self.open_tree({})
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied("""
                    const m = TreeView.state.model, at = (i) => { const b = TreeView.nodePixels(TreeView.state, i); return stylesOf(b.x0, b.y0, b.x, b.y); };
                    return {leaf: at(m.leafNode[0]), clade: at(2), big: at(6), root: at(14), branch: TreeView.rgbOf(TreeView.state.palette.branch)};""")
                for name in ("leaf", "clade", "big"):
                    self.assertEqual(found[name], [PLAIN], name)
                self.assertNotIn(found["branch"], [found["leaf"][0]])

    def test_a_mixed_clade_is_drawn_in_the_page_branch_colour_and_the_plain_leaf_inside_it_in_plain(self):
        self.open_tree()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied("""
                    const m = TreeView.state.model, p = TreeView.state.palette, at = (i) => { const b = TreeView.nodePixels(TreeView.state, i); return stylesOf(b.x0, b.y0, b.x, b.y); };
                    return {h: at(m.leafNode[7]), gh: at(12), efgh: at(13), root: at(14), ab: at(2), branch: TreeView.rgbOf(p.branch)};""")
                self.assertEqual(found["h"], [PLAIN])
                # (gh) is half green and half plain: neither colour, so the neutral one; as are (efgh) and the root.
                self.assertEqual(found["gh"], [found["branch"]])
                self.assertEqual(found["efgh"], [found["branch"]])
                self.assertEqual(found["ab"], [[255, 0, 0]])

    def test_a_clade_that_is_nine_tenths_plain_is_plain_and_one_that_is_not_is_mixed(self):
        text = "#NEXUS\nBEGIN TREES;\n  TREE t = [&R] (j:1,a:1,b:1,c:1,d:1,e:1,f:1,g:1,h:1,i:1);\nEND;\n"
        headers = list("abcdefghij")
        for painted, expected in (({0: "#ff0000"}, 0), ({0: "#ff0000", 1: "#ff0000"}, NO_CLASS)):
            with self.subTest(coloured=len(painted)):
                self.open_tree(painted, text=text, headers=headers)
                self.assertEqual(self.state("TreeView.state.colors.nodeClass[TreeView.state.model.root]"), expected)

    def test_the_plain_colour_is_the_viewers_initial_node_colour_it_is_set_to(self):
        import EMAPSSN_Config as cfg

        with mock.patch.object(cfg, "INITIAL_NODE_COLOR", "#aa5500"):
            self.open_tree({})
            self.choose("rectangular")
            found = self.spied("""
                const m = TreeView.state.model, b = TreeView.nodePixels(TreeView.state, m.leafNode[0]);
                return stylesOf(b.x0, b.y0, b.x, b.y);""")
        self.assertEqual(found, [[0xAA, 0x55, 0x00]])

    def test_the_legend_leaves_plain_out_and_its_counts_are_of_the_colours_alone(self):
        self.open_tree()
        found = self.state("""({legend: TreeView.state.colors.legend.map(e => [e.cls, e.count]),
            plain: TreeView.state.colors.leafClass.filter(c => c === 0).length,
            rows: document.querySelectorAll('#legend-list .legend-row').length})""")
        self.assertEqual(found["legend"], [[1, 4], [2, 2], [3, 1]])
        self.assertEqual((found["plain"], found["rows"]), (1, 3))
        self.open_tree({})
        self.assertEqual(self.state("TreeView.state.colors.legend.length"), 0)
        self.assertFalse(self.state("document.getElementById('legend').classList.contains('shown')"))

    def test_a_leaf_that_is_not_a_node_of_the_network_is_faded_not_plain(self):
        self.open_tree(network=list("abcdefg"))
        self.choose("rectangular")
        found = self.spied("""
            const m = TreeView.state.model, b = TreeView.nodePixels(TreeView.state, m.leafNode[7]);
            return {h: stylesOf(b.x0, b.y0, b.x, b.y), faded: TreeView.rgbOf(TreeView.state.palette.faded)};""")
        self.assertEqual(found["h"], [found["faded"]])

    def test_a_folded_clade_of_plain_nodes_is_tinted_in_plain_and_its_bar_is_plain(self):
        self.open_tree({})
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.script("TreeView.openAll(); TreeView.toggleFold(6); return 1")
                found = self.spied("""
                    const s = TreeView.state;
                    return {fills: drawn.filter(e => e.op === 'fill' && e.main && !e.bare).map(e => e.style),
                            bars: drawn.filter(e => e.op === 'stroke' && e.main && e.width === 4).map(e => e.style),
                            tint: TreeView.mixColors('rgb(68, 136, 255)', s.palette.background, 0.45)};""")
                tint = [int(part) for part in found["tint"][found["tint"].index("(") + 1:found["tint"].index(")")].split(",")]
                self.assertIn(tint, found["fills"])
                self.assertEqual(found["bars"], [PLAIN])

    def test_the_overview_map_has_the_plain_colour_too(self):
        self.open_tree({})
        bluish = self.state("""(() => { TreeView.drawNow();
            const mini = document.getElementById('minimap'), data = mini.getContext('2d').getImageData(0, 0, mini.width, mini.height).data;
            let n = 0;
            for (let i = 0; i < data.length; i += 4) if (data[i + 2] > 200 && data[i] < 120 && data[i + 3] > 200) n++;
            return n; })()""")
        self.assertGreater(bluish, 3)

    def test_the_plain_colour_is_seen_in_both_themes(self):
        for theme in ("light", "dark"):
            with self.subTest(theme=theme):
                self.open_tree({})
                found = self.state(f"""(() => {{ document.documentElement.setAttribute('data-theme', {json.dumps(theme)});
                    const s = TreeView.state; s.palette = TreeView.readPalette();
                    const ink = TreeView.plainInk(s.colors, s.palette);
                    return {{ink, ratio: TreeView.contrast(ink, s.palette.background), network: s.colors.css[0]}}; }})()""")
                self.assertEqual(found["ink"], found["network"])
                self.assertGreaterEqual(found["ratio"], 3)

    def test_a_plain_colour_the_page_could_not_see_is_moved_toward_the_text_colour_until_it_can_be(self):
        import EMAPSSN_Config as cfg

        for theme, color, moved in (("light", "#ffffff", True), ("dark", "#18181b", True), ("dark", "#ffffff", False)):
            with self.subTest(theme=theme, color=color), mock.patch.object(cfg, "INITIAL_NODE_COLOR", color):
                self.open_tree({})
                found = self.state(f"""(() => {{ document.documentElement.setAttribute('data-theme', {json.dumps(theme)});
                    const s = TreeView.state; s.palette = TreeView.readPalette();
                    const ink = TreeView.plainInk(s.colors, s.palette);
                    return {{ink, network: s.colors.css[0], ratio: TreeView.contrast(ink, s.palette.background),
                            drawn: TreeView.contrast(s.colors.css[0], s.palette.background)}}; }})()""")
                self.assertGreaterEqual(found["ratio"], 3)
                self.assertEqual(found["ink"] != found["network"], moved)


class SelectionColourTests(AppearanceCase):
    """The nodes the network has selected are drawn in the colour the Viewer outlines them in, which the Viewer sends."""

    def test_the_page_declares_no_colour_of_its_own_for_them(self):
        html = (SRC_DIR / "web_ui" / "tree.html").read_text(encoding="utf-8")
        for source in ("tree.html", "tree_render.js"):
            self.assertNotIn("--tree-selected", (SRC_DIR / "web_ui" / source).read_text(encoding="utf-8"), source)
        # A hex colour in a script would be a copy of a setting the Viewer owns.
        for name in ("tree_render.js", "tree_select.js", "tree_nav.js", "tree_session.js", "tree_minimap.js"):
            text = (SRC_DIR / "web_ui" / name).read_text(encoding="utf-8")
            for value in ("#ffaa00", "#ff0000", "#4488ff"):
                self.assertNotIn(value, text.lower(), f"{name} copies {value}")
        self.assertIn("--accent", html)

    def test_selected_leaves_clades_and_the_ruler_are_in_the_highlight_colour(self):
        self.open_tree()
        self.select([0, 1, 4])
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied("""
                    const s = TreeView.state;
                    const outline = drawn.filter(e => (e.op === 'stroke' || e.op === 'strokeRect') && e.main && same(e.style, [255, 170, 0]));
                    return {a: marksOf(0), e: marksOf(4), c: marksOf(2), marks: s.marks,
                            outlined: outline.length};""")
                self.assertEqual(found["a"], [SELECTED])
                self.assertEqual(found["e"], [SELECTED])
                self.assertEqual(found["c"], [])
                self.assertEqual(found["marks"]["selected"], "rgb(255, 170, 0)")
                # The smallest clade that holds the selection is outlined in it too (the root, here, at the dashed outline).
                self.assertGreater(found["outlined"], 0)

    def test_the_colour_follows_the_viewers_setting(self):
        import EMAPSSN_Config as cfg

        self.open_tree()
        self.select([0])
        with mock.patch.object(cfg, "HOVER_COLOR", "#00aa55"):
            for listener in self.viewer.node_update_listeners:
                listener()
            self.wait_for("TreeView.state.marks.selected === 'rgb(0, 170, 85)'")
            self.choose("rectangular")
            found = self.spied("return marksOf(0)")
        self.assertEqual(found, [[0, 170, 85]])

    def test_the_ruler_and_the_overview_map_show_it_in_that_colour(self):
        self.open_tree()
        self.choose("rectangular")
        self.select([0, 1])
        found = self.spied("""
            return {ruler: drawn.filter(e => e.op === 'fillRect' && e.main && same(e.style, [255, 170, 0]) && e.rect[2] === 10).length};""")
        self.assertEqual(found["ruler"], 2)  # one tick for each of the two leaves
        amber = self.state("""(() => { TreeView.drawNow();
            const mini = document.getElementById('minimap'), data = mini.getContext('2d').getImageData(0, 0, mini.width, mini.height).data;
            let n = 0;
            for (let i = 0; i < data.length; i += 4) if (data[i] > 225 && data[i + 1] > 140 && data[i + 1] < 190 && data[i + 2] < 45 && data[i + 3] > 200) n++;
            return n; })()""")
        self.assertGreater(amber, 0)


class ConnectedTests(AppearanceCase):
    """The nodes next to the selected ones, while the Viewer outlines them in its connected node colour."""

    EDGES = [[0, 4], [0, 5], [1, 2], [6, 7]]

    def test_the_neighbours_leaves_are_marked_in_the_connected_colour_in_every_layout(self):
        self.open_tree(edges=self.EDGES)
        self.select([0], neighbours=2)
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied("return {a: marksOf(0), e: marksOf(4), f: marksOf(5), b: marksOf(1), c: marksOf(2), on: TreeView.state.marks.connectedOn}")
                self.assertEqual(found["a"], [SELECTED])
                self.assertEqual(found["e"], [CONNECTED])
                self.assertEqual(found["f"], [CONNECTED])
                self.assertEqual((found["b"], found["c"]), ([], []))
                self.assertTrue(found["on"])

    def test_a_node_that_is_selected_and_next_to_one_is_drawn_as_selected(self):
        self.open_tree(edges=self.EDGES)
        self.select([0, 4], neighbours=1)  # 0 and 4 touch; 5 is the only other neighbour
        self.choose("rectangular")
        found = self.spied("return {a: marksOf(0), e: marksOf(4), f: marksOf(5)}")
        self.assertEqual(found["e"], [SELECTED])
        self.assertEqual(found["f"], [CONNECTED])
        self.assertEqual(self.state("Array.from(TreeView.state.selection.connected)"), [5])

    def test_where_both_are_named_the_selected_colour_is_drawn_last_and_so_shows(self):
        # The Viewer draws a node that is selected and next to one as selected; the server leaves the selected out of
        # the neighbours, and the page keeps the order all the same.
        self.open_tree(edges=self.EDGES)
        self.select([0], neighbours=2)
        self.script("TreeView.state.selection.connected = Int32Array.of(0, 4, 5); return 1")
        self.choose("rectangular")
        found = self.spied("return marksOf(0)")
        self.assertEqual(found, [CONNECTED, SELECTED])

    def test_the_neighbours_follow_the_selection_and_go_with_it(self):
        self.open_tree(edges=self.EDGES)
        self.select([0], neighbours=2)
        self.select([1], neighbours=1)
        self.assertEqual(self.state("Array.from(TreeView.state.selection.connected)"), [2])
        self.select([], neighbours=0)
        self.assertEqual(self.state("Array.from(TreeView.state.selection.connected)"), [])

    def test_a_page_opened_after_the_selection_has_the_neighbours_too(self):
        self.open_tree(edges=self.EDGES)
        self.viewer.selected_indices = [0]
        self.open_page()
        self.wait_for("TreeView.state.selection.connected.length === 2")
        self.assertEqual(self.state("Array.from(TreeView.state.selection.leaves)"), [0])

    def test_with_identification_off_the_viewer_outlines_none_and_so_does_the_tree(self):
        import EMAPSSN_Config as cfg

        with mock.patch.object(cfg, "NODE_BOUNDARY_COLOR", cfg.CONNECTED_NODE_COLOR):
            self.open_tree(edges=self.EDGES)
            self.select([0])
            self.pump(0.5)
            self.choose("rectangular")
            found = self.spied("return {e: marksOf(4), a: marksOf(0), on: TreeView.state.marks.connectedOn}")
            self.assertEqual((found["e"], found["a"], found["on"]), ([], [SELECTED], False))
            self.assertEqual(self.state("TreeView.state.selection.connected.length"), 0)
            # No request is made for neighbours the Viewer does not outline: one when the page opened, none for the selection.
            asked = self.state("performance.getEntriesByType('resource').filter(e => e.name.includes('/api/tree/selection')).length")
            self.select([1])
            self.pump(0.3)
            self.assertEqual(self.state("performance.getEntriesByType('resource').filter(e => e.name.includes('/api/tree/selection')).length"), asked)

    def test_an_answer_about_an_older_selection_is_not_used(self):
        self.open_tree(edges=self.EDGES)
        # The first request is answered late, with the neighbours of a selection that is gone; the second at once.
        self.script("""
            const encode = (nodes) => { const bytes = new Uint8Array(Int32Array.of(...nodes).buffer); let s = ''; for (const b of bytes) s += String.fromCharCode(b); return btoa(s); };
            const answer = (nodes) => new Response(JSON.stringify({status: 'ok', connected_on: true, connected_count: nodes.length, connected: encode(nodes)}), {headers: {'Content-Type': 'application/json'}});
            const real = window.fetch; let calls = 0; const applied = [];
            window.fetch = (url, options) => {
                if (!String(url).includes('/api/tree/selection')) return real(url, options);
                calls++;
                return calls === 1 ? new Promise((resolve) => setTimeout(() => resolve(answer([4])), 400)) : Promise.resolve(answer([5]));
            };
            const apply = TreeView.applyConnected;
            TreeView.applyConnected = (nodes) => { applied.push(Array.from(nodes)); return apply(nodes); };
            window.__applied = applied; window.__real = real; return 1;""")
        self.viewer.selected_indices = [0]
        self.viewer.broadcast_event({"type": "selection_changed", "indices": [0]})
        self.pump(0.1)
        self.viewer.selected_indices = [1]
        self.viewer.broadcast_event({"type": "selection_changed", "indices": [1]})
        self.wait_for("window.__applied.length >= 1")
        self.pump(0.8)
        self.assertEqual(self.state("window.__applied"), [[5]])
        self.script("window.fetch = window.__real; return 1")

    def test_the_viewers_switch_taking_effect_while_the_page_is_open_is_followed(self):
        import EMAPSSN_Config as cfg

        self.open_tree(edges=self.EDGES)
        self.select([0], neighbours=2)
        with mock.patch.object(cfg, "NODE_BOUNDARY_COLOR", cfg.CONNECTED_NODE_COLOR):
            for listener in self.viewer.node_update_listeners:
                listener()
            self.wait_for("TreeView.state.marks.connectedOn === false")
            self.wait_for("TreeView.state.selection.connected.length === 0")

    def test_no_edge_reaches_the_page(self):
        self.open_tree(edges=self.EDGES)
        self.select([0], neighbours=2)
        found = self.script("""const reply = await (await fetch('/api/tree/selection')).json(); return Object.keys(reply).sort();""")
        self.assertEqual(found, ["connected", "connected_count", "connected_on", "count", "focus", "indices", "status"])

    def test_a_hidden_neighbour_is_not_drawn_and_the_tree_is_not_given_room_for_it(self):
        self.open_tree(edges=self.EDGES)
        self.select([0], neighbours=2)
        self.hide([4])
        self.choose("rectangular")
        found = self.spied("""
            const s = TreeView.state;
            return {f: marksOf(5), live: Array.from(s.model.leafNode).map(n => s.layout.live[n]), rows: s.layout.height};""")
        self.assertEqual(found["f"], [CONNECTED])
        self.assertEqual((found["live"][4], found["rows"]), (0, 7))

    def test_the_tooltip_of_a_neighbour_says_so(self):
        self.open_tree(edges=self.EDGES)
        self.select([0], neighbours=2)
        self.assertEqual(self.state("TreeView.describeNode(TreeView.state.model.leafNode[4]).lines.includes('Next to a selected node of the network')"), True)
        self.assertEqual(self.state("TreeView.describeNode(TreeView.state.model.leafNode[2]).lines.includes('Next to a selected node of the network')"), False)
        self.assertEqual(self.state("TreeView.describeNode(TreeView.state.model.root).lines.includes('Next to a selected node of the network: 2')"), True)

    def test_forty_four_thousand_selected_nodes_leave_the_page_responsive(self):
        count = 44000
        headers = headers_of(count)
        rng = np.random.default_rng(42)
        edges = rng.integers(0, count, size=(count * 10, 2))
        self.serve_headers(headers, colors=node_colors(count))
        self.enable_selection()
        self.give_network(edges)
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)
        self.viewer.selected_indices = list(range(0, count, 2))
        started = time.perf_counter()
        self.viewer.broadcast_event({"type": "selection_changed", "indices": self.viewer.selected_indices})
        self.wait_for("TreeView.state.selection.connected.length > 10000", timeout=60)
        self.assertLess(time.perf_counter() - started, 20)
        self.assertEqual(self.console_errors(), [])


class AccentTests(AppearanceCase):
    """What the search and the legend mark is drawn in the theme colour: the page's --accent, which is the desktop's."""

    def test_the_pages_accent_is_the_studio_theme_accent_in_every_theme_colour_light_and_dark(self):
        from desktop import Studio_Theme
        from desktop.Theme_Colors import THEME_COLORS

        def luminance(color):
            channels = [int(color[k:k + 2], 16) / 255 for k in (1, 3, 5)]
            channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
            return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

        def contrast(a, b):
            high, low = sorted((luminance(a), luminance(b)), reverse=True)
            return (high + 0.05) / (low + 0.05)

        for name in THEME_COLORS:
            with self.subTest(color=name):
                gui = Studio_Theme.theme_tokens(name)["accent"]
                light = Studio_Theme.web_page_properties(name)
                dark = Studio_Theme.web_page_properties(name, dark=True)
                # The window and the light page have the same accent; the dark page has the family's lighter shade.
                self.assertEqual(light["--accent"], gui)
                self.assertNotEqual(dark["--accent"], "")
                self.assertGreaterEqual(contrast(light["--accent"], light["--bg-card"]), 4.5)
                self.assertGreaterEqual(contrast(dark["--accent"], dark["--bg-card"]), 4.5)

    def test_the_served_page_has_the_accent_of_the_theme_colour_in_use(self):
        from desktop import Studio_Theme

        original = Studio_Theme.theme_color()
        self.addCleanup(Studio_Theme.set_theme_color, original)
        Studio_Theme.set_theme_color("teal")
        self.open_tree()
        for theme, dark in (("light", False), ("dark", True)):
            found = self.state(f"""(() => {{ document.documentElement.setAttribute('data-theme', {json.dumps(theme)});
                return getComputedStyle(document.documentElement).getPropertyValue('--accent').trim(); }})()""")
            self.assertEqual(found, Studio_Theme.web_page_properties("teal", dark=dark)["--accent"])
        self.assertEqual(Studio_Theme.web_page_properties("teal")["--accent"], Studio_Theme.theme_tokens("teal")["accent"])

    def test_search_marks_are_in_the_accent_in_every_layout_and_theme(self):
        self.open_tree()
        for theme in ("light", "dark"):
            for kind in KINDS:
                with self.subTest(theme=theme, kind=kind):
                    self.script(f"document.documentElement.setAttribute('data-theme', {json.dumps(theme)}); return 1")
                    self.choose(kind)
                    self.script("TreeView.runSearch('c'); TreeView.state.current = -1; return 1")
                    found = self.spied("""
                        const s = TreeView.state, accent = TreeView.rgbOf(s.palette.accent), b = TreeView.nodePixels(s, s.model.leafNode[2]);
                        const ring = drawn.filter(e => e.main && e.bare && e.op === 'stroke');
                        return {accent, rings: ring.map(e => e.style), fills: drawn.filter(e => e.main && e.bare && e.op === 'fill').map(e => e.style),
                                found: Array.from(s.matches)};""")
                    self.assertEqual(found["found"], [2])
                    self.assertIn(found["accent"], found["rings"])
                    self.assertNotIn([245, 158, 11], found["rings"])
                    self.assertNotIn([251, 191, 36], found["rings"])

    def test_the_search_stands_on_one_match_with_a_solid_mark_and_the_others_see_through(self):
        self.open_tree()
        self.choose("rectangular")
        self.script("TreeView.runSearch('a'); return 1")  # no header of a to h but 'a' holds
        found = self.spied("""
            const s = TreeView.state;
            s.current = 0; TreeView.drawNow();
            const solid = drawn.filter(e => e.main && e.bare && e.op === 'fill').map(e => e.style);
            return {solid, accent: TreeView.rgbOf(s.palette.accent), matches: s.matches.length};""")
        self.assertEqual(found["matches"], 1)
        self.assertIn(found["accent"], found["solid"])

    def test_a_legend_click_marks_its_leaves_in_the_accent_and_a_second_click_takes_them_off(self):
        self.open_tree()
        self.choose("rectangular")
        self.script("document.querySelector('#legend-list .legend-row').click(); return 1")
        self.wait_for("TreeView.state.legendClass >= 0")
        found = self.spied("""
            const s = TreeView.state, accent = TreeView.rgbOf(s.palette.accent);
            return {accent, rings: drawn.filter(e => e.main && e.bare && e.op === 'stroke').map(e => e.style),
                    marked: Array.from(s.matches), ticks: drawn.filter(e => e.op === 'fillRect' && e.main && same(e.style, accent)).length};""")
        self.assertEqual(found["marked"], [0, 1, 2, 3])
        self.assertIn(found["accent"], found["rings"])
        self.script("document.querySelector('#legend-list .legend-row').click(); return 1")
        self.wait_for("TreeView.state.legendClass === -1")
        self.assertEqual(self.state("TreeView.state.matches.length"), 0)

    def test_the_ruler_and_the_overview_map_mark_the_matches_in_the_accent(self):
        self.open_tree()
        self.choose("rectangular")
        self.script("TreeView.runSearch('c'); return 1")
        found = self.spied("""
            const accent = TreeView.rgbOf(TreeView.state.palette.accent);
            return {ruler: drawn.filter(e => e.op === 'fillRect' && e.main && e.rect[2] === 10 && same(e.style, accent)).length};""")
        self.assertEqual(found["ruler"], 1)


def hidden_cases():
    """Which nodes to hide in the eight-leaf tree, and what that leaves: a leaf, a whole pair, three in two clades."""
    return (
        ("a leaf", [0]),
        ("a whole clade", [4, 5]),
        ("three in two clades", [1, 2, 6]),
    )


class HiddenNodeTests(AppearanceCase):
    """A node the Viewer hides is not on the tree: its branch goes, the others close up, and nothing is left faded."""

    def leaves_left(self, hidden):
        return [k for k in range(8) if k not in hidden]

    def test_hidden_leaves_have_no_row_no_angle_and_no_place_and_the_others_take_them_in_order(self):
        for label, hidden in hidden_cases():
            self.open_tree()
            self.hide(hidden)
            for kind in KINDS:
                with self.subTest(case=label, kind=kind):
                    self.choose(kind)
                    found = self.state("""(() => { const s = TreeView.state, m = s.model, L = s.layout;
                        return {live: Array.from(m.leafNode).map(n => L.live[n]), y: Array.from(m.leafNode).map(n => L.y[n]), x: Array.from(m.leafNode).map(n => L.x[n]),
                                angle: L.angle ? Array.from(m.leafNode).map(n => L.angle[n]) : null, height: L.height, slot: L.slot}; })()""")
                    left = self.leaves_left(hidden)
                    self.assertEqual([k for k in range(8) if found["live"][k]], left)
                    if kind == "rectangular":
                        self.assertEqual([found["y"][k] for k in left], [row + 0.5 for row in range(len(left))])
                        self.assertEqual(found["height"], len(left))
                    elif kind == "circular":
                        slot = 2 * np.pi / len(left)
                        for row, k in enumerate(left):
                            self.assertAlmostEqual(found["angle"][k], -np.pi / 2 + slot * (row + 0.5), places=9)
                        self.assertAlmostEqual(found["slot"], slot, places=12)

    def test_the_tree_drawn_is_the_tree_of_the_leaves_left_none_faded_none_visited_that_is_gone(self):
        for label, hidden in hidden_cases():
            self.open_tree()
            self.hide(hidden)
            for kind in KINDS:
                with self.subTest(case=label, kind=kind):
                    self.choose(kind)
                    found = self.spied("""
                        const s = TreeView.state, faded = TreeView.rgbOf(s.palette.faded);
                        const live = Array.from(s.layout.live).reduce((a, b) => a + b, 0);
                        return {visited: TreeView.lastStats.visited, live, dots: TreeView.lastStats.dots,
                                faded: drawn.filter(e => e.main && e.parts.length && same(e.style, faded)).length, leaves: s.layout.live ? s.model.leafCount : 0};""")
                    self.assertEqual(found["visited"], found["live"])
                    self.assertEqual(found["dots"], 8 - len(hidden))
                    self.assertEqual(found["faded"], 0)

    def test_the_width_of_the_tree_is_of_the_leaves_left_not_of_the_one_that_was_farthest(self):
        text = "#NEXUS\nBEGIN TREES;\n  TREE t = [&R] (((a:1,b:1):1,(c:1,d:1):1):1,((e:1,f:1):1,(g:1,h:10):1):1);\nEND;\n"
        self.open_tree(text=text)
        self.choose("rectangular")
        self.assertEqual(self.state("TreeView.state.layout.width"), 12)
        self.hide([7])
        self.choose("rectangular")
        self.assertEqual(self.state("TreeView.state.layout.width"), 3)
        self.choose("circular")
        self.assertEqual(self.state("TreeView.state.layout.width"), 6)

    def test_every_path_between_two_leaves_left_keeps_its_length(self):
        for label, hidden in hidden_cases():
            self.open_tree()
            self.hide(hidden)
            self.choose("rectangular")
            found = self.state("""(() => { const s = TreeView.state, m = s.model, L = s.layout;
                const depth = (k) => { let d = 0; for (let i = m.leafNode[k]; m.parent[i] >= 0; i = m.parent[i]) d += m.length[i]; return d; };
                return Array.from({length: m.leafCount}, (_, k) => [L.live[m.leafNode[k]], L.x[m.leafNode[k]], depth(k)]); })()""")
            with self.subTest(case=label):
                for live, x, depth in found:
                    if live:
                        self.assertAlmostEqual(x, depth, places=5)

    def test_a_node_left_with_one_child_has_no_join_and_sits_where_the_child_does(self):
        self.open_tree()
        self.hide([1, 5])  # (ab) keeps a, (cd) keeps c
        for kind in ("rectangular", "circular"):
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.state("""(() => { const L = TreeView.state.layout, out = new Float64Array(8), m = TreeView.state.model;
                    L.join(2, out); const pass = out[0]; L.join(6, out); const joined = out[0];
                    return {pass, joined, ab: [L.x[2], L.y[2]], a: [L.x[0], L.y[0]], live: [L.live[2], L.live[1]]}; })()""")
                self.assertEqual((found["pass"], found["joined"]), (0, 1 if kind == "rectangular" else 2))
                if kind == "circular":
                    self.assertNotEqual(found["ab"], found["a"])  # the node is nearer the root, on the same ray
                else:
                    self.assertEqual(found["ab"][1], found["a"][1])
                self.assertEqual(found["live"], [1, 0])

    def test_a_clade_whose_leaves_are_all_hidden_is_not_there(self):
        self.open_tree()
        self.hide([4, 5])
        found = self.state("Array.from(TreeView.state.layout.live)")
        # Nodes: e is 7, f is 8, (ef) is 9, (gh) is 12, (efgh) is 13.
        self.assertEqual((found[7], found[8], found[9]), (0, 0, 0))
        self.assertEqual((found[12], found[13]), (1, 1))

    def test_a_hidden_node_cannot_be_pointed_at_searched_for_or_gone_to(self):
        self.open_tree()
        self.hide([2])
        self.choose("rectangular")
        found = self.state("""(() => { const s = TreeView.state, m = s.model, v = s.view;
            // Where leaf c would have been, were it left: the middle of its old row, the row's place now.
            const hit = TreeView.hitTest(v.ox + s.layout.x[m.leafNode[2]] * v.sx, v.oy + 2.5 * v.sy);
            const before = {...s.view};
            TreeView.focusLeaf(2); TreeView.zoomToNode(m.leafNode[2]);
            const same = JSON.stringify(before) === JSON.stringify(s.view);
            return {found: TreeView.runSearch('c'), same, node: hit ? hit.node : null}; })()""")
        self.assertEqual(found["found"], 0)
        self.assertTrue(found["same"])
        self.assertNotEqual(found["node"], 2)

    def test_a_search_made_before_a_node_is_hidden_is_made_again_among_the_nodes_left(self):
        self.open_tree()
        self.script("document.getElementById('search-input').value = 'c'; TreeView.runSearch('c'); return 1")
        self.assertEqual(self.state("TreeView.state.matches.length"), 1)
        self.hide([2])
        self.assertEqual(self.state("TreeView.state.matches.length"), 0)
        self.hide([])
        self.assertEqual(self.state("Array.from(TreeView.state.matches)"), [2])

    def test_the_legend_counts_are_of_the_nodes_left(self):
        self.open_tree()
        self.hide([0, 1])
        counts = self.state("TreeView.state.colors.legend.map(e => [e.cls, e.count])")
        self.assertEqual(counts[0], [1, 2])  # two of the four red leaves are left
        self.assertEqual(self.state("document.querySelector('#legend-list .legend-row .count').textContent"), "2")

    def test_the_summary_says_how_many_are_left_out(self):
        self.open_tree()
        self.assertNotIn("hidden", self.state("document.getElementById('summary').textContent"))
        self.hide([0, 1])
        self.assertIn("2 hidden in the network, left out", self.state("document.getElementById('summary').textContent"))
        self.hide([])
        self.assertNotIn("hidden", self.state("document.getElementById('summary').textContent"))

    def test_showing_them_again_gives_the_whole_tree_back_exactly(self):
        self.open_tree()
        before = self.state("""(() => { const L = TreeView.state.layout; return [Array.from(L.x), Array.from(L.y), L.live]; })()""")
        self.hide([0, 3, 5])
        self.assertNotEqual(self.state("Array.from(TreeView.state.layout.y)"), before[1])
        self.hide([])
        after = self.state("""(() => { const L = TreeView.state.layout; return [Array.from(L.x), Array.from(L.y), L.live]; })()""")
        self.assertEqual(after, before)
        self.assertIsNone(after[2])

    def test_every_node_hidden_shows_the_whole_tree_faded_as_before(self):
        self.open_tree()
        self.hide(range(8))
        found = self.spied("""
            const s = TreeView.state, faded = TreeView.rgbOf(s.palette.faded);
            return {live: s.layout.live, removed: s.removed, faded: drawn.filter(e => e.main && e.parts.length && same(e.style, faded)).length > 0};""")
        self.assertEqual((found["live"], found["removed"], found["faded"]), (None, None, True))

    def test_a_view_that_shows_all_of_the_tree_shows_all_of_what_is_left(self):
        self.open_tree()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.hide([])
                self.choose(kind)
                self.hide([0, 1, 2, 3])
                found = self.state("""(() => { const s = TreeView.state;
                    return {fitted: s.fitted, view: s.view, fit: s.fit, bounds: s.layout.bounds}; })()""")
                self.assertTrue(found["fitted"])
                self.assertEqual(found["view"], found["fit"])

    def test_a_view_that_was_moved_keeps_its_scale_and_the_leaf_at_its_middle_where_it_was(self):
        self.open_tree()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.hide([])
                self.choose(kind)
                before = self.state("""(() => { const s = TreeView.state, c = TreeView.canvas;
                    TreeView.focusLeaf(5); TreeView.zoomAbout(c.clientWidth / 2, c.clientHeight / 2, 1.5, 1.5); TreeView.drawNow();
                    const m = s.model, L = s.layout, v = s.view; let best = -1, near = Infinity;
                    for (let k = 0; k < m.leafCount; k++) { const n = m.leafNode[k], d = Math.hypot(v.ox + L.x[n] * v.sx - c.clientWidth / 2, v.oy + L.y[n] * v.sy - c.clientHeight / 2);
                        if (d < near) { near = d; best = k; } }
                    const n = m.leafNode[best];
                    return {leaf: best, at: [v.ox + L.x[n] * v.sx, v.oy + L.y[n] * v.sy], sx: v.sx, sy: v.sy, fitted: s.fitted}; })()""")
                self.assertFalse(before["fitted"])
                keep = [k for k in range(8) if k != 0 and k != 7]
                hidden = [k for k in range(8) if k not in keep]
                self.assertNotIn(before["leaf"], hidden)
                self.hide(hidden)
                after = self.state(f"""(() => {{ const s = TreeView.state, m = s.model, v = s.view, n = m.leafNode[{before["leaf"]}];
                    return {{at: [v.ox + s.layout.x[n] * v.sx, v.oy + s.layout.y[n] * v.sy], sx: v.sx, sy: v.sy, fitted: s.fitted}}; }})()""")
                self.assertFalse(after["fitted"])
                self.assertAlmostEqual(after["at"][0], before["at"][0], places=4)
                self.assertAlmostEqual(after["at"][1], before["at"][1], places=4)
                self.assertAlmostEqual(after["sx"] / before["sx"], 1, delta=0.5 if kind == "rectangular" else 1e-9)

    def test_folding_search_selection_the_map_and_the_ruler_work_on_the_tree_left(self):
        self.open_tree()
        self.hide([1])
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                self.select([0, 2])
                self.script("TreeView.openAll(); TreeView.toggleFold(6); TreeView.drawNow(); return 1")
                found = self.state("""(() => { const s = TreeView.state, L = s.layout, out = new Float64Array(8);
                    L.wedge(6, out); const rows = kind => 0;
                    return {folded: s.collapsed[6], selected: Array.from(s.selection.leaves), clade: s.selection.clade, wedge: Array.from(out),
                            hiddenLeaves: s.leafHidden ? Array.from(s.leafHidden) : null}; })()""")
                self.assertEqual((found["folded"], found["selected"]), (1, [0, 2]))
                if kind == "rectangular":
                    # (abcd) now holds three rows, a, c, d, from row 0 to 3.
                    self.assertEqual((found["wedge"][4], found["wedge"][6]), (0, 3))
                self.assertEqual(self.console_errors(), [])
                self.script("TreeView.openAll(); return 1")

    def test_the_midpoint_root_note_and_the_rooting_wording_stay(self):
        self.open_tree(text=EIGHT_LEAVES.replace("[&R]", "[&U]"))
        self.script("const select = document.getElementById('layout-select'); select.value = 'rectangular'; select.dispatchEvent(new Event('change')); return 1")
        self.hide([0, 7])
        found = self.state("""({banner: Array.from(document.querySelectorAll('#warning-list li.rooting-note')).map(li => li.textContent),
            details: document.getElementById('details-rooting').textContent, kind: TreeView.state.kind})""")
        self.assertEqual(found["banner"], ["Drawn from the midpoint root; the tree itself is unrooted."])
        self.assertEqual(found["details"], "Drawn from the midpoint root; the tree itself is unrooted.")

    def test_the_overview_map_is_of_the_tree_left(self):
        self.open_tree()
        self.hide([4, 5, 6, 7])
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.state("""(() => { TreeView.drawNow(); const s = TreeView.state, v = TreeView.minimapView;
                    return {bounds: s.layout.bounds, view: v}; })()""")
                self.assertGreater(found["view"]["right"], found["view"]["left"])
                self.assertEqual(self.console_errors(), [])

    def test_the_whole_cycle_writes_nothing_to_the_console(self):
        self.open_tree(edges=ConnectedTests.EDGES)
        for kind in KINDS:
            self.choose(kind)
            self.select([0, 1], neighbours=3)
            for hidden in ([2], [0, 4], range(8), []):
                self.hide(hidden)
                self.script("TreeView.runSearch('c'); TreeView.drawNow(); return 1")
        self.assertEqual(self.console_errors(), [])


class RelayoutCostTests(AppearanceCase):
    """What leaving nodes out costs a tree of 44,000 leaves, which is what a command that hides nodes makes the page pay."""

    def test_hiding_and_showing_forty_four_thousand_leaves_is_fast_in_every_layout(self):
        count = 44000
        headers = headers_of(count)
        self.serve_headers(headers, colors=node_colors(count))
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        for kind in KINDS:
            with self.subTest(kind=kind):
                found = self.script(f"""
                    const s = TreeView.state, m = s.model;
                    TreeView.setLayout({json.dumps(kind)}, 'lengths');
                    const removed = new Uint8Array(m.leafCount);
                    for (let k = 0; k < m.leafCount; k++) if (k % 3 === 0) removed[k] = 1;
                    const t0 = performance.now(); TreeView.applyRemoved(removed); const hide = performance.now() - t0;
                    const frame0 = performance.now(); TreeView.drawNow(); const frame = performance.now() - frame0;
                    const t1 = performance.now(); TreeView.applyRemoved(null); const show = performance.now() - t1;
                    return {{hide, show, frame, live: removed.length}};""")
                if __import__("os").environ.get("TREE_PERFORMANCE_FILE"):
                    with open(__import__("os").environ["TREE_PERFORMANCE_FILE"], "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"case": "hide a third", "kind": kind, **found}) + "\n")
                self.assertLess(found["hide"], 1500)
                self.assertLess(found["show"], 1500)


if __name__ == "__main__":
    unittest.main()
