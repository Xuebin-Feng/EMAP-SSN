# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's page: the colours of the network, the legend and its key, the notes, text and safety, and fitting names.

Opened offscreen over the real web server like the rest (tests/tree_page_harness.py); the first half of the page's
tests is test_tree_page.py, the pass-through and large-tree half test_tree_detail_page.py. The helpers are in
tests/tree_page_helpers.py."""

import json
import os
import pathlib
import re
import sys
import time
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for folder in (SRC_DIR, PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from PySide6.QtCore import QUrl, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from tests.tree_page_harness import TreePageMixin, http_get, page_headers  # noqa: E402
from tests.tree_fixtures import (  # noqa: E402
    ALIGNMENT_NAME,
    headers_of,
    make_alignment_folder,
    make_viewer,
    node_colors,
)
from tests.tree_page_helpers import (  # noqa: E402
    BASE_PAINT,
    BLUE,
    EIGHT_LEAVES,
    LEGEND_NOT_BOOTSTRAP,
    LayoutCase,
    NO_CLASS,
    PAGE_FILES,
    RED,
    SPY,
    WEB_DIR,
    nj_options,
    write_tree_file,
)


class ColorCase(TreePageMixin, unittest.TestCase):
    """A page over the eight-leaf tree above, whose network the Viewer has coloured."""

    def open_colored(self, painted=None, *, hidden=(), clusters=None, network=None, text=EIGHT_LEAVES, key=None):
        headers = list("abcdefgh")
        folder = self.make_folder()
        write_tree_file(folder, headers, text)
        count = len(network or headers)
        self.serve_headers(headers, folder=folder, tree=None, network=network,
                           colors=node_colors(count, BASE_PAINT if painted is None else painted), hidden=hidden, clusters=clusters, key=key)
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
                    unlabelled: c.unlabelled.map(e => ({cls: e.cls, count: e.count, css: e.css, hex: e.hex})), css: c.css}; })()""")

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

    def test_plain_is_drawn_but_is_no_choice_so_the_unlabelled_colours_leave_it_out(self):
        self.open_colored()
        found = self.info()
        self.assertEqual([entry["cls"] for entry in found["unlabelled"]], [1, 2, 3])
        self.assertEqual([entry["count"] for entry in found["unlabelled"]], [4, 2, 1])
        # Nothing coloured: a tree in the plain colour, and no legend.
        self.open_colored({})
        self.assertEqual(self.info()["unlabelled"], [])
        self.assertEqual(self.state("document.getElementById('legend').classList.contains('shown')"), False)

    def test_hidden_nodes_and_nodes_outside_the_network_have_no_colour_and_do_not_count(self):
        self.open_colored(hidden=[2])
        found = self.info()
        self.assertEqual(found["leafClass"][2], NO_CLASS)
        # (cd) holds one leaf that is shown, which is red: the clade is red.
        self.assertEqual((found["present"][5], found["nodeClass"][5]), (1, 1))
        self.assertEqual(found["unlabelled"][0]["count"], 3)
        # A leaf that is not a node of the network has no colour either; (gh) is then green all through.
        self.open_colored(network=list("abcdefg"))
        found = self.info()
        self.assertEqual(found["leafClass"][7], NO_CLASS)
        self.assertEqual((found["present"][12], found["nodeClass"][12]), (1, 3))

    def test_a_clade_too_small_to_draw_takes_the_colour_most_of_its_shown_leaves_have(self):
        tops = """(() => { const c = TreeView.state.colors, m = TreeView.state.model;
            return Array.from({length: m.nodeCount}, (_, i) => TreeView.cladeTop(c, m, i)); })()"""
        self.open_colored()
        # (efgh) is half blue, a quarter green and a quarter plain: blue, though no colour has more than half of it; the
        # root, half red, is red. (gh), one green leaf and one plain, is a tie: the lower class, plain, as cladeMix has it.
        self.assertEqual(self.state(tops), [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 0, 0, 2, 1])
        # Counted once and kept: the same again.
        self.assertEqual(self.state(tops), [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 0, 0, 2, 1])
        # A clade none of whose leaves is a node of the network has no colour; the leaves that are not do not count.
        self.open_colored({0: RED, 1: RED, 2: RED, 3: RED, 4: BLUE, 5: BLUE}, network=list("abcdef"))
        found = self.state(tops)
        self.assertEqual(found[10:13], [NO_CLASS, NO_CLASS, NO_CLASS])
        self.assertEqual((found[13], found[14]), (2, 1))

    def test_colours_are_compared_in_classes_of_sixteen_levels_so_that_a_gradient_forms_clades(self):
        self.open_colored({0: "#fa0000", 1: "#ff0000", 2: "#e60000", 3: "#f10000"})
        found = self.info()
        self.assertEqual(found["leafClass"][:4], [1, 1, 2, 1])
        # A class is drawn in the first colour met in it, exactly.
        self.assertEqual(found["css"][1], "rgb(250, 0, 0)")
        self.assertEqual(self.state("TreeView.state.colors.unlabelled[0].hex"), "#fa0000")

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

    def test_a_colour_is_never_called_a_cluster_by_a_guess_it_is_its_hex_unless_the_colour_key_says_more(self):
        # Every leaf of a colour is in one cluster, and the legend still says only the colour: the Viewer's key is what names it.
        self.open_colored(clusters=[0, 0, 0, 0, 1, 1, 2, -1])
        names = self.state("TreeView.state.colors.unlabelled.map(e => TreeView.classLabel(e.cls))")
        self.assertEqual(names, ["#ff0000", "#0000ff", "#00ff00"])
        rows = self.state("[...document.querySelectorAll('#legend-list .legend-row .name')].map(e => e.textContent)")
        self.assertEqual(rows, ["#ff0000", "#0000ff", "#00ff00"])
        self.assertEqual(self.state("typeof TreeView.clusterOfClass"), "undefined")

    def test_the_legend_lists_every_colour_not_twelve(self):
        headers = headers_of(30)
        painted = {n: "#%02x%02x%02x" % (8 + 8 * n, 255 - 8 * n, (n * 37) % 256) for n in range(30)}
        self.serve_headers(headers, colors=node_colors(30, painted))
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        info = self.state("""({rows: document.querySelectorAll('#legend-list .legend-row').length,
                               more: document.getElementById('legend').textContent.includes('more colours'),
                               counts: TreeView.state.colors.unlabelled.map(e => e.count)})""")
        self.assertEqual(info["rows"], 30)
        self.assertFalse(info["more"])
        self.assertEqual(info["counts"], sorted(info["counts"], reverse=True))

    def test_a_viewer_with_no_colours_leaves_the_tree_in_its_own(self):
        self.serve_headers(headers_of(20))
        self.open_page()
        self.pump(0.5)
        self.assertEqual(self.state("TreeView.state.colors"), None)
        self.assertEqual(self.state("document.getElementById('legend').classList.contains('shown')"), False)
        self.assertEqual(self.console_errors(), [])


def cluster_key(labels=("Cluster 1", "Cluster 2"), title="Clusters"):
    """A key for the eight-leaf network: its first four nodes are the cluster labels[0] and the next two labels[1]."""
    from desktop import Node_Colors

    def declare(viewer):
        red, blue = tuple(float(v) for v in viewer.current_colors[0]), tuple(float(v) for v in viewer.current_colors[4])
        Node_Colors.declare_categories(viewer, title, [(labels[0], red), (labels[1], blue)], [0, 0, 0, 0, 1, 1], nodes=range(6))

    return declare


def gradient_key(values=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)):
    """A key for the eight-leaf network: a viridis gradient on its first six nodes, the seventh with no value."""
    import matplotlib as mpl

    from desktop import Node_Colors

    def declare(viewer):
        cmap = mpl.colormaps["viridis"]
        low, high = min(values), max(values)
        viewer.current_colors[:6] = cmap((np.asarray(values) - low) / (high - low))
        gray = (0.8, 0.8, 0.8, 1.0)
        viewer.current_colors[6] = gray
        Node_Colors.declare_gradient(viewer, "Property length", "viridis", cmap, low, high, range(6), values,
                                     extras=[("No value", gray, [6])])

    return declare


class KeyLegendTests(ColorCase):
    """The legend reads the Viewer's colour key: its layers, newest first, then the colours it does not cover."""

    def repaint(self):
        """What the Viewer does when it draws its nodes again: tell its listeners."""
        for listener in list(self.viewer.node_update_listeners):
            listener()

    def titles(self):
        return self.state("[...document.querySelectorAll('#legend-list .legend-title')].map(e => e.textContent)")

    def rows(self):
        return self.state("""[...document.querySelectorAll('#legend-list .legend-row')].map(b => ({name: b.querySelector('.name').textContent,
            count: b.querySelector('.count').textContent, pressed: b.getAttribute('aria-pressed')}))""")

    def click_row(self, index):
        self.js(f"document.querySelectorAll('#legend-list .legend-row')[{index}].click()")

    def matches(self):
        return self.state("Array.from(TreeView.state.matches)")

    def test_a_layer_of_categories_is_a_titled_group_of_rows_with_their_leaves_and_the_rest_are_unlabelled(self):
        self.open_colored(key=cluster_key())
        self.assertEqual(self.titles(), ["Clusters", "Unlabelled colours"])
        self.assertEqual(self.rows(), [
            {"name": "Cluster 1", "count": "4", "pressed": "false"},
            {"name": "Cluster 2", "count": "2", "pressed": "false"},
            {"name": "#00ff00", "count": "1", "pressed": "false"},
        ])

    def test_two_categories_that_share_a_colour_are_two_rows_that_each_mark_their_own_leaves(self):
        from desktop import Node_Colors

        def declare(viewer):
            red = tuple(float(v) for v in viewer.current_colors[0])
            Node_Colors.declare_categories(viewer, "Clusters", [("Cluster 1", red), ("Cluster 2", red)], [0, 0, 1, 1], nodes=range(4))

        self.open_colored(key=declare)
        self.assertEqual([row["name"] for row in self.rows()][:2], ["Cluster 1", "Cluster 2"])
        self.click_row(1)
        self.assertEqual(self.matches(), [2, 3])
        self.assertEqual([row["pressed"] for row in self.rows()][:2], ["false", "true"])
        self.click_row(0)
        self.assertEqual(self.matches(), [0, 1])
        self.click_row(0)
        self.assertEqual(self.matches(), [])

    def test_an_unlabelled_colour_marks_only_its_unlabelled_leaves(self):
        # Blue is both a labelled cluster (e, f) and, once g is painted blue without a key, an unlabelled colour.
        self.open_colored(key=cluster_key())
        self.viewer.current_colors[6] = self.viewer.current_colors[4]
        version = self.info()["version"]
        self.repaint()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        names = [row["name"] for row in self.rows()]
        self.assertEqual(names, ["Cluster 1", "Cluster 2", "#0000ff"])
        self.click_row(2)
        self.assertEqual(self.matches(), [6])

    def test_nodes_repainted_without_a_declaration_move_to_unlabelled_colours(self):
        self.open_colored(key=cluster_key())
        self.viewer.current_colors[2] = (1.0, 0.5, 0.0, 1.0)
        version = self.info()["version"]
        self.repaint()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        rows = self.rows()
        self.assertEqual([(row["name"], row["count"]) for row in rows],
                         [("Cluster 1", "3"), ("Cluster 2", "2"), ("#ff8000", "1"), ("#00ff00", "1")])

    def test_a_key_that_comes_after_the_colours_replaces_the_unlabelled_rows_and_one_that_goes_brings_them_back(self):
        self.open_colored()
        self.assertEqual(self.titles(), ["Unlabelled colours"])
        from desktop import Node_Colors

        cluster_key()(self.viewer)
        self.repaint()
        self.wait_for("TreeView.state.colors.layers.length === 1")
        self.assertEqual(self.titles(), ["Clusters", "Unlabelled colours"])
        Node_Colors.clear_key(self.viewer)
        self.repaint()
        self.wait_for("TreeView.state.colors.layers.length === 0")
        self.assertEqual(self.titles(), ["Unlabelled colours"])
        self.assertEqual(len(self.rows()), 3)

    def test_a_key_that_only_means_something_else_refreshes_the_page(self):
        self.open_colored(key=cluster_key())
        version = self.info()["version"]
        self.viewer.color_key_layers[0]["title"] = "Subclusters of cluster 3"
        self.repaint()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        self.assertEqual(self.titles()[0], "Subclusters of cluster 3")

    def test_the_newest_layer_is_listed_first(self):
        from desktop import Node_Colors

        def declare(viewer):
            cluster_key()(viewer)
            green = tuple(float(v) for v in viewer.current_colors[6])
            Node_Colors.declare_categories(viewer, "Colors", [("P1 \u2192 green", green)], [0], nodes=[6])

        self.open_colored(key=declare)
        self.assertEqual(self.titles(), ["Colors", "Clusters"])
        self.assertEqual([row["name"] for row in self.rows()], ["P1 \u2192 green", "Cluster 1", "Cluster 2"])

    def test_a_layer_of_categories_with_more_than_twelve_rows_lists_them_all_in_a_scrolling_area(self):
        from desktop import Node_Colors

        count = 40
        headers = headers_of(count)
        painted = {n: "#%02x%02x%02x" % (8 + 6 * n, 250 - 6 * n, (n * 37) % 256) for n in range(count)}

        def declare(viewer):
            Node_Colors.declare_categories(
                viewer, "Clusters", [(f"Cluster {n}", tuple(float(v) for v in viewer.current_colors[n])) for n in range(count)],
                list(range(count)))

        self.serve_headers(headers, colors=node_colors(count, painted), key=declare)
        self.open_page()
        self.wait_for("TreeView.state.colors !== null")
        info = self.state("""(() => { const list = document.getElementById('legend-list');
            return {rows: list.querySelectorAll('.legend-row').length, scrolls: list.scrollHeight > list.clientHeight,
                    overflow: getComputedStyle(list).overflowY, last: [...list.querySelectorAll('.name')].pop().textContent}; })()""")
        self.assertEqual((info["rows"], info["overflow"], info["last"]), (count, "auto", "Cluster 39"))
        self.assertTrue(info["scrolls"])

    def test_titles_and_labels_are_text_never_markup(self):
        hostile = ("<b>bold</b>", "<img src=x onerror=window.__pwned=1>")
        self.open_colored(key=cluster_key(labels=hostile, title="<i>Clusters</i><script>window.__pwned=1</script>"))
        self.pump(0.3)
        info = self.state("""({names: [...document.querySelectorAll('#legend-list .name')].map(e => e.textContent).slice(0, 2),
            title: document.querySelector('#legend-list .legend-title').textContent,
            markup: document.querySelectorAll('#legend-list b, #legend-list i, #legend-list img, #legend-list script').length,
            pwned: typeof window.__pwned})""")
        self.assertEqual(info["names"], list(hostile))
        self.assertEqual(info["title"], "<i>Clusters</i><script>window.__pwned=1</script>")
        self.assertEqual((info["markup"], info["pwned"]), (0, "undefined"))
        self.assertEqual(self.console_errors(), [])

    def test_a_label_that_is_not_text_is_empty_and_a_malformed_key_is_no_key(self):
        self.open_colored(key=cluster_key())
        payload = self.script("""const p = await (await fetch('/api/tree/colors')).json();
            p.key.layers[0].categories[0].label = {x: 1}; p.key.layers[0].title = 7; p.key.layers[0].categories[1].rgba = 'bad';
            const colors = TreeView.decodeColors(p, TreeView.state.model);
            const broken = JSON.parse(JSON.stringify(p)); broken.key.owner = 'AAAA';
            const none = TreeView.decodeColors(broken, TreeView.state.model);
            return {rows: colors.layers[0].rows.map(r => [r.label, r.css]), title: colors.layers[0].title,
                    layers: none.layers.length, unlabelled: none.unlabelled.length};""")
        self.assertEqual(payload["rows"], [["", "rgb(255, 0, 0)"], ["Cluster 2", "rgb(0, 0, 0)"]])
        self.assertEqual((payload["title"], payload["layers"], payload["unlabelled"]), ("", 0, 3))

    def test_spectrum_colours_are_never_called_clusters(self):
        # Every leaf of a colour is in one cluster, as after a spectrum that follows a clustering: no row says so.
        self.open_colored(clusters=[0, 0, 0, 0, 1, 1, 2, -1], key=gradient_key())
        names = self.state("[...document.querySelectorAll('#legend-list .name, #legend-list .legend-title')].map(e => e.textContent)")
        self.assertFalse([name for name in names if name.startswith("Cluster")], names)
        self.assertEqual(names[0], "Property length")

    def gradient_info(self):
        return self.state("""(() => { const bar = document.querySelector('#legend-list .legend-bar');
            const ends = [...document.querySelectorAll('#legend-list .legend-ends span')].map(e => e.textContent);
            const range = document.querySelector('#legend-list .legend-range');
            return {title: document.querySelector('#legend-list .legend-title').textContent,
                    scheme: document.querySelector('#legend-list .legend-scheme').textContent, ends,
                    background: bar.style.backgroundImage.startsWith('linear-gradient'), range: range.hidden ? '' : range.textContent,
                    band: !document.querySelector('#legend-list .legend-band').hidden}; })()""")

    def drag(self, start, end, release=True):
        """Press on the bar at the share ``start`` of its width, move to ``end`` (None for no move) and let go (or not)."""
        steps = [("pointerdown", start)]
        if end is not None:
            steps.append(("pointermove", end))
        if release:
            steps.append(("pointerup", start if end is None else end))
        return self.script("""const bar = document.querySelector('#legend-list .legend-bar'), box = bar.getBoundingClientRect();
            for (const [type, share] of %s) {
                bar.dispatchEvent(new PointerEvent(type, {bubbles: true, clientX: box.left + box.width * share, clientY: box.top + 5, button: 0, pointerId: 7}));
            }
            return 1""" % json.dumps(steps))

    def test_a_gradient_is_a_bar_with_its_title_scheme_minimum_and_maximum_and_its_extras_as_rows(self):
        self.open_colored(key=gradient_key())
        info = self.gradient_info()
        self.assertEqual((info["title"], info["scheme"], info["ends"], info["background"]), ("Property length", "viridis", ["0", "5"], True))
        self.assertEqual((info["range"], info["band"]), ("", False))
        self.assertEqual(self.rows(), [{"name": "No value", "count": "1", "pressed": "false"}])
        self.click_row(0)
        self.assertEqual(self.matches(), [6])

    def test_dragging_on_the_bar_marks_the_leaves_whose_value_is_in_the_range_and_shows_it_while_dragging(self):
        self.open_colored(key=gradient_key())
        # From 18% to 62% of 0 to 5 is 0.9 to 3.1: the values 1, 2 and 3.
        self.drag(0.18, 0.62, release=False)
        self.assertEqual(self.matches(), [1, 2, 3])
        info = self.gradient_info()
        self.assertEqual((info["range"], info["band"]), ("0.9 to 3.1", True))
        self.assertEqual(self.state("TreeView.state.legendMark"), "range:0")
        self.drag(0.18, 0.62)  # letting go leaves the marks
        self.assertEqual(self.matches(), [1, 2, 3])
        # Dragging back to the right edge and over the left edge covers the whole bar.
        self.drag(0.5, 1.4)
        self.assertEqual(self.matches(), [3, 4, 5])
        self.drag(0.95, -0.5)
        self.assertEqual(self.matches(), [0, 1, 2, 3, 4])

    def test_a_press_on_the_bar_covers_a_narrow_range_round_the_point_and_the_clear_button_takes_it_off(self):
        self.open_colored(key=gradient_key())
        self.drag(0.4, None)  # value 2.0, in 1.84 to 2.16
        self.assertEqual(self.matches(), [2])
        self.drag(0.5, None)  # value 2.5: between two values
        self.assertEqual(self.matches(), [])
        self.assertEqual(self.state("TreeView.state.legendMark"), "range:0")
        self.js("document.querySelector('#legend-list .legend-clear').click()")
        self.assertEqual((self.matches(), self.state("TreeView.state.legendMark")), ([], None))
        self.assertEqual(self.gradient_info()["band"], False)

    def test_the_bar_answers_the_arrow_keys_and_escape(self):
        self.open_colored(key=gradient_key())
        key = lambda name: self.script(f"""const bar = document.querySelector('#legend-list .legend-bar'); bar.focus();
            bar.dispatchEvent(new KeyboardEvent('keydown', {{key: '{name}', bubbles: true, cancelable: true}})); return 1""")
        key("Home")  # the first of sixteen bands: 0 to 0.3125
        self.assertEqual(self.matches(), [0])
        key("End")  # the last: 4.6875 to 5
        self.assertEqual(self.matches(), [5])
        key("ArrowLeft")
        self.assertEqual(self.matches(), [])
        key("Escape")
        self.assertEqual(self.state("TreeView.state.legendMark"), None)
        self.assertEqual(self.console_errors(), [])

    def test_the_bar_is_a_slider_whose_value_and_text_say_the_range_from_the_first_step(self):
        self.open_colored(key=gradient_key())
        read = lambda: self.state("""(() => { const bar = document.querySelector('#legend-list .legend-bar');
            return ['role', 'aria-valuemin', 'aria-valuemax', 'aria-valuenow', 'aria-valuetext'].map(name => bar.getAttribute(name)); })()""")
        key = lambda name: self.script(f"""const bar = document.querySelector('#legend-list .legend-bar'); bar.focus();
            bar.dispatchEvent(new KeyboardEvent('keydown', {{key: '{name}', bubbles: true, cancelable: true}})); return 1""")
        # Before any use the range line is hidden, but the slider already says what it is and that nothing is marked.
        self.assertEqual(read(), ["slider", "0", "5", "0", "No range marked"])
        self.assertTrue(self.state("document.querySelector('#legend-list .legend-range').hidden"))
        key("ArrowRight")  # the first step of the bar is the first band, and is said at once
        self.assertEqual(read(), ["slider", "0", "5", "0", "0 to 0.3125"])
        key("ArrowRight")
        self.assertEqual(read(), ["slider", "0", "5", "0.3125", "0.3125 to 0.625"])
        key("End")
        self.assertEqual(read()[3:], ["4.6875", "4.688 to 5"])
        key("Escape")
        self.assertEqual(read(), ["slider", "0", "5", "0", "No range marked"])
        # A drag says its range too.
        self.drag(0.18, 0.62)
        self.assertEqual(read()[3:], ["0.9", "0.9 to 3.1"])
        self.assertEqual(self.state("document.querySelector('#legend-list .legend-range').hasAttribute('aria-live')"), False)

    def test_a_pressed_row_has_no_ring_and_a_label_cut_short_is_the_title_of_its_text(self):
        self.open_colored(key=cluster_key())
        before = self.state("getComputedStyle(document.querySelector('#legend-list .legend-row')).boxShadow")
        self.assertEqual(before, "none")
        self.click_row(0)
        found = self.state("""(() => { const row = document.querySelector('#legend-list .legend-row');
            return {shadow: getComputedStyle(row).boxShadow, pressed: row.getAttribute('aria-pressed'),
                    names: [...document.querySelectorAll('#legend-list .legend-row .name')].every(n => n.title === n.textContent),
                    titles: [...document.querySelectorAll('#legend-list .legend-title')].every(n => n.title === n.textContent),
                    action: row.title}; })()""")
        self.assertEqual(found["pressed"], "true")
        self.assertEqual(found["shadow"], "none")
        self.assertTrue(found["names"])
        self.assertTrue(found["titles"])
        self.assertEqual(found["action"], "Mark these leaves")

    def test_a_gradient_with_no_leaf_value_has_no_bar_and_a_layer_with_no_leaf_has_no_group(self):
        # Hide every node the gradient coloured: its bar and its rows have nothing to count.
        self.open_colored(key=gradient_key(), hidden=[0, 1, 2, 3, 4, 5])
        self.assertEqual(self.state("document.querySelectorAll('#legend-list .legend-bar').length"), 0)
        self.assertEqual(self.titles(), ["Property length"])
        self.open_colored(key=gradient_key(), hidden=[0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(self.titles(), [])

    def test_new_colours_take_the_marks_of_a_range_off(self):
        self.open_colored(key=gradient_key())
        self.drag(0.18, 0.62)
        version = self.info()["version"]
        self.viewer.current_colors[7] = (1.0, 0.0, 1.0, 1.0)
        self.repaint()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        self.assertEqual((self.matches(), self.state("TreeView.state.legendMark")), ([], None))
        self.assertEqual(self.gradient_info()["band"], False)


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

    # The tree at a size that makes (abcd) and (efgh) under three pixels, and the colours of their lines.
    SMALL_TREE = """
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
                mark: TreeView.rgbOf(s.palette.mark), branch: TreeView.rgbOf(s.palette.branch)};"""

    def test_a_clade_too_small_to_draw_is_one_line_in_the_colour_most_of_its_leaves_have(self):
        self.open_colored()
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied(self.SMALL_TREE)
                self.assertGreater(found["bars"], 0)
                self.assertFalse(found["rootTiny"])
                # (The clade's stem and its line are a pixel apart at this size, so each is found; both are in its colour.)
                self.assertEqual({tuple(style) for style in found["abcd"]}, {(255, 0, 0)})
                # (efgh) is half blue, a quarter green and a quarter plain: its line is blue, its stem the branch colour,
                # and nothing of it the grey that marks a leaf that is not in the network.
                self.assertIn([0, 0, 255], found["efgh"])
                self.assertEqual({tuple(style) for style in found["efgh"]} - {(0, 0, 255), tuple(found["branch"])}, set())

    def test_a_clade_too_small_to_draw_with_no_leaf_in_the_network_is_grey(self):
        self.open_colored({0: RED, 1: RED, 2: RED, 3: RED}, network=list("abcd"))
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind=kind):
                self.choose(kind)
                found = self.spied(self.SMALL_TREE)
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
        self.assertEqual([entry["count"] for entry in after["unlabelled"]], [3, 2, 1, 1])
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
        self.assertEqual(self.state("TreeView.state.colors.cluster[0]"), 0)
        version = self.info()["version"]
        self.viewer.cluster_labels = np.array([4, 4, 4, 4, 1, 1, 2, -1], dtype=np.int32)
        self.repaint()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        self.assertEqual(self.state("TreeView.state.colors.cluster[0]"), 4)

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
        self.assertEqual(self.state("TreeView.state.colors.unlabelled[0].count"), 2)
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


class UnmatchedLeavesTests(ColorCase):
    """The Unmatched leaves button: the leaves that are not nodes of the network drawn faded, or left out as hidden nodes are."""

    # g and h (leaves 6 and 7) are not nodes of this network.
    SIX = {0: RED, 1: RED, 2: RED, 3: RED, 4: BLUE, 5: BLUE}

    def button(self):
        return self.state("""(() => { const b = document.getElementById('unmatched-btn');
            return {hidden: b.hidden, disabled: b.disabled, pressed: b.getAttribute('aria-pressed'), text: b.textContent.trim()}; })()""")

    def press(self):
        self.script("document.getElementById('unmatched-btn').click(); return 1")

    def seen(self):
        """What the page shows of the leaves: which are laid out, what the warning line and the status line say."""
        return self.state("""(() => { const s = TreeView.state, m = s.model, live = s.layout.live;
            return {laidOut: Array.from({length: m.leafCount}, (_, k) => !live || !!live[m.leafNode[k]]),
                    out: s.unmatchedOut, banner: [...document.querySelectorAll('#warning-list li')].map(li => li.textContent).join(' | '),
                    summary: document.getElementById('summary').textContent}; })()""")

    def test_the_button_is_there_only_for_a_tree_with_leaves_in_the_network_and_leaves_outside_it(self):
        self.open_colored()
        self.assertTrue(self.button()["hidden"])
        self.open_colored(self.SIX, network=list("abcdef"))
        self.assertEqual(self.button(), {"hidden": False, "disabled": False, "pressed": "true", "text": "Unmatched leaves"})
        # Beside Overview, after it.
        self.assertEqual(self.state("document.getElementById('overview-btn').nextElementSibling.id"), "unmatched-btn")

    def test_pressing_it_leaves_them_out_and_pressing_again_draws_them_faded(self):
        self.open_colored(self.SIX, network=list("abcdef"))
        found = self.seen()
        self.assertEqual(found["laidOut"], [True] * 8)
        self.assertIn("They are drawn faded.", found["banner"])
        self.press()
        found = self.seen()
        self.assertEqual(self.button()["pressed"], "false")
        self.assertEqual(found["laidOut"], [True] * 6 + [False, False])
        self.assertTrue(found["out"])
        self.assertIn("Leaves that are not nodes of this network: 2. They are left out.", found["banner"])
        self.assertIn("2 not in this network, left out", found["summary"])
        # The whole of what is left is in view, as for hidden nodes.
        self.assertTrue(self.state("TreeView.state.fitted"))
        self.press()
        found = self.seen()
        self.assertEqual(self.button()["pressed"], "true")
        self.assertEqual(found["laidOut"], [True] * 8)
        self.assertFalse(found["out"])
        self.assertIn("They are drawn faded.", found["banner"])
        self.assertNotIn("not in this network, left out", found["summary"])
        self.assertEqual(self.console_errors(), [])

    def test_hidden_nodes_and_unmatched_leaves_are_left_out_together_and_told_apart(self):
        self.open_colored(self.SIX, network=list("abcdef"), hidden=[2])
        self.assertEqual(self.seen()["laidOut"], [True, True, False, True, True, True, True, True])
        self.press()
        self.assertEqual(self.seen()["laidOut"], [True, True, False, True, True, True, False, False])
        # The Viewer's colours change: both stay out.
        self.assertTrue(self.script("return await TreeView.loadColors()"))
        self.assertEqual(self.seen()["laidOut"], [True, True, False, True, True, True, False, False])
        lines = self.state("TreeView.describeNode(TreeView.state.model.root).lines")
        self.assertIn("Leaves in this clade: 5", lines)
        self.assertIn("Of them in this network: 5", lines)
        self.assertIn("Hidden in the network: 1", lines)
        self.assertIn("Not in this network, left out: 2", lines)
        summary = self.seen()["summary"]
        self.assertIn("1 hidden in the network, left out", summary)
        self.assertIn("2 not in this network, left out", summary)

    def test_the_choice_holds_when_the_tree_is_read_again(self):
        self.open_colored(self.SIX, network=list("abcdef"))
        self.press()
        self.assertTrue(self.script("return await TreeView.loadTree()"))
        self.wait_for("TreeView.state.colors !== null")
        self.assertEqual(self.button()["pressed"], "false")
        self.assertEqual(self.seen()["laidOut"], [True] * 6 + [False, False])
        self.assertIn("They are left out.", self.seen()["banner"])


class LegendTests(ColorCase):
    def rows(self):
        return self.state("""[...document.querySelectorAll('#legend-list .legend-row')].map(b => ({name: b.querySelector('.name').textContent,
            count: b.querySelector('.count').textContent, pressed: b.getAttribute('aria-pressed'), swatch: b.querySelector('.swatch').style.backgroundColor}))""")

    def test_colours_with_no_key_are_listed_under_unlabelled_colours_by_hex_with_their_number_of_leaves(self):
        self.open_colored()
        self.assertEqual(self.rows(), [
            {"name": "#ff0000", "count": "4", "pressed": "false", "swatch": "rgb(255, 0, 0)"},
            {"name": "#0000ff", "count": "2", "pressed": "false", "swatch": "rgb(0, 0, 255)"},
            {"name": "#00ff00", "count": "1", "pressed": "false", "swatch": "rgb(0, 255, 0)"},
        ])
        self.assertEqual(self.state("[...document.querySelectorAll('#legend-list .legend-title')].map(e => e.textContent)"),
                         ["Unlabelled colours"])
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
        self.assertEqual(self.state("[Array.from(TreeView.state.matches), TreeView.state.legendMark]"), [[], None])

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

    def test_a_leaf_says_its_cluster_and_a_clade_says_the_colour_most_of_its_leaves_have_in_hex(self):
        self.open_colored(clusters=[0, 0, 0, 0, 1, 1, 2, -1])
        self.choose("rectangular")
        leaf = self.hover_text(0)
        self.assertIn("Cluster: 0", leaf)
        clade = self.hover_text(6)
        self.assertIn("Colour of 100% of its leaves: #ff0000", clade)

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



    def says(self, node):
        return [line for line in self.hover_text(node) if "its leaves" in line]

    def categories(self, title, labels, slots, nodes):
        from desktop import Node_Colors

        def declare(viewer):
            red = tuple(float(v) for v in viewer.current_colors[0])
            Node_Colors.declare_categories(viewer, title, [(label, red) for label in labels], slots, nodes=nodes)

        return declare

    def test_two_clusters_that_share_one_colour_are_told_apart_by_the_clade_hover(self):
        self.open_colored(key=self.categories("Clusters", ["Cluster 1", "Cluster 2"], [0, 0, 0, 1], range(4)))
        self.choose("rectangular")
        self.assertEqual(self.says(6), ["Colour of 75% of its leaves: Clusters: Cluster 1"])
        self.assertEqual(self.says(2), ["Colour of 100% of its leaves: Clusters: Cluster 1"])

    def test_a_layer_of_residues_names_the_layer_and_the_residue(self):
        self.open_colored(key=self.categories("Residue at 120", ["K", "R"], [1, 1, 1, 0], range(4)))
        self.choose("rectangular")
        self.assertEqual(self.says(6), ["Colour of 75% of its leaves: Residue at 120: R"])

    def test_a_gradient_clade_says_the_layer_and_the_range_of_its_values(self):
        self.open_colored(key=gradient_key((0.42, 0.5, 0.61, 0.87, 4.0, 5.0)))
        self.choose("rectangular")
        self.assertEqual(self.says(6), ["Property length: 0.42 to 0.87 across 100% of its leaves"])

    def test_an_unlabelled_colour_keeps_its_hex_beside_a_labelled_clade(self):
        self.open_colored(key=cluster_key())
        self.viewer.current_colors[6] = self.viewer.current_colors[4]
        self.viewer.current_colors[7] = self.viewer.current_colors[4]
        version = self.info()["version"]
        for listener in list(self.viewer.node_update_listeners):
            listener()
        self.wait_for(f"TreeView.state.colors.version !== {version}")
        self.choose("rectangular")
        self.assertEqual(self.says(12), ["Colour of 100% of its leaves: #0000ff"])
        self.assertEqual(self.says(9), ["Colour of 100% of its leaves: Clusters: Cluster 2"])

    def test_a_clade_where_no_group_has_most_of_the_leaves_says_nothing(self):
        self.open_colored(key=self.categories("Clusters", ["Cluster 1", "Cluster 2"], [0, 0, 1, 1], range(4)))
        self.choose("rectangular")
        self.assertEqual(self.says(6), [])



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

    # The texts a frame draws, but the scale bar's length, which is no name.
    FILLTEXT = """(() => {
        const proto = CanvasRenderingContext2D.prototype, original = proto.fillText, bar = TreeView.drawScaleBar;
        let count = 0, inBar = false;
        TreeView.drawScaleBar = function () { inBar = true; try { return bar.apply(this, arguments); } finally { inBar = false; } };
        proto.fillText = function () { if (!inBar) count++; return original.apply(this, arguments); };
        try { TreeView.drawNow(); } finally { proto.fillText = original; TreeView.drawScaleBar = bar; }
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
    # The scale bar's length is no name.
    NAME_BOXES = """(() => {
        const proto = CanvasRenderingContext2D.prototype, original = proto.fillText, boxes = [], bar = TreeView.drawScaleBar;
        let inBar = false;
        TreeView.drawScaleBar = function () { inBar = true; try { return bar.apply(this, arguments); } finally { inBar = false; } };
        proto.fillText = function (text, x, y) {
            if (inBar) return original.apply(this, arguments);
            const m = this.getTransform(), width = this.measureText(text).width;
            const size = parseFloat(this.font), left = this.textAlign === 'right' ? x - width : x;
            const ratio = TreeView.canvas.width / TreeView.canvas.clientWidth;
            const corners = [[left, y - size * 0.6], [left + width, y - size * 0.6], [left, y + size * 0.6], [left + width, y + size * 0.6]]
                .map(([px, py]) => [(m.a * px + m.c * py + m.e) / ratio, (m.b * px + m.d * py + m.f) / ratio]);
            boxes.push({text, x0: Math.min(...corners.map(c => c[0])), x1: Math.max(...corners.map(c => c[0])),
                        y0: Math.min(...corners.map(c => c[1])), y1: Math.max(...corners.map(c => c[1]))});
            return original.apply(this, arguments);
        };
        try { TreeView.drawNow(); } finally { proto.fillText = original; TreeView.drawScaleBar = bar; }
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
