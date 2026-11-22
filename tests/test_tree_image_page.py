# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's scale bar and Save an image (tree_render.js TV.scaleBar, tree_image.js): the tree as it is in view, as a PNG or an SVG.

The scale bar is drawn at the foot of the drawing beside the overview map, a round length that follows the zoom; a
cladogram has none. The camera button between Fit and Details opens a card like Details: the format, the size, and what
goes in the band under the tree. The image is the page's own drawing in the light theme's colours, without the overview
map, the strip at the right edge or the pointer's highlight. The page is opened offscreen over the real web server, on a
tree file of 40 leaves in two coloured clusters, with branch lengths and support.
"""

import json
import pathlib
import random
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

from tests.tree_fixtures import give_edges, node_colors  # noqa: E402
from tests.tree_page_harness import TreePageMixin  # noqa: E402
from web_ui import tree_backend  # noqa: E402

LEAVES = [f"leaf{k:02d}" for k in range(40)]
RED, BLUE = "#d62728", "#1f77b4"


def newick(lo, hi, rng):
    """A tree over LEAVES[lo:hi], split in halves, with lengths and a support value at each clade."""
    length = round(rng.uniform(0.02, 0.2), 3)
    if hi - lo == 1:
        return f"{LEAVES[lo]}:{length}"
    middle = (lo + hi) // 2
    return f"({newick(lo, middle, rng)},{newick(middle, hi, rng)}){rng.randint(40, 100)}:{length}"


TREE = newick(0, 40, random.Random(3)) + ";"


def declare(viewer):
    """The colour key: the first 20 nodes are Cluster A (red), the others Cluster B (blue)."""
    from matplotlib import colors as mcolors

    from desktop import Node_Colors
    categories = [("Cluster A", tuple(float(v) for v in mcolors.to_rgba(RED))), ("Cluster B", tuple(float(v) for v in mcolors.to_rgba(BLUE)))]
    Node_Colors.declare_categories(viewer, "Clusters", categories, [0 if k < 20 else 1 for k in range(40)], nodes=range(40))


# The image TreeView.buildImage makes for `options`, read back: a PNG's size and a few of its pixels, an SVG's text.
BUILD = """
    const made = await TreeView.buildImage(Object.assign({format: 'png', size: 100, transparent: true, scale: true, legend: true, support: false}, OPTIONS));
    if (made.blob.type === 'image/svg+xml') return {name: made.name, type: made.blob.type, text: await made.blob.text()};
    const bitmap = await createImageBitmap(made.blob);
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const pixel = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data);
    const bytes = new Uint8Array(await made.blob.arrayBuffer());
    let hash = 0;
    for (const b of bytes) hash = (hash * 31 + b) >>> 0;
    return {name: made.name, type: made.blob.type, width: bitmap.width, height: bitmap.height, corner: pixel(1, 1),
            hash, size: bytes.length};
"""


class ImageCase(TreePageMixin, unittest.TestCase):
    """A page over a network of LEAVES with the tree file TREE open, its colours in, the Details card closed."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(tree_backend._THRESHOLD_CACHE, {"order": None, "count": 0, "built": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self):
        painted = {k: RED if k < 20 else BLUE for k in range(40)}
        self.serve_headers(LEAVES, colors=node_colors(40, painted), key=declare, tree=None)
        give_edges(self.viewer, [(n, n + 1) for n in range(39)], np.arange(39, 0, -1), threshold=3.5)
        self.open_page()
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)
        self.script(f"TreeView.openText({json.dumps(TREE)}, {{name: 'clusters.nwk', size: {len(TREE)}, pasted: false}}); return 1")
        self.wait_for("TreeView.state.source === 'uploaded' && document.body.dataset.view === 'tree'")
        self.wait_for("TreeView.state.colors !== null && TreeView.state.colors.layers.length > 0")
        self.script("TreeView.closeCard('details'); TreeView.drawNow(); return 1")

    def build(self, **options):
        return self.script(BUILD.replace("OPTIONS", json.dumps(options)), timeout=60)

    def svg(self, **options):
        """The SVG for `options`, parsed in the page: its size, the texts in the drawing and in the band, and its paths."""
        made = self.build(format="svg", **options)
        return self.script("""const doc = new DOMParser().parseFromString(TEXT, 'image/svg+xml');
            const root = doc.documentElement;
            const group = root.querySelector('g[clip-path]');
            const texts = (nodes) => Array.from(nodes).map((node) => ({text: node.textContent, transform: node.getAttribute('transform')}));
            return {error: !!doc.querySelector('parsererror'), width: Number(root.getAttribute('width')), height: Number(root.getAttribute('height')),
                    drawing: texts(group.querySelectorAll('text')), band: texts(root.querySelectorAll(':scope > text')),
                    paths: group.querySelectorAll('path').length, fills: Array.from(root.querySelectorAll('[fill]'), (node) => node.getAttribute('fill')),
                    clip: root.querySelector('clipPath rect').getAttribute('height'),
                    rects: Array.from(root.querySelectorAll(':scope > rect')).filter((r) => r.getAttribute('width') === root.getAttribute('width')).length};""".replace("TEXT", json.dumps(made["text"])))

    def zoom_into_a(self):
        """Zoom to a clade of four to eight leaves well inside Cluster A, where the names are drawn."""
        return self.script("""const m = TreeView.state.model;
            for (let i = 0; i < m.nodeCount; i++) {
                const n = m.lastLeaf[i] - m.firstLeaf[i] + 1;
                if (m.childCount[i] > 0 && n >= 4 && n <= 8 && m.firstLeaf[i] >= 4 && m.lastLeaf[i] <= 15) { TreeView.zoomToNode(i); return TreeView.drawNow(); }
            }
            return null;""")


class ScaleBarTests(ImageCase):
    """The scale bar: a round length, at most 120 pixels long, beside the overview map; none for a cladogram."""

    def test_the_length_is_one_two_or_five_times_a_power_of_ten_at_most_120_pixels(self):
        self.start()
        found = self.script("""const bar = (sx) => TreeView.scaleBar({model: {hasLengths: true}, layout: {mode: 'lengths'}, view: {sx}});
            return [1250, 1000, 50, 30, 20, 0.004].map(bar);""")
        self.assertEqual([round(b["length"], 12) for b in found], [0.05, 0.1, 2, 2, 5, 20000])
        self.assertEqual([round(b["px"], 6) for b in found], [62.5, 100, 100, 60, 100, 80])
        self.assertEqual([b["text"] for b in found][:3], ["0.05", "0.1", "2"])
        self.assertIsNone(self.state("TreeView.scaleBar({model: {hasLengths: true}, layout: {mode: 'cladogram'}, view: {sx: 50}})"))
        self.assertIsNone(self.state("TreeView.scaleBar({model: {hasLengths: false}, layout: {mode: 'lengths'}, view: {sx: 50}})"))

    def test_the_bar_follows_the_zoom_and_sits_beside_the_overview_map(self):
        self.start()
        fitted = self.state("TreeView.drawNow().scale")
        self.assertTrue(60 <= fitted["px"] <= 120)
        # Wider by four: a quarter of the length is as long, so the bar is a shorter round length.
        zoomed = self.script("TreeView.zoomAbout(200, 200, 4, 1); return TreeView.drawNow().scale")
        self.assertLess(zoomed["length"], fitted["length"])
        self.assertTrue(60 <= zoomed["px"] <= 120)
        place = self.state("""(() => { const map = document.getElementById('minimap');
            return {x: TreeView.state.scaleBarX, right: map.offsetLeft + map.offsetWidth, hidden: map.hidden}; })()""")
        self.assertFalse(place["hidden"])
        self.assertEqual(place["x"], place["right"] + 20)
        self.script("document.getElementById('overview-btn').click(); TreeView.drawNow(); return 1")
        self.assertEqual(self.state("TreeView.state.scaleBarX"), 20)

    def test_a_cladogram_has_no_scale_bar(self):
        self.start()
        self.script("document.getElementById('cladogram-btn').click(); return 1")
        self.wait_for("TreeView.state.layout.mode === 'cladogram'")
        self.assertIsNone(self.state("TreeView.drawNow().scale"))
        self.script("document.getElementById('layout-select').value = 'circular'; document.getElementById('layout-select').dispatchEvent(new Event('change')); "
                    "document.getElementById('phylogram-btn').click(); return 1")
        self.wait_for("TreeView.state.layout.kind === 'circular' && TreeView.state.layout.mode === 'lengths'")
        self.assertIsNotNone(self.state("TreeView.drawNow().scale"))


class ImageCardTests(ImageCase):
    """The camera button between Fit and Details, icons all three, and the card it opens."""

    def test_fit_image_and_details_are_icon_buttons_in_that_order_with_names(self):
        self.start()
        found = self.state("""[...document.querySelectorAll('.toolbar > button')].slice(-3).map((b) => ({
            id: b.id, label: b.getAttribute('aria-label'), text: b.textContent.trim(), icon: b.classList.contains('btn-icon'), disabled: b.disabled}))""")
        self.assertEqual([b["id"] for b in found], ["fit-btn", "image-btn", "details-btn"])
        self.assertEqual([b["label"] for b in found], ["Fit", "Save an image", "Details"])
        self.assertEqual({b["text"] for b in found}, {""})
        self.assertTrue(all(b["icon"] and not b["disabled"] for b in found))
        self.script("TreeView.show('loading'); return 1")
        self.assertTrue(self.state("document.getElementById('image-btn').disabled"))

    def test_the_card_opens_in_place_of_details_and_gives_the_focus_back(self):
        self.start()
        self.script("document.getElementById('image-btn').focus(); document.getElementById('image-btn').click(); return 1")
        found = self.state("""({shown: document.getElementById('image-card').classList.contains('shown'),
            expanded: document.getElementById('image-btn').getAttribute('aria-expanded'), active: document.activeElement.id})""")
        self.assertEqual(found, {"shown": True, "expanded": "true", "active": "image-card"})
        # Details takes its place, and the camera button takes it back.
        self.script("document.getElementById('details-btn').click(); return 1")
        self.assertEqual(self.state("['image-card', 'details'].map((id) => document.getElementById(id).classList.contains('shown'))"), [False, True])
        self.script("document.getElementById('image-btn').click(); return 1")
        self.assertEqual(self.state("['image-card', 'details'].map((id) => document.getElementById(id).classList.contains('shown'))"), [True, False])
        # Esc closes it and the focus goes back to the button; so does Cancel.
        self.script("document.getElementById('image-card').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true})); return 1")
        self.assertEqual(self.state("[document.getElementById('image-card').classList.contains('shown'), document.activeElement.id]"), [False, "image-btn"])
        self.script("document.getElementById('image-btn').click(); document.getElementById('image-cancel').click(); return 1")
        self.assertEqual(self.state("[document.getElementById('image-card').classList.contains('shown'), document.activeElement.id]"), [False, "image-btn"])

    def test_the_choices_fit_the_tree_and_the_view(self):
        self.start()
        self.script("document.getElementById('image-btn').click(); return 1")
        found = self.state("""(() => { const $ = (id) => document.getElementById(id);
            return {scale: $('image-scale').disabled, legend: $('image-legend').disabled, support: $('image-support').disabled,
                    label: $('image-support-label').textContent, pixels: $('image-pixels').textContent, size: $('image-size').value,
                    max: Number($('image-size').max), width: TreeView.canvas.clientWidth, ratio: devicePixelRatio}; })()""")
        self.assertFalse(found["scale"])
        self.assertFalse(found["legend"])
        # The support dots are off, so their legend has nothing to say.
        self.assertTrue(found["support"])
        self.assertEqual(found["label"], "Support legend")
        self.assertEqual(found["size"], "200")
        self.assertTrue(found["pixels"].startswith(f"{round(found['width'] * found['ratio'] * 2):,} × "), found["pixels"])
        self.assertGreaterEqual(found["max"], 200)
        # SVG has no size; a cladogram no scale bar; shown support dots a legend.
        self.script("document.getElementById('image-svg').click(); return 1")
        self.assertEqual(self.state("[document.getElementById('image-size').disabled, document.getElementById('image-pixels').textContent]"),
                         [True, "Lines and text, at any size"])
        self.script("document.getElementById('image-cancel').click(); document.getElementById('support-btn').click(); "
                    "document.getElementById('cladogram-btn').click(); return 1")
        self.wait_for("TreeView.state.layout.mode === 'cladogram'")
        self.script("document.getElementById('image-btn').click(); return 1")
        self.assertEqual(self.state("['image-scale', 'image-support'].map((id) => document.getElementById(id).disabled)"), [True, False])

    def test_a_size_typed_past_the_largest_is_brought_back_to_it(self):
        self.start()
        self.script("document.getElementById('image-btn').click(); const input = document.getElementById('image-size');"
                    "input.value = '100000'; input.dispatchEvent(new Event('input')); input.dispatchEvent(new Event('change')); return 1")
        found = self.state("[document.getElementById('image-size').value, document.getElementById('image-size').max]")
        self.assertEqual(found[0], found[1])

    def test_save_hands_the_file_to_the_browser_closes_the_card_and_says_so(self):
        self.start()
        self.script("""window.__saved = []; TreeView.downloadImage = (blob, name) => window.__saved.push([name, blob.type, blob.size]);
            document.getElementById('image-btn').click(); document.getElementById('image-save').click(); return 1""")
        self.wait_for("window.__saved.length === 1")
        self.wait_for("!document.getElementById('image-card').classList.contains('shown')")
        saved = self.state("window.__saved[0]")
        self.assertEqual(saved[:2], ["tree-view.png", "image/png"])
        self.assertGreater(saved[2], 1000)
        self.assertEqual(self.state("[document.activeElement.id, document.getElementById('announcer').textContent]"), ["image-btn", "Saved tree-view.png."])

    def test_an_image_that_cannot_be_made_is_said_in_the_card_which_stays_open(self):
        self.start()
        self.script("""TreeView.buildImage = () => Promise.reject(new Error('no memory'));
            document.getElementById('image-btn').click(); document.getElementById('image-save').click(); return 1""")
        self.wait_for("document.getElementById('image-error').textContent !== ''")
        self.assertTrue(self.state("document.getElementById('image-card').classList.contains('shown')"))
        self.assertIn("smaller size", self.state("document.getElementById('image-error').textContent"))


class PngTests(ImageCase):
    """A PNG: the drawing's pixels times the size, a transparent background, the band under the tree, light colours."""

    def test_a_png_is_the_drawing_at_its_size_with_the_band_under_it(self):
        self.start()
        size = self.state("[TreeView.canvas.clientWidth, TreeView.canvas.clientHeight, devicePixelRatio]")
        plain = self.build(scale=False, legend=False)
        self.assertEqual((plain["width"], plain["height"]), (round(size[0] * size[2]), round(size[1] * size[2])))
        self.assertEqual(plain["corner"][3], 0)
        double = self.build(size=200, scale=False, legend=False)
        self.assertEqual((double["width"], double["height"]), (round(size[0] * size[2] * 2), round(size[1] * size[2] * 2)))
        banded = self.build()
        self.assertEqual(banded["width"], plain["width"])
        self.assertGreater(banded["height"], plain["height"])

    def test_without_transparency_the_background_is_the_light_themes_card(self):
        self.start()
        found = self.build(transparent=False)
        self.assertEqual(found["corner"], [255, 255, 255, 255])

    def test_dark_mode_saves_the_same_image_and_stays_dark(self):
        self.start()
        light = self.build(transparent=False)
        self.script("document.documentElement.setAttribute('data-theme', 'dark'); return 1")
        self.pump(0.2)
        dark = self.build(transparent=False)
        self.assertEqual((dark["hash"], dark["size"]), (light["hash"], light["size"]))
        self.assertEqual(self.state("document.documentElement.getAttribute('data-theme')"), "dark")
        self.assertEqual(self.state("TreeView.state.palette.background"), self.state("getComputedStyle(document.documentElement).getPropertyValue('--bg-card').trim()"))


class SvgTests(ImageCase):
    """An SVG: the same drawing as paths and text, its colours as #rrggbb, clipped to the drawing, the band under it."""

    def test_an_svg_has_the_drawings_names_paths_and_size(self):
        self.start()
        stats = self.zoom_into_a()
        self.assertGreater(stats["labels"], 0)
        found = self.svg()
        self.assertFalse(found["error"])
        size = self.state("[TreeView.canvas.clientWidth, TreeView.canvas.clientHeight]")
        self.assertEqual(found["width"], size[0])
        self.assertGreater(found["height"], size[1])
        self.assertEqual(float(found["clip"]), size[1])
        names = [t["text"] for t in found["drawing"] if t["text"].startswith("leaf")]
        self.assertEqual(len(names), stats["labels"])
        # The branches of one colour, and the leaves' dots.
        self.assertGreaterEqual(found["paths"], 2)
        self.assertTrue(all(fill == "none" or fill.startswith("#") for fill in found["fills"]), set(found["fills"]))
        # Transparent: nothing under the drawing; with a background, one rectangle the size of the image.
        self.assertEqual(found["rects"], 0)
        self.assertEqual(self.svg(transparent=False)["rects"], 1)

    def test_the_band_lists_the_scale_and_only_the_colours_in_view(self):
        self.start()
        fitted = [t["text"] for t in self.svg()["band"]]
        self.assertIn("Clusters:", fitted)
        self.assertIn("Cluster A", fitted)
        self.assertIn("Cluster B", fitted)
        self.zoom_into_a()
        zoomed = [t["text"] for t in self.svg()["band"]]
        self.assertIn("Cluster A", zoomed)
        self.assertNotIn("Cluster B", zoomed)
        bar = self.state("TreeView.scaleBar(TreeView.state)")
        self.assertIn(bar["text"], zoomed)
        self.script("document.getElementById('support-btn').click(); return 1")
        with_support = [t["text"] for t in self.svg(support=True)["band"]]
        self.assertIn("Support", with_support)
        self.assertIn("100%", with_support)

    def test_circular_names_are_turned_with_a_transform(self):
        self.start()
        self.script("document.getElementById('layout-select').value = 'circular'; document.getElementById('layout-select').dispatchEvent(new Event('change')); return 1")
        self.wait_for("TreeView.state.layout.kind === 'circular'")
        stats = self.state("TreeView.drawNow()")
        self.assertGreater(stats["labels"], 0)
        names = [t for t in self.svg()["drawing"] if t["text"].startswith("leaf")]
        self.assertEqual(len(names), stats["labels"])
        self.assertTrue(all(t["transform"] and t["transform"].startswith("matrix(") for t in names))

    def test_arcs_are_written_as_a_canvas_draws_them(self):
        self.start()
        found = self.script("""const write = (draw) => { const ctx = new TreeView.SvgContext(); ctx.beginPath(); draw(ctx); ctx.stroke(); return ctx.out[0].match(/d="([^"]*)"/)[1]; };
            return {
                circle: write((c) => c.arc(10, 10, 5, 0, 2 * Math.PI)),
                quarter: write((c) => c.arc(0, 0, 10, 0, Math.PI / 2)),
                back: write((c) => c.arc(0, 0, 10, 0, Math.PI / 2, true)),
                wrapped: write((c) => c.arc(0, 0, 10, 0, -Math.PI / 2)),
                joined: write((c) => { c.moveTo(0, 0); c.arc(20, 0, 10, Math.PI, 0); }),
            };""")
        self.assertEqual(found["circle"], "M15 10A5 5 0 1 1 5 10A5 5 0 1 1 15 10")
        self.assertEqual(found["quarter"], "M10 0A10 10 0 0 1 0 10")
        self.assertEqual(found["back"], "M10 0A10 10 0 1 0 0 10")
        self.assertEqual(found["wrapped"], "M10 0A10 10 0 1 1 0 -10")
        self.assertEqual(found["joined"], "M0 0L10 0A10 10 0 0 1 30 0")


if __name__ == "__main__":
    unittest.main()
