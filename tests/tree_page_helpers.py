# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What Tree View's page tests share (not collected by unittest).

The tests of the page are in test_tree_page.py, test_tree_detail_page.py, test_tree_color_page.py,
test_tree_selection_page.py, test_tree_appearance_page.py, test_tree_source_page.py and test_tree_open_page.py, which
each open tree.html in Qt WebEngine over tests/tree_page_harness.py. This module holds what more than one of them
uses: the page's file names, JavaScript snippets, the eight-leaf tree and its colours, the spies on the drawing, and
the page over a neighbor-joining tree of 60 leaves. Import with ``from tests.tree_page_helpers import ...``.
"""

import json
import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for folder in (SRC_DIR, PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.tree_fixtures import TREE_NAME, headers_of, make_alignment_folder, neighbor_joining_tree  # noqa: E402
from tests.tree_page_harness import TreePageMixin  # noqa: E402

WEB_DIR = SRC_DIR / "web_ui"
PAGE_FILES = ["tree.html", "tree_model.js", "tree_color.js", "tree_layout.js", "tree_render.js", "tree_nav.js", "tree_select.js",
              "tree_minimap.js", "tree_legend.js", "tree_session.js"]

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
