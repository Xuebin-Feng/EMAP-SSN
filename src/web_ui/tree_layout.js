/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's layouts: where each node of a model (tree_model.js) goes.
 *
 * Three layouts, each with a branch-length form (a phylogram: a node is as far from the root as its summed
 * branch lengths) and a cladogram form (every branch counts one, or the leaves stand in a row, so the lines
 * only show who is next to whom):
 *
 *   rectangular  the root at the left, a node's x its distance from the root and each leaf a row of its own
 *                (y numbered from the top); a clade sits at the middle of its first and last child. The
 *                two axes are free, so a view zooms rows and width apart.
 *   circular     the same tree wound round the root: a node's radius is its distance and its angle the
 *                middle of its leaves' angles, the leaves sharing the circle equally.
 *   unrooted     the equal-angle layout: each clade gets the share of the full circle that its leaves are of
 *                all, its root-ward branch runs along the middle of that share and its own clades divide
 *                the share again. The root is where the walk starts, not a place of the tree.
 *
 * A layout gives every node a position (x, y) and a bounding box of its subtree in its own units, and tells how
 * each node is drawn in them, so that the drawing, the pointer and the overview work for every layout alike:
 *
 *   stem(i, out)   the line from where node i's branch starts to node i: out = [x0, y0, x1, y1]
 *   join(i, out)   what joins node i's children: out[0] is 0 (nothing), 1 (a line: out[1..4] as a stem) or
 *                  2 (an arc about the origin: out[1] radius, out[2] and out[3] the angles it runs between)
 *   wedge(i, out)  the shape a collapsed clade is drawn as: out[0] 0, a triangle (out[1..6], three points), or
 *                  1, a slice (out[1], out[2] its point; out[3], out[4] the centre of its arc; out[5] the
 *                  radius; out[6] and out[7] the angles)
 *   tiny(i, sx, sy, limit)   whether the clade is smaller than `limit` pixels where it matters
 *   mark(i, out)   the line a tiny clade is drawn as, from its node to where it reaches: out = [x0, y0, x1, y1]
 *   thickness(i, sx, sy)   how thick that line is to be, in pixels: how wide the clade is where it is widest
 *
 * Nothing recurses, and a layout of 44,000 leaves takes a few milliseconds.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const TWO_PI = 2 * Math.PI;
    // The first leaf of the circular layout is at the top, and the leaves go round clockwise.
    const START = -Math.PI / 2;

    TV.LAYOUT_KINDS = ["rectangular", "circular", "unrooted"];

    // The distance of each node from the root, as a branch-length sum ("lengths") or in steps with the leaves all
    // at the end ("cladogram"). A tree with no branch lengths, or lengths that sum to nothing, can only be a
    // cladogram, whatever is asked. Returns { depth, mode } with the mode used.
    function nodeDepths(model, mode) {
        const n = model.nodeCount;
        const depth = new Float64Array(n);
        let used = mode === "cladogram" || !model.hasLengths ? "cladogram" : "lengths";
        if (used === "lengths") {
            // Parents come after their children, so the root is first when counting down.
            for (let i = n - 2; i >= 0; i--) depth[i] = depth[model.parent[i]] + Math.max(0, model.length[i]);
            let farthest = 0;
            for (let i = 0; i < n; i++) if (depth[i] > farthest) farthest = depth[i];
            if (!(farthest > 0)) used = "cladogram";
        }
        if (used === "cladogram") {
            // A leaf is 0 steps from the leaves, and a clade is one step more than its deepest child.
            const height = new Int32Array(n);
            for (let i = 0; i < n - 1; i++) {
                const p = model.parent[i];
                if (height[i] + 1 > height[p]) height[p] = height[i] + 1;
            }
            const rootHeight = Math.max(1, height[model.root]);
            for (let i = 0; i < n; i++) depth[i] = rootHeight - height[i];
        }
        return { depth, mode: used };
    }

    // The farthest depth in each node's subtree.
    function reachOf(model, depth) {
        const reach = Float64Array.from(depth);
        for (let i = 0; i < model.nodeCount - 1; i++) {
            const p = model.parent[i];
            if (reach[i] > reach[p]) reach[p] = reach[i];
        }
        return reach;
    }

    // Four arrays that hold the bounding box of each node's subtree: minX, minY, maxX, maxY.
    function newBoxes(n) {
        return { x0: new Float64Array(n).fill(Infinity), y0: new Float64Array(n).fill(Infinity),
                 x1: new Float64Array(n).fill(-Infinity), y1: new Float64Array(n).fill(-Infinity) };
    }

    function addPoint(boxes, i, x, y) {
        if (x < boxes.x0[i]) boxes.x0[i] = x;
        if (x > boxes.x1[i]) boxes.x1[i] = x;
        if (y < boxes.y0[i]) boxes.y0[i] = y;
        if (y > boxes.y1[i]) boxes.y1[i] = y;
    }

    // An arc about the origin from angle a0 to a1 (a0 <= a1) at radius r: its ends, and each quarter turn between.
    function addArc(boxes, i, r, a0, a1) {
        addPoint(boxes, i, r * Math.cos(a0), r * Math.sin(a0));
        addPoint(boxes, i, r * Math.cos(a1), r * Math.sin(a1));
        for (let quarter = Math.ceil(a0 / (Math.PI / 2)); quarter * (Math.PI / 2) < a1; quarter++) {
            addPoint(boxes, i, r * Math.cos(quarter * Math.PI / 2), r * Math.sin(quarter * Math.PI / 2));
        }
    }

    // Each node's box takes in its children's, which are already finished: they come first.
    function mergeUp(model, boxes) {
        for (let i = 0; i < model.nodeCount - 1; i++) {
            const p = model.parent[i];
            if (boxes.x0[i] < boxes.x0[p]) boxes.x0[p] = boxes.x0[i];
            if (boxes.y0[i] < boxes.y0[p]) boxes.y0[p] = boxes.y0[i];
            if (boxes.x1[i] > boxes.x1[p]) boxes.x1[p] = boxes.x1[i];
            if (boxes.y1[i] > boxes.y1[p]) boxes.y1[p] = boxes.y1[i];
        }
    }

    // The bounds of the whole layout, padded where it has no size (a single leaf).
    function boundsOf(boxes, root) {
        const bounds = { minX: boxes.x0[root], minY: boxes.y0[root], maxX: boxes.x1[root], maxY: boxes.y1[root] };
        if (!(bounds.maxX - bounds.minX > 1e-9)) { bounds.minX -= 0.5; bounds.maxX += 0.5; }
        if (!(bounds.maxY - bounds.minY > 1e-9)) { bounds.minY -= 0.5; bounds.maxY += 0.5; }
        return bounds;
    }

    function firstChild(model, i) { return model.childList[model.childStart[i]]; }
    function lastChild(model, i) { return model.childList[model.childStart[i + 1] - 1]; }

    function rectangular(model, depth, mode) {
        const n = model.nodeCount;
        const x = depth;
        const y = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            y[i] = model.leafOrdinal[i] >= 0 ? model.leafOrdinal[i] + 0.5 : (y[firstChild(model, i)] + y[lastChild(model, i)]) / 2;
        }
        const reach = reachOf(model, depth);
        const boxes = newBoxes(n);
        for (let i = 0; i < n; i++) {
            const p = model.parent[i];
            boxes.x0[i] = p >= 0 ? x[p] : x[i];
            boxes.x1[i] = reach[i];
            boxes.y0[i] = model.firstLeaf[i];
            boxes.y1[i] = model.lastLeaf[i] + 1;
        }
        return {
            kind: "rectangular", mode, uniform: false, x, y, reach, boxes,
            bounds: boundsOf(boxes, model.root), width: reach[model.root], height: model.leafCount,
            stem(i, out) {
                const p = model.parent[i];
                out[0] = p >= 0 ? x[p] : x[i]; out[1] = y[i]; out[2] = x[i]; out[3] = y[i];
            },
            join(i, out) {
                if (model.childCount[i] === 0) { out[0] = 0; return; }
                out[0] = 1; out[1] = x[i]; out[2] = y[firstChild(model, i)]; out[3] = x[i]; out[4] = y[lastChild(model, i)];
            },
            wedge(i, out) {
                out[0] = 0; out[1] = x[i]; out[2] = y[i];
                out[3] = reach[i]; out[4] = model.firstLeaf[i]; out[5] = reach[i]; out[6] = model.lastLeaf[i] + 1;
            },
            // A clade is tiny when its rows are, however far it reaches to the right.
            tiny(i, sx, sy, limit) {
                return model.lastLeaf[i] > model.firstLeaf[i] && (model.lastLeaf[i] + 1 - model.firstLeaf[i]) * sy < limit;
            },
            mark(i, out) { out[0] = x[i]; out[1] = y[i]; out[2] = reach[i]; out[3] = y[i]; },
            thickness(i, sx, sy) { return (model.lastLeaf[i] + 1 - model.firstLeaf[i]) * sy; },
        };
    }

    function circular(model, depth, mode) {
        const n = model.nodeCount;
        const leaves = model.leafCount;
        const angle = new Float64Array(n);
        const x = new Float64Array(n);
        const y = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            angle[i] = model.leafOrdinal[i] >= 0
                ? START + TWO_PI * (model.leafOrdinal[i] + 0.5) / leaves
                : (angle[firstChild(model, i)] + angle[lastChild(model, i)]) / 2;
            x[i] = depth[i] * Math.cos(angle[i]);
            y[i] = depth[i] * Math.sin(angle[i]);
        }
        const reach = reachOf(model, depth);
        // The angles between which a clade's leaves lie, each leaf having its slot of the circle.
        const slot = TWO_PI / leaves;
        const lowAngle = (i) => START + slot * model.firstLeaf[i];
        const highAngle = (i) => START + slot * (model.lastLeaf[i] + 1);
        const boxes = newBoxes(n);
        for (let i = 0; i < n; i++) {
            const p = model.parent[i];
            addPoint(boxes, i, x[i], y[i]);
            if (p >= 0) addPoint(boxes, i, depth[p] * Math.cos(angle[i]), depth[p] * Math.sin(angle[i]));
            if (model.childCount[i] > 0) {
                addArc(boxes, i, depth[i], angle[firstChild(model, i)], angle[lastChild(model, i)]);
                addArc(boxes, i, reach[i], lowAngle(i), highAngle(i));
            }
        }
        mergeUp(model, boxes);
        return {
            kind: "circular", mode, uniform: true, x, y, angle, depth, reach, boxes, slot,
            bounds: boundsOf(boxes, model.root), width: 2 * reach[model.root], height: 2 * reach[model.root],
            stem(i, out) {
                const p = model.parent[i];
                const r = p >= 0 ? depth[p] : depth[i];
                out[0] = r * Math.cos(angle[i]); out[1] = r * Math.sin(angle[i]); out[2] = x[i]; out[3] = y[i];
            },
            join(i, out) {
                if (model.childCount[i] === 0) { out[0] = 0; return; }
                out[0] = 2; out[1] = depth[i]; out[2] = angle[firstChild(model, i)]; out[3] = angle[lastChild(model, i)];
            },
            wedge(i, out) {
                out[0] = 1; out[1] = x[i]; out[2] = y[i]; out[3] = 0; out[4] = 0;
                out[5] = reach[i]; out[6] = lowAngle(i); out[7] = highAngle(i);
            },
            // Tiny where it is widest: the arc its leaves take at its far end.
            tiny(i, sx, sy, limit) {
                return model.lastLeaf[i] > model.firstLeaf[i]
                    && reach[i] * (model.lastLeaf[i] + 1 - model.firstLeaf[i]) * slot * sx < limit;
            },
            mark(i, out) { out[0] = x[i]; out[1] = y[i]; out[2] = reach[i] * Math.cos(angle[i]); out[3] = reach[i] * Math.sin(angle[i]); },
            thickness(i, sx, sy) { return reach[i] * (model.lastLeaf[i] + 1 - model.firstLeaf[i]) * slot * sx; },
        };
    }

    function unrooted(model, mode) {
        const n = model.nodeCount;
        const x = new Float64Array(n);
        const y = new Float64Array(n);
        const low = new Float64Array(n);
        const high = new Float64Array(n);
        const branch = new Float64Array(n);
        let uses = mode === "lengths" && model.hasLengths;
        if (uses) {
            let total = 0;
            for (let i = 0; i < n - 1; i++) { branch[i] = Math.max(0, model.length[i]); total += branch[i]; }
            if (!(total > 0)) uses = false;
        }
        if (!uses) branch.fill(1);
        // The farthest a clade's leaves are from its node along the branches: the radius of its slice.
        const far = new Float64Array(n);
        for (let i = 0; i < n - 1; i++) {
            const p = model.parent[i];
            if (far[i] + branch[i] > far[p]) far[p] = far[i] + branch[i];
        }
        // Counting down, a node has its share of the circle before its children are given theirs.
        low[model.root] = -Math.PI;
        high[model.root] = Math.PI;
        for (let i = n - 1; i >= 0; i--) {
            if (model.childCount[i] === 0) continue;
            const span = high[i] - low[i];
            const total = model.lastLeaf[i] - model.firstLeaf[i] + 1;
            let start = low[i];
            for (let k = model.childStart[i]; k < model.childStart[i + 1]; k++) {
                const child = model.childList[k];
                const share = span * (model.lastLeaf[child] - model.firstLeaf[child] + 1) / total;
                low[child] = start;
                high[child] = start + share;
                start += share;
                const direction = (low[child] + high[child]) / 2;
                x[child] = x[i] + branch[child] * Math.cos(direction);
                y[child] = y[i] + branch[child] * Math.sin(direction);
            }
        }
        return buildUnrooted(model, uses, x, y, low, high, far);
    }

    function buildUnrooted(model, uses, x, y, low, high, far) {
        const n = model.nodeCount;
        const boxes = newBoxes(n);
        for (let i = 0; i < n; i++) {
            const p = model.parent[i];
            addPoint(boxes, i, x[i], y[i]);
            if (p >= 0) addPoint(boxes, i, x[p], y[p]);
            if (model.childCount[i] > 0) {
                // The slice a collapsed clade is drawn as has its point at the node: its arc, shifted to it.
                const r = far[i];
                addPoint(boxes, i, x[i] + r * Math.cos(low[i]), y[i] + r * Math.sin(low[i]));
                addPoint(boxes, i, x[i] + r * Math.cos(high[i]), y[i] + r * Math.sin(high[i]));
                for (let quarter = Math.ceil(low[i] / (Math.PI / 2)); quarter * (Math.PI / 2) < high[i]; quarter++) {
                    addPoint(boxes, i, x[i] + r * Math.cos(quarter * Math.PI / 2), y[i] + r * Math.sin(quarter * Math.PI / 2));
                }
            }
        }
        mergeUp(model, boxes);
        const bounds = boundsOf(boxes, model.root);
        return {
            kind: "unrooted", mode: uses ? "lengths" : "cladogram", uniform: true, x, y, low, high, far, boxes, bounds,
            slot: TWO_PI / model.leafCount,
            width: bounds.maxX - bounds.minX, height: bounds.maxY - bounds.minY,
            stem(i, out) {
                const p = model.parent[i];
                out[0] = p >= 0 ? x[p] : x[i]; out[1] = p >= 0 ? y[p] : y[i]; out[2] = x[i]; out[3] = y[i];
            },
            join(i, out) { out[0] = 0; },
            wedge(i, out) {
                out[0] = 1; out[1] = x[i]; out[2] = y[i]; out[3] = x[i]; out[4] = y[i];
                out[5] = far[i]; out[6] = low[i]; out[7] = high[i];
            },
            // Tiny when the box round the clade is, or when its slice is a sliver: every leaf of an equal-angle clade lies in
            // the slice of the circle that is the clade's share, from its node as far as its farthest leaf, so a clade that
            // is long but under a few pixels wide at its far end is a line, however big the box round it is.
            tiny(i, sx, sy, limit) {
                return model.lastLeaf[i] > model.firstLeaf[i]
                    && (Math.max(boxes.x1[i] - boxes.x0[i], boxes.y1[i] - boxes.y0[i]) * sx < limit
                        || far[i] * (high[i] - low[i]) * sx < limit);
            },
            mark(i, out) {
                const middle = (low[i] + high[i]) / 2;
                out[0] = x[i]; out[1] = y[i]; out[2] = x[i] + far[i] * Math.cos(middle); out[3] = y[i] + far[i] * Math.sin(middle);
            },
            thickness(i, sx, sy) {
                const box = Math.max(boxes.x1[i] - boxes.x0[i], boxes.y1[i] - boxes.y0[i]) * sx;
                return Math.min(box, far[i] * (high[i] - low[i]) * sx) * 0.5;
            },
        };
    }

    // The layout of a model: kind is one of TV.LAYOUT_KINDS, mode "lengths" or "cladogram" (the mode used comes back
    // in the layout, which can only be a cladogram for a tree with no lengths).
    TV.layoutTree = function (model, kind, mode) {
        if (kind === "unrooted") return unrooted(model, mode);
        const { depth, mode: used } = nodeDepths(model, mode);
        return kind === "circular" ? circular(model, depth, used) : rectangular(model, depth, used);
    };

    TV.layoutRectangular = function (model, mode) {
        return TV.layoutTree(model, "rectangular", mode);
    };

    // The layout a tree is shown in at first: a rooted tree (UPGMA's) as a rectangular tree from its root, an unrooted
    // one (neighbor-joining's, rooted at a midpoint only so that it can be written down) from the middle, and a tree
    // that says neither as the rooted kind.
    TV.defaultLayoutKind = function (model) {
        return model.rooted === false ? "unrooted" : "rectangular";
    };
})(window.TreeView);
