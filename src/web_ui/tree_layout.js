/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's layouts: where each node of a model (tree_model.js) goes.
 *
 * A layout gives every node a position (x, y) in its own units, and for each node the farthest x its
 * subtree reaches, which is what lets the drawing skip what is out of view. The rectangular layout puts
 * the root at the left, a node's x at its distance from the root and each leaf in a row of its own,
 * numbered from the top; a clade sits at the middle of its first and last child.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    // The rectangular layout of a model. mode is "lengths" (a node's x is its summed branch length from the
    // root: a phylogram) or "cladogram" (every leaf at the right, the branches only showing who is next to whom).
    // A tree with no branch lengths, or with lengths that sum to nothing, is drawn as a cladogram whatever mode asks.
    TV.layoutRectangular = function (model, mode) {
        const n = model.nodeCount;
        const x = new Float64Array(n);
        const y = new Float64Array(n);
        const reach = new Float64Array(n);

        for (let i = 0; i < n; i++) {
            if (model.childCount[i] === 0) {
                y[i] = model.leafOrdinal[i] + 0.5;
            } else {
                y[i] = (y[model.childList[model.childStart[i]]] + y[model.childList[model.childStart[i + 1] - 1]]) / 2;
            }
        }

        let used = mode === "cladogram" || !model.hasLengths ? "cladogram" : "lengths";
        if (used === "lengths") {
            // Parents come after their children, so the root is first when counting down.
            for (let i = n - 2; i >= 0; i--) x[i] = x[model.parent[i]] + Math.max(0, model.length[i]);
            let farthest = 0;
            for (let i = 0; i < n; i++) if (x[i] > farthest) farthest = x[i];
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
            for (let i = 0; i < n; i++) x[i] = rootHeight - height[i];
        }

        for (let i = 0; i < n; i++) reach[i] = x[i];
        for (let i = 0; i < n - 1; i++) {
            const p = model.parent[i];
            if (reach[i] > reach[p]) reach[p] = reach[i];
        }
        return { kind: "rectangular", mode: used, x, y, reach, width: reach[model.root], height: model.leafCount };
    };
})(window.TreeView);
