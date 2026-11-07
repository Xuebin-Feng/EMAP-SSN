/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's links to the network's selection, both ways.
 *
 * Network to tree: the Viewer tells every page which nodes are selected (selection_changed) and which one has the
 * click focus (highlight_row), as numbers of its own nodes. A leaf of the tree stands for one of them (the model's
 * viewerNode), so the selection is found among the leaves in one pass, and the smallest clade that holds all of them
 * is the one whose range of leaves is the first selected leaf to the last. The selection is a layer of its own: the
 * search and the legend mark leaves too, and neither takes the selection's place. The nodes next to the selected ones
 * (the Viewer outlines them in a colour of their own while it identifies connected nodes) are a layer beside it: the
 * Viewer's answer names them, never the edges between them, and they are marked in the tree the same way.
 *
 * Tree to network: the page asks the Viewer to select the nodes of some leaves (the tree_select action) or to focus
 * on one (tree_focus), and learns what it came to from the same two events as every other page does, never from the
 * answer. This file holds what such a request is made of; which gesture sends one is the page's to decide.
 *
 * Nothing here has a text to show, so nothing here is translated.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const state = TV.state;
    // The selection the network has, as the tree sees it (state.selection). leaves: the numbers of the leaves
    // selected, in tree order; clade: the smallest clade that holds them (-1 for none: no leaf, or one); focus: the
    // leaf with the click focus, or -1; count: how many nodes the network says are selected, of which some may not
    // be leaves of this tree; connected: the leaves next to the selected nodes of the network, in tree order.
    TV.emptySelection = function () {
        return { leaves: new Int32Array(0), connected: new Int32Array(0), clade: -1, focus: -1, count: 0 };
    };

    // For each node of the network, the number of the leaf that stands for it, or -1: the inverse of viewerNode.
    TV.leafOfNode = function (model) {
        if (model.leafOfNode) return model.leafOfNode;
        let largest = -1;
        for (let k = 0; k < model.leafCount; k++) if (model.viewerNode[k] > largest) largest = model.viewerNode[k];
        const inverse = new Int32Array(largest + 1).fill(-1);
        for (let k = 0; k < model.leafCount; k++) if (model.viewerNode[k] >= 0) inverse[model.viewerNode[k]] = k;
        model.leafOfNode = inverse;
        return inverse;
    };

    // The leaves that stand for the nodes `indices` (the Viewer's node numbers, in any order, with repeats, and of
    // which any number may not be a leaf of the tree), in tree order, without repeats.
    TV.leavesOfNodes = function (model, indices) {
        const inverse = TV.leafOfNode(model);
        const picked = new Uint8Array(model.leafCount);
        let found = 0;
        for (let n = 0; n < indices.length; n++) {
            const index = indices[n];
            if (index < 0 || index >= inverse.length) continue;
            const leaf = inverse[index];
            if (leaf >= 0 && !picked[leaf]) {
                picked[leaf] = 1;
                found++;
            }
        }
        const leaves = new Int32Array(found);
        let next = 0;
        for (let k = 0; k < model.leafCount; k++) if (picked[k]) leaves[next++] = k;
        return leaves;
    };

    // The smallest clade whose leaves hold all of `leaves` (sorted): the lowest ancestor of the first one that
    // reaches the last one. -1 for none, and for one leaf, which is a leaf and no clade.
    TV.smallestClade = function (model, leaves) {
        if (leaves.length < 2) return -1;
        const last = leaves[leaves.length - 1];
        let node = model.leafNode[leaves[0]];
        while (model.lastLeaf[node] < last) node = model.parent[node];
        return node;
    };

    // The clade to outline for `selection` in what `layout` draws: the smallest clade that holds the selected leaves that are
    // left in the tree, or -1 for none, or one leaf. With no leaf left out it is the selection's own clade. The lowest node
    // that holds two leaves that are left has two children that are left, so it is a clade, never a pass-through point; the
    // leaves that are left out are not counted, so that a selection that still names a node the Viewer has just hidden
    // leaves the outline where it belongs. Worked out once for each selection and layout.
    const outlines = new WeakMap();
    TV.outlineClade = function (model, layout, selection) {
        if (!layout || !layout.live) return selection.clade;
        const leaves = selection.leaves;
        const kept = outlines.get(leaves);
        if (kept && kept.layout === layout) return kept.clade;
        const shown = (leaf) => layout.live[model.leafNode[leaf]] === 1;
        let low = 0;
        let high = leaves.length - 1;
        while (low <= high && !shown(leaves[low])) low++;
        while (high >= low && !shown(leaves[high])) high--;
        let clade = -1;
        if (high > low) {
            clade = model.leafNode[leaves[low]];
            while (model.lastLeaf[clade] < leaves[high]) clade = model.parent[clade];
        }
        outlines.set(leaves, { layout, clade });
        return clade;
    };

    // Whether the tree says how stable any of its splits is.
    TV.hasSupport = function (model) {
        if (model.hasSupport === undefined) model.hasSupport = model.support.some((value) => value >= 0);
        return model.hasSupport;
    };

    // Take the Viewer's selection as the tree's (the Viewer's node numbers), and with it the click focus (a node number,
    // null or -1 for none) and the nodes next to the selection (node numbers, or null to say nothing of them); left out,
    // the focus stays where it was, as the selection changing does not move it, and the neighbours stay until the
    // Viewer says which they are now, unless nothing is selected, which has none.
    TV.applySelection = function (indices, focus, connected) {
        const model = state.model;
        if (!model) return;
        const leaves = TV.leavesOfNodes(model, indices);
        const inverse = TV.leafOfNode(model);
        state.selection = {
            leaves,
            connected: connected ? TV.leavesOfNodes(model, connected) : indices.length ? state.selection.connected : new Int32Array(0),
            clade: TV.smallestClade(model, leaves),
            focus: focus === undefined ? state.selection.focus
                : Number.isInteger(focus) && focus >= 0 && focus < inverse.length ? inverse[focus] : -1,
            count: indices.length,
        };
        TV.selectionChanged();
    };

    // The nodes next to the selected ones are these (node numbers of the network).
    TV.applyConnected = function (nodes) {
        const model = state.model;
        if (!model) return;
        state.selection = { ...state.selection, connected: TV.leavesOfNodes(model, nodes) };
        TV.selectionChanged();
    };

    // The click focus moved to node `focus` of the network (or off it).
    TV.applyFocus = function (focus) {
        const model = state.model;
        if (!model) return;
        const inverse = TV.leafOfNode(model);
        state.selection = {
            ...state.selection,
            focus: Number.isInteger(focus) && focus >= 0 && focus < inverse.length ? inverse[focus] : -1,
        };
        TV.selectionChanged();
    };

    TV.clearSelection = function () {
        state.selection = TV.emptySelection();
        TV.selectionChanged();
    };

    // The nodes of the network that a clade (or a leaf) stands for and that can be selected: its leaves that are
    // nodes of the network and, when the colours are known, not hidden. In tree order.
    TV.nodesOfClade = function (node) {
        const model = state.model;
        const colors = state.colors && state.colors.leafClass.length === model.leafCount ? state.colors : null;
        const nodes = [];
        for (let k = model.firstLeaf[node]; k <= model.lastLeaf[node]; k++) {
            if (model.viewerNode[k] < 0) continue;
            if (colors && colors.leafClass[k] === TV.NO_CLASS) continue;
            nodes.push(model.viewerNode[k]);
        }
        return nodes;
    };

    function post(body) {
        return fetch("/api/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        }).then((response) => response.ok, () => false);
    }

    // Ask the Viewer to select the nodes of a clade's leaves, as a box selection does: mode "replace" (the default),
    // "add" or "toggle". Resolves to what was asked: { sent: false } when the clade has no node to select, else
    // { sent: true, count, ok } with ok whether the Viewer took it. The new selection arrives as an event.
    TV.selectClade = async function (node, mode) {
        if (!state.model || node < 0 || node >= state.model.nodeCount) return { sent: false, count: 0 };
        const indices = TV.nodesOfClade(node);
        if (!indices.length) return { sent: false, count: 0 };
        const ok = await post({ action: "tree_select", indices, mode: mode || "replace" });
        return { sent: true, count: indices.length, ok };
    };

    // The leaves whose tips are in the box (edges included) with corners at (left, top) and (right, bottom), in pixels
    // of the canvas, in tree order. A leaf counts when it is drawn, at the point its selection ring is drawn at: left
    // in the layout, not inside a folded clade (a folded clade has no tips), not hidden by the colours, and a node of
    // the network. Names do not count, only tips.
    TV.leavesInBox = function (scene, left, top, right, bottom) {
        const { model, layout, view } = scene;
        if (!model || !layout) return new Int32Array(0);
        const colors = scene.colors && scene.colors.leafClass.length === model.leafCount ? scene.colors.leafClass : null;
        const folded = scene.leafHidden;
        const live = layout.live;
        const { leafNode, viewerNode } = model;
        const { x, y } = layout;
        const { ox, oy, sx, sy } = view;
        const NO_CLASS = TV.NO_CLASS;
        const found = new Int32Array(model.leafCount);
        let count = 0;
        for (let k = 0; k < found.length; k++) {
            const node = leafNode[k];
            const px = ox + x[node] * sx;
            if (px < left || px > right) continue;
            const py = oy + y[node] * sy;
            if (py < top || py > bottom) continue;
            if (viewerNode[k] < 0 || (live && !live[node]) || (folded && folded[k]) || (colors && colors[k] === NO_CLASS)) continue;
            found[count++] = k;
        }
        return found.slice(0, count);
    };

    // The sorted leaf numbers that are in exactly one of the sorted lists `a` and `b`: a selection with some leaves
    // toggled.
    TV.toggleLeaves = function (a, b) {
        const out = new Int32Array(a.length + b.length);
        let i = 0, j = 0, n = 0;
        while (i < a.length && j < b.length) {
            if (a[i] < b[j]) out[n++] = a[i++];
            else if (a[i] > b[j]) out[n++] = b[j++];
            else { i++; j++; }
        }
        while (i < a.length) out[n++] = a[i++];
        while (j < b.length) out[n++] = b[j++];
        return out.slice(0, n);
    };

    // Ask the Viewer to select the nodes of these leaves (numbers of the tree, each a node of the network), as a box
    // selection does: mode "replace" (an empty list clears the selection) or "toggle". Resolves to whether the Viewer took it.
    TV.selectLeaves = function (leaves, mode) {
        const indices = Array.from(leaves, (leaf) => state.model.viewerNode[leaf]);
        return post({ action: "tree_select", indices, mode });
    };

    // Ask the Viewer to clear its selection.
    TV.clearNetworkSelection = function () {
        return post({ action: "tree_select", indices: [], mode: "replace" });
    };

    // Ask the Viewer to put its click focus on the node of a leaf (the Viewer's node number), or off it (null).
    TV.focusNode = function (index) {
        return post({ action: "tree_focus", index });
    };
})(window.TreeView);
