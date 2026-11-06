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
 * search and the legend mark leaves too, and neither takes the selection's place.
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
    // be leaves of this tree.
    TV.emptySelection = function () {
        return { leaves: new Int32Array(0), clade: -1, focus: -1, count: 0 };
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

    // Whether the tree says how stable any of its splits is.
    TV.hasSupport = function (model) {
        if (model.hasSupport === undefined) model.hasSupport = model.support.some((value) => value >= 0);
        return model.hasSupport;
    };

    // Take the Viewer's selection as the tree's (the Viewer's node numbers), and with it the click focus (a node number,
    // null or -1 for none); left out, the focus stays where it was, as the selection changing does not move it.
    TV.applySelection = function (indices, focus) {
        const model = state.model;
        if (!model) return;
        const leaves = TV.leavesOfNodes(model, indices);
        const inverse = TV.leafOfNode(model);
        state.selection = {
            leaves,
            clade: TV.smallestClade(model, leaves),
            focus: focus === undefined ? state.selection.focus
                : Number.isInteger(focus) && focus >= 0 && focus < inverse.length ? inverse[focus] : -1,
            count: indices.length,
        };
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

    // Ask the Viewer to put its click focus on the node of a leaf (the Viewer's node number), or off it (null).
    TV.focusNode = function (index) {
        return post({ action: "tree_focus", index });
    };
})(window.TreeView);
