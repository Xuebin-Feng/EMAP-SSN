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
 * answer. This file holds what such a request is made of; which gesture sends one is the page's to decide. A box
 * that is still open is sent as it changes (see "A box that is still open" below), so that the network follows it as
 * the tree follows the network's own box.
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

    // ---- A box that is still open: the network's selection follows it ------------------------------------------
    //
    // While the right button is down the Viewer is sent the leaves in the box as they change, so that the network's
    // selection follows the box as the tree follows the network's own. The Viewer answers a request before it runs
    // it, so the page cannot pace itself on the replies: it sends an update only when the set in the box changed, at
    // most once for each frame drawn (the caller is the drawing), and with two requests still out it keeps only the
    // newest to send when one is back. The Viewer skips an update that a newer one has overtaken. An update is
    //   { action: "tree_select", gesture: { id, phase: "update", seq }, mode: "replace" | "toggle", nodes }
    // with nodes the Viewer's node numbers of the leaves as a bit set in base64 (bit n % 8 of byte n // 8 is node n:
    // at most 1/8 of a byte for each node of the network, however many leaves are in the box), and seq 1, 2, 3...
    // The end carries the final set itself, so that the result never rests on every update having arrived, and a
    // cancel (Esc) puts the selection back. While the gesture is open a request of "alive" goes every second: a
    // Viewer that hears nothing for a few seconds takes the page to be gone and cancels.
    const MAX_IN_FLIGHT = 2;
    const ALIVE_MS = 1000;
    const RETRY_MS = 300;
    let gestureCount = 0;
    // The id of the last gesture this page opened: the Viewer's selection_changed events that it makes while the
    // gesture is open carry it (TV.isOwnGestureEcho).
    TV.ownGestureId = null;

    TV.newGesture = function () {
        gestureCount++;
        const id = "g" + Date.now().toString(36) + "-" + gestureCount;
        TV.ownGestureId = id;
        return { id, seq: 0, last: null, lastShift: false, sent: false, inFlight: 0, waiting: null, timer: 0, over: false };
    };

    // The leaves as a bit set of the network's nodes, in base64.
    TV.encodeLeaves = function (leaves) {
        const model = state.model;
        const bytes = new Uint8Array((TV.leafOfNode(model).length + 7) >> 3);
        const { viewerNode } = model;
        for (let n = 0; n < leaves.length; n++) {
            const node = viewerNode[leaves[n]];
            bytes[node >> 3] |= 1 << (node & 7);
        }
        let text = "";
        for (let at = 0; at < bytes.length; at += 8192) text += String.fromCharCode.apply(null, bytes.subarray(at, at + 8192));
        return btoa(text);
    };

    function sameLeaves(a, b) {
        if (!a || a.length !== b.length) return false;
        for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false;
        return true;
    }

    function sendGesture(gesture, phase, leaves, shift) {
        const body = { action: "tree_select", gesture: { id: gesture.id, phase } };
        if (phase === "update" || phase === "end") {
            body.gesture.seq = ++gesture.seq;
            body.mode = shift ? "toggle" : "replace";
            body.nodes = TV.encodeLeaves(leaves);
        }
        return post(body);
    }

    // An end or a cancel is sent again once if it did not get through: the Viewer cancels a gesture it hears no more of.
    function sendFinal(gesture, phase, leaves, shift) {
        sendGesture(gesture, phase, leaves, shift).then((ok) => {
            if (!ok) setTimeout(() => sendGesture(gesture, phase, leaves, shift), RETRY_MS);
        });
    }

    function stopAlive(gesture) {
        gesture.over = true;
        gesture.waiting = null;
        clearInterval(gesture.timer);
        gesture.timer = 0;
    }

    // The leaves in the box are these now (sorted; ``shift`` toggles them against the selection the gesture started
    // with): tell the Viewer if that is not what it was last told.
    TV.gestureUpdate = function (gesture, leaves, shift) {
        if (gesture.over || !state.model) return;
        if (gesture.sent && gesture.lastShift === shift && sameLeaves(gesture.last, leaves)) return;
        gesture.last = leaves;
        gesture.lastShift = shift;
        if (!gesture.timer) gesture.timer = setInterval(() => sendGesture(gesture, "alive"), ALIVE_MS);
        if (gesture.inFlight >= MAX_IN_FLIGHT) {
            gesture.sent = true;
            gesture.waiting = { leaves, shift };
            return;
        }
        pushUpdate(gesture, leaves, shift);
    };

    function pushUpdate(gesture, leaves, shift) {
        gesture.sent = true;
        gesture.inFlight++;
        sendGesture(gesture, "update", leaves, shift).then(() => {
            gesture.inFlight--;
            const next = gesture.waiting;
            gesture.waiting = null;
            if (next && !gesture.over) pushUpdate(gesture, next.leaves, next.shift);
        });
    }

    // The button was let go with these leaves in the box: the end carries them. The selection the tree shows is the
    // one the end makes at once, so that it does not look as if it went back to an earlier update's until the
    // Viewer's last event comes.
    // ``base`` is the leaves the network had selected when the box opened, which a Shift box toggles against.
    TV.gestureEnd = function (gesture, leaves, shift, base) {
        stopAlive(gesture);
        const result = shift ? TV.toggleLeaves(base, leaves) : leaves;
        const { viewerNode } = state.model;
        TV.applySelection(Array.from(result, (leaf) => viewerNode[leaf]));
        sendFinal(gesture, "end", leaves, shift);
    };

    // Esc, a cancelled pointer or a lost capture, or the page going away: the Viewer puts its selection back.
    TV.gestureCancel = function (gesture) {
        stopAlive(gesture);
        if (gesture.sent) sendFinal(gesture, "cancel");
    };

    // Whether a selection_changed message (its text, not parsed) is the echo of an update of this page's own open
    // gesture, which the page leaves alone: it draws the box's selection itself, and an echo that arrives late would
    // draw an older one over it. The message that ends or cancels a gesture says "final" and is taken as any other.
    const OWN_ECHO = /^\{"type":\s*"selection_changed",\s*"gesture":\s*\{"id":\s*"([A-Za-z0-9_-]*)"(,\s*"final":\s*true)?\}/;
    TV.isOwnGestureEcho = function (text) {
        const found = OWN_ECHO.exec(text);
        return Boolean(found) && !found[2] && found[1] === TV.ownGestureId;
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
