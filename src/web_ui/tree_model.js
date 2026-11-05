/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's model: the tree the Viewer sends (/api/tree/data), as typed arrays over its nodes, with the
 * arrays that make drawing and looking things up cheap.
 *
 * The Viewer numbers the nodes in post-order: every child comes before its parent, the root is the last
 * node, a node's subtree is the block of nodes that ends at it, and the leaves, in node order, run top to
 * bottom as the file has them. Nothing here recurses, so a tree as deep as it is long is no trouble.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    // The bytes a base64 string stands for.
    function bytesOf(text) {
        const binary = atob(text);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
    }

    // A typed array of Type with `count` elements, read from the base64 text of its little-endian bytes.
    function typedArray(text, Type, count, name) {
        if (typeof text !== "string") throw new Error(name + " is missing");
        const bytes = bytesOf(text);
        if (bytes.length !== count * Type.BYTES_PER_ELEMENT) throw new Error(name + " has the wrong size");
        return new Type(bytes.buffer);
    }

    TV.typedArray = typedArray;

    // The model of a payload whose status is "ok". Throws an Error with a plain reason when the payload cannot be
    // a tree the page can draw.
    TV.decodeTree = function (payload) {
        const nodeCount = payload.counts.nodes;
        const leafCount = payload.counts.leaves;
        if (!Number.isInteger(nodeCount) || !Number.isInteger(leafCount) || leafCount < 1 || nodeCount < leafCount) {
            throw new Error("the counts do not describe a tree");
        }
        const parent = typedArray(payload.parent, Int32Array, nodeCount, "parent");
        const length = typedArray(payload.length, Float32Array, nodeCount, "length");
        const support = typedArray(payload.support, Int8Array, nodeCount, "support");
        const viewerNode = typedArray(payload.viewer_node, Int32Array, leafCount, "viewer_node");
        const labels = payload.labels;
        if (!Array.isArray(labels) || labels.length !== leafCount) throw new Error("the leaf names do not match the leaves");

        const root = nodeCount - 1;
        if (parent[root] !== -1) throw new Error("the last node is not the root");
        const childCount = new Int32Array(nodeCount);
        for (let i = 0; i < root; i++) {
            if (parent[i] <= i || parent[i] >= nodeCount) throw new Error("the nodes are not in post-order");
            childCount[parent[i]]++;
        }

        // The children of each node, side by side in node order, which is left to right.
        const childStart = new Int32Array(nodeCount + 1);
        for (let i = 0; i < nodeCount; i++) childStart[i + 1] = childStart[i] + childCount[i];
        const childList = new Int32Array(childStart[nodeCount]);
        const fill = childStart.slice(0, nodeCount);
        for (let i = 0; i < root; i++) childList[fill[parent[i]]++] = i;

        // What a bottom-up pass finds: the size of each subtree, the leaves it holds (a range of leaf
        // numbers, as the leaves are in order) and the leaves' numbering.
        const size = new Int32Array(nodeCount).fill(1);
        const firstLeaf = new Int32Array(nodeCount);
        const lastLeaf = new Int32Array(nodeCount);
        const leafOrdinal = new Int32Array(nodeCount).fill(-1);
        const leafNode = new Int32Array(leafCount);
        let leaves = 0;
        for (let i = 0; i < nodeCount; i++) {
            if (childCount[i] === 0) {
                if (leaves >= leafCount) throw new Error("the tree has more leaves than it says");
                leafOrdinal[i] = leaves;
                leafNode[leaves] = i;
                firstLeaf[i] = leaves;
                lastLeaf[i] = leaves;
                leaves++;
            } else {
                firstLeaf[i] = firstLeaf[childList[childStart[i]]];
                lastLeaf[i] = lastLeaf[childList[childStart[i + 1] - 1]];
            }
            if (i < root) size[parent[i]] += size[i];
        }
        if (leaves !== leafCount) throw new Error("the tree has fewer leaves than it says");

        // shownBefore[k]: how many of the leaves before leaf k are nodes of the network, so that the number below
        // any clade is a subtraction.
        const shownBefore = new Int32Array(leafCount + 1);
        for (let k = 0; k < leafCount; k++) shownBefore[k + 1] = shownBefore[k] + (viewerNode[k] >= 0 ? 1 : 0);

        return {
            nodeCount, leafCount, root, parent, length, support, viewerNode, labels,
            childCount, childStart, childList, size, firstLeaf, lastLeaf, leafOrdinal, leafNode, shownBefore,
            hasLengths: !!payload.has_lengths,
            rooted: payload.rooted === true ? true : payload.rooted === false ? false : null,
            counts: payload.counts,
            notes: payload.notes || {},
            warnings: payload.warnings || [],
        };
    };

    // How many leaves are below node i (1 for a leaf), and how many of them are nodes of the network.
    TV.leavesUnder = function (model, i) {
        return model.lastLeaf[i] - model.firstLeaf[i] + 1;
    };
    TV.shownUnder = function (model, i) {
        return model.shownBefore[model.lastLeaf[i] + 1] - model.shownBefore[model.firstLeaf[i]];
    };

    // The child of node i whose leaves hold leaf number k (i is internal and holds k); a binary search, as a node
    // may have thousands of children.
    TV.childHolding = function (model, i, k) {
        let low = model.childStart[i];
        let high = model.childStart[i + 1] - 1;
        while (low < high) {
            const middle = (low + high + 1) >> 1;
            if (model.firstLeaf[model.childList[middle]] <= k) low = middle; else high = middle - 1;
        }
        return model.childList[low];
    };

    // The nodes from node `node` up to the root, as an array of node numbers beginning with it.
    TV.pathToRoot = function (model, node) {
        const path = [];
        for (let i = node; i >= 0; i = model.parent[i]) path.push(i);
        return path;
    };
})(window.TreeView);
