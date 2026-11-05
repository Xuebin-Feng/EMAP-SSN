/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's colours: the colours of the network's nodes (/api/tree/colors) as the tree shows them.
 *
 * A leaf takes the colour of the node it stands for. A clade takes the colour most of its leaves have when
 * at least nine in ten of the leaves that are shown have it, and else none, so that the overview of a tree of
 * thousands of leaves says at once whether the colours (clusters, say) are clades. Colours are compared in classes,
 * a colour kept to four bits of each channel, so that a gradient (spectrum) forms clades too and a clade is drawn
 * in a few dozen colours, not thousands. The colour a node has until something colours it is plain, and plain
 * is no colour: the tree is drawn in its own colours where nothing has coloured the network.
 *
 * Everything is one pass over the nodes in post-order (children before parents): the colour most of a clade has
 * is found by a majority vote from its children (which finds the colour held by more than half, if there is one),
 * and its share is counted exactly from the sorted places of that colour's leaves.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    // The class of a leaf that has no colour to show (it is not a node of the network, or is hidden), and of a clade
    // with no colour of its own; and the class of the colour a node has until something colours it.
    TV.NO_CLASS = 0xFFFF;
    TV.PLAIN_CLASS = 0;
    // The share of a clade's leaves that must have one colour for the clade to be drawn in it.
    TV.PURE_SHARE = 0.9;
    // The most colours the legend lists, and the most a folded clade's bar shows.
    TV.LEGEND_ROWS = 12;
    TV.MIX_PARTS = 4;
    // A colour is called a cluster's when this share of the leaves that have it are in that cluster.
    const CLUSTER_SHARE = 0.9;

    function classKey(r, g, b) {
        return ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4);
    }

    function hexOf(rgb) {
        return "#" + rgb.map((value) => value.toString(16).padStart(2, "0")).join("");
    }

    // The first index in values[from, to) of a value that is at least `bound` (values is sorted there).
    function lowerBound(values, from, to, bound) {
        let low = from;
        let high = to;
        while (low < high) {
            const middle = (low + high) >> 1;
            if (values[middle] < bound) low = middle + 1; else high = middle;
        }
        return low;
    }

    // How many leaves of class `cls` have a number from `first` to `last`.
    function countInRange(colors, cls, first, last) {
        const from = colors.starts[cls];
        const to = colors.starts[cls + 1];
        return lowerBound(colors.positions, from, to, last + 1) - lowerBound(colors.positions, from, to, first);
    }

    // The colours of a payload whose status is "ok", over the leaves of `model`. Throws an Error with a plain reason
    // when the payload cannot be colours the page can use.
    TV.decodeColors = function (payload, model) {
        const count = payload.count;
        if (!Number.isInteger(count) || count < 1) throw new Error("the colours do not describe a network");
        const rgba = TV.typedArray(payload.rgba, Uint8Array, count * 4, "rgba");
        const hiddenBits = TV.typedArray(payload.hidden, Uint8Array, Math.ceil(count / 8), "hidden");
        const cluster = payload.cluster
            ? TV.typedArray(payload.cluster, payload.cluster_type === "i2" ? Int16Array : Int32Array, count, "cluster") : null;
        const plain = payload.plain;
        if (!Array.isArray(plain) || plain.length < 3) throw new Error("the plain colour is missing");

        // The leaves' classes: class 0 is plain, the others are numbered as they are met. A class stands for the
        // first colour met in it, exactly.
        const ids = new Map([[classKey(plain[0], plain[1], plain[2]), TV.PLAIN_CLASS]]);
        const rep = [[plain[0], plain[1], plain[2]]];
        const leafCount = model.leafCount;
        const leafClass = new Uint16Array(leafCount).fill(TV.NO_CLASS);
        for (let k = 0; k < leafCount; k++) {
            const node = model.viewerNode[k];
            if (node < 0 || node >= count) continue;
            if ((hiddenBits[node >> 3] >> (node & 7)) & 1) continue;
            const r = rgba[node * 4];
            const g = rgba[node * 4 + 1];
            const b = rgba[node * 4 + 2];
            const key = classKey(r, g, b);
            let cls = ids.get(key);
            if (cls === undefined) {
                cls = rep.length;
                ids.set(key, cls);
                rep.push([r, g, b]);
            }
            leafClass[k] = cls;
        }

        // The places of each class's leaves, in leaf order, side by side.
        const classCount = rep.length;
        const starts = new Int32Array(classCount + 1);
        for (let k = 0; k < leafCount; k++) if (leafClass[k] !== TV.NO_CLASS) starts[leafClass[k] + 1]++;
        for (let c = 0; c < classCount; c++) starts[c + 1] += starts[c];
        const positions = new Int32Array(starts[classCount]);
        const fill = starts.slice(0, classCount);
        for (let k = 0; k < leafCount; k++) if (leafClass[k] !== TV.NO_CLASS) positions[fill[leafClass[k]]++] = k;

        const colors = {
            version: payload.version, count, cluster, leafClass, starts, positions, rep, classCount,
            css: rep.map((rgb) => "rgb(" + rgb.join(", ") + ")"),
            mixCache: new Map(), labelCache: new Map(),
        };

        // One pass, children before parents. cand/weight are the majority vote; present is how many leaves are shown.
        const nodeCount = model.nodeCount;
        const { childCount, childStart, childList, leafOrdinal, firstLeaf, lastLeaf } = model;
        const candidate = new Uint16Array(nodeCount).fill(TV.NO_CLASS);
        const weight = new Int32Array(nodeCount);
        const present = new Int32Array(nodeCount);
        const dominant = new Uint16Array(nodeCount).fill(TV.NO_CLASS);
        const purity = new Float32Array(nodeCount);
        const nodeClass = new Uint16Array(nodeCount).fill(TV.NO_CLASS);
        for (let i = 0; i < nodeCount; i++) {
            if (childCount[i] === 0) {
                const cls = leafClass[leafOrdinal[i]];
                if (cls !== TV.NO_CLASS) {
                    candidate[i] = cls;
                    weight[i] = 1;
                    present[i] = 1;
                    dominant[i] = cls;
                    purity[i] = 1;
                    if (cls !== TV.PLAIN_CLASS) nodeClass[i] = cls;
                }
                continue;
            }
            let voted = TV.NO_CLASS;
            let votes = 0;
            let shown = 0;
            for (let k = childStart[i]; k < childStart[i + 1]; k++) {
                const child = childList[k];
                shown += present[child];
                const w = weight[child];
                if (w === 0) continue;
                const c = candidate[child];
                if (votes === 0) { voted = c; votes = w; }
                else if (voted === c) votes += w;
                else if (votes >= w) votes -= w;
                else { voted = c; votes = w - votes; }
            }
            present[i] = shown;
            candidate[i] = voted;
            weight[i] = votes;
            if (shown > 0 && voted !== TV.NO_CLASS) {
                const share = countInRange(colors, voted, firstLeaf[i], lastLeaf[i]) / shown;
                dominant[i] = voted;
                purity[i] = share;
                if (voted !== TV.PLAIN_CLASS && share >= TV.PURE_SHARE) nodeClass[i] = voted;
            }
        }
        Object.assign(colors, { nodeClass, dominant, purity, present });

        // The legend: the colours with the most leaves, other than plain.
        const entries = [];
        for (let c = 1; c < classCount; c++) {
            const leaves = starts[c + 1] - starts[c];
            if (leaves > 0) entries.push({ cls: c, count: leaves });
        }
        entries.sort((a, b) => b.count - a.count || a.cls - b.cls);
        colors.colored = entries.length;
        colors.legend = entries.slice(0, TV.LEGEND_ROWS).map((entry) => ({ ...entry, css: colors.css[entry.cls], hex: hexOf(rep[entry.cls]) }));
        colors.more = entries.length - colors.legend.length;
        return colors;
    };

    // The cluster number that at least nine in ten of the leaves of class `cls` are in, or -1 when there is none or
    // the Viewer has no clusters. Worked out once for each class.
    TV.clusterOfClass = function (colors, model, cls) {
        if (!colors.cluster) return -1;
        if (colors.labelCache.has(cls)) return colors.labelCache.get(cls);
        const tally = new Map();
        const from = colors.starts[cls];
        const to = colors.starts[cls + 1];
        for (let p = from; p < to; p++) {
            const id = colors.cluster[model.viewerNode[colors.positions[p]]];
            tally.set(id, (tally.get(id) || 0) + 1);
        }
        let best = -1;
        let most = 0;
        for (const [id, n] of tally) if (n > most) { best = id; most = n; }
        const found = best >= 0 && most >= CLUSTER_SHARE * (to - from) ? best : -1;
        colors.labelCache.set(cls, found);
        return found;
    };

    // The colour of class `cls` as #rrggbb.
    TV.classHex = function (colors, cls) {
        return hexOf(colors.rep[cls]);
    };

    // The leaf numbers (sorted) that have class `cls`.
    TV.leavesOfClass = function (colors, cls) {
        return colors.positions.slice(colors.starts[cls], colors.starts[cls + 1]);
    };

    // What the colours of a folded clade are, for the bar along its far edge: { shown, parts: [[class, share], ...]
    // the colours with the most leaves (not plain), as shares of the leaves shown }. Kept for the colours in hand.
    TV.cladeMix = function (colors, model, i) {
        const known = colors.mixCache.get(i);
        if (known) return known;
        const shown = colors.present[i];
        const tally = new Map();
        for (let k = model.firstLeaf[i]; k <= model.lastLeaf[i]; k++) {
            const cls = colors.leafClass[k];
            if (cls !== TV.NO_CLASS && cls !== TV.PLAIN_CLASS) tally.set(cls, (tally.get(cls) || 0) + 1);
        }
        const parts = [...tally].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, TV.MIX_PARTS).map(([cls, n]) => [cls, n / shown]);
        const mix = { shown, parts };
        if (colors.mixCache.size > 20000) colors.mixCache.clear();
        colors.mixCache.set(i, mix);
        return mix;
    };
})(window.TreeView);
