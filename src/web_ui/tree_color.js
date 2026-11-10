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
 * in a few dozen colours, not thousands. The colour a node has until something colours it is plain: it is a
 * class like the others and the tree is drawn in it, but it is the network's default, not a choice, so the legend
 * leaves it out. What a colour means is not guessed from the colour: the Viewer's colour key (payload.key) says which
 * layer painted a node and which of its categories or what value the node has, and the legend lists the leaves by
 * that; only the leaves the key does not cover are listed by colour. A clade that is neither plain nor one colour has no class and is drawn in the page's branch colour.
 *
 * A node of the network that the Viewer has hidden has no place in the tree at all: `removed` marks its leaf, and
 * the layout gives it no room.
 *
 * Everything is one pass over the nodes in post-order (children before parents): the colour most of a clade has
 * is found by a majority vote from its children (which finds the colour held by more than half, if there is one),
 * and its share is counted exactly from the sorted places of that colour's leaves.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    // The class of a leaf that has no colour to show (it is not a node of the network, or is hidden), and of a clade
    // with no colour of its own; and the class of the colour a node has until something colours it (the plain colour).
    TV.NO_CLASS = 0xFFFF;
    TV.PLAIN_CLASS = 0;
    // The share of a clade's leaves that must have one colour for the clade to be drawn in it.
    TV.PURE_SHARE = 0.9;
    // The most colours a folded clade's bar shows.
    TV.MIX_PARTS = 4;

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

    // A text from the key is text; anything else is nothing.
    function plainText(value) {
        return typeof value === "string" ? value : "";
    }

    function cssOf(rgba) {
        const channel = (v) => (Number.isFinite(v) ? Math.min(255, Math.max(0, Math.round(v))) : 0);
        return "rgb(" + channel(rgba[0]) + ", " + channel(rgba[1]) + ", " + channel(rgba[2]) + ")";
    }

    // The rows (label, colour) of a layer's categories or extras; entries that are not well formed read as empty.
    function entriesOf(list) {
        return (Array.isArray(list) ? list : []).map((entry) => {
            const rgba = entry && Array.isArray(entry.rgba) && entry.rgba.length >= 3 ? entry.rgba : [0, 0, 0];
            return { label: plainText(entry && entry.label), css: cssOf(rgba), leaves: [] };
        });
    }

    // The colour key of a payload against the leaves of `model`: { layers, unlabelledCounts }. layers lists, newest first,
    // each layer of the key that has a leaf shown: { kind, title, index, rows: [{ label, css, leaves, count }] and, for a
    // gradient, scheme, min, max, stops (css colours) and bar: { leaves, values, count } }. A leaf the key does not
    // cover, or whose node no longer has the colour the key expects, is counted in unlabelledCounts by its colour class.
    // A key that does not fit the network is no key.
    function decodeKey(key, colors, model, leafClass, count) {
        const unlabelledCounts = new Int32Array(colors.classCount);
        const covered = new Uint8Array(model.leafCount);
        let owner = null;
        let category = null;
        let labelled = null;
        let values = null;
        let layers = null;
        try {
            if (key && Array.isArray(key.layers) && key.layers.length > 0) {
                owner = TV.typedArray(key.owner, key.owner_type === "i2" ? Int16Array : Int32Array, count, "owner");
                category = TV.typedArray(key.category, key.category_type === "i2" ? Int16Array : Int32Array, count, "category");
                labelled = TV.typedArray(key.labelled, Uint8Array, Math.ceil(count / 8), "labelled");
                if (key.values) values = TV.typedArray(key.values, Float32Array, count, "values");
                layers = key.layers.every((layer) => layer && typeof layer === "object") ? key.layers : null;
            }
        } catch (error) {
            layers = null;
        }
        const built = (layers || []).map((layer) => {
            const gradient = layer.kind === "gradient";
            const made = { kind: gradient ? "gradient" : "categories", title: plainText(layer.title), rows: entriesOf(gradient ? layer.extras : layer.categories), bar: null };
            if (gradient) {
                const stops = Array.isArray(layer.stops) ? layer.stops.filter((stop) => Array.isArray(stop) && stop.length >= 3) : [];
                Object.assign(made, {
                    scheme: plainText(layer.scheme), min: Number.isFinite(layer.min) ? layer.min : null,
                    max: Number.isFinite(layer.max) ? layer.max : null, stops: stops.map(cssOf), bar: { leaves: [], values: [], count: 0 },
                });
            }
            return made;
        });
        for (let k = 0; k < model.leafCount; k++) {
            const cls = leafClass[k];
            if (cls === TV.NO_CLASS) continue;
            const node = model.viewerNode[k];
            let placed = false;
            if (layers && (labelled[node >> 3] >> (node & 7)) & 1) {
                const layer = owner[node];
                if (layer >= 0 && layer < built.length) {
                    const made = built[layer];
                    const slot = category[node];
                    if (slot >= 0 && slot < made.rows.length) {
                        made.rows[slot].leaves.push(k);
                        placed = true;
                    } else if (made.bar && slot === -1 && values && Number.isFinite(values[node])) {
                        made.bar.leaves.push(k);
                        made.bar.values.push(values[node]);
                        placed = true;
                    }
                }
            }
            if (placed) covered[k] = 1; else unlabelledCounts[cls]++;
        }
        const shown = [];
        // What a hover can say of a clade: each labelled row of a category layer, and each gradient bar, is one group of
        // leaves (leafGroup, -1 for the leaves no layer covers); groups[g] names it.
        const leafGroup = new Int32Array(model.leafCount).fill(-1);
        const groups = [];
        for (let index = built.length - 1; index >= 0; index--) {
            const made = built[index];
            made.index = index;
            made.rows = made.rows.filter((row) => row.leaves.length > 0);
            for (const row of made.rows) {
                row.count = row.leaves.length;
                row.leaves = Int32Array.from(row.leaves);
                for (let p = 0; p < row.leaves.length; p++) leafGroup[row.leaves[p]] = groups.length;
                groups.push({ layer: made, row });
            }
            if (made.bar) {
                for (let p = 0; p < made.bar.leaves.length; p++) leafGroup[made.bar.leaves[p]] = groups.length;
                if (made.bar.leaves.length > 0) groups.push({ layer: made, row: null });
                made.bar.count = made.bar.leaves.length;
                made.bar.leaves = Int32Array.from(made.bar.leaves);
                made.bar.values = Float32Array.from(made.bar.values);
            }
            if (made.rows.length > 0 || (made.bar && made.bar.count > 0)) shown.push(made);
        }
        return { layers: shown, unlabelledCounts, covered, leafGroup, groups };
    }

    // The leaf numbers (sorted) of class `cls` that no layer of the colour key covers: what "Unlabelled colours" counts.
    TV.unlabelledLeavesOfClass = function (colors, cls) {
        const found = [];
        for (let p = colors.starts[cls]; p < colors.starts[cls + 1]; p++) if (!colors.covered[colors.positions[p]]) found.push(colors.positions[p]);
        return Int32Array.from(found);
    };

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
        // The leaves whose nodes the Viewer has hidden.
        const removed = new Uint8Array(leafCount);
        let removedCount = 0;
        for (let k = 0; k < leafCount; k++) {
            const node = model.viewerNode[k];
            if (node < 0 || node >= count) continue;
            if ((hiddenBits[node >> 3] >> (node & 7)) & 1) {
                removed[k] = 1;
                removedCount++;
                continue;
            }
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
            version: payload.version, count, cluster, leafClass, starts, positions, rep, classCount, removed, removedCount,
            css: rep.map((rgb) => "rgb(" + rgb.join(", ") + ")"),
            mixCache: new Map(),
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
                    nodeClass[i] = cls;
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
                if (share >= TV.PURE_SHARE) nodeClass[i] = voted;
            }
        }
        Object.assign(colors, { nodeClass, dominant, purity, present });

        // The legend: what the colour key says (its layers, newest first), then the colours of the leaves it does not
        // cover, other than plain, most leaves first, all of them.
        const keyed = decodeKey(payload.key, colors, model, leafClass, count);
        colors.layers = keyed.layers;
        colors.covered = keyed.covered;
        colors.leafGroup = keyed.leafGroup;
        colors.groups = keyed.groups;
        colors.tally = new Int32Array(keyed.groups.length + classCount);
        colors.sayCache = new Map();
        const entries = [];
        for (let c = 1; c < classCount; c++) {
            const leaves = keyed.unlabelledCounts[c];
            if (leaves > 0) entries.push({ cls: c, count: leaves, css: colors.css[c], hex: hexOf(rep[c]) });
        }
        entries.sort((a, b) => b.count - a.count || a.cls - b.cls);
        colors.unlabelled = entries;
        return colors;
    };

    // The leaf numbers (sorted) of a gradient layer's bar whose value is from `low` to `high`.
    TV.leavesInRange = function (layer, low, high) {
        const found = [];
        const { leaves, values } = layer.bar;
        for (let p = 0; p < leaves.length; p++) if (values[p] >= low && values[p] <= high) found.push(leaves[p]);
        return Int32Array.from(found);
    };

    // The colour of class `cls` as #rrggbb.
    TV.classHex = function (colors, cls) {
        return hexOf(colors.rep[cls]);
    };

    // What most of the shown leaves of clade i have in common, as the hover says it, or null when no group has more than
    // half of them. A group is a labelled row of a category layer, a gradient layer (its leaves of the bar), or a colour
    // class of the leaves no layer covers (plain excepted). { share, group: { layer, row } } for a labelled group, with
    // low and high (the least and the greatest value of that layer among the clade's leaves) for a gradient; { share,
    // cls } for a colour. Kept for the colours in hand, so a hover on the same clade costs a lookup.
    TV.cladeSays = function (colors, model, i) {
        if (colors.sayCache.has(i)) return colors.sayCache.get(i);
        const shown = colors.present[i];
        let said = null;
        if (shown > 0) {
            const { leafGroup, leafClass, tally, groups } = colors;
            const offset = groups.length;
            let best = -1;
            let most = 0;
            const first = model.firstLeaf[i];
            const last = model.lastLeaf[i];
            for (let k = first; k <= last; k++) {
                const cls = leafClass[k];
                if (cls === TV.NO_CLASS) continue;
                const g = leafGroup[k];
                const slot = g >= 0 ? g : (cls === TV.PLAIN_CLASS ? -1 : offset + cls);
                if (slot < 0) continue;
                if (++tally[slot] > most) { most = tally[slot]; best = slot; }
            }
            for (let k = first; k <= last; k++) {
                const cls = leafClass[k];
                if (cls === TV.NO_CLASS) continue;
                const g = leafGroup[k];
                tally[g >= 0 ? g : offset + cls] = 0;
            }
            if (best >= 0 && most / shown > 0.5) {
                if (best >= offset) said = { share: most / shown, cls: best - offset };
                else {
                    said = { share: most / shown, group: groups[best] };
                    if (!groups[best].row) {
                        const { leaves, values } = groups[best].layer.bar;
                        let low = Infinity;
                        let high = -Infinity;
                        let p = lowerBound(leaves, 0, leaves.length, first);
                        for (; p < leaves.length && leaves[p] <= last; p++) {
                            if (values[p] < low) low = values[p];
                            if (values[p] > high) high = values[p];
                        }
                        said.low = low;
                        said.high = high;
                    }
                }
            }
        }
        if (colors.sayCache.size > 20000) colors.sayCache.clear();
        colors.sayCache.set(i, said);
        return said;
    };

    // The leaves that have no room in the tree because the Viewer has hidden their nodes (1 for each), or null when
    // there are none to leave out, or none that would be left: a network with every node hidden shows the whole tree, faded.
    TV.removedLeaves = function (colors) {
        return colors && colors.removedCount > 0 && colors.removedCount < colors.removed.length ? colors.removed : null;
    };

    // The leaf numbers (sorted) that have class `cls`.
    TV.leavesOfClass = function (colors, cls) {
        return colors.positions.slice(colors.starts[cls], colors.starts[cls + 1]);
    };

    // What the colours of a folded clade are, for the bar along its far edge: { shown, parts: [[class, share], ...]
    // the colours with the most leaves (plain among them), as shares of the leaves shown }. Kept for the colours in hand.
    TV.cladeMix = function (colors, model, i) {
        const known = colors.mixCache.get(i);
        if (known) return known;
        const shown = colors.present[i];
        const tally = new Map();
        for (let k = model.firstLeaf[i]; k <= model.lastLeaf[i]; k++) {
            const cls = colors.leafClass[k];
            if (cls !== TV.NO_CLASS) tally.set(cls, (tally.get(cls) || 0) + 1);
        }
        const parts = [...tally].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, TV.MIX_PARTS).map(([cls, n]) => [cls, n / shown]);
        const mix = { shown, parts };
        if (colors.mixCache.size > 20000) colors.mixCache.clear();
        colors.mixCache.set(i, mix);
        return mix;
    };
})(window.TreeView);
