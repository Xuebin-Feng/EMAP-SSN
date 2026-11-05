/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's drawing: a tree (tree_model.js) in a layout (tree_layout.js) on a canvas, in the page's theme.
 *
 * Only what is in view is visited. A clade that comes out smaller than a few pixels, or that a person has
 * collapsed, is drawn as one shape (a wedge) however many leaves it holds, so what a frame costs follows what can
 * be seen, not the size of the tree. Lines are batched into one path per style, and a leaf is marked with a dot,
 * then named, only when there is room for it. One traversal (paintTree) draws every layout, for the page and for
 * the overview map alike: the layout says how a node's lines run (layout.stem, layout.join, layout.wedge).
 *
 * A view maps the layout's units to pixels: screen x = ox + x * sx, screen y = oy + y * sy. A rectangular layout
 * keeps its two scales apart, as a long tree zooms in its rows far more than in its width; the others have one.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    // A clade smaller than this many pixels is one shape; a rectangular row or a circular slot this tall gets a dot,
    // and one this tall its name.
    TV.COLLAPSE_PX = 3;
    TV.DOT_PX = 4;
    TV.LABEL_PX = 12;
    // The width of the strip at the right edge that shows where the view is in the whole rectangular tree, and the matches.
    TV.RULER_WIDTH = 14;
    // A name is cut here, whatever its length.
    TV.MAX_LABEL_CHARACTERS = 160;
    // The most leaves an unrooted tree dots, and names, at once: it has no row to say whether there is room.
    TV.UNROOTED_DOTS = 2500;
    TV.UNROOTED_LABELS = 150;
    // How far from a leaf the name of one can reach: a clade that far outside the view can still show names in it.
    const LABEL_REACH_PX = 520;
    // The size, in pixels, of the square that folds a clade up, and of the smallest clade that has one. A square is
    // shown for a clade that is folded, hovered or selected, not for every clade at once.
    TV.HANDLE_PX = 9;
    TV.HANDLE_CLADE_PX = 12;

    // The red, green and blue (0 to 255) of a CSS colour, as the canvas reads it.
    const reader = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
    TV.rgbOf = function (css) {
        reader.clearRect(0, 0, 1, 1);
        reader.fillStyle = "#000000";
        reader.fillStyle = css;
        reader.fillRect(0, 0, 1, 1);
        return Array.from(reader.getImageData(0, 0, 1, 1).data.slice(0, 3));
    };

    // A colour between two CSS colours, as an opaque CSS colour: `share` of the first. Drawing a colour that is mixed
    // already, where a see-through one would be laid on the background, is far cheaper to draw.
    TV.mixColors = function (first, second, share) {
        const a = TV.rgbOf(first);
        const b = TV.rgbOf(second);
        return "rgb(" + a.map((value, k) => Math.round(value * share + b[k] * (1 - share))).join(", ") + ")";
    };

    // The colours the canvas draws in, read from the page's theme (its custom properties) so that light, dark and
    // the Viewer's theme colour all apply.
    TV.readPalette = function () {
        const style = getComputedStyle(document.documentElement);
        const get = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
        const background = get("--bg-card", "#ffffff");
        const branch = get("--text-secondary", "#52525b");
        return {
            // The line a tiny clade is drawn as, and the fill of a folded one: the branch colour, thinned out.
            mark: TV.mixColors(branch, background, 0.55),
            wedge: TV.mixColors(branch, background, 0.3),
            background: get("--bg-card", "#ffffff"),
            branch: get("--text-secondary", "#52525b"),
            faded: get("--text-disabled", "#a1a1aa"),
            label: get("--text-primary", "#09090b"),
            accent: get("--accent", "#5b5bd6"),
            selection: get("--selection", "rgba(91, 91, 214, 0.1)"),
            match: get("--highlight-ring", "#f59e0b"),
            matchSoft: get("--highlight-bg", "rgba(245, 158, 11, 0.15)"),
            ruler: get("--bg-muted", "#f4f4f5"),
            thumb: get("--scroll-hover", "#a1a1aa"),
            font: get("--font-family", "sans-serif"),
        };
    };

    // Give the canvas one pixel per device pixel, and return its size in CSS pixels and the ratio.
    TV.sizeCanvas = function (canvas) {
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.floor(canvas.clientWidth));
        const height = Math.max(1, Math.floor(canvas.clientHeight));
        const wanted = [Math.round(width * ratio), Math.round(height * ratio)];
        if (canvas.width !== wanted[0] || canvas.height !== wanted[1]) {
            canvas.width = wanted[0];
            canvas.height = wanted[1];
        }
        return { width, height, ratio };
    };

    const scratch = new Float64Array(8);

    // The pixel position of node i's own point (x, y), where its branch begins (x0, y0), and the box round its subtree
    // (left, top, right, bottom). For a rectangular layout `reach` is where its farthest leaf is.
    TV.nodePixels = function (scene, i) {
        const { layout, view } = scene;
        const boxes = layout.boxes;
        layout.stem(i, scratch);
        return {
            x: view.ox + scratch[2] * view.sx,
            y: view.oy + scratch[3] * view.sy,
            x0: view.ox + scratch[0] * view.sx,
            y0: view.oy + scratch[1] * view.sy,
            reach: view.ox + (layout.kind === "rectangular" ? layout.reach[i] : boxes.x1[i]) * view.sx,
            left: view.ox + boxes.x0[i] * view.sx,
            right: view.ox + boxes.x1[i] * view.sx,
            top: view.oy + boxes.y0[i] * view.sy,
            bottom: view.oy + boxes.y1[i] * view.sy,
        };
    };

    function shortened(text) {
        return text.length > TV.MAX_LABEL_CHARACTERS ? text.slice(0, TV.MAX_LABEL_CHARACTERS - 1) + "…" : text;
    }

    // The first index of the sorted array `values` whose value is at least `bound`.
    function lowerBound(values, bound) {
        let low = 0;
        let high = values.length;
        while (low < high) {
            const middle = (low + high) >> 1;
            if (values[middle] < bound) low = middle + 1; else high = middle;
        }
        return low;
    }

    // How many of the sorted leaf numbers `matches` are in the range [first, last].
    TV.matchesIn = function (matches, first, last) {
        return matches && matches.length ? lowerBound(matches, last + 1) - lowerBound(matches, first) : 0;
    };

    // Whether there is room to mark each leaf with a dot, and to name it: { dots, labels, size } for a layout in a view
    // with `leafCount` leaves drawn on their own. A rectangular row says it by its height; a circular tree by the arc
    // that a leaf has at its farthest, and an unrooted one the same (its leaves are as far as that, or nearer, and
    // none lie closer to each other than their slice of the circle there), and by how many are to be marked or named.
    TV.leafDetail = function (layout, view, leafCount) {
        if (layout.kind === "rectangular") {
            return { dots: view.sy >= TV.DOT_PX, labels: view.sy >= TV.LABEL_PX, size: view.sy };
        }
        if (layout.kind === "circular") {
            const slot = layout.reach[layout.reach.length - 1] * layout.slot * view.sx;
            return { dots: slot >= TV.DOT_PX, labels: slot >= TV.LABEL_PX, size: slot };
        }
        const slot = layout.far[layout.far.length - 1] * layout.slot * view.sx;
        return { dots: slot >= TV.DOT_PX && leafCount <= TV.UNROOTED_DOTS, labels: slot >= TV.LABEL_PX && leafCount <= TV.UNROOTED_LABELS, size: 14 };
    };

    // A rectangular tree is all horizontal and vertical lines, and a line that falls on the middle of a pixel row
    // draws many times faster than one that is spread over two (60 times, measured, where there is no GPU), so its
    // coordinates are put on the middle of the device pixel they fall in. Every line of the page goes through the
    // same function, so those that meet still meet. The other layouts have lines at every angle, which cannot be.
    TV.snapFor = function (layout) {
        if (layout.kind !== "rectangular") return null;
        const ratio = window.devicePixelRatio || 1;
        return (value) => (Math.floor(value * ratio) + 0.5) / ratio;
    };

    // The rectangle that a line must touch to be worth adding to a path: the canvas and a margin round it. A visited
    // clade's own line can lie far outside it (the clade is visited for its box, which is bigger than its line, in a
    // circular or unrooted tree), and a path with thousands of lines nobody sees costs where there is no GPU.
    let drawW = Infinity;
    let drawH = Infinity;
    const DRAW_MARGIN = 8;

    function offScreen(x0, y0, x1, y1) {
        return (x0 < -DRAW_MARGIN && x1 < -DRAW_MARGIN) || (x0 > drawW + DRAW_MARGIN && x1 > drawW + DRAW_MARGIN)
            || (y0 < -DRAW_MARGIN && y1 < -DRAW_MARGIN) || (y0 > drawH + DRAW_MARGIN && y1 > drawH + DRAW_MARGIN);
    }

    function addStem(path, out, view, snap) {
        let x0 = view.ox + out[0] * view.sx, y0 = view.oy + out[1] * view.sy;
        let x1 = view.ox + out[2] * view.sx, y1 = view.oy + out[3] * view.sy;
        if (offScreen(x0, y0, x1, y1)) return;
        if (snap) { x0 = snap(x0); y0 = snap(y0); x1 = snap(x1); y1 = snap(y1); }
        path.moveTo(x0, y0);
        path.lineTo(x1, y1);
    }

    // The join of node i's children, in a path (an arc is about the origin, which the view puts at (ox, oy)).
    function addJoin(path, out, view, snap) {
        if (out[0] === 1) {
            let x0 = view.ox + out[1] * view.sx, y0 = view.oy + out[2] * view.sy;
            let x1 = view.ox + out[3] * view.sx, y1 = view.oy + out[4] * view.sy;
            if (offScreen(x0, y0, x1, y1)) return;
            if (snap) { x0 = snap(x0); y0 = snap(y0); x1 = snap(x1); y1 = snap(y1); }
            path.moveTo(x0, y0);
            path.lineTo(x1, y1);
        } else if (out[0] === 2) {
            // A circle that is all outside the canvas, or all round it, has no arc in view.
            const radius = out[1] * view.sx;
            const near = Math.hypot(Math.max(-view.ox, 0, view.ox - drawW), Math.max(-view.oy, 0, view.oy - drawH));
            const far = Math.hypot(Math.max(view.ox, drawW - view.ox), Math.max(view.oy, drawH - view.oy));
            if (radius + DRAW_MARGIN < near || radius - DRAW_MARGIN > far) return;
            path.moveTo(view.ox + out[1] * Math.cos(out[2]) * view.sx, view.oy + out[1] * Math.sin(out[2]) * view.sy);
            path.arc(view.ox, view.oy, radius, out[2], out[3]);
        }
    }

    // The shape of a collapsed clade, as a closed path.
    function addWedge(path, out, view) {
        if (out[0] === 0) {
            path.moveTo(view.ox + out[1] * view.sx, view.oy + out[2] * view.sy);
            path.lineTo(view.ox + out[3] * view.sx, view.oy + out[4] * view.sy);
            path.lineTo(view.ox + out[5] * view.sx, view.oy + out[6] * view.sy);
        } else {
            const radius = out[5] * view.sx;
            const cx = view.ox + out[3] * view.sx;
            const cy = view.oy + out[4] * view.sy;
            path.moveTo(view.ox + out[1] * view.sx, view.oy + out[2] * view.sy);
            path.lineTo(cx + radius * Math.cos(out[6]), cy + radius * Math.sin(out[6]));
            path.arc(cx, cy, radius, out[6], out[7]);
        }
        path.closePath();
    }
    TV.addWedge = addWedge;

    // The bar that shows what colours a folded clade holds: along the far edge of its wedge, one piece for each of
    // its colours, as long as the share of the clade's leaves that have it. `paths` holds a path for each colour.
    function addMixBar(paths, out, view, mix) {
        let at = 0;
        for (const [cls, share] of mix.parts) {
            let path = paths.get(cls);
            if (!path) paths.set(cls, path = new Path2D());
            const from = at;
            const to = Math.min(1, at + share);
            at = to;
            if (out[0] === 0) {
                // A triangle: its far edge runs from one corner to the other.
                const x0 = view.ox + out[3] * view.sx, y0 = view.oy + out[4] * view.sy;
                const x1 = view.ox + out[5] * view.sx, y1 = view.oy + out[6] * view.sy;
                path.moveTo(x0 + (x1 - x0) * from, y0 + (y1 - y0) * from);
                path.lineTo(x0 + (x1 - x0) * to, y0 + (y1 - y0) * to);
            } else {
                // A slice: its far edge is the arc, a little inside it so that the bar is not half outside the wedge.
                const cx = view.ox + out[3] * view.sx, cy = view.oy + out[4] * view.sy;
                const radius = Math.max(0, out[5] * view.sx - 2);
                const a = out[6] + (out[7] - out[6]) * from, b = out[6] + (out[7] - out[6]) * to;
                path.moveTo(cx + radius * Math.cos(a), cy + radius * Math.sin(a));
                path.arc(cx, cy, radius, a, b);
            }
        }
    }

    // The colour a folded clade is filled with when most of its leaves have one: that colour, washed out into the
    // background, as an opaque colour (cheaper to draw than a see-through one). Kept for the colours and theme in hand.
    function tintOf(colors, palette, cls) {
        const key = palette.background + "|" + cls;
        if (!colors.tints) colors.tints = new Map();
        let tint = colors.tints.get(key);
        if (tint === undefined) colors.tints.set(key, tint = TV.mixColors(colors.css[cls], palette.background, 0.45));
        return tint;
    }

    // Draw the tree of scene on the context, which is `width` x `height` CSS pixels: the lines, the shapes of the
    // collapsed and the tiny, and the dots, handles and names that have room. scene: { model, layout, view, palette,
    // collapsed, matches }. mini is true for the overview map, which draws only the shapes of the tree. Returns
    // what was drawn: { visited, bars, collapsed, labels, dots, handles }, and in `pieces` what the callers add to.
    TV.paintTree = function (ctx, scene, width, height, mini) {
        const { model, layout, view, palette, collapsed } = scene;
        const stats = { visited: 0, bars: 0, collapsed: 0, labels: 0, dots: 0, handles: 0 };
        const { sx, sy, ox, oy } = view;
        const boxes = layout.boxes;
        const { parent, childCount, childStart, childList, firstLeaf, lastLeaf, leafOrdinal, viewerNode } = model;
        const margin = 8;
        const rectangular = layout.kind === "rectangular";
        const labelsPossible = !mini && (rectangular ? view.sy >= TV.LABEL_PX : layout.kind === "circular"
            ? TV.leafDetail(layout, view, 0).labels : true);
        // Names reach out of a leaf: to the right of a rectangular tree, to any side of the others.
        const padOut = labelsPossible ? LABEL_REACH_PX : 0;
        const padLeft = padOut;
        const padRight = rectangular ? 0 : padOut;
        const limit = mini ? 2 : TV.COLLAPSE_PX;
        const matches = scene.matches;
        drawW = width;
        drawH = height;
        // The colours of the network, when they are for this tree: a node's class says what it is drawn in (NO_CLASS
        // for none, the colour of the branches).
        const colors = scene.colors && scene.colors.nodeClass.length === model.nodeCount ? scene.colors : null;
        const nodeClass = colors ? colors.nodeClass : null;
        const NO = TV.NO_CLASS;

        const branches = new Path2D();
        const faded = new Path2D();
        const wedges = new Path2D();
        const matched = new Path2D();
        const handles = [];
        const leaves = [];
        const out = new Float64Array(8);
        // What is drawn in a colour goes in the path of its class: one stroke (or fill) for each colour in view.
        const inks = new Map();
        const wedgeFills = new Map();
        const mixBars = new Map();
        const inkOf = (cls) => {
            if (cls === NO) return branches;
            let path = inks.get(cls);
            if (!path) inks.set(cls, path = new Path2D());
            return path;
        };
        // A leaf is drawn as part of the network when it is a node of it and, with colours, is not hidden.
        const shownLeaf = (k) => viewerNode[k] >= 0 && (!colors || colors.leafClass[k] !== NO);
        // A tiny clade is a line, in the bucket of its thickness (to the half pixel, one to three) and its colour: one
        // stroke for each.
        const marks = new Map();
        const snap = TV.snapFor(layout);

        const wedgeOf = (i) => {
            layout.wedge(i, out);
            addWedge(wedges, out, view);
            const cls = nodeClass ? nodeClass[i] : NO;
            let fill = wedgeFills.get(cls);
            if (!fill) wedgeFills.set(cls, fill = new Path2D());
            addWedge(fill, out, view);
            if (matches && matches.length && TV.matchesIn(matches, firstLeaf[i], lastLeaf[i])) addWedge(matched, out, view);
            if (colors) addMixBar(mixBars, out, view, TV.cladeMix(colors, model, i));
        };
        const markOf = (i) => {
            const width = Math.min(3, Math.max(1, Math.round(layout.thickness(i, sx, sy) * 2) / 2));
            const cls = nodeClass ? nodeClass[i] : NO;
            const key = width * 2 * 65536 + cls;
            let entry = marks.get(key);
            if (!entry) marks.set(key, entry = { path: new Path2D(), width, cls });
            layout.mark(i, out);
            addStem(entry.path, out, view, snap);
            if (matches && matches.length && TV.matchesIn(matches, firstLeaf[i], lastLeaf[i])) addStem(matched, out, view, snap);
        };

        const stack = [model.root];
        while (stack.length) {
            const i = stack.pop();
            const left = ox + boxes.x0[i] * sx;
            const right = ox + boxes.x1[i] * sx;
            const top = oy + boxes.y0[i] * sy;
            const bottom = oy + boxes.y1[i] * sy;
            if (bottom + (rectangular ? 0 : padOut) < -margin || top - (rectangular ? 0 : padOut) > height + margin) continue;
            if (right + padLeft < -margin || left - padRight > width + margin) continue;
            stats.visited++;
            layout.stem(i, out);
            // Where node i is, kept before the shapes below use the array for their own.
            const nodeX = ox + out[2] * sx;
            const nodeY = oy + out[3] * sy;

            const cls = nodeClass ? nodeClass[i] : NO;
            if (childCount[i] === 0) {
                addStem(shownLeaf(leafOrdinal[i]) ? inkOf(cls) : faded, out, view, snap);
                leaves.push(i);
            } else if (collapsed && collapsed[i]) {
                addStem(inkOf(cls), out, view, snap);
                wedgeOf(i);
                stats.collapsed++;
                if (!mini) handles.push([i, nodeX, nodeY, true]);
            } else if (layout.tiny(i, sx, sy, limit)) {
                addStem(inkOf(cls), out, view, snap);
                markOf(i);
                stats.bars++;
            } else {
                // A clade that is one colour is drawn in it, all its lines.
                const ink = inkOf(cls);
                addStem(ink, out, view, snap);
                layout.join(i, out);
                addJoin(ink, out, view, snap);
                if (!mini && (i === scene.hover || i === scene.pinned) && i !== model.root
                    && Math.max(right - left, bottom - top) >= TV.HANDLE_CLADE_PX) {
                    handles.push([i, nodeX, nodeY, false]);
                }
                for (let k = childStart[i]; k < childStart[i + 1]; k++) stack.push(childList[k]);
            }
        }

        // Everything is drawn opaque: a see-through shape costs several times as much where there is no GPU.
        ctx.lineWidth = 1;
        ctx.lineCap = "butt";
        for (const [cls, path] of wedgeFills) {
            ctx.fillStyle = cls === NO ? palette.wedge : tintOf(colors, palette, cls);
            ctx.fill(path);
        }
        for (const { path, width, cls } of marks.values()) {
            ctx.strokeStyle = cls === NO ? palette.mark : colors.css[cls];
            ctx.lineWidth = width;
            ctx.stroke(path);
        }
        ctx.strokeStyle = palette.mark;
        ctx.lineWidth = 1;
        ctx.stroke(wedges);
        // The colours a folded clade holds, along its far edge.
        if (mixBars.size) {
            ctx.lineWidth = mini ? 2 : 4;
            for (const [cls, path] of mixBars) {
                ctx.strokeStyle = colors.css[cls];
                ctx.stroke(path);
            }
            ctx.lineWidth = 1;
        }
        ctx.strokeStyle = palette.faded;
        ctx.stroke(faded);
        ctx.strokeStyle = palette.branch;
        ctx.stroke(branches);
        for (const [cls, path] of inks) {
            ctx.strokeStyle = colors.css[cls];
            ctx.stroke(path);
        }
        ctx.lineWidth = mini ? 1.5 : 2;
        ctx.strokeStyle = palette.match;
        ctx.stroke(matched);
        if (mini) return stats;

        // The leaves drawn on their own: a dot where there is room, a name where there is more.
        const detail = TV.leafDetail(layout, view, leaves.length);
        const dotRadius = Math.min(3.2, Math.max(1.1, detail.size * 0.25));
        const dots = new Path2D();
        const fadedDots = new Path2D();
        const dotsBy = new Map();
        const names = [];
        for (const i of leaves) {
            layout.stem(i, out);
            const x = ox + out[2] * sx;
            const y = oy + out[3] * sy;
            const inNetwork = shownLeaf(leafOrdinal[i]);
            if (detail.dots) {
                let path = inNetwork ? dots : fadedDots;
                if (inNetwork && nodeClass && nodeClass[i] !== NO) {
                    path = dotsBy.get(nodeClass[i]);
                    if (!path) dotsBy.set(nodeClass[i], path = new Path2D());
                }
                path.moveTo(x + dotRadius, y);
                path.arc(x, y, dotRadius, 0, 2 * Math.PI);
                stats.dots++;
            }
            if (detail.labels) {
                // The direction the leaf points in, for a name that runs away from the tree.
                const direction = rectangular ? 0 : layout.kind === "circular" ? layout.angle[i]
                    : Math.atan2(out[3] - out[1], out[2] - out[0]);
                names.push([model.labels[leafOrdinal[i]], x, y, inNetwork, direction]);
            }
        }
        if (detail.dots) {
            ctx.fillStyle = palette.faded;
            ctx.fill(fadedDots);
            ctx.fillStyle = palette.branch;
            ctx.fill(dots);
            for (const [cls, path] of dotsBy) {
                ctx.fillStyle = colors.css[cls];
                ctx.fill(path);
            }
        }

        // The handles that fold a clade up (a minus) and the ones that open it again (a plus).
        const half = TV.HANDLE_PX / 2;
        for (const [, x, y, isCollapsed] of handles) {
            ctx.fillStyle = palette.background;
            ctx.strokeStyle = palette.branch;
            ctx.lineWidth = 1;
            ctx.fillRect(x - half, y - half, TV.HANDLE_PX, TV.HANDLE_PX);
            ctx.strokeRect(x - half + 0.5, y - half + 0.5, TV.HANDLE_PX - 1, TV.HANDLE_PX - 1);
            ctx.beginPath();
            ctx.moveTo(x - half + 2.5, y);
            ctx.lineTo(x + half - 2.5, y);
            if (isCollapsed) { ctx.moveTo(x, y - half + 2.5); ctx.lineTo(x, y + half - 2.5); }
            ctx.stroke();
            stats.handles++;
        }

        if (names.length) {
            ctx.font = Math.min(13, Math.max(9, detail.size * 0.8)).toFixed(1) + "px " + palette.font;
            ctx.textBaseline = "middle";
            for (const [text, x, y, inNetwork, direction] of names) {
                ctx.fillStyle = inNetwork ? palette.label : palette.faded;
                const offset = dotRadius + 4;
                if (rectangular) {
                    ctx.textAlign = "left";
                    ctx.fillText(shortened(text), x + offset, y);
                } else if (layout.kind === "circular") {
                    // Along the radius, turned over on the left half so that no name is upside down.
                    const flipped = Math.cos(direction) < 0;
                    ctx.save();
                    ctx.translate(x, y);
                    ctx.rotate(flipped ? direction + Math.PI : direction);
                    ctx.textAlign = flipped ? "right" : "left";
                    ctx.fillText(shortened(text), flipped ? -offset : offset, 0);
                    ctx.restore();
                } else {
                    const toTheLeft = Math.cos(direction) < 0;
                    ctx.textAlign = toTheLeft ? "right" : "left";
                    ctx.fillText(shortened(text), toTheLeft ? x - offset : x + offset, y);
                }
            }
            stats.labels = names.length;
        }
        stats.leafSize = detail.size;
        return stats;
    };

    // The highlight behind the lines for node i: a band across a rectangular tree, the shape of the clade in the others.
    function drawHighlight(ctx, scene, node, fill) {
        const { model, layout, view } = scene;
        ctx.fillStyle = fill;
        if (layout.kind === "rectangular") {
            const box = TV.nodePixels(scene, node);
            const named = view.sy >= TV.LABEL_PX;
            ctx.fillRect(box.x0, box.top, Math.max(2, box.reach - box.x0) + (named ? LABEL_REACH_PX : 6), Math.max(2, box.bottom - box.top));
            return;
        }
        if (model.childCount[node] === 0) {
            const box = TV.nodePixels(scene, node);
            ctx.beginPath();
            ctx.arc(box.x, box.y, 9, 0, 2 * Math.PI);
            ctx.fill();
            return;
        }
        const path = new Path2D();
        layout.wedge(node, scratch);
        addWedge(path, scratch, view);
        ctx.fill(path);
    }

    // Draw the scene on the canvas: the background, the highlights, the tree, then the hovered branch, the search's
    // matches, and for a rectangular tree the ruler. Returns what was drawn, for the tests and the status line.
    TV.drawTree = function (canvas, scene) {
        const { width, height, ratio } = TV.sizeCanvas(canvas);
        const ctx = canvas.getContext("2d");
        const palette = scene.palette;
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.fillStyle = palette.background;
        ctx.fillRect(0, 0, width, height);
        const { model, layout, view } = scene;
        if (!model || !layout) return { visited: 0, bars: 0, collapsed: 0, labels: 0, dots: 0, handles: 0, width, height };

        for (const [node, fill] of [[scene.pinned, palette.selection], [scene.hover, palette.selection]]) {
            if (node !== undefined && node >= 0) drawHighlight(ctx, scene, node, fill);
        }
        const stats = TV.paintTree(ctx, scene, width, height, false);
        stats.width = width;
        stats.height = height;

        // The hovered branch, and the search's matches, over the lines.
        if (scene.hover !== undefined && scene.hover >= 0 && !(scene.collapsed && scene.collapsed[scene.hover])) {
            ctx.strokeStyle = palette.accent;
            ctx.lineWidth = 2;
            const path = new Path2D();
            const snap = TV.snapFor(layout);
            layout.stem(scene.hover, scratch);
            addStem(path, scratch, view, snap);
            layout.join(scene.hover, scratch);
            addJoin(path, scratch, view, snap);
            ctx.stroke(path);
        }
        TV.drawMatches(ctx, scene, width, height);
        if (layout.kind === "rectangular") TV.drawRuler(ctx, scene, width, height);
        return stats;
    };

    // A ring at the tip of each leaf the search found that is in view; a short tick where its row is too thin for one.
    // A leaf inside a collapsed clade has no tip: the clade's shape is outlined instead (paintTree).
    TV.drawMatches = function (ctx, scene, width, height) {
        const matches = scene.matches;
        if (!matches || !matches.length) return;
        const { model, layout, view, palette } = scene;
        const rectangular = layout.kind === "rectangular";
        const hidden = scene.leafHidden;
        // A rectangular tree's matches are sorted by row, so those above the view are skipped by a search.
        const first = rectangular ? lowerBound(matches, Math.floor((-4 - view.oy) / view.sy)) : 0;
        // A thin row, or so many matches that rings would be a smear, get a small mark instead of a ring.
        const thin = rectangular ? view.sy < 6 : matches.length > 1500;
        ctx.lineWidth = 1.5;
        for (let m = first; m < matches.length; m++) {
            const leaf = matches[m];
            if (hidden && hidden[leaf]) continue;
            const node = model.leafNode[leaf];
            const y = view.oy + layout.y[node] * view.sy;
            if (rectangular && y > height + 8) break;
            const x = view.ox + layout.x[node] * view.sx;
            if (x < -LABEL_REACH_PX || x > width + 8 || y < -8 || y > height + 8) continue;
            const current = m === scene.current;
            ctx.strokeStyle = current ? palette.accent : palette.match;
            ctx.fillStyle = palette.matchSoft;
            if (thin) {
                ctx.fillStyle = current ? palette.accent : palette.match;
                if (rectangular) ctx.fillRect(x, y - 1, current ? 22 : 14, 2); else ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
            } else {
                const size = rectangular ? view.sy : TV.leafDetail(layout, view, 1).size;
                const radius = Math.min(8, Math.max(3.5, size * 0.45));
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
        }
    };

    // The strip at the right edge of a rectangular tree: the whole tree top to bottom, the rows in view as a thumb, a
    // tick for each match.
    TV.drawRuler = function (ctx, scene, width, height) {
        const { model, view, palette } = scene;
        const left = width - TV.RULER_WIDTH;
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = palette.ruler;
        ctx.fillRect(left, 0, TV.RULER_WIDTH, height);
        ctx.globalAlpha = 1;
        const rows = model.leafCount;
        const top = Math.max(0, -view.oy / view.sy / rows) * height;
        const bottom = Math.min(1, (height - view.oy) / view.sy / rows) * height;
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = palette.thumb;
        ctx.fillRect(left + 1, top, TV.RULER_WIDTH - 2, Math.max(6, bottom - top));
        ctx.globalAlpha = 1;
        if (scene.matches && scene.matches.length) {
            ctx.fillStyle = palette.match;
            // One tick per pixel row at most: a few thousand matches cost no more than the ruler is tall.
            let drawn = -1;
            for (let m = 0; m < scene.matches.length; m++) {
                const y = Math.round((scene.matches[m] + 0.5) / rows * height);
                if (y !== drawn) {
                    ctx.fillRect(left + 2, y - 1, TV.RULER_WIDTH - 4, 2);
                    drawn = y;
                }
            }
        }
    };
})(window.TreeView);
