/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's navigation: the layout in use, the view (zoom and pan), what is folded up, what is under the pointer,
 * the search and the tooltip.
 *
 * A rectangular tree zooms its rows and its width apart, as a long tree needs far more zoom in its rows than in its
 * width: the wheel zooms the rows about the pointer, Shift and the wheel the width, Ctrl (or a pinch) both. The
 * circular and the unrooted tree have one scale. Double-click a clade to bring it to fill the view. A square at a
 * clade's root folds the clade up into one shape, and a click on the shape opens it again. What is under the pointer
 * is found by going down the tree only into the clades whose box holds the pointer, so a look-up costs what is
 * near the pointer and not the size of the tree.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const state = TV.state = {
        model: null,
        layout: null,
        kind: "rectangular",
        mode: "lengths",
        view: { sx: 1, sy: 1, ox: 0, oy: 0 },
        fit: { sx: 1, sy: 1, ox: 0, oy: 0 },
        fitted: true,
        hover: -1,
        pinned: -1,
        palette: null,
        matches: new Int32Array(0),
        current: -1,
        query: "",
        // Which clades are folded up (1 at the clade's root), how many, and which leaves are inside one.
        collapsed: null,
        collapsedCount: 0,
        leafHidden: null,
    };
    TV.canvas = null;
    TV.lastStats = null;

    // How close, in pixels, the pointer must be to a line to be on it.
    const HIT_PX = 5;
    const MAX_ROW_PX = 80;
    // How many leaves, at most, a circular or unrooted tree has in view when one is brought up to be read: few
    // enough for their dots and names to be told apart.
    const READABLE_LEAVES = 60;
    const PAD = { left: 24, right: 24, top: 16, bottom: 16 };
    const TWO_PI = 2 * Math.PI;

    function clamp(value, low, high) {
        return Math.min(high, Math.max(low, value));
    }

    TV.limits = function () {
        const fit = state.fit;
        if (state.layout && state.layout.uniform) {
            return { sxMin: fit.sx * 0.25, sxMax: fit.sx * 8000, syMin: fit.sy * 0.25, syMax: fit.sy * 8000 };
        }
        return {
            sxMin: fit.sx * 0.25,
            sxMax: fit.sx * 2000,
            syMin: Math.min(fit.sy, MAX_ROW_PX) * 0.25,
            syMax: Math.max(MAX_ROW_PX, fit.sy),
        };
    };

    // Keep some of the tree in view, so that it cannot be dragged out of sight and lost.
    function keepInView() {
        const view = state.view;
        const bounds = state.layout.bounds;
        const width = TV.canvas.clientWidth;
        const height = TV.canvas.clientHeight;
        view.ox = clamp(view.ox, 60 - 300 - bounds.maxX * view.sx, width - 60 - bounds.minX * view.sx);
        view.oy = clamp(view.oy, 60 - bounds.maxY * view.sy, height - 60 - bounds.minY * view.sy);
    }

    // The view that shows the whole tree: a rectangular tree from the left, centred when it is short; the others
    // centred, in one scale.
    TV.fitView = function () {
        if (!state.layout || !TV.canvas) return;
        const layout = state.layout;
        const bounds = layout.bounds;
        const width = TV.canvas.clientWidth;
        const height = TV.canvas.clientHeight;
        if (layout.uniform) {
            let scale = Math.min((width - PAD.left - PAD.right) / (bounds.maxX - bounds.minX),
                                 (height - PAD.top - PAD.bottom) / (bounds.maxY - bounds.minY));
            // No more than a leaf in forty pixels of the circle: a short tree is not drawn as a giant one.
            if (layout.kind === "circular") scale = Math.min(scale, 40 / (layout.reach[layout.reach.length - 1] * layout.slot));
            scale = Math.max(scale, 1e-6);
            state.view = {
                sx: scale, sy: scale,
                ox: width / 2 - (bounds.minX + bounds.maxX) / 2 * scale,
                oy: height / 2 - (bounds.minY + bounds.maxY) / 2 * scale,
            };
        } else {
            const span = layout.width > 0 ? layout.width : 1;
            const sx = Math.max(1e-6, (width - PAD.left - PAD.right - TV.RULER_WIDTH) / span);
            const sy = clamp((height - PAD.top - PAD.bottom) / layout.height, 1e-6, 28);
            state.view = { sx, sy, ox: PAD.left, oy: Math.max(PAD.top, (height - sy * layout.height) / 2) };
        }
        state.fit = { ...state.view };
        state.fitted = true;
        TV.requestDraw();
    };

    // Draw the tree in another layout (a kind of TV.LAYOUT_KINDS, and "lengths" or "cladogram") and show all of it.
    // What is folded up stays folded, and what is selected stays so: the nodes are the same.
    TV.setLayout = function (kind, mode) {
        if (!state.model) return;
        state.layout = TV.layoutTree(state.model, kind, mode);
        state.kind = kind;
        state.mode = state.layout.mode;
        state.hover = -1;
        TV.minimapChanged();
        TV.fitView();
    };

    // Zoom by factors fx and fy about the pixel (px, py). A layout with one scale takes the one factor that is not 1.
    TV.zoomAbout = function (px, py, fx, fy) {
        if (!state.layout) return;
        if (state.layout.uniform) fx = fy = fx !== 1 ? fx : fy;
        const view = state.view;
        const limit = TV.limits();
        const sx = clamp(view.sx * fx, limit.sxMin, limit.sxMax);
        const sy = state.layout.uniform ? sx : clamp(view.sy * fy, limit.syMin, limit.syMax);
        view.ox = px - (px - view.ox) / view.sx * sx;
        view.oy = py - (py - view.oy) / view.sy * sy;
        view.sx = sx;
        view.sy = sy;
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    TV.pan = function (dx, dy) {
        if (!state.layout) return;
        state.view.ox += dx;
        state.view.oy += dy;
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    // Bring node i's clade, with the branch that leads to it, to fill the view.
    TV.zoomToNode = function (i) {
        const { model, layout } = state;
        if (!model || !(i >= 0 && i < model.nodeCount)) return;
        if (model.childCount[i] === 0) {
            TV.focusLeaf(model.leafOrdinal[i]);
            return;
        }
        const width = TV.canvas.clientWidth;
        const height = TV.canvas.clientHeight;
        const limit = TV.limits();
        const boxes = layout.boxes;
        const room = { left: 24, right: TV.RULER_WIDTH + 24, top: 24, bottom: 24 };
        const x0 = boxes.x0[i];
        const y0 = boxes.y0[i];
        if (layout.uniform) {
            const spanX = Math.max(boxes.x1[i] - x0, 1e-9);
            const spanY = Math.max(boxes.y1[i] - y0, 1e-9);
            const labelRoom = Math.min(160, width * 0.12);
            const scale = clamp(Math.min((width - 2 * room.left - 2 * labelRoom) / spanX, (height - room.top - room.bottom) / spanY),
                                limit.sxMin, limit.sxMax);
            state.view = {
                sx: scale, sy: scale,
                ox: width / 2 - (x0 + boxes.x1[i]) / 2 * scale,
                oy: height / 2 - (y0 + boxes.y1[i]) / 2 * scale,
            };
        } else {
            const rows = model.lastLeaf[i] - model.firstLeaf[i] + 1;
            const sy = clamp((height - room.top - room.bottom) / rows, limit.syMin, limit.syMax);
            const labelRoom = sy >= TV.LABEL_PX ? Math.min(300, width * 0.3) : 0;
            const span = Math.max(layout.reach[i] - x0, layout.width * 1e-3, 1e-9);
            const sx = clamp((width - room.left - room.right - labelRoom) / span, limit.sxMin, limit.sxMax);
            state.view = {
                sx,
                sy,
                ox: room.left - x0 * sx,
                oy: room.top + (height - room.top - room.bottom - rows * sy) / 2 - model.firstLeaf[i] * sy,
            };
        }
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    // How many leaves have their tips in the view.
    TV.leavesInView = function () {
        const { model, layout, view } = state;
        if (!model || !layout || !TV.canvas) return 0;
        const width = TV.canvas.clientWidth;
        const height = TV.canvas.clientHeight;
        let count = 0;
        for (let k = 0; k < model.leafCount; k++) {
            const node = model.leafNode[k];
            const px = view.ox + layout.x[node] * view.sx;
            const py = view.oy + layout.y[node] * view.sy;
            if (px >= 0 && px <= width && py >= 0 && py <= height) count++;
        }
        return count;
    };

    // Put leaf number k in the middle of the view, where its name can be read: a rectangular tree at a row tall enough,
    // the others in the clade of a few leaves round it, and then in until a few dozen leaves are in view. A leaf
    // inside a folded clade opens the clade.
    TV.focusLeaf = function (k) {
        const { model, layout, view } = state;
        if (!model) return;
        const node = model.leafNode[k];
        TV.expandToShow(node);
        if (layout.uniform) {
            if (model.nodeCount === 1) {
                TV.fitView();
                return;
            }
            let around = model.parent[node];
            while (around >= 0 && TV.leavesUnder(model, around) < 16 && model.parent[around] >= 0) around = model.parent[around];
            TV.zoomToNode(around >= 0 ? around : model.root);
            // A clade of sixteen leaves can reach across the whole tree, and its leaves still be specks.
            if (TV.leavesInView() > READABLE_LEAVES) {
                TV.centerOn(layout.x[node], layout.y[node]);
                for (let step = 0; step < 80 && TV.leavesInView() > READABLE_LEAVES; step++) {
                    const before = state.view.sx;
                    TV.zoomAbout(TV.canvas.clientWidth / 2, TV.canvas.clientHeight / 2, 1.4, 1.4);
                    if (state.view.sx === before) break;
                }
            }
            return;
        }
        const limit = TV.limits();
        view.sy = clamp(Math.max(view.sy, 16), limit.syMin, limit.syMax);
        view.oy = TV.canvas.clientHeight / 2 - (k + 0.5) * view.sy;
        const tip = view.ox + layout.x[node] * view.sx;
        if (tip < 40 || tip > TV.canvas.clientWidth - 300) {
            view.ox = TV.canvas.clientWidth * 0.5 - layout.x[node] * view.sx;
        }
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    // Move the view so that the rows at the fraction `fraction` (0 to 1) of a rectangular tree are in its middle: the
    // ruler's jump.
    TV.jumpToFraction = function (fraction) {
        if (!state.layout || state.layout.kind !== "rectangular") return;
        const view = state.view;
        view.oy = TV.canvas.clientHeight / 2 - clamp(fraction, 0, 1) * state.layout.height * view.sy;
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    // Move the view so that the point (x, y) of the layout is in the middle of it: the overview's jump.
    TV.centerOn = function (x, y) {
        if (!state.layout) return;
        const view = state.view;
        view.ox = TV.canvas.clientWidth / 2 - x * view.sx;
        view.oy = TV.canvas.clientHeight / 2 - y * view.sy;
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    // ---- Folding clades up ----------------------------------------------------------------------------------

    // Which leaves are inside a folded clade, from the clades folded.
    function foldedLeaves() {
        const { model, collapsed } = state;
        const difference = new Int32Array(model.leafCount + 1);
        let count = 0;
        for (let i = 0; i < model.nodeCount; i++) {
            if (!collapsed[i]) continue;
            count++;
            difference[model.firstLeaf[i]]++;
            difference[model.lastLeaf[i] + 1]--;
        }
        const hidden = new Uint8Array(model.leafCount);
        let inside = 0;
        for (let k = 0; k < model.leafCount; k++) {
            inside += difference[k];
            hidden[k] = inside > 0 ? 1 : 0;
        }
        state.collapsedCount = count;
        state.leafHidden = count ? hidden : null;
    }

    // Forget what is folded: the model is a new one.
    TV.resetFolds = function () {
        state.collapsed = state.model ? new Uint8Array(state.model.nodeCount) : null;
        state.collapsedCount = 0;
        state.leafHidden = null;
    };

    // Whether node d is inside the clade of node i (its own subtree is the block of nodes that ends at i).
    TV.isInside = function (d, i) {
        return d < i && d > i - state.model.size[i];
    };

    // Fold clade i up into one shape, or open it again. The root has no clade to fold. Returns whether it changed.
    TV.toggleFold = function (i) {
        const { model } = state;
        if (!model || i < 0 || i === model.root || model.childCount[i] === 0) return false;
        state.collapsed[i] = state.collapsed[i] ? 0 : 1;
        foldedLeaves();
        // What was selected or pointed at inside a clade that is now one shape is the shape.
        if (state.collapsed[i]) {
            if (state.pinned >= 0 && TV.isInside(state.pinned, i)) state.pinned = i;
            if (state.hover >= 0 && TV.isInside(state.hover, i)) state.hover = i;
        }
        TV.minimapChanged();
        TV.requestDraw();
        if (TV.onFoldsChanged) TV.onFoldsChanged();
        return true;
    };

    TV.openAll = function () {
        if (!state.model || !state.collapsedCount) return false;
        state.collapsed.fill(0);
        foldedLeaves();
        TV.minimapChanged();
        TV.requestDraw();
        if (TV.onFoldsChanged) TV.onFoldsChanged();
        return true;
    };

    // Open every folded clade that holds node d, so that it can be seen. Returns whether any was.
    TV.expandToShow = function (d) {
        const { model } = state;
        if (!model || !state.collapsedCount) return false;
        let opened = false;
        for (let a = model.parent[d]; a >= 0; a = model.parent[a]) {
            if (state.collapsed[a]) {
                state.collapsed[a] = 0;
                opened = true;
            }
        }
        if (opened) {
            foldedLeaves();
            TV.minimapChanged();
            if (TV.onFoldsChanged) TV.onFoldsChanged();
        }
        return opened;
    };

    // ---- Drawing, a frame at a time --------------------------------------------------------------------------

    let drawQueued = false;

    // The scene the drawing is given: the state, whose view and layout are the ones in use.
    TV.drawNow = function () {
        drawQueued = false;
        if (!TV.canvas) return null;
        if (!state.palette) state.palette = TV.readPalette();
        TV.lastStats = TV.drawTree(TV.canvas, state);
        if (TV.drawMinimap) TV.drawMinimap();
        return TV.lastStats;
    };

    // Draw at the next frame, or after a moment where frames do not come (a page behind others).
    TV.requestDraw = function () {
        if (drawQueued) return;
        drawQueued = true;
        requestAnimationFrame(() => { if (drawQueued) TV.drawNow(); });
        setTimeout(() => { if (drawQueued) TV.drawNow(); }, 100);
    };

    // Tell the overview that what it draws of the tree changed (the layout, the folds, the matches, the theme).
    TV.minimapChanged = function () {
        if (TV.invalidateMinimap) TV.invalidateMinimap();
    };

    // ---- What is under the pointer ---------------------------------------------------------------------------

    const measure = document.createElement("canvas").getContext("2d");

    // The width in pixels of a leaf's name at the size the page draws it in.
    function nameWidth(text, size) {
        measure.font = Math.min(13, Math.max(9, size * 0.8)).toFixed(1) + "px " + (state.palette ? state.palette.font : "sans-serif");
        return measure.measureText(text.length > TV.MAX_LABEL_CHARACTERS ? text.slice(0, TV.MAX_LABEL_CHARACTERS) : text).width;
    }

    // Pixels from the point (px, py) to the segment (x0, y0)-(x1, y1).
    function segmentDistance(px, py, x0, y0, x1, y1) {
        const dx = x1 - x0;
        const dy = y1 - y0;
        const squared = dx * dx + dy * dy;
        const t = squared > 0 ? clamp(((px - x0) * dx + (py - y0) * dy) / squared, 0, 1) : 0;
        return Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
    }

    // Pixels from the point to the arc about (cx, cy) of radius r, from angle a0 to a1 (a0 <= a1, less than a turn apart).
    function arcDistance(px, py, cx, cy, r, a0, a1) {
        const dx = px - cx;
        const dy = py - cy;
        const turn = (((Math.atan2(dy, dx) - a0) % TWO_PI) + TWO_PI) % TWO_PI;
        if (turn <= a1 - a0) return Math.abs(Math.hypot(dx, dy) - r);
        return Math.min(Math.hypot(px - (cx + r * Math.cos(a0)), py - (cy + r * Math.sin(a0))),
                        Math.hypot(px - (cx + r * Math.cos(a1)), py - (cy + r * Math.sin(a1))));
    }

    // Pixels from the point to a wedge (layout.wedge's out): 0 inside it.
    function wedgeDistance(out, px, py, view) {
        if (out[0] === 0) {
            const ax = view.ox + out[1] * view.sx, ay = view.oy + out[2] * view.sy;
            const bx = view.ox + out[3] * view.sx, by = view.oy + out[4] * view.sy;
            const cx = view.ox + out[5] * view.sx, cy = view.oy + out[6] * view.sy;
            const side = (x0, y0, x1, y1) => (px - x1) * (y0 - y1) - (x0 - x1) * (py - y1);
            const d1 = side(ax, ay, bx, by);
            const d2 = side(bx, by, cx, cy);
            const d3 = side(cx, cy, ax, ay);
            if (!((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0))) return 0;
            return Math.min(segmentDistance(px, py, ax, ay, bx, by), segmentDistance(px, py, bx, by, cx, cy),
                            segmentDistance(px, py, cx, cy, ax, ay));
        }
        // A slice: between the radius of its point and its own radius, and between its two angles, about the centre of its arc.
        const cx = view.ox + out[3] * view.sx;
        const cy = view.oy + out[4] * view.sy;
        const radius = out[5] * view.sx;
        const dx = px - cx;
        const dy = py - cy;
        const r = Math.hypot(dx, dy);
        const apexR = Math.hypot(view.ox + out[1] * view.sx - cx, view.oy + out[2] * view.sy - cy);
        const across = r > radius ? r - radius : r < apexR ? apexR - r : 0;
        const turn = (((Math.atan2(dy, dx) - out[6]) % TWO_PI) + TWO_PI) % TWO_PI;
        const span = out[7] - out[6];
        const along = turn <= span || r < 1 ? 0 : r * Math.min(turn - span, TWO_PI - turn);
        return Math.hypot(across, along);
    }

    // Whether the last frame, or the view, has room to name the leaves.
    function namesShown() {
        const { layout, view } = state;
        if (layout.kind === "rectangular") return view.sy >= TV.LABEL_PX;
        if (layout.kind === "circular") return TV.leafDetail(layout, view, 0).labels;
        return !!(TV.lastStats && TV.lastStats.labels > 0);
    }

    // The node at the pixel (px, py) and which part of it is there, or null: { node, part } with part "leaf" (its
    // branch, dot or name), "branch", "connector" (the line that joins a clade's children), "clade" (a clade drawn as
    // one shape), or "handle" (the square that folds a clade up or opens it). Goes down only into the clades whose box
    // holds the point, and takes the nearest line.
    TV.hitTest = function (px, py) {
        const { model, layout, view, collapsed } = state;
        if (!model || !layout) return null;
        const boxes = layout.boxes;
        const rectangular = layout.kind === "rectangular";
        const named = namesShown();
        // Whether there is room for dots and names: a row or a slot says it, an unrooted tree has the last frame's word.
        const detail = layout.kind === "unrooted"
            ? { dots: !!(TV.lastStats && TV.lastStats.dots > 0), labels: named, size: 14 }
            : TV.leafDetail(layout, view, 0);
        const pad = named ? 520 : 0;
        const out = new Float64Array(8);
        const stemOut = new Float64Array(8);
        let best = null;
        let bestDistance = HIT_PX + 1e-9;
        const offer = (node, part, distance) => {
            if (distance < bestDistance || (distance === bestDistance && best === null)) {
                best = { node, part };
                bestDistance = distance;
            }
        };

        const stack = [model.root];
        while (stack.length) {
            const i = stack.pop();
            const left = view.ox + boxes.x0[i] * view.sx;
            const right = view.ox + boxes.x1[i] * view.sx;
            const top = view.oy + boxes.y0[i] * view.sy;
            const bottom = view.oy + boxes.y1[i] * view.sy;
            // Names reach out of the box: right of a rectangular one, any side of the others.
            if (px < left - HIT_PX - (rectangular ? 0 : pad) || px > right + HIT_PX + pad
                || py < top - HIT_PX - (rectangular ? 0 : pad) || py > bottom + HIT_PX + (rectangular ? 0 : pad)) continue;
            layout.stem(i, stemOut);
            const x0 = view.ox + stemOut[0] * view.sx;
            const y0 = view.oy + stemOut[1] * view.sy;
            const x = view.ox + stemOut[2] * view.sx;
            const y = view.oy + stemOut[3] * view.sy;

            if (model.childCount[i] === 0) {
                // A leaf: its branch (a rectangular row is as tall as its row), its dot, and its name.
                if (rectangular) {
                    const tolerance = Math.max(HIT_PX, view.sy / 2);
                    if (Math.abs(py - y) <= tolerance && px >= x0 - HIT_PX && px <= x + HIT_PX) offer(i, "leaf", Math.abs(py - y) * HIT_PX / tolerance);
                } else {
                    offer(i, "leaf", segmentDistance(px, py, x0, y0, x, y));
                }
                if (detail.dots && Math.hypot(px - x, py - y) <= HIT_PX) offer(i, "leaf", 0);
                if (named) {
                    const text = model.labels[model.leafOrdinal[i]];
                    const width = nameWidth(text, detail.size);
                    const offset = Math.min(3.2, Math.max(1.1, detail.size * 0.25)) + 4;
                    const half = Math.max(HIT_PX, Math.min(13, Math.max(9, detail.size * 0.8)) / 2 + 2);
                    let hit = false;
                    if (rectangular) {
                        hit = Math.abs(py - y) <= Math.max(HIT_PX, view.sy / 2) && px >= x && px <= x + offset + width;
                    } else {
                        // Along the leaf's own direction in a circular tree, level in an unrooted one.
                        const direction = layout.kind === "circular" ? layout.angle[i] : Math.atan2(y - y0, x - x0);
                        if (layout.kind === "circular") {
                            const along = (px - x) * Math.cos(direction) + (py - y) * Math.sin(direction);
                            const across = -(px - x) * Math.sin(direction) + (py - y) * Math.cos(direction);
                            hit = along >= 0 && along <= offset + width && Math.abs(across) <= half;
                        } else {
                            const toTheLeft = Math.cos(direction) < 0;
                            hit = Math.abs(py - y) <= half && (toTheLeft ? px >= x - offset - width && px <= x : px >= x && px <= x + offset + width);
                        }
                    }
                    if (hit) offer(i, "leaf", 0);
                }
                continue;
            }

            // The clade drawn as one shape (folded up, or too small to be more): the shape, and a handle to open it.
            const folded = collapsed && collapsed[i];
            if (folded || layout.tiny(i, view.sx, view.sy, TV.COLLAPSE_PX)) {
                layout.wedge(i, out);
                offer(i, "clade", wedgeDistance(out, px, py, view));
                if (folded && Math.abs(px - x) <= TV.HANDLE_PX / 2 + 1 && Math.abs(py - y) <= TV.HANDLE_PX / 2 + 1) offer(i, "handle", -1);
                offer(i, "branch", segmentDistance(px, py, x0, y0, x, y));
                continue;
            }

            offer(i, "branch", segmentDistance(px, py, x0, y0, x, y));
            layout.join(i, out);
            if (out[0] === 1) {
                offer(i, "connector", segmentDistance(px, py, view.ox + out[1] * view.sx, view.oy + out[2] * view.sy,
                                                      view.ox + out[3] * view.sx, view.oy + out[4] * view.sy));
            } else if (out[0] === 2) {
                offer(i, "connector", arcDistance(px, py, view.ox, view.oy, out[1] * view.sx, out[2], out[3]));
            }
            // The square of a clade shows for the one that is hovered or selected, and is a target only then.
            if ((i === state.hover || i === state.pinned) && i !== model.root && Math.max(right - left, bottom - top) >= TV.HANDLE_CLADE_PX
                && Math.abs(px - x) <= TV.HANDLE_PX / 2 + 1 && Math.abs(py - y) <= TV.HANDLE_PX / 2 + 1) offer(i, "handle", -1);
            for (let k = model.childStart[i]; k < model.childStart[i + 1]; k++) stack.push(model.childList[k]);
        }
        return best;
    };

    // ---- What to say about a node ----------------------------------------------------------------------------

    function formatLength(value) {
        return String(Number(value.toPrecision(4)));
    }

    // The lines of text that describe node i: { title, lines }. Every header is the file's text, shown as text.
    TV.describeNode = function (i) {
        const { model } = state;
        const lines = [];
        let title;
        if (model.childCount[i] === 0) {
            const leaf = model.leafOrdinal[i];
            title = model.labels[leaf];
            lines.push(model.viewerNode[leaf] >= 0 ? t("A node of this network") : t("Not a node of this network"));
        } else {
            const count = TV.leavesUnder(model, i);
            title = i === model.root ? t("The root of the tree") : t("A clade");
            lines.push(t("Leaves in this clade: {count}", { count: count.toLocaleString() }));
            lines.push(t("Of them in this network: {count}", { count: TV.shownUnder(model, i).toLocaleString() }));
            if (model.support[i] >= 0) {
                lines.push(t("Stable in {percent}% of the replicate trees", { percent: model.support[i] }));
            }
            if (state.collapsed && state.collapsed[i]) lines.push(t("Folded up: click it to open it"));
        }
        if (model.hasLengths && i !== model.root) lines.push(t("Branch length: {length}", { length: formatLength(model.length[i]) }));
        return { title, lines };
    };

    // ---- The tooltip -----------------------------------------------------------------------------------------

    TV.showTooltip = function (tooltip, px, py, node, part) {
        const description = TV.describeNode(node);
        tooltip.replaceChildren();
        const name = document.createElement("div");
        name.className = "name";
        name.textContent = description.title;
        tooltip.appendChild(name);
        const lines = part === "handle"
            ? [state.collapsed && state.collapsed[node] ? t("Click to open this clade") : t("Click to fold this clade up"), ...description.lines]
            : description.lines;
        for (const text of lines) {
            const line = document.createElement("div");
            line.className = "line";
            line.textContent = text;
            tooltip.appendChild(line);
        }
        tooltip.style.display = "block";
        const stage = tooltip.parentElement;
        const left = px + 16 + tooltip.offsetWidth > stage.clientWidth ? px - 16 - tooltip.offsetWidth : px + 16;
        const top = py + 18 + tooltip.offsetHeight > stage.clientHeight ? py - 18 - tooltip.offsetHeight : py + 18;
        tooltip.style.left = Math.max(4, left) + "px";
        tooltip.style.top = Math.max(4, top) + "px";
    };

    TV.hideTooltip = function (tooltip) {
        tooltip.style.display = "none";
    };

    // ---- Search ----------------------------------------------------------------------------------------------

    // Find the leaves whose header holds `query`, in any case, or equals it when it is in double quotes. Returns
    // how many; the matches are kept, in tree order, for the drawing.
    TV.runSearch = function (query) {
        const { model } = state;
        const text = query.trim();
        state.query = text;
        state.current = -1;
        TV.minimapChanged();
        if (!model || !text) {
            state.matches = new Int32Array(0);
            return 0;
        }
        const found = [];
        if (text.length >= 2 && text[0] === "\"" && text[text.length - 1] === "\"") {
            const exact = text.slice(1, -1);
            for (let k = 0; k < model.leafCount; k++) if (model.labels[k] === exact) found.push(k);
        } else {
            if (!model.lowerLabels) model.lowerLabels = model.labels.map((label) => label.toLowerCase());
            const needle = text.toLowerCase();
            for (let k = 0; k < model.leafCount; k++) if (model.lowerLabels[k].includes(needle)) found.push(k);
        }
        state.matches = Int32Array.from(found);
        return found.length;
    };

    // Go to the next (delta 1) or previous (delta -1) match, round the ends, and bring it into view. Returns its
    // place among the matches, from 0, or -1 when there are none.
    TV.stepMatch = function (delta) {
        const count = state.matches.length;
        if (!count) return -1;
        state.current = (state.current + delta + count) % count;
        TV.focusLeaf(state.matches[state.current]);
        return state.current;
    };

    // ---- The gestures ----------------------------------------------------------------------------------------

    // The hint at the foot of the page.
    TV.hintText = function () {
        if (state.layout && state.layout.uniform) {
            return t("Wheel: zoom. Drag: move. Double-click a clade: zoom to it. Click its square to fold it up.");
        }
        return t("Wheel: zoom the rows. Shift+wheel: zoom the width. Drag: move. Double-click a clade: zoom to it. Click its square to fold it up.");
    };

    // Attach the page's pointer, wheel and key handling to the canvas. `ui` holds what the page does in answer:
    // tooltip (the element), onPin(node) to say what the status line shows.
    TV.attachNavigation = function (canvas, ui) {
        let drag = null;
        let hoverPoint = null;
        let hoverQueued = false;

        const local = (event) => {
            const rect = canvas.getBoundingClientRect();
            return [event.clientX - rect.left, event.clientY - rect.top];
        };
        const inRuler = (px) => state.layout && state.layout.kind === "rectangular" && px >= canvas.clientWidth - TV.RULER_WIDTH;

        function updateHover() {
            hoverQueued = false;
            if (!hoverPoint || drag) return;
            const hit = TV.hitTest(hoverPoint[0], hoverPoint[1]);
            const node = hit ? hit.node : -1;
            canvas.classList.toggle("pointing", !!hit);
            if (hit) TV.showTooltip(ui.tooltip, hoverPoint[0], hoverPoint[1], node, hit.part); else TV.hideTooltip(ui.tooltip);
            if (node !== state.hover) {
                state.hover = node;
                TV.requestDraw();
            }
        }

        canvas.addEventListener("wheel", (event) => {
            event.preventDefault();
            const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
            const amount = (event.shiftKey ? event.deltaX || event.deltaY : event.deltaY) * unit;
            const factor = Math.exp(-amount * 0.0015);
            const [px, py] = local(event);
            if (event.ctrlKey || event.metaKey) TV.zoomAbout(px, py, factor, factor);
            else if (event.shiftKey) TV.zoomAbout(px, py, factor, 1);
            else TV.zoomAbout(px, py, 1, factor);
            TV.hideTooltip(ui.tooltip);
        }, { passive: false });

        canvas.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || !state.layout) return;
            const [px, py] = local(event);
            // A pointer that is gone already cannot be captured; the drag goes on without it.
            try { canvas.setPointerCapture(event.pointerId); } catch (error) {}
            drag = { x: px, y: py, moved: false, ruler: inRuler(px) };
            TV.hideTooltip(ui.tooltip);
            if (drag.ruler) TV.jumpToFraction(py / canvas.clientHeight);
            canvas.classList.add("dragging");
            canvas.focus({ preventScroll: true });
        });

        canvas.addEventListener("pointermove", (event) => {
            const [px, py] = local(event);
            if (drag) {
                if (drag.ruler) {
                    TV.jumpToFraction(py / canvas.clientHeight);
                    return;
                }
                if (Math.abs(px - drag.x) + Math.abs(py - drag.y) > 3) drag.moved = true;
                if (drag.moved) TV.pan(px - drag.x, py - drag.y);
                drag.x = px;
                drag.y = py;
                return;
            }
            hoverPoint = [px, py];
            if (!hoverQueued) {
                hoverQueued = true;
                requestAnimationFrame(updateHover);
                setTimeout(() => { if (hoverQueued) updateHover(); }, 100);
            }
        });

        function endDrag(event) {
            if (!drag) return;
            const wasClick = !drag.moved && !drag.ruler;
            drag = null;
            canvas.classList.remove("dragging");
            try { canvas.releasePointerCapture(event.pointerId); } catch (error) {}
            if (!wasClick) return;
            const [px, py] = local(event);
            const hit = TV.hitTest(px, py);
            // A square folds its clade up, and a folded clade opens when it is clicked.
            if (hit && (hit.part === "handle" || (hit.part === "clade" && state.collapsed && state.collapsed[hit.node]))) {
                TV.toggleFold(hit.node);
                hoverPoint = [px, py];
                TV.hideTooltip(ui.tooltip);
                ui.onPin(state.pinned);
                return;
            }
            state.pinned = hit && hit.node !== state.pinned ? hit.node : -1;
            ui.onPin(state.pinned);
            TV.requestDraw();
        }
        canvas.addEventListener("pointerup", endDrag);
        canvas.addEventListener("pointercancel", endDrag);

        canvas.addEventListener("pointerleave", () => {
            hoverPoint = null;
            TV.hideTooltip(ui.tooltip);
            canvas.classList.remove("pointing");
            if (state.hover !== -1) {
                state.hover = -1;
                TV.requestDraw();
            }
        });

        canvas.addEventListener("dblclick", (event) => {
            if (!state.layout) return;
            const [px, py] = local(event);
            if (inRuler(px)) return;
            const hit = TV.hitTest(px, py);
            if (hit && hit.part === "handle") return;
            if (hit) TV.zoomToNode(hit.node); else TV.fitView();
        });

        canvas.addEventListener("keydown", (event) => {
            if (!state.layout || event.ctrlKey || event.altKey || event.metaKey) return;
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            const step = event.shiftKey ? 400 : 80;
            let handled = true;
            switch (event.key) {
                case "ArrowLeft": TV.pan(step, 0); break;
                case "ArrowRight": TV.pan(-step, 0); break;
                case "ArrowUp": TV.pan(0, step); break;
                case "ArrowDown": TV.pan(0, -step); break;
                case "+": case "=": TV.zoomAbout(width / 2, height / 2, 1, 1.25); break;
                case "-": case "_": TV.zoomAbout(width / 2, height / 2, 1, 0.8); break;
                case "]": TV.zoomAbout(width / 2, height / 2, 1.25, 1); break;
                case "[": TV.zoomAbout(width / 2, height / 2, 0.8, 1); break;
                case "Home": case "0": TV.fitView(); break;
                case "c": case "C": handled = state.pinned >= 0 && TV.toggleFold(state.pinned); break;
                case "Escape":
                    state.pinned = -1;
                    ui.onPin(-1);
                    TV.requestDraw();
                    break;
                default: handled = false;
            }
            if (handled) event.preventDefault();
        });
    };
})(window.TreeView);
