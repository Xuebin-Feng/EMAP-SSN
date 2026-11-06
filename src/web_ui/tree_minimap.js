/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's overview map: the whole tree in a small canvas at the corner of the page, with the rectangle of the
 * view on it. Press or drag on it to move the view there.
 *
 * The tree is drawn once, with the one traversal of the page (TV.paintTree, in its small form) into a canvas the
 * map keeps, and again only when what it draws changes (the layout, what is folded up, the matches, the theme, the
 * size). A frame of the page then costs the copy of that picture and the rectangle.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const state = TV.state;
    const PAD = 6;
    let canvas = null;
    let base = null;
    let dirty = true;
    let baseSize = "";
    // How the layout maps onto the map: map x = ox + x * sx, and so for y.
    let map = { sx: 1, sy: 1, ox: 0, oy: 0 };

    TV.invalidateMinimap = function () {
        dirty = true;
    };

    // The scale and offset that put the layout's bounds in a map of width x height pixels.
    function fitMap(layout, width, height) {
        const bounds = layout.bounds;
        const bw = bounds.maxX - bounds.minX;
        const bh = bounds.maxY - bounds.minY;
        if (layout.uniform) {
            const scale = Math.min((width - 2 * PAD) / bw, (height - 2 * PAD) / bh);
            return {
                sx: scale, sy: scale,
                ox: width / 2 - (bounds.minX + bounds.maxX) / 2 * scale,
                oy: height / 2 - (bounds.minY + bounds.maxY) / 2 * scale,
            };
        }
        const sx = (width - 2 * PAD) / bw;
        const sy = (height - 2 * PAD) / bh;
        return { sx, sy, ox: PAD - bounds.minX * sx, oy: PAD - bounds.minY * sy };
    }

    // The tree, drawn small into the canvas the map keeps.
    function paintBase(width, height, ratio) {
        if (!base) base = document.createElement("canvas");
        base.width = Math.round(width * ratio);
        base.height = Math.round(height * ratio);
        const ctx = base.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        map = fitMap(state.layout, width, height);
        const scene = {
            model: state.model, layout: state.layout, view: map, palette: state.palette,
            collapsed: state.collapsed, matches: state.matches, colors: state.colors, selection: state.selection,
        };
        TV.paintTree(ctx, scene, width, height, true);
        // The leaves the network has selected, as points, one to a pixel; the matches go over them.
        if (state.selection.leaves.length) {
            ctx.fillStyle = state.palette.selected;
            const hidden = state.leafHidden;
            const seen = new Set();
            for (const leaf of state.selection.leaves) {
                if (hidden && hidden[leaf]) continue;
                const node = state.model.leafNode[leaf];
                const x = Math.round(map.ox + state.layout.x[node] * map.sx);
                const y = Math.round(map.oy + state.layout.y[node] * map.sy);
                const key = (x + 4096) * 1048576 + (y + 4096);
                if (seen.has(key)) continue;
                seen.add(key);
                ctx.fillRect(x - 1, y - 1, 2, 2);
            }
        }
        // A leaf the search found, as a point; a leaf inside a folded clade is in its outlined shape.
        if (state.matches.length) {
            ctx.fillStyle = state.palette.match;
            const hidden = state.leafHidden;
            for (const leaf of state.matches) {
                if (hidden && hidden[leaf]) continue;
                const node = state.model.leafNode[leaf];
                ctx.fillRect(map.ox + state.layout.x[node] * map.sx - 1, map.oy + state.layout.y[node] * map.sy - 1, 2, 2);
            }
        }
    }

    // Draw the map: the picture of the tree, and the rectangle of the view on it.
    TV.drawMinimap = function () {
        if (!canvas || canvas.hidden || !state.model || !state.layout) return;
        if (!state.palette) state.palette = TV.readPalette();
        const { width, height, ratio } = TV.sizeCanvas(canvas);
        const size = width + "x" + height + "@" + ratio;
        if (dirty || size !== baseSize) {
            paintBase(width, height, ratio);
            dirty = false;
            baseSize = size;
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(base, 0, 0);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

        // The part of the layout that the page shows, on the map.
        const view = state.view;
        const pageWidth = TV.canvas.clientWidth;
        const pageHeight = TV.canvas.clientHeight;
        const x0 = map.ox + (0 - view.ox) / view.sx * map.sx;
        const x1 = map.ox + (pageWidth - view.ox) / view.sx * map.sx;
        const y0 = map.oy + (0 - view.oy) / view.sy * map.sy;
        const y1 = map.oy + (pageHeight - view.oy) / view.sy * map.sy;
        const left = Math.max(0, Math.min(x0, width - 4));
        const top = Math.max(0, Math.min(y0, height - 4));
        const right = Math.min(width, Math.max(x1, left + 4));
        const bottom = Math.min(height, Math.max(y1, top + 4));
        // Kept for the tests: where the view is on the map, in the map's pixels.
        TV.minimapView = { left, top, right, bottom, width, height };
        ctx.fillStyle = state.palette.selection;
        ctx.fillRect(left, top, right - left, bottom - top);
        ctx.strokeStyle = state.palette.accent;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(left + 0.75, top + 0.75, Math.max(1, right - left - 1.5), Math.max(1, bottom - top - 1.5));
    };

    // The point of the layout under the pixel (px, py) of the map.
    TV.minimapPoint = function (px, py) {
        return [(px - map.ox) / map.sx, (py - map.oy) / map.sy];
    };

    // Attach the map's pointer handling to its canvas: pressing or dragging on it centres the view there.
    TV.attachMinimap = function (element) {
        canvas = element;
        let pressed = false;
        const move = (event) => {
            if (!state.layout) return;
            const rect = canvas.getBoundingClientRect();
            const [x, y] = TV.minimapPoint(event.clientX - rect.left - canvas.clientLeft, event.clientY - rect.top - canvas.clientTop);
            TV.centerOn(x, y);
        };
        canvas.addEventListener("pointerdown", (event) => {
            if (event.button !== 0) return;
            pressed = true;
            try { canvas.setPointerCapture(event.pointerId); } catch (error) {}
            move(event);
        });
        canvas.addEventListener("pointermove", (event) => { if (pressed) move(event); });
        const release = (event) => {
            pressed = false;
            try { canvas.releasePointerCapture(event.pointerId); } catch (error) {}
        };
        canvas.addEventListener("pointerup", release);
        canvas.addEventListener("pointercancel", release);
    };
})(window.TreeView);
