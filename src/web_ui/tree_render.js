/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's drawing: a tree (tree_model.js) in a layout (tree_layout.js) on a canvas, in the page's theme.
 *
 * Only what is in view is visited. A clade that comes out shorter than a few pixels is drawn as one bar,
 * however many leaves it holds, so what a frame costs follows what a person can see, not the size of the
 * tree. Lines are batched into one path per style, and a leaf is marked with a dot, then named, only when its
 * row is tall enough to show it.
 *
 * A view maps the layout's units to pixels: screen x = ox + x * sx, screen y = oy + y * sy. The two scales are
 * kept apart, as a long tree zooms in its rows far more than in its width.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    // A clade shorter than this many pixels is one bar; a row this tall gets a dot, and one this tall its name.
    TV.COLLAPSE_PX = 3;
    TV.DOT_PX = 4;
    TV.LABEL_PX = 12;
    // The width of the strip at the right edge that shows where the view is in the whole tree, and the matches.
    TV.RULER_WIDTH = 14;
    // A name is cut here, whatever its length.
    TV.MAX_LABEL_CHARACTERS = 160;
    // How far to the right of a leaf the name of one can reach: a clade whose leaves are that far left of the view
    // can still show names in it.
    const LABEL_REACH_PX = 520;

    // The colours the canvas draws in, read from the page's theme (its custom properties) so that light, dark and
    // the Viewer's theme colour all apply.
    TV.readPalette = function () {
        const style = getComputedStyle(document.documentElement);
        const get = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
        return {
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

    // The pixel position of node i's own point, and where its branch begins (its parent's x).
    TV.nodePixels = function (scene, i) {
        const { model, layout, view } = scene;
        const p = model.parent[i];
        const x = view.ox + layout.x[i] * view.sx;
        return {
            x,
            y: view.oy + layout.y[i] * view.sy,
            x0: p >= 0 ? view.ox + layout.x[p] * view.sx : x,
            reach: view.ox + layout.reach[i] * view.sx,
            top: view.oy + model.firstLeaf[i] * view.sy,
            bottom: view.oy + (model.lastLeaf[i] + 1) * view.sy,
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

    // Draw the scene on the canvas. scene: { model, layout, view, palette, hover, pinned, matches, current },
    // hover and pinned a node number or -1, matches the sorted leaf numbers found by the search and current the
    // index in it of the one the search is at (or -1). Returns what was drawn, for the tests and the status line.
    TV.drawTree = function (canvas, scene) {
        const { width, height, ratio } = TV.sizeCanvas(canvas);
        const ctx = canvas.getContext("2d");
        const palette = scene.palette;
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.fillStyle = palette.background;
        ctx.fillRect(0, 0, width, height);
        const stats = { visited: 0, bars: 0, labels: 0, dots: 0, width, height };
        const { model, layout, view } = scene;
        if (!model || !layout) return stats;

        const { sx, sy, ox, oy } = view;
        const { x: layoutX, y: layoutY, reach: layoutReach } = layout;
        const { parent, childCount, childStart, childList, firstLeaf, lastLeaf, leafOrdinal, viewerNode } = model;
        const margin = 4;
        const showDots = sy >= TV.DOT_PX;
        const showLabels = sy >= TV.LABEL_PX;
        const labelRoom = showLabels ? LABEL_REACH_PX : 0;
        const dotRadius = Math.min(3.2, Math.max(1.1, sy * 0.25));

        // Highlights sit behind the lines.
        for (const [node, fill] of [[scene.pinned, palette.selection], [scene.hover, palette.selection]]) {
            if (node === undefined || node < 0) continue;
            const box = TV.nodePixels(scene, node);
            ctx.fillStyle = fill;
            ctx.fillRect(box.x0, box.top, Math.max(2, box.reach - box.x0) + (showLabels ? LABEL_REACH_PX : 6), Math.max(2, box.bottom - box.top));
        }

        const branches = new Path2D();
        const faded = new Path2D();
        const bars = new Path2D();
        const dots = new Path2D();
        const fadedDots = new Path2D();
        const labels = [];

        const stack = [model.root];
        while (stack.length) {
            const i = stack.pop();
            const p = parent[i];
            const xi = ox + layoutX[i] * sx;
            const x0 = p >= 0 ? ox + layoutX[p] * sx : xi;
            const top = oy + firstLeaf[i] * sy;
            const bottom = oy + (lastLeaf[i] + 1) * sy;
            if (bottom < -margin || top > height + margin) continue;
            if (x0 > width + margin || ox + layoutReach[i] * sx + labelRoom < -margin) continue;
            stats.visited++;
            const yi = oy + layoutY[i] * sy;

            if (childCount[i] === 0) {
                const inNetwork = viewerNode[leafOrdinal[i]] >= 0;
                const path = inNetwork ? branches : faded;
                path.moveTo(x0, yi);
                path.lineTo(xi, yi);
                if (showDots) {
                    (inNetwork ? dots : fadedDots).moveTo(xi + dotRadius, yi);
                    (inNetwork ? dots : fadedDots).arc(xi, yi, dotRadius, 0, 2 * Math.PI);
                    stats.dots++;
                }
                if (showLabels) labels.push([model.labels[leafOrdinal[i]], xi + dotRadius + 4, yi, inNetwork]);
            } else if (lastLeaf[i] > firstLeaf[i] && bottom - top < TV.COLLAPSE_PX) {
                branches.moveTo(x0, yi);
                branches.lineTo(xi, yi);
                const barHeight = Math.max(1.5, bottom - top);
                bars.rect(xi, yi - barHeight / 2, Math.max(1, ox + layoutReach[i] * sx - xi), barHeight);
                stats.bars++;
            } else {
                branches.moveTo(x0, yi);
                branches.lineTo(xi, yi);
                const first = childList[childStart[i]];
                const last = childList[childStart[i + 1] - 1];
                branches.moveTo(xi, oy + layoutY[first] * sy);
                branches.lineTo(xi, oy + layoutY[last] * sy);
                for (let k = childStart[i]; k < childStart[i + 1]; k++) stack.push(childList[k]);
            }
        }

        ctx.lineWidth = 1;
        ctx.lineCap = "butt";
        ctx.globalAlpha = 0.35;
        ctx.fillStyle = palette.branch;
        ctx.fill(bars);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = palette.faded;
        ctx.stroke(faded);
        ctx.strokeStyle = palette.branch;
        ctx.stroke(branches);
        if (showDots) {
            ctx.fillStyle = palette.faded;
            ctx.fill(fadedDots);
            ctx.fillStyle = palette.branch;
            ctx.fill(dots);
        }

        if (labels.length) {
            ctx.font = Math.min(13, Math.max(9, sy * 0.8)).toFixed(1) + "px " + palette.font;
            ctx.textBaseline = "middle";
            ctx.textAlign = "left";
            for (const [text, x, y, inNetwork] of labels) {
                ctx.fillStyle = inNetwork ? palette.label : palette.faded;
                ctx.fillText(shortened(text), x, y);
            }
            stats.labels = labels.length;
        }

        // The hovered branch, and the search's matches, over the lines.
        if (scene.hover !== undefined && scene.hover >= 0) {
            const box = TV.nodePixels(scene, scene.hover);
            ctx.strokeStyle = palette.accent;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(box.x0, box.y);
            ctx.lineTo(box.x, box.y);
            if (model.childCount[scene.hover] > 0) {
                const first = childList[childStart[scene.hover]];
                const last = childList[childStart[scene.hover + 1] - 1];
                ctx.moveTo(box.x, oy + layoutY[first] * sy);
                ctx.lineTo(box.x, oy + layoutY[last] * sy);
            }
            ctx.stroke();
        }
        TV.drawMatches(ctx, scene, width, height);
        TV.drawRuler(ctx, scene, width, height);
        return stats;
    };

    // A ring at the tip of each leaf the search found that is in view; a short tick where its row is too thin for one.
    TV.drawMatches = function (ctx, scene, width, height) {
        const matches = scene.matches;
        if (!matches || !matches.length) return;
        const { model, layout, view, palette } = scene;
        const first = lowerBound(matches, Math.floor((-4 - view.oy) / view.sy));
        ctx.lineWidth = 1.5;
        for (let m = first; m < matches.length; m++) {
            const leaf = matches[m];
            const y = view.oy + (leaf + 0.5) * view.sy;
            if (y > height + 8) break;
            const x = view.ox + layout.x[model.leafNode[leaf]] * view.sx;
            if (x < -LABEL_REACH_PX || x > width + 8) continue;
            const current = m === scene.current;
            ctx.strokeStyle = current ? palette.accent : palette.match;
            ctx.fillStyle = palette.matchSoft;
            if (view.sy < 6) {
                ctx.fillStyle = current ? palette.accent : palette.match;
                ctx.fillRect(x, y - 1, current ? 22 : 14, 2);
            } else {
                const radius = Math.min(8, Math.max(3.5, view.sy * 0.45));
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
        }
    };

    // The strip at the right edge: the whole tree top to bottom, the rows in view as a thumb, a tick for each match.
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
