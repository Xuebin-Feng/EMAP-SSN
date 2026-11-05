/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's navigation: the view (zoom and pan), what is under the pointer, the search and the tooltip.
 *
 * The rows and the width zoom apart, as a long tree needs far more zoom in its rows than in its width: the
 * wheel zooms the rows about the pointer, Shift and the wheel the width, Ctrl (or a pinch) both. Double-click
 * a clade to bring it to fill the view. What is under the pointer is found by going down from the root along
 * the row it is in, so a look-up costs the depth of the tree and not its size.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const state = TV.state = {
        model: null,
        layout: null,
        view: { sx: 1, sy: 1, ox: 0, oy: 0 },
        fit: { sx: 1, sy: 1, ox: 0, oy: 0 },
        fitted: true,
        hover: -1,
        pinned: -1,
        palette: null,
        matches: new Int32Array(0),
        current: -1,
        query: "",
    };
    TV.canvas = null;
    TV.lastStats = null;

    // How close, in pixels, the pointer must be to a line to be on it.
    const HIT_PX = 5;
    const MAX_ROW_PX = 80;
    const PAD = { left: 24, right: 24, top: 16, bottom: 16 };

    function clamp(value, low, high) {
        return Math.min(high, Math.max(low, value));
    }

    TV.limits = function () {
        const fit = state.fit;
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
        const width = TV.canvas.clientWidth;
        const height = TV.canvas.clientHeight;
        const treeWidth = state.layout.width * view.sx;
        const treeHeight = state.layout.height * view.sy;
        view.ox = clamp(view.ox, 60 - treeWidth - 300, width - 60);
        view.oy = clamp(view.oy, 60 - treeHeight, height - 60);
    }

    // The view that shows the whole tree, centred when it is short.
    TV.fitView = function () {
        if (!state.layout || !TV.canvas) return;
        const width = TV.canvas.clientWidth;
        const height = TV.canvas.clientHeight;
        const span = state.layout.width > 0 ? state.layout.width : 1;
        const sx = Math.max(1e-6, (width - PAD.left - PAD.right - TV.RULER_WIDTH) / span);
        const sy = clamp((height - PAD.top - PAD.bottom) / state.layout.height, 1e-6, 28);
        state.view = { sx, sy, ox: PAD.left, oy: Math.max(PAD.top, (height - sy * state.layout.height) / 2) };
        state.fit = { ...state.view };
        state.fitted = true;
        TV.requestDraw();
    };

    // Zoom by factors fx and fy about the pixel (px, py).
    TV.zoomAbout = function (px, py, fx, fy) {
        if (!state.layout) return;
        const view = state.view;
        const limit = TV.limits();
        const sx = clamp(view.sx * fx, limit.sxMin, limit.sxMax);
        const sy = clamp(view.sy * fy, limit.syMin, limit.syMax);
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
        if (!model) return;
        if (model.childCount[i] === 0) {
            TV.focusLeaf(model.leafOrdinal[i]);
            return;
        }
        const width = TV.canvas.clientWidth;
        const height = TV.canvas.clientHeight;
        const limit = TV.limits();
        const rows = model.lastLeaf[i] - model.firstLeaf[i] + 1;
        const room = { left: 24, right: TV.RULER_WIDTH + 24, top: 24, bottom: 24 };
        const sy = clamp((height - room.top - room.bottom) / rows, limit.syMin, limit.syMax);
        const labelRoom = sy >= TV.LABEL_PX ? Math.min(300, width * 0.3) : 0;
        const p = model.parent[i];
        const x0 = layout.x[p >= 0 ? p : i];
        const span = Math.max(layout.reach[i] - x0, layout.width * 1e-3, 1e-9);
        const sx = clamp((width - room.left - room.right - labelRoom) / span, limit.sxMin, limit.sxMax);
        state.view = {
            sx,
            sy,
            ox: room.left - x0 * sx,
            oy: room.top + (height - room.top - room.bottom - rows * sy) / 2 - model.firstLeaf[i] * sy,
        };
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    // Put leaf number k in the middle of the view, at a row tall enough to read its name.
    TV.focusLeaf = function (k) {
        const { model, layout, view } = state;
        if (!model) return;
        const limit = TV.limits();
        view.sy = clamp(Math.max(view.sy, 16), limit.syMin, limit.syMax);
        view.oy = TV.canvas.clientHeight / 2 - (k + 0.5) * view.sy;
        const tip = view.ox + layout.x[model.leafNode[k]] * view.sx;
        if (tip < 40 || tip > TV.canvas.clientWidth - 300) {
            view.ox = TV.canvas.clientWidth * 0.5 - layout.x[model.leafNode[k]] * view.sx;
        }
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    // Move the view so that the rows at the fraction `fraction` (0 to 1) of the tree are in its middle: the ruler's
    // jump.
    TV.jumpToFraction = function (fraction) {
        if (!state.layout) return;
        const view = state.view;
        view.oy = TV.canvas.clientHeight / 2 - clamp(fraction, 0, 1) * state.layout.height * view.sy;
        state.fitted = false;
        keepInView();
        TV.requestDraw();
    };

    // ---- Drawing, a frame at a time --------------------------------------------------------------------------

    let drawQueued = false;

    TV.drawNow = function () {
        drawQueued = false;
        if (!TV.canvas) return null;
        if (!state.palette) state.palette = TV.readPalette();
        TV.lastStats = TV.drawTree(TV.canvas, state);
        return TV.lastStats;
    };

    // Draw at the next frame, or after a moment where frames do not come (a page behind others).
    TV.requestDraw = function () {
        if (drawQueued) return;
        drawQueued = true;
        requestAnimationFrame(() => { if (drawQueued) TV.drawNow(); });
        setTimeout(() => { if (drawQueued) TV.drawNow(); }, 100);
    };

    // ---- What is under the pointer ---------------------------------------------------------------------------

    const measure = document.createElement("canvas").getContext("2d");

    // The width in pixels of a leaf's name at the current row height.
    function nameWidth(text) {
        measure.font = Math.min(13, Math.max(9, state.view.sy * 0.8)).toFixed(1) + "px " + (state.palette ? state.palette.font : "sans-serif");
        return measure.measureText(text.length > TV.MAX_LABEL_CHARACTERS ? text.slice(0, TV.MAX_LABEL_CHARACTERS) : text).width;
    }

    // The node at the pixel (px, py) and which part of it is there ("leaf", "branch", "connector" or "clade"), or null.
    // Rectangular layout: from the root, go down the row the pointer is in; each node on the way may be on its own
    // branch, or on the line that joins its children, or be a clade drawn as one bar.
    TV.hitTest = function (px, py) {
        const { model, layout, view } = state;
        if (!model || !layout) return null;
        const row = (py - view.oy) / view.sy;
        let node = model.root;
        for (;;) {
            const box = TV.nodePixels(state, node);
            const inRows = row >= model.firstLeaf[node] - HIT_PX / view.sy && row <= model.lastLeaf[node] + 1 + HIT_PX / view.sy;
            if (!inRows) return null;
            if (model.childCount[node] === 0) {
                const named = view.sy >= TV.LABEL_PX;
                const right = box.x + (named ? 4 + nameWidth(model.labels[model.leafOrdinal[node]]) : HIT_PX);
                const half = Math.max(HIT_PX, view.sy / 2);
                return Math.abs(py - box.y) <= half && px >= box.x0 - HIT_PX && px <= right ? { node, part: "leaf" } : null;
            }
            if (model.lastLeaf[node] > model.firstLeaf[node] && box.bottom - box.top < TV.COLLAPSE_PX) {
                const on = px >= box.x0 - HIT_PX && px <= box.reach + HIT_PX && py >= box.top - HIT_PX && py <= box.bottom + HIT_PX;
                return on ? { node, part: "clade" } : null;
            }
            if (Math.abs(py - box.y) <= HIT_PX && px >= box.x0 - HIT_PX && px <= box.x + HIT_PX) return { node, part: "branch" };
            const first = model.childList[model.childStart[node]];
            const last = model.childList[model.childStart[node + 1] - 1];
            const joinTop = view.oy + layout.y[first] * view.sy;
            const joinBottom = view.oy + layout.y[last] * view.sy;
            if (Math.abs(px - box.x) <= HIT_PX && py >= joinTop - HIT_PX && py <= joinBottom + HIT_PX) return { node, part: "connector" };
            if (px < box.x - HIT_PX || row < model.firstLeaf[node] || row >= model.lastLeaf[node] + 1) return null;
            node = TV.childHolding(model, node, Math.floor(row));
        }
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
        }
        if (model.hasLengths && i !== model.root) lines.push(t("Branch length: {length}", { length: formatLength(model.length[i]) }));
        return { title, lines };
    };

    // ---- The tooltip -----------------------------------------------------------------------------------------

    TV.showTooltip = function (tooltip, px, py, node) {
        const description = TV.describeNode(node);
        tooltip.replaceChildren();
        const name = document.createElement("div");
        name.className = "name";
        name.textContent = description.title;
        tooltip.appendChild(name);
        for (const text of description.lines) {
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
        return t("Wheel: zoom the rows. Shift+wheel: zoom the width. Drag: move. Double-click a clade: zoom to it.");
    };

    // Attach the page's pointer, wheel and key handling to the canvas. `ui` holds what the page does in answer:
    // tooltip (the element), onPin(node) and onHover() to say what the status line shows.
    TV.attachNavigation = function (canvas, ui) {
        let drag = null;
        let hoverPoint = null;
        let hoverQueued = false;

        const local = (event) => {
            const rect = canvas.getBoundingClientRect();
            return [event.clientX - rect.left, event.clientY - rect.top];
        };
        const inRuler = (px) => px >= canvas.clientWidth - TV.RULER_WIDTH;

        function updateHover() {
            hoverQueued = false;
            if (!hoverPoint || drag) return;
            const hit = TV.hitTest(hoverPoint[0], hoverPoint[1]);
            const node = hit ? hit.node : -1;
            canvas.classList.toggle("pointing", !!hit);
            if (hit) TV.showTooltip(ui.tooltip, hoverPoint[0], hoverPoint[1], node); else TV.hideTooltip(ui.tooltip);
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
            if (wasClick) {
                const [px, py] = local(event);
                const hit = TV.hitTest(px, py);
                state.pinned = hit && hit.node !== state.pinned ? hit.node : -1;
                ui.onPin(state.pinned);
                TV.requestDraw();
            }
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
