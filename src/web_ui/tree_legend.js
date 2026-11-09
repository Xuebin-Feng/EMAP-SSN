/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's legend: what the colours of the network mean, as the Viewer's colour key says (tree_color.js decodes it).
 *
 * The key's layers are listed newest first. A layer of categories (clusters, colours chosen by name...) is a titled
 * group of rows, a swatch, a label and the number of leaves; a click marks the leaves of that row, as a search marks
 * what it finds, and the same row again takes the marks off. A layer of a gradient (a spectrum) is a bar of its colours
 * from its minimum to its maximum, with the rows of what the gradient leaves out (no value, say) under it; pressing or
 * dragging on the bar marks the leaves whose value is in the range it covers, and the arrow keys step through the bar
 * in sixteen bands. The leaves the key does not cover are listed last, by colour, under "Unlabelled colours".
 *
 * Every title and label comes from the Viewer, and a layout cache that somebody shared can put anything in them: they are
 * set as text, never as markup.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const state = TV.state;
    const $ = (id) => document.getElementById(id);
    // How many bands the arrow keys step through on a bar, and how much of the bar a press without a drag covers.
    const BANDS = 16;
    const CLICK_SPAN = 1 / 32;
    let legendOpen = true;

    function setIconOf(svg, name) {
        const use = svg.querySelector("use");
        if (use) use.setAttribute("href", "#i-" + name);
    }

    function number(value) {
        return String(Number(value.toPrecision(4)));
    }

    function element(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    // One row: a button that marks (or unmarks) `leaves`, named `key`.
    function rowOf(key, css, label, count, title, leaves) {
        const item = element("li");
        const button = element("button", "legend-row");
        button.type = "button";
        button.dataset.key = key;
        button.setAttribute("aria-pressed", String(state.legendMark === key));
        button.title = title;
        const swatch = element("span", "swatch");
        swatch.style.backgroundColor = css;
        const name = element("span", "name", label);
        const total = element("span", "count", count.toLocaleString());
        button.append(swatch, name, total);
        button.addEventListener("click", () => {
            if (state.legendMark === key) TV.clearLegendMark(); else markWith(key, leaves(), null);
        });
        item.appendChild(button);
        return item;
    }

    function markWith(key, leaves, range) {
        state.legendMark = key;
        state.legendRange = range;
        TV.markLeaves(leaves);
        refreshLegendMarks();
    }

    // Take the marks of the legend off (a search that replaces them, a new colour state, or the same entry pressed again).
    TV.clearLegendMark = function () {
        if (state.legendMark === null) return;
        state.legendMark = null;
        state.legendRange = null;
        TV.markLeaves(new Int32Array(0));
        refreshLegendMarks();
    };

    // The marks without a new tree view of the list: pressed rows, the bands and the range lines of the bars.
    function refreshLegendMarks() {
        for (const button of $("legend-list").querySelectorAll(".legend-row")) {
            button.setAttribute("aria-pressed", String(state.legendMark === button.dataset.key));
        }
        for (const wrap of $("legend-list").querySelectorAll(".legend-scale")) {
            const range = state.legendRange && String(state.legendRange.layer) === wrap.dataset.layer ? state.legendRange : null;
            showRange(wrap, range);
        }
    }

    // The range a bar shows as marked, or none.
    function showRange(wrap, range) {
        const band = wrap.querySelector(".legend-band");
        const line = wrap.querySelector(".legend-range");
        const clear = wrap.querySelector(".legend-clear");
        const min = Number(wrap.dataset.min);
        const max = Number(wrap.dataset.max);
        band.hidden = !range;
        line.hidden = !range;
        clear.hidden = !range;
        if (!range) return;
        const span = max - min;
        const from = span > 0 ? (range.low - min) / span : 0;
        const to = span > 0 ? (range.high - min) / span : 1;
        band.style.left = (Math.min(1, Math.max(0, from)) * 100) + "%";
        band.style.width = (Math.min(1, Math.max(0, to - from)) * 100) + "%";
        line.textContent = t("{low} to {high}", { low: number(range.low), high: number(range.high) });
    }

    // Mark the leaves of `layer` whose value is from the share `a` to the share `b` of the bar (0 to 1, either way round).
    TV.markBarRange = function (layer, a, b) {
        const lowShare = Math.min(1, Math.max(0, Math.min(a, b)));
        const highShare = Math.min(1, Math.max(0, Math.max(a, b)));
        const span = layer.max - layer.min;
        const range = { layer: layer.index, low: layer.min + span * lowShare, high: layer.min + span * highShare };
        markWith("range:" + layer.index, TV.leavesInRange(layer, range.low, range.high), range);
    };

    function barOf(layer) {
        const wrap = element("div", "legend-scale");
        wrap.dataset.layer = String(layer.index);
        wrap.dataset.min = String(layer.min);
        wrap.dataset.max = String(layer.max);
        const bar = element("div", "legend-bar");
        bar.tabIndex = 0;
        bar.setAttribute("role", "group");
        const hint = t("Press or drag to mark the leaves with a value in that range");
        bar.title = hint;
        bar.setAttribute("aria-label", layer.title + ": " + hint);
        bar.style.backgroundImage = layer.stops.length > 1 ? "linear-gradient(to right, " + layer.stops.join(", ") + ")" : "none";
        if (layer.stops.length === 1) bar.style.backgroundColor = layer.stops[0];
        const band = element("div", "legend-band");
        band.hidden = true;
        bar.appendChild(band);
        const ends = element("div", "legend-ends");
        const low = element("span", "", number(layer.min));
        const high = element("span", "", number(layer.max));
        ends.append(low, high);
        const line = element("div", "legend-range");
        line.hidden = true;
        line.setAttribute("aria-live", "polite");
        const clear = element("button", "legend-clear", t("Clear"));
        clear.type = "button";
        clear.hidden = true;
        clear.title = t("Take the marks off");
        clear.addEventListener("click", () => TV.clearLegendMark());
        const footer = element("div", "legend-range-line");
        footer.append(line, clear);
        wrap.append(bar, ends, footer);

        const shareAt = (event) => {
            const box = bar.getBoundingClientRect();
            return box.width > 0 ? Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)) : 0;
        };
        let start = null;
        let moved = false;
        const cover = (end) => {
            if (!moved) TV.markBarRange(layer, start - CLICK_SPAN, start + CLICK_SPAN);
            else TV.markBarRange(layer, start, end);
        };
        bar.addEventListener("pointerdown", (event) => {
            if (event.button !== undefined && event.button !== 0) return;
            start = shareAt(event);
            moved = false;
            try { bar.setPointerCapture(event.pointerId); } catch (error) { /* a pointer the browser does not know */ }
            cover(start);
            event.preventDefault();
        });
        bar.addEventListener("pointermove", (event) => {
            if (start === null) return;
            const here = shareAt(event);
            if (!moved && Math.abs(here - start) * bar.getBoundingClientRect().width < 3) return;
            moved = true;
            cover(here);
        });
        const finish = () => { start = null; };
        bar.addEventListener("pointerup", finish);
        bar.addEventListener("pointercancel", finish);
        bar.addEventListener("keydown", (event) => {
            const marked = state.legendMark === "range:" + layer.index && state.legendRange;
            let band = marked ? Math.round(((state.legendRange.low - layer.min) / (layer.max - layer.min || 1)) * BANDS) : -1;
            if (event.key === "Escape") {
                TV.clearLegendMark();
            } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
                band = Math.min(BANDS - 1, band + 1);
            } else if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
                band = Math.max(0, band - 1);
            } else if (event.key === "Home") {
                band = 0;
            } else if (event.key === "End") {
                band = BANDS - 1;
            } else {
                return;
            }
            event.preventDefault();
            if (event.key !== "Escape") TV.markBarRange(layer, band / BANDS, (band + 1) / BANDS);
        });
        return wrap;
    }

    function groupOf(title) {
        const group = element("section", "legend-group");
        group.appendChild(element("h3", "legend-title", title));
        return group;
    }

    // The legend: the layers of the colour key, newest first, then the colours the key does not cover.
    TV.showLegend = function () {
        const colors = state.colors;
        const legend = $("legend");
        const list = $("legend-list");
        list.replaceChildren();
        const layers = colors ? colors.layers : [];
        const unlabelled = colors ? colors.unlabelled : [];
        legend.classList.toggle("shown", layers.length > 0 || unlabelled.length > 0);
        if (layers.length > 0 || unlabelled.length > 0) {
            const rowTitle = t("Mark these leaves");
            for (const layer of layers) {
                const group = groupOf(layer.title);
                if (layer.kind === "gradient") {
                    if (layer.scheme) group.appendChild(element("div", "legend-scheme", layer.scheme));
                    if (layer.bar.count > 0 && layer.min !== null && layer.max !== null) group.appendChild(barOf(layer));
                }
                if (layer.rows.length > 0) {
                    const rows = element("ul", "legend-rows");
                    layer.rows.forEach((row, slot) => rows.appendChild(
                        rowOf("row:" + layer.index + ":" + slot, row.css, row.label, row.count, rowTitle, () => row.leaves)));
                    group.appendChild(rows);
                }
                list.appendChild(group);
            }
            if (unlabelled.length > 0) {
                const group = groupOf(t("Unlabelled colours"));
                const rows = element("ul", "legend-rows");
                for (const entry of unlabelled) {
                    rows.appendChild(rowOf("colour:" + entry.cls, entry.css, entry.hex, entry.count,
                        t("Mark the leaves of this colour"), () => TV.unlabelledLeavesOfClass(colors, entry.cls)));
                }
                group.appendChild(rows);
                list.appendChild(group);
            }
        }
        list.hidden = !legendOpen;
        $("legend-toggle").setAttribute("aria-expanded", String(legendOpen));
        setIconOf($("legend-icon"), legendOpen ? "chevron-up" : "chevron-down");
        refreshLegendMarks();
    };

    TV.refreshLegendMarks = refreshLegendMarks;

    TV.toggleLegend = function () {
        legendOpen = !legendOpen;
        TV.showLegend();
    };
})(window.TreeView);
