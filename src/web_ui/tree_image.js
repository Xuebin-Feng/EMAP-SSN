/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's Save an image: the tree as it is in view, as a PNG or an SVG.
 *
 * The image is the page's own drawing (TV.drawScene), in the light theme's colours whatever the page shows, so that
 * names never come out white on white paper. What belongs to the page alone is left out: the overview map, the strip
 * at the right edge, the pointer's highlight and the fold squares. Under the tree goes a band with what was asked
 * for: the scale bar, the colours of the leaves in view, and the legend of the support dots. A PNG is `size` percent
 * of the drawing's pixels on the screen; an SVG is the same drawing as lines and text, at any size: SvgContext stands
 * in for the canvas and writes down what the drawing does with it.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const state = TV.state;
    const $ = (id) => document.getElementById(id);

    // The largest PNG offered: a side of at most MAX_SIDE pixels and at most MAX_AREA pixels in all, which browsers can
    // still make a canvas of.
    const MAX_SIDE = 16384;
    const MAX_AREA = 16384 * 8192;
    const FILE_NAME = "tree-view";

    // What the card asks, kept while the page is open.
    const choices = { format: "png", size: 200, transparent: true, scale: true, legend: true, support: true };

    // ---- What goes in the image ------------------------------------------------------------------------------------

    // The light theme's colours: the page switches to the light theme for the instant it takes to read them, and back.
    function lightPalette() {
        const root = document.documentElement;
        const theme = root.getAttribute("data-theme");
        if (theme !== "dark") return TV.readPalette();
        root.setAttribute("data-theme", "light");
        try {
            return TV.readPalette();
        } finally {
            root.setAttribute("data-theme", theme);
        }
    }

    // The scene the image is drawn from: the page's, in `palette`, with nothing under the pointer, nothing pinned, no
    // box being drawn, and the names drawn even while the view moves.
    function figureScene(palette) {
        return Object.assign({}, state, { palette, hover: -1, pinned: -1, box: null, moving: false });
    }

    // The drawing's size in CSS pixels, as the canvas has it.
    function drawingSize() {
        const canvas = TV.canvas;
        return { width: Math.max(1, Math.floor(canvas.clientWidth)), height: Math.max(1, Math.floor(canvas.clientHeight)) };
    }

    // For each leaf, 1 when its tip is in view.
    function leavesInView(width, height) {
        const { model, layout, view } = state;
        const seen = new Uint8Array(model.leafCount);
        for (let k = 0; k < model.leafCount; k++) {
            const node = model.leafNode[k];
            if (layout.live && !layout.live[node]) continue;
            const px = view.ox + layout.x[node] * view.sx;
            const py = view.oy + layout.y[node] * view.sy;
            if (px >= 0 && px <= width && py >= 0 && py <= height) seen[k] = 1;
        }
        return seen;
    }

    // The colour legend for the leaves in view, as groups { title, rows: [{ css, label }], gradient }: each layer of the
    // colour key with the rows that have a leaf in view, a gradient's bar when it has a value, and the unlabelled colours.
    function legendGroups(width, height) {
        const colors = state.colors;
        if (!colors || !state.model || colors.nodeClass.length !== state.model.nodeCount) return [];
        const seen = leavesInView(width, height);
        const groups = [];
        for (const layer of colors.layers || []) {
            const rows = layer.rows.filter((row) => row.leaves.some((k) => seen[k])).map((row) => ({ css: row.css, label: row.label }));
            const gradient = layer.kind === "gradient" && layer.bar && layer.bar.count > 0 && layer.min !== null && layer.max !== null
                && layer.stops.length ? { stops: layer.stops, min: layer.min, max: layer.max } : null;
            if (rows.length || gradient) groups.push({ title: layer.title, rows, gradient });
        }
        const unlabelled = (colors.unlabelled || []).filter((entry) => {
            for (let p = colors.starts[entry.cls]; p < colors.starts[entry.cls + 1]; p++) {
                const k = colors.positions[p];
                if (seen[k] && !colors.covered[k]) return true;
            }
            return false;
        });
        if (unlabelled.length) groups.push({ title: t("Unlabelled colours"), rows: unlabelled.map((entry) => ({ css: entry.css, label: entry.hex })), gradient: null });
        return groups;
    }

    // ---- The band under the tree -----------------------------------------------------------------------------------

    const BAND_FONT_PX = 12;
    const BAND_ROW_PX = 26;
    const BAND_PAD_PX = 10;
    const BAND_SIDE_PX = 20;
    const BAND_GAP_PX = 20;
    const SWATCH_PX = 10;
    const RAMP_STEP_PX = 14;
    const measurer = document.createElement("canvas").getContext("2d");

    const number = (value) => String(Number(value.toPrecision(4)));

    // What the band holds, each item { width, draw(ctx, x, y) } (y the middle of its row): the scale bar, then the
    // colours (a group's title goes with its first row, so that a title never ends a row), then the support dots.
    function bandItems(options, palette, width, height) {
        const font = BAND_FONT_PX + "px " + palette.font;
        const bold = "600 " + font;
        const textWidth = (text, weight) => {
            measurer.font = weight ? bold : font;
            return measurer.measureText(text).width;
        };
        const write = (ctx, text, x, y, weight) => {
            ctx.font = weight ? bold : font;
            ctx.fillStyle = palette.label;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x, y);
        };
        const items = [];
        const bar = options.scale ? TV.scaleBar(state) : null;
        if (bar) items.push({ width: bar.px, draw: (ctx, x, y) => TV.drawScaleBar(ctx, bar, palette, x, y + 7, false) });
        if (options.legend) {
            for (const group of legendGroups(width, height)) {
                const title = group.title ? group.title + ":" : "";
                const titleWidth = title ? textWidth(title, true) + 8 : 0;
                const pieces = group.rows.map((row) => ({
                    width: SWATCH_PX + 6 + textWidth(row.label),
                    draw(ctx, x, y) {
                        ctx.fillStyle = row.css;
                        ctx.fillRect(x, y - SWATCH_PX / 2, SWATCH_PX, SWATCH_PX);
                        write(ctx, row.label, x + SWATCH_PX + 6, y);
                    },
                }));
                if (group.gradient) {
                    const { stops, min, max } = group.gradient;
                    const low = number(min), high = number(max);
                    const step = Math.max(2, Math.round(84 / stops.length));
                    pieces.unshift({
                        width: textWidth(low) + 6 + step * stops.length + 6 + textWidth(high),
                        draw(ctx, x, y) {
                            write(ctx, low, x, y);
                            let at = x + textWidth(low) + 6;
                            for (const stop of stops) {
                                ctx.fillStyle = stop;
                                ctx.fillRect(at, y - 4, step, 8);
                                at += step;
                            }
                            write(ctx, high, at + 6, y);
                        },
                    });
                }
                if (!pieces.length) continue;
                const first = pieces[0];
                items.push({
                    width: titleWidth + first.width,
                    draw(ctx, x, y) {
                        if (title) write(ctx, title, x, y, true);
                        first.draw(ctx, x + titleWidth, y);
                    },
                });
                items.push(...pieces.slice(1));
            }
        }
        if (options.support) {
            const title = $("support-title").textContent;
            const low = "0%", high = "100%";
            const steps = palette.support.length;
            items.push({
                width: textWidth(title, true) + 8 + textWidth(low) + 6 + steps * RAMP_STEP_PX + 6 + textWidth(high),
                draw(ctx, x, y) {
                    write(ctx, title, x, y, true);
                    let at = x + textWidth(title, true) + 8;
                    write(ctx, low, at, y);
                    at += textWidth(low) + 6;
                    palette.support.forEach((color, k) => {
                        ctx.fillStyle = color;
                        ctx.fillRect(at + k * RAMP_STEP_PX, y - 4, RAMP_STEP_PX, 8);
                    });
                    write(ctx, high, at + steps * RAMP_STEP_PX + 6, y);
                },
            });
        }
        return items;
    }

    // The items in rows that fit `room` pixels.
    function rowsOf(items, room) {
        const rows = [];
        let row = [];
        let used = 0;
        for (const item of items) {
            if (row.length && used + BAND_GAP_PX + item.width > room) {
                rows.push(row);
                row = [];
                used = 0;
            }
            used += (row.length ? BAND_GAP_PX : 0) + item.width;
            row.push(item);
        }
        if (row.length) rows.push(row);
        return rows;
    }

    // What the image is made of: the drawing's size, the band's rows and height, in CSS pixels.
    function plan(options, palette) {
        const { width, height } = drawingSize();
        const rows = rowsOf(bandItems(options, palette, width, height), width - 2 * BAND_SIDE_PX);
        return { width, height, rows, band: rows.length ? rows.length * BAND_ROW_PX + 2 * BAND_PAD_PX : 0, palette };
    }

    function drawBand(ctx, made) {
        made.rows.forEach((row, r) => {
            const y = made.height + BAND_PAD_PX + r * BAND_ROW_PX + BAND_ROW_PX / 2;
            let x = BAND_SIDE_PX;
            for (const item of row) {
                item.draw(ctx, x, y);
                x += item.width + BAND_GAP_PX;
            }
        });
    }

    // The options the card has now: a switch that does not apply (no scale bar for a cladogram) is off.
    function currentOptions() {
        return {
            format: choices.format,
            size: choices.size,
            transparent: choices.transparent,
            scale: choices.scale && !$("image-scale").disabled,
            legend: choices.legend && !$("image-legend").disabled,
            support: choices.support && !$("image-support").disabled,
        };
    }

    // ---- PNG -------------------------------------------------------------------------------------------------------

    // The device pixels a PNG of `size` percent has, for the image `made` plans.
    function pngPixels(made, size) {
        const ratio = (window.devicePixelRatio || 1) * size / 100;
        return { ratio, width: Math.round(made.width * ratio), height: Math.round((made.height + made.band) * ratio) };
    }

    // The largest size, in steps of 25 percent, whose PNG browsers can still make.
    function largestSize(made) {
        const one = pngPixels(made, 100);
        const most = Math.min(MAX_SIDE / Math.max(one.width, one.height), Math.sqrt(MAX_AREA / (one.width * one.height)));
        return Math.max(25, Math.floor(most * 4) * 25);
    }

    // Why a PNG could not be made, for the console: the card says it in words of its own.
    const failed = (why) => Promise.reject(new Error(why));

    function pngBlob(options) {
        const made = plan(options, lightPalette());
        const { ratio, width, height } = pngPixels(made, options.size);
        if (width > MAX_SIDE || height > MAX_SIDE || width * height > MAX_AREA) return failed("larger than a canvas can be");
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return failed("no 2D context");
        if (!options.transparent) {
            ctx.fillStyle = made.palette.background;
            ctx.fillRect(0, 0, width, height);
        }
        // The tree is kept to its own rectangle: names that run over its edges stop there, as on the screen.
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, made.width * ratio, made.height * ratio);
        ctx.clip();
        TV.drawScene(ctx, figureScene(made.palette), made.width, made.height, ratio, { transparent: true });
        ctx.restore();
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        drawBand(ctx, made);
        return new Promise((resolve) => canvas.toBlob(resolve, "image/png")).then((blob) => blob || failed("the canvas made no PNG"));
    }

    // ---- SVG -------------------------------------------------------------------------------------------------------

    // A number as SVG writes it: to the hundredth.
    const num = (value) => String(Math.round(value * 100) / 100 || 0);
    const XML = { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&apos;" };
    const escapeXml = (text) => String(text).replace(/[&<>"']/g, (c) => XML[c]);

    // A CSS colour as SVG paint, { color, alpha }: the colour as #rrggbb and its alpha apart, as drawing programs read
    // it best. The canvas says what any CSS colour is in one of its two forms.
    const paints = new Map();
    function paintOf(css) {
        let paint = paints.get(css);
        if (paint) return paint;
        measurer.fillStyle = "#000000";
        measurer.fillStyle = css;
        const said = String(measurer.fillStyle);
        const rgba = /^rgba?\(\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\s*\)$/.exec(said);
        if (rgba) {
            const hex = "#" + rgba.slice(1, 4).map((v) => Number(v).toString(16).padStart(2, "0")).join("");
            paint = { color: hex, alpha: rgba[4] === undefined ? 1 : Number(rgba[4]) };
        } else {
            paint = { color: said, alpha: 1 };
        }
        paints.set(css, paint);
        return paint;
    }

    // What a canvas font says: { style, weight, size, family }.
    function fontOf(font) {
        const found = /^\s*(?:(italic|oblique)\s+)?(?:(normal|bold|bolder|lighter|[1-9]00)\s+)?([\d.]+)px\s+(.+)$/i.exec(font);
        if (!found) return { style: null, weight: null, size: 10, family: "sans-serif" };
        return { style: found[1] || null, weight: found[2] || null, size: Number(found[3]), family: found[4] };
    }

    // A path as SVG writes it, made the way a Path2D is (moveTo, lineTo, arc, rect, closePath).
    class SvgPath {
        constructor() {
            this.parts = [];
            this.x = null;
            this.y = null;
            this.startX = 0;
            this.startY = 0;
        }
        moveTo(x, y) {
            this.parts.push("M" + num(x) + " " + num(y));
            this.x = this.startX = x;
            this.y = this.startY = y;
        }
        lineTo(x, y) {
            if (this.x === null) {
                this.moveTo(x, y);
                return;
            }
            this.parts.push("L" + num(x) + " " + num(y));
            this.x = x;
            this.y = y;
        }
        closePath() {
            if (this.x === null) return;
            this.parts.push("Z");
            this.x = this.startX;
            this.y = this.startY;
        }
        rect(x, y, width, height) {
            this.moveTo(x, y);
            this.parts.push("h" + num(width) + "v" + num(height) + "h" + num(-width) + "Z");
        }
        // As a canvas draws an arc: a line from where the path is to the arc's start, then the arc, clockwise unless
        // `anticlockwise`; a whole turn or more is the whole circle, in two halves (one SVG arc cannot close on itself).
        arc(x, y, radius, start, end, anticlockwise) {
            const turn = 2 * Math.PI;
            const x0 = x + radius * Math.cos(start);
            const y0 = y + radius * Math.sin(start);
            if (this.x === null) this.moveTo(x0, y0);
            else if (Math.abs(this.x - x0) > 1e-6 || Math.abs(this.y - y0) > 1e-6) this.lineTo(x0, y0);
            if (!(radius > 0)) return;
            const r = num(radius);
            const sweep = anticlockwise ? 0 : 1;
            let span = anticlockwise ? start - end : end - start;
            if (span >= turn) {
                const xm = x - radius * Math.cos(start);
                const ym = y - radius * Math.sin(start);
                this.parts.push("A" + r + " " + r + " 0 1 " + sweep + " " + num(xm) + " " + num(ym)
                    + "A" + r + " " + r + " 0 1 " + sweep + " " + num(x0) + " " + num(y0));
                return;
            }
            span = ((span % turn) + turn) % turn;
            if (span === 0) return;
            const x1 = x + radius * Math.cos(end);
            const y1 = y + radius * Math.sin(end);
            this.parts.push("A" + r + " " + r + " 0 " + (span > Math.PI ? 1 : 0) + " " + sweep + " " + num(x1) + " " + num(y1));
            this.x = x1;
            this.y = y1;
        }
        get d() {
            return this.parts.join("");
        }
    }

    // What the drawing does with a 2D canvas, written down as SVG: the calls tree_render.js and the band make, no more.
    class SvgContext {
        constructor() {
            this.out = [];
            this.stack = [];
            this.fillStyle = "#000000";
            this.strokeStyle = "#000000";
            this.lineWidth = 1;
            this.lineCap = "butt";
            this.lineJoin = "miter";
            this.globalAlpha = 1;
            this.font = "10px sans-serif";
            this.textAlign = "start";
            this.textBaseline = "alphabetic";
            this.dash = [];
            this.matrix = [1, 0, 0, 1, 0, 0];
            this.path = new SvgPath();
        }
        newPath() {
            return new SvgPath();
        }
        save() {
            const { fillStyle, strokeStyle, lineWidth, lineCap, lineJoin, globalAlpha, font, textAlign, textBaseline } = this;
            this.stack.push({ fillStyle, strokeStyle, lineWidth, lineCap, lineJoin, globalAlpha, font, textAlign, textBaseline,
                dash: this.dash.slice(), matrix: this.matrix.slice() });
        }
        restore() {
            const saved = this.stack.pop();
            if (saved) Object.assign(this, saved);
        }
        setTransform(a, b, c, d, e, f) {
            this.matrix = [a, b, c, d, e, f];
        }
        transform(a, b, c, d, e, f) {
            const [A, B, C, D, E, F] = this.matrix;
            this.matrix = [A * a + C * b, B * a + D * b, A * c + C * d, B * c + D * d, A * e + C * f + E, B * e + D * f + F];
        }
        translate(x, y) {
            this.transform(1, 0, 0, 1, x, y);
        }
        rotate(angle) {
            const c = Math.cos(angle), s = Math.sin(angle);
            this.transform(c, s, -s, c, 0, 0);
        }
        scale(x, y) {
            this.transform(x, 0, 0, y, 0, 0);
        }
        setLineDash(list) {
            this.dash = Array.from(list || []);
        }
        getLineDash() {
            return this.dash.slice();
        }
        beginPath() {
            this.path = new SvgPath();
        }
        moveTo(x, y) { this.path.moveTo(x, y); }
        lineTo(x, y) { this.path.lineTo(x, y); }
        arc(x, y, radius, start, end, anticlockwise) { this.path.arc(x, y, radius, start, end, anticlockwise); }
        rect(x, y, width, height) { this.path.rect(x, y, width, height); }
        closePath() { this.path.closePath(); }
        clearRect() {}
        measureText(text) {
            measurer.font = this.font;
            return measurer.measureText(text);
        }

        placed() {
            const [a, b, c, d, e, f] = this.matrix;
            if (Math.abs(a - 1) < 1e-9 && Math.abs(b) < 1e-9 && Math.abs(c) < 1e-9 && Math.abs(d - 1) < 1e-9
                && Math.abs(e) < 1e-9 && Math.abs(f) < 1e-9) return "";
            const k = (v) => String(Math.round(v * 10000) / 10000 || 0);
            return ` transform="matrix(${k(a)} ${k(b)} ${k(c)} ${k(d)} ${num(e)} ${num(f)})"`;
        }
        filled() {
            const paint = paintOf(this.fillStyle);
            const alpha = paint.alpha * this.globalAlpha;
            return ` fill="${escapeXml(paint.color)}"` + (alpha < 1 ? ` fill-opacity="${Math.round(alpha * 1000) / 1000}"` : "");
        }
        stroked() {
            const paint = paintOf(this.strokeStyle);
            const alpha = paint.alpha * this.globalAlpha;
            return ` fill="none" stroke="${escapeXml(paint.color)}" stroke-width="${num(this.lineWidth)}"`
                + (this.lineCap !== "butt" ? ` stroke-linecap="${this.lineCap}"` : "")
                + (this.lineJoin !== "miter" ? ` stroke-linejoin="${this.lineJoin}"` : "")
                + (this.dash.length ? ` stroke-dasharray="${this.dash.map(num).join(" ")}"` : "")
                + (alpha < 1 ? ` stroke-opacity="${Math.round(alpha * 1000) / 1000}"` : "");
        }
        fill(path) {
            const d = (path instanceof SvgPath ? path : this.path).d;
            if (d) this.out.push(`<path d="${d}"${this.filled()}${this.placed()}/>`);
        }
        stroke(path) {
            const d = (path instanceof SvgPath ? path : this.path).d;
            if (d) this.out.push(`<path d="${d}"${this.stroked()}${this.placed()}/>`);
        }
        box(x, y, width, height) {
            return ` x="${num(Math.min(x, x + width))}" y="${num(Math.min(y, y + height))}" width="${num(Math.abs(width))}" height="${num(Math.abs(height))}"`;
        }
        fillRect(x, y, width, height) {
            this.out.push(`<rect${this.box(x, y, width, height)}${this.filled()}${this.placed()}/>`);
        }
        strokeRect(x, y, width, height) {
            this.out.push(`<rect${this.box(x, y, width, height)}${this.stroked()}${this.placed()}/>`);
        }
        // A canvas places text by its baseline; SVG readers differ on dominant-baseline, so the middle and top are a shift.
        fillText(text, x, y) {
            const font = fontOf(this.font);
            const anchor = { center: "middle", right: "end", end: "end" }[this.textAlign];
            const shift = { middle: "0.35em", top: "0.8em", hanging: "0.8em", bottom: "-0.25em", ideographic: "-0.25em" }[this.textBaseline];
            this.out.push(`<text x="${num(x)}" y="${num(y)}"` + (shift ? ` dy="${shift}"` : "")
                + ` font-family="${escapeXml(font.family)}" font-size="${num(font.size)}"`
                + (font.weight && font.weight !== "normal" ? ` font-weight="${font.weight}"` : "")
                + (font.style ? ` font-style="${font.style}"` : "")
                + (anchor ? ` text-anchor="${anchor}"` : "")
                + `${this.filled()}${this.placed()}>${escapeXml(text)}</text>`);
        }
        open(attributes) {
            this.out.push(`<g${attributes}>`);
        }
        close() {
            this.out.push("</g>");
        }
        document(width, height, defs) {
            return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${num(width)}" height="${num(height)}" viewBox="0 0 ${num(width)} ${num(height)}">\n`
                + (defs ? defs + "\n" : "") + this.out.join("\n") + "\n</svg>\n";
        }
    }
    TV.SvgContext = SvgContext;

    function svgText(options) {
        const made = plan(options, lightPalette());
        const width = made.width, height = made.height + made.band;
        const ctx = new SvgContext();
        if (!options.transparent) {
            ctx.fillStyle = made.palette.background;
            ctx.fillRect(0, 0, width, height);
        }
        ctx.open(' clip-path="url(#tree-view-drawing)"');
        TV.drawScene(ctx, figureScene(made.palette), made.width, made.height, 1, { transparent: true, vector: true });
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.close();
        drawBand(ctx, made);
        const defs = `<defs><clipPath id="tree-view-drawing"><rect width="${num(made.width)}" height="${num(made.height)}"/></clipPath></defs>`;
        return ctx.document(width, height, defs);
    }

    // ---- Saving ----------------------------------------------------------------------------------------------------

    // The image for `options` ({ format, size, transparent, scale, legend, support }): { blob, name }.
    TV.buildImage = async function (options) {
        if (options.format === "svg") {
            return { blob: new Blob([svgText(options)], { type: "image/svg+xml" }), name: FILE_NAME + ".svg" };
        }
        return { blob: await pngBlob(options), name: FILE_NAME + ".png" };
    };

    // Hand the file to the browser, which saves it where it saves downloads (the page runs in the user's browser).
    TV.downloadImage = function (blob, name) {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = name;
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 10000);
    };

    async function save() {
        const button = $("image-save");
        $("image-error").textContent = "";
        button.disabled = true;
        try {
            const { blob, name } = await TV.buildImage(currentOptions());
            TV.downloadImage(blob, name);
            TV.closeCard("image-card", true);
            TV.say(t("Saved {file}.", { file: name }));
        } catch (error) {
            console.warn("Could not save the image:", error);
            $("image-error").textContent = t("The image could not be made. A smaller size may work.");
        } finally {
            button.disabled = false;
            const card = $("image-card");
            if (card.classList.contains("shown") && !card.contains(document.activeElement)) button.focus();
        }
    }

    // ---- The card --------------------------------------------------------------------------------------------------

    // Say what the image will be: a PNG's size in pixels, or that an SVG has none.
    function showPixels() {
        const svg = choices.format === "svg";
        const input = $("image-size");
        input.disabled = svg;
        if (svg) {
            $("image-pixels").textContent = t("Lines and text, at any size");
            return;
        }
        const made = plan(currentOptions(), state.palette || TV.readPalette());
        input.max = String(largestSize(made));
        const pixels = pngPixels(made, Math.min(choices.size, Number(input.max)));
        $("image-pixels").textContent = t("{width} × {height} px", {
            width: pixels.width.toLocaleString(), height: pixels.height.toLocaleString(),
        });
    }

    function chooseFormat(format) {
        choices.format = format;
        $("image-png").setAttribute("aria-pressed", String(format === "png"));
        $("image-svg").setAttribute("aria-pressed", String(format === "svg"));
        showPixels();
    }

    // Fit the card to the tree and the view in hand, as it opens: a switch with nothing to show is off and greyed.
    TV.prepareImageCard = function () {
        const model = state.model;
        $("image-error").textContent = "";
        $("image-scale").disabled = !TV.scaleBar(state);
        const colors = state.colors;
        $("image-legend").disabled = !(colors && ((colors.layers && colors.layers.length) || (colors.unlabelled && colors.unlabelled.length)));
        $("image-support").disabled = !(model && TV.hasSupport(model) && state.showSupport);
        $("image-support-label").textContent = model && model.source === "uploaded" ? t("Support legend") : t("Split stability legend");
        for (const [id, key] of [["image-transparent", "transparent"], ["image-scale", "scale"], ["image-legend", "legend"], ["image-support", "support"]]) {
            const box = $(id);
            box.checked = choices[key] && !box.disabled;
        }
        $("image-size").value = String(choices.size);
        chooseFormat(choices.format);
    };

    $("image-png").addEventListener("click", () => chooseFormat("png"));
    $("image-svg").addEventListener("click", () => chooseFormat("svg"));
    $("image-size").addEventListener("input", (event) => {
        const value = Number(event.target.value);
        if (Number.isFinite(value) && value > 0) {
            choices.size = value;
            showPixels();
        }
    });
    // Once typed, the size is put in the range the PNG can have.
    $("image-size").addEventListener("change", (event) => {
        const input = event.target;
        const most = Number(input.max) || 800;
        const value = Number(input.value);
        choices.size = Math.min(most, Math.max(25, Number.isFinite(value) && value > 0 ? Math.round(value) : 200));
        input.value = String(choices.size);
        showPixels();
    });
    for (const [id, key] of [["image-transparent", "transparent"], ["image-scale", "scale"], ["image-legend", "legend"], ["image-support", "support"]]) {
        $(id).addEventListener("change", (event) => {
            choices[key] = event.target.checked;
            showPixels();
        });
    }
    $("image-close").addEventListener("click", () => TV.closeCard("image-card", true));
    $("image-cancel").addEventListener("click", () => TV.closeCard("image-card", true));
    $("image-save").addEventListener("click", save);
})(window.TreeView);
