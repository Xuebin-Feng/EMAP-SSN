/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's Color by card: one choice of coloring for the whole scene, its legend, and a compact legend on the canvas.
 */

// The scene has one coloring at a time (pLDDT, a structure's own color, a
// layer of the Viewer's residue data, or one of Mol*'s secondary-structure
// and residue-position colorings), and each structure follows it; a
// structure's row can give it a coloring of its own. A structure that lacks
// the chosen layer shows the listing's grey and says "No data", because
// keeping its pLDDT colors would put two meanings under one legend. What
// is drawn is read from Mol* (a representation's color theme and the layer
// that is attached), not kept in a copy; only the scene's choice is kept
// here (colorMode), and a restored session works it out again from the
// structures. The layers' own logic is esmfold_layers.js (applyLayer,
// restorePlddt, reconcileLayers), which this file calls and does not change.
// Loaded after esmfold_session.js.

const HEX_COLOR = /^#[0-9a-f]{6}$/i;
const BUILTIN_PREFIX = 'builtin:';
const STRUCTURE_COLORING = 'builtin:structure';
const SECONDARY_COLORING = 'builtin:secondary-structure';
const POSITION_COLORING = 'builtin:residue-position';
// What a structure shows when it has no values for the chosen layer: not a choice, a state.
const NO_DATA = 'nodata';
const OTHER_COLORING = 'other';
// The most layers shown as buttons before the card adds a filter and a scroll area.
const MAX_PLAIN_LAYERS = 8;
let colorMode = PLDDT_LAYER;
let noDataColor = 0xdcdcdc;
let colorFilter = '';
let moreColorings = false;
let colorRenderId = 0;
let renderedColors = '';
// The listings of the Viewer as they came, by file, so that a row can offer a structure's layers without waiting.
const resolvedListings = new Map();
const structureColors = new Map();
const themeLegends = new Map();

function hexToInt(hex) {
    return parseInt(hex.slice(1), 16);
}

function intToHex(value) {
    return '#' + value.toString(16).padStart(6, '0');
}

// ---- What each structure shows ---------------------------------------------------------
// A structure's own color: the first of the palette not taken by another, and the same one each time it is asked.
function structureColor(ref) {
    const present = new Set(loadedStructures().map(structure => structure.cell.transform.ref));
    for (const known of [...structureColors.keys()]) {
        if (!present.has(known)) structureColors.delete(known);
    }
    if (!structureColors.has(ref)) {
        const used = new Set(structureColors.values());
        structureColors.set(ref, STRUCTURE_COLORS.find(color => !used.has(color)) || STRUCTURE_COLORS[structureColors.size % STRUCTURE_COLORS.length]);
    }
    return structureColors.get(ref);
}

function colorThemeOf(structure) {
    return representationRefs(structure)[0]?.cell.transform.params.colorTheme;
}

// The coloring a structure shows now, as an id: PLDDT_LAYER, a layer's id, a built-in's, NO_DATA or OTHER_COLORING.
function coloringOf(structure) {
    const theme = colorThemeOf(structure);
    if (!theme) return PLDDT_LAYER;
    switch (theme.name) {
        case 'mvs-annotation': return activeLayerId(structure);
        case 'plddt-confidence': return PLDDT_LAYER;
        case 'secondary-structure': return SECONDARY_COLORING;
        case 'sequence-id': return POSITION_COLORING;
        case 'uniform': {
            const value = theme.params?.value;
            if (value === noDataColor) return NO_DATA;
            return STRUCTURE_COLORS.some(color => hexToInt(color) === value) ? STRUCTURE_COLORING : OTHER_COLORING;
        }
        default: return OTHER_COLORING;
    }
}

function themeFor(structure, id) {
    if (id === STRUCTURE_COLORING) {
        return { color: 'uniform', colorParams: { value: hexToInt(structureColor(structure.cell.transform.ref)) } };
    }
    if (id === SECONDARY_COLORING) return { color: 'secondary-structure' };
    if (id === POSITION_COLORING) return { color: 'sequence-id' };
    return { color: 'uniform', colorParams: { value: noDataColor } };
}

// What a color is called, for a button, a select and a chip.
function coloringLabel(id, layers = []) {
    if (id === PLDDT_LAYER) return 'pLDDT';
    if (id === STRUCTURE_COLORING) return t("Structure color");
    if (id === SECONDARY_COLORING) return t("Secondary structure");
    if (id === POSITION_COLORING) return t("Residue position");
    if (id === NO_DATA) return t("No data");
    if (id === OTHER_COLORING) return t("Other (set in Advanced)");
    return layers.find(layer => layer.id === id)?.label ?? id;
}

// ---- Changing it ----------------------------------------------------------------------------
async function themePlddt(structure) {
    try {
        await themeStructure(structure, { color: 'plddt-confidence' });
    } catch (e1) {
        try {
            await plugin.managers.structure.component.updateRepresentationsTheme(structure.components, {
                colorTheme: { name: 'plddt-confidence' }
            });
        } catch (e2) {
            console.warn("Failed to apply pLDDT theme:", e1, e2);
        }
    }
}

// Colors first, then lets go of the layer, while the annotation theme is still applicable.
async function colorWithoutLayer(structure, theme) {
    await themeStructure(structure, theme);
    if (activeLayerId(structure) !== PLDDT_LAYER) {
        await setLayerProperties(structure, null);
        layersChanged();
    }
}

// Colors one structure by `id`, whatever it shows now; a layer it has no values for is grey.
async function colorStructure(structure, id) {
    if (coloringOf(structure) === id) return;
    if (id === PLDDT_LAYER) {
        if (activeLayerId(structure) !== PLDDT_LAYER) await restorePlddt(structure);
        else await themePlddt(structure);
    } else if (id.startsWith(BUILTIN_PREFIX)) {
        await colorWithoutLayer(structure, themeFor(structure, id));
    } else {
        const file = structureFile(structure);
        const listing = file && await layerListing(file);
        if (listing?.layers.some(entry => entry.id === id && entry.annotation)) await applyLayer(structure, id);
        else await colorWithoutLayer(structure, themeFor(structure, NO_DATA));
    }
}

function colorsChanged() {
    renderColorCard();
    scheduleStructures();
    setTimeout(saveSession, 1500);
}

// The scene's choice: every structure follows it, together, one after another.
function setColorMode(id) {
    colorMode = id;
    renderColorCard();
    return runExclusive(async () => {
        for (const structure of loadedStructures()) {
            try {
                await colorStructure(structure, id);
            } catch (err) {
                console.warn("Could not apply the layer:", err);
                showStatus(t("Could not color {file} by that layer.", {file: structureName(structure)}));
            }
        }
        colorsChanged();
    }).catch(err => reportFailure("Could not change the colors:", err));
}

// One structure on its own; "" gives it back to the scene's.
function setStructureColoring(ref, id) {
    return runExclusive(async () => {
        const structure = structureByRef(ref);
        if (structure) await colorStructure(structure, id || colorMode);
        colorsChanged();
    }).catch(err => reportFailure("Could not change the colors:", err));
}

// A structure just loaded takes the scene's coloring (the others keep theirs), except that one opened from a
// file starts in its own color when the scene's is pLDDT: its B-factors are not known to be a confidence. Its
// row has that choice, and pLDDT is one click away there; the scene's next choice is for it too.
async function colorNewStructure(structure) {
    try {
        const own = !isFolded(structure) && colorMode === PLDDT_LAYER;
        await colorStructure(structure, own ? STRUCTURE_COLORING : colorMode);
    } catch (err) {
        console.warn("Could not color the new structure:", err);
    }
}

// After a session came back (or Undo, or an opened file): the scene's choice is the one most structures show.
// A structure that was opened from a file is not counted: it starts in a color of its own, and opening one
// never changes the scene's choice.
function deriveColorMode() {
    const counts = new Map();
    for (const structure of loadedStructures()) {
        if (!isFolded(structure)) {
            // (Its own color is its own: the swatch keeps matching it.)
            const own = colorThemeOf(structure);
            if (coloringOf(structure) === STRUCTURE_COLORING) structureColors.set(structure.cell.transform.ref, intToHex(own.params.value));
            continue;
        }
        const id = coloringOf(structure);
        if (id === NO_DATA || id === OTHER_COLORING) continue;
        counts.set(id, (counts.get(id) || 0) + 1);
        const theme = colorThemeOf(structure);
        if (id === STRUCTURE_COLORING) structureColors.set(structure.cell.transform.ref, intToHex(theme.params.value));
    }
    let best = null;
    for (const [id, count] of counts) {
        if (best === null || count > counts.get(best)) best = id;
    }
    colorMode = best ?? PLDDT_LAYER;
    renderColorCard();
}

// ---- What the card shows --------------------------------------------------------------------
function gatherColoring() {
    const structures = loadedStructures().map(structure => ({
        structure, ref: structure.cell.transform.ref, name: structureName(structure), file: structureFile(structure)
    }));
    return Promise.all(structures.map(item => (item.file ? layerListing(item.file) : null))).then(listings => {
        const layers = [];
        const unavailable = [];
        let plddt = null;
        structures.forEach((item, index) => {
            const listing = listings[index];
            if (item.file && listing) resolvedListings.set(item.file, listing);
            if (listing?.background) noDataColor = hexToInt(listing.background);
            if (!listing) return;
            plddt = plddt || listing.layers.find(entry => entry.id === PLDDT_LAYER) || null;
            if (listing.unavailable) unavailable.push({ name: item.name, reason: listing.unavailable });
            for (const entry of listing.layers) {
                if (entry.id === PLDDT_LAYER || !entry.annotation) continue;
                let layer = layers.find(known => known.id === entry.id);
                if (!layer) {
                    layer = { id: entry.id, label: entry.label, unit: entry.unit, description: entry.description,
                              source: entry.source, legend: entry.legend, have: [], partial: false };
                    layers.push(layer);
                }
                layer.have.push(item.name);
                if (entry.covered < entry.residues) layer.partial = true;
            }
        });
        for (const layer of layers) layer.lack = structures.map(item => item.name).filter(name => !layer.have.includes(name));
        return { structures, layers, unavailable, plddt };
    });
}

// A color theme's own legend, from Mol*, for the built-in colorings.
function themeLegend(name) {
    const first = loadedStructures()[0];
    if (!first) return null;
    if (!themeLegends.has(name)) {
        const registry = plugin.representation.structure.themes.colorThemeRegistry;
        themeLegends.set(name, registry.create(name, { structure: first.cell.obj.data }, {}).legend ?? null);
    }
    return themeLegends.get(name);
}

// pLDDT's four bands, very high first, in the Viewer's words when a listing has them (`entry` is its pLDDT layer).
function plddtBands(entry) {
    return entry?.legend.bands ?? [
        { color: '#0053d6', label: t("Very high (pLDDT > 90)") }, { color: '#65cbf3', label: t("Confident (pLDDT 70 to 90)") },
        { color: '#ffdb13', label: t("Low (pLDDT 50 to 70)") }, { color: '#ff7d45', label: t("Very low (pLDDT < 50)") }
    ];
}

// The legend of the chosen coloring, as plain data: {title, unit, kind, ...}.
function legendOf(model) {
    const id = colorMode;
    if (id === PLDDT_LAYER) {
        const bands = plddtBands(model.plddt);
        return { heading: coloringLabel(id), unit: '', kind: 'bands', bands, description: model.plddt?.description ?? '', source: '' };
    }
    if (id === STRUCTURE_COLORING) {
        return { heading: coloringLabel(id), unit: '', kind: 'note', note: t("Each structure has its own color, the one beside its name.") };
    }
    if (id === SECONDARY_COLORING) {
        const table = new Map((themeLegend('secondary-structure')?.table ?? []).map(([name, color]) => [name, intToHex(color)]));
        const bands = [['alphaHelix', t("Helix")], ['betaStrand', t("Strand")], ['coil', t("Coil")]]
            .filter(([key]) => table.has(key)).map(([key, label]) => ({ color: table.get(key), label }));
        return { heading: coloringLabel(id), unit: '', kind: 'table', bands };
    }
    if (id === POSITION_COLORING) {
        const colors = (themeLegend('sequence-id')?.colors ?? []).map(intToHex);
        return { heading: coloringLabel(id), unit: '', kind: 'scale', colors, low: t("N-terminus"), high: t("C-terminus") };
    }
    const layer = model.layers.find(known => known.id === id);
    if (!layer) return null;
    return { heading: layer.label, unit: layer.unit, kind: layer.legend.kind, legend: layer.legend, description: layer.description,
             source: layer.source, partial: layer.partial, lack: layer.lack };
}

// The model of the card: all that it draws, as data, so that it is drawn again only when it changes.
async function colorModel() {
    const gathered = await gatherColoring();
    const model = { mode: colorMode, filter: colorFilter, more: moreColorings, layers: gathered.layers,
                    unavailable: gathered.unavailable, plddt: gathered.plddt, count: gathered.structures.length };
    model.legend = gathered.structures.length ? legendOf(model) : null;
    model.noData = gathered.structures.filter(item => coloringOf(item.structure) === NO_DATA).map(item => item.name);
    // Opened files showing pLDDT: the colors are the file's B-factors.
    model.fromFiles = gathered.structures.filter(item => !isFolded(item.structure) && coloringOf(item.structure) === PLDDT_LAYER)
        .map(item => item.name);
    return model;
}

function colorChip(id, label, model, extra) {
    const chip = make('button', 'seg-btn');
    chip.type = 'button';
    chip.title = label;
    const text = make('span', 'seg-label', label);
    // The same text, bold and unseen, keeps the chip as wide as it is when chosen (esmfold.html).
    text.dataset.text = label;
    chip.append(text);
    chip.setAttribute('role', 'radio');
    chip.setAttribute('aria-checked', model.mode === id ? 'true' : 'false');
    chip.dataset.coloring = id;
    chip.dataset.focusKey = 'color:' + id;
    if (extra) {
        const badge = make('span', 'cov', extra);
        chip.append(badge);
    }
    chip.addEventListener('click', () => {
        if (colorMode !== id) setColorMode(id);
    });
    return chip;
}

function groupLabel(text) {
    return make('div', 'group-label', text);
}

// The colors of a legend's bar as [position from 0 to 1, color] stops, and the names of its two ends. A gradient
// runs through its stops; a scale through its colors; bands are blocks, each as wide as its pLDDT band (or equal parts).
function legendRamp(legend) {
    if (legend.kind === 'gradient') {
        const span = legend.legend.max - legend.legend.min;
        const unit = legend.unit ? ' ' + legend.unit : '';
        return { stops: legend.legend.stops.filter(([, color]) => HEX_COLOR.test(color))
                     .map(([value, color]) => [(value - legend.legend.min) / span, color]),
                 low: legend.legend.min + unit, high: legend.legend.max + unit };
    }
    if (legend.kind === 'scale') {
        return { stops: legend.colors.map((color, index) => [index / Math.max(1, legend.colors.length - 1), color]),
                 low: legend.low, high: legend.high };
    }
    // Very low to very high, as wide as the bands are in pLDDT (0 to 50, 50 to 70, 70 to 90, 90 to 100).
    const ordered = [...legend.bands].reverse();
    const confidence = legend.heading === coloringLabel(PLDDT_LAYER);
    const widths = confidence && ordered.length === 4 ? [0.5, 0.2, 0.2, 0.1] : ordered.map(() => 1 / ordered.length);
    let at = 0;
    const stops = [];
    ordered.forEach((band, index) => {
        stops.push([at, band.color]);
        at += widths[index];
        stops.push([at, band.color]);
    });
    return { stops, low: confidence ? '0' : '', high: confidence ? '100' : '' };
}

// A bar of the legend's colors with its two ends named.
function legendBar(legend, compact) {
    const wrap = make('div', compact ? 'legend-compact' : 'legend-body');
    const bar = make('div', 'ramp');
    const ramp = legendRamp(legend);
    bar.style.background = 'linear-gradient(to right, ' + ramp.stops.map(([at, color]) => `${color} ${(100 * at).toFixed(1)}%`).join(', ') + ')';
    const ends = make('div', 'ticks');
    const lowEnd = make('span', undefined, ramp.low);
    const highEnd = make('span', undefined, ramp.high);
    ends.append(lowEnd, highEnd);
    wrap.append(bar, ends);
    return wrap;
}

function legendKey(color, text) {
    const key = make('div', 'band');
    const swatch = make('i');
    swatch.style.setProperty('--bc', HEX_COLOR.test(color) ? color : 'transparent');
    const label = make('span', undefined, text);
    key.append(swatch, label);
    return key;
}

function legendTitle(legend, tag) {
    const title = make(tag, 't', legend.heading);
    if (legend.unit) {
        const unit = make('small', undefined, '(' + legend.unit + ')');
        title.append(' ', unit);
    }
    return title;
}

function cardLegend(model) {
    const legend = model.legend;
    const box = make('div', 'legend');
    box.id = 'color-legend';
    const heading = legendTitle(legend, 'div');
    box.append(heading);
    if (legend.kind === 'gradient' || legend.kind === 'scale') {
        box.append(legendBar(legend, false));
    } else if (legend.kind === 'bands' || legend.kind === 'table') {
        const keys = make('div', 'bands');
        keys.append(...legend.bands.map(band => legendKey(band.color, band.label)));
        box.append(keys);
    } else if (legend.kind === 'note') {
        const note = make('div', 'legend-note', legend.note);
        box.append(note);
    }
    if (legend.partial) {
        const some = legendKey(intToHex(noDataColor), t("No data"));
        some.classList.add('band-nodata');
        box.append(some);
    }
    if (colorMode === PLDDT_LAYER && model.fromFiles.length) {
        const caution = make('div', 'caution', t("{names}: the colors are the file's B-factor column. In an experimental structure that is not confidence.", {names: model.fromFiles.join(', ')}));
        box.append(caution);
    }
    if (model.noData.length && colorMode !== PLDDT_LAYER && !colorMode.startsWith(BUILTIN_PREFIX)) {
        const warning = make('div', 'nodata');
        const swatch = make('i');
        const words = make('span', undefined, t("No data: {names}. This layer has no values for them.", {names: model.noData.join(', ')}));
        warning.append(swatch, words);
        box.append(warning);
    }
    const text = legend.source || legend.description;
    if (text) {
        const details = make('details', 'src');
        const summary = make('summary');
        const info = icon("info");
        const line = make('span', undefined, legend.source || legend.description);
        const chevron = icon("chevron-down", 'chev');
        summary.append(info, line, chevron);
        details.append(summary);
        if (legend.source && legend.description) {
            const more = make('p', undefined, legend.description);
            details.append(more);
        }
        box.append(details);
    }
    return box;
}

function compactLegend(model) {
    const legend = model.legend;
    const box = byId('canvas-legend');
    if (!legend || legend.kind === 'note') {
        box.replaceChildren();
        box.dataset.shown = '0';
        return;
    }
    box.dataset.shown = '1';
    const parts = [legendTitle(legend, 'div')];
    if (legend.kind === 'table') {
        const keys = make('div', 'bands compact-keys');
        keys.append(...legend.bands.map(band => legendKey(band.color, band.label)));
        parts.push(keys);
    } else {
        parts.push(legendBar(legend, true));
    }
    if (model.noData.length && !colorMode.startsWith(BUILTIN_PREFIX) && colorMode !== PLDDT_LAYER) {
        parts.push(legendKey(intToHex(noDataColor), t("No data")));
    }
    box.replaceChildren(...parts);
}

function renderColorBody(model) {
    const body = byId('color-body');
    const focused = body.contains(document.activeElement) ? document.activeElement.dataset.focusKey : null;
    const filterFocused = document.activeElement === body.querySelector('.filter');
    const nodes = [];
    const group = make('div', 'seg-group');
    group.setAttribute('role', 'radiogroup');
    group.setAttribute('aria-label', t("Color by"));

    const modelLabel = groupLabel(t("Model"));
    const modelChips = make('div', 'seg');
    modelChips.append(colorChip(STRUCTURE_COLORING, coloringLabel(STRUCTURE_COLORING), model),
                      colorChip(PLDDT_LAYER, coloringLabel(PLDDT_LAYER), model));
    group.append(modelLabel, modelChips);

    const layersLabel = groupLabel(t("Layers"));
    group.append(layersLabel);
    const many = model.layers.length > MAX_PLAIN_LAYERS;
    if (many) {
        const filter = make('input', 'filter');
        filter.type = 'search';
        filter.placeholder = t("Filter layers");
        filter.setAttribute('aria-label', t("Filter layers"));
        filter.value = model.filter;
        filter.addEventListener('input', () => {
            colorFilter = filter.value;
            renderColorCard(true);
        });
        group.append(filter);
    }
    const shown = model.layers.filter(layer => !many || layer.label.toLowerCase().includes(model.filter.trim().toLowerCase()));
    if (shown.length) {
        const layerChips = make('div', many ? 'seg scroll' : 'seg');
        layerChips.append(...shown.map(layer => colorChip(layer.id, layer.label, model,
            layer.lack.length ? t("{n} of {total}", {n: layer.have.length, total: model.count}) : '')));
        group.append(layerChips);
    } else if (many) {
        const none = make('div', 'hint', t("No layer matches."));
        group.append(none);
    }
    nodes.push(group);

    if (model.count && (!model.layers.length || model.unavailable.length)) {
        const empty = make('div', 'empty-layers');
        const lines = make('div');
        if (!model.layers.length) {
            const advice = make('div', undefined, t("Layers such as conservation need an alignment: fold with one loaded in the Viewer."));
            lines.append(advice);
        }
        for (const item of model.unavailable) {
            const reason = make('div', undefined, item.name + ': ' + item.reason);
            lines.append(reason);
        }
        empty.append(lines);
        nodes.push(empty);
    }

    const toggle = make('button', 'more-toggle');
    toggle.type = 'button';
    toggle.setAttribute('aria-expanded', model.more ? 'true' : 'false');
    toggle.dataset.focusKey = 'color:more';
    const toggleLabel = make('span', undefined, t("More colorings"));
    const toggleIcon = icon("chevron-down");
    toggle.append(toggleIcon, toggleLabel);
    toggle.addEventListener('click', () => {
        moreColorings = !moreColorings;
        renderColorCard(true);
    });
    nodes.push(toggle);
    if (model.more) {
        const more = make('div', 'seg more-colorings');
        more.setAttribute('role', 'radiogroup');
        more.setAttribute('aria-label', t("More colorings"));
        more.append(colorChip(SECONDARY_COLORING, coloringLabel(SECONDARY_COLORING), model),
                    colorChip(POSITION_COLORING, coloringLabel(POSITION_COLORING), model));
        nodes.push(more);
    }
    if (model.legend) nodes.push(cardLegend(model));
    body.replaceChildren(...nodes);
    // One tab stop for the group: the chosen chip, or the first.
    const chips = [...body.querySelectorAll('[role="radio"]')];
    const current = chips.find(chip => chip.getAttribute('aria-checked') === 'true') || chips[0];
    for (const chip of chips) chip.tabIndex = chip === current ? 0 : -1;
    if (filterFocused) {
        const filter = body.querySelector('.filter');
        filter.focus();
        filter.setSelectionRange(filter.value.length, filter.value.length);
    } else if (focused) {
        body.querySelector('[data-focus-key="' + focused.replace(/["\\]/g, '\\$&') + '"]')?.focus();
    }
}

// Draws the card and the legend on the canvas from the structures as they are; `force` draws even if nothing changed.
async function renderColorCard(force = false) {
    if (!plugin) return;
    const renderId = ++colorRenderId;
    let model;
    try {
        model = await colorModel();
    } catch (err) {
        console.warn("Could not draw the colors:", err);
        return;
    }
    // A newer draw replaces this one, which an await made stale.
    if (renderId !== colorRenderId) return;
    const signature = JSON.stringify(model);
    if (!force && signature === renderedColors) return;
    renderedColors = signature;
    renderColorBody(model);
    compactLegend(model);
}

// Arrow keys move through the colorings and choose them, as radio buttons do.
byId('color-body').addEventListener('keydown', event => {
    if (!event.target.matches?.('[role="radio"]')) return;
    const chips = [...byId('color-body').querySelectorAll('[role="radio"]')];
    const at = chips.indexOf(event.target);
    const target = { ArrowRight: at + 1, ArrowDown: at + 1, ArrowLeft: at - 1, ArrowUp: at - 1, Home: 0, End: chips.length - 1 }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    const next = chips[(target + chips.length) % chips.length];
    next.focus();
    next.click();
});

// ---- What a row offers ------------------------------------------------------------------------
// The colorings a structure can be given on its own: pLDDT, its own color, and the layers it has.
function structureColoringOptions(structure) {
    const listing = resolvedListings.get(structureFile(structure));
    const own = (listing?.layers ?? []).filter(entry => entry.id !== PLDDT_LAYER && entry.annotation);
    return [[PLDDT_LAYER, 'pLDDT'], [STRUCTURE_COLORING, coloringLabel(STRUCTURE_COLORING)],
            ...own.map(entry => [entry.id, entry.label])];
}

window.addEventListener('esmfold-layers', () => scheduleStructures());
