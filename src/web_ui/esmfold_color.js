/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's Color by card: one choice of coloring for the whole scene, its legend, and a compact legend on the canvas.
 */

// The scene has one coloring at a time (pLDDT, a structure's own color, a
// layer of the Viewer's residue data, or one of Mol*'s own colorings: the
// properties it works out from a structure (secondary structure, solvent
// accessibility, hydrophobicity, charge) and the residue position), and each
// structure follows it; a
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
// The properties Mol* works out from the structure itself (themes of its own, in their default parameters).
const ASA_COLORING = 'builtin:accessible-surface-area';
const HYDROPHOBICITY_COLORING = 'builtin:hydrophobicity';
const CHARGE_COLORING = 'builtin:residue-charge';
// The Mol* color theme of each built-in that is one of Mol*'s own, and the other way round.
const BUILTIN_THEMES = {
    [SECONDARY_COLORING]: 'secondary-structure', [POSITION_COLORING]: 'sequence-id', [ASA_COLORING]: 'accessible-surface-area',
    [HYDROPHOBICITY_COLORING]: 'hydrophobicity', [CHARGE_COLORING]: 'residue-charge'
};
const BUILTIN_OF_THEME = Object.fromEntries(Object.entries(BUILTIN_THEMES).map(([id, name]) => [name, id]));
// What a structure shows when it has no values for the chosen layer: not a choice, a state.
const NO_DATA = 'nodata';
const OTHER_COLORING = 'other';
// The names of the three radio groups of colorings (the card's own chips, the Properties, and those under More colorings).
const MODEL_CHIPS = 'coloring';
const PROPERTY_CHIPS = 'coloring-properties';
const MORE_CHIPS = 'coloring-more';
// The most layers shown as buttons before the card adds a filter and a scroll area.
const MAX_PLAIN_LAYERS = 8;
let colorMode = PLDDT_LAYER;
let noDataColor = 0xdcdcdc;
let colorFilter = '';
// "More colorings" starts open. It is held here only (a person's choice is not stored, in the session or in the browser), and it opens by
// itself when the scene's coloring becomes one that is inside it, however that came about (a session restored, a fall-back, the page's own
// calls); closed by hand while such a coloring is drawn, it stays closed until the scene's coloring next changes to one inside it.
let moreColorings = true;
const MORE_COLORINGS = [STRUCTURE_COLORING, POSITION_COLORING];
let lastColorMode = colorMode;
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
        case 'secondary-structure': case 'sequence-id': case 'accessible-surface-area': case 'hydrophobicity': case 'residue-charge':
            return BUILTIN_OF_THEME[theme.name];
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
    if (BUILTIN_THEMES[id]) return { color: BUILTIN_THEMES[id] };
    return { color: 'uniform', colorParams: { value: noDataColor } };
}

// What a color is called, for a button, a select and a chip.
function coloringLabel(id, layers = []) {
    if (id === PLDDT_LAYER) return 'pLDDT';
    if (id === STRUCTURE_COLORING) return t("Structure color");
    if (id === SECONDARY_COLORING) return t("Secondary structure");
    if (id === POSITION_COLORING) return t("Residue position");
    if (id === ASA_COLORING) return t("Solvent accessibility");
    if (id === HYDROPHOBICITY_COLORING) return t("Hydrophobicity");
    if (id === CHARGE_COLORING) return t("Charge");
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
    if (id === ASA_COLORING) {
        // Mol*'s scale runs from the first color (buried: a relative area of 0) to the last (exposed: 1 or more).
        const colors = (themeLegend('accessible-surface-area')?.colors ?? []).map(intToHex);
        return { heading: coloringLabel(id), unit: '', kind: 'scale', colors, low: t("Buried"), high: t("Exposed"),
                 note: t("How much of each residue the solvent can reach, as a share of the most a residue of its kind can expose. Each structure is worked out on its own, with a water probe of 1.4 Å and only protein atoms in the way.") };
    }
    if (id === HYDROPHOBICITY_COLORING) {
        // Mol*'s scale runs from the most hydrophilic residue type (the first color, red) to the most hydrophobic (the last, green).
        const colors = (themeLegend('hydrophobicity')?.colors ?? []).map(intToHex);
        return { heading: coloringLabel(id), unit: '', kind: 'scale', colors, low: t("Hydrophilic"), high: t("Hydrophobic"),
                 note: t("One value for each kind of residue, from the Wimley and White scale for membrane interfaces (water to membrane), the same wherever the residue sits.") };
    }
    if (id === CHARGE_COLORING) {
        // The colors Mol* gives arginine and lysine, aspartate and glutamate, histidine and the rest, read from its own table.
        const table = new Map((themeLegend('residue-charge')?.table ?? []).map(([name, color]) => [name, intToHex(color)]));
        const bands = [['ARG', t("Positive")], ['ASP', t("Negative")], ['HIS', t("Histidine")], ['ALA', t("Neutral")]]
            .filter(([key]) => table.has(key)).map(([key, label]) => ({ color: table.get(key), label }));
        return { heading: coloringLabel(id), unit: '', kind: 'table', bands,
                 note: t("Arginine and lysine count as positive, aspartate and glutamate as negative; histidine has a color of its own and every other residue is neutral. It is the charge of the residue's name, not worked out for the structure.") };
    }
    const layer = model.layers.find(known => known.id === id);
    if (!layer) return null;
    return { heading: layer.label, unit: layer.unit, kind: layer.legend.kind, legend: layer.legend, description: layer.description,
             source: layer.source, partial: layer.partial, lack: layer.lack };
}

// The model of the card: all that it draws, as data, so that it is drawn again only when it changes.
async function colorModel() {
    const gathered = await gatherColoring();
    if (colorMode !== lastColorMode) {
        lastColorMode = colorMode;
        if (MORE_COLORINGS.includes(colorMode)) moreColorings = true;
    }
    const model = { mode: colorMode, filter: colorFilter, more: moreColorings, layers: gathered.layers,
                    unavailable: gathered.unavailable, plddt: gathered.plddt, count: gathered.structures.length };
    model.legend = gathered.structures.length ? legendOf(model) : null;
    model.noData = gathered.structures.filter(item => coloringOf(item.structure) === NO_DATA).map(item => item.name);
    // Opened files showing pLDDT: the colors are the file's B-factors.
    model.fromFiles = gathered.structures.filter(item => !isFolded(item.structure) && coloringOf(item.structure) === PLDDT_LAYER)
        .map(item => item.name);
    return model;
}

// A coloring's choice: a native radio in a label that looks like a chip (esmfold.html), so that the browser does the keys: the chips
// with one `name` are one Tab stop (the chosen one), and the arrow keys move along them and choose.
function colorChip(id, label, model, extra, name) {
    const chip = make('label', 'seg-btn');
    chip.title = label;
    chip.dataset.coloring = id;
    const input = make('input');
    input.type = 'radio';
    input.name = name;
    input.value = id;
    input.checked = model.mode === id;
    input.dataset.focusKey = 'color:' + id;
    const text = make('span', 'seg-label', label);
    // The same text, bold and unseen, keeps the chip as wide as it is when chosen (esmfold.html).
    text.dataset.text = label;
    chip.append(input, text);
    if (extra) {
        const badge = make('span', 'cov', extra);
        chip.append(badge);
    }
    input.addEventListener('change', () => {
        if (input.checked && colorMode !== id) setColorMode(id);
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
    // pLDDT's band limits are named between its ends (50, 70 and 90), at the places where the bands meet.
    const marks = confidence && ordered.length === 4 ? [[0.5, '50'], [0.7, '70'], [0.9, '90']] : [];
    return { stops, low: confidence ? '0' : '', high: confidence ? '100' : '', marks };
}

// A bar of the legend's colors with its two ends named.
function legendBar(legend, compact) {
    const wrap = make('div', compact ? 'legend-compact' : 'legend-body');
    const bar = make('div', 'ramp');
    const ramp = legendRamp(legend);
    bar.style.background = 'linear-gradient(to right, ' + ramp.stops.map(([at, color]) => `${color} ${(100 * at).toFixed(1)}%`).join(', ') + ')';
    // The bar is a picture: what it says is the text of its ends and its marks, which a screen reader reads in order.
    bar.setAttribute('aria-hidden', 'true');
    const ends = make('div', 'ticks');
    const lowEnd = make('span', undefined, ramp.low);
    const highEnd = make('span', undefined, ramp.high);
    // A label cut with an ellipsis (esmfold.html) has all of its text on hover.
    lowEnd.title = ramp.low;
    highEnd.title = ramp.high;
    // The marks are above the bar, each centred on its place, and take no room in the row of the ends.
    const marks = (ramp.marks ?? []).map(([at, text]) => {
        const mark = make('span', 'tick-mark', text);
        mark.style.left = (100 * at).toFixed(1) + '%';
        return mark;
    });
    ends.append(lowEnd, ...marks, highEnd);
    wrap.classList.toggle('has-marks', marks.length > 0);
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

// What the card says of the chosen coloring besides what the canvas's legend draws (its colors, in compactLegend): the texts. pLDDT's
// description and the caution for opened files, a layer's source (which opens to its description) and the warning of the structures
// that have no values for it, and the note of Structure color, which the canvas has no legend for. The texts read well under the chips
// with no name of their own; a coloring with none (Secondary structure, Residue position) has no box at all.
function cardLegend(model) {
    const legend = model.legend;
    const parts = [];
    if (legend.note) {
        const note = make('div', 'legend-note', legend.note);
        parts.push(note);
    }
    if (colorMode === PLDDT_LAYER && model.fromFiles.length) {
        const caution = make('div', 'caution', t("{names}: the colors are the file's B-factor column. In an experimental structure that is not confidence.", {names: model.fromFiles.join(', ')}));
        parts.push(caution);
    }
    if (model.noData.length && colorMode !== PLDDT_LAYER && !colorMode.startsWith(BUILTIN_PREFIX)) {
        const warning = make('div', 'nodata');
        const swatch = make('i');
        const words = make('span', undefined, t("No data: {names}. This layer has no values for them.", {names: model.noData.join(', ')}));
        warning.append(swatch, words);
        parts.push(warning);
    }
    const text = legend.source || legend.description;
    if (text && legend.source && legend.description) {
        // Where the numbers came from, which opens to what they mean.
        const details = make('details', 'src');
        const summary = make('summary');
        const info = icon("info");
        const line = make('span', undefined, legend.source);
        const chevron = icon("chevron-down", 'chev');
        summary.append(info, line, chevron);
        const more = make('p', undefined, legend.description);
        details.append(summary, more);
        parts.push(details);
    } else if (text) {
        parts.push(legendNote(text));
    }
    if (!parts.length) return null;
    const box = make('div', 'legend');
    box.id = 'color-legend';
    box.append(...parts);
    return box;
}

// The one text a legend has to say besides its colors (pLDDT's description, or a layer's that has no source, or no description): two lines of
// it, and a chevron to the whole of it when it is longer. A text that fits has no chevron and nothing to press, since a control that opens
// nothing is a control that does not work. The room a text has changes with the panel's width (it is dragged, the window is resized, a
// hidden panel or tab is shown), so whether it fits is worked out again whenever the note's width changes; a watcher of no height
// does that, since the note's own height changes when it is worked out.
let openedNote = null;

function legendNote(text) {
    const box = make('div', 'src');
    const info = icon("info");
    const line = make('span', 'note-text', text);
    const chevron = icon("chevron-down", 'chev');
    // A plain row while the text fits; a <button> when there is more to show, so that it is a control only then.
    const rowAs = tag => {
        const made = make(tag, 'note-row clamp');
        if (tag === 'button') made.type = 'button';
        made.append(info, line, chevron);
        return made;
    };
    let row = rowAs('div');
    const watch = make('span', 'note-watch');
    watch.setAttribute('aria-hidden', 'true');
    box.append(row, watch);
    const fit = () => {
        if (!row.isConnected) return;
        // Worked out in two lines with all the room, as a note with no chevron (a chevron takes room from the text).
        row.classList.add('clamp', 'measuring');
        const longer = line.scrollHeight > line.clientHeight + 1;
        const opened = longer && openedNote === text;
        row.classList.remove('measuring');
        row.classList.toggle('clamp', !opened);
        const tag = longer ? 'BUTTON' : 'DIV';
        if (row.tagName !== tag) {
            const next = rowAs(tag.toLowerCase());
            next.classList.toggle('clamp', !opened);
            row.replaceWith(next);
            row = next;
        }
        if (longer) row.setAttribute('aria-expanded', opened ? 'true' : 'false');
        else row.removeAttribute('aria-expanded');
    };
    box.addEventListener('click', event => {
        if (!event.target.closest('button.note-row')) return;
        openedNote = openedNote === text ? null : text;
        fit();
    });
    const watcher = new ResizeObserver(() => {
        if (row.isConnected) fit();
        else watcher.disconnect();
    });
    watcher.observe(watch);
    // The text is drawn in a font that may arrive after the first drawing, and wraps differently in it.
    document.fonts?.ready.then(fit);
    return box;
}

// Whether a legend has the "No data" key: a layer that some structure lacks, or that a structure covers in part.
function legendHasNoData(model) {
    if (colorMode === PLDDT_LAYER || colorMode.startsWith(BUILTIN_PREFIX)) return false;
    return model.noData.length > 0 || !!model.legend?.partial;
}

// The legend over the canvas: the one place the colors of the coloring are drawn. A labelled group, named by its own title, so that a
// screen reader has the title, the ends and marks of the bar, and the keys.
function compactLegend(model) {
    const legend = model.legend;
    const box = byId('canvas-legend');
    if (!legend || legend.kind === 'note') {
        box.replaceChildren();
        box.dataset.shown = '0';
        box.removeAttribute('role');
        box.removeAttribute('aria-labelledby');
        return;
    }
    box.dataset.shown = '1';
    const title = legendTitle(legend, 'div');
    title.id = 'canvas-legend-title';
    box.setAttribute('role', 'group');
    box.setAttribute('aria-labelledby', title.id);
    const parts = [title];
    if (legend.kind === 'table') {
        const keys = make('div', 'bands compact-keys');
        keys.append(...legend.bands.map(band => legendKey(band.color, band.label)));
        parts.push(keys);
    } else {
        parts.push(legendBar(legend, true));
    }
    if (legendHasNoData(model)) {
        parts.push(legendKey(intToHex(noDataColor), t("No data")));
    }
    box.replaceChildren(...parts);
}

// The layer filter is one <input> for the page's life. A card drawn again leaves it where it is, never taken out and put back, which would
// lose its focus and its caret and end a composition in it (an IME's pinyin, before a word is chosen); its text is the person's, and
// the card never writes it. What is typed filters the layer buttons only, the rest of the card stays, and not while an IME is composing:
// the text then is not yet what is meant, and the filter follows when the composition ends.
let layerFilterInput = null;
// The model the card was last drawn from, which the filter draws the buttons again from.
let drawnColorModel = null;

function layerFilter() {
    if (!layerFilterInput) {
        const filter = make('input', 'filter');
        filter.type = 'search';
        filter.placeholder = t("Filter layers");
        filter.setAttribute('aria-label', t("Filter layers"));
        filter.addEventListener('input', event => {
            if (!event.isComposing) filterLayers();
        });
        filter.addEventListener('compositionend', filterLayers);
        layerFilterInput = filter;
    }
    return layerFilterInput;
}

function filterLayers() {
    if (colorFilter === layerFilterInput.value) return;
    colorFilter = layerFilterInput.value;
    if (!drawnColorModel) return;
    // The card as drawn, with the new filter: what renderColorCard would draw, so that it does not draw it all again.
    drawnColorModel = { ...drawnColorModel, filter: colorFilter };
    renderedColors = JSON.stringify(drawnColorModel);
    const body = byId('color-body');
    const choices = layerChoices(drawnColorModel);
    if (choices) body.querySelector('.layer-choices')?.replaceWith(choices);
}

// The layer buttons the filter lets through (with eight layers or fewer there is no filter: all of them), or what says none does.
function layerChoices(model) {
    const many = model.layers.length > MAX_PLAIN_LAYERS;
    const wanted = model.filter.trim().toLowerCase();
    const shown = model.layers.filter(layer => !many || layer.label.toLowerCase().includes(wanted));
    if (shown.length) {
        const layerChips = make('div', (many ? 'seg scroll' : 'seg') + ' layer-choices');
        layerChips.append(...shown.map(layer => colorChip(layer.id, layer.label, model,
            layer.lack.length ? t("{n} of {total}", {n: layer.have.length, total: model.count}) : '', MODEL_CHIPS)));
        return layerChips;
    }
    return many ? make('div', 'hint layer-choices', t("No layer matches.")) : null;
}

// Puts `before`, `keep` and `after` in `parent`, in that order, in place of what it held. A `keep` that is in `parent` already stays where
// it is, and what is around it changes.
function replaceAround(parent, keep, before, after) {
    if (keep.parentNode !== parent) {
        parent.replaceChildren(...before, keep, ...after);
        return;
    }
    for (const child of [...parent.childNodes]) {
        if (child !== keep) child.remove();
    }
    keep.before(...before);
    keep.after(...after);
}

function renderColorBody(model) {
    const body = byId('color-body');
    const focused = body.contains(document.activeElement) ? document.activeElement.dataset.focusKey : null;
    drawnColorModel = model;
    const nodes = [];
    const many = model.layers.length > MAX_PLAIN_LAYERS;
    const filter = many ? layerFilter() : null;
    // The group the filter is in, if the card has it already: it stays, with the filter in it.
    let group = filter && filter.parentNode?.parentNode === body ? filter.parentNode : null;
    if (!group) {
        group = make('div', 'seg-group');
        group.setAttribute('role', 'radiogroup');
        group.setAttribute('aria-label', t("Color by"));
    }

    const modelLabel = groupLabel(t("Model"));
    const modelChips = make('div', 'seg');
    modelChips.append(colorChip(PLDDT_LAYER, coloringLabel(PLDDT_LAYER), model, '', MODEL_CHIPS));
    // The properties Mol* works out from the structure: always shown, and a radio group of their own (one Tab stop, the arrow keys inside).
    const propertiesLabel = groupLabel(t("Properties"));
    const propertyChips = make('div', 'seg properties');
    propertyChips.setAttribute('role', 'radiogroup');
    propertyChips.setAttribute('aria-label', t("Properties"));
    propertyChips.append(...[SECONDARY_COLORING, ASA_COLORING, HYDROPHOBICITY_COLORING, CHARGE_COLORING]
        .map(id => colorChip(id, coloringLabel(id), model, '', PROPERTY_CHIPS)));
    const layersLabel = groupLabel(t("Layers"));
    const choices = layerChoices(model);
    const before = [modelLabel, modelChips, propertiesLabel, propertyChips, layersLabel];
    if (filter) replaceAround(group, filter, before, choices ? [choices] : []);
    else group.replaceChildren(...before, ...(choices ? [choices] : []));

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
        more.append(colorChip(STRUCTURE_COLORING, coloringLabel(STRUCTURE_COLORING), model, '', MORE_CHIPS),
                    colorChip(POSITION_COLORING, coloringLabel(POSITION_COLORING), model, '', MORE_CHIPS));
        nodes.push(more);
    }
    const legendBox = model.legend ? cardLegend(model) : null;
    if (legendBox) nodes.push(legendBox);
    replaceAround(body, group, [], nodes);
    // (The filter, which stayed, keeps the focus by itself.)
    if (focused && !body.contains(document.activeElement)) keyedControl(body, focused)?.focus();
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

// ---- What a row offers ------------------------------------------------------------------------
// The colorings a structure can be given on its own: pLDDT, its own color, and the layers it has.
function structureColoringOptions(structure) {
    const listing = resolvedListings.get(structureFile(structure));
    const own = (listing?.layers ?? []).filter(entry => entry.id !== PLDDT_LAYER && entry.annotation);
    return [[PLDDT_LAYER, 'pLDDT'], [STRUCTURE_COLORING, coloringLabel(STRUCTURE_COLORING)],
            ...own.map(entry => [entry.id, entry.label])];
}

window.addEventListener('esmfold-layers', () => scheduleStructures());
