/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's own interface around the Mol* canvas.
 *
 * Mol* draws its interface itself; this page hides it (the Viewer's options
 * below) and builds a header, a structure list and the rest of its controls
 * from Mol*'s managers, in the look of the Agent and Meta Data pages. Mol*'s
 * own panels stay one click away: the Advanced button brings them back around
 * the same canvas (plugin.layout.updateProps), over the same scene. The
 * controls read Mol*'s state and the layer functions of esmfold_layers.js;
 * they keep no copy of it. Loaded last: esmfold_superposition.js and
 * esmfold_layers.js are before it, and the page's texts are marked with t()
 * (page_text.js), which the Viewer's web server translates as it serves this file.
 */

// ---- The page's state ----------------------------------------------------
let plugin;
let viewerInstance;
let isRestoring = true;
let pendingStructures = [];
let everConnected = false;
// What the alignment of each structure to the first found, by the structure's ref.
const fitStats = new Map();
const STRUCTURE_COLORS = ['#5b5bd6', '#0f9d8f', '#c2750c', '#c2185b', '#2563eb', '#65a30d'];

function byId(id) {
    return document.getElementById(id);
}

function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

// An icon of the page's sprite (<symbol id="i-NAME">), drawn in the text color.
function icon(name, className = '') {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', ('icon ' + className).trim());
    svg.setAttribute('aria-hidden', 'true');
    svg.append(document.createElementNS('http://www.w3.org/2000/svg', 'use'));
    setIcon(svg, name);
    return svg;
}

function setIcon(svg, name) {
    svg.firstElementChild.setAttribute('href', '#i-' + name);
}

// A square button showing only an icon (a drawn icon(NAME)); `label` names it for the pointer and for screen readers.
function iconButton(label, drawing, onClick, className = '') {
    const button = make('button', ('btn btn-ghost btn-icon btn-sm ' + className).trim());
    button.type = 'button';
    button.title = label;
    button.setAttribute('aria-label', label);
    button.append(drawing);
    button.addEventListener('click', onClick);
    return button;
}

// ---- Messages: a toast over the canvas, and a polite live region -----------
let toastTimer = null;

function hideStatus() {
    byId('toast').classList.remove('show');
}

// `duration` 0 keeps the message until the next one. `options`: {error, label, run}, an action button.
function showStatus(text, duration = 3000, options = {}) {
    const toast = byId('toast');
    clearTimeout(toastTimer);
    toast.className = 'toast show' + (options.error ? ' error' : '');
    const symbol = icon("info");
    setIcon(symbol, options.error ? "triangle-alert" : "info");
    const message = make('span', undefined, text);
    toast.replaceChildren(symbol, message);
    if (options.run) {
        const action = make('button', 'toast-action', options.label);
        action.type = 'button';
        action.addEventListener('click', () => {
            hideStatus();
            options.run();
        });
        toast.append(action);
    }
    byId('live').textContent = text;
    if (duration > 0) toastTimer = setTimeout(hideStatus, duration);
}

// ---- Connection to the Viewer --------------------------------------------
function setConnection(state) {
    const badge = byId('connection-status');
    badge.dataset.state = state;
    if (state === 'connected') badge.textContent = t("Connected");
    else if (state === 'reconnecting') badge.textContent = t("Reconnecting...");
    else badge.textContent = t("Connecting...");
    document.body.dataset.offline = state === 'reconnecting' && everConnected ? '1' : '0';
}

// ---- Theme: the choice the other pages keep (ssn_theme), else the system's ------
function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

// The canvas color of the theme, as '#rrggbb'.
function canvasBackground() {
    return getComputedStyle(document.documentElement).getPropertyValue('--canvas-bg').trim() || '#fafafa';
}

function syncBackground() {
    if (!plugin || !plugin.canvas3d) return;
    plugin.canvas3d.setProps({renderer: {backgroundColor: parseInt(canvasBackground().slice(1), 16)}});
}

// Mol*'s own panels (Advanced) come in its light or its dark skin, whichever the theme asks for.
function applySkin(theme) {
    byId('skin-light').disabled = theme === 'dark';
    byId('skin-dark').disabled = theme !== 'dark';
}

function applyTheme(theme, remember = true) {
    document.documentElement.setAttribute('data-theme', theme);
    if (remember) {
        try { localStorage.setItem('ssn_theme', theme); } catch (error) {}
    }
    applySkin(theme);
    const dark = theme === 'dark';
    const themeIcon = byId('theme-toggle-icon');
    byId('theme-toggle-label').textContent = dark ? t("Light Mode") : t("Dark Mode");
    setIcon(themeIcon, dark ? "sun" : "moon");
    syncBackground();
}

byId('theme-toggle-btn').addEventListener('click', () => applyTheme(currentTheme() === 'dark' ? 'light' : 'dark'));
window.addEventListener('storage', event => {
    if (event.key === 'ssn_theme' && (event.newValue === 'dark' || event.newValue === 'light')) applyTheme(event.newValue, false);
});

// ---- What the canvas area shows: starting, no WebGL, restoring, empty or the structures ----
function updateView() {
    let view = 'ready';
    const count = plugin ? loadedStructures().length : 0;
    if (!plugin) view = 'starting';
    else if (!plugin.canvas3d) view = 'webgl';
    else if (isRestoring) view = 'restoring';
    else if (count === 0) view = 'empty';
    document.body.dataset.view = view;
    document.body.dataset.count = String(count);
}

// ---- Mol*: its interface hidden, only the canvas. Without layoutShowRemoteState: false,
// Mol* lists the public states at webchem.ncbr.muni.cz on every load.
molstar.Viewer.create('viewer', {
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowRemoteState: false,
    layoutShowSequence: true,
    layoutShowLog: true,
    layoutShowLeftPanel: true,
    viewportShowControls: false,
    viewportShowExpand: false,
    viewportShowToggleFullscreen: false,
    viewportShowSettings: false,
    viewportShowSelectionMode: false,
    viewportShowAnimation: false,
    viewportShowReset: false,
    viewportShowScreenshotControls: false,
    viewportBackgroundColor: canvasBackground()
}).then(v => {
    viewerInstance = v;
    plugin = v.plugin;
    // The hierarchy changes as structures come and go; a structure's visibility is only a cell's state.
    plugin.managers.structure.hierarchy.behaviors.selection.subscribe(() => scheduleStructures());
    plugin.state.data.events.cell.stateUpdated.subscribe(() => scheduleStructures());
    syncBackground();
    updateView();
    renderStructures();
    showStatus(t("Mol* initialized. Connecting to the Viewer..."));
    setupSSE();
}).catch(error => {
    console.error("Mol* could not start:", error);
    document.body.dataset.view = 'webgl';
    showStatus(t("Mol* could not start: {reason}", {reason: error.message}), 0, {error: true});
});

// ---- The Viewer's side: events, and the session it keeps ----------------------
// Request session load from python server
function requestSessionLoad() {
    fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'load_molstar_session' })
    }).catch(err => {
        console.error("Failed to load session:", err);
        showStatus(t("Connection error: could not fetch the session."));
    });
}

function finishRestoring() {
    isRestoring = false;
    updateView();
}

// Setup SSE connection to listen for events
function setupSSE() {
    const sse = new EventSource('/api/events?client=esmfold');

    // Backup safety timeout for session restore handshake
    setTimeout(() => {
        if (isRestoring) {
            console.warn("Session restore handshake timed out; proceeding with fresh session.");
            finishRestoring();
            processPendingStructures();
        }
    }, 4000);

    async function processPendingStructures() {
        for (const struct of pendingStructures) {
            showStatus(t("Processing queued structure: {name}...", {name: struct.node_id}), 0);
            await enqueueStructure(struct.pdb_url, struct.node_id);
        }
        pendingStructures = [];
    }

    sse.onmessage = async (event) => {
        try {
            const msg = JSON.parse(event.data);

            if (msg.type === 'restore_session') {
                if (isRestoring) {
                    if (msg.session) {
                        showStatus(t("Restoring the saved session..."));
                        await plugin.state.setSnapshot(msg.session);
                        await reconcileLayers();
                        showStatus(t("Session restored successfully."));
                    } else {
                        showStatus(t("Started with a fresh blank session."));
                    }
                    finishRestoring();
                    await processPendingStructures();
                }
            } else if (msg.type === 'esmfold_pdb') {
                const { pdb_url, node_id } = msg;
                if (isRestoring) {
                    // Queue incoming structure while waiting for the session state response
                    pendingStructures.push({ pdb_url, node_id });
                } else {
                    showStatus(t("New structure received: {name}. Loading...", {name: node_id}), 0);
                    await enqueueStructure(pdb_url, node_id);
                }
            }
        } catch (e) {
            console.error("Error processing SSE message:", e);
        }
    };

    sse.onopen = () => {
        everConnected = true;
        setConnection('connected');
        requestSessionLoad();
    };

    sse.onerror = (err) => {
        console.error("SSE Connection Error:", err);
        setConnection('reconnecting');
    };
}

// Structures load one at a time, so each load aligns the structure it
// added (the newest) even when fold events arrive back to back.
let structureQueue = Promise.resolve();
function enqueueStructure(url, label) {
    structureQueue = structureQueue
        .then(() => loadAndAlignStructure(url, label))
        .catch(err => console.error("Error loading structure:", err));
    return structureQueue;
}

// Load structure, apply cartoon/pLDDT coloring, align to pivot, and auto-save
async function loadAndAlignStructure(url, label) {
    try {
        showStatus(t("Loading structure {name}...", {name: label}), 0);
        plugin.log.info(`Loading structure ${label}...`);

        // Use the official high-level loadStructureFromUrl method
        await viewerInstance.loadStructureFromUrl(url, 'pdb');

        // Apply pLDDT confidence coloring to all loaded structures,
        // except those the user has coloured by a per-residue layer
        const structures = plugin.managers.structure.hierarchy.current.structures;
        for (const s of structures) {
            if (activeLayerId(s) !== PLDDT_LAYER) continue;
            const components = s.components;
            try {
                await plugin.managers.structure.component.updateRepresentationsTheme(components, {
                    color: 'plddt-confidence'
                });
            } catch (e1) {
                try {
                    await plugin.managers.structure.component.updateRepresentationsTheme(components, {
                        colorTheme: { name: 'plddt-confidence' }
                    });
                } catch (e2) {
                    console.warn("Failed to apply pLDDT theme:", e1, e2);
                }
            }
        }
        if (structures.length > 1) {
            showStatus(t("Aligning {name} to the pivot structure...", {name: label}), 0);
            plugin.log.info(`Aligning ${label} to pivot structure...`);
            const pivot = structures[0];
            const target = structures[structures.length - 1];

            try {
                const fit = await superposeOnto(pivot, target);
                fitStats.set(target.cell.transform.ref, fit);
                showStatus(t("Loaded and aligned {name}.", {name: label}));
                plugin.log.info(`Successfully loaded and aligned ${label}: RMSD ${fit.rmsd.toFixed(2)} Å over ${fit.used} of ${fit.aligned} aligned CA pairs (${Math.round(100 * fit.identity)}% sequence identity).`);
            } catch (alignErr) {
                console.warn("Structural alignment skipped/failed: ", alignErr);
                plugin.log.warn(`Structural alignment skipped/failed: ${alignErr.message}`);
                showStatus(t("Loaded {name}.", {name: label}));
            }
        } else {
            showStatus(t("Loaded {name} as the pivot structure.", {name: label}));
            plugin.log.info(`Loaded ${label} as the pivot structure.`);
        }
    } catch (err) {
        console.error("Error loading structure:", err);
        plugin.log.error(`Error loading structure ${label}: ${err.message}`);
        showStatus(t("Failed to load structure {name}.", {name: label}), 6000, {error: true});
    } finally {
        // A structure written again under the same name has new residue data.
        const file = decodeURIComponent(url.split('?')[0].split('/').pop());
        layerListing(file, true).then(() => renderLayerPanel());
        scheduleStructures();
        // Wait 1.5 seconds for representation state to finalize, then save session snapshot
        setTimeout(saveSession, 1500);
    }
}

// What went wrong with a save goes to the Viewer's terminal log (developers read it; no one is shown it).
function reportDebug(detail) {
    fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'console_debug_err', detail })
    });
}

// Save session state to layout cache
function saveSession() {
    if (!plugin) return;
    try {
        const snapshot = plugin.state.getSnapshot();

        fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'save_molstar_session', session: snapshot })
        }).then(response => {
            if (response.ok) {
                console.log("Mol* session snapshot auto-saved.");
            } else {
                console.error("Failed to auto-save session state.");
                reportDebug("Failed to auto-save session state. Status: " + response.status);
            }
        }).catch(err => {
            console.error("Network error during auto-save:", err);
            reportDebug("Network error during auto-save: " + (err.stack || err.message));
        });
    } catch (err) {
        console.error("Serialization error during saveSession:", err);
        reportDebug("Serialization error during saveSession: " + (err.stack || err.message));
    }
}

// ---- The Structures card ---------------------------------------------------
let structuresTimer = null;
const residueCounts = new Map();

function scheduleStructures() {
    clearTimeout(structuresTimer);
    structuresTimer = setTimeout(renderStructures, 30);
}

function structureName(structure) {
    const file = structureFile(structure);
    if (file) return file.replace(/\.pdb$/i, '');
    return (structure.cell.obj && structure.cell.obj.label) || t("Structure");
}

function structureHidden(structure) {
    return structure.components.length > 0 && structure.components.every(component => component.cell.state.isHidden);
}

function residueCount(structure) {
    const ref = structure.cell.transform.ref;
    if (!residueCounts.has(ref)) residueCounts.set(ref, readCaTrace(structure.cell.obj.data).codes.length);
    return residueCounts.get(ref);
}

function structureByRef(ref) {
    return loadedStructures().find(structure => structure.cell.transform.ref === ref);
}

function chip(text, className, title) {
    const node = make('span', 'chip ' + className, text);
    if (title) node.title = title;
    return node;
}

function structureRow(structure, index) {
    const ref = structure.cell.transform.ref;
    const name = structureName(structure);
    const hidden = structureHidden(structure);
    const row = make('div', 'srow' + (hidden ? ' hidden-structure' : ''));
    row.dataset.ref = ref;
    const main = make('div', 'srow-main');
    const swatch = make('span', 'srow-swatch');
    swatch.style.setProperty('--sw', STRUCTURE_COLORS[index % STRUCTURE_COLORS.length]);
    const text = make('div', 'srow-text');
    const nameLine = make('div', 'srow-name', name);
    text.append(nameLine);
    const sub = make('div', 'srow-sub');
    const fit = fitStats.get(ref);
    let badge = null;
    if (index === 0) {
        badge = chip(t("Pivot"), 'pivot', t("The other structures are aligned to this one."));
    } else if (fit) {
        badge = chip(t("RMSD {rmsd} Å", {rmsd: fit.rmsd.toFixed(1)}), 'ok',
            t("Aligned to the pivot: {used} of {aligned} Cα pairs, {identity}% sequence identity.",
                {used: fit.used, aligned: fit.aligned, identity: Math.round(100 * fit.identity)}));
    } else if (structure.transform) {
        badge = chip(t("Aligned"), 'ok', t("Aligned to the pivot."));
    }
    const count = make('span', undefined, t("{n} residues", {n: residueCount(structure)}));
    sub.append(...(badge ? [badge] : []), count);
    text.append(sub);

    const eyeLabel = hidden ? t("Show {name}", {name}) : t("Hide {name}", {name});
    const eyeIcon = icon("eye");
    setIcon(eyeIcon, hidden ? "eye-off" : "eye");
    const eye = iconButton(eyeLabel, eyeIcon, () => {
        plugin.managers.structure.component.toggleVisibility(structure.components);
    });
    eye.setAttribute('aria-pressed', hidden ? 'true' : 'false');
    const zoom = iconButton(t("Zoom to {name}", {name}), icon("crosshair"), () => zoomToStructure(structure));
    const remove = iconButton(t("Remove {name}", {name}), icon("trash-2"), () => removeStructure(structure), 'btn-danger-ghost');
    main.append(swatch, text, eye, zoom, remove);
    row.append(main);
    return row;
}

function renderStructures() {
    if (!plugin) return;
    const structures = loadedStructures();
    const list = byId('struct-list');
    if (structures.length) list.replaceChildren(...structures.map(structureRow));
    else {
        const hint = make('div', 'hint', t("Structures appear here as they are folded."));
        list.replaceChildren(hint);
    }
    byId('struct-count').textContent = structures.length ? '(' + structures.length + ')' : '';
    updateView();
}

function zoomToStructure(structure) {
    const loci = molstar.lib.structure.Structure.toStructureElementLoci(structure.cell.obj.data);
    plugin.managers.camera.focusLoci(loci);
}

// Removing a structure can be undone for a few seconds: the session as it was comes back.
function removeStructure(structure) {
    const ref = structure.cell.transform.ref;
    const name = structureName(structure);
    let before = null;
    try {
        before = plugin.state.getSnapshot();
    } catch (error) {
        console.warn("Could not keep the scene for Undo:", error);
    }
    runExclusive(async () => {
        const target = structureByRef(ref);
        if (target) await plugin.managers.structure.hierarchy.remove([target]);
        fitStats.delete(ref);
        renderLayerPanel();
        scheduleStructures();
        setTimeout(saveSession, 1500);
    });
    const undo = before ? {label: t("Undo"), run: () => undoRemoval(before)} : {};
    showStatus(t("Removed {name}.", {name}), 6000, undo);
}

function undoRemoval(snapshot) {
    runExclusive(async () => {
        await plugin.state.setSnapshot(snapshot);
        await reconcileLayers();
        scheduleStructures();
        setTimeout(saveSession, 1500);
    });
}

// ---- The toolbar over the canvas ---------------------------------------------
byId('t-reset').addEventListener('click', () => {
    if (plugin) plugin.managers.camera.reset();
});

byId('t-full').addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(error => console.warn("Full screen was refused:", error));
});
document.addEventListener('fullscreenchange', () => {
    byId('t-full').setAttribute('aria-pressed', document.fullscreenElement ? 'true' : 'false');
});

// Advanced: Mol*'s own panels around the canvas, over the same scene.
function setAdvanced(on) {
    if (!plugin) return;
    document.body.dataset.adv = on ? '1' : '0';
    byId('t-adv').setAttribute('aria-pressed', on ? 'true' : 'false');
    plugin.layout.updateProps({showControls: on});
    setTimeout(() => viewerInstance.handleResize(), 50);
    (on ? byId('adv-back') : byId('t-adv')).focus();
}
byId('t-adv').addEventListener('click', () => setAdvanced(document.body.dataset.adv !== '1'));
byId('adv-back').addEventListener('click', () => setAdvanced(false));
byId('open-file').addEventListener('click', () => setAdvanced(true));

// ---- The panel: beside the canvas, a drawer on a narrow window, a strip on a phone ----
function setPanel(open) {
    document.body.dataset.panel = open ? '1' : '0';
    byId('panel-toggle').setAttribute('aria-expanded', open ? 'true' : 'false');
    byId('side-grip').setAttribute('aria-expanded', open ? 'true' : 'false');
    if (plugin) setTimeout(() => viewerInstance.handleResize(), 220);
}
byId('panel-toggle').addEventListener('click', () => setPanel(document.body.dataset.panel !== '1'));
byId('side-grip').addEventListener('click', () => setPanel(document.body.dataset.panel !== '1'));
(() => {
    const width = byId('app').clientWidth;
    setPanel(!(width <= 1040 && width > 700));
})();

// ---- Help ---------------------------------------------------------------------
let helpOpener = null;

function closeHelp() {
    byId('help-modal').classList.remove('open');
    if (helpOpener) helpOpener.focus();
}

byId('help-btn').addEventListener('click', () => {
    helpOpener = byId('help-btn');
    byId('help-modal').classList.add('open');
    byId('help-ok').focus();
});
byId('help-ok').addEventListener('click', closeHelp);
byId('help-modal').addEventListener('click', event => {
    if (event.target.id === 'help-modal') closeHelp();
});

document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    if (byId('help-modal').classList.contains('open')) closeHelp();
    else if (document.body.dataset.adv === '1') setAdvanced(false);
});

// ---- Start ----------------------------------------------------------------------
applyTheme(currentTheme(), false);
setConnection('connecting');
updateView();
