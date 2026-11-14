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
 * they keep no copy of it. esmfold_superposition.js and esmfold_layers.js are
 * loaded before this file, and esmfold_structures.js (the Structures card) and
 * esmfold_session.js (the Session menu) after it; the page's texts are marked
 * with t() (page_text.js), which the Viewer's web server translates as it serves
 * this file.
 */

// ---- The page's state ----------------------------------------------------
let plugin;
let viewerInstance;
let isRestoring = true;
let pendingStructures = [];
let everConnected = false;
// 'connecting', 'connected' or 'reconnecting': the Viewer's side of the page's stream.
let connectionState = 'connecting';
// Set when Mol* could not be started (failStart): the reason, as text. The page then has no 3D view to offer.
let startFailure = null;

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
// A message goes a while after it is shown (its duration), but not while the pointer is over it or the focus is in it (on its Undo, say):
// its time starts again once both have left it. A person reading it, or about to press its button, keeps it.
let toastTimer = null;
let toastDuration = 0;
let toastHovered = false;
let toastFocused = false;

function hideStatus() {
    clearTimeout(toastTimer);
    toastTimer = null;
    // Nothing holds a message that is not shown (the next one starts afresh).
    toastHovered = false;
    toastFocused = false;
    byId('toast').classList.remove('show');
}

function startToastTimer() {
    clearTimeout(toastTimer);
    toastTimer = null;
    if (toastDuration > 0 && !toastHovered && !toastFocused && byId('toast').classList.contains('show')) {
        toastTimer = setTimeout(hideStatus, toastDuration);
    }
}

// While a dialog is open the page under it is inert, the toast and the live region with it: a message would be under the dialog's backdrop
// and unheard. It waits for the dialog to close (the latest one only, as the toast shows only the latest), and is shown and said then.
let heldStatus = null;

// `duration` 0 keeps the message until the next one. `options`: {error, label, run, focus}, an action button: `run` does it (and may return
// a promise of its end), and `focus` names the control that takes the focus once it is done (see focusAfterAction).
function showStatus(text, duration = 3000, options = {}) {
    if (document.querySelector('dialog[open]')) {
        heldStatus = [text, duration, options];
        return;
    }
    // (A message held for a dialog is older than this one.)
    heldStatus = null;
    const toast = byId('toast');
    toast.className = 'toast show' + (options.error ? ' error' : '');
    const symbol = icon("info");
    setIcon(symbol, options.error ? "triangle-alert" : "info");
    const message = make('span', undefined, text);
    toast.replaceChildren(symbol, message);
    // (The focus was on the message before, if it was in the toast at all.)
    toastFocused = false;
    if (options.run) {
        // What had the focus as the message came: the control whose doing it reports (the Session button, for Clear scene).
        const origin = document.activeElement;
        const action = make('button', 'toast-action', options.label);
        action.type = 'button';
        action.addEventListener('click', () => {
            const focused = toast.contains(document.activeElement);
            hideStatus();
            const done = Promise.resolve(options.run());
            if (focused) done.then(() => focusAfterAction(options.focus, origin));
        });
        toast.append(action);
    }
    // A screen reader is told the message's text and that its button is there (it is in the message, which goes after a while).
    byId('live').textContent = options.run ? text + ' ' + t("{action} is available.", {action: options.label}) : text;
    toastDuration = duration;
    startToastTimer();
}

for (const dialog of document.querySelectorAll('dialog')) {
    dialog.addEventListener('close', () => {
        if (!heldStatus || document.querySelector('dialog[open]')) return;
        const [text, duration, options] = heldStatus;
        heldStatus = null;
        showStatus(text, duration, options);
    });
}

// A message's button goes with the message, and the focus it had would drop to the page. Once its action is done (an Undo, which brings a
// structure back), the focus goes to the control the action names (`choose`: the structure's own Remove, back in its row), else to what had
// it when the message came (`origin`), else to the Structures card's heading: unless the person has gone on to something else meanwhile.
function focusAfterAction(choose, origin) {
    const now = document.activeElement;
    if (now && now !== document.body && !byId('toast').contains(now)) return;
    for (const target of [choose?.(), origin, byId('h-structures')]) {
        if (!target || !target.isConnected || target === document.body || byId('toast').contains(target)) continue;
        target.focus();
        if (document.activeElement === target) return;
    }
}

byId('toast').addEventListener('pointerenter', () => {
    toastHovered = true;
    clearTimeout(toastTimer);
});
byId('toast').addEventListener('pointerleave', () => {
    toastHovered = false;
    startToastTimer();
});
byId('toast').addEventListener('focusin', () => {
    toastFocused = true;
    clearTimeout(toastTimer);
});
byId('toast').addEventListener('focusout', event => {
    if (byId('toast').contains(event.relatedTarget)) return;
    toastFocused = false;
    startToastTimer();
});

// ---- The focus, when a part of the page is drawn again ------------------------------------------------------
// A list or a card drawn again gives the focus back to the control that had it, found by the key the control carries (data-focus-key,
// focusable() in esmfold_structures.js). A control that went with what it acted on (a structure's Remove, a failure's Dismiss, the pin's
// Clear) has nothing to go back to: the focus goes to the same control of the row that took its place, else of the row before it, else to
// the card's heading, so that it stays where the person was and never drops to the page.
function focusPlace(container) {
    const node = document.activeElement;
    if (!node || node === document.body || !container.contains(node)) return null;
    const row = node.closest('.srow');
    return { key: node.dataset.focusKey || null, row: row ? [...container.querySelectorAll('.srow')].indexOf(row) : -1 };
}

function keyedControl(container, key) {
    return container.querySelector('[data-focus-key="' + key.replace(/["\\]/g, '\\$&') + '"]');
}

// `place` is what focusPlace(container) said before the container was drawn again; `heading` takes the focus when nothing else can.
function restoreFocus(container, place, heading) {
    if (!place || container.contains(document.activeElement)) return;
    let target = place.key ? keyedControl(container, place.key) : null;
    // The same control of another row is the one whose key ends as this one's does (":remove", ":dismiss").
    const action = place.key && place.key.includes(':') ? place.key.slice(place.key.lastIndexOf(':')) : null;
    if (!target && action && place.row >= 0) {
        const rows = [...container.querySelectorAll('.srow')];
        for (const row of [rows[place.row], rows[place.row - 1]]) {
            target = row ? [...row.querySelectorAll('[data-focus-key]')].find(node => node.dataset.focusKey.endsWith(action)) : null;
            if (target) break;
        }
    }
    target?.focus();
    if (!target || document.activeElement !== target) heading?.focus();
}

// ---- Connection to the Viewer --------------------------------------------
function setConnection(state) {
    connectionState = state;
    const badge = byId('connection-status');
    // Without Mol* the page does not connect (failStart), and its header says what is wrong, not "Connecting...".
    badge.dataset.state = startFailure ? 'failed' : state;
    if (startFailure) badge.textContent = t("3D view unavailable");
    else if (state === 'connected') badge.textContent = t("Connected");
    else if (state === 'reconnecting') badge.textContent = t("Reconnecting...");
    else badge.textContent = t("Connecting...");
    document.body.dataset.offline = state === 'reconnecting' && everConnected ? '1' : '0';
    window.dispatchEvent(new CustomEvent('esmfold-connection', { detail: { state } }));
}

// ---- Theme: the choice the other pages keep (ssn_theme), else the system's ------
function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

// The canvas color of the theme, as '#rrggbb'.
function canvasBackground() {
    return getComputedStyle(document.documentElement).getPropertyValue('--canvas-bg').trim() || '#fafafa';
}

// What the page sets on the canvas whatever a session says (a saved one has its own background): the
// theme's background, and the axes gizmo in the bottom right corner, clear of the status line and the legend.
function syncCanvasProps() {
    if (!plugin || !plugin.canvas3d) return;
    const axes = plugin.canvas3d.props.camera.helper.axes;
    plugin.canvas3d.setProps({
        renderer: {backgroundColor: parseInt(canvasBackground().slice(1), 16)},
        camera: {helper: {axes: axes.name === 'on' ? {name: 'on', params: {...axes.params, location: 'bottom-right'}} : axes}}
    });
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
    syncCanvasProps();
    measureScrollbar();
}

byId('theme-toggle-btn').addEventListener('click', () => applyTheme(currentTheme() === 'dark' ? 'light' : 'dark'));
// While no theme has been chosen (on this page or another: ssn_theme) the page follows the system's, as it changes.
function followSystemTheme(dark) {
    let chosen = null;
    try { chosen = localStorage.getItem('ssn_theme'); } catch (error) {}
    if (chosen === 'light' || chosen === 'dark') return;
    applyTheme(dark ? 'dark' : 'light', false);
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => followSystemTheme(event.matches));
window.addEventListener('storage', event => {
    if (event.key === 'ssn_theme' && (event.newValue === 'dark' || event.newValue === 'light')) applyTheme(event.newValue, false);
});

// ---- What the canvas area shows: starting, Mol* not started, no WebGL, restoring, empty or the structures ----
function updateView() {
    let view = 'ready';
    const count = plugin && !startFailure ? loadedStructures().length : 0;
    if (startFailure) view = 'failed';
    else if (!plugin) view = 'starting';
    else if (!plugin.canvas3d) view = 'webgl';
    else if (isRestoring) view = 'restoring';
    else if (count === 0) view = typeof jobRunning === 'function' && jobRunning() ? 'folding' : 'empty';
    document.body.dataset.view = view;
    document.body.dataset.count = String(count);
}

// ---- Mol*: its interface hidden, only the canvas. Without layoutShowRemoteState: false,
// Mol* lists the public states at webchem.ncbr.muni.cz on every load.
// The page's scripts after this one define what the start below calls; it waits for them.
const scriptsLoaded = document.readyState === 'loading'
    ? new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve, { once: true }))
    : Promise.resolve();

// An async function, so that every way of not starting (the bundle is not there, it fails as it runs, its
// Viewer cannot be made, the setup below throws) is a rejection that failStart answers, not an error that stops here.
async function startMolstar() {
    // The bundle did not load (esmfold.html's head kept why): there is no Mol* to call.
    if (typeof molstar === 'undefined' || !molstar.Viewer) throw new Error(bundleProblem());
    const [v] = await Promise.all([molstar.Viewer.create('viewer', {
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
    }), scriptsLoaded]);
    viewerInstance = v;
    plugin = v.plugin;
    // The hierarchy changes as structures come and go; a structure's visibility is only a cell's state.
    plugin.managers.structure.hierarchy.behaviors.selection.subscribe(() => scheduleStructures());
    plugin.state.data.events.cell.stateUpdated.subscribe(() => scheduleStructures());
    // The list put off while Mol* replaced the scene (renderStructures): the update has ended, and Mol* has
    // rebuilt its hierarchy from the state just before it says so.
    plugin.state.data.behaviors.isUpdating.subscribe(updating => {
        if (updating || !renderRowsAfterUpdate) return;
        renderRowsAfterUpdate = false;
        scheduleStructures();
    });
    syncCanvasProps();
    updateView();
    renderStructures();
    setupResidueEvents();
    setupToolEvents();
    showStatus(t("Mol* initialized. Connecting to the Viewer..."));
    setupSSE();
}

// Why there is no `molstar`: what the page's head kept of the bundle's file, in words.
function bundleProblem() {
    const load = window.molstarLoad || {};
    if (load.message) return load.message;
    if (typeof molstar !== 'undefined') return t("Mol* loaded without its viewer.");
    return t("The file {file} did not load.", {file: '/esmfold/molstar.js'});
}

// Mol* could not be started, whatever the cause. The canvas area says so, with the reason to copy; the header
// says so; and what needs Mol* is not offered: the toolbar, the cards of the panel, the Session menu (esmfold.html
// hides them while the view is "failed"). The page does not connect to the Viewer either: it would be sent folded
// structures it cannot show, and the Viewer would think a Fold View is open; and it never saves, so the session the
// Viewer holds is still there when a reload starts Mol*. The theme and Help keep working.
function failStart(error) {
    console.error("Mol* could not start:", error);
    const reason = String((error && error.message) || error);
    startFailure = reason;
    byId('start-reason').textContent = t("Reason: {reason}", {reason});
    byId('session-btn').disabled = true;
    setConnection(connectionState);
    updateView();
    const heading = byId('start-title').textContent;
    byId('live').textContent = heading;
}
startMolstar().catch(failStart);
byId('start-retry').addEventListener('click', () => location.reload());

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
            await (struct.saved ? enqueueSavedStructure : enqueueStructure)(struct.pdb_url, struct.node_id);
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
                        syncCanvasProps();
                        deriveColorMode();
                        resetPointerState();
                        showStatus(t("Session restored successfully."));
                    } else {
                        showStatus(t("Started with a fresh blank session."));
                    }
                    finishRestoring();
                    await processPendingStructures();
                }
            } else if (msg.type === 'esmfold_job') {
                handleJobEvent(msg);
            } else if (msg.type === 'viewer_info') {
                handleViewerInfo(msg);
            } else if (msg.type === 'esmfold_pdb') {
                const { pdb_url, node_id, saved } = msg;
                if (isRestoring) {
                    // Queue incoming structure while waiting for the session state response
                    pendingStructures.push({ pdb_url, node_id, saved });
                } else {
                    showStatus(t("New structure received: {name}. Loading...", {name: node_id}), 0);
                    await (saved ? enqueueSavedStructure : enqueueStructure)(pdb_url, node_id);
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
    return enqueueLoad(() => viewerInstance.loadStructureFromUrl(url, 'pdb'), label, url);
}

// A structure the Viewer shows from a file it saved earlier (`saved` on the event), not one just folded: it is
// loaded unless the scene has it already, as a restored session or an earlier request may have put it there.
// The Viewer may have written the file's residue data again since, so its layers are asked for again.
function enqueueSavedStructure(url, label) {
    structureQueue = structureQueue
        .then(() => {
            const file = fileOfUrl(url);
            if (!loadedStructures().some(structure => structureFile(structure) === file)) {
                return loadAndAlignStructure(() => viewerInstance.loadStructureFromUrl(url, 'pdb'), label, url);
            }
            showStatus(t("{name} is already in the scene.", {name: label}));
            return layerListing(file, true).then(() => renderColorCard());
        })
        .catch(err => console.error("Error loading structure:", err));
    return structureQueue;
}

// `load` makes Mol* load a structure (from a URL, or from a file's text); `url` is where it came from, if anywhere.
function enqueueLoad(load, label, url) {
    structureQueue = structureQueue
        .then(() => loadAndAlignStructure(load, label, url))
        .catch(err => console.error("Error loading structure:", err));
    return structureQueue;
}

// Load structure, apply cartoon/pLDDT coloring, align to pivot, and auto-save
async function loadAndAlignStructure(load, label, url) {
    try {
        showStatus(t("Loading structure {name}...", {name: label}), 0);
        plugin.log.info(`Loading structure ${label}...`);

        const before = loadedStructures().length;
        await load();

        // Mol* logs a file it cannot get or read and carries on: no new structure is the failure.
        const structures = loadedStructures();
        if (structures.length <= before) throw new Error(t("Mol* could not get a structure from it."));
        // The new structure (the newest) takes the scene's coloring, pLDDT unless the
        // Color by card says otherwise; the structures already there keep what they show.
        await colorNewStructure(structures[structures.length - 1]);
        if (structures.length > 1) {
            showStatus(t("Aligning {name} to the pivot structure...", {name: label}), 0);
            plugin.log.info(`Aligning ${label} to pivot structure...`);
            const target = structures[structures.length - 1];
            const pivot = pivotStructure(target.cell.transform.ref);

            try {
                const fit = await superposeOnto(pivot, target);
                recordFit(target.cell.transform.ref, pivot.cell.transform.ref, fit);
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
        addFailure(label, err.message);
    } finally {
        // A structure written again under the same name has new residue data.
        if (url) {
            layerListing(fileOfUrl(url), true).then(() => renderColorCard());
        }
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
// The session as Mol* gives it; a camera only where there is a canvas to ask.
function takeSnapshot() {
    return withoutFocus(plugin.state.getSnapshot({ camera: !!plugin.canvas3d }));
}

// Resolves with whether the Viewer took it, and says so to the Session menu (esmfold-saved).
function saveSession() {
    const done = ok => {
        window.dispatchEvent(new CustomEvent('esmfold-saved', { detail: { ok } }));
        return ok;
    };
    if (!plugin) return Promise.resolve(false);
    try {
        const snapshot = takeSnapshot();

        return fetch('/api/action', {
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
            return done(response.ok);
        }).catch(err => {
            console.error("Network error during auto-save:", err);
            reportDebug("Network error during auto-save: " + (err.stack || err.message));
            return done(false);
        });
    } catch (err) {
        console.error("Serialization error during saveSession:", err);
        reportDebug("Serialization error during saveSession: " + (err.stack || err.message));
        return Promise.resolve(done(false));
    }
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
    showSideWidth();
    setTimeout(() => viewerInstance.handleResize(), 50);
    (on ? byId('adv-back') : byId('t-adv')).focus();
}
byId('t-adv').addEventListener('click', () => setAdvanced(document.body.dataset.adv !== '1'));
byId('adv-back').addEventListener('click', () => setAdvanced(false));

// ---- The panel: beside the canvas, a drawer on a narrow window, a strip on a phone ----
// The drawer (a window of 701 to 1040 px, the one layout that shows the Scene button, #panel-toggle) lies over the canvas. Closed, it is
// out of sight, and inert, so that Tab and a screen reader pass it by; open, it covers the toolbar over the canvas, which is inert while
// it does. Beside the canvas and in a phone's strip the panel is always in view, and nothing is inert. Which layout it is follows the
// page's width (esmfold.html's container queries), so this is worked out again whenever that changes.
function panelIsDrawer() {
    return getComputedStyle(byId('panel-toggle')).display !== 'none';
}

function syncPanelInert() {
    const drawer = panelIsDrawer();
    const open = document.body.dataset.panel === '1';
    byId('side').inert = drawer && !open;
    document.querySelector('.view-toolbar').inert = drawer && open;
}

function setPanel(open) {
    // A drawer that closes takes its inert with it, and the focus inside it would drop to the page: it goes to the Scene button.
    if (!open && panelIsDrawer() && byId('side').contains(document.activeElement)) byId('panel-toggle').focus();
    document.body.dataset.panel = open ? '1' : '0';
    byId('panel-toggle').setAttribute('aria-expanded', open ? 'true' : 'false');
    byId('side-grip').setAttribute('aria-expanded', open ? 'true' : 'false');
    syncPanelInert();
    if (plugin) setTimeout(() => viewerInstance.handleResize(), 220);
}
// An open drawer lies over its Scene button, so the focus goes in with it, to its Close.
byId('panel-toggle').addEventListener('click', () => {
    setPanel(document.body.dataset.panel !== '1');
    if (document.body.dataset.panel === '1') byId('side-close').focus({ preventScroll: true });
});
byId('side-grip').addEventListener('click', () => setPanel(document.body.dataset.panel !== '1'));
byId('side-close').addEventListener('click', () => {
    setPanel(false);
    byId('panel-toggle').focus();
});

// Closes the open drawer, and says whether it was one (the Scene button is where the focus goes back to).
function closeDrawer() {
    if (!panelIsDrawer() || document.body.dataset.panel !== '1') return false;
    setPanel(false);
    byId('panel-toggle').focus();
    return true;
}
(() => {
    const width = byId('app').clientWidth;
    setPanel(!(width <= 1040 && width > 700));
})();
new ResizeObserver(syncPanelInert).observe(byId('app'));

// The skip link goes to the panel. A closed drawer is inert, so the link opens it first. The panel takes the focus (Tab goes on from there
// to its first control) without scrolling to it: the drawer slides in over the canvas, and the workspace under it must not scroll.
document.querySelector('.skip-link').addEventListener('click', event => {
    event.preventDefault();
    if (panelIsDrawer() && document.body.dataset.panel !== '1') setPanel(true);
    const side = byId('side');
    side.tabIndex = -1;
    side.focus({ preventScroll: true });
});
// The panel is a place to land, not a stop of Tab's.
byId('side').addEventListener('blur', () => byId('side').removeAttribute('tabindex'));

// ---- The panel's width: beside the canvas a handle on its left edge makes it wider ----
// Beside the canvas (a window wider than 1040 px) the panel is 340 px wide, and can be made wider, up to half the window and
// 640 px: by dragging the handle on its left edge, or with Left and Right on the handle (20 px a press, 100 with Shift) and End
// (the widest); a double click or Home gives it its 340 px back. The canvas takes the room the panel leaves. esmfold.html holds the
// width in --side-width and clamps it to the window's (the same limits are worked out here for the handle's values), so a
// window that gets smaller takes the panel in with it, and the width comes back when the window does. The width is kept in this
// browser (localStorage, as the theme is) and is no part of the saved session. The drawer of a narrower window and the strip of a
// phone have a width of their own and no handle.
const SIDE_MIN = 340;
const SIDE_CAP = 640;
const SIDE_SHARE = 0.5;
const SIDE_STEP = 20;
const SIDE_BIG_STEP = 100;
const SIDE_STORE = 'ssn_fold_panel_width';
const sideResizer = byId('side-resizer');
let sideDrag = null;

// The narrowest and the widest the panel can be in the window as it is.
function sideLimits() {
    return { min: SIDE_MIN, max: Math.max(SIDE_MIN, Math.min(SIDE_CAP, Math.floor(byId('app').clientWidth * SIDE_SHARE))) };
}

function sideWidth() {
    return Math.round(byId('side').getBoundingClientRect().width);
}

// The handle says what the panel is and what it can be (what the panel is while it is shown: Advanced hides it).
function showSideWidth() {
    const { min, max } = sideLimits();
    sideResizer.setAttribute('aria-valuemin', String(min));
    sideResizer.setAttribute('aria-valuemax', String(max));
    const width = sideWidth();
    if (width > 0) sideResizer.setAttribute('aria-valuenow', String(width));
}

function rememberSideWidth(width) {
    try {
        if (width === null || width <= SIDE_MIN) localStorage.removeItem(SIDE_STORE);
        else localStorage.setItem(SIDE_STORE, String(width));
    } catch (error) {}
}

// Mol* follows its container's size on its own; this says so to it in step with a drag too.
let sideResizeFrame = 0;
function resizeCanvas() {
    if (sideResizeFrame) return;
    sideResizeFrame = requestAnimationFrame(() => {
        sideResizeFrame = 0;
        if (viewerInstance) viewerInstance.handleResize();
    });
}

// Makes the panel `width` pixels wide, or as near as the limits allow (the window's are applied by esmfold.html, so a width the
// window has no room for waits for one that has); `keep` keeps what is shown for the next visit.
function setSideWidth(width, keep = true) {
    byId('app').style.setProperty('--side-width', Math.round(Math.min(SIDE_CAP, Math.max(SIDE_MIN, width))) + 'px');
    showSideWidth();
    if (keep) rememberSideWidth(sideWidth());
    resizeCanvas();
}

function resetSideWidth() {
    byId('app').style.removeProperty('--side-width');
    showSideWidth();
    rememberSideWidth(null);
    resizeCanvas();
}

sideResizer.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    event.preventDefault();
    try {
        sideResizer.setPointerCapture(event.pointerId);
    } catch (error) {}
    // Where the pointer took hold of the edge, so that the edge stays under it.
    sideDrag = { id: event.pointerId, grab: event.clientX - byId('side').getBoundingClientRect().left };
    document.body.dataset.resizing = '1';
    sideResizer.focus();
});
sideResizer.addEventListener('pointermove', event => {
    if (!sideDrag || event.pointerId !== sideDrag.id) return;
    setSideWidth(byId('side').getBoundingClientRect().right - (event.clientX - sideDrag.grab), false);
});
function endSideDrag(event) {
    if (!sideDrag || event.pointerId !== sideDrag.id) return;
    sideDrag = null;
    delete document.body.dataset.resizing;
    rememberSideWidth(sideWidth());
    resizeCanvas();
}
sideResizer.addEventListener('pointerup', endSideDrag);
sideResizer.addEventListener('pointercancel', endSideDrag);
sideResizer.addEventListener('lostpointercapture', endSideDrag);
sideResizer.addEventListener('dblclick', resetSideWidth);
sideResizer.addEventListener('keydown', event => {
    const step = event.shiftKey ? SIDE_BIG_STEP : SIDE_STEP;
    // The panel is on the right: the edge moving left makes it wider.
    if (event.key === 'ArrowLeft') setSideWidth(sideWidth() + step);
    else if (event.key === 'ArrowRight') setSideWidth(sideWidth() - step);
    else if (event.key === 'Home') resetSideWidth();
    else if (event.key === 'End') setSideWidth(SIDE_CAP);
    else return;
    event.preventDefault();
});
new ResizeObserver(showSideWidth).observe(byId('app'));

// ---- The platform's scrollbar: its width is taken from the room round the panel's cards (--sb, esmfold.html) ----
// A classic bar is 15 px or more, an overlay one has none; scrollbar-gutter keeps the room, so it is measured whether or not the panel scrolls.
// (Not while the panel is not drawn, as in the phone's closed strip: the last width stays.)
function measureScrollbar() {
    const scroller = byId('side-scroll');
    if (scroller.offsetWidth === 0) return;
    byId('app').style.setProperty('--sb', (scroller.offsetWidth - scroller.clientWidth) + 'px');
}
new ResizeObserver(measureScrollbar).observe(byId('side-scroll'));
window.addEventListener('resize', measureScrollbar);
measureScrollbar();
(() => {
    let stored = null;
    try {
        stored = Number(localStorage.getItem(SIDE_STORE));
    } catch (error) {}
    // Only a width the handle could have made counts: a whole number from 341 px to the cap (a smaller window shows less of it).
    if (Number.isInteger(stored) && stored > SIDE_MIN && stored <= SIDE_CAP) setSideWidth(stored, false);
    else showSideWidth();
})();

// ---- The panel's tabs: on a phone the cards take turns in the strip; elsewhere all are in view ----
function setTab(name) {
    document.body.dataset.tab = name;
    for (const tab of document.querySelectorAll('.side-tabs [role="tab"]')) {
        tab.setAttribute('aria-selected', tab.dataset.tab === name ? 'true' : 'false');
        tab.tabIndex = tab.dataset.tab === name ? 0 : -1;
    }
}
for (const tab of document.querySelectorAll('.side-tabs [role="tab"]')) {
    tab.addEventListener('click', () => setTab(tab.dataset.tab));
    tab.addEventListener('keydown', event => {
        const tabs = [...document.querySelectorAll('.side-tabs [role="tab"]')];
        const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
        if (!step) return;
        event.preventDefault();
        const next = tabs[(tabs.indexOf(tab) + step + tabs.length) % tabs.length];
        setTab(next.dataset.tab);
        next.focus();
    });
}
setTab('structures');

// ---- Help ---------------------------------------------------------------------
// The page's dialogs (Help here, the question in esmfold_session.js, Save an image in esmfold_image.js) are <dialog>s opened with
// showModal(): the page under one is inert until it closes. Each puts the focus on its first control as it opens, and gives it back to
// what opened it as it closes (the browser does so too; a dialog says so itself, in case the browser's choice has gone from the page).
let helpOpener = null;

// Closes the help if it is open, and says whether it was.
function closeHelp() {
    const modal = byId('help-modal');
    if (!modal.open) return false;
    modal.close();
    if (helpOpener && helpOpener.isConnected) helpOpener.focus();
    return true;
}

byId('help-btn').addEventListener('click', () => {
    helpOpener = byId('help-btn');
    const modal = byId('help-modal');
    if (!modal.open) modal.showModal();
    byId('help-ok').focus();
});
byId('help-ok').addEventListener('click', closeHelp);
byId('help-modal').addEventListener('click', event => {
    if (event.target.id === 'help-modal') closeHelp();
});

// Escape closes the layer on top: a menu, a question, the help, the drawer (a narrow window's panel), a tool, the pinned residue, then the Advanced view. (An open dialog's
// Escape would be the browser's too, after this: this closes the dialog first, as its buttons do, and leaves the browser nothing to close.)
document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    if (closeSessionMenu() || closeConfirm() || closeImage() || closeHelp() || closeDrawer()) return;
    if (leaveTool() || clearPin()) return;
    if (document.body.dataset.adv === '1') setAdvanced(false);
});

// ---- Start ----------------------------------------------------------------------
applyTheme(currentTheme(), false);
setConnection('connecting');
updateView();
