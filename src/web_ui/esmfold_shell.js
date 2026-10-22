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
}

byId('theme-toggle-btn').addEventListener('click', () => applyTheme(currentTheme() === 'dark' ? 'light' : 'dark'));
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
    return enqueueLoad(() => viewerInstance.loadStructureFromUrl(url, 'pdb'), label, url);
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
        const structures = plugin.managers.structure.hierarchy.current.structures;
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
            const file = decodeURIComponent(url.split('?')[0].split('/').pop());
            layerListing(file, true).then(() => renderColorCard());
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
    setTimeout(() => viewerInstance.handleResize(), 50);
    (on ? byId('adv-back') : byId('t-adv')).focus();
}
byId('t-adv').addEventListener('click', () => setAdvanced(document.body.dataset.adv !== '1'));
byId('adv-back').addEventListener('click', () => setAdvanced(false));

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

// Escape closes the layer on top: a menu, a question, the help, a tool, the pinned residue, then the Advanced view.
document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    if (closeSessionMenu() || closeConfirm() || closeImage()) return;
    if (byId('help-modal').classList.contains('open')) closeHelp();
    else if (leaveTool() || clearPin()) return;
    else if (document.body.dataset.adv === '1') setAdvanced(false);
});

// ---- Start ----------------------------------------------------------------------
applyTheme(currentTheme(), false);
setConnection('connecting');
updateView();
