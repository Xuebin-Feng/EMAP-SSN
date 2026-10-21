/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's Session menu and its question dialog.
 */

// The page saves its session to the Viewer after every change and takes it
// back when it opens (esmfold_shell.js). The menu shows that, saves on
// demand, hands the session over as a file (Mol*'s own .molj, which
// Mol*'s panels in Advanced read too), takes one back from a file, and
// clears the scene. Opening and clearing replace the scene, so each offers
// Undo for a few seconds, as removing a structure does. Loaded after
// esmfold_shell.js and esmfold_structures.js.

const sessionMenu = byId('session-menu');
const sessionButton = byId('session-btn');
let lastSaved = null;
let lastSaveFailed = false;

// ---- What the menu says about saving ------------------------------------------
function clockTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function updateSessionNote() {
    const connected = connectionState === 'connected';
    const header = byId('saved-note');
    const note = byId('menu-saved');
    header.textContent = '';
    if (!connected) {
        note.textContent = t("Not saved: no connection to the Viewer.");
    } else if (lastSaveFailed) {
        header.textContent = t("Not saved");
        note.textContent = t("The last save failed. Try Save now.");
    } else {
        if (lastSaved) header.textContent = t("Saved {time}", {time: clockTime(lastSaved)});
        note.textContent = t("Saved automatically after every change");
    }
    byId('mi-save').disabled = !connected;
}

window.addEventListener('esmfold-connection', updateSessionNote);
window.addEventListener('esmfold-saved', event => {
    lastSaveFailed = !event.detail.ok;
    if (event.detail.ok) lastSaved = new Date();
    updateSessionNote();
});
updateSessionNote();

// ---- The menu ----------------------------------------------------------------------
function menuItems() {
    return [...sessionMenu.querySelectorAll('.menu-item:not([disabled])')];
}

function setMenu(open) {
    if (open) {
        byId('mi-clear').disabled = !(plugin && loadedStructures().length);
        updateSessionNote();
    }
    sessionMenu.classList.toggle('open', open);
    sessionButton.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) menuItems()[0]?.focus();
}

// Closes the menu if it is open, and says whether it was; Escape then gives the focus back to its button.
function closeSessionMenu() {
    if (!sessionMenu.classList.contains('open')) return false;
    setMenu(false);
    sessionButton.focus();
    return true;
}

sessionButton.addEventListener('click', () => setMenu(!sessionMenu.classList.contains('open')));
document.addEventListener('click', event => {
    if (!event.target.closest('.menu-wrap') && sessionMenu.classList.contains('open')) setMenu(false);
});
sessionMenu.addEventListener('keydown', event => {
    const items = menuItems();
    const at = items.indexOf(document.activeElement);
    const moves = { ArrowDown: (at + 1) % items.length, ArrowUp: (at - 1 + items.length) % items.length, Home: 0, End: items.length - 1 };
    if (event.key in moves) {
        event.preventDefault();
        items[moves[event.key]].focus();
    } else if (event.key === 'Tab') {
        setMenu(false);
    }
});

// ---- The question dialog ---------------------------------------------------------
let confirmOpener = null;
let confirmAction = null;

function askToConfirm(title, message, confirmLabel, action) {
    confirmOpener = document.activeElement;
    confirmAction = action;
    byId('confirm-title').textContent = title;
    byId('confirm-text').textContent = message;
    byId('confirm-ok').textContent = confirmLabel;
    byId('confirm-modal').classList.add('open');
    // The safe answer has the focus.
    byId('confirm-cancel').focus();
}

// Closes the dialog if it is open, and says whether it was.
function closeConfirm() {
    const modal = byId('confirm-modal');
    if (!modal.classList.contains('open')) return false;
    modal.classList.remove('open');
    confirmAction = null;
    if (confirmOpener && confirmOpener.isConnected) confirmOpener.focus();
    return true;
}

byId('confirm-cancel').addEventListener('click', closeConfirm);
byId('confirm-modal').addEventListener('click', event => {
    if (event.target.id === 'confirm-modal') closeConfirm();
});
byId('confirm-ok').addEventListener('click', () => {
    const action = confirmAction;
    closeConfirm();
    if (action) action();
});

// ---- What the items do --------------------------------------------------------------
function saveNow() {
    saveSession().then(ok => {
        if (ok) showStatus(t("Session saved."));
        else showStatus(t("The session could not be saved."), 6000, {error: true});
    });
}

// A file for the browser to keep: Mol*'s own session format, which holds every structure, pose and color.
async function downloadSession() {
    const blob = await plugin.managers.snapshot.serialize({ params: { camera: !!plugin.canvas3d } });
    const link = make('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'fold-view-session.molj';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
}

// Mol* reads a .molj file and, as the page saves it, a bare session; anything else it only logs.
function looksLikeSession(text) {
    try {
        const parsed = JSON.parse(text);
        return !!parsed && typeof parsed === 'object' && (Array.isArray(parsed.entries) || !!parsed.data?.tree);
    } catch (error) {
        return false;
    }
}

async function openSessionFile(file) {
    if (!/\.molx$/i.test(file.name) && !looksLikeSession(await file.text())) {
        showStatus(t("{file} is not a Mol* session file.", {file: file.name}), 6000, {error: true});
        return;
    }
    const before = snapshotForUndo();
    runExclusive(async () => {
        await plugin.managers.snapshot.open(file);
        await reconcileLayers();
        syncCanvasProps();
        deriveColorMode();
        resetPointerState();
        scheduleStructures();
        setTimeout(saveSession, 1500);
    }).catch(err => reportFailure("Could not open the session:", err));
    showStatus(t("Opened {file}.", {file: file.name}), 6000, before ? {label: t("Undo"), run: () => undoSceneChange(before)} : {});
}

function clearScene() {
    const before = snapshotForUndo();
    runExclusive(async () => {
        const state = plugin.state.data;
        const update = state.build();
        for (const ref of state.tree.children.get(state.tree.root.ref)?.toArray() ?? []) update.delete(ref);
        await plugin.runTask(state.updateTree(update));
        fitStats.clear();
        expandedRows.clear();
        chosenPivotRef = null;
        renderColorCard();
        scheduleStructures();
        setTimeout(saveSession, 1500);
    }).catch(err => reportFailure("Could not clear the scene:", err));
    showStatus(t("The scene was cleared."), 6000, before ? {label: t("Undo"), run: () => undoSceneChange(before)} : {});
}

function chooseItem(action) {
    closeSessionMenu();
    if (action === 'save') saveNow();
    else if (action === 'download') downloadSession().catch(err => reportFailure("Could not download the session:", err));
    else if (action === 'open') byId('open-session-input').click();
    else if (action === 'open-structure') byId('open-structure-input').click();
    else if (action === 'clear') {
        askToConfirm(t("Clear the scene?"),
            t("This removes every structure from the view and from the saved session. You can undo it for a few seconds."),
            t("Clear scene"), clearScene);
    }
}

for (const item of sessionMenu.querySelectorAll('.menu-item')) {
    item.addEventListener('click', () => chooseItem(item.dataset.action));
}
byId('open-session-input').addEventListener('change', event => {
    const [file] = event.target.files;
    event.target.value = '';
    if (file) openSessionFile(file);
});
