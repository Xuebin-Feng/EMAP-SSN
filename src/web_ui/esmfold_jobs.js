/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's states while structures are being folded: progress, a failure, and a Viewer with no window.
 */

// The Viewer's folding worker tells the page how it is doing (esmfold_job
// events: started, progress, structure_failed, failed, finished), and the
// Viewer says on connect whether it has a window (viewer_info). Neither is
// needed for the page to work: an older Viewer sends neither, and the page
// then shows what it always showed. While a run is going the Structures card
// has a row for the sequence being folded and, with no structure yet, the
// canvas says so with a progress bar; a structure that failed to fold, or to
// load, has a row of its own with the reason, until it is dismissed.
// Loaded after esmfold_image.js.

let job = null;
let failures = [];
let failureCount = 0;
let jobTimer = null;

function jobRunning() {
    return !!job && !job.finished;
}

function clockSpan(milliseconds) {
    const seconds = Math.max(0, Math.floor(milliseconds / 1000));
    return String(Math.floor(seconds / 60)).padStart(2, '0') + ':' + String(seconds % 60).padStart(2, '0');
}

// "Folding 2 of 3", or the step before the first.
function jobHeading() {
    if (!job || !job.index) return t("Getting the model ready...");
    return t("Folding {index} of {total}", {index: job.index, total: job.total});
}

// ---- What the Viewer says -----------------------------------------------------------------------------
function addFailure(name, reason, log) {
    failures.push({ id: ++failureCount, name, reason, log: log || '' });
    scheduleStructures();
}

function handleJobEvent(message) {
    if (message.event === 'started') {
        job = { total: message.total || 0, index: 0, name: '', startedAt: Date.now(), finished: false };
        clearInterval(jobTimer);
        jobTimer = setInterval(tickJob, 1000);
    } else if (message.event === 'progress') {
        if (!jobRunning()) job = { total: message.total || 0, index: 0, name: '', startedAt: Date.now(), finished: false };
        job.index = message.index || job.index;
        job.total = message.total || job.total;
        job.name = message.name || '';
        if (!jobTimer) jobTimer = setInterval(tickJob, 1000);
    } else if (message.event === 'structure_failed') {
        addFailure(message.name || t("A structure"), message.reason || t("The folding worker stopped."), message.log);
        showStatus(t("Folding {name} failed: {reason}", {name: message.name || '', reason: message.reason || ''}), 8000, {error: true});
    } else if (message.event === 'failed') {
        addFailure(job?.name || t("Folding"), message.reason || t("The folding worker stopped."), message.log);
        if (job) job.finished = true;
        showStatus(t("The folding worker stopped: {reason}", {reason: message.reason || ''}), 8000, {error: true});
    } else if (message.event === 'finished') {
        if (job) job.finished = true;
        if (!message.failed) showStatus(t("Folded {done} of {total}.", {done: message.succeeded ?? 0, total: message.total ?? 0}));
    }
    if (job?.finished) {
        clearInterval(jobTimer);
        jobTimer = null;
    }
    renderJob();
    scheduleStructures();
}

function handleViewerInfo(message) {
    document.body.dataset.hidden = message.hidden ? '1' : '0';
    byId('hidden-badge').hidden = !message.hidden;
}

// ---- Drawing it -----------------------------------------------------------------------------------------------
function tickJob() {
    if (!jobRunning()) return;
    for (const span of document.querySelectorAll('.job-elapsed')) span.textContent = clockSpan(Date.now() - job.startedAt);
}

// The canvas's own state (no structure yet): the heading, the sequence, the bar.
function renderJob() {
    updateView();
    if (!job) return;
    byId('folding-title').textContent = jobHeading();
    byId('folding-name').textContent = job.name;
    const bar = byId('folding-bar');
    const done = job.index ? job.index - 1 : 0;
    bar.setAttribute('aria-valuemax', String(job.total));
    bar.setAttribute('aria-valuenow', String(done));
    bar.firstElementChild.style.width = job.total ? (100 * done / job.total) + '%' : '0%';
    byId('folding-elapsed').textContent = clockSpan(Date.now() - job.startedAt);
}

// The rows after the structures': the sequence being folded, and each failure.
function jobRows() {
    const rows = [];
    if (jobRunning()) {
        const row = make('div', 'srow pending');
        const main = make('div', 'srow-main');
        const swatch = make('span', 'srow-swatch');
        const text = make('div', 'srow-text');
        const spinner = icon("loader-circle", 'icon-spin');
        const label = make('span', undefined, job.name || t("Getting the model ready..."));
        const nameLine = make('div', 'srow-name');
        nameLine.append(spinner, label);
        const sub = make('div', 'srow-sub');
        const progress = make('span', undefined, job.index ? t("Folding {index} of {total}", {index: job.index, total: job.total}) : jobHeading());
        const elapsed = make('span', 'job-elapsed', clockSpan(Date.now() - job.startedAt));
        sub.append(progress, elapsed);
        text.append(nameLine, sub);
        main.append(swatch, text);
        row.append(main);
        rows.push(row);
    }
    for (const failure of failures) {
        const row = make('div', 'srow failed');
        row.setAttribute('role', 'group');
        row.setAttribute('aria-label', failure.name);
        const main = make('div', 'srow-main');
        const swatch = make('span', 'srow-swatch');
        const text = make('div', 'srow-text');
        const name = make('span', undefined, failure.name);
        const chipNode = chip(t("Failed"), 'err');
        const nameLine = make('div', 'srow-name');
        nameLine.append(name, chipNode);
        const reason = make('div', 'srow-sub', failure.reason);
        text.append(nameLine, reason);
        main.append(swatch, text);
        const body = make('div', 'srow-body');
        const actions = make('div', 'row-actions');
        if (failure.log) {
            const copy = make('button', 'btn btn-sm', t("Copy log path"));
            copy.type = 'button';
            copy.addEventListener('click', () => copyText(failure.log));
            actions.append(copy);
        }
        const dismiss = focusable(make('button', 'btn btn-sm', t("Dismiss")), 'failure:' + failure.id);
        dismiss.type = 'button';
        dismiss.addEventListener('click', () => {
            failures = failures.filter(item => item.id !== failure.id);
            scheduleStructures();
        });
        actions.append(dismiss);
        body.append(actions);
        row.append(main, body);
        rows.push(row);
    }
    return rows;
}

function copyText(text) {
    const done = () => showStatus(t("Copied the log path."));
    if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).then(done, () => showStatus(t("Could not copy: select the path in the Viewer's terminal."), 6000, {error: true}));
    } else {
        showStatus(t("Could not copy: select the path in the Viewer's terminal."), 6000, {error: true});
    }
}
