/* Pending images belong to the composer until the server accepts a message. */
let pendingAttachments = [];
let pendingSubmission = null;
let composerEnabled = false;
const captureRequests = new Map();
const MAX_ATTACHMENTS = 10;
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

function attachmentNotice(message) {
    document.getElementById('attachment-notice').textContent = message;
}

function refreshComposer() {
    const processing = pendingAttachments.some(item => item.processing);
    const locked = !!pendingSubmission;
    document.getElementById('chat-input-field').disabled = !composerEnabled || locked;
    document.getElementById('chat-send-btn').disabled = !composerEnabled || processing || (locked && !pendingSubmission.transportFailed);
    document.getElementById('capture-viewer-btn').disabled = locked || pendingAttachments.length >= MAX_ATTACHMENTS;
    document.querySelectorAll('#attachment-tray button').forEach(button => { button.disabled = locked; });
}

function renderAttachments() {
    const tray = document.getElementById('attachment-tray');
    tray.replaceChildren();
    pendingAttachments.forEach(item => {
        const tile = document.createElement('div');
        tile.className = 'attachment-tile';
        if (item.data_url) {
            const img = document.createElement('img');
            img.src = item.data_url;
            img.alt = item.name;
            tile.appendChild(img);
        }
        const label = document.createElement('span');
        label.textContent = item.processing ? 'Processing…' : item.name;
        label.title = item.name;
        tile.appendChild(label);
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.textContent = '×';
        remove.setAttribute('aria-label', `Remove attachment: ${item.name}`);
        remove.title = 'Remove attachment';
        remove.onclick = () => {
            pendingAttachments = pendingAttachments.filter(candidate => candidate !== item);
            renderAttachments();
        };
        tile.appendChild(remove);
        tray.appendChild(tile);
    });
    refreshComposer();
}

function reserveAttachment(name) {
    if (pendingSubmission) return null;
    if (pendingAttachments.length >= MAX_ATTACHMENTS) {
        attachmentNotice('A message can contain at most 10 image attachments.');
        return null;
    }
    const item = {name, processing: true};
    pendingAttachments.push(item);
    renderAttachments();
    return item;
}

async function normalizeImage(file) {
    if (!file.size || file.size > MAX_IMAGE_BYTES) throw new Error('Each image must be nonempty and at most 20 MiB.');
    if (file.type && !file.type.startsWith('image/')) throw new Error('Only image files are accepted.');
    const response = await fetch('/api/agent/image', {method: 'POST', body: file, signal: AbortSignal.timeout(20000)});
    const info = await response.json();
    if (!response.ok) throw new Error(info.error || 'Image validation failed.');
    const url = URL.createObjectURL(new Blob([file], {type: info.mime_type}));
    try {
        const img = new Image();
        img.src = url;
        await img.decode();
        const scale = Math.min(1, 1600 / Math.max(img.naturalWidth, img.naturalHeight));
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
        canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/png');
    } catch (error) {
        throw new Error('This browser cannot decode this image. Use a supported static image format.');
    } finally {
        URL.revokeObjectURL(url);
    }
}

function addImageFiles(files) {
    if (pendingSubmission) return;
    attachmentNotice('');
    for (const file of files) {
        const item = reserveAttachment(file.name || 'Pasted image');
        if (!item) continue;
        normalizeImage(file).then(dataURL => {
            item.data_url = dataURL;
            item.processing = false;
        }).catch(error => {
            if (pendingAttachments.includes(item)) {
                pendingAttachments = pendingAttachments.filter(candidate => candidate !== item);
                attachmentNotice(`${item.name}: ${error.message}`);
            }
        }).finally(renderAttachments);
    }
}

function captureViewerAttachment() {
    const item = reserveAttachment('Viewer capture');
    if (!item) return;
    const token = crypto.randomUUID();
    const timer = setTimeout(() => finishCapture({capture_token: token, error: 'Viewer capture timed out. Try again.'}), 20000);
    captureRequests.set(token, {item, timer});
    sendAction({action: 'capture_view', capture_token: token}).catch(error => {
        finishCapture({capture_token: token, error: error.message});
    });
}

function finishCapture(event) {
    const request = captureRequests.get(event.capture_token);
    if (!request) return; // Includes captures requested by other browser pages.
    clearTimeout(request.timer);
    captureRequests.delete(event.capture_token);
    if (!pendingAttachments.includes(request.item)) return;
    if (event.error) {
        pendingAttachments = pendingAttachments.filter(item => item !== request.item);
        attachmentNotice(event.error);
    } else {
        request.item.data_url = 'data:image/png;base64,' + event.image_base64;
        request.item.processing = false;
    }
    renderAttachments();
}

function clearAttachmentDraft() {
    pendingAttachments = [];
    pendingSubmission = null;
    for (const request of captureRequests.values()) clearTimeout(request.timer);
    captureRequests.clear();
    document.getElementById('chat-input-field').value = '';
    attachmentNotice('');
    renderAttachments();
}

function setupAttachments() {
    const composer = document.getElementById('chat-composer');
    document.getElementById('capture-viewer-btn').onclick = captureViewerAttachment;
    composer.addEventListener('dragover', event => {
        event.preventDefault();
        composer.classList.add('drag-over');
    });
    composer.addEventListener('dragleave', event => {
        if (!composer.contains(event.relatedTarget)) composer.classList.remove('drag-over');
    });
    composer.addEventListener('drop', event => {
        event.preventDefault();
        composer.classList.remove('drag-over');
        addImageFiles(Array.from(event.dataTransfer.files));
    });
    composer.addEventListener('paste', event => {
        const files = Array.from(event.clipboardData.items).filter(item => item.kind === 'file').map(item => item.getAsFile()).filter(Boolean);
        if (!files.length) return; // Leave ordinary text paste to the browser.
        event.preventDefault();
        addImageFiles(files);
        const text = event.clipboardData.getData('text/plain');
        const input = document.getElementById('chat-input-field');
        if (text && !input.disabled) input.setRangeText(text, input.selectionStart, input.selectionEnd, 'end');
    });
}

async function sendAgentQuery() {
    if (!composerEnabled || pendingAttachments.some(item => item.processing)) return;
    if (pendingSubmission && !pendingSubmission.transportFailed) return;
    const input = document.getElementById('chat-input-field');
    if (!pendingSubmission) {
        const query = input.value.trim();
        if (!query && !pendingAttachments.length) return;
        pendingSubmission = {
            submission_id: crypto.randomUUID(), query,
            attachments: pendingAttachments.map(({name, data_url}) => ({name, data_url})),
            accepted: false
        };
    }
    const draft = pendingSubmission;
    draft.transportFailed = false;
    attachmentNotice('');
    refreshComposer();
    // Retry the same ID on an uncertain transport outcome, never a new turn.
    const timer = setTimeout(() => submissionTransportFailed(draft), 20000);
    try {
        await sendAction({action: 'agent_query', query: draft.query, attachments: draft.attachments, submission_id: draft.submission_id});
    } catch (error) {
        submissionTransportFailed(draft);
    } finally {
        if (draft.accepted || pendingSubmission !== draft) clearTimeout(timer);
    }
}

function submissionTransportFailed(draft) {
    if (pendingSubmission !== draft || draft.accepted) return;
    draft.transportFailed = true;
    attachmentNotice('Could not confirm submission. Press Send to retry the same message. Your draft is retained.');
    refreshComposer();
}

function acceptSubmission(event) {
    const draft = pendingSubmission;
    if (!draft || event.submission_id !== draft.submission_id || draft.accepted) return;
    draft.accepted = true;
    draft.transportFailed = false;
    draft.bubble = appendUserMsg(draft.query, draft.attachments);
    pendingAttachments = [];
    document.getElementById('chat-input-field').value = '';
    attachmentNotice('');
    renderAttachments();
}

function finishSubmission(event, failed = false) {
    const draft = pendingSubmission;
    if (event.submission_id && (!draft || event.submission_id !== draft.submission_id)) return false;
    if (draft) {
        if (failed) {
            if (draft.bubble) draft.bubble.remove();
            document.getElementById('chat-input-field').value = draft.query;
            pendingAttachments = draft.attachments.map(item => ({...item, processing: false}));
        } else if (!draft.accepted) {
            acceptSubmission(event);
        }
        pendingSubmission = null;
    }
    renderAttachments();
    return true;
}

function reconcileSubmission(history) {
    const draft = pendingSubmission;
    if (!draft) return;
    if (history.some(message => message.submission_id === draft.submission_id)) {
        pendingSubmission = null;
        pendingAttachments = [];
        document.getElementById('chat-input-field').value = '';
        attachmentNotice('');
        renderAttachments();
        return;
    }
    if (draft.accepted) draft.bubble = appendUserMsg(draft.query, draft.attachments);
    // A reconnect may have missed acceptance or a provider error. Request the
    // cached outcome using the same ID, without starting another model call.
    sendAction({action: 'agent_query', query: draft.query, attachments: draft.attachments, submission_id: draft.submission_id})
        .catch(() => submissionTransportFailed(draft));
}
