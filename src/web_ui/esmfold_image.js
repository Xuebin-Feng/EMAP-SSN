/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's "Save an image": Mol*'s picture of the canvas, with the legend drawn into it.
 */

// Mol*'s own image export has no legend, and a figure needs one. The picture
// is Mol*'s (its screenshot helper, at the window's size or a multiple of it,
// with or without the background); the legend of the chosen coloring is drawn
// onto it in the bottom left from the same data the page's legend is drawn
// from, scaled to the picture. Loaded after esmfold_tools.js.

let imageOpener = null;

function openImageDialog() {
    imageOpener = document.activeElement;
    byId('image-modal').classList.add('open');
    byId('image-legend').focus();
}

// Closes the dialog if it is open, and says whether it was.
function closeImage() {
    const modal = byId('image-modal');
    if (!modal.classList.contains('open')) return false;
    modal.classList.remove('open');
    if (imageOpener && imageOpener.isConnected) imageOpener.focus();
    return true;
}

// ---- The legend on the picture -----------------------------------------------------------------------
function roundedBox(context, x, y, width, height, radius) {
    context.beginPath();
    context.moveTo(x + radius, y);
    context.arcTo(x + width, y, x + width, y + height, radius);
    context.arcTo(x + width, y + height, x, y + height, radius);
    context.arcTo(x, y + height, x, y, radius);
    context.arcTo(x, y, x + width, y, radius);
    context.closePath();
}

// `text`, cut with an ellipsis to fit `limit` pixels in the context's font.
function fitText(context, text, limit) {
    if (context.measureText(text).width <= limit) return text;
    let kept = 0, high = text.length;
    while (kept < high) {
        const middle = Math.ceil((kept + high) / 2);
        if (context.measureText(text.slice(0, middle) + '…').width <= limit) kept = middle;
        else high = middle - 1;
    }
    return text.slice(0, kept) + '…';
}

// Draws the legend of `model` (colorModel()) at the bottom left of a picture `width` by `height`, `k` times
// the size it has on the page; returns the box it took, or null when the coloring has no legend to draw.
function drawLegend(context, width, height, k, model) {
    const legend = model.legend;
    if (!legend || legend.kind === 'note') return null;
    const family = getComputedStyle(document.body).fontFamily;
    const small = 11 * k, normal = 12 * k, pad = 10 * k, gap = 5 * k;
    const heading = legend.unit ? legend.heading + ' (' + legend.unit + ')' : legend.heading;
    context.font = `600 ${normal}px ${family}`;
    let textWidth = context.measureText(heading).width;
    const keys = legend.kind === 'table' ? legend.bands.map(band => [band.color, band.label]) : [];
    if (model.noData.length && !colorMode.startsWith(BUILTIN_PREFIX) && colorMode !== PLDDT_LAYER) {
        keys.push([intToHex(noDataColor), t("No data")]);
    }
    context.font = `${small}px ${family}`;
    for (const [, label] of keys) textWidth = Math.max(textWidth, context.measureText(label).width + 20 * k);
    // As wide as its words ask, at least 150 and at most 200 of the page's pixels (the legend over the canvas is no wider), and
    // within the picture; a heading or a key too long for that is cut with an ellipsis, as the page's own legend breaks it.
    const barWidth = Math.min(Math.max(150 * k, textWidth), Math.max(100 * k, Math.min(200 * k, width - 2 * (12 * k + pad))));
    context.font = `600 ${normal}px ${family}`;
    const title = fitText(context, heading, barWidth);
    context.font = `${small}px ${family}`;
    const shownKeys = keys.map(([color, label]) => [color, fitText(context, label, barWidth - 20 * k)]);
    const hasBar = legend.kind !== 'table';
    const boxWidth = Math.ceil(barWidth + 2 * pad);
    const boxHeight = Math.ceil(2 * pad + normal + gap + (hasBar ? 8 * k + gap + small : 0) + keys.length * (small + 4 * k));
    const x = 12 * k, y = height - 12 * k - boxHeight;
    context.save();
    context.globalAlpha = 0.95;
    context.fillStyle = '#ffffff';
    roundedBox(context, x, y, boxWidth, boxHeight, 8 * k);
    context.fill();
    context.globalAlpha = 1;
    context.strokeStyle = '#d4d4d8';
    context.lineWidth = Math.max(1, k);
    context.stroke();
    context.textBaseline = 'top';
    context.fillStyle = '#18181b';
    context.font = `600 ${normal}px ${family}`;
    context.fillText(title, x + pad, y + pad);
    let at = y + pad + normal + gap;
    if (hasBar) {
        const ramp = legendRamp(legend);
        const gradient = context.createLinearGradient(x + pad, 0, x + pad + barWidth, 0);
        for (const [offset, color] of ramp.stops) gradient.addColorStop(Math.min(1, Math.max(0, offset)), color);
        context.fillStyle = gradient;
        roundedBox(context, x + pad, at, barWidth, 8 * k, 4 * k);
        context.fill();
        at += 8 * k + gap;
        context.fillStyle = '#71717a';
        context.font = `${small}px ${family}`;
        context.textAlign = 'left';
        context.fillText(ramp.low, x + pad, at);
        context.textAlign = 'right';
        context.fillText(ramp.high, x + pad + barWidth, at);
        context.textAlign = 'left';
        at += small;
    }
    for (const [color, label] of shownKeys) {
        context.fillStyle = color;
        context.fillRect(x + pad, at + 1 * k, 14 * k, small - 2 * k);
        context.fillStyle = '#18181b';
        context.font = `${small}px ${family}`;
        context.fillText(label, x + pad + 20 * k, at);
        at += small + 4 * k;
    }
    context.restore();
    return { x, y, width: boxWidth, height: boxHeight };
}

// ---- The picture ---------------------------------------------------------------------------------------------
// A PNG of the canvas: `scale` times its size (device pixels), with or without Mol*'s background, with the legend if asked.
async function buildImage({ legend, transparent, scale }) {
    const helper = plugin.helpers.viewportScreenshot;
    const before = helper.behaviors.values.value;
    const canvas = document.querySelector('#viewer canvas');
    const resolution = scale === 1 ? { name: 'viewport', params: {} }
        : { name: 'custom', params: { width: canvas.width * scale, height: canvas.height * scale } };
    helper.behaviors.values.next({ ...before, transparent, axes: { name: 'off', params: {} }, resolution });
    let uri;
    try {
        uri = await helper.getImageDataUri();
    } finally {
        helper.behaviors.values.next(before);
    }
    const image = new Image();
    image.src = uri;
    await image.decode();
    const picture = document.createElement('canvas');
    picture.width = image.width;
    picture.height = image.height;
    const context = picture.getContext('2d');
    context.drawImage(image, 0, 0);
    if (legend) drawLegend(context, image.width, image.height, image.width / canvas.getBoundingClientRect().width, await colorModel());
    return new Promise(resolve => picture.toBlob(resolve, 'image/png'));
}

async function saveImage() {
    const blob = await buildImage({
        legend: byId('image-legend').checked, transparent: byId('image-transparent').checked, scale: Number(byId('image-size').value)
    });
    const link = make('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'fold-view.png';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
}

byId('t-image').addEventListener('click', openImageDialog);
byId('image-cancel').addEventListener('click', closeImage);
byId('image-modal').addEventListener('click', event => {
    if (event.target.id === 'image-modal') closeImage();
});
byId('image-save').addEventListener('click', () => {
    const button = byId('image-save');
    button.disabled = true;
    saveImage().then(() => {
        closeImage();
        showStatus(t("Saved fold-view.png."));
    }).catch(err => reportFailure("Could not save the image:", err)).finally(() => {
        button.disabled = false;
    });
});
