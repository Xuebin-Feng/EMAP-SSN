/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Fold View page's sequence strip, and stepping through the residues with the keyboard.
 */

// Under the canvas: the residues of one chain of one structure, in blocks of
// ten with their numbers, each with a bar in the color it has in 3D (read
// from the color theme Mol* draws it with, so the strip shows what the
// canvas shows). Pointing at a residue lights it in 3D and fills the Residue
// card; a click or Enter pins it, as a click on the canvas does. With the
// keyboard, the strip is one tab stop whose arrow keys move through the
// residues, and so is the canvas, whose arrow keys pin the next or previous
// residue; both say the residue they are on in the page's polite live
// region. Loaded after esmfold_residue.js.

const RESIDUE_LETTERS = 'ARNDCQEGHILKMFPSTWYV';
let sequenceOpen = true;
// The chain shown: {ref, chain}, kept while it is still there.
let sequenceTarget = null;
let sequenceSignature = '';
// The residues drawn: [{index, chain, number, code, color}], and the one with the tab stop.
let sequenceResidues = [];
let sequenceFocus = 0;
let sequenceTimer = null;

function scheduleSequence() {
    clearTimeout(sequenceTimer);
    sequenceTimer = setTimeout(renderSequence, 60);
}

// ---- Which chain ----------------------------------------------------------------------------
function sequenceChoices() {
    const choices = [];
    for (const structure of loadedStructures()) {
        const chains = [...new Set(nativeTrace(structure).ids.map(([chain]) => chain))];
        for (const chain of chains) choices.push({ ref: structure.cell.transform.ref, chain, name: structureName(structure), several: chains.length > 1 });
    }
    return choices;
}

function chooseSequenceTarget(choices) {
    const same = candidate => sequenceTarget && candidate.ref === sequenceTarget.ref && candidate.chain === sequenceTarget.chain;
    if (!choices.some(same)) {
        const wanted = pinnedResidue || hoveredResidue;
        sequenceTarget = choices.find(candidate => wanted && candidate.ref === wanted.ref && candidate.chain === wanted.chain) || choices[0] || null;
    }
    return sequenceTarget;
}

// ---- The residues of a chain, with their colors ---------------------------------------------------
// The CA trace of a structure as it was folded, with each residue's color in the theme it is drawn in.
function coloredTrace(structure) {
    const data = structure.cell.obj.data;
    let theme = null;
    const colorTheme = colorThemeOf(structure);
    if (colorTheme) {
        try {
            theme = plugin.representation.structure.themes.colorThemeRegistry.create(colorTheme.name, { structure: data }, colorTheme.params);
        } catch (err) {
            console.warn("Could not read the colors of the structure:", err);
        }
    }
    return readCaTrace(data, theme);
}

function residueAtIndex(structure, trace, k) {
    return {
        ref: structure.cell.transform.ref, chain: trace.ids[k][0], number: trace.ids[k][1], name: AMINO_ACIDS[trace.codes[k]],
        plddt: trace.plddt[k], secondary: null
    };
}

// ---- Drawing the strip --------------------------------------------------------------------------------
function sequenceResidueButton(item, structure, trace) {
    const letter = make('button', 'seq-res', RESIDUE_LETTERS[item.code]);
    letter.type = 'button';
    letter.tabIndex = -1;
    letter.dataset.k = String(item.index);
    letter.style.setProperty('--rc', item.color);
    const name = AMINO_ACIDS[item.code];
    letter.setAttribute('aria-label', name.charAt(0) + name.slice(1).toLowerCase() + ' ' + item.number);
    return letter;
}

function renderSequence() {
    const bar = byId('seqbar');
    if (!plugin || !loadedStructures().length) {
        sequenceSignature = '';
        bar.dataset.empty = '1';
        return;
    }
    bar.dataset.empty = '0';
    const choices = sequenceChoices();
    const target = chooseSequenceTarget(choices);
    const structure = target && structureByRef(target.ref);
    if (!structure) return;
    const trace = coloredTrace(structure);
    const residues = [];
    trace.ids.forEach(([chain, number], k) => {
        if (chain === target.chain) residues.push({ index: k, chain, number, code: trace.codes[k], color: intToHex(trace.colors?.[k] ?? 0xcccccc) });
    });
    const signature = JSON.stringify([choices.map(choice => [choice.ref, choice.chain, choice.name]), target, residues.map(item => [item.number, item.code, item.color])]);
    if (signature === sequenceSignature) return refreshSequenceMarks();
    sequenceSignature = signature;
    sequenceResidues = residues;
    sequenceFocus = Math.min(sequenceFocus, Math.max(0, residues.length - 1));

    // The structure and chain, when there is a choice.
    const select = byId('seq-structure');
    select.replaceChildren(...choices.map(choice => {
        const option = make('option', undefined, choice.several ? t("{name} · chain {chain}", {name: choice.name, chain: choice.chain}) : choice.name);
        option.value = choice.ref + '|' + choice.chain;
        return option;
    }));
    select.value = target.ref + '|' + target.chain;
    select.hidden = choices.length < 2;
    // The select shows as much of a long name as it has room for: the whole of it is on hover.
    select.title = select.selectedOptions[0]?.textContent ?? '';

    const body = byId('seq-body');
    const blocks = [];
    for (let at = 0; at < residues.length; at += 10) {
        const block = make('div', 'seq-block');
        const number = make('div', 'seq-num', String(residues[at].number));
        const letters = make('div', 'seq-letters');
        letters.append(...residues.slice(at, at + 10).map(item => sequenceResidueButton(item, structure, trace)));
        block.append(number, letters);
        blocks.push(block);
    }
    body.replaceChildren(...blocks);
    body.setAttribute('aria-label', t("Sequence of {name}, chain {chain}", {name: structureName(structure), chain: target.chain}));
    refreshSequenceMarks();
}

// The tab stop, the hovered residue and the pinned one, marked on the buttons that are drawn.
function refreshSequenceMarks() {
    if (!sequenceTarget) return;
    const body = byId('seq-body');
    const marked = residue => (residue && residue.ref === sequenceTarget.ref && residue.chain === sequenceTarget.chain
        ? sequenceResidues.findIndex(item => item.number === residue.number) : -1);
    const pinned = marked(pinnedResidue);
    const hovered = marked(hoveredResidue);
    const buttons = body.querySelectorAll('.seq-res');
    const stop = pinned >= 0 && !body.contains(document.activeElement) ? pinned : sequenceFocus;
    buttons.forEach((button, at) => {
        button.classList.toggle('pinned', at === pinned);
        button.classList.toggle('hover', at === hovered && at !== pinned);
        if (at === pinned) button.setAttribute('aria-current', 'true');
        else button.removeAttribute('aria-current');
        button.tabIndex = at === stop ? 0 : -1;
    });
}

// ---- Pointing at, pinning and stepping through residues ---------------------------------------------------
function sequenceResidue(button) {
    const structure = structureByRef(sequenceTarget.ref);
    return residueAtIndex(structure, nativeTrace(structure), Number(button.dataset.k));
}

function announceResidue(residue) {
    const structure = structureByRef(residue.ref);
    byId('live').textContent = isFolded(structure)
        ? t("Residue {name}, chain {chain}, pLDDT {value}.", {name: residueTitle(residue), chain: residue.chain, value: residue.plddt.toFixed(1)})
        : t("Residue {name}, chain {chain}, B-factor {value}.", {name: residueTitle(residue), chain: residue.chain, value: residue.plddt.toFixed(1)});
}

function lightResidue(residue) {
    try {
        plugin.managers.interactivity.lociHighlights.highlightOnly({ loci: residueLoci(residue) });
    } catch (err) {
        console.warn("Could not light the residue:", err);
    }
}

function unlightResidues() {
    plugin.managers.interactivity.lociHighlights.clearHighlights();
}

function pointAtSequence(button) {
    const residue = sequenceResidue(button);
    hoveredResidue = residue;
    lightResidue(residue);
    if (!pinnedResidue) renderResidueCard();
    refreshSequenceMarks();
}

function leaveSequence() {
    hoveredResidue = null;
    unlightResidues();
    if (!pinnedResidue) renderResidueCard();
    refreshSequenceMarks();
}

byId('seq-body').addEventListener('pointerover', event => {
    const button = event.target.closest?.('.seq-res');
    if (button) pointAtSequence(button);
});
byId('seq-body').addEventListener('pointerleave', () => {
    if (!byId('seq-body').contains(document.activeElement)) leaveSequence();
});
byId('seq-body').addEventListener('click', event => {
    const button = event.target.closest?.('.seq-res');
    if (!button) return;
    const residue = sequenceResidue(button);
    sequenceFocus = sequenceResidues.findIndex(item => item.index === Number(button.dataset.k));
    pinResidue(residue, residueLoci(residue));
});
byId('seq-body').addEventListener('focusin', event => {
    const button = event.target.closest?.('.seq-res');
    if (!button) return;
    sequenceFocus = sequenceResidues.findIndex(item => item.index === Number(button.dataset.k));
    pointAtSequence(button);
    announceResidue(sequenceResidue(button));
});
byId('seq-body').addEventListener('focusout', event => {
    if (!byId('seq-body').contains(event.relatedTarget)) leaveSequence();
});
byId('seq-body').addEventListener('keydown', event => {
    const steps = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 10, ArrowUp: -10 };
    const buttons = [...byId('seq-body').querySelectorAll('.seq-res')];
    const at = buttons.indexOf(document.activeElement);
    if (at < 0) return;
    let target = null;
    if (event.key in steps) target = Math.min(buttons.length - 1, Math.max(0, at + steps[event.key]));
    else if (event.key === 'Home') target = 0;
    else if (event.key === 'End') target = buttons.length - 1;
    else if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        buttons[at].click();
        return;
    }
    if (target === null) return;
    event.preventDefault();
    buttons[at].tabIndex = -1;
    buttons[target].tabIndex = 0;
    buttons[target].focus();
});

byId('seq-structure').addEventListener('change', event => {
    const [ref, chain] = event.target.value.split('|');
    sequenceTarget = { ref, chain };
    sequenceFocus = 0;
    renderSequence();
});

byId('seq-toggle').addEventListener('click', () => {
    sequenceOpen = !sequenceOpen;
    byId('seqbar').dataset.open = sequenceOpen ? '1' : '0';
    byId('seq-toggle').setAttribute('aria-expanded', sequenceOpen ? 'true' : 'false');
    if (plugin) setTimeout(() => viewerInstance.handleResize(), 220);
});

// ---- Stepping with the keyboard on the canvas ----------------------------------------------------------------
// The canvas is one tab stop. Arrow keys pin the next or previous residue of the chain being followed (the
// pinned one's, or the first); Home and End the first and last; R resets the view and F comes to the pinned residue.
function stepResidue(to) {
    const choices = sequenceChoices();
    const target = chooseSequenceTarget(choices);
    const structure = target && structureByRef(target.ref);
    if (!structure) return;
    const trace = nativeTrace(structure);
    const chain = trace.ids.map((id, k) => k).filter(k => trace.ids[k][0] === target.chain);
    if (!chain.length) return;
    const current = pinnedResidue && pinnedResidue.ref === target.ref && pinnedResidue.chain === target.chain
        ? chain.findIndex(k => trace.ids[k][1] === pinnedResidue.number) : -1;
    const next = to === 'first' ? 0 : to === 'last' ? chain.length - 1
        : current < 0 ? (to > 0 ? 0 : chain.length - 1) : Math.min(chain.length - 1, Math.max(0, current + to));
    const residue = residueAtIndex(structure, trace, chain[next]);
    pinResidue(residue, residueLoci(residue));
    announceResidue(residue);
}

(() => {
    const view = byId('viewer');
    view.tabIndex = 0;
    view.setAttribute('role', 'application');
    view.setAttribute('aria-label', t("3D structure view. Arrow keys step through the residues, F comes to the pinned one, R resets the view, Escape lets go of it."));
    view.addEventListener('keydown', event => {
        if (event.target !== view || event.altKey || event.ctrlKey || event.metaKey) return;
        const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: 'first', End: 'last' };
        if (event.key in keys) {
            event.preventDefault();
            stepResidue(keys[event.key]);
        } else if (event.key === 'r' || event.key === 'R') {
            event.preventDefault();
            plugin.managers.camera.reset();
        } else if ((event.key === 'f' || event.key === 'F') && pinnedResidue) {
            event.preventDefault();
            zoomToResidue(pinnedResidue);
        }
    });
})();
