/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Opening a tree in Tree View: a file chosen in the Tree list, dropped on the page, or pasted with Ctrl+V when no field
 * has the focus. The page reads the text with the reader copied from Cladewise (tree_read.js and its two companions: Newick,
 * NEXUS, PhyloXML, NeXML and JSON), asks which tree when a file holds several, and sends the tree to the Viewer
 * (/api/tree/upload, the tree_upload action), which checks everything, matches the leaves to the network's nodes by full
 * sequence name, and keeps the tree for the session. What it answers is drawn as any other tree, and the card of how the
 * leaves matched is shown. A tree none of whose leaves is a node of the network is not drawn: the card shows a few names of
 * the tree beside a few headers of the network instead.
 *
 * The Viewer sends facts, never sentences: a note on a refusal, counts on a match. Every word shown is written here, through
 * t(). A name, a header or the file's own words are shown as text, never as markup.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const state = TV.state;
    const $ = (id) => document.getElementById(id);
    // A file of several trees that is waiting for a choice: { reader, source }.
    let pending = null;
    // How many openings were begun, so that a slow one an newer one has overtaken is dropped.
    let openings = 0;
    let copiedTimer = null;
    // The most names the box of unmatched leaves lists (the copy has them all).
    const LISTED = 5000;
    // How long the last opening took, in milliseconds: reading the text, the Viewer's answer, and the first frame.
    TV.lastOpening = null;

    // ---- What a refusal says -----------------------------------------------------------------------------------

    // One sentence for the note the Viewer gives when it does not take a tree.
    function refusalText(note, detail) {
        detail = detail || {};
        switch (note) {
            case "bad_request": return t("The Viewer could not read the request.");
            case "too_large": return t("The tree is too large for the Viewer to take in one request.");
            case "no_leaves": return t("The tree has no leaves.");
            case "too_few_leaves": return t("The tree has fewer than two leaves, so there is nothing to draw.");
            case "too_many_leaves": return t("The tree has more than {limit} leaves, which is more than Tree View takes.", { limit: Number(detail.limit).toLocaleString() });
            case "bad_labels": return t("The names of the leaves are not text.");
            case "empty_label": return t("A leaf has no name.");
            case "label_too_long": return t("A name is longer than {limit} characters.", { limit: Number(detail.limit).toLocaleString() });
            case "duplicate_label": return t("Two leaves are called {label}. A name must name one leaf, so that it can be matched to one node of the network.", { label: String(detail.label) });
            case "bad_arrays": return t("The tree's numbers are not in a form the Viewer accepts.");
            case "size_mismatch": return t("The tree's lists do not have one entry for each node.");
            case "too_many_nodes": return t("The tree has more nodes than Tree View takes.");
            case "not_post_order": return t("The tree's nodes do not form one tree: a node's parent must come after it, and the last node must be the root.");
            case "leaf_count_mismatch": return t("The number of names does not match the number of leaves of the tree.");
            case "bad_length": return t("A branch length is not a finite number of 0 or more.");
            case "bad_support": return t("A support value is not a percentage from 0 to 100.");
            case "bad_rooted": return t("What the tree says of its rooting is not in a form the Viewer accepts.");
            case "bad_text": return t("A name or a note of the file is not text the Viewer accepts.");
            case "bad_warnings": return t("The notes the reader made are not in a form the Viewer accepts.");
            case "bad_clade_names": return t("The names of the clades are not in a form the Viewer accepts.");
            default: return t("The Viewer did not take the tree: {note}", { note: String(note) });
        }
    }
    TV.refusalText = refusalText;

    // The card for a text that is no tree the reader takes (reason: the reader's own words, with the line and column).
    function errorCard(source, reason) {
        pending = null;
        TV.showCard({
            title: t("This is not a tree Tree View can read"),
            body: t("{name} could not be opened.", { name: source.name }),
            more: reason,
            open: t("Open another file"),
            retry: t("Back to the tree"),
        });
    }

    function refusalCard(source, result) {
        pending = null;
        TV.showCard({
            title: t("The Viewer did not take this tree"),
            body: t("{name} was read, but: {reason}", { name: source.name, reason: refusalText(result.note, result.detail) }),
            open: t("Open another file"),
            retry: t("Back to the tree"),
        });
    }

    // ---- The report of the match ------------------------------------------------------------------------------

    // The names of the leaves that are no node of the network, in the tree's order.
    function unmatchedNames(model) {
        const names = [];
        for (let leaf = 0; leaf < model.leafCount; leaf++) if (model.viewerNode[leaf] < 0) names.push(model.labels[leaf]);
        return names;
    }

    function copyText(text, button) {
        const label = button.querySelector("span");
        const original = label.dataset.label || label.textContent;
        label.dataset.label = original;
        const done = () => {
            label.textContent = t("Copied");
            clearTimeout(copiedTimer);
            copiedTimer = setTimeout(() => { label.textContent = original; }, 1500);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, () => {});
    }

    // A list of short texts (a name each) in a column, in the page's monospace.
    function listOf(items) {
        const list = document.createElement("ul");
        for (const item of items) {
            const entry = document.createElement("li");
            entry.textContent = item;
            list.appendChild(entry);
        }
        return list;
    }

    function countLine(text) {
        const item = document.createElement("li");
        item.textContent = text;
        return item;
    }

    // The card of how the leaves of the opened tree matched the nodes of the network: the counts, which step matched how
    // many, the network's nodes the tree has no leaf for, and the leaves that matched no node, as a list to copy.
    function fillReport(panel, model) {
        const match = model.match;
        panel.replaceChildren();
        const head = document.createElement("div");
        head.className = "head";
        const title = document.createElement("h2");
        title.textContent = t("How the tree matched the network");
        const close = document.createElement("button");
        close.type = "button";
        close.className = "btn btn-icon btn-ghost";
        close.title = t("Close");
        close.setAttribute("aria-label", t("Close"));
        close.appendChild(TV.icon("x"));
        close.addEventListener("click", () => TV.showReport(false));
        head.append(title, close);
        panel.appendChild(head);

        const summary = document.createElement("p");
        summary.className = "note";
        summary.textContent = t("{matched} of the {leaves} leaves of {file} are nodes of this network.", {
            matched: match.matched.toLocaleString(), leaves: match.leaves.toLocaleString(), file: model.uploaded.file,
        });
        panel.appendChild(summary);
        const steps = document.createElement("ul");
        steps.className = "step-list";
        steps.append(
            countLine(t("Matched by the exact full name: {count}", { count: match.exact.toLocaleString() })),
            countLine(t("Matched by the name after the Viewer's header rules: {count}", { count: match.sanitized.toLocaleString() })),
            countLine(t("Leaves that are not in the network: {count}", { count: match.unmatched.toLocaleString() })),
            countLine(t("Nodes of the network that are not in the tree: {count}", { count: match.network_nodes_not_in_tree.toLocaleString() })),
        );
        if (match.ambiguous) {
            steps.appendChild(countLine(t("Leaves whose name reaches a node that another leaf has too, left unmatched: {count}", { count: match.ambiguous.toLocaleString() })));
        }
        panel.appendChild(steps);

        if (match.unmatched) {
            const names = unmatchedNames(model);
            const subtitle = document.createElement("div");
            subtitle.className = "group-title";
            subtitle.textContent = t("Leaves that are not in the network");
            const box = document.createElement("textarea");
            box.readOnly = true;
            box.setAttribute("aria-label", t("Leaves that are not in the network"));
            box.value = names.slice(0, LISTED).join("\n") + (names.length > LISTED ? "\n" + t("... and {count} more", { count: (names.length - LISTED).toLocaleString() }) : "");
            const copy = document.createElement("button");
            copy.type = "button";
            copy.className = "btn";
            copy.id = "report-copy";
            copy.appendChild(TV.icon("copy"));
            const label = document.createElement("span");
            label.textContent = t("Copy the list");
            copy.appendChild(label);
            copy.addEventListener("click", () => copyText(names.join("\n"), copy));
            panel.append(subtitle, box, copy);
        }
    }

    // Show the card of the match, or take it away.
    TV.showReport = function (on) {
        const panel = $("match-report");
        if (!on || !state.model || !state.model.match || !state.model.uploaded) {
            panel.classList.remove("shown");
            return;
        }
        $("details").classList.remove("shown");
        TV.showMismatch(false);
        fillReport(panel, state.model);
        panel.classList.add("shown");
    };

    // ---- The leaves and nodes the banner's warnings count ----------------------------------------------------------

    // How many answers were asked of the Viewer, so that an answer to an older question (another source, or a card closed
    // and opened again) is not used.
    let mismatchAsks = 0;

    // One list of the card: a heading with the count, the names in a box (the first LISTED, then how many more) and a
    // button that copies every name.
    function mismatchList(panel, heading, names) {
        const subtitle = document.createElement("div");
        subtitle.className = "group-title";
        subtitle.textContent = heading;
        const box = document.createElement("textarea");
        box.readOnly = true;
        box.setAttribute("aria-label", heading);
        box.value = names.slice(0, LISTED).join("\n") + (names.length > LISTED ? "\n" + t("... and {count} more", { count: (names.length - LISTED).toLocaleString() }) : "");
        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "btn mismatch-copy";
        copy.appendChild(TV.icon("copy"));
        const label = document.createElement("span");
        label.textContent = t("Copy the list");
        copy.appendChild(label);
        copy.addEventListener("click", () => copyText(names.join("\n"), copy));
        panel.append(subtitle, box, copy);
    }

    // The heading of each list the Viewer answers, with its count.
    function mismatchHeading(code, count) {
        const values = { count: count.toLocaleString() };
        switch (code) {
            case "leaves_not_in_network": return t("Leaves that are not in the network: {count}", values);
            case "network_nodes_not_in_tree": return t("Nodes of the network that are not in the tree: {count}", values);
            case "leaves_not_in_alignment": return t("Leaves whose sequence is not in the loaded alignment: {count}", values);
            default: return t("Leaves whose name reaches a node that another leaf has too, left unmatched: {count}", values);
        }
    }

    // Fill the card: its heading and close button, then a list for each warning the banner has that names leaves or
    // nodes. answer is what the Viewer sent, or null when it could not be asked.
    function fillMismatch(panel, answer) {
        panel.replaceChildren();
        const head = document.createElement("div");
        head.className = "head";
        const title = document.createElement("h2");
        title.textContent = t("The leaves and nodes the warnings count");
        const close = document.createElement("button");
        close.type = "button";
        close.className = "btn btn-icon btn-ghost";
        close.title = t("Close");
        close.setAttribute("aria-label", t("Close"));
        close.appendChild(TV.icon("x"));
        close.addEventListener("click", () => TV.showMismatch(false));
        head.append(title, close);
        panel.appendChild(head);
        if (!answer) {
            const note = document.createElement("p");
            note.className = "note";
            note.textContent = t("The Viewer could not list them.");
            panel.appendChild(note);
            return;
        }
        const warned = new Set(state.model.warnings.map((warning) => warning.code));
        for (const code of ["leaves_not_in_network", "network_nodes_not_in_tree", "leaves_not_in_alignment", "ambiguous_names"]) {
            const found = answer[code];
            if (warned.has(code) && found && found.count) mismatchList(panel, mismatchHeading(code, found.count), found.names);
        }
    }

    // Ask the Viewer which leaves and nodes the warnings of the tree now shown count, and show them in the card.
    async function loadMismatch() {
        const panel = $("mismatch-card");
        const ask = ++mismatchAsks;
        const source = state.model.source;
        fillMismatch(panel, { });
        let answer = null;
        try {
            const response = await fetch("/api/tree/mismatch?source=" + encodeURIComponent(source), { cache: "no-store" });
            if (response.ok) answer = await response.json();
        } catch (error) {
            answer = null;
        }
        if (ask !== mismatchAsks || !panel.classList.contains("shown")) return;
        fillMismatch(panel, answer);
    }

    // Show the card of the leaves and nodes the banner's warnings count, or take it away. Showing it (again, after the tree
    // or the source changed) asks the Viewer; the card closes the other cards, and the button says whether it is open.
    TV.showMismatch = function (on) {
        const panel = $("mismatch-card"), button = $("mismatch-btn");
        const open = on && state.model && !button.hidden;
        button.setAttribute("aria-expanded", String(Boolean(open)));
        if (!open) {
            mismatchAsks++;
            panel.classList.remove("shown");
            return;
        }
        $("details").classList.remove("shown");
        $("match-report").classList.remove("shown");
        panel.classList.add("shown");
        loadMismatch();
    };

    // The card for a tree none of whose leaves is a node of the network: it is not drawn, and a few names of the tree stand
    // beside a few headers of the network, so that what a program did to the names (IQ-TREE writes a space as _, for one) shows.
    TV.noMatchCard = function (payload) {
        const match = payload.match || {};
        const info = payload.uploaded || {};
        return {
            title: t("No leaf of this tree is a node of the network"),
            body: t("None of the {leaves} leaves of {file} has the full name of a node of this network, so there is nothing to show beside it. A leaf matches when its name is the node's whole header, or becomes it under the rules the Viewer writes headers by.", {
                leaves: Number(match.leaves || 0).toLocaleString(), file: info.file || "",
            }),
            build(container) {
                const lists = document.createElement("div");
                lists.className = "report-lists";
                const left = document.createElement("div");
                const leftTitle = document.createElement("h3");
                leftTitle.textContent = t("Names in the tree");
                left.append(leftTitle, listOf(match.tree_examples || []));
                const right = document.createElement("div");
                const rightTitle = document.createElement("h3");
                rightTitle.textContent = t("Headers in the network");
                right.append(rightTitle, listOf(match.network_examples || []));
                lists.append(left, right);
                container.appendChild(lists);
            },
            more: t("Open a tree whose leaves carry the network's full headers, or compare the two lists: a tree program may have changed the names."),
            open: t("Open another tree file"),
        };
    };

    // ---- Opening -----------------------------------------------------------------------------------------------

    function base64Of(typed) {
        const bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
        let text = "";
        const piece = 0x8000;
        for (let at = 0; at < bytes.length; at += piece) text += String.fromCharCode.apply(null, bytes.subarray(at, at + piece));
        return btoa(text);
    }

    // What the Viewer is sent for a tree the reader made: the arrays of the nodes in post-order (as the base64 of their
    // bytes), the leaves' names, and what the file said of the tree. A branch that has a negative length is drawn with none.
    function bodyOf(tree, reader, source) {
        const length = new Float32Array(tree.length);
        for (let i = 0; i < length.length; i++) if (length[i] < 0) length[i] = 0;
        const names = {};
        if (tree.cladeNames) tree.cladeNames.forEach((name, node) => { names[node] = name; });
        return {
            name: source.name,
            format: reader.format,
            tree_name: tree.name || "",
            labels: tree.labels,
            parent: base64Of(tree.parent),
            length: base64Of(length),
            support: base64Of(tree.support),
            rooted: tree.rooted === true ? true : tree.rooted === false ? false : null,
            root_stated: !!tree.rootStated,
            has_lengths: !!tree.hasLengths,
            support_name: tree.supportSets && tree.supportSets[0] ? tree.supportSets[0].name || "" : "",
            dialects: tree.dialects || [],
            clade_names: names,
            warnings: (tree.warnings || []).map((warning) => (warning.count === undefined ? { code: warning.code } : { code: warning.code, count: warning.count })),
        };
    }

    // Send tree `index` of `reader` (read from `source`: { name, size, pasted }) to the Viewer, and draw what it answers: the
    // tree, with the card of the match; or the card that says why not. Resolves to whether a tree is drawn.
    TV.sendTree = async function (reader, source, index) {
        const opening = ++openings;
        const started = performance.now();
        pending = null;
        TV.show("loading");
        // Let the page say it is working before it does.
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (opening !== openings) return false;
        let body;
        try {
            body = bodyOf(reader.tree(index), reader, source);
        } catch (error) {
            errorCard(source, String(error.message || error));
            return false;
        }
        const read = performance.now();
        let result;
        try {
            const response = await fetch("/api/tree/upload", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!response.ok) throw new Error(response.status + " " + response.statusText);
            result = await response.json();
        } catch (error) {
            if (opening !== openings) return false;
            TV.showCard({
                title: t("Tree View could not reach the Viewer"),
                body: t("The Viewer did not answer: {reason}", { reason: String(error.message || error) }),
                more: t("Check that the Viewer is still running, and press Back to the tree."),
                retry: t("Back to the tree"),
            });
            return false;
        }
        if (opening !== openings) return false;
        const answered = performance.now();
        if (!result.ok) {
            refusalCard(source, result);
            return false;
        }
        const shown = TV.applyPayload(result.state);
        if (shown) TV.showReport(true);
        TV.lastOpening = { read: read - started, answered: answered - read, drawn: performance.now() - answered };
        return shown;
    };

    // The trees of a file of several, for a list to choose from.
    function choiceLabels(reader) {
        return reader.entries.map((entry, index) => (index + 1).toLocaleString() + ". " + (entry.name || t("Tree {number}", { number: index + 1 })));
    }

    // Open the tree text of `source` ({ name, size, pasted }): read what trees it holds, and send the tree, or ask which of
    // several. Resolves to whether a tree is drawn.
    TV.openText = async function (text, source) {
        const opening = ++openings;
        TV.show("loading");
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (opening !== openings) return false;
        let reader;
        try {
            if (text.length > TV.MAX_FILE_BYTES) throw new Error(t("The tree text is larger than {size} MB.", { size: TV.MAX_FILE_BYTES / (1024 * 1024) }));
            reader = TV.readTreeText(text);
        } catch (error) {
            errorCard(source, String(error.message || error));
            return false;
        }
        if (reader.entries.length > 1) {
            pending = { reader, source };
            TV.showCard({
                title: t("Choose a tree"),
                body: t("{name} holds {count} trees.", { name: source.name, count: reader.entries.length.toLocaleString() }),
                choices: choiceLabels(reader),
                open: t("Open this tree"),
                more: t("Double-click a tree, or press Enter, to open it."),
                retry: t("Back to the tree"),
            });
            return false;
        }
        return TV.sendTree(reader, source, 0);
    };

    // Open a file (a File of the page's file chooser or of a drop).
    TV.openFile = async function (file) {
        const source = { name: file.name, size: file.size, pasted: false };
        if (file.size > TV.MAX_FILE_BYTES) {
            ++openings;
            errorCard(source, t("The file is larger than {size} MB.", { size: TV.MAX_FILE_BYTES / (1024 * 1024) }));
            return false;
        }
        let text;
        try {
            text = await file.text();
        } catch (error) {
            ++openings;
            errorCard(source, t("The file could not be read: {reason}", { reason: String(error.message || error) }));
            return false;
        }
        return TV.openText(text, source);
    };

    TV.chooseFile = function () {
        $("file-input").click();
    };

    // The tree the card's list has chosen.
    function openChosen() {
        if (!pending) return;
        const index = Number($("empty-choice").value);
        if (Number.isInteger(index) && index >= 0) TV.sendTree(pending.reader, pending.source, index);
    }

    // Whether an element takes typed text, so that a paste into it is the field's and not a tree.
    function takesText(element) {
        if (!element || element.nodeType !== 1) return false;
        if (element.isContentEditable || element.tagName === "TEXTAREA" || element.tagName === "SELECT") return true;
        if (element.tagName !== "INPUT") return false;
        return !["checkbox", "radio", "color", "range", "button", "submit", "reset", "file", "image"].includes(element.type);
    }
    TV.takesText = takesText;

    document.addEventListener("DOMContentLoaded", () => {
        $("empty-open").addEventListener("click", () => {
            if (!$("empty-choice").hidden && pending) openChosen(); else TV.chooseFile();
        });
        $("empty-choice").addEventListener("dblclick", openChosen);
        $("empty-choice").addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                openChosen();
            }
        });
        $("file-input").addEventListener("change", (event) => {
            const file = event.target.files[0];
            event.target.value = "";
            if (file) TV.openFile(file);
        });

        // A file dropped anywhere on the page.
        let dragging = 0;
        const hasFiles = (event) => !!event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files");
        document.addEventListener("dragenter", (event) => {
            if (!hasFiles(event)) return;
            event.preventDefault();
            dragging++;
            document.body.classList.add("dropping");
        });
        document.addEventListener("dragover", (event) => {
            if (!hasFiles(event)) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
        });
        document.addEventListener("dragleave", (event) => {
            if (!hasFiles(event)) return;
            dragging = Math.max(0, dragging - 1);
            if (!dragging) document.body.classList.remove("dropping");
        });
        document.addEventListener("drop", (event) => {
            if (!hasFiles(event)) return;
            event.preventDefault();
            dragging = 0;
            document.body.classList.remove("dropping");
            const file = event.dataTransfer.files[0];
            if (file) TV.openFile(file);
        });

        // Tree text pasted when no field has the focus.
        document.addEventListener("paste", (event) => {
            if (takesText(event.target)) return;
            const text = event.clipboardData ? event.clipboardData.getData("text/plain") : "";
            if (!text.trim()) return;
            event.preventDefault();
            TV.openText(text, { name: t("Pasted text"), size: text.length, pasted: true });
        });
    });
})(window.TreeView);
