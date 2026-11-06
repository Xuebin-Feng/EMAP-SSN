/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tree View's page: it asks the Viewer for the tree of the loaded alignment, says why when there is none to draw,
 * says how the tree fits the alignment and the network, and wires the search, the buttons and the theme.
 *
 * The Viewer sends facts, never sentences: a status, counts and warning codes. Every word the page shows is
 * written here, through t(), in the Viewer's language. A header, a path or a note of the file is shown as it is,
 * as text, never as markup.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const state = TV.state;
    const $ = (id) => document.getElementById(id);
    let eventSource = null;
    let loading = false;
    let reconnecting = false;
    let searchTimer = null;
    let copiedTimer = null;
    // Whether the overview map is wanted (it is shown only with a tree).
    let overviewOn = true;
    // The node of the network this page last asked the Viewer to focus on (-1 for none), so that it asks again only
    // when the pinned leaf changes; and how many selection events came, so that a selection asked for before the
    // events it has not heard of is not taken over them.
    let postedFocus = -1;
    let selectionEvents = 0;
    // Whether the banner's note on the rooting was dismissed with its x. It lasts as long as this page does and is
    // kept nowhere: opening the page again shows the note again.
    let rootingNoteDismissed = false;
    // How many times the network's selection changed, and whether the nodes next to it are being asked for: they are
    // asked for after each change, one request at a time, and an answer to an older selection is not used.
    let selectionChanges = 0;
    let connectedBusy = false;
    let connectedAgain = false;

    // ---- Icons -----------------------------------------------------------------------------------------------

    // An icon from the page's sprite (its <symbol id="i-NAME">), drawn in the text colour.
    const SVG_NS = "http://www.w3.org/2000/svg";
    function icon(name) {
        const svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("class", "icon");
        svg.setAttribute("aria-hidden", "true");
        const use = document.createElementNS(SVG_NS, "use");
        use.setAttribute("href", "#i-" + name);
        svg.appendChild(use);
        return svg;
    }

    function setIcon(svg, name) {
        svg.querySelector("use").setAttribute("href", "#i-" + name);
    }

    // ---- What the page shows ---------------------------------------------------------------------------------

    // Show one of the page's three states: "loading", "empty" (the card that says why there is no tree) or
    // "tree" (the canvas). The controls that need a tree are on only with one.
    function show(which) {
        $("loading").classList.toggle("shown", which === "loading");
        $("empty").classList.toggle("shown", which === "empty");
        const hasTree = which === "tree";
        for (const id of ["search-input", "fit-btn", "details-btn", "layout-select", "overview-btn"]) $(id).disabled = !hasTree;
        showSupportControl(hasTree);
        $("minimap").hidden = !hasTree || !overviewOn;
        if (!hasTree) {
            for (const id of ["search-prev", "search-next", "phylogram-btn", "cladogram-btn", "unfold-btn"]) $(id).disabled = true;
            $("search-count").textContent = "";
            $("details").classList.remove("shown");
            $("banner").classList.remove("shown");
            $("summary").textContent = "";
            TV.hideTooltip($("tooltip"));
        }
        document.body.dataset.view = which;
    }

    // The card that says why there is no tree. spec: { title, body, path, more }, path shown as a path to copy.
    function showEmpty(spec) {
        state.model = null;
        state.layout = null;
        state.hover = state.pinned = -1;
        state.matches = new Int32Array(0);
        state.selection = TV.emptySelection();
        postedFocus = -1;
        $("empty-title").textContent = spec.title;
        $("empty-body").textContent = spec.body;
        const path = $("empty-path");
        path.hidden = !spec.path;
        path.textContent = spec.path || "";
        const more = $("empty-more");
        more.hidden = !spec.more;
        more.textContent = spec.more || "";
        $("empty-copy").hidden = !spec.path;
        show("empty");
        TV.drawNow();
    }

    // What to say for each status the Viewer gives when there is nothing to draw. payload: the Viewer's answer, with
    // reason added when this page could not use the tree it was sent.
    function emptyCard(payload) {
        const alignment = payload.alignment || "";
        switch (payload.status) {
            case "no_alignment":
                return {
                    title: t("No alignment is loaded"),
                    body: t("Tree View shows the guide tree of the alignment that is loaded in the Viewer. Load an alignment, and its tree appears here."),
                };
            case "unnamed_alignment":
                return {
                    title: t("This alignment has no guide tree"),
                    body: t("{alignment} was not named by the Embedding MSA tool, which saves a guide tree for each alignment it makes, beside it, in a file named after the alignment: a name like Set_[model]_alignment.fasta. A renamed or external alignment has none.", { alignment }),
                };
            case "no_tree":
                return {
                    title: t("No guide tree was found for this alignment"),
                    body: t("The guide tree of {alignment} should be the file below, which the Embedding MSA tool saves in the Guide_Trees folder beside the alignment. It is not there.", { alignment }),
                    path: payload.tree_path,
                    more: t("Run the Embedding MSA tool again to make it, and press Check again."),
                };
            case "unreadable":
                return {
                    title: t("The guide tree could not be read"),
                    body: t("The file is there, but this page cannot read it as a guide tree: {reason}", { reason: payload.reason || "" }),
                    path: payload.tree_path,
                };
            case "no_overlap":
                return {
                    title: t("The tree does not match this network"),
                    body: t("None of the {leaves} leaves of this tree is a node of the network, so there is nothing to show beside it. The tree and the network may come from different sequence sets.", { leaves: ((payload.counts || {}).leaves || 0).toLocaleString() }),
                    path: payload.tree_path,
                };
            case "no_network":
                return {
                    title: t("No network is loaded"),
                    body: t("The network threshold tree is built from the edges of the network the Viewer has loaded, and none is loaded."),
                };
            case "no_edges":
                return {
                    title: t("This network has no edges"),
                    body: t("The network threshold tree is built from the edges of the network the Viewer has loaded, and this network has none, so there is nothing to join."),
                };
            case "unreachable":
                return {
                    title: t("Tree View could not reach the Viewer"),
                    body: t("The Viewer did not answer: {reason}", { reason: payload.reason || "" }),
                    more: t("Check that the Viewer is still running, and press Check again."),
                };
            default:
                return {
                    title: t("Tree View could not read the Viewer's answer"),
                    body: t("The Viewer answered with a status this page does not know: {status}", { status: String(payload.status) }),
                };
        }
    }

    // ---- Which tree is shown -----------------------------------------------------------------------------------

    // Fill in the chooser from the trees the Viewer offers (payload.sources): each is listed, one that cannot be shown is
    // disabled and says why when the pointer rests on it, and the one in use is selected. The chooser stays on while the
    // Viewer is reachable, so that another tree can be chosen when this one has nothing to draw.
    function showSources(payload) {
        const select = $("source-select");
        if (Array.isArray(payload.sources)) state.sources = payload.sources;
        if (payload.source) state.source = payload.source;
        for (const option of select.options) {
            const entry = state.sources.find((item) => item.id === option.value);
            option.disabled = !!entry && !entry.available;
            option.title = entry && !entry.available && entry.reason ? emptyCard({ status: entry.reason, alignment: payload.alignment }).title : "";
        }
        select.value = state.source;
        select.disabled = !state.sources.length;
    }

    // Choose another tree: the Viewer keeps the choice for the session. What is folded and where the view is start over,
    // as they do for any new tree; the colours, the selection and the nodes the network hides come through the nodes the
    // leaves stand for, so they carry over to it.
    function chooseSource(value) {
        if (value === state.source && state.model) return;
        TV.loadTree(value).then((ok) => { $("source-select").value = state.source; return ok; });
    }

    // ---- The colours of the network ----------------------------------------------------------------------------

    let legendOpen = true;
    let colorsLoading = false;
    let colorsAgain = false;

    // The legend: the colours the leaves have, most leaves first, each of them a button that marks its leaves.
    function showLegend() {
        const colors = state.colors;
        const legend = $("legend");
        const list = $("legend-list");
        list.replaceChildren();
        const rows = colors ? colors.legend : [];
        legend.classList.toggle("shown", rows.length > 0);
        if (!rows.length) return;
        for (const entry of rows) {
            const item = document.createElement("li");
            const button = document.createElement("button");
            button.className = "legend-row";
            button.setAttribute("aria-pressed", String(state.legendClass === entry.cls));
            button.title = t("Mark the leaves of this colour");
            const swatch = document.createElement("span");
            swatch.className = "swatch";
            swatch.style.backgroundColor = entry.css;
            const name = document.createElement("span");
            name.className = "name";
            name.textContent = TV.classLabel(entry.cls);
            const count = document.createElement("span");
            count.className = "count";
            count.textContent = entry.count.toLocaleString();
            button.append(swatch, name, count);
            button.addEventListener("click", () => markColor(entry.cls));
            item.appendChild(button);
            list.appendChild(item);
        }
        const more = $("legend-more");
        more.hidden = !(colors.more > 0);
        if (colors.more > 0) more.textContent = t("{count} more colours", { count: colors.more.toLocaleString() });
        list.hidden = !legendOpen;
        more.hidden = more.hidden || !legendOpen;
        $("legend-toggle").setAttribute("aria-expanded", String(legendOpen));
        setIcon($("legend-icon"), legendOpen ? "chevron-up" : "chevron-down");
    }

    // Mark the leaves of one colour as a search marks what it found; the same button again takes the marks off.
    function markColor(cls) {
        const colors = state.colors;
        if (!colors) return;
        if (state.legendClass === cls) {
            state.legendClass = -1;
            state.matches = new Int32Array(0);
        } else {
            state.legendClass = cls;
            state.matches = TV.leavesOfClass(colors, cls);
        }
        state.current = -1;
        state.query = "";
        $("search-input").value = "";
        $("search-box").classList.remove("has-text");
        showSearch(0);
        TV.minimapChanged();
        TV.requestDraw();
        showLegend();
    }

    // The colours the Viewer outlines its selected nodes and their neighbours in, as the page draws them, and whether it
    // outlines the neighbours at all.
    function marksOf(raw) {
        const css = (rgba) => "rgb(" + rgba.slice(0, 3).join(", ") + ")";
        return { selected: css(raw.selected), connected: css(raw.connected), connectedOn: !!raw.connected_on };
    }

    // The colours the Viewer has now, drawn; none (the tree in its own colours) when the Viewer has none to give. The
    // nodes it has hidden are left out of the tree, which is laid out again without them.
    function applyColors(payload) {
        let colors = null;
        if (payload && payload.status === "ok" && state.model) {
            try {
                colors = TV.decodeColors(payload, state.model);
            } catch (error) {
                colors = null;
            }
        }
        const marksBefore = JSON.stringify(state.marks);
        if (payload && payload.marks && payload.marks.selected && payload.marks.connected) state.marks = marksOf(payload.marks);
        // A colour the legend marked that is no longer there marks nothing.
        const marked = state.legendClass;
        state.colors = colors;
        if (marked >= 0) {
            state.legendClass = -1;
            state.matches = new Int32Array(0);
            state.current = -1;
        }
        // What the search found is found again among the leaves that are left.
        if (TV.applyRemoved(TV.removedLeaves(colors))) {
            if (state.query) showSearch(TV.runSearch(state.query));
            showLayoutControls();
        }
        showLegend();
        if (state.pinned < 0) showPinned(-1);
        TV.minimapChanged();
        TV.drawNow();
        // The neighbours of the selection depend on what the Viewer outlines, which a change of its colours can alter.
        if (marksBefore !== "null" && marksBefore !== JSON.stringify(state.marks)) TV.loadSelection();
    }

    // Ask the Viewer for the colours of the network. A change that comes while one is asked for is asked for after.
    TV.loadColors = async function () {
        if (!state.model) return false;
        if (colorsLoading) {
            colorsAgain = true;
            return false;
        }
        colorsLoading = true;
        try {
            const response = await fetch("/api/tree/colors", { cache: "no-store" });
            if (!response.ok) throw new Error(response.status + " " + response.statusText);
            applyColors(await response.json());
            return true;
        } catch (error) {
            applyColors(null);
            return false;
        } finally {
            colorsLoading = false;
            if (colorsAgain) {
                colorsAgain = false;
                TV.loadColors();
            }
        }
    };

    // ---- The layout ------------------------------------------------------------------------------------------

    // Show in the controls the layout in use, and which forms of it there are: a tree with no branch lengths has no
    // phylogram.
    function showLayoutControls() {
        const model = state.model;
        $("layout-select").value = state.kind;
        const lengths = state.mode === "lengths";
        $("phylogram-btn").setAttribute("aria-pressed", String(lengths));
        $("cladogram-btn").setAttribute("aria-pressed", String(!lengths));
        $("phylogram-btn").disabled = !model.hasLengths;
        $("cladogram-btn").disabled = !model.hasLengths && !lengths;
        $("unfold-btn").disabled = !state.collapsedCount;
        $("hints").textContent = TV.hintText();
    }

    // What the page does when something is folded up or opened.
    TV.onFoldsChanged = function () {
        $("unfold-btn").disabled = !state.collapsedCount;
    };

    function chooseLayout(kind, mode) {
        TV.setLayout(kind, mode);
        showLayoutControls();
        showBanner();
        // The note can change the room the drawing has: fit it to what is left, not to what there was.
        if (state.fitted) TV.fitView();
        TV.drawNow();
    }

    // ---- How the tree fits the alignment and the network -----------------------------------------------------

    // One sentence for each warning the Viewer found. A mismatch is shown, never an error: the tree is drawn all the same.
    function warningText(warning) {
        switch (warning.code) {
            case "alignment_differs":
                return t("This tree was made for the alignment {tree}, but the loaded alignment is {loaded}.", { tree: warning.tree, loaded: warning.loaded });
            case "name_differs":
                return t("This tree was made for {tree}, but the loaded alignment is for {loaded}.", { tree: warning.tree, loaded: warning.loaded });
            case "alignment_older":
                return t("The loaded alignment is older than this tree, so the alignment step of the run that made the tree may not have finished.");
            case "leaves_not_in_network":
                return t("Leaves that are not nodes of this network: {count}. They are drawn faded.", { count: Number(warning.count).toLocaleString() });
            case "network_nodes_not_in_tree":
                return t("Nodes of this network that are not leaves of the tree: {count}.", { count: Number(warning.count).toLocaleString() });
            case "leaves_not_in_alignment":
                return t("Leaves whose sequence is not in the loaded alignment: {count}.", { count: Number(warning.count).toLocaleString() });
            default:
                return t("The tree and the Viewer differ in a way this page has no words for: {code}", { code: String(warning.code) });
        }
    }

    // The note on an unrooted tree drawn from a root, or null. A rectangular tree and a circular one grow from a root,
    // and an unrooted tree has none but the one its file was saved with, so the picture would say what the tree does
    // not. A rooted tree in any layout, and an unrooted one in the unrooted layout, say nothing untrue: no note.
    function rootingNote() {
        const model = state.model;
        if (!model || model.rooted === true || state.kind === "unrooted") return null;
        return t("Drawn from the midpoint root; the tree itself is unrooted.");
    }

    // The banner under the controls: a sentence for each warning the Viewer found, and the note on the rooting when
    // the layout in use calls for it, which the Details panel repeats.
    function showBanner() {
        if (!state.model) return;
        const list = $("warning-list");
        list.replaceChildren();
        const add = (symbol, text, className) => {
            const item = document.createElement("li");
            if (className) item.className = className;
            item.appendChild(symbol);
            const words = document.createElement("span");
            words.textContent = text;
            item.appendChild(words);
            list.appendChild(item);
        };
        for (const warning of state.model.warnings) add(icon("triangle-alert"), warningText(warning));
        const note = rootingNote();
        if (note && !rootingNoteDismissed) {
            add(icon("info"), note, "rooting-note");
            const close = document.createElement("button");
            close.type = "button";
            close.className = "note-close";
            close.title = t("Hide this note");
            close.setAttribute("aria-label", t("Hide this note"));
            close.appendChild(icon("x"));
            close.addEventListener("click", dismissRootingNote);
            list.lastChild.appendChild(close);
        }
        $("banner").classList.toggle("shown", list.children.length > 0);
        const detail = $("details-rooting");
        if (detail) {
            detail.textContent = note || "";
            detail.hidden = !note;
        }
    }

    // The x of the note: take it out of the banner for the rest of this page's life, and give the drawing the room.
    function dismissRootingNote() {
        rootingNoteDismissed = true;
        showBanner();
        if (state.fitted) TV.fitView();
        TV.drawNow();
    }

    // The counts, as the status line says them, and how many nodes the network has selected.
    function summaryText(model) {
        const counts = t("{leaves} leaves, {shown} of them nodes of this network", {
            leaves: model.leafCount.toLocaleString(),
            shown: model.counts.leaves_in_network.toLocaleString(),
        });
        const hidden = state.colors ? state.colors.removedCount : 0;
        const shown = hidden ? counts + " · " + t("{count} hidden in the network, left out", { count: hidden.toLocaleString() }) : counts;
        const selection = state.selection;
        if (!selection.count) return shown;
        const chosen = selection.leaves.length;
        const selected = chosen === selection.count
            ? t("{count} selected in the network", { count: chosen.toLocaleString() })
            : t("{count} selected in the network, {leaves} of them in this tree", {
                count: selection.count.toLocaleString(), leaves: chosen.toLocaleString(),
            });
        return shown + " · " + selected;
    }

    // The status line: what is pinned, else the counts.
    function showPinned(node) {
        if (!state.model) return;
        if (node < 0) {
            $("summary").textContent = summaryText(state.model);
            return;
        }
        const description = TV.describeNode(node);
        $("summary").textContent = [description.title, ...description.lines].join(" · ");
    }

    // ---- The network's selection and focus, and the stability of the splits ------------------------------------

    // The network's click focus follows the leaf that is pinned: pinning a leaf focuses on its node in the network,
    // and unpinning it, or pinning anything else, takes the focus off it. It is asked only when the leaf changes.
    function syncFocus(node) {
        const model = state.model;
        if (!model) return;
        const leaf = node >= 0 && model.childCount[node] === 0 ? model.leafOrdinal[node] : -1;
        const target = leaf >= 0 ? model.viewerNode[leaf] : -1;
        if (target === postedFocus) return;
        postedFocus = target;
        TV.focusNode(target >= 0 ? target : null);
    }

    // The selection changed in the network: the status line says how many nodes, unless it shows a pinned node.
    TV.onSelectionChanged = function () {
        if (state.model && state.pinned < 0) showPinned(-1);
    };

    // Ask the Viewer which nodes are selected and which one has the focus, as a page opened later than the selection
    // must. A selection event that came meanwhile is newer than the answer, and is kept.
    TV.loadSelection = async function () {
        if (!state.model) return false;
        const before = selectionEvents;
        try {
            const response = await fetch("/api/tree/selection", { cache: "no-store" });
            if (!response.ok) throw new Error(response.status + " " + response.statusText);
            const payload = await response.json();
            if (before !== selectionEvents || !state.model || payload.status !== "ok") return false;
            const connected = payload.connected_on ? TV.typedArray(payload.connected, Int32Array, payload.connected_count, "connected") : [];
            TV.applySelection(TV.typedArray(payload.indices, Int32Array, payload.count, "indices"), payload.focus, connected);
            return true;
        } catch (error) {
            return false;
        }
    };

    // Ask the Viewer which nodes are next to the selected ones, which it names while it identifies connected nodes. One
    // request at a time; a change of the selection while one is out asks again, and the answer to the old one is dropped.
    async function loadConnected() {
        if (!state.model) return;
        if (connectedBusy) {
            connectedAgain = true;
            return;
        }
        if (!state.selection.count || (state.marks && !state.marks.connectedOn)) {
            if (state.selection.connected.length) TV.applyConnected([]);
            return;
        }
        connectedBusy = true;
        const stamp = selectionChanges;
        try {
            const response = await fetch("/api/tree/selection", { cache: "no-store" });
            if (!response.ok) throw new Error(response.status + " " + response.statusText);
            const payload = await response.json();
            if (stamp === selectionChanges && state.model && payload.status === "ok") {
                TV.applyConnected(payload.connected_on ? TV.typedArray(payload.connected, Int32Array, payload.connected_count, "connected") : []);
            }
        } catch (error) {
            // The neighbours are left as they were.
        } finally {
            connectedBusy = false;
            if (connectedAgain) {
                connectedAgain = false;
                loadConnected();
            }
        }
    }

    // The switch that shows how stable each split is, with the legend that goes with it once it is on: both only for a
    // tree that says how stable its splits are.
    function showSupportControl(hasTree) {
        const usable = !!(hasTree && state.model && TV.hasSupport(state.model));
        $("support-legend").hidden = !usable;
        $("support-btn").setAttribute("aria-pressed", String(usable && state.showSupport));
        $("support-body").hidden = !(usable && state.showSupport);
    }

    // The legend's ramp, in the palette's five steps.
    function showSupportRamp() {
        const ramp = $("support-ramp");
        ramp.replaceChildren();
        for (const color of state.palette.support) {
            const step = document.createElement("span");
            step.style.backgroundColor = color;
            ramp.appendChild(step);
        }
    }

    // The Details panel: the file, how the run made the tree, and the counts.
    function showDetails(payload, model) {
        const panel = $("details");
        panel.replaceChildren();
        const heading = document.createElement("h2");
        heading.textContent = t("This tree");
        panel.appendChild(heading);

        const addRows = (rows) => {
            const list = document.createElement("dl");
            for (const [name, value] of rows) {
                if (value === undefined || value === null || value === "") continue;
                const term = document.createElement("dt");
                term.textContent = name;
                const definition = document.createElement("dd");
                definition.textContent = String(value);
                list.append(term, definition);
            }
            panel.appendChild(list);
        };
        const addTitle = (text) => {
            const title = document.createElement("div");
            title.className = "group-title";
            title.textContent = text;
            panel.appendChild(title);
        };

        if (model.score && model.threshold) {
            showThresholdDetails(panel, model, addRows, addTitle);
            return;
        }
        const notes = model.notes;
        const kind = model.rooted === true ? t("Rooted") : model.rooted === false ? t("Unrooted (rooted at its midpoint in the file)") : t("Rooting not stated");
        addRows([
            [t("File"), payload.tree_path],
            [t("Alignment"), payload.alignment],
            [t("Saved"), notes.WRITTEN],
            [t("Program"), notes.PROGRAM],
            [t("Sequence set"), notes.SEQUENCE_SET],
            [t("Model"), notes.MODEL],
            [t("Tree method"), notes.TREE_METHOD],
            [t("Kind"), kind],
        ]);
        // The note on drawing an unrooted tree from a root, which showBanner fills in and clears with the layout.
        const rooting = document.createElement("p");
        rooting.className = "note";
        rooting.id = "details-rooting";
        rooting.hidden = true;
        panel.appendChild(rooting);

        addTitle(t("How it fits"));
        const counts = model.counts;
        addRows([
            [t("Leaves"), counts.leaves.toLocaleString()],
            [t("Leaves in this network"), counts.leaves_in_network.toLocaleString()],
            [t("Leaves not in this network"), counts.leaves_not_in_network.toLocaleString()],
            [t("Network nodes not in the tree"), counts.network_nodes_not_in_tree.toLocaleString()],
            [t("Leaves not in the alignment"), counts.leaves_not_in_alignment === undefined ? undefined : counts.leaves_not_in_alignment.toLocaleString()],
        ]);

        const words = [notes.ROOT, notes.BRANCH_LENGTHS, notes.INTERNAL_LABELS].filter(Boolean);
        if (words.length) {
            addTitle(t("About this tree"));
            for (const sentence of words) {
                const paragraph = document.createElement("p");
                paragraph.className = "note";
                paragraph.textContent = sentence;
                panel.appendChild(paragraph);
            }
        }
    }

    // The Details of the tree built from the network's edges: what it is, which score and which edges, and how it falls
    // into groups; the words are the page's, the numbers the Viewer's.
    function showThresholdDetails(panel, model, addRows, addTitle) {
        const info = model.threshold;
        const score = (value) => (value === null || value === undefined ? undefined : String(Number(value.toPrecision(5))));
        const kept = info.kept === "top_k"
            ? t("The best {k} edges of each node, from the kept edges", { k: Number(info.k).toLocaleString() })
            : info.cutoff === null || info.cutoff === undefined
                ? t("All the edges of the network")
                : t("Edges scoring at least {cutoff}", { cutoff: score(info.cutoff) });
        addRows([
            [t("Source"), t("This network's edges (single linkage)")],
            [t("Score"), info.score_kind === "blast" ? t("BLAST score") : t("Similarity")],
            [t("Kept edges"), kept],
            [t("Edges"), info.edges.toLocaleString()],
            [t("Highest score"), score(info.top)],
            [t("Lowest score where groups join"), score(info.bottom)],
            [t("Separate groups"), info.forest ? t("{count} groups share no kept edge", { count: info.groups.toLocaleString() }) : t("One: every node is joined")],
            [t("Built in"), t("{seconds} s", { seconds: String(info.build_seconds) })],
            [t("Kind"), t("Rooted")],
        ]);
        addTitle(t("How it fits"));
        addRows([[t("Leaves"), model.counts.leaves.toLocaleString()]]);
        addTitle(t("About this tree"));
        const words = [
            t("Each branch point is where two groups of nodes first share an edge, at that edge's score. Cutting the tree at the threshold gives the groups of nodes the Viewer shows joined."),
        ];
        if (info.forest) words.push(t("Groups that share no kept edge hang from one root below the lowest score; the broken branches show that their lengths are no score."));
        for (const sentence of words) {
            const paragraph = document.createElement("p");
            paragraph.className = "note";
            paragraph.textContent = sentence;
            panel.appendChild(paragraph);
        }
    }

    // ---- Loading ---------------------------------------------------------------------------------------------

    // Draw the tree of a payload, or say why there is none.
    TV.applyPayload = function (payload) {
        showSources(payload);
        if (payload.status !== "ok") {
            showEmpty(emptyCard(payload));
            return false;
        }
        let model;
        try {
            model = TV.decodeTree(payload);
        } catch (error) {
            showEmpty(emptyCard({ status: "unreadable", tree_path: payload.tree_path, reason: String(error.message || error) }));
            return false;
        }
        state.model = model;
        // Where the threshold slider is, for the cut line of a tree built from the network's scores.
        state.cut = model.threshold && Number.isFinite(model.threshold.current) ? model.threshold.current : null;
        state.colors = null;
        state.removed = null;
        state.legendClass = -1;
        TV.resetFolds();
        // A rooted tree opens rectangular; an unrooted tree, or one that does not say, opens unrooted.
        state.kind = TV.defaultLayoutKind(model);
        state.layout = TV.layoutTree(model, state.kind, "lengths");
        state.mode = state.layout.mode;
        TV.minimapChanged();
        state.hover = state.pinned = -1;
        state.matches = new Int32Array(0);
        state.current = -1;
        state.selection = TV.emptySelection();
        postedFocus = -1;
        $("search-input").value = "";
        $("search-box").classList.remove("has-text");
        show("tree");
        TV.fitView();
        showLayoutControls();
        showDetails(payload, model);
        showBanner();
        showPinned(-1);
        showLegend();
        TV.drawNow();
        TV.loadColors();
        TV.loadSelection();
        return true;
    };

    // Ask the Viewer for a tree, and show it: the one named (it keeps the choice), else the one it has kept, else its default
    // (the guide tree when the loaded alignment has one, else the tree of the network's edges).
    TV.loadTree = async function (source) {
        if (loading) return false;
        loading = true;
        show("loading");
        try {
            const query = typeof source === "string" ? "?source=" + encodeURIComponent(source) : "";
            const response = await fetch("/api/tree/data" + query, { cache: "no-store" });
            if (!response.ok) throw new Error(response.status + " " + response.statusText);
            return TV.applyPayload(await response.json());
        } catch (error) {
            showEmpty(emptyCard({ status: "unreachable", reason: String(error.message || error) }));
            return false;
        } finally {
            loading = false;
        }
    };

    // ---- Search ----------------------------------------------------------------------------------------------

    function showSearch(count) {
        const input = $("search-input");
        $("search-box").classList.toggle("has-text", input.value.length > 0);
        for (const id of ["search-prev", "search-next"]) $(id).disabled = count < 1;
        const label = $("search-count");
        if (!input.value.trim()) label.textContent = "";
        else if (!count) label.textContent = t("No match");
        else if (state.current < 0) label.textContent = t("{count} found", { count: count.toLocaleString() });
        else label.textContent = t("{current} of {count}", { current: (state.current + 1).toLocaleString(), count: count.toLocaleString() });
    }

    function search(value) {
        // What a search finds replaces what the legend marked.
        if (state.legendClass >= 0) {
            state.legendClass = -1;
            showLegend();
        }
        const count = TV.runSearch(value);
        if (count) TV.stepMatch(1); else TV.requestDraw();
        showSearch(count);
    }

    function step(delta) {
        if (TV.stepMatch(delta) >= 0) showSearch(state.matches.length);
    }

    // ---- Theme -----------------------------------------------------------------------------------------------

    // The head script set the starting theme. The choice is kept as ssn_theme, which the other pages keep too, so
    // that they match.
    function toggleTheme() {
        const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        try { localStorage.setItem("ssn_theme", next); } catch (error) {}
    }

    function showThemeButton() {
        const dark = document.documentElement.getAttribute("data-theme") === "dark";
        $("theme-toggle-label").textContent = dark ? t("Light Mode") : t("Dark Mode");
        const themeIcon = $("theme-toggle-icon");
        setIcon(themeIcon, dark ? "sun" : "moon");
        // The canvas draws in the theme's colours, read again whenever it changes.
        state.palette = TV.readPalette();
        if ($("support-ramp")) showSupportRamp();
        TV.minimapChanged();
        TV.requestDraw();
    }

    // ---- The Viewer's event stream ---------------------------------------------------------------------------

    // Connecting tells the Viewer this page is open (it opens no second one). The first message of a connection is
    // an init message that carries no state (the manifest declares initial_state false), which is how a reconnect is noticed.
    function connectEventStream() {
        const badge = $("connection-status");
        eventSource = new EventSource("/api/events?client=tree");
        eventSource.onopen = () => {
            badge.textContent = t("Connected");
            badge.dataset.state = "connected";
        };
        eventSource.onerror = () => {
            badge.textContent = t("Reconnecting...");
            badge.dataset.state = "reconnecting";
            reconnecting = true;
        };
        eventSource.onmessage = (event) => {
            // A Viewer that came back may hold another alignment: ask again.
            if (reconnecting && /^\{"type":\s*"init"/.test(event.data)) {
                reconnecting = false;
                TV.loadTree();
                return;
            }
            // Only the events that matter here are read: the first message of a connection is the Viewer's whole state,
            // and most of what follows is for other pages.
            const kind = /^\{"type":\s*"(tree_colors_changed|alignment_changed|selection_changed|highlight_row|threshold_changed)"/.exec(event.data);
            if (!kind) return;
            if (kind[1] === "threshold_changed") {
                // The Viewer's threshold slider moved: the cut line goes with it, and nothing else is drawn again.
                let value = null;
                try { value = JSON.parse(event.data).value; } catch (error) { return; }
                if (typeof value === "number" && Number.isFinite(value)) {
                    state.cut = value;
                    if (state.model && state.model.score) TV.requestDraw();
                }
                return;
            }
            if (kind[1] === "alignment_changed") {
                TV.loadTree();
                return;
            }
            if (kind[1] === "selection_changed" || kind[1] === "highlight_row") {
                let message = null;
                try { message = JSON.parse(event.data); } catch (error) { return; }
                if (kind[1] === "selection_changed") {
                    selectionEvents++;
                    selectionChanges++;
                    if (Array.isArray(message.indices)) {
                        TV.applySelection(message.indices);
                        loadConnected();
                    }
                } else {
                    selectionEvents++;
                    TV.applyFocus(message.index);
                }
                return;
            }
            let version = null;
            try { version = JSON.parse(event.data).version; } catch (error) {}
            if (!state.colors || state.colors.version !== version) TV.loadColors();
        };
    }

    // ---- Start -----------------------------------------------------------------------------------------------

    function copyPath() {
        const path = $("empty-path").textContent;
        const done = () => {
            const label = $("empty-copy").querySelector("span");
            label.textContent = t("Copied");
            clearTimeout(copiedTimer);
            copiedTimer = setTimeout(() => { label.textContent = t("Copy the path"); }, 1500);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(path).then(done, () => {});
    }

    document.addEventListener("DOMContentLoaded", () => {
        TV.canvas = $("tree-canvas");
        state.palette = TV.readPalette();
        TV.attachNavigation(TV.canvas, { tooltip: $("tooltip"), onPin: (node) => { showPinned(node); syncFocus(node); } });
        showSupportRamp();
        $("hints").textContent = TV.hintText();
        show("loading");

        TV.attachMinimap($("minimap"));
        $("fit-btn").addEventListener("click", () => TV.fitView());
        $("source-select").addEventListener("change", (event) => chooseSource(event.target.value));
        $("layout-select").addEventListener("change", (event) => chooseLayout(event.target.value, state.mode));
        $("phylogram-btn").addEventListener("click", () => chooseLayout(state.kind, "lengths"));
        $("cladogram-btn").addEventListener("click", () => chooseLayout(state.kind, "cladogram"));
        $("unfold-btn").addEventListener("click", () => TV.openAll());
        $("support-btn").addEventListener("click", () => {
            state.showSupport = !state.showSupport;
            showSupportControl(true);
            TV.requestDraw();
        });
        $("overview-btn").addEventListener("click", () => {
            overviewOn = !overviewOn;
            $("overview-btn").setAttribute("aria-pressed", String(overviewOn));
            $("minimap").hidden = !overviewOn;
            if (overviewOn) TV.requestDraw();
        });
        $("details-btn").addEventListener("click", () => $("details").classList.toggle("shown"));
        $("legend-toggle").addEventListener("click", () => {
            legendOpen = !legendOpen;
            showLegend();
        });
        $("empty-retry").addEventListener("click", () => TV.loadTree());
        $("empty-copy").addEventListener("click", copyPath);
        $("search-prev").addEventListener("click", () => step(-1));
        $("search-next").addEventListener("click", () => step(1));
        $("search-clear").addEventListener("click", () => {
            $("search-input").value = "";
            search("");
            $("search-input").focus();
        });
        $("search-input").addEventListener("input", (event) => {
            const value = event.target.value;
            $("search-box").classList.toggle("has-text", value.length > 0);
            clearTimeout(searchTimer);
            searchTimer = setTimeout(() => search(value), 150);
        });
        $("search-input").addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                clearTimeout(searchTimer);
                if (state.query !== event.target.value.trim()) search(event.target.value); else step(event.shiftKey ? -1 : 1);
            } else if (event.key === "Escape") {
                event.target.value = "";
                search("");
                TV.canvas.focus();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey && !$("search-input").disabled
                && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
                event.preventDefault();
                $("search-input").focus();
            }
        });

        showThemeButton();
        $("theme-toggle-btn").addEventListener("click", toggleTheme);
        new MutationObserver(showThemeButton).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
        // A choice made on another page while this one is open applies here at once.
        window.addEventListener("storage", (event) => {
            if (event.key === "ssn_theme" && (event.newValue === "dark" || event.newValue === "light")) {
                document.documentElement.setAttribute("data-theme", event.newValue);
            }
        });

        new ResizeObserver(() => {
            if (!state.layout) return;
            if (state.fitted) TV.fitView(); else TV.requestDraw();
        }).observe($("stage"));

        connectEventStream();
        TV.loadTree();
    });
})(window.TreeView);
