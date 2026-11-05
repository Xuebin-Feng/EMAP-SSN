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
                    more: t("Run the Embedding MSA tool again to make it, or put the tree there, and press Check again."),
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

    function showWarnings(warnings) {
        const list = $("warning-list");
        list.replaceChildren();
        for (const warning of warnings) {
            const item = document.createElement("li");
            item.appendChild(icon("triangle-alert"));
            const text = document.createElement("span");
            text.textContent = warningText(warning);
            item.appendChild(text);
            list.appendChild(item);
        }
        $("banner").classList.toggle("shown", warnings.length > 0);
    }

    // The counts, as the status line says them.
    function summaryText(model) {
        return t("{leaves} leaves, {shown} of them nodes of this network", {
            leaves: model.leafCount.toLocaleString(),
            shown: model.counts.leaves_in_network.toLocaleString(),
        });
    }

    // The status line: what is selected, else the counts.
    function showPinned(node) {
        if (!state.model) return;
        if (node < 0) {
            $("summary").textContent = summaryText(state.model);
            return;
        }
        const description = TV.describeNode(node);
        $("summary").textContent = [description.title, ...description.lines].join(" · ");
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

    // ---- Loading ---------------------------------------------------------------------------------------------

    // Draw the tree of a payload, or say why there is none.
    TV.applyPayload = function (payload) {
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
        TV.resetFolds();
        // A rooted tree is shown from its root, an unrooted one (neighbor-joining's) from its middle.
        state.kind = TV.defaultLayoutKind(model);
        state.layout = TV.layoutTree(model, state.kind, "lengths");
        state.mode = state.layout.mode;
        TV.minimapChanged();
        state.hover = state.pinned = -1;
        state.matches = new Int32Array(0);
        state.current = -1;
        $("search-input").value = "";
        $("search-box").classList.remove("has-text");
        show("tree");
        TV.fitView();
        showLayoutControls();
        showWarnings(model.warnings);
        showDetails(payload, model);
        showPinned(-1);
        TV.drawNow();
        return true;
    };

    // Ask the Viewer for the tree of the loaded alignment, and show it.
    TV.loadTree = async function () {
        if (loading) return false;
        loading = true;
        show("loading");
        try {
            const response = await fetch("/api/tree/data", { cache: "no-store" });
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
        TV.minimapChanged();
        TV.requestDraw();
    }

    // ---- The Viewer's event stream ---------------------------------------------------------------------------

    // Connecting tells the Viewer this page is open (it opens no second one). The first message of a connection is
    // the Viewer's whole state, which this page does not read, so it is not parsed.
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
            }
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
        TV.attachNavigation(TV.canvas, { tooltip: $("tooltip"), onPin: showPinned });
        $("hints").textContent = TV.hintText();
        show("loading");

        TV.attachMinimap($("minimap"));
        $("fit-btn").addEventListener("click", () => TV.fitView());
        $("layout-select").addEventListener("change", (event) => chooseLayout(event.target.value, state.mode));
        $("phylogram-btn").addEventListener("click", () => chooseLayout(state.kind, "lengths"));
        $("cladogram-btn").addEventListener("click", () => chooseLayout(state.kind, "cladogram"));
        $("unfold-btn").addEventListener("click", () => TV.openAll());
        $("overview-btn").addEventListener("click", () => {
            overviewOn = !overviewOn;
            $("overview-btn").setAttribute("aria-pressed", String(overviewOn));
            $("minimap").hidden = !overviewOn;
            if (overviewOn) TV.requestDraw();
        });
        $("details-btn").addEventListener("click", () => $("details").classList.toggle("shown"));
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
