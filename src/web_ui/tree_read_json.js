/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copied from Cladewise (the same author's standalone tree viewer, same licence), js/tree_json.js at its commit ba06967. What differs
 * here: every text the reader says goes through t() (page_text.js), so the Viewer shows it in its language; the strings
 * that are names of notations (NHX, Jplace, FigTree) are kept as they are.
 *
 * Cladewise's JSON readers: Nextstrain Auspice v2, plain nested trees, and Jplace, as the same tree arrays the Newick
 * reader gives (tree_parse.js).
 *
 *   Auspice v2: { "version": "v2", "meta": {...}, "tree": { "name", "node_attrs": { "div": ... }, "children": [...] } }.
 *     A node's `div` is its distance from the root, so a branch length is the node's div less its parent's. A node_attrs
 *     entry that is a value (or a { "value": ... }) is an annotation; support, bootstrap and posterior are support. The
 *     `tree` may also be a list of such trees (they are chosen from). Internal nodes called NODE_0000012 are unnamed.
 *   Nested: { "name", "length" (or branch_length, branchLength), "children": [...] }, the root alone or a list of roots.
 *     Support is the member support, bootstrap, posterior, confidence or prob; every other value is an annotation.
 *   Jplace: { "tree": "((A:0.1{0},B:0.2{1}):0.3{2},C:0.4{3}){4};", "placements": [...], ... }. The tree string is
 *     Newick with {n} edge numbers (read and ignored); the placements are ignored.
 *
 * JSON.parse is not recursive in the engines that matter, and the walk here keeps its open nodes in a list, so a tree
 * 50,000 deep loads. A syntax error says its line and column; a tree that is not what it should be says the path of
 * the member (tree.children[3].children[0]).
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const TreeParseError = TV.TreeParseError;
    const NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
    const SUPPORT_MEMBERS = ["posterior", "support", "bootstrap", "confidence", "prob"];
    const AUSPICE_SUPPORT = ["posterior", "support", "bootstrap"];
    const RESERVED = new Set(["name", "label", "children", "length", "branch_length", "branchLength", "rooted"]);

    const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

    function numberOf(value) {
        if (typeof value === "number") return Number.isFinite(value) ? value : NaN;
        if (typeof value === "string" && NUMBER.test(value.trim())) return Number(value.trim());
        if (isObject(value) && "value" in value) return numberOf(value.value);
        return NaN;
    }

    function textOf(value) {
        if (isObject(value) && "value" in value) return textOf(value.value);
        if (typeof value === "string") return value.length > 300 ? value.slice(0, 300) + "…" : value;
        if (typeof value === "number" || typeof value === "boolean") return String(value);
        return null;
    }

    // The error JSON.parse threw, as a TreeParseError with the line and column of the mistake.
    function syntaxError(error, text) {
        const message = String(error && error.message || error);
        const at = /at position (\d+)/.exec(message);
        const reason = t("The JSON is not well formed: {problem}", { problem: message.replace(/\s*\(line \d+ column \d+\)\s*$/, "").replace(/\s+in JSON at position \d+/, "").replace(/\s+at position \d+/, "") });
        if (at) return new TreeParseError(reason, text, Number(at[1]));
        const lc = /line (\d+) column (\d+)/.exec(message);
        if (lc) {
            const error2 = new TreeParseError(reason, null, 0);
            error2.message = reason + " (" + t("line {line}", { line: lc[1] }) + ", " + t("column {column}", { column: lc[2] }) + ")";
            error2.line = Number(lc[1]);
            error2.column = Number(lc[2]);
            return error2;
        }
        return new TreeParseError(reason, null, 0);
    }

    // The trees of a JSON text: { format, entries, tree(index) }, as TV.readTreeText answers. `lead` is where the text's
    // first character is.
    TV.readJsonText = function (text, lead, options) {
        const maxLeaves = (options && options.maxLeaves) || TV.MAX_LEAVES;
        let data;
        try {
            data = JSON.parse(text);
        } catch (error) {
            throw syntaxError(error, text);
        }
        const nothing = () => new TreeParseError(t("The JSON holds no tree: it needs a \"tree\" (a nested object, or a Newick string as Jplace has), or a \"children\" list"), null, 0);

        // Jplace: the tree is a Newick string.
        if (isObject(data) && typeof data.tree === "string") {
            const newick = data.tree;
            if (!newick.trim()) throw new TreeParseError(t("The Jplace \"tree\" string is empty"), null, 0, "tree");
            return reader("Jplace", [{ name: "" }], () => {
                try {
                    const tree = TV.readNewickRange(newick, 0, newick.length, "", null, maxLeaves);
                    tree.dialects = tree.dialects.concat(tree.dialects.includes("Jplace edge numbers") ? [] : ["Jplace edge numbers"]);
                    return tree;
                } catch (error) {
                    if (error instanceof TreeParseError) throw new TreeParseError(t("{reason}, in the \"tree\" string", { reason: error.reason }), newick, error.offset);
                    throw error;
                }
            });
        }

        let roots;
        let format = "JSON (nested)";
        let auspice = false;
        if (isObject(data) && (isObject(data.tree) || Array.isArray(data.tree))) {
            roots = Array.isArray(data.tree) ? data.tree : [data.tree];
            const first = roots.find(isObject);
            auspice = (typeof data.version === "string" && /^v\d/i.test(data.version)) || isObject(data.meta) || !!(first && isObject(first.node_attrs));
            if (auspice) format = "JSON (Auspice)";
        } else if (Array.isArray(data)) {
            roots = data;
        } else if (isObject(data) && (Array.isArray(data.children) || typeof data.name === "string")) {
            roots = [data];
            auspice = isObject(data.node_attrs);
            if (auspice) format = "JSON (Auspice)";
        } else {
            throw nothing();
        }
        if (!roots.length) throw nothing();
        roots.forEach((root, index) => {
            const where = roots.length > 1 || Array.isArray(data) ? "[" + index + "]" : "tree";
            if (!isObject(root)) throw new TreeParseError(t("A tree is not an object"), null, 0, where);
            if (!Array.isArray(root.children) && typeof root.name !== "string") throw nothing();
        });
        const rootedFlag = isObject(data) && typeof data.rooted === "boolean" ? data.rooted : null;
        const entries = roots.map((root, index) => ({ name: typeof root.name === "string" && root.name && roots.length > 1 ? root.name : "", index }));
        return reader(format, entries, (entry) => buildNested(roots[entry.index], auspice, rootedFlag, maxLeaves, Array.isArray(data) || roots.length > 1 ? "[" + entry.index + "]" : "tree"));
    };

    function reader(format, entries, build) {
        let cached = null;
        let cachedIndex = -1;
        return {
            format,
            entries,
            tree(index) {
                if (!(index >= 0 && index < entries.length)) throw new RangeError("There is no tree " + index);
                if (cachedIndex !== index) {
                    cached = null;
                    cached = build(entries[index]);
                    cachedIndex = index;
                }
                return cached;
            },
        };
    }

    // One nested tree, read without recursion: the nodes being read are a list of frames, a node is made when all its
    // children are, so the children come before it. `base` is the path of the tree, for an error.
    function buildNested(root, auspice, rootedFlag, maxLeaves, base) {
        const built = new TV.Builder(maxLeaves);
        const frames = [{ node: root, index: 0, position: 0, ids: [] }];
        const rooted = typeof root.rooted === "boolean" ? root.rooted : rootedFlag;
        let rootChildren = 0;
        const supportKeys = auspice ? AUSPICE_SUPPORT : SUPPORT_MEMBERS;

        // Where the top frame is, as a path, for an error: the last few steps.
        const where = () => {
            const steps = frames.slice(1).map((frame) => ".children[" + frame.index + "]");
            return base + (steps.length > 5 ? ".children[…]" + steps.slice(-4).join("") : steps.join(""));
        };
        const fail = (reason) => { throw new TreeParseError(reason, null, 0, where()); };
        const divOf = (node) => (isObject(node.node_attrs) ? numberOf(node.node_attrs.div) : NaN);

        while (frames.length) {
            const frame = frames[frames.length - 1];
            const node = frame.node;
            const children = node.children;
            if (children !== undefined && children !== null && !Array.isArray(children)) fail(t("\"children\" is not a list"));
            const count = children ? children.length : 0;
            if (frame.position < count) {
                const child = children[frame.position];
                frames.push({ node: isObject(child) ? child : {}, index: frame.position, position: 0, ids: [] });
                frame.position++;
                if (!isObject(child)) fail(t("A child is not an object"));
                continue;
            }

            // Every child is made: make this node.
            const named = typeof node.name === "string" ? node.name : (typeof node.label === "string" ? node.label : (typeof node.name === "number" ? String(node.name) : ""));
            let id;
            if (count === 0) {
                id = built.leaf(named, fail);
            } else {
                id = built.internal(frame.ids, fail);
                if (named && !(auspice && /^NODE_\d+$/.test(named))) built.setName(id, named, fail);
            }

            // The branch above it: an Auspice node's div less its parent's, else a length member.
            const above = frames.length > 1 ? frames[frames.length - 2].node : null;
            let length = NaN;
            if (auspice && above) {
                const div = divOf(node);
                const parentDiv = divOf(above);
                if (Number.isFinite(div) && Number.isFinite(parentDiv)) length = div - parentDiv;
            }
            if (!Number.isFinite(length) && above) {
                for (const key of ["length", "branch_length", "branchLength"]) {
                    if (node[key] === undefined || node[key] === null) continue;
                    length = numberOf(node[key]);
                    if (Number.isNaN(length)) fail(t("A branch length is not a number: {value}", { value: JSON.stringify(String(node[key]).slice(0, 40)) }));
                    break;
                }
            }
            if (Number.isFinite(length)) built.setLength(id, length);

            // Support, and every other value as an annotation.
            const members = auspice && isObject(node.node_attrs) ? node.node_attrs : (auspice ? {} : node);
            let supported = false;
            for (const key of Object.keys(members)) {
                if (RESERVED.has(key) || key === "div") continue;
                const value = members[key];
                if (!supported && supportKeys.includes(key.toLowerCase())) {
                    const number = numberOf(value);
                    if (!Number.isNaN(number)) {
                        built.setSupport(id, number, 0, "");
                        supported = true;
                        continue;
                    }
                }
                const shown = textOf(value);
                if (shown !== null && shown !== "") built.annotate(id, key, shown);
            }

            frames.pop();
            if (frames.length) frames[frames.length - 1].ids.push(id); else rootChildren = count;
        }
        return TV.finishTree(built, { name: "", rooted: rooted === true || rooted === false ? rooted : null, rootChildren });
    }
})(window.TreeView);
