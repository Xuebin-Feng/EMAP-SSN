/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copied from Cladewise (the same author's standalone tree viewer, same licence), js/tree_parse.js at its commit ba06967. What differs
 * here: every text the reader says goes through t() (page_text.js), so the Viewer shows it in its language; the strings
 * that are names of notations (NHX, Jplace, FigTree) are kept as they are.
 *
 * Cladewise's tree reader: tree text to the arrays tree_model.js takes. This file reads Newick (with its NHX, paired
 * support, RAxML and Jplace dialects) and NEXUS (with BEAST, MrBayes, TreeAnnotator and FigTree annotations), and sends
 * a text that is XML (tree_xml.js: PhyloXML, NeXML) or JSON (tree_json.js: Auspice, nested, Jplace) to its reader. The
 * format is found from the content, never from a file's name.
 *
 * TV.readTreeText(text) finds the trees of a text without reading them (a NEXUS TREES block holds any number, a
 * Newick file one or more, each ending in ";"), and answers { format, entries: [{ name }], tree(index) }: tree(index)
 * reads one of them. Every format answers the same kind of tree, { name, rooted, rootStated, hasLengths, hasSupport,
 * parent, length, support, labels, cladeNames, warnings, nodeCount, leafCount }, and may add:
 *   supportSets: [{ name, support (Int8Array), raw (Float32Array), scale }] when the file gives more than one support
 *     value to a node (IQ-TREE's 85.2/97, PhyloXML's confidence types); `support` is the one the page starts with.
 *   annotations: a Map from a node to [key, value, key, value, ...] (NHX, [&...] and the like), or null.
 *   nodeColors: a Map from a node to [r, g, b] (FigTree's !color, PhyloXML's color), or null.
 *   dialects: the names of the dialects the tree used, which the Details panel lists.
 * TV.buildModel turns a tree into the page's model:
 *
 *   The nodes are numbered in post-order: a leaf when it is read and an internal node when its ")" closes it, so every
 *   child comes before its parent, the root is the last node, and the leaves, in node order, run top to bottom as the
 *   file has them. parent: Int32Array (-1 for the root). length: Float32Array, the branch length to the parent (0 for
 *   the root and for a node the file gives none; hasLengths says whether any was given). support: Int8Array, a
 *   percentage from 0 to 100 or -1. labels: the leaf names. cladeNames: a Map from an internal node to its name, or null.
 *
 * What a label means. An unquoted underscore stays an underscore, as Biopython, ETE and iTOL read it (the strict NEXUS
 * rule, a space, would turn the accession XP_007399765.1 into XP 007399765.1), and a quoted label is kept as it is, its
 * doubled quotes made one. A label after ")" that is a number is the node's support; any other is the clade's
 * name. A metacomment [&key=value,...] after a node can give support too, as posterior, support or bootstrap. A tree
 * whose support values are all from 0 to 1 has them scaled to percent; others are kept as they are, from 0 to 100.
 *
 * Rooting. [&R] says the tree is rooted and [&U] unrooted. With neither, a root with three or more children is unrooted
 * and one with fewer is rooted.
 *
 * Nothing here recurses: the open nodes are a stack, so a tree as deep as it is long (a 50,000-leaf caterpillar) is no
 * trouble. The limits are those of EMAP-SSN's guide-tree reader. An error is a TreeParseError that says what is wrong,
 * and its line and column (counted from 1) in the text.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    // The most a tree file may be, hold and nest. A tree of 44,000 leaves is about 5 MB; the limits keep a file that is
    // not a tree from taking the page's time or memory. The leaf limit also bounds the internal nodes, so a flood of
    // "(" is refused before it fills memory.
    TV.MAX_FILE_BYTES = 128 * 1024 * 1024;
    TV.MAX_LEAVES = 250000;
    TV.MAX_LABEL_CHARACTERS = 65536;

    // An error of the text. It says where: the line and column of `offset` in `text` (counted from 1), and, for XML and
    // JSON, the path of the element or member (`where`) as well. A JSON structure error has a path and no text position.
    class TreeParseError extends Error {
        constructor(reason, text, offset, where) {
            const at = typeof text === "string" ? lineAndColumn(text, offset) : null;
            const place = [];
            if (at) place.push(t("line {line}", { line: at.line }), t("column {column}", { column: at.column }));
            if (where) place.push(t("at {where}", { where }));
            super(place.length ? reason + " (" + place.join(", ") + ")" : reason);
            this.name = "TreeParseError";
            this.reason = reason;
            this.line = at ? at.line : null;
            this.column = at ? at.column : null;
            this.offset = offset;
            this.where = where || "";
        }
    }
    TV.TreeParseError = TreeParseError;

    // The line and column, from 1, of the character at `offset`. A line ends at "\n", "\r\n" or a lone "\r".
    function lineAndColumn(text, offset) {
        const stop = Math.max(0, Math.min(offset, text.length));
        let line = 1;
        let lineStart = 0;
        for (let i = 0; i < stop; i++) {
            const c = text.charCodeAt(i);
            if (c === 10 || (c === 13 && text.charCodeAt(i + 1) !== 10)) {
                line++;
                lineStart = i + 1;
            }
        }
        return { line, column: stop - lineStart + 1 };
    }
    TV.lineAndColumn = lineAndColumn;

    // ---- Tokens --------------------------------------------------------------------------------------------------

    // One token of the text: white space, a quoted word, the start of a comment, punctuation, a Jplace edge number {n}, or
    // a word. A quote that never closes, or a stray "]", matches none of them, and the reader says so. A word may hold a
    // brace, but a {n} ends it: Jplace writes 0.1{3}.
    const TOKEN = /(\s+)|('[^']*(?:''[^']*)*')|(\[)|([(),:;=])|(\{\d+\})|((?:[^\s()\[\],:;='{]|\{(?!\d+\}))+)/y;
    const BRACKET = /[\[\]]/g;

    // The text after the comment that opens at `open` (its "["), as the offset just past its "]"; -1 when it never closes.
    // A comment may hold comments.
    function endOfComment(text, open) {
        BRACKET.lastIndex = open + 1;
        let depth = 1;
        for (;;) {
            const found = BRACKET.exec(text);
            if (!found) return -1;
            depth += found[0] === "[" ? 1 : -1;
            if (!depth) return found.index + 1;
        }
    }

    // A reader of the tokens of text[start, end): next() puts the next token's kind in `kind` ("w" a word, "q" a quoted
    // word, "p" one of ( ) , : ; =, "j" a Jplace edge number, "c" a metacomment such as &R or a comment that is a number,
    // as RAxML writes its support, "end" for none), its text in `value` and its offset in `at`. Other comments are skipped.
    function lexer(text, start, end) {
        const lex = { kind: "end", value: "", at: start, pos: start };
        lex.next = function () {
            for (;;) {
                if (lex.pos >= end) {
                    lex.kind = "end";
                    lex.value = "";
                    lex.at = end;
                    return;
                }
                TOKEN.lastIndex = lex.pos;
                const found = TOKEN.exec(text);
                if (!found) {
                    const c = text[lex.pos];
                    throw new TreeParseError(c === "'" ? t("A quote that never closes") : c === "]" ? t("A ] with no [ before it")
                        : t("The text cannot be read here"), text, lex.pos);
                }
                const at = lex.pos;
                lex.pos = TOKEN.lastIndex;
                if (found[1] !== undefined) continue;
                lex.at = at;
                if (found[6] !== undefined) {
                    lex.kind = "w";
                    lex.value = found[6];
                    return;
                }
                if (found[5] !== undefined) {
                    lex.kind = "j";
                    lex.value = found[5];
                    return;
                }
                if (found[4] !== undefined) {
                    lex.kind = "p";
                    lex.value = found[4];
                    return;
                }
                if (found[2] !== undefined) {
                    lex.kind = "q";
                    lex.value = found[2].slice(1, -1).replace(/''/g, "'");
                    return;
                }
                const close = endOfComment(text, at);
                if (close < 0) throw new TreeParseError(t("A comment in [ ] never closes"), text, at);
                lex.pos = close;
                const comment = text.slice(at + 1, close - 1).trim();
                if (comment[0] === "&" || NUMBER.test(comment)) {
                    lex.kind = "c";
                    lex.value = comment;
                    return;
                }
            }
        };
        return lex;
    }

    // The offset just past the ";" that ends the command or tree that begins at `from`, skipping quotes and comments; or
    // `end` (and found false) when there is none. Reads no tokens: a quick scan for the three characters that matter.
    const SKIP = /[';\[]/g;
    function skipToSemicolon(text, from, end) {
        let pos = from;
        for (;;) {
            SKIP.lastIndex = pos;
            const found = SKIP.exec(text);
            if (!found || found.index >= end) return { at: end, found: false };
            const c = found[0];
            if (c === ";") return { at: found.index + 1, found: true };
            if (c === "'") {
                // A quote closes at the next one that is not doubled.
                let close = found.index + 1;
                for (;;) {
                    close = text.indexOf("'", close);
                    if (close < 0) throw new TreeParseError(t("A quote that never closes"), text, found.index);
                    if (text[close + 1] !== "'") break;
                    close += 2;
                }
                pos = close + 1;
            } else {
                const close = endOfComment(text, found.index);
                if (close < 0) throw new TreeParseError(t("A comment in [ ] never closes"), text, found.index);
                pos = close;
            }
        }
    }

    // ---- What a tree is, once it is read ----------------------------------------------------------------------------

    const NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
    // The keys of a metacomment that give a node's support, best first.
    const SUPPORT_KEYS = ["posterior", "support", "bootstrap", "prob"];
    // The most annotations kept for a node, and the longest value kept: a file cannot fill the page with them.
    const MAX_ANNOTATIONS = 40;
    const MAX_VALUE_CHARACTERS = 300;
    // The most support values a node can have (IQ-TREE writes up to three).
    const MAX_SUPPORT_SLOTS = 8;
    const ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth"];

    // A growing typed array.
    function grow(array, size) {
        const bigger = new array.constructor(Math.max(size, array.length * 2));
        bigger.set(array);
        return bigger;
    }

    // The RGB of a colour a file gives for a node: #rrggbb or #rgb, FigTree's older signed integer (ARGB), or a CSS colour
    // name; null when it is none of them.
    function colourValue(value) {
        const word = String(value).trim().replace(/^"|"$/g, "");
        let match = /^#([0-9a-f]{6})$/i.exec(word);
        if (match) return [parseInt(match[1].slice(0, 2), 16), parseInt(match[1].slice(2, 4), 16), parseInt(match[1].slice(4), 16)];
        match = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(word);
        if (match) return [parseInt(match[1] + match[1], 16), parseInt(match[2] + match[2], 16), parseInt(match[3] + match[3], 16)];
        if (/^-?\d+$/.test(word)) {
            const number = Number(word) >>> 0;
            return [(number >> 16) & 255, (number >> 8) & 255, number & 255];
        }
        return TV.parseColor ? TV.parseColor(word) : null;
    }
    TV.colourValue = colourValue;

    // The key=value pairs of a metacomment's text (after its &, or after &&NHX:), split at `separator` where it is outside
    // braces and quotes. A value in quotes loses them; a long value is cut.
    function splitPairs(body, separator) {
        const pairs = [];
        const add = (pair) => {
            const equals = pair.indexOf("=");
            if (equals <= 0) return;
            let value = pair.slice(equals + 1).trim();
            if (value.length >= 2 && value[0] === '"' && value[value.length - 1] === '"') value = value.slice(1, -1);
            if (value.length > MAX_VALUE_CHARACTERS) value = value.slice(0, MAX_VALUE_CHARACTERS) + "…";
            pairs.push([pair.slice(0, equals).trim(), value]);
        };
        if (body.indexOf("{") < 0 && body.indexOf('"') < 0) {
            for (const pair of body.split(separator)) add(pair);
            return pairs;
        }
        let depth = 0;
        let quoted = false;
        let start = 0;
        for (let i = 0; i <= body.length; i++) {
            const c = body[i];
            if (c === '"') quoted = !quoted;
            else if (!quoted && c === "{") depth++;
            else if (!quoted && c === "}") depth--;
            else if (i === body.length || (!quoted && depth === 0 && c === separator)) {
                add(body.slice(start, i));
                start = i + 1;
            }
        }
        return pairs;
    }

    // What is built while a tree is read, whatever its format: the arrays of the nodes (in post-order), their support
    // values in as many slots as the file gives, annotations, colours and the dialects used. finishTree turns it into the
    // tree the page takes. The Newick reader keeps its own arrays and ends here too; the XML and JSON readers use a
    // Builder from the start.
    class Builder {
        constructor(maxLeaves) {
            this.maxLeaves = maxLeaves;
            this.maxNodes = 2 * maxLeaves;
            this.nodes = 0;
            this.parent = new Int32Array(1024);
            this.length = new Float32Array(1024);
            this.slots = [new Float32Array(1024).fill(-1)];
            this.slotNames = [""];
            this.labels = [];
            this.cladeNames = new Map();
            this.annotations = new Map();
            this.colors = new Map();
            this.dialects = new Set();
            this.hasLengths = false;
            this.unnamed = 0;
        }

        // A new node, whose parent is not known yet. `fail(reason)` throws the reader's own error.
        newNode(fail) {
            if (this.nodes >= this.maxNodes) fail(t("The tree has more than {count} nodes", { count: this.maxNodes.toLocaleString("en-US") }));
            if (this.nodes === this.parent.length) {
                this.parent = grow(this.parent, this.nodes + 1);
                this.length = grow(this.length, this.nodes + 1);
                this.slots = this.slots.map((slot) => {
                    const bigger = new Float32Array(this.parent.length).fill(-1);
                    bigger.set(slot);
                    return bigger;
                });
            }
            this.parent[this.nodes] = -1;
            return this.nodes++;
        }

        leaf(label, fail) {
            if (this.labels.length >= this.maxLeaves) fail(t("The tree has more than {count} leaves", { count: this.maxLeaves.toLocaleString("en-US") }));
            let named = label;
            if (!named) {
                named = "unnamed_" + (this.labels.length + 1);
                this.unnamed++;
            }
            if (named.length > TV.MAX_LABEL_CHARACTERS) fail(t("A name is longer than {count} characters", { count: TV.MAX_LABEL_CHARACTERS.toLocaleString("en-US") }));
            const node = this.newNode(fail);
            this.labels.push(named);
            return node;
        }

        // An internal node over `children` (node numbers, all already made).
        internal(children, fail) {
            const node = this.newNode(fail);
            for (const child of children) this.parent[child] = node;
            return node;
        }

        setLength(node, value) {
            this.length[node] = value;
            this.hasLengths = true;
        }

        // Support value `value` in slot `slot` (named `name`, when it has not been).
        setSupport(node, value, slot, name) {
            if (!Number.isFinite(value) || slot >= MAX_SUPPORT_SLOTS) return;
            while (this.slots.length <= slot) {
                this.slots.push(new Float32Array(this.parent.length).fill(-1));
                this.slotNames.push("");
            }
            if (name && !this.slotNames[slot]) this.slotNames[slot] = name;
            this.slots[slot][node] = value;
        }

        setName(node, name, fail) {
            if (name.length > TV.MAX_LABEL_CHARACTERS) fail(t("A name is longer than {count} characters", { count: TV.MAX_LABEL_CHARACTERS.toLocaleString("en-US") }));
            this.cladeNames.set(node, name);
        }

        annotate(node, key, value) {
            let list = this.annotations.get(node);
            if (!list) this.annotations.set(node, list = []);
            if (list.length < 2 * MAX_ANNOTATIONS) list.push(key, value);
        }
    }

    // The tree the page takes, from what was built: `built` is a Builder-shaped object (the Newick reader passes its own
    // arrays in one), `info` is { name, rooted (true, false or null for "the root decides"), rootChildren }.
    function finishTree(built, info) {
        const nodes = built.nodes;
        const labels = built.labels;
        const warnings = [];
        const rootStated = info.rooted !== null && info.rooted !== undefined;
        const rooted = rootStated ? info.rooted : info.rootChildren < 3;

        // Support: values from 0 to 1 are shares, scaled to percent; others are percentages from 0 to 100. Each slot is
        // scaled by its own largest value.
        const sets = [];
        let ignored = 0;
        // A leaf has no split to be sure of: a support value some file gives it (NHX's B on a leaf) is not kept.
        const inner = new Uint8Array(nodes);
        for (let i = 0; i < nodes; i++) if (built.parent[i] >= 0) inner[built.parent[i]] = 1;
        built.slots.forEach((values, index) => {
            for (let i = 0; i < nodes; i++) if (!inner[i]) values[i] = -1;
            let max = -Infinity;
            let any = false;
            for (let i = 0; i < nodes; i++) {
                if (values[i] >= 0) {
                    any = true;
                    if (values[i] > max) max = values[i];
                }
            }
            if (!any) return;
            const scale = max <= 1 ? 100 : 1;
            const support = new Int8Array(nodes).fill(-1);
            let kept = 0;
            for (let i = 0; i < nodes; i++) {
                const value = values[i];
                if (value < 0) continue;
                const percent = value * scale;
                if (percent <= 100) { support[i] = Math.round(percent); kept++; } else ignored++;
            }
            if (kept) sets.push({ name: built.slotNames[index], slot: index, support, raw: values.slice(0, nodes), scale });
        });
        if (ignored) warnings.push({ code: "support_out_of_range", count: ignored });
        // A name for a set that the file did not name.
        for (const set of sets) {
            if (!set.name) set.name = sets.length > 1 ? ORDINALS[set.slot] + " value" : "Support";
        }
        const support = sets.length ? sets[0].support : new Int8Array(nodes).fill(-1);

        if (built.unnamed) warnings.push({ code: "unnamed_leaves", count: built.unnamed });
        // Warnings about the tree: names that repeat, and branches of negative length.
        const seen = new Set();
        let repeated = 0;
        for (const leaf of labels) {
            if (seen.has(leaf)) repeated++; else seen.add(leaf);
        }
        if (repeated) warnings.push({ code: "duplicate_names", count: repeated });
        let negative = 0;
        for (let i = 0; i < nodes; i++) if (built.length[i] < 0) negative++;
        if (negative) warnings.push({ code: "negative_lengths", count: negative });
        for (const warning of info.warnings || []) warnings.unshift(warning);

        return {
            name: info.name, rooted, rootStated, hasLengths: built.hasLengths,
            hasSupport: sets.length > 0,
            parent: built.parent.slice(0, nodes),
            length: built.length.slice(0, nodes),
            support,
            supportSets: sets.length > 1 ? sets : null,
            labels,
            cladeNames: built.cladeNames.size ? built.cladeNames : null,
            annotations: built.annotations.size ? built.annotations : null,
            nodeColors: built.colors.size ? built.colors : null,
            dialects: Array.from(built.dialects),
            warnings,
            nodeCount: nodes,
            leafCount: labels.length,
        };
    }

    TV.Builder = Builder;
    TV.finishTree = finishTree;
    TV.splitPairs = splitPairs;

    // ---- One Newick tree -----------------------------------------------------------------------------------------

    // What the next token of a Newick tree may be.
    const NODE = 0, LEAF = 1, CLOSED = 2, NAMED = 3, LENGTHED = 4;

    // The tree of the Newick text in text[start, end), whose ";" ends it (or the end does, which is a warning).
    // translate: a Map from a NEXUS TRANSLATE key to its name, or null.
    function readNewick(text, start, end, name, translate, maxLeaves) {
        const lex = lexer(text, start, end);
        const built = new Builder(maxLeaves);
        const maxNodes = built.maxNodes;
        const fail = (reason, at) => { throw new TreeParseError(reason, text, at); };
        const labels = built.labels;
        const cladeNames = built.cladeNames;
        const dialects = built.dialects;
        // The children of the nodes still open, one after another, and where each open node's children begin.
        const pending = [];
        const begins = [];
        let state = NODE;
        let last = -1;
        let lastIsLeaf = false;
        let started = false;
        let rooted = null;
        let rootChildren = 0;
        let ended = false;

        // What a comment after a node says about it: a metacomment [&key=value,...], NHX [&&NHX:key=value:...], FigTree's
        // !color, or RAxML's [100] (a number: the node's support, when it is an internal node).
        const comment = (text) => {
            if (!started || last < 0) return;
            if (text[0] !== "&") {
                if (!lastIsLeaf) {
                    if (built.slots[0][last] < 0) built.setSupport(last, Number(text), 0, "");
                    dialects.add("RAxML branch support");
                }
                return;
            }
            const nhx = /^&&NHX/i.test(text);
            const pairs = splitPairs(nhx ? text.slice(5).replace(/^:/, "") : text.slice(1), nhx ? ":" : ",");
            if (!pairs.length) return;
            if (nhx) dialects.add("NHX");
            let rank = SUPPORT_KEYS.length;
            let given = NaN;
            for (const [key, value] of pairs) {
                const lower = key.toLowerCase();
                if (lower[0] === "!") {
                    if (lower === "!color") {
                        const rgb = colourValue(value);
                        if (rgb) {
                            built.colors.set(last, rgb);
                            dialects.add("FigTree colours");
                        }
                    }
                    continue;
                }
                let found = SUPPORT_KEYS.indexOf(lower);
                if (nhx && lower === "b") found = 1;
                if (found >= 0 && NUMBER.test(value)) {
                    if (found < rank) {
                        rank = found;
                        given = Number(value);
                    }
                    continue;
                }
                built.annotate(last, key, value);
                if (!nhx) dialects.add("[&...] annotations");
            }
            if (!Number.isNaN(given) && built.slots[0][last] < 0) built.setSupport(last, given, 0, "");
        };

        // A label on a node that a ")" closed: its support when it is a number (or the numbers a/b/c of IQ-TREE and
        // RAxML-NG, each a support value), else its name.
        const label = (value, at) => {
            if (NUMBER.test(value)) {
                built.setSupport(last, Number(value), 0, "");
                return;
            }
            if (value.indexOf("/") > 0 && value.length < 200) {
                const parts = value.split("/");
                if (parts.length > 1 && parts.length <= MAX_SUPPORT_SLOTS && parts.every((part) => NUMBER.test(part))) {
                    parts.forEach((part, slot) => built.setSupport(last, Number(part), slot, ""));
                    dialects.add("paired support values");
                    return;
                }
            }
            if (value.length > TV.MAX_LABEL_CHARACTERS) fail(t("A name is longer than {count} characters", { count: TV.MAX_LABEL_CHARACTERS.toLocaleString("en-US") }), at);
            cladeNames.set(last, value);
        };

        for (;;) {
            lex.next();
            const { kind, value, at } = lex;
            if (kind === "end") break;
            if (kind === "c") {
                if (!started) {
                    const flag = value.replace(/\s/g, "").toUpperCase();
                    if (flag === "&R") rooted = true; else if (flag === "&U") rooted = false;
                } else if (state !== NODE) {
                    comment(value);
                }
                continue;
            }
            started = true;
            if (kind === "j") {
                // Jplace's edge number after a node: read and ignored.
                if (state === NODE) fail(t("A { } where no node ends"), at);
                dialects.add("Jplace edge numbers");
                continue;
            }
            if (kind === "p") {
                if (value === "(") {
                    if (state !== NODE) fail(t("A ( where no node can start"), at);
                    begins.push(pending.length);
                    if (begins.length > maxNodes) fail(t("The tree has more than {count} nested nodes", { count: maxNodes.toLocaleString("en-US") }), at);
                } else if (value === ":") {
                    if (state !== LEAF && state !== CLOSED && state !== NAMED) {
                        fail(t("A branch length where no node ends, or a leaf with no name"), at);
                    }
                    // The number, which a metacomment may come before.
                    let number;
                    for (;;) {
                        lex.next();
                        if (lex.kind !== "c") { number = lex; break; }
                        comment(lex.value);
                    }
                    if (number.kind !== "w" || !NUMBER.test(number.value)) {
                        fail(t("A branch length is not a number: {value}", { value: JSON.stringify(number.value.slice(0, 40)) }), number.at);
                    }
                    const branch = Number(number.value);
                    if (!Number.isFinite(branch)) fail(t("A branch length is not finite: {value}", { value: number.value }), number.at);
                    built.setLength(last, branch);
                    state = LENGTHED;
                } else if (value === "," || value === ")" || value === ";") {
                    if (state === NODE) fail(t("A node is missing before {value}", { value: JSON.stringify(value) }), at);
                    if (value === ",") {
                        if (!begins.length) fail(t("A , outside any parentheses"), at);
                        state = NODE;
                    } else if (value === ")") {
                        if (!begins.length) fail(t("A ) with no ( before it"), at);
                        const from = begins.pop();
                        const node = built.newNode((reason) => fail(reason, at));
                        const count = pending.length - from;
                        for (let k = from; k < pending.length; k++) built.parent[pending[k]] = node;
                        pending.length = from;
                        if (begins.length) pending.push(node); else rootChildren = count;
                        last = node;
                        lastIsLeaf = false;
                        state = CLOSED;
                    } else {
                        if (begins.length) fail(t("The tree ends before every ( is closed"), at);
                        ended = true;
                        break;
                    }
                } else {
                    fail(t("An = inside the tree"), at);
                }
            } else if (state === NODE) {
                let named = value;
                if (translate && translate.has(value)) named = translate.get(value);
                const node = built.leaf(named, (reason) => fail(reason, at));
                if (begins.length) pending.push(node);
                last = node;
                lastIsLeaf = true;
                state = LEAF;
            } else if (state === CLOSED) {
                label(value, at);
                state = NAMED;
            } else {
                fail(t("A name ({value}) where none can be", { value: JSON.stringify(value.slice(0, 40)) }), at);
            }
        }
        const warnings = [];
        if (!ended) {
            if (begins.length) fail(t("The tree ends before every ( is closed"), end);
            if (state === NODE) fail(t("A node is missing at the end of the tree"), end);
            warnings.push({ code: "no_semicolon" });
        }
        if (!built.nodes) fail(t("The tree has no leaves"), start);
        return finishTree(built, { name, rooted, rootChildren, warnings });
    }
    TV.readNewickRange = readNewick;

    // ---- A text of trees -----------------------------------------------------------------------------------------

    // The offset of the first character at or after `from` that is not white space or a comment; `end` when there is none.
    function skipBlank(text, from, end) {
        let pos = from;
        for (;;) {
            while (pos < end && /\s/.test(text[pos])) pos++;
            if (pos < end && text[pos] === "[") {
                const close = endOfComment(text, pos);
                if (close < 0) throw new TreeParseError(t("A comment in [ ] never closes"), text, pos);
                pos = close;
            } else {
                return pos;
            }
        }
    }

    // The entries of a plain Newick text: each tree from where it begins to its ";".
    function scanNewick(text) {
        const entries = [];
        let pos = 0;
        for (;;) {
            // A tree begins at its first character, which can be the comment that says [&R].
            let begin = pos;
            while (begin < text.length && /\s/.test(text[begin])) begin++;
            const meaning = skipBlank(text, begin, text.length);
            if (meaning >= text.length) break;
            const stop = skipToSemicolon(text, meaning, text.length);
            entries.push({ name: "", start: begin, end: stop.at });
            pos = stop.at;
        }
        return entries;
    }

    // What a NEXUS TRANSLATE command gives: a Map from each key to its name. `lex` is just past the word TRANSLATE.
    function readTranslate(text, lex) {
        const table = new Map();
        for (;;) {
            lex.next();
            if (lex.kind === "end") throw new TreeParseError(t("A TRANSLATE command that never ends with ;"), text, lex.at);
            if (lex.kind === "p" && lex.value === ";") return table;
            if (lex.kind === "c") continue;
            if (lex.kind === "p" && lex.value === ",") continue;
            if (lex.kind !== "w" && lex.kind !== "q") throw new TreeParseError(t("A TRANSLATE command must read key name, key name, ...;"), text, lex.at);
            const key = lex.value;
            lex.next();
            if (lex.kind !== "w" && lex.kind !== "q") throw new TreeParseError(t("A TRANSLATE key ({value}) has no name", { value: JSON.stringify(key.slice(0, 40)) }), text, lex.at);
            table.set(key, lex.value);
        }
    }

    // The entries of a NEXUS text, and its TRANSLATE table.
    function scanNexus(text, lex) {
        const entries = [];
        let translate = null;
        let block = null;
        const skipCommand = () => {
            const stop = skipToSemicolon(text, lex.pos, text.length);
            lex.pos = stop.at;
        };
        for (;;) {
            lex.next();
            if (lex.kind === "end") break;
            if (lex.kind === "c" || (lex.kind === "p" && lex.value === ";")) continue;
            const word = lex.kind === "w" ? lex.value.toUpperCase() : "";
            if (word === "BEGIN") {
                lex.next();
                block = lex.kind === "w" ? lex.value.toUpperCase() : "";
                skipCommand();
            } else if (word === "END" || word === "ENDBLOCK") {
                block = null;
                skipCommand();
            } else if (block === "TREES" && (word === "TREE" || word === "UTREE")) {
                const names = [];
                let equals = -1;
                for (;;) {
                    lex.next();
                    if (lex.kind === "end") throw new TreeParseError(t("A TREE command has no tree"), text, text.length);
                    if (lex.kind === "p" && lex.value === "=") { equals = lex.pos; break; }
                    if (lex.kind === "p") throw new TreeParseError(t("A TREE command must read TREE name = tree;"), text, lex.at);
                    if ((lex.kind === "w" || lex.kind === "q") && lex.value !== "*") names.push(lex.value);
                }
                const stop = skipToSemicolon(text, equals, text.length);
                entries.push({ name: names.join(" "), start: equals, end: stop.at });
                lex.pos = stop.at;
            } else if (block === "TREES" && word === "TRANSLATE") {
                translate = readTranslate(text, lex);
            } else {
                skipCommand();
            }
        }
        return { entries, translate };
    }

    // The trees of a text of either format. Throws a TreeParseError for a text that is empty, is NEXUS with no tree, or
    // is not well-formed; reading the tree itself (tree(index)) can throw one too.
    TV.readTreeText = function (text, options) {
        const maxLeaves = (options && options.maxLeaves) || TV.MAX_LEAVES;
        if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
        // The format comes from the content. XML begins with <, JSON with { (or a [ that holds an object or list); a
        // Newick tree may begin with a comment, so its first meaning is found past any.
        let lead = 0;
        while (lead < text.length && /\s/.test(text[lead])) lead++;
        if (lead < text.length) {
            const c = text[lead];
            if (c === "<") {
                if (!TV.readXmlText) throw new TreeParseError(t("This page cannot read XML trees"), text, lead);
                return TV.readXmlText(text, lead, { maxLeaves });
            }
            if (c === "{" || (c === "[" && /^\[\s*[{\[]/.test(text.slice(lead, lead + 40)))) {
                if (!TV.readJsonText) throw new TreeParseError(t("This page cannot read JSON trees"), text, lead);
                return TV.readJsonText(text, lead, { maxLeaves });
            }
        }
        const first = skipBlank(text, 0, text.length);
        if (first >= text.length) throw new TreeParseError(t("The text is empty"), text, 0);
        let format = "Newick";
        let entries;
        let translate = null;
        if (text.slice(first, first + 6).toUpperCase() === "#NEXUS" && !/[^\s]/.test(text.charAt(first + 6) || " ")) {
            format = "NEXUS";
            const lex = lexer(text, first + 6, text.length);
            ({ entries, translate } = scanNexus(text, lex));
            if (!entries.length) throw new TreeParseError(t("The file holds no tree: its TREES block has no TREE command"), text, first);
        } else if (text[first] === ">") {
            throw new TreeParseError(t("The file looks like FASTA sequences, not a tree: a tree begins with (, <, { or #NEXUS"), text, first);
        } else if (text[first] === "#") {
            throw new TreeParseError(t("The file is neither Newick nor NEXUS: a NEXUS file begins with #NEXUS"), text, first);
        } else {
            entries = scanNewick(text);
        }
        const cache = new Map();
        return {
            format,
            entries,
            tree(index) {
                if (!(index >= 0 && index < entries.length)) throw new RangeError("There is no tree " + index);
                if (!cache.has(index)) {
                    cache.clear();
                    const entry = entries[index];
                    cache.set(index, readNewick(text, entry.start, entry.end, entry.name, translate, maxLeaves));
                }
                return cache.get(index);
            },
        };
    };
})(window.TreeView);
