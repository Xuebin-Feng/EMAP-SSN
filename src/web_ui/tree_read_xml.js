/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copied from Cladewise (the same author's standalone tree viewer, same licence), js/tree_xml.js at its commit ba06967. What differs
 * here: every text the reader says goes through t() (page_text.js), so the Viewer shows it in its language; the strings
 * that are names of notations (NHX, Jplace, FigTree) are kept as they are.
 *
 * Cladewise's XML readers: PhyloXML and NeXML text to the same tree arrays as the Newick reader (tree_parse.js).
 *
 * The browser's DOMParser is not used: Chromium refuses XML nested more than about 5,000 elements deep ("Excessive node
 * nesting"), and a tree can be as deep as it is long. This file has a small reader of its own that keeps its open
 * elements in a list and never recurses, checks that the text is well formed (tags that match, quoted attributes, one
 * root, known entities), and says where an error is: the line and column, and the path of the elements it is in.
 * Namespace prefixes are ignored (phy:clade is clade). A DOCTYPE is skipped, and no entity of its own is defined, so
 * nothing in a file can expand into more text.
 *
 * PhyloXML (root phyloxml; each phylogeny is a tree): clade, name, branch_length (element or attribute), confidence (each
 * type is a support value of its own, so two types give the page a choice), taxonomy (scientific_name, then code, is the
 * label of a clade with no name; with common_name it is also an annotation), sequence (name, accession, symbol: a label
 * after the taxonomy's, and annotations), color (red, green, blue: a colour for the clade and, unless a clade below has
 * its own, for its leaves), and the phylogeny's rooted attribute and name. Everything else is ignored.
 *
 * NeXML (root nexml; each tree element of a trees element is a tree): the otu labels of the otus elements, node (label,
 * otu), edge (source, target, length), and a meta element of a node or an edge with a numeric content whose property
 * names support, bootstrap, posterior or confidence is the support of the node (of the edge's target). Other meta
 * properties are annotations. Edges may come in any order; the tree must be one root with every node below it.
 */
window.TreeView = window.TreeView || {};

(function (TV) {
    "use strict";

    const TreeParseError = TV.TreeParseError;
    const NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

    // The elements a tag opens: <name attr="value" ... > or <name ... />.
    const OPEN = /<([^\s\/>=<"']+)((?:\s+[^\s\/>=<"']+\s*=\s*(?:"[^"<]*"|'[^'<]*'))*)\s*(\/?)>/y;
    const CLOSE = /<\/([^\s>]+)\s*>/y;
    const ATTRIBUTE = /([^\s=]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
    const ENTITY = /&([^;\s&<]*);?/g;
    const NAMED = { lt: "<", gt: ">", amp: "&", quot: '"', apos: "'" };

    // The path of the elements that are open, for an error: the innermost few.
    function pathOf(stack) {
        if (!stack.length) return "";
        return (stack.length > 6 ? "…/" : "") + stack.slice(-6).join("/");
    }

    // `raw` with its entities (&lt; &gt; &amp; &quot; &apos; &#9; &#x41;) turned into characters. `bad(reason)` throws.
    function decode(raw, bad) {
        if (raw.indexOf("&") < 0) return raw;
        return raw.replace(ENTITY, (whole, body) => {
            if (whole[whole.length - 1] !== ";") bad(t("An & that is not an entity (write &amp;)"));
            if (Object.prototype.hasOwnProperty.call(NAMED, body)) return NAMED[body];
            const match = /^#(?:x([0-9a-fA-F]{1,6})|(\d{1,7}))$/.exec(body);
            if (match) {
                const code = match[1] ? parseInt(match[1], 16) : Number(match[2]);
                if (code > 0 && code <= 0x10FFFF && !(code >= 0xD800 && code <= 0xDFFF)) return String.fromCodePoint(code);
            }
            return bad(t("The entity &{name}; is not one XML defines", { name: body.slice(0, 20) }));
        });
    }

    // The attributes of a tag's text, by local name (xsi:type is type), decoded.
    function attributesOf(raw, bad) {
        const found = {};
        if (!raw) return found;
        ATTRIBUTE.lastIndex = 0;
        let match;
        while ((match = ATTRIBUTE.exec(raw))) {
            const colon = match[1].lastIndexOf(":");
            const name = colon >= 0 ? match[1].slice(colon + 1) : match[1];
            found[name] = decode(match[2] !== undefined ? match[2] : match[3], bad);
        }
        return found;
    }

    // Read text[from, to), calling h.open(local, rawAttributes, at, stack) at each start tag, h.text(raw, at, literal) at
    // each run of text that is not white space (a CDATA section is literal: no entities in it), and h.close(local, end,
    // stack) at each end tag (stack, the list of the open elements' local names, is the one after the element is closed
    // for close, and includes it for open; an element with no content closes at once). Throws a TreeParseError for text
    // that is not well formed.
    function scanXml(text, from, to, h) {
        const full = [];
        const local = [];
        let pos = from;
        let rootDone = false;
        let sawRoot = false;
        const bad = (reason, at) => { throw new TreeParseError(reason, text, at, pathOf(local)); };
        while (pos < to) {
            const lt = text.indexOf("<", pos);
            const stop = lt < 0 || lt > to ? to : lt;
            if (stop > pos) {
                const raw = text.slice(pos, stop);
                if (/\S/.test(raw)) {
                    if (!local.length) bad(t("Text outside the root element"), pos + raw.search(/\S/));
                    h.text(raw, pos, false);
                }
            }
            if (stop >= to) break;
            pos = stop;
            const next = text.charCodeAt(pos + 1);
            if (next === 33) {
                // <!-- comment -->, <![CDATA[ ... ]]>, <!DOCTYPE ...>
                if (text.startsWith("<!--", pos)) {
                    const close = text.indexOf("-->", pos + 4);
                    if (close < 0 || close + 3 > to) bad(t("A comment that never closes"), pos);
                    pos = close + 3;
                } else if (text.startsWith("<![CDATA[", pos)) {
                    const close = text.indexOf("]]>", pos + 9);
                    if (close < 0 || close + 3 > to) bad(t("A CDATA section that never closes"), pos);
                    if (!local.length) bad(t("Text outside the root element"), pos);
                    h.text(text.slice(pos + 9, close), pos, true);
                    pos = close + 3;
                } else if (text.startsWith("<!DOCTYPE", pos) && !sawRoot) {
                    // Skip to its ">", past an internal subset in [ ] and any quote.
                    let at = pos + 9;
                    let depth = 0;
                    let quote = "";
                    for (; at < to; at++) {
                        const c = text[at];
                        if (quote) { if (c === quote) quote = ""; continue; }
                        if (c === '"' || c === "'") quote = c;
                        else if (c === "[") depth++;
                        else if (c === "]") depth--;
                        else if (c === ">" && depth <= 0) break;
                    }
                    if (at >= to) bad(t("A DOCTYPE that never closes"), pos);
                    pos = at + 1;
                } else {
                    bad(t("The text cannot be read here"), pos);
                }
            } else if (next === 63) {
                const close = text.indexOf("?>", pos + 2);
                if (close < 0 || close + 2 > to) bad(t("A <? that never closes"), pos);
                pos = close + 2;
            } else if (next === 47) {
                CLOSE.lastIndex = pos;
                const match = CLOSE.exec(text);
                if (!match || match.index !== pos) bad(t("A closing tag that cannot be read"), pos);
                if (!local.length) bad(t("A closing tag </{name}> with nothing open", { name: match[1] }), pos);
                if (full[full.length - 1] !== match[1]) {
                    bad(t("A closing tag </{name}> where </{expected}> was expected", { name: match[1].slice(0, 40), expected: full[full.length - 1].slice(0, 40) }), pos);
                }
                full.pop();
                const name = local.pop();
                pos = CLOSE.lastIndex;
                if (!local.length) rootDone = true;
                h.close(name, pos, local);
            } else {
                OPEN.lastIndex = pos;
                const match = OPEN.exec(text);
                if (!match || match.index !== pos || OPEN.lastIndex > to) bad(t("A tag that cannot be read"), pos);
                if (rootDone) bad(t("More than one root element"), pos);
                sawRoot = true;
                const colon = match[1].lastIndexOf(":");
                const name = colon >= 0 ? match[1].slice(colon + 1) : match[1];
                const at = pos;
                pos = OPEN.lastIndex;
                full.push(match[1]);
                local.push(name);
                h.open(name, match[2], at, local);
                if (match[3]) {
                    full.pop();
                    local.pop();
                    if (!local.length) rootDone = true;
                    h.close(name, pos, local);
                }
            }
        }
        if (local.length) bad(t("The element <{name}> never closes", { name: full[full.length - 1].slice(0, 40) }), to);
        if (!sawRoot) bad(t("The text holds no element"), from);
    }

    // ---- The list of trees ------------------------------------------------------------------------------------------

    // The trees of an XML text: { format, entries, tree(index) }, as TV.readTreeText answers. `lead` is where the text's
    // first element begins.
    TV.readXmlText = function (text, lead, options) {
        const maxLeaves = (options && options.maxLeaves) || TV.MAX_LEAVES;
        // The root element's name: the first tag that is not a declaration or a comment.
        const head = /<(?![?!])([^\s\/>]+)/.exec(text.slice(lead, lead + 100000).replace(/<!--[\s\S]*?-->/g, (m) => " ".repeat(m.length)).replace(/<!DOCTYPE[^>\[]*(\[[\s\S]*?\])?\s*>/gi, (m) => " ".repeat(m.length)));
        const rootName = head ? head[1].slice(head[1].lastIndexOf(":") + 1) : "";
        const lower = rootName.toLowerCase();
        if (lower === "phyloxml") return phyloReader(text, maxLeaves);
        if (lower === "nexml") return nexmlReader(text, maxLeaves);
        throw new TreeParseError(t("The XML is neither PhyloXML nor NeXML: its root element is <{name}>, not <phyloxml> or <nexml>", { name: rootName.slice(0, 40) || "?" }), text, lead);
    };

    function wrap(format, entries, build) {
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

    // ---- PhyloXML ---------------------------------------------------------------------------------------------------

    function phyloReader(text, maxLeaves) {
        const entries = [];
        let current = null;
        let naming = false;
        const bad = (reason, at, stack) => { throw new TreeParseError(reason, text, at, pathOf(stack || [])); };
        scanXml(text, 0, text.length, {
            open(name, raw, at, stack) {
                if (name === "phylogeny" && stack.length === 2 && !current) {
                    current = { name: "", start: at, end: at, attributes: raw };
                } else if (current && name === "name" && stack.length === 3 && stack[1] === "phylogeny") {
                    naming = true;
                }
            },
            text(raw, at, literal) {
                if (naming) current.name += literal ? raw : decode(raw, (reason) => bad(reason, at, ["phylogeny", "name"]));
            },
            close(name, end, stack) {
                if (naming && name === "name") naming = false;
                if (current && name === "phylogeny" && stack.length === 1) {
                    current.end = end;
                    current.name = current.name.trim();
                    entries.push(current);
                    current = null;
                }
            },
        });
        if (!entries.length) throw new TreeParseError(t("The file holds no phylogeny element"), text, 0);
        return wrap("PhyloXML", entries, (entry) => buildPhylo(text, entry, maxLeaves));
    }

    // One phylogeny of a PhyloXML text, read without recursion: the clades being read are a list of frames, a clade
    // becomes a node when its end tag comes (so the children come before it, as the model needs).
    function buildPhylo(text, entry, maxLeaves) {
        const built = new TV.Builder(maxLeaves);
        const frames = [];
        const slots = new Map();
        let capture = null;
        let top = 0;
        let rootChildren = 0;
        let rooted = null;
        const failAt = (reason, at, stack) => { throw new TreeParseError(reason, text, at, pathOf(stack)); };

        const labelOf = (frame) => frame.name || frame.scientific || frame.code || frame.sequenceName || frame.accession || "";

        scanXml(text, entry.start, entry.end, {
            open(name, raw, at, stack) {
                const depth = stack.length;
                if (name === "phylogeny" && depth === 1) {
                    const given = attributesOf(raw, (reason) => failAt(reason, at, stack)).rooted;
                    if (given !== undefined) rooted = /^(true|1)$/i.test(given.trim()) ? true : /^(false|0)$/i.test(given.trim()) ? false : null;
                    return;
                }
                if (name === "clade") {
                    const frame = { children: [], at, name: "", length: "", lengthAt: at, confidences: [], scientific: "", code: "", common: "",
                        sequenceName: "", accession: "", symbol: "", red: "", green: "", blue: "", hasColor: false };
                    if (raw) {
                        const given = attributesOf(raw, (reason) => failAt(reason, at, stack)).branch_length;
                        if (given !== undefined) frame.length = given;
                    }
                    frames.push(frame);
                    return;
                }
                if (!frames.length) return;
                const parent = stack[depth - 2];
                const grand = stack[depth - 3];
                const frame = frames[frames.length - 1];
                let field = null;
                if (parent === "clade") {
                    if (name === "name") field = "name";
                    else if (name === "branch_length") field = "length";
                    else if (name === "confidence") field = "confidence";
                } else if (grand === "clade") {
                    if (parent === "taxonomy") {
                        if (name === "scientific_name") field = "scientific";
                        else if (name === "code") field = "code";
                        else if (name === "common_name") field = "common";
                    } else if (parent === "sequence") {
                        if (name === "name") field = "sequenceName";
                        else if (name === "accession") field = "accession";
                        else if (name === "symbol") field = "symbol";
                    } else if (parent === "color") {
                        if (name === "red" || name === "green" || name === "blue") field = name;
                    }
                }
                if (!field) return;
                capture = { field, element: name, frame, text: "", at, type: "" };
                if (field === "confidence") capture.type = (attributesOf(raw, (reason) => failAt(reason, at, stack)).type || "").trim();
            },
            text(raw, at, literal) {
                if (capture) capture.text += literal ? raw : decode(raw, (reason) => failAt(reason, at, ["clade"]));
            },
            close(name, end, stack) {
                if (capture && name === capture.element && capture.frame === frames[frames.length - 1]) {
                    const { field, frame, at } = capture;
                    const value = capture.text.trim();
                    if (field === "confidence") frame.confidences.push({ type: capture.type, value, at });
                    else if (field === "length") { frame.length = value; frame.lengthAt = at; }
                    else if (field === "red" || field === "green" || field === "blue") { frame[field] = value; frame.hasColor = true; }
                    else frame[field] = value;
                    capture = null;
                    return;
                }
                if (name !== "clade") return;
                const frame = frames.pop();
                const fail = (reason) => failAt(reason, frame.at, stack.concat("clade"));
                const leaf = frame.children.length === 0;
                const node = leaf ? built.leaf(labelOf(frame), fail) : built.internal(frame.children, fail);
                if (frame.length !== "") {
                    if (!NUMBER.test(frame.length) || !Number.isFinite(Number(frame.length))) failAt(t("A branch length is not a number: {value}", { value: JSON.stringify(frame.length.slice(0, 40)) }), frame.lengthAt, stack.concat("clade"));
                    built.setLength(node, Number(frame.length));
                }
                for (const confidence of frame.confidences) {
                    if (NUMBER.test(confidence.value)) {
                        const type = confidence.type || "confidence";
                        if (!slots.has(type)) slots.set(type, slots.size);
                        built.setSupport(node, Number(confidence.value), slots.get(type), type);
                    } else if (confidence.value) {
                        built.annotate(node, "confidence" + (confidence.type ? " (" + confidence.type + ")" : ""), confidence.value);
                    }
                }
                if (!leaf && frame.name) built.setName(node, frame.name, fail);
                const label = leaf ? labelOf(frame) : "";
                if (frame.scientific && frame.scientific !== label) built.annotate(node, "Scientific name", frame.scientific);
                if (frame.code && frame.code !== label) built.annotate(node, "Taxonomy code", frame.code);
                if (frame.common) built.annotate(node, "Common name", frame.common);
                if (frame.accession && frame.accession !== label) built.annotate(node, "Accession", frame.accession);
                if (frame.symbol) built.annotate(node, "Symbol", frame.symbol);
                if (frame.hasColor) {
                    const channels = [frame.red, frame.green, frame.blue].map((value) => (NUMBER.test(value) ? Number(value) : NaN));
                    if (channels.every((value) => value >= 0 && value <= 255)) built.colors.set(node, channels.map(Math.round));
                    else failAt(t("A color has a red, green or blue that is not a number from 0 to 255"), frame.at, stack.concat("clade"));
                }
                if (frames.length) {
                    frames[frames.length - 1].children.push(node);
                } else {
                    top++;
                    rootChildren = frame.children.length;
                }
            },
        });
        if (top !== 1) throw new TreeParseError(top ? t("The phylogeny holds {count} top-level clades; it needs one", { count: top }) : t("The phylogeny holds no clade"), text, entry.start, "phylogeny");
        return TV.finishTree(built, { name: entry.name, rooted, rootChildren });
    }

    // ---- NeXML ------------------------------------------------------------------------------------------------------

    const SUPPORT_PROPERTY = /support|bootstrap|posterior|confidence/i;

    function nexmlReader(text, maxLeaves) {
        const entries = [];
        const otus = new Map();
        let current = null;
        const failAt = (reason, at, stack) => { throw new TreeParseError(reason, text, at, pathOf(stack)); };
        scanXml(text, 0, text.length, {
            open(name, raw, at, stack) {
                const parent = stack[stack.length - 2];
                if (name === "otu" && parent === "otus") {
                    const given = attributesOf(raw, (reason) => failAt(reason, at, stack));
                    if (given.id !== undefined) otus.set(given.id, given.label !== undefined && given.label !== "" ? given.label : given.id);
                } else if (name === "tree" && parent === "trees" && !current) {
                    const given = attributesOf(raw, (reason) => failAt(reason, at, stack));
                    current = { name: given.label || given.id || "", start: at, end: at };
                }
            },
            text() {},
            close(name, end, stack) {
                if (current && name === "tree" && stack[stack.length - 1] === "trees") {
                    current.end = end;
                    entries.push(current);
                    current = null;
                }
            },
        });
        if (!entries.length) throw new TreeParseError(t("The file holds no tree element in a trees element"), text, 0);
        return wrap("NeXML", entries, (entry) => buildNexml(text, entry, otus, maxLeaves));
    }

    // One tree of a NeXML text. Nodes and edges are read in file order, then joined: a node's children are the targets of
    // the edges that name it as the source, in the order the edges come. The tree is walked from its one root with a list
    // in place of recursion.
    function buildNexml(text, entry, otus, maxLeaves) {
        const nodeIds = new Map();
        const nodes = [];
        const edges = [];
        let owner = null;
        const failAt = (reason, at, stack) => { throw new TreeParseError(reason, text, at, pathOf(stack)); };
        const place = (item, kind) => kind + (item.id ? "[id=" + item.id.slice(0, 40) + "]" : "");

        scanXml(text, entry.start, entry.end, {
            open(name, raw, at, stack) {
                const parent = stack[stack.length - 2];
                if (name === "node" && parent === "tree") {
                    const given = attributesOf(raw, (reason) => failAt(reason, at, stack));
                    const node = { id: given.id, label: given.label || "", otu: given.otu || "", at, meta: [] };
                    if (node.id === undefined || node.id === "") failAt(t("A node has no id"), at, stack);
                    if (nodeIds.has(node.id)) failAt(t("Two nodes share the id {id}", { id: JSON.stringify(node.id.slice(0, 40)) }), at, stack);
                    nodeIds.set(node.id, nodes.length);
                    nodes.push(node);
                    owner = node;
                } else if (name === "edge" && parent === "tree") {
                    const given = attributesOf(raw, (reason) => failAt(reason, at, stack));
                    const edge = { id: given.id, source: given.source, target: given.target, length: given.length, at, meta: [] };
                    edges.push(edge);
                    owner = edge;
                } else if (name === "meta" && owner && (parent === "node" || parent === "edge")) {
                    const given = attributesOf(raw, (reason) => failAt(reason, at, stack));
                    if (given.property !== undefined && given.content !== undefined) owner.meta.push([given.property, given.content]);
                }
            },
            text() {},
            close(name) {
                if (name === "node" || name === "edge") owner = null;
            },
        });

        const count = nodes.length;
        if (!count) throw new TreeParseError(t("The tree holds no node"), text, entry.start, "tree");
        const parentOf = new Int32Array(count).fill(-1);
        const sourceOf = new Int32Array(edges.length);
        const targetOf = new Int32Array(edges.length);
        const childCount = new Int32Array(count + 1);
        edges.forEach((edge, e) => {
            const at = (reason) => failAt(reason, edge.at, ["tree", place(edge, "edge")]);
            const source = nodeIds.get(edge.source);
            const target = nodeIds.get(edge.target);
            if (source === undefined) at(t("The edge names a source node that is not in the tree: {id}", { id: JSON.stringify(String(edge.source).slice(0, 40)) }));
            if (target === undefined) at(t("The edge names a target node that is not in the tree: {id}", { id: JSON.stringify(String(edge.target).slice(0, 40)) }));
            if (source === target) at(t("The edge joins a node to itself"));
            if (parentOf[target] >= 0) at(t("The node {id} has two parents", { id: JSON.stringify(edge.target.slice(0, 40)) }));
            parentOf[target] = source;
            sourceOf[e] = source;
            targetOf[e] = target;
            childCount[source]++;
        });
        let root = -1;
        let roots = 0;
        for (let i = 0; i < count; i++) if (parentOf[i] < 0) { roots++; if (root < 0) root = i; }
        if (roots !== 1) {
            throw new TreeParseError(roots ? t("The tree has {count} nodes with no parent; it needs exactly one root", { count: roots }) : t("Every node has a parent: the edges form a cycle"), text, entry.start, "tree");
        }
        // The children of each node, side by side, in the order of the edges.
        const start = new Int32Array(count + 1);
        for (let i = 0; i < count; i++) start[i + 1] = start[i] + childCount[i];
        const kids = new Int32Array(edges.length);
        const fill = start.slice(0, count);
        for (let e = 0; e < edges.length; e++) kids[fill[sourceOf[e]]++] = targetOf[e];
        const edgeOf = new Int32Array(count).fill(-1);
        for (let e = 0; e < edges.length; e++) edgeOf[targetOf[e]] = e;

        const built = new TV.Builder(maxLeaves);
        const treeFail = (reason) => failAt(reason, entry.start, ["tree"]);
        const made = new Int32Array(count).fill(-1);
        const cursor = new Int32Array(count);
        const stack = [root];
        const order = [];
        let walked = 0;
        while (stack.length) {
            const node = stack[stack.length - 1];
            const first = start[node];
            const at = cursor[node];
            if (first + at < start[node + 1]) {
                cursor[node]++;
                stack.push(kids[first + at]);
                continue;
            }
            stack.pop();
            walked++;
            const item = nodes[node];
            const label = item.label || (item.otu ? (otus.get(item.otu) || item.otu) : "");
            let id;
            if (start[node + 1] === start[node]) {
                id = built.leaf(label, treeFail);
            } else {
                const children = [];
                for (let k = start[node]; k < start[node + 1]; k++) children.push(made[kids[k]]);
                id = built.internal(children, treeFail);
                if (item.label) built.setName(id, item.label, treeFail);
            }
            made[node] = id;
            order.push(node);
        }
        if (walked !== count) throw new TreeParseError(t("{count} nodes are not joined to the root (a cycle, or a second tree)", { count: count - walked }), text, entry.start, "tree");
        const rootChildren = childCount[root];

        // Length, support and annotations, now that every node has its number.
        for (const node of order) {
            const id = made[node];
            const item = nodes[node];
            const edge = edgeOf[node] >= 0 ? edges[edgeOf[node]] : null;
            if (edge && edge.length !== undefined && edge.length !== "") {
                const length = edge.length.trim();
                if (!NUMBER.test(length) || !Number.isFinite(Number(length))) {
                    failAt(t("A branch length is not a number: {value}", { value: JSON.stringify(length.slice(0, 40)) }), edge.at, ["tree", place(edge, "edge")]);
                }
                built.setLength(id, Number(length));
            }
            const lists = edge ? item.meta.concat(edge.meta) : item.meta;
            for (const [property, content] of lists) {
                if (SUPPORT_PROPERTY.test(property) && NUMBER.test(content.trim()) && built.slots[0][id] < 0) {
                    built.setSupport(id, Number(content.trim()), 0, "");
                } else {
                    built.annotate(id, property, content.length > 300 ? content.slice(0, 300) + "…" : content);
                }
            }
        }
        return TV.finishTree(built, { name: entry.name, rooted: null, rootChildren });
    }
})(window.TreeView);
