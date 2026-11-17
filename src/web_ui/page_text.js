/* Copyright 2026 Xuebin Feng
 * Author affiliation: University of Toronto
 * SPDX-License-Identifier: Apache-2.0
 *
 * t(text, values): one of a page's texts, its values filled in.
 *
 * Mark each text a page's scripts show with t(), passing the English as a
 * plain string literal: t("Agent activated: {model}", {model: name}). The
 * Viewer's web server has already written the text in the page's language
 * (src/web_ui/Page_Texts.py), so t() only fills in each {name} from values,
 * and turns {{ and }} into single braces, as a Message does in Python.
 *
 * platformKeyText(text, mac): on a Mac the page's shortcuts are made with Cmd,
 * so the texts' "Ctrl" reads "Cmd" there, and "Ctrl+Y" reads "Cmd+Shift+Z",
 * the Mac's redo key. t() applies it to the text before its values are filled
 * in, and applyPlatformKeys() to the page's marked interface text once the
 * page has loaded (data-i18n elements and title, aria-label, placeholder and
 * alt attributes, nothing else, never the page's data). It does what
 * utilities/Platform_Keys.py does for the Viewer's Qt windows: keep the two alike.
 */
const PAGE_IS_MAC = /Mac/.test((typeof navigator !== "undefined" && navigator.platform) || "");

function platformKeyText(text, mac = PAGE_IS_MAC) {
    if (!mac) return text;
    return text.replace(/(?<![A-Za-z0-9_])Ctrl\+Y(?![A-Za-z0-9_])/g, "Cmd+Shift+Z")
        .replace(/(?<![A-Za-z0-9_])Ctrl(?![A-Za-z0-9_])/g, "Cmd");
}

// Only the page's own interface text is rewritten: the content of an element marked data-i18n, and the title,
// aria-label, placeholder and alt attributes (the texts src/web_ui/Page_Texts.py finds), never what lies in a
// script, style, textarea, input, pre, code, table or grid, in an element with translate="no", or in the
// page's data ("Ctrl" is a common name for a control in sample names and metadata).
const PLATFORM_KEYS_SKIPPED = "script, style, template, textarea, input, select, pre, code, table, [role=grid], " +
    ".tabulator, [translate=no]";
const PLATFORM_KEYS_ATTRIBUTES = ["title", "aria-label", "placeholder", "alt"];

function applyPlatformKeys(root = document, mac = PAGE_IS_MAC) {
    if (!mac) return;
    const skipped = (element) => !element || element.closest(PLATFORM_KEYS_SKIPPED) !== null;
    for (const marked of root.querySelectorAll("[data-i18n]")) {
        if (skipped(marked)) continue;
        const walker = document.createTreeWalker(marked, NodeFilter.SHOW_TEXT, {
            acceptNode: (node) => skipped(node.parentElement) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
        });
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            const text = platformKeyText(node.nodeValue, mac);
            if (text !== node.nodeValue) node.nodeValue = text;
        }
    }
    const selector = PLATFORM_KEYS_ATTRIBUTES.map((name) => "[" + name + "]").join(", ");
    for (const element of root.querySelectorAll(selector)) {
        if (skipped(element)) continue;
        for (const name of PLATFORM_KEYS_ATTRIBUTES) {
            const value = element.getAttribute(name);
            if (value === null) continue;
            const text = platformKeyText(value, mac);
            if (text !== value) element.setAttribute(name, text);
        }
    }
}

if (PAGE_IS_MAC && typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => applyPlatformKeys());
    else applyPlatformKeys();
}

function t(text, values = {}) {
    return platformKeyText(text).replace(/\{\{|\}\}|\{(\w+)\}/g, (field, name) => {
        if (name === undefined) return field[0];
        return Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : field;
    });
}
