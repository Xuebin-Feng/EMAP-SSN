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
 * in, and applyPlatformKeys() to the page's own text and its title and
 * aria-label attributes once the page has loaded. It does what
 * utilities/Platform_Keys.py does for the Viewer's Qt windows: keep the two alike.
 */
const PAGE_IS_MAC = /Mac/.test((typeof navigator !== "undefined" && navigator.platform) || "");

function platformKeyText(text, mac = PAGE_IS_MAC) {
    if (!mac) return text;
    return text.replace(/(?<![A-Za-z0-9_])Ctrl\+Y(?![A-Za-z0-9_])/g, "Cmd+Shift+Z")
        .replace(/(?<![A-Za-z0-9_])Ctrl(?![A-Za-z0-9_])/g, "Cmd");
}

function applyPlatformKeys(root = document, mac = PAGE_IS_MAC) {
    if (!mac) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const text = platformKeyText(node.nodeValue, mac);
        if (text !== node.nodeValue) node.nodeValue = text;
    }
    for (const element of root.querySelectorAll("[title], [aria-label]")) {
        for (const name of ["title", "aria-label"]) {
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
