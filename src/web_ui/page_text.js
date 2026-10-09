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
 */
function t(text, values = {}) {
    return text.replace(/\{\{|\}\}|\{(\w+)\}/g, (field, name) => {
        if (name === undefined) return field[0];
        return Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : field;
    });
}
