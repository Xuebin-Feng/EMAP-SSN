# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Helpers for checking a window's text against the pseudo-language.

Under the pseudo-language (Desktop_App.install_translations(app,
PSEUDO_LANGUAGE)), a text that came from a catalog is bracketed, so in
visible_texts() the unbracketed ones are text never marked for translation,
and cut_off_texts() lists translated text a window doesn't show whole.
outside_the_catalog() finds the untranslated part of a message the code put
together. pseudo_language() installs the pseudo-language for one test.

Not collected by unittest; import with ``from tests.translation_fixtures import ...``.
"""
import ast
import re

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextDocumentFragment
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabBar,
    QTabWidget,
    QWidget,
)

from desktop.Desktop_App import (
    NAME_ITEM_ROLE,
    PSEUDO_LANGUAGE,
    SYSTEM_LANGUAGE,
    LanguageSelector,
    ToggleSwitch,
    clipped_button_text,
    install_translations,
)
from utilities.Localization import LANGUAGES_DIR, is_pseudo_translated


def _describe(widget, part):
    name = widget.objectName()
    return f"{type(widget).__name__}{f' {name}' if name else ''} {part}"


def _shown_label_text(label):
    """The text a label shows: a rich-text label's without its markup."""
    text = label.text()
    rich = label.textFormat() == Qt.TextFormat.RichText or (
        label.textFormat() == Qt.TextFormat.AutoText and QtGui.Qt.mightBeRichText(text)
    )
    return QTextDocumentFragment.fromHtml(text).toPlainText() if rich else text


def visible_texts(window):
    """(where, text) for every text a user can read in window.

    That is its title, labels, buttons, group and tab titles, list choices,
    placeholders, spin box units, tooltips, and the names and descriptions
    a screen reader speaks, on every tab, shown or not.
    A rich-text label counts with the text it shows, without its markup, and
    a toggle switch with both its texts, on and off.
    What a user typed or picked (an edit's text, a spin box's value) is
    their data, not the window's text, so it is left out, as is text
    without a letter, such as a number, an arrow or an emoji alone. So are
    a list choice that shows an outside name, such as a device's
    (NAME_ITEM_ROLE), and the language names in the Language dropdown,
    which each show their own language on purpose.
    """
    found = []

    def add(widget, part, text):
        if text and any(character.isalpha() for character in text):
            found.append((_describe(widget, part), text))

    add(window, "title", window.windowTitle())
    for widget in [window, *window.findChildren(QWidget)]:
        if isinstance(widget, QLabel):
            add(widget, "text", _shown_label_text(widget))
        if isinstance(widget, ToggleSwitch):
            for state, text in zip(("off", "on"), widget.state_texts()):
                add(widget, f"{state} text", text)
        elif isinstance(widget, QAbstractButton):
            add(widget, "text", widget.text())
        if isinstance(widget, QGroupBox):
            add(widget, "title", widget.title())
        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                add(widget, f"tab {index}", widget.tabText(index))
        if isinstance(widget, QComboBox):
            for index in range(widget.count()):
                if isinstance(widget, LanguageSelector) and widget.itemData(index) != SYSTEM_LANGUAGE:
                    continue
                if widget.itemData(index, NAME_ITEM_ROLE):
                    continue
                add(widget, f"choice {index}", widget.itemText(index))
        if isinstance(widget, QLineEdit):
            add(widget, "placeholder", widget.placeholderText())
        if isinstance(widget, QAbstractSpinBox) and hasattr(widget, "suffix"):
            add(widget, "prefix", widget.prefix())
            add(widget, "suffix", widget.suffix())
        add(widget, "tooltip", widget.toolTip())
        # The scroll buttons Qt puts in a tab bar take their names from Qt's catalogs.
        if not isinstance(widget.parentWidget(), QTabBar):
            add(widget, "accessible name", widget.accessibleName())
            add(widget, "accessible description", widget.accessibleDescription())
    return found


def cut_off_texts(window):
    """(where, text) for translated text that window shows only in part.

    Only shown buttons and labels are measured, since a widget on a hidden
    tab has no final size yet. A button is measured with the room its style
    leaves for text; a label that neither wraps nor holds markup, with the
    room inside its margins.
    """
    found = []
    for button in window.findChildren(QPushButton):
        if button.isVisible() and is_pseudo_translated(button.text()) and clipped_button_text(button):
            found.append((_describe(button, "text"), button.text()))
    for label in window.findChildren(QLabel):
        text = label.text()
        if not (label.isVisible() and is_pseudo_translated(text)) or label.wordWrap() or "<" in text:
            continue
        room = label.contentsRect().width() - 2 * label.margin() - max(label.indent(), 0)
        needed = max(label.fontMetrics().horizontalAdvance(line) for line in text.split("\n"))
        if needed > room:
            found.append((_describe(label, "text"), text))
    return found


# What puts text on the Viewer's canvas, with the position and the name of
# its text argument: the console line and the background-job status line,
# the helpers that pass their text on to them, and the name of what failed
# that report_selection_error starts the console line with.
CANVAS_TEXT_CALLS = {
    "show_status": (1, "message"),
    "print_help": (1, "msg"),
    "set_background_job_status": (0, "message"),
    "_set_viewer_status": (0, "message"),
    "_set_console_message": (1, "message"),
    "_set_console_text": (1, "message"),
    "report_selection_error": (3, "operation"),
}
# The HUD's text visuals, whose .text a file sets itself.
HUD_TEXTS = frozenset({"instr_text", "zoom_text", "hidden_text", "background_job_status_text"})
# Calls whose result is marked text: a message, or a translation.
MARKING_CALLS = frozenset({"Message", "JoinedMessage", "translate", "tr", "display_text"})


def _called_name(call):
    function = call.func
    return function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)


def _has_letter(text):
    return any(character.isalpha() for character in text)


def _unmarked(node, assignments):
    """Whether node, an expression a file shows on the canvas, holds text no catalog supplies.

    Text with a letter written in the code, as a string or in an f-string, is
    unmarked, and so is str() of a value, which shows a message's English, and
    text joined from parts. A name is followed to what its function assigns
    to it. Anything else, such as a parameter or another function's result,
    is the caller's or that function's to mark.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and _has_letter(node.value)
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(part, ast.Constant) and _has_letter(part.value) for part in node.values)
    if isinstance(node, ast.BinOp):
        return _unmarked(node.left, assignments) or _unmarked(node.right, assignments)
    if isinstance(node, ast.IfExp):
        return _unmarked(node.body, assignments) or _unmarked(node.orelse, assignments)
    if isinstance(node, ast.Name):
        return any(_unmarked(value, assignments) for value in assignments.get(node.id, ()))
    if isinstance(node, ast.Call):
        name = _called_name(node)
        if name in MARKING_CALLS:
            return False
        if name == "format" and isinstance(node.func, ast.Attribute):
            return _unmarked(node.func.value, assignments)
        if name == "join" and isinstance(node.func, ast.Attribute) and node.args \
                and isinstance(node.args[0], (ast.List, ast.Tuple)):
            # Parts written out, such as a dialog's translated file filters.
            return _unmarked(node.func.value, assignments) or any(
                _unmarked(part, assignments) for part in node.args[0].elts
            )
        return name in ("str", "join")
    return False


def _shown_argument(call, canvas_calls):
    """The argument call shows on the canvas, if it is a canvas call, else None."""
    position, name = canvas_calls.get(_called_name(call), (None, None))
    if position is None:
        return None
    if len(call.args) > position:
        return call.args[position]
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _canvas_calls(tree):
    """CANVAS_TEXT_CALLS, and the functions of tree that pass a parameter on to one.

    A command's helper such as _report_error(viewer, msg), which hands msg to
    print_help, puts its text on the canvas just as print_help does, and so
    does a helper that hands it to that helper.
    """
    calls = dict(CANVAS_TEXT_CALLS)
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    changed = True
    while changed:
        changed = False
        for function in functions:
            if function.name in calls:
                continue
            parameters = [argument.arg for argument in function.args.args]
            for node in ast.walk(function):
                shown = _shown_argument(node, calls) if isinstance(node, ast.Call) else None
                if isinstance(shown, ast.Name) and shown.id in parameters:
                    position = parameters.index(shown.id)
                    if parameters[0] in ("self", "cls"):
                        position -= 1  # Called as self.helper(...).
                    calls[function.name] = (position, shown.id)
                    changed = True
                    break
    return calls


def unmarked_canvas_texts(source):
    """(line, code) for each text source shows on the Viewer's canvas unmarked.

    That is the text given to the console line and the background-job status
    line (CANVAS_TEXT_CALLS) and set on the HUD's text visuals (HUD_TEXTS),
    whether written there or assigned to a name passed there, in the same
    function or, for a name the function doesn't assign, at the module's top
    level, as a help text is. A marked text is a Message, a JoinedMessage, a
    translate() result, or display_text() of a value.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    found = []

    def assignments_in(nodes):
        assignments = {}
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.value)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and isinstance(node.target, ast.Name) and node.value:
                assignments.setdefault(node.target.id, []).append(node.value)
        return assignments

    module_assignments = assignments_in(tree.body)
    canvas_calls = _canvas_calls(tree)

    def check(function):
        assignments = {**module_assignments, **assignments_in(ast.walk(function))}
        for node in ast.walk(function):
            shown = None
            if isinstance(node, ast.Call):
                shown = _shown_argument(node, canvas_calls)
                if shown is None and _called_name(node) == "Text":
                    shown = next((keyword.value for keyword in node.keywords if keyword.arg == "text"), None)
            elif isinstance(node, ast.Assign):
                target = node.targets[0]
                if (isinstance(target, ast.Attribute) and target.attr == "text"
                        and isinstance(target.value, ast.Attribute) and target.value.attr in HUD_TEXTS):
                    shown = node.value
            if shown is not None and _unmarked(shown, assignments):
                found.append((node.lineno, lines[node.lineno - 1].strip()))

    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for function in functions:
        check(function)
    # A function nested in another is walked with it too; count each line once.
    return sorted(set(found))


# Qt calls that open a dialog with text given after the parent widget.
DIALOG_CALLS = frozenset({
    "critical", "warning", "information", "question", "about",
    "getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory",
})


def unmarked_dialog_texts(source):
    """(line, code) for each dialog call in source given text no catalog supplies.

    Every argument after the parent widget is checked as unmarked_canvas_texts
    checks canvas text: a title, a message, a file filter.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    module_assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_assignments.setdefault(target.id, []).append(node.value)
    found = []
    for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        assignments = dict(module_assignments)
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.value)
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and _called_name(node) in DIALOG_CALLS:
                if any(_unmarked(argument, assignments) for argument in node.args[1:]):
                    found.append((node.lineno, lines[node.lineno - 1].strip()))
    return sorted(set(found))


def shows_canvas_text(source):
    """Whether source puts any text on the Viewer's canvas, marked or not."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _called_name(node) in CANVAS_TEXT_CALLS:
            return True
    return False


def outside_the_catalog(text):
    """The letters of text that no catalog supplied, under the pseudo-language.

    A translated text is bracketed with the values filled into it, and a
    message may join several, or hold one inside another, so bracketed
    pieces are taken out innermost first. What is left came from no catalog.
    """
    previous = None
    while previous != text:
        previous, text = text, re.sub(r"\[[^\[\]]*\]", "", text)
    return "".join(character for character in text if character.isalpha())


def pseudo_language(test_case, app, catalog_dir=LANGUAGES_DIR):
    """Install the pseudo-language made from <catalog_dir>/emapssn.ts until test_case ends."""
    installed = install_translations(app, PSEUDO_LANGUAGE, catalog_dir=catalog_dir)
    test_case.addCleanup(installed.remove)
    return installed
