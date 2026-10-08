# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Helpers for checking a window's text against the pseudo-language.

Under the pseudo-language (Desktop_App.install_translations(app,
PSEUDO_LANGUAGE)), a text that came from a catalog is bracketed, so in
visible_texts() the unbracketed ones are text never marked for translation,
and cut_off_texts() lists translated text a window doesn't show whole.
pseudo_language() installs the pseudo-language for one test.

Not collected by unittest; import with ``from tests.translation_fixtures import ...``.
"""
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QWidget,
)

from desktop.Desktop_App import (
    PSEUDO_LANGUAGE,
    SYSTEM_LANGUAGE,
    LanguageSelector,
    clipped_button_text,
    install_translations,
)
from utilities.Localization import LANGUAGES_DIR, is_pseudo_translated


def _describe(widget, part):
    name = widget.objectName()
    return f"{type(widget).__name__}{f' {name}' if name else ''} {part}"


def visible_texts(window):
    """(where, text) for every text a user can read in window.

    That is its title, labels, buttons, group and tab titles, list choices,
    placeholders, spin box units and tooltips, on every tab, shown or not.
    What a user typed or picked (an edit's text, a spin box's value) is
    their data, not the window's text, so it is left out, as is text
    without a letter, such as a number, an arrow or an emoji alone. So are
    the language names in the Language dropdown, which each show their own
    language on purpose.
    """
    found = []

    def add(widget, part, text):
        if text and any(character.isalpha() for character in text):
            found.append((_describe(widget, part), text))

    add(window, "title", window.windowTitle())
    for widget in [window, *window.findChildren(QWidget)]:
        if isinstance(widget, (QLabel, QAbstractButton)):
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
                add(widget, f"choice {index}", widget.itemText(index))
        if isinstance(widget, QLineEdit):
            add(widget, "placeholder", widget.placeholderText())
        if isinstance(widget, QAbstractSpinBox) and hasattr(widget, "suffix"):
            add(widget, "prefix", widget.prefix())
            add(widget, "suffix", widget.suffix())
        add(widget, "tooltip", widget.toolTip())
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


def pseudo_language(test_case, app, catalog_dir=LANGUAGES_DIR):
    """Install the pseudo-language made from <catalog_dir>/emapssn.ts until test_case ends."""
    installed = install_translations(app, PSEUDO_LANGUAGE, catalog_dir=catalog_dir)
    test_case.addCleanup(installed.remove)
    return installed
