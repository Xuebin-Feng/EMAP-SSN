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
    placeholders, spin box units and tooltips, on every tab, shown or not.
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
