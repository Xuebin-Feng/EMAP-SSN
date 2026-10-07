# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Text for the user: English where it is recorded, translatable where it is shown.

A Message keeps an English template and the values that fill it. str(message)
is the English sentence that goes to the terminal, the command portal (MCP
clients and the agent page) and the tests. message.display() is what a window
shows: the template passes through the installed translator first. Until a
translator is installed, both are the same text.

Nothing here imports Qt, so the VR viewer process can use it as well.
"""

_translator = None


def set_translator(translator):
    """Install translator(template) -> shown template, or None to remove it.

    Returns the translator it replaces, so a caller can restore it.
    """
    global _translator
    previous, _translator = _translator, translator
    return previous


def translate(template):
    """Return the shown form of an English template."""
    return template if _translator is None else _translator(template)


class Message:
    """An English template and the values that fill it."""

    __slots__ = ("template", "values")

    def __init__(self, template, **values):
        self.template = template
        self.values = values

    def __str__(self):
        return self.template.format(**self.values)

    def __repr__(self):
        return f"Message({self.template!r}, **{self.values!r})"

    def display(self):
        """The sentence a window shows: the translated template, filled in."""
        return translate(self.template).format(**self.values)


def display_text(message):
    """What a window shows for message: a Message's display text, or the text itself."""
    return message.display() if isinstance(message, Message) else str(message)
