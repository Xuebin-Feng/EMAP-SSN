# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Text for the user: English where it is recorded, translatable where it is shown.

A Message keeps an English template and the values that fill it. str(message)
is the English sentence that goes to the terminal, the command portal (MCP
clients and the agent page) and the tests. message.display() is what a window
shows: the template passes through the installed translator first. Until a
translator is installed, both are the same text.

A counted text holds %n, Qt's count, and marks its English plural endings,
as in "%n file(s)": English shows "1 file" and "2 files" (english_plural),
and a translation gives each plural form its language has.

Translations live in Qt catalogs in src/resources/languages: emapssn.ts lists
every text the program can show, and emapssn_<language>.ts holds one
language's translations. Update_Translations.py in the same folder collects the
texts from the code, Message templates included, which it files under
MESSAGE_CONTEXT.

The pseudo-language is a test-only language made from emapssn.ts: every text
comes out accented, longer and bracketed, so text that skipped the catalog
shows up in English, and text cut off at its end loses its closing bracket.

Nothing here imports Qt, so the VR viewer process can use it as well.
"""

from dataclasses import dataclass
import operator
from pathlib import Path
import re
from xml.etree import ElementTree

LANGUAGES_DIR = Path(__file__).resolve().parents[1] / "resources" / "languages"
CATALOG_NAME = "emapssn"
TEMPLATE_CATALOG = LANGUAGES_DIR / f"{CATALOG_NAME}.ts"

# The catalog context of every Message template. A window's translator looks
# the templates up here, and the update command files them here.
MESSAGE_CONTEXT = "Message"

_translator = None


def set_translator(translator):
    """Install translator(template, n) -> shown template, or None to remove it.

    n is the count of a counted template, whose shown form has %n filled
    in, and -1 for any other template. Returns the translator it replaces,
    so a caller can restore it.
    """
    global _translator
    previous, _translator = _translator, translator
    return previous


def translate(template, n=-1):
    """Return the shown form of an English template, counted by n (-1 for none)."""
    return english_text(template, n) if _translator is None else _translator(template, n)


# ---------------------------------------------------------------------
# English plurals
# ---------------------------------------------------------------------

# A plural ending marked right after its word, as in "file(s)" or "match(es)".
_PLURAL_ENDING = re.compile(r"(?<=[^\W\d_])\((e?s)\)")


def has_plural_ending(text):
    """Whether text marks an English plural ending, as "%n file(s)" does."""
    return _PLURAL_ENDING.search(text) is not None


def english_plural(text, n):
    """text with its marked plural endings as English writes them for the count n.

    A count of one drops each ending, and any other count keeps it without
    its brackets: "%n file(s)" reads "%n file" for one and "%n files"
    otherwise, and "%n match(es)" works alike. A negative n counts nothing
    and leaves text as it is. %n itself stays, for Qt or english_text to
    fill in.
    """
    if n < 0:
        return text
    return _PLURAL_ENDING.sub("" if n == 1 else r"\1", text)


def english_text(template, n=-1):
    """The English shown for template counted by n: its plural endings chosen and %n filled in."""
    return template if n < 0 else english_plural(template, n).replace("%n", str(n))


class Message:
    """An English template and the values that fill it.

    A counted message's template holds %n, Qt's count, and the value n is
    the count: Message("Removed %n group(s).", n=2) reads "Removed 2
    groups." The template marks its plural endings, as english_plural
    explains, and a translation takes the form its language uses for n.
    """

    __slots__ = ("template", "values")

    def __init__(self, template, **values):
        if "%n" in template and "n" in values:
            values["n"] = operator.index(values["n"])  # A count is a whole number, such as len().
        self.template = template
        self.values = values

    @property
    def count(self):
        """n for a counted message, and -1 for any other."""
        return self.values.get("n", -1) if "%n" in self.template else -1

    def __str__(self):
        return english_text(self.template, self.count).format(**self.values)

    def __repr__(self):
        return f"Message({self.template!r}, **{self.values!r})"

    def display(self):
        """The sentence a window shows: the translated template, filled in.

        A value that is itself a Message, or an error raised with one, is
        shown translated too. A translation whose placeholders don't match
        the template's would fail to fill in; the English sentence is shown
        instead, so the message is never lost.
        """
        values = {
            key: display_text(value) if isinstance(value, (Message, BaseException)) else value
            for key, value in self.values.items()
        }
        try:
            return translate(self.template, self.count).format(**values)
        except (KeyError, IndexError, ValueError):
            return str(self)


class JoinedMessage(Message):
    """Messages shown one after another, such as the sentences of a report.

    str() joins their English, and display() their translations, so a
    message put together from parts is still one Message: the console line
    shows it translated and the terminal prints it in English. A part may
    also be plain text, such as a file name, which shows as it is.
    """

    __slots__ = ("parts", "separator")

    def __init__(self, parts, separator=" "):
        super().__init__("")
        self.parts = tuple(parts)
        self.separator = separator

    def __str__(self):
        return self.separator.join(str(part) for part in self.parts)

    def __repr__(self):
        return f"JoinedMessage({list(self.parts)!r}, separator={self.separator!r})"

    def display(self):
        return self.separator.join(display_text(part) for part in self.parts)


def display_text(message):
    """What a window shows for message: a Message's display text, or the text itself.

    An error raised with a Message, as in ValueError(Message(...)), shows
    that Message, while str(error) stays English for the terminal.
    """
    if isinstance(message, BaseException) and len(message.args) == 1:
        message = message.args[0]
    return message.display() if isinstance(message, Message) else str(message)


def QT_TRANSLATE_NOOP(context, text):
    """Mark text for the catalog under context, and return it as it is.

    Qt's function of this name, for code that must not import Qt, such as a
    module's table of labels: the catalog lists the text, and the window
    translates it where it shows it, with QCoreApplication.translate(context, text).
    """
    return text


# ---------------------------------------------------------------------
# Placeholders: the parts of a text the program fills in or reads back
# ---------------------------------------------------------------------

_FILL_IN_PATTERNS = (
    r"\{\{|\}\}|\{[^{}]*\}",                                # str.format fields, {{ and }}
    r"%L?(?:n|[1-9]\d?)",                                   # Qt's %n and %1 ... %99
    # printf codes. Without the space flag, so the percent sign of
    # "50% disk space" or "2.00% coverage" stays plain text.
    r"%(?:\([^)]*\))?[-#0+]*\d*(?:\.\d+)?[sdifrxXeEgGc%]",
)
_FILL_INS = re.compile("|".join(_FILL_IN_PATTERNS))
_KEPT_AS_WRITTEN = re.compile("|".join(_FILL_IN_PATTERNS + (
    r"</?[A-Za-z][^<>]*>",                        # markup tags
    r"&(?:[A-Za-z]+|#\d+|#[xX][0-9A-Fa-f]+);",    # character entities
    r"&&",                                        # an "&" shown in a shortcut text
    r"https?://\S+",                              # links
    r"\*[^\s;)]*",                                # file patterns such as *.fasta
)))


def fill_ins(text):
    """The placeholders a translation of text must keep, sorted.

    These are str.format fields such as {count}, Qt's %n and %1, and printf
    codes such as %s. The escaped braces {{ and }} and %% are not counted.
    """
    return sorted(
        match.group() for match in _FILL_INS.finditer(text)
        if match.group() not in ("{{", "}}", "%%")
    )


# ---------------------------------------------------------------------
# The test-only pseudo-language
# ---------------------------------------------------------------------

PSEUDO_OPEN = "["
PSEUDO_CLOSE = "]"
_PSEUDO_LETTERS = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "åƀçđéƒĝĥîĵķļṁñöþǫŕšţûṽŵẋýžÅƁÇĐÉƑĜĤÎĴĶĻṀÑÖÞǪŔŠŢÛṼŴẊÝŽ",
)
_PSEUDO_VOWELS = re.compile("[åéîöûÅÉÎÖÛ]")


def _pseudo_letters(text):
    return _PSEUDO_VOWELS.sub(lambda vowel: vowel.group() * 2, text.translate(_PSEUDO_LETTERS))


def pseudo_translate(text):
    """The pseudo-language's form of an English text.

    Every letter takes an accent and every vowel is doubled, which lengthens
    text about as much as many real translations do, and the text is
    bracketed: "Save" becomes "[Šååṽéé]". What the program fills in or
    reads back stays as written: {fields}, %n, markup, links and file
    patterns. Spaces around the text stay outside the brackets.
    """
    body = text.strip()
    if not body:
        return text
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):]
    pieces, position = [], 0
    for kept in _KEPT_AS_WRITTEN.finditer(body):
        pieces.append(_pseudo_letters(body[position:kept.start()]))
        pieces.append(kept.group())
        position = kept.end()
    pieces.append(_pseudo_letters(body[position:]))
    return f"{leading}{PSEUDO_OPEN}{''.join(pieces)}{PSEUDO_CLOSE}{trailing}"


def is_pseudo_translated(text):
    """Whether shown text came whole through the pseudo-language: it is bracketed."""
    body = text.strip()
    return body.startswith(PSEUDO_OPEN) and body.endswith(PSEUDO_CLOSE)


# ---------------------------------------------------------------------
# Reading a catalog
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CatalogMessage:
    """One text in a Qt translation catalog (.ts file).

    comment is Qt's disambiguation, which tells apart two uses of one
    English text. translations holds the translation, or one translation per
    plural form when numerus is true. status is "" for a finished
    translation, "unfinished", or "vanished"/"obsolete" for a text that left
    the code.
    """

    context: str
    source: str
    comment: str = ""
    numerus: bool = False
    translations: tuple[str, ...] = ()
    status: str = ""

    @property
    def key(self):
        """What a translator looks a text up by."""
        return (self.context, self.source, self.comment)


def read_catalog(path):
    """Every message in the Qt translation catalog at path, in file order."""
    messages = []
    for context in ElementTree.parse(path).getroot().iter("context"):
        context_name = context.findtext("name", default="")
        for message in context.iter("message"):
            translation = message.find("translation")
            if translation is None:
                texts, status = (), ""
            else:
                forms = translation.findall("numerusform")
                texts = tuple(form.text or "" for form in forms) if forms else (translation.text or "",)
                status = translation.get("type", "")
            messages.append(CatalogMessage(
                context=context_name,
                source=message.findtext("source", default=""),
                comment=message.findtext("comment", default=""),
                numerus=message.get("numerus") == "yes",
                translations=texts,
                status=status,
            ))
    return messages
