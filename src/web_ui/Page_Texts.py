# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The texts of the bundled web pages, and the pages shown in a language.

A page marks its texts for translation, and the update command
(src/resources/languages/Update_Translations.py) files them in the catalogs
under the page's context (PAGE_CONTEXTS), beside the windows' texts. A text
is marked one of three ways:

* An element whose content is one text carries data-i18n, as in
  <button data-i18n>Clear Chat</button>. The content may hold markup, such
  as <b> or <br>, which a translation keeps; its spaces and line breaks
  count as one space.
* A title, placeholder, aria-label or alt attribute that holds a letter is
  a text.
* A script passes a text to t() as a plain string literal, as in
  t("Agent activated: {model}", {model: name}). t(), from page_text.js,
  fills in the values the way a Message does.

A <script>, a <style> and an element with translate="no" hold no texts
but their scripts' t() calls.

The Viewer's web server shows the pages in the Viewer's language: as it
serves a page or one of its scripts, translated_file() writes each text's
translation in place of its English. So a page needs no catalog of its
own, and t() only fills in values. Nothing here imports Qt.
"""

import bisect
from dataclasses import dataclass, field
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

SRC_DIR = Path(__file__).resolve().parents[1]

# Each file that holds a page's texts, relative to src, and the catalog
# context its texts are filed under.
PAGE_CONTEXTS = {
    "web_ui/agent.html": "AgentPage",
    "resources/agent/attachments.js": "AgentPage",
    "web_ui/meta.html": "MetadataPage",
}
MARK = "data-i18n"
TEXT_ATTRIBUTES = frozenset({"title", "placeholder", "aria-label", "alt"})
# Elements without an end tag, so without content to mark.
_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "source", "track", "wbr",
})
_UNSHOWN_ELEMENTS = frozenset({"script", "style", "template"})


def has_letter(text):
    """Whether text holds a letter, which makes it language."""
    return any(character.isalpha() for character in text)


@dataclass(frozen=True)
class PageText:
    """One marked text: its English, and where its file writes it.

    start and end span what a translation replaces: an element's content
    without the spaces around it ("content"), an attribute's value with its
    quotes ("attribute"), or a string literal with its quotes ("script").
    line is start's line, counted from 1.
    """

    text: str
    start: int
    end: int
    line: int
    kind: str


@dataclass
class PageSource:
    """What read_page found in a page's source.

    texts are the marked texts, in file order. problems are (line,
    explanation) for texts no catalog can list. unmarked are (line, text)
    for text an element shows outside any marked element, and scripts the
    tokens of each script: an inline <script>, or a whole .js file.
    """

    texts: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    unmarked: list = field(default_factory=list)
    scripts: list = field(default_factory=list)


# ---------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------

class ScriptError(ValueError):
    """A script the lexer can't read, such as one with an unclosed string."""


@dataclass(frozen=True)
class ScriptToken:
    """A token of a script: a "name", "number", "string", "template", "regex" or "punct".

    A string's value is the text it stands for; any other token's is its
    source. A template keeps its literal parts as (source, start, end) and
    the tokens of each ${...} in it.
    """

    kind: str
    value: str
    start: int
    end: int
    parts: tuple = ()
    expressions: tuple = ()


_STRING = re.compile(r""""(?:[^"\\\r\n]|\\.)*"|'(?:[^'\\\r\n]|\\.)*'""", re.DOTALL)
_REGEX = re.compile(r"/(?:[^/\\\[\r\n]|\\.|\[(?:[^\]\\\r\n]|\\.)*\])+/[A-Za-z]*")
_NAME = re.compile(r"[A-Za-z_$][\w$]*")
_NUMBER = re.compile(r"\.?\d[\w.]*")
_PUNCTUATORS = (
    ">>>=", "...", "===", "!==", "**=", "<<=", ">>=", ">>>", "&&=", "||=", "??=",
    "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--", "+=", "-=",
    "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "**",
)
# Words after which a slash starts a regular expression, not a division.
_BEFORE_EXPRESSION = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "throw", "case", "do", "else", "yield", "await",
})
_ESCAPE = re.compile(r"\\(u\{[0-9A-Fa-f]+\}|u[0-9A-Fa-f]{4}|x[0-9A-Fa-f]{2}|\r\n|.)", re.DOTALL)
_SINGLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def _string_value(literal):
    """The text a JavaScript string literal, quotes and all, stands for."""
    def read(match):
        escape = match.group(1)
        if escape[0] in "ux" and len(escape) > 1:
            return chr(int(escape.strip("ux{}"), 16))
        if escape in ("\n", "\r", "\r\n", "\u2028", "\u2029"):
            return ""  # A backslash at the end of a line continues the string.
        return _SINGLE_ESCAPES.get(escape, escape)

    return _ESCAPE.sub(read, literal[1:-1])


def _regex_may_follow(tokens):
    if not tokens:
        return True
    last = tokens[-1]
    if last.kind == "name":
        return last.value in _BEFORE_EXPRESSION
    return last.kind == "punct" and last.value not in (")", "]")


def lex_script(source, start=0, end=None):
    """The tokens of the script source[start:end], without its comments and spaces.

    Raises ScriptError for a script it can't read through.
    """
    tokens, _ = _lex(source, start, len(source) if end is None else end, closing=False)
    return tokens


def _lex(source, position, end, closing):
    """Tokens from position up to end, or with closing, up to the } that ends a ${...}."""
    tokens = []
    depth = 0
    while position < end:
        character = source[position]
        if character.isspace():
            position += 1
            continue
        if source.startswith("//", position):
            newline = source.find("\n", position, end)
            position = end if newline < 0 else newline
            continue
        if source.startswith("/*", position):
            close = source.find("*/", position + 2, end)
            if close < 0:
                raise ScriptError(f"a comment at offset {position} is never closed")
            position = close + 2
            continue
        if character in "\"'":
            match = _STRING.match(source, position, end)
            if match is None:
                raise ScriptError(f"a string at offset {position} is never closed")
            tokens.append(ScriptToken("string", _string_value(match.group()), position, match.end()))
            position = match.end()
            continue
        if character == "`":
            token = _template(source, position, end)
            tokens.append(token)
            position = token.end
            continue
        if character == "/" and _regex_may_follow(tokens):
            match = _REGEX.match(source, position, end)
            if match is not None:
                tokens.append(ScriptToken("regex", match.group(), position, match.end()))
                position = match.end()
                continue
        for kind, pattern in (("name", _NAME), ("number", _NUMBER)):
            match = pattern.match(source, position, end)
            if match is not None:
                tokens.append(ScriptToken(kind, match.group(), position, match.end()))
                position = match.end()
                break
        else:
            if closing and character == "}" and depth == 0:
                return tokens, position
            punctuator = next((p for p in _PUNCTUATORS if source.startswith(p, position)), character)
            depth += {"{": 1, "}": -1}.get(punctuator, 0)
            tokens.append(ScriptToken("punct", punctuator, position, position + len(punctuator)))
            position += len(punctuator)
    if closing:
        raise ScriptError("a template's ${ is never closed")
    return tokens, position


def _template(source, start, end):
    """The template literal whose backquote is at start."""
    parts, expressions = [], []
    position = part_start = start + 1
    while position < end:
        character = source[position]
        if character == "\\":
            position += 2
            continue
        if character == "`":
            parts.append((source[part_start:position], part_start, position))
            return ScriptToken(
                "template", source[start:position + 1], start, position + 1, tuple(parts), tuple(expressions),
            )
        if source.startswith("${", position):
            parts.append((source[part_start:position], part_start, position))
            tokens, close = _lex(source, position + 2, end, closing=True)
            expressions.append(tuple(tokens))
            position = part_start = close + 1
            continue
        position += 1
    raise ScriptError(f"a template at offset {start} is never closed")


def is_t_call(tokens, index):
    """Whether tokens[index] is the name of a call to the pages' t()."""
    token = tokens[index]
    if token.kind != "name" or token.value != "t":
        return False
    before = tokens[index - 1] if index else None
    if before is not None and (before.value in (".", "?.") or before.value == "function"):
        return False  # A method named t, or t's own definition.
    return index + 1 < len(tokens) and tokens[index + 1].kind == "punct" and tokens[index + 1].value == "("


_T_ADVICE = (
    "t() needs its English text as a plain string literal, so the catalog can list it. "
    'Fill values in afterwards, as in t("Saved {name}.", {name: value}).'
)


def _script_texts(tokens, page, line_of):
    """Add the texts tokens pass to t(), and the t() calls no catalog can list, to page."""
    for index, token in enumerate(tokens):
        for expression in token.expressions:
            _script_texts(expression, page, line_of)
        if not is_t_call(tokens, index):
            continue
        argument = tokens[index + 2:index + 4]
        if len(argument) == 2 and argument[0].kind == "string" and argument[1].value in (",", ")"):
            literal = argument[0]
            page.texts.append(PageText(literal.value, literal.start, literal.end, line_of(literal.start), "script"))
        else:
            page.problems.append((line_of(token.start), _T_ADVICE))


def _read_script(source, start, end, page, line_of):
    try:
        tokens = lex_script(source, start, end)
    except ScriptError as error:
        page.problems.append((line_of(start), f"a script can't be read, so its texts were not collected: {error}"))
        return
    page.scripts.append(tokens)
    _script_texts(tokens, page, line_of)


# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------

_TAG_NAME = re.compile(r"<[^\s/>]+")
_ATTRIBUTE = re.compile(r"""([^\s"'<>/=]+)(?:\s*=\s*("[^"]*"|'[^']*'|[^\s"'=<>`]+))?""")
_MARKUP = re.compile(r"<[^>]*>")


@dataclass
class _Element:
    tag: str
    content_start: int
    marked: bool
    unshown: bool


class _PageParser(HTMLParser):
    """Reads one page into a PageSource."""

    def __init__(self, source, page, line_of):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.page = page
        self.line_of = line_of
        self.line_starts = [0] + [match.end() for match in re.finditer("\n", source)]
        self.open = []

    def _position(self):
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def inside(self, flag):
        return any(getattr(element, flag) for element in self.open)

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs, closed=tag in _VOID_ELEMENTS)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs, closed=True)

    def _start(self, tag, attrs, closed):
        start = self._position()
        raw = self.get_starttag_text()
        attributes = dict(attrs)
        marked = MARK in attributes
        unshown = (
            self.inside("unshown") or tag in _UNSHOWN_ELEMENTS
            or (attributes.get("translate") or "").lower() == "no"
        )
        if not unshown and not self.inside("marked"):
            self._attribute_texts(raw, start)
        if marked and closed:
            self.page.problems.append((self.line_of(start), f"{MARK} marks an element's content, and <{tag}> has none."))
        elif marked and self.inside("marked"):
            self.page.problems.append(
                (self.line_of(start), f"<{tag} {MARK}> is inside another marked element, so its text is marked twice.")
            )
        if not closed:
            self.open.append(_Element(tag, start + len(raw), marked, unshown))

    def _attribute_texts(self, raw, start):
        for match in _ATTRIBUTE.finditer(raw, _TAG_NAME.match(raw).end()):
            name, value = match.group(1).lower(), match.group(2)
            if name not in TEXT_ATTRIBUTES or value is None:
                continue
            text = html.unescape(value[1:-1] if value[0] in "\"'" else value)
            # A browser reads a line break in the value as \n, whatever the file holds.
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            if has_letter(text):
                value_start = start + match.start(2)
                self.page.texts.append(
                    PageText(text, value_start, start + match.end(2), self.line_of(value_start), "attribute")
                )

    def handle_endtag(self, tag):
        end = self._position()
        if not any(element.tag == tag for element in self.open):
            return
        while self.open:
            element = self.open.pop()
            if element.tag == "script":
                _read_script(self.source, element.content_start, end, self.page, self.line_of)
            elif element.marked:
                self._content_text(element, end)
            if element.tag == tag:
                return

    def _content_text(self, element, end):
        content = self.source[element.content_start:end]
        start = element.content_start + len(content) - len(content.lstrip())
        if not has_letter(_MARKUP.sub("", content)):
            self.page.problems.append((self.line_of(start), f"<{element.tag} {MARK}> holds no text."))
            return
        stripped = content.strip()
        self.page.texts.append(
            PageText(" ".join(stripped.split()), start, start + len(stripped), self.line_of(start), "content")
        )

    def handle_data(self, data):
        if self.inside("unshown") or self.inside("marked") or not has_letter(data):
            return
        self.page.unmarked.append((self.line_of(self._position()), " ".join(data.split())))


def read_page(source, suffix):
    """What a page's source holds: an .html page, or a .js script by suffix."""
    page = PageSource()
    line_starts = [0] + [match.end() for match in re.finditer("\n", source)]

    def line_of(offset):
        return bisect.bisect_right(line_starts, offset)

    if suffix.lower() == ".js":
        _read_script(source, 0, len(source), page, line_of)
    else:
        parser = _PageParser(source, page, line_of)
        parser.feed(source)
        parser.close()
    for text in page.texts:
        if "%n" in text.text:
            page.problems.append((text.line, f"{text.text!r} holds %n, but a page's text can't be counted. "
                                             "Rephrase it without the count."))
        if text.kind == "script" and text.text.rstrip().endswith(":"):
            # A label the script puts in front of a value, with its own space,
            # leaves that space in every language: Chinese would read "错误： …".
            page.problems.append((text.line, f"{text.text!r} is a label cut off from what follows it, so a "
                                             "translation can't place its own spacing. Mark the whole line, as in "
                                             't("<b>Error:</b> {error}", {error: value}).'))
    page.texts.sort(key=lambda text: text.start)
    page.problems.sort()
    return page


# ---------------------------------------------------------------------
# A page in a language
# ---------------------------------------------------------------------

def _written(translation, kind):
    """translation as the file writes it where the text was."""
    if kind == "attribute":
        return '"' + html.escape(translation, quote=True) + '"'
    if kind == "script":
        literal = json.dumps(translation, ensure_ascii=False)
        # Inside a page's <script>, "</script" would end it. JSON leaves the
        # line and paragraph separators raw, which older JavaScript refused.
        return literal.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return translation


def translated_source(source, suffix, translate):
    """source, a page or a script by suffix, with translate(text) in place of each marked text."""
    pieces, position = [], 0
    for text in read_page(source, suffix).texts:
        pieces += [source[position:text.start], _written(translate(text.text), text.kind)]
        position = text.end
    pieces.append(source[position:])
    return "".join(pieces)


_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:_[A-Z][a-z]{3})?(?:_[A-Z]{2})?$")
_HTML_LANG = re.compile(r"""(<html\b[^>]*?\slang\s*=\s*)("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)
_HEAD_END = re.compile(r"</head\s*>", re.IGNORECASE)


def language_tag(language):
    """A catalog's language as HTML names it, such as zh-CN for zh_CN; None for a test language."""
    return language.replace("_", "-") if _LANGUAGE_CODE.match(language or "") else None


def page_context(path):
    """The catalog context of the page file at path, or None for any other file."""
    try:
        relative = Path(path).resolve().relative_to(SRC_DIR).as_posix()
    except (OSError, ValueError):
        return None
    return PAGE_CONTEXTS.get(relative)


def translated_file(path, body, language, translate, styles=""):
    """The bytes to serve for the file at path, whose own bytes are body.

    A page's file comes in language: translate(context, text) gives each
    text's translation, and <html lang> names a real language. styles, CSS
    such as the language's fonts, goes at the end of such a page's head.
    Any other file, and any file for language None (English), comes as it is.
    """
    context = page_context(path)
    if context is None or not language:
        return body
    suffix = Path(path).suffix.lower()
    shown = translated_source(body.decode("utf-8"), suffix, lambda text: translate(context, text))
    tag = language_tag(language)
    if suffix == ".html" and tag:
        shown = _HTML_LANG.sub(lambda match: f'{match.group(1)}"{tag}"', shown, count=1)
        if styles:
            shown = _HEAD_END.sub(lambda match: f"<style>\n{styles}\n</style>\n{match.group(0)}", shown, count=1)
    return shown.encode("utf-8")
